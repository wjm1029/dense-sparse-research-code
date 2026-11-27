"""Sparse long-video self-forcing then bounded OPSD coordinate distillation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from train_wan22_sparse_ddp import sha,cpu,rng,restore_rng
from train_wan22_sparse_prompt_full_ddp import coordinate_loss
from wan22_sparse_task_router import TASK_PROMPTS
from wan22_sparse_causal_ar import pack_coordinate_groups
from wan22_sparse_long477 import LongSparseDataset,LongSparseModule,chunk_raw_span,feedback_groups,clean_teacher_forward
from wan22_ar_self_forcing import detach_kv_cache
from wan22_opsd_prefix_cache import update_ema_


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--phase',choices=('self_forcing','opsd_local'),required=True)
 p.add_argument('--index_dir',required=True); p.add_argument('--output_dir',required=True)
 p.add_argument('--init_checkpoint',required=True); p.add_argument('--max_steps',required=True,type=int)
 p.add_argument('--model_dir',default='/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B')
 p.add_argument('--resume'); p.add_argument('--save_steps',type=int,default=25)
 p.add_argument('--eval_steps',type=int,default=200); p.add_argument('--seed',type=int,default=7)
 p.add_argument('--ema_decay',type=float,default=.9999); p.add_argument('--distill_weight',type=float,default=1.)
 p.add_argument('--loss_start_frame',type=int,default=21)
 p.add_argument('--run_steps',type=int,default=0,help='Smoke cap; does not change formal stage contract/target')
 p.add_argument('--skip_initial_eval',action='store_true')
 p.add_argument('--min_free_gib',type=float,default=40.)
 a=p.parse_args()
 if a.max_steps<=0 or a.run_steps<0 or a.loss_start_frame<0 or a.loss_start_frame%3 or a.loss_start_frame>=120:
  raise ValueError('Invalid step/chunk settings')
 rank,local,world=[int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE')]
 if world!=4: raise ValueError('Formal and smoke stages require four ranks')
 torch.cuda.set_device(local); device=torch.device('cuda',local)
 dist.init_process_group('nccl',timeout=timedelta(hours=2))
 control=dist.new_group(backend='gloo',timeout=timedelta(hours=2))
 free,total=torch.cuda.mem_get_info(device)
 enough=torch.tensor(int(free>=a.min_free_gib*2**30),device=device); dist.all_reduce(enough,op=dist.ReduceOp.MIN)
 if not enough: raise RuntimeError('Not enough free GPU memory on every card; other workloads are preserved')
 random.seed(a.seed+rank); np.random.seed(a.seed+rank); torch.manual_seed(a.seed+rank); torch.cuda.manual_seed_all(a.seed+rank)
 out=Path(a.output_dir).resolve(); here=Path(__file__).resolve().parent
 if rank==0:
  if out.exists() and any(out.iterdir()) and not a.resume: raise FileExistsError(out)
  out.mkdir(parents=True,exist_ok=True)
 dist.barrier(group=control)
 train,val=LongSparseDataset(a.index_dir),LongSparseDataset(a.index_dir,True)
 scripts=('train_wan22_sparse_long477_ddp.py','wan22_sparse_long477.py','wan22_sparse_causal_ar.py',
  'train_wan22_sparse_prompt_full_ddp.py','train_wan22_sparse_ddp.py','wan22_sparse_task_router.py',
  'wan22_dense6_task_router.py','wan22_causal_attention.py','wan22_ar_self_forcing.py',
  'wan22_opsd_prefix_cache.py','wan22_opsd_bounded_causal_attention.py')
 contract={'schema':'wan22_sparse_long477_self_forcing_opsd_v1','phase':a.phase,'max_steps':a.max_steps,
  'index_hash':sha(Path(a.index_dir)/'manifest.json'),'reference_index_hash':train.manifest['reference_index_sha256'],
  'init_checkpoint':str(Path(a.init_checkpoint).resolve()),'init_sha256':sha(a.init_checkpoint),
  'source_hashes':{s:sha(here/s) for s in scripts},'model_dir':a.model_dir,
  'pipeline_hash':sha('diffsynth/pipelines/wan_video.py'),'dit_hash':sha('diffsynth/models/wan_video_dit.py'),
  'prompts':list(TASK_PROMPTS),'world_size':4,'seed':a.seed,'frames':477,'latent_frames':120,
  'chunk':3,'chunks_per_optimizer_step':40,'rank':32,'router_hidden':512,
  'attention':'full_prefix' if a.phase=='self_forcing' else 'sink3_local12_absolute_positions',
  'history':'previous_predicted_coordinates_stopgrad','bptt':'stopgrad_history_and_KV_each_chunk',
  'loss':'coordinate_MSE_plus_0.001_router'+('_plus_masked_coordinate_EMA_distillation' if a.phase=='opsd_local' else ''),
  'loss_start_frame':a.loss_start_frame,'distill_weight':a.distill_weight,'ema_decay':a.ema_decay,
  'learning_rate':2e-5 if a.phase=='self_forcing' else 1e-5,
  'router_lr':2e-6 if a.phase=='self_forcing' else 1e-6,'warmup':20,'weight_decay':.01,
  'train_clips':len(train),'val_clips':len(val),'validation':'predicted_history_only_clip_holdout',
  'teacher_context':'delayed_one_chunk_clean_GT_prefix_replay' if a.phase=='opsd_local' else 'none'}
 with safe_open(a.init_checkpoint,framework='pt',device='cpu') as f:
  source_meta=f.metadata(); source_contract=json.loads(source_meta['contract'])
 if a.phase=='self_forcing':
  if source_contract.get('stage')!='ar' or int(source_meta['step'])!=21750: raise ValueError('Expected sparse AR21750 source')
  if source_contract['index_hash']!=contract['reference_index_hash']: raise ValueError('81-frame target reference mismatch')
  for name,key in [('train_wan22_sparse_ddp.py','base_sparse_hash'),('wan22_sparse_task_router.py','sparse_router_hash'),
    ('wan22_dense6_task_router.py','dense_router_hash'),('wan22_sparse_causal_ar.py','sparse_causal_ar_hash')]:
   if source_contract[key]!=sha(here/name): raise ValueError('Initialization runtime changed: '+name)
 else:
  if source_contract.get('phase')!='self_forcing' or int(source_meta['step'])!=2400: raise ValueError('Expected completed sparse SF2400 source')
  if source_contract['index_hash']!=contract['index_hash'] or source_contract['source_hashes']!=contract['source_hashes']:
   raise ValueError('Self-forcing index/source code mismatch')
 for key in ('model_dir','pipeline_hash','dit_hash','prompts'):
  if source_contract[key]!=contract[key]: raise ValueError('Source contract mismatch: '+key)
 module=LongSparseModule(a.model_dir,a.phase); module.prepare(device)
 names={n for n,q in module.named_parameters() if q.requires_grad}|{'context','task_router.task_feature_table','task_router.features_ready','sparse_head.xy_grid'}
 weights=load_file(a.init_checkpoint,device='cpu')
 if set(weights)!=names: raise ValueError('Inherited parameter/buffer set mismatch')
 for key in ('context','task_router.task_feature_table','sparse_head.xy_grid'):
  if not torch.equal(weights[key],module.state_dict()[key].cpu()): raise ValueError('Condition buffer mismatch: '+key)
 result=module.load_state_dict(weights,strict=False)
 if result.unexpected_keys: raise ValueError(result.unexpected_keys)
 del weights
 groups=[{'params':[q for n,q in module.named_parameters() if q.requires_grad and '.lora_' in n],'lr':contract['learning_rate']},
  {'params':module.sparse_head.parameters(),'lr':contract['learning_rate']},
  {'params':module.task_router.parameters(),'lr':contract['router_lr']}]
 optimizer=torch.optim.AdamW(groups,weight_decay=.01)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.,(s+1)/20))
 named=[(n,q) for n,q in module.named_parameters() if q.requires_grad]
 assigned=[id(q) for g in groups for q in g['params']]
 if len(assigned)!=len(set(assigned)) or set(assigned)!={id(q) for _,q in named}: raise ValueError('Optimizer coverage')
 ema={n:q.detach().clone() for n,q in named} if a.phase=='opsd_local' else None
 step=epoch=cursor=0
 if a.resume:
  state=torch.load(a.resume,map_location='cpu',weights_only=False)
  if state['contract']!=contract or set(state['weights'])!=names: raise ValueError('Resume contract/tensors mismatch')
  module.load_state_dict(state['weights'],strict=False); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
  step,epoch,cursor=state['step'],state['epoch'],state['next_batch']; restore_rng(state['rng'][rank])
  if ema is not None:
   if set(state['teacher_ema'])!=set(ema): raise ValueError('Missing EMA resume weights')
   ema={n:v.to(device) for n,v in state['teacher_ema'].items()}
  del state
 ddp=DDP(module,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
 sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=a.seed,drop_last=True)
 loader=DataLoader(train,batch_size=1,sampler=sampler,num_workers=2,pin_memory=True,
  generator=torch.Generator().manual_seed(a.seed+rank))
 if rank==0:
  (out/'config.json').write_text(json.dumps({'args':vars(a),'contract':contract},indent=2))
  snapshots=out/'runtime_source'; snapshots.mkdir(exist_ok=True)
  for name in scripts: shutil.copyfile(here/name,snapshots/name)
  print('LONG_SPARSE_INITIALIZED '+json.dumps({'phase':a.phase,'source_step':source_meta['step'],'step':step,
   'train':len(train),'val':len(val),'frames':477,'chunks':40,'history':'predicted_only','attention':contract['attention']}),flush=True)
 writer=ThreadPoolExecutor(max_workers=1) if rank==0 else None; pending=[]; uploaded=[]
 spool=Path('/tmp')/('wan22_sparse_long_'+out.name)
 def upload(w,s,saved_step):
  for src,dst in ((w,out/w.name),(s,out/'trainer_state_latest.pt')):
   temp=out/(dst.name+f'.pending-{saved_step}'); shutil.copyfile(src,temp); os.replace(temp,dst)
  print(f'PFS_CHECKPOINT_COMPLETE phase={a.phase} step={saved_step}',flush=True)
  uploaded.append((w,s))
  if len(uploaded)>2:
   for old in uploaded.pop(0):
    assert old.parent==spool; old.unlink()
 def save(next_epoch,next_cursor):
  states=[None]*world if rank==0 else None; dist.gather_object(rng(),states,dst=0,group=control)
  if rank==0:
   for future in pending:
    if future.done(): future.result()
   pending[:]=[f for f in pending if not f.done()]
   if len(pending)>=2: pending.pop(0).result()
   spool.mkdir(exist_ok=True)
   if shutil.disk_usage(spool).free<6*2**30: raise OSError('Insufficient local checkpoint space')
   weights={n:cpu(v).contiguous() for n,v in module.state_dict().items() if n in names}
   wp=spool/f'step-{step}.safetensors'; sp=spool/f'trainer_state_step-{step}.pt'
   if wp.exists() or (out/wp.name).exists(): raise FileExistsError('Refuse checkpoint overwrite')
   save_file(weights,str(wp),metadata={'contract':json.dumps(contract),'step':str(step)})
   state={'weights':weights,'optimizer':cpu(optimizer.state_dict()),'scheduler':scheduler.state_dict(),
    'step':step,'epoch':next_epoch,'next_batch':next_cursor,'rng':states,'contract':contract,
    'teacher_ema':cpu(ema) if ema is not None else None}
   torch.save(state,sp)
   link=spool/'latest.pending'; link.symlink_to(sp.name); os.replace(link,spool/'trainer_state_latest.pt')
   pending.append(writer.submit(upload,wp,sp,step))
   print(f'LOCAL_CHECKPOINT_COMPLETE phase={a.phase} step={step} state={sp}',flush=True)
  dist.barrier(group=control)
 @torch.no_grad()
 def evaluate(limit=None):
  module.eval(); sums=torch.zeros(2,3,device=device,dtype=torch.float64)
  for i in range(rank,min(len(val),limit or len(val)),world):
   batch=[v[None].to(device) for v in val[i]]
   for task,prompt in enumerate(TASK_PROMPTS):
    with torch.autocast('cuda',dtype=torch.bfloat16): pred=module.rollout(batch[0],prompt)
    data,error=coordinate_loss(pred,*batch[1:],task)
    sums[task]+=torch.stack([data.double(),error.double(),data.new_tensor(1).double()])
  dist.all_reduce(sums)
  if rank==0:
   for task,prompt in enumerate(TASK_PROMPTS):
    row={'step':step,'phase':a.phase,'prompt':prompt,'mse':float(sums[task,0]/sums[task,2]),
     'error':float(sums[task,1]/sums[task,2]),'unit':'pixels' if task==0 else 'actor_height',
     'n':int(sums[task,2]),'scope':'long477_clip_holdout_predicted_history_only'}
    with (out/'evaluation.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print('EVALUATION '+json.dumps(row),flush=True)
  module.train()
 if not a.skip_initial_eval: evaluate()
 start=time.perf_counter(); initial_step=step
 cap=min(a.max_steps,step+a.run_steps) if a.run_steps else a.max_steps
 while step<cap:
  sampler.set_epoch(epoch)
  for position,batch in enumerate(loader):
   if position<cursor: continue
   source,xy,xyz,mask=[v.to(device,non_blocking=True) for v in batch]
   task=(rank+step)%2; prompt=TASK_PROMPTS[task]; coords=xy if task==0 else xyz
   gt_groups=pack_coordinate_groups(coords,mask if task==0 else None)
   valid=[bool(mask[:,slice(*chunk_raw_span(o))].any()) if task==0 else True for o in range(0,120,3)]
   valid_count=sum(valid); distill_count=sum(v and o>=a.loss_start_frame for v,o in zip(valid,range(0,120,3)))
   if not valid_count or (ema is not None and not distill_count): raise ValueError('No supervised rollout chunks')
   optimizer.zero_grad(set_to_none=True); module.train()
   student_cache=module.causal_controller.new_kv_cache()
   teacher_cache=module.causal_controller.new_kv_cache() if ema is not None else None
   history=source.new_zeros(1,3,4,24,3); sums=torch.zeros(3,device=device)
   for chunk,offset in enumerate(range(0,120,3)):
    raw_start,raw_end=chunk_raw_span(offset)
    if ema is not None:
     with torch.autocast('cuda',dtype=torch.bfloat16):
      teacher=clean_teacher_forward(module,source,history,prompt,offset,teacher_cache,gt_groups,ema,named)
    sync=nullcontext() if chunk==39 else ddp.no_sync()
    with sync:
     with torch.autocast('cuda',dtype=torch.bfloat16):
      pred,groups,reg=ddp(source[:,:,offset:offset+3],history,prompt,offset,student_cache)
     data,error=coordinate_loss(pred,xy[:,raw_start:raw_end],xyz[:,raw_start:raw_end],mask[:,raw_start:raw_end],task)
     distill=pred.float().sum()*0.
     if ema is not None and valid[chunk] and offset>=a.loss_start_frame:
      distill=coordinate_loss(pred,teacher,teacher,mask[:,raw_start:raw_end],task)[0]
     loss=data/valid_count+.001*reg/40
     if ema is not None: loss=loss+a.distill_weight*distill/distill_count
     if not torch.isfinite(loss): raise FloatingPointError(f'Invalid loss offset={offset}')
     loss.backward()
    sums+=torch.stack([data.detach(),distill.detach(),reg.detach()])
    history=feedback_groups(groups,task,offset); detach_kv_cache(student_cache); module.task_router.clear()
   del student_cache,teacher_cache,history,source,gt_groups
   norm=torch.nn.utils.clip_grad_norm_([q for _,q in named],1.,error_if_nonfinite=True)
   optimizer.step(); scheduler.step()
   if ema is not None: update_ema_(ema,named,a.ema_decay)
   step+=1
   record=torch.tensor([task,float(sums[0]/valid_count),float(sums[1]/max(1,distill_count)),float(sums[2]/40),float(norm)],device=device)
   gathered=[torch.empty_like(record) for _ in range(world)]; dist.all_gather(gathered,record)
   if rank==0:
    elapsed=time.perf_counter()-start
    rows=[{'step':step,'task':int(r[0]),'coordinate_mse':float(r[1]),'distill_mse':float(r[2]),
     'router_reg':float(r[3]),'grad_norm':float(r[4]),'elapsed':elapsed} for r in gathered]
    with (out/'train_metrics.jsonl').open('a') as f:
     for row in rows: f.write(json.dumps(row)+'\n')
    print(f'TRAIN phase={a.phase} step={step}/{a.max_steps} mse={sum(r["coordinate_mse"] for r in rows)/world:.7f} distill={sum(r["distill_mse"] for r in rows)/world:.7f} sec_per_step={elapsed/(step-initial_step):.2f} max_mem_gib={torch.cuda.max_memory_allocated(device)/2**30:.2f}',flush=True)
   next_epoch,next_cursor=(epoch+1,0) if position+1==len(loader) else (epoch,position+1)
   if step%a.eval_steps==0 or step==cap: evaluate(8 if a.run_steps else None)
   if step%a.save_steps==0 or step==cap: save(next_epoch,next_cursor)
   if step>=cap: break
  epoch+=1; cursor=0
 if rank==0:
  writer.shutdown(wait=True)
  for future in pending: future.result()
  (out/('TRAINING_COMPLETE' if step==a.max_steps else 'SMOKE_COMPLETE')).write_text(json.dumps({'step':step,'phase':a.phase,'contract':contract},indent=2))
 dist.barrier(group=control); dist.destroy_process_group()

if __name__=='__main__': main()
