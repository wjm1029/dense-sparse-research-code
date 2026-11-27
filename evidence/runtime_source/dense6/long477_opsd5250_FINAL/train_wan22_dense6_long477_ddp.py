"""Four-GPU Dense6 477-frame self-forcing, paired checkpoints and held-out rollout."""
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
from torch.utils.data import DataLoader
from safetensors import safe_open
from safetensors.torch import save_file
from wan22_dense6_long477 import LongDenseModule,LongDenseDataset
from wan22_dense6_dispersion import SameTaskSampler
from wan22_cache_integrity import sha256_file as sha
from wan22_dense6_checkpoint import load_dense6_checkpoint,REQUIRED_PROVENANCE_KEYS,OPTIONAL_PROVENANCE_KEYS
from wan22_ar_self_forcing import detach_kv_cache

def cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    return value

def rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--extra_cache',required=True); p.add_argument('--output_dir',required=True)
    p.add_argument('--source',default='/pfs/mayuema/dense/wan/outputs/wan22_dense6_ar_tf_dispersion_43500_20261013/step-43500.safetensors')
    p.add_argument('--model_dir',default='/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B')
    p.add_argument('--max_steps',type=int,default=4800); p.add_argument('--save_steps',type=int,default=25)
    p.add_argument('--eval_steps',type=int,default=400); p.add_argument('--resume')
    p.add_argument('--smoke',action='store_true'); p.add_argument('--seed',type=int,default=7)
    a=p.parse_args(); rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    assert world==4 and a.max_steps>0
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(hours=2))
    control=dist.new_group(backend='gloo',timeout=timedelta(hours=2))
    random.seed(a.seed+rank); np.random.seed(a.seed+rank); torch.manual_seed(a.seed+rank); torch.cuda.manual_seed_all(a.seed+rank)
    with safe_open(a.source,framework='pt',device='cpu') as f: meta=f.metadata()
    assert meta['step']=='43500' and meta['causal_frames_per_chunk']=='3' and meta['causal_num_frames']=='21'
    assert meta['dispersion_weight']=='0.01' and meta['dispersion_tau']=='1.0'
    prompts=json.loads(meta['output_prompt_texts'])
    out=Path(a.output_dir)
    if rank==0:
        if out.exists() and any(out.iterdir()) and not a.resume: raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True)
    dist.barrier(group=control)
    train=LongDenseDataset(a.extra_cache,prompts,smoke=a.smoke)
    val=LongDenseDataset(a.extra_cache,prompts,True,a.smoke)
    here=Path(__file__).resolve().parent
    scripts=('train_wan22_dense6_long477_ddp.py','wan22_dense6_long477.py','wan22_dense6_dispersion.py',
      'train_wan22_dense6_dispersion_ddp.py','wan22_ar_self_forcing.py','wan22_ar_teacher_forcing.py',
      'wan22_causal_attention.py','wan22_dense6_task_router.py','wan22_dense6_checkpoint.py','wan22_masked_loss.py')
    contract={'schema':'wan22_dense6_long477_self_forcing_dispersion_v1','max_steps':a.max_steps,'world_size':4,
      'source':a.source,'source_sha256':sha(a.source),'source_runtime_receipt':meta['runtime_source_patch_sha256'],
      'cache_manifests':train.identities,'prompts':prompts,'model_dir':a.model_dir,'seed':a.seed,
      'scripts':{s:sha(here/s) for s in scripts},'pipeline_sha256':sha('diffsynth/pipelines/wan_video.py'),
      'dit_sha256':sha('diffsynth/models/wan_video_dit.py'),'latent_shape':[48,120,30,52],
      'chunk':3,'chunks_per_update':40,'attention':'full_prefix','history':'previous_prediction_detached',
      'loss':'chunk_mean_latent_MSE + 0.001 router + 0.01 full_block9_chunk_feature_dispersion_tau1',
      'sampler':'six task cycle; four distinct clips; same-prompt feature pairs only',
      'lr':2e-5,'router_lr':2e-6,'warmup':20,'weight_decay':.01,
      'validation':'SHA256(dense6-long-v1:clip_id)%20==0; predicted-history; clip holdout, not actor/action disjoint',
      'train_clips':len(train.rows),'val_clips':len(val.rows),'smoke':a.smoke}
    receipt=Path('/pfs/mayuema/dense/wan/docs/wan22_dense6_ar_dispersion_20261013_runtime.json')
    assert sha(receipt)==meta['runtime_source_patch_sha256']
    inherited=json.loads(receipt.read_text())
    for name,checksum in inherited['source_hashes'].items(): assert sha(here/name)==checksum, name
    module=LongDenseModule(a.model_dir)
    module.prepare_condition_tables(prompts,device)
    module.pipe.dit.to(device); module.task_router.to(device); module.pipe.device=device
    provenance={k:meta[k] for k in REQUIRED_PROVENANCE_KEYS|OPTIONAL_PROVENANCE_KEYS if k in meta}
    load_dense6_checkpoint(module,a.source,expected_provenance=provenance,lora_rank=32,
      lora_targets='q,k,v,o,ffn.0,ffn.2',router_hidden_dim=512,expected_router_mode='learned')
    named=[(n,q) for n,q in module.named_parameters() if q.requires_grad]
    names={n for n,_ in named}|{n for n in module.state_dict() if n.startswith('task_router.')}
    groups=[{'params':[q for n,q in named if n.startswith('pipe.dit.')],'lr':2e-5},
            {'params':[q for n,q in named if n.startswith('task_router.')],'lr':2e-6}]
    assert {id(q) for g in groups for q in g['params']}=={id(q) for _,q in named}
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.,(s+1)/20))
    step=epoch=cursor=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert state['contract']==contract and set(state['weights'])==names
        module.load_state_dict(state['weights'],strict=False)
        assert torch.equal(module.context_table.cpu(),state['contexts'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        step,epoch,cursor=state['step'],state['epoch'],state['next_batch']
        r=state['rng'][rank]; random.setstate(r['python']); np.random.set_state(r['numpy'])
        torch.set_rng_state(r['torch']); torch.cuda.set_rng_state(r['cuda']); del state
    ddp=DDP(module,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    sampler=SameTaskSampler(len(train.rows),num_replicas=world,rank=rank,seed=a.seed)
    loader=DataLoader(train,batch_size=1,sampler=sampler,num_workers=2,pin_memory=True,
      generator=torch.Generator().manual_seed(a.seed+rank))
    if rank==0:
        (out/'config.json').write_text(json.dumps({'args':vars(a),'contract':contract},indent=2))
        snapshot=out/'runtime_source'; snapshot.mkdir(exist_ok=True)
        for s in scripts: shutil.copyfile(here/s,snapshot/s)
        print('DENSE6_LONG_INITIALIZED '+json.dumps({'step':step,'train_clips':len(train.rows),'val_clips':len(val.rows),
          'inherited':'Dense6 AR43500 LoRA/router/history','fresh_stage_optimizer':not bool(a.resume),'smoke':a.smoke}),flush=True)
    spool=Path('/tmp')/('wan22_dense6_long_'+out.name)
    writer=ThreadPoolExecutor(max_workers=1) if rank==0 else None; pending=[]; uploaded=[]
    def upload(w,s,saved_step):
        for src,dst in ((w,out/w.name),(s,out/'trainer_state_latest.pt')):
            temp=out/(dst.name+f'.pending-{saved_step}'); shutil.copyfile(src,temp); os.replace(temp,dst)
        print(f'PFS_CHECKPOINT_COMPLETE step={saved_step}',flush=True)
        uploaded.append((w,s))
        if len(uploaded)>2:
            for old in uploaded.pop(0): old.unlink()
    def save(next_epoch,next_cursor):
        states=[None]*world if rank==0 else None
        dist.gather_object(rng(),states,dst=0,group=control)
        if rank==0:
            for future in pending:
                if future.done(): future.result()
            pending[:]=[f for f in pending if not f.done()]
            if len(pending)>=2: pending.pop(0).result()
            spool.mkdir(exist_ok=True)
            assert shutil.disk_usage(spool).free>6*2**30,'Local checkpoint guard'
            weights={n:cpu(v).contiguous() for n,v in module.state_dict().items() if n in names}
            w=spool/f'step-{step}.safetensors'; s=spool/f'trainer_state_step-{step}.pt'
            assert not w.exists() and not (out/w.name).exists()
            save_file({**weights,'condition_contexts':module.context_table.cpu().contiguous()},str(w),
              metadata={'schema':contract['schema'],'step':str(step),'contract':json.dumps(contract)})
            torch.save({'weights':weights,'contexts':cpu(module.context_table),'optimizer':cpu(optimizer.state_dict()),
              'scheduler':scheduler.state_dict(),'step':step,'epoch':next_epoch,'next_batch':next_cursor,
              'rng':states,'contract':contract},s)
            link=spool/'latest.pending'; link.symlink_to(s.name); os.replace(link,spool/'trainer_state_latest.pt')
            pending.append(writer.submit(upload,w,s,step))
            print(f'LOCAL_CHECKPOINT_COMPLETE step={step}',flush=True)
        dist.barrier(group=control)
    @torch.no_grad()
    def evaluate():
        module.eval(); values=torch.zeros(6,2,device=device,dtype=torch.float64)
        for i in range(rank,len(val),world):
            rgb,target,task,prompt,_=val[i]
            rgb=rgb[None].to(device); target=target[None].to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=module.rollout(rgb,torch.tensor([task],device=device),torch.tensor([prompt],device=device))
            values[task,0]+=(pred.float()-target.float()).square().mean().double(); values[task,1]+=1
        dist.all_reduce(values)
        if rank==0:
            for task,(loss,count) in enumerate(values.tolist()):
                row={'step':step,'task':task,'latent_mse':loss/count,'n':int(count),'scope':'smoke_same4' if a.smoke else 'long477_clip_holdout_free_rollout'}
                with (out/'evaluation.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print('EVALUATION '+json.dumps(row),flush=True)
        module.train()
    # Smoke tests all six tasks then validates; formal initial held-out baseline.
    if not a.smoke and not a.resume: evaluate()
    start=time.monotonic(); initial_step=step
    while step<a.max_steps:
        sampler.set_epoch(epoch)
        for position,batch in enumerate(loader):
            if position<cursor: continue
            rgb,target,task,prompt,_=batch
            rgb,target,task,prompt=[x.to(device,non_blocking=True) for x in (rgb,target,task,prompt)]
            module.train(); optimizer.zero_grad(set_to_none=True)
            caches=module.causal_controller.new_kv_cache(); history=torch.zeros_like(rgb[:,:,:3])
            sums=torch.zeros(3,device=device)
            for chunk,offset in enumerate(range(0,120,3)):
                with (nullcontext() if chunk==39 else ddp.no_sync()):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred,reg,dispersion=ddp(rgb[:,:,offset:offset+3],history,task,prompt,offset,caches)
                        mse=(pred.float()-target[:,:,offset:offset+3].float()).square().mean()
                        loss=(mse+.001*reg+.01*dispersion)/40
                    if not torch.isfinite(loss): raise FloatingPointError((step,offset))
                    loss.backward()
                sums+=torch.stack([mse.detach(),reg.detach(),dispersion.detach()])
                history=pred.detach(); detach_kv_cache(caches); module.task_router.clear()
            del caches,history,pred,rgb,target
            norm=torch.nn.utils.clip_grad_norm_([q for _,q in named],1.,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step+=1
            record=torch.cat((sums/40,norm.detach()[None]))
            dist.all_reduce(record); record/=world
            if rank==0:
                row={'step':step,'task':int(task[0]),'mse':float(record[0]),'router':float(record[1]),
                  'dispersion':float(record[2]),'grad_norm':float(record[3]),'elapsed':time.monotonic()-start}
                with (out/'train_metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print(f'TRAIN step={step}/{a.max_steps} task={int(task[0])} mse={record[0]:.7f} dispersion={record[2]:.6f} sec_per_step={row["elapsed"]/(step-initial_step):.2f} peak_gib={torch.cuda.max_memory_allocated()/2**30:.2f}',flush=True)
            next_epoch,next_cursor=(epoch+1,0) if position+1==len(loader) else (epoch,position+1)
            if step%a.eval_steps==0 or step==a.max_steps: evaluate()
            if step%a.save_steps==0 or step==a.max_steps: save(next_epoch,next_cursor)
            if step==a.max_steps: break
        epoch+=1; cursor=0
    if rank==0:
        writer.shutdown(wait=True)
        for future in pending: future.result()
        (out/('SMOKE_COMPLETE' if a.smoke else 'TRAINING_COMPLETE')).write_text(json.dumps({'step':step,'contract':contract},indent=2))
    dist.barrier(group=control); dist.destroy_process_group()

if __name__=='__main__': main()
