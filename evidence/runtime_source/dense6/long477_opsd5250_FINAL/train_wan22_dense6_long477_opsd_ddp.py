"""Clean-data Dense6 long OPSD from native SF4800; student checkpoints + EMA resume."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
import json,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from wan22_dense6_long477 import LongDenseDataset
from wan22_dense6_long477_opsd import Dense6OPSDModule,teacher_forward
from wan22_dense6_dispersion import SameTaskSampler
from wan22_opsd_prefix_cache import update_ema_
from wan22_ar_self_forcing import detach_kv_cache
from wan22_cache_integrity import sha256_file as sha
from train_wan22_dense6_long477_ddp import cpu,rng

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output_dir',required=True)
    p.add_argument('--source',default='/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_self_forcing_dispersion_4800_20261017/step-4800.safetensors')
    p.add_argument('--extra_cache',default='/pfs/mayuema/dense/wan/cache/wan22_dense6_long477_extra_flow32_v1_20261017')
    p.add_argument('--audit_dir',default='/pfs/mayuema/dense/wan/outputs/long1079_texture_audit_20261018')
    p.add_argument('--model_dir',default='/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B')
    p.add_argument('--max_steps',type=int,default=5250); p.add_argument('--save_steps',type=int,default=100)
    p.add_argument('--eval_steps',type=int,default=500); p.add_argument('--resume'); p.add_argument('--run_steps',type=int)
    p.add_argument('--skip_initial_eval',action='store_true'); p.add_argument('--seed',type=int,default=19)
    a=p.parse_args(); rank,local,world=[int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE')]
    assert world==4 and a.max_steps>0
    torch.cuda.set_device(local); torch.set_num_threads(1); device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(hours=2))
    control=dist.new_group(backend='gloo',timeout=timedelta(hours=2))
    random.seed(a.seed+rank); np.random.seed(a.seed+rank); torch.manual_seed(a.seed+rank); torch.cuda.manual_seed_all(a.seed+rank)
    with safe_open(a.source,framework='pt',device='cpu') as f: meta=f.metadata()
    assert meta['schema']=='wan22_dense6_long477_self_forcing_dispersion_v1' and meta['step']=='4800'
    parent=json.loads(meta['contract']); prompts=parent['prompts']; here=Path(__file__).resolve().parent
    for name,checksum in parent['scripts'].items(): assert sha(here/name)==checksum,name
    assert sha('diffsynth/pipelines/wan_video.py')==parent['pipeline_sha256']
    assert sha('diffsynth/models/wan_video_dit.py')==parent['dit_sha256']
    audit=Path(a.audit_dir); assert (audit/'AUDIT_COMPLETE').is_file()
    rows=json.loads((audit/'CLIPS.json').read_text()); summary=json.loads((audit/'SUMMARY.json').read_text())
    assert len(rows)==1079 and len({r['id'] for r in rows})==1079
    assert not summary['cache_audit_errors'] and not summary['mapping_errors']
    assert all(r['status']=='checked' for r in rows)
    excluded={r['id'] for r in rows if r['broken_color_reference']}
    assert len(excluded)==426
    train=LongDenseDataset(a.extra_cache,prompts); val=LongDenseDataset(a.extra_cache,prompts,True)
    assert train.identities==parent['cache_manifests']
    assert {r['id'] for r in train.rows+val.rows}=={r['id'] for r in rows}
    train.rows=[r for r in train.rows if r['id'] not in excluded]
    val.rows=[r for r in val.rows if r['id'] not in excluded]
    assert len(train.rows)==626 and len(val.rows)==27
    scripts=tuple(dict.fromkeys((*parent['scripts'],'train_wan22_dense6_long477_opsd_ddp.py',
        'wan22_dense6_long477_opsd.py','wan22_opsd_prefix_cache.py','wan22_opsd_bounded_causal_attention.py')))
    contract={**parent,'schema':'wan22_dense6_long477_opsd_clean_v1','phase':'opsd_local',
        'max_steps':a.max_steps,'source':a.source,'source_sha256':sha(a.source),'seed':a.seed,
        'scripts':{s:sha(here/s) for s in scripts},'train_clips':626,'val_clips':27,
        'attention':'sink3_local12_absolute_positions','history':'previous_prediction_detached',
        'loss':'chunk_mean_latent_MSE + 0.001 router + 0.01 block9_dispersion + post_burnin_EMA_latent_MSE',
        'teacher_context':'delayed_one_chunk_clean_GT_KV_replay; same student current history',
        'ema_decay':.9999,'distill_weight':1.,'loss_start_frame':21,
        'audit_sha256':sha(audit/'CLIPS.json'),'audit_summary_sha256':sha(audit/'SUMMARY.json'),
        'exclusion':'broken_color_reference; remove clip across all six tasks from train and val',
        'train_ids':[r['id'] for r in train.rows],'val_ids':[r['id'] for r in val.rows],
        'excluded_ids':sorted(excluded),'smoke':False}
    out=Path(a.output_dir)
    if rank==0:
        if out.exists() and any(out.iterdir()) and not a.resume: raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True)
    dist.barrier(group=control)
    weights=load_file(a.source,device='cpu'); contexts=weights.pop('condition_contexts')
    module=Dense6OPSDModule(a.model_dir); module.pipe.text_encoder=None; module.pipe.tokenizer=None
    module.context_table=contexts.to(device); module.pipe.dit.to(device); module.task_router.to(device); module.pipe.device=device
    named=[(n,q) for n,q in module.named_parameters() if q.requires_grad]
    names={n for n,_ in named}|{n for n in module.state_dict() if n.startswith('task_router.')}
    assert set(weights)==names
    result=module.load_state_dict(weights,strict=False); assert not result.unexpected_keys
    del weights,contexts
    groups=[{'params':[q for n,q in named if n.startswith('pipe.dit.')],'lr':2e-5},
            {'params':[q for n,q in named if n.startswith('task_router.')],'lr':2e-6}]
    assert {id(q) for g in groups for q in g['params']}=={id(q) for _,q in named}
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.,(s+1)/20))
    ema={n:q.detach().clone() for n,q in named}; step=epoch=cursor=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert state['contract']==contract and set(state['weights'])==names
        assert set(state['teacher_ema'])==set(ema) and torch.equal(module.context_table.cpu(),state['contexts'])
        module.load_state_dict(state['weights'],strict=False)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        ema={n:state['teacher_ema'][n].to(device=q.device,dtype=q.dtype) for n,q in named}
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
        print(f'DENSE6_OPSD_INITIALIZED step={step} train=626 val=27 excluded=426 max_steps={a.max_steps}',flush=True)
    spool=Path('/tmp')/('wan22_dense6_opsd_'+out.name)
    writer=ThreadPoolExecutor(max_workers=1) if rank==0 else None; pending=[]; uploaded=[]
    def upload(w,s,saved_step):
        for src,dst in ((w,out/w.name),(s,out/'trainer_state_latest.pt')):
            temp=out/(dst.name+f'.pending-{saved_step}'); shutil.copyfile(src,temp); os.replace(temp,dst)
        print(f'PFS_CHECKPOINT_COMPLETE step={saved_step}',flush=True); uploaded.append((w,s))
        if len(uploaded)>2:
            for old in uploaded.pop(0): old.unlink()
    def save(next_epoch,next_cursor):
        states=[None]*world if rank==0 else None; dist.gather_object(rng(),states,dst=0,group=control)
        if rank==0:
            for future in pending:
                if future.done(): future.result()
            pending[:]=[f for f in pending if not f.done()]
            if len(pending)>=2: pending.pop(0).result()
            spool.mkdir(exist_ok=True); assert shutil.disk_usage(spool).free>8*2**30
            weights={n:cpu(v).contiguous() for n,v in module.state_dict().items() if n in names}
            w=spool/f'step-{step}.safetensors'; s=spool/f'trainer_state_step-{step}.pt'
            assert not w.exists() and not (out/w.name).exists()
            save_file({**weights,'condition_contexts':cpu(module.context_table).contiguous()},str(w),
                metadata={'schema':contract['schema'],'step':str(step),'contract':json.dumps(contract)})
            torch.save({'weights':weights,'contexts':cpu(module.context_table),'teacher_ema':cpu(ema),
                'optimizer':cpu(optimizer.state_dict()),'scheduler':scheduler.state_dict(),
                'step':step,'epoch':next_epoch,'next_batch':next_cursor,'rng':states,'contract':contract},s)
            pending.append(writer.submit(upload,w,s,step))
            print(f'LOCAL_CHECKPOINT_COMPLETE step={step}',flush=True)
        dist.barrier(group=control)
    @torch.no_grad()
    def evaluate(limit=None):
        module.eval(); values=torch.zeros(6,2,device=device,dtype=torch.float64)
        for i in range(rank,min(len(val),limit or len(val)),world):
            rgb,target,task,prompt,_=val[i]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=module.rollout(rgb[None].to(device),torch.tensor([task],device=device),torch.tensor([prompt],device=device))
            values[task,0]+=(pred.float()-target[None].to(device).float()).square().mean().double(); values[task,1]+=1
        dist.all_reduce(values)
        if rank==0:
            for task,(loss,count) in enumerate(values.tolist()):
                if count==0: continue
                row={'step':step,'task':task,'latent_mse':loss/count,'n':int(count),'scope':'clean27_local12_free_rollout' if limit is None else 'smoke_clean_subset'}
                with (out/'evaluation.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print('EVALUATION '+json.dumps(row),flush=True)
        module.train()
    if not a.skip_initial_eval: evaluate()
    cap=min(a.max_steps,step+a.run_steps) if a.run_steps else a.max_steps
    start=time.monotonic(); initial_step=step
    while step<cap:
        sampler.set_epoch(epoch)
        for position,batch in enumerate(loader):
            if position<cursor: continue
            rgb,target,task,prompt,_=batch
            rgb,target,task,prompt=[x.to(device,non_blocking=True) for x in (rgb,target,task,prompt)]
            module.train(); optimizer.zero_grad(set_to_none=True)
            caches=module.causal_controller.new_kv_cache(); teacher_cache=module.causal_controller.new_kv_cache()
            history=torch.zeros_like(rgb[:,:,:3]); sums=torch.zeros(4,device=device)
            for chunk,offset in enumerate(range(0,120,3)):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    teacher=teacher_forward(module,rgb,target,history,task,prompt,offset,teacher_cache,ema,named)
                with (nullcontext() if chunk==39 else ddp.no_sync()):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred,reg,dispersion=ddp(rgb[:,:,offset:offset+3],history,task,prompt,offset,caches)
                        mse=(pred.float()-target[:,:,offset:offset+3].float()).square().mean()
                        distilled=(pred.float()-teacher.float()).square().mean() if offset>=21 else pred.float().sum()*0
                        loss=(mse+.001*reg+.01*dispersion)/40+distilled/33
                    if not torch.isfinite(loss): raise FloatingPointError((step,offset))
                    loss.backward()
                sums+=torch.stack([mse.detach(),reg.detach(),dispersion.detach(),distilled.detach()])
                history=pred.detach(); detach_kv_cache(caches); module.task_router.clear()
            del caches,teacher_cache,history,pred,teacher,rgb,target
            norm=torch.nn.utils.clip_grad_norm_([q for _,q in named],1.,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); update_ema_(ema,named,.9999); step+=1
            record=torch.cat((sums/torch.tensor([40,40,40,33],device=device),norm.detach()[None]))
            dist.all_reduce(record); record/=world
            if rank==0:
                row={'step':step,'task':int(task[0]),'mse':float(record[0]),'router':float(record[1]),
                    'dispersion':float(record[2]),'distill':float(record[3]),'grad_norm':float(record[4]),'elapsed':time.monotonic()-start}
                with (out/'train_metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print(f'TRAIN step={step}/{a.max_steps} task={int(task[0])} mse={record[0]:.7f} distill={record[3]:.7f} dispersion={record[2]:.6f} sec_per_step={row["elapsed"]/(step-initial_step):.2f} peak_gib={torch.cuda.max_memory_allocated()/2**30:.2f}',flush=True)
            ne,nc=(epoch+1,0) if position+1==len(loader) else (epoch,position+1)
            if step%a.eval_steps==0 or step==cap: evaluate(24 if a.run_steps else None)
            if step%a.save_steps==0 or step==cap: save(ne,nc)
            if step==cap: break
        epoch+=1; cursor=0
    if rank==0:
        writer.shutdown(wait=True)
        for future in pending: future.result()
        (out/('TRAINING_COMPLETE' if step==a.max_steps else 'SMOKE_COMPLETE')).write_text(json.dumps({'step':step,'contract':contract},indent=2))
    dist.barrier(group=control); dist.destroy_process_group()

if __name__=='__main__': main()
