"""Complete new JTA bidirectional trainer: DDP, overfit, holdout eval, paired resume.

The new multi-person model has not been adapted to causal/AR/OPSD. Those stages
for the historical models are provided separately with their original sources.
"""
import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time
from package_runtime import setup, PACKAGE
setup()
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from jta_cache import sha, atomic_json, atomic_torch
from sparse_model import build_fresh_jta_model, make_optimizer
from sparse_loss import LossConfig, sparse_objective

SCHEMA='jta_multi_person_bidirectional_train_v1'


class CacheDataset:
    def __init__(self, root, split, verify_files=True):
        self.root=Path(root);self.config=json.loads((self.root/'cache_config.json').read_text())
        receipt=json.loads((self.root/'CACHE_COMPLETE.json').read_text())
        if receipt['manifest_sha256']!=sha(self.root/'manifest.jsonl') or receipt['cache_config_sha256']!=sha(self.root/'cache_config.json'):
            raise ValueError('Cache manifest/config integrity failure')
        self.rows=[json.loads(line) for line in (self.root/'manifest.jsonl').read_text().splitlines()]
        self.rows=[r for r in self.rows if r['split']==split]
        self.positive=[i for i,r in enumerate(self.rows) if not r['empty']]
        self.empty=[i for i,r in enumerate(self.rows) if r['empty']]
        self.verify_files=verify_files;self.checked=set()

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];path=self.root/row['cache_file']
        if self.verify_files and index not in self.checked:
            if sha(path)!=row['sha256']:raise ValueError(f'Cache payload changed: {path}')
            self.checked.add(index)
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['cache_config_sha256']!=sha(self.root/'cache_config.json') or p['row']['id']!=row['id']:
            raise ValueError('Wrong cached contract/identity')
        z=p['source_latent'];target=p['targets']
        if tuple(z.shape)!=(48,21,30,52) or z.dtype!=torch.bfloat16 or not torch.isfinite(z).all():
            raise ValueError('Wrong source latent')
        return z,target,row


def choose_index(dataset, step, rank, world, seed, empty_fraction, overfit=None):
    if overfit is not None:return overfit
    rng=random.Random(seed+(step*world+rank)*1000003)
    pool=dataset.empty if rng.random()<empty_fraction else dataset.positive
    if not pool:pool=dataset.positive or dataset.empty
    if not pool:raise ValueError('Empty dataset')
    return pool[rng.randrange(len(pool))]


def adapter_state(model):
    names={n for n,p in model.named_parameters() if p.requires_grad}|{
        n for n,_ in model.named_buffers() if n=='context' or n.startswith(('task_router.','sparse_head.'))}
    state={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n in names}
    if set(state)!=names:raise ValueError('Incomplete trainable/buffer state')
    return state


def load_adapter(model, weights):
    expected=set(adapter_state(model))
    if set(weights)!=expected:raise ValueError(f'Checkpoint keys differ: {set(weights)^expected}')
    result=model.load_state_dict(weights,strict=False)
    if result.unexpected_keys or expected.intersection(result.missing_keys):raise ValueError('Incomplete adapter load')


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def pose_metrics(pred,gt,assignment):
    """Matched-to-GT diagnostics, NOT official JTA AP/MOTA or true detection metrics."""
    qi,ni=assignment;count=pred['presence_logits'][0].sigmoid().ge(.5).sum(-1)
    result=dict(count_mae=float((count-gt['present'].sum(-1)).abs().float().mean()))
    if len(qi):
        visible=gt['visible'][:,ni].bool()&gt['present'][:,ni,None].bool()
        delta=(pred['xy'][0][:,qi]-gt['xy'][:,ni])*pred['xy'].new_tensor([832,480])
        distance=delta.square().sum(-1).sqrt()
        result.update(matched_visible_2d_distance_sum=float(distance[visible].sum()),visible_joints=int(visible.sum()))
        if 'xyz' in pred:
            valid=gt['valid_3d'][:,ni].bool()
            delta3=pred['xyz'][0][:,qi]-gt['xyz'][:,ni]
            result.update(matched_3d_distance_mm_sum=float(delta3.square().sum(-1).sqrt()[valid].sum()*1000),valid_3d_joints=int(valid.sum()))
    return result


@torch.no_grad()
def evaluate(model, dataset, indices, device, config, rank=0,world=1,group=None,
             overfit=False, output=None, step=0):
    model.eval();records=[]
    for position,index in enumerate(indices):
        if position%world!=rank:continue
        z,gt,row=dataset[index];z=z[None].to(device)
        gt={k:v.to(device) for k,v in gt.items()}
        for task in (0,1):
            with torch.autocast(device.type,dtype=torch.bfloat16):pred,gates=model(z,gt['intrinsic'][None],task,False)
            loss,terms,pairs=sparse_objective(pred,[gt],task,gates,config)
            records.append(dict(id=row['id'],task=task,loss=float(loss),**{k:float(v) for k,v in terms.items()},**pose_metrics(pred,gt,pairs[0])))
            model.task_router.clear()
            if overfit and output is not None and rank==0:
                atomic_torch(Path(output)/f'overfit_task{task}_latest.pt',dict(step=step,id=row['id'],
                    prediction=cpu_tree(pred),target=cpu_tree(gt),assignment=cpu_tree(pairs[0])))
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,records,group=group)
        records=[r for part in gathered for r in part]
    tasks={}
    for task in (0,1):
        selected=[r for r in records if r['task']==task]
        if not selected:continue
        summary=dict(clips=len(selected),loss=sum(r['loss'] for r in selected)/len(selected),
            count_mae=sum(r['count_mae'] for r in selected)/len(selected))
        visible=sum(r.get('visible_joints',0) for r in selected)
        if visible:summary['matched_visible_2d_error_px']=sum(r.get('matched_visible_2d_distance_sum',0) for r in selected)/visible
        valid=sum(r.get('valid_3d_joints',0) for r in selected)
        if valid:summary['matched_root_relative_mpjpe_mm']=sum(r.get('matched_3d_distance_mm_sum',0) for r in selected)/valid
        tasks[str(task)]=summary
    model.train()
    return dict(step=step,scope='same train clip overfit' if overfit else 'official sequence-disjoint validation',
                metric_note='GT-matched pose diagnostics; not official JTA AP/tracking metrics',tasks=tasks,records=records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--max-steps',type=int,default=30000);p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--eval-every',type=int,default=500);p.add_argument('--eval-clips',type=int,default=32)
    p.add_argument('--seed',type=int,default=20261127);p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--queries',type=int,default=128);p.add_argument('--empty-fraction',type=float,default=.1)
    p.add_argument('--overfit-id');p.add_argument('--resume',help='paired trainer_latest.pt, not weights-only')
    p.add_argument('--loss-config',help='JSON overrides for LossConfig');p.add_argument('--stop-file')
    a=p.parse_args()
    if min(a.max_steps,a.save_every,a.eval_every,a.eval_clips,a.queries)<1 or a.warmup<1 or not 0<=a.empty_fraction<=1:
        p.error('Invalid training settings')
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);device=torch.device('cuda',local);group=None
    if world>1:
        dist.init_process_group('nccl',timeout=datetime.timedelta(hours=2))
        group=dist.new_group(backend='gloo',timeout=datetime.timedelta(hours=2))
    def barrier():
        if world>1:dist.barrier(group=group)
    config=LossConfig(**(json.loads(Path(a.loss_config).read_text()) if a.loss_config else {}))
    out=Path(a.output);train=CacheDataset(a.cache,'train');val=CacheDataset(a.cache,'val')
    if not len(train):raise ValueError('No cached training records')
    overfit=None
    if a.overfit_id:
        candidates=[i for i,r in enumerate(train.rows) if r['id']==a.overfit_id]
        if len(candidates)!=1:raise ValueError('overfit-id must name a training crop')
        overfit=candidates[0]
    elif not len(val):raise ValueError('Non-overfit training requires official validation records')
    if any(r['track_count']>a.queries for r in train.rows+val.rows):raise ValueError('Not enough person query slots')
    if rank==0:
        if out.exists() and any(out.iterdir()) and not a.resume:raise FileExistsError('Use a new output directory or --resume')
        out.mkdir(parents=True,exist_ok=True)
    barrier()
    # Use identical initial model weights on all ranks; subsequent sampling is rank-specific.
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    code_paths=[Path(__file__),Path(__file__).with_name('jta_cache.py')]+[PACKAGE/'core'/name for name in
        ('task_gating.py','sparse_model.py','sparse_decode.py','sparse_loss.py','jta_tiles.py')]
    contract=dict(schema=SCHEMA,attention='bidirectional',initialization='official Wan, fresh LoRA/router/head; no old Sparse weights',
        world_size=world,seed=a.seed,queries=a.queries,max_steps=a.max_steps,warmup=a.warmup,
        empty_fraction=a.empty_fraction,overfit_id=a.overfit_id,loss=asdict(config),
        cache_config_sha256=sha(Path(a.cache)/'cache_config.json'),cache_manifest_sha256=sha(Path(a.cache)/'manifest.jsonl'),
        code_sha256={str(path.relative_to(PACKAGE)):sha(path) for path in code_paths})
    if a.resume and (out/'config.json').is_file():
        if json.loads((out/'config.json').read_text())['contract']!=contract:
            raise ValueError('Existing output contract differs; refuse to overwrite config')
    model=build_fresh_jta_model(str(Path(a.project_root)/'models/Wan2.2-TI2V-5B'),device,a.queries)
    optimizer=make_optimizer(model)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/a.warmup))
    start=0
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        if saved['contract']!=contract:raise ValueError('Resume contract mismatch')
        load_adapter(model,saved['weights']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start=saved['step'];restore_rng(saved['rng'][rank]);del saved
    if rank==0:atomic_json(out/'config.json',dict(args=vars(a),contract=contract,train_clips=len(train),val_clips=len(val)))
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True) if world>1 else model
    def save(step):
        state=rng_state();states=[None]*world if rank==0 else None
        if world>1:dist.gather_object(state,states,dst=0,group=group)
        else:states=[state]
        if rank==0:
            weights=adapter_state(model)
            atomic_torch(out/f'adapter_step_{step:06d}.pt',dict(schema=SCHEMA,step=step,contract=contract,weights=weights))
            atomic_torch(out/'trainer_latest.pt',dict(schema=SCHEMA,step=step,contract=contract,weights=weights,
                optimizer=cpu_tree(optimizer.state_dict()),scheduler=scheduler.state_dict(),rng=states))
        barrier()
    eval_dataset=train if overfit is not None else val
    # Deterministic validation subset covers disjoint sequences as evenly as possible.
    by_sequence={}
    for index,row in enumerate(val.rows):by_sequence.setdefault(row['sequence'],[]).append(index)
    eval_indices=[]
    if overfit is not None:eval_indices=[overfit]
    else:
        for offset in range(max(map(len,by_sequence.values()))):
            for seq in sorted(by_sequence):
                if offset<len(by_sequence[seq]):eval_indices.append(by_sequence[seq][offset])
                if len(eval_indices)>=a.eval_clips:break
            if len(eval_indices)>=a.eval_clips:break
    if not a.resume:save(0)
    begun=time.monotonic();completed=start;reason='max_steps'
    for step in range(start,a.max_steps):
        index=choose_index(train,step,rank,world,a.seed,a.empty_fraction,overfit)
        z,gt,row=train[index];z=z[None].to(device);gt={k:v.to(device) for k,v in gt.items()}
        task=(step+rank)%2;optimizer.zero_grad(set_to_none=True);ddp.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):pred,gates=ddp(z,gt['intrinsic'][None],task,True)
        loss,terms,_=sparse_objective(pred,[gt],task,gates,config)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();model.task_router.clear();completed=step+1
        # Free graph references before full model eval/checkpoint.
        record=dict(step=completed,rank=rank,task=task,id=row['id'],loss=float(loss.detach()),
            grad_norm=float(norm),elapsed_seconds=time.monotonic()-begun,**{k:float(v) for k,v in terms.items()})
        with (out/f'train_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        del loss,terms,pred,gates,z,gt
        if completed==start+1 or completed%10==0:
            if rank==0:print(json.dumps(record),flush=True);atomic_json(out/'STATUS.json',record)
        if completed%a.eval_every==0 or completed==a.max_steps:
            report=evaluate(model,eval_dataset,eval_indices,device,config,rank,world,group,overfit is not None,out,completed)
            if rank==0:atomic_json(out/f'validation_step_{completed:06d}.json',report);print(json.dumps(dict(event='validation',step=completed,tasks=report['tasks'])),flush=True)
        stop=torch.tensor(int(bool(a.stop_file and Path(a.stop_file).is_file())),device=device)
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if completed%a.save_every==0 or completed==a.max_steps or bool(stop):save(completed)
        if bool(stop):reason='stop_file';break
    if rank==0:atomic_json(out/('STOPPED.json' if reason=='stop_file' else 'TRAINING_COMPLETE.json'),dict(step=completed,reason=reason,scope='overfit' if a.overfit_id else 'official train/val'))
    barrier()
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
