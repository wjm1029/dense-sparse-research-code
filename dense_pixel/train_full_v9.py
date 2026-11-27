"""Bounded-memory training over the complete qualified video dataset.

Frozen native backbone; task-specific FP32 pixel heads; train-only regularized
feature transforms. Validation is reporting-only and unseen scenes are tested
after the fixed training budget. Resumable sampler, optimizer and checkpoints.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import random
import time
import torch
from torch import nn
import run_probe as v1
from full_data_v9 import RUN,atomic,emit
from cache_full_v9 import paths
from train_multiclip_v8 import Branch,pack_regular,save_atomic,infer,folded_infer
from decode_unit_v5 import pack,measure

OUT=RUN/'training'


def rows(split):return [json.loads(x) for x in (RUN/f'{split}.jsonl').read_text().splitlines()]


def load_case(row,task):
    data=torch.load(paths(row,task)[0],map_location='cpu',weights_only=False)
    assert data['row']['sample']==row['sample']
    assert data['checkpoint_sha256']=='51ee931e662cf59347f96b2f9b29136f2c62e8db77c958af6a2e225ec57d6d97'
    return data


@torch.no_grad()
def fit_stats(train,task,fingerprint):
    path=OUT/f'{task}_geometry.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['manifest_sha256']==fingerprint;return saved['stats']
    sums={}
    for name,o in [('first',1024),('regular',4096)]:
        sums[name]=dict(n=0,sx=torch.zeros(3072,device='cuda',dtype=torch.float64),sy=torch.zeros(o,device='cuda',dtype=torch.float64),
            xx=torch.zeros(3072,3072,device='cuda',dtype=torch.float64),xy=torch.zeros(3072,o,device='cuda',dtype=torch.float64))
    for begin in range(0,len(train),64):
        batch=[torch.load(paths(row,task)[1],map_location='cpu',weights_only=False) for row in train[begin:begin+64]]
        assert all(b['split']=='train' for b in batch)
        for name,s in sums.items():
            x=torch.cat([b[name]['x'] for b in batch]).cuda().double();y=torch.cat([b[name]['y'] for b in batch]).cuda().double()
            s['n']+=len(x);s['sx']+=x.sum(0);s['sy']+=y.sum(0);s['xx']+=x.T@x;s['xy']+=x.T@y
        if begin%512==0:emit('statistics_progress',task=task,clips=min(begin+64,len(train)),total=len(train))
    stats={}
    for name,s in sums.items():
        mean=s['sx']/s['n'];my=s['sy']/s['n'];cov=(s['xx']-s['sx'][:,None]*mean[None])/s['n']
        cross=(s['xy']-s['sx'][:,None]*my[None])/s['n'];cov=(cov+cov.T)/2
        ridge=float(cov.diagonal().mean())*.001;e,v=torch.linalg.eigh(cov);assert e.min()>-1e-8*e.max()
        denominator=e.clamp_min(0)+ridge;projection=v/denominator.sqrt()[None]
        weight=v@((v.T@cross)/denominator[:,None]);bias=my-mean@weight
        receipt=dict(patches=s['n'],clips=len(train),ridge=ridge,minimum_eigenvalue=float(e.min()),maximum_eigenvalue=float(e.max()))
        stats[name]=dict(mean=mean.cpu(),projection=projection.cpu(),base_weight=weight.float().cpu(),base_bias=bias.float().cpu(),receipt=receipt)
        emit('statistics_ready',task=task,branch=name,**receipt)
    save_atomic(dict(stats=stats,manifest_sha256=fingerprint),path);return stats


@torch.no_grad()
def prepare_case(heads,case):
    bank={}
    for name in ['first','regular']:
        first=name=='first';x=(case['features'][:1] if first else case['features'][1:]).reshape(-1,3072)
        y=(pack(case['targets'][:1]) if first else pack_regular(case['targets'])).cuda().float()
        z,base=heads[name].transform(x);bank[name]=(z,(y-base)/heads[name].scale)
    return bank


def stratified_probe(cases,limit):
    groups={}
    for row in cases:groups.setdefault(row['scene'],[]).append(row)
    for values in groups.values():values.sort(key=lambda r:hashlib.sha256(('v9-probe/'+r['sample']).encode()).hexdigest())
    result=[];index=0
    while len(result)<min(limit,len(cases)):
        for scene in sorted(groups):
            if index<len(groups[scene]):result.append(groups[scene][index])
            if len(result)==min(limit,len(cases)):break
        index+=1
    return result


@torch.no_grad()
def evaluate(heads,cases,task,step,label):
    heads.eval();records=[]
    for i,row in enumerate(cases):
        case=load_case(row,task);pred=infer(heads,case['features']);gt=case['targets'].cuda()
        record=dict(sample=row['sample'],clamped=v1.metrics(pred,gt,task),unclamped=measure(pred,gt,task))
        if case['baseline'] is not None:record['official_vae']=v1.metrics(case['baseline'],case['targets'],task)
        records.append(record)
        if i==0:save_atomic(dict(prediction=pred.cpu(),row=row,step=step,metrics=record),OUT/f'{task}_{label}_preview.pt')
    keys=['clamped','unclamped']+(['official_vae'] if all('official_vae' in r for r in records) else [])
    means={kind:{k:sum(r[kind][k] for r in records)/len(records) for k in records[0][kind]} for kind in keys}
    report=dict(count=len(records),mean=means,cases=records)
    atomic(OUT/f'{task}_{label}_{step}.json',report);emit('evaluation',task=task,step=step,split=label,count=len(records),mean=means)
    heads.train();return report


def main():
    global TASK
    parser=argparse.ArgumentParser();parser.add_argument('--task',required=True,choices=['depth','matte'])
    parser.add_argument('--steps',type=int,default=100000);a=parser.parse_args();TASK=a.task
    torch.set_num_threads(4);torch.manual_seed(20261130);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest');rng=random.Random(20261130)
    assert torch.cuda.mem_get_info()[0]>20*2**30
    data_ready=json.loads((RUN/'DATA_READY.json').read_text())
    train,validation,test=rows('train'),rows('validation'),rows('test')
    fingerprint=hashlib.sha256((RUN/'train.jsonl').read_bytes()).hexdigest()
    val_probe=stratified_probe(validation,64);train_probe=stratified_probe(train,16)
    request=dict(task=TASK,steps=a.steps,train_clips=len(train),validation_clips=len(validation),test_clips=len(test),
        manifest_sha256=fingerprint,data_summary=data_ready,backbone_frozen=True,
        architecture='Independent first-frame and regular-frame ridge + 3072->2048->1024/4096 GELU residual heads',
        initialization='Global training-only ridge; zero residual output, no single-clip or v8 trained weights',
        transform='Regularized whitening from deterministic 32 first/128 regular patches of EVERY training clip; ridge 0.001 mean variance',
        precision='BF16 native feature cache, FP64 statistics/transforms, FP32 pixel MLP',
        loss='Uniform pixel MSE; first/regular loss weights 1/81 and 80/81, residual scale 0.05',
        optimizer='AdamW lr 3e-4 warmup100 cosine to3e-6, wd0.001',
        sampling='16-clip GPU bank, shuffled exhaustive clip cycles, replace one clip every8 steps; each batch samples8 bank clips',
        batch=dict(first=512,regular=4096),checkpoint_every=5000,
        evaluation='16 train +64 stratified validation clips every5000; all validation and unseen-scene test after fixed100000; no holdout checkpoint selection',
        code_sha256=v1.sha(__file__))
    atomic(OUT/f'{TASK}_REQUEST.json',request)
    stats=fit_stats(train,TASK,fingerprint);heads=nn.ModuleDict({n:Branch(s,2048,.05) for n,s in stats.items()}).cuda()
    opt=torch.optim.AdamW(heads.parameters(),lr=3e-4,weight_decay=.001)
    order=list(range(len(train)));rng.shuffle(order);cursor=0;start=0
    latest=OUT/f'{TASK}_latest.pt';saved=None
    if latest.exists():
        saved=torch.load(latest,map_location='cuda',weights_only=False)
        assert saved['request']['manifest_sha256']==fingerprint and saved['request']['steps']==a.steps
        heads.load_state_dict(saved['heads']);opt.load_state_dict(saved['optimizer']);start=saved['step']
        order=saved['order'];cursor=saved['cursor'];rng.setstate(saved['sampler_rng'])
    def next_index():
        nonlocal cursor,order
        if cursor==len(order):rng.shuffle(order);cursor=0
        value=order[cursor];cursor+=1;return value
    bank_ids=saved['bank_ids'] if saved else [next_index() for _ in range(16)]
    bank=[prepare_case(heads,load_case(train[i],TASK)) for i in bank_ids]
    if saved:
        torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    else:
        evaluate(heads,train_probe,TASK,0,'train_probe');evaluate(heads,val_probe,TASK,0,'validation_probe')
    prefetch_id=saved['prefetch_id'] if saved else next_index()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(load_case,train[prefetch_id],TASK)
        emit('training_started',task=TASK,start=start,steps=a.steps,parameters=sum(p.numel() for p in heads.parameters()),train_clips=len(train))
        for step in range(start+1,a.steps+1):
            if step%8==0:
                slot=(step//8-1)%len(bank);case=future.result();bank[slot]=prepare_case(heads,case);bank_ids[slot]=prefetch_id
                del case;prefetch_id=next_index();future=pool.submit(load_case,train[prefetch_id],TASK)
            opt.zero_grad(set_to_none=True);selected=rng.sample(range(len(bank)),8);losses={}
            for name,per_clip,weight in [('first',64,1/81),('regular',512,80/81)]:
                xs=[];ys=[]
                for index in selected:
                    z,y=bank[index][name];j=torch.randint(len(z),(per_clip,),device='cuda');xs.append(z[j]);ys.append(y[j])
                prediction=heads[name].net(torch.cat(xs));loss=(prediction-torch.cat(ys)).square().mean()
                assert torch.isfinite(loss);(loss*weight).backward();losses[name]=float(loss.detach())*.05**2
            norm=nn.utils.clip_grad_norm_(heads.parameters(),10.);assert torch.isfinite(norm)
            lr=(3e-6+(3e-4-3e-6)*.5*(1+math.cos(math.pi*step/a.steps)))*min(step/100,1)
            for group in opt.param_groups:group['lr']=lr
            opt.step()
            if step==start+1 or step%100==0:emit('train_step',task=TASK,step=step,pixel_mse=losses,lr=lr,gradient_norm=float(norm),gpu_gb=torch.cuda.memory_allocated()/2**30)
            if step%5000==0 or step==a.steps:
                train_report=evaluate(heads,train_probe,TASK,step,'train_probe');val_report=evaluate(heads,val_probe,TASK,step,'validation_probe')
                save_atomic(dict(step=step,heads=heads.state_dict(),optimizer=opt.state_dict(),request=request,
                    order=order,cursor=cursor,bank_ids=bank_ids,prefetch_id=prefetch_id,sampler_rng=rng.getstate(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),latest)
                exported={n:h.export() for n,h in heads.items()}
                with torch.no_grad():
                    case=load_case(val_probe[0],TASK);direct=infer(heads,case['features']);folded=folded_infer(exported,case['features'])
                    delta=float((direct-folded).abs().max());assert delta<.001,delta
                package=dict(heads=exported,task=TASK,step=step,request=request,folded_max_difference=delta)
                save_atomic(package,OUT/f'{TASK}_head_latest.pt')
                if step%25000==0 or step==a.steps:save_atomic(package,OUT/f'{TASK}_head_{step}.pt')
                emit('checkpoint_saved',task=TASK,step=step,folded_max_difference=delta)
    final_val=evaluate(heads,validation,TASK,a.steps,'validation_full')
    final_test=evaluate(heads,test,TASK,a.steps,'test_full')
    atomic(OUT/f'{TASK}_COMPLETE.json',dict(step=a.steps,validation=final_val['mean'],test=final_test['mean'],request=request))
    emit('training_complete',task=TASK,step=a.steps)


if __name__=='__main__':
    TASK='startup';OUT.mkdir(parents=True,exist_ok=True)
    try:main()
    except BaseException as error:
        atomic(OUT/f'{TASK}_FAILED.json',dict(error=repr(error)));emit('training_failed',task=TASK,error=repr(error));raise
