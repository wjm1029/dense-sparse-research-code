"""Train separate multi-clip FP32 pixel heads on frozen native DiT features.

Training-only regularized whitening and ridge baselines; nonlinear token MLPs.
The held-out split is reporting-only, with no checkpoint/parameter selection.
The existing cache contract is BF16 features and scalar [-1,1] targets.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
import torch
from torch import nn
from PIL import Image, ImageDraw
import run_probe as v1
from decode_unit_v5 import pack, unpack, measure

SHA='51ee931e662cf59347f96b2f9b29136f2c62e8db77c958af6a2e225ec57d6d97'
RUN=v1.EXP/'multi_clip_v8'


def emit(event,**kw):
    d={'event':event,'task':TASK,'elapsed_seconds':time.monotonic()-START,**kw}
    print(json.dumps(d),flush=True)
    with (RUN/f'{TASK}_events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')


def save_atomic(value,path):
    temporary=path.with_suffix('.pending.pt');torch.save(value,temporary);os.replace(temporary,path)


def pack_regular(y):
    return y[1:].reshape(20,4,15,32,26,32).permute(0,2,4,1,3,5).reshape(-1,4096)


def video_from_patches(first,regular):
    return torch.cat([unpack(first,1)]+[unpack(y,4) for y in regular.reshape(20,390,4096)])


def load_cases(rows):
    out=[]
    for i,row in enumerate(rows):
        data=v1.load_sample(str(v1.cache_path(row)))
        assert data['checkpoint_sha256']==SHA and data['row']['sample']==row['sample']
        x=data['features'][TASK];y=data['targets'][TASK]
        assert x.shape==(21,15,26,3072) and x.dtype==torch.bfloat16
        assert y.shape==(81,480,832) and torch.isfinite(x).all() and torch.isfinite(y).all()
        assert torch.equal(unpack(pack(y[:1]),1),y[:1])
        assert torch.equal(unpack(pack_regular(y)[:390],4),y[1:5])
        out.append(dict(row=row,features=x,targets=y,baseline=data['baseline'][TASK]))
        emit('cache_loaded',sample=row['sample'],index=i+1,total=len(rows))
    return out


@torch.no_grad()
def geometry(data,first):
    name='first' if first else 'regular';d=3072;o=1024 if first else 4096
    sx=torch.zeros(d,dtype=torch.float64,device='cuda');sy=torch.zeros(o,dtype=torch.float64,device='cuda')
    xx=torch.zeros(d,d,dtype=torch.float64,device='cuda');xy=torch.zeros(d,o,dtype=torch.float64,device='cuda');n=0
    for case in data:
        x=(case['features'][:1] if first else case['features'][1:]).reshape(-1,d).cuda().double()
        y=(pack(case['targets'][:1]) if first else pack_regular(case['targets'])).cuda().double()
        sx+=x.sum(0);sy+=y.sum(0);xx+=x.T@x;xy+=x.T@y;n+=len(x)
    mean=sx/n;my=sy/n;cov=(xx-sx[:,None]*mean[None])/n;cross=(xy-sx[:,None]*my[None])/n
    cov=(cov+cov.T)/2
    # Same fixed ridge strength as v2; no held-out tuning. Soft whitening caps amplification.
    ridge=float(cov.diagonal().mean())*.001
    values,vectors=torch.linalg.eigh(cov)
    assert values.min()>-1e-8*values.max()
    projection=vectors/(values.clamp_min(0)+ridge).sqrt()[None]
    weight=vectors@((vectors.T@cross)/(values.clamp_min(0)+ridge)[:,None])
    bias=my-mean@weight
    receipt=dict(patches=n,feature_dimensions=d,ridge=ridge,
        eigenvalue_min=float(values.min()),eigenvalue_max=float(values.max()),
        unregularized_condition=float(values.max()/values.min().clamp_min(1e-30)),
        regularized_condition=float((values.max()+ridge)/(values.min().clamp_min(0)+ridge)),
        max_whitening_gain=float(1/(values.min().clamp_min(0)+ridge).sqrt()))
    emit('geometry_ready',branch=name,**receipt)
    return dict(mean=mean.cpu(),projection=projection.cpu(),base_weight=weight.float().cpu(),
                base_bias=bias.float().cpu(),receipt=receipt)


class Branch(nn.Module):
    def __init__(self,stats,width,scale):
        super().__init__();self.scale=scale
        for key in ['mean','projection','base_weight','base_bias']:self.register_buffer(key,stats[key].clone())
        output=stats['base_bias'].numel()
        self.net=nn.Sequential(nn.Linear(3072,width),nn.GELU(),nn.Linear(width,output))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def transform(self,x):
        # Precompute once with high precision; optimizer updates only FP32 MLPs.
        x=x.cuda().float()
        return ((x.double()-self.mean)@self.projection).float(),x@self.base_weight+self.base_bias

    def forward(self,x):
        z,base=self.transform(x)
        return base+self.scale*self.net(z)

    @torch.no_grad()
    def export(self):
        w=(self.projection@self.net[0].weight.T.double()).float()
        b=(self.net[0].bias.double()-self.mean@w.double()).float()
        return dict(base_weight=self.base_weight.cpu(),base_bias=self.base_bias.cpu(),
            residual_weight1=w.cpu(),residual_bias1=b.cpu(),
            residual_weight2=self.net[2].weight.cpu(),residual_bias2=self.net[2].bias.cpu(),scale=self.scale)


@torch.no_grad()
def infer(heads,features):
    f=features.cuda().to(torch.bfloat16).float()
    first=heads['first'](f[0].reshape(-1,3072))
    regular=torch.cat([heads['regular'](chunk) for chunk in f[1:].reshape(-1,3072).split(2048)])
    return video_from_patches(first,regular)


@torch.no_grad()
def folded_infer(export,features):
    f=features.cuda().to(torch.bfloat16).float();values=[]
    for name,x in [('first',f[0].reshape(-1,3072)),('regular',f[1:].reshape(-1,3072))]:
        w={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in export[name].items()}
        hidden=torch.nn.functional.gelu(x@w['residual_weight1']+w['residual_bias1'])
        values.append(x@w['base_weight']+w['base_bias']+w['scale']*torch.nn.functional.linear(hidden,w['residual_weight2'],w['residual_bias2']))
    return video_from_patches(*values)


def preview(case,pred,step,split):
    canvas=Image.new('RGB',(1248,792));draw=ImageDraw.Draw(canvas)
    for ri,fi in enumerate([0,40,80]):
        for ci,(title,value) in enumerate([('GT',case['targets']),('Original VAE',case['baseline']),('Multiclip pixel head',pred)]):
            arr=((value[fi].float().cpu().clamp(-1,1)+1)*127.5).round().byte().numpy()
            canvas.paste(Image.fromarray(arr).convert('RGB').resize((416,240)),(ci*416,ri*264+24))
            draw.text((ci*416+4,ri*264+4),f'{title} / frame {fi+1}',fill='white')
    canvas.save(RUN/f'{TASK}_{split}_{step}.png')


@torch.no_grad()
def evaluate(heads,train,holdout,step):
    heads.eval();report={}
    for split,data in [('train',train),('holdout',holdout)]:
        records=[]
        for i,case in enumerate(data):
            pred=infer(heads,case['features']);gt=case['targets'].cuda()
            records.append(dict(sample=case['row']['sample'],clamped=v1.metrics(pred,gt,TASK),
                unclamped=measure(pred,gt,TASK),official_vae=v1.metrics(case['baseline'],case['targets'],TASK)))
            if i==0:preview(case,pred,step,split)
        means={kind:{k:sum(r[kind][k] for r in records)/len(records) for k in records[0][kind]}
               for kind in ['clamped','unclamped','official_vae']}
        report[split]=dict(count=len(records),mean=means,cases=records)
    v1.atomic_json(RUN/f'{TASK}_eval_{step}.json',report)
    emit('evaluation',step=step,train=report['train']['mean']['clamped'],holdout=report['holdout']['mean']['clamped'])
    heads.train();return report


def main():
    global TASK
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=list(v1.TASKS))
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--resume',action='store_true');a=p.parse_args();TASK=a.task
    torch.set_num_threads(4);torch.manual_seed(20261129)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    train_rows=v1.rows('train_candidates');holdout_rows=v1.rows('eval_candidates')
    assert len(train_rows)==18 and len(holdout_rows)==6
    assert not ({r['sample'] for r in train_rows}&{r['sample'] for r in holdout_rows})
    assert not ({r['scene'] for r in train_rows}&{r['scene'] for r in holdout_rows})
    manifest=dict(train=train_rows,holdout=holdout_rows)
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    request=dict(task=TASK,steps=a.steps,eval_every=a.eval_every,train_clips=18,holdout_clips=6,
        train_frames=18*81,resolution=[832,480],manifest=manifest,manifest_sha256=fingerprint,
        backbone_sha256=SHA,backbone_frozen=True,code_sha256=v1.sha(__file__),
        transform='Per-task/per-branch training-only covariance; regularized whitening eigenvalues + 0.001 * mean covariance diagonal',
        base='Training-only ridge solve with same regularizer; frozen during MLP fitting',
        nonlinear='Separate first/regular 3072->2048->1024/4096 GELU residual MLPs, zero last layer',
        residual_scale=.05,precision='FP64 stats/precompute, FP32 pixel head, BF16 cached native input features',
        loss='Uniform pixel MSE; first frame weight 1/81, regular frames 80/81; no crop, no augmentation',
        batch=dict(first=512,regular=4096),optimizer=dict(name='AdamW',lr=.0003,weight_decay=.001,warmup=100,min_lr=.000003),
        checkpoint_policy='Latest optimizer/RNG checkpoint every 1000; fixed 2000/5000/final heads; no holdout selection',
        single_clip_weights_reused=False,validation_policy='6 held-out clips reporting only; no gradients or whitening statistics')
    if not a.resume:assert not (RUN/f'{TASK}_latest.pt').exists(),'Use --resume for existing training'
    v1.atomic_json(RUN/f'{TASK}_REQUEST.json',request)
    emit('started',pid=os.getpid(),device=torch.cuda.get_device_name(),free_gb=torch.cuda.mem_get_info()[0]/2**30)
    assert torch.cuda.mem_get_info()[0]>20*2**30
    train=load_cases(train_rows);holdout=load_cases(holdout_rows)
    stats_path=RUN/f'{TASK}_geometry.pt'
    if stats_path.exists():
        saved=torch.load(stats_path,map_location='cpu',weights_only=False);assert saved['manifest_sha256']==fingerprint
        stats=saved['stats']
    else:
        stats={name:geometry(train,name=='first') for name in ['first','regular']}
        save_atomic(dict(stats=stats,manifest_sha256=fingerprint),stats_path)
    heads=nn.ModuleDict({name:Branch(s,2048,.05) for name,s in stats.items()}).cuda()
    optimizer=torch.optim.AdamW(heads.parameters(),lr=.0003,weight_decay=.001)
    prepared={}
    with torch.no_grad():
        for name in ['first','regular']:
            zs=[];rs=[]
            for case in train:
                first=name=='first'
                x=(case['features'][:1] if first else case['features'][1:]).reshape(-1,3072)
                y=(pack(case['targets'][:1]) if first else pack_regular(case['targets'])).cuda().float()
                z,base=heads[name].transform(x);zs.append(z);rs.append((y-base)/.05)
            prepared[name]=(torch.cat(zs),torch.cat(rs))
        del zs,rs,z,base,y
    start=0
    if a.resume:
        checkpoint=torch.load(RUN/f'{TASK}_latest.pt',map_location='cuda',weights_only=False)
        assert checkpoint['request']['manifest_sha256']==fingerprint and checkpoint['request']['steps']==a.steps
        heads.load_state_dict(checkpoint['heads']);optimizer.load_state_dict(checkpoint['optimizer']);start=checkpoint['step']
        torch.set_rng_state(checkpoint['cpu_rng'].cpu());torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu())
    else:
        evaluate(heads,train,holdout,0)
    emit('training_started',step=start,parameters=sum(p.numel() for p in heads.parameters()),
        regular_patches=len(prepared['regular'][0]),first_patches=len(prepared['first'][0]),
        gpu_gb=torch.cuda.memory_allocated()/2**30)
    train_start=time.monotonic()
    for step in range(start+1,a.steps+1):
        optimizer.zero_grad(set_to_none=True);losses={}
        for name,batch,weight in [('first',512,1/81),('regular',4096,80/81)]:
            z,target=prepared[name];idx=torch.randint(len(z),(batch,),device='cuda')
            prediction=heads[name].net(z[idx]);loss=(prediction-target[idx]).square().mean()
            assert torch.isfinite(loss);(loss*weight).backward();losses[name]=float(loss)*.05**2
        norm=nn.utils.clip_grad_norm_(heads.parameters(),10.)
        assert torch.isfinite(norm)
        lr=(3e-6+(3e-4-3e-6)*.5*(1+math.cos(math.pi*step/a.steps)))*min(step/100,1)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
        if step==start+1 or step%100==0:
            emit('train_step',step=step,pixel_mse=losses,lr=lr,gradient_norm=float(norm),
                training_seconds=time.monotonic()-train_start,gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30)
        if step%a.eval_every==0 or step==a.steps:
            report=evaluate(heads,train,holdout,step)
            save_atomic(dict(step=step,heads=heads.state_dict(),optimizer=optimizer.state_dict(),
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),request=request,metrics=report),RUN/f'{TASK}_latest.pt')
            export={name:head.export() for name,head in heads.items()}
            with torch.no_grad():
                original=infer(heads,holdout[0]['features']);folded=folded_infer(export,holdout[0]['features'])
                difference=float((original-folded).abs().max());assert difference<.001,difference
            package=dict(task=TASK,step=step,heads=export,request=request,folded_max_difference=difference)
            save_atomic(package,RUN/f'{TASK}_head_latest.pt')
            if step in [2000,5000,a.steps]:save_atomic(package,RUN/f'{TASK}_head_{step}.pt')
            emit('checkpoint_saved',step=step,folded_max_difference=difference)
    v1.atomic_json(RUN/f'{TASK}_COMPLETE.json',dict(step=step,metrics=report,seconds=time.monotonic()-START))
    emit('complete',step=step)


if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True);START=time.monotonic();TASK='startup'
    try:main()
    except BaseException as error:
        v1.atomic_json(RUN/f'{TASK}_FAILED.json',dict(error=repr(error)));emit('failed',error=repr(error));raise
