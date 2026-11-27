"""Full-patch ridge readout and halo-trained residual refinement diagnostic.

Reuses immutable v1 predicted-history features. No backbone or VAE changes.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import run_probe as v1

RUN=v1.EXP/'run_v2'
TASKS=v1.TASKS


def emit(event,**kw):
    row=dict(event=event,unix_time=time.time(),**kw)
    print(json.dumps(row),flush=True)
    with (RUN/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')


class RidgeHead(nn.Module):
    def __init__(self):
        super().__init__();self.first=nn.Linear(3072,1024);self.regular=nn.Linear(3072,4096)

    def forward(self,x,first=False):
        b,d,h,w=x.shape;frames=1 if first else 4
        y=(self.first if first else self.regular)(x.permute(0,2,3,1).float())
        return y.reshape(b,h,w,frames,32,32).permute(0,3,1,4,2,5).reshape(b,frames,h*32,w*32)


class ResidualHead(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.base.requires_grad_(False)
        self.feature=nn.Conv2d(3072,4*8,1)
        self.stem=nn.Conv2d(12,24,3,padding=1)
        self.body=nn.Sequential(nn.SiLU(),nn.Conv2d(24,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,24,3,padding=1))
        self.out=nn.Conv2d(24,4,3,padding=1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x,first=False):
        base=self.base(x,first=first);b,f,h,w=base.shape
        packed=F.pixel_unshuffle(base.reshape(b*f,1,h,w),2)
        features=self.feature(x).reshape(b,4,8,x.shape[-2],x.shape[-1])[:,:f]
        features=F.interpolate(features.reshape(b*f,8,x.shape[-2],x.shape[-1]),size=(h//2,w//2),mode='bilinear',align_corners=False)
        y=self.stem(torch.cat([packed,features],1));y=y+self.body(y)
        correction=F.pixel_shuffle(self.out(F.silu(y)),2).reshape(b,f,h,w)
        return base+correction


def paired_patches(data,task,first):
    target=data['targets'][task].float().cuda()
    if first:
        x=data['features'][task][:1].reshape(-1,3072).float().cuda()
        y=target[:1].reshape(15,32,26,32).permute(0,2,1,3).reshape(-1,1024)
    else:
        x=data['features'][task][1:].reshape(-1,3072).float().cuda()
        y=target[1:].reshape(20,4,15,32,26,32).permute(0,2,4,1,3,5).reshape(-1,4096)
    return x,y


def fit_ridge(cases):
    models=nn.ModuleDict({t:RidgeHead() for t in TASKS}).cuda()
    diagnostics=[]
    # FP64 accumulators and solve reduce cancellation in centered normal equations.
    for task in TASKS:
        for first in [True,False]:
            outdim=1024 if first else 4096
            xx=torch.zeros(3072,3072,dtype=torch.float64,device='cuda')
            xy=torch.zeros(3072,outdim,dtype=torch.float64,device='cuda')
            sx=torch.zeros(3072,dtype=torch.float64,device='cuda')
            sy=torch.zeros(outdim,dtype=torch.float64,device='cuda');n=0
            started=time.monotonic()
            for index,row in enumerate(cases):
                d=v1.load_sample(str(v1.cache_path(row)));x,y=paired_patches(d,task,first)
                # Compute Gram products without TF32 truncation; then accumulate FP64.
                xd=x.double();yd=y.double()
                xx.add_(xd.T@xd);xy.add_(xd.T@yd);sx.add_(xd.sum(0));sy.add_(yd.sum(0));n+=len(x)
                del xd,yd,x,y
                emit('ridge_accumulate',task=task,first=first,clip=index+1,total=len(cases))
            mx=sx/n;my=sy/n
            covariance=xx-sx[:,None]*mx[None,:]
            cross=xy-sx[:,None]*my[None,:]
            covariance=(covariance+covariance.T)/2
            strength=float(covariance.diagonal().mean())*.001
            covariance.diagonal().add_(strength)
            weights=torch.linalg.solve(covariance,cross)
            bias=my-mx@weights
            layer=models[task].first if first else models[task].regular
            layer.weight.data.copy_(weights.T.float());layer.bias.data.copy_(bias.float())
            assert torch.isfinite(layer.weight).all()
            residual=float((covariance@weights-cross).norm()/cross.norm())
            diag=dict(task=task,first=first,training_patches=n,ridge_lambda=strength,solve_relative_residual=residual,seconds=time.monotonic()-started)
            emit('ridge_solved',**diag);diagnostics.append(diag)
    v1.atomic_json(RUN/'ridge_diagnostics.json',diagnostics)
    return models


def evaluate(models,cases,label,step):
    # Shared evaluation with its output directory redirected; original cache remains v1.
    original=v1.RUN;old_cache=v1.cache_path;old_emit=v1.emit
    v1.RUN=RUN;v1.cache_path=lambda row:original/'cache'/(Path(row['path']).name+'.pt');v1.emit=emit
    try:v1.evaluate(models,cases,label,step)
    finally:v1.RUN=original;v1.cache_path=old_cache;v1.emit=old_emit


def sample_crop(cases,task):
    row=random.choice(cases);data=v1.load_sample(str(v1.cache_path(row)))
    li=0 if random.random()<.1 else random.randrange(1,21)
    # Two token halo: 192px input -> central64px supervised. Never pads clip edges artificially.
    outer=6;inner=2;halo=2;y=random.randrange(15-outer+1);x=random.randrange(26-outer+1)
    if random.random()<.5:
        fg=torch.nonzero(data['targets']['matte'][v1.frame_slice(li)].float().mean(0)>0)
        if len(fg):
            cy,cx=fg[random.randrange(len(fg))].tolist()
            y=max(0,min(15-outer,cy//32-outer//2));x=max(0,min(26-outer,cx//32-outer//2))
    feature=data['features'][task][li,y:y+outer,x:x+outer].permute(2,0,1)[None].cuda()
    target=data['targets'][task][v1.frame_slice(li),(y+halo)*32:(y+halo+inner)*32,(x+halo)*32:(x+halo+inner)*32][None].cuda()
    return feature,target,li==0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=4000);a=parser.parse_args()
    for p in [RUN,RUN/'previews']:p.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);torch.manual_seed(20261123);random.seed(20261123)
    torch.backends.cuda.matmul.allow_tf32=False
    cases=v1.rows('train_candidates');holdout=v1.rows('eval_candidates')
    assert len(cases)==18 and len(holdout)==6
    try:
        emit('started',pid=os.getpid(),steps=a.steps,code_sha256=v1.sha(__file__))
        ridge=fit_ridge(cases)
        torch.save(ridge.state_dict(),RUN/'ridge_heads.pt')
        ridge.requires_grad_(False)
        evaluate(nn.ModuleDict({'ridge':ridge}),cases[:2],'ridge_fit',0)
        evaluate(nn.ModuleDict({'ridge':ridge}),holdout,'ridge_holdout',0)
        refiners=nn.ModuleDict({t:ResidualHead(ridge[t]) for t in TASKS}).cuda()
        optimizer=torch.optim.AdamW([p for p in refiners.parameters() if p.requires_grad],lr=1e-4,weight_decay=.001)
        emit('refiner_training_started',parameters=sum(p.numel() for p in refiners.parameters() if p.requires_grad))
        started=time.monotonic()
        for step in range(1,a.steps+1):
            task=list(TASKS)[(step-1)%2];optimizer.zero_grad(set_to_none=True)
            total=0.
            for micro in range(4):
                feature,target,first=sample_crop(cases,task)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=refiners[task](feature,first=first)[:,:,64:128,64:128]
                    loss=v1.pixel_loss(pred.float(),target.float())/4
                    if task=='matte':
                        # Equal contribution of foreground and background in addition to L1.
                        fg=target>-.98
                        if fg.any() and (~fg).any():
                            error=(pred.float()-target.float()).abs()
                            loss=loss+.125*(error[fg].mean()+error[~fg].mean())
                assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
            norm=nn.utils.clip_grad_norm_([p for p in refiners.parameters() if p.requires_grad],1.)
            assert torch.isfinite(norm);optimizer.step()
            if step==1 or step%100==0:emit('train_step',step=step,task=task,loss=total,elapsed_seconds=time.monotonic()-started)
            if step%1000==0 or step==a.steps:
                checkpoint=dict(models=refiners.state_dict(),optimizer=optimizer.state_dict(),step=step,backbone_frozen=True)
                tmp=RUN/'refiner_latest.pending.pt';torch.save(checkpoint,tmp);os.replace(tmp,RUN/'refiner_latest.pt')
                evaluate(nn.ModuleDict({'ridge':ridge,'residual':refiners}),cases[:2],'refiner_fit',step)
                # Holdout is used for fixed-schedule reporting, never optimizer/lambda selection.
                evaluate(nn.ModuleDict({'ridge':ridge,'residual':refiners}),holdout,'refiner_holdout',step)
        v1.atomic_json(RUN/'COMPLETE.json',dict(steps=a.steps,seconds=time.monotonic()-started))
    except BaseException as e:
        v1.atomic_json(RUN/'FAILED.json',dict(error=repr(e)));raise


if __name__=='__main__':main()
