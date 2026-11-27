"""Cross-patch residual head on the fixed v9 MLP; same data and MSE objective.

This is a spatial convolution ablation, not a full PixVid reproduction.
No backbone/VAE execution or held-out checkpoint selection during training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
from train_full_v9 import RUN as DATA, OUT as BASE, rows, load_case, stratified_probe
from train_multiclip_v8 import folded_infer, save_atomic
from full_data_v9 import EXP, atomic, emit
import run_probe as v1

RUN = EXP/'conv_head_v10'
SEED = 20261123


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.SiLU(),
                                 nn.Conv2d(width,width,3,padding=1))
    def forward(self,x): return x + .1*self.net(x)


class ConvHead(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.token = nn.Conv2d(3072,width,1)
        self.pixel = nn.Conv2d(4,width,3,padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(3)])
        self.output = nn.Conv2d(width,4,3,padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self,base,features):
        # base [N,1,H,W], features [N,3072,H/32,W/32].
        # All convolutions operate across patch boundaries on one shared grid.
        f = F.layer_norm(features.permute(0,2,3,1).float(),(3072,)).permute(0,3,1,2)
        cond = F.interpolate(self.token(f),size=(base.shape[-2]//2,base.shape[-1]//2),mode='nearest')
        h = self.pixel(F.pixel_unshuffle(base.float(),2)) + cond
        correction = F.pixel_shuffle(self.output(self.blocks(F.silu(h))),2)
        return base.float() + .1*correction


def frame_tokens(indices):
    return torch.where(indices==0,0,(indices-1)//4+1)


@torch.no_grad()
def prepare(case,export):
    base = folded_infer(export,case['features'])
    return dict(base=base,features=case['features'].cuda(),target=case['targets'].cuda())


def crop_batch(bank,rng,fixed=False):
    xs=[];fs=[];ys=[]
    selected=list(range(min(4,len(bank)))) if fixed else rng.sample(range(len(bank)),min(4,len(bank)))
    for i in selected:
        b=bank[i]; start=39 if fixed else rng.randrange(80)
        iy=3 if fixed else rng.randrange(8); ix=8 if fixed else rng.randrange(19)
        ids=torch.tensor([start,start+1],device='cuda'); ti=frame_tokens(ids)
        y,x=iy*32,ix*32
        xs.append(b['base'][ids,y:y+256,x:x+256,None].permute(0,3,1,2))
        ys.append(b['target'][ids,y:y+256,x:x+256,None].permute(0,3,1,2).float())
        fs.append(b['features'][ti,iy:iy+8,ix:ix+8].permute(0,3,1,2).float())
    return torch.cat(xs),torch.cat(fs),torch.cat(ys)


@torch.no_grad()
def infer_video(model,base,features):
    out=[]
    for start in range(0,81,4):
        ids=torch.arange(start,min(start+4,81),device='cuda')
        f=features[frame_tokens(ids)].permute(0,3,1,2).float()
        out.append(model(base[ids,None],f)[:,0])
    return torch.cat(out)


def seam_score(value,period=32):
    """PixVid-style seam excess, adapted to our 32px grid; squared 8-bit units.

    At a boundary, subtract median of 3 neighboring positions on either side;
    subtract the same excess measured at the +4px control location. Median over
    boundaries, then average axes. Negative values are retained, not clipped.
    """
    x=(value.float().clamp(-1,1)+1)*127.5; scores=[]
    for axis in [-1,-2]:
        d=x.diff(dim=axis).square()
        profile=d.mean(dim=(0,1)) if axis==-1 else d.mean(dim=(0,2))
        borders=torch.arange(period-1,len(profile)-7,period,device=x.device)
        offsets=torch.tensor([-3,-2,-1,1,2,3],device=x.device)
        def excess(p):return profile[p]-profile[p[:,None]+offsets].median(dim=1).values
        scores.append((excess(borders)-excess(borders+4)).median())
    return float(torch.stack(scores).mean())


def quality(value,gt,task):
    result=v1.metrics(value,gt,task)
    result['patch_seam32']=seam_score(value)
    # GT-referenced gradient error restricted to actual patch borders.
    e=value.float().clamp(-1,1)-gt.float()
    result['patch_border_gradient_mae']=float((e.diff(dim=-1)[...,31::32].abs().mean()+
                                             e.diff(dim=-2)[:,31::32,:].abs().mean())/2)
    return result


@torch.no_grad()
def evaluate(model,export,cases,task,step,label,out):
    model.eval(); records=[]
    for i,row in enumerate(cases):
        c=load_case(row,task); b=prepare(c,export)
        prediction=infer_video(model,b['base'],b['features'])
        record=dict(sample=row['sample'],conv=quality(prediction,b['target'],task),
                    mlp_v9=quality(b['base'],b['target'],task),gt_seam32=seam_score(b['target']))
        if c['baseline'] is not None:
            record['official_vae']=quality(c['baseline'].cuda(),b['target'],task)
        records.append(record)
        if i==0:save_atomic(dict(prediction=prediction.cpu(),row=row,metrics=record,step=step),out/f'{label}_preview.pt')
        if (i+1)%100==0:emit('evaluation_progress',task=task,step=step,split=label,done=i+1,total=len(cases))
        del b,c,prediction
    kinds=['conv','mlp_v9']+(['official_vae'] if all('official_vae' in r for r in records) else [])
    mean={k:{m:sum(r[k][m] for r in records)/len(records) for m in records[0][k]} for k in kinds}
    report=dict(step=step,count=len(records),mean=mean,cases=records)
    atomic(out/f'{label}_{step}.json',report)
    emit('evaluation',task=task,step=step,split=label,mean=mean);model.train();return report


def setup(task):
    torch.set_num_threads(4);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=BASE/f'{task}_head_100000.pt'
    export=torch.load(p,map_location='cpu',weights_only=False)
    assert export['task']==task and export['step']==100000
    return ConvHead().cuda(),export['heads'],p


def smoke(task):
    model,export,_=setup(task);out=RUN/'smoke'/task;out.mkdir(parents=True,exist_ok=True)
    c=load_case(rows('train')[0],task);bank=[prepare(c,export)]
    b,f,y=crop_batch(bank,random.Random(SEED),fixed=True)
    assert torch.equal(model(b,f),b),'Zero-init head must exactly preserve MLP baseline'
    expected=v1.metrics(bank[0]['base'],bank[0]['target'],task)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    losses=[]
    for i in range(24):
        opt.zero_grad(set_to_none=True);p=model(b,f)
        loss=(p[...,32:-32,32:-32]-y[...,32:-32,32:-32]).square().mean()
        assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(norm);opt.step();losses.append(float(loss))
    assert losses[-1]<losses[0],losses
    # Convolution must transmit a pixel change across the x=128 patch border.
    model.eval();inp=b[:1].clone().requires_grad_(True)
    value=model(inp,f[:1])[0,0,128,128]
    grad=torch.autograd.grad(value,inp)[0]
    cross=float(grad[0,0,120:137,120:128].abs().sum());assert cross>0
    with torch.no_grad():
        pred=model(b,f);save_atomic(model.state_dict(),out/'head.pt')
        replay=ConvHead().cuda();replay.load_state_dict(torch.load(out/'head.pt',weights_only=True))
        delta=float((pred-replay(b,f)).abs().max());assert delta<1e-6
        # Interior crop and full-frame inference should match: 32px halo exceeds receptive field.
        ids=torch.tensor([39,40],device='cuda');ti=frame_tokens(ids)
        full=model(bank[0]['base'][ids,None],bank[0]['features'][ti].permute(0,3,1,2).float())
        crop_delta=float((pred[...,32:-32,32:-32]-full[...,128:320,288:480]).abs().max())
        assert crop_delta<1e-5,crop_delta
        blank=torch.zeros(4,96,128,device='cuda');grid=blank.clone();grid[:,:,32:64]=.2
        flat_score=seam_score(blank);grid_score=seam_score(grid)
        assert abs(flat_score)<1e-6 and grid_score>0,(flat_score,grid_score)
    result=dict(status='passed',task=task,steps=24,initial_mse=losses[0],final_mse=losses[-1],
                cross_patch_gradient=cross,reload_max_difference=delta,crop_full_max_difference=crop_delta,
                zero_init_identity=True,baseline=expected,parameters=sum(p.numel() for p in model.parameters()))
    atomic(out/'COMPLETE.json',result);emit('smoke_passed',**result)


def train(task,steps):
    model,export,base_path=setup(task);out=RUN/task;out.mkdir(parents=True,exist_ok=True)
    train_rows,validation,test=rows('train'),rows('validation'),rows('test')
    fingerprint=hashlib.sha256((DATA/'train.jsonl').read_bytes()).hexdigest()
    request=dict(task=task,steps=steps,train_clips=len(train_rows),validation_clips=len(validation),test_clips=len(test),
        manifest_sha256=fingerprint,baseline_checkpoint=str(base_path),baseline_sha256=v1.sha(base_path),
        architecture='Frozen v9 MLP + zero-initialized conditional spatial residual CNN; half-resolution 32 channels, three two-conv residual blocks, pixel shuffle x2',
        conditioning='Per-token LayerNorm then 1x1 projection to32 channels, nearest expansion on half-resolution grid; first token to frame0, remaining tokens to4 frames each',
        loss='Uniform pixel MSE only on crop core; no perceptual/adversarial/edge/temporal losses',
        precision='FP32 CNN and predictions; BF16 frozen features; TF32 disabled',
        batch='4 clips x2 adjacent frames x256x256 crops; supervise center192x192 with32px halo',
        sampling='8-clip GPU bank; replace one clip every2 steps in shuffled exhaustive dataset order',
        optimizer='AdamW 2e-4 warmup100 cosine to2e-6 wd0.001 gradclip1',
        evaluation='8 train/24 validation every2000; all validation/test at fixed final step; no test selection',
        parameters=sum(p.numel() for p in model.parameters()),code_sha256=v1.sha(__file__))
    latest=out/'latest.pt';saved=None
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        for k in ['manifest_sha256','baseline_sha256','steps','code_sha256']:assert saved['request'][k]==request[k],k
    atomic(out/'REQUEST.json',request)
    rng=random.Random(SEED);order=list(range(len(train_rows)));rng.shuffle(order);cursor=0;start=0
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.001)
    if saved:
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);start=saved['step']
        order=saved['order'];cursor=saved['cursor'];rng.setstate(saved['rng'])
    def next_index():
        nonlocal cursor,order
        if cursor==len(order):rng.shuffle(order);cursor=0
        i=order[cursor];cursor+=1;return i
    bank_ids=saved['bank_ids'] if saved else [next_index() for _ in range(8)]
    bank=[prepare(load_case(train_rows[i],task),export) for i in bank_ids]
    if saved:
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    prefetch=saved['prefetch'] if saved else next_index()
    train_probe=stratified_probe(train_rows,8);val_probe=stratified_probe(validation,24)
    if not saved:evaluate(model,export,val_probe,task,0,'validation_probe',out)
    started=time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(load_case,train_rows[prefetch],task)
        emit('training_started',task=task,steps=steps,start=start,parameters=request['parameters'])
        for step in range(start+1,steps+1):
            if step%2==0:
                slot=(step//2-1)%8;c=future.result();bank[slot]=prepare(c,export);bank_ids[slot]=prefetch;del c
                prefetch=next_index();future=pool.submit(load_case,train_rows[prefetch],task)
            b,f,y=crop_batch(bank,rng);opt.zero_grad(set_to_none=True);pred=model(b,f)
            loss=(pred[...,32:-32,32:-32]-y[...,32:-32,32:-32]).square().mean()
            assert torch.isfinite(loss);loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
            lr=(2e-6+(2e-4-2e-6)*.5*(1+math.cos(math.pi*step/steps)))*min(step/100,1)
            for g in opt.param_groups:g['lr']=lr
            opt.step()
            if step==start+1 or step%100==0:
                elapsed=time.time()-started
                status=dict(task=task,phase='training',step=step,steps=steps,mse=float(loss.detach()),lr=lr,
                    elapsed_seconds=elapsed,steps_per_second=(step-start)/max(elapsed,1e-6),
                    train_clip_loads=8+step//2,approx_dataset_passes=(8+step//2)/len(train_rows),
                    gpu_gb=torch.cuda.memory_allocated()/2**30,time=time.time())
                atomic(out/'STATUS.json',status);emit('train_step',status=status)
            del b,f,y,pred,loss
            if step%2000==0 or step==steps:
                package=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,request=request,
                    order=order,cursor=cursor,rng=rng.getstate(),bank_ids=bank_ids,prefetch=prefetch,
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
                save_atomic(package,latest)
                save_atomic(dict(model=model.state_dict(),step=step,request=request),out/f'head_{step}.pt')
                emit('checkpoint_saved',task=task,step=step)
                evaluate(model,export,train_probe,task,step,'train_probe',out)
                evaluate(model,export,val_probe,task,step,'validation_probe',out)
    atomic(out/'STATUS.json',dict(task=task,phase='final_evaluation',step=steps,time=time.time()))
    val=evaluate(model,export,validation,task,steps,'validation_full',out)
    test_result=evaluate(model,export,test,task,steps,'test_full',out)
    atomic(out/'TRAINING_COMPLETE.json',dict(step=steps,validation=val['mean'],test=test_result['mean'],request=request,time=time.time()))
    # Export the exact same three cases used for the v9 visual review.
    from export_conv_v10 import export_examples
    export_examples(task,model,export,out)
    atomic(out/'COMPLETE.json',dict(step=steps,validation=val['mean'],test=test_result['mean'],time=time.time()))
    atomic(out/'STATUS.json',dict(task=task,phase='complete',step=steps,time=time.time()))
    emit('training_complete',task=task,step=steps)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['depth','matte'])
    p.add_argument('--steps',type=int,default=30000);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:
        if a.smoke:smoke(a.task)
        else:train(a.task,a.steps)
    except BaseException as error:
        out=RUN/('smoke' if a.smoke else '')/a.task
        atomic(out/'FAILED.json',dict(error=repr(error),time=time.time()));raise
