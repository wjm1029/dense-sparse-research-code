"""DC-preserving depth convolution: global training-only fit then matched adaptive arms.

The 3x3 weights sum to one and have no bias. The adaptive versions predict convex
local kernels, so token conditioning cannot introduce a patchwise depth offset.
This is a decoder diagnostic, not a reproduction of PixVid.
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
from torch.nn import functional as F
import train_conv_v10 as v
from train_full_v9 import RUN as DATA,OUT as BASE,rows,load_case,stratified_probe
from train_multiclip_v8 import save_atomic
from full_data_v9 import EXP,atomic,emit
import run_probe as v1

RUN=EXP/'depth_conv_diagnosis_v11'


def neighbors(x):
    h,w=x.shape[-2:];p=F.pad(x,(1,1,1,1),mode='replicate')
    return torch.cat([p[:,:,dy:dy+h,dx:dx+w] for dy in range(3) for dx in range(3)],dim=1)


class AdaptiveKernel(nn.Module):
    def __init__(self,kernel,conditional=False):
        super().__init__();self.conditional=conditional
        self.register_buffer('initial_logits',kernel.flatten().clamp_min(1e-6).log()[None,:,None,None])
        self.pixel=nn.Sequential(nn.Conv2d(9,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU())
        self.token=nn.Conv2d(3072,16,1)
        self.output=nn.Conv2d(16,9,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        if not conditional:self.token.requires_grad_(False)

    def forward(self,base,features):
        patch=neighbors(base.float());h=self.pixel(patch-base.float())
        if self.conditional:
            f=F.layer_norm(features.permute(0,2,3,1).float(),(3072,)).permute(0,3,1,2)
            h=h+F.interpolate(self.token(f),size=base.shape[-2:],mode='nearest')
        weights=(self.initial_logits+self.output(h)).softmax(dim=1)
        return (weights*patch).sum(dim=1,keepdim=True)


class StaticKernel(nn.Module):
    def __init__(self,kernel):
        super().__init__();self.register_buffer('kernel',kernel.reshape(1,1,3,3))
    def forward(self,base,features=None):
        return F.conv2d(F.pad(base,(1,1,1,1),mode='replicate'),self.kernel)


def configure():
    torch.set_num_threads(4);torch.manual_seed(v.SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=BASE/'depth_head_100000.pt'
    return torch.load(p,map_location='cpu',weights_only=False)['heads'],p


def sample_system(b,seed,n=8192):
    gen=torch.Generator(device='cuda').manual_seed(seed)
    fi=torch.randint(81,(n,),device='cuda',generator=gen)
    yy=torch.randint(480,(n,),device='cuda',generator=gen);xx=torch.randint(832,(n,),device='cuda',generator=gen)
    base=b['base'];center=base[fi,yy,xx]
    offsets=[(dy,dx) for dy in [-1,0,1] for dx in [-1,0,1] if (dy,dx)!=(0,0)]
    x=torch.stack([base[fi,(yy+dy).clamp(0,479),(xx+dx).clamp(0,831)]-center for dy,dx in offsets],dim=1).double()
    y=(b['target'][fi,yy,xx].float()-center).double()
    return x.T@x,x.T@y,n


def fit():
    export,base_path=configure();out=RUN/'full_data_filter';out.mkdir(parents=True,exist_ok=True)
    training=rows('train');fingerprint=hashlib.sha256((DATA/'train.jsonl').read_bytes()).hexdigest()
    base_sha=v1.sha(base_path);code_sha=v1.sha(__file__)
    cov=torch.zeros(8,8,device='cuda',dtype=torch.float64);cross=torch.zeros(8,device='cuda',dtype=torch.float64)
    cursor=0;count=0;state=out/'statistics.pt'
    if state.exists():
        s=torch.load(state,map_location='cuda',weights_only=False)
        assert s['manifest_sha256']==fingerprint and s['base_sha256']==base_sha and s['code_sha256']==code_sha
        cov=s['cov'];cross=s['cross'];cursor=s['cursor'];count=s['count']
    started=time.time();initial_cursor=cursor
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(load_case,training[cursor],'depth') if cursor<len(training) else None
        for i in range(cursor,len(training)):
            case=future.result()
            future=pool.submit(load_case,training[i+1],'depth') if i+1<len(training) else None
            b=v.prepare(case,export);xx,xy,n=sample_system(b,v.SEED+i)
            cov+=xx;cross+=xy;count+=n;del case,b
            if (i+1)%100==0 or i+1==len(training):
                save_atomic(dict(cov=cov.cpu(),cross=cross.cpu(),cursor=i+1,count=count,
                    manifest_sha256=fingerprint,base_sha256=base_sha,code_sha256=code_sha),state)
                status=dict(phase='fitting_kernel',clips=i+1,total=len(training),sampled_pixels=count,
                    elapsed_seconds=time.time()-started,clips_per_second=(i+1-initial_cursor)/max(time.time()-started,1e-6),time=time.time())
                atomic(out/'STATUS.json',status);emit('kernel_fit_progress',status=status)
    coef=torch.linalg.solve(cov+1e-8*cov.diagonal().mean()*torch.eye(8,device='cuda'),cross).float()
    weight=torch.zeros(3,3,device='cuda');weight[1,1]=1-coef.sum();i=0
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            if (dy,dx)!=(0,0):weight[dy+1,dx+1]=coef[i];i+=1
    raw=weight.clone();weight=weight.clamp_min(1e-6);weight/=weight.sum()
    receipt=dict(kernel=weight.cpu(),unconstrained_kernel=raw.cpu(),training_clips=len(training),sampled_pixels=count,
        manifest_sha256=fingerprint,base_sha256=base_sha,code_sha256=code_sha,
        negative_coefficients_projected=bool((raw<0).any()))
    save_atomic(receipt,out/'kernel.pt')
    model=StaticKernel(weight).cuda()
    probe=v.evaluate(model,export,stratified_probe(rows('validation'),24),'depth',0,'validation_probe',out)
    atomic(out/'COMPLETE.json',dict(training_clips=len(training),sampled_pixels=count,kernel=weight.tolist(),
        unconstrained_kernel=raw.tolist(),validation=probe['mean'],time=time.time()))
    atomic(out/'STATUS.json',dict(phase='complete',clips=len(training),time=time.time()))
    emit('kernel_fit_complete',kernel=weight.tolist(),validation=probe['mean'])


def smoke():
    export,_=configure();r=RUN/'adaptive_smoke';r.mkdir(parents=True,exist_ok=True)
    weights=torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]],device='cuda')/16
    bank=[v.prepare(load_case(stratified_probe(rows('train'),1)[0],'depth'),export)]
    b,f,y=v.crop_batch(bank,random.Random(v.SEED),fixed=True);result={}
    for variant in ['pixel_only','token_condition']:
        torch.manual_seed(v.SEED);model=AdaptiveKernel(weights,variant=='token_condition').cuda()
        with torch.no_grad():
            reference=StaticKernel(weights)(b)
            delta=float((reference-model(b,f)).abs().max());assert delta<1e-6
            flat=torch.ones_like(b)*.37;flat_error=float((model(flat,f)-flat).abs().max());assert flat_error<1e-6
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        losses=[]
        for step in range(100):
            opt.zero_grad(set_to_none=True);p=model(b,f)
            loss=(p[...,32:-32,32:-32]-y[...,32:-32,32:-32]).square().mean()
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(float(loss.detach()))
        assert losses[-1]<losses[0]
        save_atomic(model.state_dict(),r/f'{variant}.pt')
        replay=AdaptiveKernel(weights,variant=='token_condition').cuda();replay.load_state_dict(torch.load(r/f'{variant}.pt',weights_only=True))
        with torch.no_grad():
            assert float((model(b,f)-replay(b,f)).abs().max())==0
            flat_error=float((model(flat,f)-flat).abs().max());assert flat_error<1e-6
            ids=torch.tensor([39,40],device='cuda')
            full=model(bank[0]['base'][ids,None],bank[0]['features'][v.frame_tokens(ids)].permute(0,3,1,2).float())
            crop_delta=float((model(b,f)[...,32:-32,32:-32]-full[...,128:320,288:480]).abs().max())
            assert crop_delta<1e-5
        result[variant]=dict(initial_mse=losses[0],final_mse=losses[-1],constant_preservation_error=flat_error,
            crop_full_max_difference=crop_delta,reload_max_difference=0)
    atomic(r/'COMPLETE.json',dict(status='passed',variants=result));emit('adaptive_smoke_passed',variants=result)


@torch.no_grad()
def evaluate(model,export,cases,step,label,out,kernel):
    model.eval();static=StaticKernel(kernel.cuda()).cuda();records=[]
    for i,row in enumerate(cases):
        case=load_case(row,'depth');b=v.prepare(case,export)
        prediction=v.infer_video(model,b['base'],b['features'])
        fixed=v.infer_video(static,b['base'],b['features'])
        record=dict(sample=row['sample'],adaptive=v.quality(prediction,b['target'],'depth'),
            static_filter=v.quality(fixed,b['target'],'depth'),mlp_v9=v.quality(b['base'],b['target'],'depth'))
        if case['baseline'] is not None:record['official_vae']=v.quality(case['baseline'].cuda(),b['target'],'depth')
        records.append(record)
        if i==0:save_atomic(dict(prediction=prediction.cpu(),row=row,metrics=record,step=step),out/f'{label}_preview.pt')
        if (i+1)%100==0:emit('evaluation_progress',variant=out.name,step=step,split=label,done=i+1,total=len(cases))
        del case,b,prediction,fixed
    kinds=['adaptive','static_filter','mlp_v9']+(['official_vae'] if all('official_vae' in r for r in records) else [])
    means={k:{m:sum(r[k][m] for r in records)/len(records) for m in records[0][k]} for k in kinds}
    report=dict(step=step,count=len(records),mean=means,cases=records)
    atomic(out/f'{label}_{step}.json',report);emit('evaluation',variant=out.name,step=step,split=label,mean=means)
    model.train();return report


def train(variant,steps):
    export,base_path=configure();out=RUN/'adaptive'/variant;out.mkdir(parents=True,exist_ok=True)
    kernel_file=RUN/'full_data_filter/kernel.pt';fitted=torch.load(kernel_file,map_location='cpu',weights_only=False)
    torch.manual_seed(v.SEED);model=AdaptiveKernel(fitted['kernel'],variant=='token_condition').cuda()
    training,validation,test=rows('train'),rows('validation'),rows('test')
    request=dict(variant=variant,steps=steps,task='depth',train_clips=len(training),validation_clips=len(validation),test_clips=len(test),
        manifest_sha256=fitted['manifest_sha256'],base_sha256=v1.sha(base_path),kernel_sha256=v1.sha(kernel_file),code_sha256=v1.sha(__file__),
        architecture='Spatially adaptive convex3x3 kernel; pixel-difference CNN16ch; optional LayerNorm token projection16ch',
        initialization='Same fitted global kernel from every training video; zero logit residual; both variants exactly match at initialization',
        loss='MSE only,192x192 core of256x256 crops with32px halo',
        sampling='Same seed and8-clip bank, replace1clip every2steps in shuffled exhaustive order,4clips x2frames per batch',
        optimizer='AdamW2e-4 warmup100 cosine to2e-6 wd0.001 gradclip1; FP32 TF32 disabled',
        evaluation='8train/24validation every2000; full445validation and471test at fixed final step',
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert request['manifest_sha256']==hashlib.sha256((DATA/'train.jsonl').read_bytes()).hexdigest()
    latest=out/'latest.pt';saved=None
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        for k in ['variant','steps','manifest_sha256','kernel_sha256','code_sha256']:assert saved['request'][k]==request[k]
    atomic(out/'REQUEST.json',request)
    rng=random.Random(v.SEED);order=list(range(len(training)));rng.shuffle(order);cursor=0;start=0
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.001)
    if saved:
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);start=saved['step']
        order=saved['order'];cursor=saved['cursor'];rng.setstate(saved['rng'])
    def next_index():
        nonlocal cursor,order
        if cursor==len(order):rng.shuffle(order);cursor=0
        i=order[cursor];cursor+=1;return i
    bank_ids=saved['bank_ids'] if saved else [next_index() for _ in range(8)]
    bank=[v.prepare(load_case(training[i],'depth'),export) for i in bank_ids]
    if saved:torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    prefetch=saved['prefetch'] if saved else next_index()
    train_probe=stratified_probe(training,8);val_probe=stratified_probe(validation,24)
    if not saved:evaluate(model,export,val_probe,0,'validation_probe',out,fitted['kernel'])
    started=time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(load_case,training[prefetch],'depth')
        for step in range(start+1,steps+1):
            if step%2==0:
                slot=(step//2-1)%8;c=future.result();bank[slot]=v.prepare(c,export);bank_ids[slot]=prefetch;del c
                prefetch=next_index();future=pool.submit(load_case,training[prefetch],'depth')
            b,f,y=v.crop_batch(bank,rng);opt.zero_grad(set_to_none=True);prediction=model(b,f)
            loss=(prediction[...,32:-32,32:-32]-y[...,32:-32,32:-32]).square().mean()
            assert torch.isfinite(loss);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
            lr=(2e-6+(2e-4-2e-6)*.5*(1+math.cos(math.pi*step/steps)))*min(step/100,1)
            for group in opt.param_groups:group['lr']=lr
            opt.step()
            if step==start+1 or step%100==0:
                elapsed=time.time()-started
                status=dict(variant=variant,phase='training',step=step,steps=steps,mse=float(loss.detach()),
                    elapsed_seconds=elapsed,steps_per_second=(step-start)/max(elapsed,1e-6),train_clip_loads=8+step//2,time=time.time())
                atomic(out/'STATUS.json',status);emit('train_step',status=status)
            del b,f,y,prediction,loss
            if step%2000==0 or step==steps:
                save_atomic(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,request=request,order=order,cursor=cursor,
                    rng=rng.getstate(),bank_ids=bank_ids,prefetch=prefetch,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),latest)
                save_atomic(dict(model=model.state_dict(),step=step,request=request),out/f'head_{step}.pt')
                evaluate(model,export,train_probe,step,'train_probe',out,fitted['kernel'])
                evaluate(model,export,val_probe,step,'validation_probe',out,fitted['kernel'])
    atomic(out/'STATUS.json',dict(phase='final_evaluation',step=steps,time=time.time()))
    val=evaluate(model,export,validation,steps,'validation_full',out,fitted['kernel'])
    tested=evaluate(model,export,test,steps,'test_full',out,fitted['kernel'])
    atomic(out/'TRAINING_COMPLETE.json',dict(step=steps,validation=val['mean'],test=tested['mean'],request=request,time=time.time()))
    from export_conv_v10 import export_examples
    export_examples('depth',model,export,out,output_title='Depth v11 | '+variant)
    atomic(out/'COMPLETE.json',dict(step=steps,validation=val['mean'],test=tested['mean'],time=time.time()))
    atomic(out/'STATUS.json',dict(phase='complete',step=steps,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','smoke','train'])
    p.add_argument('--variant',choices=['pixel_only','token_condition']);p.add_argument('--steps',type=int,default=12000);a=p.parse_args()
    try:
        if a.action=='fit':fit()
        elif a.action=='smoke':smoke()
        else:
            assert a.variant;train(a.variant,a.steps)
    except BaseException as error:
        out=RUN/('full_data_filter' if a.action=='fit' else 'adaptive_smoke' if a.action=='smoke' else 'adaptive/'+str(a.variant))
        atomic(out/'FAILED.json',dict(error=repr(error),time=time.time()));raise
