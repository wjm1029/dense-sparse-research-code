"""Matched pixel-only/conditional depth CNN diagnostics on training data only."""
import argparse
import json
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
import train_conv_v10 as v
from full_data_v9 import EXP,atomic,emit
from train_multiclip_v8 import save_atomic
from train_full_v9 import rows,load_case,stratified_probe

RUN=EXP/'depth_conv_diagnosis_v11'


class PixelOnly(v.ConvHead):
    def __init__(self):
        super().__init__()
        self.token.requires_grad_(False)
    def forward(self,base,features=None):
        h=self.pixel(F.pixel_unshuffle(base.float(),2))
        return base.float()+.1*F.pixel_shuffle(self.output(self.blocks(F.silu(h))),2)


class LinearFilter(nn.Module):
    def __init__(self,weight,bias):
        super().__init__();self.register_buffer('weight',weight.reshape(1,1,3,3));self.register_buffer('bias',bias.reshape(1))
    def forward(self,base,features=None):
        return F.conv2d(F.pad(base,(1,1,1,1),mode='replicate'),self.weight,self.bias)


def make_model(variant):
    torch.manual_seed(v.SEED)
    if variant=='pixel_only':return PixelOnly().cuda()
    if variant=='token_condition':return v.ConvHead().cuda()
    raise ValueError(variant)


def core(x):return x[...,32:-32,32:-32]


@torch.no_grad()
def fit_filter(b,y):
    patches=F.unfold(b,kernel_size=3,padding=1).reshape(len(b),9,256,256)
    x=core(patches).permute(0,2,3,1).reshape(-1,9).double()
    x=torch.cat([x,torch.ones(len(x),1,device='cuda',dtype=torch.float64)],dim=1)
    target=core(y).reshape(-1).double();cov=x.T@x/len(x);cross=x.T@target/len(x)
    ridge=1e-8*cov.diagonal().mean();w=torch.linalg.solve(cov+ridge*torch.eye(10,device='cuda'),cross).float()
    return LinearFilter(w[:9],w[9]).cuda()


def run(steps):
    _,export,_=v.setup('depth');out=RUN/'single_video_random';out.mkdir(parents=True,exist_ok=True)
    row=stratified_probe(rows('train'),1)[0];bank=[v.prepare(load_case(row,'depth'),export)]
    rng=random.Random(v.SEED);parts=[v.crop_batch(bank,rng) for _ in range(4)]
    b,f,y=[torch.cat([p[i] for p in parts]) for i in range(3)]
    base_mse=float((core(b)-core(y)).square().mean())
    weights=torch.tensor([1.,2.,1.],device='cuda');blur=LinearFilter(weights[:,None]*weights[None,:]/16,torch.tensor(0.,device='cuda'))
    linear=fit_filter(b,y)
    reference={}
    for name,model in [('gaussian3',blur),('least_squares3',linear)]:
        with torch.no_grad():p=v.infer_video(model,bank[0]['base'],bank[0]['features'])
        reference[name]=dict(crop_mse=float((core(model(b))-core(y)).square().mean()),full=v.quality(p,bank[0]['target'],'depth'))
    reference['base']=dict(crop_mse=base_mse,full=v.quality(bank[0]['base'],bank[0]['target'],'depth'))
    save_atomic(linear.state_dict(),out/'least_squares_filter.pt')
    models={k:make_model(k) for k in ['pixel_only','token_condition']}
    assert all(torch.equal(models['pixel_only'].state_dict()[k],x) for k,x in models['token_condition'].state_dict().items())
    for model in models.values():assert torch.equal(model(b,f),b)
    optimizers={k:torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.001) for k,model in models.items()}
    atomic(out/'REQUEST.json',dict(sample=row['sample'],training_only=True,steps=steps,seed=v.SEED,
        fixed_batch='Monitor batch only; training samples new spatial crops and frames from the complete video every step, shared identically across both arms',
        objective='center192x192 MSE; same AdamW2e-4, weight_decay.001, gradclip1; only token conditioning differs',
        reference=reference))
    monitor=(b,f,y);train_rng=random.Random(v.SEED+1)
    started=time.time();curves=[]
    for step in range(1,steps+1):
        parts=[v.crop_batch(bank,train_rng) for _ in range(4)]
        b,f,y=[torch.cat([p[i] for p in parts]) for i in range(3)]
        stats={}
        for name,model in models.items():
            opt=optimizers[name];opt.zero_grad(set_to_none=True);p=model(b,f)
            loss=(core(p)-core(y)).square().mean();assert torch.isfinite(loss);loss.backward()
            gradients={layer:float(getattr(model,layer).weight.grad.norm()) if getattr(model,layer).weight.grad is not None else None for layer in ['pixel','token','output']}
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
            stats[name]=dict(mse=float(loss.detach()),gradients=gradients)
        if step==1 or step%100==0:
            d=dict(step=step,elapsed_seconds=time.time()-started,variants=stats);curves.append(d)
            atomic(out/'STATUS.json',d);emit('diagnostic_step',**d)
    b,f,y=monitor
    results={}
    for name,model in models.items():
        model.eval()
        with torch.no_grad():
            p=v.infer_video(model,bank[0]['base'],bank[0]['features']);correct=p-bank[0]['base']
            crop_mse=float((core(model(b,f))-core(y)).square().mean())
        save_atomic(dict(model=model.state_dict(),steps=steps,variant=name),out/f'{name}.pt')
        results[name]=dict(crop_mse=crop_mse,full=v.quality(p,bank[0]['target'],'depth'),
            correction_std=float(correct.std()),correction_mean=float(correct.mean()))
        replay=make_model(name);replay.load_state_dict(torch.load(out/f'{name}.pt',weights_only=False)['model'])
        with torch.no_grad():delta=float((model(b,f)-replay(b,f)).abs().max())
        assert delta==0;results[name]['reload_max_difference']=delta
    pixel=results['pixel_only'];base=reference['base']
    gate=dict(crop_mse_reduced=pixel['crop_mse']<base['crop_mse']*.98,
              seams_reduced=pixel['full']['patch_seam32']<base['full']['patch_seam32']*.8,
              edge_preserved=pixel['full']['gt_edge_region_mae']<=base['full']['gt_edge_region_mae']*1.05)
    report=dict(sample=row['sample'],steps=steps,reference=reference,variants=results,gate=gate,
        ready_for_full_data=all(gate.values()),curves=curves,time=time.time())
    atomic(out/'COMPLETE.json',report);emit('diagnostic_complete',reference=reference,variants=results,gate=gate)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    try:run(a.steps)
    except BaseException as error:
        atomic(RUN/'single_video_random/FAILED.json',dict(error=repr(error),time=time.time()));raise
