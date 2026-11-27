"""Matched linear-head optimization tests on immutable native features.

All variants begin with the same predictions. Whitening is estimated from the
diagnostic training inputs only and is not a validated deployable preprocessing.
"""
import json
import math
import time
import torch
from torch import nn
import run_probe as v1
import run_probe_v2 as v2
from decode_unit_v5 import pack, measure

RUN = v1.EXP/'decode_condition_v6'


def emit(event, **kw):
    obj = dict(event=event, seconds=time.monotonic()-START, **kw)
    print(json.dumps(obj), flush=True)
    with (RUN/'events.jsonl').open('a') as f: f.write(json.dumps(obj)+'\n')


def geometry(x):
    xd = x.double(); mean = xd.mean(0); xc = xd-mean
    if x.shape[0] < x.shape[1]:
        values, u = torch.linalg.eigh(xc@xc.T)
        keep = values > values[-1]*1e-10
        values = values[keep]; v = xc.T@u[:,keep]/values.sqrt()[None]
    else:
        values, v = torch.linalg.eigh(xc.T@xc)
        keep = values > values[-1]*1e-10
        values, v = values[keep], v[:,keep]
    scale = (values/x.shape[0]).sqrt()
    projection = v/scale[None]
    z = (xc@projection).float()
    receipt = dict(n=x.shape[0], d=x.shape[1], retained_rank=len(values),
        centered_covariance_condition=float(values[-1]/values[0]),
        feature_rms=float(xd.square().mean().sqrt()), centered_rms=float(xc.square().mean().sqrt()),
        mean_energy_fraction=float(mean.square().sum()/xd.square().sum(1).mean()),
        whiten_covariance_max_error=float((z.T.double()@z.double()/len(z)-torch.eye(z.shape[1],device='cuda')).abs().max()))
    return mean, xc.std(0, unbiased=False).clamp_min(1e-6), projection, v*scale[None], z, receipt


def optimize(z, y, weight, bias, label, task, steps=1000):
    layer = nn.Linear(z.shape[1], y.shape[1]).cuda().float()
    with torch.no_grad(): layer.weight.copy_(weight.T); layer.bias.copy_(bias)
    opt = torch.optim.Adam(layer.parameters(), lr=3e-4)
    initial = None
    for step in range(steps+1):
        pred = layer(z); loss = (pred-y).square().mean()
        assert torch.isfinite(loss)
        if step%100==0 or step==steps:
            m = measure(pred, y, task)
            if initial is None: initial = m
            emit('fit', label=label, task=task, step=step, metrics=m)
        if step==steps: break
        opt.zero_grad(set_to_none=True); loss.backward()
        for group in opt.param_groups:
            group['lr'] = 3e-6+(3e-4-3e-6)*.5*(1+math.cos(math.pi*step/steps))
        opt.step()
    return layer, dict(initial=initial, final=m)


def main():
    torch.set_num_threads(4); torch.manual_seed(20261127)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    row = v1.rows('overfit8')[0]; data = v1.load_sample(str(v1.cache_path(row)))
    state = torch.load(v2.RUN/'refiner_latest.pt',map_location='cpu',weights_only=False)['models']
    v1.atomic_json(RUN/'REQUEST.json', dict(sample=row, code_sha256=v1.sha(__file__),
        precision='FP32 training/matmul, FP64 whitening', loss='MSE only', steps=1000,
        controls='Identical initial linear predictions; raw, centered, standardized, whitened input reparameterizations',
        scope='Fixed native features; training-set diagnostic only; no backbone updates'))
    results = {}
    for task in v1.TASKS:
        f = data['features'][task].cuda().float(); gt=data['targets'][task].cuda().float()
        task_report = {}
        for unit in [0, 1, 'full_regular']:
            first = unit == 0
            name = 'first' if first else 'regular'
            w0 = state[f'{task}.base.{name}.weight'].T.cuda().float()
            b0 = state[f'{task}.base.{name}.bias'].cuda().float()
            if unit == 'full_regular':
                x=f[1:].reshape(-1,3072)
                y=torch.cat([pack(gt[1+4*i:5+4*i]) for i in range(20)])
            else:
                x=f[unit].reshape(-1,3072); y=pack(gt[v1.frame_slice(unit)])
            mean,std,projection,inverse,zwhite,receipt=geometry(x)
            emit('geometry', task=task, scope=str(unit), **receipt)
            record={'geometry':receipt, 'variants':{}}
            kinds=['raw','centered','standardized','whitened'] if unit!='full_regular' else ['whitened']
            reference=x@w0+b0
            for kind in kinds:
                if kind=='raw': z,w,b=x,w0,b0
                elif kind=='centered': z=(x.double()-mean).float(); w=w0; b=(mean@w0.double()+b0).float()
                elif kind=='standardized':
                    z=((x.double()-mean)/std).float(); w=(std[:,None]*w0.double()).float(); b=(mean@w0.double()+b0).float()
                else:
                    z=zwhite; w=(inverse.T@w0.double()).float(); b=(mean@w0.double()+b0).float()
                discrepancy=float((z@w+b-reference).abs().max())
                assert discrepancy<1e-3, discrepancy
                layer, r = optimize(z,y,w,b,f'{unit}_{kind}',task)
                r['initial_output_max_difference']=discrepancy
                record['variants'][kind]=r
                if kind=='whitened':
                    with torch.no_grad():
                        folded_w=(projection@layer.weight.T.double()).float()
                        folded_b=(layer.bias.double()-mean@folded_w.double()).float()
                        folded_pred=x@folded_w+folded_b
                        record['folded_fp32']=measure(folded_pred,y,task)
                    torch.save(dict(mean=mean.cpu(),projection=projection.cpu(),
                                    weight=layer.weight.detach().cpu(),bias=layer.bias.detach().cpu(),
                                    folded_weight=folded_w.cpu(),folded_bias=folded_b.cpu()),
                               RUN/f'{task}_{unit}_whitened.pt')
                del layer
            task_report[str(unit)]=record
            v1.atomic_json(RUN/f'{task}_partial.json',task_report)
        results[task]=task_report
    v1.atomic_json(RUN/'COMPLETE.json',dict(results=results,seconds=time.monotonic()-START))
    emit('complete')


if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True); START=time.monotonic()
    try: main()
    except BaseException as error:
        v1.atomic_json(RUN/'FAILED.json',dict(error=repr(error))); raise
