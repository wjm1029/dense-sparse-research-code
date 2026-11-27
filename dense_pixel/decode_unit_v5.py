"""Frozen-feature decoder diagnostics: packing, linear fit, and tiny-set Adam.

Training-set reconstruction only; no claim about generalization or paper reproduction.
Features/targets come from the unchanged native model cache for v4's same clip.
"""
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from PIL import Image, ImageDraw
import run_probe as v1
import run_probe_v2 as v2

RUN = v1.EXP / 'decode_unit_v5'


def emit(event, **kw):
    obj = dict(event=event, **kw)
    print(json.dumps(obj), flush=True)
    with (RUN / 'events.jsonl').open('a') as f:
        f.write(json.dumps(obj) + '\n')


def pack(y):
    f, h, w = y.shape
    return y.reshape(f, h//32, 32, w//32, 32).permute(1, 3, 0, 2, 4).reshape(-1, f*1024)


def unpack(y, frames):
    return y.reshape(15, 26, frames, 32, 32).permute(2, 0, 3, 1, 4).reshape(frames, 480, 832)


def measure(p, g, task):
    d = p.float() - g.float()
    r = dict(mae=float(d.abs().mean()), rmse=float(d.square().mean().sqrt()),
             max_abs=float(d.abs().max()))
    if task == 'matte':
        fg = g > -.98
        r['foreground_mae'] = float(d[fg].abs().mean())
    return r


def render(gt, pred, name):
    canvas = Image.new('RGB', (832, 264))
    draw = ImageDraw.Draw(canvas)
    for i, (label, v) in enumerate([('GT', gt), ('Prediction', pred)]):
        arr = ((v.detach().float().cpu().clamp(-1, 1)+1)*127.5).round().byte().numpy()
        canvas.paste(Image.fromarray(arr).convert('RGB').resize((416, 240)), (416*i, 24))
        draw.text((416*i+4, 4), label, fill='white')
    canvas.save(RUN / (name+'.png'))


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261126)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    row = v1.rows('overfit8')[0]
    data = v1.load_sample(str(v1.cache_path(row)))
    init = torch.load(v2.RUN/'refiner_latest.pt', map_location='cpu', weights_only=False)
    v1.atomic_json(RUN/'REQUEST.json', dict(sample=row, features='Original frozen native predicted-history cache',
        source_checkpoint=data['checkpoint_sha256'], precision='FP64 linear solve; FP32 Adam; BF16 evaluation control',
        unit_steps=1000, optimizer='Adam lr 3e-4 cosine to 3e-6; pure MSE; no clipping',
        regularizer='1e-10 * mean Gram diagonal', code_sha256=v1.sha(__file__)))
    report = {}
    started = time.monotonic()
    for task in v1.TASKS:
        features = data['features'][task].cuda().float()
        gt = data['targets'][task].cuda().float()
        assert features.shape == (21, 15, 26, 3072)
        assert torch.equal(unpack(pack(gt[:1]), 1), gt[:1])
        assert torch.equal(unpack(pack(gt[1:5]), 4), gt[1:5])
        result = {'packing_roundtrip_exact': True,
                  'official_vae_full_clip': measure(data['baseline'][task].cuda(), gt, task)}
        linear = v2.RidgeHead().cuda().float()
        # Dual solve for first frame; primal solve for all 20 regular groups.
        for first in [True, False]:
            x = features[:1 if first else 21] if first else features[1:]
            x = x.reshape(-1, 3072).double()
            y = pack(gt[:1]) if first else torch.cat([pack(gt[1+4*i:5+4*i]) for i in range(20)])
            y = y.double()
            mx, my = x.mean(0), y.mean(0)
            xc, yc = x-mx, y-my
            if first:
                gram = xc@xc.T
                reg = float(gram.diagonal().mean())*1e-10
                gram.diagonal().add_(reg)
                weights = xc.T@torch.linalg.solve(gram, yc)
            else:
                gram = xc.T@xc
                reg = float(gram.diagonal().mean())*1e-10
                gram.diagonal().add_(reg)
                weights = torch.linalg.solve(gram, xc.T@yc)
            bias = my-mx@weights
            layer = linear.first if first else linear.regular
            with torch.no_grad():
                layer.weight.copy_(weights.T.float()); layer.bias.copy_(bias.float())
            key = 'linear_first_frame' if first else 'linear_all_regular_frames'
            result[key] = dict(fp64=measure(x@weights+bias, y, task), ridge_strength=reg,
                               examples=x.shape[0], feature_dimensions=x.shape[1])
            emit('linear_solve', task=task, scope=key, metrics=result[key])
            del x, y, xc, yc, gram, weights, bias
        with torch.no_grad():
            pred = torch.cat([linear(features[i].permute(2,0,1)[None], first=i==0)[0] for i in range(21)])
            result['linear_full_clip_fp32'] = measure(pred, gt, task)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                bf = torch.cat([linear(features[i].permute(2,0,1)[None], first=i==0)[0] for i in range(21)])
            result['linear_full_clip_bf16'] = measure(bf, gt, task)
            render(gt[40], pred[40], task+'_linear_frame41')
            torch.save(linear.state_dict(), RUN/(task+'_linear_weights.pt'))
        del pred, bf, linear
        for li in [0, 1]:
            head = v2.ResidualHead(v2.RidgeHead()).cuda().float()
            head.load_state_dict({k[len(task)+1:]:v for k,v in init['models'].items() if k.startswith(task+'.')})
            head.requires_grad_(True)
            x = features[li].permute(2,0,1)[None]
            y = gt[v1.frame_slice(li)][None]
            opt = torch.optim.Adam(head.parameters(), lr=3e-4)
            initial = None
            best = None
            for step in range(1001):
                pred = head(x, first=li==0)
                mse = (pred-y).square().mean()
                assert torch.isfinite(mse)
                if step%100==0:
                    m = measure(pred, y, task)
                    if initial is None: initial = m
                    if best is None or m['rmse'] < best['metrics']['rmse']:
                        best = dict(step=step, metrics=m)
                    emit('adam_unit', task=task, latent_index=li, step=step, metrics=m,
                         seconds=time.monotonic()-started)
                if step==1000: break
                opt.zero_grad(set_to_none=True)
                mse.backward()
                if step==0:
                    active = head.base.first if li==0 else head.base.regular
                    norms = {name:float(p.grad.norm()) for name,p in active.named_parameters()}
                    assert all(math.isfinite(v) and v>0 for v in norms.values())
                    emit('gradient_check', task=task, latent_index=li, norms=norms)
                for group in opt.param_groups:
                    group['lr'] = 3e-6 + (3e-4-3e-6)*.5*(1+math.cos(math.pi*step/1000))
                opt.step()
            result['adam_first_frame' if li==0 else 'adam_regular_4frames'] = dict(initial=initial, final=m, best=best)
            render(y[0,0], pred[0,0], f'{task}_adam_latent{li}')
            torch.save(head.state_dict(), RUN/f'{task}_adam_latent{li}.pt')
            del head, opt, pred, mse
        report[task] = result
        v1.atomic_json(RUN/'RESULTS.json', report)
    v1.atomic_json(RUN/'COMPLETE.json', dict(results=report, seconds=time.monotonic()-started))
    emit('complete', seconds=time.monotonic()-started)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        RUN.mkdir(exist_ok=True, parents=True)
        v1.atomic_json(RUN/'FAILED.json', dict(error=repr(error)))
        raise
