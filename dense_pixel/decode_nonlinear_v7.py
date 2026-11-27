"""Full 81-frame fixed-clip decoder overfit after linear conditioning diagnosis.

First-frame head is v6's gradient-trained linear fit. Regular frames use v6's
gradient-trained linear fit plus a token MLP residual. No pixel coordinates,
frame IDs, GT inputs, or per-patch trainable embeddings are used at inference.
Training-clip whitening is folded into the MLP for a raw-feature replay check.
"""
import argparse
import json
import math
import time
import torch
from torch import nn
from PIL import Image, ImageDraw
import run_probe as v1
from decode_unit_v5 import pack, unpack, measure

COND=v1.EXP/'decode_condition_v6'
RUN=v1.EXP/'decode_nonlinear_v7'


def emit(event, **kw):
    d={'event':event,'seconds':time.monotonic()-START,**kw}
    print(json.dumps(d),flush=True)
    with (RUN/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')


def accepted(m, task):
    return (m['mae']<.005 and m['rmse']<.015) if task=='depth' else (m['mae']<.001 and m['foreground_mae']<.015)


def video_from_patches(first,regular):
    return torch.cat([unpack(first,1)]+[unpack(y,4) for y in regular.reshape(20,390,4096)])


def preview(data,task,pred):
    canvas=Image.new('RGB',(416*3,264*3));draw=ImageDraw.Draw(canvas)
    for r,fi in enumerate([0,40,80]):
        for c,(title,tensor) in enumerate([('GT',data['targets'][task]),('Original VAE',data['baseline'][task]),('Frozen backbone + MLP',pred)]):
            arr=((tensor[fi].float().cpu().clamp(-1,1)+1)*127.5).round().byte().numpy()
            canvas.paste(Image.fromarray(arr).convert('RGB').resize((416,240)),(c*416,r*264+24))
            draw.text((c*416+4,r*264+4),f'{title} / frame {fi+1}',fill='white')
    canvas.save(RUN/f'{task}_comparison.png')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=list(v1.TASKS),required=True)
    parser.add_argument('--steps',type=int,default=2000);args=parser.parse_args();task=args.task
    torch.set_num_threads(4);torch.manual_seed(20261128)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    row=v1.rows('overfit8')[0]; data=v1.load_sample(str(v1.cache_path(row)))
    feat=data['features'][task].cuda().float();gt=data['targets'][task].cuda().float()
    x=feat[1:].reshape(-1,3072)
    y=torch.cat([pack(gt[1+4*i:5+4*i]) for i in range(20)])
    c=torch.load(COND/f'{task}_full_regular_whitened.pt',map_location='cuda',weights_only=True)
    first=torch.load(COND/f'{task}_0_whitened.pt',map_location='cuda',weights_only=True)
    with torch.no_grad():
        z=((x.double()-c['mean'])@c['projection']).float()
        base=x@c['folded_weight']+c['folded_bias']
        first_pred=feat[0].reshape(-1,3072)@first['folded_weight']+first['folded_bias']
        target_residual=(y-base)/.02
    model=nn.Sequential(nn.Linear(z.shape[1],2048),nn.GELU(),nn.Linear(2048,4096)).cuda().float()
    nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    request=dict(task=task,sample=row,steps=args.steps,hidden_width=2048,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        input='Frozen native features; full 81 frames; original predicted history',
        first_head='V6 gradient-trained linear head; frozen',regular_base='V6 gradient-trained linear head; frozen',
        residual='Whitened features -> Linear 2048 -> GELU -> Linear 4096; zero output initialization',
        loss='Full-batch MSE on residual / 0.02, equivalent to pixel MSE up to constant',
        lr='1e-3 cosine to 1e-5',precision='FP32; whitening computed once in FP64',
        acceptance='V4 thresholds for 3 consecutive evaluations; report unclamped and clamped errors',
        scope='Training-clip overfit diagnostic, not generalization',code_sha256=v1.sha(__file__))
    v1.atomic_json(RUN/f'{task}_REQUEST.json',request)
    streak=0
    for step in range(args.steps+1):
        residual=model(z);loss=(residual-target_residual).square().mean()
        assert torch.isfinite(loss)
        if step%100==0 or step==args.steps:
            with torch.no_grad():
                pixels=video_from_patches(first_pred,base+.02*residual)
                metric=measure(pixels,gt,task)
                clipped=v1.metrics(pixels,gt,task)
            emit('evaluation',task=task,step=step,unclamped=metric,clamped=clipped,
                 residual_mse=float(loss),gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30)
            streak=streak+1 if accepted(metric,task) else 0
            if streak>=3:break
        if step==args.steps:break
        optimizer.zero_grad(set_to_none=True);loss.backward()
        for group in optimizer.param_groups:
            group['lr']=1e-5+(1e-3-1e-5)*.5*(1+math.cos(math.pi*step/args.steps))
        optimizer.step()
    # Fold the feature whitening into layer 1; verify ordinary raw-feature inference.
    with torch.no_grad():
        folded_weight=(c['projection']@model[0].weight.T.double()).float()
        folded_bias=(model[0].bias.double()-c['mean']@folded_weight.double()).float()
        hidden=torch.nn.functional.gelu(x@folded_weight+folded_bias)
        replay_regular=base+.02*torch.nn.functional.linear(hidden,model[2].weight,model[2].bias)
        replay=video_from_patches(first_pred,replay_regular)
        replay_metrics=measure(replay,gt,task)
        prediction_difference=float((replay-pixels).abs().max())
        assert prediction_difference<.001, prediction_difference
        with torch.autocast('cuda',dtype=torch.bfloat16):
            bfbase=x@c['folded_weight']+c['folded_bias']
            bfhidden=torch.nn.functional.gelu(x@folded_weight+folded_bias)
            bfregular=bfbase+.02*torch.nn.functional.linear(bfhidden,model[2].weight,model[2].bias)
            bffirst=feat[0].reshape(-1,3072)@first['folded_weight']+first['folded_bias']
        bfmetrics=measure(video_from_patches(bffirst,bfregular),gt,task)
    torch.save(dict(task=task,step=step,first_weight=first['folded_weight'].cpu(),first_bias=first['folded_bias'].cpu(),
        regular_weight=c['folded_weight'].cpu(),regular_bias=c['folded_bias'].cpu(),
        residual_weight1=folded_weight.cpu(),residual_bias1=folded_bias.cpu(),
        residual_weight2=model[2].weight.detach().cpu(),residual_bias2=model[2].bias.detach().cpu(),
        residual_scale=.02,source_checkpoint_sha256=data['checkpoint_sha256'],request=request),RUN/f'{task}_head.pt')
    torch.save(dict(prediction=replay.cpu(),step=step,metrics=replay_metrics),RUN/f'{task}_prediction.pt')
    preview(data,task,replay)
    result=dict(task=task,step=step,accepted=streak>=3 and accepted(replay_metrics,task),
        full_clip_metrics=replay_metrics,clamped_metrics=v1.metrics(replay,gt,task),
        white_input_metrics=metric,folded_max_prediction_difference=prediction_difference,
        bf16_folded_metrics=bfmetrics,seconds=time.monotonic()-START)
    v1.atomic_json(RUN/f'{task}_COMPLETE.json',result);emit('complete',**result)


if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True);START=time.monotonic()
    try:main()
    except BaseException as error:
        emit('failed',error=repr(error));raise
