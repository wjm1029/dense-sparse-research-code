"""Resumable four-GPU head warmup, with a single-example acceptance gate."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from full_data_v9 import EXP, atomic, emit
from train_full_v9 import RUN as DATA, rows, load_case, stratified_probe
from train_multiclip_v8 import save_atomic, folded_infer
from train_conv_v10 import quality
from pixvid_detailer_v12 import Detailer, expand_features, infer_video
import run_probe as v1

RUN = EXP/'pixvid_detailer_v12'
SEED = 20261123
sys.path.insert(0, str(RUN/'deps'))


def setup():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


def perceptual():
    import lpips
    return lpips.LPIPS(net='vgg', verbose=False).cuda().eval().requires_grad_(False)


def crop(case, rng, fixed=False):
    # 8x8 tokens ->256px, supervise center192px. Full causal warmup precedes loss.
    start = 0 if fixed or rng.random() < .2 else rng.randrange(1, 74)
    length = 9 if fixed else 8
    begin = max(0, start-Detailer.temporal_context)
    ids = torch.arange(begin, start+length, device='cuda')
    iy, ix = (3, 8) if fixed else (rng.randrange(8), rng.randrange(19))
    feat = case['features'][:, iy:iy+8, ix:ix+8].cuda()
    x = expand_features(feat, ids)
    gt = case['targets'][start:start+length, iy*32+32:(iy+8)*32-32,
                         ix*32+32:(ix+8)*32-32].cuda().float()[None]
    return x, ids, gt, start-begin


def objective(pred, gt, lp, generator, fixed=False):
    mse = (pred-gt).square().mean()
    delta = pred.diff(dim=1)-gt.diff(dim=1)
    temporal = (delta.square()+1e-6).sqrt().mean()-1e-3
    count = min(4, pred.shape[1])
    indices = list(range(count)) if fixed else generator.sample(range(pred.shape[1]), count)
    # Scalar normalized depth replicated across RGB for the paper's VGG LPIPS.
    p = pred[:, indices].reshape(-1, 1, *pred.shape[-2:]).expand(-1, 3, -1, -1)
    g = gt[:, indices].reshape_as(p[:, :1]).expand(-1, 3, -1, -1)
    percept = lp(p, g).mean()
    return mse+.2*temporal+.2*percept, dict(mse=float(mse.detach()), temporal=float(temporal.detach()),
        lpips=float(percept.detach()), mae=float((pred-gt).detach().abs().mean()))


def forward_loss(model, case, rng, lp, fixed=False):
    x, ids, gt, warm = crop(case, rng, fixed)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(x, ids)[:, warm:, 32:-32, 32:-32]
    loss, info = objective(pred, gt, lp, rng, fixed)
    return loss, info


def structural_checks(model):
    model.eval()
    # Small grid but full model, FP32 verifies temporal stream equivalence/causality.
    x = torch.randn(1, 19, 2, 3, 3072, device='cuda')*.5
    ids = torch.arange(19, device='cuda')
    with torch.no_grad():
        full = model(x, ids)
        state = {}; parts = []
        for s in range(0, 19, 4):
            parts.append(model(x[:, s:s+4], ids[s:s+4], state))
        stream_error = float((full-torch.cat(parts, 1)).abs().max())
        future = x.clone(); future[:, 10:] += torch.randn_like(future[:, 10:])*10
        future_error = float((full[:, :10]-model(future, ids)[:, :10]).abs().max())
        assert stream_error < 2e-5 and future_error == 0, (stream_error, future_error)
        assert expand_features(torch.arange(21,device='cuda')[:,None,None,None], torch.arange(9,device='cuda')).flatten().tolist()==[0,1,1,1,1,2,2,2,2]
    # Gradient must propagate across the 32px token boundary.
    xx = x[:, :3].clone().requires_grad_(True)
    val = model(xx, ids[:3])[0, -1, 32, 32]
    grad = torch.autograd.grad(val, xx)[0]
    cross = float(grad[..., 0, 0, :].abs().sum())
    assert cross > 0, cross
    return dict(stream_full_max_error=stream_error, future_leak_max_error=future_error,
                cross_token_gradient=cross, frame_alignment=True)


def smoke(steps):
    setup(); out=RUN/'single_sample'; out.mkdir(parents=True, exist_ok=True)
    model = Detailer().cuda()
    checks = structural_checks(model)
    lp=perceptual(); case=load_case(rows('train')[0], 'depth'); rng=random.Random(SEED)
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=0)
    first=None; started=time.monotonic()
    for step in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        loss, info=forward_loss(model,case,rng,lp,True)
        assert torch.isfinite(loss)
        loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),1.); assert torch.isfinite(norm)
        opt.step()
        if first is None:first=info
        if step==1 or step%20==0:
            status=dict(step=step,steps=steps,elapsed_seconds=time.monotonic()-started,**info)
            atomic(out/'STATUS.json',status);emit('single_sample',**status)
    model.eval()
    with torch.no_grad():
        x,ids,gt,warm=crop(case,rng,True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x,ids)[:,warm:,32:-32,32:-32]
        _,last=objective(pred,gt,lp,rng,True)
        save_atomic(dict(model=model.state_dict(),step=steps),out/'head.pt')
        replay=Detailer().cuda().eval();replay.load_state_dict(torch.load(out/'head.pt',weights_only=False)['model'])
        with torch.autocast('cuda',dtype=torch.bfloat16):again=replay(x,ids)[:,warm:,32:-32,32:-32]
        checks['reload_max_error']=float((pred-again).abs().max())
        assert checks['reload_max_error']==0
        save_atomic(dict(prediction=pred.cpu(),target=gt.cpu(),sample=case['row']['sample']),out/'preview.pt')
    # This gate verifies learnability on one fixed 9-frame crop, not generalization.
    passed=last['mae']<.03 and last['mse']<.003 and last['mse']<first['mse']*.15
    receipt=dict(passed=passed,scope='one train video, first9 frames, fixed center192x192 crop',
        sample=case['row']['sample'],initial=first,final=last,checks=checks,steps=steps,
        seconds=time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters()))
    atomic(out/'COMPLETE.json',receipt);emit('single_sample_complete',**receipt)
    assert passed, receipt


@torch.no_grad()
def evaluate(model,cases,step,label,out,rank=0,world=1):
    records=[]; model.eval()
    for i,row in enumerate(cases[rank::world]):
        case=load_case(row,'depth'); pred=infer_video(model,case['features']);gt=case['targets'].cuda()
        r=dict(sample=row['sample'],detailer=quality(pred,gt,'depth'))
        if case['baseline'] is not None:r['official_vae']=quality(case['baseline'].cuda(),gt,'depth')
        records.append(r)
        if (i+1)%20==0:emit('eval_progress',rank=rank,step=step,split=label,done=i+1)
    if world>1:
        gathered=[None]*world;dist.all_gather_object(gathered,records);records=[r for part in gathered for r in part]
    kinds=['detailer']+(['official_vae'] if all('official_vae' in r for r in records) else [])
    mean={k:{m:sum(r[k][m] for r in records)/len(records) for m in records[0][k]} for k in kinds}
    report=dict(step=step,count=len(records),mean=mean,cases=records)
    if rank==0:atomic(out/f'{label}_{step}.json',report);emit('evaluation',step=step,split=label,mean=mean)
    return report


def train(steps,accum):
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);dist.init_process_group('nccl');setup()
    out=RUN/'depth';out.mkdir(parents=True,exist_ok=True)
    gate=json.loads((RUN/'single_sample/COMPLETE.json').read_text());assert gate['passed']
    model=Detailer().cuda();ema=copy.deepcopy(model).eval().requires_grad_(False)
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False)
    lp=perceptual();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=0)
    train_rows=rows('train');val=rows('validation');test=rows('test')
    order=list(range(len(train_rows)));random.Random(SEED).shuffle(order);order=order[rank::world]
    rng=random.Random(SEED+rank);cursor=0;epoch=0;start=0
    request=dict(stage='head warmup; frozen Dense6 native DiT',task='depth',steps=steps,world_size=world,
        train_clips=len(train_rows),validation_clips=len(val),test_clips=len(test),
        architecture='SiLU -> causal3x1x1 3072->3072 -> five (ConvTranspose1x2x2 + causal3x3x3 + SiLU) -> causal3x1x1 scalar',
        widths=list(Detailer.widths),parameters=sum(p.numel() for p in model.parameters()),
        temporal_adaptation='Native token0 maps frame0; token1..20 each maps4raw frames, with learned phase embedding before causal detailer',
        input='Existing BF16 last-layer features after native fixed-t=0 head normalization/modulation; not raw pre-head s+e_t',
        differences_from_paper=['Frozen backbone during warmup','Native RGB VAE encoder and latent AR history retained',
            'Temporal expansion for native compressed tokens','Scalar depth instead of RGB; LPIPS on replicated depth',
            'Deterministic t=0 MSE, no flow/noise reweighting','Five levels for spatial patch32; widths adapted to D3072'],
        loss='pixel MSE +0.2 temporal Charbonnier(delta=0.001) +0.2 VGG LPIPS on4frames',
        optimizer='AdamW lr1e-4 beta0.9/0.95 wd0 clip1 EMA0.999; constant LR as paper',
        sampling=f'{world} GPUs x{accum} crops/update; each crop8targetframes plus up to14contextframes,256x256->192x192 core; four-case CPU bank, refresh1 per2steps perGPU; disjoint exhaustive rank partitions',
        precision='BF16 conv autocast, FP32 final scalar output/loss/master weights',
        evaluation='16 validation clips every2000; full445 validation/full471 test at fixed final step, no test checkpoint selection',
        manifest_sha256=hashlib.sha256((DATA/'train.jsonl').read_bytes()).hexdigest(),
        code_sha256={f:v1.sha(Path(__file__).parent/f) for f in ['train_pixvid_v12.py','pixvid_detailer_v12.py']})
    latest=out/f'latest_rank{rank}.pt';saved=None
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        assert saved['request']==request,'Resume contract changed'
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);opt.load_state_dict(saved['optimizer'])
        start=saved['step'];cursor=saved['cursor'];epoch=saved['epoch'];order=saved['order'];rng.setstate(saved['rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    if rank==0:atomic(out/'REQUEST.json',request)
    def next_id():
        nonlocal cursor,epoch
        if cursor==len(order):rng.shuffle(order);cursor=0;epoch+=1
        i=order[cursor];cursor+=1;return i
    bank_ids=saved['bank_ids'] if saved else [next_id() for _ in range(4)]
    bank=[load_case(train_rows[i],'depth') for i in bank_ids]
    prefetch=saved['prefetch'] if saved else next_id()
    del saved
    begun=time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(load_case,train_rows[prefetch],'depth')
        for step in range(start+1,steps+1):
            if step%2==0:
                slot=(step//2-1)%len(bank);bank[slot]=future.result();bank_ids[slot]=prefetch
                prefetch=next_id();future=pool.submit(load_case,train_rows[prefetch],'depth')
            ddp.train();opt.zero_grad(set_to_none=True);values=[]
            for micro in range(accum):
                from contextlib import nullcontext
                ctx=ddp.no_sync() if micro<accum-1 else nullcontext()
                with ctx:
                    loss,info=forward_loss(ddp,bank[rng.randrange(len(bank))],rng,lp)
                    assert torch.isfinite(loss);(loss/accum).backward();values.append(info)
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
            with torch.no_grad():
                for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,.001)
            if step==start+1 or step%20==0:
                status=dict(phase='training',step=step,steps=steps,rank=rank,epoch=epoch,
                    seconds_per_step=(time.monotonic()-begun)/(step-start),gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30,
                    **{k:sum(v[k] for v in values)/len(values) for k in values[0]})
                atomic(out/f'STATUS_rank{rank}.json',status)
                if rank==0:atomic(out/'STATUS.json',status);emit('train_step',**status)
            if step%500==0 or step==steps:
                save_atomic(dict(step=step,model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
                    request=request,order=order,cursor=cursor,epoch=epoch,bank_ids=bank_ids,prefetch=prefetch,rng=rng.getstate(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),latest)
                if rank==0:save_atomic(dict(step=step,model=ema.state_dict(),request=request),out/'head_latest.pt')
            if step%2000==0 or step==steps:
                evaluate(ema,stratified_probe(val,16),step,'validation_probe',out,rank,world)
                if rank==0:
                    save_atomic(dict(step=step,model=ema.state_dict(),request=request),out/f'head_{step}.pt')
                    export_examples(ema,out,step)
                dist.barrier()
    if rank==0:atomic(out/'STATUS.json',dict(phase='full_evaluation',step=steps))
    v=evaluate(ema,val,steps,'validation_full',out,rank,world)
    t=evaluate(ema,test,steps,'test_full',out,rank,world)
    if rank==0:
        atomic(out/'COMPLETE.json',dict(step=steps,validation=v['mean'],test=t['mean'],request=request,time=time.time()))
        atomic(out/'STATUS.json',dict(phase='complete',step=steps))
    dist.destroy_process_group()


@torch.no_grad()
def export_examples(model,out,step):
    import numpy as np
    import subprocess
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    dest=out/f'visual_{step}';dest.mkdir(exist_ok=True)
    examples=json.loads((DATA/'visual_review/MANIFEST.json').read_text())['examples']
    lookup={r['sample']:r for r in rows('test')}
    # Intermediate progress uses validation only. Test example IDs are exported at final step.
    final=step==json.loads((out/'REQUEST.json').read_text())['steps']
    selected=[(e['label'],lookup[e['sample']]) for e in examples] if final else [('validation',stratified_probe(rows('validation'),16)[0])]
    old=torch.load(DATA/'training/depth_head_100000.pt',map_location='cpu',weights_only=False)['heads']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    for label,row in selected:
        c=load_case(row,'depth');pred=infer_video(model,c['features']);base=folded_infer(old,c['features'])
        arr=[((a.float().clamp(-1,1)+1)*127.5).round().byte().cpu().numpy() for a in [c['targets'],c['baseline'],base,pred]]
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s','4160x528','-r','16','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','16',
            '-pix_fmt','yuv420p','-threads','4','-movflags','+faststart',str(dest/f'{label}_depth.mp4')]
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for fi in range(81):
                canvas=Image.new('RGB',(4160,528),(20,23,28));d=ImageDraw.Draw(canvas)
                canvas.paste(Image.open(Path(row['path'])/'rgb'/f'{fi+1:06d}.png').convert('RGB'),(0,48))
                for j,a in enumerate(arr,1):canvas.paste(Image.fromarray(a[fi]).convert('RGB'),(j*832,48))
                for j,title in enumerate(['RGB','GT','Original VAE','MLP v9',f'PixVid detailer v12 | {step}']):
                    d.text((j*832+12,12),title,font=font,fill='white')
                p.stdin.write(np.asarray(canvas).tobytes())
                if fi==40:canvas.save(dest/f'{label}_frame41.png')
            p.stdin.close();assert p.wait()==0
        finally:
            if p.poll() is None:p.kill();p.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','train'],required=True)
    parser.add_argument('--steps',type=int,required=True);parser.add_argument('--accum',type=int,default=2);a=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    try:
        smoke(a.steps) if a.mode=='smoke' else train(a.steps,a.accum)
    except BaseException as e:
        atomic(RUN/f'FAILED_{a.mode}_rank{os.environ.get("RANK","0")}.json',dict(error=repr(e),time=time.time()))
        raise
