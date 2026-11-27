"""Frozen Dense6 OPSD feature-readout pilot; not a reproduction of DWD/PixVid.

Uses predicted history, unmodified native adapters, t=0, and scalar task targets.
Original RGB VAE input and chunk3 latency remain. No backbone updates.
"""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path('/xs-train-nas/LingXiTeam/workspace/msh/my_code/video_dense')
EXP = ROOT/'experiments/pixvid_decode_probe_20261122'
SCRIPTS = ROOT/'code/DiffSynth-Studio/examples/wanvideo/model_training/dense_lora'
sys.path.insert(0, str(SCRIPTS))
TASKS = {'depth': 0, 'matte': 2}
RUN = EXP/'run_v1'


def emit(event, **kw):
    row = dict(event=event, unix_time=time.time(), **kw)
    print(json.dumps(row), flush=True)
    with (RUN/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')


def atomic_json(path, obj):
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(obj, indent=2)); os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def rows(name):
    return [json.loads(x) for x in (EXP/(name+'.jsonl')).read_text().splitlines()]


def cache_path(row):
    return RUN/'cache'/(Path(row['path']).name+'.pt')


def read_pixels(row):
    from dataset import _read_depth_exr, _ray_distance_to_camera_z
    p = Path(row['path']); meta = json.loads((p/'metadata.json').read_text())
    rgb, inv, matte = [], [], []
    for i in range(1, 82):
        rgb.append(np.array(Image.open(p/'rgb'/f'{i:06d}.png').convert('RGB')))
        depth = _read_depth_exr(p/'depth'/f'{i:06d}.exr')
        if meta['coordinates']['depth_type'] == 'blender_z_ray_distance_meters':
            depth = _ray_distance_to_camera_z(depth, meta['frames_data'][i-1]['K'])
        assert np.isfinite(depth).all() and (depth > 0).all()
        inv.append(np.reciprocal(depth))
        m = np.array(Image.open(p/'matte'/f'{i:06d}.png'))
        assert m.ndim == 2, m.shape
        scale = 65535. if m.dtype == np.uint16 or m.max() > 255 else 255.
        matte.append(m.astype(np.float32)/scale)
    rgb = np.stack(rgb); inv = np.stack(inv); matte = np.stack(matte)
    # Recompute clip-global percentiles; never use old sqrt-depth metadata.
    p2, p98 = np.percentile(inv, [2, 98]); assert p98 > p2 > 0
    depth = np.clip((inv-p2)/(p98-p2), 0, 1).astype(np.float32)*2-1
    assert rgb.mean() >= 40, ('too dark', row['sample'], rgb.mean())
    targets = {'depth': torch.from_numpy(depth).half(), 'matte': torch.from_numpy(matte*2-1).half()}
    return rgb, targets, dict(p2=float(p2), p98=float(p98), domain='inverse_camera_z',
                              percentiles='numpy exact over all 81 frames/pixels', rgb_mean=float(rgb.mean()))


def metrics(p, g, task=None):
    p = p.float().clamp(-1, 1); g = g.float()
    d = p-g
    dx = d[..., 1:]-d[..., :-1]; dy = d[..., 1:, :]-d[..., :-1, :]
    result = {'mae': float(d.abs().mean()), 'rmse': float(d.square().mean().sqrt()),
            'gradient_mae': float((dx.abs().mean()+dy.abs().mean())/2),
            'temporal_delta_error': float((d[1:]-d[:-1]).abs().mean())}
    edge=torch.zeros_like(g,dtype=torch.bool)
    edge[...,1:] |= (g[...,1:]-g[...,:-1]).abs()>.05
    edge[...,1:,:] |= (g[...,1:,:]-g[...,:-1,:]).abs()>.05
    if edge.any(): result['gt_edge_region_mae']=float(d.abs()[edge].mean())
    if task=='matte':
        fg=g>-.98
        if fg.any(): result['foreground_mae']=float(d.abs()[fg].mean())
        if (~fg).any(): result['background_mae']=float(d.abs()[~fg].mean())
        if fg.any() and (~fg).any(): result['balanced_mae']=(result['foreground_mae']+result['background_mae'])/2
    return result


def prepare(limit):
    from safetensors.torch import load_file
    from wan22_dense6_long477_opsd import Dense6OPSDModule
    from infer_wan22_dense4 import load_official_vae
    checkpoint = ROOT/'weights/current/dense6/long477_opsd5250_FINAL/step-5250.safetensors'
    cfg = json.loads((checkpoint.parent/'config.json').read_text())['contract']
    expected_sha = '51ee931e662cf59347f96b2f9b29136f2c62e8db77c958af6a2e225ec57d6d97'
    assert sha(checkpoint) == expected_sha
    for name, expected in cfg['scripts'].items():
        # Verify the bound runtime source when present, otherwise the main source.
        src = checkpoint.parent/'runtime_source'/name
        if not src.exists(): src = SCRIPTS/name
        assert sha(src) == expected, ('source hash mismatch', name)
        assert sha(SCRIPTS/name) == expected, ('active source hash mismatch', name)
    model_dir = ROOT/'models/Wan2.2-TI2V-5B'
    emit('loading_native_adapter', checkpoint_sha256=expected_sha)
    module = Dense6OPSDModule(str(model_dir))
    weights = load_file(str(checkpoint), device='cpu')
    contexts = weights.pop('condition_contexts')
    expected = {n for n,q in module.named_parameters() if q.requires_grad} | {
        n for n in module.state_dict() if n.startswith('task_router.')}
    assert set(weights) == expected, {'missing': sorted(expected-set(weights)), 'extra': sorted(set(weights)-expected)}
    result = module.load_state_dict(weights, strict=False)
    assert not result.unexpected_keys
    assert not (set(result.missing_keys) & expected)
    module.pipe.text_encoder = None; module.pipe.tokenizer = None
    module.context_table = contexts.to('cuda', dtype=torch.bfloat16)
    module.requires_grad_(False); module.eval()
    module.pipe.dit.to('cuda'); module.task_router.to('cuda'); module.pipe.device = torch.device('cuda')
    del weights, contexts
    import gc
    gc.collect(); torch.cuda.empty_cache()
    vae = load_official_vae(ROOT/'code/Wan2.2', model_dir, torch.device('cuda'))
    atomic_json(RUN/'native_load_receipt.json', dict(checkpoint=str(checkpoint), sha256=expected_sha,
        native_key_count=len(expected), unexpected_keys=result.unexpected_keys,
        missing_trainable_keys=sorted(set(result.missing_keys)&expected), backbone_frozen=True,
        scripts=cfg['scripts'], torch=torch.__version__, gpu=torch.cuda.get_device_name()))
    ordered = rows('overfit8') + rows('train_candidates') + rows('eval_candidates')
    unique = list({r['sample']: r for r in ordered}.values())
    if limit: unique = unique[:limit]
    for idx, row in enumerate(unique):
        out = cache_path(row)
        if out.exists():
            cached=torch.load(out,map_location='cpu',weights_only=False)
            assert cached['checkpoint_sha256']==expected_sha and cached['row']['sample']==row['sample']
            emit('cache_reused', sample=row['sample']); continue
        started = time.monotonic(); rgb, targets, preprocess = read_pixels(row)
        source = torch.from_numpy(rgb.copy()).permute(3,0,1,2).to('cuda',dtype=torch.bfloat16)/127.5-1
        with torch.no_grad(): z = vae.encode([source])[0]
        assert z.shape == (48,21,30,52) and torch.isfinite(z).all()
        payload = dict(row=row, checkpoint_sha256=expected_sha, targets=targets, preprocess=preprocess,
                       features={}, baseline={}, baseline_metrics={}, timings={})
        for task, tid in TASKS.items():
            pid = cfg['prompts'].index('output: depth' if task=='depth' else 'output: person alpha matte')
            captures=[]
            handle=module.pipe.dit.head.head.register_forward_pre_hook(
                lambda layer,args: captures.append(args[0].detach().cpu().to(torch.bfloat16)))
            caches=module.causal_controller.new_kv_cache(); history=torch.zeros_like(z[None,:,:3]); preds=[]
            module.pipe.dit.train()  # Original stream helper requires this; there is no dropout.
            torch.cuda.synchronize(); t=time.monotonic()
            try:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for offset in range(0,21,3):
                        pred,_,_=module(z[None,:,offset:offset+3],history,
                            torch.tensor([tid],device='cuda'),torch.tensor([pid],device='cuda'),offset,caches,False)
                        preds.append(pred); history=pred.detach(); module.task_router.clear()
            finally:
                handle.remove(); module.pipe.dit.eval(); module.task_router.clear()
            torch.cuda.synchronize(); dit_sec=time.monotonic()-t
            pred=torch.cat(preds,dim=2)[0]
            features=torch.cat(captures,dim=1).reshape(21,15,26,3072).contiguous()
            assert features.shape==(21,15,26,3072) and torch.isfinite(features).all()
            torch.cuda.synchronize(); t=time.monotonic()
            with torch.no_grad(): decoded=vae.decode([pred])[0]
            torch.cuda.synchronize(); decode_sec=time.monotonic()-t
            assert decoded.shape==(3,81,480,832) and torch.isfinite(decoded).all()
            base=decoded.float().mean(0).cpu().half()
            payload['features'][task]=features; payload['baseline'][task]=base
            payload['baseline_metrics'][task]=metrics(base,targets[task],task)
            payload['timings'][task]={'dit_with_feature_copy_seconds':dit_sec,'vae_decode_seconds':decode_sec}
            emit('baseline',sample=row['sample'],task=task,metrics=payload['baseline_metrics'][task],
                 vae_decode_seconds=decode_sec,dit_with_feature_copy_seconds=dit_sec)
            del decoded,pred,preds,caches,features,base
        tmp=out.with_suffix('.pending.pt'); torch.save(payload,tmp); os.replace(tmp,out)
        emit('cache_ready',index=idx+1,total=len(unique),sample=row['sample'],seconds=time.monotonic()-started)
    atomic_json(RUN/'PREPARE_COMPLETE.json',dict(samples=len(unique),limit=limit))


class LinearPixels(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.LayerNorm(3072)
        self.regular=nn.Linear(3072,4*32*32); self.first=nn.Linear(3072,32*32)
        nn.init.normal_(self.regular.weight,std=.001); nn.init.zeros_(self.regular.bias)
        nn.init.normal_(self.first.weight,std=.001); nn.init.zeros_(self.first.bias)

    def forward(self, x, first=False):
        # B,D,H,W -> B,F,H*32,W*32
        x=x.permute(0,2,3,1); b,h,w,_=x.shape
        x=(self.first if first else self.regular)(self.norm(x)); f=1 if first else 4
        return x.reshape(b,h,w,f,32,32).permute(0,3,1,4,2,5).reshape(b,f,h*32,w*32)


class ConvPixels(nn.Module):
    """Small spatial detailer with explicit four-frame unpacking; no future chunks.

    Not the original PixVid architecture. Uses per-frame spatial convolutions.
    """
    def __init__(self):
        super().__init__(); self.norm=nn.LayerNorm(3072)
        self.project=nn.Conv2d(3072,4*64,1)
        widths=[64,32,24,16,8,4]; blocks=[]
        for a,b in zip(widths,widths[1:]):
            blocks += [nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(a,b,3,padding=1),nn.SiLU()]
        self.decode=nn.Sequential(*blocks,nn.Conv2d(4,1,3,padding=1))

    def forward(self,x,first=False):
        b,_,h,w=x.shape
        x=self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x=self.project(x).reshape(b,4,64,h,w)
        if first: x=x[:,:1]
        f=x.shape[1]; return self.decode(x.reshape(b*f,64,h,w)).reshape(b,f,h*32,w*32)


@lru_cache(maxsize=32)
def load_sample(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def frame_slice(i):
    return slice(0,1) if i==0 else slice(1+(i-1)*4,1+i*4)


def pixel_loss(p,g):
    l1=(p-g).abs().mean()
    edge=((p[...,1:]-p[...,:-1] - (g[...,1:]-g[...,:-1])).abs().mean()+
          (p[...,1:,:]-p[...,:-1,:] - (g[...,1:,:]-g[...,:-1,:])).abs().mean())/2
    temp=(p[:,1:]-p[:,:-1]-(g[:,1:]-g[:,:-1])).abs().mean() if p.shape[1]>1 else l1*0
    return l1+.25*edge+.1*temp


def predict(model,features):
    result=[]
    for i in range(21):
        x=features[i].permute(2,0,1)[None].to('cuda',dtype=torch.bfloat16)
        result.append(model(x,first=i==0)[0])
    return torch.cat(result,0)


def evaluate(models, cases, stage, step):
    output=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for r in cases:
            d=load_sample(str(cache_path(r)))
            for task in TASKS:
                target=d['targets'][task]
                item={'sample':r['sample'],'task':task,'stage':stage,'step':step,
                      'baseline':metrics(d['baseline'][task],target,task), 'vae_decode_seconds':d['timings'][task]['vae_decode_seconds']}
                previews={}
                for name,bytask in models.items():
                    model=bytask[task]; model.eval()
                    # Warm up, then time compute only with features already on GPU.
                    features=d['features'][task].to('cuda')
                    predict(model,features)
                    torch.cuda.synchronize(); t=time.monotonic(); p=predict(model,features)
                    torch.cuda.synchronize(); elapsed=time.monotonic()-t
                    pred=p.float().cpu(); item[name]=dict(metrics=metrics(pred,target,task),decode_seconds=elapsed)
                    previews[name]=pred; model.train(); del features,p
                if len(output)<2:
                    frames=[0,40,80]; panels=[]
                    for fi in frames:
                        images=[]
                        for label,a in [('GT',target),('VAE',d['baseline'][task]),*previews.items()]:
                            arr=((a[fi].float().clamp(-1,1).numpy()+1)*127.5).astype(np.uint8)
                            im=Image.fromarray(arr).convert('RGB').resize((416,240))
                            panel=Image.new('RGB',(416,262));panel.paste(im,(0,22))
                            ImageDraw.Draw(panel).text((6,5),f'{label} frame {fi+1}',fill='white');images.append(panel)
                        canvas=Image.new('RGB',(416*len(images),262))
                        for j,im in enumerate(images):canvas.paste(im,(j*416,0))
                        panels.append(canvas)
                    canvas=Image.new('RGB',(panels[0].width,262*3))
                    for j,im in enumerate(panels):canvas.paste(im,(0,j*262))
                    canvas.save(RUN/'previews'/f'{stage}_{step}_{Path(r["path"]).name}_{task}.jpg')
                output.append(item)
    atomic_json(RUN/f'eval_{stage}_{step}.json',output)
    for task in TASKS:
        subset=[x for x in output if x['task']==task]
        emit('evaluation',stage=stage,step=step,task=task,cases=len(subset),
             baseline_mae=float(np.mean([x['baseline']['mae'] for x in subset])),
             **{name+'_mae':float(np.mean([x[name]['metrics']['mae'] for x in subset])) for name in models})


def train(stage,steps):
    torch.manual_seed(20261122); random.seed(20261122)
    cases=rows('overfit8' if stage in ('overfit','smoke') else 'train_candidates')
    validation=rows('eval_candidates') if stage!='smoke' else []
    if stage=='smoke': cases=cases[:1]
    assert all(cache_path(r).exists() for r in cases+validation), 'Feature preparation incomplete'
    models=nn.ModuleDict({name:nn.ModuleDict({task:cls() for task in TASKS})
                          for name,cls in [('linear',LinearPixels),('conv',ConvPixels)]}).cuda()
    optim={name:torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=.001) for name,m in models.items()}
    if stage=='pilot' and (RUN/'head_overfit_latest.pt').exists():
        saved=torch.load(RUN/'head_overfit_latest.pt',map_location='cpu',weights_only=False)
        models.load_state_dict(saved['models'])
    emit('training_started',stage=stage,steps=steps,train_cases=len(cases),eval_cases=len(validation),
         trainable_parameters={name:sum(p.numel() for p in m.parameters()) for name,m in models.items()})
    started=time.monotonic()
    for step in range(1,steps+1):
        # Same clip, task, latent time and crop for both methods.
        task=list(TASKS)[(step-1)%2]; row=cases[((step-1)//2)%len(cases)]
        d=load_sample(str(cache_path(row))); li=0 if ((step-1)//2)%10==0 else random.randrange(1,21)
        crop=3; y=random.randrange(15-crop+1); x=random.randrange(26-crop+1)
        # Half the crops target foreground, avoiding an all-background matte shortcut.
        if step%4<2:
            fg=torch.nonzero(d['targets']['matte'][frame_slice(li)].float().mean(0)>0)
            if len(fg):
                cy,cx=fg[random.randrange(len(fg))].tolist()
                y=max(0,min(15-crop,cy//32-crop//2));x=max(0,min(26-crop,cx//32-crop//2))
        feature=d['features'][task][li,y:y+crop,x:x+crop].permute(2,0,1)[None].cuda()
        target=d['targets'][task][frame_slice(li),y*32:(y+crop)*32,x*32:(x+crop)*32][None].cuda()
        losses={}
        for name,bytask in models.items():
            optim[name].zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=bytask[task](feature,first=li==0); loss=pixel_loss(prediction.float(),target.float())
            assert torch.isfinite(loss)
            loss.backward(); norm=nn.utils.clip_grad_norm_(bytask.parameters(),1.)
            assert torch.isfinite(norm);optim[name].step(); losses[name]=float(loss.detach())
        if step==1 or step%20==0:
            emit('train_step',stage=stage,step=step,task=task,losses=losses,
                 elapsed_seconds=time.monotonic()-started,gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30)
        if step%200==0 or step==steps:
            tmp=RUN/f'head_{stage}_latest.pending.pt'
            torch.save(dict(models=models.state_dict(),optimizers={n:o.state_dict() for n,o in optim.items()},
                step=step,stage=stage,seed=20261122,backbone_frozen=True),tmp)
            os.replace(tmp,RUN/f'head_{stage}_latest.pt')
            evaluate(models, [cases[0]], stage+'_fit',step)
            if validation:
                evaluate(models, validation[:2] if step<steps else validation,stage+'_holdout',step)
    atomic_json(RUN/f'{stage.upper()}_COMPLETE.json',dict(steps=steps,seconds=time.monotonic()-started))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','overfit','pilot','smoke'])
    p.add_argument('--limit',type=int,default=0);p.add_argument('--steps',type=int,default=400);a=p.parse_args()
    for d in [RUN,RUN/'cache',RUN/'previews']:d.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>20*2**30
    emit('process_started',action=a.action,pid=os.getpid(),code_sha256=sha(__file__))
    try:
        if a.action=='prepare':prepare(a.limit)
        else:train(a.action,a.steps)
    except BaseException as e:
        atomic_json(RUN/(a.action+'_FAILED.json'),dict(error=repr(e),unix_time=time.time()))
        raise


if __name__=='__main__':main()
