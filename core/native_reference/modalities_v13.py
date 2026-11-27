"""Independent matte/normal PixVid detailers with boundary-inclusive training."""
import json
from pathlib import Path
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from pixvid_detailer_v12 import Detailer as DepthDetailer, CausalConv, expand_features
from full_data_v9 import EXP
from train_full_v9 import RUN as DATA, load_case as load_scalar, rows, stratified_probe
from train_conv_v10 import quality as scalar_quality, seam_score
from prepare_normal_v13 import cache_path, SCHEMA

RUN=EXP/'pixvid_modalities_v13'


class Detailer(DepthDetailer):
    def __init__(self,task):
        super().__init__()
        self.task=task;self.channels=3 if task=='normal' else 1
        if self.channels==3:self.output=CausalConv(self.widths[-1],3)

    def forward(self,features,frame_ids,state=None):
        phase=torch.where(frame_ids==0,4,(frame_ids-1).remainder(4))
        x=features.permute(0,4,1,2,3)
        e=self.phase(phase).permute(1,0)[None,:,:,None,None]
        x=self.bottleneck(F.silu(x+e),state,'bottleneck')
        for i,b in enumerate(self.blocks):
            x=checkpoint(b,x,use_reentrant=False) if self.training and self.checkpoint_blocks and state is None else b(x,state,f'up{i}')
        with torch.autocast('cuda',enabled=False):
            return self.output(x.float(),state,'output').permute(0,2,1,3,4)


def load_case(row,task):
    if task=='matte':
        c=load_scalar(row,task);c['targets']=c['targets'][:,None]
        if c['baseline'] is not None:c['baseline']=c['baseline'][:,None]
        return c
    c=torch.load(cache_path(row),map_location='cpu',weights_only=False)
    assert c['schema']==SCHEMA and c['normal_transform']==['x','-y','-z']
    assert c['row']['sample']==row['sample']
    assert c['checkpoint_sha256']=='51ee931e662cf59347f96b2f9b29136f2c62e8db77c958af6a2e225ec57d6d97'
    assert c['targets'].shape==(81,3,480,832)
    return c


def crop(case,rng,fixed=False):
    start=0 if fixed or rng.random()<.2 else rng.randrange(1,74)
    length=9 if fixed else 8;begin=max(0,start-14)
    # A 6x6-token supervised region can start at EVERY valid location, including
    # image corners. Only artificial crop boundaries lose their one-token halo.
    cy,cx=(4,9) if fixed else (rng.randrange(10),rng.randrange(21))
    if not fixed and case['targets'].shape[1]==1 and rng.random()<.5:
        foreground=torch.nonzero(case['targets'][start,0]>-.98)
        if len(foreground):
            fy,fx=foreground[rng.randrange(len(foreground))].tolist()
            cy=max(0,min(9,fy//32-rng.randrange(6)))
            cx=max(0,min(20,fx//32-rng.randrange(6)))
    y0,x0=max(0,cy-1),max(0,cx-1);y1,x1=min(15,cy+7),min(26,cx+7)
    ids=torch.arange(begin,start+length,device='cuda')
    x=expand_features(case['features'][:,y0:y1,x0:x1].cuda(),ids)
    gt=case['targets'][start:start+length,:,cy*32:(cy+6)*32,cx*32:(cx+6)*32].cuda().float()[None]
    return x,ids,gt,start-begin,((cy-y0)*32,(cx-x0)*32)


def objective(pred,gt,lp,generator,fixed=False):
    mse=(pred-gt).square().mean()
    delta=pred.diff(dim=1)-gt.diff(dim=1)
    temporal=(delta.square()+1e-6).sqrt().mean()-1e-3
    count=min(4,pred.shape[1]);idx=list(range(count)) if fixed else generator.sample(range(pred.shape[1]),count)
    p=pred[:,idx].reshape(-1,pred.shape[2],*pred.shape[-2:])
    g=gt[:,idx].reshape_as(p)
    if pred.shape[2]==1:p=p.expand(-1,3,-1,-1);g=g.expand(-1,3,-1,-1)
    percept=lp(p,g).mean()
    info=dict(mse=float(mse.detach()),temporal=float(temporal.detach()),lpips=float(percept.detach()),mae=float((pred-gt).detach().abs().mean()))
    if pred.shape[2]==3:
        valid=gt.square().sum(2)>.25
        cosine=(F.normalize(pred.detach(),dim=2)*F.normalize(gt,dim=2)).sum(2).clamp(-1,1)
        info['angle_deg']=float(torch.rad2deg(cosine[valid].acos()).mean())
    return mse+.2*temporal+.2*percept,info


def forward_loss(model,case,rng,lp,fixed=False):
    x,ids,gt,warm,(oy,ox)=crop(case,rng,fixed)
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x,ids)[:,warm:,:,oy:oy+192,ox:ox+192]
    return objective(pred,gt,lp,rng,fixed)


@torch.no_grad()
def infer_video(model,features,chunk=4):
    model.eval();features=features.cuda();state={};result=[]
    for s in range(0,81,chunk):
        ids=torch.arange(s,min(s+chunk,81),device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):result.append(model(expand_features(features,ids),ids,state)[0])
    return torch.cat(result)


@torch.no_grad()
def quality(pred,gt,task):
    if task=='matte':return scalar_quality(pred[:,0],gt[:,0],task)
    p=pred.float().clamp(-1,1);g=gt.float();e=p-g
    valid=g.square().sum(1)>.25
    cosine=(F.normalize(p,dim=1)*F.normalize(g,dim=1)).sum(1).clamp(-1,1)
    angles=torch.rad2deg(cosine[valid].acos())
    # Zero-length predictions count as 90deg through cosine0, never disappear.
    edge=torch.zeros_like(valid)
    edge[...,1:]|=(g[...,1:]-g[...,:-1]).square().sum(1).sqrt()>.1
    edge[:,1:,:]|=(g[:,:,1:,:]-g[:,:,:-1,:]).square().sum(1).sqrt()>.1
    result=dict(mae=float(e.abs().mean()),rmse=float(e.square().mean().sqrt()),
        gradient_mae=float((e.diff(dim=-1).abs().mean()+e.diff(dim=-2).abs().mean())/2),
        temporal_delta_error=float(e.diff(dim=0).abs().mean()),
        angle_mean_deg=float(angles.mean()),angle_rmse_deg=float(angles.square().mean().sqrt()),
        angle_below_11_25=float((angles<11.25).float().mean()),angle_below_22_5=float((angles<22.5).float().mean()),
        angle_below_30=float((angles<30).float().mean()),valid_pixels=int(valid.sum()),
        valid_norm_error=float((p.square().sum(1).sqrt()[valid]-1).abs().mean()),
        patch_seam32=sum(seam_score(p[:,c]) for c in range(3))/3,
        patch_border_gradient_mae=float((e.diff(dim=-1)[...,31::32].abs().mean()+e.diff(dim=-2)[...,31::32,:].abs().mean())/2))
    if (edge&valid).any():result['gt_edge_region_mae']=float(e.abs().mean(1)[edge&valid].mean())
    return result


def structural_checks(model):
    model.eval();x=torch.randn(1,19,2,3,3072,device='cuda')*.5;ids=torch.arange(19,device='cuda')
    with torch.no_grad():
        full=model(x,ids);state={};chunks=[model(x[:,s:s+4],ids[s:s+4],state) for s in range(0,19,4)]
        stream=float((full-torch.cat(chunks,1)).abs().max())
        changed=x.clone();changed[:,10:]+=torch.randn_like(changed[:,10:])*10
        causal=float((full[:,:10]-model(changed,ids)[:,:10]).abs().max())
        assert stream<2e-5 and causal==0,(stream,causal)
    xx=x[:,:3].clone().requires_grad_(True)
    grad=torch.autograd.grad(model(xx,ids[:3])[0,-1,0,32,32],xx)[0]
    cross=float(grad[...,0,0,:].abs().sum());assert cross>0
    assert expand_features(torch.arange(21,device='cuda')[:,None,None,None],torch.arange(9,device='cuda')).flatten().tolist()==[0,1,1,1,1,2,2,2,2]
    # Verify the revised crop supervision covers all native480x832 pixels.
    coverage=torch.zeros(480,832,dtype=torch.bool)
    for cy in range(10):
        for cx in range(21):coverage[cy*32:(cy+6)*32,cx*32:(cx+6)*32]=True
    assert coverage.all()
    return dict(stream_full_max_error=stream,future_leak_max_error=causal,cross_token_gradient=cross,
                frame_alignment=True,full_image_supervision_coverage=True)


@torch.no_grad()
def export_examples(model,out,step):
    import numpy as np
    import subprocess
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    task=model.task;dest=out/f'visual_{step}';dest.mkdir(exist_ok=True)
    examples=json.loads((DATA/'visual_review/MANIFEST.json').read_text())['examples']
    lookup={r['sample']:r for r in rows('test')}
    final=step==json.loads((out/'REQUEST.json').read_text())['steps']
    selected=[(e['label'],lookup[e['sample']]) for e in examples] if final else [('validation',stratified_probe(rows('validation'),16)[0])]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    for label,row in selected:
        c=load_case(row,task);pred=infer_video(model,c['features'])
        values=[c['targets'],c['baseline'],pred];arrays=[]
        for a in values:
            a=a.float().clamp(-1,1)
            if task=='normal':a=F.normalize(a,dim=1) # visualization only, same rule for GT/VAE/pred
            a=((a+1)*127.5).round().byte()
            if a.shape[1]==1:a=a.expand(-1,3,-1,-1)
            arrays.append(a.permute(0,2,3,1).cpu().numpy())
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s','3328x528','-r','16','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','16',
            '-pix_fmt','yuv420p','-threads','4','-movflags','+faststart',str(dest/f'{label}_{task}.mp4')]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for fi in range(81):
                canvas=Image.new('RGB',(3328,528),(20,23,28));d=ImageDraw.Draw(canvas)
                canvas.paste(Image.open(Path(row['path'])/'rgb'/f'{fi+1:06d}.png').convert('RGB'),(0,48))
                for j,a in enumerate(arrays,1):canvas.paste(Image.fromarray(a[fi]),(j*832,48))
                for j,title in enumerate(['RGB',f'GT {task}','Original VAE',f'Detailer v13 | {step}']):d.text((j*832+12,12),title,font=font,fill='white')
                proc.stdin.write(np.asarray(canvas).tobytes())
                if fi==40:canvas.save(dest/f'{label}_frame41.png')
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
