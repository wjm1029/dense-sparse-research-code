"""Two real RGB cases, native OPSD5250 plus final depth/normal detailers and VAE controls."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import torch
from torch.nn import functional as F
import run_probe as v1
import run_probe_v3b as native
from benchmark_decode_v14 import rollout_gpu
from pixvid_detailer_v12 import Detailer as DepthHead, infer_video as depth_infer
from modalities_v13 import Detailer as NormalHead, infer_video as normal_infer

ROOT=v1.EXP/'real_case_v16_20261124'
OUT=ROOT/'outputs'


def atomic(name,value):
    path=OUT/name;tmp=path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)


def emit(event,**kw):
    d=dict(event=event,time=time.time(),**kw);print(json.dumps(d),flush=True);atomic('STATUS.json',d)


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();x=fn();torch.cuda.synchronize()
    return x,time.perf_counter()-start


def export_video(path,render,frames=81,width=2496,height=1096):
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
         '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','16','-i','pipe:0',
         '-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p',
         '-threads','4','-movflags','+faststart',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i in range(frames):
            canvas=render(i);proc.stdin.write(np.asarray(canvas,dtype=np.uint8).tobytes())
            if i in [0,40,80] and height>480:canvas.save(path.with_name(path.stem+f'_frame{i+1:02d}.png'))
        proc.stdin.close();assert proc.wait()==0
    finally:
        if proc.poll() is None:proc.kill();proc.wait()


@torch.no_grad()
def main():
    OUT.mkdir(parents=True,exist_ok=True);native.RUN=OUT
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest');v1.TASKS['normal']=1
    inputs=json.loads((ROOT/'inputs/INPUTS.json').read_text())
    lut_meta=json.loads((v1.EXP/'spectral_lut_v15.json').read_text())
    lut=np.asarray(lut_meta['rgb_uint8'],dtype=np.uint8)
    report=dict(checkpoint_sha256=native.SHA,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        model='Dense6OPSDModule, step5250, chunk3, sink3/local12, predicted latent history',
        layout='Rows: depth / normal. Columns: RGB / OPSD+official VAE / OPSD+final detailer',
        depth_mapping={k:v for k,v in lut_meta.items() if k!='rgb_uint8'},
        normal_visualization='Clamp[-1,1], normalize signed xyz, map to RGB; identical for both routes',
        preprocessing='Decode supplied RGB mp4 without crop/resize/resampling; preserve all81frames and16fps playback',
        ground_truth=False,old_bidirectional_exports_used=False,cases=[])
    emit('loading_native')
    (module,vae,contract),load_seconds=timed(native.load_native)
    report['native_load_seconds']=load_seconds
    heads={}
    for task in ['depth','normal']:
        p=v1.EXP/('pixvid_detailer_v12/depth/head_30000.pt' if task=='depth' else 'pixvid_modalities_v13/normal/head_30000.pt')
        head=(DepthHead() if task=='depth' else NormalHead('normal')).cuda().eval()
        head.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'],strict=True)
        heads[task]=head;report[task+'_head']=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for case in inputs['cases']:
        emit('reading_rgb',case=case['id'])
        path=ROOT/'inputs'/case['input_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==case['sha256']
        reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24');meta=next(reader)
        assert tuple(meta['size'])==(832,480) and meta['fps']==16
        rgb=np.stack([np.frombuffer(buf,dtype=np.uint8).reshape(480,832,3) for buf in reader])
        assert rgb.shape==(81,480,832,3)
        source=torch.from_numpy(rgb.copy()).permute(3,0,1,2).cuda().to(torch.bfloat16)/127.5-1
        z,encode_seconds=timed(lambda:vae.encode([source])[0]);assert z.shape==(48,21,30,52)
        result=dict(input=case,encode_seconds=encode_seconds,tasks={});visuals={}
        for task in ['depth','normal']:
            emit('inference',case=case['id'],task=task)
            prompt='output: depth' if task=='depth' else 'output: surface normal';pid=contract['prompts'].index(prompt)
            (latent,features),dit_seconds=timed(lambda:rollout_gpu(module,z,task,pid,True))
            vae_pred,vae_seconds=timed(lambda:vae.decode([latent])[0].float())
            fn=depth_infer if task=='depth' else normal_infer
            pred,head_seconds=timed(lambda:fn(heads[task],features))
            vae_pred=vae_pred.mean(0) if task=='depth' else vae_pred.permute(1,0,2,3)
            assert pred.shape==vae_pred.shape
            assert torch.isfinite(pred).all() and torch.isfinite(vae_pred).all()
            x=pred.float();y=vae_pred.float()
            info=dict(dit_seconds=dit_seconds,vae_decode_seconds=vae_seconds,new_decode_seconds=head_seconds,
                      timing_note='Single un-warmed real-case pass, not controlled benchmark',
                      prediction_vs_vae_mae=float((x.clamp(-1,1)-y.clamp(-1,1)).abs().mean()),
                      prediction_range=[float(x.min()),float(x.max())],
                      prediction_fraction_outside_signed_unit=float((x.abs()>1).float().mean()),
                      these_are_not_accuracy_metrics=True)
            arrays=[]
            for a in [y,x]:
                a=a.clamp(-1,1)
                if task=='depth':arr=lut[((a+1)*.5*(len(lut)-1)).round().long().cpu().numpy()]
                else:
                    a=F.normalize(a,dim=1)
                    arr=((a+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
                arrays.append(arr)
            visuals[task]=arrays;result['tasks'][task]=info
            del latent,features,vae_pred,pred,x,y
        emit('exporting',case=case['id'])
        name=case['id']+'_depth_normal_comparison.mp4'
        def render(i):
            canvas=Image.new('RGB',(2496,1096),(20,23,28));d=ImageDraw.Draw(canvas)
            for task,base_y in [('depth',0),('normal',524)]:
                arrays=[rgb,*visuals[task]]
                for j,arr in enumerate(arrays):canvas.paste(Image.fromarray(arr[i]),(832*j,base_y+44))
                for j,title in enumerate([f'RGB | {task}','OPSD + official VAE','OPSD + final detailer']):
                    d.text((832*j+14,base_y+8),title,font=font,fill='white')
            d.text((14,1060),f'{case["id"]} | frame {i+1:02d}/81 | 16 fps playback | real video, no GT',font=small,fill='white')
            d.text((1340,1060),'Depth: fixed Spectral [0,1]',font=small,fill='white')
            bar=lut[np.rint(np.linspace(0,len(lut)-1,540)).astype(int)][None].repeat(18,axis=0)
            canvas.paste(Image.fromarray(bar),(1860,1058));d.text((1825,1056),'0',font=small,fill='white');d.text((2414,1056),'1',font=small,fill='white')
            return canvas
        export_video(OUT/name,render)
        result['comparison_video']=name
        for task in ['depth','normal']:
            suffix='depth_spectral' if task=='depth' else 'normal'
            name=case['id']+'_'+suffix+'_new.mp4'
            export_video(OUT/name,lambda i:visuals[task][1][i],width=832,height=480)
            result['tasks'][task]['new_head_video']=name
        report['cases'].append(result);atomic('RESULTS.json',report)
        emit('case_complete',case=case['id'])
        del source,z,rgb,visuals;gc.collect()
    atomic('COMPLETE.json',report);emit('complete')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);atomic('FAILED.json',dict(error=repr(e)));raise
