"""Held-out Dense6 long free-rollout metrics plus six-task decoded comparison videos."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from safetensors.torch import load_file
from wan22_cache_integrity import sha256_file as sha
from wan22_dense6_long477 import LongDenseModule,LongDenseDataset,MAIN
from wan22_dense6_task_router import TASK_TO_ID,ID_TO_TASK
from infer_wan22_dense4 import load_official_vae

FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)

@torch.inference_mode()
def decode(vae,latent,device):
    video=vae.decode([latent.to(device=device,dtype=torch.bfloat16)])[0]
    assert video.shape==(3,477,480,832) and torch.isfinite(video).all()
    return video.detach().cpu().to(torch.float32).contiguous()

def display_frame(value,task):
    value=np.clip(value,-1,1)
    if task in ('depth','matte'):
        image=np.repeat(np.mean(value,axis=0)[...,None],3,-1)
        return np.rint((image+1)*127.5).astype(np.uint8)
    if task=='optical_flow':
        import cv2
        u,v=value[0]*32,value[1]*32
        mag=np.sqrt(u*u+v*v)
        hue=np.mod((np.arctan2(v,u)+np.pi)*90/np.pi,180).astype(np.uint8)
        hsv=np.stack([hue,np.full_like(hue,255),np.clip(mag/20*255,0,255).astype(np.uint8)],-1)
        rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        rgb[value[2]<=0]=0
        return rgb
    return np.rint((np.moveaxis(value,0,-1)+1)*127.5).astype(np.uint8)

def render(path,clip,gt,pred,task,step,mse):
    fps=float(json.loads((clip/'metadata.json').read_text())['video']['fps'])
    temp=path.with_name(path.stem+'.partial.mp4')
    ff=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
      '-s','1248x280','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20',
      '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    try:
        for t in range(477):
            image=Image.new('RGB',(1248,280),(15,18,25)); draw=ImageDraw.Draw(image)
            draw.text((8,4),f'step{step} | {task} | frame{t+1}/477 | latent MSE {mse:.5f}',font=FONT,fill='white')
            for x,title in ((0,'RGB input'),(416,'GT target (VAE reconstruction)'),(832,'Prediction (free-running history)')):
                draw.text((x+8,22),title,font=FONT,fill='white')
            with Image.open(clip/'rgb'/f'{t+1:06d}.png') as raw: rgb=raw.convert('RGB').resize((416,240),Image.Resampling.LANCZOS)
            image.paste(rgb,(0,40))
            for x,value in ((416,gt[:,t].numpy()),(832,pred[:,t].numpy())):
                panel=Image.fromarray(display_frame(value,task)).resize((416,240),Image.Resampling.BILINEAR)
                image.paste(panel,(x,40))
            ff.stdin.write(image.tobytes())
            if t==238: image.save(path.with_suffix('.jpg'),quality=90)
        ff.stdin.close(); assert ff.wait()==0
    except BaseException:
        if ff.stdin and not ff.stdin.closed: ff.stdin.close()
        if ff.poll() is None: ff.terminate(); ff.wait()
        raise
    assert not path.exists(); os.replace(temp,path)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
      '-show_entries','stream=nb_read_frames,width,height','-of','json',str(path)],text=True))['streams'][0]
    assert int(probe['nb_read_frames'])==477
    return fps,sha(path)

def decoded_metrics(pred,gt,task,clip):
    sums=0.; absolute=0.; count=0; angle_sum=0.; angle_count=0
    epe_sum=0.; moving_sum=0.; flow_count=0; moving_count=0
    from build_wan22_albedo_flow_target_cache import _read_exr
    for start in range(0,477,8):
        p=pred[:,start:start+8].clamp(-1,1); g=gt[:,start:start+8].clamp(-1,1)
        diff=p-g; sums+=float(diff.square().sum()); absolute+=float(diff.abs().sum()); count+=diff.numel()
        if task=='normal':
            norm=torch.linalg.vector_norm(g,dim=0); valid=norm>.2
            cosine=torch.nn.functional.cosine_similarity(p,g,dim=0).clamp(-1,1)
            angle_sum+=float(torch.rad2deg(torch.acos(cosine))[valid].sum()); angle_count+=int(valid.sum())
        if task=='optical_flow':
            for j in range(p.shape[1]):
                t=start+j
                if t==0: continue
                raw=_read_exr(clip/'optical_flow'/f'{t:06d}.exr')
                uv=torch.from_numpy(raw[...,[2,1]].copy()).permute(2,0,1)
                valid=torch.from_numpy(raw[...,0]>.5)
                error=torch.linalg.vector_norm(p[:2,j]*32-uv,dim=0)
                moving=valid&(torch.linalg.vector_norm(uv,dim=0)>1)
                epe_sum+=float(error[valid].sum()); flow_count+=int(valid.sum())
                moving_sum+=float(error[moving].sum()); moving_count+=int(moving.sum())
    result={'decoded_mse_vs_vae_target':sums/count,'decoded_mae_vs_vae_target':absolute/count}
    if angle_count: result['normal_angle_deg_vs_vae_target']=angle_sum/angle_count
    if flow_count: result.update(raw_flow_epe_px=epe_sum/flow_count,raw_moving_flow_epe_px=moving_sum/max(1,moving_count))
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output_dir',required=True); a=p.parse_args()
    out=Path(a.output_dir); request=json.loads((out/'REQUEST.json').read_text()); contract=request['contract']
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); torch.set_num_threads(1); device=torch.device('cuda',local)
    assert torch.cuda.mem_get_info(device)[0]>45*2**30,'Insufficient free GPU memory; preserve other jobs'
    checkpoint=Path(request['checkpoint']); assert sha(checkpoint)==request['checkpoint_sha256']
    here=Path(__file__).resolve().parent
    for name,checksum in contract['scripts'].items(): assert sha(here/name)==checksum,name
    assert sha('diffsynth/pipelines/wan_video.py')==contract['pipeline_sha256']
    assert sha('diffsynth/models/wan_video_dit.py')==contract['dit_sha256']
    dataset=LongDenseDataset('/pfs/mayuema/dense/wan/cache/wan22_dense6_long477_extra_flow32_v1_20261017',contract['prompts'],True)
    assert dataset.identities==contract['cache_manifests'] and [r['id'] for r in dataset.rows]==[r['id'] for r in request['validation']]
    weights=load_file(str(checkpoint),device='cpu'); contexts=weights.pop('condition_contexts')
    module=LongDenseModule(contract['model_dir'])
    # Text embeddings are saved in this native long checkpoint; no re-encoding.
    module.pipe.text_encoder=None; module.pipe.tokenizer=None
    module.context_table=contexts.to(device=device,dtype=torch.bfloat16)
    module.pipe.dit.to(device); module.task_router.to(device); module.pipe.device=device
    names={n for n,q in module.named_parameters() if q.requires_grad}|{n for n in module.state_dict() if n.startswith('task_router.')}
    assert set(weights)==names
    result=module.load_state_dict(weights,strict=False); assert not result.unexpected_keys
    del weights; module.eval(); torch.cuda.empty_cache()
    vae=load_official_vae(Path('/pfs/mayuema/dense/wan/Wan2.2'),Path(contract['model_dir']),device)
    start=time.monotonic(); print(f'DENSE_LONG_EVAL_INITIALIZED rank={rank} step={request["step"]} val=41',flush=True)
    raw_root=Path('/pfs/mayuema/dense/long481_prod3000_20260906/outputs/train')
    for i in range(rank,len(dataset),world):
        case=i//6; task_id=i%6; task=ID_TO_TASK[task_id]; cid=dataset.rows[case]['id']
        case_out=out/f'{case+1:02d}_{cid.replace("/","_")}'; case_out.mkdir(exist_ok=True)
        record=case_out/f'{task}.json'
        if record.exists(): continue
        source,target,_,pid,_=dataset[i]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred=module.rollout(source[None].to(device),torch.tensor([task_id],device=device),torch.tensor([pid],device=device))[0]
        pred=pred.cpu().to(torch.bfloat16); assert torch.isfinite(pred).all()
        frame_error=(pred.float()-target.float()).square().mean((0,2,3))
        row={'step':request['step'],'clip_id':cid,'validation_index':case,'task':task,'prompt':contract['prompts'][pid],
          'latent_mse':float(frame_error.mean()),'mse_first_quarter':float(frame_error[:30].mean()),
          'mse_last_quarter':float(frame_error[90:].mean()),'scope':'41_clip_holdout_predicted_history_only',
          'preview':cid in request['preview_ids']}
        if row['preview']:
            torch.cuda.empty_cache()
            pv=decode(vae,pred,device); gv=decode(vae,target,device)
            row.update(decoded_metrics(pv,gv,task,raw_root/cid))
            video=case_out/f'{task}.mp4'
            row['fps'],row['video_sha256']=render(video,raw_root/cid,gv,pv,task,request['step'],row['latent_mse'])
            row['video']=str(video.relative_to(out)); row['poster']=str(video.with_suffix('.jpg').relative_to(out))
            del pv,gv
        temp=record.with_suffix('.pending'); temp.write_text(json.dumps(row,indent=2)); os.replace(temp,record)
        print('DENSE_EVAL_CASE '+json.dumps(row)+f' elapsed={time.monotonic()-start:.1f}',flush=True)
    (out/f'RANK_{rank}_COMPLETE').write_text('success\n')

if __name__=='__main__': main()
