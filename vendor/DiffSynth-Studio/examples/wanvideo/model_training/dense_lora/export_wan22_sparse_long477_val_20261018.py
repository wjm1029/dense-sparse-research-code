"""Export all held-out sparse OPSD student rollouts as side-by-side MP4 cases.

GT is used only for metrics/drawing, never passed to the prediction rollout.
Outputs are written to PFS; no large temporary files on the nearly full host disk.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from safetensors import safe_open
from safetensors.torch import load_file
from train_wan22_sparse_ddp import sha
from train_wan22_sparse_prompt_full_ddp import coordinate_loss
from wan22_sparse_long477 import LongSparseDataset,LongSparseModule
from wan22_sparse_task_router import TASK_PROMPTS

GREEN=(18,210,82); MAGENTA=(245,55,190)
FONT_PATH='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT=ImageFont.truetype(FONT_PATH,17)
SMALL=ImageFont.truetype(FONT_PATH,14)

def skeleton(image,points,parents,valid,color,width=2,radius=2):
    draw=ImageDraw.Draw(image)
    for j,parent in enumerate(parents):
        if parent>=0 and valid[j] and valid[parent]:
            draw.line([tuple(points[j]),tuple(points[parent])],fill=color,width=width)
    for j,(x,y) in enumerate(points):
        if valid[j]: draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color)

def projection_bounds(gt,pred,axes):
    points=np.concatenate((gt[...,axes].reshape(-1,2),pred[...,axes].reshape(-1,2)),0)
    assert np.isfinite(points).all()
    lo=points.min(0); hi=points.max(0)
    return (lo+hi)/2,np.maximum(hi-lo,.2)*1.15

def draw3d(canvas,gt,pred,parents,axes,bounds,origin,title):
    x,y=origin; w,h=600,270
    draw=ImageDraw.Draw(canvas); draw.rectangle((x,y,x+w,y+h),fill=(26,30,39),outline=(65,75,90))
    draw.text((x+12,y+8),title,font=SMALL,fill=(230,235,245))
    center,span=bounds; scale=min((w-50)/span[0],(h-55)/span[1])
    for values,color in ((gt,GREEN),(pred,MAGENTA)):
        points=(values[:,axes]-center)*scale+np.array([x+w/2,y+h/2+12])
        skeleton(canvas,points,parents,np.ones(24,dtype=bool),color,2,3)

def render(path,clip,parents,xy,xyz,mask,pred2,pred3,metrics,case_index):
    metadata=json.loads((clip/'metadata.json').read_text())
    fps=float(metadata['video']['fps']); assert 0<fps<=120
    temp=path.with_name(path.stem+'.partial.mp4')
    process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
      '-pix_fmt','rgb24','-s','1248x640','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264',
      '-preset','fast','-crf','20','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
    front=projection_bounds(xyz,pred3,[0,1]); side=projection_bounds(xyz,pred3,[2,1])
    try:
        for frame in range(477):
            with Image.open(clip/'rgb'/f'{frame+1:06d}.png') as raw:
                rgb=raw.convert('RGB').resize((416,240),Image.Resampling.LANCZOS)
            canvas=Image.new('RGB',(1248,640),(15,18,25)); draw=ImageDraw.Draw(canvas)
            draw.text((12,7),f'Validation {case_index+1:02d}/27 | OPSD student step2625 | frame {frame+1:03d}/477',font=FONT,fill=(240,244,250))
            draw.text((12,30),f'2D {metrics[0]:.3f}px | 3D {metrics[1]:.5f} x actor height | GT green / prediction magenta',font=SMALL,fill=(200,210,225))
            for col,label in enumerate(('RGB + GT 2D','RGB + predicted 2D','RGB + GT / predicted 2D')):
                draw.text((col*416+10,53),label,font=SMALL,fill=(235,240,248))
            gt_points=xy[frame]*[416,240]; predicted_points=pred2[frame]*[416,240]
            # Prediction visibility never comes from GT validity.
            pred_valid=np.isfinite(pred2[frame]).all(-1)&((pred2[frame]>=0)&(pred2[frame]<=1)).all(-1)
            left=rgb.copy(); middle=rgb.copy(); right=rgb.copy()
            skeleton(left,gt_points,parents,mask[frame],GREEN)
            skeleton(middle,predicted_points,parents,pred_valid,MAGENTA)
            skeleton(right,gt_points,parents,mask[frame],GREEN)
            skeleton(right,predicted_points,parents,pred_valid,MAGENTA)
            for col,image in enumerate((left,middle,right)): canvas.paste(image,(col*416,76))
            draw3d(canvas,xyz[frame],pred3[frame],parents,[0,1],front,(12,340),'3D front: camera X right / Y down')
            draw3d(canvas,xyz[frame],pred3[frame],parents,[2,1],side,(636,340),'3D side: camera Z right / Y down')
            draw.text((12,617),clip.parent.name+'/'+clip.name,font=SMALL,fill=(170,185,205))
            process.stdin.write(canvas.tobytes())
            if frame==238: canvas.save(path.with_suffix('.jpg'),quality=90)
        process.stdin.close()
        if process.wait()!=0: raise RuntimeError('ffmpeg failed: '+str(path))
    except BaseException:
        if process.stdin and not process.stdin.closed: process.stdin.close()
        if process.poll() is None: process.terminate(); process.wait()
        raise
    assert not path.exists(), 'Refuse completed-video overwrite'
    os.replace(temp,path)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
      '-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)],text=True))['streams'][0]
    assert int(info['nb_read_frames'])==477 and info['width']==1248 and info['height']==640
    return fps,info

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output_dir',required=True)
    p.add_argument('--checkpoint',default='/pfs/mayuema/dense/wan/outputs/wan22_sparse_long477_opsd_local12_2625_20261016/step-2625.safetensors')
    p.add_argument('--index_dir',default='/pfs/mayuema/dense/wan/cache/wan22_sparse_long477_coordinates_v1_20261016')
    a=p.parse_args(); rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0')); torch.cuda.set_device(local); torch.set_num_threads(1)
    device=torch.device('cuda',local); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    with safe_open(a.checkpoint,framework='pt',device='cpu') as f: metadata=f.metadata()
    contract=json.loads(metadata['contract']); assert metadata['step']=='2625' and contract['phase']=='opsd_local'
    assert contract['index_hash']==sha(Path(a.index_dir)/'manifest.json')
    here=Path(__file__).resolve().parent
    for name,checksum in contract['source_hashes'].items(): assert sha(here/name)==checksum,name
    assert contract['pipeline_hash']==sha('diffsynth/pipelines/wan_video.py')
    assert contract['dit_hash']==sha('diffsynth/models/wan_video_dit.py')
    dataset=LongSparseDataset(a.index_dir,True); assert len(dataset)==27
    checkpoint_hash=sha(a.checkpoint)
    module=LongSparseModule(contract['model_dir'],'opsd_local'); module.prepare(device)
    weights=load_file(a.checkpoint,device='cpu')
    expected={n for n,q in module.named_parameters() if q.requires_grad}|{'context','task_router.task_feature_table','task_router.features_ready','sparse_head.xy_grid'}
    assert set(weights)==expected
    for key in ('context','task_router.task_feature_table','sparse_head.xy_grid'):
        assert torch.equal(weights[key],module.state_dict()[key].cpu()),key
    result=module.load_state_dict(weights,strict=False); assert not result.unexpected_keys
    del weights; module.eval(); start=time.monotonic()
    print(f'EXPORT_INITIALIZED rank={rank} total_val=27 checkpoint=2625 student=True bounded_local12=True',flush=True)
    for index in range(rank,len(dataset),world):
        data_index=int(dataset.indices[index]); cid=dataset.manifest['ids'][data_index]
        stem=f'{index+1:02d}_'+cid.replace('/','_'); video=out/(stem+'.mp4'); record=out/(stem+'.json')
        if record.exists():
            previous=json.loads(record.read_text()); assert previous['checkpoint_sha256']==checkpoint_hash and video.exists()
            continue
        source,xy,xyz,mask=dataset[index]
        clip=Path(dataset.manifest['dataset_root'])/cid
        with np.load(clip/'genception/keypoints_2d/keypoints_2d.npz',allow_pickle=False) as labels:
            parents=labels['parents'].astype(int)
            assert len(parents)==24
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            source=source[None].to(device)
            pred2=module.rollout(source,TASK_PROMPTS[0]); pred3=module.rollout(source,TASK_PROMPTS[1])
        assert pred2.shape==(1,477,24,2) and pred3.shape==(1,477,24,3)
        assert torch.isfinite(pred2).all() and torch.isfinite(pred3).all()
        xy_gpu,xyz_gpu,mask_gpu=[v[None].to(device) for v in (xy,xyz,mask)]
        _,error2=coordinate_loss(pred2,xy_gpu,xyz_gpu,mask_gpu,0)
        _,error3=coordinate_loss(pred3,xy_gpu,xyz_gpu,mask_gpu,1)
        errors=(float(error2),float(error3))
        arrays=[v.cpu().numpy() for v in (xy,xyz,mask)]
        predictions=[pred2[0].cpu().float().numpy(),pred3[0].cpu().float().numpy()]
        np.savez_compressed(out/(stem+'.npz'),prediction_2d=predictions[0],prediction_3d=predictions[1],
          target_2d=arrays[0],target_3d=arrays[1],mask_2d=arrays[2],parents=parents,frame_numbers=np.arange(1,478))
        fps,probe=render(video,clip,parents,*arrays,*predictions,errors,index)
        payload={'validation_index':index,'clip_id':cid,'checkpoint':a.checkpoint,'checkpoint_sha256':checkpoint_hash,
          'student_not_ema':True,'prompts':list(TASK_PROMPTS),'frames':477,'fps':fps,'error_2d_pixels':errors[0],
          'error_3d_actor_height':errors[1],'video':video.name,'poster':stem+'.jpg','coordinates':stem+'.npz',
          'video_sha256':sha(video),'probe':probe,'inference_history':'predicted only; GT never model input',
          'visualization':'GT green, prediction magenta; 3D root-relative camera axes, actor-height units'}
        temp=record.with_suffix('.pending'); temp.write_text(json.dumps(payload,indent=2)); os.replace(temp,record)
        print(f'CASE_COMPLETE rank={rank} index={index+1}/27 clip={cid} 2d_px={errors[0]:.4f} 3d_height={errors[1]:.6f} elapsed={time.monotonic()-start:.1f}',flush=True)
    (out/f'RANK_{rank}_COMPLETE').write_text('success\n')

if __name__=='__main__': main()
