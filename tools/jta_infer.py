"""JTA multi-person inference on all nine native-resolution tiles; no GT used.

Cross-tile duplicate suppression below is an explicit heuristic, not a trained
tracker. No cross-clip global identities or absolute 3D root are predicted.
"""
import argparse
import json
from pathlib import Path
from package_runtime import setup
setup()
import cv2
import numpy as np
import torch
from jta_tiles import TILES, LIMBS
from jta_cache import read_native_clip, atomic_json, atomic_torch
from jta_train import load_adapter, SCHEMA
from sparse_model import build_fresh_jta_model
from sparse_decode import decode_people


def to_plain(row, tile):
    count=len(row['slot']);people=[]
    for i in range(count):
        person=dict(tile=tile,slot=int(row['slot'][i]),score=float(row['score'][i]),
            xy_pixels=row['xy_pixels'][i].cpu().tolist(),
            in_crop_probability=row['in_crop_probability'][i].cpu().tolist(),
            visible_probability=row['visible_probability'][i].cpu().tolist(),source_tiles=[tile])
        if 'xyz_root_relative_m' in row:person['xyz_root_relative_m']=row['xyz_root_relative_m'][i].cpu().tolist()
        people.append(person)
    return people


def duplicate(a,b,minimum_shared=6,relative_threshold=.08):
    """Same-frame/cross-tile only; joint-correspondence median pixel distance."""
    if set(a['source_tiles'])&set(b['source_tiles']):return False
    x=np.asarray(a['xy_pixels']);y=np.asarray(b['xy_pixels'])
    valid=(np.asarray(a['in_crop_probability'])>.5)&(np.asarray(b['in_crop_probability'])>.5)
    valid&=(np.asarray(a['visible_probability'])>.5)&(np.asarray(b['visible_probability'])>.5)
    if valid.sum()<minimum_shared:return False
    # Native pixel scale, bounded threshold avoids swallowing close neighboring people.
    span=min(np.ptp(x[valid],axis=0).max(),np.ptp(y[valid],axis=0).max())
    threshold=float(np.clip(span*relative_threshold,3.,15.))
    return bool(np.median(np.linalg.norm(x[valid]-y[valid],axis=-1))<threshold)


def merge_tiles(people):
    def quality(p):return p['score']*float(np.mean(p['in_crop_probability']))
    kept=[]
    for person in sorted(people,key=quality,reverse=True):
        match=next((p for p in kept if duplicate(p,person)),None)
        if match is None:kept.append(dict(person))
        else:match['source_tiles']=sorted(set(match['source_tiles']+person['source_tiles']))
    return kept


def render(video,start,frames,output):
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,start-1)
    fps=cap.get(cv2.CAP_PROP_FPS);writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080))
    if not writer.isOpened():raise RuntimeError('Unable to open MP4 writer')
    try:
        for people in frames:
            ok,image=cap.read()
            if not ok:raise RuntimeError('Failed to read visualization frame')
            for index,p in enumerate(people):
                xy=np.asarray(p['xy_pixels']);valid=np.asarray(p['in_crop_probability'])>.5
                valid&=(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
                for j,k in LIMBS:
                    if valid[j] and valid[k]:cv2.line(image,tuple(np.rint(xy[j]).astype(int)),tuple(np.rint(xy[k]).astype(int)),(0,255,0),1,cv2.LINE_AA)
                for j in np.flatnonzero(valid):cv2.circle(image,tuple(np.rint(xy[j]).astype(int)),2,(0,180,255),-1)
            writer.write(image)
    finally:cap.release();writer.release()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--video',required=True)
    p.add_argument('--start-frame',type=int,default=1);p.add_argument('--task',choices=['2d','3d'],default='3d')
    p.add_argument('--threshold',type=float,default=.5);p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',required=True);p.add_argument('--render',action='store_true')
    a=p.parse_args()
    if a.start_frame<1 or not 0<a.threshold<1:p.error('Invalid frame/threshold')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if state.get('schema')!=SCHEMA or state['contract'].get('attention')!='bidirectional':raise ValueError('Expected new JTA bidirectional checkpoint')
    if state['step']<1:raise ValueError('Step0 is an untrained initialization')
    device=torch.device(a.device);torch.cuda.set_device(device);root=Path(a.project_root)
    from infer_wan22_dense4 import load_official_vae
    model=build_fresh_jta_model(str(root/'models/Wan2.2-TI2V-5B'),device,state['contract']['queries'])
    load_adapter(model,state['weights']);model.requires_grad_(False).eval();del state
    vae=load_official_vae(root/'code/Wan2.2',root/'models/Wan2.2-TI2V-5B',device)
    task=int(a.task=='3d');raw_per_tile=[];frames=[[] for _ in range(81)]
    with torch.no_grad():
        for tile,origin in enumerate(TILES):
            row=dict(video=a.video,start_frame=a.start_frame,num_frames=81,origin=origin,id=f'inference_tile_{tile}')
            rgb=read_native_clip(row)
            source=torch.from_numpy(rgb.copy()).permute(3,0,1,2).to(device=device,dtype=torch.bfloat16)/127.5-1
            z=vae.encode([source])[0][None]
            x,y=origin;k=torch.tensor([[[1158.,0,960-x],[0,1158,540-y],[0,0,1.]]],device=device)
            with torch.autocast(device.type,dtype=torch.bfloat16):pred,_=model(z,k,task,False)
            model.task_router.clear()
            decoded=decode_people(pred,torch.tensor([origin],device=device),threshold=a.threshold)[0]
            for ti,people in enumerate(decoded):frames[ti].extend(to_plain(people,tile))
            raw_per_tile.append(dict(tile=tile,origin=origin,prediction={k:v.cpu() for k,v in pred.items()}))
            print(json.dumps(dict(event='tile_complete',tile=tile)),flush=True)
    merged=[merge_tiles(people) for people in frames]
    atomic_json(out/'prediction.json',dict(video=a.video,start_frame_1based=a.start_frame,frames=merged,
        task=a.task,presence_threshold=a.threshold,merge='heuristic cross-tile pose suppression; not global tracking',
        xyz='camera axes, mid-hip-relative meters; no absolute root or world coordinates',
        source_size=[1920,1080],tiles=TILES,resize=False))
    atomic_torch(out/'raw_predictions.pt',raw_per_tile)
    if a.render:render(a.video,a.start_frame,merged,out/'prediction.mp4')


if __name__=='__main__':main()
