"""Add scale32 delayed flow, albedo and metadata-derived raymaps to fixed long RGB index.

Read-only source data. Official Wan2.2 whole-477 encode, no independent windows.
The raymap matches the existing quantized Rothko target convention, including
full-source-video origin scale. No image/pose labels are written into raw data.
"""
import argparse
from collections import Counter
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
sys.path.insert(0,'/pfs/mayuema/dense/code')
from backfill_genception_camera import camera_arrays
from materialize_genception_raymaps import base_directions
import build_wan22_flow32_target_cache as reader
from wan22_cache_integrity import sha256_file as sha, sha256_json

SCHEMA='wan22_dense6_long477_extra_v1'
SHAPE=(48,120,30,52)
TASKS=('albedo','optical_flow','raymap')
PROMPTS=('output: albedo','output: optical flow','output: camera raymap')
TASK_IDS=(4,5,3)

def atomic_json(path,data):
    temp=path.with_name(path.name+f'.pending-{os.getpid()}')
    temp.write_text(json.dumps(data,indent=2)); os.replace(temp,path)

def load_video(clip,task):
    video=torch.empty((477,3,480,832),dtype=torch.bfloat16)
    stats=Counter()
    if task=='raymap':
        metadata=json.loads((clip/'metadata.json').read_text())
        assert metadata['video']['frames']==481
        assert [r['frame'] for r in metadata['frames_data']]==list(range(1,482))
        arrays=camera_arrays(metadata)
        # Match the established float32 camera-pose / quantized PNG path.
        intrinsics=arrays['K'].astype(np.float32)
        relative=arrays['relative_camera_to_world_opencv'].astype(np.float32)
        centers=arrays['camera_centers_reference'].astype(np.float32)
        scale=max(float(np.abs(centers).max()),1e-6)
        cached_K=None; directions_camera=None
    for i in range(477):
        if task=='albedo':
            raw=reader._read_exr(clip/task/f'{i+1:06d}.exr')
            assert raw.shape==(480,832,3)
            value=np.clip(raw[...,::-1],0,1)*2-1
        elif task=='optical_flow':
            value=np.zeros((480,832,3),dtype=np.float32)
            if i==0: value[...,2]=-1
            else:
                raw=reader._read_exr(clip/task/f'{i:06d}.exr')
                assert raw.shape==(480,832,4)
                u,v,valid=raw[...,2],raw[...,1],raw[...,0]>0.5
                assert np.max(np.abs(raw[...,0]-valid.astype(np.float32)))<1e-6
                assert not np.any((~valid)&((np.abs(u)>1e-6)|(np.abs(v)>1e-6)))
                value[...,0]=np.clip(u/32.,-1,1); value[...,1]=np.clip(v/32.,-1,1)
                value[...,2]=valid.astype(np.float32)*2-1
                stats['valid_pixels']+=int(valid.sum())
                stats['clipped_u']+=int((valid&(np.abs(u)>32)).sum())
                stats['clipped_v']+=int((valid&(np.abs(v)>32)).sum())
        else:
            K=intrinsics[i]
            if cached_K is None or not np.array_equal(cached_K,K):
                directions_camera=base_directions(K,480,832); cached_K=K.copy()
            directions=directions_camera@relative[i,:3,:3].T
            directions/=np.maximum(np.linalg.norm(directions,axis=-1,keepdims=True),1e-12)
            value=np.clip((directions+1)*.5,0,1)
            value[120:360,208:624]=np.clip((centers[i]/scale+1)*.5,0,1)
            value=np.rint(value*255).astype(np.uint8).astype(np.float32)/255*2-1
        assert np.isfinite(value).all()
        video[i].copy_(torch.from_numpy(value.copy()).permute(2,0,1))
    return video,dict(stats)

def valid(path,contract,cid,task):
    if not path.exists(): return False
    p=torch.load(path,map_location='cpu',weights_only=False); z=p['latent']
    assert p['cache_contract_hash']==contract and p['clip_id']==cid and p['modality']==task
    assert z.shape==SHAPE and z.dtype==torch.bfloat16 and torch.isfinite(z).all()
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache_dir',required=True)
    p.add_argument('--limit_per_rank',type=int,default=0,help='Preflight prefix; never writes CACHE_COMPLETE')
    a=p.parse_args(); rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    dist.init_process_group('gloo',timeout=timedelta(hours=24))
    torch.set_num_threads(1)
    root=Path('/pfs/mayuema/dense/long481_prod3000_20260906/outputs/train')
    source=Path('/pfs/mayuema/dense/wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40')
    index=json.loads((source/'source_index.json').read_text()); rows=index['samples']
    source_manifest=json.loads((source/'manifest.json').read_text())
    assert json.loads((source/'CACHE_COMPLETE').read_text())['manifest_sha256']==sha(source/'manifest.json')
    out=Path(a.cache_dir); out.mkdir(parents=True,exist_ok=True)
    manifest={'schema':SCHEMA,'source_cache':str(source),'dataset_root':str(root),
      'source_manifest_sha256':sha(source/'manifest.json'),'source_index_sha256':sha(source/'source_index.json'),
      'source_contract':source_manifest['cache_contract_hash'],'num_clips':len(rows),'tasks':list(TASKS),
      'shape':list(SHAPE),'frames':[1,477],'source_frames':481,'flow_scale_px':32.,
      'flow_alignment':'first invalid; target_t = forward_flow_(t-1)_to_t; original source grid',
      'raymap':'metadata first-camera Rothko; float32; origin scale over all481; round8bit; center[208,120,624,360]',
      'vae_checkpoint_sha256':source_manifest['vae_checkpoint_sha256'],
      'vae_source_sha256':source_manifest['official_vae_source_sha256'],
      'builder_sha256':sha(__file__),'reader_sha256':sha(reader.__file__),
      'camera_code_sha256':sha('/pfs/mayuema/dense/code/backfill_genception_camera.py'),
      'raymap_code_sha256':sha('/pfs/mayuema/dense/code/materialize_genception_raymaps.py')}
    manifest['cache_contract_hash']=sha256_json(manifest); contract=manifest['cache_contract_hash']
    if rank==0:
        if (out/'manifest.json').exists(): assert json.loads((out/'manifest.json').read_text())==manifest
        else: atomic_json(out/'manifest.json',manifest)
    dist.barrier()
    official=Path('/pfs/mayuema/dense/wan/Wan2.2')
    model=Path('/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B')
    assert sha(model/'Wan2.2_VAE.pth')==manifest['vae_checkpoint_sha256']
    assert sha(official/'wan/modules/vae2_2.py')==manifest['vae_source_sha256']
    vae=reader.load_official_vae(argparse.Namespace(official_repo=str(official),model_dir=str(model)),device)
    reader.LATENT_SHAPE=SHAPE
    start=time.monotonic()
    assigned=rows[rank::world]
    if a.limit_per_rank: assigned=assigned[:a.limit_per_rank]
    for j,row in enumerate(assigned):
        cid=row['id']; clip=root/cid
        assert sha(clip/'metadata.json')==sha(Path(index['dataset_root'])/cid/'metadata.json'), 'RGB/labels metadata identity differs'
        for task,taskid,prompt in zip(TASKS,TASK_IDS,PROMPTS):
            target=out/task/(cid+'.pt'); target.parent.mkdir(parents=True,exist_ok=True)
            if valid(target,contract,cid,task): continue
            video,stats=load_video(clip,task)
            latent=reader.encode_video(vae,video,device); del video
            temporary=target.with_suffix(f'.pending-{rank}')
            torch.save({'schema':SCHEMA,'cache_contract_hash':contract,'clip_id':cid,'modality':task,
              'task_id':taskid,'prompt':prompt,'latent':latent,'raw_statistics':stats,
              'metadata_sha256':sha(clip/'metadata.json')},temporary)
            os.replace(temporary,target)
            print(f'EXTRA_CACHE rank={rank} clip={j+1}/{len(assigned)} id={cid} task={task} elapsed={time.monotonic()-start:.1f}',flush=True)
    dist.barrier()
    if rank==0 and not a.limit_per_rank:
        for row in rows:
            for task in TASKS: assert (out/task/(row['id']+'.pt')).is_file()
        atomic_json(out/'CACHE_COMPLETE',{'schema':SCHEMA,'cache_contract_hash':contract,'manifest_sha256':sha(out/'manifest.json'),'num_clips':len(rows)})
        print('EXTRA_CACHE_COMPLETE',flush=True)
    dist.destroy_process_group()

if __name__=='__main__': main()
