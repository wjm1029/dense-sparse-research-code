"""Build/finalize native-pixel JTA RGB VAE caches, with aligned multi-person labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from package_runtime import setup
setup()
import cv2
import numpy as np
import torch
from jta_tiles import load_people, make_targets

SCHEMA='jta_native_rgb_vae_cache_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.pending',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if Path(name).exists():Path(name).unlink()


def atomic_torch(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.pending',dir=path.parent);os.close(fd)
    try:
        torch.save(data,name);os.replace(name,path)
    finally:
        if Path(name).exists():Path(name).unlink()


def read_native_clip(row):
    """Source labels use 1-based frames. Decode whole source frames, then crop; never resize."""
    cap=cv2.VideoCapture(row['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,row['start_frame']-1)
    frames=[];x,y=row['origin'];width,height=832,480
    try:
        for index in range(row['num_frames']):
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Video read failed: {row["id"]} frame {index}')
            if frame.shape!=(1080,1920,3):raise ValueError('Unexpected source resolution')
            if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))!=row['start_frame']+index:
                raise ValueError('Frame seeking mismatch')
            frames.append(cv2.cvtColor(frame[y:y+height,x:x+width],cv2.COLOR_BGR2RGB))
    finally:cap.release()
    return np.stack(frames)


def selected_rows(index, limit_per_split=0, only_ids=None):
    index=Path(index)
    if not (index.parent/'INDEX_COMPLETE').is_file():raise ValueError('Incomplete JTA index')
    rows=[json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate index IDs')
    if only_ids:
        wanted=set(only_ids);rows=[r for r in rows if r['id'] in wanted]
        if {r['id'] for r in rows}!=wanted:raise ValueError('Requested IDs missing from index')
    if limit_per_split:
        counts={};kept=[]
        for row in rows:
            split=row['split'];counts.setdefault(split,0)
            if counts[split]<limit_per_split:kept.append(row);counts[split]+=1
        rows=kept
    return rows


def finalize(output):
    out=Path(output);cfg=json.loads((out/'cache_config.json').read_text());entries=[]
    for shard in range(cfg['num_shards']):
        receipt=json.loads((out/f'shard_{shard:04d}_COMPLETE.json').read_text())
        file=out/f'cache_shard_{shard:04d}.jsonl'
        if sha(file)!=receipt['manifest_sha256']:raise ValueError('Shard manifest modified')
        entries.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
    if len(entries)!=cfg['selected_records'] or len({r['id'] for r in entries})!=len(entries):
        raise ValueError('Missing or duplicate cached clips')
    for row in entries:
        if sha(out/row['cache_file'])!=row['sha256']:raise ValueError(f'Cache damaged: {row["id"]}')
    entries.sort(key=lambda r:r['id'])
    dest=out/'manifest.jsonl';pending=dest.with_suffix('.pending')
    pending.write_text(''.join(json.dumps(r)+'\n' for r in entries));os.replace(pending,dest)
    atomic_json(out/'CACHE_COMPLETE.json',dict(schema=SCHEMA,records=len(entries),manifest_sha256=sha(dest),
        cache_config_sha256=sha(out/'cache_config.json'),splits={s:sum(r['split']==s for r in entries) for s in ('train','val','test')}))
    print(json.dumps(dict(event='cache_complete',records=len(entries),path=str(dest))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--finalize',action='store_true');p.add_argument('--index');p.add_argument('--project-root')
    p.add_argument('--device',default='cuda:0');p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-id',type=int,default=0);p.add_argument('--limit-per-split',type=int,default=0)
    p.add_argument('--only-ids',nargs='+');p.add_argument('--capacity',type=int,default=128)
    a=p.parse_args()
    if a.finalize:finalize(a.output);return
    if not a.index or not a.project_root:p.error('--index and --project-root required for building')
    if a.num_shards<1 or not 0<=a.shard_id<a.num_shards or a.limit_per_split<0:p.error('Invalid sharding/limit')
    root=Path(a.project_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=selected_rows(a.index,a.limit_per_split,a.only_ids)
    if not rows:raise ValueError('No selected records')
    if any(r['track_count']>a.capacity for r in rows):raise ValueError('Index exceeds model capacity')
    vae_file=root/'models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth'
    config=dict(schema=SCHEMA,index_sha256=sha(a.index),num_shards=a.num_shards,
        selected_records=len(rows),selected_ids_sha256=hashlib.sha256('\n'.join(r['id'] for r in rows).encode()).hexdigest(),
        vae_sha256=sha(vae_file),frames=81,resize=False,source_size=[1920,1080],crop_size=[832,480],
        latent_shape=[48,21,30,52],latent_dtype='bfloat16',capacity=a.capacity,
        code_sha256={name:sha(Path(__file__).parent/name) for name in ('jta_cache.py',)},
        label_adapter_sha256=sha(Path(__file__).parents[1]/'core/jta_tiles.py'))
    cfg_path=out/'cache_config.json'
    if cfg_path.exists():
        if json.loads(cfg_path.read_text())!=config:raise ValueError('Cache config mismatch; use a new output folder')
    else:atomic_json(cfg_path,config)
    config_hash=sha(cfg_path);device=torch.device(a.device)
    if device.type!='cuda':raise ValueError('Official VAE cache generation requires CUDA')
    torch.cuda.set_device(device)
    from infer_wan22_dense4 import load_official_vae
    vae=load_official_vae(root/'code/Wan2.2',root/'models/Wan2.2-TI2V-5B',device)
    manifest=[];last_annotation=None;people=None;start=time.monotonic();source_hashes={}
    for row in rows[a.shard_id::a.num_shards]:
        rel=f'clips/{row["id"]}.pt';dest=out/rel
        if dest.exists():
            payload=torch.load(dest,map_location='cpu',weights_only=False)
            if payload.get('cache_config_sha256')!=config_hash or payload.get('row')!=row:
                raise ValueError(f'Existing payload mismatch: {dest}')
        else:
            if row['annotation']!=last_annotation:
                people=load_people(row['annotation']);last_annotation=row['annotation']
            targets=make_targets(people,row['start_frame'],row['origin'],capacity=a.capacity)
            if len(targets['person_ids'])!=row['track_count']:raise ValueError('Index/target track mismatch')
            rgb=read_native_clip(row)
            source=torch.from_numpy(rgb.copy()).permute(3,0,1,2).to(device=device,dtype=torch.bfloat16)/127.5-1
            # Official Wan wrapper returns FP32 latents even with BF16 convolutions;
            # cast explicitly to the established BF16 cache contract.
            with torch.no_grad():z=vae.encode([source])[0].detach().to(device='cpu',dtype=torch.bfloat16)
            if tuple(z.shape)!=(48,21,30,52) or z.dtype!=torch.bfloat16 or not torch.isfinite(z).all():
                raise ValueError('Invalid RGB latent')
            for path in (row['annotation'],row['video']):
                if path not in source_hashes:source_hashes[path]=sha(path)
            payload=dict(schema=SCHEMA,cache_config_sha256=config_hash,row=row,source_latent=z,
                targets={k:torch.from_numpy(v.copy()) for k,v in targets.items()},
                source_sha256={p:source_hashes[p] for p in (row['annotation'],row['video'])})
            atomic_torch(dest,payload)
        z=payload['source_latent']
        if tuple(z.shape)!=(48,21,30,52) or not torch.isfinite(z).all():raise ValueError('Corrupt latent')
        manifest.append(dict(**{k:row[k] for k in ('id','split','sequence','empty','track_count')},cache_file=rel,sha256=sha(dest)))
        if len(manifest)%10==0:print(json.dumps(dict(event='cache_progress',shard=a.shard_id,records=len(manifest),seconds=time.monotonic()-start)),flush=True)
    file=out/f'cache_shard_{a.shard_id:04d}.jsonl'
    pending=file.with_suffix('.pending');pending.write_text(''.join(json.dumps(r)+'\n' for r in manifest));os.replace(pending,file)
    atomic_json(out/f'shard_{a.shard_id:04d}_COMPLETE.json',dict(records=len(manifest),manifest_sha256=sha(file)))
    if a.num_shards==1:finalize(out)


if __name__=='__main__':main()
