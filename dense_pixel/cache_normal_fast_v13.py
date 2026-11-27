"""Parallel CPU frame reader for normal-conditioned feature caching.

Uses exactly v9's qualified clips/splits. No stride8 data, no GT history.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time
import numpy as np
import torch
from PIL import Image
from full_data_v9 import EXP, SHARDS, atomic, emit
from train_full_v9 import RUN as DATA, rows
from train_multiclip_v8 import save_atomic
import run_probe as v1
import run_probe_v3b as native

RUN=EXP/'pixvid_modalities_v13'
NORMAL=RUN/'normal_data'
SCHEMA='dense6_normal_flip_yz_full81_v13'


def key(row):return row['scene']+'_'+Path(row['sample']).name


def all_rows():return rows('train')+rows('validation')+rows('test')


def cache_path(row):return RUN/'normal_cache'/(key(row)+'.pt')


def ready_path(row):return NORMAL/'ready'/(key(row)+'.json')


def valid_dir(p):
    return {x.name for x in p.glob('*.exr')}=={f'{i:06d}.exr' for i in range(1,82)}


def extract_group(shard,group):
    marker=NORMAL/'shards'/(shard+'.json')
    if marker.exists():return json.loads(marker.read_text())
    pending={}
    for row in group:
        original=Path(row['path'])/'normal'
        dest=NORMAL/'data'/row['sample']/'normal'
        if valid_dir(original):src=original
        elif valid_dir(dest):src=dest
        else:pending[row['sample']]=row;continue
        atomic(ready_path(row),dict(row=row,normal_path=str(src),frames=81,source='existing_full81'))
    if pending:
        archive=SHARDS/shard
        info=json.loads((SHARDS/(shard+'.json')).read_text())
        old=json.loads((DATA/'shards'/f'{info["shard_index"]:05d}.json').read_text())
        assert info['sha256']==old['sha256'] and archive.stat().st_size==info['archive_size_bytes']
        stage=NORMAL/'staging'/shard
        count={s:0 for s in pending}
        p=subprocess.Popen(['zstd','-q','-d','-c','--',str(archive)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=p.stdout,mode='r|') as tar:
                for item in tar:
                    path=PurePosixPath(item.name)
                    assert not path.is_absolute() and '..' not in path.parts
                    if not item.isfile() or len(path.parts)!=5:continue
                    sample='/'.join(path.parts[:3])
                    if sample not in pending or path.parts[3]!='normal':continue
                    assert path.parts[4] in {f'{i:06d}.exr' for i in range(1,82)}
                    dest=stage/path;dest.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(item) as source,dest.open('wb') as target:shutil.copyfileobj(source,target,1024*1024)
                    count[sample]+=1
            p.stdout.close();assert p.wait()==0,p.stderr.read().decode()
        finally:
            if p.poll() is None:p.kill();p.wait()
        for sample,row in pending.items():
            src=stage/sample/'normal';assert count[sample]==81 and valid_dir(src),(sample,count[sample])
            dst=NORMAL/'data'/sample/'normal';dst.parent.mkdir(parents=True,exist_ok=True)
            os.replace(src,dst)
            atomic(ready_path(row),dict(row=row,normal_path=str(dst),frames=81,source='full_archive',
                archive_sha256_verified_by_v9=info['sha256']))
    result=dict(shard=shard,clips=len(group),newly_extracted=len(pending))
    atomic(marker,result);emit('normal_shard_ready',**result);return result


def extract(workers):
    groups={}
    for r in all_rows():groups.setdefault(r['source_shard'],[]).append(r)
    # Register all existing81-frame normals immediately, before expensive scans.
    for r in all_rows():
        p=Path(r['path'])/'normal'
        if valid_dir(p):atomic(ready_path(r),dict(row=r,normal_path=str(p),frames=81,source='existing_full81'))
    ordered=sorted(groups,key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(extract_group,s,groups[s]) for s in ordered]
        for i,f in enumerate(as_completed(futures)):
            f.result();atomic(NORMAL/'EXTRACTION_STATUS.json',dict(done=i+1,total=len(groups)))
    assert all(ready_path(r).exists() for r in all_rows())
    atomic(NORMAL/'EXTRACTION_COMPLETE.json',dict(clips=len(all_rows()),shards=len(groups),time=time.time()))


def read_normal(row):
    from dense4_dataset import _read_normal
    meta=json.loads((Path(row['path'])/'metadata.json').read_text())
    assert meta['coordinates']['normal_space']=='opencv_camera',meta['coordinates']
    source=Path(json.loads(ready_path(row).read_text())['normal_path'])
    def frame(i):
        a=_read_normal(source/f'{i:06d}.exr')
        assert a.shape==(480,832,3) and np.isfinite(a).all()
        a[...,1:]*=-1
        return torch.from_numpy(a).permute(2,0,1).half()
    with ThreadPoolExecutor(max_workers=8) as pool:
        video=torch.stack(list(pool.map(frame,range(1,82))))
    lengths=video.float().square().sum(1).sqrt();valid=lengths>1e-6
    assert valid.any() and (lengths[valid]-1).abs().max()<.002
    return video


def read_rgb(row):
    def frame(i):
        with Image.open(Path(row['path'])/'rgb'/f'{i:06d}.png') as im:
            return np.asarray(im.convert('RGB'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        return np.stack(list(pool.map(frame,range(1,82))))


def cache(rank,world,limit=0):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    v1.TASKS['normal']=1
    worker=RUN/'normal_workers'/str(rank);worker.mkdir(parents=True,exist_ok=True);native.RUN=worker
    assigned=all_rows()[rank::world];remaining=list(assigned);done=0;model=vae=contract=None
    while remaining:
        progress=False
        for row in list(remaining):
            dest=cache_path(row)
            if dest.exists():
                remaining.remove(row);done+=1;continue
            if not ready_path(row).exists():continue
            started=time.monotonic();targets=read_normal(row)
            if model is None:
                assert torch.cuda.mem_get_info()[0]>45*2**30
                model,vae,contract=native.load_native()
            rgb=read_rgb(row)
            with torch.no_grad():
                source=torch.from_numpy(rgb.copy()).permute(3,0,1,2).cuda().to(torch.bfloat16)/127.5-1
                z=vae.encode([source])[0];assert z.shape==(48,21,30,52)
                result=native.rollout(model,z,'normal',contract['prompts'].index('output: surface normal'))
                baseline=None
                if row['split']!='train':
                    baseline=vae.decode([result['latent'].cuda().to(torch.bfloat16)])[0].float().permute(1,0,2,3).cpu().half()
                    assert baseline.shape==(81,3,480,832) and torch.isfinite(baseline).all()
            features=result['features'].to(torch.bfloat16)
            assert features.shape==(21,15,26,3072) and torch.isfinite(features).all()
            dest.parent.mkdir(parents=True,exist_ok=True)
            save_atomic(dict(row=row,checkpoint_sha256=native.SHA,schema=SCHEMA,
                normal_transform=['x','-y','-z'],cache_reader='parallel8_bitwise_verified',features=features,targets=targets,baseline=baseline),dest)
            del result,baseline,features,source,z,rgb,targets
            remaining.remove(row);done+=1;progress=True
            status=dict(done=done,total=len(assigned),remaining=len(remaining),last_sample=row['sample'],seconds=time.monotonic()-started)
            atomic(worker/'STATUS.json',status);emit('normal_features_ready',rank=rank,**status)
            if limit and done>=limit:return
        if not progress and remaining:
            assert not (NORMAL/'EXTRACTION_FAILED.json').exists(),'Normal extraction failed'
            time.sleep(10)
    atomic(RUN/'normal_workers'/f'{rank}_COMPLETE.json',dict(clips=done,world=world,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','cache']);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=2);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    try:
        extract(a.workers) if a.mode=='extract' else cache(a.rank,a.world,a.limit)
    except BaseException as e:
        f=NORMAL/'EXTRACTION_FAILED.json' if a.mode=='extract' else RUN/'normal_workers'/f'{a.rank}_FAILED.json'
        atomic(f,dict(error=repr(e),time=time.time()));raise
