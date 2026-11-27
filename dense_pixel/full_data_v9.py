"""Full-dataset extraction, quality checks and fixed splits for pixel-head training."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import time

EXP=Path('/xs-train-nas/LingXiTeam/workspace/msh/my_code/video_dense/experiments/pixvid_decode_probe_20261122')
RUN=EXP/'full_data_v9'
SHARDS=Path('/xs-train-nas/LingXiTeam/workspace/msh/my_code/image_dense/dataset/data/shards')
TEST_SCENES={'d1c15f6add3824d0','9865d7b36b7b6cb1'}
FRAME_FILES={mod:{f'{i:06d}'+('.exr' if mod=='depth' else '.png') for i in range(1,82)} for mod in ['rgb','depth','matte']}


def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending.json');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(tmp,path)


def emit(event,**kw):print(json.dumps(dict(event=event,time=time.time(),**kw)),flush=True)


def classify(path,member,shard):
    meta=json.loads((path/'metadata.json').read_text());job=json.loads((path/'job.json').read_text())
    quality=json.loads((path/'quality.json').read_text())
    reasons=[]
    if quality.get('ok') is not True:reasons.append('original_quality_failed')
    actor_type=meta.get('shot',{}).get('actor_type')
    human=actor_type=='human' or (actor_type is None and bool(job.get('character')))
    if not human:reasons.append('not_human_for_person_matte_route')
    video=meta['video']
    if (video['frames'],video['height'],video['width'])!=(81,480,832):reasons.append('unexpected_video_shape')
    for mod,ext in [('rgb','.png'),('depth','.exr'),('matte','.png')]:
        if {p.name for p in (path/mod).glob('*'+ext)}!={f'{i:06d}{ext}' for i in range(1,82)}:
            reasons.append(mod+'_missing_frames')
    scene=member.split('/')[1]
    actor=job.get('character',job.get('animal',{})).get('id','unknown')
    animation=job.get('animation',{}).get('id','unknown')
    # Group all occurrences of an actor/action pair into train or validation.
    group=f'{actor}/{animation}'
    bucket=int(hashlib.sha256(('v9/'+group).encode()).hexdigest()[:12],16)%10
    split='test' if scene in TEST_SCENES else ('validation' if bucket==0 else 'train')
    return dict(sample=member,path=str(path),scene=scene,actor=actor,animation=animation,
        actor_action_group=group,split=split,source_shard=shard,accepted=not reasons,rejection_reasons=reasons)


def extract_one(info):
    index=info['shard_index'];marker=RUN/'shards'/f'{index:05d}.json'
    if marker.exists():
        prior=json.loads(marker.read_text());assert prior['sha256']==info['sha256'];return prior
    archive=SHARDS/info['shard_name'];expected=set(info['members'])
    for member in expected:
        p=PurePosixPath(member);assert len(p.parts)==3 and p.parts[0]=='train' and not p.is_absolute() and '..' not in p.parts
    # Reuse previously verified full-frame extractions; never use the stride-8 tree.
    reused={m:EXP/'data'/m for m in expected if (EXP/'data'/m/'metadata.json').exists()}
    todo=expected-set(reused)
    digest=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):digest.update(b)
    assert digest.hexdigest()==info['sha256'],f'SHA mismatch: {archive}'
    stage=RUN/'staging'/f'{index:05d}';stage.mkdir(parents=True,exist_ok=True)
    if todo:
        proc=subprocess.Popen(['zstd','-q','-d','-c','--',str(archive)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        try:
            with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
                for item in tar:
                    path=PurePosixPath(item.name)
                    assert not path.is_absolute() and '..' not in path.parts,'Unsafe archive member'
                    if not item.isfile() or len(path.parts)<4:continue
                    prefix='/'.join(path.parts[:3]);tail=path.parts[3:]
                    if prefix not in todo:continue
                    allowed=(len(tail)==1 and tail[0] in {'metadata.json','job.json','quality.json'}) or (
                        len(tail)==2 and tail[0] in {'rgb','depth','matte'} and
                        tail[1] in FRAME_FILES[tail[0]])
                    if not allowed:continue
                    target=stage/path;target.parent.mkdir(parents=True,exist_ok=True)
                    source=tar.extractfile(item)
                    with source,target.open('wb') as out:shutil.copyfileobj(source,out,1024*1024)
            proc.stdout.close();assert proc.wait()==0,f'zstd failed: {archive}'
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
    rows=[]
    for member in sorted(expected):
        if member in reused:path=reused[member]
        else:
            source=stage/member;dest=RUN/'data'/member;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():os.replace(source,dest)
            path=dest
        rows.append(classify(path,member,info['shard_name']))
    result=dict(index=index,archive=info['shard_name'],sha256=info['sha256'],rows=rows,
        accepted=sum(r['accepted'] for r in rows),reused=len(reused))
    atomic(marker,result);emit('shard_ready',index=index,accepted=result['accepted'],total=len(rows))
    return result


def prepare(workers):
    infos=[json.loads(p.read_text()) for p in sorted(SHARDS.glob('*.tar.zst.json'))]
    members=[m for s in infos for m in s['members']];assert len(members)==len(set(members))==9110
    plan=dict(shards=infos,raw_clips=len(members),compressed_bytes=sum(s['archive_size_bytes'] for s in infos),
        scene_counts=dict(Counter(m.split('/')[1] for m in members)),test_scenes=sorted(TEST_SCENES),
        split='Two unseen scenes for test; remaining scenes split 90/10 by global actor/action pair hash',
        modalities=['rgb','depth','matte'],frames=81,workers=workers,
        eligibility='Original quality pass, human actor for person-matte route, correct full-frame structure; pixel checks follow in feature extraction',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic(RUN/'PLAN.json',plan);emit('plan_ready',raw_clips=len(members),shards=len(infos),compressed_tb=plan['compressed_bytes']/1e12)
    errors=[];rows=[]
    # Interleave scenes instead of processing only one scene at the beginning.
    infos.sort(key=lambda s:hashlib.sha256(s['shard_name'].encode()).hexdigest())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(extract_one,s):s for s in infos}
        for future in as_completed(futures):
            info=futures[future]
            try:rows.extend(future.result()['rows'])
            except Exception as error:
                record=dict(shard=info['shard_name'],error=repr(error));errors.append(record);emit('shard_failed',**record)
                atomic(RUN/'extraction_errors.json',errors)
    summary=dict(raw_clips=len(members),audited_clips=len(rows),accepted=sum(r['accepted'] for r in rows),
        split_counts=dict(Counter(r['split'] for r in rows if r['accepted'])),
        rejection_counts=dict(Counter(reason for r in rows for reason in r['rejection_reasons'])),errors=errors)
    atomic(RUN/('EXTRACTION_FAILED.json' if errors else 'EXTRACTION_COMPLETE.json'),summary)
    assert not errors,errors


def finalize():
    plan=json.loads((RUN/'PLAN.json').read_text());assert (RUN/'EXTRACTION_COMPLETE.json').exists()
    world=json.loads((RUN/'LAUNCH.json').read_text())['world_size']
    for rank in range(world):assert (RUN/'workers'/f'{rank}_COMPLETE.json').exists()
    records=[json.loads(p.read_text()) for p in sorted((RUN/'processed').glob('*.json'))]
    eligible=[r for r in records if r['status']=='ready']
    expected=[]
    for p in (RUN/'shards').glob('*.json'):expected.extend(r for r in json.loads(p.read_text())['rows'] if r['accepted'])
    assert {r['row']['sample'] for r in records}=={r['sample'] for r in expected}
    splits={s:sorted([r['row'] for r in eligible if r['row']['split']==s],key=lambda r:r['sample']) for s in ['train','validation','test']}
    assert len(splits['train'])>=1000 and len(splits['validation'])>=100 and len(splits['test'])>=100
    assert not ({r['actor_action_group'] for r in splits['train']}&{r['actor_action_group'] for r in splits['validation']})
    assert not ({r['scene'] for r in splits['test']}&{r['scene'] for r in splits['train']+splits['validation']})
    for split,rows in splits.items():
        (RUN/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=dict(raw_clips=plan['raw_clips'],ready=len(eligible),split_counts={s:len(r) for s,r in splits.items()},
        filtered_after_pixel_checks=[r for r in records if r['status']!='ready'],
        per_split_scenes={s:dict(Counter(r['scene'] for r in rows)) for s,rows in splits.items()})
    atomic(RUN/'DATA_READY.json',summary);emit('data_ready',split_counts=summary['split_counts'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','finalize']);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':prepare(a.workers)
    else:finalize()
