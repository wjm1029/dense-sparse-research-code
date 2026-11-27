"""Four resumable workers consume full-frame shard receipts and cache native features."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
import run_probe as v1
import run_probe_v3b as native
from train_multiclip_v8 import pack_regular,save_atomic
from decode_unit_v5 import pack
from full_data_v9 import RUN,atomic,emit


def key(row):return row['scene']+'_'+Path(row['sample']).name


def paths(row,task):
    return RUN/'cache'/task/(key(row)+'.pt'),RUN/'statistics_samples'/task/(key(row)+'.pt')


def save_task(row,task,features,targets,baseline):
    cache,stats=paths(row,task);cache.parent.mkdir(parents=True,exist_ok=True);stats.parent.mkdir(parents=True,exist_ok=True)
    features=features.cpu().to(torch.bfloat16);targets=targets.cpu().half()
    assert features.shape==(21,15,26,3072) and targets.shape==(81,480,832)
    assert torch.isfinite(features).all() and torch.isfinite(targets).all()
    save_atomic(dict(row=row,checkpoint_sha256=native.SHA,features=features,targets=targets,
        baseline=None if baseline is None else baseline.cpu().half()),cache)
    generator=torch.Generator().manual_seed(int(hashlib.sha256(row['sample'].encode()).hexdigest()[:15],16))
    i=torch.randperm(390,generator=generator)[:32];j=torch.randperm(7800,generator=generator)[:128]
    sample=dict(first=dict(x=features[0].reshape(-1,3072)[i],y=pack(targets[:1])[i]),
        regular=dict(x=features[1:].reshape(-1,3072)[j],y=pack_regular(targets)[j]),sample=row['sample'],split=row['split'])
    save_atomic(sample,stats)


def main():
    global RANK
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,required=True);parser.add_argument('--world-size',type=int,required=True);a=parser.parse_args()
    RANK=a.rank
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    worker=RUN/'workers'/str(a.rank);worker.mkdir(parents=True,exist_ok=True);native.RUN=worker
    assert torch.cuda.mem_get_info()[0]>45*2**30,'Insufficient free GPU memory for native feature worker'
    plan=json.loads((RUN/'PLAN.json').read_text())
    assigned={s['shard_index'] for s in plan['shards'] if s['shard_index']%a.world_size==a.rank}
    processed_shards=set();counts={'ready':0,'rejected_pixels':0,'reused':0};model=vae=contract=None
    while processed_shards!=assigned:
        ready=[]
        for marker in sorted((RUN/'shards').glob('*.json')):
            index=int(marker.stem)
            if index in assigned and index not in processed_shards:ready.append((index,marker))
        if not ready:
            if (RUN/'EXTRACTION_FAILED.json').exists():raise RuntimeError('Extraction failed; inspect extraction_errors.json')
            time.sleep(10);continue
        for index,marker in ready:
            receipt=json.loads(marker.read_text())
            for row in receipt['rows']:
                if not row['accepted']:continue
                done=RUN/'processed'/(key(row)+'.json')
                if done.exists():
                    old=json.loads(done.read_text());assert old['row']['sample']==row['sample'];continue
                start=time.monotonic();old_cache=v1.cache_path(row)
                if old_cache.exists():
                    data=v1.load_sample(str(old_cache));assert data['checkpoint_sha256']==native.SHA
                    assert data['row']['sample']==row['sample']
                    for task in v1.TASKS:save_task(row,task,data['features'][task],data['targets'][task],data['baseline'][task])
                    counts['reused']+=1;del data
                else:
                    try:rgb,targets,preprocess=v1.read_pixels(row)
                    except AssertionError as error:
                        # Invalid input/targets are recorded, never silently converted into examples.
                        atomic(done,dict(status='rejected_pixels',row=row,error=repr(error)));counts['rejected_pixels']+=1
                        emit('pixel_rejected',rank=a.rank,sample=row['sample'],error=repr(error));continue
                    if model is None:model,vae,contract=native.load_native()
                    with torch.no_grad():
                        source=torch.from_numpy(rgb.copy()).permute(3,0,1,2).cuda().to(torch.bfloat16)/127.5-1
                        z=vae.encode([source])[0]
                    assert z.shape==(48,21,30,52)
                    baseline_needed=row['split']!='train' or int(hashlib.sha256(row['sample'].encode()).hexdigest()[:8],16)%100==0
                    for task in v1.TASKS:
                        prompt='output: depth' if task=='depth' else 'output: person alpha matte'
                        result=native.rollout(model,z,task,contract['prompts'].index(prompt));baseline=None
                        if baseline_needed:
                            with torch.no_grad():baseline=vae.decode([result['latent'].cuda().to(torch.bfloat16)])[0].float().mean(0).cpu().half()
                        save_task(row,task,result['features'],targets[task],baseline)
                        del result,baseline
                    del z,source,rgb,targets
                atomic(done,dict(status='ready',row=row,checkpoint_sha256=native.SHA,seconds=time.monotonic()-start))
                counts['ready']+=1
                atomic(worker/'PROGRESS.json',dict(rank=a.rank,counts=counts,last_sample=row['sample'],shards_done=len(processed_shards)))
                emit('features_ready',rank=a.rank,sample=row['sample'],split=row['split'],seconds=time.monotonic()-start,counts=counts)
            processed_shards.add(index)
    atomic(RUN/'workers'/f'{a.rank}_COMPLETE.json',dict(rank=a.rank,counts=counts,shards=len(processed_shards)))
    emit('worker_complete',rank=a.rank,counts=counts)


if __name__=='__main__':
    RANK='startup'
    try:main()
    except BaseException as error:
        atomic(RUN/'workers'/f'{RANK}_FAILED.json',dict(error=repr(error)))
        emit('worker_failed',error=repr(error));raise
