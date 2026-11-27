"""Coordinate already-started extraction/cache jobs and automatically start training."""
import json
import os
from pathlib import Path
import subprocess
import time
from full_data_v9 import RUN,EXP,atomic,emit

WRAPPER=EXP.parent.parent/'tools/wan22_python.sh'


def alive(pid,expected):
    try:
        raw=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return expected in raw
    except FileNotFoundError:return False


def main():
    launch=json.loads((RUN/'LAUNCH.json').read_text())
    assert (EXP/'train_full_v9.py').exists() and WRAPPER.exists()
    last=None
    while True:
        if (RUN/'EXTRACTION_FAILED.json').exists():raise RuntimeError('Full extraction reported failures')
        if not (RUN/'EXTRACTION_COMPLETE.json').exists() and not alive(launch['extraction_pid'],'full_data_v9.py'):
            raise RuntimeError('Extraction process exited without completion receipt')
        for worker in launch['workers']:
            rank=worker['rank']
            if (RUN/'workers'/f'{rank}_FAILED.json').exists():raise RuntimeError(f'Cache worker {rank} reported failure')
            if not (RUN/'workers'/f'{rank}_COMPLETE.json').exists() and not alive(worker['pid'],'cache_full_v9.py'):
                raise RuntimeError(f'Cache worker {rank} exited without completion receipt')
        summary=dict(phase='extract_and_cache',extracted_shards=len(list((RUN/'shards').glob('*.json'))),
            processed_clips=len(list((RUN/'processed').glob('*.json'))),
            cache_workers_complete=sum((RUN/'workers'/f'{w["rank"]}_COMPLETE.json').exists() for w in launch['workers']),time=time.time())
        atomic(RUN/'STATUS.json',summary)
        current=(summary['extracted_shards']//25,summary['processed_clips']//100)
        if current!=last:emit('pipeline_progress',status=summary);last=current
        if (RUN/'EXTRACTION_COMPLETE.json').exists() and summary['cache_workers_complete']==launch['world_size']:break
        time.sleep(20)
    subprocess.run(['bash',str(WRAPPER),str(EXP/'full_data_v9.py'),'finalize'],check=True)
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(s.split(',')[0]):int(s.split(',')[1]) for s in raw.splitlines()}
        available=[g for g in launch['gpus'] if free[g]>25*1024]
        if len(available)>=2:break
        atomic(RUN/'STATUS.json',dict(phase='waiting_for_training_gpus',free_memory_mib=free,time=time.time()));time.sleep(30)
    jobs=[]
    for task,gpu in zip(['depth','matte'],available[:2]):
        log=(RUN/'logs'/f'train_{task}.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        process=subprocess.Popen(['bash',str(WRAPPER),str(EXP/'train_full_v9.py'),'--task',task,'--steps',str(launch['train_steps_per_task'])],
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        log.close();jobs.append((task,gpu,process))
    atomic(RUN/'TRAINING_LAUNCH.json',dict(jobs=[dict(task=t,gpu=g,pid=p.pid) for t,g,p in jobs],time=time.time()))
    atomic(RUN/'STATUS.json',dict(phase='training',time=time.time()))
    while any(p.poll() is None for _,_,p in jobs):
        for task,_,p in jobs:
            if p.poll() is not None and p.returncode!=0:raise RuntimeError(f'{task} training failed, exit={p.returncode}')
        time.sleep(20)
    for task,_,p in jobs:
        assert p.returncode==0 and (RUN/'training'/f'{task}_COMPLETE.json').exists()
    atomic(RUN/'COMPLETE.json',dict(tasks={t:json.loads((RUN/'training'/f'{t}_COMPLETE.json').read_text()) for t,_,_ in jobs},time=time.time()))
    atomic(RUN/'STATUS.json',dict(phase='complete',time=time.time()));emit('pipeline_complete')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        atomic(RUN/'PIPELINE_FAILED.json',dict(error=repr(error),time=time.time()));emit('pipeline_failed',error=repr(error));raise
