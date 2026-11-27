"""Independent, durable pipelines: matte now; normal cache -> full training."""
import argparse
import json
import os
import subprocess
import time
from full_data_v9 import EXP,atomic
from prepare_normal_v13 import RUN, NORMAL, all_rows, cache_path

PY=EXP.parents[1]/'tools/wan22_python.sh'


def phase(task,name,**data):
    atomic(RUN/task/'PIPELINE_STATUS.json',dict(phase=name,time=time.time(),**data))


def launch(args,gpus,log):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),PYTHONPATH=str(EXP/'pixvid_detailer_v12/deps'))
    with log.open('a') as f:
        return subprocess.Popen(['bash',str(PY)]+args,cwd=EXP,env=env,stdout=f,stderr=subprocess.STDOUT)


def main(task):
    out=RUN/task;out.mkdir(parents=True,exist_ok=True)
    gate_path=out/'single_sample/COMPLETE.json'
    if not gate_path.exists():phase(task,'single_sample_gate')
    while not gate_path.exists():
        assert not (out/'FAILED_smoke_rank0.json').exists(),'Single-example check failed before completion'
        time.sleep(10)
    gate=json.loads(gate_path.read_text())
    assert gate['task']==task and gate['passed'],'Single-example gate must pass first'
    gpus=[4,5] if task=='matte' else [6,7]
    if task=='normal':
        assert (NORMAL/'EXTRACTION_COMPLETE.json').exists(),'Full normal extraction is not complete'
        workers=[]
        for rank,gpu in enumerate(gpus):
            # A worker may already be caching while the other GPU runs the gate.
            record=RUN/'normal_workers'/f'{rank}_LAUNCH.json'
            if record.exists():
                pid=json.loads(record.read_text())['pid']
                procfile=__import__('pathlib').Path(f'/proc/{pid}/cmdline')
                if procfile.exists() and b'cache_normal_fast_v13.py' in procfile.read_bytes():
                    workers.append((rank,pid,None));continue
            if (RUN/'normal_workers'/f'{rank}_COMPLETE.json').exists():continue
            p=launch(['cache_normal_fast_v13.py','cache','--rank',str(rank),'--world','2'],[gpu],RUN/f'normal_cache_rank{rank}.log')
            atomic(record,dict(pid=p.pid,gpu=gpu,time=time.time()))
            workers.append((rank,p.pid,p))
        phase(task,'feature_caching',gpus=gpus,worker_pids=[pid for rank,pid,p in workers],clips=len(all_rows()))
        try:
            while not all((RUN/'normal_workers'/f'{rank}_COMPLETE.json').exists() for rank in range(2)):
                for rank,pid,p in workers:
                    assert not (RUN/'normal_workers'/f'{rank}_FAILED.json').exists(),f'Normal cache worker {rank} failed'
                    if p is not None:assert p.poll() in (None,0),f'Normal cache worker {pid} failed: {p.returncode}'
                    elif not (RUN/'normal_workers'/f'{rank}_COMPLETE.json').exists():os.kill(pid,0)
                time.sleep(10)
            for rank,pid,p in workers:
                if p is not None:assert p.wait()==0
                else:
                    procfile=__import__('pathlib').Path(f'/proc/{pid}/cmdline')
                    while procfile.exists() and b'cache_normal_fast_v13.py' in procfile.read_bytes():time.sleep(1)
        finally:
            for rank,pid,p in workers:
                if p is not None and p.poll() is None:p.terminate();p.wait()
        for rank in range(2):assert (RUN/'normal_workers'/f'{rank}_COMPLETE.json').exists()
        assert all(cache_path(r).exists() for r in all_rows())
        atomic(RUN/'normal_cache/COMPLETE.json',dict(clips=len(all_rows()),train=4696,validation=445,test=471,time=time.time()))
    free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    usage={int(a):int(b) for a,b in [line.split(',') for line in free.splitlines()]}
    assert all(usage[g]<1000 for g in gpus),f'GPU availability changed: {usage}'
    p=launch(['-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_modalities_v13.py',
              '--task',task,'--mode','train','--steps','30000','--accum','4'],gpus,out/'train.log')
    phase(task,'training',gpus=gpus,pid=p.pid,steps=30000)
    assert p.wait()==0,f'{task} training failed; inspect train.log'
    phase(task,'complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['matte','normal'],required=True);a=parser.parse_args()
    try:main(a.task)
    except BaseException as e:
        phase(a.task,'failed',error=repr(e));raise
