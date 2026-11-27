"""Run the learnability gate, then launch the authorized four-GPU depth warmup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from full_data_v9 import EXP, atomic

RUN=EXP/'pixvid_detailer_v12'
PY=EXP.parents[1]/'tools/wan22_python.sh'


def phase(name, **kwargs):
    atomic(RUN/'PIPELINE_STATUS.json',dict(phase=name,time=time.time(),**kwargs))


def main():
    weight=RUN/'vgg16-397923af.pth'
    digest=hashlib.sha256()
    with weight.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):digest.update(b)
    assert digest.hexdigest()=='397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'
    cache=EXP.parents[1]/'runtime/wan22_deploy_20261122/cache/torch/hub/checkpoints/vgg16-397923af.pth'
    cache.parent.mkdir(parents=True,exist_ok=True)
    if not cache.exists():cache.symlink_to(weight)
    phase('single_sample_gate',steps=500)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',PYTHONPATH=str(RUN/'deps'))
    with (RUN/'single_sample.log').open('a') as log:
        p=subprocess.Popen(['bash',str(PY),str(EXP/'train_pixvid_v12.py'),'--mode','smoke','--steps','500'],
            env=env,cwd=EXP,stdout=log,stderr=subprocess.STDOUT)
        phase('single_sample_gate',steps=500,pid=p.pid)
        assert p.wait()==0,'Single-sample gate failed; inspect single_sample.log'
    gate=json.loads((RUN/'single_sample/COMPLETE.json').read_text());assert gate['passed']
    free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    usage={int(a):int(b) for a,b in [line.split(',') for line in free.splitlines()]}
    assert all(usage[g]<1000 for g in [4,5,6,7]),f'GPU availability changed: {usage}'
    env['CUDA_VISIBLE_DEVICES']='4,5,6,7'
    with (RUN/'train.log').open('a') as log:
        p=subprocess.Popen(['bash',str(PY),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            str(EXP/'train_pixvid_v12.py'),'--mode','train','--steps','30000','--accum','2'],
            env=env,cwd=EXP,stdout=log,stderr=subprocess.STDOUT)
        phase('training',pid=p.pid,gpus=[4,5,6,7],steps=30000)
        assert p.wait()==0,'Training failed; inspect train.log'
    phase('complete')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        phase('failed',error=repr(e));raise
