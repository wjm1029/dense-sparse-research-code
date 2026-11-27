"""Fit on every train clip, then run matched depth-kernel ablations on GPUs4/5."""
import json
import os
import subprocess
import time
from full_data_v9 import EXP,atomic,emit

RUN=EXP/'depth_conv_diagnosis_v11'
WRAPPER=EXP.parent.parent/'tools/wan22_python.sh'
SCRIPT=EXP/'adaptive_depth_v11.py'


def launch(args,gpu,logpath):
    logpath.parent.mkdir(parents=True,exist_ok=True)
    with logpath.open('a') as log:
        p=subprocess.Popen(['bash',str(WRAPPER),str(SCRIPT),*args],stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),close_fds=True)
    return p


def main():
    assert json.loads((RUN/'adaptive_smoke/COMPLETE.json').read_text())['status']=='passed'
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(x.split(',')[0]):int(x.split(',')[1]) for x in raw.splitlines()}
    assert min(free[g] for g in [4,5])>30000
    if not (RUN/'full_data_filter/COMPLETE.json').exists():
        p=launch(['fit'],4,RUN/'full_data_filter/fit.log')
        atomic(RUN/'STATUS.json',dict(phase='fitting_full_dataset_kernel',pid=p.pid,time=time.time()))
        atomic(RUN/'FIT_LAUNCH.json',dict(pid=p.pid,gpu=4,time=time.time()))
        code=p.wait();assert code==0,f'Kernel fit exited {code}'
    fitted=json.loads((RUN/'full_data_filter/COMPLETE.json').read_text())
    m=fitted['validation'];assert fitted['training_clips']==4696
    # A useful starting point must reduce seams and avoid a large MAE regression.
    # Edge error is reported separately; recovering smoothing damage is the adaptive experiment.
    gate=dict(seams_reduced=m['conv']['patch_seam32']<.5*m['mlp_v9']['patch_seam32'],
              mae_stable=m['conv']['mae']<=1.02*m['mlp_v9']['mae'],
              edge_error_ratio=m['conv']['gt_edge_region_mae']/m['mlp_v9']['gt_edge_region_mae'])
    atomic(RUN/'FILTER_GATE.json',gate)
    assert gate['seams_reduced'] and gate['mae_stable'],'Global filter failed the declared starting-point checks'
    jobs=[]
    for variant,gpu in [('pixel_only',4),('token_condition',5)]:
        if (RUN/'adaptive'/variant/'COMPLETE.json').exists():continue
        p=launch(['train','--variant',variant,'--steps','12000'],gpu,RUN/'adaptive'/variant/'train.log')
        jobs.append((variant,gpu,p))
    atomic(RUN/'TRAINING_LAUNCH.json',dict(jobs=[dict(variant=v,gpu=g,pid=p.pid,steps=12000) for v,g,p in jobs],time=time.time()))
    atomic(RUN/'STATUS.json',dict(phase='matched_adaptive_training',time=time.time()))
    while any(p.poll() is None for _,_,p in jobs):
        for variant,_,p in jobs:
            if p.poll() not in [None,0]:raise RuntimeError(f'{variant} failed with exit {p.returncode}')
        time.sleep(15)
    result={}
    for variant in ['pixel_only','token_condition']:
        result[variant]=json.loads((RUN/'adaptive'/variant/'COMPLETE.json').read_text())
    atomic(RUN/'COMPLETE.json',dict(variants=result,filter_gate=gate,time=time.time()))
    atomic(RUN/'STATUS.json',dict(phase='complete',time=time.time()));emit('pipeline_complete')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        atomic(RUN/'PIPELINE_FAILED.json',dict(error=repr(error),time=time.time()));raise
