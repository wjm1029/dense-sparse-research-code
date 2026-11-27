"""Additional-head-training control for v3; original backbone stays unchanged.

Uses original cached free-running features, so no new latent trajectory is made.
Shares loss, schedule and sample RNG seed with the one-block experiment.
"""
import os
import random
import time
import json
import torch
from torch import nn
import run_probe as v1
import run_probe_v2 as v2
import run_probe_v3b as v3

RUN=v3.RUN/'head_only_control'


def emit(event,**kw):
    x=dict(event=event,unix_time=time.time(),**kw);print(json.dumps(x),flush=True)
    with (RUN/'events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')


def main():
    for p in [RUN,RUN/'previews']:p.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);torch.manual_seed(20261124);random.seed(20261124)
    ridge=nn.ModuleDict({t:v2.RidgeHead() for t in v1.TASKS}).cuda()
    heads=nn.ModuleDict({t:v2.ResidualHead(ridge[t]) for t in v1.TASKS}).cuda()
    saved=torch.load(v2.RUN/'refiner_latest.pt',map_location='cpu',weights_only=False)
    heads.load_state_dict(saved['models']);heads.requires_grad_(True)
    optimizer=torch.optim.AdamW(heads.parameters(),lr=2e-5,weight_decay=.001)
    train=v1.rows('train_candidates');val=v1.rows('eval_candidates')
    original_run=v1.RUN;original_cache=v1.cache_path;original_emit=v1.emit
    def evaluate(step):
        v1.RUN=RUN;v1.cache_path=lambda row:original_run/'cache'/(v1.Path(row['path']).name+'.pt');v1.emit=emit
        try:v1.evaluate(nn.ModuleDict({'head_only':heads}),val,'holdout',step)
        finally:v1.RUN=original_run;v1.cache_path=original_cache;v1.emit=original_emit
    try:
        emit('started',code_sha256=v1.sha(__file__),pid=os.getpid(),steps=800)
        started=time.monotonic()
        for step in range(1,801):
            task=list(v1.TASKS)[(step-1)%2];row=random.choice(train)
            data=v1.load_sample(str(v1.cache_path(row)))
            features=data['features'][task].cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=v3.pixel_crops(heads[task],features,data['targets'][task],data['targets']['matte'],task)
            assert torch.isfinite(loss);loss.backward()
            norm=nn.utils.clip_grad_norm_(heads.parameters(),1.);assert torch.isfinite(norm)
            optimizer.param_groups[0]['lr']=2e-5*min(step/50,1);optimizer.step()
            if step==1 or step%100==0:emit('train_step',step=step,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-started)
            if step in [200,400,800]:evaluate(step)
        torch.save(dict(pixel_heads=heads.state_dict(),step=800),RUN/'heads_latest.pt')
        v1.atomic_json(RUN/'COMPLETE.json',dict(steps=800,seconds=time.monotonic()-started))
    except BaseException as e:
        v1.atomic_json(RUN/'FAILED.json',dict(error=repr(e)));raise


if __name__=='__main__':main()
