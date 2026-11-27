"""One-clip, full-frame AR overfit diagnostic. No train/val generalization claim.

Current predicted latent history is used in training and evaluation. Last four
DiT blocks and both scalar pixel readouts adapt; no GT history is supplied.
Gradients accumulate over all seven chunks before one optimizer update.
"""
import json
import gc
import os
import random
import time
from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn
import run_probe as v1
import run_probe_v2 as v2
import run_probe_v3b as native

RUN=v1.EXP/'single_overfit_v4'


def emit(event,**kw):
    d=dict(event=event,unix_time=time.time(),**kw);print(json.dumps(d),flush=True)
    with (RUN/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')


def save_weights(module,heads,step,metrics):
    path=RUN/'weights_latest.pending.pt'
    torch.save(dict(blocks={str(i):module.pipe.dit.blocks[i].state_dict() for i in range(26,30)},
        pixel_heads=heads.state_dict(),step=step,metrics=metrics,
        base_checkpoint_sha256=native.SHA,optimizer_saved=False),path)
    os.replace(path,RUN/'weights_latest.pt')


@torch.no_grad()
def evaluate(module,heads,z,targets,latent_targets,contract,row,step):
    heads.eval();report={};original=v1.load_sample(str(v1.cache_path(row)))
    for task in v1.TASKS:
        pid=contract['prompts'].index('output: depth' if task=='depth' else 'output: person alpha matte')
        result=native.rollout(module,z,task,pid)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=v1.predict(heads[task],result['features'].cuda()).float().cpu()
        metric=v1.metrics(pred,targets[task].cpu(),task)
        metric['latent_mse_vs_encoded_gt']=float((result['latent'].float()-latent_targets[task].cpu().float()).square().mean())
        report[task]=metric
        native.preview(row,task,targets[task].cpu(),original['baseline'][task],pred,step)
        if step%100==0 or step==0:
            torch.save(dict(prediction=pred,source_case=row,metrics=metric,step=step),RUN/f'preview_{task}_latest.pt')
    v1.atomic_json(RUN/f'eval_{step}.json',report)
    emit('full_clip_evaluation',step=step,metrics=report)
    heads.train();module.pipe.dit.train()
    return report


def good_fit(m):
    return (m['depth']['mae']<.005 and m['depth']['rmse']<.015 and
            m['matte']['mae']<.001 and m['matte'].get('foreground_mae',1)<.015)


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    for folder in [RUN,RUN/'previews']:folder.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);torch.manual_seed(20261125);random.seed(20261125)
    assert torch.cuda.mem_get_info()[0]>45*2**30
    native.RUN=RUN;native.emit=emit
    try:
        row=v1.rows('overfit8')[0]
        emit('started',pid=os.getpid(),max_steps=a.steps,sample=row['sample'],code_sha256=v1.sha(__file__))
        module,vae,contract=native.load_native()
        rgb,targets,preprocess=v1.read_pixels(row)
        with torch.no_grad():
            video=torch.from_numpy(rgb.copy()).permute(3,0,1,2).cuda().to(torch.bfloat16)/127.5-1
            source=vae.encode([video])[0]
            latent_targets={}
            for task in v1.TASKS:
                gt=targets[task][None].expand(3,-1,-1,-1).contiguous().cuda().to(torch.bfloat16)
                latent_targets[task]=vae.encode([gt])[0].detach()
        del vae,video,gt;gc.collect();torch.cuda.empty_cache()
        targets={t:x.cuda() for t,x in targets.items()}
        dit=module.pipe.dit
        for block in dit.blocks[26:]:block.float();block.requires_grad_(True)
        ridge=nn.ModuleDict({t:v2.RidgeHead() for t in v1.TASKS}).cuda()
        heads=nn.ModuleDict({t:v2.ResidualHead(ridge[t]) for t in v1.TASKS}).cuda()
        init=torch.load(v2.RUN/'refiner_latest.pt',map_location='cpu',weights_only=False)
        assert init['step']==4000;heads.load_state_dict(init['models']);heads.requires_grad_(True)
        tail=[p for block in dit.blocks[26:] for p in block.parameters()]
        base=[p for h in heads.values() for p in h.base.parameters()]
        refine=[p for h in heads.values() for n,p in h.named_parameters() if not n.startswith('base.')]
        groups=[dict(params=tail,lr=5e-6,initial_lr=5e-6,name='last4_blocks'),
                dict(params=base,lr=1e-6,initial_lr=1e-6,name='linear_readout'),
                dict(params=refine,lr=1e-4,initial_lr=1e-4,name='residual_readout')]
        optimizer=torch.optim.AdamW(groups,weight_decay=0.)
        trainable=[n for n,p in module.named_parameters() if p.requires_grad]
        assert trainable and all(any(n.startswith(f'pipe.dit.blocks.{i}.') for i in range(26,30)) for n in trainable)
        request=dict(sample=row,sample_count=1,raw_frames=81,resolution=[832,480],
            tasks=['depth','matte'],max_steps=a.steps,trainable_blocks=[26,27,28,29],
            tail_parameters=sum(p.numel() for p in tail),head_parameters=sum(p.numel() for p in heads.parameters()),
            learning_rates={g['name']:g['initial_lr'] for g in groups},warmup=20,
            source='Full-clip official VAE-encoded RGB; fixed input, no data augmentation.',
            history='Current prediction only; independent task caches reset every full rollout.',
            optimizer_update='Accumulate seven native chunk forward/backwards; one update per full clip; alternate tasks.',
            pixel_loss='Full-frame L1+0.25gradientL1+0.1temporal; matte also0.5*(FG MAE+BG MAE).',
            latent_loss='Weight1 latent MSE against VAE-encoded raw GT, supervision only; GT never enters history.',
            acceptance={'depth_mae':.005,'depth_rmse':.015,'matte_mae':.001,'matte_foreground_mae':.015,
                        'consecutive_evaluations':3},
            scope='Single-sample capacity/optimization diagnostic; no claim of held-out quality or reproduction.',
            base_checkpoint_sha256=native.SHA,preprocess=preprocess,
            initialization='Native final5250 plus v2 fixed4000 pixel heads; no v3/v3b adapted block weights.',
            checkpoint='Weights only; optimizer is not persisted. Original files remain separate.',
            code_sha256=v1.sha(__file__))
        v1.atomic_json(RUN/'REQUEST.json',request)
        report=evaluate(module,heads,source,targets,latent_targets,contract,row,0)
        start=time.monotonic();streak=0
        emit('training_started',tail_parameters=request['tail_parameters'],head_parameters=request['head_parameters'])
        for step in range(1,a.steps+1):
            task=list(v1.TASKS)[(step-1)%2];pid=contract['prompts'].index(
                'output: depth' if task=='depth' else 'output: person alpha matte')
            captures=[]
            hook=dit.head.head.register_forward_pre_hook(lambda layer,args:captures.append(args[0]))
            caches=module.causal_controller.new_kv_cache();history=torch.zeros_like(source[None,:,:3])
            previous_pixels=None;optimizer.zero_grad(set_to_none=True);sums={'pixel':0.,'latent':0.,'total':0.}
            try:
                for offset in range(0,21,3):
                    captures.clear()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        pred,_,_=module(source[None,:,offset:offset+3],history,
                            torch.tensor([v1.TASKS[task]],device='cuda'),torch.tensor([pid],device='cuda'),offset,caches,False)
                        assert len(captures)==1
                        features=captures[0].reshape(3,15,26,3072)
                        pixels=[]
                        for j in range(3):
                            pixels.append(heads[task](features[j].permute(2,0,1)[None],first=offset+j==0))
                        pixels=torch.cat(pixels,dim=1)
                        begin=0 if offset==0 else 4*offset-3;end=4*(offset+3)-3
                        target=targets[task][None,begin:end]
                        assert pixels.shape==target.shape
                        pixel=v1.pixel_loss(pixels.float(),target.float())
                        if task=='matte':
                            fg=target>-.98
                            if fg.any() and (~fg).any():
                                error=(pixels.float()-target.float()).abs()
                                pixel=pixel+.5*(error[fg].mean()+error[~fg].mean())
                        if previous_pixels is not None:
                            delta=pixels[:,0].float()-previous_pixels.float()-(target[:,0].float()-targets[task][None,begin-1].float())
                            pixel=pixel+.1*delta.abs().mean()
                        latent=(pred.float()-latent_targets[task][None,:,offset:offset+3].float()).square().mean()
                        loss=pixel*((end-begin)/81)+latent/7
                    assert torch.isfinite(loss)
                    loss.backward()
                    history=pred.detach();previous_pixels=pixels[:,-1].detach()
                    # Truncated BPTT and native bounded cache semantics at chunk boundaries.
                    for cache in caches:
                        if cache.key is not None:cache.key=cache.key.detach()
                        if cache.value is not None:cache.value=cache.value.detach()
                    module.task_router.clear()
                    sums['pixel']+=float(pixel.detach())*(end-begin)/81
                    sums['latent']+=float(latent.detach())/7;sums['total']+=float(loss.detach())
                    del pred,pixels,pixel,latent,loss,features,target
            finally:
                hook.remove();module.task_router.clear();captures.clear()
            norm=nn.utils.clip_grad_norm_(tail+base+refine,1.);assert torch.isfinite(norm)
            for group in optimizer.param_groups:group['lr']=group['initial_lr']*min(step/20,1)
            optimizer.step()
            if step==1 or step%5==0:emit('train_step',step=step,task=task,losses=sums,
                gradient_norm=float(norm),elapsed_seconds=time.monotonic()-start,
                gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30)
            if step%25==0 or step==a.steps:
                report=evaluate(module,heads,source,targets,latent_targets,contract,row,step)
                streak=streak+1 if good_fit(report) else 0
                if step%100==0 or step==a.steps or streak>=3:save_weights(module,heads,step,report)
                if streak>=3:
                    v1.atomic_json(RUN/'OVERFIT_ACCEPTED.json',dict(step=step,metrics=report,streak=streak));break
        v1.atomic_json(RUN/'COMPLETE.json',dict(step=step,metrics=report,
            accepted=streak>=3,seconds=time.monotonic()-start))
    except BaseException as error:
        v1.atomic_json(RUN/'FAILED.json',dict(error=repr(error)));raise


if __name__=='__main__':main()
