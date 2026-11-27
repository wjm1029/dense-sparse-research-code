"""One-block adaptation with native-output anchoring and true AR holdout checks.

Training prefix activations use the frozen original model's predicted history.
Final acceptance reruns the updated full DiT with its own predicted history.
No original source/checkpoint is modified. Depth/matte only; other routes untested.
"""
import argparse
import gc
import json
import os
import random
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image,ImageDraw
import run_probe as v1
import run_probe_v2 as v2

RUN=v1.EXP/'run_v3'
SHA='51ee931e662cf59347f96b2f9b29136f2c62e8db77c958af6a2e225ec57d6d97'


def emit(event,**kw):
    value=dict(event=event,unix_time=time.time(),**kw)
    print(json.dumps(value),flush=True)
    with (RUN/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')


def ppath(row):return RUN/'prefix_cache'/(Path(row['path']).name+'.pt')


def load_native():
    from safetensors.torch import load_file
    from wan22_dense6_long477_opsd import Dense6OPSDModule
    from infer_wan22_dense4 import load_official_vae
    checkpoint=v1.ROOT/'weights/current/dense6/long477_opsd5250_FINAL/step-5250.safetensors'
    assert v1.sha(checkpoint)==SHA
    contract=json.loads((checkpoint.parent/'config.json').read_text())['contract']
    for name,checksum in contract['scripts'].items():assert v1.sha(v1.SCRIPTS/name)==checksum,name
    model=v1.ROOT/'models/Wan2.2-TI2V-5B'
    emit('native_loading',checkpoint_sha256=SHA)
    module=Dense6OPSDModule(str(model))
    weights=load_file(str(checkpoint),device='cpu');contexts=weights.pop('condition_contexts')
    expected={n for n,p in module.named_parameters() if p.requires_grad}|{n for n in module.state_dict() if n.startswith('task_router.')}
    assert set(weights)==expected
    loaded=module.load_state_dict(weights,strict=False)
    assert not loaded.unexpected_keys and not(set(loaded.missing_keys)&expected)
    module.pipe.text_encoder=None;module.pipe.tokenizer=None
    module.context_table=contexts.cuda().to(torch.bfloat16)
    module.requires_grad_(False);module.eval()
    module.pipe.dit.cuda();module.task_router.cuda();module.pipe.device=torch.device('cuda')
    del weights,contexts;gc.collect();torch.cuda.empty_cache()
    vae=load_official_vae(v1.ROOT/'code/Wan2.2',model,torch.device('cuda'))
    v1.atomic_json(RUN/'native_load_receipt.json',dict(checkpoint_sha256=SHA,native_keys=len(expected),
        unexpected_keys=loaded.unexpected_keys,missing_trainable_keys=sorted(set(loaded.missing_keys)&expected),scripts=contract['scripts']))
    return module,vae,contract


@torch.no_grad()
def rollout(module,z,task,pid,capture_prefix=False):
    dit=module.pipe.dit;features=[];velocities=[];inputs=[];headtimes=[]
    def take_input(layer,args):
        inputs.append(tuple(x.detach().cpu() for x in args))
    hooks=[dit.head.head.register_forward_pre_hook(lambda layer,args:features.append(args[0].detach().cpu())),
           dit.head.head.register_forward_hook(lambda layer,args,out:velocities.append(out.detach().cpu()))]
    if capture_prefix:
        hooks.extend([dit.blocks[-1].register_forward_pre_hook(take_input),
                      dit.head.register_forward_pre_hook(lambda layer,args:headtimes.append(args[1].detach().cpu()))])
    dit.train();caches=module.causal_controller.new_kv_cache();history=torch.zeros_like(z[None,:,:3]);preds=[]
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for offset in range(0,21,3):
                p,_,_=module(z[None,:,offset:offset+3],history,
                    torch.tensor([v1.TASKS[task]],device='cuda'),torch.tensor([pid],device='cuda'),offset,caches,False)
                preds.append(p);history=p.detach();module.task_router.clear()
    finally:
        for h in hooks:h.remove()
        module.task_router.clear();dit.eval()
    result=dict(features=torch.cat(features,1).reshape(21,15,26,3072).contiguous(),
                reference_velocity=torch.cat(velocities,1).contiguous(),latent=torch.cat(preds,2)[0].cpu())
    if capture_prefix:
        result.update(tokens=torch.cat([x[0] for x in inputs],1).contiguous(),
                      context=inputs[0][1],t_mod=inputs[0][2],freqs=torch.cat([x[3] for x in inputs],0),head_t=headtimes[0])
    return result


def prepare(module,vae,contract,cases):
    for i,row in enumerate(cases):
        file=ppath(row)
        if file.exists():
            saved=torch.load(file,map_location='cpu',weights_only=False)
            assert saved['checkpoint_sha256']==SHA
            emit('prefix_reused',index=i+1,total=len(cases));continue
        started=time.monotonic();p=Path(row['path'])
        rgb=np.stack([np.array(Image.open(p/'rgb'/f'{j:06d}.png').convert('RGB')) for j in range(1,82)])
        source=torch.from_numpy(rgb.copy()).permute(3,0,1,2).cuda().to(torch.bfloat16)/127.5-1
        with torch.no_grad():z=vae.encode([source])[0]
        assert z.shape==(48,21,30,52)
        payload=dict(row=row,source=z.cpu(),checkpoint_sha256=SHA,tasks={})
        previous=v1.load_sample(str(v1.cache_path(row)))
        for task in v1.TASKS:
            prompt='output: depth' if task=='depth' else 'output: person alpha matte';pid=contract['prompts'].index(prompt)
            data=rollout(module,z,task,pid,True)
            delta=(data['features'].float()-previous['features'][task].float()).square().mean()
            emit('prefix_feature_check',sample=row['sample'],task=task,vs_v1_feature_mse=float(delta))
            payload['tasks'][task]=data
        tmp=file.with_suffix('.pending.pt');torch.save(payload,tmp);os.replace(tmp,file)
        emit('prefix_ready',index=i+1,total=len(cases),seconds=time.monotonic()-started)
    v1.atomic_json(RUN/'PREPARE_COMPLETE.json',dict(samples=len(cases)))


@lru_cache(maxsize=24)
def cached(path):return torch.load(path,map_location='cpu',weights_only=False)


def normalized_features(dit,tokens,t):
    head=dit.head
    shift,scale=(head.modulation.to(dtype=t.dtype,device=t.device)+t.unsqueeze(1)).chunk(2,dim=1)
    return head.norm(tokens)*(1+scale)+shift


def tail_forward(module,data):
    from wan22_causal_attention import CausalAttentionConfig
    dit=module.pipe.dit;block=dit.blocks[-1];attention=block.self_attn.attn
    original=attention.config
    attention.config=CausalAttentionConfig(21,390,3,12,3)
    try:
        tokens=block(data['tokens'].cuda(),data['context'].cuda(),data['t_mod'].cuda(),data['freqs'].cuda())
        features=normalized_features(dit,tokens,data['head_t'].cuda())
        velocity=dit.head.head(features)
    finally:attention.config=original
    return features.reshape(21,15,26,3072),velocity


def preview(row,task,gt,baseline,new,step):
    canvas=Image.new('RGB',(416*4,272*3))
    for ri,fi in enumerate([0,40,80]):
        raw=Image.open(Path(row['path'])/'rgb'/f'{fi+1:06d}.png').convert('RGB').resize((416,240))
        canvas.paste(raw,(0,ri*272+32));draw=ImageDraw.Draw(canvas)
        draw.text((4,ri*272+8),f'RGB | {task} | frame {fi+1}',fill='white')
        for ci,(label,value) in enumerate([('Raw GT',gt),('Official VAE',baseline),('Tail adapted',new)],1):
            a=((value[fi].float().clamp(-1,1).numpy()+1)*127.5).round().astype(np.uint8)
            canvas.paste(Image.fromarray(a).convert('RGB').resize((416,240)),(ci*416,ri*272+32))
            draw.text((ci*416+4,ri*272+8),label,fill='white')
    canvas.save(RUN/'previews'/f'online_{step}_{Path(row["path"]).name}_{task}.jpg',quality=95)


@torch.no_grad()
def evaluate(module,heads,contract,cases,step):
    output=[];heads.eval()
    for i,row in enumerate(cases):
        d=cached(str(ppath(row)));original=v1.load_sample(str(v1.cache_path(row)));z=d['source'].cuda()
        for task in v1.TASKS:
            pid=contract['prompts'].index('output: depth' if task=='depth' else 'output: person alpha matte')
            updated=rollout(module,z,task,pid)
            with torch.autocast('cuda',dtype=torch.bfloat16):p=v1.predict(heads[task],updated['features'].cuda()).float().cpu()
            original_task=d['tasks'][task]
            drift=(updated['latent'].float()-original_task['latent'].float()).square().mean((0,2,3))
            rowout=dict(sample=row['sample'],task=task,step=step,scope='updated_full_model_predicted_history',
                baseline=v1.metrics(original['baseline'][task],original['targets'][task],task),
                adapted=v1.metrics(p,original['targets'][task],task),latent_drift_mse=float(drift.mean()),
                latent_drift_first_chunk=float(drift[:3].mean()),latent_drift_last_chunk=float(drift[-3:].mean()))
            output.append(rowout)
            if i==0:preview(row,task,original['targets'][task],original['baseline'][task],p,step)
    v1.atomic_json(RUN/f'eval_online_{step}.json',output)
    for task in v1.TASKS:
        sub=[r for r in output if r['task']==task]
        emit('online_evaluation',step=step,task=task,cases=len(sub),
            adapted_mae=float(np.mean([r['adapted']['mae'] for r in sub])),
            baseline_mae=float(np.mean([r['baseline']['mae'] for r in sub])),
            latent_drift_mse=float(np.mean([r['latent_drift_mse'] for r in sub])))
    heads.train();module.pipe.dit.blocks[-1].train()


def pixel_crops(head,features,targets,matte):
    losses=[]
    for _ in range(4):
        li=0 if random.random()<.1 else random.randrange(1,21)
        y=random.randrange(10);x=random.randrange(21)
        if random.random()<.5:
            fg=torch.nonzero(matte[v1.frame_slice(li)].float().mean(0)>0)
            if len(fg):
                cy,cx=fg[random.randrange(len(fg))].tolist();y=max(0,min(9,cy//32-3));x=max(0,min(20,cx//32-3))
        feature=features[li,y:y+6,x:x+6].permute(2,0,1)[None]
        target=targets[v1.frame_slice(li),(y+2)*32:(y+4)*32,(x+2)*32:(x+4)*32][None].cuda()
        pred=head(feature,first=li==0)[:,:,64:128,64:128]
        losses.append(v1.pixel_loss(pred.float(),target.float()))
    return torch.stack(losses).mean()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=800);a=parser.parse_args()
    for p in [RUN,RUN/'prefix_cache',RUN/'previews']:p.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);torch.manual_seed(20261124);random.seed(20261124)
    assert torch.cuda.mem_get_info()[0]>45*2**30
    train=v1.rows('train_candidates');val=v1.rows('eval_candidates')
    emit('started',pid=os.getpid(),steps=a.steps,code_sha256=v1.sha(__file__))
    try:
        module,vae,contract=load_native();prepare(module,vae,contract,train+val)
        # Encoder is no longer needed. Full native model is retained for honest AR checks.
        del vae;gc.collect();torch.cuda.empty_cache()
        dit=module.pipe.dit;block=dit.blocks[-1]
        block.float();block.requires_grad_(True)
        ridge=nn.ModuleDict({t:v2.RidgeHead() for t in v1.TASKS}).cuda()
        heads=nn.ModuleDict({t:v2.ResidualHead(ridge[t]) for t in v1.TASKS}).cuda()
        previous=torch.load(v2.RUN/'refiner_latest.pt',map_location='cpu',weights_only=False)
        assert previous['step']==4000;heads.load_state_dict(previous['models']);heads.requires_grad_(True)
        optimizer=torch.optim.AdamW([{'params':list(block.parameters()),'lr':1e-6,'initial_lr':1e-6},
            {'params':list(heads.parameters()),'lr':2e-5,'initial_lr':2e-5}],weight_decay=.001)
        trainable=[n for n,p in module.named_parameters() if p.requires_grad]
        assert trainable and all(n.startswith('pipe.dit.blocks.29.') for n in trainable)
        v1.atomic_json(RUN/'trainable_receipt.json',dict(block_parameters=sum(p.numel() for p in block.parameters()),
            head_parameters=sum(p.numel() for p in heads.parameters()),native_trainable_names=trainable,
            frozen_native_output_head=True,router_frozen=True,original_checkpoint_sha256=SHA))
        # Measure full-sequence replay rounding drift before taking any gradient step.
        row=train[0];data=cached(str(ppath(row)))['tasks']['depth']
        module.task_router(torch.tensor([0],device='cuda'))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):feat,velocity=tail_forward(module,data)
        replay_mse=float((velocity.float().cpu()-data['reference_velocity'].float()).square().mean())
        emit('replay_initial_drift',velocity_mse=replay_mse)
        assert replay_mse<.001,('Unacceptable replay numerical difference',replay_mse)
        module.task_router.clear();evaluate(module,heads,contract,val,0)
        emit('training_started',steps=a.steps,train_clips=len(train),validation_clips=len(val),anchor_weight=5.)
        started=time.monotonic()
        for step in range(1,a.steps+1):
            task=list(v1.TASKS)[(step-1)%2];row=random.choice(train)
            data=cached(str(ppath(row)))['tasks'][task];target=v1.load_sample(str(v1.cache_path(row)))
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():module.task_router(torch.tensor([v1.TASKS[task]],device='cuda'))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                feature,velocity=tail_forward(module,data)
                pixel=pixel_crops(heads[task],feature,target['targets'][task],target['targets']['matte'])
                anchor=(velocity.float()-data['reference_velocity'].cuda().float()).square().mean()
                loss=pixel+5*anchor
            assert torch.isfinite(loss)
            loss.backward();norm=nn.utils.clip_grad_norm_(list(block.parameters())+list(heads.parameters()),1.)
            assert torch.isfinite(norm)
            warm=min(step/50,1)
            for group in optimizer.param_groups:group['lr']=group['initial_lr']*warm
            optimizer.step();module.task_router.clear()
            if step==1 or step%20==0:
                emit('train_step',step=step,task=task,loss=float(loss.detach()),pixel_loss=float(pixel.detach()),
                    anchor_mse=float(anchor.detach()),gradient_norm=float(norm),elapsed_seconds=time.monotonic()-started,
                    gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30)
            if step in [200,400] or step==a.steps:
                evaluate(module,heads,contract,val,step)
                if step>=400 or step==a.steps:
                    tmp=RUN/'tail_latest.pending.pt'
                    torch.save(dict(block29=block.state_dict(),pixel_heads=heads.state_dict(),optimizer=optimizer.state_dict(),
                        step=step,base_checkpoint_sha256=SHA),tmp)
                    os.replace(tmp,RUN/'tail_latest.pt')
        v1.atomic_json(RUN/'COMPLETE.json',dict(steps=a.steps,seconds=time.monotonic()-started))
    except BaseException as e:
        v1.atomic_json(RUN/'FAILED.json',dict(error=repr(e)));raise


if __name__=='__main__':main()
