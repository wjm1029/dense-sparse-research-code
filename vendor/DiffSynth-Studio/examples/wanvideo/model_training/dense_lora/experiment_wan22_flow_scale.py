"""Isolated flow-scale reconstruction audit and paired small-set overfit."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from build_wan22_albedo_flow_target_cache import _read_exr
from infer_wan22_dense4 import load_official_vae, decode_video

p=argparse.ArgumentParser()
p.add_argument('--output_dir',required=True)
p.add_argument('--phase',choices=['scale','overfit'],required=True)
p.add_argument('--variant',choices=['baseline','selected'],default='selected')
p.add_argument('--steps',type=int,default=200)
a=p.parse_args()
os.environ['TOKENIZERS_PARALLELISM']='false'
torch.manual_seed(17); np.random.seed(17); random.seed(17)
root=Path('/pfs/mayuema/dense/wan'); out=Path(a.output_dir)
model=root/'Wan2.2-TI2V-5B'; device=torch.device('cuda:0')
depth=root/'cache/wan22_depth_latents_bf16_bright40'
extra=root/'cache/wan22_albedo_delayed_flow_targets_v1_bf16_bright40'
indices=[0,1837,3675,5512]
ids=[json.loads((depth/'dataset_index.json').read_text())['samples'][i]['id'] for i in indices]

def raw_flow(cid):
    uv=torch.zeros(2,81,480,832); valid=torch.zeros(81,480,832,dtype=torch.bool)
    for t in range(1,81):
        x=_read_exr(Path('/pfs/mayuema/dense/dataset/prod_final_9110_20260816/train')/cid/'optical_flow'/f'{t:06d}.exr')
        uv[0,t]=torch.from_numpy(x[...,2].copy()); uv[1,t]=torch.from_numpy(x[...,1].copy())
        valid[t]=torch.from_numpy(x[...,0]>0.5)
    return uv,valid

def metrics(decoded,uv,valid,scale):
    pred=decoded[:2]*torch.tensor(scale)[:,None,None,None]
    err=(pred-uv).square().sum(0).sqrt(); mag=uv.square().sum(0).sqrt()
    moving=valid&(mag>1); static=valid&(mag<=1)
    pv=decoded[2]>0
    return dict(epe_px=float(err[valid].mean()),moving_epe_px=float(err[moving].mean()) if moving.any() else None,
        static_epe_px=float(err[static].mean()) if static.any() else None,mean_magnitude_px=float(mag[valid].mean()),
        valid_iou=float((pv[1:]&valid[1:]).sum()/(pv[1:]|valid[1:]).sum().clamp_min(1)))

vae=load_official_vae(root/'Wan2.2',model,device)
if a.phase=='scale':
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    with torch.inference_mode():
        for cid in ids:
            uv,valid=raw_flow(cid)
            for name,scale in [('baseline',(832.,480.)),('scale16',(16.,16.)),('scale32',(32.,32.)),('scale64',(64.,64.))]:
                vec=(uv/torch.tensor(scale)[:,None,None,None]).clamp(-1,1)
                encoded_input=torch.cat([vec,(valid.float()*2-1)[None]],0)
                if name=='baseline':
                    z=torch.load(extra/'optical_flow'/f'{cid}.pt',map_location='cpu',weights_only=True)['target_latent']
                else:
                    z=vae.encode([encoded_input.to(device=device,dtype=torch.bfloat16)])[0].detach().cpu()
                decoded=decode_video(vae,z,device)
                row=dict(clip_id=cid,variant=name,scale=list(scale),**metrics(decoded,uv,valid,scale))
                clipped=(uv.abs()>torch.tensor(scale)[:,None,None,None]).any(0)&valid
                row['clipped_fraction']=float(clipped.sum()/valid.sum())
                row['clip_only_epe_px']=float(((vec*torch.tensor(scale)[:,None,None,None]-uv).square().sum(0).sqrt())[valid].mean())
                path=out/name/f'{cid}.pt'; path.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'target_latent':z,'clip_id':cid,'scale':list(scale)},path)
                rows.append(row); (out/'scale_metrics.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(row),flush=True)
    summary={name:{k:float(np.mean([r[k] for r in rows if r['variant']==name])) for k in ('epe_px','moving_epe_px','static_epe_px','clipped_fraction','clip_only_epe_px')} for name in ('baseline','scale16','scale32','scale64')}
    best=min(('scale16','scale32','scale64'),key=lambda name:summary[name]['epe_px'])
    result={'clip_ids':ids,'indices':indices,'summary':summary,'selected_variant':best,'selection':'lowest equal-clip mean valid-pixel reconstruction EPE; inspect clipping and moving EPE too','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'scale_summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
    (out/'SCALE_COMPLETE').touch()
else:
    from train_wan22_dense6_ddp import Dense6LoRAModule
    from wan22_dense6_checkpoint import checkpoint_metadata,build_provenance_metadata,load_dense6_checkpoint
    result=json.loads((out/'scale_summary.json').read_text())
    variant='baseline' if a.variant=='baseline' else result['selected_variant']
    run=out/f'overfit_{variant}'; run.mkdir(exist_ok=False)
    checkpoint=root/'outputs/wan22_dense6_hyperlora_bidir_stage1_50k/step-50000.safetensors'
    meta=checkpoint_metadata(checkpoint); prompts=json.loads(meta['output_prompt_texts']); pid=prompts.index('output: optical flow')
    module=Dense6LoRAModule(str(model),int(meta['lora_rank']),meta['lora_targets'],int(meta['router_hidden_dim']),meta['router_mode'],0.)
    module.task_router.to(device=device,dtype=torch.float32); module.prepare_condition_tables(prompts,device)
    provenance=build_provenance_metadata(module,model_dir=model,prompt_texts=prompts,prompt_dir='/pfs/mayuema/dense/prompts',depth_cache_dir=depth,target_cache_dir=root/'cache/wan22_dense4_targets_bf16_bright40',code_patch_path=root/'docs/wan22_dense6_bidir_runtime_source.patch',repo_dir=Path(__file__).resolve().parents[4],auxiliary_cache_dirs={'albedo_delayed_flow':extra})
    # The experiment commit adds a driver; verify unchanged checkpoint implementation
    # before using the checkpoint's original training revision for its provenance.
    repo=Path(__file__).resolve().parents[4]
    runtime_revision=provenance['diffsynth_git_head']
    subprocess.run(['git','merge-base','--is-ancestor',meta['diffsynth_git_head'],runtime_revision],cwd=repo,check=True)
    for name in ('train_wan22_dense6_ddp.py','wan22_dense6_checkpoint.py','wan22_dense6_task_router.py','wan22_masked_loss.py'):
        subprocess.run(['git','diff','--exit-code',meta['diffsynth_git_head'],'--',f'examples/wanvideo/model_training/dense_lora/{name}'],cwd=repo,check=True)
    provenance['diffsynth_git_head']=meta['diffsynth_git_head']
    load_dense6_checkpoint(module,checkpoint,expected_provenance=provenance,lora_rank=int(meta['lora_rank']),lora_targets=meta['lora_targets'],router_hidden_dim=int(meta['router_hidden_dim']),expected_router_mode=meta['router_mode'])
    module.pipe.dit.to(device)
    samples=[]
    for cid in ids:
        src=torch.load(depth/f'{cid}.pt',map_location='cpu',weights_only=True)['source_latent']
        payload=torch.load(out/variant/f'{cid}.pt',map_location='cpu',weights_only=True)
        uv,valid=raw_flow(cid)
        samples.append((cid,src,payload['target_latent'],payload['scale'],uv,valid))
    params=[v for v in module.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW([{'params':[v for n,v in module.named_parameters() if v.requires_grad and '.lora_' in n],'lr':1e-4},{'params':list(module.task_router.parameters()),'lr':1e-5}],weight_decay=1e-2)
    tid=torch.tensor([5],device=device); promptid=torch.tensor([pid],device=device)
    records=[]
    def evaluate(step):
        module.eval()
        with torch.inference_mode():
            for cid,src,gt,scale,uv,valid in samples:
                velocity,_=module.predict_velocity(src[None].to(device),tid,promptid)
                module.task_router.clear(); pred=(-velocity[0]).detach().cpu().to(torch.bfloat16)
                decoded=decode_video(vae,pred,device)
                row=dict(step=step,variant=variant,clip_id=cid,latent_mse=float((pred.float()-gt.float()).square().mean()),**metrics(decoded,uv,valid,scale))
                records.append(row); print(json.dumps(row),flush=True)
                if step==a.steps: torch.save({'pred_latent':pred,'scale':scale},run/f'prediction_{ids.index(cid)}.pt')
        (run/'eval_metrics.json').write_text(json.dumps(records,indent=2)); module.train()
    evaluate(0)
    log=[]
    for step in range(1,a.steps+1):
        _,src,gt,_,_,_=samples[(step-1)%len(samples)]
        optimizer.zero_grad(set_to_none=True)
        loss,_,_=module(src[None].to(device),gt[None].to(device),tid,promptid,torch.ones(1,1,21,30,52,device=device),True)
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.); optimizer.step(); module.task_router.clear()
        if step%10==0:
            row={'step':step,'loss':float(loss.detach()),'grad_norm':float(norm)};log.append(row);print(json.dumps(row),flush=True)
        if step in {50,a.steps}: evaluate(step)
    torch.save({'trainable_state':{n:v.detach().cpu() for n,v in module.named_parameters() if v.requires_grad},'variant':variant,'steps':a.steps,'initial_checkpoint':str(checkpoint),'initial_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'runtime_revision':runtime_revision,'seed':17,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},run/'experimental_trainable.pt')
    (run/'train_log.json').write_text(json.dumps(log,indent=2)); (run/'OVERFIT_COMPLETE').touch()
