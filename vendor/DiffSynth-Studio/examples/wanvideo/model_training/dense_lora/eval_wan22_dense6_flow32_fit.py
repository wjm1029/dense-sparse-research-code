"""Fixed training-sample fit audit; decoded GT is the VAE reconstruction ceiling."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from train_wan22_dense6_ddp import CachedDense6Dataset, Dense6LoRAModule
from wan22_dense6_checkpoint import checkpoint_metadata, build_provenance_metadata, load_dense6_checkpoint
from wan22_dense6_task_router import TASK_TO_ID
from infer_wan22_dense4 import load_official_vae, decode_video, task_frames, rgb_frames, label, write_video
from build_wan22_albedo_flow_target_cache import _read_exr

p = argparse.ArgumentParser()
p.add_argument('--output_dir', required=True)
p.add_argument('--clip_indices', default='0,1837,3675,5512')
p.add_argument('--checkpoint', default='/pfs/mayuema/dense/wan/outputs/wan22_dense6_hyperlora_bidir_flow32_stage1_50k/step-50000.safetensors')
a = p.parse_args()
root = Path('/pfs/mayuema/dense/wan')
repo = Path(__file__).resolve().parents[4]
depth = root/'cache/wan22_depth_latents_bf16_bright40'
target = root/'cache/wan22_dense4_targets_bf16_bright40'
extra = root/'cache/wan22_albedo_delayed_flow32_targets_v1_bf16_bright40'
checkpoint = Path(a.checkpoint)
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=False)
ds = CachedDense6Dataset(depth, target, extra, allow_legacy_v1_cache=True)
meta = checkpoint_metadata(checkpoint)
prompts = json.loads(meta['output_prompt_texts'])
device = torch.device('cuda:0')
module = Dense6LoRAModule(str(root/'Wan2.2-TI2V-5B'), int(meta['lora_rank']), meta['lora_targets'], int(meta['router_hidden_dim']), meta['router_mode'], 0.)
module.task_router.to(device=device, dtype=torch.float32)
module.prepare_condition_tables(prompts, device)
provenance = build_provenance_metadata(module, model_dir=root/'Wan2.2-TI2V-5B', prompt_texts=prompts, prompt_dir='/pfs/mayuema/dense/prompts', depth_cache_dir=depth, target_cache_dir=target, code_patch_path=root/'docs/wan22_dense6_bidir_flow32_runtime_source.patch', repo_dir=repo, auxiliary_cache_dirs={'albedo_delayed_flow':extra})
if provenance['diffsynth_git_head'] != meta['diffsynth_git_head']:
    subprocess.run(['git','merge-base','--is-ancestor',meta['diffsynth_git_head'],provenance['diffsynth_git_head']],cwd=repo,check=True)
    for name in ('train_wan22_dense6_ddp.py','wan22_dense6_checkpoint.py','wan22_dense6_task_router.py','wan22_masked_loss.py','wan22_dense6_task_sampler.py'):
        subprocess.run(['git','diff','--exit-code',meta['diffsynth_git_head'],'--',f'examples/wanvideo/model_training/dense_lora/{name}'],cwd=repo,check=True)
    provenance['diffsynth_git_head'] = meta['diffsynth_git_head']
load_dense6_checkpoint(module, checkpoint, expected_provenance=provenance, lora_rank=int(meta['lora_rank']), lora_targets=meta['lora_targets'], router_hidden_dim=int(meta['router_hidden_dim']), expected_router_mode=meta['router_mode'])
module.pipe.dit.to(device).eval()
module.eval()
vae = load_official_vae(root/'Wan2.2',root/'Wan2.2-TI2V-5B',device)
rows = []
with torch.inference_mode():
    for index in map(int,a.clip_indices.split(',')):
        cid = ds.clip_ids[index]
        clipout = out/f'clip_{index:04d}'
        clipout.mkdir()
        source = ds[index*6][0]
        rgb = rgb_frames(decode_video(vae,source,device))
        for task,tid in TASK_TO_ID.items():
            source,gt,_,pid,_ = ds[index*6+tid]
            velocity,_ = module.predict_velocity(source[None].to(device),torch.tensor([tid],device=device),torch.tensor([pid],device=device))
            module.task_router.clear()
            pred = (-velocity[0]).detach().cpu().to(torch.bfloat16)
            if not torch.isfinite(pred).all(): raise ValueError(f'Nonfinite prediction: {cid}/{task}')
            mse = float((pred.float()-gt.float()).square().mean())
            pv = decode_video(vae,pred,device)
            gv = decode_video(vae,gt,device)
            row = dict(clip_id=cid,clip_index=index,task=task,prompt=prompts[pid],latent_mse=mse,decoded_mae=float((pv-gv).abs().mean()))
            if task=='optical_flow':
                valid=gv[2,1:]>0
                diff=(pv[:2,1:]-gv[:2,1:])*32.
                row['vae_reference_flow_epe_px']=float(diff.square().sum(0).sqrt()[valid].mean())
                row['vae_reference_valid_iou']=float(((pv[2,1:]>0)&valid).sum()/((pv[2,1:]>0)|valid).sum().clamp_min(1))
                def flow_frames(v):
                    import cv2
                    u=v[0].numpy()*32.; w=v[1].numpy()*32.
                    hue=((np.arctan2(w,u)+np.pi)*90/np.pi).astype(np.uint8)
                    mag=np.sqrt(u*u+w*w)
                    hsv=np.stack([hue,np.full_like(hue,255),np.clip(mag/20*255,0,255).astype(np.uint8)],axis=-1)
                    return np.stack([cv2.cvtColor(f,cv2.COLOR_HSV2RGB) for f in hsv])
                pf,gf=flow_frames(pv),flow_frames(gv)
                raw_uv=torch.zeros(2,81,480,832)
                raw_valid=torch.zeros(81,480,832,dtype=torch.bool)
                for t in range(1,81):
                    x=_read_exr(Path('/pfs/mayuema/dense/dataset/prod_final_9110_20260816/train')/cid/'optical_flow'/f'{t:06d}.exr')
                    raw_uv[0,t]=torch.from_numpy(x[...,2].copy())
                    raw_uv[1,t]=torch.from_numpy(x[...,1].copy())
                    raw_valid[t]=torch.from_numpy(x[...,0]>0.5)
                mag=raw_uv.square().sum(0).sqrt()
                moving=raw_valid&(mag>1)
                for kind,v in [('prediction',pv),('vae_ceiling',gv)]:
                    epe=(v[:2]*32.-raw_uv).square().sum(0).sqrt()
                    row[f'raw_{kind}_epe_px']=float(epe[raw_valid].mean())
                    row[f'raw_{kind}_moving_epe_px']=float(epe[moving].mean()) if moving.any() else None
                    vv=v[2]>0
                    row[f'raw_{kind}_valid_iou']=float((vv[1:]&raw_valid[1:]).sum()/(vv[1:]|raw_valid[1:]).sum().clamp_min(1))
                row['raw_zero_flow_epe_px']=float(mag[raw_valid].mean())
                row['raw_zero_flow_moving_epe_px']=float(mag[moving].mean()) if moving.any() else None
                row['raw_clipped_fraction']=float(((raw_uv.abs()>32.).any(0)&raw_valid).sum()/raw_valid.sum())
                raw_video=torch.cat([raw_uv/32.,(raw_valid.float()*2-1)[None]],0)
                rf=flow_frames(raw_video)
                raw_panels=np.stack([np.concatenate([label(rgb[t].copy(),'RGB'),label(rf[t].copy(),'Raw EXR GT'),label(gf[t].copy(),'VAE ceiling'),label(pf[t].copy(),'Prediction')],axis=1) for t in range(81)])
                write_video(clipout/'optical_flow_raw_comparison.mp4',raw_panels,16.)
                Image.fromarray(np.concatenate([raw_panels[t] for t in (1,20,40,60,80)],axis=0)).resize((1664,1200)).save(clipout/'optical_flow_raw_contact.jpg')
            else: pf,gf=task_frames(pv,task),task_frames(gv,task)
            panels=np.stack([np.concatenate([label(rgb[t].copy(),'RGB'),label(gf[t].copy(),'GT (VAE decoded)'),label(pf[t].copy(),'Prediction')],axis=1) for t in range(81)])
            write_video(clipout/f'{task}_comparison.mp4',panels,16.)
            strip=np.concatenate([panels[t] for t in (0,20,40,60,80)],axis=0)
            Image.fromarray(strip).resize((1248,1200)).save(clipout/f'{task}_contact.jpg')
            rows.append(row)
            (out/'metrics.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
summary={task:{key:float(np.mean([r[key] for r in rows if r['task']==task])) for key in ('latent_mse','decoded_mae')} for task in TASK_TO_ID}
flow_rows=[r for r in rows if r['task']=='optical_flow']
summary['optical_flow'].update({key:float(np.mean([r[key] for r in flow_rows if r[key] is not None])) for key in flow_rows[0] if key.startswith('raw_') or key.startswith('vae_reference_')})
(out/'summary.json').write_text(json.dumps({'split':'training_fit_only','checkpoint':str(checkpoint),'num_clips':len(rows)//6,'metrics':summary},indent=2))
(out/'EVAL_COMPLETE').touch()
print(json.dumps(summary,indent=2),flush=True)
