"""Fixed training-sample fit audit; decoded GT is the VAE reconstruction ceiling."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from train_wan22_dense6_ddp import CachedDense6Dataset, Dense6LoRAModule
from wan22_dense6_checkpoint import checkpoint_metadata, build_provenance_metadata, load_dense6_checkpoint
from wan22_dense6_task_router import TASK_TO_ID
from infer_wan22_dense4 import load_official_vae, decode_video, task_frames, rgb_frames, label, write_video

p = argparse.ArgumentParser()
p.add_argument('--output_dir', required=True)
p.add_argument('--clip_indices', default='0,1837,3675,5512')
a = p.parse_args()
root = Path('/pfs/mayuema/dense/wan')
repo = Path(__file__).resolve().parents[4]
depth = root/'cache/wan22_depth_latents_bf16_bright40'
target = root/'cache/wan22_dense4_targets_bf16_bright40'
extra = root/'cache/wan22_albedo_delayed_flow_targets_v1_bf16_bright40'
checkpoint = root/'outputs/wan22_dense6_hyperlora_bidir_stage1_50k/step-50000.safetensors'
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=False)
ds = CachedDense6Dataset(depth, target, extra, allow_legacy_v1_cache=True)
meta = checkpoint_metadata(checkpoint)
prompts = json.loads(meta['output_prompt_texts'])
device = torch.device('cuda:0')
module = Dense6LoRAModule(str(root/'Wan2.2-TI2V-5B'), int(meta['lora_rank']), meta['lora_targets'], int(meta['router_hidden_dim']), meta['router_mode'], 0.)
module.task_router.to(device=device, dtype=torch.float32)
module.prepare_condition_tables(prompts, device)
provenance = build_provenance_metadata(module, model_dir=root/'Wan2.2-TI2V-5B', prompt_texts=prompts, prompt_dir='/pfs/mayuema/dense/prompts', depth_cache_dir=depth, target_cache_dir=target, code_patch_path=root/'docs/wan22_dense6_bidir_runtime_source.patch', repo_dir=repo, auxiliary_cache_dirs={'albedo_delayed_flow':extra})
load_dense6_checkpoint(module, checkpoint, expected_provenance=provenance, lora_rank=int(meta['lora_rank']), lora_targets=meta['lora_targets'], router_hidden_dim=int(meta['router_hidden_dim']), expected_router_mode=meta['router_mode'])
module.pipe.dit.to(device).eval()
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
                diff=(pv[:2,1:]-gv[:2,1:])*torch.tensor([832.,480.])[:,None,None,None]
                row['vae_reference_flow_epe_px']=float(diff.square().sum(0).sqrt()[valid].mean())
                row['vae_reference_valid_iou']=float(((pv[2,1:]>0)&valid).sum()/((pv[2,1:]>0)|valid).sum().clamp_min(1))
                def flow_frames(v):
                    import cv2
                    u=v[0].numpy()*832; w=v[1].numpy()*480
                    hue=((np.arctan2(w,u)+np.pi)*90/np.pi).astype(np.uint8)
                    mag=np.sqrt(u*u+w*w)
                    hsv=np.stack([hue,np.full_like(hue,255),np.clip(mag/20*255,0,255).astype(np.uint8)],axis=-1)
                    return np.stack([cv2.cvtColor(f,cv2.COLOR_HSV2RGB) for f in hsv])
                pf,gf=flow_frames(pv),flow_frames(gv)
            else: pf,gf=task_frames(pv,task),task_frames(gv,task)
            panels=np.stack([np.concatenate([label(rgb[t].copy(),'RGB'),label(gf[t].copy(),'GT (VAE decoded)'),label(pf[t].copy(),'Prediction')],axis=1) for t in range(81)])
            write_video(clipout/f'{task}_comparison.mp4',panels,16.)
            strip=np.concatenate([panels[t] for t in (0,20,40,60,80)],axis=0)
            Image.fromarray(strip).resize((1248,1200)).save(clipout/f'{task}_contact.jpg')
            rows.append(row)
            (out/'metrics.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
summary={task:{key:float(np.mean([r[key] for r in rows if r['task']==task])) for key in ('latent_mse','decoded_mae')} for task in TASK_TO_ID}
(out/'summary.json').write_text(json.dumps({'split':'training_fit_only','checkpoint':str(checkpoint),'num_clips':len(rows)//6,'metrics':summary},indent=2))
(out/'EVAL_COMPLETE').touch()
print(json.dumps(summary,indent=2),flush=True)
