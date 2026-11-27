import json
from pathlib import Path
import torch
from build_wan22_albedo_flow_target_cache import load_delayed_flow
from infer_wan22_dense4 import load_official_vae,decode_video
root=Path('/pfs/mayuema/dense/wan')
cid='0a1afa764b9fb300/008f8fd4bbb646de9069'
raw,_=load_delayed_flow(Path('/pfs/mayuema/dense/dataset/prod_final_9110_20260816/train')/cid)
raw=raw.permute(1,0,2,3)
payload=torch.load(root/'cache/wan22_albedo_delayed_flow_targets_v1_bf16_bright40/optical_flow'/f'{cid}.pt',map_location='cpu',weights_only=True)
vae=load_official_vae(root/'Wan2.2',root/'Wan2.2-TI2V-5B',torch.device('cuda:0'))
with torch.inference_mode(): decoded=decode_video(vae,payload['target_latent'],torch.device('cuda:0'))
valid=raw[2,1:]>0
scale=torch.tensor([832.,480.])[:,None,None,None]
epe=((decoded[:2,1:]-raw[:2,1:])*scale).square().sum(0).sqrt()
mag=(raw[:2,1:]*scale).square().sum(0).sqrt()
result={'clip_id':cid,'raw_reference_vae_epe_px':float(epe[valid].mean()),'raw_mean_magnitude_px':float(mag[valid].mean()),'raw_moving_gt1px_vae_epe_px':float(epe[valid&(mag>1)].mean()),'valid_fraction':float(valid.float().mean())}
print(json.dumps(result),flush=True)
(root/'outputs/dense6_train_fit_50k_20261001/flow_vae_audit.json').write_text(json.dumps(result,indent=2))
