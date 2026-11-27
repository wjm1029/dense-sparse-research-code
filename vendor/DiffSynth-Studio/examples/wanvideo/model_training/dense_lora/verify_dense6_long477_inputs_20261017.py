"""Compare the new long flow preprocessing with the audited short flow32 reader."""
import json
from pathlib import Path
import torch
import build_wan22_flow32_target_cache as reference
from build_wan22_dense6_long477_extra import load_video
from wan22_cache_integrity import sha256_file as sha

torch.set_num_threads(1)
source=Path('/pfs/mayuema/dense/wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40')
row=json.loads((source/'source_index.json').read_text())['samples'][0]
clip=Path('/pfs/mayuema/dense/long481_prod3000_20260906/outputs/train')/row['id']
expected,_=reference.load_delayed_flow(clip)
actual,stats=load_video(clip,'optical_flow')
assert actual.shape==(477,3,480,832)
assert torch.equal(actual[:81],expected.to(torch.bfloat16))
assert torch.count_nonzero(actual[0,:2])==0 and torch.all(actual[0,2]==-1)
result={'clip_id':row['id'],'first81_flow_inputs_exactly_equal_to_audited_flow32_reader':True,
        'first_frame_invalid':True,'statistics':stats,'verifier_sha256':sha(__file__)}
out=Path('/pfs/mayuema/dense/wan/cache/wan22_dense6_long477_extra_flow32_v1_20261017')
(out/'INPUT_VERIFICATION.json').write_text(json.dumps(result,indent=2))
print('LONG_INPUT_VERIFIED '+json.dumps(result),flush=True)
