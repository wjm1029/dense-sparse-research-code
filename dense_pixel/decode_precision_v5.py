"""Check solved weights through actual pixel-head forward, FP32 vs BF16."""
import torch
import run_probe as v1
import run_probe_v2 as v2
from decode_unit_v5 import RUN, measure

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
data = v1.load_sample(str(v1.cache_path(v1.rows('overfit8')[0])))
report = {}
with torch.no_grad():
    for task in v1.TASKS:
        head = v2.RidgeHead().cuda().float()
        head.load_state_dict(torch.load(RUN/(task+'_linear_weights.pt'), map_location='cuda', weights_only=True))
        x = data['features'][task][0].cuda().float().permute(2,0,1)[None]
        y = data['targets'][task][:1].cuda().float()
        p = head(x, first=True)[0]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            b = head(x, first=True)[0]
        report[task] = dict(first_frame_fp32=measure(p, y, task), first_frame_bf16=measure(b, y, task))
v1.atomic_json(RUN/'PRECISION.json', report)
print(report)
