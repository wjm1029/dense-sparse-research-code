"""Pin the latest fully uploaded immutable long Dense6 checkpoint and fixed holdouts."""
import hashlib
import json
from pathlib import Path
import re
from safetensors import safe_open

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()

base=Path('/pfs/mayuema/dense/wan')
training=base/'outputs/wan22_dense6_long477_self_forcing_dispersion_4800_20261017'
log=(base/'logs/dense6_long477_sf4800_20261017.log').read_text()
step=max(map(int,re.findall(r'^PFS_CHECKPOINT_COMPLETE step=(\d+)$',log,re.M)))
checkpoint=training/f'step-{step}.safetensors'
with safe_open(str(checkpoint),framework='pt',device='cpu') as f: meta=f.metadata()
contract=json.loads(meta['contract']); assert int(meta['step'])==step and not contract['smoke']
rows=json.loads((base/'cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40/source_index.json').read_text())['samples']
validation=[r for r in rows if int(hashlib.sha256(('dense6-long-v1:'+r['id']).encode()).hexdigest()[:8],16)%20==0]
assert len(validation)==41
ordered=sorted(validation,key=lambda r:hashlib.sha256(('dense6-long-preview-v1:'+r['id']).encode()).digest())
previews=[]
for subject in sorted({r['subject_label'] for r in validation}):
    previews.append(next(r['id'] for r in ordered if r['subject_label']==subject))
for row in ordered:
    if len(previews)>=6: break
    if row['id'] not in previews: previews.append(row['id'])
out=base/f'outputs/wan22_dense6_long477_val_step{step}_20261018'
request={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'step':step,
  'contract':contract,'validation':validation,'preview_ids':previews,'num_previews':len(previews),
  'selection':'fixed seed dense6-long-preview-v1; subject coverage first, no metric-based selection',
  'mode':'causal chunk3 full-prefix; previous prediction only; all477 frames',
  'comparison_reference':'cached GT decoded through the same frozen official VAE, not raw-GT quality ceiling'}
out.mkdir(exist_ok=False)
(out/'REQUEST.json').write_text(json.dumps(request,indent=2))
print(json.dumps({'output':str(out),'step':step,'validation_clips':41,'preview_clips':len(previews),'preview_ids':previews},indent=2))
