"""Pin last complete OPSD2200 checkpoint, exact27 clean holdouts and fixed6 previews."""
import hashlib,json
from pathlib import Path
from safetensors import safe_open
from wan22_cache_integrity import sha256_file as sha

base=Path('/pfs/mayuema/dense/wan')
train=base/'outputs/wan22_dense6_long477_opsd_clean_local12_5250_20261018'
checkpoint=train/'step-2200.safetensors'
with safe_open(str(checkpoint),framework='pt',device='cpu') as f: meta=f.metadata()
c=json.loads(meta['contract']); assert meta['step']=='2200' and c['schema']=='wan22_dense6_long477_opsd_clean_v1'
assert c['attention']=='sink3_local12_absolute_positions'
assert 'PFS_CHECKPOINT_COMPLETE step=2200' in (base/'logs/dense6_long477_opsd_clean5250_20261018.log').read_text()
rows=json.loads((base/'cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40/source_index.json').read_text())['samples']
validation=[r for r in rows if r['id'] in set(c['val_ids'])]
assert len(validation)==27 and [r['id'] for r in validation]==c['val_ids']
assert not (set(c['val_ids'])&set(c['excluded_ids']))
ordered=sorted(validation,key=lambda r:hashlib.sha256(('dense6-long-preview-v1:'+r['id']).encode()).digest())
# Preserve the previously delivered human case, then five fixed hash-selected clips.
previous='114b73905af1cb7f/b4f81d3949c39c8528d3'
assert previous in c['val_ids']
previews=[previous,'114b73905af1cb7f/13d01e9478eb65f432e2']
out=Path('/tmp/wan22_dense6_opsd2200_validation_20261020'); out.mkdir(exist_ok=False)
request={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'step':2200,'contract':c,
    'validation':validation,'preview_ids':previews,'num_previews':2,'world_size':2,
    'selection':'previously shown human clip + first rendered fixed-hash human clip; no metric selection',
    'mode':'causal chunk3 sink3_local12; predicted-history only; full477 frames',
    'training_status':'not completed; stopped2267 with ENOSPC; latest uploaded checkpoint2200',
    'comparison_reference':'cached GT decoded via same frozen official VAE; middle panel is not raw GT'}
(out/'REQUEST.json').write_text(json.dumps(request,indent=2))
print(json.dumps({'output':str(out),'step':2200,'val':27,'preview_ids':previews},indent=2))
