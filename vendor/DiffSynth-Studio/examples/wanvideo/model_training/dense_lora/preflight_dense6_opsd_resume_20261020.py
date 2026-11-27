"""Read-only paired-state audit and recoverable write probe before Dense6 OPSD resume."""
import argparse,json,os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from wan22_cache_integrity import sha256_file as sha

p=argparse.ArgumentParser(); p.add_argument('--state',required=True); p.add_argument('--output',required=True); a=p.parse_args()
source=Path(a.state); out=Path(a.output)
state=torch.load(source,map_location='cpu',weights_only=False)
step=state['step']; assert 2200<=step<5250
c=state['contract']; assert c['schema']=='wan22_dense6_long477_opsd_clean_v1' and c['max_steps']==5250
assert c['train_clips']==626 and c['val_clips']==27 and len(c['excluded_ids'])==426
assert not (set(c['train_ids']+c['val_ids'])&set(c['excluded_ids']))
assert len(state['rng'])==4 and state['scheduler']['last_epoch']==step
assert len(state['teacher_ema'])==610 and len(state['optimizer']['state'])==610
assert all(int(v['step'])==step for v in state['optimizer']['state'].values())
assert all(torch.isfinite(v).all() for v in state['teacher_ema'].values())
weights_path=source.parent/f'step-{step}.safetensors'
with safe_open(str(weights_path),framework='pt',device='cpu') as f:
    meta=f.metadata(); assert int(meta['step'])==step and json.loads(meta['contract'])==c
weights=load_file(str(weights_path),device='cpu'); contexts=weights.pop('condition_contexts')
assert set(weights)==set(state['weights']) and len(weights)==612
assert all(torch.equal(v,state['weights'][n]) and torch.isfinite(v).all() for n,v in weights.items())
assert torch.equal(contexts,state['contexts'])
here=Path(__file__).resolve().parent
for name,checksum in c['scripts'].items(): assert sha(here/name)==checksum,name
assert sha('diffsynth/pipelines/wan_video.py')==c['pipeline_sha256']
assert sha('diffsynth/models/wan_video_dit.py')==c['dit_sha256']
out.mkdir(exist_ok=True)
temporary=out/'WRITE_PROBE_20261020.pending'; committed=out/'WRITE_PROBE_20261020.ok'
assert not temporary.exists() and not committed.exists()
with temporary.open('xb') as f:
    f.write(bytes(1024*1024)); f.flush(); os.fsync(f.fileno())
os.replace(temporary,committed); assert committed.stat().st_size==1024*1024; committed.unlink()
receipt={'status':'PASS','resume_step':step,'target':5250,'state':str(source),
    'weights':str(weights_path),'weights_sha256':sha(weights_path),'state_sha256':sha(source),
    'train_clips':626,'val_clips':27,'excluded_clips':426,'paired_tensors':len(weights),
    'ema_tensors':610,'optimizer_tensors':610,'four_rank_rng':True,'write_fsync_rename_probe':'PASS'}
(out/f'RESUME_PREFLIGHT_step{step}_20261020.json').write_text(json.dumps(receipt,indent=2))
print('RESUME_PREFLIGHT '+json.dumps(receipt),flush=True)
