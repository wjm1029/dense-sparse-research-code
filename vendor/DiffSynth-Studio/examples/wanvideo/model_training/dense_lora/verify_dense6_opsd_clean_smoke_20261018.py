"""Verify all six real updates, clean split, EMA and native paired checkpoint."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from wan22_cache_integrity import sha256_file as sha

root=Path('/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_opsd_clean_smoke6_20261018')
assert (root/'SMOKE_COMPLETE').is_file()
rows=[json.loads(line) for line in (root/'train_metrics.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows]==list(range(1,7))
assert [r['task'] for r in rows]==list(range(6))
assert all(r['grad_norm']>0 and r['distill']>0 for r in rows)
assert all(all(torch.isfinite(torch.tensor(r[k])) for k in ('mse','router','dispersion','distill','grad_norm')) for r in rows)
state=torch.load(root/'trainer_state_latest.pt',map_location='cpu',weights_only=False)
weights=load_file(str(root/'step-6.safetensors')); contexts=weights.pop('condition_contexts')
assert state['step']==6 and state['scheduler']['last_epoch']==6
assert len(state['rng'])==4 and state['next_batch']==6
assert set(weights)==set(state['weights'])
assert all(torch.equal(weights[n],state['weights'][n]) for n in weights)
assert torch.equal(contexts,state['contexts'])
assert state['teacher_ema'] and all(torch.isfinite(v).all() for v in state['teacher_ema'].values())
assert set(state['teacher_ema'])=={n for n in weights if n in state['teacher_ema']}
assert all(int(v['step'])==6 for v in state['optimizer']['state'].values())
c=state['contract']; assert c['train_clips']==626 and c['val_clips']==27
assert len(c['train_ids'])==626 and len(c['val_ids'])==27 and len(c['excluded_ids'])==426
assert not (set(c['train_ids'])&set(c['val_ids']))
assert not (set(c['train_ids']+c['val_ids'])&set(c['excluded_ids']))
assert c['loss_start_frame']==21 and c['attention']=='sink3_local12_absolute_positions'
evaluation=[json.loads(line) for line in (root/'evaluation.jsonl').read_text().splitlines()]
assert len(evaluation)==6 and all(r['n']==4 for r in evaluation)
receipt={'status':'PASS','steps':6,'tasks':6,'train_clips':626,'val_clips':27,'excluded_clips':426,
    'paired_tensors':len(weights),'ema_tensors':len(state['teacher_ema']),
    'optimizer_tensors':len(state['optimizer']['state']),'weights_sha256':sha(root/'step-6.safetensors')}
(root/'SMOKE_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
