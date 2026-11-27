"""Verify all-six-task smoke completion and exact paired checkpoint contents."""
import json
from pathlib import Path
import torch
from safetensors import safe_open

root=Path('/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_smoke6_20261017')
marker=json.loads((root/'SMOKE_COMPLETE').read_text()); assert marker['step']==6
state=torch.load(root/'trainer_state_latest.pt',map_location='cpu',weights_only=False)
assert state['step']==6 and state['scheduler']['last_epoch']==6 and len(state['rng'])==4
assert marker['contract']==state['contract'] and state['contract']['smoke']
assert all(set(r)=={'python','numpy','torch','cuda'} for r in state['rng'])
metrics=[json.loads(line) for line in (root/'train_metrics.jsonl').read_text().splitlines()]
assert {r['task'] for r in metrics}==set(range(6)) and len(metrics)==6
assert all(r['mse']>=0 and r['grad_norm']>0 for r in metrics)
with safe_open(str(root/'step-6.safetensors'),framework='pt',device='cpu') as f:
    assert set(f.keys())==set(state['weights'])|{'condition_contexts'}
    assert json.loads(f.metadata()['contract'])==state['contract']
    for key,value in state['weights'].items(): assert torch.equal(value,f.get_tensor(key)),key
    assert torch.equal(f.get_tensor('condition_contexts'),state['contexts'])
evaluation=[json.loads(line) for line in (root/'evaluation.jsonl').read_text().splitlines()]
assert {r['task'] for r in evaluation}==set(range(6)) and all(r['n']==4 for r in evaluation)
result={'step':6,'all6_tasks_completed':True,'chunks_per_step':40,'all6_tasks_free_rollout_evaluated':True,
  'validation_scope':'same4_smoke_only','paired_weight_tensors_equal':True,
  'optimizer_parameter_states':len(state['optimizer']['state']),'rng_ranks':len(state['rng']),
  'metrics':metrics,'evaluation':evaluation}
(root/'SMOKE_VERIFIED.json').write_text(json.dumps(result,indent=2))
print('DENSE6_LONG_SMOKE_VERIFIED '+json.dumps(result),flush=True)
