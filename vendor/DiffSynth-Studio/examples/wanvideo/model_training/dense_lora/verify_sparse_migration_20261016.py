"""Verify the immutable completed PFS state/weights before target-host resume."""
import json
from pathlib import Path
import torch
from safetensors import safe_open

out=Path('/pfs/mayuema/dense/wan/outputs/wan22_sparse_long477_self_forcing_2400_20261016')
receipt=json.loads((out/'migration_stop_receipt_20261016.json').read_text())
state=torch.load(out/'trainer_state_latest.pt',map_location='cpu',weights_only=False)
step=state['step']
assert step==receipt['last_uploaded_step'] and step>=50
assert state['contract']['world_size']==4 and state['contract']['phase']=='self_forcing'
assert len(state['rng'])==4 and all({'python','numpy','torch','cuda'}==set(r) for r in state['rng'])
assert state['scheduler']['last_epoch']==step
with safe_open(str(out/f'step-{step}.safetensors'),framework='pt',device='cpu') as f:
 assert int(f.metadata()['step'])==step and json.loads(f.metadata()['contract'])==state['contract']
 assert set(f.keys())==set(state['weights'])
 for name in f.keys():
  assert torch.equal(f.get_tensor(name),state['weights'][name]), name
record={'target_host':'sz6671','source_host':'sz6670','step':step,'epoch':state['epoch'],
        'next_batch':state['next_batch'],'optimizer_parameter_states':len(state['optimizer']['state']),
        'learning_rates':[g['lr'] for g in state['optimizer']['param_groups']],
        'scheduler_last_epoch':state['scheduler']['last_epoch'],'rng_ranks':len(state['rng']),
        'all_weight_tensors_identical':True}
(out/'migration_verified_sz6671_20261016.json').write_text(json.dumps(record,indent=2))
print('MIGRATION_STATE_VERIFIED '+json.dumps(record),flush=True)
