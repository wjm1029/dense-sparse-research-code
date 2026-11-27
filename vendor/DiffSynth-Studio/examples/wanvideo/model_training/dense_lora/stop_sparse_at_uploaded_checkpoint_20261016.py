"""Stop only the identified sparse torchrun after a new PFS checkpoint is complete."""
import json
import os
from pathlib import Path
import signal
import time

pid=3007584
log=Path('/pfs/mayuema/dense/wan/logs/sparse_long477_sf2400_20261016.log')
out=Path('/pfs/mayuema/dense/wan/outputs/wan22_sparse_long477_self_forcing_2400_20261016')
expected='train_wan22_sparse_long477_ddp.py'
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
 cmd=Path(f'/proc/{pid}/cmdline')
 if not cmd.exists(): raise RuntimeError('Source process exited before controlled migration')
 command=cmd.read_bytes().replace(b'\0',b' ').decode()
 if expected not in command or '--phase self_forcing' not in command:
  raise RuntimeError('Refuse to signal an unrelated/reused PID')
 steps=[]
 for line in log.read_text().splitlines():
  if line.startswith('PFS_CHECKPOINT_COMPLETE phase=self_forcing step='):
   steps.append(int(line.rsplit('=',1)[1]))
 saved=max(steps,default=0)
 if saved>=50:
  if not (out/f'step-{saved}.safetensors').is_file() or not (out/'trainer_state_latest.pt').is_file():
   raise RuntimeError('Completion log lacks corresponding checkpoint files')
  os.kill(pid,signal.SIGTERM)
  receipt={'source_host':'sz6670','target_host':'sz6671','torchrun_pid':pid,'last_uploaded_step':saved,
           'state':str(out/'trainer_state_latest.pt'),'weight':str(out/f'step-{saved}.safetensors')}
  (out/'migration_stop_receipt_20261016.json').write_text(json.dumps(receipt,indent=2))
  print('MIGRATION_STOP_REQUESTED '+json.dumps(receipt),flush=True)
  break
 time.sleep(2)
else: raise TimeoutError('No new complete checkpoint within 30 minutes; no process was stopped')
