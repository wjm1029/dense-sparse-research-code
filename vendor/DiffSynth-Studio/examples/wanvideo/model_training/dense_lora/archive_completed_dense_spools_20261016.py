"""Recoverably archive only this task's completed Dense6 local recovery files."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path('/pfs/mayuema/dense/wan')
TARGET=ROOT/'outputs/checkpoint_spool_archive_sz6671_20261016'
PAIRS={
 '/tmp/wan22_dense6_flow32_recovery':ROOT/'outputs/wan22_dense6_hyperlora_bidir_flow32_stage1_50k/trainer_state_bundled_latest.pt',
 '/tmp/wan22_dense6_dispersion_wan22_dense6_flow32_dispersion_100k_to120k_20261012':ROOT/'outputs/wan22_dense6_flow32_dispersion_100k_to120k_20261012/trainer_state_bundled_latest.pt',
 '/tmp/wan22_dense6_stage_wan22_dense6_causal_dispersion_45k_20261013':ROOT/'outputs/wan22_dense6_causal_dispersion_45k_20261013/trainer_state_bundled_latest.pt',
 '/tmp/wan22_dense6_stage_wan22_dense6_ar_tf_dispersion_43500_20261013':ROOT/'outputs/wan22_dense6_ar_tf_dispersion_43500_20261013/trainer_state_bundled_latest.pt',
}
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''): h.update(chunk)
 return h.hexdigest()
for directory,remote_latest in PAIRS.items():
 folder=Path(directory)
 for name in ('trainer_state_bundled_latest.pt','trainer_state_bundled_previous.pt'):
  source=folder/name
  if not source.exists(): continue
  if source.is_symlink() or source.parent!=folder or not source.is_file(): raise ValueError(source)
  checksum=digest(source); size=source.stat().st_size
  dest=remote_latest if name.endswith('latest.pt') else TARGET/folder.name/name
  if name.endswith('latest.pt'):
   if not dest.is_file() or digest(dest)!=checksum: raise ValueError('Latest is not identically backed up: '+str(source))
  else:
   dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    if digest(dest)!=checksum: raise ValueError('Archive target differs: '+str(dest))
   else:
    tmp=dest.with_suffix('.pending'); shutil.copyfile(source,tmp)
    if digest(tmp)!=checksum: raise ValueError('Archive copy verification failed')
    os.replace(tmp,dest)
  receipt={'source':str(source),'recoverable_at':str(dest),'bytes':size,'sha256':checksum}
  receipts=TARGET/'receipts'; receipts.mkdir(parents=True,exist_ok=True)
  (receipts/(folder.name+'_'+name+'.json')).write_text(json.dumps(receipt,indent=2))
  source.unlink()
  print('ARCHIVED_AND_REMOVED_LOCAL '+json.dumps(receipt),flush=True)
print('FREE_GIB',shutil.disk_usage('/tmp').free/2**30,flush=True)
