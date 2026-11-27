"""After SF completion only, archive/remove its own local spool before OPSD."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

out=Path('/pfs/mayuema/dense/wan/outputs/wan22_sparse_long477_self_forcing_2400_20261016')
spool=Path('/tmp/wan22_sparse_long_wan22_sparse_long477_self_forcing_2400_20261016')
archive=Path('/pfs/mayuema/dense/wan/outputs/checkpoint_spool_archive_sz6671_20261016')/spool.name
marker=json.loads((out/'TRAINING_COMPLETE').read_text())
if marker['step']!=2400 or marker['phase']!='self_forcing': raise ValueError('SF not complete')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(8*1024**2),b''): h.update(data)
 return h.hexdigest()
if spool.exists():
 for source in sorted(spool.iterdir()):
  if source.is_symlink():
   if source.name!='trainer_state_latest.pt' or Path(os.readlink(source)).name!=os.readlink(source):
    raise ValueError('Unexpected spool link')
   continue
  if not re.fullmatch(r'(step-\d+\.safetensors|trainer_state_step-\d+\.pt)',source.name):
   raise ValueError('Unexpected spool file: '+str(source))
  expected=out/source.name if source.suffix=='.safetensors' else archive/source.name
  if source.name=='trainer_state_step-2400.pt': expected=out/'trainer_state_latest.pt'
  checksum=sha(source)
  if not expected.exists():
   expected.parent.mkdir(parents=True,exist_ok=True)
   temp=expected.with_suffix('.pending'); shutil.copyfile(source,temp)
   if sha(temp)!=checksum: raise ValueError('Archive checksum mismatch')
   os.replace(temp,expected)
  if sha(expected)!=checksum: raise ValueError('Existing archive mismatch')
  archive.mkdir(parents=True,exist_ok=True)
  (archive/(source.name+'.receipt.json')).write_text(json.dumps({'source':str(source),'recoverable_at':str(expected),'sha256':checksum},indent=2))
  source.unlink()
  print('ARCHIVED_FINISHED_SF_LOCAL',str(source),'->',str(expected),flush=True)
 link=spool/'trainer_state_latest.pt'
 if link.is_symlink(): link.unlink()
print('FREE_GIB',shutil.disk_usage('/tmp').free/2**30,flush=True)
