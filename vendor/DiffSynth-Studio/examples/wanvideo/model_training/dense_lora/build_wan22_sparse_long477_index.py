"""Audit long human coordinates against the existing whole-video RGB VAE cache."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import numpy as np

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=argparse.ArgumentParser()
p.add_argument('--output_dir',required=True)
p.add_argument('--source_cache',default='/pfs/mayuema/dense/wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40')
p.add_argument('--reference_index',default='/pfs/mayuema/dense/wan/cache/wan22_sparse_coordinates_v1_20261011')
a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=False)
cache=Path(a.source_cache); index=json.loads((cache/'source_index.json').read_text())
manifest=json.loads((cache/'manifest.json').read_text()); complete=json.loads((cache/'CACHE_COMPLETE').read_text())
assert complete['manifest_sha256']==sha(cache/'manifest.json')
assert complete['cache_contract_hash']==manifest['cache_contract_hash']
assert manifest['source_index_sha256']==sha(cache/'source_index.json')
assert manifest['selected_frames']==[1,477] and manifest['latent_shape_cthw']==[48,120,30,52]
root=Path(index['dataset_root']); reference=json.loads((Path(a.reference_index)/'manifest.json').read_text())
names=reference['joint_names']; ids=[]; xy=[]; xyz=[]; masks=[]; splits=[]; hashes=[]; rejected=[]; counts=Counter()
for r in index['samples']:
 if r['subject_label']!='person': counts['nonhuman_skipped']+=1; continue
 cid=r['id']; g=root/cid/'genception'
 try:
  files=[g/'keypoints_2d/keypoints_2d.npz',g/'keypoints_3d/keypoints_3d.npz',g/'keypoints_3d/root_relative_camera_per_height.npy',g/'keypoints_3d/schema.json',g/'.keypoints_complete.json']
  assert all(f.is_file() for f in files), 'incomplete keypoints'
  assert (cache/'source'/(cid+'.pt')).is_file(), 'missing source latent'
  schema=json.loads(files[3].read_text())
  assert schema['joint_order']==names and schema['3d']['model_target']=='root_relative_camera_per_height.npy'
  with np.load(files[0],allow_pickle=False) as d2,np.load(files[1],allow_pickle=False) as d3:
   assert list(d2['joint_names'])==list(d3['joint_names'])==names
   assert np.array_equal(d2['frame_numbers'],np.arange(1,482)) and np.array_equal(d3['frame_numbers'],np.arange(1,482)), 'frame numbering'
   t2=d2['image_xy_01'][:477].astype(np.float32); mask=(d2['in_front']&d2['in_frame'])[:477]
   assert t2.shape==(477,24,2) and mask.shape==(477,24) and mask.any()
   assert np.isfinite(t2).all() and np.max(np.abs(t2-d2['pixel_xy'][:477]/[832,480]))<1e-5
   t3=np.load(files[2],allow_pickle=False)[:477].astype(np.float32)
   assert t3.shape==(477,24,3) and np.isfinite(t3).all() and np.max(np.abs(t3[:,0]))<1e-5
   camera=d3['camera_opencv_m'][:477]; centered=camera-camera[:,:1]
   height=float((centered*t3).sum()/np.square(t3).sum())
   assert np.isfinite(height) and height>0 and np.max(np.abs(centered/height-t3))<5e-4, 'camera/actor-height geometry'
  val=int(hashlib.sha256(('sparse-v1:'+cid).encode()).hexdigest()[:8],16)%20==0
  ids.append(cid); xy.append(t2); xyz.append(t3); masks.append(mask); splits.append(val)
  hashes.append({'id':cid,'labels_sha256':[sha(f) for f in files]})
  counts['val' if val else 'train']+=1
 except Exception as e: rejected.append({'id':cid,'error':str(e)}); counts['rejected']+=1
 if len(ids)%50==0: print(json.dumps({'accepted':len(ids),'counts':dict(counts)}),flush=True)
assert counts['train'] and counts['val']
np.savez_compressed(out/'targets.npz',xy=np.stack(xy),xyz=np.stack(xyz),mask=np.stack(masks),is_val=np.array(splits))
result={'schema':'wan22_sparse_long477_coordinates_v1','source_cache':str(cache.resolve()),'dataset_root':str(root),
 'source_index_sha256':sha(cache/'source_index.json'),'source_manifest_sha256':sha(cache/'manifest.json'),
 'source_contract':manifest['cache_contract_hash'],'source_shape':[48,120,30,52],'num_frames':477,'selected_frames':[1,477],
 'ids':ids,'joint_names':names,'counts':dict(counts),'rejected':rejected,'label_identities':hashes,
 'targets_sha256':sha(out/'targets.npz'),'reference_index_sha256':sha(Path(a.reference_index)/'manifest.json'),
 'split':'SHA256(sparse-v1:clip_id)%20==0; clip-level holdout, not actor/action-disjoint',
 'targets':reference['targets'],'builder_sha256':sha(__file__)}
(out/'manifest.json').write_text(json.dumps(result,indent=2))
(out/'INDEX_COMPLETE').write_text(sha(out/'manifest.json')+'\n')
print(json.dumps({'counts':dict(counts),'rejected':rejected},indent=2),flush=True)
