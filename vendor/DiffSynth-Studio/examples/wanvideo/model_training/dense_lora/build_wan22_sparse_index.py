"""Audit sparse coordinates against the existing RGB cache; never encode targets as images."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--output_dir', required=True)
p.add_argument('--dataset_root', default='/pfs/mayuema/dense/dataset/prod_final_9110_20260816/train')
p.add_argument('--source_cache', default='/pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40')
a = p.parse_args()
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=False)
cache, root = Path(a.source_cache), Path(a.dataset_root)
index_file = cache/'dataset_index.json'
index = json.loads(index_file.read_text())
assert (index['num_frames'],index['height'],index['width']) == (81,480,832)
assert (cache/'CACHE_COMPLETE').is_file()
ids, xy, xyz, masks, splits, rejected = [], [], [], [], [], []
joint_names = None
counts = Counter()
for i, sample in enumerate(index['samples']):
    cid = sample['id']; g = root/cid/'genception'
    paths = [g/'keypoints_2d/keypoints_2d.npz',g/'keypoints_3d/keypoints_3d.npz',g/'keypoints_3d/root_relative_camera_per_height.npy',g/'keypoints_3d/schema.json']
    if not all(f.is_file() for f in paths):
        counts['missing_sparse'] += 1; continue
    try:
        assert (cache/(cid+'.pt')).is_file(), 'missing source cache'
        schema = json.loads(paths[3].read_text())
        assert schema['3d']['model_target'] == 'root_relative_camera_per_height.npy'
        with np.load(paths[0],allow_pickle=False) as d2, np.load(paths[1],allow_pickle=False) as d3:
            names = list(d2['joint_names'])
            if joint_names is None: joint_names = names
            assert names == joint_names == list(d3['joint_names']) == schema['joint_order'], 'joint order mismatch'
            frame_ids = np.array([int(Path(f).stem) for f in sample['rgb_names']])
            assert np.array_equal(frame_ids,d2['frame_numbers']) and np.array_equal(frame_ids,d3['frame_numbers']), 'frame alignment'
            target2 = d2['image_xy_01'].astype(np.float32)
            mask = d2['in_front'] & d2['in_frame']
            assert target2.shape == (81,24,2) and mask.shape == (81,24)
            assert np.isfinite(target2).all()
            err = np.abs(target2-d2['pixel_xy']/np.array([832,480])).max()
            assert err < 1e-5, f'2D normalization mismatch {err}'
            target3 = np.load(paths[2],allow_pickle=False).astype(np.float32)
            assert target3.shape == (81,24,3) and np.isfinite(target3).all()
            assert np.abs(target3[:,0]).max() < 1e-5, '3D pelvis is not zero'
            centered = d3['camera_opencv_m']-d3['camera_opencv_m'][:,:1]
            height = float((centered*target3).sum()/np.square(target3).sum())
            assert np.isfinite(height) and height > 0
            assert np.abs(centered/height-target3).max() < 5e-4, '3D coordinate/height mismatch'
            assert mask.any(), 'no valid 2D joints'
        # Deterministic clip-level holdout, not a scene/actor generalization benchmark.
        val = int(hashlib.sha256(('sparse-v1:'+cid).encode()).hexdigest()[:8],16)%20 == 0
        ids.append(cid); xy.append(target2); xyz.append(target3); masks.append(mask); splits.append(val)
        counts['val' if val else 'train'] += 1
    except Exception as e:
        counts['invalid_sparse'] += 1; rejected.append({'id':cid,'reason':str(e)})
    if (i+1)%500 == 0: print(f'audited={i+1} accepted={len(ids)} counts={dict(counts)}',flush=True)
assert counts['train'] > 0 and counts['val'] > 0
np.savez_compressed(out/'targets.npz',xy=np.stack(xy),xyz=np.stack(xyz),mask=np.stack(masks),is_val=np.array(splits))
manifest = {'schema':'wan22_sparse_coordinates_v1','dataset_root':str(root),'source_cache':str(cache),'source_index_sha256':hashlib.sha256(index_file.read_bytes()).hexdigest(),'ids':ids,'joint_names':joint_names,'counts':dict(counts),'rejected':rejected,'prompt':'output: 2D and 3D human keypoints','targets':{'2d':'image_xy_01, pixel_xy/[832,480], mask=in_front AND in_frame','3d':'root_relative_camera_per_height.npy; +X right,+Y down,+Z forward; pelvis=0'},'split':'SHA256(sparse-v1:clip_id)%20==0; clip-level 5%, not actor/scene-disjoint','source_shape':[48,21,30,52]}
manifest['targets_sha256'] = hashlib.sha256((out/'targets.npz').read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(out/'INDEX_COMPLETE').touch()
print(json.dumps({'counts':dict(counts),'rejected_first10':rejected[:10]},ensure_ascii=False),flush=True)
