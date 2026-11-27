"""Native-resolution JTA multi-person clips. Source annotations are never modified."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np

WIDTH, HEIGHT, CLIP = 832, 480, 81
XS, YS = (0, 544, 1088), (0, 300, 600)
TILES = [(x, y) for y in YS for x in XS]
LIMBS = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(2,7),(7,8),
         (8,9),(9,10),(2,11),(11,12),(12,13),(13,14),(14,15),
         (15,16),(16,17),(17,18),(15,19),(19,20),(20,21)]
CONTRACT = dict(schema='jta_native_tiles_v1', source_size=[1920,1080],
    tile_size=[WIDTH,HEIGHT], tile_origins=TILES, resize=False,
    frames=CLIP, frame_stride=1, fps=30, joints=22,
    target='all people with >=1 visible, in-crop joint in a frame',
    visibility='not occluded AND not self_occluded',
    presence='observable in this frame; not physical existence behind opaque occluders',
    xy='(source_pixel - tile_origin) / [832,480]; never clamp',
    xyz='camera meters, relative to midpoint of right/left hip (16,19)',
    root_camera='absolute midpoint of hips, stored separately in meters',
    intrinsic='fx=fy=1158; cx=960-x0; cy=540-y0',
    masks='in_crop/visible/valid_3d retained separately; no loss for out-of-crop joints',
    split='official source-sequence split, before windows and tiles',
    person_identity='(official split, sequence, person_id); clip slot is NOT person_id',
    inference='all nine fixed tiles, without GT filtering; merge duplicate poses in original pixels',
    empty_tiles='retained in index for negative sampling; positives cannot be inferred from GT at inference',
    fully_hidden='not a positive detection target; original rows preserved in source',
    overflow='fail if clip track count exceeds model capacity; never truncate people')

def load_people(path):
    """Return one row per complete person-frame, with original 1-based frame IDs."""
    a = np.asarray(json.loads(Path(path).read_text()), dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 10 or not np.isfinite(a).all():
        raise ValueError(f'Invalid annotation: {path}')
    # Do not rely on undocumented file ordering.
    if len(a) % 22:
        raise ValueError(f'Incomplete joint groups: {path}')
    a = a[np.lexsort((a[:,2], a[:,1], a[:,0]))].reshape(-1,22,10)
    assert np.all(a[:,:,2] == np.arange(22))
    assert np.all(a[:,:,0] == a[:,:1,0]) and np.all(a[:,:,1] == a[:,:1,1])
    assert np.all(a[:,:,:3] == np.floor(a[:,:,:3]))
    assert np.isin(a[:,:,8:], [0,1]).all()
    return a

def geometry(a, origin):
    xy = a[:,:,3:5] - np.array(origin, dtype=np.float32)
    xyz = a[:,:,5:8]
    valid3 = np.isfinite(xyz).all(-1) & (xyz[:,:,2] > 0)
    inside = ((xy[:,:,0] >= 0) & (xy[:,:,0] < WIDTH) &
              (xy[:,:,1] >= 0) & (xy[:,:,1] < HEIGHT) & valid3)
    visible = inside & (a[:,:,8] == 0) & (a[:,:,9] == 0)
    present = visible.any(-1)
    return xy, xyz, inside, visible, valid3, present

def make_targets(a, start_frame, origin, num_frames=CLIP, capacity=None):
    a = a[(a[:,0,0] >= start_frame) & (a[:,0,0] < start_frame + num_frames)]
    xy, xyz, inside, visible, valid3, present = geometry(a, origin)
    ids = np.unique(a[present,0,1]).astype(np.int64)
    if capacity is not None and len(ids) > capacity:
        raise ValueError(f'{len(ids)} tracks > capacity {capacity}; refuse silent truncation')
    n = len(ids)
    result = dict(person_ids=ids, xy=np.zeros((num_frames,n,22,2),np.float32),
        xyz=np.zeros((num_frames,n,22,3),np.float32),
        root_camera=np.zeros((num_frames,n,3),np.float32),
        present=np.zeros((num_frames,n),bool), annotated=np.zeros((num_frames,n),bool),
        in_crop=np.zeros((num_frames,n,22),bool), visible=np.zeros((num_frames,n,22),bool),
        valid_3d=np.zeros((num_frames,n,22),bool),
        tile_origin=np.asarray(origin,np.int32), frame_ids=np.arange(start_frame,start_frame+num_frames),
        intrinsic=np.array([[1158,0,960-origin[0]],[0,1158,540-origin[1]],[0,0,1]],np.float32))
    if not n:
        return result
    keep = np.isin(a[:,0,1],ids)
    frame = a[keep,0,0].astype(np.int64) - start_frame
    slot = np.searchsorted(ids, a[keep,0,1].astype(np.int64))
    assert len(np.unique(frame*n+slot)) == len(frame)
    root = (xyz[keep,16] + xyz[keep,19]) * .5
    result['xy'][frame,slot] = xy[keep] / np.array([WIDTH,HEIGHT],np.float32)
    result['xyz'][frame,slot] = xyz[keep] - root[:,None]
    result['root_camera'][frame,slot] = root
    result['annotated'][frame,slot] = True
    for key,value in [('present',present),('in_crop',inside),('visible',visible),('valid_3d',valid3)]:
        result[key][frame,slot] = value[keep]
    # Explicit masks, including hip validity, prevent padded/absent labels entering losses.
    root_valid = result['valid_3d'][:,:,16] & result['valid_3d'][:,:,19]
    result['valid_3d'] &= root_valid[:,:,None] & result['in_crop'] & result['present'][:,:,None]
    return result

def scan_sequence(job):
    import cv2
    root, split, path = job
    p = Path(path); a = load_people(p)
    video = Path(root)/'videos'/split/(p.stem+'.mp4')
    if not video.is_file(): raise FileNotFoundError(video)
    cap = cv2.VideoCapture(str(video))
    last = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    cap.release()
    assert size == [1920,1080] and abs(fps-30)<1e-3 and last>=CLIP, (video,size,fps,last)
    first = 1
    # Some videos end with no people: annotation extent is not video duration.
    assert a[:,0,0].min() >= first and a[:,0,0].max() <= last, (path,last)
    starts = sorted(set(list(range(first,last-CLIP+2,CLIP)) + [last-CLIP+1]))
    rows = []
    for tile,(x,y) in enumerate(TILES):
        _,_,inside,visible,_,present = geometry(a,(x,y))
        for start in starts:
            selected = present & (a[:,0,0] >= start) & (a[:,0,0] < start+CLIP)
            frames = a[selected,0,0].astype(np.int64)
            ids = np.unique(a[selected,0,1]).astype(int)
            counts = np.bincount(frames-start,minlength=CLIP)
            rows.append(dict(id=f'{split}_{p.stem}_f{start:04d}_t{tile}',split=split,
                sequence=p.stem,video=str(video),annotation=str(p),start_frame=start,
                num_frames=CLIP,tile=tile,origin=[x,y],track_count=len(ids),
                max_people_per_frame=int(counts.max()),positive_frames=int((counts>0).sum()),
                person_frames=int(counts.sum()),empty=not bool(selected.any())))
    return dict(sequence=p.stem,split=split,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                person_frames=len(a),video_frames=last,
                annotated_frame_count=len(np.unique(a[:,0,0])),
                annotation_frame_range=[int(a[:,0,0].min()),int(a[:,0,0].max())]),rows

def build_index(root, output, splits, workers):
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    (out/'contract.json').write_text(json.dumps(CONTRACT,indent=2))
    jobs=[(root,s,str(p)) for s in splits for p in sorted((Path(root)/'annotations'/s).glob('*.json'))]
    if not jobs: raise ValueError('No annotations found')
    summaries=[]; n=positive=max_tracks=max_people=0; hist={}; start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool, (out/'index.jsonl').open('w') as f:
        for source,rows in pool.map(scan_sequence,jobs):
            summaries.append(source)
            for r in rows:
                f.write(json.dumps(r)+'\n'); n+=1; positive+=int(not r['empty'])
                max_tracks=max(max_tracks,r['track_count']); max_people=max(max_people,r['max_people_per_frame'])
                hist[r['track_count']]=hist.get(r['track_count'],0)+1
            f.flush()
            print(json.dumps(dict(sequences=len(summaries),total=len(jobs),clips=n,positive=positive,
                max_tracks=max_tracks,max_people=max_people,elapsed_sec=round(time.time()-start,1))),flush=True)
    report=dict(sequences=summaries,total_clips=n,positive_clips=positive,empty_clips=n-positive,
                max_clip_tracks=max_tracks,max_people_per_frame=max_people,track_count_histogram=hist,
                elapsed_sec=time.time()-start,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    (out/'INDEX_COMPLETE').touch()

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--splits',nargs='+',default=['train','val']); p.add_argument('--workers',type=int,default=4)
    args=p.parse_args(); build_index(args.root,args.output,args.splits,args.workers)
