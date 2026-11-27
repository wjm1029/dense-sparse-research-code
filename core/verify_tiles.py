"""Exercise geometry/padding/overflow and export inspectable native-size GT crops."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from jta_tiles import *

def check_contract():
    cover=np.zeros((1080,1920),np.int16)
    for x,y in TILES: cover[y:y+HEIGHT,x:x+WIDTH]+=1
    assert cover.min() >= 1
    a=np.zeros((3,22,10),np.float32)
    a[:,:,0]=np.array([1,2,2])[:,None]; a[:,:,1]=np.array([7,7,99])[:,None]
    a[:,:,2]=np.arange(22); a[:,:,3:5]=[600,350]; a[:,:,5:8]=[1,2,10]
    # A visible 40px-tall skeleton, partial crop, absent frame and fully hidden frame.
    a[0,:,4]=np.linspace(330,370,22)
    a[0,0,3]=500
    a[1,:,8]=1
    a[2,:,3:5]=[700,400]
    t=make_targets(a,1,(544,300),num_frames=3)
    assert t['person_ids'].tolist()==[7,99]
    assert t['present'].tolist()==[[True,False],[False,True],[False,False]]
    assert not t['in_crop'][0,0,0] and t['xy'][0,0,0,0]<0
    assert not t['valid_3d'][1,0].any() and not t['annotated'][2].any()
    assert np.isclose(np.ptp(t['xy'][0,0,:,1])*HEIGHT,40)
    assert np.allclose(t['xyz'],0)
    assert np.allclose(t['xy'][0,0]*[WIDTH,HEIGHT]+[544,300],a[0,:,3:5])
    point=np.array([1,2,10.]); k=t['intrinsic']; project=k@point
    assert np.allclose(project[:2]/project[2], [1158*.1+960-544,1158*.2+540-300])
    try: make_targets(a,1,(544,300),num_frames=3,capacity=1)
    except ValueError: pass
    else: raise AssertionError('Overflow was silently accepted')
    empty=make_targets(a,1,(1088,600),num_frames=3)
    assert empty['xy'].shape==(3,0,22,2)
    return dict(grid_coverage_min=int(cover.min()),grid_coverage_max=int(cover.max()),
                coordinate_roundtrip=True,forty_pixel_height_preserved=True,
                crop_intrinsics=True,padding_and_visibility=True,overflow_rejected=True)

def preview(root,out,sequence):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=root/'annotations'/'train'/f'seq_{sequence}.json'
    a=load_people(path);frame_id=301
    cap=cv2.VideoCapture(str(root/'videos'/'train'/f'seq_{sequence}.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id-1);ok,frame=cap.read();cap.release()
    assert ok and frame.shape==(1080,1920,3)
    source=a[a[:,0,0]==frame_id]
    marked=frame.copy();tiles=[];details=[];source_seen=set();tile_seen=set()
    for row in source:
        visible=(row[:,8:]==0).all(-1)&(row[:,3]>=0)&(row[:,3]<1920)&(row[:,4]>=0)&(row[:,4]<1080)&(row[:,7]>0)
        for joint in np.flatnonzero(visible): source_seen.add((int(row[0,1]),int(joint)))
    for tile,(x,y) in enumerate(TILES):
        t=make_targets(a,frame_id,(x,y),num_frames=81)
        np.savez_compressed(out/f'seq_{sequence}_f0301_t{tile}_targets.npz',**t)
        raw=frame[y:y+HEIGHT,x:x+WIDTH].copy()
        assert raw.shape==(HEIGHT,WIDTH,3)
        # PNG preserves source decoded RGB exactly; no resize/re-encoding artifacts.
        rawpath=out/f'seq_{sequence}_t{tile}_rgb.png';cv2.imwrite(str(rawpath),raw)
        assert np.array_equal(cv2.imread(str(rawpath)),frame[y:y+HEIGHT,x:x+WIDTH])
        crop=raw.copy();xy=t['xy'][0]*[WIDTH,HEIGHT]
        for slot,pid in enumerate(t['person_ids']):
            if not t['present'][0,slot]: continue
            color=tuple(int(v) for v in np.random.default_rng(int(pid)).integers(80,256,3))
            inside=t['in_crop'][0,slot];vis=t['visible'][0,slot]
            for j in np.flatnonzero(vis):tile_seen.add((int(pid),int(j)))
            for j,k in LIMBS:
                if inside[j] and inside[k]:
                    cv2.line(crop,tuple(np.rint(xy[slot,j]).astype(int)),tuple(np.rint(xy[slot,k]).astype(int)),color,1,cv2.LINE_AA)
            for j in np.flatnonzero(inside):
                cv2.circle(crop,tuple(np.rint(xy[slot,j]).astype(int)),2,color if vis[j] else (100,100,100),-1)
            anchor=xy[slot,np.flatnonzero(vis)[0]].astype(int)
            cv2.putText(crop,str(pid),tuple(anchor),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1,cv2.LINE_AA)
        cv2.putText(crop,f'tile {tile} origin=({x},{y}) people={t["present"][0].sum()} | GT, no resize',
                    (12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(out/f'seq_{sequence}_t{tile}_gt.png'),crop);tiles.append(crop)
        cv2.rectangle(marked,(x,y),(x+WIDTH-1,y+HEIGHT-1),(0,255,255),2)
        cv2.putText(marked,str(tile),(x+10,y+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        details.append(dict(tile=tile,origin=[x,y],visible_people=int(t['present'][0].sum()),
                            clip_tracks=len(t['person_ids']),raw_crop_exact=True))
    assert source_seen==tile_seen, 'Visible joints were lost by tiling'
    cv2.imwrite(str(out/f'seq_{sequence}_grid.jpg'),marked)
    cv2.imwrite(str(out/f'seq_{sequence}_all_tiles.jpg'),np.concatenate([np.concatenate(tiles[i:i+3],1) for i in (0,3,6)],0))
    return dict(sequence=sequence,frame=frame_id,visible_joints=len(source_seen),
                all_visible_joints_covered=True,tiles=details)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();checks=check_contract()
    previews=[preview(args.root,args.out,s) for s in (85,170)]
    report=dict(checks=checks,previews=previews,note='Ground truth only; no trained predictions.')
    Path(args.out,'VERIFY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
