"""Small, lossless RGB+target pilot bundles for a future multi-person overfit run."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from jta_tiles import CLIP, TILES, WIDTH, HEIGHT, load_people, make_targets

def main(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=[]
    # Same sequence-level official training split, with easy/crowded/empty probes.
    for seq, tile_ids in [(85,[0]),(170,[4,0])]:
        a=load_people(root/'annotations'/'train'/f'seq_{seq}.json')
        cap=cv2.VideoCapture(str(root/'videos'/'train'/f'seq_{seq}.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES,300)
        buffers={tile:np.empty((CLIP,HEIGHT,WIDTH,3),np.uint8) for tile in tile_ids}
        for i in range(CLIP):
            ok,frame=cap.read();assert ok
            assert int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) == 301+i
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            for tile in tile_ids:
                x,y=TILES[tile];buffers[tile][i]=frame[y:y+HEIGHT,x:x+WIDTH]
        cap.release()
        for tile in tile_ids:
            name=f'train_seq_{seq}_f0301_t{tile}'
            targets=make_targets(a,301,TILES[tile])
            np.save(out/(name+'_rgb.npy'),buffers[tile],allow_pickle=False)
            np.savez_compressed(out/(name+'_targets.npz'),**targets)
            assert np.array_equal(np.load(out/(name+'_rgb.npy'),mmap_mode='r'),buffers[tile])
            manifest.append(dict(id=name,source_sequence=seq,split='train',tile=tile,
                origin=TILES[tile],start_frame_1based=301,frames=CLIP,
                rgb_shape=list(buffers[tile].shape),rgb_order='RGB',rgb_dtype='uint8',resize=False,
                rgb_file=name+'_rgb.npy',target_file=name+'_targets.npz',
                track_count=len(targets['person_ids']),
                max_people_per_frame=int(targets['present'].sum(1).max()),
                positive_frames=int(targets['present'].any(1).sum())))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'PILOT_DATA_COMPLETE').touch()
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();main(args.root,args.out)
