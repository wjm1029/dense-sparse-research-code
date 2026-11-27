"""Sample raw actor-region albedo pixels, not model predictions, at three frames."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)
out=Path('/pfs/mayuema/dense/wan/outputs/long1079_texture_audit_20261018')
payload=json.loads((out/'INPUT.json').read_text())
root=Path('/pfs/mayuema/dense/long481_prod3000_20260906/outputs/train')
def inspect(row):
    frames=[]
    try:
        for frame in (1,239,477):
            raw=cv2.imread(str(root/row['id']/'albedo'/f'{frame:06d}.exr'),cv2.IMREAD_UNCHANGED)
            assert raw is not None and raw.shape==(480,832,3)
            with Image.open(root/row['id']/'matte'/f'{frame:06d}.png') as image:
                matte=np.asarray(image)
            if matte.ndim==3: matte=matte[...,0]
            mask=matte>(32767 if matte.max()>255 else 127)
            rgb=raw[...,::-1][mask]; assert len(rgb)>0 and np.isfinite(rgb).all()
            r,g,b=rgb.T
            purple=(r>.2)&(b>.2)&(g<np.minimum(r,b)*.15)&(np.abs(r-b)<np.maximum(r,b)*.25)
            frames.append({'frame':frame,'foreground_pixels':len(rgb),'magenta_fraction':float(purple.mean()),
                           'median_rgb':np.median(rgb,axis=0).tolist()})
        return {'id':row['id'],'frames':frames,'strong_magenta':sum(f['magenta_fraction']>=.2 for f in frames)>=2,'error':None}
    except Exception as exc: return {'id':row['id'],'frames':frames,'error':str(exc)}
records=[]
with ThreadPoolExecutor(max_workers=8) as pool:
    for i,result in enumerate(pool.map(inspect,payload['clips'])):
        records.append(result)
        if (i+1)%100==0: print('PIXEL_AUDIT '+str(i+1)+'/1079',flush=True)
(out/'PIXELS.json').write_text(json.dumps({'frames':[1,239,477],
  'rule':'actor albedo R,B>.2; G<.15*min(R,B); abs(R-B)<.25*max(R,B); fraction>=.2 at >=2/3 frames',
  'interpretation':'visual evidence only; true purple materials can match; cross-check material path audit',
  'clips':records},indent=2))
print('PIXEL_AUDIT_COMPLETE '+json.dumps({'clips':len(records),'errors':sum(r['error'] is not None for r in records),
  'strong_magenta':sum(r.get('strong_magenta',False) for r in records)}),flush=True)
