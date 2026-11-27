"""Read-only three-frame raw RGB/albedo appearance audit of actual Dense6 bidir clips.

Appearance screening is not a full historical asset-path audit; report it as an
estimate with explicit thresholds, not universal proof of clean/corrupt samples.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import numpy as np
from PIL import Image,ImageDraw
from dense4_dataset import load_subject_map

cv2.setNumThreads(1)
base=Path('/pfs/mayuema/dense/wan')
index_path=base/'cache/wan22_depth_latents_bf16_bright40/dataset_index.json'
index=json.loads(index_path.read_text()); root=Path(index['root']); rows=index['samples']
assert len(rows)==7350 and index['num_frames']==81
subjects=load_subject_map('/pfs/mayuema/dense/prompts')
expected=json.loads((base/'cache/wan22_dense4_targets_bf16_bright40/manifest.json').read_text())['subject_counts']
assert Counter(subjects[r['id']] for r in rows)==expected
out=base/'outputs/bidir7350_texture_appearance_audit_20261018'; out.mkdir(exist_ok=False)

def fraction(rgb,mask):
    values=rgb[mask]; assert len(values)>0 and np.isfinite(values).all()
    r,g,b=values.T
    purple=(r>.2)&(b>.2)&(g<np.minimum(r,b)*.15)&(np.abs(r-b)<np.maximum(r,b)*.25)
    return float(purple.mean()),np.median(values,axis=0).tolist()

def inspect(row):
    cid=row['id']; clip=root/cid; frames=[]
    try:
        meta=json.loads((clip/'metadata.json').read_text())
        for frame in (1,41,81):
            a=cv2.imread(str(clip/'albedo'/f'{frame:06d}.exr'),cv2.IMREAD_UNCHANGED)
            assert a is not None and a.shape==(480,832,3)
            with Image.open(clip/'matte'/f'{frame:06d}.png') as im: matte=np.asarray(im)
            if matte.ndim==3: matte=matte[...,0]
            mask=matte>(32767 if matte.max()>255 else 127)
            if not mask.any():
                frames.append({'frame':frame,'foreground_pixels':0,'albedo_magenta_fraction':None,
                  'rgb_magenta_fraction':None,'albedo_median_rgb':None,'rgb_median_rgb':None})
                continue
            with Image.open(clip/'rgb'/f'{frame:06d}.png') as im: rgb=np.asarray(im.convert('RGB'),dtype=np.float32)/255
            af,am=fraction(a[...,::-1],mask); rf,rm=fraction(rgb,mask)
            frames.append({'frame':frame,'foreground_pixels':int(mask.sum()),'albedo_magenta_fraction':af,
              'rgb_magenta_fraction':rf,'albedo_median_rgb':am,'rgb_median_rgb':rm})
        return {'id':cid,'subject':subjects[cid],'character_id':meta['character']['id'],
          'character_path':meta['character']['path'],'frames':frames,
          'visible_frame_count':sum(f['foreground_pixels']>0 for f in frames),
          'albedo_strong_magenta':sum(f['albedo_magenta_fraction'] is not None and f['albedo_magenta_fraction']>=.2 for f in frames)>=2,
          'rgb_strong_magenta':sum(f['rgb_magenta_fraction'] is not None and f['rgb_magenta_fraction']>=.2 for f in frames)>=2,'error':None}
    except Exception as exc: return {'id':cid,'subject':subjects[cid],'frames':frames,'error':str(exc)}

results=[]
with ThreadPoolExecutor(max_workers=12) as pool:
    for i,row in enumerate(pool.map(inspect,rows)):
        results.append(row)
        if (i+1)%250==0: print('BIDIR_PIXEL_AUDIT '+str(i+1)+'/7350',flush=True)
groups={}; assets={}; categories=Counter()
for r in results:
    category='error' if r['error'] else 'insufficient_visibility' if r['visible_frame_count']<2 else 'both' if r['albedo_strong_magenta'] and r['rgb_strong_magenta'] else 'albedo_only' if r['albedo_strong_magenta'] else 'rgb_only' if r['rgb_strong_magenta'] else 'neither_flagged'
    r['category']=category; categories[category]+=1
    for name in ('all',r['subject']):
        g=groups.setdefault(name,Counter()); g['total']+=1; g[category]+=1
        if not r['error']:
            g['albedo_flagged']+=int(r['albedo_strong_magenta']); g['rgb_flagged']+=int(r['rgb_strong_magenta'])
    if not r['error']:
        g=assets.setdefault(r['character_id'],{'path':r['character_path'],'total':0,'albedo_flagged':0,'rgb_flagged':0})
        g['total']+=1; g['albedo_flagged']+=int(r['albedo_strong_magenta']); g['rgb_flagged']+=int(r['rgb_strong_magenta'])
(out/'CLIPS.json').write_text(json.dumps(results,indent=2))
summary={'scope':'actual7350 clips used by Dense6 bidirectional flow32 + dispersion stages; min_rgb_mean40',
  'index_sha256':hashlib.sha256(index_path.read_bytes()).hexdigest(),'frames':[1,41,81],
  'rule':'actor pixels R,B>.2; G<.15*min(R,B); abs(R-B)<.25*max(R,B); >=20% pixels at >=2 of3 frames',
  'caveat':'appearance estimate only; genuine purple materials may match, subtle failures can be missed; no historical texture-path audit',
  'groups':{k:dict(v) for k,v in groups.items()},'categories':dict(categories),'assets':assets,
  'errors':[r for r in results if r['error']]}
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
# Representative raw-data panels for manual verification (no model inference).
examples=[]
for subject in groups:
    if subject=='all': continue
    candidates=[r for r in results if r['subject']==subject and r['category'] in ('both','albedo_only','rgb_only')]
    if not candidates: continue
    r=max(candidates,key=lambda r:np.mean([f['albedo_magenta_fraction'] for f in r['frames'] if f['albedo_magenta_fraction'] is not None]))
    clip=root/r['id']; canvas=Image.new('RGB',(832,780),'white'); draw=ImageDraw.Draw(canvas)
    for y,frame in enumerate((1,41,81)):
        with Image.open(clip/'rgb'/f'{frame:06d}.png') as im: rgb=im.convert('RGB').resize((416,240))
        a=cv2.imread(str(clip/'albedo'/f'{frame:06d}.exr'),cv2.IMREAD_UNCHANGED)[...,::-1]
        albedo=Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).resize((416,240))
        canvas.paste(rgb,(0,y*260+20)); canvas.paste(albedo,(416,y*260+20))
        draw.text((5,y*260+4),f'{subject} frame{frame}: raw RGB | raw albedo; {r["category"]}',fill='black')
    name=f'example_{subject}.jpg'; canvas.save(out/name,quality=90); examples.append({'subject':subject,'id':r['id'],'image':name})
(out/'EXAMPLES.json').write_text(json.dumps(examples,indent=2))
(out/'AUDIT_COMPLETE').write_text('success\n')
print(json.dumps(summary,indent=2),flush=True)
