"""Read-only mapping from the fixed1079 long training clips to exact actor caches."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

base=Path('/pfs/mayuema/dense')
source=base/'wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40/source_index.json'
index=json.loads(source.read_text()); cfg=json.loads((base/'long481_prod3000_20260906/config/long_481.json').read_text())
out=base/'wan/outputs/long1079_texture_audit_20261018'; out.mkdir(exist_ok=False)
def process(row):
    path=Path(index['dataset_root'])/row['id']/'metadata.json'; meta=json.loads(path.read_text())
    actor=cfg['actor']; camera=cfg['camera']
    identity={'schema':'e2p-actor-animation-v3','blender_version':[4,5,12],
      'character_id':meta['character']['id'],'character_path':str(Path(meta['character']['path']).resolve()),
      'animation_id':meta['animation']['id'],'animation_path':str(Path(meta['animation']['path']).resolve()),'frames':481,
      'target_height_m':actor.get('target_height_m'),'animal_target_height_m':actor.get('animal_target_height_m'),
      'hair_budget':actor.get('hair_budget'),'foot_clearance_m':.25,
      'collision_frame_stride':int(actor.get('collision_frame_stride',4)),
      'visibility_surface_points':int(camera.get('visibility_surface_points',256))}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:32]
    directory=Path(cfg['paths']['actor_cache_root'])/key
    try:
        manifest=json.loads((directory/'manifest.json').read_text())
        assert manifest['identity']==identity and manifest['cache_key']==key
        assert (directory/'DONE').exists() and (directory/'actor.blend').exists()
        error=None
    except Exception as exc: error=str(exc)
    val=int(hashlib.sha256(('dense6-long-v1:'+row['id']).encode()).hexdigest()[:8],16)%20==0
    return dict(row,cache_key=key,actor_blend=str(directory/'actor.blend'),actor_source=meta['character']['path'],
      dataset_root=index['dataset_root'],validation=val,mapping_error=error)
with ThreadPoolExecutor(max_workers=12) as pool: rows=list(pool.map(process,index['samples']))
caches={r['cache_key']:{'cache_key':r['cache_key'],'path':r['actor_blend'],'source_path':r['actor_source']} for r in rows if not r['mapping_error']}
payload={'scope':'fixed1079 synthetic long clips currently used by Dense6; actor caches only',
  'source_index':str(source),'source_index_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'clips':rows,'caches':list(caches.values()),'mapping_errors':[r for r in rows if r['mapping_error']]}
(out/'INPUT.json').write_text(json.dumps(payload,indent=2))
print(json.dumps({'output':str(out),'clips':len(rows),'caches':len(caches),'mapping_errors':len(payload['mapping_errors']),
  'subjects':dict(Counter(r['subject_label'] for r in rows))},indent=2),flush=True)
