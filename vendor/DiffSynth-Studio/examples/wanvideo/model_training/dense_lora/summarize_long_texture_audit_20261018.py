"""Weighted clip counts, separating broken texture references from visible magenta."""
from collections import Counter
import csv
import json
from pathlib import Path

out=Path('/pfs/mayuema/dense/wan/outputs/long1079_texture_audit_20261018')
assert all((out/f'RANK_{i}_COMPLETE').exists() for i in range(4))
source=json.loads((out/'INPUT.json').read_text()); pixel=json.loads((out/'PIXELS.json').read_text())
caches={r['cache_key']:r for path in sorted(out.glob('cache_rank*.jsonl')) for r in map(json.loads,path.read_text().splitlines())}
assert len(caches)==len(source['caches'])==116
pixels={r['id']:r for r in pixel['clips']}; assert len(pixels)==1079
rows=[]; groups={}; counts=Counter()
for clip in source['clips']:
    cache=caches.get(clip['cache_key']); color=pixels[clip['id']]
    status='audit_error' if clip['mapping_error'] or not cache or cache['audit_error'] or color['error'] else 'checked'
    broken=bool(cache and cache.get('missing_images')); broken_color=bool(cache and cache.get('missing_color_images'))
    magenta=color.get('strong_magenta',False)
    row={'id':clip['id'],'subject':clip['subject_label'],'validation':clip['validation'],'cache_key':clip['cache_key'],
      'status':status,'broken_texture_reference':broken,'broken_color_reference':broken_color,
      'strong_magenta':magenta,'broken_plus_visible_magenta':broken and magenta,
      'sampled_magenta_fractions':[f['magenta_fraction'] for f in color['frames']],
      'missing_images':[r['name'] for r in cache.get('missing_images',[])] if cache else []}
    rows.append(row)
    for name in ('all',clip['subject_label'],'validation' if clip['validation'] else 'train'):
        g=groups.setdefault(name,Counter()); g['total']+=1
        for key in ('broken_texture_reference','broken_color_reference','strong_magenta','broken_plus_visible_magenta'):
            g[key]+=int(row[key])
        g['audit_errors']+=int(status!='checked')
    counts['total']+=1
    for key in ('broken_texture_reference','broken_color_reference','strong_magenta','broken_plus_visible_magenta'):
        counts[key]+=int(row[key])
summary={'scope':source['scope'],'method':'material-output-reachable image textures; actual UDIM tiles; packed image handling; cached actor.blend path resolution',
  'pixel_method':pixel['rule'],'pixel_caveat':pixel['interpretation'],'clip_counts':dict(counts),
  'groups':{k:dict(v) for k,v in groups.items()},'unique_caches':len(caches),
  'broken_caches':sum(bool(r.get('missing_images')) for r in caches.values()),
  'cache_audit_errors':[r for r in caches.values() if r['audit_error']],
  'mapping_errors':source['mapping_errors'],'pixel_errors':[r for r in pixel['clips'] if r['error']],
  'not_audited':'short81 dataset, scene/background material paths; no renders or source data changed'}
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
(out/'CLIPS.json').write_text(json.dumps(rows,indent=2))
with (out/'clips.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['id','subject','validation','cache_key','status','broken_texture_reference',
      'broken_color_reference','strong_magenta','broken_plus_visible_magenta','sampled_magenta_fractions','missing_images'])
    writer.writeheader(); writer.writerows(rows)
lines=['# Long1079 actor texture-path audit','',summary['scope'],'',
  '| Group | Clips | Broken texture refs | Broken color refs | Broken refs + strong magenta |',
  '|---|---:|---:|---:|---:|']
for name,g in summary['groups'].items():
    lines.append(f'| {name} | {g["total"]} | {g["broken_texture_reference"]} | {g["broken_color_reference"]} | {g["broken_plus_visible_magenta"]} |')
lines+=['','Broken references are a structural risk; not every referenced missing map necessarily makes visible pixels purple.',
  'Strong-magenta is a three-frame heuristic, not a full481-frame pixel audit; true purple materials can match.',
  'Cross-check the two signals; neither alone proves all affected pixels/frames are corrupt.',
  'Actor texture paths only. Background scene assets and the short81 dataset have not been audited. No data/training was changed.']
(out/'README.md').write_text('\n'.join(lines)+'\n')
(out/'AUDIT_COMPLETE').write_text('success\n')
print(json.dumps(summary,indent=2),flush=True)
