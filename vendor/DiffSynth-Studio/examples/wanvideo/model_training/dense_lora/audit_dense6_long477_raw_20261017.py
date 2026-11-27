"""Read-only audit of newly supplied labels against the fixed RGB cache index."""
import concurrent.futures
import json
import os
from pathlib import Path

ROOT = Path('/pfs/mayuema/dense/long481_prod3000_20260906/outputs/train')
CACHE = Path('/pfs/mayuema/dense/wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40')
index = json.loads((CACHE/'source_index.json').read_text())
expected = {f'{i:06d}.exr' for i in range(1,482)}

def audit(row):
    cid = row['id']; clip = ROOT/cid
    result = {'id':cid, 'counts':{}, 'errors':[]}
    for task in ('albedo','optical_flow'):
        try:
            names = {e.name for e in os.scandir(clip/task) if e.is_file() and not e.name.startswith('.')}
            result['counts'][task] = len(names)
            if names != expected:
                result['errors'].append({'task':task,'missing':sorted(expected-names)[:5],'extra':sorted(names-expected)[:5]})
        except OSError as exc: result['errors'].append(str(exc))
    marker=clip/'.albedo_flow_complete.json'
    if marker.exists(): result['marker']=json.loads(marker.read_text())
    return result

with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    rows=list(pool.map(audit,index['samples']))
bad=[r for r in rows if r['errors']]
print(json.dumps({'clips':len(rows),'complete':len(rows)-len(bad),'bad_count':len(bad),
                  'bad_examples':bad[:8],'first_sample':rows[0]},ensure_ascii=False),flush=True)
