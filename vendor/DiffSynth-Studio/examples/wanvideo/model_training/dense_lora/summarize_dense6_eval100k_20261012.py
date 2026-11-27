"""Merge fixed-clip evaluation shards and compare exactly matching 50k/70k samples."""
import json
from pathlib import Path
import numpy as np

root=Path('/pfs/mayuema/dense/wan/outputs')
out=root/'dense6_flow32_train_fit_100k_20261012'
rows=[]
for rank in range(4):
    shard=out/f'shard_{rank}'
    assert (shard/'EVAL_COMPLETE').is_file()
    rows.extend(json.loads((shard/'metrics.json').read_text()))
rows.sort(key=lambda r:(r['clip_index'],r['task']))
assert len(rows)==24 and len({(r['clip_id'],r['task']) for r in rows})==24
baselines={}
for label,directory in [('50k','dense6_flow32_train_fit_50k_20261010'),('70k','dense6_flow32_train_fit_70k_20261011')]:
    baseline=json.loads((root/directory/'metrics.json').read_text())
    assert {(r['clip_id'],r['task']) for r in baseline}=={(r['clip_id'],r['task']) for r in rows}
    baselines[label]=baseline
def aggregate(values):
    result={}
    for task in sorted({r['task'] for r in values}):
        selected=[r for r in values if r['task']==task]
        keys=['latent_mse','decoded_mae']
        if task=='optical_flow': keys += [k for k in selected[0] if k.startswith(('raw_','vae_reference_'))]
        result[task]={key:float(np.mean([r[key] for r in selected if r.get(key) is not None])) for key in keys}
    return result
summary=aggregate(rows)
comparison={label:aggregate(values) for label,values in baselines.items()}
comparison['100k']=summary
(out/'metrics.json').write_text(json.dumps(rows,indent=2))
(out/'summary.json').write_text(json.dumps({'split':'training_fit_only','num_clips':4,'checkpoint':'/pfs/mayuema/dense/wan/outputs/wan22_dense6_hyperlora_bidir_flow32_stage1_50k/step-100000.safetensors','metrics':summary},indent=2))
(out/'comparison_50k_70k_100k.json').write_text(json.dumps(comparison,indent=2))
lines=['# Dense6 flow32: 100k fixed training-clip fit evaluation','','Same four training clips and evaluation implementation as 50k/70k. This is not held-out validation or a real-case generalization benchmark.','','| Task / metric | 50k | 70k | 100k | Change vs 70k |','|---|---:|---:|---:|---:|']
for task in ('depth','normal','matte','raymap','albedo','optical_flow'):
    key='raw_prediction_epe_px' if task=='optical_flow' else 'decoded_mae'
    values=[comparison[label][task][key] for label in ('50k','70k','100k')]
    delta=(values[2]/values[1]-1)*100
    lines.append(f'| {task}: {key} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {delta:+.2f}% |')
flow=summary['optical_flow']
lines.extend(['',f'Moving-pixel raw flow EPE (>1px GT motion): {flow["raw_prediction_moving_epe_px"]:.6f}px.',f'VAE reference raw flow EPE: {flow["raw_vae_ceiling_epe_px"]:.6f}px; zero-flow baseline: {flow["raw_zero_flow_epe_px"]:.6f}px.','','Decoded MAE compares to VAE-decoded cached GT, not raw physical targets. Flow EPE compares to raw EXR in pixels, scale32, delayed one frame. Clip-level means are unweighted; four training clips are not representative of the full dataset.'])
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
(out/'EVAL_COMPLETE').touch()
print(json.dumps(comparison,indent=2),flush=True)
