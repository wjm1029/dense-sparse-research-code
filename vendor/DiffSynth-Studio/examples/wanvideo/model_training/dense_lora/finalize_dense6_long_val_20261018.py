"""Aggregate held-out Dense6 results and verify local video transfers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser(); p.add_argument('--directory',required=True); p.add_argument('--verify_local',action='store_true'); a=p.parse_args()
root=Path(a.directory)
tasks=('depth','normal','matte','raymap','albedo','optical_flow')
if a.verify_local:
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,checksum in manifest['files_sha256'].items(): assert sha(root/name)==checksum,name
    videos=[name for name in manifest['files_sha256'] if name.endswith('.mp4')]
    assert len(videos)==36
    for name in videos:
        stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
          '-show_entries','stream=nb_frames,codec_name,width,height','-of','json',str(root/name)],text=True))['streams'][0]
        assert int(stream['nb_frames'])==477 and stream['codec_name']=='h264'
        assert (stream['width'],stream['height'])==(1248,280)
    print('LOCAL_VERIFIED 41 validation clips;246 task metrics;36 videos;477 frames each;all checksums match')
else:
    assert all((root/f'RANK_{r}_COMPLETE').is_file() for r in range(3))
    request=json.loads((root/'REQUEST.json').read_text()); rows=[]; files={'REQUEST.json':sha(root/'REQUEST.json')}
    for path in sorted(root.glob('[0-9][0-9]_*/*.json')):
        row=json.loads(path.read_text()); assert row['step']==request['step'] and row['task'] in tasks
        rows.append(row); files[str(path.relative_to(root))]=sha(path)
        if row['preview']:
            assert sha(root/row['video'])==row['video_sha256']
            files[row['video']]=row['video_sha256']; files[row['poster']]=sha(root/row['poster'])
    assert len(rows)==246 and len({(r['clip_id'],r['task']) for r in rows})==246
    assert {r['clip_id'] for r in rows}=={r['id'] for r in request['validation']}
    baseline_path=Path('/pfs/mayuema/dense/wan/outputs/wan22_dense6_long477_self_forcing_dispersion_4800_20261017/evaluation.jsonl')
    baseline={tasks[r['task']]:r['latent_mse'] for r in map(json.loads,baseline_path.read_text().splitlines()) if r['step']==0}
    summary={}
    for task in tasks:
        selected=[r for r in rows if r['task']==task]; assert len(selected)==41
        preview=[r for r in selected if r['preview']]; assert len(preview)==6
        mean=lambda key,items:sum(r[key] for r in items)/len(items)
        mse=mean('latent_mse',selected)
        summary[task]={'n':41,'latent_mse':mse,'initial_latent_mse':baseline[task],
          'reduction_percent':100*(1-mse/baseline[task]),'mse_first_quarter':mean('mse_first_quarter',selected),
          'mse_last_quarter':mean('mse_last_quarter',selected),
          'decoded_preview_n':6,'preview_decoded_mae_vs_vae_target':mean('decoded_mae_vs_vae_target',preview)}
        for key in ('raw_flow_epe_px','raw_moving_flow_epe_px','normal_angle_deg_vs_vae_target'):
            if key in preview[0]: summary[task]['preview_'+key]=mean(key,preview)
    payload={'checkpoint':request['checkpoint'],'checkpoint_sha256':request['checkpoint_sha256'],
      'step':request['step'],'validation_clips':41,'preview_clips':6,'videos':36,'metrics':summary,
      'files_sha256':files,'results':rows,'comparison_reference':request['comparison_reference']}
    (root/'MANIFEST.json').write_text(json.dumps(payload,indent=2))
    (root/'summary.json').write_text(json.dumps({k:v for k,v in payload.items() if k not in ('files_sha256','results')},indent=2))
    lines=[f'# Dense6 长视频验证：step-{request["step"]}','',
      '41 条留出验证长视频，六个任务均为完整 477 帧、chunk3 因果 free-running 推理。只反馈上一块预测，不喂 GT history。',
      '以下六条可视化按固定规则选择，优先覆盖不同主体，不按误差挑样例。共36条视频，保留原始帧率。',
      '视频左：原 RGB；中：缓存 GT 经同一冻结 VAE 重建；右：模型预测经 VAE 解码。中栏不是原始渲染 GT。',
      'Flow 保持 scale32、延迟一帧、原 source-grid 定义；颜色亮度固定在20px饱和，无效像素黑色。',
      '下表是全部41条的 latent MSE；decoded 指标和 raw-flow EPE 仅覆盖6条可视化样例，见 summary.json。','',
      '| 模态 | 初始 MSE | 当前 MSE | 下降 |','|---|---:|---:|---:|']
    for task in tasks:
        s=summary[task]; lines.append(f'| {task} | {s["initial_latent_mse"]:.5f} | {s["latent_mse"]:.5f} | {s["reduction_percent"]:.1f}% |')
    for cid in request['preview_ids']:
        rowmeta=next(r for r in request['validation'] if r['id']==cid)
        lines+=['',f'## {rowmeta["subject_label"]} — {cid}','']
        for task in tasks:
            row=next(r for r in rows if r['clip_id']==cid and r['task']==task)
            lines.append(f'- [{task}]({row["video"]}) · latent MSE {row["latent_mse"]:.5f}')
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    (root/'EVAL_COMPLETE').write_text('success\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('files_sha256','results')},indent=2))
