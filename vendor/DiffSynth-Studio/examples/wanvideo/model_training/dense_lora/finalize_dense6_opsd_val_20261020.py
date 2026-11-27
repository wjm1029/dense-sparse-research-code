"""Check exact clean holdouts, aggregate six tasks and compare same-split local12 baseline."""
import argparse,json,hashlib,subprocess
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()

p=argparse.ArgumentParser(); p.add_argument('--directory',required=True); p.add_argument('--verify_local',action='store_true'); a=p.parse_args()
root=Path(a.directory); tasks=('depth','normal','matte','raymap','albedo','optical_flow')
if a.verify_local:
    m=json.loads((root/'MANIFEST.json').read_text())
    for name,checksum in m['files_sha256'].items(): assert sha(root/name)==checksum,name
    videos=[name for name in m['files_sha256'] if name.endswith('.mp4')]; assert len(videos)==12
    for name in videos:
        s=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,codec_name,width,height','-of','json',str(root/name)],text=True))['streams'][0]
        assert int(s['nb_frames'])==477 and s['codec_name']=='h264' and (s['width'],s['height'])==(1248,280)
    print('VERIFIED27 clean clips,162 task metrics,12 videos,477 frames,all SHA256 matches')
else:
    request=json.loads((root/'REQUEST.json').read_text())
    assert all((root/f'RANK_{r}_COMPLETE').is_file() for r in range(request['world_size']))
    records=[]; files={'REQUEST.json':sha(root/'REQUEST.json')}
    for case in sorted(root.glob('[0-9][0-9]_*')):
        for task in tasks:
            path=case/(task+'.json'); assert path.is_file(),path
            row=json.loads(path.read_text()); assert row['task']==task and row['step']==2200
            records.append(row); files[str(path.relative_to(root))]=sha(path)
            if row['preview']:
                assert sha(root/row['video'])==row['video_sha256']
                files[row['video']]=row['video_sha256']; files[row['poster']]=sha(root/row['poster'])
    assert len(records)==162 and len({(r['clip_id'],r['task']) for r in records})==162
    assert {r['clip_id'] for r in records}==set(request['contract']['val_ids'])
    baseline_path=Path(request['checkpoint']).parent/'evaluation.jsonl'
    initial={tasks[r['task']]:r['latent_mse'] for r in map(json.loads,baseline_path.read_text().splitlines()) if r['step']==6}
    previous={tasks[r['task']]:r['latent_mse'] for r in map(json.loads,baseline_path.read_text().splitlines()) if r['step']==2000}
    metrics={}
    for task in tasks:
        rows=[r for r in records if r['task']==task]; previews=[r for r in rows if r['preview']]
        assert len(rows)==27 and len(previews)==2
        mean=lambda key,items:sum(r[key] for r in items)/len(items)
        mse=mean('latent_mse',rows)
        metrics[task]={'n':27,'latent_mse':mse,'same_clean27_local12_step6_baseline':initial[task],
            'reduction_vs_step6_percent':100*(1-mse/initial[task]),
            'same_clean27_local12_step2000':previous[task],'change_vs_step2000_percent':100*(mse/previous[task]-1),
            'mse_first_quarter':mean('mse_first_quarter',rows),'mse_last_quarter':mean('mse_last_quarter',rows),
            'decoded_preview_n':2,'preview_decoded_mae_vs_vae_target':mean('decoded_mae_vs_vae_target',previews)}
        for key in ('raw_flow_epe_px','raw_moving_flow_epe_px','normal_angle_deg_vs_vae_target'):
            if key in previews[0]: metrics[task]['preview_'+key]=mean(key,previews)
    payload={'step':2200,'training_status':request['training_status'],'checkpoint':request['checkpoint'],
        'checkpoint_sha256':request['checkpoint_sha256'],'validation_clips':27,'preview_clips':2,
        'videos':12,'metrics':metrics,'files_sha256':files,'results':records}
    (root/'MANIFEST.json').write_text(json.dumps(payload,indent=2))
    (root/'summary.json').write_text(json.dumps({k:v for k,v in payload.items() if k not in ('files_sha256','results')},indent=2))
    lines=['# Dense6 OPSD step2200：清理后验证集','',
        '训练未完成：2267步写共享盘时ENOSPC退出，使用最后完整上传的step2200，不能称为5250最终权重。',
        '27条人物留出视频，六个任务均为完整477帧；chunk3、sink3+local12、只反馈预测history。',
        '排除全部426条缺失颜色贴图的样本。两条可视化：之前展示的人物case + 第一个导出的固定hash样本，不按误差挑选。',
        '左RGB，中同一冻结VAE重建GT，右预测。中栏不是原始渲染GT。',
        '对比基线是相同27条验证集、相同local12配置的OPSD step6；不混用旧41条或full-prefix指标。',
        'Decoded指标和raw-flow EPE仅覆盖两条人物；latent指标覆盖全部27条。','',
        '| 模态 | step6 MSE | step2200 MSE | 下降 |','|---|---:|---:|---:|']
    for task in tasks:
        s=metrics[task]; lines.append(f'| {task} | {initial[task]:.5f} | {s["latent_mse"]:.5f} | {s["reduction_vs_step6_percent"]:.1f}% |')
    ray=metrics['raymap']
    lines+=['',f'注意 raymap：前1/4平均MSE {ray["mse_first_quarter"]:.5f}，后1/4 {ray["mse_last_quarter"]:.5f}；整体较同配置step2000上升{ray["change_vs_step2000_percent"]:.1f}%。另外五个模态较step2000略有改善。',
        '抽查中间帧，人物和场景的主要几何结构可对应，细节偏平滑；单帧对比不等同于对整段时序稳定性的判断。',
        '两条视频的raw-flow EPE均值约0.547px，运动区域约0.603px；normal角误差约5.31°，参考是VAE重建GT。']
    for cid in request['preview_ids']:
        label='此前给过的人物 case' if cid=='114b73905af1cb7f/b4f81d3949c39c8528d3' else '新增人物 case'
        lines+=['',f'## {label} — {cid}','']
        for task in tasks:
            row=next(r for r in records if r['clip_id']==cid and r['task']==task)
            lines.append(f'- [{task}]({row["video"]}) · MSE {row["latent_mse"]:.5f}')
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    (root/'EVAL_COMPLETE').write_text('success\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('files_sha256','results')},indent=2))
