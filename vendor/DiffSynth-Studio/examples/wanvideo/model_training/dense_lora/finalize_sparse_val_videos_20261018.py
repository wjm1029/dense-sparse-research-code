"""Build a portable manifest, or verify the transferred validation artifacts."""
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

p=argparse.ArgumentParser(); p.add_argument('--directory',required=True); p.add_argument('--verify_local',action='store_true')
a=p.parse_args(); root=Path(a.directory)
if not a.verify_local:
    assert all((root/f'RANK_{rank}_COMPLETE').exists() for rank in range(4))
    cases=[json.loads(path.read_text()) for path in sorted(root.glob('[0-9][0-9]_*.json'))]
    assert len(cases)==27 and {r['validation_index'] for r in cases}==set(range(27))
    assert len({r['checkpoint_sha256'] for r in cases})==1
    assert all(r['frames']==477 and r['student_not_ema'] for r in cases)
    files={}
    for row in cases:
        assert sha(root/row['video'])==row['video_sha256']
        for name in (row['video'],row['poster'],row['coordinates']): files[name]=sha(root/name)
        record=Path(row['video']).with_suffix('.json').name; files[record]=sha(root/record)
    e2=sum(r['error_2d_pixels'] for r in cases)/27; e3=sum(r['error_3d_actor_height'] for r in cases)/27
    assert abs(e2-3.508029990726047)<.001 and abs(e3-.015722376777342072)<.00001
    manifest={'checkpoint':cases[0]['checkpoint'],'checkpoint_sha256':cases[0]['checkpoint_sha256'],
      'num_clips':27,'frames_per_clip':477,'mean_error_2d_pixels':e2,'mean_error_3d_actor_height':e3,
      'cases':cases,'files_sha256':files}
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    lines=['# Sparse long477 validation videos — OPSD2625','',
      'All27 held-out clips, all477 frames at original frame rate. Student weights, not EMA.',
      'Two separate prompts: output: 2D keypoints / output: 3D keypoints.',
      'Prediction uses RGB latents + previous predicted history only. GT is for drawing/metrics only.','',
      'Top: RGB+GT2D | RGB+predicted2D | RGB+overlay. Bottom: 3D front(X/Y) and side(Z/Y).',
      'Green=GT, magenta=prediction. 3D is root-relative camera coordinates / actor height, not meters.',
      'Prediction2D visibility is inferred from prediction bounds, not hidden by GT validity.',
      '3D plot axes share fixed per-clip GT+prediction bounds for display only.','',
      f'Mean validation error:2D {e2:.6f}px;3D {e3:.8f} actor height.','',
      '| Video | 2D px | 3D / height |','|---|---:|---:|']
    lines += [f"| [{r['video']}]({r['video']}) | {r['error_2d_pixels']:.4f} | {r['error_3d_actor_height']:.6f} |" for r in cases]
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:manifest[k] for k in ('num_clips','mean_error_2d_pixels','mean_error_3d_actor_height')},indent=2))
else:
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,checksum in manifest['files_sha256'].items(): assert sha(root/name)==checksum,name
    for row in manifest['cases']:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
          '-show_entries','stream=width,height,nb_frames,codec_name','-of','json',str(root/row['video'])],text=True))['streams'][0]
        assert info['codec_name']=='h264' and int(info['nb_frames'])==477
        assert (info['width'],info['height'])==(1248,640)
    print('LOCAL_VERIFIED 27 videos;477 frames each;all108 artifact checksums match')
