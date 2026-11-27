"""Export fixed-checkpoint unseen-scene examples with disclosed selection rules."""
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from train_full_v9 import RUN, OUT, rows, load_case, folded_infer
import run_probe as v1

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
DEST = RUN / 'visual_review'
DEST.mkdir(exist_ok=True)
cases = rows('test')
reports = {t: json.loads((OUT / f'{t}_test_full_100000.json').read_text()) for t in ['depth', 'matte']}
metrics = {t: {x['sample']: x for x in d['cases']} for t, d in reports.items()}
chosen = []
for i, scene in enumerate(sorted({r['scene'] for r in cases}), 1):
    group = [r for r in cases if r['scene'] == scene]
    ranks = {}
    for task in ['depth', 'matte']:
        ordered = sorted(group, key=lambda r: (metrics[task][r['sample']]['clamped']['mae'], r['sample']))
        ranks[task] = {r['sample']: n / max(1, len(ordered)-1) for n, r in enumerate(ordered)}
    row = min(group, key=lambda r: (sum(abs(ranks[t][r['sample']]-.5) for t in ranks), r['sample']))
    chosen.append((f'scene{i}_typical', row, 'Closest to median depth and matte MAE ranks within this unseen scene'))
ordered = sorted(cases, key=lambda r: (metrics['matte'][r['sample']]['clamped']['mae'], r['sample']))
chosen.append(('matte_hard', ordered[round(.9*(len(ordered)-1))], '90th percentile matte MAE in the test set (higher error)'))
font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
font = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default()
exports = {t: torch.load(OUT / f'{t}_head_100000.pt', map_location='cpu', weights_only=False)['heads'] for t in ['depth', 'matte']}
summary = []
for label, row, rule in chosen:
    record = dict(label=label, sample=row['sample'], selection=rule, tasks={})
    meta = json.loads((Path(row['path'])/'metadata.json').read_text())
    fps = float(meta['video'].get('fps', 16))
    record['fps'] = fps
    for task in ['depth', 'matte']:
        case = load_case(row, task)
        with torch.no_grad():
            prediction = folded_infer(exports[task], case['features']).cpu()
        measured = v1.metrics(prediction, case['targets'], task)
        expected = metrics[task][row['sample']]['clamped']['mae']
        assert abs(measured['mae'] - expected) < 1e-5, (measured['mae'], expected)
        tensors = [case['targets'], case['baseline'], prediction]
        assert all(x is not None and x.shape == (81,480,832) for x in tensors)
        arrays = [((x.float().clamp(-1,1)+1)*127.5).round().byte().numpy() for x in tensors]
        path = DEST/f'{label}_{task}.mp4'
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner','-loglevel','error','-y',
               '-f','rawvideo','-pix_fmt','rgb24','-s','3328x528','-r',str(fps),'-i','pipe:0',
               '-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p',
               '-threads','4','-movflags','+faststart',str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for fi in range(81):
                canvas = Image.new('RGB',(3328,528),(20,23,28))
                draw = ImageDraw.Draw(canvas)
                titles = [f'RGB | {label} | {task} | {fi+1}/81', 'Ground truth', 'Original VAE', 'Direct pixel head | 100k steps']
                raw = Image.open(Path(row['path'])/'rgb'/f'{fi+1:06d}.png').convert('RGB')
                canvas.paste(raw,(0,48))
                for col, arr in enumerate(arrays,1):
                    canvas.paste(Image.fromarray(arr[fi]).convert('RGB'),(832*col,48))
                for col, title in enumerate(titles):
                    draw.text((832*col+12,12),title,font=font,fill='white')
                proc.stdin.write(np.asarray(canvas).tobytes())
                if fi == 40:
                    canvas.save(DEST/f'{label}_{task}_frame41.png')
                    # Native-resolution 2x2 still for easier inspection in narrow chat panes.
                    grid = Image.new('RGB',(1664,1056))
                    for col in range(4):
                        grid.paste(canvas.crop((832*col,0,832*(col+1),528)),((col%2)*832,(col//2)*528))
                    grid.save(DEST/f'{label}_{task}_detail.png')
            proc.stdin.close()
            assert proc.wait() == 0
        finally:
            if proc.poll() is None: proc.kill(); proc.wait()
        reader = imageio_ffmpeg.read_frames(str(path)); metadata = next(reader)
        count = sum(1 for _ in reader)
        assert count == 81 and tuple(metadata['size']) == (3328,528)
        record['tasks'][task] = dict(video=path.name, frames=count, measured=measured,
                                    reported=metrics[task][row['sample']], bytes=path.stat().st_size)
        print(json.dumps(dict(event='exported',label=label,task=task,frames=count,mae=measured['mae'])),flush=True)
        del case, prediction, tensors, arrays
    summary.append(record)
(DEST/'MANIFEST.json').write_text(json.dumps(dict(checkpoint_step=100000,split='unseen-scene test',
    source='Saved native-rollout feature cache; reloaded exported heads; MAE checked against full evaluation',
    display='Fixed [-1,1] to [0,255], identical for GT/VAE/direct; no frame-wise normalization or smoothing',
    examples=summary),indent=2)+'\n')
print('EXPORT_COMPLETE',flush=True)
