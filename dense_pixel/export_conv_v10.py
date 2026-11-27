"""Same held-out examples and display scale as v9; export after fixed training."""
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from train_full_v9 import RUN as DATA,load_case,rows


@torch.no_grad()
def export_examples(task,model,export,out,output_title='MLP + convolution v10'):
    from train_conv_v10 import prepare,infer_video,quality
    dest=out/'visual_review';dest.mkdir(exist_ok=True)
    manifest=json.loads((DATA/'visual_review/MANIFEST.json').read_text())
    lookup={r['sample']:r for r in rows('test')}
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fontpath),21) if fontpath.exists() else ImageFont.load_default()
    records=[];model.eval()
    for example in manifest['examples']:
        row=lookup[example['sample']];label=example['label'];c=load_case(row,task);b=prepare(c,export)
        pred=infer_video(model,b['base'],b['features'])
        arrays=[((x.float().clamp(-1,1)+1)*127.5).round().byte().cpu().numpy()
                for x in [c['targets'],c['baseline'],b['base'],pred]]
        path=dest/f'{label}_{task}.mp4';fps=example['fps']
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
             '-f','rawvideo','-pix_fmt','rgb24','-s','4160x528','-r',str(fps),'-i','pipe:0',
             '-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-threads','4',
             '-movflags','+faststart',str(path)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for fi in range(81):
                canvas=Image.new('RGB',(4160,528),(20,23,28));draw=ImageDraw.Draw(canvas)
                canvas.paste(Image.open(Path(row['path'])/'rgb'/f'{fi+1:06d}.png').convert('RGB'),(0,48))
                titles=[f'RGB | {task} | {fi+1}/81','Ground truth','Original VAE','MLP v9 | 100k steps',output_title]
                for col,arr in enumerate(arrays,1):canvas.paste(Image.fromarray(arr[fi]).convert('RGB'),(832*col,48))
                for col,title in enumerate(titles):draw.text((832*col+12,12),title,font=font,fill='white')
                proc.stdin.write(np.asarray(canvas).tobytes())
                if fi==40:canvas.save(dest/f'{label}_{task}_frame41.png')
            proc.stdin.close();assert proc.wait()==0
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
        reader=imageio_ffmpeg.read_frames(str(path));next(reader);count=sum(1 for _ in reader);assert count==81
        records.append(dict(sample=row['sample'],selection=example['selection'],video=path.name,frames=count,
                            metrics=quality(pred,b['target'],task)))
        del b,c,pred,arrays
    (dest/'MANIFEST.json').write_text(json.dumps(records,indent=2)+'\n')
