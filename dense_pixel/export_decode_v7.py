"""Export verified 81-frame reconstructions for visual inspection."""
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio_ffmpeg
import run_probe as v1
from decode_nonlinear_v7 import RUN

torch.set_num_threads(4)
assert json.loads((RUN/'LIVE_VERIFIED.json').read_text())['all_passed']
row=v1.rows('overfit8')[0];data=v1.load_sample(str(v1.cache_path(row)))
for task in v1.TASKS:
    pred=torch.load(RUN/f'{task}_live_prediction.pt',map_location='cpu',weights_only=True)
    path=RUN/f'{task}_comparison.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
         '-f','rawvideo','-pix_fmt','rgb24','-s','1664x272','-r','16','-i','pipe:0',
         '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2',
         '-movflags','+faststart',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for fi in range(81):
            canvas=Image.new('RGB',(1664,272));draw=ImageDraw.Draw(canvas)
            raw=Image.open(Path(row['path'])/'rgb'/f'{fi+1:06d}.png').convert('RGB').resize((416,240))
            canvas.paste(raw,(0,32));draw.text((4,8),f'RGB / {task} / frame {fi+1}',fill='white')
            for ci,(title,tensor) in enumerate([('Ground truth',data['targets'][task]),('Original VAE',data['baseline'][task]),('Frozen backbone + pixel head',pred)],1):
                arr=((tensor[fi].float().clamp(-1,1)+1)*127.5).round().byte().numpy()
                canvas.paste(Image.fromarray(arr).convert('RGB').resize((416,240)),(ci*416,32))
                draw.text((ci*416+4,8),title,fill='white')
            proc.stdin.write(np.asarray(canvas).tobytes())
        proc.stdin.close();assert proc.wait()==0
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
    reader=imageio_ffmpeg.read_frames(str(path));metadata=next(reader);count=sum(1 for _ in reader)
    assert count==81,(path,count)
    print(json.dumps(dict(path=str(path),frames=count,size=metadata['size'])),flush=True)
