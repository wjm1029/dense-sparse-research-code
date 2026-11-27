"""Export full-video comparisons and fixed-final-checkpoint summary."""
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image,ImageDraw
import imageio_ffmpeg
import run_probe as v1
import run_probe_v2 as v2


def main():
    out=v2.RUN;assert (out/'COMPLETE.json').exists()
    torch.set_num_threads(4)
    ridge=nn.ModuleDict({t:v2.RidgeHead() for t in v1.TASKS}).cuda()
    ridge.load_state_dict(torch.load(out/'ridge_heads.pt',map_location='cpu',weights_only=True))
    refiners=nn.ModuleDict({t:v2.ResidualHead(ridge[t]) for t in v1.TASKS}).cuda()
    saved=torch.load(out/'refiner_latest.pt',map_location='cpu',weights_only=False)
    assert saved['step']==4000
    refiners.load_state_dict(saved['models']);ridge.eval();refiners.eval()
    evaluations=json.loads((out/'eval_refiner_holdout_4000.json').read_text())
    summary={'stage':'v2_final4000','holdout_clips':6,'target_range':[-1,1],'tasks':{},
        'limitations':['Frozen-backbone engineering pilot, not a paper reproduction.',
        'Only 18 train clips and 6 holdout clips; historical backbone overlap unknown.',
        'Timing is preliminary head compute only; no end-to-end speedup claim.']}
    for task in v1.TASKS:
        subset=[d for d in evaluations if d['task']==task];summary['tasks'][task]={}
        for method in ['baseline','ridge','residual']:
            metrics=[d[method] if method=='baseline' else d[method]['metrics'] for d in subset]
            keys=set.intersection(*[set(m) for m in metrics])
            summary['tasks'][task][method]={k:float(np.mean([m[k] for m in metrics])) for k in keys}
    row=v1.rows('eval_candidates')[0];data=v1.load_sample(str(v1.cache_path(row)))
    for task in v1.TASKS:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            features=data['features'][task].cuda()
            predictions={'ridge':v1.predict(ridge[task],features).float().cpu(),
                         'refined':v1.predict(refiners[task],features).float().cpu()}
        path=out/f'comparison_{task}.mp4'
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
          '-f','rawvideo','-pix_fmt','rgb24','-s','2080x272','-r','16','-i','pipe:0',
          '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','2',
          '-movflags','+faststart',str(path)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for i in range(81):
                canvas=Image.new('RGB',(2080,272));draw=ImageDraw.Draw(canvas)
                raw=Image.open(Path(row['path'])/'rgb'/f'{i+1:06d}.png').convert('RGB').resize((416,240))
                canvas.paste(raw,(0,32));draw.text((5,7),f'RGB | {task} | frame {i+1}/81',fill='white')
                values=[('Raw GT',data['targets'][task]),('Official VAE',data['baseline'][task]),*predictions.items()]
                for j,(label,value) in enumerate(values,1):
                    array=((value[i].float().clamp(-1,1).numpy()+1)*127.5).round().astype(np.uint8)
                    canvas.paste(Image.fromarray(array).convert('RGB').resize((416,240)),(j*416,32))
                    draw.text((j*416+5,7),label,fill='white')
                process.stdin.write(np.asarray(canvas).tobytes())
                if i==40:canvas.save(out/f'comparison_{task}.jpg',quality=95)
            process.stdin.close();assert process.wait()==0
        finally:
            if process.poll() is None:process.kill();process.wait()
        summary.setdefault('videos',{})[task]=str(path)
    v1.atomic_json(out/'RESULT_SUMMARY.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
