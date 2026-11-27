"""Render raw GT/VAE/final detailer predictions with a shared fixed Spectral depth LUT."""
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import torch
from torch.nn import functional as F
import run_probe as v1
from train_full_v9 import rows, load_case as load_depth
from pixvid_detailer_v12 import Detailer as DepthHead, infer_video as infer_depth
from modalities_v13 import Detailer as ModalHead, load_case as load_modal, infer_video as infer_modal

EXP=v1.EXP
OUT=EXP/'final_visual_v15_20261124'


@torch.no_grad()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    color=json.loads((EXP/'spectral_lut_v15.json').read_text())
    lut=np.asarray(color['rgb_uint8'],dtype=np.uint8)
    examples=json.loads((EXP/'full_data_v9/visual_review/MANIFEST.json').read_text())['examples']
    lookup={r['sample']:r for r in rows('test')}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    manifest=dict(frames=81,playback_fps=16,layout='top left RGB; top right GT; bottom left original VAE; bottom right final detailer',
                  depth_colormap={k:v for k,v in color.items() if k!='rgb_uint8'},
                  depth_units='Normalized dataset depth, not meters. Fixed mapping across all frames and methods; no per-image stretching.',
                  normal_visualization='Clamp signed xyz to[-1,1], unit-normalize all GT/VAE/pred consistently, then map to RGB.',
                  examples=[])
    for task in ['depth','matte','normal']:
        path=EXP/('pixvid_detailer_v12/depth/head_30000.pt' if task=='depth' else f'pixvid_modalities_v13/{task}/head_30000.pt')
        model=(DepthHead() if task=='depth' else ModalHead(task)).cuda().eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        for e in examples:
            row=lookup[e['sample']]
            case=load_depth(row,task) if task=='depth' else load_modal(row,task)
            pred=(infer_depth if task=='depth' else infer_modal)(model,case['features']).cpu()
            arrays=[]
            for x in [case['targets'],case['baseline'],pred]:
                x=x.float().clamp(-1,1)
                if task=='depth':
                    idx=((x+1)*.5*(len(lut)-1)).round().long().numpy()
                    arr=lut[idx]
                else:
                    if task=='normal':x=F.normalize(x,dim=1)
                    if x.shape[1]==1:x=x.expand(-1,3,-1,-1)
                    arr=((x+1)*127.5).round().byte().permute(0,2,3,1).numpy()
                arrays.append(arr)
            suffix='depth_spectral' if task=='depth' else task
            stem=f"{e['label']}_{suffix}"
            video=OUT/f'{stem}.mp4'
            cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
                 '-f','rawvideo','-pix_fmt','rgb24','-s','1664x1104','-r','16','-i','pipe:0',
                 '-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p',
                 '-threads','4','-movflags','+faststart',str(video)]
            p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
            try:
                for i in range(81):
                    canvas=Image.new('RGB',(1664,1104),(20,23,28));draw=ImageDraw.Draw(canvas)
                    with Image.open(Path(row['path'])/'rgb'/f'{i+1:06d}.png') as rgb:canvas.paste(rgb.convert('RGB'),(0,44))
                    for a,xy in zip(arrays,[(832,44),(0,568),(832,568)]):canvas.paste(Image.fromarray(a[i]),xy)
                    titles=['RGB input',f'Ground truth | {task}','Original VAE',f'Final detailer | {task} | step 30000']
                    for title,(x,y) in zip(titles,[(0,0),(832,0),(0,524),(832,524)]):draw.text((x+14,y+9),title,font=font,fill='white')
                    draw.text((14,1065),f'{e["label"]} | frame {i+1:02d}/81 | playback 16 fps',font=small,fill='white')
                    if task=='depth':
                        draw.text((740,1065),'Spectral | fixed normalized depth',font=small,fill='white')
                        bar=lut[np.rint(np.linspace(0,len(lut)-1,480)).astype(int)][None].repeat(16,axis=0)
                        canvas.paste(Image.fromarray(bar),(1130,1060))
                        draw.text((1130,1079),'0',font=small,fill='white');draw.text((1598,1079),'1',font=small,fill='white')
                    p.stdin.write(np.asarray(canvas).tobytes())
                    if i==40:canvas.save(OUT/f'{stem}_frame41.png')
                p.stdin.close();assert p.wait()==0
            finally:
                if p.poll() is None:p.kill();p.wait()
            item=dict(task=task,label=e['label'],sample=e['sample'],head=str(path),video=video.name,
                      still=f'{stem}_frame41.png',bytes=video.stat().st_size)
            manifest['examples'].append(item)
            (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
            print(json.dumps(dict(event='exported',**item)),flush=True)
            del case,pred,arrays
        del model
        torch.cuda.empty_cache()
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(time=time.time(),videos=len(manifest['examples']))))


if __name__=='__main__':main()
