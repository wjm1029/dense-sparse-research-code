"""Render 2D GT/prediction on RGB to inspect coordinate alignment and fit."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

p = argparse.ArgumentParser()
p.add_argument('--clip_id',required=True)
p.add_argument('--prediction',default=None)
p.add_argument('--output',required=True)
a = p.parse_args()
root = Path('/pfs/mayuema/dense/dataset/prod_final_9110_20260816/train')/a.clip_id
with np.load(root/'genception/keypoints_2d/keypoints_2d.npz',allow_pickle=False) as z:
    target = z['image_xy_01'].copy(); parents = z['parents'].copy(); mask = z['in_front'] & z['in_frame']
prediction = None
if a.prediction:
    with np.load(a.prediction,allow_pickle=False) as z:
        prediction = z['prediction_2d'][0].copy()
        np.testing.assert_allclose(target,z['target_2d'][0],atol=1e-6)
indices = [0,20,40,60,80]
sheet = Image.new('RGB',(416*5,260),(255,255,255))
for column,i in enumerate(indices):
    image = Image.open(root/'rgb'/f'{i+1:06d}.png').convert('RGB').resize((416,240))
    draw = ImageDraw.Draw(image)
    for values,color in [(target,(0,255,0)),(prediction,(255,0,255))]:
        if values is None: continue
        points = values[i]*np.array([416,240])
        for j,parent in enumerate(parents):
            if parent >= 0 and mask[i,j] and mask[i,parent]: draw.line([tuple(points[j]),tuple(points[parent])],fill=color,width=2)
        for j,(x,y) in enumerate(points):
            if mask[i,j]: draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
    sheet.paste(image,(column*416,20))
    ImageDraw.Draw(sheet).text((column*416+4,2),f'frame {i+1}; GT green, prediction magenta',fill=(0,0,0))
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
sheet.save(a.output)
print(a.output)
