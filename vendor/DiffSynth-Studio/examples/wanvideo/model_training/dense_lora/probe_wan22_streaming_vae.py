"""Isolated feasibility probe; never changes production VAE or latent caches."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import imageio.v2 as imageio
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--input_video',required=True)
p.add_argument('--output_dir',required=True)
p.add_argument('--frames',type=int,default=81)
a=p.parse_args()
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=False)
src=Path('/pfs/mayuema/dense/wan/Wan2.2/wan/modules/vae2_2.py')
spec=importlib.util.spec_from_file_location('official_probe_vae',src)
vmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(vmod)
device=torch.device('cuda:0'); torch.cuda.set_device(device)
vae=vmod.Wan2_2_VAE(vae_pth='/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B/Wan2.2_VAE.pth',dtype=torch.bfloat16,device=str(device))
reader=imageio.get_reader(a.input_video)
try:
    frames=np.stack([np.asarray(f)[...,:3] for f in reader])[:a.frames]
finally: reader.close()
assert frames.shape==(a.frames,480,832,3) and (a.frames-1)%4==0
x=torch.from_numpy(frames.copy()).to(device).permute(3,0,1,2).float().div(127.5).sub(1).to(torch.bfloat16)[None]

class WindowEncoder:
    def __init__(self): self.reset()
    def reset(self):
        self.cache=[None]*vmod.count_conv3d(vae.model.encoder)
        self.outputs=[]; self.n=0
    def push(self,chunk):
        expected=1 if not self.outputs else 4
        if chunk.shape[2]!=expected: raise ValueError('Require first 1 frame then exact 4-frame groups')
        z=vae.model.encoder(vmod.patchify(chunk,patch_size=2),feat_cache=self.cache,feat_idx=[0])
        self.outputs.append(z); self.n+=chunk.shape[2]
    def finish(self):
        if not self.outputs: raise ValueError('Empty window')
        mu,_=vae.model.conv1(torch.cat(self.outputs,2)).chunk(2,dim=1)
        result=(mu-vae.scale[0].view(1,48,1,1,1))*vae.scale[1].view(1,48,1,1,1)
        self.reset(); return result.float()

def measured(fn):
    torch.cuda.synchronize(); start=time.perf_counter()
    z=fn(); torch.cuda.synchronize()
    return z,(time.perf_counter()-start)*1000
def full(value): return vae.model.encode(value,vae.scale).float()
def stream(value):
    session=WindowEncoder(); timings=[]
    for i in range(1+(value.shape[2]-1)//4):
        c=value[:,:,:1] if i==0 else value[:,:,1+4*(i-1):1+4*i]
        _,ms=measured(lambda:session.push(c)); timings.append(ms)
    result,finish_ms=measured(session.finish)
    return result,timings,finish_ms
rows={}
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    full(x) # warmup
    ref,full_ms=measured(lambda:full(x))
    z,groups,finish_ms=stream(x)
    err=(z-ref).abs()
    rows.update(full_encode_ms=full_ms,stream_group_ms=groups,stream_finish_ms=finish_ms,
        stream_total_ms=sum(groups)+finish_ms,latent_max_abs_error=float(err.max()),latent_mean_abs_error=float(err.mean()))
    torch.testing.assert_close(z,ref,rtol=0,atol=1e-5)
    # A second window must not inherit the preceding window's padding/state.
    second=x.flip(2).contiguous()
    second_ref,_=measured(lambda:full(second))
    second_z,_,_=stream(second)
    rows['reset_second_window_max_abs_error']=float((second_z-second_ref).abs().max())
    torch.testing.assert_close(second_z,second_ref,rtol=0,atol=1e-5)
    # Changing only the final 4 raw frames must not change earlier latent groups.
    changed=x.clone(); changed[:,:,-4:]*=-1
    changed_z,_,_=stream(changed)
    rows['future_perturb_prefix_max_abs_error']=float((z[:,:,:-1]-changed_z[:,:,:-1]).abs().max())
    torch.testing.assert_close(z[:,:,:-1],changed_z[:,:,:-1],rtol=0,atol=1e-5)
    session=WindowEncoder()
    try: session.push(x[:,:,:2])
    except ValueError: rows['invalid_first_group_rejected']=True
    else: raise AssertionError('Invalid group accepted')
    # Full-window decoded pixels must also match when the latent window matches.
    pixels,_=measured(lambda:vae.model.decode(ref,vae.scale).float().clamp(-1,1))
    stream_pixels,_=measured(lambda:vae.model.decode(z,vae.scale).float().clamp(-1,1))
    rows['decoded_max_abs_error']=float((pixels-stream_pixels).abs().max())
    torch.testing.assert_close(pixels,stream_pixels,rtol=0,atol=1e-5)
rows.update(input_video=a.input_video,frames=a.frames,shape=list(ref.shape),dtype='BF16',
    protocol='one warmup, one timed full encode, synchronized per-group streaming; real-arrival waits excluded',
    scope='window-scoped input encoder prototype only; no DiT, no live camera, no production changes',
    note='Decoded equality uses full decode of equal latents; streaming output decoder is NOT tested',
    peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,passed=True)
(out/'results.json').write_text(json.dumps(rows,indent=2)); (out/'PROBE_COMPLETE').touch()
print(json.dumps(rows,indent=2),flush=True)
