"""Isolated window-scoped streaming VAE measurements, not production inference."""
import argparse
import importlib.util
import json
import time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--output_dir', required=True)
p.add_argument('--long_video', required=True)
a = p.parse_args()
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('probe_vae', '/pfs/mayuema/dense/wan/Wan2.2/wan/modules/vae2_2.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
torch.cuda.set_device(0)
vae = v.Wan2_2_VAE(vae_pth='/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B/Wan2.2_VAE.pth', dtype=torch.bfloat16, device='cuda:0')

def timed(fn):
    torch.cuda.synchronize(); start = time.perf_counter()
    value = fn(); torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000

class Encoder:
    def __init__(self): self.reset()
    def reset(self):
        self.cache = [None] * v.count_conv3d(vae.model.encoder)
        self.features = []
    def push(self, x):
        if x.shape[2] != (1 if not self.features else 4):
            raise ValueError('Expected first 1 frame, then 4 frames')
        self.features.append(vae.model.encoder(v.patchify(x, patch_size=2), feat_cache=self.cache, feat_idx=[0]))
    def finish(self):
        if not self.features: raise ValueError('Empty window')
        mu = vae.model.conv1(torch.cat(self.features, 2)).chunk(2, 1)[0]
        z = (mu - vae.scale[0].view(1,48,1,1,1)) * vae.scale[1].view(1,48,1,1,1)
        self.reset(); return z.float()

class Decoder:
    def __init__(self): self.reset()
    def reset(self):
        self.cache = [None] * v.count_conv3d(vae.model.decoder)
        self.first = True
    def push(self, z):
        if z.shape[2] != 1: raise ValueError('One latent frame per decoder push')
        x = z / vae.scale[1].view(1,48,1,1,1) + vae.scale[0].view(1,48,1,1,1)
        x = vae.model.conv2(x)
        x = vae.model.decoder(x, feat_cache=self.cache, feat_idx=[0], first_chunk=self.first)
        self.first = False
        return v.unpatchify(x, patch_size=2).float()

def groups(x):
    yield x[:,:,:1]
    for t in range(1, x.shape[2], 4): yield x[:,:,t:t+4]

def stream(x, session=None):
    session = session or Encoder(); times = []
    for chunk in groups(x):
        _, ms = timed(lambda: session.push(chunk)); times.append(ms)
    z, finish = timed(session.finish)
    return z, times, finish

def arrival_tail(times, finish, fps):
    end = 0.0
    for i, duration in enumerate(times):
        arrival = 4*i*1000/fps
        end = max(end, arrival) + duration
    return end + finish - (len(times)-1)*4*1000/fps

base = Path('/pfs/mayuema/dense/wan/inputs/selected_real_videos_20260928/preprocessed')
cases = [(base/name,81) for name in ['batch1_01_人物_秘鲁妇女纺线.mp4', 'batch1_03_人物_陶艺师制作陶器.mp4', 'batch1_07_动物_河边的象群.mp4', 'batch1_09_动物_正在玩耍的猫.mp4', 'batch1_10_物品_咖啡机萃取咖啡.mp4']]
cases.append((Path(a.long_video),297))
report = {'protocol':'BF16, 480x832, one warmup per case, three timed encoding repeats; synchronized per group; no DiT/live camera', 'arrival_model':'Measured GPU durations scheduled at 16/30 fps; excludes IO/queueing and competition with DiT', 'cases':[]}
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    for path, count in cases:
        reader = imageio.get_reader(str(path))
        frames = []
        try:
            fps = reader.get_meta_data().get('fps')
            for frame in reader:
                frames.append(frame[:,:,:3])
                if len(frames) == count: break
        finally: reader.close()
        array = np.stack(frames)
        assert array.shape == (count,480,832,3)
        x = torch.from_numpy(array).cuda().permute(3,0,1,2).float().div(127.5).sub(1).to(torch.bfloat16)[None]
        del array, frames
        torch.cuda.reset_peak_memory_stats()
        vae.model.encode(x,vae.scale)
        row = {'input':str(path),'frames':count,'source_fps':fps,'full_encode_ms':[],'stream_encode_ms':[],'post_last_frame_ms_16fps':[],'post_last_frame_ms_30fps':[]}
        for repeat in range(3):
            ref, ms = timed(lambda: vae.model.encode(x,vae.scale).float())
            z, durations, finish = stream(x)
            row['full_encode_ms'].append(ms)
            row['stream_encode_ms'].append(sum(durations)+finish)
            row['post_last_frame_ms_16fps'].append(arrival_tail(durations,finish,16))
            row['post_last_frame_ms_30fps'].append(arrival_tail(durations,finish,30))
            row['latent_max_error'] = float((ref-z).abs().max())
            torch.testing.assert_close(ref,z,rtol=0,atol=1e-5)
        pixels, row['full_decode_ms'] = timed(lambda: vae.model.decode(ref,vae.scale).float())
        decoder = Decoder(); maximum = 0.; decoded_times = []; offset = 0
        for i in range(z.shape[2]):
            chunk, ms = timed(lambda: decoder.push(z[:,:,i:i+1]))
            decoded_times.append(ms)
            maximum = max(maximum,float((chunk-pixels[:,:,offset:offset+chunk.shape[2]]).abs().max()))
            offset += chunk.shape[2]
        assert offset == count
        row.update(stream_decode_ms=sum(decoded_times),first_decoded_frame_ms=decoded_times[0],decode_group_ms=decoded_times,decoded_max_error=maximum)
        assert maximum <= 1e-5, maximum
        del pixels, chunk
        # Independent states must survive interleaved streams and a reset.
        if count == 81:
            e1,e2 = Encoder(),Encoder()
            flipped = x.flip(2).contiguous()
            for c1,c2 in zip(groups(x),groups(flipped)): e1.push(c1); e2.push(c2)
            zz1,zz2 = e1.finish(),e2.finish()
            r2 = vae.model.encode(flipped,vae.scale).float()
            row['interleaved_encoder_max_error'] = max(float((zz1-ref).abs().max()),float((zz2-r2).abs().max()))
            d1,d2 = Decoder(),Decoder(); dref = Decoder(); maxerr = 0.
            for i in range(z.shape[2]):
                y1 = d1.push(z[:,:,i:i+1]); d2.push(r2[:,:,i:i+1])
                expected = dref.push(z[:,:,i:i+1])
                maxerr = max(maxerr,float((y1-expected).abs().max()))
            row['interleaved_decoder_max_error'] = maxerr
            e1.push(x[:,:,:1]); e1.reset()
            reset_z,_,_ = stream(x,e1)
            row['abort_reset_max_error'] = float((reset_z-ref).abs().max())
            changed = x.clone(); changed[:,:,-4:] *= -1
            changed_z,_,_ = stream(changed)
            row['future_change_prefix_max_error'] = float((changed_z[:,:,:-1]-z[:,:,:-1]).abs().max())
            del flipped, zz1, zz2, r2, y1, expected, reset_z, changed, changed_z, e1,e2,d1,d2,dref
        row['peak_gpu_gib'] = torch.cuda.max_memory_allocated()/2**30
        report['cases'].append(row)
        (out/'results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(row,ensure_ascii=False),flush=True)
        del x,z,ref,decoder
        torch.cuda.empty_cache()
report['complete'] = True
(out/'results.json').write_text(json.dumps(report,indent=2))
(out/'PROBE_COMPLETE').touch()
