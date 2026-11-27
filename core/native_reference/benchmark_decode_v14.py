"""Paired GPU-resident decoder and RGB-to-pixels benchmarks; no file IO timed."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch
import run_probe as v1
import run_probe_v3b as native
from pixvid_detailer_v12 import Detailer as DepthHead, infer_video as depth_infer
from modalities_v13 import Detailer as ModalHead, infer_video as modal_infer

EXP = v1.EXP
OUT = EXP / os.environ.get('BENCH_RUN', 'speed_v14_20261124')


def save(name, value):
    p = OUT / name
    t = p.with_suffix('.pending.json')
    t.write_text(json.dumps(value, indent=2))
    t.replace(p)


def emit(event, **kw):
    x = dict(event=event, time=time.time(), **kw)
    print(json.dumps(x), flush=True)
    save('STATUS.json', x)


def monitor_gpu():
    """Keep contemporaneous occupancy evidence outside timing-critical threads."""
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    while True:
        outputs = {}
        for name, query in [('gpu', '--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,power.draw'),
                            ('processes', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory')]:
            p = subprocess.run(['nvidia-smi', '-i', gpu, query, '--format=csv,noheader'],
                               capture_output=True, text=True)
            outputs[name] = p.stdout.strip()
        with (OUT/'GPU_OCCUPANCY.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(), **outputs))+'\n')
        time.sleep(5)


@torch.no_grad()
def rollout_gpu(module, z, task, pid, capture):
    """Same OPSD invocation as native.rollout; omit diagnostic CPU copies."""
    dit = module.pipe.dit
    features = []
    hook = dit.head.head.register_forward_pre_hook(
        lambda layer, args: features.append(args[0].detach())) if capture else None
    caches = module.causal_controller.new_kv_cache()
    history = torch.zeros_like(z[None, :, :3])
    preds = []
    task_id = torch.tensor([v1.TASKS[task]], device='cuda')
    prompt_id = torch.tensor([pid], device='cuda')
    dit.train()  # Required by original training_streaming_forward, even in no_grad.
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for offset in range(0, 21, 3):
                pred, _, _ = module(z[None, :, offset:offset+3], history,
                    task_id, prompt_id, offset, caches, False)
                preds.append(pred)
                history = pred.detach()
                module.task_router.clear()
    finally:
        if hook is not None:
            hook.remove()
        module.task_router.clear()
        dit.eval()
    return torch.cat(preds, 2)[0], (torch.cat(features, 1).reshape(21, 15, 26, 3072)
                                  if capture else None)


def measure(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    begin.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t
    item = dict(wall_ms=wall*1000, gpu_ms=begin.elapsed_time(end),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                incremental_peak_gib=(torch.cuda.max_memory_allocated()-base)/2**30)
    if isinstance(result, torch.Tensor):
        assert torch.isfinite(result).all(), 'nonfinite benchmark output'
        item['output_shape'] = list(result.shape)
    del result
    return item


def summary(items):
    wall = [x['wall_ms'] for x in items]
    return dict(n=len(items), median_ms=statistics.median(wall),
                min_ms=min(wall), max_ms=max(wall), p90_ms=float(np.percentile(wall, 90)),
                effective_fps=81000/statistics.median(wall),
                median_gpu_ms=statistics.median(x['gpu_ms'] for x in items),
                max_incremental_peak_gib=max(x['incremental_peak_gib'] for x in items))


def paired(functions, warmup, repeats, task, case, stage):
    for _ in range(warmup):
        for fn in functions.values():
            measure(fn)
    samples = {k: [] for k in functions}
    for i in range(repeats):
        names = list(functions)
        if i % 2:
            names.reverse()
        for name in names:
            item = measure(functions[name])
            samples[name].append(item)
            emit('timing', task=task, case=case, stage=stage, route=name, repeat=i, **item)
    return dict(samples=samples, summary={k: summary(v) for k, v in samples.items()})


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=monitor_gpu, daemon=True).start()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    v1.TASKS['normal'] = 1
    native.RUN = OUT
    config = dict(gpu=torch.cuda.get_device_name(), physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
        torch=torch.__version__, cuda=torch.version.cuda, frames=81, height=480, width=832,
        batch=1, decoder_warmup=2, decoder_repeats=7, pipeline_warmup=1, pipeline_repeats=3,
        precision='Native/VAE BF16; head BF16 autocast with FP32 weights and final convolution',
        timing='Synchronized wall time and CUDA events; alternating routes; input and output GPU-resident',
        pipeline='GPU RGB tensor -> official VAE encoder -> native OPSD rollout -> decoder',
        excluded='File IO, model loading, host/device transfer, video compression, rendering',
        detailer='Unmodified evaluation infer_video, 4 raw frames per decoder call',
        vae='Unmodified official Wan2_2_VAE.decode, 81-frame input, default internal temporal chunks',
        streaming_latency='Not measured; full-clip sequential pipeline, no overlap',
        checkpoint_sha256=native.SHA)
    save('CONFIG.json', config)
    emit('loading_models')
    module, vae, contract = native.load_native()
    rows = [json.loads(x) for x in (EXP/'full_data_v9/validation.jsonl').read_text().splitlines()]
    chosen = [rows[0], next(r for r in reversed(rows) if r['scene'] != rows[0]['scene'])]
    sources = []
    for row in chosen:
        def read_frame(i):
            with Image.open(Path(row['path'])/'rgb'/f'{i:06d}.png') as im:
                return np.asarray(im.convert('RGB'))
        with ThreadPoolExecutor(8) as pool:
            rgb = np.stack(list(pool.map(read_frame, range(1, 82))))
        source = torch.from_numpy(rgb.copy()).permute(3, 0, 1, 2).cuda().to(torch.bfloat16)/127.5-1
        sources.append(source)
    results = dict(config=config, cases=chosen, tasks={})
    prompts = dict(depth='output: depth', matte='output: person alpha matte', normal='output: surface normal')
    for task in os.environ.get('BENCH_TASKS', 'depth,matte,normal').split(','):
        path = (EXP/'pixvid_detailer_v12/depth/head_30000.pt' if task == 'depth'
                else EXP/f'pixvid_modalities_v13/{task}/head_30000.pt')
        head = (DepthHead() if task == 'depth' else ModalHead(task)).cuda().eval()
        saved = torch.load(path, map_location='cpu', weights_only=False)
        head.load_state_dict(saved['model'], strict=True)
        del saved
        infer = depth_infer if task == 'depth' else modal_infer
        pid = contract['prompts'].index(prompts[task])
        task_result = dict(head=str(path), head_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=[])
        results['tasks'][task] = task_result

        def decode_vae(latent):
            x = vae.decode([latent])[0].float()
            if task == 'normal':
                return x.permute(1, 0, 2, 3)
            x = x.mean(0)
            return x[:, None] if task == 'matte' else x

        def pipeline(source, new):
            z = vae.encode([source])[0]
            latent, features = rollout_gpu(module, z, task, pid, new)
            return infer(head, features) if new else decode_vae(latent)

        for case_id, source in enumerate(sources):
            emit('preparing_case', task=task, case=case_id)
            z = vae.encode([source])[0]
            latent, features = rollout_gpu(module, z, task, pid, True)
            if case_id == 0:
                reference = native.rollout(module, z, task, pid)
                checks = dict(feature_max_abs=float((features-reference['features'].cuda()).abs().max()),
                    latent_max_abs=float((latent-reference['latent'].cuda()).abs().max()))
                assert checks['feature_max_abs'] == 0 and checks['latent_max_abs'] == 0, checks
                task_result['native_parity'] = checks
                del reference
                emit('native_parity_passed', task=task, **checks)
            decoder = paired(dict(official_vae=lambda: decode_vae(latent),
                                  detailer=lambda: infer(head, features)), 2, 7, task, case_id, 'decoder')
            del z, latent, features
            gc.collect()
            pipeline_result = paired(dict(official_vae=lambda: pipeline(source, False),
                                          detailer=lambda: pipeline(source, True)), 1, 3, task, case_id, 'pipeline')
            task_result['cases'].append(dict(sample=chosen[case_id]['sample'], decoder=decoder, pipeline=pipeline_result))
            save('RESULTS.json', results)
        for stage in ['decoder', 'pipeline']:
            task_result[stage] = {route: summary([s for case in task_result['cases']
                for s in case[stage]['samples'][route]]) for route in ['official_vae', 'detailer']}
            task_result[stage]['speedup'] = task_result[stage]['official_vae']['median_ms']/task_result[stage]['detailer']['median_ms']
        emit('task_complete', task=task, decoder=task_result['decoder'], pipeline=task_result['pipeline'])
        save('RESULTS.json', results)
        del head
        gc.collect()
        torch.cuda.empty_cache()
    save('COMPLETE.json', results)
    emit('complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        save('FAILED.json', dict(error=repr(error), time=time.time()))
        raise
