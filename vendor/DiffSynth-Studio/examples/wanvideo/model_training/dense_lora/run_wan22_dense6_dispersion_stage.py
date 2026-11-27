"""Isolated 100k->120k DispLoss continuation, paired optimizer/RNG recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import train_wan22_dense6_dispersion_ddp as training
from wan22_dense6_checkpoint import checkpoint_metadata

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--output', required=True)
p.add_argument('--target_step', type=int, default=120000)
p.add_argument('--weight', type=float, default=0.01)
p.add_argument('--tau', type=float, default=1.0)
p.add_argument('--resume_stage', action='store_true')
o = p.parse_args()
source, output = Path(o.source).resolve(), Path(o.output).resolve()
if source == output or o.weight < 0 or o.tau <= 0:
    raise ValueError('Invalid stage configuration')
here = Path(__file__).resolve().parent
contract = {'schema': 'dense6_dispersion_stage_v1', 'source': str(source),
            'initial_step': 100000, 'weight': o.weight, 'tau': o.tau,
            'feature_block_index': 8, 'distance': 'full_flattened_mean_squared',
            'pair_selection': 'same_prompt_with_diagonal', 'sampling': 'same_task_distinct_clips_cycle6',
            'code_sha256': {name: hashlib.sha256((here/name).read_bytes()).hexdigest() for name in
                ('run_wan22_dense6_dispersion_stage.py', 'train_wan22_dense6_dispersion_ddp.py', 'wan22_dense6_dispersion.py')}}
control = None
original_init = dist.init_process_group
original_barrier = dist.barrier
def init(*args, **kwargs):
    global control
    original_init(*args, **kwargs)
    control = dist.new_group(backend='gloo', timeout=timedelta(seconds=7200))
def barrier(*args, **kwargs):
    if not args and kwargs.get('group') is None:
        kwargs['group'] = control
    return original_barrier(*args, **kwargs)
dist.init_process_group, dist.barrier = init, barrier

def stage_args():
    values = json.loads((source/'training_args.json').read_text())
    values.update(output_path=str(output), resume_from=str(output) if o.resume_stage else None,
                  stage_source=str(source), init_checkpoint=None, max_train_steps=o.target_step,
                  dispersion_weight=o.weight, dispersion_tau=o.tau)
    return argparse.Namespace(**values)
training.parse_args = stage_args

original_load = training.load_dense6_checkpoint
def strict_load(module, path, **kwargs):
    expected = dict(kwargs['expected_provenance'])
    metadata = checkpoint_metadata(path)
    old_revision = metadata['diffsynth_git_head']
    if old_revision != expected['diffsynth_git_head']:
        repo = here.parents[3]
        subprocess.run(['git', 'merge-base', '--is-ancestor', old_revision, 'HEAD'], cwd=repo, check=True)
        for name in ('train_wan22_dense6_ddp.py', 'wan22_dense6_checkpoint.py', 'wan22_dense6_task_router.py', 'wan22_masked_loss.py', 'wan22_dense6_task_sampler.py'):
            subprocess.run(['git','diff','--exit-code', old_revision,'--', str(here.relative_to(repo)/name)], cwd=repo, check=True)
        expected['diffsynth_git_head'] = old_revision
    kwargs['expected_provenance'] = expected
    return original_load(module, path, **kwargs)

def load_state(resume_from, output_dir, rank, module, args, optimizer, scheduler):
    root = output if o.resume_stage else source
    state = torch.load(root/'trainer_state_bundled_latest.pt', map_location='cpu', weights_only=False)
    if state['world_size'] != dist.get_world_size():
        raise ValueError('World size mismatch')
    if o.resume_stage and state.get('dispersion_contract') != contract:
        raise ValueError('Stage code/config contract mismatch')
    if not o.resume_stage and (state.get('schema') != 'dense6_bundled_recovery_v1' or state['global_step'] != 100000):
        raise ValueError('Expected completed baseline at 100k')
    step = int(state['global_step'])
    meta = strict_load(module, root/f'step-{step}.safetensors', expected_provenance=module.provenance_metadata,
        lora_rank=args.lora_rank, lora_targets=args.lora_target_modules,
        router_hidden_dim=args.router_hidden_dim, expected_router_mode=args.router_mode)
    if int(meta['step']) != step:
        raise ValueError('Unpaired optimizer and weights')
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    rng = state['rng_by_rank'][rank]
    random.setstate(rng['python_rng']); np.random.set_state(rng['numpy_rng'])
    torch.set_rng_state(rng['torch_rng']); torch.cuda.set_rng_state(rng['cuda_rng'])
    epoch, offset = (int(state['epoch']), int(state['next_batch_in_epoch'])) if o.resume_stage else (0, 0)
    if rank == 0:
        (output/'dispersion_contract.json').write_text(json.dumps(contract, indent=2))
        print(f'STAGE_RECOVERY step={step} epoch={epoch} offset={offset} original_optimizer_and_rng_restored=True sampler_cursor_reset={not o.resume_stage}', flush=True)
    return step, epoch, offset
training.load_trainer_state = load_state

original_save = training.save_dense6_checkpoint
def save_weights(module, path, **kwargs):
    if Path(path).exists():
        raise FileExistsError(path)
    kwargs['extra_metadata'] = {**kwargs.get('extra_metadata', {}), 'dispersion_contract': json.dumps(contract, sort_keys=True)}
    return original_save(module, path, **kwargs)
training.save_dense6_checkpoint = save_weights

def save_state(output_dir, rank, *, global_step, epoch, next_batch_in_epoch, optimizer, scheduler):
    rng = {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
           'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}
    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(rng, gathered, dst=0, group=control)
    if rank != 0:
        return
    local = Path('/tmp') / ('wan22_dense6_dispersion_' + output.name)
    local.mkdir(exist_ok=True)
    if shutil.disk_usage(local).free < 3 * 1024**3:
        raise OSError('Insufficient local checkpoint space (<3GiB)')
    state = {'schema': 'dense6_dispersion_stage_v1', 'world_size': dist.get_world_size(),
             'global_step': global_step, 'epoch': epoch, 'next_batch_in_epoch': next_batch_in_epoch,
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'rng_by_rank': gathered, 'dispersion_contract': contract}
    tmp = local/'writing.pt'; latest = local/'trainer_state_bundled_latest.pt'
    torch.save(state, tmp)
    if latest.exists():
        os.replace(latest, local/'trainer_state_bundled_previous.pt')
    os.replace(tmp, latest)
    remote_tmp = output/'trainer_state_bundled_latest.tmp'
    shutil.copyfile(latest, remote_tmp)
    os.replace(remote_tmp, output/'trainer_state_bundled_latest.pt')
    print(f'DISPERSION_CHECKPOINT_SAVED step={global_step}', flush=True)
training.save_trainer_state = save_state
training.main()
