"""Recovery driver: paired optimizer/weights, bundled RNG, CPU checkpoint barriers."""
import argparse
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import train_wan22_dense6_ddp as training
from wan22_dense6_checkpoint import checkpoint_metadata
from wan22_cache_integrity import sha256_file

p=argparse.ArgumentParser()
p.add_argument('--output_dir',required=True)
p.add_argument('--runtime_patch',required=True)
p.add_argument('--max_train_steps',type=int,default=50000)
options=p.parse_args()
if options.max_train_steps <= 0:
    p.error('--max_train_steps must be positive')
output=Path(options.output_dir).resolve()
repo=Path(__file__).resolve().parents[4]
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
receipt={'recovery_driver_revision':revision,'recovery_driver_sha256':sha256_file(Path(__file__)),
         'recovery_runtime_patch_sha256':sha256_file(Path(options.runtime_patch)),
         'recovery_control_timeout_seconds':'7200',
         'recovery_target_step':str(options.max_train_steps)}
control=None
cursor={'epoch':-1,'offset':0,'length':7350*6//4}

def args_for_resume():
    args=json.loads((output/'training_args.json').read_text())
    args.update(resume_from=str(output),output_path=str(output),max_train_steps=options.max_train_steps)
    return argparse.Namespace(**args)
training.parse_args=args_for_resume

old_init=dist.init_process_group
old_barrier=dist.barrier
def init_with_control(*args,**kwargs):
    global control
    old_init(*args,**kwargs)
    control=dist.new_group(backend='gloo',timeout=timedelta(seconds=7200))
def checkpoint_barrier(*args,**kwargs):
    if control is not None and not args and kwargs.get('group') is None:
        kwargs['group']=control
    return old_barrier(*args,**kwargs)
dist.init_process_group=init_with_control
dist.barrier=checkpoint_barrier

old_load=training.load_dense6_checkpoint
def strict_load_with_revision_check(module,path,**kwargs):
    meta=checkpoint_metadata(path)
    expected=dict(kwargs['expected_provenance'])
    if meta['diffsynth_git_head']!=expected['diffsynth_git_head']:
        subprocess.run(['git','merge-base','--is-ancestor',meta['diffsynth_git_head'],revision],cwd=repo,check=True)
        for name in ('train_wan22_dense6_ddp.py','wan22_dense6_checkpoint.py','wan22_dense6_task_router.py','wan22_masked_loss.py','wan22_dense6_task_sampler.py'):
            subprocess.run(['git','diff','--exit-code',meta['diffsynth_git_head'],'--',f'examples/wanvideo/model_training/dense_lora/{name}'],cwd=repo,check=True)
        expected['diffsynth_git_head']=meta['diffsynth_git_head']
    kwargs['expected_provenance']=expected
    return old_load(module,path,**kwargs)
training.load_dense6_checkpoint=strict_load_with_revision_check

old_save_weights=training.save_dense6_checkpoint
def save_weights_with_receipt(module,path,**kwargs):
    path=Path(path)
    if path.exists():
        archive=output/'recovery_backups'/path.name
        archive.parent.mkdir(exist_ok=True)
        if archive.exists():
            raise FileExistsError(f'Recovery backup already exists: {archive}')
        shutil.copy2(path,archive)
    kwargs['extra_metadata']={**kwargs.get('extra_metadata',{}),**receipt}
    return old_save_weights(module,path,**kwargs)
training.save_dense6_checkpoint=save_weights_with_receipt

def load_consistent_state(resume_from,output_dir,rank,module,args,optimizer,scheduler):
    bundled=output/'trainer_state_bundled_latest.pt'
    state=torch.load(bundled if bundled.exists() else output/'trainer_state_latest.pt',map_location='cpu',weights_only=False)
    step=int(state['global_step']); epoch=int(state['epoch']); offset=int(state['next_batch_in_epoch'])
    path=output/f'step-{step}.safetensors'
    meta=strict_load_with_revision_check(module,path,expected_provenance=module.provenance_metadata,
         lora_rank=args.lora_rank,lora_targets=args.lora_target_modules,
         router_hidden_dim=args.router_hidden_dim,expected_router_mode=args.router_mode)
    if int(meta['step'])!=step: raise ValueError('Optimizer and weight step mismatch')
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
    if bundled.exists():
        if state.get('schema')!='dense6_bundled_recovery_v1' or state.get('world_size')!=dist.get_world_size():
            raise ValueError('Bundled recovery contract mismatch')
        rng=state['rng_by_rank'][rank]
        random.setstate(rng['python_rng']); np.random.set_state(rng['numpy_rng'])
        torch.set_rng_state(rng['torch_rng']); torch.cuda.set_rng_state(rng['cuda_rng'])
        mode='bundled_rng_restore'
    else:
        # Legacy latest RNG files straddle steps 11000 and 11500. Do not mix them.
        seed=args.seed+step+rank
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        mode='legacy_mixed_rng_discarded_and_reseeded'
    cursor.update(epoch=epoch,offset=offset)
    entry={'rank':rank,'step':step,'epoch':epoch,'next_batch_in_epoch':offset,'rng_mode':mode,**receipt}
    (output/f'recovery_receipt_rank{rank}.json').write_text(json.dumps(entry,indent=2))
    print('CONSISTENT_RECOVERY '+json.dumps(entry),flush=True)
    return step,epoch,0
training.load_trainer_state=load_consistent_state

# Skip sampler indices, not 11000 cached videos per rank. Preserve cursor semantics.
old_iter=training.SixTaskStratifiedDistributedSampler.__iter__
def resumed_iterator(self):
    offset=cursor['offset'] if self.epoch==cursor['epoch'] else 0
    yield from itertools.islice(old_iter(self),offset,None)
training.SixTaskStratifiedDistributedSampler.__iter__=resumed_iterator

def save_bundle(output_dir,rank,*,global_step,epoch,next_batch_in_epoch,optimizer,scheduler):
    rng={'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
         'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
    all_rng=[None]*dist.get_world_size() if rank==0 else None
    dist.gather_object(rng,all_rng,dst=0,group=control)
    if rank!=0: return
    if epoch==cursor['epoch']: next_batch_in_epoch+=cursor['offset']
    if next_batch_in_epoch>=cursor['length']: epoch+=1; next_batch_in_epoch=0
    bundle={'schema':'dense6_bundled_recovery_v1','world_size':dist.get_world_size(),
       'global_step':global_step,'epoch':epoch,'next_batch_in_epoch':next_batch_in_epoch,
       'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
       'rng_by_rank':all_rng,'recovery_receipt':receipt}
    local=Path('/tmp/wan22_dense6_flow32_recovery'); local.mkdir(exist_ok=True)
    fd,local_tmp=tempfile.mkstemp(prefix='bundle-',suffix='.pt',dir=local); os.close(fd)
    try:
        torch.save(bundle,local_tmp)
        local_latest=local/'trainer_state_bundled_latest.pt'
        if local_latest.exists(): os.replace(local_latest,local/'trainer_state_bundled_previous.pt')
        os.replace(local_tmp,local_latest)
        final=output/'trainer_state_bundled_latest.pt'
        temp=output/f'trainer_state_bundled_step{global_step}.tmp-p{os.getpid()}'
        shutil.copyfile(local_latest,temp)
        os.replace(temp,final)
        print(f'BUNDLED_RECOVERY_SAVED step={global_step} local={local_latest} remote={final}',flush=True)
    finally:
        if Path(local_tmp).exists(): Path(local_tmp).unlink()
training.save_trainer_state=save_bundle

if args_for_resume().lingbot_depth_cache_dir:
    raise ValueError('This recovery driver is scoped to the synthetic stratified flow32 stage')
training.main()
