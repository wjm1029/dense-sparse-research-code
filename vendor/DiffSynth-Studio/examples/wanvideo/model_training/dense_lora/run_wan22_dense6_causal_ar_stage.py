"""Dense6 causal/AR stage driver; retain dispersion and strict provenance."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import numpy as np
import torch
import torch.distributed as dist
from wan22_dense6_checkpoint import checkpoint_metadata

p = argparse.ArgumentParser()
p.add_argument('--stage', choices=('causal','ar'), required=True)
p.add_argument('--source', required=True)
p.add_argument('--source_patch', required=True)
p.add_argument('--output', required=True)
p.add_argument('--max_steps', type=int, required=True)
p.add_argument('--resume', action='store_true')
o = p.parse_args()
if o.stage == 'causal':
    import train_wan22_dense6_causal_ddp as adapter
else:
    import train_wan22_dense6_ar_tf_ddp as adapter
training = adapter.dense6
here = Path(__file__).resolve().parent
source, output = Path(o.source).resolve(), Path(o.output).resolve()
source_patch = Path(o.source_patch).resolve()
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
source_meta = checkpoint_metadata(source)
if sha(source_patch) != source_meta['runtime_source_patch_sha256']:
    raise ValueError('Initialization source patch/receipt hash mismatch')
if o.max_steps <= 0 or output == source.parent:
    raise ValueError('Invalid stage target/output')
if o.stage == 'ar' and source_meta.get('causal_frames_per_chunk') != '1':
    raise ValueError('AR must initialize from the frame-causal stage')
names = ('run_wan22_dense6_causal_ar_stage.py','train_wan22_dense6_causal_ddp.py',
         'train_wan22_dense6_ar_tf_ddp.py','train_wan22_dense6_dispersion_ddp.py',
         'wan22_dense6_dispersion.py','train_wan22_dense6_ddp.py',
         'wan22_dense6_checkpoint.py','wan22_dense6_task_router.py',
         'wan22_masked_loss.py','wan22_causal_attention.py','wan22_ar_teacher_forcing.py')
contract = {'schema':'dense6_causal_ar_dispersion_v1','stage':o.stage,
            'source_checkpoint':str(source),'source_sha256':sha(source),
            'source_patch_sha256':sha(source_patch),'source_step':source_meta['step'],
            'max_steps':o.max_steps,'world_size':4,'dispersion_weight':0.01,'dispersion_tau':1.0,
            'frames_per_chunk':1 if o.stage=='causal' else 3,'local_attn_frames':-1,'sink_frames':0,
            'source_hashes':{name:sha(here/name) for name in names}}
receipt = Path('/pfs/mayuema/dense/wan/docs') / f'wan22_dense6_{o.stage}_dispersion_20261013_runtime.json'
if int(os.environ.get('RANK','0')) == 0:
    contents = json.dumps(contract,sort_keys=True,indent=2)+'\n'
    if receipt.exists() and receipt.read_text() != contents:
        raise ValueError('Immutable runtime receipt already exists with different code/config')
    if not receipt.exists():
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(contents); os.replace(temporary, receipt)

control = None
init_original, barrier_original = dist.init_process_group, dist.barrier
def init(*args,**kwargs):
    global control
    init_original(*args,**kwargs)
    control = dist.new_group(backend='gloo',timeout=timedelta(seconds=7200))
    dist.barrier(group=control)
    if not receipt.is_file() or json.loads(receipt.read_text()) != contract:
        raise ValueError('Runtime receipt contract mismatch')
def barrier(*args,**kwargs):
    if not args and kwargs.get('group') is None:
        kwargs['group'] = control
    return barrier_original(*args,**kwargs)
dist.init_process_group, dist.barrier = init, barrier

def args_for_stage():
    cfg = json.loads((Path('/pfs/mayuema/dense/wan/outputs/wan22_dense6_flow32_dispersion_100k_to120k_20261012')/'training_args.json').read_text())
    cfg.update(output_path=str(output), max_train_steps=o.max_steps,
               resume_from=str(output) if o.resume else None, init_checkpoint=None,
               init_code_patch_path=None, stage_source=str(source), code_patch_path=str(receipt),
               dispersion_weight=0.01, dispersion_tau=1.0)
    if cfg.get('lingbot_depth_cache_dir'):
        raise ValueError('This stage is scoped to the six-task synthetic dataset')
    return argparse.Namespace(**cfg)
training.parse_args = args_for_stage
training.Dense6LoRAModule = (adapter.CausalDense6LoRAModule if o.stage == 'causal'
                             else adapter.ARTeacherForcingDense6LoRAModule)
stage_load = training.load_dense6_checkpoint if o.stage == 'causal' else adapter._load_ar_checkpoint
stage_save = adapter._save_causal_checkpoint if o.stage == 'causal' else adapter._save_ar_checkpoint

def save_weights(module,path,**kwargs):
    if Path(path).exists():
        raise FileExistsError(path)
    kwargs['extra_metadata'] = {**kwargs.get('extra_metadata',{}),
        'stage_contract':json.dumps(contract,sort_keys=True),
        'initialization_checkpoint':str(source), 'initialization_checkpoint_sha256':contract['source_sha256'],
        'dispersion_weight':'0.01','dispersion_tau':'1.0','dispersion_feature_block':'8',
        'stage_step_numbering':'local_to_stage_v1'}
    return stage_save(module,path,**kwargs)
training.save_dense6_checkpoint = save_weights

def load_state(resume_from,output_dir,rank,module,args,optimizer,scheduler):
    cfg=module.causal_controller.config
    if (cfg.frames_per_chunk,cfg.local_attn_frames,cfg.sink_frames) != (contract['frames_per_chunk'],-1,0):
        raise ValueError('Unexpected attention configuration/environment')
    expected = dict(module.provenance_metadata)
    if o.resume:
        state = torch.load(output/'trainer_state_bundled_latest.pt',map_location='cpu',weights_only=False)
        if state.get('stage_contract') != contract or state['world_size'] != dist.get_world_size():
            raise ValueError('Stage resume contract mismatch')
        step = int(state['global_step']); path = output/f'step-{step}.safetensors'
        expected_stage = checkpoint_metadata(path)
        for key,value in module.causal_controller.config.metadata().items():
            if expected_stage.get(key) != value:
                raise ValueError(f'Causal resume metadata mismatch: {key}')
        stage_load(module,path,expected_provenance=expected,lora_rank=args.lora_rank,
            lora_targets=args.lora_target_modules,router_hidden_dim=args.router_hidden_dim,
            expected_router_mode=args.router_mode)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        rng = state['rng_by_rank'][rank]
        random.setstate(rng['python_rng']); np.random.set_state(rng['numpy_rng'])
        torch.set_rng_state(rng['torch_rng']); torch.cuda.set_rng_state(rng['cuda_rng'])
        return step,int(state['epoch']),int(state['next_batch_in_epoch'])
    expected['runtime_source_patch_sha256'] = sha(source_patch)
    current_revision = expected['diffsynth_git_head']
    old_revision = source_meta['diffsynth_git_head']
    if old_revision != current_revision:
        repo = here.parents[3]
        subprocess.run(['git','merge-base','--is-ancestor',old_revision,current_revision],cwd=repo,check=True)
        for name in ('train_wan22_dense6_ddp.py','wan22_dense6_checkpoint.py','wan22_dense6_task_router.py','wan22_masked_loss.py','wan22_dense6_task_sampler.py'):
            subprocess.run(['git','diff','--exit-code',old_revision,'--',str(here.relative_to(repo)/name)],cwd=repo,check=True)
    expected['diffsynth_git_head'] = old_revision
    loaded = stage_load(module,source,expected_provenance=expected,lora_rank=args.lora_rank,
        lora_targets=args.lora_target_modules,router_hidden_dim=args.router_hidden_dim,
        expected_router_mode=args.router_mode)
    if loaded['step'] != source_meta['step']:
        raise ValueError('Initialization step mismatch')
    if rank == 0:
        (output/'stage_contract.json').write_text(json.dumps(contract,indent=2))
        snapshots = output/'runtime_source'; snapshots.mkdir()
        for name in names:
            shutil.copyfile(here/name,snapshots/name)
        print(f'STAGE_INITIALIZED stage={o.stage} source_step={loaded["step"]} stage_step=0 LoRA_router_inherited=True optimizer=fresh_stage dispersion=0.01',flush=True)
    return 0,0,0
training.load_trainer_state = load_state

def save_state(output_dir,rank,*,global_step,epoch,next_batch_in_epoch,optimizer,scheduler):
    rng={'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
         'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
    all_rng=[None]*dist.get_world_size() if rank==0 else None
    dist.gather_object(rng,all_rng,dst=0,group=control)
    if rank != 0:
        return
    spool = Path('/tmp')/('wan22_dense6_stage_'+output.name); spool.mkdir(exist_ok=True)
    if shutil.disk_usage(spool).free < 3*1024**3:
        raise OSError('Insufficient local space for checkpoint (<3GiB)')
    state={'schema':'dense6_causal_ar_dispersion_v1','stage_contract':contract,
           'world_size':dist.get_world_size(),'global_step':global_step,'epoch':epoch,
           'next_batch_in_epoch':next_batch_in_epoch,'optimizer':optimizer.state_dict(),
           'scheduler':scheduler.state_dict(),'rng_by_rank':all_rng}
    temporary=spool/'writing.pt'; latest=spool/'trainer_state_bundled_latest.pt'
    torch.save(state,temporary)
    if latest.exists(): os.replace(latest,spool/'trainer_state_bundled_previous.pt')
    os.replace(temporary,latest)
    remote_tmp=output/'trainer_state_bundled_latest.tmp'
    shutil.copyfile(latest,remote_tmp); os.replace(remote_tmp,output/'trainer_state_bundled_latest.pt')
    print(f'STAGE_CHECKPOINT_COMPLETE stage={o.stage} step={global_step}',flush=True)
training.save_trainer_state = save_state
training.main()
if int(os.environ.get('RANK','0')) == 0:
    (output/'TRAINING_COMPLETE').write_text(json.dumps({'stage':o.stage,'step':o.max_steps,'stage_contract':contract},indent=2))
