"""Independent Wan2.2 sparse LoRA + joint-query coordinate head, bidirectional stage 1."""
from __future__ import annotations
import argparse
import copy
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from safetensors.torch import save_file
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline, WanVideoUnit_PromptEmbedder

PROMPT = 'output: 2D and 3D human keypoints'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class SparseDataset(Dataset):
    def __init__(self, root, validation=False):
        root = Path(root)
        assert (root/'INDEX_COMPLETE').is_file()
        self.manifest = json.loads((root/'manifest.json').read_text())
        assert sha(root/'targets.npz') == self.manifest['targets_sha256']
        assert sha(Path(self.manifest['source_cache'])/'dataset_index.json') == self.manifest['source_index_sha256']
        with np.load(root/'targets.npz',allow_pickle=False) as z:
            self.xy,self.xyz,self.mask = z['xy'].copy(),z['xyz'].copy(),z['mask'].copy()
            self.indices = np.flatnonzero(z['is_val'] == validation)
    def __len__(self): return len(self.indices)
    def __getitem__(self, index):
        i = self.indices[index]; cid = self.manifest['ids'][i]
        payload = torch.load(Path(self.manifest['source_cache'])/(cid+'.pt'),map_location='cpu',weights_only=False)
        z = payload['source_latent']
        assert payload['clip_id'] == cid and tuple(z.shape) == (48,21,30,52)
        assert z.dtype == torch.bfloat16 and torch.isfinite(z).all()
        return z,torch.from_numpy(self.xy[i]),torch.from_numpy(self.xyz[i]),torch.from_numpy(self.mask[i])

class JointHead(nn.Module):
    """24 spatial joint queries; each temporal latent predicts its own 1/4-frame group."""
    def __init__(self, dim=3072, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.project = nn.Linear(dim,hidden)
        self.position = nn.Linear(2,hidden)
        self.queries = nn.Parameter(torch.randn(24,hidden)*0.02)
        self.attention = nn.MultiheadAttention(hidden,4,batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,512),nn.GELU(),nn.Linear(512,20))
        yy,xx = torch.meshgrid(torch.linspace(-1,1,15),torch.linspace(-1,1,26),indexing='ij')
        self.register_buffer('xy_grid',torch.stack([xx,yy],-1).reshape(390,2))
    def forward(self, features):
        b,n,c = features.shape
        assert n == 21*390
        kv = self.project(self.norm(features)).reshape(b*21,390,-1)
        kv = kv + self.position(self.xy_grid)[None]
        q = self.queries[None].expand(b*21,-1,-1)
        pooled = self.attention(q,kv,kv,need_weights=False)[0]+q
        groups = self.output(pooled).reshape(b,21,24,4,5).permute(0,1,3,2,4)
        coordinates = torch.cat([groups[:,0,:1],groups[:,1:].reshape(b,80,24,5)],1).float()
        return coordinates[...,:2],coordinates[...,2:]

class SparseModule(DiffusionTrainingModule):
    def __init__(self, model_dir, rank=32):
        super().__init__()
        root = Path(model_dir)
        shards = sorted(root.glob('diffusion_pytorch_model-*.safetensors'))
        assert len(shards) == 3
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cpu',model_configs=[ModelConfig(path=[str(p) for p in shards]),ModelConfig(path=str(root/'models_t5_umt5-xxl-enc-bf16.pth'))],tokenizer_config=ModelConfig(path=str(root/'google/umt5-xxl')))
        assert self.pipe.dit.in_dim == 48 and self.pipe.dit.dim == 3072 and len(self.pipe.dit.blocks) == 30
        self.switch_pipe_to_training_mode(self.pipe,trainable_models=None,lora_base_model='dit',lora_target_modules='q,k,v,o,ffn.0,ffn.2',lora_rank=rank,task='sft')
        adapters = [(n,p) for n,p in self.pipe.dit.named_parameters() if p.requires_grad]
        assert adapters and all('.lora_' in n for n,p in adapters)
        for n,p in adapters: p.data = p.data.float()
        bparams = [p for n,p in adapters if '.lora_B.' in n]
        assert len(bparams) == 300 and all(torch.count_nonzero(p).item() == 0 for p in bparams), 'LoRA delta must start at zero'
        assert any(torch.count_nonzero(p).item() > 0 for n,p in adapters if '.lora_A.' in n), 'Do not initialize BOTH A/B at zero'
        self.sparse_head = JointHead()
        self.features = None
        self.pipe.dit.head.register_forward_pre_hook(self.capture_features)
        self.register_buffer('context',None)
    def capture_features(self, module, inputs): self.features = inputs[0]
    @torch.no_grad()
    def prepare(self, device):
        self.pipe.device = device
        self.pipe.text_encoder.to(device)
        self.context = WanVideoUnit_PromptEmbedder().encode_prompt(self.pipe,PROMPT).detach().to(device)
        self.pipe.text_encoder.to('cpu'); self.pipe.text_encoder = None; self.pipe.tokenizer = None
        gc.collect(); torch.cuda.empty_cache()
        self.pipe.dit.to(device); self.sparse_head.to(device)
    def forward(self, source, checkpointing=True):
        self.pipe.model_fn(dit=self.pipe.dit,latents=source,timestep=torch.zeros(source.shape[0],device=source.device,dtype=torch.bfloat16),context=self.context.expand(source.shape[0],-1,-1),use_gradient_checkpointing=checkpointing,use_gradient_checkpointing_offload=False,cfg_merge=False)
        features = self.features; self.features = None
        assert features is not None
        return self.sparse_head(features)

def loss_and_metrics(pred2,pred3,xy,xyz,mask):
    valid = mask.float(); denom = valid.sum().clamp_min(1)
    l2 = ((pred2-xy).square().sum(-1)*valid).sum()/(denom*2)
    l3 = (pred3-xyz).square().mean()
    pixel = torch.linalg.vector_norm((pred2-xy)*pred2.new_tensor([832,480]),dim=-1)
    e2 = (pixel*valid).sum()/denom
    e3 = torch.linalg.vector_norm(pred3-xyz,dim=-1).mean()
    return l2+l3,l2,l3,e2,e3

def cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)

def rng(): return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch']); torch.cuda.set_rng_state(r['cuda'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index_dir',required=True); p.add_argument('--output_dir',required=True)
    p.add_argument('--model_dir',default='/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B')
    p.add_argument('--max_steps',type=int,default=30000); p.add_argument('--save_steps',type=int,default=500)
    p.add_argument('--eval_steps',type=int,default=500); p.add_argument('--eval_samples',type=int,default=32)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--resume',default=None)
    p.add_argument('--overfit_clip',default=None,help='Clip ID or auto; repeat exactly one training clip on both ranks')
    a = p.parse_args()
    rank, local, world = [int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE')]
    assert world == 2, 'This launch is explicitly two GPU'
    torch.cuda.set_device(local); device = torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    control = dist.new_group(backend='gloo',timeout=datetime.timedelta(minutes=30))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = Path(a.output_dir)
    if rank == 0:
        if not a.resume: out.mkdir(parents=True,exist_ok=False)
        assert out.is_dir()
    dist.barrier(group=control)
    train, val = SparseDataset(a.index_dir),SparseDataset(a.index_dir,True)
    chosen_clip = None
    if a.overfit_clip:
        if a.overfit_clip == 'auto':
            # Prefer a fully visible skeleton so loss cannot look good by masking most joints.
            chosen = max(train.indices,key=lambda i:int(train.mask[i].sum()))
        else:
            chosen = train.manifest['ids'].index(a.overfit_clip)
            assert chosen in train.indices, 'Choose a training, not held-out, clip'
        chosen_clip = train.manifest['ids'][chosen]
        train.indices = np.full(512,chosen,dtype=np.int64)
        val.indices = np.array([chosen],dtype=np.int64)
        if rank == 0: print(f'OVERFIT_ONLY clip={chosen_clip} valid_2d={int(train.mask[chosen].sum())}/1944 unique_train_clips=1 same_clip_on_both_ranks=YES',flush=True)
    contract = {'schema':'wan22_sparse_lora_jointhead_v1','model_dir':a.model_dir,'index_sha256':sha(Path(a.index_dir)/'manifest.json'),'script_sha256':sha(__file__),'pipeline_sha256':sha('diffsynth/pipelines/wan_video.py'),'dit_sha256':sha('diffsynth/models/wan_video_dit.py'),'prompt':PROMPT,'lora_rank':32,'lora_targets':'q,k,v,o,ffn.0,ffn.2','head':'24 joint queries, hidden256, attention4heads, temporal1+4n direct regression','attention':'bidirectional','seed':a.seed,'world_size':world,'lora_lr':1e-4,'head_lr':1e-4,'warmup':100,'weight_decay':0.01,'gradient_clip':1.0,'2d_weight':1.0,'3d_weight':1.0,'train_clips':1 if chosen_clip else len(train),'val_clips':0 if chosen_clip else len(val),'overfit_clip':chosen_clip}
    module = SparseModule(a.model_dir); module.prepare(device)
    optimizer = torch.optim.AdamW([{'params':[p for n,p in module.named_parameters() if p.requires_grad and '.lora_' in n],'lr':1e-4},{'params':module.sparse_head.parameters(),'lr':1e-4}],weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.0,(s+1)/100))
    step, epoch, cursor = 0,0,0
    if a.resume:
        state = torch.load(a.resume,map_location='cpu',weights_only=False)
        assert state['contract'] == contract, 'Resume contract mismatch'
        result = module.load_state_dict(state['weights'],strict=False)
        assert not result.unexpected_keys
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        step,epoch,cursor = state['step'],state['epoch'],state['next_batch']
        restore_rng(state['rng'][rank]); print(f'RESUMED rank={rank} step={step} epoch={epoch} next_batch={cursor}',flush=True)
    if rank == 0:
        (out/'config.json').write_text(json.dumps({'args':vars(a),'contract':contract,'initialization':'official frozen base, random LoRA A + zero B (zero delta); new random sparse head; NO dense checkpoint'},indent=2))
        shutil.copyfile(__file__,out/'train_source.py')
        print(json.dumps(contract),flush=True)
        print('INITIAL_ZERO_DELTA_LORA_VERIFIED sites=300; sparse_head=new; dense_checkpoint=NONE',flush=True)
    ddp = DDP(module,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
    sampler = DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=a.seed,drop_last=True)
    loader = DataLoader(train,batch_size=1,sampler=sampler,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(a.seed+rank))

    @torch.no_grad()
    def evaluate():
        module.eval(); values = torch.zeros(6,device=device,dtype=torch.float64)
        for i in range(rank,min(a.eval_samples,len(val)),world):
            batch = [v[None].to(device) for v in val[i]]
            with torch.autocast('cuda',dtype=torch.bfloat16): pred2,pred3 = module(batch[0],False)
            metrics = loss_and_metrics(pred2,pred3,*batch[1:])
            if chosen_clip and rank == 0:
                np.savez_compressed(out/f'overfit_prediction_step{step:06d}.npz',prediction_2d=pred2.cpu().numpy(),prediction_3d=pred3.cpu().numpy(),target_2d=batch[1].cpu().numpy(),target_3d=batch[2].cpu().numpy(),valid_2d=batch[3].cpu().numpy())
            values[:5] += torch.stack([m.detach().double() for m in metrics]); values[5] += 1
        dist.all_reduce(values)
        if rank == 0:
            row = dict(zip(['loss','mse2d','mse3d','error2d_pixels','mpjpe3d_per_height'],(values[:5]/values[5]).tolist()))
            row.update(step=step,n=int(values[5]),note='3D error is normalized by actor height, NOT meters; '+('TRAIN SAMPLE FIT ONLY, not validation' if chosen_clip else 'clip-level holdout'))
            with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print('VALIDATION '+json.dumps(row),flush=True)
        module.train()

    def save(next_epoch,next_cursor):
        states = [None]*world if rank == 0 else None
        dist.gather_object(rng(),states,dst=0,group=control)
        if rank == 0:
            weights = {n:cpu(p).contiguous() for n,p in module.named_parameters() if p.requires_grad}
            local_out = Path('/tmp')/('wan22_sparse_'+out.name); local_out.mkdir(exist_ok=True)
            weight_temp = local_out/f'step-{step}.safetensors'
            save_file(weights,str(weight_temp),metadata={'contract':json.dumps(contract),'step':str(step)})
            target = out/weight_temp.name; temp = out/(weight_temp.name+'.tmp')
            shutil.copyfile(weight_temp,temp); os.replace(temp,target); weight_temp.unlink()
            bundle = {'weights':weights,'optimizer':cpu(optimizer.state_dict()),'scheduler':scheduler.state_dict(),'rng':states,'step':step,'epoch':next_epoch,'next_batch':next_cursor,'contract':contract}
            pending = local_out/'trainer_state.pending.pt'; latest = local_out/'trainer_state_latest.pt'
            torch.save(bundle,pending)
            if latest.is_file(): os.replace(latest,local_out/'trainer_state_previous.pt')
            os.replace(pending,latest)
            remote_pending = out/'trainer_state.pending.pt'; shutil.copyfile(latest,remote_pending); os.replace(remote_pending,out/'trainer_state_latest.pt')
            print(f'SAVED step={step} weights={target} bundled_optimizer_rng_cursor=YES',flush=True)
        dist.barrier(group=control)

    if not a.resume: save(0,0)
    evaluate()
    start = time.perf_counter(); recent = []
    while step < a.max_steps:
        sampler.set_epoch(epoch)
        for position,batch in enumerate(loader):
            if position < cursor: continue
            batch = [v.to(device,non_blocking=True) for v in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): pred2,pred3 = ddp(batch[0],True)
            metrics = loss_and_metrics(pred2,pred3,*batch[1:]); loss = metrics[0]
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            values = torch.stack([m.detach() for m in metrics]); dist.all_reduce(values); values /= world
            recent.append(values.cpu().tolist())
            next_epoch, next_cursor = (epoch+1,0) if position+1 == len(loader) else (epoch,position+1)
            if step%10 == 0 and rank == 0:
                avg = np.mean(recent,axis=0); recent.clear()
                row = {'step':step,'loss':float(avg[0]),'mse2d':float(avg[1]),'mse3d':float(avg[2]),'error2d_pixels':float(avg[3]),'mpjpe3d_per_height':float(avg[4]),'grad_norm':float(norm),'lr':scheduler.get_last_lr(),'elapsed_sec':time.perf_counter()-start,'max_mem_gib':torch.cuda.max_memory_allocated()/2**30}
                with (out/'train_metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print('TRAIN '+json.dumps(row),flush=True)
            if rank != 0 and step%10 == 0: recent.clear()
            if step%a.eval_steps == 0: evaluate()
            if step%a.save_steps == 0 or step == a.max_steps: save(next_epoch,next_cursor)
            if step >= a.max_steps: break
        epoch += 1; cursor = 0
    if rank == 0: (out/('SMOKE_COMPLETE' if a.max_steps <= 5 else 'TRAINING_COMPLETE')).touch()
    dist.barrier(group=control); dist.destroy_process_group()

if __name__ == '__main__': main()
