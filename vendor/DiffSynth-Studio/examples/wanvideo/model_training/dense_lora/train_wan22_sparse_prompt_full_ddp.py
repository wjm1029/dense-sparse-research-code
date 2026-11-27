"""Prompt-selected 2D OR 3D coordinate prediction, shared sparse LoRA + task HyperRouter."""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from safetensors.torch import save_file
from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
from train_wan22_sparse_ddp import SparseModule,SparseDataset,JointHead,cpu,rng,restore_rng,sha
from wan22_sparse_task_router import SparseTaskRouter,discover_lora_sites,TASK_PROMPTS

class ConditionalHead(JointHead):
    def __init__(self,dim=3072,hidden=256):
        super().__init__(dim,hidden)
        self.output = self.output[:-1]
        self.readouts = nn.ModuleList([nn.Linear(512,8),nn.Linear(512,12)])
    def forward(self,features,task):
        b,n,c = features.shape
        assert n == 21*390 and task in (0,1)
        kv = self.project(self.norm(features)).reshape(b*21,390,-1)
        kv = kv+self.position(self.xy_grid)[None]
        q = self.queries[None].expand(b*21,-1,-1)
        pooled = self.attention(q,kv,kv,need_weights=False)[0]+q
        # Only the selected readout runs. This is not joint 5-coordinate output.
        d = 2 if task == 0 else 3
        groups = self.readouts[task](self.output(pooled)).reshape(b,21,24,4,d).permute(0,1,3,2,4)
        return torch.cat([groups[:,0,:1],groups[:,1:].reshape(b,80,24,d)],1).float()

class PromptSparseModule(SparseModule):
    def __init__(self,model_dir):
        super().__init__(model_dir)
        self.sparse_head = ConditionalHead()
        sites = discover_lora_sites(self.pipe.dit)
        assert len(sites) == 300
        self.task_router = SparseTaskRouter([n for n,_ in sites])
        self.task_router.bind(self.pipe.dit)
    @torch.no_grad()
    def prepare(self,device):
        self.pipe.device = device; self.pipe.text_encoder.to(device)
        encoder = WanVideoUnit_PromptEmbedder()
        contexts = [encoder.encode_prompt(self.pipe,p).detach() for p in TASK_PROMPTS]
        self.context = torch.cat(contexts,0).to(device=device,dtype=torch.bfloat16)
        features = torch.stack([c[0,0].float().cpu() for c in contexts])
        assert not torch.equal(features[0],features[1]), 'Task embeddings must be distinct'
        self.task_router.set_task_feature_table(features)
        self.pipe.text_encoder.to('cpu'); self.pipe.text_encoder = None; self.pipe.tokenizer = None
        gc.collect(); torch.cuda.empty_cache()
        self.pipe.dit.to(device); self.sparse_head.to(device); self.task_router.to(device)
    def forward(self,source,prompt,checkpointing=True):
        if prompt not in TASK_PROMPTS: raise ValueError(f'Unsupported prompt {prompt!r}; use {TASK_PROMPTS}')
        task = TASK_PROMPTS.index(prompt)
        gates = self.task_router(torch.full((source.shape[0],),task,device=source.device,dtype=torch.long))
        self.task_router.call_with_error_cleanup(lambda:self.pipe.model_fn(dit=self.pipe.dit,latents=source,timestep=torch.zeros(source.shape[0],device=source.device,dtype=torch.bfloat16),context=self.context[task:task+1].expand(source.shape[0],-1,-1),use_gradient_checkpointing=checkpointing,use_gradient_checkpointing_offload=False,cfg_merge=False))
        features = self.features; self.features = None
        return self.sparse_head(features,task),self.task_router.identity_regularization(gates)

def coordinate_loss(prediction,xy,xyz,mask,task):
    if task == 0:
        valid = mask.float(); denom = valid.sum().clamp_min(1)
        data = ((prediction-xy).square().sum(-1)*valid).sum()/(2*denom)
        error = (torch.linalg.vector_norm((prediction-xy)*prediction.new_tensor([832,480]),dim=-1)*valid).sum()/denom
    else:
        data = (prediction-xyz).square().mean()
        error = torch.linalg.vector_norm(prediction-xyz,dim=-1).mean()
    return data,error

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index_dir',required=True); p.add_argument('--output_dir',required=True)
    p.add_argument('--model_dir',default='/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B')
    p.add_argument('--max_steps',type=int,default=2000); p.add_argument('--overfit_clip',default=None)
    p.add_argument('--save_steps',type=int,default=500); p.add_argument('--eval_steps',type=int,default=100)
    p.add_argument('--eval_samples',type=int,default=372); p.add_argument('--max_hours',type=float,default=0)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--resume',default=None)
    p.add_argument('--stop_file',default='/tmp/wan22_sparse_prompt_STOP_REQUESTED_20261012')
    a = p.parse_args()
    rank,local,world = [int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE')]
    assert world in (2,4)
    torch.cuda.set_device(local); device = torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(hours=2))
    control = dist.new_group(backend='gloo',timeout=datetime.timedelta(hours=2))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = Path(a.output_dir)
    if rank == 0:
        if not a.resume: out.mkdir(parents=True,exist_ok=False)
    dist.barrier(group=control)
    train,val = SparseDataset(a.index_dir),SparseDataset(a.index_dir,True)
    chosen_clip = None
    if a.overfit_clip:
        chosen = max(train.indices,key=lambda i:int(train.mask[i].sum())) if a.overfit_clip == 'auto' else train.manifest['ids'].index(a.overfit_clip)
        assert chosen in train.indices
        chosen_clip = train.manifest['ids'][chosen]
        train.indices = np.full(512,chosen,dtype=np.int64); val.indices = np.array([chosen],dtype=np.int64)
    contract = {'schema':'wan22_sparse_prompt_hyperlora_v1','model_dir':a.model_dir,'index_hash':sha(Path(a.index_dir)/'manifest.json'),'script_hash':sha(__file__),'base_sparse_hash':sha(Path(__file__).with_name('train_wan22_sparse_ddp.py')),'sparse_router_hash':sha(Path(__file__).with_name('wan22_sparse_task_router.py')),'dense_router_hash':sha(Path(__file__).with_name('wan22_dense6_task_router.py')),'pipeline_hash':sha('diffsynth/pipelines/wan_video.py'),'dit_hash':sha('diffsynth/models/wan_video_dit.py'),'prompts':list(TASK_PROMPTS),'world_size':world,'seed':a.seed,'overfit_clip':chosen_clip,'train_clips':1 if chosen_clip else len(train),'val_clips':0 if chosen_clip else len(val),'rank':32,'router_hidden':512,'router_lr':1e-5,'lora_head_lr':1e-4,'router_identity_weight':1e-3,'warmup_steps':100,'attention':'bidirectional','init':'official base, zero-delta new LoRA, new heads and identity-initialized task router; no old checkpoint'}
    module = PromptSparseModule(a.model_dir); module.prepare(device)
    opt = torch.optim.AdamW([{'params':[p for n,p in module.named_parameters() if p.requires_grad and '.lora_' in n],'lr':1e-4},{'params':module.sparse_head.parameters(),'lr':1e-4},{'params':module.task_router.parameters(),'lr':1e-5}],weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.0,(s+1)/100))
    step,epoch,cursor = 0,0,0
    if a.resume:
        state = torch.load(a.resume,map_location='cpu',weights_only=False)
        assert state['contract'] == contract
        expected = {n for n,p in module.named_parameters() if p.requires_grad}|{'context','task_router.task_feature_table','task_router.features_ready','sparse_head.xy_grid'}
        assert set(state['weights']) == expected
        result = module.load_state_dict(state['weights'],strict=False); assert not result.unexpected_keys
        opt.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        step,epoch,cursor = state['step'],state['epoch'],state['next_batch']; restore_rng(state['rng'][rank])
    ddp = DDP(module,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
    sampler = DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=a.seed,drop_last=True)
    loader = DataLoader(train,batch_size=1,sampler=sampler,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(a.seed+rank))
    local_out = Path('/tmp')/('wan22_sparse_prompt_'+out.name)
    if rank == 0:
        local_out.mkdir(exist_ok=True)
        (out/'config.json').write_text(json.dumps({'args':vars(a),'contract':contract},indent=2))
        shutil.copyfile(__file__,out/'train_source.py')
        print('PROMPT_CONTRACT '+json.dumps(contract),flush=True)
        print('INITIAL_ZERO_DELTA_LORA_VERIFIED; two distinct frozen UMT5 task embeddings; router identity gates; selected head only',flush=True)
    writer = ThreadPoolExecutor(max_workers=1) if rank == 0 else None
    pending_uploads = []
    uploaded_checkpoints = []
    def upload(weight_path,state_path,saved_step):
        for source,target in [(weight_path,out/weight_path.name),(state_path,out/'trainer_state_latest.pt')]:
            temporary = out/(target.name+f'.pending-{saved_step}')
            shutil.copyfile(source,temporary); os.replace(temporary,target)
        print(f'PFS_CHECKPOINT_COMPLETE step={saved_step}',flush=True)
        uploaded_checkpoints.append((weight_path,state_path))
        # Keep the last two fully uploaded local checkpoints plus every pending
        # one. All historical weights remain on PFS; only our local spool is pruned.
        if len(uploaded_checkpoints) > 2:
            for old_path in uploaded_checkpoints.pop(0):
                assert old_path.parent == local_out
                old_path.unlink()
    def save(next_epoch,next_cursor):
        states = [None]*world if rank == 0 else None
        dist.gather_object(rng(),states,dst=0,group=control)
        if rank == 0:
            for future in pending_uploads:
                if future.done(): future.result()
            pending_uploads[:] = [f for f in pending_uploads if not f.done()]
            if len(pending_uploads) >= 3:
                print('PFS_BACKPRESSURE waiting for oldest upload; bounded local spool',flush=True)
                pending_uploads.pop(0).result()
            if shutil.disk_usage(local_out).free < 4*2**30:
                raise RuntimeError('Local checkpoint spool has less than 4 GiB free; refuse unsafe save')
            names = {n for n,p in module.named_parameters() if p.requires_grad}|{'context','task_router.task_feature_table','task_router.features_ready','sparse_head.xy_grid'}
            weights = {n:cpu(v).contiguous() for n,v in module.state_dict().items() if n in names}
            assert set(weights) == names
            weight_path = local_out/f'step-{step}.safetensors'; state_path = local_out/f'trainer_state_step-{step}.pt'
            save_file(weights,str(weight_path),metadata={'contract':json.dumps(contract),'step':str(step)})
            torch.save({'weights':weights,'optimizer':cpu(opt.state_dict()),'scheduler':scheduler.state_dict(),'rng':states,'step':step,'epoch':next_epoch,'next_batch':next_cursor,'contract':contract},state_path)
            link = local_out/'trainer_state_latest.pending'; link.symlink_to(state_path.name); os.replace(link,local_out/'trainer_state_latest.pt')
            pending_uploads.append(writer.submit(upload,weight_path,state_path,step))
            print(f'LOCAL_CHECKPOINT_COMPLETE step={step} path={state_path} PFS_upload=asynchronous',flush=True)
        dist.barrier(group=control)
    @torch.no_grad()
    def evaluate():
        module.eval(); sums = torch.zeros(2,3,device=device,dtype=torch.float64)
        for i in range(rank,min(a.eval_samples,len(val)),world):
            batch = [v[None].to(device) for v in val[i]]
            for task,prompt in enumerate(TASK_PROMPTS):
                with torch.autocast('cuda',dtype=torch.bfloat16): pred,reg = module(batch[0],prompt,False)
                data,error = coordinate_loss(pred,*batch[1:],task)
                sums[task] += torch.stack([data.double(),error.double(),data.new_tensor(1).double()])
                if chosen_clip and rank == 0:
                    np.savez_compressed(out/f'fit_task{task}_step{step:06d}.npz',prediction=pred.cpu().numpy(),target=batch[1 if task==0 else 2].cpu().numpy(),mask=batch[3].cpu().numpy())
        dist.all_reduce(sums)
        if rank == 0:
            for task,prompt in enumerate(TASK_PROMPTS):
                row = {'step':step,'prompt':prompt,'mse':float(sums[task,0]/sums[task,2]),'error':float(sums[task,1]/sums[task,2]),'unit':'pixels' if task==0 else 'actor_height','n':int(sums[task,2]),'scope':'same training clip fit' if chosen_clip else 'clip-level holdout'}
                with (out/'evaluation.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print('EVALUATION '+json.dumps(row),flush=True)
        module.train()
    if not a.resume: save(0,0)
    evaluate(); start = time.perf_counter(); totals = torch.zeros(2,4,device=device,dtype=torch.float64); stop_reason = None
    while step < a.max_steps:
        sampler.set_epoch(epoch)
        for position,batch in enumerate(loader):
            if position < cursor: continue
            # Half the ranks do each task; swapped every step. Tasks are selected by prompt.
            task = (rank+step)%2; prompt = TASK_PROMPTS[task]
            batch = [v.to(device,non_blocking=True) for v in batch]; opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): pred,reg = ddp(batch[0],prompt,True)
            data,error = coordinate_loss(pred,*batch[1:],task); loss = data+1e-3*reg
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); norm = nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
            opt.step(); scheduler.step(); step += 1
            totals[task] += torch.stack([data.detach().double(),error.detach().double(),reg.detach().double(),data.new_tensor(1).double()])
            stop = torch.zeros(1,device=device,dtype=torch.int32)
            if rank == 0:
                if Path(a.stop_file).is_file(): stop.fill_(1)
                elif a.max_hours > 0 and time.perf_counter()-start >= a.max_hours*3600: stop.fill_(2)
            dist.broadcast(stop,src=0); stop_reason = int(stop.item()) or None
            next_epoch,next_cursor = (epoch+1,0) if position+1 == len(loader) else (epoch,position+1)
            if step%10 == 0:
                dist.all_reduce(totals)
                if rank == 0:
                    for future in pending_uploads:
                        if future.done(): future.result()
                    for t in range(2):
                        values = (totals[t,:3]/totals[t,3]).tolist()
                        row = {'step':step,'prompt':TASK_PROMPTS[t],'mse':values[0],'error':values[1],'unit':'pixels' if t==0 else 'actor_height','router_reg':values[2],'grad_norm':float(norm),'elapsed_sec':time.perf_counter()-start}
                        with (out/'train_metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                        print('TRAIN '+json.dumps(row),flush=True)
                totals.zero_()
            if step%a.eval_steps == 0: evaluate()
            if step%a.save_steps == 0 or step == a.max_steps or stop_reason: save(next_epoch,next_cursor)
            if step == a.max_steps or stop_reason: break
        epoch += 1; cursor = 0
        if stop_reason: break
    dist.barrier(group=control); dist.destroy_process_group()
    if rank == 0:
        (local_out/'TRAINING_COMPUTE_COMPLETE').touch()
        print(f'COMPUTE_COMPLETE step={step}; waiting for queued PFS checkpoint uploads',flush=True)
        writer.shutdown(wait=True)
        for future in pending_uploads: future.result()
        (out/('STOPPED' if stop_reason else 'TRAINING_COMPLETE')).touch()


if __name__ == '__main__': main()
