"""Long sparse rollout and OPSD clean-prefix policies, reusing sparse heads/router."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from train_wan22_sparse_ddp import sha
from train_wan22_sparse_prompt_full_ddp import coordinate_loss
from wan22_sparse_causal_ar import CausalARSparseModule, pack_coordinate_groups
from wan22_causal_attention import CausalAttentionConfig
from wan22_opsd_bounded_causal_attention import install_opsd_bounded_causal_attention
from wan22_opsd_prefix_cache import (clone_prefix,replace_chunk_from_replay,swap_trainable_parameters_)


class LongSparseDataset(Dataset):
    def __init__(self,root,validation=False):
        root=Path(root); self.manifest=json.loads((root/'manifest.json').read_text())
        assert (root/'INDEX_COMPLETE').read_text().strip()==sha(root/'manifest.json')
        assert self.manifest['schema']=='wan22_sparse_long477_coordinates_v1'
        assert sha(root/'targets.npz')==self.manifest['targets_sha256']
        self.cache=Path(self.manifest['source_cache'])
        assert sha(self.cache/'source_index.json')==self.manifest['source_index_sha256']
        assert sha(self.cache/'manifest.json')==self.manifest['source_manifest_sha256']
        with np.load(root/'targets.npz',allow_pickle=False) as z:
            self.xy,self.xyz,self.mask=z['xy'].copy(),z['xyz'].copy(),z['mask'].copy()
            self.indices=np.flatnonzero(z['is_val']==validation)
    def __len__(self): return len(self.indices)
    def __getitem__(self,index):
        i=self.indices[index]; cid=self.manifest['ids'][i]
        p=torch.load(self.cache/'source'/(cid+'.pt'),map_location='cpu',weights_only=False)
        assert p['schema']=='wan22_long477_dense_latent_v1' and p['modality']=='source'
        assert p['cache_contract_hash']==self.manifest['source_contract'] and p['clip_id']==cid
        source=p['latent']
        assert source.shape==(48,120,30,52) and source.dtype==torch.bfloat16 and torch.isfinite(source).all()
        return source,torch.from_numpy(self.xy[i]),torch.from_numpy(self.xyz[i]),torch.from_numpy(self.mask[i])


def chunk_raw_span(offset,frames=3):
    return (0,1+4*(frames-1)) if offset==0 else (1+4*(offset-1),1+4*(offset+frames-1))


def feedback_groups(groups,task,offset):
    if task==0:
        valid=torch.isfinite(groups).all(-1)&((groups>=0)&(groups<=1)).all(-1)
        result=torch.cat([torch.where(valid[...,None],groups,torch.zeros_like(groups)),valid[...,None].to(groups.dtype)],-1)
    else: result=groups
    if offset==0:
        result=result.clone(); result[:,0,1:]=0
    return result.detach()


def stream_features(controller,dit,x,timestep,context,offset,caches):
    """Training/eval shared stream math; no checkpoint replay may mutate KV twice."""
    frames=x.shape[2]
    if offset<0 or offset%3 or frames!=3 or offset+frames>controller.config.num_frames:
        raise ValueError('Invalid sparse chunk boundary')
    t=dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim,timestep).to(x.dtype))
    t_mod=dit.time_projection(t).unflatten(1,(6,dit.dim))
    context=dit.text_embedding(context); tokens=dit.patchify(x); f,h,w=tokens.shape[2:]
    tokens=rearrange(tokens,'b c f h w -> b (f h w) c').contiguous()
    freqs=torch.cat((dit.freqs[0][offset:offset+f].view(f,1,1,-1).expand(f,h,w,-1),
        dit.freqs[1][:h].view(1,h,1,-1).expand(f,h,w,-1),
        dit.freqs[2][:w].view(1,1,w,-1).expand(f,h,w,-1)),dim=-1).reshape(f*h*w,1,-1).to(x.device)
    with controller.use_kv_cache(caches):
        for block in dit.blocks: tokens=block(tokens,context,t_mod,freqs)
    # Trigger the original sparse feature hook, while retaining native Wan semantics.
    dit.head(tokens,t)


class LongSparseModule(CausalARSparseModule):
    def __init__(self,model_dir,phase):
        super().__init__(model_dir,'ar')
        self.configure_long(phase)

    def configure_long(self,phase,num_frames=120):
        if phase not in ('self_forcing','opsd_local'): raise ValueError(phase)
        self.phase=phase
        if phase=='self_forcing':
            config=CausalAttentionConfig(num_frames,390,3,-1,0)
            self.causal_controller.config=config
            for module in self.causal_controller.modules: module.config=config
        else:
            self.causal_controller=install_opsd_bounded_causal_attention(self.pipe.dit,
                latent_shape=(48,num_frames,30,52),frames_per_chunk=3,local_attn_frames=12,sink_frames=3)

    def forward(self,rgb,history,prompt,offset,caches):
        task,gates,timestep,context=self._activate(rgb,prompt)
        self.coordinate_history.active=history
        try:
            self.task_router.call_with_error_cleanup(lambda:stream_features(self.causal_controller,
                self.pipe.dit,rgb,timestep,context,offset,caches))
            features=self.features; self.features=None
            groups=self.sparse_head.groups(features,task)
            prediction=(torch.cat([groups[:,0,:1],groups[:,1:].flatten(1,2)],1)
                if offset==0 else groups.flatten(1,2))
        finally: self.coordinate_history.active=None; self.features=None
        return prediction,groups,self.task_router.identity_regularization(gates)

    @torch.no_grad()
    def rollout(self,source,prompt):
        if self.training or self.pipe.dit.training: raise ValueError('Validation must use eval mode')
        task=0 if prompt=='output: 2D keypoints' else 1
        caches=self.causal_controller.new_kv_cache(); result=[]
        history=source.new_zeros(source.shape[0],3,4,24,3)
        for offset in range(0,source.shape[2],3):
            prediction,groups,_=self(source[:,:,offset:offset+3],history,prompt,offset,caches)
            result.append(prediction); history=feedback_groups(groups,task,offset)
            self.task_router.clear()
        return torch.cat(result,1)


def clean_teacher_forward(module,source,history,prompt,offset,teacher_cache,gt_groups,ema,named):
    """Same student trajectory, EMA teacher; delayed one-chunk clean KV replacement."""
    backup=swap_trainable_parameters_(named,ema)
    try:
        with torch.no_grad():
            prediction,_,_=module(source[:,:,offset:offset+3],history,prompt,offset,teacher_cache)
            module.task_router.clear()
            if offset>=3:
                previous=offset-3; start=previous*390
                clean_history=torch.zeros_like(history) if previous==0 else gt_groups[:,previous-3:previous]
                replay=clone_prefix(teacher_cache,start)
                module(source[:,:,previous:previous+3],clean_history,prompt,previous,replay)
                module.task_router.clear()
                replace_chunk_from_replay(teacher_cache,replay,start_token=start,
                    chunk_tokens=module.causal_controller.config.chunk_tokens)
        return prediction
    finally:
        with torch.no_grad():
            for name,p in named: p.copy_(backup[name])
        module.task_router.clear()
