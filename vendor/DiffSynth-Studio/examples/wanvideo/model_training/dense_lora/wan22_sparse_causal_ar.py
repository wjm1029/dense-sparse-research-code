"""Prompt-selected sparse causal/AR adapter; no current-chunk GT leakage."""
import types
import torch
from torch import nn
from train_wan22_sparse_prompt_full_ddp import PromptSparseModule, ConditionalHead
from wan22_sparse_task_router import TASK_PROMPTS
from wan22_causal_attention import install_causal_attention


class StreamingConditionalHead(ConditionalHead):
    """Same parameter names/shapes as the bidirectional head, variable length."""
    def groups(self, features, task):
        b,n,c=features.shape
        if n%390 or task not in (0,1): raise ValueError('Sparse token/task layout mismatch')
        frames=n//390
        kv=self.project(self.norm(features)).reshape(b*frames,390,-1)
        kv=kv+self.position(self.xy_grid)[None]
        q=self.queries[None].expand(b*frames,-1,-1)
        pooled=self.attention(q,kv,kv,need_weights=False)[0]+q
        d=2 if task==0 else 3
        return self.readouts[task](self.output(pooled)).reshape(b,frames,24,4,d).permute(0,1,3,2,4).float()

    def forward(self,features,task,frame_offset=0):
        groups=self.groups(features,task)
        b,f,s,j,d=groups.shape
        if frame_offset==0:
            return torch.cat([groups[:,0,:1],groups[:,1:].reshape(b,(f-1)*4,24,d)],1)
        return groups.reshape(b,f*4,24,d)


def pack_coordinate_groups(coords, mask=None):
    """Map raw 1+4n frames to latent-frame slot groups, with explicit padding."""
    b,t,j,d=coords.shape
    if (t-1)%4 or j!=24 or d not in (2,3): raise ValueError('Coordinate layout mismatch')
    if d==2:
        if mask is None: mask=torch.isfinite(coords).all(-1)&((coords>=0)&(coords<=1)).all(-1)
        safe=torch.where(mask[...,None].bool(),coords,torch.zeros_like(coords))
        values=torch.cat([safe,mask[...,None].to(coords.dtype)],-1)
    else:
        if not torch.isfinite(coords).all(): raise ValueError('Nonfinite 3D coordinate history')
        values=coords
    groups=coords.new_zeros(b,1+(t-1)//4,4,24,3)
    groups[:,0,0]=values[:,0]
    groups[:,1:]=values[:,1:].reshape(b,(t-1)//4,4,24,3)
    return groups


def shifted_coordinate_history(coords,mask=None,chunk=3):
    groups=pack_coordinate_groups(coords,mask)
    history=torch.zeros_like(groups)
    history[:,chunk:]=groups[:,:-chunk]
    return history


class SparseCoordinateHistory:
    def __init__(self,dit):
        self.dit=dit; self.active=None
        projection=nn.Linear(4*24*3,dit.dim,device=dit.patch_embedding.weight.device,dtype=torch.float32)
        nn.init.zeros_(projection.weight); nn.init.zeros_(projection.bias)
        dit.add_module('lora_sparse_history',projection)
        self.original_patchify=dit.patchify
        controller=self
        def patchify(model,x,*args,**kwargs):
            tokens=controller.original_patchify(x,*args,**kwargs)
            history=controller.active
            if history is None or history.shape[:2]!=(x.shape[0],x.shape[2]):
                raise ValueError('Coordinate history not set for this temporal span')
            bias=model.lora_sparse_history(history.flatten(2).float())
            return tokens+bias.permute(0,2,1)[:,:,:,None,None].to(tokens.dtype)
        dit.patchify=types.MethodType(patchify,dit)

    def assert_zero(self):
        if any(torch.count_nonzero(p) for p in self.dit.lora_sparse_history.parameters()):
            raise ValueError('New coordinate history projection is not zero')


class CausalARSparseModule(PromptSparseModule):
    def __init__(self,model_dir,stage):
        super().__init__(model_dir)
        self.sparse_head=StreamingConditionalHead()
        if stage not in ('causal','ar'): raise ValueError(stage)
        self.stage=stage
        self.causal_controller=install_causal_attention(self.pipe.dit,latent_shape=(48,21,30,52),
            frames_per_chunk=1 if stage=='causal' else 3,local_attn_frames=-1,sink_frames=0)
        self.coordinate_history=SparseCoordinateHistory(self.pipe.dit) if stage=='ar' else None

    def _activate(self,source,prompt):
        if prompt not in TASK_PROMPTS: raise ValueError('Unsupported sparse prompt')
        task=TASK_PROMPTS.index(prompt)
        gates=self.task_router(torch.full((source.shape[0],),task,device=source.device,dtype=torch.long))
        return task,gates,torch.zeros(source.shape[0],device=source.device,dtype=torch.bfloat16),self.context[task:task+1].expand(source.shape[0],-1,-1)

    def forward(self,source,prompt,checkpointing=True,target=None,mask=None):
        if self.stage=='ar' and target is None:
            if self.training: raise ValueError('AR training requires teacher-forcing targets')
            return self.free_running(source,prompt)
        task,gates,timestep,context=self._activate(source,prompt)
        if self.stage=='ar':
            self.coordinate_history.active=shifted_coordinate_history(target,mask if task==0 else None)
        try:
            self.task_router.call_with_error_cleanup(lambda:self.pipe.model_fn(dit=self.pipe.dit,
                latents=source,timestep=timestep,context=context,use_gradient_checkpointing=checkpointing,
                use_gradient_checkpointing_offload=False,cfg_merge=False))
            features=self.features; self.features=None
            prediction=self.sparse_head(features,task)
        finally:
            if self.coordinate_history: self.coordinate_history.active=None
        return prediction,self.task_router.identity_regularization(gates)

    @torch.no_grad()
    def free_running(self,source,prompt):
        if self.pipe.dit.training or self.training: raise ValueError('AR inference requires eval')
        task,gates,timestep,context=self._activate(source,prompt)
        caches=self.causal_controller.new_kv_cache()
        history=source.new_zeros(source.shape[0],3,4,24,3)
        results=[]
        try:
            for start in range(0,source.shape[2],3):
                self.coordinate_history.active=history
                self.causal_controller.streaming_forward(self.pipe.dit,source[:,:,start:start+3],
                    timestep,context,frame_offset=start,caches=caches)
                features=self.features; self.features=None
                groups=self.sparse_head.groups(features,task)
                prediction=(torch.cat([groups[:,0,:1],groups[:,1:].reshape(source.shape[0],8,24,groups.shape[-1])],1)
                    if start==0 else groups.flatten(1,2))
                results.append(prediction)
                if task==0:
                    valid=torch.isfinite(groups).all(-1)&((groups>=0)&(groups<=1)).all(-1)
                    history=torch.cat([torch.where(valid[...,None],groups,torch.zeros_like(groups)),valid[...,None].to(groups.dtype)],-1)
                else: history=groups
                if start==0:
                    history=history.clone(); history[:,0,1:]=0
        finally:
            self.coordinate_history.active=None; self.features=None
        return torch.cat(results,1),self.task_router.identity_regularization(gates)
