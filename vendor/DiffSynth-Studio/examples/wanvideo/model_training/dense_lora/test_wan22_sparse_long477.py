"""Tiny exact source-head inheritance plus differentiable/EMA prefix checks."""
from types import SimpleNamespace
import torch
from torch import nn
from diffsynth.models.wan_video_dit import WanModel
from wan22_causal_attention import install_causal_attention
from wan22_sparse_causal_ar import StreamingConditionalHead,SparseCoordinateHistory,pack_coordinate_groups
from wan22_sparse_long477 import LongSparseModule,chunk_raw_span,feedback_groups,clean_teacher_forward
from wan22_ar_self_forcing import detach_kv_cache
from wan22_opsd_prefix_cache import update_ema_
from wan22_sparse_task_router import TASK_PROMPTS

spans=[chunk_raw_span(o) for o in range(0,120,3)]
assert spans[0]==(0,9) and spans[-1]==(465,477)
assert [i for s,e in spans for i in range(s,e)]==list(range(477))

class Router(nn.Module):
 def __init__(self): super().__init__(); self.table=nn.Embedding(2,2)
 def forward(self,tasks): return self.table(tasks)
 def identity_regularization(self,gates): return (gates-1).square().mean()
 def call_with_error_cleanup(self,fn): return fn()
 def clear(self): pass

torch.manual_seed(12)
model=WanModel(dim=24,in_dim=4,ffn_dim=48,out_dim=4,text_dim=12,freq_dim=8,
 eps=1e-6,patch_size=(1,2,2),num_heads=2,num_layers=2,has_image_input=False,
 require_vae_embedding=False,require_clip_embedding=False).to(dtype=torch.bfloat16)
causal=install_causal_attention(model,latent_shape=(4,18,30,52),frames_per_chunk=3)
module=LongSparseModule.__new__(LongSparseModule); nn.Module.__init__(module)
module.pipe=nn.Module(); module.pipe.dit=model
module.stage='ar'; module.sparse_head=StreamingConditionalHead(dim=24,hidden=32)
module.task_router=Router(); module.context=torch.randn(2,5,12,dtype=torch.bfloat16)
module.features=None; module.causal_controller=causal
model.head.register_forward_pre_hook(module.capture_features)
module.coordinate_history=SparseCoordinateHistory(model)
source=torch.randn(1,4,18,30,52,dtype=torch.bfloat16)
for phase in ('self_forcing','opsd_local'):
 module.configure_long(phase,18); module.train(); model.train()
 for task,prompt in enumerate(TASK_PROMPTS):
  caches=module.causal_controller.new_kv_cache(); history=source.new_zeros(1,3,4,24,3)
  teacher_cache=module.causal_controller.new_kv_cache()
  target=torch.randn(1,69,24,2 if task==0 else 3)
  gt=pack_coordinate_groups(target,torch.ones(1,69,24) if task==0 else None)
  named=[(n,p) for n,p in module.named_parameters() if p.requires_grad]
  ema={n:p.detach().clone() for n,p in named}
  before={n:p.detach().clone() for n,p in named}
  for offset in range(0,18,3):
   module.zero_grad()
   with torch.autocast('cpu',dtype=torch.bfloat16):
    if phase=='opsd_local':
     teacher=clean_teacher_forward(module,source,history,prompt,offset,teacher_cache,gt,ema,named)
     assert not teacher.requires_grad
     for n,p in named: torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    pred,groups,reg=module(source[:,:,offset:offset+3],history,prompt,offset,caches)
    loss=pred.square().mean()+.001*reg
    if phase=='opsd_local': loss=loss+(pred-teacher).square().mean()
   assert torch.isfinite(loss)
   loss.backward()
   assert model.lora_sparse_history.weight.grad is not None
   assert all(p.grad is None or torch.isfinite(p.grad).all() for _,p in named)
   history=feedback_groups(groups,task,offset); detach_kv_cache(caches)
   assert not history.requires_grad and all(not c.key.requires_grad for c in caches)
   if phase=='opsd_local':
    assert all(c.positions.numel()<=15*390 for c in caches)
    assert all(c.seen_tokens==(offset+3)*390 for c in teacher_cache)
  update_ema_(ema,named,.9999)
  module.eval(); model.eval()
  with torch.autocast('cpu',dtype=torch.bfloat16): result=module.rollout(source,prompt)
  assert result.shape==target.shape and torch.isfinite(result).all()
  module.train(); model.train()
print('PASS: 477-frame exact coverage, both sparse tasks, TBPTT stopgrad, bounded positions/eviction, delayed clean EMA replay, weight restoration, free-running rollout')
