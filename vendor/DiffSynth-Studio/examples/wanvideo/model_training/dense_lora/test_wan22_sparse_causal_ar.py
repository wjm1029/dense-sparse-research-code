from types import SimpleNamespace
import torch
from torch import nn
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video import model_fn_wan_video
from train_wan22_sparse_prompt_full_ddp import ConditionalHead
from wan22_sparse_causal_ar import StreamingConditionalHead,pack_coordinate_groups,shifted_coordinate_history,SparseCoordinateHistory,CausalARSparseModule
from wan22_sparse_task_router import TASK_PROMPTS
from wan22_causal_attention import install_causal_attention

torch.manual_seed(10)
old=ConditionalHead(dim=24,hidden=32).eval(); new=StreamingConditionalHead(dim=24,hidden=32).eval()
new.load_state_dict(old.state_dict(),strict=True)
features=torch.randn(1,21*390,24)
for task in (0,1):
    torch.testing.assert_close(old(features,task),new(features,task),rtol=0,atol=0)
    streamed=torch.cat([new(features[:,start*390:(start+3)*390],task,start) for start in range(0,21,3)],1)
    torch.testing.assert_close(streamed,new(features,task),rtol=1e-5,atol=1e-6)
coords=torch.randn(1,81,24,2); mask=torch.randint(0,2,(1,81,24)).bool()
history=shifted_coordinate_history(coords,mask)
assert torch.count_nonzero(history[:,:3])==0
changed=coords.clone(); changed[:,9:]+=100
torch.testing.assert_close(history[:,:6],shifted_coordinate_history(changed,mask)[:,:6],rtol=0,atol=0)
groups=pack_coordinate_groups(coords,mask)
assert torch.count_nonzero(groups[:,0,1:])==0
assert torch.count_nonzero(groups[...,:2][groups[...,2]==0])==0

class Router(nn.Module):
    def __init__(self): super().__init__(); self.table=nn.Embedding(2,2)
    def forward(self,tasks): return self.table(tasks)
    def identity_regularization(self,gates): return (gates-1).square().mean()
    def call_with_error_cleanup(self,fn): return fn()

model=WanModel(dim=24,in_dim=4,ffn_dim=48,out_dim=4,text_dim=12,freq_dim=8,
    eps=1e-6,patch_size=(1,2,2),num_heads=2,num_layers=2,has_image_input=False,
    require_vae_embedding=False,require_clip_embedding=False).to(dtype=torch.bfloat16)
causal=install_causal_attention(model,latent_shape=(4,6,30,52),frames_per_chunk=3)
module=CausalARSparseModule.__new__(CausalARSparseModule); nn.Module.__init__(module)
module.pipe=SimpleNamespace(dit=model,model_fn=model_fn_wan_video)
module.stage='ar'; module.sparse_head=StreamingConditionalHead(dim=24,hidden=32)
module.task_router=Router(); module.context=torch.randn(2,5,12,dtype=torch.bfloat16)
module.features=None; module.causal_controller=causal
model.head.register_forward_pre_hook(module.capture_features)
module.coordinate_history=SparseCoordinateHistory(model)
rgb=torch.randn(1,4,6,30,52,dtype=torch.bfloat16)
module.coordinate_history.active=torch.randn(1,6,4,24,3)
with torch.no_grad():
    torch.testing.assert_close(model.patchify(rgb),module.coordinate_history.original_patchify(rgb),rtol=0,atol=0)
module.coordinate_history.active=None
for task,prompt in enumerate(TASK_PROMPTS):
    target=torch.randn(1,21,24,2 if task==0 else 3)
    module.train(); model.train(); model.zero_grad()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        pred,reg=module(rgb,prompt,True,target,torch.ones(1,21,24))
    assert pred.shape==target.shape and torch.isfinite(pred).all()
    pred.square().mean().backward()
    assert model.lora_sparse_history.weight.grad.abs().max()>0
    module.eval(); model.eval()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        free,_=module(rgb,prompt,False)
    torch.testing.assert_close(pred.detach(),free,rtol=0.08,atol=0.015)
    assert module.coordinate_history.active is None and module.features is None
print('PASS: inherited head exact parity, chunk head parity, masked history/no GT leakage, zero initialization, both task gradients and free-running cached AR')
