"""Four-rank tiny-model coverage of six-task streaming DispLoss and TBPTT."""
from types import SimpleNamespace
import torch
from torch import nn
import torch.distributed as dist
from diffsynth.models.wan_video_dit import WanModel
from wan22_dense6_long477 import LongDenseModule
from wan22_causal_attention import install_causal_attention
from wan22_ar_teacher_forcing import ARTeacherForcingConfig,install_ar_history_adapter
from wan22_ar_self_forcing import detach_kv_cache
from wan22_dense6_dispersion import SameTaskSampler

class Router(nn.Module):
    def __init__(self): super().__init__(); self.table=nn.Embedding(6,2)
    def forward(self,t): return self.table(t)
    def identity_regularization(self,g): return (g-1).square().mean()
    def clear(self): pass

dist.init_process_group('gloo'); rank=dist.get_rank()
torch.manual_seed(1)
model=WanModel(dim=24,in_dim=4,ffn_dim=48,out_dim=4,text_dim=12,freq_dim=8,
    eps=1e-6,patch_size=(1,2,2),num_heads=2,num_layers=9,has_image_input=False,
    require_vae_embedding=False,require_clip_embedding=False).to(dtype=torch.bfloat16).train()
cfg=ARTeacherForcingConfig(6,3,4)
controller=install_causal_attention(model,latent_shape=(4,6,4,4),frames_per_chunk=3)
ar=install_ar_history_adapter(model,controller,cfg)
module=LongDenseModule.__new__(LongDenseModule); nn.Module.__init__(module)
module.pipe=SimpleNamespace(dit=model); module.task_router=Router()
module.context_table=torch.randn(6,5,12,dtype=torch.bfloat16)
module.ar_config=cfg; module.ar_controller=ar; module.causal_controller=controller
torch.manual_seed(30+rank)
for task in range(6):
    model.zero_grad(); module.task_router.zero_grad()
    rgb=torch.randn(1,4,6,4,4,dtype=torch.bfloat16); target=torch.randn_like(rgb)
    ids=torch.tensor([task]); caches=controller.new_kv_cache(); history=torch.zeros_like(rgb[:,:,:3])
    for offset in (0,3):
        pred,reg,disp=module(rgb[:,:,offset:offset+3],history,ids,ids,offset,caches)
        assert torch.isfinite(disp) and disp<0
        loss=((pred.float()-target[:,:,offset:offset+3].float()).square().mean()+.001*reg+.01*disp)/2
        loss.backward(); history=pred.detach(); detach_kv_cache(caches)
        assert not history.requires_grad and all(c.key.grad_fn is None for c in caches)
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in ar.history_parameters)
    assert any(q.grad.abs().max()>0 for q in ar.history_parameters)
for epoch in (0,1):
    sampler=SameTaskSampler(11,num_replicas=4,rank=rank,seed=7); sampler.set_epoch(epoch)
    local=list(sampler); gathered=[None]*4; dist.all_gather_object(gathered,local)
    for indices in zip(*gathered):
        assert len({i//6 for i in indices})==4 and len({i%6 for i in indices})==1
if rank==0: print('PASS Dense6 long: all6 tasks, two-chunk TBPTT, feature gradients, history gradients, distinct same-task DDP sampler',flush=True)
dist.destroy_process_group()
