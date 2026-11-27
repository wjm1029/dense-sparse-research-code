"""Exercise the new AR forward for all six task IDs without base-model loading."""
from types import SimpleNamespace
import torch
from torch import nn
import torch.distributed as dist
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video import model_fn_wan_video
from train_wan22_dense6_ar_tf_ddp import ARTeacherForcingDense6LoRAModule
from wan22_causal_attention import install_causal_attention
from wan22_ar_teacher_forcing import install_ar_history_adapter, ARTeacherForcingConfig

class Router(nn.Module):
    def __init__(self):
        super().__init__(); self.table=nn.Embedding(6,2)
    def forward(self,tasks): return self.table(tasks)
    def identity_regularization(self,gates): return (gates-1).square().mean()

dist.init_process_group('gloo')
rank=dist.get_rank()
torch.manual_seed(1)
model=WanModel(dim=24,in_dim=4,ffn_dim=48,out_dim=4,text_dim=12,freq_dim=8,
    eps=1e-6,patch_size=(1,2,2),num_heads=2,num_layers=9,has_image_input=False,
    require_vae_embedding=False,require_clip_embedding=False).to(dtype=torch.bfloat16).train()
config=ARTeacherForcingConfig(num_frames=6,frames_per_chunk=3,latent_channels=4)
causal=install_causal_attention(model,latent_shape=(4,6,4,4),frames_per_chunk=3)
ar=install_ar_history_adapter(model,causal,config)
module=ARTeacherForcingDense6LoRAModule.__new__(ARTeacherForcingDense6LoRAModule)
nn.Module.__init__(module)
module.pipe=SimpleNamespace(dit=model,model_fn=model_fn_wan_video)
module.task_router=Router(); module.router_mode='learned'
module.router_identity_weight=.001; module.context_table=torch.randn(6,5,12,dtype=torch.bfloat16)
module.ar_controller=ar; module.dispersion_weight=.01; module.dispersion_tau=1.
module.probe_count=6
torch.manual_seed(30+rank)
for task in range(6):
    model.zero_grad(); module.task_router.zero_grad()
    source=torch.randn(1,4,6,4,4,dtype=torch.bfloat16); target=torch.randn_like(source)
    tasks=torch.tensor([task]); mask=torch.ones(1,1,6,4,4)
    loss,data,identity=module(source,target,tasks,tasks,mask,True)
    dispersion,pairs,distance=module.dispersion_stats
    assert pairs==6 and dispersion < 0 and torch.isfinite(loss)
    torch.testing.assert_close(loss.detach(),data+.001*identity+.01*dispersion)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ar.history_parameters)
    assert any(p.grad.abs().max()>0 for p in ar.history_parameters)
if rank==0: print('PASS: all six AR tasks retain DispLoss, checkpointed ninth-block graph, finite backward and history gradients',flush=True)
dist.destroy_process_group()
