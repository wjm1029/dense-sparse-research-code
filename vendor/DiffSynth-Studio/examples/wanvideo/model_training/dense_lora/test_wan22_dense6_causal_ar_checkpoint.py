"""Small real Dense6 checkpoint roundtrip, including AR-only parameters."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch import nn
from wan22_cache_integrity import sha256_json
from wan22_dense6_checkpoint import REQUIRED_PROVENANCE_KEYS, tensor_sha256, save_dense6_checkpoint, load_dense6_checkpoint
from train_wan22_dense6_ar_tf_ddp import _load_ar_checkpoint, AR_CONFIG
from train_wan22_dense6_dispersion_ddp import optimizer_param_groups

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(6,3))
        self.register_buffer('task_feature_table',torch.arange(18.).reshape(6,3))
        self.register_buffer('features_ready',torch.tensor(True))
        self.site_names=('probe',)
        self._hook_handles=[True]
    def routing_summary(self): return {'test':True}

class Module(nn.Module):
    def __init__(self,ar=False):
        super().__init__()
        self.pipe=nn.Module(); self.pipe.dit=nn.Module()
        self.pipe.dit.lora_probe=nn.Parameter(torch.randn(3,3))
        self.task_router=Router()
        if ar:
            self.pipe.dit.lora_history_patch=nn.Linear(3,3)
            nn.init.zeros_(self.pipe.dit.lora_history_patch.weight)
            nn.init.zeros_(self.pipe.dit.lora_history_patch.bias)
            def assert_zero():
                assert all(torch.count_nonzero(p)==0 for p in self.pipe.dit.lora_history_patch.parameters())
            self.ar_controller=SimpleNamespace(history_parameters=tuple(self.pipe.dit.lora_history_patch.parameters()),assert_zero_initialized=assert_zero)
    def trainable_param_names(self):
        return [n for n,p in self.named_parameters() if p.requires_grad]

base=Module()
provenance={key:'test' for key in REQUIRED_PROVENANCE_KEYS}
prompts=['output: depth','output: optical flow']
provenance.update(output_prompt_texts=json.dumps(prompts),output_prompt_texts_sha256=sha256_json(prompts),
                  task_feature_table_sha256=tensor_sha256(base.task_router.task_feature_table))
load_kwargs=dict(expected_provenance=provenance,lora_rank=32,lora_targets='q,k,v,o,ffn.0,ffn.2',router_hidden_dim=512,expected_router_mode='learned')
save_kwargs=dict(step=45000,world_size=4,lora_rank=32,lora_targets='q,k,v,o,ffn.0,ffn.2',router_hidden_dim=512,router_mode='learned',provenance_metadata=provenance)
with tempfile.TemporaryDirectory(prefix='dense6-ar-checkpoint-') as tmp:
    causal=Path(tmp)/'causal.safetensors'
    save_dense6_checkpoint(base,causal,extra_metadata={'causal_attention':'block_prefix_v1'},**save_kwargs)
    ar=Module(ar=True)
    _load_ar_checkpoint(ar,causal,**load_kwargs)
    torch.testing.assert_close(ar.pipe.dit.lora_probe,base.pipe.dit.lora_probe)
    torch.testing.assert_close(ar.task_router.weight,base.task_router.weight)
    ar.ar_controller.assert_zero_initialized()
    args=SimpleNamespace(learning_rate=1e-4,weight_decay=.01,router_learning_rate=1e-5,router_weight_decay=.01)
    groups=optimizer_param_groups(ar,args)
    assert {id(p) for p in ar.ar_controller.history_parameters} <= {id(p) for g in groups for p in g['params']}
    with torch.no_grad(): ar.pipe.dit.lora_history_patch.weight.fill_(.25)
    checkpoint=Path(tmp)/'ar.safetensors'
    save_dense6_checkpoint(ar,checkpoint,extra_metadata=AR_CONFIG.metadata(),**save_kwargs)
    reloaded=Module(ar=True)
    _load_ar_checkpoint(reloaded,checkpoint,**load_kwargs)
    for key,value in ar.state_dict().items(): torch.testing.assert_close(reloaded.state_dict()[key],value)
print('PASS: six-task router inherited, AR zero-init transition, history optimizer group and exact checkpoint roundtrip')
