"""把新 JTA decoder/router 接到已有 Wan 双向主干；所有适配器重新初始化。"""
import gc
from pathlib import Path
import sys
import torch
from task_gating import JTATaskRouter, SPARSE_PROMPTS, discover_lora_sites, text_feature_from_context
from sparse_decode import MultiPersonDecode
from sparse_loss import sparse_objective, LossConfig


def build_fresh_jta_model(model_dir, device='cuda', queries=128):
    """依赖项目中匹配的 DiffSynth 版本；只加载官方 Wan 基座，绝不加载旧 Sparse。

    延迟 import 让 head/loss/router 的 CPU 测试不需要安装完整 DiffSynth。
    旧 SparseModule 在这里仅复用官方模型构造、LoRA 初始化与 feature hook；
    其旧 24-joint head 立即被全新多人 head 替换，旧权重没有任何加载入口。
    """
    reference=str(Path(__file__).resolve().parent/'native_reference')
    if reference not in sys.path:sys.path.insert(0,reference)
    from train_wan22_sparse_ddp import SparseModule
    from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder

    class FreshJTAModule(SparseModule):
        def __init__(self):
            super().__init__(model_dir,rank=32)
            self.sparse_head=MultiPersonDecode(queries=queries)
            sites=discover_lora_sites(self.pipe.dit)
            if len(sites)!=300:raise ValueError('Expected 300 Wan2.2 LoRA sites')
            self.task_router=JTATaskRouter([n for n,_ in sites])
            self.task_router.bind(self.pipe.dit)

        @torch.no_grad()
        def prepare(self, device):
            device=torch.device(device);self.pipe.device=device
            self.pipe.text_encoder.to(device)
            embedder=WanVideoUnit_PromptEmbedder()
            contexts=[embedder.encode_prompt(self.pipe,p).detach() for p in SPARSE_PROMPTS]
            self.context=torch.cat(contexts).to(device=device,dtype=torch.bfloat16)
            self.task_router.set_task_feature_table(torch.stack([text_feature_from_context(c) for c in contexts]))
            self.pipe.text_encoder.to('cpu');self.pipe.text_encoder=None;self.pipe.tokenizer=None
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            self.pipe.dit.to(device);self.sparse_head.to(device);self.task_router.to(device)

        def forward(self, source_latent, intrinsic, task, checkpointing=True):
            if task not in (0,1):raise ValueError('task: 0=2D, 1=3D')
            if tuple(source_latent.shape[1:])!=(48,21,30,52):
                raise ValueError('Expected RGB VAE latent [B,48,21,30,52]')
            b=source_latent.shape[0]
            gates=self.task_router(torch.full((b,),task,device=source_latent.device,dtype=torch.long))
            self.task_router.call_with_error_cleanup(lambda:self.pipe.model_fn(
                dit=self.pipe.dit,latents=source_latent,
                timestep=torch.zeros(b,device=source_latent.device,dtype=torch.bfloat16),
                context=self.context[task:task+1].expand(b,-1,-1),
                use_gradient_checkpointing=checkpointing,use_gradient_checkpointing_offload=False,
                cfg_merge=False))
            features=self.features;self.features=None
            if features is None:raise RuntimeError('Missing pre-head DiT features')
            # Sparse 用 dit.head 的输入；与 Dense 像素 decoder 的 hook 位置不同。
            return self.sparse_head(features,intrinsic,task),gates

    model=FreshJTAModule();model.prepare(device)
    return model


def make_optimizer(model):
    """沿用主干/头 1e-4、router 1e-5；router 零初始化末层与偏置不衰减。"""
    groups={k:[] for k in ('lora','head','router_decay','router_nodecay')}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if '.lora_' in name:groups['lora'].append(p)
        elif name.startswith('sparse_head.'):groups['head'].append(p)
        elif name.startswith('task_router.'):
            key='router_nodecay' if name.endswith('.bias') or '.hypernet.0.' in name or '.layernets.' in name else 'router_decay'
            groups[key].append(p)
        else:raise ValueError(f'Unexpected trainable parameter {name}')
    return torch.optim.AdamW([dict(params=params,lr=1e-5 if name.startswith('router') else 1e-4,
                                  weight_decay=0 if name=='router_nodecay' else .01)
                             for name,params in groups.items() if params])


def training_step(model, optimizer, source_latent, targets, task, config=LossConfig()):
    """单步示例。外层可加 DDP/采样/保存；使用 DDP 时 find_unused_parameters=True。

    只完成当前任务的 forward/backward 后才能切换 router，避免 checkpoint 重算
    错用另一任务门控。3D step 也训练辅助 2D，用于匹配；它不是旧 Sparse loss。
    """
    model.train();optimizer.zero_grad(set_to_none=True)
    k=torch.stack([gt['intrinsic'] for gt in targets]).to(source_latent.device)
    with torch.autocast(source_latent.device.type,dtype=torch.bfloat16):
        prediction,gates=model(source_latent,k,task,checkpointing=True)
    loss,metrics,_=sparse_objective(prediction,targets,task,gates,config)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    underlying=getattr(model,'module',model);underlying.task_router.clear()
    return float(loss.detach()),{k:float(v) for k,v in metrics.items()}
