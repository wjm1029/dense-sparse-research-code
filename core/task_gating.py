"""任务 embedding -> LoRA rank 门控。Dense 原实现保留；Sparse 为独立双任务 router。

每个 LoRA 层: y = W_frozen(x) + scale * B(g_task * A(dropout(x))).
门控只作用于 rank 通道，不改变冻结的 W；不是在多个 LoRA 专家之间做 softmax。
"""
import torch
from native_reference.wan22_dense6_task_router import TaskHyperRouter, discover_lora_sites

DENSE_TASKS = ('depth', 'normal', 'matte', 'raymap', 'albedo', 'optical_flow')
SPARSE_PROMPTS = ('output: 2D keypoints', 'output: 3D keypoints')


class JTATaskRouter(TaskHyperRouter):
    """复用现有门控参数结构；新建 2×4096 的冻结文本特征表。"""
    def __init__(self, site_names, rank=32, task_feature_dim=4096, hidden_dim=512):
        super().__init__(site_names, rank=rank, task_feature_dim=task_feature_dim,
                         hidden_dim=hidden_dim)
        self.task_feature_table = torch.zeros(2, task_feature_dim)

    def set_task_feature_table(self, features):
        if features.shape != self.task_feature_table.shape or not torch.isfinite(features).all():
            raise ValueError('Expected finite [2, task_feature_dim] task embeddings')
        self.task_feature_table.copy_(features.detach().to(self.task_feature_table))
        self.features_ready.fill_(True)

    def forward(self, task_ids):
        if task_ids.ndim != 1 or task_ids.dtype != torch.long:
            raise ValueError('task_ids must be int64 [B]')
        if ((task_ids < 0) | (task_ids > 1)).any():
            raise ValueError('JTA task IDs: 0=2D, 1=3D')
        return super().forward(task_ids)

    @torch.no_grad()
    def routing_summary(self):
        logits = self.layernets(self.hypernet(self.task_feature_table)).view(2, -1, self.rank)
        gates = 1 + .5 * logits.float().tanh()
        return dict(gate_min=float(gates.min()), gate_max=float(gates.max()),
                    task_pair_mean_abs_diff=float((gates[0]-gates[1]).abs().mean()))


def text_feature_from_context(context):
    """与现有代码相同：冻结 UMT5 输出的第 0 个 token；不是 mean pooling。"""
    if context.ndim != 3 or context.shape[0] != 1:
        raise ValueError('Expected one prompt context [1,L,4096]')
    return context[0, 0].detach().float()

# 门控公式由原类实现：
# logits = layernets(MLP(LayerNorm(task_feature)))  -> [B,300,32]
# gates  = 1 + 0.5*tanh(logits), 初值恰好为 1，范围 [0.5,1.5]
# router_regularization = mean((gates-1)**2)
# LoRA A 随机、B 为零：初始残差为零；不能同时把 A/B 清零。
# gradient checkpointing 会在 backward 重跑 LoRA A：在 backward 完成前不要
# clear router 或激活其他任务。每个任务必须 forward -> loss.backward() 后再切换。
