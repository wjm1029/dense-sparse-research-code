"""Same task-embedding/HyperRouter mechanism as dense, with two sparse tasks."""
import torch
from wan22_dense6_task_router import TaskHyperRouter as DenseRouter, discover_lora_sites

TASK_PROMPTS = ('output: 2D keypoints','output: 3D keypoints')

class SparseTaskRouter(DenseRouter):
    def __init__(self, site_names, rank=32):
        super().__init__(site_names,rank=rank)
        self.task_feature_table = torch.zeros(2,self.task_feature_dim)
    def set_task_feature_table(self,features):
        assert tuple(features.shape) == (2,self.task_feature_dim) and torch.isfinite(features).all()
        self.task_feature_table.copy_(features.detach().to(self.task_feature_table))
        self.features_ready.fill_(True)
    def forward(self,task_ids):
        if torch.any((task_ids < 0)|(task_ids > 1)): raise ValueError('Only sparse 2D/3D task IDs are valid')
        return super().forward(task_ids)
    @torch.no_grad()
    def routing_summary(self):
        logits = self.layernets(self.hypernet(self.task_feature_table)).reshape(2,len(self.site_names),self.rank)
        gates = 1+0.5*torch.tanh(logits.float())
        return {'gate_min':float(gates.min()),'gate_max':float(gates.max()),'task_pair_mean_abs_diff':float((gates[0]-gates[1]).abs().mean())}

