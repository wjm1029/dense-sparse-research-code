"""DispLoss mean-squared feature distance; differentiable global batch."""
import math
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.utils.data import Sampler


class SameTaskSampler(Sampler):
    def __init__(self, num_clips, *, num_replicas, rank, seed):
        if num_clips < num_replicas:
            raise ValueError('Need distinct clips in each global batch')
        self.num_clips, self.world, self.rank, self.seed = num_clips, num_replicas, rank, seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(self.num_clips / self.world) * 6

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        perms = [torch.randperm(self.num_clips, generator=gen).tolist() for _ in range(6)]
        for group in range(math.ceil(self.num_clips / self.world)):
            for t in range(6):
                task = (t + self.epoch) % 6
                position = (group * self.world + self.rank) % self.num_clips
                yield perms[task][position] * 6 + task


def feature_dispersion(features, prompt_ids, tau=1.0):
    # Full spatiotemporal feature, NOT a spatial or temporal pooling approximation.
    # all_gather backward sums rank contributions; subsequent DDP averaging yields
    # the exact gradient of one global-batch objective (no additional world factor).
    z = torch.cat(all_gather(features.contiguous()), dim=0).flatten(1).float()
    prompts = [torch.empty_like(prompt_ids) for _ in range(dist.get_world_size())]
    dist.all_gather(prompts, prompt_ids)
    prompts = torch.cat(prompts)
    logits = []
    distances = []
    active = 0
    for i in range(z.shape[0]):
        logits.append(z[i].sum() * 0)  # include diagonal, connected zero
        for j in range(i + 1, z.shape[0]):
            if int(prompts[i]) != int(prompts[j]):
                continue  # avoid spreading features merely because prompts differ
            d = (z[i] - z[j]).square().mean()
            distances.append(d.detach())
            logits.extend([-d / tau, -d / tau])
            active += 1
    stacked = torch.stack(logits)
    loss = torch.logsumexp(stacked, dim=0) - math.log(len(logits))
    mean_distance = torch.stack(distances).mean() if distances else z.new_zeros(())
    return loss, active, mean_distance
