"""Deterministic step-stratified sampling for four-task DDP training."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class FourTaskStratifiedDistributedSampler(Sampler[int]):
    """Give every global step exactly one sample from each of four tasks.

    Dataset indices must be laid out as ``clip_index * 4 + task_id``.  Task to
    rank assignment rotates every step, so no rank is permanently tied to a
    modality. Each task visits every clip exactly once per epoch.
    """

    def __init__(
        self,
        num_clips: int,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if num_replicas != 4:
            raise ValueError(f"Four-task stratification requires 4 replicas, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid rank {rank} for {num_replicas} replicas")
        if num_clips <= 0:
            raise ValueError("num_clips must be positive")
        self.num_clips = int(num_clips)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_clips

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        clip_permutations = [
            torch.randperm(self.num_clips, generator=generator).tolist()
            for _ in range(self.num_replicas)
        ]
        for step_in_epoch in range(self.num_clips):
            task_id = (step_in_epoch + self.rank + self.epoch) % self.num_replicas
            clip_index = clip_permutations[task_id][step_in_epoch]
            yield clip_index * self.num_replicas + task_id
