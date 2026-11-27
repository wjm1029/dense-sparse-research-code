"""Deterministic samplers for six-task Wan2.2 Dense6 training."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


NUM_TASKS = 6


class SixTaskStratifiedDistributedSampler(Sampler[int]):
    """Use every clip/task pair once, balanced over each three 4-GPU steps."""

    def __init__(self, num_clips: int, *, num_replicas: int, rank: int, seed: int) -> None:
        if num_replicas not in (1, 4):
            raise ValueError(f"Dense6 stratification supports 1 or 4 replicas, got {num_replicas}")
        if not 0 <= rank < num_replicas or num_clips <= 0:
            raise ValueError("Invalid sampler rank or clip count")
        total = int(num_clips) * NUM_TASKS
        if total % num_replicas:
            raise ValueError("Dense6 clip/task count must divide evenly across replicas")
        self.num_clips = int(num_clips)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_clips * NUM_TASKS // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutations = [
            torch.randperm(self.num_clips, generator=generator).tolist()
            for _ in range(NUM_TASKS)
        ]
        task_order = [
            (task_id + self.epoch) % NUM_TASKS for task_id in range(NUM_TASKS)
        ]
        global_indices = [
            permutations[task_id][clip_position] * NUM_TASKS + task_id
            for clip_position in range(self.num_clips)
            for task_id in task_order
        ]
        yield from global_indices[self.rank :: self.num_replicas]


class ProportionalTaskPoolDistributedSampler(Sampler[int]):
    """Use every heterogeneous six-task pool sample once with minimal padding."""

    def __init__(
        self,
        task_pools: dict[int, list[int] | tuple[int, ...]],
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if num_replicas not in (1, 4) or not 0 <= rank < num_replicas:
            raise ValueError("Proportional Dense6 sampling supports one or four valid ranks")
        if set(task_pools) != set(range(NUM_TASKS)):
            raise ValueError("Dense6 task pools must have IDs 0..5")
        if any(not task_pools[task] for task in range(NUM_TASKS)):
            raise ValueError("All six task pools must be non-empty")
        flattened = tuple(
            int(index) for task in range(NUM_TASKS) for index in task_pools[task]
        )
        if len(flattened) != len(set(flattened)) or any(index < 0 for index in flattened):
            raise ValueError("Task-pool dataset indices must be unique and non-negative")
        self.indices = flattened
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = (len(flattened) + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.indices), generator=generator).tolist()
        shuffled = [self.indices[position] for position in order]
        shuffled.extend(shuffled[: self.total_size - len(shuffled)])
        yield from shuffled[self.rank : self.total_size : self.num_replicas]
