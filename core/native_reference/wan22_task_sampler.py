"""Deterministic task-balanced sampling for Wan2.2 Dense4 training and smoke tests."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class FourTaskStratifiedDistributedSampler(Sampler[int]):
    """Use all four tasks per global step on 4 GPUs, or sequentially on one smoke GPU."""

    def __init__(self, num_clips: int, *, num_replicas: int, rank: int, seed: int) -> None:
        if num_replicas not in (1, 4):
            raise ValueError(f"Dense4 stratification supports 1 (smoke) or 4 replicas, got {num_replicas}")
        if not 0 <= rank < num_replicas or num_clips <= 0:
            raise ValueError("Invalid sampler rank or clip count")
        self.num_clips = int(num_clips)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_clips if self.num_replicas == 4 else self.num_clips * 4

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutations = [torch.randperm(self.num_clips, generator=generator).tolist() for _ in range(4)]
        if self.num_replicas == 1:
            for step in range(self.num_clips * 4):
                task_id = (step + self.epoch) % 4
                clip_index = permutations[task_id][step // 4]
                yield clip_index * 4 + task_id
            return
        for step in range(self.num_clips):
            task_id = (step + self.rank + self.epoch) % 4
            clip_index = permutations[task_id][step]
            yield clip_index * 4 + task_id


class FourTaskPoolDistributedSampler(Sampler[int]):
    """Balance four heterogeneous task pools at every global optimizer step.

    Depth/normal may contain additional datasets while matte/raymap retain the
    original pool.  The longest pool defines an epoch; shorter pools reshuffle
    and cycle, so every four-rank step still contains exactly one sample from
    each task.  This preserves DDP gradient semantics and prevents a new domain
    from silently changing the task mixture.
    """

    def __init__(
        self,
        task_pools: dict[int, list[int] | tuple[int, ...]],
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if num_replicas not in (1, 4):
            raise ValueError(f"Dense4 stratification supports 1 or 4 replicas, got {num_replicas}")
        if not 0 <= rank < num_replicas or set(task_pools) != set(range(4)):
            raise ValueError("Invalid sampler rank or task pool IDs")
        pools = tuple(tuple(int(index) for index in task_pools[task]) for task in range(4))
        if any(not pool for pool in pools):
            raise ValueError("Every task pool must be non-empty")
        flattened = [index for pool in pools for index in pool]
        if len(flattened) != len(set(flattened)) or any(index < 0 for index in flattened):
            raise ValueError("Task pool dataset indices must be unique and non-negative")
        self.task_pools = pools
        self.steps_per_epoch = max(len(pool) for pool in pools)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch if self.num_replicas == 4 else self.steps_per_epoch * 4

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample(self, task_id: int, task_step: int) -> int:
        pool = self.task_pools[task_id]
        cycle, position = divmod(task_step, len(pool))
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * 1_000_003 + task_id * 10_007 + cycle
        )
        permutation = torch.randperm(len(pool), generator=generator)
        return pool[int(permutation[position])]

    def __iter__(self):
        if self.num_replicas == 4:
            for step in range(self.steps_per_epoch):
                task_id = (step + self.rank + self.epoch) % 4
                yield self._sample(task_id, step)
            return
        task_steps = [0, 0, 0, 0]
        for slot in range(self.steps_per_epoch * 4):
            task_id = (slot + self.epoch) % 4
            yield self._sample(task_id, task_steps[task_id])
            task_steps[task_id] += 1


class ProportionalTaskPoolDistributedSampler(Sampler[int]):
    """Use every heterogeneous-pool sample once per epoch, with minimal DDP padding."""

    def __init__(
        self,
        task_pools: dict[int, list[int] | tuple[int, ...]],
        *,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        if num_replicas not in (1, 4) or not 0 <= rank < num_replicas:
            raise ValueError("Proportional Dense4 sampling supports one or four valid ranks")
        if set(task_pools) != set(range(4)) or any(not task_pools[task] for task in range(4)):
            raise ValueError("All four task pools must be non-empty")
        flattened = tuple(int(index) for task in range(4) for index in task_pools[task])
        if len(flattened) != len(set(flattened)) or any(index < 0 for index in flattened):
            raise ValueError("Task pool dataset indices must be unique and non-negative")
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
