from __future__ import annotations

from collections import Counter

from wan22_dense6_task_sampler import SixTaskStratifiedDistributedSampler


def test_four_gpu_dense6_schedule_is_exactly_balanced() -> None:
    num_clips = 10
    sequences = [
        list(
            SixTaskStratifiedDistributedSampler(
                num_clips, num_replicas=4, rank=rank, seed=17
            )
        )
        for rank in range(4)
    ]
    assert all(len(sequence) == 15 for sequence in sequences)
    flattened = []
    for step in range(len(sequences[0])):
        flattened.extend(sequences[rank][step] for rank in range(4))
        if (step + 1) % 3 == 0:
            task_window = [
                sequences[rank][position] % 6
                for position in range(step - 2, step + 1)
                for rank in range(4)
            ]
            assert Counter(task_window) == Counter({task_id: 2 for task_id in range(6)})
    assert sorted(flattened) == list(range(num_clips * 6))


def test_single_gpu_dense6_schedule_covers_all_tasks() -> None:
    sampler = SixTaskStratifiedDistributedSampler(
        3, num_replicas=1, rank=0, seed=23
    )
    values = list(sampler)
    assert len(values) == 18
    assert Counter(value % 6 for value in values) == Counter(
        {task_id: 3 for task_id in range(6)}
    )
