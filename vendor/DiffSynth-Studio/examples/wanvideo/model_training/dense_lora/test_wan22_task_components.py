"""CPU contract tests for the Wan2.2 Dense4 task router and samplers."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from wan22_task_router import TASK_INSTRUCTIONS, TASK_TO_ID, TaskHyperRouter
from wan22_task_sampler import (
    FourTaskPoolDistributedSampler,
    FourTaskStratifiedDistributedSampler,
    ProportionalTaskPoolDistributedSampler,
)
from wan22_masked_loss import masked_latent_mse


class FakeLoRASite(nn.Module):
    def __init__(self, input_dim: int, rank: int) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(input_dim, rank, bias=False)})

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.lora_A["default"](value)


class FakeDiT(nn.Module):
    def __init__(self, count: int, input_dim: int, rank: int) -> None:
        super().__init__()
        self.sites = nn.ModuleList([FakeLoRASite(input_dim, rank) for _ in range(count)])


def main() -> None:
    assert set(TASK_TO_ID) == {"depth", "normal", "matte", "raymap"}
    assert TASK_INSTRUCTIONS == {
        0: "output: depth",
        1: "output: surface normal",
        2: "output: alpha matte",
        3: "output: camera raymap",
    }

    per_rank = [
        list(FourTaskStratifiedDistributedSampler(7, num_replicas=4, rank=rank, seed=1))
        for rank in range(4)
    ]
    for step in range(7):
        assert {indices[step] % 4 for indices in per_rank} == {0, 1, 2, 3}
    smoke = list(FourTaskStratifiedDistributedSampler(1, num_replicas=1, rank=0, seed=1))
    assert [index % 4 for index in smoke] == [0, 1, 2, 3]
    epoch_sampler = FourTaskStratifiedDistributedSampler(9, num_replicas=4, rank=2, seed=17)
    epoch_sampler.set_epoch(3)
    full_order = list(epoch_sampler)
    resumed_order = list(epoch_sampler)[4:]
    assert resumed_order == full_order[4:]

    pools = {0: [0, 4, 8], 1: [1, 5, 9], 2: [2, 6], 3: [3]}
    pooled_per_rank = [
        list(FourTaskPoolDistributedSampler(pools, num_replicas=4, rank=rank, seed=23))
        for rank in range(4)
    ]
    assert all(len(indices) == 3 for indices in pooled_per_rank)
    for step in range(3):
        assert {pooled_per_rank[rank][step] % 4 for rank in range(4)} == {0, 1, 2, 3}
    pooled = FourTaskPoolDistributedSampler(pools, num_replicas=1, rank=0, seed=23)
    pooled.set_epoch(2)
    assert list(pooled)[5:] == list(pooled)[5:]
    assert {index % 4 for index in pooled} == {0, 1, 2, 3}
    proportional = [
        list(ProportionalTaskPoolDistributedSampler(pools, num_replicas=4, rank=rank, seed=31))
        for rank in range(4)
    ]
    assert all(len(indices) == 3 for indices in proportional)
    proportional_flat = [index for rank_indices in proportional for index in rank_indices]
    assert set(proportional_flat) == {index for values in pools.values() for index in values}
    assert len(proportional_flat) == 12  # ten real samples plus two deterministic DDP pads
    resume_sampler = ProportionalTaskPoolDistributedSampler(pools, num_replicas=4, rank=2, seed=31)
    resume_sampler.set_epoch(5)
    assert list(resume_sampler)[1:] == list(resume_sampler)[1:]

    prediction = torch.tensor([[[[[1.0, 10.0]]], [[[3.0, 20.0]]]]])
    target = torch.zeros_like(prediction)
    ones = torch.ones((1, 1, 1, 1, 2))
    assert torch.equal(masked_latent_mse(prediction, target, ones), F.mse_loss(prediction, target))
    first_only = torch.tensor([[[[[1.0, 0.0]]]]])
    assert torch.equal(masked_latent_mse(prediction, target, first_only), torch.tensor(5.0))
    try:
        masked_latent_mse(prediction, target, torch.zeros_like(first_only))
    except ValueError:
        pass
    else:
        raise AssertionError("All-zero loss mask must fail")

    fake = FakeDiT(count=3, input_dim=8, rank=4)
    router = TaskHyperRouter(
        [f"sites.{index}" for index in range(3)], rank=4, task_feature_dim=16, hidden_dim=8
    )
    router.bind(fake)
    router.set_task_feature_table(torch.randn(4, 16))
    gates = router(torch.tensor([0, 1, 2, 3]))
    assert gates.shape == (4, 3, 4)
    assert torch.equal(gates, torch.ones_like(gates))
    value_hooked = torch.randn(4, 8, requires_grad=True)
    value_plain = value_hooked.detach().clone().requires_grad_(True)
    output = fake.sites[0](value_hooked)
    plain = F.linear(value_plain, fake.sites[0].lora_A["default"].weight)
    assert torch.equal(output, plain)
    output.square().sum().backward()
    hooked_input_grad = value_hooked.grad.clone()
    hooked_weight_grad = fake.sites[0].lora_A["default"].weight.grad.clone()
    fake.sites[0].lora_A["default"].weight.grad = None
    plain.square().sum().backward()
    assert torch.equal(hooked_input_grad, value_plain.grad)
    assert torch.equal(hooked_weight_grad, fake.sites[0].lora_A["default"].weight.grad)
    router.clear()

    clone = TaskHyperRouter(
        [f"sites.{index}" for index in range(3)], rank=4, task_feature_dim=16, hidden_dim=8
    )
    clone.load_state_dict(copy.deepcopy(router.state_dict()))
    assert torch.equal(clone(torch.tensor([0, 1, 2, 3])), gates)
    clone.clear()

    def fail_after_activation():
        router(torch.tensor([0]))
        raise RuntimeError("intentional")

    try:
        router.call_with_error_cleanup(fail_after_activation)
    except RuntimeError as error:
        assert str(error) == "intentional"
    else:
        raise AssertionError("Expected guarded routing call to fail")
    assert router._active_gates is None

    trainer_source = Path(__file__).with_name("train_wan22_dense4_ddp.py").read_text(encoding="utf-8")
    tree = ast.parse(trainer_source)
    predict = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "predict_velocity"
    )
    calls = [node for node in ast.walk(predict) if isinstance(node, ast.Call)]
    called_attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    assert called_attributes.count("model_fn") == 1
    assert not ({"add_noise", "training_target", "step"} & set(called_attributes))
    assert "scheduler" not in ast.unparse(predict)

    forward = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
    )
    forward_text = ast.unparse(forward)
    assert "-velocity" in forward_text and "masked_latent_mse" in forward_text

    production = TaskHyperRouter([f"site.{index}" for index in range(300)], rank=32)
    assert sum(parameter.numel() for parameter in production.parameters()) == 7_293_312
    print("Wan2.2 Dense4 CPU component contracts: PASS")


if __name__ == "__main__":
    main()
