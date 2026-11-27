"""Six-task UniControl-inspired task HyperNet for a single shared Wan LoRA.

UniControl modulates the trainable zero-convolution residual branch with a
task-instruction embedding.  Wan LoRA has no zero-conv branch, so the exact
analogue is the LoRA bottleneck: the frozen Wan projection is untouched while
the HyperNet mixes the rank channels of every trainable LoRA residual site.
"""

from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Callable
from typing import TypeVar

import torch
from torch import nn


TASK_TO_ID = {
    "depth": 0,
    "normal": 1,
    "matte": 2,
    "raymap": 3,
    "albedo": 4,
    "optical_flow": 5,
}
ID_TO_TASK = {value: key for key, value in TASK_TO_ID.items()}
TASK_INSTRUCTIONS = {
    0: "output: depth",
    1: "output: surface normal",
    2: "output: alpha matte",
    3: "output: camera raymap",
    4: "output: albedo",
    5: "output: optical flow",
}

T = TypeVar("T")


def discover_lora_sites(dit: nn.Module, adapter_name: str = "default") -> list[tuple[str, nn.Module]]:
    """Return PEFT LoRA-A modules in stable name order."""
    sites = []
    for name, module in dit.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is not None and adapter_name in lora_a:
            sites.append((name, lora_a[adapter_name]))
    sites.sort(key=lambda item: item[0])
    if not sites:
        raise RuntimeError("No PEFT LoRA sites found in Wan DiT")
    return sites


class TaskHyperRouter(nn.Module):
    """Map frozen task-instruction features to per-site LoRA rank mixtures.

    The final layer is zero initialized. ``1 + 0.5 * tanh(logits)`` therefore
    starts at an all-ones gate, exactly preserving vanilla LoRA behavior while
    keeping every rank channel in the conservative interval [0.5, 1.5].
    """

    def __init__(
        self,
        site_names: Sequence[str],
        *,
        rank: int,
        task_feature_dim: int = 4096,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if len(set(site_names)) != len(site_names) or not site_names:
            raise ValueError("LoRA site names must be non-empty and unique")
        self.site_names = tuple(site_names)
        self.rank = int(rank)
        self.task_feature_dim = int(task_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.hypernet = nn.Sequential(
            nn.LayerNorm(self.task_feature_dim),
            nn.Linear(self.task_feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        # This single matrix is equivalent to UniControl's parallel layernets:
        # one independent rank-vector head for each modulated residual site.
        self.layernets = nn.Linear(self.hidden_dim, len(self.site_names) * self.rank)
        nn.init.zeros_(self.layernets.weight)
        nn.init.zeros_(self.layernets.bias)
        self.register_buffer(
            "task_feature_table",
            torch.zeros(len(TASK_TO_ID), self.task_feature_dim),
            persistent=True,
        )
        # This must be persistent: an otherwise complete checkpoint reload used
        # to fail its first forward because the old Python bool was not restored.
        self.register_buffer("features_ready", torch.tensor(False), persistent=True)
        self._active_gates: torch.Tensor | None = None
        self._hook_handles: list = []

    def set_task_feature_table(self, features: torch.Tensor) -> None:
        expected = (len(TASK_TO_ID), self.task_feature_dim)
        if tuple(features.shape) != expected or not torch.isfinite(features).all():
            raise ValueError(f"Expected finite task feature table {expected}, got {tuple(features.shape)}")
        self.task_feature_table.copy_(features.detach().to(self.task_feature_table))
        self.features_ready.fill_(True)

    def bind(self, dit: nn.Module, adapter_name: str = "default") -> None:
        if self._hook_handles:
            raise RuntimeError("TaskHyperRouter is already bound")
        sites = discover_lora_sites(dit, adapter_name)
        names = tuple(name for name, _ in sites)
        if names != self.site_names:
            raise ValueError("LoRA site ordering differs from checkpoint router schema")

        for site_index, (_, lora_a) in enumerate(sites):
            def route_rank_channels(_module, _inputs, output, index=site_index):
                gates = self._active_gates
                if gates is None:
                    raise RuntimeError("LoRA route used without an active task")
                gate = gates[:, index, :]
                while gate.ndim < output.ndim:
                    gate = gate.unsqueeze(1)
                return output * gate.to(dtype=output.dtype)

            self._hook_handles.append(lora_a.register_forward_hook(route_rank_channels))

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        if not bool(self.features_ready.item()):
            raise RuntimeError("Task instruction features have not been initialized")
        if task_ids.ndim != 1:
            raise ValueError(f"task_ids must be [B], got {tuple(task_ids.shape)}")
        if torch.any((task_ids < 0) | (task_ids >= len(TASK_TO_ID))):
            raise ValueError(f"Invalid task ids: {task_ids.tolist()}")
        features = self.task_feature_table[task_ids]
        logits = self.layernets(self.hypernet(features))
        logits = logits.view(task_ids.shape[0], len(self.site_names), self.rank)
        gates = (1.0 + 0.5 * torch.tanh(logits.float())).to(logits.dtype)
        self._active_gates = gates
        return gates

    def clear(self) -> None:
        self._active_gates = None

    def call_with_error_cleanup(self, call: Callable[[], T]) -> T:
        """Preserve gates for checkpoint backward, but clear them on forward failure."""
        try:
            return call()
        except Exception:
            self.clear()
            raise

    def activate_identity(self, task_ids: torch.Tensor) -> torch.Tensor:
        """Activate vanilla shared-LoRA routing for the required g=1 ablation."""
        gates = torch.ones(
            task_ids.shape[0], len(self.site_names), self.rank,
            device=task_ids.device, dtype=self.task_feature_table.dtype,
        )
        self._active_gates = gates
        return gates

    def identity_regularization(self, gates: torch.Tensor) -> torch.Tensor:
        """Keep routing near the vanilla-LoRA identity early in training."""
        return (gates.float() - 1.0).square().mean()

    @torch.no_grad()
    def routing_summary(self) -> dict[str, float]:
        features = self.task_feature_table
        logits = self.layernets(self.hypernet(features)).view(
            len(TASK_TO_ID), len(self.site_names), self.rank
        )
        gates = 1.0 + 0.5 * torch.tanh(logits.float())
        derivative = 0.5 * (1.0 - torch.tanh(logits.float()).square())
        summary = {
            "gate_min": float(gates.min()),
            "gate_max": float(gates.max()),
            "gate_std": float(gates.std()),
            "task_pair_mean_abs_diff": float(
                torch.stack(
                    [
                        (gates[i] - gates[j]).abs().mean()
                        for i in range(len(TASK_TO_ID))
                        for j in range(i + 1, len(TASK_TO_ID))
                    ]
                ).mean()
            ),
        }
        for task_id, task_name in ID_TO_TASK.items():
            task_gates = gates[task_id]
            task_derivative = derivative[task_id]
            summary[f"{task_name}_gate_mean"] = float(task_gates.mean())
            summary[f"{task_name}_near_lower_frac"] = float((task_gates < 0.55).float().mean())
            summary[f"{task_name}_near_upper_frac"] = float((task_gates > 1.45).float().mean())
            summary[f"{task_name}_mean_derivative"] = float(task_derivative.mean())
        return summary
