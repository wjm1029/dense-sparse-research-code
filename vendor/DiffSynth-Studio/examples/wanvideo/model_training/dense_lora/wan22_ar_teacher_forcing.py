"""Chunkwise autoregressive target-history conditioning for Wan2.2 Dense4.

The official Wan implementation remains untouched.  RGB tokens keep the
pretrained patch embedding while a zero-initialized parallel patch projection
adds the previous target chunk.  At training time that history is a one-chunk
right shift of ground truth (teacher forcing); at inference time it is the
previous predicted chunk (free-running autoregression).
"""

from __future__ import annotations

from dataclasses import dataclass
import types
from typing import Sequence

import torch
import torch.nn as nn

from wan22_causal_attention import CausalWan22Controller, LayerKVCache


@dataclass(frozen=True)
class ARTeacherForcingConfig:
    num_frames: int = 21
    frames_per_chunk: int = 3
    latent_channels: int = 48

    def __post_init__(self) -> None:
        if self.num_frames <= 0 or self.frames_per_chunk <= 0 or self.latent_channels <= 0:
            raise ValueError("AR dimensions must be positive")
        if self.num_frames % self.frames_per_chunk:
            raise ValueError("num_frames must be divisible by frames_per_chunk")

    @property
    def num_chunks(self) -> int:
        return self.num_frames // self.frames_per_chunk

    def metadata(self) -> dict[str, str]:
        return {
            "ar_teacher_forcing": "previous_gt_chunk_v1",
            "ar_inference_history": "previous_predicted_chunk_v1",
            "ar_history_adapter": "zero_init_additive_patch_v1",
            "ar_num_frames": str(self.num_frames),
            "ar_frames_per_chunk": str(self.frames_per_chunk),
            "ar_num_chunks": str(self.num_chunks),
            "ar_latent_channels": str(self.latent_channels),
        }


def teacher_forcing_history(target: torch.Tensor, config: ARTeacherForcingConfig) -> torch.Tensor:
    """Right-shift target by one complete chunk without current-chunk leakage."""

    if target.ndim != 5:
        raise ValueError("target must have shape [B, C, F, H, W]")
    if target.shape[1] != config.latent_channels or target.shape[2] != config.num_frames:
        raise ValueError(
            f"Expected C/F={config.latent_channels}/{config.num_frames}, got "
            f"{target.shape[1]}/{target.shape[2]}"
        )
    history = torch.zeros_like(target)
    k = config.frames_per_chunk
    history[:, :, k:] = target[:, :, :-k]
    return history


def combine_rgb_and_history(
    rgb: torch.Tensor,
    history: torch.Tensor,
    config: ARTeacherForcingConfig,
) -> torch.Tensor:
    if rgb.shape != history.shape:
        raise ValueError(f"RGB/history shapes differ: {rgb.shape} != {history.shape}")
    if rgb.ndim != 5 or rgb.shape[1] != config.latent_channels:
        raise ValueError("RGB/history must be [B, latent_channels, F, H, W]")
    return torch.cat((rgb, history), dim=1)


class ARHistoryAdapterController:
    """Own the added history projection and chunkwise free-running rollout."""

    def __init__(
        self,
        dit: nn.Module,
        causal_controller: CausalWan22Controller,
        config: ARTeacherForcingConfig,
        original_patchify,
    ) -> None:
        self.dit = dit
        self.causal_controller = causal_controller
        self.config = config
        self.original_patchify = original_patchify

    @property
    def history_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.dit.lora_history_patch.parameters())

    def assert_zero_initialized(self) -> None:
        if any(torch.count_nonzero(parameter.detach()) for parameter in self.history_parameters):
            raise RuntimeError("AR history adapter is not zero-initialized")

    def teacher_forced_input(self, rgb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return combine_rgb_and_history(rgb, teacher_forcing_history(target, self.config), self.config)

    def new_kv_cache(self) -> list[LayerKVCache]:
        return self.causal_controller.new_kv_cache()

    @torch.no_grad()
    def streaming_teacher_forced(
        self,
        rgb: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        caches: Sequence[LayerKVCache] | None = None,
    ) -> torch.Tensor:
        history = teacher_forcing_history(target, self.config)
        return self._stream_chunks(rgb, history, timestep, context, caches=caches)

    def _stream_chunks(
        self,
        rgb: torch.Tensor,
        history: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        caches: Sequence[LayerKVCache] | None = None,
    ) -> torch.Tensor:
        if rgb.shape != history.shape or rgb.shape[2] != self.config.num_frames:
            raise ValueError("Invalid RGB/history shape for AR streaming")
        if caches is None:
            caches = self.new_kv_cache()
        outputs = []
        k = self.config.frames_per_chunk
        for start in range(0, self.config.num_frames, k):
            ar_input = combine_rgb_and_history(
                rgb[:, :, start : start + k],
                history[:, :, start : start + k],
                self.config,
            )
            outputs.append(
                self.causal_controller.streaming_forward(
                    self.dit,
                    ar_input,
                    timestep,
                    context,
                    frame_offset=start,
                    caches=caches,
                )
            )
        return torch.cat(outputs, dim=2)

    @torch.no_grad()
    def free_running(
        self,
        rgb: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        output_sign: float = -1.0,
        caches: Sequence[LayerKVCache] | None = None,
    ) -> torch.Tensor:
        """Generate chunks sequentially and feed each prediction to the next."""

        if rgb.ndim != 5 or rgb.shape[1] != self.config.latent_channels:
            raise ValueError("RGB must be [B, latent_channels, F, H, W]")
        if rgb.shape[2] != self.config.num_frames:
            raise ValueError("RGB frame count differs from AR configuration")
        if caches is None:
            caches = self.new_kv_cache()
        k = self.config.frames_per_chunk
        previous = torch.zeros_like(rgb[:, :, :k])
        outputs = []
        for start in range(0, self.config.num_frames, k):
            ar_input = combine_rgb_and_history(
                rgb[:, :, start : start + k], previous, self.config
            )
            raw = self.causal_controller.streaming_forward(
                self.dit,
                ar_input,
                timestep,
                context,
                frame_offset=start,
                caches=caches,
            )
            prediction = raw * output_sign
            outputs.append(prediction)
            previous = prediction.detach()
        return torch.cat(outputs, dim=2)


def install_ar_history_adapter(
    dit: nn.Module,
    causal_controller: CausalWan22Controller,
    config: ARTeacherForcingConfig,
) -> ARHistoryAdapterController:
    """Add a zero-init history patch projection without editing official Wan code."""

    if hasattr(dit, "lora_history_patch"):
        raise RuntimeError("AR history adapter is already installed")
    if causal_controller.config.num_frames != config.num_frames:
        raise ValueError("Causal and AR frame counts differ")
    if causal_controller.config.frames_per_chunk != config.frames_per_chunk:
        raise ValueError("Causal and AR chunk sizes differ")
    patch = dit.patch_embedding
    if not isinstance(patch, nn.Conv3d) or patch.in_channels != config.latent_channels:
        raise ValueError("Unexpected Wan patch embedding contract")

    history_patch = nn.Conv3d(
        config.latent_channels,
        patch.out_channels,
        kernel_size=patch.kernel_size,
        stride=patch.stride,
        padding=patch.padding,
        dilation=patch.dilation,
        groups=patch.groups,
        bias=patch.bias is not None,
        padding_mode=patch.padding_mode,
        device=patch.weight.device,
        dtype=torch.float32,
    )
    nn.init.zeros_(history_patch.weight)
    if history_patch.bias is not None:
        nn.init.zeros_(history_patch.bias)
    dit.add_module("lora_history_patch", history_patch)
    original_patchify = dit.patchify

    def ar_patchify(self, x, control_camera_latents_input=None, enable_wantodance_global=False):
        expected_channels = 2 * config.latent_channels
        if x.ndim != 5 or x.shape[1] != expected_channels:
            raise ValueError(f"AR Wan expects {expected_channels} channels, got {tuple(x.shape)}")
        rgb, history = x.split(config.latent_channels, dim=1)
        rgb_tokens = original_patchify(
            rgb,
            control_camera_latents_input,
            enable_wantodance_global=enable_wantodance_global,
        )
        history_tokens = self.lora_history_patch(history.to(dtype=self.lora_history_patch.weight.dtype))
        return rgb_tokens + history_tokens.to(dtype=rgb_tokens.dtype)

    dit.patchify = types.MethodType(ar_patchify, dit)
    controller = ARHistoryAdapterController(
        dit, causal_controller, config, original_patchify
    )
    controller.assert_zero_initialized()
    return controller
