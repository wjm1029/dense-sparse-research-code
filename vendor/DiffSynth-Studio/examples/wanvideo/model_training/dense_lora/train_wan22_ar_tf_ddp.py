"""Wan2.2 Dense4 chunkwise autoregressive teacher-forcing continuation.

This intentionally wraps the audited Dense4 trainer instead of editing it.
The existing causal-30k LoRA/router checkpoint initializes every old parameter;
only the additive target-history patch projection starts at exactly zero.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_wan22_dense4_ddp as dense4
from wan22_ar_teacher_forcing import (
    ARTeacherForcingConfig,
    install_ar_history_adapter,
)
from wan22_causal_attention import install_causal_attention
from wan22_dense4_checkpoint import checkpoint_metadata


FRAMES_PER_CHUNK = 3
AR_CONFIG = ARTeacherForcingConfig(
    num_frames=dense4.LATENT_SHAPE[1],
    frames_per_chunk=FRAMES_PER_CHUNK,
    latent_channels=dense4.LATENT_SHAPE[0],
)


class ARTeacherForcingDense4LoRAModule(dense4.Dense4LoRAModule):
    """Dense4 model conditioned on the previous ground-truth/predicted chunk."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.causal_controller = install_causal_attention(
            self.pipe.dit,
            latent_shape=dense4.LATENT_SHAPE,
            frames_per_chunk=AR_CONFIG.frames_per_chunk,
            local_attn_frames=-1,
            sink_frames=0,
        )
        self.ar_controller = install_ar_history_adapter(
            self.pipe.dit, self.causal_controller, AR_CONFIG
        )

    def _activate_task(
        self, task_ids: torch.Tensor, prompt_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.context_table is None:
            raise RuntimeError("Prompt context table has not been prepared")
        if self.router_mode == "learned":
            gates = self.task_router(task_ids)
        else:
            gates = self.task_router.activate_identity(task_ids)
        timestep = torch.zeros(
            task_ids.shape[0], device=task_ids.device, dtype=torch.bfloat16
        )
        return gates, timestep, self.context_table[prompt_ids]

    def predict_velocity_teacher_forced(
        self,
        source_latent: torch.Tensor,
        target_latent: torch.Tensor,
        task_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        gradient_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates, timestep, context = self._activate_task(task_ids, prompt_ids)
        ar_input = self.ar_controller.teacher_forced_input(source_latent, target_latent)
        velocity = self.pipe.model_fn(
            dit=self.pipe.dit,
            latents=ar_input,
            timestep=timestep,
            context=context,
            use_gradient_checkpointing=gradient_checkpointing,
            use_gradient_checkpointing_offload=False,
            cfg_merge=False,
        )
        return velocity, gates

    @torch.no_grad()
    def predict_free_running(
        self,
        source_latent: torch.Tensor,
        task_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pipe.dit.training:
            raise RuntimeError("Free-running AR inference requires dit.eval()")
        gates, timestep, context = self._activate_task(task_ids, prompt_ids)
        prediction = self.ar_controller.free_running(
            source_latent,
            timestep,
            context,
            output_sign=-1.0,
        )
        return prediction, gates

    @torch.no_grad()
    def predict_velocity(
        self,
        source_latent: torch.Tensor,
        task_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        gradient_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference-driver compatibility: free-running prediction as velocity."""

        if gradient_checkpointing:
            raise ValueError("Free-running inference does not use gradient checkpointing")
        prediction, gates = self.predict_free_running(
            source_latent, task_ids, prompt_ids
        )
        return -prediction, gates

    def forward(
        self,
        source_latent: torch.Tensor,
        target_latent: torch.Tensor,
        task_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        gradient_checkpointing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        velocity, gates = self.predict_velocity_teacher_forced(
            source_latent,
            target_latent,
            task_ids,
            prompt_ids,
            gradient_checkpointing,
        )
        identity_loss = (
            self.task_router.identity_regularization(gates)
            if self.router_mode == "learned"
            else source_latent.new_zeros((), dtype=torch.float32)
        )
        data_loss = dense4.masked_latent_mse(-velocity, target_latent, loss_mask)
        total_loss = data_loss + self.router_identity_weight * identity_loss
        return total_loss, data_loss.detach(), identity_loss.detach()


_save_dense4_checkpoint = dense4.save_dense4_checkpoint
_load_dense4_checkpoint = dense4.load_dense4_checkpoint
_build_provenance_metadata = dense4.build_provenance_metadata


def _save_ar_checkpoint(*args, extra_metadata=None, **kwargs) -> None:
    metadata = dict(extra_metadata or {})
    metadata.update(
        {
            "causal_attention": "block_prefix_v1",
            "causal_num_frames": str(dense4.LATENT_SHAPE[1]),
            "causal_tokens_per_frame": str(
                (dense4.LATENT_SHAPE[2] // 2) * (dense4.LATENT_SHAPE[3] // 2)
            ),
            "causal_frames_per_chunk": str(AR_CONFIG.frames_per_chunk),
            "causal_local_attn_frames": "-1",
            "causal_sink_frames": "0",
            "causal_streaming_kv": "per_layer_v1",
            "causal_initialization": "causal_stage1_step_30000",
            **AR_CONFIG.metadata(),
        }
    )
    _save_dense4_checkpoint(*args, extra_metadata=metadata, **kwargs)


def _load_ar_checkpoint(module, path, **kwargs):
    """Strict resume, or audited zero-init transition from causal Stage 1."""

    metadata = checkpoint_metadata(path)
    is_ar = metadata.get("ar_teacher_forcing") == AR_CONFIG.metadata()["ar_teacher_forcing"]
    if is_ar:
        for key, expected in AR_CONFIG.metadata().items():
            if metadata.get(key) != expected:
                raise ValueError(f"AR checkpoint metadata mismatch for {key}")
        return _load_dense4_checkpoint(module=module, path=path, **kwargs)

    if metadata.get("causal_attention") != "block_prefix_v1":
        raise ValueError("AR initialization requires an audited causal checkpoint")
    parameters = module.ar_controller.history_parameters
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    module.ar_controller.assert_zero_initialized()
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        loaded = _load_dense4_checkpoint(module=module, path=path, **kwargs)
    finally:
        for parameter, requires_grad in zip(parameters, original_requires_grad):
            parameter.requires_grad_(requires_grad)
    module.ar_controller.assert_zero_initialized()
    return loaded


def _cli_value(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _install_init_provenance_bridge() -> None:
    checkpoint_path = _cli_value("--init_checkpoint")
    init_patch_path = _cli_value("--init_code_patch_path")
    if checkpoint_path is None:
        return
    if init_patch_path is None:
        raise ValueError("AR initialization requires --init_code_patch_path")
    with safe_open(checkpoint_path, framework="pt", device="cpu") as stream:
        metadata = dict(stream.metadata() or {})
    actual_patch_hash = dense4.sha256_file(Path(init_patch_path).resolve())
    if actual_patch_hash != metadata.get("runtime_source_patch_sha256"):
        raise ValueError("Initialization runtime patch does not match checkpoint metadata")
    source_git_head = metadata.get("diffsynth_git_head")
    if not source_git_head:
        raise ValueError("Initialization checkpoint is missing diffsynth_git_head")
    init_patch = Path(init_patch_path).resolve()

    def build_transition_provenance(*args, code_patch_path, **kwargs):
        result = _build_provenance_metadata(
            *args, code_patch_path=code_patch_path, **kwargs
        )
        if Path(code_patch_path).resolve() == init_patch:
            result["diffsynth_git_head"] = source_git_head
        return result

    dense4.build_provenance_metadata = build_transition_provenance


def main() -> None:
    if int(os.environ.get("WAN_AR_FRAMES_PER_CHUNK", str(FRAMES_PER_CHUNK))) != FRAMES_PER_CHUNK:
        raise ValueError("This audited CF++-aligned baseline requires exactly 3 frames/chunk")
    dense4.Dense4LoRAModule = ARTeacherForcingDense4LoRAModule
    dense4.save_dense4_checkpoint = _save_ar_checkpoint
    dense4.load_dense4_checkpoint = _load_ar_checkpoint
    _install_init_provenance_bridge()
    dense4.main()


if __name__ == "__main__":
    main()
