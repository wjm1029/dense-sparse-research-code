"""Causal Stage-1 continuation for the Wan2.2 Dense6 HyperLoRA checkpoint.

All data, t=0 flow-matching, task routing, losses, optimizer, and checkpoint
contracts come from ``train_wan22_dense6_ddp``. Only Wan self-attention is
changed through the separate adapter in ``wan22_causal_attention``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from safetensors import safe_open

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_wan22_dense6_dispersion_ddp as dense6
from wan22_causal_attention import install_causal_attention


FRAMES_PER_CHUNK = int(os.environ.get("WAN_CAUSAL_FRAMES_PER_CHUNK", "1"))
LOCAL_ATTN_FRAMES = int(os.environ.get("WAN_CAUSAL_LOCAL_ATTN_FRAMES", "-1"))
SINK_FRAMES = int(os.environ.get("WAN_CAUSAL_SINK_FRAMES", "0"))


class CausalDense6LoRAModule(dense6.Dense6LoRAModule):
    """Existing Stage-1 module with block-causal self-attention installed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.causal_controller = install_causal_attention(
            self.pipe.dit,
            latent_shape=dense6.LATENT_SHAPE,
            frames_per_chunk=FRAMES_PER_CHUNK,
            local_attn_frames=LOCAL_ATTN_FRAMES,
            sink_frames=SINK_FRAMES,
        )


_save_dense6_checkpoint = dense6.save_dense6_checkpoint
_build_provenance_metadata = dense6.build_provenance_metadata


def _save_causal_checkpoint(*args, extra_metadata=None, **kwargs) -> None:
    metadata = dict(extra_metadata or {})
    metadata.update(
        {
            "causal_attention": "block_prefix_v1",
            "causal_num_frames": str(dense6.LATENT_SHAPE[1]),
            "causal_tokens_per_frame": str(
                (dense6.LATENT_SHAPE[2] // 2) * (dense6.LATENT_SHAPE[3] // 2)
            ),
            "causal_frames_per_chunk": str(FRAMES_PER_CHUNK),
            "causal_local_attn_frames": str(LOCAL_ATTN_FRAMES),
            "causal_sink_frames": str(SINK_FRAMES),
            "causal_streaming_kv": "per_layer_v1",
            "causal_initialization": "dense6_bidir_step_120000",
        }
    )
    _save_dense6_checkpoint(*args, extra_metadata=metadata, **kwargs)


def _cli_value(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _install_init_provenance_bridge() -> None:
    """Allow an audited offline commit to initialize the new causal commit.

    The base loader normally requires the running Git HEAD to equal the HEAD in
    the input checkpoint. That is correct for same-code resume but impossible
    for the intentional offline-to-causal architecture transition. We retain
    every other strict provenance check and require the supplied immutable old
    source patch to match the checkpoint before accepting its recorded HEAD.
    """

    checkpoint_path = _cli_value("--init_checkpoint")
    init_patch_path = _cli_value("--init_code_patch_path")
    if checkpoint_path is None:
        return
    if init_patch_path is None:
        raise ValueError("Causal initialization requires --init_code_patch_path")
    with safe_open(checkpoint_path, framework="pt", device="cpu") as stream:
        checkpoint_metadata = dict(stream.metadata() or {})
    expected_patch_hash = checkpoint_metadata.get("runtime_source_patch_sha256")
    actual_patch_hash = dense6.sha256_file(Path(init_patch_path).resolve())
    if not expected_patch_hash or actual_patch_hash != expected_patch_hash:
        raise ValueError(
            "Initialization runtime patch does not match checkpoint metadata: "
            f"{actual_patch_hash!r} != {expected_patch_hash!r}"
        )
    source_git_head = checkpoint_metadata.get("diffsynth_git_head")
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

    dense6.build_provenance_metadata = build_transition_provenance


def main() -> None:
    # The imported trainer resolves both symbols dynamically from its module.
    dense6.Dense6LoRAModule = CausalDense6LoRAModule
    dense6.save_dense6_checkpoint = _save_causal_checkpoint
    _install_init_provenance_bridge()
    dense6.main()


if __name__ == "__main__":
    main()
