"""Differentiable chunk streaming for truncated-BPTT AR self-forcing.

This file extends the audited causal controller without changing DiffSynth or
official Wan sources.  Cached K/V and predicted history are detached at every
chunk boundary; gradients cover the current chunk only.
"""

from __future__ import annotations

from typing import Sequence

import torch
from einops import rearrange

from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from wan22_causal_attention import CausalWan22Controller, LayerKVCache


def detach_kv_cache(caches: Sequence[LayerKVCache]) -> None:
    for cache in caches:
        if cache.key is not None:
            cache.key = cache.key.detach()
        if cache.value is not None:
            cache.value = cache.value.detach()


def training_streaming_forward(
    controller: CausalWan22Controller,
    dit: torch.nn.Module,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    *,
    frame_offset: int,
    caches: Sequence[LayerKVCache],
) -> torch.Tensor:
    """Forward one complete latent chunk with gradients for that chunk."""

    if not dit.training:
        raise RuntimeError("self-forcing training requires dit.train()")
    if frame_offset < 0 or frame_offset % controller.config.frames_per_chunk:
        raise ValueError("frame_offset must align to the chunk size")
    frames = x.shape[2]
    if frames <= 0 or frames % controller.config.frames_per_chunk:
        raise ValueError("streaming tensor must contain complete chunks")
    if frame_offset + frames > controller.config.num_frames:
        raise ValueError("streaming tensor exceeds the configured sequence")

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(x.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    embedded_context = dit.text_embedding(context)
    tokens = dit.patchify(x)
    f, h, w = tokens.shape[2:]
    tokens = rearrange(tokens, "b c f h w -> b (f h w) c").contiguous()
    temporal_start = frame_offset // dit.patch_size[0]
    freqs = torch.cat(
        (
            dit.freqs[0][temporal_start : temporal_start + f]
            .view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ),
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(tokens.device)

    with controller.use_kv_cache(caches):
        for block in dit.blocks:
            tokens = block(tokens, embedded_context, t_mod, freqs)
    tokens = dit.head(tokens, t)
    return dit.unpatchify(tokens, (f, h, w))
