"""Memory-bounded block-causal attention for Wan2.2 OPSD post-training.

This is an opt-in dispatcher used only by the OPSD entry point.  It retains a
fixed sink plus a sliding local window and records absolute token positions so
that delayed teacher-cache replacement remains well-defined after eviction.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn as nn

from wan22_causal_attention import CausalAttentionConfig, _attention, block_causal_attention


@dataclass
class BoundedLayerKVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    seen_tokens: int = 0


def bounded_streaming_attention(query, key, value, *, num_heads, config, cache):
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value shapes differ")
    if query.shape[1] == 0 or query.shape[1] % config.chunk_tokens:
        raise ValueError("streaming input must contain complete chunks")
    if config.local_attn_frames == -1:
        raise ValueError("bounded cache requires a finite local attention window")
    if cache.seen_tokens + query.shape[1] > config.sequence_tokens:
        raise ValueError("streaming input exceeds configured sequence")

    outputs = []
    sink_tokens = config.sink_frames * config.tokens_per_frame
    window_tokens = config.local_attn_frames * config.tokens_per_frame
    for start in range(0, query.shape[1], config.chunk_tokens):
        end = start + config.chunk_tokens
        absolute_start = cache.seen_tokens
        current_positions = torch.arange(
            absolute_start, absolute_start + config.chunk_tokens,
            device=key.device, dtype=torch.long,
        )
        current_key, current_value = key[:, start:end], value[:, start:end]
        if cache.key is None:
            joined_key, joined_value = current_key, current_value
            joined_positions = current_positions
        else:
            joined_key = torch.cat((cache.key, current_key), dim=1)
            joined_value = torch.cat((cache.value, current_value), dim=1)
            joined_positions = torch.cat((cache.positions, current_positions))
        absolute_end = absolute_start + config.chunk_tokens
        keep = (joined_positions < sink_tokens) | (joined_positions >= absolute_end - window_tokens)
        selected_key, selected_value = joined_key[:, keep], joined_value[:, keep]
        outputs.append(_attention(query[:, start:end], selected_key, selected_value, num_heads))
        cache.key, cache.value = selected_key, selected_value
        cache.positions = joined_positions[keep]
        cache.seen_tokens = absolute_end
    return torch.cat(outputs, dim=1)


class BoundedCausalAttentionModule(nn.Module):
    def __init__(self, num_heads, config, layer_index):
        super().__init__()
        self.num_heads, self.config, self.layer_index = num_heads, config, layer_index
        self._streaming_cache = None

    def forward(self, q, k, v):
        if self._streaming_cache is None:
            return block_causal_attention(q, k, v, num_heads=self.num_heads, config=self.config)
        return bounded_streaming_attention(
            q, k, v, num_heads=self.num_heads, config=self.config,
            cache=self._streaming_cache,
        )


class BoundedCausalWan22Controller:
    def __init__(self, modules, config):
        self.modules, self.config = tuple(modules), config

    def new_kv_cache(self):
        return [BoundedLayerKVCache() for _ in self.modules]

    @contextmanager
    def use_kv_cache(self, caches) -> Iterator[None]:
        if len(caches) != len(self.modules):
            raise ValueError("cache layer count differs from model")
        if any(module._streaming_cache is not None for module in self.modules):
            raise RuntimeError("a streaming cache context is already active")
        try:
            for module, cache in zip(self.modules, caches):
                module._streaming_cache = cache
            yield
        finally:
            for module in self.modules:
                module._streaming_cache = None


def install_opsd_bounded_causal_attention(
    dit: nn.Module, *, latent_shape: Sequence[int], frames_per_chunk: int = 3,
    local_attn_frames: int = 12, sink_frames: int = 3,
) -> BoundedCausalWan22Controller:
    if len(latent_shape) != 4:
        raise ValueError("latent_shape must be [C, F, H, W]")
    _, frames, height, width = map(int, latent_shape)
    temporal_patch, height_patch, width_patch = map(int, dit.patch_size)
    config = CausalAttentionConfig(
        num_frames=frames // temporal_patch,
        tokens_per_frame=(height // height_patch) * (width // width_patch),
        frames_per_chunk=frames_per_chunk,
        local_attn_frames=local_attn_frames,
        sink_frames=sink_frames,
    )
    keys_before = tuple(dit.state_dict())
    modules = []
    for layer_index, block in enumerate(dit.blocks):
        replacement = BoundedCausalAttentionModule(
            block.self_attn.num_heads, config, layer_index
        )
        block.self_attn.attn = replacement
        modules.append(replacement)
    if tuple(dit.state_dict()) != keys_before:
        raise RuntimeError("installing bounded attention changed state-dict keys")
    return BoundedCausalWan22Controller(modules, config)
