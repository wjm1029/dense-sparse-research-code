"""Inference-only pre-RoPE, fixed-capacity KV pool for Wan2.2 streaming.

This is an opt-in alternative to the training-time bounded cache.  It keeps
both pre-RoPE keys (for future position remapping) and the currently materialized
RoPE keys (for fast exact inference).  Storage is allocated once per layer as a
global sink plus a circular window of complete chunks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import types
from typing import Iterator, Sequence

import torch
import torch.nn as nn
from einops import rearrange

from diffsynth.models.wan_video_dit import rope_apply, sinusoidal_embedding_1d
from wan22_causal_attention import CausalAttentionConfig, _attention, block_causal_attention


@dataclass
class FixedPreRoPELayerKVCache:
    sink_key_pre: torch.Tensor | None = None
    sink_key: torch.Tensor | None = None
    sink_value: torch.Tensor | None = None
    tail_key_pre: torch.Tensor | None = None
    tail_key: torch.Tensor | None = None
    tail_value: torch.Tensor | None = None
    assembled_key: torch.Tensor | None = None
    assembled_value: torch.Tensor | None = None
    anchor_key_pre: torch.Tensor | None = None
    anchor_value: torch.Tensor | None = None
    intervention_freqs: torch.Tensor | None = None
    intervention_active: bool = False
    anchor_valid: bool = False
    sink_chunks: int = 0
    tail_chunks: int = 0
    tail_write_chunk: int = 0
    seen_chunks: int = 0
    last_assembled_tokens: int = 0
    chunk_tokens: int = 0

    def data_ptrs(self) -> tuple[int, ...]:
        tensors = (
            self.sink_key_pre, self.sink_key, self.sink_value,
            self.tail_key_pre, self.tail_key, self.tail_value,
            self.assembled_key, self.assembled_value,
            self.anchor_key_pre, self.anchor_value,
        )
        return tuple(t.data_ptr() for t in tensors if t is not None)


def _allocate(cache, key, value, config):
    chunk_tokens = config.chunk_tokens
    sink_chunks = config.sink_frames // config.frames_per_chunk
    tail_chunks = config.local_attn_frames // config.frames_per_chunk
    if config.sink_frames % config.frames_per_chunk:
        raise ValueError("sink_frames must align to complete chunks")
    if config.local_attn_frames % config.frames_per_chunk:
        raise ValueError("local_attn_frames must align to complete chunks")
    shape = (key.shape[0], chunk_tokens, key.shape[2])
    sink_shape = (shape[0], sink_chunks * chunk_tokens, shape[2])
    tail_shape = (shape[0], tail_chunks * chunk_tokens, shape[2])
    assembled_shape = (shape[0], (sink_chunks + tail_chunks) * chunk_tokens, shape[2])
    cache.sink_key_pre = key.new_empty(sink_shape)
    cache.sink_key = key.new_empty(sink_shape)
    cache.sink_value = value.new_empty(sink_shape)
    cache.tail_key_pre = key.new_empty(tail_shape)
    cache.tail_key = key.new_empty(tail_shape)
    cache.tail_value = value.new_empty(tail_shape)
    cache.assembled_key = key.new_empty(assembled_shape)
    cache.assembled_value = value.new_empty(assembled_shape)
    cache.chunk_tokens = chunk_tokens


def snapshot_latest_chunk_as_anchor_(cache: FixedPreRoPELayerKVCache) -> None:
    """Preserve the most recently cached chunk as an intervention anchor."""
    if cache.tail_key_pre is None or cache.tail_chunks == 0:
        raise RuntimeError("Cannot snapshot an anchor before a chunk is cached")
    chunk_tokens = cache.chunk_tokens
    if cache.anchor_key_pre is None:
        shape = (cache.tail_key_pre.shape[0], chunk_tokens, cache.tail_key_pre.shape[2])
        cache.anchor_key_pre = cache.tail_key_pre.new_empty(shape)
        cache.anchor_value = cache.tail_value.new_empty(shape)
    slot = (cache.tail_write_chunk - 1) % (cache.tail_key_pre.shape[1] // chunk_tokens)
    source = slice(slot * chunk_tokens, (slot + 1) * chunk_tokens)
    cache.anchor_key_pre.copy_(cache.tail_key_pre[:, source])
    cache.anchor_value.copy_(cache.tail_value[:, source])
    cache.anchor_valid = True


def fixed_pool_streaming_attention(
    query_pre, key_pre, value, freqs, *, num_heads, config, cache
):
    if query_pre.shape != key_pre.shape or query_pre.shape != value.shape:
        raise ValueError("query/key/value shapes differ")
    if query_pre.shape[1] == 0 or query_pre.shape[1] % config.chunk_tokens:
        raise ValueError("streaming input must contain complete chunks")
    if config.local_attn_frames == -1:
        raise ValueError("fixed pool requires a finite local window")
    if cache.sink_key_pre is None:
        _allocate(cache, key_pre, value, config)

    outputs = []
    chunk_tokens = config.chunk_tokens
    sink_capacity = config.sink_frames // config.frames_per_chunk
    tail_capacity = config.local_attn_frames // config.frames_per_chunk
    for start in range(0, query_pre.shape[1], chunk_tokens):
        end = start + chunk_tokens
        q = rope_apply(query_pre[:, start:end], freqs[start:end], num_heads)
        k_pre = key_pre[:, start:end]
        k = rope_apply(k_pre, freqs[start:end], num_heads)
        v = value[:, start:end]

        absolute_chunk = cache.seen_chunks
        if absolute_chunk < sink_capacity:
            sink_slice = slice(absolute_chunk * chunk_tokens, (absolute_chunk + 1) * chunk_tokens)
            cache.sink_key_pre[:, sink_slice].copy_(k_pre)
            cache.sink_key[:, sink_slice].copy_(k)
            cache.sink_value[:, sink_slice].copy_(v)
            cache.sink_chunks += 1

        slot = cache.tail_write_chunk
        tail_slice = slice(slot * chunk_tokens, (slot + 1) * chunk_tokens)
        cache.tail_key_pre[:, tail_slice].copy_(k_pre)
        cache.tail_key[:, tail_slice].copy_(k)
        cache.tail_value[:, tail_slice].copy_(v)
        cache.tail_write_chunk = (slot + 1) % tail_capacity
        cache.tail_chunks = min(cache.tail_chunks + 1, tail_capacity)
        cache.seen_chunks += 1

        cursor = 0
        if cache.sink_chunks:
            count = cache.sink_chunks * chunk_tokens
            cache.assembled_key[:, :count].copy_(cache.sink_key[:, :count])
            cache.assembled_value[:, :count].copy_(cache.sink_value[:, :count])
            cursor = count

        first_slot = 0 if cache.tail_chunks < tail_capacity else cache.tail_write_chunk
        for order in range(cache.tail_chunks):
            tail_slot = (first_slot + order) % tail_capacity
            tail_absolute_chunk = cache.seen_chunks - cache.tail_chunks + order
            if tail_absolute_chunk < cache.sink_chunks:
                continue
            source = slice(tail_slot * chunk_tokens, (tail_slot + 1) * chunk_tokens)
            target = slice(cursor, cursor + chunk_tokens)
            replace_previous = (
                cache.intervention_active
                and cache.anchor_valid
                and cache.intervention_freqs is not None
                and tail_absolute_chunk == absolute_chunk - 1
            )
            if replace_previous:
                disguised_key = rope_apply(
                    cache.anchor_key_pre, cache.intervention_freqs, num_heads
                )
                cache.assembled_key[:, target].copy_(disguised_key)
                cache.assembled_value[:, target].copy_(cache.anchor_value)
            else:
                cache.assembled_key[:, target].copy_(cache.tail_key[:, source])
                cache.assembled_value[:, target].copy_(cache.tail_value[:, source])
            cursor += chunk_tokens
        outputs.append(
            _attention(
                q, cache.assembled_key[:, :cursor], cache.assembled_value[:, :cursor], num_heads
            )
        )
        cache.last_assembled_tokens = cursor
    return torch.cat(outputs, dim=1)


class PreRoPEFixedPoolAttention(nn.Module):
    def __init__(self, num_heads, config, layer_index):
        super().__init__()
        self.num_heads, self.config, self.layer_index = num_heads, config, layer_index
        self._streaming_cache = None

    def forward(self, q_pre, k_pre, value, freqs):
        if self._streaming_cache is None:
            q = rope_apply(q_pre, freqs, self.num_heads)
            k = rope_apply(k_pre, freqs, self.num_heads)
            return block_causal_attention(q, k, value, num_heads=self.num_heads, config=self.config)
        return fixed_pool_streaming_attention(
            q_pre, k_pre, value, freqs, num_heads=self.num_heads,
            config=self.config, cache=self._streaming_cache,
        )


def _prerope_self_attention_forward(self, x, freqs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)
    return self.o(self.attn(q, k, v, freqs))


class PreRoPEFixedPoolController:
    def __init__(self, modules, config):
        self.modules, self.config = tuple(modules), config
        self._intervention_schedule: tuple[bool, ...] | None = None
        self._runtime_intervention_active = False
        self._intervention_activations = 0
        self._intervened_chunks = 0

    def new_kv_cache(self):
        self._runtime_intervention_active = False
        self._intervention_activations = 0
        self._intervened_chunks = 0
        return [FixedPreRoPELayerKVCache() for _ in self.modules]

    def set_cache_intervention_schedule(self, schedule: Sequence[bool] | None) -> None:
        if schedule is not None and len(schedule) != self.config.num_frames // self.config.frames_per_chunk:
            raise ValueError("Cache-intervention schedule length differs from chunk count")
        self._intervention_schedule = None if schedule is None else tuple(bool(x) for x in schedule)

    def cache_intervention_stats(self) -> dict[str, int | bool]:
        return {
            "enabled": self._intervention_schedule is not None,
            "activations": self._intervention_activations,
            "intervened_chunks": self._intervened_chunks,
        }

    def _prepare_cache_intervention(self, caches, chunk_index, previous_freqs) -> None:
        active = bool(
            self._intervention_schedule is not None
            and self._intervention_schedule[chunk_index]
            and chunk_index > 0
        )
        if active and not self._runtime_intervention_active:
            for cache in caches:
                snapshot_latest_chunk_as_anchor_(cache)
            self._intervention_activations += 1
        if active:
            self._intervened_chunks += 1
        for cache in caches:
            cache.intervention_active = active
            cache.intervention_freqs = previous_freqs if active else None
        self._runtime_intervention_active = active

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

    @torch.no_grad()
    def streaming_forward(
        self, dit, x, timestep, context, *, frame_offset, caches,
        clip_feature=None, y=None,
    ):
        if dit.training:
            raise RuntimeError("streaming_forward requires dit.eval()")
        frames = x.shape[2]
        if frame_offset < 0 or frame_offset % self.config.frames_per_chunk:
            raise ValueError("frame_offset must align to chunks")
        if frames <= 0 or frames % self.config.frames_per_chunk:
            raise ValueError("input must contain complete chunks")
        if frame_offset + frames > self.config.num_frames:
            raise ValueError("stream exceeds configured sequence")
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(x.dtype))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        context = dit.text_embedding(context)
        if y is not None and dit.require_vae_embedding:
            x = torch.cat((x, y), dim=1)
        if clip_feature is not None and dit.require_clip_embedding:
            context = torch.cat((dit.img_emb(clip_feature), context), dim=1)
        x = dit.patchify(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        temporal_start = frame_offset // dit.patch_size[0]
        freqs = torch.cat((
            dit.freqs[0][temporal_start:temporal_start + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ), dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        chunk_index = frame_offset // self.config.frames_per_chunk
        previous_freqs = None
        if self._intervention_schedule is not None and chunk_index > 0:
            previous_temporal_start = temporal_start - self.config.frames_per_chunk
            previous_freqs = torch.cat((
                dit.freqs[0][previous_temporal_start:previous_temporal_start + f]
                .view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ), dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        self._prepare_cache_intervention(caches, chunk_index, previous_freqs)
        with self.use_kv_cache(caches):
            for block in dit.blocks:
                x = block(x, context, t_mod, freqs)
        x = dit.head(x, t)
        return dit.unpatchify(x, (f, h, w))


def install_prerope_fixed_pool_attention(
    dit: nn.Module, *, latent_shape: Sequence[int], frames_per_chunk=3,
    local_attn_frames=12, sink_frames=3,
) -> PreRoPEFixedPoolController:
    _, frames, height, width = map(int, latent_shape)
    temporal_patch, height_patch, width_patch = map(int, dit.patch_size)
    if temporal_patch != 1:
        raise ValueError("Wan2.2 implementation requires temporal patch size 1")
    config = CausalAttentionConfig(
        num_frames=frames, tokens_per_frame=(height // height_patch) * (width // width_patch),
        frames_per_chunk=frames_per_chunk, local_attn_frames=local_attn_frames,
        sink_frames=sink_frames,
    )
    keys_before = tuple(dit.state_dict())
    modules = []
    for layer_index, block in enumerate(dit.blocks):
        block.self_attn.attn = PreRoPEFixedPoolAttention(
            block.self_attn.num_heads, config, layer_index
        )
        block.self_attn.forward = types.MethodType(
            _prerope_self_attention_forward, block.self_attn
        )
        modules.append(block.self_attn.attn)
    if tuple(dit.state_dict()) != keys_before:
        raise RuntimeError("installing pre-RoPE attention changed state-dict keys")
    return PreRoPEFixedPoolController(modules, config)
