"""Block-causal Wan2.2 self-attention with an equivalent streaming KV path.

This module deliberately does not modify DiffSynth's official Wan implementation.
``install_causal_attention`` replaces only the parameter-free attention dispatchers
below each self-attention layer. Q/K/V/O projections (including PEFT wrappers),
RoPE, task routing, and every state-dict key remain unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d


@dataclass(frozen=True)
class CausalAttentionConfig:
    """Temporal block layout used by both training and streaming inference."""

    num_frames: int
    tokens_per_frame: int
    frames_per_chunk: int = 1
    local_attn_frames: int = -1
    sink_frames: int = 0

    def __post_init__(self) -> None:
        if self.num_frames <= 0 or self.tokens_per_frame <= 0:
            raise ValueError("num_frames and tokens_per_frame must be positive")
        if self.frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be positive")
        if self.num_frames % self.frames_per_chunk:
            raise ValueError("num_frames must be divisible by frames_per_chunk")
        if self.local_attn_frames != -1 and self.local_attn_frames < self.frames_per_chunk:
            raise ValueError("local_attn_frames must be -1 or at least one chunk")
        if self.sink_frames < 0:
            raise ValueError("sink_frames cannot be negative")
        if self.local_attn_frames != -1 and self.sink_frames >= self.local_attn_frames:
            raise ValueError("sink_frames must be smaller than local_attn_frames")

    @property
    def chunk_tokens(self) -> int:
        return self.frames_per_chunk * self.tokens_per_frame

    @property
    def sequence_tokens(self) -> int:
        return self.num_frames * self.tokens_per_frame

    def metadata(self) -> dict[str, str]:
        return {
            "causal_attention": "block_prefix_v1",
            "causal_num_frames": str(self.num_frames),
            "causal_tokens_per_frame": str(self.tokens_per_frame),
            "causal_frames_per_chunk": str(self.frames_per_chunk),
            "causal_local_attn_frames": str(self.local_attn_frames),
            "causal_sink_frames": str(self.sink_frames),
            "causal_streaming_kv": "per_layer_v1",
        }


@dataclass
class LayerKVCache:
    """RoPE-applied keys and values for one DiT self-attention layer."""

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    seen_tokens: int = 0

    def reset(self) -> None:
        self.key = None
        self.value = None
        self.seen_tokens = 0


def _as_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    return rearrange(x, "b s (n d) -> b n s d", n=num_heads)


def _from_heads(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b n s d -> b s (n d)")


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Use PyTorch SDPA so CUDA can select its fused FlashAttention backend."""

    output = F.scaled_dot_product_attention(
        _as_heads(q, num_heads),
        _as_heads(k, num_heads),
        _as_heads(v, num_heads),
        dropout_p=0.0,
        is_causal=False,
    )
    return _from_heads(output)


def _select_prefix(
    key: torch.Tensor,
    value: torch.Tensor,
    end: int,
    config: CausalAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.local_attn_frames == -1:
        return key[:, :end], value[:, :end]

    window_tokens = config.local_attn_frames * config.tokens_per_frame
    sink_tokens = config.sink_frames * config.tokens_per_frame
    start = max(sink_tokens, end - window_tokens)
    if sink_tokens == 0 or start == 0:
        return key[:, start:end], value[:, start:end]
    return (
        torch.cat((key[:, :sink_tokens], key[:, start:end]), dim=1),
        torch.cat((value[:, :sink_tokens], value[:, start:end]), dim=1),
    )


def block_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    config: CausalAttentionConfig,
) -> torch.Tensor:
    """Full-sequence block-causal attention used during single-forward training.

    Tokens inside the current temporal chunk attend bidirectionally to one another
    and to allowed prefix chunks, but never to a future chunk. Splitting queries
    avoids materializing an O(sequence_length^2) boolean mask.
    """

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical [B, S, D] shapes")
    if query.shape[1] != config.sequence_tokens:
        raise ValueError(
            f"Expected {config.sequence_tokens} tokens from the configured latent layout, "
            f"got {query.shape[1]}"
        )

    chunks = []
    for start in range(0, config.sequence_tokens, config.chunk_tokens):
        end = start + config.chunk_tokens
        prefix_key, prefix_value = _select_prefix(key, value, end, config)
        chunks.append(_attention(query[:, start:end], prefix_key, prefix_value, num_heads))
    return torch.cat(chunks, dim=1)


def streaming_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    config: CausalAttentionConfig,
    cache: LayerKVCache,
) -> torch.Tensor:
    """Attend one or more complete chunks while updating a per-layer KV cache."""

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical [B, S, D] shapes")
    if query.shape[1] == 0 or query.shape[1] % config.chunk_tokens:
        raise ValueError("streaming input must contain a positive whole number of chunks")
    if cache.seen_tokens + query.shape[1] > config.sequence_tokens:
        raise ValueError("streaming input exceeds the configured sequence length")

    outputs = []
    for start in range(0, query.shape[1], config.chunk_tokens):
        end = start + config.chunk_tokens
        current_key = key[:, start:end]
        current_value = value[:, start:end]
        cache.key = current_key if cache.key is None else torch.cat((cache.key, current_key), dim=1)
        cache.value = current_value if cache.value is None else torch.cat((cache.value, current_value), dim=1)
        cache.seen_tokens += config.chunk_tokens
        prefix_key, prefix_value = _select_prefix(cache.key, cache.value, cache.seen_tokens, config)
        outputs.append(_attention(query[:, start:end], prefix_key, prefix_value, num_heads))
    return torch.cat(outputs, dim=1)


class CausalAttentionModule(nn.Module):
    """Drop-in replacement for DiffSynth's parameter-free AttentionModule."""

    def __init__(self, num_heads: int, config: CausalAttentionConfig, layer_index: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.config = config
        self.layer_index = layer_index
        self._streaming_cache: LayerKVCache | None = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self._streaming_cache is None:
            return block_causal_attention(q, k, v, num_heads=self.num_heads, config=self.config)
        return streaming_attention(
            q, k, v,
            num_heads=self.num_heads,
            config=self.config,
            cache=self._streaming_cache,
        )


class CausalWan22Controller:
    """Owns the causal layout and per-layer streaming cache lifecycle."""

    def __init__(self, modules: Sequence[CausalAttentionModule], config: CausalAttentionConfig) -> None:
        self.modules = tuple(modules)
        self.config = config

    def new_kv_cache(self) -> list[LayerKVCache]:
        return [LayerKVCache() for _ in self.modules]

    @contextmanager
    def use_kv_cache(self, caches: Sequence[LayerKVCache]) -> Iterator[None]:
        if len(caches) != len(self.modules):
            raise ValueError(f"Expected {len(self.modules)} layer caches, got {len(caches)}")
        if any(module._streaming_cache is not None for module in self.modules):
            raise RuntimeError("A streaming KV-cache context is already active")
        try:
            for module, cache in zip(self.modules, caches):
                module._streaming_cache = cache
            yield
        finally:
            for module in self.modules:
                module._streaming_cache = None

    @torch.no_grad()
    def streaming_forward(
        self,
        dit: nn.Module,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        frame_offset: int,
        caches: Sequence[LayerKVCache],
        clip_feature: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a latent chunk with global temporal RoPE and per-layer KV caches."""

        if dit.training:
            raise RuntimeError("streaming_forward requires dit.eval()")
        if frame_offset < 0 or frame_offset % self.config.frames_per_chunk:
            raise ValueError("frame_offset must align to frames_per_chunk")
        frames = x.shape[2]
        if frames <= 0 or frames % self.config.frames_per_chunk:
            raise ValueError("streaming tensor must contain complete temporal chunks")
        if frame_offset + frames > self.config.num_frames:
            raise ValueError("streaming tensor exceeds the configured frame count")

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
        if f != frames // dit.patch_size[0]:
            raise ValueError("Unexpected temporal patch layout")
        temporal_start = frame_offset // dit.patch_size[0]
        freqs = torch.cat(
            (
                dit.freqs[0][temporal_start : temporal_start + f]
                .view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ),
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)

        with self.use_kv_cache(caches):
            for block in dit.blocks:
                x = block(x, context, t_mod, freqs)
        x = dit.head(x, t)
        return dit.unpatchify(x, (f, h, w))


def install_causal_attention(
    dit: nn.Module,
    *,
    latent_shape: Sequence[int],
    frames_per_chunk: int = 1,
    local_attn_frames: int = -1,
    sink_frames: int = 0,
) -> CausalWan22Controller:
    """Install causal dispatchers without changing any parameter/state-dict key."""

    if len(latent_shape) != 4:
        raise ValueError("latent_shape must be [C, F, H, W]")
    _, frames, height, width = map(int, latent_shape)
    temporal_patch, height_patch, width_patch = map(int, dit.patch_size)
    if temporal_patch != 1:
        raise ValueError("Wan2.2 causal baseline currently requires temporal patch size 1")
    if height % height_patch or width % width_patch:
        raise ValueError("latent spatial dimensions must be divisible by the patch size")
    config = CausalAttentionConfig(
        num_frames=frames // temporal_patch,
        tokens_per_frame=(height // height_patch) * (width // width_patch),
        frames_per_chunk=frames_per_chunk,
        local_attn_frames=local_attn_frames,
        sink_frames=sink_frames,
    )

    state_keys_before = tuple(dit.state_dict().keys())
    modules = []
    for layer_index, block in enumerate(dit.blocks):
        dispatcher = block.self_attn.attn
        if isinstance(dispatcher, CausalAttentionModule):
            raise RuntimeError("Causal attention is already installed")
        replacement = CausalAttentionModule(
            num_heads=block.self_attn.num_heads,
            config=config,
            layer_index=layer_index,
        )
        block.self_attn.attn = replacement
        modules.append(replacement)
    if tuple(dit.state_dict().keys()) != state_keys_before:
        raise RuntimeError("Installing causal attention changed Wan state-dict keys")
    return CausalWan22Controller(modules, config)
