"""Numerical and allocation checks for the inference-only fixed KV pool."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from diffsynth.models.wan_video_dit import rope_apply
from wan22_causal_attention import CausalAttentionConfig
from wan22_opsd_bounded_causal_attention import (
    BoundedLayerKVCache,
    bounded_streaming_attention,
)
from wan22_prerope_kv_pool import (
    FixedPreRoPELayerKVCache,
    fixed_pool_streaming_attention,
    snapshot_latest_chunk_as_anchor_,
)


def main() -> None:
    torch.manual_seed(20260918)
    config = CausalAttentionConfig(
        num_frames=30, tokens_per_frame=2, frames_per_chunk=3,
        local_attn_frames=12, sink_frames=3,
    )
    heads, head_dim, batch = 2, 8, 1
    width = heads * head_dim
    reference = BoundedLayerKVCache()
    candidate = FixedPreRoPELayerKVCache()
    pointers = None
    max_error = 0.0
    for chunk in range(config.num_frames // config.frames_per_chunk):
        tokens = config.chunk_tokens
        q_pre = torch.randn(batch, tokens, width)
        k_pre = torch.randn(batch, tokens, width)
        value = torch.randn(batch, tokens, width)
        angles = torch.randn(tokens, 1, head_dim // 2, dtype=torch.float64)
        freqs = torch.polar(torch.ones_like(angles), angles)
        q = rope_apply(q_pre, freqs, heads)
        k = rope_apply(k_pre, freqs, heads)
        expected = bounded_streaming_attention(
            q, k, value, num_heads=heads, config=config, cache=reference
        )
        actual = fixed_pool_streaming_attention(
            q_pre, k_pre, value, freqs, num_heads=heads,
            config=config, cache=candidate,
        )
        max_error = max(max_error, float((expected - actual).abs().max()))
        current_pointers = candidate.data_ptrs()
        if pointers is None:
            pointers = current_pointers
        elif current_pointers != pointers:
            raise AssertionError("fixed KV pool reallocated storage")
        expected_tokens = min(
            (chunk + 1) * config.chunk_tokens,
            (config.sink_frames + config.local_attn_frames) * config.tokens_per_frame,
        )
        actual_tokens = candidate.last_assembled_tokens
        if actual_tokens != expected_tokens:
            raise AssertionError((chunk, actual_tokens, expected_tokens))
    if max_error > 1e-6:
        raise AssertionError(f"pre-RoPE pool differs from reference: {max_error}")

    intervention = FixedPreRoPELayerKVCache()
    generated = []
    for chunk in range(4):
        tokens = config.chunk_tokens
        q_pre = torch.randn(batch, tokens, width)
        k_pre = torch.randn(batch, tokens, width)
        value = torch.randn(batch, tokens, width)
        angles = torch.randn(tokens, 1, head_dim // 2, dtype=torch.float64)
        freqs = torch.polar(torch.ones_like(angles), angles)
        generated.append((q_pre, k_pre, value, freqs))
    for chunk, (q_pre, k_pre, value, freqs) in enumerate(generated):
        if chunk == 2:
            snapshot_latest_chunk_as_anchor_(intervention)
        if chunk >= 2:
            intervention.intervention_active = True
            intervention.intervention_freqs = generated[chunk - 1][3]
        fixed_pool_streaming_attention(
            q_pre, k_pre, value, freqs, num_heads=heads,
            config=config, cache=intervention,
        )
    chunk_tokens = config.chunk_tokens
    previous_slot = slice(2 * chunk_tokens, 3 * chunk_tokens)
    expected_disguised_anchor = rope_apply(
        intervention.anchor_key_pre, generated[2][3], heads
    )
    if not torch.allclose(
        intervention.assembled_key[:, previous_slot], expected_disguised_anchor, atol=1e-6
    ):
        raise AssertionError("frozen anchor was not substituted at the previous-chunk position")
    print({
        "chunks": config.num_frames // config.frames_per_chunk,
        "max_abs_error": max_error,
        "fixed_storage_tensors": len(pointers or ()),
        "intervention_anchor_substitution": True,
        "status": "PASS",
    })


if __name__ == "__main__":
    main()
