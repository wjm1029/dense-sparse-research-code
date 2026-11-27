from __future__ import annotations

import torch

from wan22_causal_attention import CausalAttentionConfig, block_causal_attention
from wan22_opsd_bounded_causal_attention import BoundedLayerKVCache, bounded_streaming_attention


def test_bounded_streaming_matches_full_local_attention():
    torch.manual_seed(31)
    config = CausalAttentionConfig(
        num_frames=18, tokens_per_frame=2, frames_per_chunk=3,
        local_attn_frames=6, sink_frames=2,
    )
    q = torch.randn(1, 36, 12)
    k = torch.randn(1, 36, 12)
    v = torch.randn(1, 36, 12)
    full = block_causal_attention(q, k, v, num_heads=3, config=config)
    cache = BoundedLayerKVCache()
    chunks = []
    for start in range(0, 36, config.chunk_tokens):
        end = start + config.chunk_tokens
        chunks.append(bounded_streaming_attention(
            q[:, start:end], k[:, start:end], v[:, start:end],
            num_heads=3, config=config, cache=cache,
        ))
    torch.testing.assert_close(full, torch.cat(chunks, dim=1), rtol=0, atol=0)
    assert cache.key.shape[1] <= (config.local_attn_frames + config.sink_frames) * config.tokens_per_frame
    assert cache.seen_tokens == config.sequence_tokens


if __name__ == "__main__":
    test_bounded_streaming_matches_full_local_attention()
    print("bounded causal attention test passed", flush=True)
