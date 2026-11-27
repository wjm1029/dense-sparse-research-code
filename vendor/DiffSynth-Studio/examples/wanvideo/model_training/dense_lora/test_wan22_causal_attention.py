from __future__ import annotations

import torch

from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video import model_fn_wan_video
from wan22_causal_attention import (
    CausalAttentionConfig,
    LayerKVCache,
    block_causal_attention,
    install_causal_attention,
    streaming_attention,
)


def test_attention_visibility_matrix_is_frame_block_causal() -> None:
    """Recover the effective attention matrix and compare every token pair.

    Zero Q/K makes attention uniform over exactly the keys presented to SDPA.
    Identity V then turns each output row into its effective attention weights,
    so this checks history, same-frame, and future visibility directly without
    relying on a separately implemented mask helper.
    """

    for num_frames, tokens_per_frame in ((2, 1), (3, 2), (4, 5)):
        config = CausalAttentionConfig(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            frames_per_chunk=1,
        )
        sequence_tokens = config.sequence_tokens
        q = torch.zeros(1, sequence_tokens, sequence_tokens)
        k = torch.zeros_like(q)
        v = torch.eye(sequence_tokens).unsqueeze(0)
        effective_weights = block_causal_attention(q, k, v, num_heads=1, config=config)[0]

        expected = torch.zeros_like(effective_weights)
        for query_token in range(sequence_tokens):
            query_frame = query_token // tokens_per_frame
            allowed_end = (query_frame + 1) * tokens_per_frame
            expected[query_token, :allowed_end] = 1.0 / allowed_end

            # Tokens within a frame are bidirectional, including later spatial tokens.
            same_frame = slice(query_frame * tokens_per_frame, allowed_end)
            assert torch.all(effective_weights[query_token, same_frame] > 0)
            # Every key belonging to a future frame must be exactly invisible.
            assert torch.count_nonzero(effective_weights[query_token, allowed_end:]) == 0

        torch.testing.assert_close(effective_weights, expected, rtol=1e-6, atol=1e-7)

        # Positive control: ordinary full attention exposes every future token.
        full_attention = torch.full_like(expected, 1.0 / sequence_tokens)
        if num_frames > 1:
            assert not torch.equal(effective_weights, full_attention)


def test_future_chunks_do_not_change_prefix() -> None:
    torch.manual_seed(7)
    config = CausalAttentionConfig(num_frames=4, tokens_per_frame=3, frames_per_chunk=1)
    q = torch.randn(1, 12, 8)
    k = torch.randn(1, 12, 8)
    v = torch.randn(1, 12, 8)
    reference = block_causal_attention(q, k, v, num_heads=2, config=config)
    changed = block_causal_attention(
        q,
        torch.cat((k[:, :6], torch.randn_like(k[:, 6:]) * 100), dim=1),
        torch.cat((v[:, :6], torch.randn_like(v[:, 6:]) * 100), dim=1),
        num_heads=2,
        config=config,
    )
    torch.testing.assert_close(reference[:, :6], changed[:, :6], rtol=0, atol=0)


def test_prefix_gradient_is_zero_for_future_keys_and_values() -> None:
    torch.manual_seed(9)
    config = CausalAttentionConfig(num_frames=4, tokens_per_frame=3, frames_per_chunk=1)
    q = torch.randn(1, 12, 8, requires_grad=True)
    k = torch.randn(1, 12, 8, requires_grad=True)
    v = torch.randn(1, 12, 8, requires_grad=True)

    # A loss over frames 0..1 may use keys/values from frames 0..1, never 2..3.
    prefix_tokens = 2 * config.tokens_per_frame
    output = block_causal_attention(q, k, v, num_heads=2, config=config)
    output[:, :prefix_tokens].square().sum().backward()

    assert torch.count_nonzero(k.grad[:, prefix_tokens:]) == 0
    assert torch.count_nonzero(v.grad[:, prefix_tokens:]) == 0
    assert torch.count_nonzero(q.grad[:, prefix_tokens:]) == 0
    assert torch.count_nonzero(k.grad[:, :prefix_tokens]) > 0
    assert torch.count_nonzero(v.grad[:, :prefix_tokens]) > 0


def test_full_prefix_matches_streaming_kv() -> None:
    torch.manual_seed(11)
    config = CausalAttentionConfig(num_frames=6, tokens_per_frame=2, frames_per_chunk=2)
    q = torch.randn(2, 12, 12)
    k = torch.randn(2, 12, 12)
    v = torch.randn(2, 12, 12)
    full = block_causal_attention(q, k, v, num_heads=3, config=config)
    cache = LayerKVCache()
    streamed = []
    for start in range(0, 12, config.chunk_tokens):
        end = start + config.chunk_tokens
        streamed.append(
            streaming_attention(
                q[:, start:end], k[:, start:end], v[:, start:end],
                num_heads=3, config=config, cache=cache,
            )
        )
    torch.testing.assert_close(full, torch.cat(streamed, dim=1), rtol=0, atol=0)


def _tiny_wan() -> WanModel:
    return WanModel(
        dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=12,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=2,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
    ).eval()


def test_install_preserves_state_dict_and_model_streaming_matches_full() -> None:
    torch.manual_seed(19)
    model = _tiny_wan()
    keys_before = tuple(model.state_dict())
    controller = install_causal_attention(model, latent_shape=(4, 4, 4, 4))
    assert tuple(model.state_dict()) == keys_before

    latent = torch.randn(1, 4, 4, 4, 4)
    timestep = torch.zeros(1)
    context = torch.randn(1, 5, 12)
    with torch.no_grad():
        full = model_fn_wan_video(
            dit=model,
            latents=latent,
            timestep=timestep,
            context=context,
        )
        caches = controller.new_kv_cache()
        chunks = [
            controller.streaming_forward(
                model,
                latent[:, :, frame : frame + 1],
                timestep,
                context,
                frame_offset=frame,
                caches=caches,
            )
            for frame in range(4)
        ]
    torch.testing.assert_close(full, torch.cat(chunks, dim=2), rtol=2e-5, atol=2e-6)


def test_causal_attention_backpropagates() -> None:
    torch.manual_seed(23)
    config = CausalAttentionConfig(num_frames=3, tokens_per_frame=2)
    q = torch.randn(1, 6, 8, requires_grad=True)
    k = torch.randn(1, 6, 8, requires_grad=True)
    v = torch.randn(1, 6, 8, requires_grad=True)
    block_causal_attention(q, k, v, num_heads=2, config=config).square().mean().backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (q, k, v))


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_"))
    for name, test in tests:
        print(f"RUN {name}", flush=True)
        test()
    print("all causal attention tests passed", flush=True)
