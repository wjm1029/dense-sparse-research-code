from __future__ import annotations

import torch

from diffsynth.models.wan_video_dit import WanModel
from diffsynth.pipelines.wan_video import model_fn_wan_video
from wan22_ar_teacher_forcing import (
    ARTeacherForcingConfig,
    combine_rgb_and_history,
    install_ar_history_adapter,
    teacher_forcing_history,
)
from wan22_causal_attention import install_causal_attention


def _tiny_ar_model():
    model = WanModel(
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
    config = ARTeacherForcingConfig(num_frames=6, frames_per_chunk=3, latent_channels=4)
    causal = install_causal_attention(
        model, latent_shape=(4, 6, 4, 4), frames_per_chunk=3
    )
    ar = install_ar_history_adapter(model, causal, config)
    return model, config, ar


def _full(model, value, context):
    return model_fn_wan_video(
        dit=model,
        latents=value,
        timestep=torch.zeros(value.shape[0]),
        context=context,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        cfg_merge=False,
    )


def test_chunk_shift_has_no_current_or_future_target_leakage() -> None:
    config = ARTeacherForcingConfig(num_frames=6, frames_per_chunk=3, latent_channels=1)
    target = torch.arange(6.0).view(1, 1, 6, 1, 1)
    history = teacher_forcing_history(target, config)
    torch.testing.assert_close(
        history.flatten(), torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    )

    changed = target.clone()
    changed[:, :, 3:] += 1000
    changed_history = teacher_forcing_history(changed, config)
    # Changing current/future GT chunk cannot change either current chunk input.
    torch.testing.assert_close(history, changed_history, rtol=0, atol=0)


def test_zero_init_history_adapter_preserves_causal_checkpoint_output() -> None:
    torch.manual_seed(1)
    model, config, ar = _tiny_ar_model()
    rgb = torch.randn(1, 4, 6, 4, 4)
    history = torch.randn_like(rgb) * 100
    expected = ar.original_patchify(rgb)
    actual = model.patchify(combine_rgb_and_history(rgb, history, config))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    ar.assert_zero_initialized()


def test_history_adapter_receives_gradient_at_zero_initialization() -> None:
    torch.manual_seed(2)
    model, config, ar = _tiny_ar_model()
    model.train()
    rgb = torch.randn(1, 4, 6, 4, 4)
    target = torch.randn_like(rgb)
    context = torch.randn(1, 5, 12)
    output = _full(model, ar.teacher_forced_input(rgb, target), context)
    output[:, :, 3:].square().mean().backward()
    assert all(parameter.grad is not None for parameter in ar.history_parameters)
    assert any(torch.count_nonzero(parameter.grad) for parameter in ar.history_parameters)


def test_previous_target_chunk_changes_next_output_but_not_prefix() -> None:
    torch.manual_seed(3)
    model, config, ar = _tiny_ar_model()
    with torch.no_grad():
        ar.dit.lora_history_patch.weight.normal_(std=0.05)
    rgb = torch.randn(1, 4, 6, 4, 4)
    target_a = torch.randn_like(rgb)
    target_b = target_a.clone()
    target_b[:, :, :3] += 5
    context = torch.randn(1, 5, 12)
    with torch.no_grad():
        output_a = _full(model, ar.teacher_forced_input(rgb, target_a), context)
        output_b = _full(model, ar.teacher_forced_input(rgb, target_b), context)
    torch.testing.assert_close(output_a[:, :, :3], output_b[:, :, :3], rtol=0, atol=0)
    assert not torch.equal(output_a[:, :, 3:], output_b[:, :, 3:])


def test_full_teacher_forcing_matches_chunkwise_kv_streaming() -> None:
    torch.manual_seed(4)
    model, _, ar = _tiny_ar_model()
    with torch.no_grad():
        ar.dit.lora_history_patch.weight.normal_(std=0.02)
    rgb = torch.randn(1, 4, 6, 4, 4)
    target = torch.randn_like(rgb)
    context = torch.randn(1, 5, 12)
    timestep = torch.zeros(1)
    with torch.no_grad():
        full = _full(model, ar.teacher_forced_input(rgb, target), context)
        streamed = ar.streaming_teacher_forced(rgb, target, timestep, context)
    torch.testing.assert_close(full, streamed, rtol=2e-5, atol=2e-6)


def test_free_running_returns_complete_sequence_and_updates_all_caches() -> None:
    torch.manual_seed(5)
    model, config, ar = _tiny_ar_model()
    with torch.no_grad():
        ar.dit.lora_history_patch.weight.normal_(std=0.02)
    rgb = torch.randn(1, 4, 6, 4, 4)
    context = torch.randn(1, 5, 12)
    caches = ar.new_kv_cache()
    prediction = ar.free_running(
        rgb, torch.zeros(1), context, output_sign=-1.0, caches=caches
    )
    assert prediction.shape == rgb.shape
    expected_tokens = config.num_frames * 4  # 4x4 latent with 2x2 spatial patch
    assert all(cache.seen_tokens == expected_tokens for cache in caches)
    assert torch.isfinite(prediction).all()


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_"))
    for name, test in tests:
        print(f"RUN {name}", flush=True)
        test()
    print("all AR teacher-forcing tests passed", flush=True)
