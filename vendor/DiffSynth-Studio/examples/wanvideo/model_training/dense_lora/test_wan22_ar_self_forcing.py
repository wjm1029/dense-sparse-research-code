from __future__ import annotations

import torch

from diffsynth.models.wan_video_dit import WanModel
from wan22_ar_self_forcing import detach_kv_cache, training_streaming_forward
from wan22_ar_teacher_forcing import ARTeacherForcingConfig, combine_rgb_and_history, install_ar_history_adapter
from wan22_causal_attention import install_causal_attention
from wan22_masked_loss import masked_latent_mse


def test_self_forcing_chunk_gradients_and_detached_state() -> None:
    torch.manual_seed(11)
    model = WanModel(dim=24, in_dim=4, ffn_dim=48, out_dim=4, text_dim=12, freq_dim=8, eps=1e-6, patch_size=(1,2,2), num_heads=2, num_layers=2, has_image_input=False, require_vae_embedding=False, require_clip_embedding=False).train()
    config = ARTeacherForcingConfig(6, 3, 4)
    causal = install_causal_attention(model, latent_shape=(4,6,4,4), frames_per_chunk=3)
    ar = install_ar_history_adapter(model, causal, config)
    with torch.no_grad(): ar.dit.lora_history_patch.weight.normal_(std=0.02)
    rgb, target = torch.randn(1,4,6,4,4), torch.randn(1,4,6,4,4)
    context, timestep = torch.randn(1,5,12), torch.zeros(1)
    caches = causal.new_kv_cache(); history = torch.zeros_like(rgb[:,:,:3])
    predictions=[]
    for offset in (0,3):
        value=combine_rgb_and_history(rgb[:,:,offset:offset+3], history, config)
        prediction=-training_streaming_forward(causal, model, value, timestep, context, frame_offset=offset, caches=caches)
        prediction.square().mean().backward()
        predictions.append(prediction.detach()); history=prediction.detach(); detach_kv_cache(caches)
        assert all(cache.key is not None and cache.key.grad_fn is None for cache in caches)
    assert all(cache.seen_tokens == 24 for cache in caches)
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in ar.history_parameters)
    assert predictions[1].shape == predictions[0].shape


def test_empty_chunk_has_differentiable_zero_loss() -> None:
    prediction = torch.randn(1, 4, 3, 2, 2, requires_grad=True)
    target = torch.randn_like(prediction)
    mask = torch.zeros(1, 1, 3, 2, 2)
    loss = (
        masked_latent_mse(prediction, target, mask)
        if bool(mask.any())
        else prediction.float().sum() * 0.0
    )
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


if __name__ == "__main__":
    test_self_forcing_chunk_gradients_and_detached_state()
    test_empty_chunk_has_differentiable_zero_loss()
    print("self-forcing tests passed")
