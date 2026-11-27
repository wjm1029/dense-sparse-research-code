"""Masked latent objectives shared by Wan2.2 training and CPU tests."""

from __future__ import annotations

import torch


def masked_latent_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Channel-shared weighted MSE, normalized over only valid latent elements."""
    if prediction.shape != target.shape or mask.ndim != prediction.ndim or mask.shape[1] != 1:
        raise ValueError(f"Invalid masked MSE shapes: {prediction.shape}, {target.shape}, {mask.shape}")
    weights = mask.float()
    if not torch.isfinite(weights).all() or bool((weights < 0).any()) or bool((weights > 1).any()):
        raise ValueError("Loss mask must be finite and within [0, 1]")
    denominator = weights.sum() * prediction.shape[1]
    if float(denominator.detach()) <= 0:
        raise ValueError("Loss mask contains no valid latent support")
    return (((prediction.float() - target.float()).square()) * weights).sum() / denominator
