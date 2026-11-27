"""OPSD-style delayed clean-prefix replacement for Wan2.2 KV caches.

The student cache always follows generated history.  The teacher cache keeps
the newest generated chunk, then replaces that chunk with a clean-target
encoding only after the following chunk has been evaluated.  This mirrors the
causal context policy in OPSD-V without depending on its Wan2.1 implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def extract_latest_chunk(caches: Sequence, chunk_tokens: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Copy the newest generated chunk before an in-place full-cache replay."""

    result = []
    for layer_index, cache in enumerate(caches):
        if cache.key is None or cache.value is None or cache.seen_tokens < chunk_tokens:
            raise ValueError(f"layer {layer_index} cannot provide the latest chunk")
        result.append((
            cache.key[:, -chunk_tokens:].detach().clone(),
            cache.value[:, -chunk_tokens:].detach().clone(),
        ))
    return result


def truncate_full_cache_(caches: Sequence, seen_tokens: int) -> None:
    """Roll a full-prefix cache back while retaining per-layer storage views."""

    for layer_index, cache in enumerate(caches):
        if getattr(cache, "positions", None) is not None:
            raise ValueError("truncate_full_cache_ does not accept bounded caches")
        if cache.seen_tokens < seen_tokens:
            raise ValueError(f"layer {layer_index} is shorter than rollback point")
        if seen_tokens == 0:
            cache.key = cache.value = None
        else:
            cache.key = cache.key[:, :seen_tokens]
            cache.value = cache.value[:, :seen_tokens]
        cache.seen_tokens = seen_tokens


def append_cached_chunk_(caches: Sequence, chunks, expected_start: int) -> None:
    """Reattach a generated chunk whose K/V was computed before GT replacement."""

    if len(caches) != len(chunks):
        raise ValueError("cache and saved chunk layer counts differ")
    for layer_index, (cache, (key, value)) in enumerate(zip(caches, chunks)):
        if cache.seen_tokens != expected_start:
            raise ValueError(f"layer {layer_index} has the wrong append position")
        cache.key = key if cache.key is None else torch.cat((cache.key, key), dim=1)
        cache.value = value if cache.value is None else torch.cat((cache.value, value), dim=1)
        cache.seen_tokens += key.shape[1]

def clone_prefix(
    caches: Sequence, seen_tokens: int
) -> list:
    """Clone a detached cache prefix for a side-effect-free replay forward."""

    if seen_tokens < 0:
        raise ValueError("seen_tokens cannot be negative")
    result = []
    for layer_index, cache in enumerate(caches):
        if cache.seen_tokens < seen_tokens:
            raise ValueError(
                f"layer {layer_index} has {cache.seen_tokens} tokens, "
                f"cannot clone prefix {seen_tokens}"
            )
        if cache.key is None or cache.value is None or cache.positions is None:
            if seen_tokens == 0:
                result.append(type(cache)(seen_tokens=0))
                continue
            raise ValueError(f"layer {layer_index} has no tensors")
        keep = cache.positions < seen_tokens
        result.append(
            type(cache)(
                key=cache.key[:, keep].detach().clone(),
                value=cache.value[:, keep].detach().clone(),
                positions=cache.positions[keep].detach().clone(),
                seen_tokens=seen_tokens,
            )
        )
    return result


def replace_chunk_from_replay(
    destination: Sequence,
    replay: Sequence,
    *,
    start_token: int,
    chunk_tokens: int,
) -> None:
    """Overwrite one teacher-cache chunk using a clean-prefix replay result."""

    if len(destination) != len(replay):
        raise ValueError("destination and replay layer counts differ")
    if start_token < 0 or chunk_tokens <= 0:
        raise ValueError("invalid replacement span")
    end_token = start_token + chunk_tokens
    for layer_index, (dst, src) in enumerate(zip(destination, replay)):
        if any(value is None for value in (dst.key, dst.value, dst.positions, src.key, src.value, src.positions)):
            raise ValueError(f"layer {layer_index} has incomplete cache tensors")
        if dst.seen_tokens < end_token:
            raise ValueError(f"layer {layer_index} destination is shorter than replacement")
        if src.seen_tokens != end_token:
            raise ValueError(
                f"layer {layer_index} replay must end exactly at {end_token}, "
                f"got {src.seen_tokens}"
            )
        dst_indices = torch.nonzero(
            (dst.positions >= start_token) & (dst.positions < end_token), as_tuple=False
        ).flatten()
        src_indices = torch.nonzero(
            (src.positions >= start_token) & (src.positions < end_token), as_tuple=False
        ).flatten()
        if dst_indices.numel() != chunk_tokens or src_indices.numel() != chunk_tokens:
            raise ValueError(f"layer {layer_index} replacement span was evicted or incomplete")
        if not torch.equal(dst.positions[dst_indices], src.positions[src_indices]):
            raise ValueError(f"layer {layer_index} replacement positions differ")
        dst.key.index_copy_(1, dst_indices, src.key.index_select(1, src_indices))
        dst.value.index_copy_(1, dst_indices, src.value.index_select(1, src_indices))


def update_ema_(
    ema: dict[str, torch.Tensor],
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    decay: float,
) -> None:
    """Update a device-resident EMA for the trainable post-training weights."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    with torch.no_grad():
        for name, parameter in named_parameters:
            if name not in ema:
                raise KeyError(f"EMA is missing parameter {name}")
            ema[name].lerp_(parameter.detach(), 1.0 - decay)


def swap_trainable_parameters_(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    replacement: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Swap trainable values in place and return a device-resident backup."""

    backup: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, parameter in named_parameters:
            if name not in replacement:
                raise KeyError(f"replacement is missing parameter {name}")
            backup[name] = parameter.detach().clone()
            parameter.copy_(replacement[name])
    return backup
