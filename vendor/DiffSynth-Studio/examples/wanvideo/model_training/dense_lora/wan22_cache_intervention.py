"""Training-free static-scene cache-intervention schedule for long AR inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StaticCacheInterventionConfig:
    enter_threshold: float = 0.5
    exit_threshold: float = 1.0
    confirm_chunks: int = 3
    max_hold_chunks: int = 20
    scene_cut_threshold: float = 20.0

    def validate(self) -> None:
        if not 0.0 <= self.enter_threshold < self.exit_threshold:
            raise ValueError("Require 0 <= enter_threshold < exit_threshold")
        if self.scene_cut_threshold <= self.exit_threshold:
            raise ValueError("scene_cut_threshold must exceed exit_threshold")
        if self.confirm_chunks < 1 or self.max_hold_chunks < 1:
            raise ValueError("confirm_chunks and max_hold_chunks must be positive")


def _last_gray(frame_group: np.ndarray) -> np.ndarray:
    frame = Image.fromarray(frame_group[-1]).convert("L").resize((64, 36), Image.Resampling.BILINEAR)
    return np.asarray(frame, dtype=np.float32)


def build_static_cache_intervention_schedule(
    frames: np.ndarray,
    num_chunks: int,
    config: StaticCacheInterventionConfig,
) -> tuple[list[bool], dict[str, object]]:
    """Detect sustained static input and return one intervention flag per chunk."""
    config.validate()
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < num_chunks:
        raise ValueError(f"Expected THWC RGB frames spanning {num_chunks} chunks, got {frames.shape}")
    groups = np.array_split(frames, num_chunks, axis=0)
    grays = [_last_gray(group) for group in groups]
    schedule = [False] * num_chunks
    diffs: list[float | None] = [None]
    events: list[dict[str, object]] = []
    static_streak = 0
    hold_chunks = 0
    active = False
    for chunk in range(1, num_chunks):
        mad = float(np.mean(np.abs(grays[chunk] - grays[chunk - 1])))
        diffs.append(mad)
        exited_this_chunk = False
        if active:
            reason = None
            if mad >= config.scene_cut_threshold:
                reason = "scene_cut"
            elif mad >= config.exit_threshold:
                reason = "motion_resumed"
            elif hold_chunks >= config.max_hold_chunks:
                reason = "max_hold"
            if reason is not None:
                active = False
                static_streak = 0
                hold_chunks = 0
                exited_this_chunk = True
                events.append({"chunk": chunk, "event": "exit", "reason": reason, "mad": mad})
            else:
                hold_chunks += 1
        if not active and not exited_this_chunk:
            if mad < config.enter_threshold:
                static_streak += 1
                if static_streak >= config.confirm_chunks:
                    active = True
                    hold_chunks = 1
                    events.append({"chunk": chunk, "event": "enter", "mad": mad})
            else:
                static_streak = 0
        schedule[chunk] = active
    report = {
        "config": {
            "enter_threshold": config.enter_threshold,
            "exit_threshold": config.exit_threshold,
            "confirm_chunks": config.confirm_chunks,
            "max_hold_chunks": config.max_hold_chunks,
            "scene_cut_threshold": config.scene_cut_threshold,
        },
        "intervened_chunks": int(sum(schedule)),
        "activations": int(sum(event["event"] == "enter" for event in events)),
        "events": events,
        "chunk_mad": diffs,
    }
    return schedule, report
