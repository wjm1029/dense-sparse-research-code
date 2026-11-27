"""Unit checks for the training-free static cache-intervention detector."""

from __future__ import annotations

import numpy as np

from wan22_cache_intervention import (
    StaticCacheInterventionConfig,
    build_static_cache_intervention_schedule,
)


def main() -> None:
    chunks = []
    values = [0, 0, 0, 0, 0, 8, 16, 16, 16, 16]
    for value in values:
        chunks.append(np.full((2, 24, 32, 3), value, dtype=np.uint8))
    frames = np.concatenate(chunks, axis=0)
    schedule, report = build_static_cache_intervention_schedule(
        frames,
        len(values),
        StaticCacheInterventionConfig(
            enter_threshold=0.5,
            exit_threshold=1.0,
            confirm_chunks=2,
            max_hold_chunks=10,
            scene_cut_threshold=20.0,
        ),
    )
    expected = [False, False, True, True, True, False, False, False, True, True]
    if schedule != expected:
        raise AssertionError((schedule, expected))
    if report["activations"] != 2 or report["intervened_chunks"] != 5:
        raise AssertionError(report)
    held_schedule, held_report = build_static_cache_intervention_schedule(
        np.zeros((16, 24, 32, 3), dtype=np.uint8),
        8,
        StaticCacheInterventionConfig(
            enter_threshold=0.5,
            exit_threshold=1.0,
            confirm_chunks=1,
            max_hold_chunks=2,
            scene_cut_threshold=20.0,
        ),
    )
    if held_schedule != [False, True, True, False, True, True, False, True]:
        raise AssertionError((held_schedule, held_report))
    event_chunks = [(event["chunk"], event["event"]) for event in held_report["events"]]
    if len(event_chunks) != len(set(event_chunks)):
        raise AssertionError("max-hold exit and re-entry must not happen in the same chunk")
    print({"status": "PASS", "schedule": schedule, "events": report["events"]})


if __name__ == "__main__":
    main()
