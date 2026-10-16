# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""History policy for VisualGen client-latency regression checks."""

from collections.abc import Mapping

_GENERATION_METRIC_PATHS = {
    "openai-videos": "gen_latency",
    "openai-images": "e2e_latency",
    "openai-image-edits": "e2e_latency",
}


def get_generation_metric_path(backend: str) -> str:
    """Return the benchmark JSON object used for generation latency.

    Args:
        backend: The backend reported by the VisualGen benchmark.

    Raises:
        ValueError: If the backend has no defined measurement path.
    """
    if backend not in _GENERATION_METRIC_PATHS:
        raise ValueError(f"Unsupported VisualGen benchmark backend: {backend!r}")
    return _GENERATION_METRIC_PATHS[backend]


def get_generation_timing_source(backend: str) -> str:
    """Describe the client measurement selected by get_generation_metric_path.

    Args:
        backend: The backend reported by the VisualGen benchmark.
    """
    return f"client_{get_generation_metric_path(backend)}"


def transform_visual_gen_history(record: Mapping[str, object]) -> dict[str, object] | None:
    """Select compatible timing data and discard inherited baseline values.

    Only records carrying the backend's current timing source enter regression inputs.
    Historical client results need source-verified metadata backfill before
    reuse; dates and commit identities alone do not establish the timing window.

    Even a client-timing row may have inherited a server-window baseline from
    CI. Drop baseline values before selecting the latest baseline/threshold
    record, while preserving threshold overrides and the original DB record.

    Args:
        record: A historical perf result from the database.

    Returns:
        A copy with compatible regression inputs, or None for excluded timing
        sources. Records from other runtimes are copied unchanged.
    """
    if record.get("s_runtime") != "visual_gen":
        return dict(record)
    backend = record.get("s_backend")
    if not isinstance(backend, str) or backend not in _GENERATION_METRIC_PATHS:
        return None
    if record.get("s_generation_timing_source") != get_generation_timing_source(backend):
        return None
    return {key: value for key, value in record.items() if not key.startswith("d_baseline_")}
