# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving-layer VisualGen metrics: metric names, timings flattener, and header formatter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.output import VisualGenMetrics

SERVER_TIMING_HEADER = "Server-Timing"
VISUAL_GEN_DENOISE_TIMING = "denoise"
VISUAL_GEN_GENERATION_TIMING = "generation"
VISUAL_GEN_TOTAL_TIMING = "total"


def build_visual_gen_server_timings(
    metrics: Optional["VisualGenMetrics"] = None,
    total: Optional[float] = None,
) -> Dict[str, float]:
    """Flatten engine ``generation``/``denoise`` + serve ``total`` into one timings dict (seconds).

    Engine metrics carry no ``total`` — the route measures it — so it is passed
    separately here. Absent values are omitted. Mirrors how the LLM serve path
    merges engine + server timings into one record.
    """
    timings: Dict[str, float] = {}
    if metrics is not None:
        timings[VISUAL_GEN_GENERATION_TIMING] = metrics.generation
        timings[VISUAL_GEN_DENOISE_TIMING] = metrics.denoise
    if total is not None:
        timings[VISUAL_GEN_TOTAL_TIMING] = total
    return timings


def _server_timing_metric(name: str, duration_seconds: float) -> str:
    # Server-Timing ``dur`` is in milliseconds; timings are stored in seconds.
    return f"{name};dur={duration_seconds * 1000:.6f}"


def build_visual_gen_timing_headers(timings: Optional[Dict[str, float]]) -> dict[str, str]:
    """Format a timings dict as a ``Server-Timing`` header (``{}`` if empty)."""
    if not timings:
        return {}
    parts = [_server_timing_metric(name, dur) for name, dur in timings.items() if dur is not None]
    return {SERVER_TIMING_HEADER: ", ".join(parts)} if parts else {}
