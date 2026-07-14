# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared metadata names for VisualGen serving metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.output import VisualGenMetrics

SERVER_TIMING_HEADER = "Server-Timing"
VISUAL_GEN_DENOISE_TIMING = "denoise"
VISUAL_GEN_GENERATION_TIMING = "generation"


def _server_timing_metric(name: str, duration_seconds: float) -> str:
    # Server-Timing ``dur`` is in milliseconds; VisualGenMetrics stores seconds.
    return f"{name};dur={duration_seconds * 1000:.6f}"


def build_visual_gen_timing_headers(
    metrics: Optional["VisualGenMetrics"],
) -> dict[str, str]:
    """Build standard Server-Timing headers for VisualGen engine timings."""
    if metrics is None:
        return {}
    return {
        SERVER_TIMING_HEADER: ", ".join(
            [
                _server_timing_metric(VISUAL_GEN_GENERATION_TIMING, metrics.generation),
                _server_timing_metric(VISUAL_GEN_DENOISE_TIMING, metrics.denoise),
            ]
        )
    }
