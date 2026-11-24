# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental schedule heuristic; not used by default.

Two token-count regions select the router, histogram and quantization geometry.
The native router K partition is a separate arithmetic contract; it must never
be selected from the geometry thresholds below.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendSchedule:
    """Only geometry options; no source, arithmetic, PDL or workspace ownership."""

    router_m: int
    router_n: int
    router_stages: int
    histogram_warps: int
    quant_segments: int


def propose_schedule(tokens: int) -> FrontendSchedule:
    """Return the two-region rule for the supported range T=64..8192.

    T=64 is a boundary point; decode dispatch is unchanged. The caller enforces
    serving phase, native arithmetic and PDL.
    """
    if type(tokens) is not int:
        raise TypeError("token count must be an integer, not bool or a device value")
    if not 64 <= tokens <= 8192:
        raise ValueError("schedule requires T in [64, 8192]; decode dispatch is unchanged")
    if tokens <= 4096:
        m, n, stages = 64, 64, 12
    else:
        m, n, stages = 128, 128, 6
    warps = 4 if tokens <= 128 else (8 if tokens <= 512 else 16)
    return FrontendSchedule(m, n, stages, warps, 3 if tokens <= 2048 else 1)
