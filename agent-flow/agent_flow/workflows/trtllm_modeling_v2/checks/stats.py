# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The binomial standard error of a score, using math alone."""

from __future__ import annotations

import math


def binomial_se(score: float, samples: int) -> float:
    """The standard error, in points, of a percentage measured on ``samples`` items.

    ``score`` is on the 0-100 scale. The proportion is clamped half a sample
    away from 0 and 1, so a score of exactly 100 still has a nonzero error.
    """
    if samples < 1:
        raise ValueError(f"a standard error needs at least one sample, got {samples}")
    p = min(max(score / 100.0, 0.5 / samples), 1.0 - 0.5 / samples)
    return 100.0 * math.sqrt(p * (1.0 - p) / samples)


__all__ = [
    "binomial_se",
]
