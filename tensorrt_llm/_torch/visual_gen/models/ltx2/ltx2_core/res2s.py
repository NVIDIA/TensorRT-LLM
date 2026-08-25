# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""res_2s coefficients used by the LTX-2.3 native sampler."""

from __future__ import annotations

import math


def phi(j: int, neg_h: float) -> float:
    """Compute phi_j(z) for z = -h."""

    if abs(neg_h) < 1e-10:
        return 1.0 / math.factorial(j)

    remainder = sum(neg_h**k / math.factorial(k) for k in range(j))
    return (math.exp(neg_h) - remainder) / (neg_h**j)


def get_res2s_coefficients(
    h: float,
    phi_cache: dict[tuple[int, float], float],
    c2: float = 0.5,
) -> tuple[float, float, float]:
    """Return midpoint res_2s Runge-Kutta coefficients."""

    def get_phi(j: int, neg_h: float) -> float:
        cache_key = (j, neg_h)
        if cache_key not in phi_cache:
            phi_cache[cache_key] = phi(j, neg_h)
        return phi_cache[cache_key]

    phi_1_c2 = get_phi(1, -h * c2)
    a21 = c2 * phi_1_c2

    phi_2_full = get_phi(2, -h)
    b2 = phi_2_full / c2
    b1 = get_phi(1, -h) - b2

    return a21, b1, b2
