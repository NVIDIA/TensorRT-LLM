# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared numerical statistics using math alone: binomial error and paired t-tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

#: One-sided significance level for a paired comparison.
DEFAULT_ALPHA = 0.05


def binomial_se(score: float, samples: int) -> float:
    """The standard error, in points, of a percentage measured on ``samples`` items.

    ``score`` is on the 0-100 scale. The proportion is clamped half a sample
    away from 0 and 1, so a score of exactly 100 still has a nonzero error.
    """
    if samples < 1:
        raise ValueError(f"a standard error needs at least one sample, got {samples}")
    p = min(max(score / 100.0, 0.5 / samples), 1.0 - 0.5 / samples)
    return 100.0 * math.sqrt(p * (1.0 - p) / samples)


# ------------------------------------------------------------------ paired tests


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` for ``a, b > 0`` and ``0 <= x <= 1``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """``P(T <= t)`` for Student's t with ``df`` degrees of freedom."""
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    tail = 0.5 * regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))
    return tail if t < 0 else 1.0 - tail


@dataclass(frozen=True)
class PairedComparison:
    """``candidate`` against ``baseline``, paired item by item.

    The direction tested is a DECREASE: ``p_value`` is the one-sided
    probability, under no change, of a mean difference at least this far
    below zero. ``regressed`` holds only when the drop is both significant
    and at least ``min_effect``.
    """

    n: int
    mean_baseline: float
    mean_candidate: float
    mean_diff: float
    #: Standard error of ``mean_diff``; ``nan`` below two pairs.
    se_diff: float
    t: float
    #: The one-sided p-value of the paired t-test.
    p_value: float
    alpha: float
    min_effect: float
    up: int
    down: int
    equal: int

    @property
    def significant(self) -> bool:
        """Whether the decrease is significant at ``alpha``."""
        return self.p_value < self.alpha

    @property
    def regressed(self) -> bool:
        """A significant decrease at least ``min_effect`` large."""
        return self.significant and -self.mean_diff >= self.min_effect

    def describe(self) -> str:
        """One line with the test, the effect and the sign counts."""
        se = "n/a" if math.isnan(self.se_diff) else f"{self.se_diff:.3f}"
        return (
            f"paired over {self.n} prompt(s): mean {self.mean_candidate:.3f} vs "
            f"{self.mean_baseline:.3f} (diff {self.mean_diff:+.3f}, SE {se}); "
            f"t {self.t:+.2f} p={self.p_value:.3f}; judged at alpha {self.alpha:g} with "
            f"minimum effect {self.min_effect:g}; up/down/equal "
            f"{self.up}/{self.down}/{self.equal}"
        )


def paired_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
    min_effect: float = 0.0,
) -> PairedComparison:
    """Compare ``candidate`` against ``baseline`` item by item. Equal lengths."""
    if len(baseline) != len(candidate):
        raise ValueError(f"paired samples differ in length: {len(baseline)} vs {len(candidate)}")
    n = len(baseline)
    diffs = [c - b for b, c in zip(baseline, candidate)]
    mean_diff = sum(diffs) / n if n else 0.0
    if n >= 2:
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        se_diff = math.sqrt(variance / n)
        if se_diff > 0:
            t = mean_diff / se_diff
            p_value = student_t_cdf(t, n - 1)
        else:
            t = 0.0 if mean_diff == 0 else math.copysign(math.inf, mean_diff)
            p_value = 0.0 if mean_diff < 0 else 1.0
    else:
        se_diff, t, p_value = float("nan"), 0.0, 1.0
    return PairedComparison(
        n=n,
        mean_baseline=sum(baseline) / n if n else 0.0,
        mean_candidate=sum(candidate) / n if n else 0.0,
        mean_diff=mean_diff,
        se_diff=se_diff,
        t=t,
        p_value=p_value,
        alpha=alpha,
        min_effect=min_effect,
        up=sum(1 for d in diffs if d > 0),
        down=sum(1 for d in diffs if d < 0),
        equal=sum(1 for d in diffs if d == 0),
    )


__all__ = [
    "DEFAULT_ALPHA",
    "PairedComparison",
    "binomial_se",
    "paired_test",
    "regularized_incomplete_beta",
    "student_t_cdf",
]
