# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy floors, independent of launch configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .stats import binomial_se

#: Floor rules ``accuracy.floor.rule`` may name.
FLOOR_STATISTICAL = "statistical"


FLOOR_TOLERANCE = "tolerance"


FLOOR_RULES: tuple[str, ...] = (FLOOR_STATISTICAL, FLOOR_TOLERANCE)


#: ``k`` in ``anchor - k * SE`` when the task does not state one.
DEFAULT_FLOOR_K = 2.0


#: Questions in the split ``trtllm-eval`` scores for each benchmark it ships,
#: for the standard error when neither the eval's own output nor the task
#: states the count.
KNOWN_BENCHMARK_SAMPLES: dict[str, int] = {
    "gsm8k": 1319,
    "mmlu": 14042,
    "mmmu": 900,
    "gpqa_diamond": 198,
    "gpqa_main": 448,
    "gpqa_extended": 546,
}


@dataclass(frozen=True)
class Floor:
    """One accuracy floor and how it was derived."""

    value: float
    reference: float
    #: :data:`FLOOR_STATISTICAL` or :data:`FLOOR_TOLERANCE`.
    rule: str
    #: The binomial SE at the reference; ``None`` only when the sample count is unknown.
    se: float | None
    samples: int | None
    #: ``k`` under the statistical rule.
    k: float | None = None
    #: Points under the tolerance rule.
    tolerance: float | None = None
    #: The reference is the unit's own score, measured as this one is (``anchor: relative``):
    #: the SE is that of the difference of two measurements.
    relative: bool = False

    def describe(self) -> str:
        """``reference 92.08 - 2.00 x SE 0.75 (n=1319) = 90.59``, or the tolerance form."""
        se = f"SE {self.se:.2f} (n={self.samples})" if self.se is not None else "SE unknown"
        if self.relative:
            what = f"the unit's own score {self.reference:.4f}"
            if self.rule == FLOOR_STATISTICAL:
                return f"{what} - {self.k:.2f} x {se} of the difference = {self.value:.4f}"
            return f"{what} - tol {self.tolerance:.2f} = {self.value:.4f} (fixed tolerance)"
        if self.rule == FLOOR_STATISTICAL:
            return f"reference {self.reference:.4f} - {self.k:.2f} x {se} = {self.value:.4f}"
        return (
            f"reference {self.reference:.4f} - tol {self.tolerance:.2f} = {self.value:.4f} "
            f"(fixed tolerance; {se})"
        )


def statistical_floor(reference: float, samples: int, k: float = DEFAULT_FLOOR_K) -> Floor:
    """``reference - k * SE``, with the SE taken at the reference score."""
    se = binomial_se(reference, samples)
    return Floor(
        value=reference - k * se,
        reference=reference,
        rule=FLOOR_STATISTICAL,
        se=se,
        samples=samples,
        k=k,
    )


def relative_floor(reference: float, samples: int, k: float = DEFAULT_FLOOR_K) -> Floor:
    """``reference - k * SE`` where ``reference`` is the unit's own score, measured as this one is.

    Both scores carry their binomial error, so the SE is that of their
    difference: sqrt(2) times one score's.
    """
    se = math.sqrt(2) * binomial_se(reference, samples)
    return Floor(
        value=reference - k * se,
        reference=reference,
        rule=FLOOR_STATISTICAL,
        se=se,
        samples=samples,
        k=k,
        relative=True,
    )


def tolerance_floor(reference: float, tolerance: float, samples: int | None = None) -> Floor:
    """``reference - tolerance``. The SE is still computed when the count is known."""
    se = binomial_se(reference, samples) if samples else None
    return Floor(
        value=reference - tolerance,
        reference=reference,
        rule=FLOOR_TOLERANCE,
        se=se,
        samples=samples,
        tolerance=tolerance,
    )


@dataclass(frozen=True)
class FloorPolicy:
    """What ``accuracy.floor`` says, resolved."""

    rule: str = FLOOR_STATISTICAL
    k: float = DEFAULT_FLOOR_K
    tolerance: float = 0.0


def resolve_floor_policy(accuracy: Mapping[str, Any]) -> FloorPolicy:
    """Read ``accuracy.floor`` (and ``accuracy.tolerance`` for the tolerance rule)."""
    block = accuracy.get("floor")
    block = block if isinstance(block, Mapping) else {}
    return FloorPolicy(
        rule=str(block.get("rule") or FLOOR_STATISTICAL),
        k=float(block.get("k", DEFAULT_FLOOR_K)),
        tolerance=float(accuracy.get("tolerance") or 0.0),
    )


def floor_for(
    policy: FloorPolicy, reference: float, samples: int | None, relative: bool = False
) -> Floor | None:
    """The floor ``policy`` puts under ``reference``; ``None`` if it needs an unknown count.

    ``relative``: ``reference`` is the unit's own score (:func:`relative_floor`).
    """
    if policy.rule == FLOOR_TOLERANCE:
        floor = tolerance_floor(reference, policy.tolerance, samples)
        return Floor(**{**floor.__dict__, "relative": relative})
    if samples is None:
        return None
    if relative:
        return relative_floor(reference, samples, policy.k)
    return statistical_floor(reference, samples, policy.k)


__all__ = [
    "DEFAULT_FLOOR_K",
    "FLOOR_RULES",
    "FLOOR_STATISTICAL",
    "FLOOR_TOLERANCE",
    "Floor",
    "FloorPolicy",
    "KNOWN_BENCHMARK_SAMPLES",
    "floor_for",
    "relative_floor",
    "resolve_floor_policy",
    "statistical_floor",
    "tolerance_floor",
]
