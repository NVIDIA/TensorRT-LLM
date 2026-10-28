# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy floors and systematic-shift policy, independent of launch configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .stats import binomial_se

#: Floor rules ``accuracy.floor.rule`` may name.
FLOOR_STATISTICAL = "statistical"


FLOOR_TOLERANCE = "tolerance"


FLOOR_RULES: tuple[str, ...] = (FLOOR_STATISTICAL, FLOOR_TOLERANCE)


#: ``k`` in ``anchor - k * SE`` when the task does not state one.
DEFAULT_FLOOR_K = 2.0


#: A unit gated over fewer members than this gets no cross-member check: one
#: member below the anchor by more than one SE happens about one time in six
#: by chance alone.
DEFAULT_SHIFT_MIN_MEMBERS = 2


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

    def describe(self) -> str:
        """``reference 92.08 - 2.00 x SE 0.75 (n=1319) = 90.59``, or the tolerance form."""
        se = f"SE {self.se:.2f} (n={self.samples})" if self.se is not None else "SE unknown"
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
    shift_min_members: int = DEFAULT_SHIFT_MIN_MEMBERS
    #: unit -> the reason a systematic shift on that unit is accepted.
    acknowledged: Mapping[str, str] | None = None


def resolve_floor_policy(accuracy: Mapping[str, Any]) -> FloorPolicy:
    """Read ``accuracy.floor`` (and ``accuracy.tolerance`` for the tolerance rule)."""
    block = accuracy.get("floor")
    block = block if isinstance(block, Mapping) else {}
    acknowledged = block.get("acknowledge")
    return FloorPolicy(
        rule=str(block.get("rule") or FLOOR_STATISTICAL),
        k=float(block.get("k", DEFAULT_FLOOR_K)),
        tolerance=float(accuracy.get("tolerance") or 0.0),
        shift_min_members=int(block.get("shift_min_members", DEFAULT_SHIFT_MIN_MEMBERS)),
        acknowledged={str(k): str(v) for k, v in acknowledged.items()}
        if isinstance(acknowledged, Mapping)
        else {},
    )


def floor_for(policy: FloorPolicy, reference: float, samples: int | None) -> Floor | None:
    """The floor ``policy`` puts under ``reference``; ``None`` if it needs an unknown count."""
    if policy.rule == FLOOR_TOLERANCE:
        return tolerance_floor(reference, policy.tolerance, samples)
    if samples is None:
        return None
    return statistical_floor(reference, samples, policy.k)


@dataclass(frozen=True)
class ShiftFinding:
    """Every member of a unit below the anchor by more than one SE."""

    unit: str | None
    reference: float
    #: member -> (measured, SE)
    members: Mapping[str, tuple[float, float]]
    acknowledged: str | None = None

    def describe(self) -> str:
        """One line naming every member, its score and how many SE it sits below."""
        shown = ", ".join(
            f"{member} {measured:.2f} ({(measured - self.reference) / se:+.1f} SE)"
            for member, (measured, se) in self.members.items()
        )
        head = (
            f"systematic shift on `{self.unit}`: every member is below the anchor "
            f"{self.reference:.2f} by more than 1 SE: {shown}"
        )
        if self.acknowledged:
            return f"{head}. Acknowledged in task.yaml: {self.acknowledged}"
        return (
            f"{head}. Each member passed its own floor, but a shift every member shares "
            f"is what a defect in shared code looks like. Delivery is blocked until it "
            f"is explained or acknowledged under `accuracy.floor.acknowledge.{self.unit}`"
        )


def systematic_shift(
    unit: str | None,
    readings: Mapping[str, tuple[float, float, float]],
    *,
    min_members: int = DEFAULT_SHIFT_MIN_MEMBERS,
    acknowledged: Mapping[str, str] | None = None,
) -> ShiftFinding | None:
    """A finding when every member scored more than one SE under its reference.

    ``readings`` maps member -> ``(measured, reference, se)``. ``None`` when
    fewer than ``min_members`` members were measured, or when any member is
    within one SE of (or above) its reference.
    """
    if len(readings) < max(1, min_members):
        return None
    below = {
        member: (measured, se)
        for member, (measured, reference, se) in readings.items()
        if se > 0 and measured < reference - se
    }
    if len(below) != len(readings):
        return None
    reference = sum(r for _, r, _ in readings.values()) / len(readings)
    reason = (acknowledged or {}).get(str(unit)) if unit is not None else None
    return ShiftFinding(unit=unit, reference=reference, members=below, acknowledged=reason)


__all__ = [
    "DEFAULT_FLOOR_K",
    "DEFAULT_SHIFT_MIN_MEMBERS",
    "FLOOR_RULES",
    "FLOOR_STATISTICAL",
    "FLOOR_TOLERANCE",
    "Floor",
    "FloorPolicy",
    "KNOWN_BENCHMARK_SAMPLES",
    "ShiftFinding",
    "floor_for",
    "resolve_floor_policy",
    "statistical_floor",
    "systematic_shift",
    "tolerance_floor",
]
