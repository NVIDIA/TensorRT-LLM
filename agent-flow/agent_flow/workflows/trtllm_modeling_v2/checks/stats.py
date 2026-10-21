# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The statistics the gates judge with. Standard library only.

The orchestrator runs on a login node whose interpreter has no numpy and no
scipy, so every quantity a verdict depends on is computed here from ``math``:

* :func:`binomial_se` and :func:`statistical_floor` -- an accuracy floor at
  ``anchor - k * SE``, where ``SE`` is the binomial standard error of a score
  measured on ``n`` samples. A fixed ``anchor - 5`` floor is about seven SE
  wide on GSM8K's 1319 questions, so a defect that moves every member by half
  a point passes it; ``k = 2`` puts the floor at about 1.5 points there.
* :func:`systematic_shift` -- every member of a unit below the anchor by more
  than one SE. Each member passing its own floor says nothing about this, and
  it is what a defect shared by every member looks like.
* :func:`paired_test` -- a paired t-test and a Wilcoxon signed-rank test over
  per-prompt values, for the acceptance gate's cross-tree and stock
  comparisons. A mean over a few prompts moves past a fixed delta when only
  the generated text changes; a paired test over enough prompts does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: Floor rules ``accuracy.floor.rule`` may name.
FLOOR_STATISTICAL = "statistical"
FLOOR_TOLERANCE = "tolerance"
FLOOR_RULES: tuple[str, ...] = (FLOOR_STATISTICAL, FLOOR_TOLERANCE)

#: ``k`` in ``anchor - k * SE`` when the task does not state one.
DEFAULT_FLOOR_K = 2.0

#: A unit gated over fewer members than this gets no cross-member check. One
#: member below the anchor by more than one SE happens about one time in six
#: by chance alone, so a finding raised on a single member would mostly be
#: noise; across two or more members the chance falls fast, and a shift that
#: every member shares is the signature of a defect in shared code.
DEFAULT_SHIFT_MIN_MEMBERS = 2

#: Paired tests ``accuracy.acceptance.regression.test`` may name.
TEST_T = "t"
TEST_WILCOXON = "wilcoxon"
PAIRED_TESTS: tuple[str, ...] = (TEST_T, TEST_WILCOXON)

#: One-sided significance level for a paired comparison.
DEFAULT_ALPHA = 0.05

#: Questions in the split ``trtllm-eval`` scores for each benchmark it ships.
#: Used for the standard error when neither the eval's own output nor the task
#: states the count. A benchmark missing here must declare ``accuracy.samples``.
KNOWN_BENCHMARK_SAMPLES: dict[str, int] = {
    "gsm8k": 1319,
    "mmlu": 14042,
    "mmmu": 900,
    "gpqa_diamond": 198,
    "gpqa_main": 448,
    "gpqa_extended": 546,
}


def binomial_se(score: float, samples: int) -> float:
    """The standard error, in points, of a percentage measured on ``samples`` items.

    ``score`` is on the 0-100 scale every TensorRT-LLM accuracy reference uses.
    The proportion is clamped half a sample away from 0 and 1, so a score of
    exactly 100 still has a nonzero error instead of a floor equal to it.
    """
    if samples < 1:
        raise ValueError(f"a standard error needs at least one sample, got {samples}")
    p = min(max(score / 100.0, 0.5 / samples), 1.0 - 0.5 / samples)
    return 100.0 * math.sqrt(p * (1.0 - p) / samples)


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


def normal_cdf(z: float) -> float:
    """``P(Z <= z)`` for the standard normal."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks with ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


#: Up to this many nonzero differences the Wilcoxon p-value is exact (the
#: permutation distribution of the signs, ties included); above it, the normal
#: approximation with tie and continuity corrections.
WILCOXON_EXACT_MAX_N = 60


def _wilcoxon_lower_p(diffs: Sequence[float]) -> tuple[float, float, int]:
    """``(W+, P(W+ <= observed), n)`` over the nonzero differences."""
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0, 0
    ranks = _ranks([abs(d) for d in nonzero])
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    if n <= WILCOXON_EXACT_MAX_N:
        doubled = [int(round(2 * r)) for r in ranks]
        counts = {0: 1}
        for r in doubled:
            nxt: dict[int, int] = dict(counts)
            for total, ways in counts.items():
                nxt[total + r] = nxt.get(total + r, 0) + ways
            counts = nxt
        observed = int(round(2 * w_plus))
        lower = sum(ways for total, ways in counts.items() if total <= observed)
        return w_plus, lower / float(2**n), n
    mean = n * (n + 1) / 4.0
    ties: dict[float, int] = {}
    for r in ranks:
        ties[r] = ties.get(r, 0) + 1
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t**3 - t for t in ties.values()) / 48.0
    if variance <= 0:
        return w_plus, 1.0, n
    z = (w_plus - mean + 0.5) / math.sqrt(variance)
    return w_plus, normal_cdf(z), n


@dataclass(frozen=True)
class PairedComparison:
    """``candidate`` against ``baseline``, paired item by item.

    The direction tested is a DECREASE: ``p_value`` is the one-sided
    probability, under no change, of a mean difference at least this far
    below zero. ``regressed`` holds only when the drop is both significant
    and at least ``min_effect`` -- significance alone flags a trivial drop on
    a large workload, and size alone flags noise on a small one.
    """

    n: int
    mean_baseline: float
    mean_candidate: float
    mean_diff: float
    #: Standard error of ``mean_diff``; ``nan`` below two pairs.
    se_diff: float
    t: float
    t_p: float
    wilcoxon_w: float
    wilcoxon_p: float
    test: str
    alpha: float
    min_effect: float
    up: int
    down: int
    equal: int

    @property
    def p_value(self) -> float:
        """The p-value of the configured test."""
        return self.wilcoxon_p if self.test == TEST_WILCOXON else self.t_p

    @property
    def significant(self) -> bool:
        """Whether the decrease is significant at ``alpha``."""
        return self.p_value < self.alpha

    @property
    def regressed(self) -> bool:
        """A significant decrease at least ``min_effect`` large."""
        return self.significant and -self.mean_diff >= self.min_effect

    def describe(self) -> str:
        """One line with both tests, the effect and the sign counts."""
        se = "n/a" if math.isnan(self.se_diff) else f"{self.se_diff:.3f}"
        return (
            f"paired over {self.n} prompt(s): mean {self.mean_candidate:.3f} vs "
            f"{self.mean_baseline:.3f} (diff {self.mean_diff:+.3f}, SE {se}); "
            f"t {self.t:+.2f} p={self.t_p:.3f}, Wilcoxon W+={self.wilcoxon_w:g} "
            f"p={self.wilcoxon_p:.3f}; judged by {self.test} at alpha {self.alpha:g} with "
            f"minimum effect {self.min_effect:g}; up/down/equal "
            f"{self.up}/{self.down}/{self.equal}"
        )


def paired_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    test: str = TEST_T,
    alpha: float = DEFAULT_ALPHA,
    min_effect: float = 0.0,
) -> PairedComparison:
    """Compare ``candidate`` against ``baseline`` item by item. Equal lengths."""
    if len(baseline) != len(candidate):
        raise ValueError(f"paired samples differ in length: {len(baseline)} vs {len(candidate)}")
    if test not in PAIRED_TESTS:
        raise ValueError(f"unknown paired test {test!r}; expected one of {PAIRED_TESTS}")
    n = len(baseline)
    diffs = [c - b for b, c in zip(baseline, candidate)]
    mean_diff = sum(diffs) / n if n else 0.0
    if n >= 2:
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        se_diff = math.sqrt(variance / n)
        if se_diff > 0:
            t = mean_diff / se_diff
            t_p = student_t_cdf(t, n - 1)
        else:
            t = 0.0 if mean_diff == 0 else math.copysign(math.inf, mean_diff)
            t_p = 0.0 if mean_diff < 0 else 1.0
    else:
        se_diff, t, t_p = float("nan"), 0.0, 1.0
    w_plus, w_p, _ = _wilcoxon_lower_p(diffs)
    return PairedComparison(
        n=n,
        mean_baseline=sum(baseline) / n if n else 0.0,
        mean_candidate=sum(candidate) / n if n else 0.0,
        mean_diff=mean_diff,
        se_diff=se_diff,
        t=t,
        t_p=t_p,
        wilcoxon_w=w_plus,
        wilcoxon_p=w_p,
        test=test,
        alpha=alpha,
        min_effect=min_effect,
        up=sum(1 for d in diffs if d > 0),
        down=sum(1 for d in diffs if d < 0),
        equal=sum(1 for d in diffs if d == 0),
    )


def median(values: Sequence[float]) -> float:
    """The median; raises on an empty sequence."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median of nothing")
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_FLOOR_K",
    "DEFAULT_SHIFT_MIN_MEMBERS",
    "FLOOR_RULES",
    "FLOOR_STATISTICAL",
    "FLOOR_TOLERANCE",
    "Floor",
    "FloorPolicy",
    "KNOWN_BENCHMARK_SAMPLES",
    "PAIRED_TESTS",
    "PairedComparison",
    "ShiftFinding",
    "TEST_T",
    "TEST_WILCOXON",
    "WILCOXON_EXACT_MAX_N",
    "binomial_se",
    "floor_for",
    "median",
    "normal_cdf",
    "paired_test",
    "regularized_incomplete_beta",
    "resolve_floor_policy",
    "statistical_floor",
    "student_t_cdf",
    "systematic_shift",
    "tolerance_floor",
]
