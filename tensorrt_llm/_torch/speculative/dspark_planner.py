# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSpark verification-budget planner: how many tokens are worth verifying.

The scheduler maximizes *system* goodput::

    tau(n)   = num_gen_requests + sum of the n largest prefix-survival values
    Theta(n) = tau(n) / step_cost_ms(num_gen_requests + n)
    budget   = argmax_n Theta(n)

Invariants: take the GLOBAL argmax (marginal cost is a staircase, so ``Theta``
is not unimodal and a first-descent stop parks on the first shelf); cost is
per step, indexed by the TOTAL verified token count including the bonus
position (a request verifying ``L`` drafts submits ``L + 1`` tokens); and
``Theta`` is a ratio, so the non-trimmable ``bias + alpha(bs)`` terms do not
cancel -- they must be supplied, not guessed (see :class:`SpsCostTable`). The
planner runs on the host because its output selects a CUDA graph, a host
decision.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ...logger import logger

__all__ = [
    "SpsCostTable",
    "check_table_fingerprint",
    "compute_verify_token_budget",
    "budget_argmax_over_uniform_lens",
    "derive_verify_len_tiers",
    "total_verify_tokens",
]


def total_verify_tokens(num_requests: int, verify_len: int) -> int:
    """Tokens the target scores for ``num_requests`` each verifying ``verify_len`` drafts.

    One bonus/anchor token per request plus its drafted positions. This is the
    unit :class:`SpsCostTable` is indexed by; the single place it is computed so
    the planner and the tier derivation cannot drift apart.
    """
    return int(num_requests) * (int(verify_len) + 1)


@dataclass(frozen=True)
class SpsCostTable:
    """Measured decode step time as a function of total verified tokens.

    Attributes:
        token_counts: strictly increasing total-token breakpoints, in
            :func:`total_verify_tokens` units (bonus tokens included).
        step_time_ms: measured ``theta(M)`` at each breakpoint.
        fixed_overhead_ms: the ``bias`` term -- per-step cost that trimming
            cannot touch -- if not already inside ``step_time_ms``.
            Understating the non-trimmable terms makes the planner over-trim:
            ``tau / T`` is a ratio, so they do not cancel.
        batch_sizes / batch_overhead_ms: optional ``alpha(bs)`` breakpoints --
            the batch-size-dependent, non-trimmable part of the step.

    Lookup is a clamped linear interpolation between breakpoints; a flat shelf
    must be MEASURED as two breakpoints with equal values (a floor lookup makes
    tokens look free right before a riser and the argmax over-spends). Queries
    outside the measured range clamp to the end values; callers must bound the
    budget so the high clamp does not happen silently.
    """

    token_counts: Sequence[int]
    step_time_ms: Sequence[float]
    fixed_overhead_ms: float = 0.0
    batch_sizes: Sequence[int] = field(default_factory=tuple)
    batch_overhead_ms: Sequence[float] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.token_counts) != len(self.step_time_ms):
            raise ValueError(
                f"token_counts ({len(self.token_counts)}) and step_time_ms "
                f"({len(self.step_time_ms)}) must have the same length"
            )
        if not self.token_counts:
            raise ValueError("SpsCostTable requires at least one measured point")
        if any(b <= a for a, b in zip(self.token_counts, self.token_counts[1:])):
            raise ValueError("token_counts must be strictly increasing")
        if any(not math.isfinite(t) or t <= 0.0 for t in self.step_time_ms):
            raise ValueError("step_time_ms entries must be positive and finite")
        if not math.isfinite(self.fixed_overhead_ms) or self.fixed_overhead_ms < 0.0:
            raise ValueError("fixed_overhead_ms must be >= 0 and finite")
        if len(self.batch_sizes) != len(self.batch_overhead_ms):
            raise ValueError(
                f"batch_sizes ({len(self.batch_sizes)}) and batch_overhead_ms "
                f"({len(self.batch_overhead_ms)}) must have the same length"
            )
        if any(b <= a for a, b in zip(self.batch_sizes, self.batch_sizes[1:])):
            raise ValueError("batch_sizes must be strictly increasing")
        if any(not math.isfinite(t) or t < 0.0 for t in self.batch_overhead_ms):
            raise ValueError("batch_overhead_ms entries must be >= 0 and finite")

    def batch_overhead(self, num_requests: int) -> float:
        """``alpha(num_requests)`` -- 0.0 when the table has no batch axis."""
        if not self.batch_sizes:
            return 0.0
        return float(
            np.interp(
                float(num_requests),
                np.asarray(self.batch_sizes, dtype=np.float64),
                np.asarray(self.batch_overhead_ms, dtype=np.float64),
            ))

    def step_time(self, num_tokens: int, num_requests: int = 0) -> float:
        """``bias + alpha(num_requests) + theta(num_tokens)``, in ms."""
        return float(self.step_times(np.asarray([int(num_tokens)]), num_requests)[0])

    def step_times(self, num_tokens: np.ndarray, num_requests: int = 0) -> np.ndarray:
        """Vectorized :meth:`step_time` over a token-count array."""
        theta = np.interp(
            np.asarray(num_tokens, dtype=np.float64),
            np.asarray(self.token_counts, dtype=np.float64),
            np.asarray(self.step_time_ms, dtype=np.float64),
        )
        return theta + self.fixed_overhead_ms + self.batch_overhead(num_requests)

    @classmethod
    def flat(cls, step_time_ms: float = 1.0) -> "SpsCostTable":
        """Explicit "not profiled" marker: every extra token looks free, so
        the planner degenerates to verify-all. Callers should warn rather than
        silently ship this."""
        return cls(token_counts=(0,), step_time_ms=(float(step_time_ms),))

    @property
    def is_flat(self) -> bool:
        """Whether the trimmable term ``theta(M)`` is constant. ``bias`` and
        ``alpha(bs)`` are identical for every candidate length, so variation
        there cannot distinguish lengths."""
        return len(set(self.step_time_ms)) <= 1


def check_table_fingerprint(*, payload: dict, live: dict) -> None:
    """Refuse a cost table profiled on a different engine than this one.

    A mismatched table produces no error downstream -- the planner just trims
    to the wrong economics. Only the keys PRESENT in both dicts are compared,
    so a table may carry facts the consumer cannot see (for humans to check),
    and a table with no fingerprint loads with a warning rather than breaking
    existing deployments.
    """
    fp = (payload.get("_meta") or {}).get("engine") or payload.get("engine")
    if not fp:
        logger.warning(
            "DSpark cost table carries no engine fingerprint; cannot verify it "
            "was profiled on this configuration. Tables are engine-specific "
            "(same cell measured 23%% apart across two configs) -- regenerate "
            "with a current profiler to get load-time checking.")
        return
    def _facts_differ(a, b) -> bool:
        try:
            return float(a) != float(b)
        except (TypeError, ValueError):
            return str(a).upper() != str(b).upper()

    mismatched = {
        key: (fp[key], live[key])
        for key in sorted(set(fp) & set(live))
        if _facts_differ(fp[key], live[key])
    }
    if mismatched:
        detail = ", ".join(f"{k}: table={a!r} engine={b!r}"
                           for k, (a, b) in mismatched.items())
        raise ValueError(
            f"DSpark cost table was profiled on a different engine "
            f"configuration ({detail}). Its step times do not describe this "
            f"engine, so the planner would trim to the wrong economics. "
            f"Re-profile on this configuration, or remove "
            f"confidence_sps_table_path to run without trimming.")
    unchecked = sorted(set(fp) - set(live))
    if unchecked:
        logger.info(
            f"DSpark cost table fingerprint: verified "
            f"{sorted(set(fp) & set(live))}; not verifiable at this site "
            f"(check manually): "
            + ", ".join(f"{k}={fp[k]!r}" for k in unchecked))


def compute_verify_token_budget(
    *,
    survival: np.ndarray,
    num_gen_requests: int,
    cost_table: SpsCostTable,
    min_verify_len: int = 1,
    max_verify_len: Optional[int] = None,
    allowed_lens: Optional[Sequence[int]] = None,
) -> int:
    """Return the batch-wide verify-token budget (tokens *above* the floor).

    Args:
        survival: ``[bs, K]`` prefix-survival probabilities. Only the columns at
            or beyond ``min_verify_len`` are schedulable; earlier positions are
            already granted to every request.
        num_gen_requests: generating requests this step. Each contributes one
            guaranteed output token, which is the constant term of ``tau``.
        cost_table: measured step cost; see :class:`SpsCostTable`.
        min_verify_len / max_verify_len: per-request bounds, mirroring
            :class:`~.dspark_schedule.DSparkScheduleConfig`.
        allowed_lens: restrict the answer to budgets a captured tier can
            realise (``n in {bs*(t - min_verify_len)}``). Pass it whenever the
            result will be executed: any non-tier total is rounded UP to a
            captured bucket, pushing a budget chosen inside a cost shelf back
            over the riser it was avoiding.

    Returns:
        ``n`` maximizing ``tau(n) / step_cost(floor_tokens + n)``, in ``[0, N]``
        where ``N`` is the number of schedulable candidates -- restricted to the
        tier-aligned indices when ``allowed_lens`` is given.
    """
    if survival.ndim != 2:
        raise ValueError(f"survival must be [bs, K], got shape {survival.shape}")
    bs, block_size = survival.shape
    cap = min(int(max_verify_len or block_size), block_size)
    schedulable = max(cap - int(min_verify_len), 0)
    if bs == 0 or num_gen_requests <= 0 or schedulable <= 0:
        return 0

    surv = np.asarray(survival, dtype=np.float64)
    candidates = np.sort(surv[:, min_verify_len : min_verify_len + schedulable].reshape(-1))[::-1]
    if candidates.size == 0:
        return 0

    # tau(n): one bonus token per request, plus the floor positions' survival
    # (unbought but part of the yield -- omitting the constant moves the argmax
    # of a ratio), plus the n admitted candidates best-first.
    base = float(num_gen_requests) + float(surv[:, :min_verify_len].sum())
    tau = base + np.concatenate(([0.0], np.cumsum(candidates)))
    # Includes the bonus position: dropping it under-reports the batch by a
    # whole ``bs``.
    floor_tokens = total_verify_tokens(bs, min_verify_len)
    tokens = floor_tokens + np.arange(tau.size)
    theta = tau / cost_table.step_times(tokens, num_gen_requests)
    if allowed_lens is None:
        return int(np.argmax(theta))
    # Tier-aligned: the same theta curve, evaluated only at totals the executor
    # can land on -- n = bs*(t - min_verify_len) gives tokens = bs*(t+1)
    # = total_verify_tokens(bs, t).
    idx = sorted({
        int(bs) * (int(t) - int(min_verify_len))
        for t in allowed_lens
        if int(min_verify_len) <= int(t) <= cap
    })
    idx = [n for n in idx if 0 <= n < theta.size]
    if not idx:
        return 0
    return int(max(idx, key=lambda n: theta[n]))


def derive_verify_len_tiers(
    *,
    cost_table: SpsCostTable,
    num_requests: int,
    block_size: int,
    min_verify_len: int = 1,
    max_tiers: int = 3,
) -> List[int]:
    """Derive the verify lengths worth capturing a CUDA graph for.

    Within a measured flat cost shelf (adjacent breakpoints with equal values)
    ``Theta = tau / cost`` strictly increases, so at a fixed batch size the
    optimum sits on a shelf's *right edge*, never in its interior; this
    returns those edges, always including ``min_verify_len`` and the full
    block. On segments that genuinely rise the right-edge set is a
    slope-change heuristic, and the property does not survive across batch
    sizes (shelves live in token space) -- derive at the modal batch size and
    measure the residual loss at the others.

    Args:
        num_requests: batch size the tiers are derived for; tiers are a
            function of it, so a deployment that captures several batch sizes
            derives a set per batch size.
        max_tiers: cap on how many lengths to return. Each extra tier is
            another captured graph, which consumes memory that would otherwise
            be KV cache. The full block and ``min_verify_len`` are kept first.
    """
    lo, hi = int(min_verify_len), int(block_size)
    if num_requests <= 0 or hi < lo:
        return [lo]

    edges = set()
    for breakpoint_tokens in cost_table.token_counts[1:]:
        # Largest length whose total (num_requests * (L + 1), see
        # total_verify_tokens) still sits below this breakpoint; inverting
        # num_requests * L instead puts the edge one shelf too far right.
        length = (int(breakpoint_tokens) - 1) // int(num_requests) - 1
        if lo <= length <= hi:
            edges.add(length)
    # The last shelf extends past the final breakpoint, so the whole block is
    # always a right edge; the floor is always runnable.
    edges.update({lo, hi})

    tiers = sorted(edges)
    if len(tiers) <= max_tiers:
        return tiers
    # Keep the endpoints; spread the remaining slots over the interior edges.
    interior = tiers[1:-1]
    step = len(interior) / float(max_tiers - 2) if max_tiers > 2 else 0
    keep = [interior[int(i * step)] for i in range(max_tiers - 2)] if max_tiers > 2 else []
    return sorted({tiers[0], *keep, tiers[-1]})


def budget_argmax_over_uniform_lens(
    *,
    survival: np.ndarray,
    num_gen_requests: int,
    cost_table: SpsCostTable,
    allowed_lens: Sequence[int],
    min_verify_len: int = 1,
) -> int:
    """Pick the best verify length from a discrete set (e.g. captured graph tiers).

    Evaluates ``Theta`` directly at each runnable length: optimizing the
    continuous budget and rounding is not equivalent, since the rounded point
    can land on the far side of a cost riser.

    Args:
        allowed_lens: verify lengths with a captured graph. Values below
            ``min_verify_len`` are ignored.
    Returns:
        The chosen per-request verify length. Falls back to ``min_verify_len``
        when no candidate is admissible.
    """
    if survival.ndim != 2:
        raise ValueError(f"survival must be [bs, K], got shape {survival.shape}")
    bs, block_size = survival.shape
    lens: List[int] = sorted(
        {int(v) for v in allowed_lens if min_verify_len <= int(v) <= block_size}
    )
    if bs == 0 or num_gen_requests <= 0 or not lens:
        return int(min_verify_len)

    surv = np.asarray(survival, dtype=np.float64)
    best_len, best_theta = lens[0], -np.inf
    for length in lens:
        # Expected yield: the bonus token plus survival of every verified
        # position, floor included.
        tau = float(num_gen_requests) + float(surv[:, :length].sum())
        theta = tau / cost_table.step_time(total_verify_tokens(bs, length), num_gen_requests)
        if theta > best_theta:
            best_len, best_theta = length, theta
    return int(best_len)
