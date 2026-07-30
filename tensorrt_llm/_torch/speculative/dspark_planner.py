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

The scheduler maximizes *system* goodput, not per-request speed::

    tau(n)   = num_gen_requests + sum of the n largest prefix-survival values
    Theta(n) = tau(n) / step_cost_ms(num_gen_requests + n)
    budget   = argmax_n Theta(n)

``tau(n)`` is the expected number of tokens the step emits if it verifies ``n``
tokens beyond the per-request floor: every generating request contributes one
bonus token for free, and each admitted draft position contributes its survival
probability. Dividing by the step cost turns that into expected tokens per
millisecond.

Two things are easy to get wrong here:

**Take the global argmax, not the first descent.** The marginal cost of a verify
token is a staircase -- flat while a token slots into the current kernel waves,
then a jump when it starts a new one -- so ``Theta`` is not unimodal. A greedy
loop that stops at the first non-improvement parks at the end of the first
shelf and leaves most of the win behind. Both SGLang and vLLM PR #47808 scan the
whole curve; so do we, and :mod:`tests` pins it against a brute-force scan.

**Cost is per step, not per token.** ``SpsCostTable`` therefore stores measured
step time against the *total* verified token count, and interpolates nothing
across a riser.

The planner runs on the host: its output selects a CUDA graph, and graph
selection is a host decision in TensorRT-LLM. Keeping it here (rather than on
device) is deliberate -- a device-side budget would have to be copied back
before the batch could be shaped, which is exactly the sync we are avoiding.
"""

import bisect
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "SpsCostTable",
    "compute_verify_token_budget",
    "budget_argmax_over_uniform_lens",
    "derive_verify_len_tiers",
]

# Fixed per-step overhead folded into every cost lookup (sampling, bookkeeping,
# launch latency). Matches vLLM PR #47808's ``_FIXED_OVERHEAD_MS``.
_FIXED_OVERHEAD_MS = 1.0


@dataclass(frozen=True)
class SpsCostTable:
    """Measured decode step time as a function of total verified tokens.

    Attributes:
        token_counts: strictly increasing total-token breakpoints.
        step_time_ms: measured step time at each breakpoint.

    Lookup is a floor (staircase) lookup, never an interpolation: between two
    measured points the cost is genuinely flat until a new kernel wave starts,
    and smoothing that away is what makes a planner over-spend right before a
    riser. Queries above the largest breakpoint clamp to the last entry, and the
    caller is expected to bound the budget so this does not happen silently.
    """

    token_counts: Sequence[int]
    step_time_ms: Sequence[float]

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
        if any(t <= 0.0 for t in self.step_time_ms):
            raise ValueError("step_time_ms entries must be positive")

    def step_time(self, num_tokens: int) -> float:
        """Step time (ms) for a step that verifies ``num_tokens`` tokens in total."""
        idx = bisect.bisect_right(self.token_counts, int(num_tokens)) - 1
        idx = min(max(idx, 0), len(self.step_time_ms) - 1)
        return float(self.step_time_ms[idx]) + _FIXED_OVERHEAD_MS

    def step_times(self, num_tokens: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`step_time`."""
        counts = np.asarray(self.token_counts)
        idx = np.clip(np.searchsorted(counts, num_tokens, side="right") - 1, 0, len(counts) - 1)
        return np.asarray(self.step_time_ms, dtype=np.float64)[idx] + _FIXED_OVERHEAD_MS

    @classmethod
    def flat(cls, step_time_ms: float = 1.0) -> "SpsCostTable":
        """A cost model with no staircase at all.

        Useful as an explicit "I have not profiled this machine" marker. With a
        flat cost every extra token is free, so the planner will always spend the
        entire budget -- i.e. it degenerates to verify-all. Callers should warn
        rather than silently ship this.
        """
        return cls(token_counts=(0,), step_time_ms=(float(step_time_ms),))

    @property
    def is_flat(self) -> bool:
        return len(set(self.step_time_ms)) <= 1


def compute_verify_token_budget(
    *,
    survival: np.ndarray,
    num_gen_requests: int,
    cost_table: SpsCostTable,
    min_verify_len: int = 1,
    max_verify_len: Optional[int] = None,
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

    Returns:
        ``n`` maximizing ``tau(n) / step_cost(floor_tokens + n)``, in ``[0, N]``
        where ``N`` is the number of schedulable candidates.
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

    # tau(n) = expected tokens emitted when n candidates are admitted:
    #   * one bonus token per generating request, emitted regardless;
    #   * the floor positions, which every request verifies for free -- their
    #     survival is part of the yield even though no budget buys them. Leaving
    #     this out shifts the whole curve and can move the argmax, since
    #     argmax (C + f(n)) / g(n) depends on C.
    #   * the n admitted candidates, best-first.
    base = float(num_gen_requests) + float(surv[:, :min_verify_len].sum())
    tau = base + np.concatenate(([0.0], np.cumsum(candidates)))
    floor_tokens = bs * int(min_verify_len)
    total_tokens = floor_tokens + np.arange(tau.size)
    theta = tau / cost_table.step_times(total_tokens)
    return int(np.argmax(theta))


def derive_verify_len_tiers(
    *,
    cost_table: SpsCostTable,
    num_requests: int,
    block_size: int,
    min_verify_len: int = 1,
    max_tiers: int = 3,
) -> List[int]:
    """Derive the verify lengths worth capturing a CUDA graph for.

    Restricting K to a handful of captured values sounds like it must cost
    something versus a freely-chosen K. **At a fixed batch size it does not**:

        Within one cost shelf the step time is constant while ``tau`` strictly
        increases with each admitted token, so ``Theta = tau / cost`` is strictly
        increasing across the shelf. The optimum therefore always sits at a
        shelf's *right edge* -- never in its interior. For that batch size, a
        tier set of the shelf right edges contains the continuous optimum
        exactly.

    **The zero-loss property does not survive across batch sizes.** The cost
    shelves live in token space, so a shelf's right edge in *length* space is
    ``(breakpoint - 1) // batch_size`` -- a function of the batch size. A
    deployment captures graphs for many batch sizes but can only capture one
    ladder, so a single tier set cannot sit on every batch size's right edges.
    Deriving tiers is more robust than hardcoding a pair like ``{1, 5}``, but it
    is still a lossy discretization once the batch size moves; the residual loss
    has to be measured, not assumed away. Callers that capture one ladder should
    derive it at the steady-state modal batch size and check the worst-case loss
    across the others.

    This returns the right-edge length for each shelf the given batch size can
    reach, always including ``min_verify_len`` and the full block.

    Args:
        num_requests: batch size the tiers are derived for. Tiers are a function
            of it (total tokens = ``num_requests * length``), so a deployment
            that captures several batch sizes derives a set per batch size.
        max_tiers: cap on how many lengths to return. Each extra tier is another
            captured graph, and captured graphs consume memory that would
            otherwise be KV cache -- so this is a real budget, not a formality.
            The full block and ``min_verify_len`` are kept first.
    """
    lo, hi = int(min_verify_len), int(block_size)
    if num_requests <= 0 or hi < lo:
        return [lo]

    edges = set()
    for breakpoint_tokens in cost_table.token_counts[1:]:
        # Largest length whose total token count still sits on the shelf below
        # this breakpoint.
        length = (int(breakpoint_tokens) - 1) // int(num_requests)
        if lo <= length <= hi:
            edges.add(length)
    # The last shelf extends past the final breakpoint, so the whole block is
    # always a right edge; the floor is always runnable.
    edges.update({lo, hi})

    tiers = sorted(edges)
    if len(tiers) <= max_tiers:
        return tiers
    # Keep the endpoints, then spread the remaining slots over the interior
    # edges rather than clustering them at one end of the curve.
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

    TensorRT-LLM can only run one draft length per step, and only lengths that
    have a captured CUDA graph. Optimizing over the continuous budget and then
    rounding is not the same as optimizing over what is actually runnable: the
    rounded-down point can sit on the far side of a cost riser. This evaluates
    ``Theta`` directly at each achievable length and returns the best one.

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
        # Every request verifies ``length`` positions: the expected yield is the
        # bonus token plus the survival of *every* verified position (including
        # the floor ones), summed over the batch.
        tau = float(num_gen_requests) + float(surv[:, :length].sum())
        theta = tau / cost_table.step_time(bs * length)
        if theta > best_theta:
            best_len, best_theta = length, theta
    return int(best_len)
