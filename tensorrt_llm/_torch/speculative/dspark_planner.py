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
step time against the *total* verified token count, interpolated between
breakpoints (see the class docstring for why the old floor lookup was wrong).

**Total verified tokens includes the bonus position.** A request that verifies
``L`` drafted positions submits ``L + 1`` tokens to the target -- the bonus
token it already holds, plus the ``L`` drafts. So a batch of ``bs`` requests at
uniform length ``L`` costs ``step_time(bs * (L + 1))``, not ``step_time(bs * L)``.
The two differ by a whole ``bs``, which is easily a shelf's width.

**The cost is two-dimensional, and only one dimension is trimmable.** SGLang
publishes ``T(bs, K) = bias + alpha(bs) + theta(M)``: a fixed cost, a
batch-size-dependent cost (the draft pass, weight movement -- untouched by
trimming), and the target's verify-token cost ``theta(M)``, which is the only
term a smaller ``M`` reduces. ``Theta = tau / T`` is a *ratio*, so the
non-trimmable terms are not a constant that cancels -- getting them wrong moves
the argmax directly. They therefore have to be supplied, not guessed:
:class:`SpsCostTable` carries ``fixed_overhead_ms`` for ``bias`` and an optional
``batch_sizes`` / ``batch_overhead_ms`` staircase for ``alpha(bs)``.

The planner runs on the host: its output selects a CUDA graph, and graph
selection is a host decision in TensorRT-LLM. Keeping it here (rather than on
device) is deliberate -- a device-side budget would have to be copied back
before the batch could be shaped, which is exactly the sync we are avoiding.
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
        token_counts: strictly increasing total-token breakpoints. "Total" means
            what :func:`total_verify_tokens` computes -- bonus tokens included.
        step_time_ms: measured ``theta(M)`` at each breakpoint.
        fixed_overhead_ms: the ``bias`` term -- per-step cost that trimming
            cannot touch (sampling, bookkeeping, launch latency), if it is not
            already inside ``step_time_ms``. Defaults to 0.0 on the assumption
            that a profiled table measured whole steps; set it only when the
            table isolates the verify term.
        batch_sizes / batch_overhead_ms: optional ``alpha(bs)`` breakpoints --
            the batch-size-dependent, *non-trimmable* part of the step (the
            draft pass, weight movement). Interpolated on ``batch_sizes``.

    Getting the non-trimmable terms wrong is not harmless. The planner maximizes
    ``tau / T``, a ratio, so ``bias + alpha(bs)`` does not cancel: understating
    it makes every verified token look proportionally more expensive than it is
    and the planner over-trims. On a large-MoE deployment ``alpha(bs)`` is tens
    of milliseconds while ``theta(M)`` is a few, so a hardcoded guess here would
    dominate the decision.

    Lookup is a clamped LINEAR INTERPOLATION between breakpoints, matching the
    additive-table consumer this was ported from (SGLang's
    ``_additive_step_time_tensor``). A floor lookup here is wrong on a sparse
    table: it bills every total below the next breakpoint at the previous
    breakpoint's price, making tokens look free right before a riser, and the
    argmax then over-spends -- measured live as the planner buying the full
    block on ~95% of decisions because its cost ratio between tiers collapsed
    below the survival ratio. A table that wants a flat shelf must MEASURE the shelf:
    two breakpoints with equal values. Queries outside the measured range
    clamp to the end values; the caller is expected to bound the budget so the
    high clamp does not happen silently.
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
        """A cost model with no staircase at all.

        Useful as an explicit "I have not profiled this machine" marker. With a
        flat cost every extra token is free, so the planner will always spend the
        entire budget -- i.e. it degenerates to verify-all. Callers should warn
        rather than silently ship this.
        """
        return cls(token_counts=(0,), step_time_ms=(float(step_time_ms),))

    @property
    def is_flat(self) -> bool:
        """Whether the *trimmable* term is constant.

        Only ``theta(M)`` matters here: ``bias`` and ``alpha(bs)`` are the same
        for every candidate length, so a table that varies only in those cannot
        distinguish a cheap verified token from an expensive one either.
        """
        return len(set(self.step_time_ms)) <= 1


def check_table_fingerprint(*, payload: dict, live: dict) -> None:
    """Refuse a cost table profiled on a different engine than this one.

    The table's numbers are only meaningful for the engine configuration they
    were measured on: the same (bs=64, M=384) cell measured 49.5 ms on a
    max_batch_size=256 engine and 61.0 ms on a max_batch_size=64 engine -- 23%
    apart against a 0.04-1.2% run-to-run noise floor -- and a table taken on
    CUTLASS MoE kernels was nearly loaded to plan MegaMoE steps. Neither
    mistake produces an error anywhere downstream; the planner just trims to
    the wrong economics.

    Only the keys PRESENT in both dicts are compared, so a table may carry
    facts the consumer cannot see (image hash, geometry) for humans to check,
    and an old table with no fingerprint loads with a warning rather than
    breaking existing deployments.
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
        allowed_lens: restrict the answer to budgets that a tier in this ladder
            can actually realise, i.e. ``n in {bs*(t - min_verify_len)}``. Pass
            it whenever the result will be executed, and omit it only to ask
            what an unconstrained scheduler would have wanted.

            The unrestricted answer is not runnable. The captured graphs are
            ``bs*(t+1)`` tokens for ``t`` in the ladder
            (``model_engine.ragged_verify_token_buckets``), and any other total
            is rounded UP to one of them -- so a budget chosen strictly inside a
            cost shelf, which is the only reason to choose one, is pushed back
            over the riser it was avoiding. The planner would then buy tokens
            priced on the cheap shelf and run on the expensive one. Realising a
            fine budget honestly would need a graph per token count, which is
            tens of MB of metadata each, taken out of KV cache.

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

    # tau(n) = expected tokens emitted when n candidates are admitted:
    #   * one bonus token per generating request, emitted regardless;
    #   * the floor positions, which every request verifies for free -- their
    #     survival is part of the yield even though no budget buys them. Leaving
    #     this out shifts the whole curve and can move the argmax, since
    #     argmax (C + f(n)) / g(n) depends on C.
    #   * the n admitted candidates, best-first.
    base = float(num_gen_requests) + float(surv[:, :min_verify_len].sum())
    tau = base + np.concatenate(([0.0], np.cumsum(candidates)))
    # Tokens actually submitted to the target at the floor: every request's
    # bonus position plus its floor draft positions. Dropping the bonus term
    # under-reports the batch by a whole ``bs``, which is easily a shelf wide.
    floor_tokens = total_verify_tokens(bs, min_verify_len)
    tokens = floor_tokens + np.arange(tau.size)
    theta = tau / cost_table.step_times(tokens, num_gen_requests)
    if allowed_lens is None:
        return int(np.argmax(theta))
    # Tier-aligned: same theta curve, evaluated only where the executor can
    # land. tokens(n) at n = bs*(t - min_verify_len) is bs*(min+1) + bs*(t-min)
    # = bs*(t+1), which is exactly total_verify_tokens(bs, t) -- the same cost
    # grid budget_argmax_over_uniform_lens uses. The only thing that changes is
    # the numerator: tau here is the sum of the top n survivals, which is what
    # schedule_verify_lens_topk actually collects, rather than the sum of a
    # uniform column, which nothing executes.
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

    Restricting K to a handful of captured values sounds like it must cost
    something versus a freely-chosen K. **At a fixed batch size, on a table
    with genuine shelves, it does not**:

        Within one cost shelf the step time is constant while ``tau`` strictly
        increases with each admitted token, so ``Theta = tau / cost`` is strictly
        increasing across the shelf. The optimum therefore always sits at a
        shelf's *right edge* -- never in its interior. For that batch size, a
        tier set of the shelf right edges contains the continuous optimum
        exactly.

    Since the consumer became an interpolation, "shelf" means a measured flat
    segment (adjacent breakpoints with equal values), not the gap between two
    sparse samples. On a table whose segments genuinely rise -- like the live
    GB300 one -- the optimum can sit inside a segment and the right-edge set
    is a slope-change heuristic rather than an exact cover; the residual loss
    has to be measured. (The old floor consumer made every gap LOOK like a
    shelf, which both made this theorem vacuously "exact" and priced a
    1512-token step at the 768-token price, disabling trimming entirely.)

    **The zero-loss property does not survive across batch sizes.** The cost
    shelves live in token space, so a shelf's right edge in *length* space is
    ``(breakpoint - 1) // batch_size - 1`` -- a function of the batch size. A
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
        # this breakpoint. Total tokens for length L is num_requests * (L + 1)
        # (see total_verify_tokens), so invert that, not num_requests * L --
        # otherwise the derived edge sits one shelf too far right.
        length = (int(breakpoint_tokens) - 1) // int(num_requests) - 1
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
        theta = tau / cost_table.step_time(total_verify_tokens(bs, length), num_gen_requests)
        if theta > best_theta:
            best_len, best_theta = length, theta
    return int(best_len)
