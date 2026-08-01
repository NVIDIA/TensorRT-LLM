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
"""Measure the DSpark SPS cost table this deployment actually has.

Why this exists
---------------
:class:`~.dspark_planner.SpsCostTable` is the only thing that tells the verify
planner a long verification costs more than a short one. Without a measured
table ``SpsCostTable.flat()`` is installed, ``is_flat`` is true, and
``DSparkVerifyPlanner.decide_verify_lens`` returns ``None`` on every step
(``stats["fallback_flat_cost"]``) -- ragged verification is then a no-op that
still produces correct output at baseline accuracy, i.e. it fails silently.
``TLLM_DSPARK_FORCE_VERIFY_LENS=1`` unblocks *correctness* testing by bypassing
that gate, but it rotates a fixed ladder and knows nothing about cost, so it
cannot answer "is trimming profitable here". Only a measured table can.

Run it::

    python -m tensorrt_llm._torch.speculative.dspark_sps_profiler \\
        --model /models/DeepSeek-V4-Pro-DSpark --tp-size 8 --ep-size 8 \\
        --enable-attention-dp --batch-sizes 8,16,32,64,128 \\
        --max-draft-len 5 --out sps_table.json

and then point the serving config at the result::

    speculative_config:
      decoding_type: DSpark
      enable_confidence_scheduling: true
      confidence_sps_table_path: sps_table.json
      confidence_verify_len_tiers: [...]   # from _meta, see "Tiers" below

What the sweep measures, and how the three terms are separated
--------------------------------------------------------------
The planner's cost model is additive::

    T(bs, M) = bias + alpha(bs) + theta(M),   M = bs * (verify_len + 1)

``theta(M)`` is the *only* term trimming reduces; ``bias`` and ``alpha(bs)`` are
paid whatever the planner picks. Because the planner maximizes ``tau / T``, a
**ratio**, the non-trimmable terms do not cancel: drop them and every verified
token looks proportionally more expensive than it is, and the planner
over-trims. So all three have to be measured, not guessed.

The sweep is a cross product of batch size ``bs`` and *uniform* verify length
``L``. Each cell is one whole steady-state decode step time ``S(bs, L)``, and
sits at ``M = bs * (L + 1)`` (:func:`~.dspark_planner.total_verify_tokens` --
the bonus token is included; getting that wrong is a whole ``bs`` of error,
easily a shelf's width). The cross product is what makes the terms separable:

* **theta(M) is isolated along a row.** Fix ``bs`` and move ``L``: ``bias`` and
  ``alpha(bs)`` are identical for every cell in the row, so the *differences*
  along it are pure ``theta``.
* **alpha(bs) is isolated along a column.** The grid supplies same-``M``,
  different-``bs`` collisions for free -- ``(bs, L)`` and ``(2*bs, L')`` land on
  the same ``M`` whenever ``L + 1 == 2 * (L' + 1)``, e.g. ``(16, 3)`` and
  ``(32, 1)`` both give ``M = 64``. Across such a collision ``theta`` cancels
  and the difference is pure ``alpha``.
  This is also why the batch-size ladder must be *connected* through those
  collisions; :func:`fit_additive_cost_model` refuses a sweep that is not,
  because each disconnected component would carry its own arbitrary offset and
  the emitted ``theta`` staircase would splice two different scales together.
* **The remaining global constant is fixed by extrapolating theta to M -> 0.**
  An additive fit only determines ``theta`` and ``alpha`` up to one shared
  constant. We shift so that ``theta`` linearly extrapolates to zero at
  ``M = 0``, which is exactly the definition that makes ``bias + alpha(bs)``
  the step's ``M``-independent floor -- the quantity the planner needs.
* **bias vs alpha(bs) is then a presentation split.** Only their sum enters
  :meth:`SpsCostTable.step_time`, so we report ``bias = min_bs (bias +
  alpha(bs))`` -- the floor present even at the smallest profiled batch -- and
  give the rest to ``alpha``. Both are emitted so that a reader can see which
  part is batch-driven.

The emitted table is therefore in the **decomposed** encoding: ``step_time_ms``
is ``theta(M)`` *only*. Do not also read it as whole-step time and re-add the
overheads -- that double-counts the floor, inflates ``T``, flattens the relative
differences between candidate lengths and drags the argmax back to verify-all.

Holding verification uniform
----------------------------
The run pins ``TLLM_DSPARK_RAGGED_VERIFY_MODE=static`` and clears
``TLLM_DSPARK_FORCE_VERIFY_LENS``. Under ``static`` the planner still chooses a
single batch-wide length, but the cost table it sees is the flat default, whose
``is_flat`` gate short-circuits to ``max_tier`` *before* any confidence is read
-- so the length is deterministically ``max_draft_len`` on every step and
``M = bs * (max_draft_len + 1)`` exactly. If the scheduler were allowed to trim
mid-sweep, each cell would be measuring a different ``M`` than the one it is
filed under, i.e. a moving target.

That determinism is also why the ``L`` axis costs one engine build per value:
nothing in the runtime pins a *uniform* verify length below the full block
(``TLLM_DSPARK_FORCE_VERIFY_LENS`` produces a ragged rotation, and there is no
analogue of SGLang's ``dspark_force_budget_frac``), so ``L`` is swept by
rebuilding with ``max_draft_len = block_size = L``. Use ``--samples-out`` /
``--fit-only`` to split a long sweep across jobs and refit without a GPU.

**Known bias of that choice:** shrinking the block also shrinks the *draft*
pass, whose cost is roughly linear in the block length, while at deployment the
block is always drafted in full and only verification is trimmed. The draft
model is a few layers against a full-size target, so the leak is a small
fraction of the measured slope -- but it is a positive fraction, so the emitted
``theta`` slope is mildly overstated and the planner is biased towards
*over*-trimming. Prefer profiling only the lengths the ladder can actually
select, and treat a marginal trim decision as noise.

Acceptance is pinned
--------------------
``--pin-acceptance`` (on by default) sets
``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` to a tiny positive value, which
makes every request commit exactly its bonus token every step. ``0.0`` means
"override disabled", hence the epsilon rather than a clean zero. This matters
because acceptance drives how fast the KV cache grows, and KV length drives
attention cost: without pinning, a long-block cell would accumulate context
faster than a short-block one and the difference would be misattributed to
``theta(M)``.

Statistics
----------
Only *aligned* steps count: a step where every attention-DP rank reports the
same generation-request count and no context requests. The first
``--warmup-steps`` aligned steps of each cell are discarded (allocator growth,
graph warmup, autotuner) and the cell reports the **median** of the rest --
never the mean. Decode step time has a long right tail from allocator and
scheduling noise, and a mean turns that tail into a fake riser, which is exactly
the failure a staircase cost model amplifies.

Refusals
--------
The tool would rather emit nothing than emit a table that looks profiled and
is not:

* :class:`FlatCostTableError` -- the curve does not move, so ``is_flat`` is true
  (or the total step time varies by less than the noise floor). The planner
  would treat the file as unprofiled and keep verifying everything.
* :class:`InertCostTableError` -- the curve moves, but not between any two tiers
  the planner may select at any profiled batch size, so the argmax can never
  leave ``max_tier``. That is a non-flat table that trims nothing, and unlike a
  flat one it does *not* trip ``fallback_flat_cost``, so it would look healthy.

Tiers
-----
``_meta.recommended_confidence_verify_len_tiers`` is derived from the measured
shelves at the modal batch size, but **nothing reads it from this file**:
``DSparkDecodingConfig.verify_len_tiers`` always returns a non-empty list, so
``dspark.py``'s ``derive_verify_len_tiers`` fallback never runs. Copy the value
into ``confidence_verify_len_tiers`` by hand. The config is the right single
source anyway -- ``_get_graphs_to_capture`` and ``ragged_verify_token_buckets``
read the same property, so a ladder that lives only here would let the planner
pick lengths that have no captured CUDA graph and silently drop the step into
eager execution.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dspark_observability import FORCE_VERIFY_LENS_ENV, RAGGED_VERIFY_MODE_ENV, RaggedVerifyMode
from .dspark_planner import (
    SpsCostTable,
    budget_argmax_over_uniform_lens,
    derive_verify_len_tiers,
    total_verify_tokens,
)

__all__ = [
    "AdditiveFit",
    "CellStat",
    "FlatCostTableError",
    "InertCostTableError",
    "InsufficientSamplesError",
    "ProfilerError",
    "StepSample",
    "SweepConfig",
    "SweepGeometryError",
    "aligned_steps_from_stats",
    "build_cost_table_payload",
    "check_table_is_informative",
    "compress_to_risers",
    "fit_additive_cost_model",
    "load_cost_table",
    "main",
    "profitability_probe",
    "run_sweep",
    "running_max",
    "summarize_cells",
]

#: Per-iteration timing fields published by ``PyExecutor._profiler``. The host
#: field is the default because it is a clean single-loop CPU wall on every
#: scheduler; ``iterLatencyMS`` is deliberately not offered, since under the
#: overlap scheduler (the shipped default) it spans ~2 loops.
HOST_STEP_TIME_KEY = "hostStepTimeMS"
GPU_FORWARD_TIME_KEY = "gpuForwardTimeMS"
TIMING_FIELDS = {"host": HOST_STEP_TIME_KEY, "gpu_forward": GPU_FORWARD_TIME_KEY}

#: ``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` treats 0.0 as "override off",
#: so pinning to bonus-token-only needs a value that floors to 0 without being 0.
#: The fractional part is the probability of one extra token, so 1e-6 fires
#: roughly never while still selecting the override path.
BONUS_TOKEN_ONLY_ACCEPTANCE = 1e-6

#: ``SpsCostTable.__post_init__`` rejects a non-positive ``step_time_ms``. After
#: normalizing theta to hit zero at M = 0 the smallest shelf can land at or below
#: zero, so it is clamped to this instead.
MIN_THETA_MS = 1e-4

#: How far the additive model may miss a cell, relative to a typical step,
#: before the fit says so. Not a repair threshold -- see
#: :func:`fit_additive_cost_model` for why the misfit is reported, not fixed.
DEFAULT_ADDITIVE_TOLERANCE = 0.02


class ProfilerError(RuntimeError):
    """Base class for every refusal this module raises."""


class InsufficientSamplesError(ProfilerError):
    """A cell did not produce enough aligned steady-state steps to trust."""


class SweepGeometryError(ProfilerError):
    """The measured cells cannot separate ``alpha(bs)`` from ``theta(M)``."""


class FlatCostTableError(ProfilerError):
    """The measured curve is flat, so the table would read as unprofiled."""


class InertCostTableError(ProfilerError):
    """The curve moves, but never between two selectable verify lengths."""


# ---------------------------------------------------------------------------
# raw samples and per-cell statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSample:
    """One aligned decode step, filed under the cell that produced it."""

    batch_size: int
    verify_len: int
    step_time_ms: float
    iteration: int = -1

    @property
    def total_verify_tokens(self) -> int:
        """``M`` for this step. Derived, never stored, so it cannot drift."""
        return total_verify_tokens(self.batch_size, self.verify_len)

    def to_json(self) -> dict:
        return {
            "batch_size": int(self.batch_size),
            "verify_len": int(self.verify_len),
            "step_time_ms": float(self.step_time_ms),
            "iteration": int(self.iteration),
        }

    @classmethod
    def from_json(cls, raw: dict) -> "StepSample":
        return cls(
            batch_size=int(raw["batch_size"]),
            verify_len=int(raw["verify_len"]),
            step_time_ms=float(raw["step_time_ms"]),
            iteration=int(raw.get("iteration", -1)),
        )


@dataclass(frozen=True)
class CellStat:
    """The summarized cost of one ``(batch_size, verify_len)`` cell."""

    batch_size: int
    verify_len: int
    step_time_ms: float
    num_samples: int
    p10_ms: float
    p90_ms: float

    @property
    def total_verify_tokens(self) -> int:
        return total_verify_tokens(self.batch_size, self.verify_len)

    def to_json(self) -> dict:
        return {
            "batch_size": int(self.batch_size),
            "verify_len": int(self.verify_len),
            "total_verify_tokens": int(self.total_verify_tokens),
            "step_time_ms": round(float(self.step_time_ms), 6),
            "num_samples": int(self.num_samples),
            "p10_ms": round(float(self.p10_ms), 6),
            "p90_ms": round(float(self.p90_ms), 6),
        }


def aligned_steps_from_stats(
    rows: Sequence[dict],
    *,
    expected_ranks: int = 1,
    timing_key: str = HOST_STEP_TIME_KEY,
) -> List[Tuple[int, int, float]]:
    """Pull ``(iteration, num_gen_requests, step_time_ms)`` out of ``get_stats``.

    A step is kept only when it is *aligned*: every expected attention-DP rank
    published a row for that iteration, no rank had context requests in flight,
    and all ranks scheduled the same number of generation requests. Anything
    else is a step whose shape is not the cell's shape -- a prefill iteration, a
    ramp step, a rank that fell behind -- and averaging it in is how a cost
    table acquires risers that are not there.

    The timing is taken from the rank-0 row rather than averaged across ranks:
    under attention-DP :mod:`~..pyexecutor.adp_iter_stats` fans out one record
    per rank with *rank-local counters* but *rank-0's* step time, so the
    per-rank timings are copies, not independent measurements.
    """
    by_iteration: Dict[int, List[dict]] = {}
    for row in rows:
        iteration = row.get("iter")
        if iteration is None:
            continue
        by_iteration.setdefault(int(iteration), []).append(row)

    aligned: List[Tuple[int, int, float]] = []
    for iteration in sorted(by_iteration):
        group = sorted(by_iteration[iteration], key=lambda r: int(r.get("attentionDpRank", 0)))
        if len({int(r.get("attentionDpRank", 0)) for r in group}) != int(expected_ranks):
            continue
        batching = [row.get("inflightBatchingStats") or {} for row in group]
        if any(int(stats.get("numContextRequests", 0) or 0) for stats in batching):
            continue
        gen_counts = {int(stats.get("numGenRequests", 0) or 0) for stats in batching}
        if len(gen_counts) != 1:
            continue
        num_gen_requests = gen_counts.pop()
        if num_gen_requests <= 0:
            continue
        # gpuForwardTimeMS is read without a device sync, so it is absent on the
        # steps whose event pair had not landed yet; those steps are simply not
        # usable for that field.
        timing = group[0].get(timing_key)
        if timing is None or float(timing) <= 0.0:
            continue
        aligned.append((iteration, num_gen_requests, float(timing)))
    return aligned


def summarize_cells(
    samples: Sequence[StepSample],
    *,
    warmup_steps: int,
    min_samples: int,
) -> List[CellStat]:
    """Median per ``(batch_size, verify_len)`` cell, after dropping warmup.

    Median, not mean: a decode step's distribution is a tight body with a long
    right tail (allocator growth, host scheduling hiccups, the occasional
    stats/log flush). The mean tracks the tail, and a staircase cost model reads
    a tail-inflated point as a riser -- i.e. as a place worth trimming before.

    Args:
        warmup_steps: leading aligned steps to discard *per cell*. They are kept
            in the raw sample file so a refit can choose a different cut.
        min_samples: post-warmup samples a cell must have to be admitted.

    Raises:
        InsufficientSamplesError: if any cell falls short. Silently dropping a
            cell would punch a hole in the grid and can disconnect the
            batch-size ladder, which changes the fit rather than just widening
            its error bars.
    """
    grouped: Dict[Tuple[int, int], List[StepSample]] = {}
    for sample in samples:
        grouped.setdefault((int(sample.batch_size), int(sample.verify_len)), []).append(sample)

    cells: List[CellStat] = []
    for (batch_size, verify_len), cell_samples in sorted(grouped.items()):
        ordered = sorted(cell_samples, key=lambda s: s.iteration)
        steady = [s.step_time_ms for s in ordered[int(warmup_steps) :]]
        if len(steady) < int(min_samples):
            raise InsufficientSamplesError(
                f"cell (batch_size={batch_size}, verify_len={verify_len}) has "
                f"{len(steady)} steady samples after discarding "
                f"{warmup_steps} warmup steps, need {min_samples}. Either the "
                f"cell never reached a steady batch of {batch_size} generating "
                f"requests (check that the batch size is captured by the CUDA "
                f"graph config and that enough requests were in flight), or the "
                f"measurement window was too short -- raise --measure-steps."
            )
        values = np.asarray(steady, dtype=np.float64)
        cells.append(
            CellStat(
                batch_size=batch_size,
                verify_len=verify_len,
                step_time_ms=float(np.median(values)),
                num_samples=int(values.size),
                p10_ms=float(np.percentile(values, 10)),
                p90_ms=float(np.percentile(values, 90)),
            )
        )
    return cells


# ---------------------------------------------------------------------------
# additive fit: T(bs, M) = bias + alpha(bs) + theta(M)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdditiveFit:
    """The separated cost terms, before they are shaped into a staircase."""

    #: ``bias + alpha(bs)`` per profiled batch size, normalized so that theta
    #: extrapolates to zero at ``M = 0``.
    intercept_ms: Dict[int, float]
    #: ``theta(M)`` on the measured token counts, non-decreasing.
    theta_ms: Dict[int, float]
    #: Largest ``|S - intercept - theta|`` over the cells, absolute and relative
    #: to the median cell time. This is the additive model's own error bar: if
    #: it is large, ``theta`` is not a function of ``M`` alone and the
    #: decomposed encoding is not safe at batch sizes far from those profiled.
    max_abs_residual_ms: float
    max_rel_residual: float
    warnings: Tuple[str, ...] = ()


def _assert_grid_is_connected(cells: Sequence[CellStat]) -> None:
    """Refuse a sweep whose ``(bs, M)`` bipartite graph has several components.

    Within one component the additive split is determined up to a single shared
    constant, which the caller fixes. Across components each has its *own*
    unknown constant, so concatenating their ``theta`` values into one staircase
    silently splices two different scales -- the resulting risers are artifacts
    of the offset, not of the hardware.
    """
    parent: Dict[Tuple[str, int], Tuple[str, int]] = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for cell in cells:
        a, b = find(("bs", cell.batch_size)), find(("m", cell.total_verify_tokens))
        if a != b:
            parent[a] = b

    components = {find(("bs", cell.batch_size)) for cell in cells}
    if len(components) > 1:
        groups: Dict[Tuple[str, int], List[int]] = {}
        for cell in cells:
            groups.setdefault(find(("bs", cell.batch_size)), []).append(cell.batch_size)
        islands = sorted(sorted(set(v)) for v in groups.values())
        raise SweepGeometryError(
            f"the swept cells split into {len(components)} disconnected groups of "
            f"batch sizes {islands}: no two groups share a total-token count, so "
            f"alpha(bs) and theta(M) cannot be put on a common scale. Two batch "
            f"sizes meet when bs1*(L1+1) == bs2*(L2+1) for swept lengths, e.g. a "
            f"doubling ladder (8,16,32,...) with verify lengths spanning a factor "
            f"of two (1..5 gives token/request ratios 2..6). Add the missing "
            f"batch sizes or widen --verify-lens."
        )


def running_max(values: Sequence[float]) -> List[float]:
    """Cheapest possible isotonic fit: enforce non-decreasing in place.

    ``SpsCostTable`` does not check monotonicity, and a noise-induced dip makes
    a *longer* verification look cheaper than a shorter one -- the planner will
    take that offer. Clamping up rather than down is the conservative direction:
    it can only make the planner trim less.
    """
    out: List[float] = []
    highest = -float("inf")
    for value in values:
        highest = max(highest, float(value))
        out.append(highest)
    return out


def _least_squares_intercept(x: Sequence[float], y: Sequence[float]) -> float:
    """Value of the best-fit line ``y = a + b*x`` at ``x = 0``."""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.size < 2 or np.ptp(xs) == 0.0:
        return float(ys.min())
    slope = float(np.cov(xs, ys, bias=True)[0, 1] / np.var(xs))
    return float(ys.mean() - slope * xs.mean())


def _solve_additive(
    cells: Sequence[CellStat],
) -> Tuple[Dict[int, float], Dict[int, float], np.ndarray]:
    """Least-squares solve of ``S = intercept[bs] + theta[M]`` over the cells.

    The design is rank-deficient by exactly one -- adding a constant to every
    ``theta`` and subtracting it from every ``intercept`` leaves every prediction
    unchanged -- so ``lstsq`` returns the minimum-norm representative and the
    caller pins the constant afterwards.

    Alternating medians were tried here and are wrong: the grid is unbalanced
    (each token count is reached by a different subset of batch sizes), so the
    per-column median correction differs from column to column and the iteration
    settles on a curve that is *not* the additive truth even when the data is
    exactly additive. Robustness is recovered by trimming outliers and refitting
    instead; see :func:`fit_additive_cost_model`.
    """
    batch_index = {bs: i for i, bs in enumerate(sorted({c.batch_size for c in cells}))}
    token_index = {m: i for i, m in enumerate(sorted({c.total_verify_tokens for c in cells}))}
    num_batches = len(batch_index)
    design = np.zeros((len(cells), num_batches + len(token_index)), dtype=np.float64)
    observed = np.empty(len(cells), dtype=np.float64)
    for row, cell in enumerate(cells):
        design[row, batch_index[cell.batch_size]] = 1.0
        design[row, num_batches + token_index[cell.total_verify_tokens]] = 1.0
        observed[row] = cell.step_time_ms

    solution, *_ = np.linalg.lstsq(design, observed, rcond=None)
    residuals = observed - design @ solution
    intercept = {bs: float(solution[i]) for bs, i in batch_index.items()}
    theta = {m: float(solution[num_batches + i]) for m, i in token_index.items()}
    return intercept, theta, residuals


def fit_additive_cost_model(
    cells: Sequence[CellStat],
    *,
    additive_tolerance: float = DEFAULT_ADDITIVE_TOLERANCE,
) -> AdditiveFit:
    """Separate ``bias + alpha(bs)`` from ``theta(M)``, and report the misfit.

    A badly measured cell is deliberately *not* repaired. The residual of the
    additive solve is the model-validity check the study calls for: cells that
    share an ``M`` but come from different ``(bs, L)`` -- e.g. ``(32, 3)`` and
    ``(16, 7)`` -- must agree once their intercepts are removed, and when they
    do not, ``theta`` is not a function of ``M`` alone. Silently dropping the
    disagreeing cell would delete exactly that evidence and leave a table that
    fits beautifully and predicts badly. So the misfit is surfaced as a warning
    and the human decides whether to trust the decomposition.

    The returned split is normalized so that ``theta`` extrapolates to zero at
    ``M = 0``; see the module docstring for why that is the split the planner
    needs rather than an arbitrary one.
    """
    if not cells:
        raise SweepGeometryError("no measured cells to fit")
    _assert_grid_is_connected(cells)

    warnings: List[str] = []
    intercept, theta, residuals = _solve_additive(cells)
    # Kept from the raw solve: this is the additive model's own error, and the
    # shaping below (monotone clamp, floor clamp) is a deliberate deviation from
    # the data that has its own warnings. Folding the two together would let a
    # clamp masquerade as a model failure and vice versa.
    typical = statistics.median(cell.step_time_ms for cell in cells)
    max_abs = float(np.max(np.abs(residuals))) if len(residuals) else 0.0
    max_rel = max_abs / typical if typical > 0 else 0.0
    if max_rel > float(additive_tolerance):
        worst = ", ".join(
            f"(bs={cells[i].batch_size}, L={cells[i].verify_len}): {residuals[i]:+.3f} ms"
            for i in np.argsort(-np.abs(residuals))[:4]
        )
        warnings.append(
            f"the additive model misses at least one cell by {max_abs:.3f} ms "
            f"({max_rel:.1%} of a typical step), above the "
            f"{float(additive_tolerance):.1%} tolerance. Largest disagreements: "
            f"{worst}. A least-squares solve spreads one bad measurement over "
            f"every cell sharing its batch size or token count, so read these as "
            f"a neighbourhood rather than as the culprit. Either one of those "
            f"steps did not replay a CUDA graph, or theta is not a function of M "
            f"alone -- in which case the decomposed encoding is unsafe away from "
            f"the profiled batch sizes."
        )

    ordered_tokens = sorted(theta)
    monotone = running_max([theta[m] for m in ordered_tokens])
    if any(abs(monotone[i] - theta[m]) > 1e-9 for i, m in enumerate(ordered_tokens)):
        warnings.append(
            "theta(M) was not monotonically increasing and has been clamped "
            "upward; a dip means the measurement noise is comparable to the "
            "risers, so treat the finer shelves as unreliable."
        )
    theta = dict(zip(ordered_tokens, monotone))

    # Fix the one free constant: push theta's M -> 0 extrapolation onto the
    # intercept, so what stays behind in `intercept` is the M-independent floor.
    shift = _least_squares_intercept(ordered_tokens, [theta[m] for m in ordered_tokens])
    # ...but never so far that a shelf goes non-positive; SpsCostTable rejects
    # that, and a zero-cost shelf would read as "these tokens are free".
    shift = min(shift, min(theta.values()) - MIN_THETA_MS)
    theta = {m: v - shift for m, v in theta.items()}
    intercept = {bs: v + shift for bs, v in intercept.items()}

    if min(intercept.values()) < 0.0:
        warnings.append(
            "the fitted M-independent floor came out negative at the smallest "
            "profiled batch size and has been clamped to zero; theta is then "
            "explaining more than the whole step, which means the additive "
            "model does not hold over this sweep."
        )
        intercept = {bs: max(v, 0.0) for bs, v in intercept.items()}

    return AdditiveFit(
        intercept_ms=dict(sorted(intercept.items())),
        theta_ms=dict(sorted(theta.items())),
        max_abs_residual_ms=max_abs,
        max_rel_residual=max_rel,
        warnings=tuple(warnings),
    )


def compress_to_risers(
    token_counts: Sequence[int],
    step_times: Sequence[float],
    *,
    min_riser_ms: float,
    max_breakpoints: int,
) -> Tuple[List[int], List[float]]:
    """Collapse a sampled curve into the staircase it is approximating.

    ``SpsCostTable`` looks up by floor and interpolates nothing, so every
    breakpoint claims "a new kernel wave starts here". Shipping one breakpoint
    per *sample* instead of one per *riser* asserts risers that do not exist,
    and (if the dead ``derive_verify_len_tiers`` path is ever revived) explodes
    the tier count so the truncation drops the real shelf edges.

    A point is kept only when it clears the previous kept shelf by
    ``min_riser_ms``. The threshold is **absolute**, not relative to ``theta``:
    ``theta`` is normalized to approach zero at ``M = 0``, so at the low end a
    fraction of a millisecond of noise is a large *relative* move and a relative
    threshold would manufacture shelves exactly where the curve carries least
    information. What matters is whether a riser is worth a measurable fraction
    of a whole step, which is what the caller scales this by.

    If more points survive than ``max_breakpoints``, the shallowest risers are
    dropped first -- they are the ones a wrong call costs least.
    """
    if len(token_counts) != len(step_times):
        raise ValueError("token_counts and step_times must have the same length")
    if not token_counts:
        raise ValueError("nothing to compress")

    kept = [0]
    for index in range(1, len(token_counts)):
        if step_times[index] - step_times[kept[-1]] > float(min_riser_ms):
            kept.append(index)

    limit = max(int(max_breakpoints), 2)
    while len(kept) > limit:
        # Only interior/trailing points are droppable: index 0 defines the base
        # shelf that every below-range query clamps onto.
        jumps = [(step_times[kept[i]] - step_times[kept[i - 1]], i) for i in range(1, len(kept))]
        _, drop = min(jumps)
        kept.pop(drop)

    return [int(token_counts[i]) for i in kept], [float(step_times[i]) for i in kept]


# ---------------------------------------------------------------------------
# table assembly, loading and refusals
# ---------------------------------------------------------------------------


def load_cost_table(payload: dict) -> SpsCostTable:
    """Build the table exactly the way ``dspark.py::_build_verify_planner`` does.

    Kept byte-for-byte in step with the loader so this module can round-trip its
    own output through the real validation instead of a copy of it.
    """
    return SpsCostTable(
        token_counts=[int(v) for v in payload["token_counts"]],
        step_time_ms=[float(v) for v in payload["step_time_ms"]],
        fixed_overhead_ms=float(payload.get("fixed_overhead_ms", 0.0)),
        batch_sizes=[int(v) for v in payload.get("batch_sizes", [])],
        batch_overhead_ms=[float(v) for v in payload.get("batch_overhead_ms", [])],
    )


def build_cost_table_payload(
    cells: Sequence[CellStat],
    *,
    riser_tolerance: float = 0.01,
    max_breakpoints: int = 8,
    additive_tolerance: float = DEFAULT_ADDITIVE_TOLERANCE,
    meta: Optional[dict] = None,
) -> dict:
    """Assemble the JSON object ``confidence_sps_table_path`` expects.

    Emits exactly the five keys the loader reads plus a ``_meta`` block; unknown
    keys are ignored by the loader, so provenance rides along for free.

    No leading ``token_counts = 0`` entry is synthesized. It would be dead
    weight: :meth:`SpsCostTable.step_time` clamps a below-range query onto index
    0 already, so a fabricated base shelf could only either duplicate the first
    measured shelf or extrapolate below everything that was measured.
    """
    fit = fit_additive_cost_model(cells, additive_tolerance=additive_tolerance)

    # A riser has to be worth a measurable slice of a *whole step* to be real,
    # so the noise floor is scaled by the typical step rather than by theta,
    # whose own scale is an artifact of the M -> 0 normalization.
    typical_step_ms = statistics.median(cell.step_time_ms for cell in cells)
    min_riser_ms = float(riser_tolerance) * typical_step_ms
    tokens, thetas = compress_to_risers(
        sorted(fit.theta_ms),
        [fit.theta_ms[m] for m in sorted(fit.theta_ms)],
        min_riser_ms=min_riser_ms,
        max_breakpoints=max_breakpoints,
    )

    batch_sizes = sorted(fit.intercept_ms)
    intercepts = running_max([fit.intercept_ms[bs] for bs in batch_sizes])
    # Only the sum bias + alpha(bs) reaches the arithmetic, so the split is a
    # reporting choice: bias is the floor that is present even at the smallest
    # profiled batch, alpha is what each larger batch adds on top of it.
    bias = min(intercepts) if intercepts else 0.0
    alphas = [value - bias for value in intercepts]

    payload = {
        "token_counts": [int(v) for v in tokens],
        "step_time_ms": [float(round(v, 6)) for v in thetas],
        "fixed_overhead_ms": float(round(bias, 6)),
        "batch_sizes": [int(v) for v in batch_sizes],
        "batch_overhead_ms": [float(round(v, 6)) for v in alphas],
    }
    payload["_meta"] = {
        "encoding": "decomposed",
        "note": (
            "step_time_ms is theta(M) only -- the trimmable term. The "
            "non-trimmable floor is fixed_overhead_ms + batch_overhead_ms[bs]. "
            "Do not read step_time_ms as whole-step time."
        ),
        "token_unit": (
            "total_verify_tokens = num_requests * (verify_len + 1); the bonus token is included"
        ),
        "cells": [
            cell.to_json() for cell in sorted(cells, key=lambda c: (c.batch_size, c.verify_len))
        ],
        "fit": {
            "max_abs_residual_ms": round(fit.max_abs_residual_ms, 6),
            "max_rel_residual": round(fit.max_rel_residual, 6),
            "additive_tolerance": float(additive_tolerance),
            "riser_tolerance": float(riser_tolerance),
            "min_riser_ms": round(min_riser_ms, 6),
            "typical_step_ms": round(typical_step_ms, 6),
            "warnings": list(fit.warnings),
        },
        **(meta or {}),
    }
    # Constructing it here means a malformed table fails in the profiler rather
    # than at engine init on a machine that has already spent minutes loading
    # weights.
    load_cost_table(payload)
    return payload


def profitability_probe(
    payload: dict,
    *,
    batch_size: int,
    tiers: Sequence[int],
    acceptance_rates: Sequence[float],
) -> List[dict]:
    """Which verify length the real planner picks, per synthetic acceptance rate.

    Runs :func:`~.dspark_planner.budget_argmax_over_uniform_lens` -- the actual
    production argmax, not a re-derivation -- against a geometric survival curve
    ``p**(j+1)``. This is the question the table exists to answer and the one
    ``TLLM_DSPARK_FORCE_VERIFY_LENS`` cannot: at what acceptance rate does
    trimming become the better trade, and does it ever?
    """
    table = load_cost_table(payload)
    ladder = sorted({int(t) for t in tiers})
    block = max(ladder)
    results: List[dict] = []
    for rate in acceptance_rates:
        survival = np.tile(
            np.array([float(rate) ** (j + 1) for j in range(block)], dtype=np.float64),
            (int(batch_size), 1),
        )
        chosen = budget_argmax_over_uniform_lens(
            survival=survival,
            num_gen_requests=int(batch_size),
            cost_table=table,
            allowed_lens=ladder,
            min_verify_len=1,
        )
        results.append(
            {
                "batch_size": int(batch_size),
                "acceptance_rate": float(rate),
                "chosen_verify_len": int(chosen),
                "max_tier": int(ladder[-1]),
                "trims": bool(int(chosen) < ladder[-1]),
            }
        )
    return results


def check_table_is_informative(
    payload: dict,
    *,
    batch_sizes: Sequence[int],
    tiers: Sequence[int],
    acceptance_rates: Sequence[float] = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95),
    min_step_time_spread: float = 0.02,
    allow_inert: bool = False,
) -> dict:
    """Refuse to ship a table that cannot change the planner's mind.

    Two distinct failures, both of which produce a healthy-looking run:

    * *flat* -- ``is_flat`` is exact float equality, so it only catches a table
      nobody measured. A measured-but-motionless curve passes it and then makes
      ``decide_verify_lens`` return ``None`` on every step anyway. So the spread
      of the **total** step time across the reachable token range is checked
      too, not the spread of theta: theta is normalized towards zero at
      ``M = 0``, which would make its own relative spread meaninglessly large.
    * *inert* -- the curve moves, but every selectable tier at every profiled
      batch size lands on one shelf. ``Theta = tau / T`` is then strictly
      increasing in length and the argmax is always ``max_tier``. Unlike the
      flat case this does *not* increment ``fallback_flat_cost``, so nothing
      downstream would report it.
    """
    table = load_cost_table(payload)
    ladder = sorted({int(t) for t in tiers})
    if len(ladder) < 2:
        raise InertCostTableError(
            f"the verify-length ladder {ladder} has a single entry, so the "
            f"planner has nothing to choose between and no cost table can "
            f"change that."
        )

    if table.is_flat:
        raise FlatCostTableError(
            f"the profiled step_time_ms {payload['step_time_ms']} has a single "
            f"distinct value, so SpsCostTable.is_flat is True. The planner "
            f"treats that as 'unprofiled': decide_verify_lens returns None on "
            f"every step (fallback_flat_cost) and ragged verification is a "
            f"no-op. Either the sweep never crossed a cost riser -- widen it "
            f"with larger batch sizes, since gains only exist at high "
            f"concurrency -- or the step cost genuinely does not depend on the "
            f"verified token count here, in which case confidence scheduling "
            f"has no headroom on this configuration and should stay off."
        )

    spreads = {}
    for batch_size in sorted({int(b) for b in batch_sizes}):
        low = table.step_time(total_verify_tokens(batch_size, ladder[0]), batch_size)
        high = table.step_time(total_verify_tokens(batch_size, ladder[-1]), batch_size)
        spreads[batch_size] = (high - low) / low if low > 0 else 0.0
    best_spread = max(spreads.values(), default=0.0)
    if best_spread < float(min_step_time_spread):
        rendered = ", ".join(f"bs={k}: {v:.3%}" for k, v in sorted(spreads.items()))
        raise FlatCostTableError(
            f"the total step time only varies by {best_spread:.3%} across the "
            f"whole tier ladder {ladder} at the best profiled batch size "
            f"({rendered}), which is at or "
            f"below the measurement noise floor of "
            f"{float(min_step_time_spread):.3%}. Such a table is technically "
            f"non-flat but the risers it asserts are noise, and the planner "
            f"would trim on them. To be worth trimming, the step must grow by "
            f"more than the acceptance yield given up -- typically >15% between "
            f"adjacent tiers."
        )

    probes: List[dict] = []
    for batch_size in sorted({int(b) for b in batch_sizes}):
        probes.extend(
            profitability_probe(
                payload,
                batch_size=batch_size,
                tiers=ladder,
                acceptance_rates=acceptance_rates,
            )
        )
    trims_anywhere = any(probe["trims"] for probe in probes)
    if not trims_anywhere and not allow_inert:
        raise InertCostTableError(
            f"the table is non-flat but inert: over batch sizes "
            f"{sorted({int(b) for b in batch_sizes})}, tiers {ladder} and "
            f"acceptance rates {list(acceptance_rates)}, the planner picks "
            f"max_tier={ladder[-1]} every time, so it would never trim a single "
            f"token. That happens when no cost riser falls strictly inside "
            f"(bs*(t_i+1), bs*(t_i+1 +1)] for any adjacent tier pair -- the "
            f"tiers all sit on one shelf. Fix the ladder so its steps straddle "
            f"the measured risers (see _meta."
            f"recommended_confidence_verify_len_tiers), or accept that trimming "
            f"cannot pay here. Pass --allow-inert to emit anyway."
        )

    return {
        "step_time_spread_by_batch_size": {k: round(v, 6) for k, v in spreads.items()},
        "best_step_time_spread": round(best_spread, 6),
        "profitability_probe": probes,
        "trims_somewhere": trims_anywhere,
    }


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """Everything the measurement loop needs; mirrors the CLI one-to-one."""

    model: str
    speculative_model: str
    batch_sizes: List[int]
    verify_lens: List[int]
    tp_size: int = 1
    ep_size: Optional[int] = None
    enable_attention_dp: bool = False
    max_draft_len: int = 5
    input_len: int = 1024
    warmup_steps: int = 16
    measure_steps: int = 64
    min_samples: int = 16
    max_seq_len: int = 4096
    max_num_tokens: int = 8192
    kv_cache_fraction: float = 0.5
    attn_backend: str = "TRTLLM"
    disable_overlap_scheduler: bool = False
    #: On by default because the deployment being profiled runs with it on, and
    #: the confidence head is real per-step work that belongs in the table.
    enable_confidence_scheduling: bool = True
    timing_key: str = HOST_STEP_TIME_KEY
    pin_acceptance: float = BONUS_TOKEN_ONLY_ACCEPTANCE
    #: How long the post-cell stats drain waits for the tail records to arrive
    #: over IPC before it decides the queue is empty.
    stats_timeout_s: float = 5.0
    token_id_range: Tuple[int, int] = (1000, 10000)
    seed: int = 1234
    extra_llm_api_options: Dict[str, object] = field(default_factory=dict)

    @property
    def dp_size(self) -> int:
        """Independent scheduling domains, i.e. rows per iteration in the stats."""
        return int(self.tp_size) if self.enable_attention_dp else 1

    def max_tokens_for(self, verify_len: int) -> int:
        """Output-token budget that guarantees the cell reaches its step count.

        With acceptance pinned every step commits exactly one token, so the
        budget *is* the step count. Unpinned, a step commits up to
        ``verify_len + 1``, so the budget is scaled to keep the lower bound on
        steps at the target -- at the price of running up to that factor longer
        when acceptance turns out to be poor.
        """
        target_steps = int(self.warmup_steps) + int(self.measure_steps)
        budget = target_steps + max(target_steps // 4, 16)
        return budget if self.pin_acceptance else budget * (int(verify_len) + 1)

    def validate(self) -> None:
        """Catch the geometry mistakes that would otherwise fail mid-sweep."""
        if not self.batch_sizes or not self.verify_lens:
            raise SweepGeometryError("both --batch-sizes and --verify-lens must be non-empty")
        if min(self.verify_lens) < 1:
            raise SweepGeometryError("verify lengths must be >= 1 (the planner's floor)")
        # A request that hits max_seq_len is evicted mid-cell, so the batch
        # drains and the steady window shrinks -- which surfaces much later as a
        # confusing "not enough steady samples".
        longest = self.input_len + max(self.max_tokens_for(v) for v in self.verify_lens)
        if longest > self.max_seq_len:
            raise SweepGeometryError(
                f"a request would reach {longest} tokens (input {self.input_len} + "
                f"generated) but --max-seq-len is {self.max_seq_len}; raise it or "
                f"lower --input-len / --measure-steps"
            )
        if self.input_len > self.max_num_tokens:
            raise SweepGeometryError(
                f"--input-len {self.input_len} exceeds --max-num-tokens "
                f"{self.max_num_tokens}, so a prompt cannot be scheduled"
            )
        widest_decode = max(self.batch_sizes) * (max(self.verify_lens) + 1)
        if widest_decode > self.max_num_tokens:
            raise SweepGeometryError(
                f"the widest decode step submits {widest_decode} tokens "
                f"(bs={max(self.batch_sizes)} x verify_len={max(self.verify_lens)} + 1) "
                f"but --max-num-tokens is {self.max_num_tokens}; the batch would be "
                f"split and no step would ever have the shape it is filed under"
            )


def _prepare_environment(config: SweepConfig) -> None:
    """Pin the knobs the table's meaning depends on, loudly.

    Set before the LLM is constructed because the worker processes inherit the
    environment at spawn and read these once.
    """
    if os.environ.get(FORCE_VERIFY_LENS_ENV):
        print(
            f"[dspark-sps] clearing {FORCE_VERIFY_LENS_ENV}: it rotates a ragged "
            f"split across the batch, so M would no longer be bs*(L+1) and every "
            f"cell would be filed under the wrong token count.",
            file=sys.stderr,
        )
        os.environ.pop(FORCE_VERIFY_LENS_ENV, None)
    os.environ[RAGGED_VERIFY_MODE_ENV] = RaggedVerifyMode.STATIC.value
    if config.pin_acceptance:
        from .interface import FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR

        os.environ[FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR] = repr(float(config.pin_acceptance))


def _build_llm(config: SweepConfig, verify_len: int):
    """One engine per verify length; see the module docstring for why."""
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, DSparkDecodingConfig, KvCacheConfig

    spec_config = DSparkDecodingConfig(
        max_draft_len=int(verify_len),
        # DSparkWorker asserts draft_model.block_size == max_draft_len, so the
        # block has to move with the swept length rather than stay at the
        # checkpoint's trained value.
        block_size=int(verify_len),
        speculative_model=config.speculative_model,
        enable_confidence_scheduling=bool(config.enable_confidence_scheduling),
        # A single tier keeps the captured-graph count identical to an
        # unscheduled run (one graph per batch size) while still exercising the
        # confidence head, which is part of the deployment's per-step cost. It
        # also makes the planner's choice a no-op: max_tier == max_draft_len, so
        # the step shape is pinned no matter what the confidence snapshot says.
        confidence_verify_len_tiers=(
            [int(verify_len)] if config.enable_confidence_scheduling else None
        ),
    )
    llm_kwargs = dict(
        model=config.model,
        tensor_parallel_size=int(config.tp_size),
        moe_expert_parallel_size=int(config.ep_size) if config.ep_size else None,
        enable_attention_dp=bool(config.enable_attention_dp),
        attn_backend=config.attn_backend,
        max_batch_size=max(config.batch_sizes),
        max_seq_len=int(config.max_seq_len),
        max_num_tokens=int(config.max_num_tokens),
        disable_overlap_scheduler=bool(config.disable_overlap_scheduler),
        speculative_config=spec_config,
        # Block reuse would let a later cell start from a warm prefix and skip
        # prefill work the earlier cells paid for, which shifts the steady-state
        # KV length between cells.
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=float(config.kv_cache_fraction),
        ),
        # Any batch size outside this list runs eager; an eager step is orders
        # of magnitude off the graph-replay cost and would poison the curve.
        cuda_graph_config=CudaGraphConfig(
            batch_sizes=sorted(set(config.batch_sizes)),
            enable_padding=True,
        ),
        enable_iter_perf_stats=True,
        # Unbounded: the buffer is drained while the sweep runs, but a stall
        # during a long cell must not silently evict the steps being measured.
        iter_stats_max_iterations=-1,
    )
    llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}
    llm_kwargs.update(config.extra_llm_api_options)
    return LLM(**llm_kwargs)


def _run_cell(llm, config: SweepConfig, *, batch_size: int, verify_len: int) -> List[StepSample]:
    """Hold ``batch_size`` generating requests per rank and record every step."""
    from tensorrt_llm import SamplingParams

    rng = random.Random(config.seed + batch_size * 1000 + verify_len)
    low, high = config.token_id_range
    num_requests = int(batch_size) * config.dp_size
    sampling_params = SamplingParams(
        # Every request must outlive the whole window so the batch never drains
        # mid-measurement and shrinks the shape being profiled.
        max_tokens=config.max_tokens_for(verify_len),
        ignore_eos=True,
        # No detokenization: it is postprocessing work that would show up in the
        # host step time without being part of what the planner is choosing.
        detokenize=False,
        temperature=0.0,
    )
    # Random ids from a mid-vocabulary window rather than real text: no tokenizer
    # round-trip is needed, and a distinct prompt per request keeps any prefix
    # sharing out of the measurement. The window avoids the low ids where
    # special/control tokens live in every large vocabulary.
    prompts = [
        {"prompt_token_ids": [rng.randint(low, high) for _ in range(config.input_len)]}
        for _ in range(num_requests)
    ]

    futures = [llm.generate_async(prompt, sampling_params) for prompt in prompts]
    try:
        for future in futures:
            future.result()
    finally:
        for future in futures:
            if not future.finished:
                future.abort()
    # Drained exactly once, after the cell has finished. `IterationResult`
    # latches itself done the first time its queue runs dry and is only reset
    # when new requests are submitted, so polling mid-cell would silently
    # discard every step after the first empty read. The timeout is the wait for
    # the last few records to cross the IPC boundary, not a poll interval.
    rows: List[dict] = list(llm.get_stats(timeout=config.stats_timeout_s) or [])

    aligned = aligned_steps_from_stats(
        rows, expected_ranks=config.dp_size, timing_key=config.timing_key
    )
    return [
        StepSample(
            batch_size=int(batch_size),
            verify_len=int(verify_len),
            step_time_ms=step_time,
            iteration=iteration,
        )
        for iteration, num_gen, step_time in aligned
        if num_gen == int(batch_size)
    ]


def run_sweep(config: SweepConfig) -> List[StepSample]:
    """Measure every ``(batch_size, verify_len)`` cell, rebuilding per length."""
    config.validate()
    _prepare_environment(config)
    samples: List[StepSample] = []
    for verify_len in sorted(config.verify_lens):
        print(
            f"[dspark-sps] building engine for verify_len={verify_len} "
            f"(block_size={verify_len}) ...",
            file=sys.stderr,
        )
        llm = _build_llm(config, verify_len)
        try:
            for batch_size in sorted(config.batch_sizes):
                started = time.time()
                cell = _run_cell(llm, config, batch_size=batch_size, verify_len=verify_len)
                print(
                    f"[dspark-sps]   bs={batch_size:>5} L={verify_len} "
                    f"M={total_verify_tokens(batch_size, verify_len):>6} "
                    f"aligned_steps={len(cell):>5} "
                    f"({time.time() - started:.1f}s)",
                    file=sys.stderr,
                )
                samples.extend(cell)
        finally:
            llm.shutdown()
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _int_list(text: str) -> List[int]:
    return [int(v) for v in str(text).replace(",", " ").split()]


def _float_list(text: str) -> List[float]:
    return [float(v) for v in str(text).replace(",", " ").split()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tensorrt_llm._torch.speculative.dspark_sps_profiler",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, help="Path to write the cost-table JSON.")
    parser.add_argument(
        "--samples-out",
        default=None,
        help="Raw per-step samples (JSONL). Defaults to <out>.samples.jsonl.",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Skip the engine entirely and refit from --samples-in. No GPU needed.",
    )
    parser.add_argument(
        "--samples-in",
        default=None,
        help="Comma-separated JSONL sample files to fit (implies --fit-only inputs).",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--model", help="Target model checkpoint path or HF id.")
    model.add_argument(
        "--speculative-model",
        default=None,
        help="DSpark draft checkpoint. Defaults to --model.",
    )
    model.add_argument("--tp-size", type=int, default=1)
    model.add_argument("--ep-size", type=int, default=None, help="moe_expert_parallel_size.")
    model.add_argument("--enable-attention-dp", action="store_true")
    model.add_argument("--attn-backend", default="TRTLLM")
    model.add_argument("--max-seq-len", type=int, default=4096)
    model.add_argument("--max-num-tokens", type=int, default=8192)
    model.add_argument("--kv-cache-fraction", type=float, default=0.5)
    model.add_argument(
        "--disable-overlap-scheduler",
        action="store_true",
        help="Profile without the overlap scheduler. The default (overlap ON) is "
        "what ships, so leave this off unless the deployment disables it.",
    )
    model.add_argument(
        "--no-confidence-scheduling",
        dest="enable_confidence_scheduling",
        action="store_false",
        help="Profile with confidence scheduling off. Only for a checkpoint "
        "without confidence-head weights: the head is real per-step work and "
        "the deployment being profiled runs with it on.",
    )
    model.add_argument(
        "--extra-llm-api-options",
        default=None,
        help="YAML file merged into the LLM(...) kwargs (moe_config, quant, ...).",
    )
    parser.set_defaults(enable_confidence_scheduling=True)

    sweep = parser.add_argument_group("sweep")
    sweep.add_argument(
        "--batch-sizes",
        default="8,16,32,64,128",
        help="Generating requests PER attention-DP rank. Every value is captured "
        "as a CUDA graph; use a ladder whose ratios are reachable by the "
        "verify-length ratios (a doubling ladder is safe).",
    )
    sweep.add_argument(
        "--max-draft-len",
        type=int,
        default=5,
        help="The deployment's block size. Sets the top of the verify-length sweep.",
    )
    sweep.add_argument(
        "--verify-lens",
        default=None,
        help="Uniform verify lengths to sweep. Defaults to 1..--max-draft-len. "
        "Each value costs one engine build.",
    )
    sweep.add_argument("--input-len", type=int, default=1024, help="Prompt length in tokens.")
    sweep.add_argument("--warmup-steps", type=int, default=16)
    sweep.add_argument("--measure-steps", type=int, default=64)
    sweep.add_argument("--min-samples", type=int, default=16)
    sweep.add_argument("--seed", type=int, default=1234)
    sweep.add_argument(
        "--timing-field",
        choices=sorted(TIMING_FIELDS),
        default="host",
        help="host = hostStepTimeMS (clean per-loop CPU wall, works under the "
        "overlap scheduler); gpu_forward = gpuForwardTimeMS (device time of the "
        "fused draft+verify forward only).",
    )
    sweep.add_argument(
        "--no-pin-acceptance",
        dest="pin_acceptance",
        action="store_false",
        help="Do not force bonus-token-only acceptance. Leaves KV growth rate "
        "dependent on the block length, which contaminates theta(M).",
    )
    parser.set_defaults(pin_acceptance=True)

    fit = parser.add_argument_group("fit")
    fit.add_argument("--riser-tolerance", type=float, default=0.01)
    fit.add_argument("--max-breakpoints", type=int, default=8)
    fit.add_argument("--min-step-time-spread", type=float, default=0.02)
    fit.add_argument(
        "--verify-len-tiers",
        default=None,
        help="Ladder to validate against; defaults to the ladder derived from the "
        "measured shelves at --reference-batch-size.",
    )
    fit.add_argument("--max-tiers", type=int, default=3)
    fit.add_argument(
        "--reference-batch-size",
        type=int,
        default=None,
        help="Modal steady-state batch size. Defaults to the largest swept value, "
        "since shelf edges move with batch size and trimming only pays at high "
        "concurrency.",
    )
    fit.add_argument("--probe-acceptance-rates", default="0.3,0.5,0.7,0.8,0.9,0.95")
    fit.add_argument(
        "--allow-inert",
        action="store_true",
        help="Emit even if the planner would never trim. Refused by default.",
    )
    return parser


def _load_samples(paths: Sequence[str]) -> List[StepSample]:
    samples: List[StepSample] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    samples.append(StepSample.from_json(json.loads(line)))
    return samples


def _write_samples(path: str, samples: Sequence[StepSample]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json()) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    verify_lens = (
        _int_list(args.verify_lens)
        if args.verify_lens
        else list(range(1, int(args.max_draft_len) + 1))
    )
    batch_sizes = _int_list(args.batch_sizes)
    samples_out = args.samples_out or f"{args.out}.samples.jsonl"

    if args.fit_only or args.samples_in:
        if not args.samples_in:
            raise SystemExit("--fit-only requires --samples-in")
        samples = _load_samples(_str_list(args.samples_in))
        provenance = {"source": "refit", "samples_in": _str_list(args.samples_in)}
    else:
        if not args.model:
            raise SystemExit("--model is required unless --fit-only is used")
        extra: Dict[str, object] = {}
        if args.extra_llm_api_options:
            import yaml

            with open(args.extra_llm_api_options, encoding="utf-8") as handle:
                extra = yaml.safe_load(handle) or {}
        config = SweepConfig(
            model=args.model,
            speculative_model=args.speculative_model or args.model,
            batch_sizes=batch_sizes,
            verify_lens=verify_lens,
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            enable_attention_dp=args.enable_attention_dp,
            max_draft_len=args.max_draft_len,
            input_len=args.input_len,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            min_samples=args.min_samples,
            max_seq_len=args.max_seq_len,
            max_num_tokens=args.max_num_tokens,
            kv_cache_fraction=args.kv_cache_fraction,
            attn_backend=args.attn_backend,
            disable_overlap_scheduler=args.disable_overlap_scheduler,
            enable_confidence_scheduling=args.enable_confidence_scheduling,
            timing_key=TIMING_FIELDS[args.timing_field],
            pin_acceptance=BONUS_TOKEN_ONLY_ACCEPTANCE if args.pin_acceptance else 0.0,
            seed=args.seed,
            extra_llm_api_options=extra,
        )
        samples = run_sweep(config)
        _write_samples(samples_out, samples)
        print(f"[dspark-sps] wrote {len(samples)} raw samples to {samples_out}", file=sys.stderr)
        provenance = {
            "source": "sweep",
            "model": config.model,
            "tensor_parallel_size": config.tp_size,
            "moe_expert_parallel_size": config.ep_size,
            "attention_dp": config.enable_attention_dp,
            "input_len": config.input_len,
            "timing_field": config.timing_key,
            "acceptance_pinned": bool(config.pin_acceptance),
            "overlap_scheduler": not config.disable_overlap_scheduler,
            "confidence_scheduling": config.enable_confidence_scheduling,
            "ragged_verify_mode": RaggedVerifyMode.STATIC.value,
        }

    cells = summarize_cells(samples, warmup_steps=args.warmup_steps, min_samples=args.min_samples)
    if not cells:
        # Distinguished from a short cell: this means no step was ever aligned,
        # which is a wiring problem (iteration stats off, the wrong rank count,
        # the batch never reaching the requested size) rather than a noisy run.
        raise InsufficientSamplesError(
            f"no cells at all: {len(samples)} raw samples produced nothing usable. "
            f"Check that enable_iter_perf_stats reached the engine, that "
            f"--enable-attention-dp matches the deployment (the aligned-step "
            f"filter expects one stats row per DP rank), and that the requested "
            f"batch sizes were actually reached."
        )
    reference_bs = int(args.reference_batch_size or max(c.batch_size for c in cells))
    block_size = max(int(args.max_draft_len), max(c.verify_len for c in cells))

    payload = build_cost_table_payload(
        cells,
        riser_tolerance=args.riser_tolerance,
        max_breakpoints=args.max_breakpoints,
        meta={
            "block_size": block_size,
            "min_verify_len": 1,
            "modal_batch_size": reference_bs,
            "warmup_steps_discarded": int(args.warmup_steps),
            "statistic": "median",
            **provenance,
        },
    )

    tiers = (
        _int_list(args.verify_len_tiers)
        if args.verify_len_tiers
        else derive_verify_len_tiers(
            cost_table=load_cost_table(payload),
            num_requests=reference_bs,
            block_size=block_size,
            min_verify_len=1,
            max_tiers=int(args.max_tiers),
        )
    )
    payload["_meta"]["recommended_confidence_verify_len_tiers"] = [int(t) for t in tiers]
    payload["_meta"]["recommended_tiers_note"] = (
        "Nothing reads this from the table file. Copy it into "
        "DSparkDecodingConfig.confidence_verify_len_tiers, which is also what "
        "sizes the captured CUDA graph set."
    )

    diagnostics = check_table_is_informative(
        payload,
        batch_sizes=sorted({c.batch_size for c in cells}),
        tiers=tiers,
        acceptance_rates=_float_list(args.probe_acceptance_rates),
        min_step_time_spread=args.min_step_time_spread,
        allow_inert=args.allow_inert,
    )
    payload["_meta"]["diagnostics"] = diagnostics

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")

    _print_report(payload, cells, diagnostics, tiers)
    return 0


def _str_list(text: str) -> List[str]:
    return [v for v in str(text).replace(",", " ").split() if v]


def _print_report(
    payload: dict,
    cells: Sequence[CellStat],
    diagnostics: dict,
    tiers: Sequence[int],
) -> None:
    meta = payload["_meta"]
    print("")
    print("DSpark SPS cost table")
    print("---------------------")
    print(f"  token_counts      : {payload['token_counts']}")
    print(f"  step_time_ms      : {payload['step_time_ms']}   (theta(M) only)")
    print(f"  fixed_overhead_ms : {payload['fixed_overhead_ms']}")
    print(f"  batch_sizes       : {payload['batch_sizes']}")
    print(f"  batch_overhead_ms : {payload['batch_overhead_ms']}")
    print("")
    print(f"  cells measured    : {len(cells)}")
    print(
        f"  additive-model fit: max residual "
        f"{meta['fit']['max_abs_residual_ms']:.4f} ms "
        f"({meta['fit']['max_rel_residual']:.2%} of a typical step)"
    )
    for warning in meta["fit"]["warnings"]:
        print(f"  WARNING: {warning}")
    print(f"  step-time spread  : {diagnostics['best_step_time_spread']:.2%} across the ladder")
    print("")
    print(f"  Copy into confidence_verify_len_tiers: {list(tiers)}")
    print("  Planner choice vs synthetic acceptance rate (batch_size / rate -> length):")
    for probe in diagnostics["profitability_probe"]:
        mark = "trim" if probe["trims"] else "    "
        print(
            f"    bs={probe['batch_size']:>5}  p={probe['acceptance_rate']:.2f}  "
            f"-> L={probe['chosen_verify_len']} (max {probe['max_tier']}) {mark}"
        )
    print("")


if __name__ == "__main__":
    sys.exit(main())
