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
A crafted (non-flat) cost table unblocks *correctness* testing by making the
planner itself trim, rather than bypassing
that gate, but it rotates a fixed ladder and knows nothing about cost, so it
cannot answer "is trimming profitable here". Only a measured table can.

Run it::

    python tests/microbenchmarks/dspark_sps_profiler.py \\
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
a crafted cost table. Under ``static`` the planner still chooses a
single batch-wide length, but the cost table it sees is the flat default, whose
``is_flat`` gate short-circuits to ``max_tier`` *before* any confidence is read
-- so the length is deterministically ``max_draft_len`` on every step and
``M = bs * (max_draft_len + 1)`` exactly. If the scheduler were allowed to trim
mid-sweep, each cell would be measuring a different ``M`` than the one it is
filed under, i.e. a moving target.

That determinism is also why the ``L`` axis costs one engine build per value:
nothing in the runtime pins a *uniform* verify length below the full block
(a ragged split would break the M = bs*(L+1) filing, and there is no
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
import pathlib
import random
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tensorrt_llm._torch.speculative.dspark_observability import RAGGED_VERIFY_MODE_ENV, RaggedVerifyMode
from tensorrt_llm._torch.speculative.dspark_planner import (
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
    #: Ragged (frac-sweep) steps: the group's executed token total, read from
    #: the stats rows. None on pinned sweeps, where M is derived instead.
    measured_tokens: Optional[int] = None

    @property
    def total_verify_tokens(self) -> int:
        """``M`` for this step. Measured when the shape was ragged; derived
        from the pin otherwise, so it cannot drift from the label."""
        if self.measured_tokens is not None:
            return int(self.measured_tokens)
        return total_verify_tokens(self.batch_size, self.verify_len)

    def to_json(self) -> dict:
        payload = {
            "batch_size": int(self.batch_size),
            "verify_len": int(self.verify_len),
            "step_time_ms": float(self.step_time_ms),
            "iteration": int(self.iteration),
        }
        if self.measured_tokens is not None:
            payload["measured_tokens"] = int(self.measured_tokens)
        return payload

    @classmethod
    def from_json(cls, raw: dict) -> "StepSample":
        measured = raw.get("measured_tokens")
        return cls(
            batch_size=int(raw["batch_size"]),
            verify_len=int(raw["verify_len"]),
            step_time_ms=float(raw["step_time_ms"]),
            iteration=int(raw.get("iteration", -1)),
            measured_tokens=None if measured is None else int(measured),
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
    #: Ragged (frac-sweep) cells: the executed token total the samples agreed
    #: on. None on pinned sweeps.
    measured_tokens: Optional[int] = None

    @property
    def total_verify_tokens(self) -> int:
        if self.measured_tokens is not None:
            return int(self.measured_tokens)
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


def ragged_steps_from_stats(
    rows: Sequence[dict],
    *,
    expected_ranks: int = 1,
    timing_key: str = HOST_STEP_TIME_KEY,
) -> List[Tuple[int, int, int, float]]:
    """``(iteration, num_gen_requests, executed_tokens, step_time_ms)`` steps.

    The ragged (frac-sweep) sibling of :func:`aligned_steps_from_stats`: on top
    of its alignment rules, every rank must also report the same
    ``numGenerationTokens``. Under attention-DP the executed token total is the
    group's agreed bucket -- every rank pads and fills to the same shape -- so
    a step whose ranks disagree is a ramp/fallback step whose shape is not one
    shape at all, and averaging it in would file its cost under a bucket it did
    not run.
    """
    by_iteration: Dict[int, List[dict]] = {}
    for row in rows:
        iteration = row.get("iter")
        if iteration is None:
            continue
        by_iteration.setdefault(int(iteration), []).append(row)

    kept: List[Tuple[int, int, int, float]] = []
    for iteration in sorted(by_iteration):
        group = by_iteration[iteration]
        if len({int(r.get("attentionDpRank", 0)) for r in group}) != int(expected_ranks):
            continue
        batching = [row.get("inflightBatchingStats") or {} for row in group]
        if any(int(stats.get("numContextRequests", 0) or 0) for stats in batching):
            continue
        gen_counts = {int(stats.get("numGenRequests", 0) or 0) for stats in batching}
        if len(gen_counts) != 1 or (num_gen := gen_counts.pop()) <= 0:
            continue
        token_counts = {row.get("numGenerationTokens") for row in group}
        if len(token_counts) != 1:
            continue
        tokens = token_counts.pop()
        if tokens is None or int(tokens) <= 0:
            continue
        timing = group[0].get(timing_key)
        if timing is None or float(timing) <= 0.0:
            continue
        kept.append((iteration, num_gen, int(tokens), float(timing)))
    return kept


def explain_alignment(
    rows: Sequence[dict],
    *,
    expected_ranks: int = 1,
    timing_key: str = HOST_STEP_TIME_KEY,
) -> str:
    """Say why :func:`aligned_steps_from_stats` kept nothing.

    A cell that reports ``aligned_steps=0`` is otherwise indistinguishable from
    a cell that never ran, and the four rejection reasons want opposite fixes:
    a rank count that is never right means ``--enable-attention-dp`` disagrees
    with the deployment, context requests on every step mean the prompts are
    still being ingested, disagreeing generation counts mean the batch is
    draining, and a missing timing field means the requested ``--timing-field``
    is not being published. Guessing between those costs a full sweep each
    time.
    """
    by_iteration: Dict[int, List[dict]] = {}
    for row in rows:
        iteration = row.get("iter")
        if iteration is None:
            continue
        by_iteration.setdefault(int(iteration), []).append(row)
    if not by_iteration:
        return (f"{len(rows)} stats rows carried no 'iter' field at all"
                if rows else "no stats rows were received at all")

    reasons: Dict[str, int] = {}
    rank_counts: Dict[int, int] = {}
    gen_seen: set = set()
    for iteration, group in by_iteration.items():
        ranks = {int(r.get("attentionDpRank", 0)) for r in group}
        rank_counts[len(ranks)] = rank_counts.get(len(ranks), 0) + 1
        if len(ranks) != int(expected_ranks):
            reasons["rank count != expected"] = reasons.get("rank count != expected", 0) + 1
            continue
        batching = [row.get("inflightBatchingStats") or {} for row in group]
        if any(int(stats.get("numContextRequests", 0) or 0) for stats in batching):
            reasons["context requests in flight"] = reasons.get("context requests in flight", 0) + 1
            continue
        gen_counts = {int(stats.get("numGenRequests", 0) or 0) for stats in batching}
        gen_seen.update(gen_counts)
        if len(gen_counts) != 1:
            reasons["ranks disagree on numGenRequests"] = reasons.get("ranks disagree on numGenRequests", 0) + 1
            continue
        if gen_counts and max(gen_counts) <= 0:
            reasons["no generation requests"] = reasons.get("no generation requests", 0) + 1
            continue
        timing = group[0].get(timing_key)
        if timing is None or float(timing) <= 0.0:
            reasons[f"missing/zero {timing_key}"] = reasons.get(f"missing/zero {timing_key}", 0) + 1
            continue
        reasons["kept"] = reasons.get("kept", 0) + 1

    parts = [f"{len(rows)} rows over {len(by_iteration)} iterations "
             f"(iter {min(by_iteration)}..{max(by_iteration)})"]
    parts.append("rejections: " + ", ".join(
        f"{reason} x{count}" for reason, count in
        sorted(reasons.items(), key=lambda kv: -kv[1])) or "none")
    parts.append("ranks-per-iteration histogram: " + ", ".join(
        f"{n}->{c}" for n, c in sorted(rank_counts.items())) +
        f" (expected {expected_ranks})")
    if gen_seen:
        parts.append(f"numGenRequests observed: {sorted(gen_seen)[:12]}")
    return "; ".join(parts)


def summarize_cells(
    samples: Sequence[StepSample],
    *,
    warmup_steps: int,
    min_samples: int,
    expected_cells: Optional[Sequence[Tuple[int, int]]] = None,
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
        expected_cells: the requested grid, when there is one. Sources that
            observe a deployment (a served sweep's iteration log, a production
            log) inevitably capture ramp and drain steps whose occupancy was
            never requested -- e.g. a batch of 2 while the client's 256
            requests were still being admitted. Those incidental cells are
            dropped (and reported) when thin; only a thin cell that was
            actually REQUESTED is an error. None means every observed cell was
            asked for (the engine-building sweep, existing callers).

    Raises:
        InsufficientSamplesError: if a requested cell falls short. Silently
            dropping one of those would punch a hole in the grid and can
            disconnect the batch-size ladder, which changes the fit rather
            than just widening its error bars.
    """
    grouped: Dict[Tuple[int, int, Optional[int]], List[StepSample]] = {}
    for sample in samples:
        key = (int(sample.batch_size), int(sample.verify_len),
               None if sample.measured_tokens is None else int(sample.measured_tokens))
        grouped.setdefault(key, []).append(sample)

    requested = None if expected_cells is None else {
        (int(b), int(l)) for b, l in expected_cells}
    dropped: List[str] = []
    cells: List[CellStat] = []
    for (batch_size, verify_len, measured_tokens), cell_samples in sorted(
            grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or -1)):
        ordered = sorted(cell_samples, key=lambda s: s.iteration)
        steady = [s.step_time_ms for s in ordered[int(warmup_steps) :]]
        if len(steady) < int(min_samples):
            # Ragged sweeps visit whichever buckets the planner picked, so a
            # thin incidental bucket is normal coverage noise, not a failed
            # cell: drop it loudly rather than aborting the fit.
            if measured_tokens is not None:
                dropped.append(f"(bs={batch_size}, M={measured_tokens}, n={len(steady)})")
                continue
            if requested is not None and (batch_size, verify_len) not in requested:
                dropped.append(f"(bs={batch_size}, L={verify_len}, n={len(steady)})")
                continue
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
                measured_tokens=measured_tokens,
            )
        )
    if dropped:
        # Reported, not silent: a fit that quietly ignored data reads as
        # "covered everything" when it did not.
        print(f"[dspark-sps] dropped {len(dropped)} thin incidental cell(s) "
              f"outside the requested grid: {', '.join(dropped[:12])}"
              + (" ..." if len(dropped) > 12 else ""), file=sys.stderr)
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
    """Collapse a sampled curve into shelf-and-riser breakpoints.

    ``SpsCostTable`` interpolates between breakpoints, so what a table CLAIMS
    is entirely in which points it keeps: dropping a shelf's interior is fine
    (interpolation across equal values is still flat), but dropping a shelf's
    LAST point turns the measured shelf into one long rising segment and
    over-bills every mid-shelf total by up to the riser height -- the
    over-trim mirror of the floor-lookup bug. So each kept riser is preceded
    by the last point of the shelf it rises from: shelves survive as
    breakpoint pairs, exactly the encoding the planner tests use for
    "genuinely flat".

    A riser is kept only when it clears the previous kept shelf by
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
            # Close the shelf we are leaving before recording the riser, so
            # interpolation stays flat across it instead of ramping from the
            # shelf's first sample to the riser top.
            if index - 1 != kept[-1]:
                kept.append(index - 1)
            kept.append(index)

    limit = max(int(max_breakpoints), 2)
    truncated_from = len(kept)
    while len(kept) > limit:
        # Drop the interior point whose removal introduces the least
        # interpolation error. The previous rule -- smallest incoming jump --
        # deleted shelf-CLOSING points first by construction (their jump is
        # below min_riser_ms), which turned every measured shelf back into a
        # ramp and re-billed mid-shelf totals upward: the exact over-charge
        # the pair encoding exists to prevent. Endpoints are never droppable:
        # index 0 anchors the below-range clamp, the last point the above.
        def _removal_error(pos: int) -> float:
            x0, x1, x2 = (token_counts[kept[pos - 1]], token_counts[kept[pos]],
                          token_counts[kept[pos + 1]])
            y0, y2 = step_times[kept[pos - 1]], step_times[kept[pos + 1]]
            interp = y0 + (y2 - y0) * (x1 - x0) / max(x2 - x0, 1)
            return abs(step_times[kept[pos]] - interp)

        drop = min(range(1, len(kept) - 1), key=_removal_error)
        kept.pop(drop)
    if truncated_from > len(kept):
        print(f"[dspark-sps] compress_to_risers: truncated {truncated_from} "
              f"breakpoints to {len(kept)} (max_breakpoints={max_breakpoints});"
              f" the kept set minimizes interpolation error, but raise the cap"
              f" if the residuals matter", file=sys.stderr)

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
        # Consumer contract marker. Tables written for the old floor consumer
        # deliberately dropped shelf-closing breakpoints; interpolating those
        # re-bills every mid-shelf total upward with nothing to say so. The
        # loader warns when this key is absent.
        "lookup": "interp",
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
    a single sweep cannot: at what acceptance rate does
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
    # 128, not a token gesture: autotuner cache misses ("using the fallback
    # tactic") and allocator growth are still visible well past step 16 on a
    # fresh engine, and a cell that starts on the cool side biases theta UP
    # for exactly the small-M cells the argmax is most sensitive to.
    warmup_steps: int = 128
    measure_steps: int = 64
    repeats: int = 1
    min_samples: int = 16
    max_seq_len: int = 4096
    # Frac sweeps only: override the engine's DERIVED tier ladder (cost-table
    # shelf edges, typically [1,3,5]). A denser ladder makes the frac knob
    # quantise to more rungs, so one sweep measures more M columns per bs --
    # the dense-tier experiments need the odd ratios {3,5} the default ladder
    # cannot execute.
    engine_tiers: Optional[List[int]] = None
    max_num_tokens: int = 8192
    kv_cache_fraction: float = 0.5
    #: Fitted STS temperatures. Not used to decide anything while the pin is on;
    #: it is what lets ``enable_ragged_verify`` pass validation without the cost
    #: table this tool exists to produce.
    sts_path: Optional[str] = None
    #: Any non-flat cost table. See the note at its use site: with a one-rung
    #: ladder its values cannot affect the result, only its presence.
    seed_table_path: Optional[str] = None
    #: Restore the original sweep geometry; see the note in _build_llm.
    block_follows_verify_len: bool = False
    #: Frac sweep (run_frac_sweep): budget fractions to visit on the live
    #: ragged path. When set, verify_lens is pinned to [max_draft_len] by the
    #: CLI and the per-cell M comes from measurement instead of the pin.
    fracs: Optional[List[float]] = None
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
    #: Mid-cell poll interval. Short enough that ~1000 rows cannot accumulate
    #: between polls (at dp_size=8 that is 125 iterations, i.e. a few seconds).
    stats_poll_s: float = 2.0
    token_id_range: Tuple[int, int] = (1000, 10000)
    seed: int = 1234
    extra_llm_api_options: Dict[str, object] = field(default_factory=dict)

    @property
    def dp_size(self) -> int:
        """Independent scheduling domains, i.e. rows per iteration in the stats."""
        return int(self.tp_size) if self.enable_attention_dp else 1

    def max_tokens_for(self, verify_len: int,
                       batch_size: Optional[int] = None) -> int:
        """Output-token budget that guarantees the cell reaches its step count.

        With acceptance pinned every step commits exactly one token, so the
        budget *is* the step count. Unpinned, a step commits up to
        ``verify_len + 1``, so the budget is scaled to keep the lower bound on
        steps at the target -- at the price of running up to that factor longer
        when acceptance turns out to be poor.

        ``batch_size`` adds the *admission ramp*, and without it no large cell
        can ever yield a sample. A cell is only counted while all ``dp_size``
        ranks hold exactly ``batch_size`` generating requests, so the request
        admitted first must still be alive when the last one is admitted.
        Measured at bs=256, input_len=1024: the scheduler admits
        roughly one context request per iteration per rank, so the ramp is
        ~``batch_size`` iterations, while the budget was 85 -- the first
        requests died ~170 iterations before the last arrived, concurrency
        plateaued at 11 of 256, and every one of the 8 repeats kept nothing.
        bs<=128 escaped only because its ramp fit inside the same 85.

        The ramp is bounded below by prefill tokens / ``max_num_tokens``, but
        that bound was off by ~60x here, so use the observed one-per-iteration
        rate instead: it is the conservative direction, and overshooting costs
        run time while undershooting costs the entire cell.
        """
        target_steps = int(self.warmup_steps) + int(self.measure_steps)
        budget = target_steps + max(target_steps // 4, 16)
        if batch_size is not None:
            budget += int(batch_size)
        return budget if self.pin_acceptance else budget * (int(verify_len) + 1)

    def validate(self) -> None:
        """Catch the geometry mistakes that would otherwise fail mid-sweep."""
        if self.fracs is not None:
            bad = [f for f in self.fracs if not 0.0 < float(f) <= 1.0]
            if bad:
                raise SweepGeometryError(
                    f"--fracs values must be in (0, 1], got {sorted(bad)}")
            if self.block_follows_verify_len:
                raise SweepGeometryError(
                    "--fracs measures the live ragged path at the constant "
                    "block; --block-follows-verify-len contradicts it")
        if not self.batch_sizes or not self.verify_lens:
            raise SweepGeometryError("both --batch-sizes and --verify-lens must be non-empty")
        if min(self.verify_lens) < 1:
            raise SweepGeometryError("verify lengths must be >= 1 (the planner's floor)")
        # A request that hits max_seq_len is evicted mid-cell, so the batch
        # drains and the steady window shrinks -- which surfaces much later as a
        # confusing "not enough steady samples".
        longest = self.input_len + max(
            self.max_tokens_for(v, b) for v in self.verify_lens for b in self.batch_sizes)
        if longest > self.max_seq_len:
            raise SweepGeometryError(
                f"a request would reach {longest} tokens (input {self.input_len} + "
                f"generated) but --max-seq-len is {self.max_seq_len}; raise it or "
                f"lower --input-len / --measure-steps"
            )
        # get_stats is served from a queue that retains on the order of 1000
        # ROWS, and attention-DP publishes one row per rank per iteration. A
        # cell longer than that window keeps only its tail, which is the drain
        # phase -- every step there has ranks disagreeing on numGenRequests, so
        # the cell yields nothing and looks like it never ran. Raising
        # --measure-steps past the window therefore makes things strictly worse;
        # use --repeats to get more samples instead.
        rows_per_iteration = self.dp_size
        window_iterations = _STATS_ROW_BUDGET // max(rows_per_iteration, 1)
        cell_iterations = int(self.warmup_steps) + int(self.measure_steps)
        if cell_iterations > window_iterations:
            print(
                f"[dspark-sps] WARNING: a cell runs {cell_iterations} iterations "
                f"but only about {window_iterations} fit in the retained stats "
                f"window ({_STATS_ROW_BUDGET} rows / {rows_per_iteration} ranks). "
                f"Only the tail survives, and the tail is the drain phase, so "
                f"cells will come back empty. Lower --measure-steps to "
                f"<= {window_iterations - int(self.warmup_steps)} and raise "
                f"--repeats instead.",
                file=sys.stderr,
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


#: Approximate number of stats rows ``get_stats`` retains for one drain. Not a
#: constant the runtime exports; measured as 1001 rows surviving a 1300-step
#: attention-DP cell, i.e. the last 125 of its iterations.
_STATS_ROW_BUDGET = 1000


def _prepare_environment(config: SweepConfig) -> None:
    """Pin the knobs the table's meaning depends on, loudly.

    Set before the LLM is constructed because the worker processes inherit the
    environment at spawn and read these once.
    """
    # COMPACT, not STATIC. Under STATIC the token axis is bs*(max_draft_len+1)
    # and the only way to vary M is to rebuild with a shorter block -- which
    # also shortens the DRAFT pass, so the measured difference between two
    # cells mixes "verified fewer tokens" with "drafted fewer tokens". The
    # deployed path drafts the full block every step and varies only the verify
    # window, so a table built that way over-states what trimming buys. COMPACT
    # plus the per-engine pin below holds the draft at max_draft_len and moves
    # only the axis the planner actually controls.
    os.environ[RAGGED_VERIFY_MODE_ENV] = RaggedVerifyMode.COMPACT.value
    if config.pin_acceptance:
        from tensorrt_llm._torch.speculative.interface import (
            FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR)

        os.environ[FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR] = repr(float(config.pin_acceptance))


def _build_llm(config: SweepConfig, verify_len: int, *,
               pin_verify_len: bool = True):
    """One engine per verify length; see the module docstring for why."""
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, DSparkDecodingConfig, KvCacheConfig

    # Held at the checkpoint's trained block for EVERY cell. Sweeping it (which
    # this did) makes the draft pass shrink along with the verify window, and
    # the two costs are then inseparable in the fitted theta(M). The previous
    # K-sweep on this hardware put the draft pass at ~17 ms and the marginal
    # verify token at ~0 -- so a table that lets draft cost leak into theta(M)
    # would attribute the draft saving to trimming and predict a benefit the
    # deployment cannot collect. The verify window is moved by the pin below.
    # The window is pinned by BOTH the env override and the ladder, and the
    # env var is SET here, never popped.
    #
    # Set, because the workers are spawned by srun with the job environment: a
    # job-wide pin reaches ranks 1..N-1 regardless of what this process does to
    # its own env, so popping it here strips rank 0 alone. A rank-0 planner
    # with no pin and no cost table falls back to uniform while its peers run
    # pinned -- a rank-divergent shape under attention-DP. With one verify
    # length per sweep process the pin always equals the ladder rung below, so
    # it can never be rejected either.
    from tensorrt_llm._torch.speculative.dspark_verify import FORCE_VERIFY_LEN_ENV

    # LAUNCH-MODE CAVEAT: this mutation reaches ranks 1..N-1 only when the
    # workers are spawned AFTER it (MPI-spawn path). On a pre-spawned world
    # (srun / trtllm-llmapi-launch) already-running ranks keep the previous
    # value, the per-L pin diverges across ranks, and cells measure a shape
    # their label does not describe. assert_swept_shape refuses such a sweep
    # downstream (verify_len_hist vs the nominal L), which is why it gates the
    # fit rather than merely warning.
    # Frac sweeps leave the pin unset: their windows come from the live
    # planner (that is the experiment), and the budget fraction travels by
    # RPC + the step's own allgather, which reaches pre-spawned worlds the
    # env mutation below cannot.
    if pin_verify_len:
        os.environ[FORCE_VERIFY_LEN_ENV] = str(int(verify_len))
    else:
        os.environ.pop(FORCE_VERIFY_LEN_ENV, None)
    # Which knob moves M.
    #
    # constant block (default): the block stays at the checkpoint's trained
    # length for every cell and only the verify window moves, which is exactly
    # what the deployment does.
    #
    # block follows verify_len: the original design. Never produces
    # max(lens) < block, so it runs today, at the price of shrinking the DRAFT
    # pass along with the window -- theta(M) then absorbs a per-L draft term.
    # That term is recoverable rather than fatal: with several batch sizes the
    # same M is reached at different L (M=192 from bs=32/L=5 and bs=64/L=2), and
    # those shared points separate a token cost from an L cost.
    block = int(verify_len) if config.block_follows_verify_len else int(config.max_draft_len)
    spec_config = DSparkDecodingConfig(
        max_draft_len=block,
        block_size=block,
        speculative_model=config.speculative_model,
        enable_confidence_scheduling=bool(config.enable_confidence_scheduling),
        # Both required together: the validator rejects confidence scheduling
        # without ragged verification (there are only two states -- schedule per
        # request, or verify the full block). The profiler had only the first
        # and could not construct its own engine at all; it failed 1:36 into
        # every two-node sweep. And ragged in turn demands a profiled artifact,
        # which is circular for the tool that produces one -- an STS path is the
        # non-circular way to satisfy it, and the pin makes the planner's own
        # choice irrelevant here anyway.
        enable_ragged_verify=bool(config.enable_confidence_scheduling),
        confidence_sts_path=config.sts_path,
        # Only has to be non-flat and present: with one rung the planner's
        # choice is already determined, so the table's VALUES cannot influence
        # this measurement -- it exists to get past the flat-cost fallback,
        # which would otherwise return None and put every step back on the
        # uniform path at M = bs*(block+1).
        confidence_sps_table_path=config.seed_table_path,
        # A single tier keeps the captured-graph count identical to an
        # unscheduled run (one graph per batch size) while still exercising the
        # confidence head, which is part of the deployment's per-step cost. It
        # also makes the planner's choice a no-op: max_tier == max_draft_len, so
        # the step shape is pinned no matter what the confidence snapshot says.
        # The pin decides the length; the ladder only has to CONTAIN it, or the
        # planner would return a length with no captured graph and every step
        # would fall back to eager -- which is not the cost being tabulated.
        # One rung: the planner's argmax has a single legal answer, so the step
        # shape is pinned no matter what the confidence snapshot says, while
        # every downstream path (bucket choice, padding reservation, graph
        # selection) runs exactly as it does in production.
        confidence_verify_len_tiers=(
            # Frac sweeps need the PRODUCTION ladder: the token-bucket grid is
            # derived from it, and a one-rung ladder would collapse every
            # captured bucket to the full block -- the sweep would then only
            # ever measure M = bs * (block + 1) regardless of the frac.
            # ``engine_tiers`` overrides the derived ladder for dense grids.
            (([int(t) for t in config.engine_tiers]
              if config.engine_tiers else None)
             if not pin_verify_len else [int(verify_len)])
            if config.enable_confidence_scheduling else None
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


def _run_cell(llm, config: SweepConfig, *, batch_size: int, verify_len: int,
              measure_ragged: bool = False) -> List[StepSample]:
    """Hold ``batch_size`` generating requests per rank and record every step.

    With ``measure_ragged`` the per-step token total is READ from the stats
    (the group's agreed bucket) instead of derived from the pin, and samples
    carry it in ``measured_tokens``; one cell then legitimately yields samples
    at several M values.
    """
    from tensorrt_llm import SamplingParams

    rng = random.Random(config.seed + batch_size * 1000 + verify_len)
    low, high = config.token_id_range
    num_requests = int(batch_size) * config.dp_size
    sampling_params = SamplingParams(
        # Every request must outlive the whole window so the batch never drains
        # mid-measurement and shrinks the shape being profiled.
        max_tokens=config.max_tokens_for(verify_len, batch_size),
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

    # Drained WHILE the cell runs, not once at the end. Only ~1000 stats rows
    # survive to a single drain -- 125 iterations at dp_size=8 -- and the
    # scheduler admits roughly one context request per iteration, so a cell
    # spends ~batch_size iterations ramping up and just as many draining. At
    # bs>=128 the whole retained window therefore lands inside the drain: every
    # step it contains has 0..11 generating requests, none has `batch_size`,
    # and the cell reports "kept nothing" while the plateau it was measuring
    # scrolled out of the buffer unseen.
    #
    # `IterationResult` latches done when its queue runs dry, so an
    # empty read ends collection. Polling only while requests are still
    # outstanding keeps the queue fed; the count of empty reads is reported so a
    # cell that lost its tail this way is visible rather than merely short.
    rows: List[dict] = []
    drain_error: List[BaseException] = []
    empty_reads = [0]
    stop = threading.Event()

    def _drain() -> None:
        while not stop.is_set():
            try:
                got = list(llm.get_stats(timeout=config.stats_poll_s) or [])
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                drain_error.append(exc)
                return
            if got:
                rows.extend(got)
            else:
                # Counted because one empty read is not merely a lost poll:
                # IterationResult latches done on the first Empty, and until
                # something un-latches it every later get_stats -- including
                # the final post-cell drain -- returns [] instantly. A cell
                # with a non-zero count here may have lost its tail.
                empty_reads[0] += 1
            stop.wait(config.stats_poll_s)

    # A daemon drainer, NOT a poll loop that owns the exit condition. Making the
    # main thread wait on `future.finished` while draining hung the sweep: the
    # loop span silently, the log stopped growing, and the stall watchdog killed
    # the job at exactly 15 minutes with no cell produced. The main thread here
    # blocks on future.result() exactly as it did before this change, so
    # termination is unaffected by anything the drainer does.
    # stats_poll_s <= 0 disables the mid-cell drain: the control arm for "does
    # the profiler perturb what it measures". The cell then relies on the single
    # post-cell drain, which only sees the last ~1000 rows -- fine for small
    # batches whose whole cell fits in that window, useless above bs~64. That is
    # exactly why this is a control and not an option.
    drainer = None
    if float(config.stats_poll_s) > 0:
        drainer = threading.Thread(target=_drain, name="dspark-sps-drain",
                                   daemon=True)
        drainer.start()
    try:
        for future in futures:
            future.result()
    finally:
        stop.set()
        if drainer is not None:
            drainer.join(timeout=max(5.0, 3.0 * config.stats_poll_s))
        for future in futures:
            if not future.finished:
                future.abort()
    if drain_error:
        print(f"[dspark-sps]   bs={batch_size} L={verify_len}: mid-cell drain "
              f"stopped early ({type(drain_error[0]).__name__}: "
              f"{drain_error[0]}); the cell may be short",
              file=sys.stderr)
    # Final drain for whatever crossed the IPC boundary after the last poll.
    # Un-latch first: a single gap in stats production longer than the poll
    # timeout latched IterationResult done, and a latched queue answers []
    # instantly -- silently truncating the cell this drain exists to finish.
    undo = getattr(getattr(getattr(llm, "_executor", None),
                           "_iter_stats_result", None), "mark_undone", None)
    if undo is not None:
        undo()
    rows.extend(list(llm.get_stats(timeout=config.stats_timeout_s) or []))
    if empty_reads[0]:
        print(f"[dspark-sps]   bs={batch_size} L={verify_len}: "
              f"{empty_reads[0]} empty stats read(s) mid-cell; the retained "
              f"window may have closed early and the cell lost its tail",
              file=sys.stderr)

    if measure_ragged:
        ragged = ragged_steps_from_stats(
            rows, expected_ranks=config.dp_size, timing_key=config.timing_key)
        kept_ragged = [s for s in ragged if s[1] == int(batch_size)]
        if not kept_ragged:
            print(
                f"[dspark-sps]   bs={batch_size:>5} (ragged) kept nothing -- "
                + explain_alignment(rows, expected_ranks=config.dp_size,
                                    timing_key=config.timing_key)
                + "; also requires numGenerationTokens on every rank row",
                file=sys.stderr)
            return []
        # Warmup is discarded per CELL by iteration order, here rather than in
        # summarize_cells: one ragged cell yields several (bs, M) groups, and a
        # per-group trim would re-charge the warmup against every bucket.
        cutoff = sorted(s[0] for s in kept_ragged)
        cutoff = cutoff[min(int(config.warmup_steps), len(cutoff) - 1)]
        return [
            StepSample(
                batch_size=int(batch_size),
                verify_len=int(verify_len),
                step_time_ms=step_time,
                iteration=iteration,
                measured_tokens=tokens,
            )
            for iteration, num_gen, tokens, step_time in kept_ragged
            if iteration >= cutoff
        ]
    aligned = aligned_steps_from_stats(
        rows, expected_ranks=config.dp_size, timing_key=config.timing_key
    )
    kept = [s for s in aligned if s[1] == int(batch_size)]
    if not kept:
        # Printed per cell rather than once at the end: which cells failed and
        # why is the whole diagnosis, and a sweep that ends in
        # InsufficientSamplesError has already thrown that away.
        print(
            f"[dspark-sps]   bs={batch_size:>5} L={verify_len} kept nothing -- "
            + explain_alignment(rows, expected_ranks=config.dp_size,
                                timing_key=config.timing_key)
            + (f"; aligned but numGenRequests never == {batch_size} "
               f"(saw {sorted({a[1] for a in aligned})[:12]})" if aligned else ""),
            file=sys.stderr,
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
            f"[dspark-sps] building engine: block_size="
            f"{verify_len if config.block_follows_verify_len else config.max_draft_len}"
            f"{' (follows L)' if config.block_follows_verify_len else ' (constant)'}"
            f", verify window {verify_len} -> M = bs*{verify_len + 1} ...",
            file=sys.stderr,
        )
        llm = _build_llm(config, verify_len)
        try:
            for batch_size in sorted(config.batch_sizes):
                started = time.time()
                cell = []
                for _ in range(max(1, int(config.repeats))):
                    # Each repeat drains the stats queue again, so repeats add
                    # samples where a longer single cell would only overflow the
                    # retained window.
                    cell.extend(_run_cell(llm, config, batch_size=batch_size,
                                          verify_len=verify_len))
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


def run_frac_sweep(config: SweepConfig) -> List[StepSample]:
    """Measure ``(batch_size, frac)`` cells on the LIVE ragged path.

    One engine build for the whole sweep: the block stays at the checkpoint's
    trained length, no verify-length pin is set, and per cell the budget
    fraction is switched at runtime (``set_dspark_budget_frac`` -- rank 0's
    queue, delivered by the decode loop's own allgather). Windows come from
    the real confidence top-k and the executed shape is the group's agreed
    bucket, so the cells sample exactly the geometry production runs --
    including the token-major presentation and the bucket fit -- instead of
    the pinned-uniform approximation. M is read per step from the stats
    (``measured_tokens``), so one cell contributes samples at every bucket the
    planner visited.
    """
    config.validate()
    _prepare_environment(config)
    fracs = sorted({float(f) for f in (config.fracs or [])})
    if not fracs:
        raise ValueError("run_frac_sweep requires a non-empty --fracs")
    print(
        f"[dspark-sps] building engine once: block_size={config.max_draft_len} "
        f"(constant), live planner windows, frac sweep {fracs} ...",
        file=sys.stderr,
    )
    samples: List[StepSample] = []
    llm = _build_llm(config, int(config.max_draft_len), pin_verify_len=False)

    def _set_frac(value):
        executor = llm._executor
        rpc = getattr(executor, "collective_rpc", None)
        if rpc is not None:
            return rpc("set_dspark_budget_frac", args=(value,))
        # Single-process executor path: the worker IS the executor.
        return executor.set_dspark_budget_frac(value)

    try:
        for frac in fracs:
            _set_frac(float(frac))
            for batch_size in sorted(config.batch_sizes):
                started = time.time()
                cell: List[StepSample] = []
                for _ in range(max(1, int(config.repeats))):
                    cell.extend(
                        _run_cell(llm, config, batch_size=batch_size,
                                  verify_len=int(config.max_draft_len),
                                  measure_ragged=True))
                buckets = sorted({s.measured_tokens for s in cell})
                print(
                    f"[dspark-sps]   bs={batch_size:>5} frac={frac:.2f} "
                    f"aligned_steps={len(cell):>5} buckets={buckets[:8]}"
                    f"{' ...' if len(buckets) > 8 else ''} "
                    f"({time.time() - started:.1f}s)",
                    file=sys.stderr,
                )
                samples.extend(cell)
    finally:
        try:
            _set_frac(None)
        except Exception:  # noqa: BLE001 - shutdown path, engine may be gone
            pass
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
        prog="python tests/microbenchmarks/dspark_sps_profiler.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, help="Path to write the cost-table JSON.")
    source = parser.add_argument_group(
        "sample source",
        "Where the (batch size, verify length, step time) triples come from. "
        "Default: build an engine in-process and sweep it. The two "
        "alternatives measure the deployment itself, so the table cannot "
        "describe a configuration nobody is running.")
    source.add_argument(
        "--from-iter-log", nargs="+", metavar="LOG",
        help="Fit from production iteration logs (print_iter_log). No server, "
             "no traffic, no machine time; coverage is whatever the "
             "deployment happened to run.")
    source.add_argument(
        "--base-url",
        help="Sweep a RUNNING trtllm-serve instead of building an engine. The "
             "verify length is pinned over /dspark/verify_len_pin, so walking "
             "the ladder costs no engine rebuild.")
    source.add_argument(
        "--control-url",
        help="Where the pin endpoint and /metrics live when they differ from "
        "--base-url: under disaggregated serving the LOAD goes through the "
        "proxy, but /dspark/verify_len_pin and the iteration stats belong to "
        "the GENERATION server. Defaults to --base-url.",
    )
    source.add_argument("--server-settle-s", type=float, default=20.0,
                        help="Seconds to let admission reach the plateau "
                             "before a cell is measured (--base-url).")
    source.add_argument("--server-poll-s", type=float, default=20.0,
                        help="Seconds of steady state to collect per cell "
                             "(--base-url).")
    source.add_argument(
        "--server-log",
        help="The server's stdout/stderr (print_iter_log). When reachable this "
             "is where --base-url takes its samples from: every step is "
             "labelled by the shape it ran, and no metrics collector has to be "
             "configured for the sweep to see anything.")
    source.add_argument("--served-model-name", default=None,
                        help="`model` field for /v1/completions (--base-url). "
                             "Omit to read it from /v1/models, which is what "
                             "the server actually registered -- for a local "
                             "checkpoint that is the directory basename, not "
                             "the path it was launched with.")
    source.add_argument(
        "--cuda-graph-batch-sizes",
        help="The deployment's captured batch-size ladder, comma separated "
             "(--from-iter-log). Padding rounds a step's rows up to the "
             "smallest entry at or above them, which is what makes the padded "
             "token total resolve to exactly one verify length. Omit and only "
             "steps whose shape is unambiguous on its own are kept.")
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
    sweep.add_argument(
        "--fracs",
        default=None,
        help="Budget fractions in (0, 1] to sweep on the LIVE ragged path "
        "(one engine build total; windows from the real confidence top-k, "
        "per-step M measured from the group's agreed bucket). Mutually "
        "exclusive with --verify-lens and --block-follows-verify-len.",
    )
    sweep.add_argument(
        "--engine-tiers",
        default=None,
        help="Frac sweeps only: comma-separated tier ladder for the SWEEP "
        "engine (e.g. 2,3,4,5), overriding the cost-table-derived default. "
        "A denser ladder quantises the frac knob to more rungs, so one sweep "
        "measures more M columns per batch size.",
    )
    sweep.add_argument("--input-len", type=int, default=1024, help="Prompt length in tokens.")
    sweep.add_argument("--block-follows-verify-len", action="store_true",
                       help="Shrink the drafted block with the verify window "
                            "(the original geometry). Avoids the "
                            "max(lens)<block_size packing assert at the price "
                            "of a per-L draft term in theta(M).")
    sweep.add_argument("--stats-poll-s", type=float, default=2.0,
                       help="Mid-cell stats drain interval. <=0 disables it, "
                            "which is the control arm for measuring whether the "
                            "drain perturbs the step time it records.")
    sweep.add_argument("--seed-table", default=None,
                       help="Any non-flat cost table. With a one-rung ladder "
                            "its values cannot affect the measurement; without "
                            "one the planner falls back to uniform and every "
                            "cell is measured at the full block.")
    sweep.add_argument("--sts-path", default=None,
                       help="Fitted STS temperatures. Required with confidence "
                            "scheduling: ragged verification will not validate "
                            "without either this or the cost table being built.")
    sweep.add_argument("--warmup-steps", type=int, default=128)
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
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="run each cell this many times and pool the samples; the way to "
             "get more samples per cell, since a longer cell overflows the "
             "retained stats window and keeps only its drain phase")
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


# ---------------------------------------------------------------------------
# Alternative sample sources
#
# The in-process sweep below builds its own engine per verify length. That
# gives total control, and costs a rebuild per length plus a configuration
# that has to be kept in sync with the deployment by hand -- when it drifts,
# the table describes a machine nobody is running (a profiled (256, 1536) cell
# read 150.2 ms against 118.8 ms measured live at the same shape, and nothing
# in the fingerprint recorded the difference).
#
# These two collectors take the measurement to the deployment instead.
# ---------------------------------------------------------------------------


def resolve_padded_shape(
    *,
    num_rows: int,
    num_generation_tokens: int,
    max_draft_len: int,
    max_batch_size: int,
    padded_batch_sizes: Optional[Sequence[int]] = None,
) -> Optional[Tuple[int, int]]:
    """Recover ``(padded_bs, verify_len)`` from a step's row and token counts.

    The token total is the PADDED one: a step with 244 real rows submits
    ``256 * (L + 1)`` tokens because the batch is padded up to a captured
    graph size, and it is that padded shape the step actually cost. So the
    length cannot be read off the real row count -- ``M / rows`` is not even
    integral -- and a filter that demands it throws away nearly every step (it
    kept 312 of 5682 on two production runs).

    Every candidate length implies a padded row count; the answer is the one
    that is integral, at least the real row count, and within the engine's
    batch ceiling. Ambiguous steps are rejected rather than guessed: filing a
    step under a shape it did not run is the mislabelling that once produced a
    clean-looking, entirely fictional table.
    """
    if num_rows <= 0 or num_generation_tokens <= 0:
        return None
    ladder = (sorted({int(b) for b in padded_batch_sizes})
              if padded_batch_sizes else None)
    if ladder:
        # Padding rounds UP to the smallest captured batch size, never past
        # it, so the row count alone fixes the padded width and the length
        # follows. Without this the low-occupancy steps are genuinely
        # ambiguous -- 768 tokens over 128 rows is L=5 at 128 rows, L=3 at
        # 192, or L=2 at 256 -- and all of them would be dropped.
        candidates = [b for b in ladder if b >= num_rows]
        if not candidates:
            return None
        padded_bs = candidates[0]
        if num_generation_tokens % padded_bs:
            # Padding declined (a resource bail), so the step ran its real
            # rows and this width does not describe it. Skip rather than
            # invent one.
            return None
        verify_len = num_generation_tokens // padded_bs - 1
        if not 0 < verify_len <= int(max_draft_len):
            return None
        return (padded_bs, verify_len)
    found: List[Tuple[int, int]] = []
    for verify_len in range(1, int(max_draft_len) + 1):
        width = verify_len + 1
        if num_generation_tokens % width:
            continue
        padded_bs = num_generation_tokens // width
        if padded_bs < num_rows or padded_bs > int(max_batch_size):
            continue
        if ladder is not None and padded_bs not in ladder:
            continue
        found.append((padded_bs, verify_len))
    return found[0] if len(found) == 1 else None


def samples_from_iter_log(
    paths: Sequence[str],
    *,
    max_draft_len: int,
    max_batch_size: int = 256,
    padded_batch_sizes: Optional[Sequence[int]] = None,
    rank_prefix: str = "0:",
    max_step_ms: float = 1000.0,
) -> List[StepSample]:
    """Read (bs, M, step time) triples out of production iteration logs.

    ``print_iter_log`` already emits everything a cell needs::

        iter = N, ..., num_scheduled_requests = R, ...,
        host_step_time = T ms, ..., states = {..., 'num_generation_tokens': M}

    so a table can be fitted from traffic that really ran, at the deployment's
    own context depths, batch mix and captured-graph set -- no synthetic
    prompts, no second engine, no machine time. The limitation is coverage:
    only shapes the deployment happened to run appear, and rarely-taken rungs
    may be thin. Use it to fit a table for a workload that is already in
    production, or to check that a swept table has not drifted; use the server
    sweep when specific cells have to be visited on purpose.

    ``verify_len`` is recovered by :func:`resolve_padded_shape`, which reads it
    out of the padded token total instead of assuming it; steps whose shape is
    ambiguous are skipped rather than filed under a length they did not run.
    Samples carry the PADDED row count, because that is the shape whose cost
    was measured.

    Pass ``padded_batch_sizes`` (the deployment's CUDA-graph ladder) when it is
    known: it removes the last ambiguity, since only a captured row count can
    have run.
    """
    line_re = re.compile(
        r"iter = (\d+).*?num_scheduled_requests = (\d+).*?"
        r"host_step_time = ([\d.]+)ms.*?'num_ctx_requests': (\d+)"
        r".*?'num_generation_tokens': (\d+)")
    out: List[StepSample] = []
    skipped_ratio = 0
    skipped_mixed = 0

    def _consume(line: str) -> None:
        nonlocal skipped_ratio, skipped_mixed
        match = line_re.search(line)
        if not match:
            return
        iteration = int(match.group(1))
        batch_size = int(match.group(2))
        step_ms = float(match.group(3))
        ctx_requests = int(match.group(4))
        tokens = int(match.group(5))
        if ctx_requests:
            # A step carrying prefill chunks runs eager at a prefill-inflated
            # time; the ladder resolution would still find a plausible
            # (padded_bs, L) for it and file a mixed step as a pure decode
            # cell -- confidently mislabelled, the exact failure this
            # collector exists to avoid.
            skipped_mixed += 1
            return
        if iteration == 1:
            # Each executor instance numbers from 1, and its first step
            # carries the wait for the first request inside its host time
            # (62 s on a live server). Identify it by the number rather than
            # by having seen one, so a log FRAGMENT is still usable.
            return
        if batch_size <= 0 or step_ms >= max_step_ms:
            return
        shape = resolve_padded_shape(
            num_rows=batch_size, num_generation_tokens=tokens,
            max_draft_len=max_draft_len, max_batch_size=max_batch_size,
            padded_batch_sizes=padded_batch_sizes)
        if shape is None:
            skipped_ratio += 1
            return
        padded_bs, verify_len = shape
        out.append(StepSample(batch_size=padded_bs, verify_len=verify_len,
                              step_time_ms=step_ms, iteration=iteration))

    for path in paths:
        text = pathlib.Path(path).read_text(errors="replace")
        candidates = [ln for ln in text.splitlines() if "iter = " in ln]
        matched = [ln for ln in candidates
                   if not rank_prefix or ln.lstrip().startswith(rank_prefix)]
        if candidates and not matched and rank_prefix:
            # Single-process deployments write no srun/mpirun rank tag; a
            # hard prefix requirement silently dropped every line and the
            # error blamed print_iter_log. When the prefix matches nothing at
            # all, the log simply is not rank-tagged -- read it as-is.
            print(f"[dspark-sps] iter-log: no lines in {path} carry the rank "
                  f"prefix {rank_prefix!r}; reading it as an untagged "
                  f"single-process log", file=sys.stderr)
            matched = candidates
        for line in matched:
            _consume(line)
    if skipped_mixed:
        print(f"[dspark-sps] iter-log: skipped {skipped_mixed} mixed "
              f"context+generation steps (prefill-inflated times)",
              file=sys.stderr)
    if skipped_ratio:
        print(f"[dspark-sps] iter-log: skipped {skipped_ratio} steps whose "
              f"padded shape could not be resolved unambiguously",
              file=sys.stderr)
    return out


def samples_from_server(
    *,
    base_url: str,
    control_url: Optional[str] = None,
    batch_sizes: Sequence[int],
    verify_lens: Sequence[int],
    input_len: int,
    model: str,
    warmup_steps: int,
    settle_s: float,
    poll_s: float,
    max_draft_len: int,
    seed: int,
    padded_batch_sizes: Optional[Sequence[int]] = None,
    server_log: Optional[str] = None,
    dp_size: int = 1,
) -> List[StepSample]:
    """Sweep a RUNNING server: pin the window over HTTP, then read /metrics.

    The equivalent of SGLang's profiler loop, and the reason
    ``/dspark/verify_len_pin`` exists: walking the verify-length ladder no
    longer costs an engine rebuild per rung, and the engine being measured is
    by construction the engine that will consume the table.

    Load is synthetic (random token ids, ignore_eos) because a cell has to hold
    a fixed batch size for long enough to measure it; the SHAPE is what the
    table indexes, and the deployment's own graphs, context depths and
    scheduler are what answer for it.

    ``dp_size`` is how many requests it takes to put ONE on each rank's
    batch. Under attention DP every rank schedules its own subset, the
    planner prices its own rank's batch, and the iteration log reports
    per-rank rows -- so the table's bs axis is PER-RANK, and a cell labeled
    ``bs`` needs ``bs * dp_size`` requests in flight. Job 2585129 swept a
    tp8-ADP server with dp_size unaccounted: every requested cell (64..256)
    actually measured per-rank batches of 8..40, and the fit refused the
    grid as disconnected.
    """
    import requests

    control_url = control_url or base_url
    rng = random.Random(seed)
    out: List[StepSample] = []

    if not model:
        # The served id is whatever the server registered -- for a local
        # checkpoint that is the directory's basename, not the path that was
        # passed to it. Posting the path gets every request rejected, no
        # traffic runs, and every cell reports zero steps with nothing to say
        # why. Ask the server instead of making the caller guess.
        listing = requests.get(f"{base_url}/v1/models", timeout=60)
        listing.raise_for_status()
        entries = (listing.json() or {}).get("data") or []
        if not entries:
            raise RuntimeError(f"{base_url}/v1/models listed no model")
        model = entries[0]["id"]
        print(f"[dspark-sps] serving model id: {model}", file=sys.stderr)

    post_errors: List[str] = []

    def _fire(body: dict) -> None:
        try:
            response = requests.post(f"{base_url}/v1/completions", json=body,
                                     timeout=3600)
            if response.status_code != 200 and len(post_errors) < 3:
                post_errors.append(
                    f"{response.status_code} {response.text[:200]}")
        except Exception as exc:  # noqa: BLE001 - reported below, not fatal
            if len(post_errors) < 3:
                post_errors.append(f"{type(exc).__name__}: {exc}")

    def pin(verify_len: Optional[int]) -> None:
        response = requests.post(f"{control_url}/dspark/verify_len_pin",
                                 json={"verify_len": verify_len}, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"pinning verify_len={verify_len} was refused by the server: "
                f"{response.status_code} {response.text}")

    def drain_metrics() -> List[dict]:
        response = requests.get(f"{control_url}/metrics", timeout=60)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, list) else [payload]

    try:
        for verify_len in sorted(verify_lens):
            try:
                pin(int(verify_len))
            except RuntimeError as exc:
                # A rung the server's ladder lacks. Raising out of the sweep
                # threw away every already-collected sample (nothing is
                # written until the sweep returns); a partial ladder is still
                # a table, and the skip is counted in the output.
                print(f"[dspark-sps] server: pin L={verify_len} refused "
                      f"({exc}); skipping the rung", file=sys.stderr)
                continue
            for batch_size in sorted(batch_sizes):
                # The load must outlive the measurement window, or the
                # cell measures ramp-down rows filed under smaller batch
                # sizes. Budget the output by the window at a conservative
                # 30 ms/step lower bound.
                window_steps = int((settle_s + poll_s) / 0.03) + 64
                max_tokens = max(64, window_steps * (1 + int(verify_len)))
                num_requests = int(batch_size) * max(1, int(dp_size))
                prompts = [[rng.randrange(1000, 30000) for _ in range(input_len)]
                           for _ in range(num_requests)]
                bodies = [{"model": model, "prompt": p, "max_tokens": max_tokens,
                           "temperature": 0.0, "ignore_eos": True, "stream": False}
                          for p in prompts]
                threads = [threading.Thread(target=_fire, args=(b, ),
                                            daemon=True) for b in bodies]
                for thread in threads:
                    thread.start()
                time.sleep(settle_s)          # let admission reach the plateau
                drain_metrics()               # discard the ramp
                time.sleep(poll_s)
                rows = drain_metrics()
                # The length is NOT read back from telemetry: /metrics carries
                # no generation-token count (iter_states reaches the log line
                # only), and it does not need to -- the sweep pinned it, and
                # the pin is refused by the server if it could not be honoured.
                # The row count is what the step really ran; the padded width
                # is resolved from the ladder when one was supplied, since a
                # partly-filled batch costs what its captured shape costs.
                #
                # hostStepTimeMS, not iterLatencyMS: under the overlap
                # scheduler the latter spans about two loops (base_worker says
                # so where it publishes both), which would inflate every cell
                # by roughly a factor of two.
                kept = 0
                cell_rows: List[StepSample] = []
                for row in rows:
                    rows_seen = int(row.get("numActiveRequests") or 0)
                    step_ms = float(row.get("hostStepTimeMS") or 0.0)
                    if rows_seen <= 0 or step_ms <= 0:
                        continue
                    padded_bs = rows_seen
                    if padded_batch_sizes:
                        wider = [int(b) for b in sorted(padded_batch_sizes)
                                 if int(b) >= rows_seen]
                        if not wider:
                            continue
                        padded_bs = wider[0]
                    cell_rows.append(StepSample(batch_size=padded_bs,
                                                verify_len=int(verify_len),
                                                step_time_ms=step_ms))
                    kept += 1
                # Applied, not just printed: the first rows after a pin change
                # can straddle the previous window's shape.
                out.extend(cell_rows[warmup_steps:])
                note = ""
                if not kept:
                    if post_errors:
                        note = f" -- requests refused: {post_errors[0]}"
                    elif not rows:
                        note = (" -- /metrics returned nothing: is "
                                "enable_iter_perf_stats on?")
                    else:
                        keys = sorted(rows[-1])[:8]
                        note = (f" -- {len(rows)} stats rows carried no usable "
                                f"(numActiveRequests, hostStepTimeMS); keys "
                                f"seen: {keys}")
                print(f"[dspark-sps] server bs={batch_size:>5} L={verify_len} "
                      f"({num_requests} requests in flight) "
                      f"kept {max(kept - warmup_steps, 0)}"
                      f" steps{note}", file=sys.stderr)
                for thread in threads:
                    thread.join(timeout=1.0)
    finally:
        # Never leave a served deployment pinned to a profiling window.
        pin(None)

    if server_log:
        # Preferred source when the server's log is reachable. /metrics needs
        # a metrics collector (return_perf_metrics) to run its tee buffer, and
        # its fallback path -- draining the stats queue directly -- came back
        # empty on an MPI deployment. The iteration log needs neither: it
        # carries the row count, the padded token total and the host step time
        # for every step, so the samples are labelled by what ran rather than
        # by what was asked for. The pin's job is then only to make sure every
        # rung is VISITED.
        from_log = samples_from_iter_log(
            [server_log], max_draft_len=max_draft_len,
            max_batch_size=(max(padded_batch_sizes) if padded_batch_sizes
                            else 256),
            padded_batch_sizes=padded_batch_sizes)
        print(f"[dspark-sps] server log yielded {len(from_log)} steps "
              f"(/metrics yielded {len(out)})", file=sys.stderr)
        if from_log:
            return from_log
    return out


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
        # Saved samples may come from any source, including deployment logs
        # full of ramp-occupancy cells; require nothing, keep what is thick.
        expected_cells = frozenset()
    elif args.from_iter_log:
        ladder = (_int_list(args.cuda_graph_batch_sizes)
                  if args.cuda_graph_batch_sizes else None)
        samples = samples_from_iter_log(
            args.from_iter_log, max_draft_len=args.max_draft_len,
            max_batch_size=max(ladder) if ladder else 256,
            padded_batch_sizes=ladder)
        if not samples:
            raise SystemExit(
                "no usable steps in the iteration logs. They must come from a "
                "run with print_iter_log AND enable_iter_perf_stats enabled: "
                "the token total lives in the `states` dict at the end of each "
                "line, and without it a step cannot be filed under a cell.")
        _write_samples(samples_out, samples)
        print(f"[dspark-sps] {len(samples)} steps from "
              f"{len(args.from_iter_log)} iteration log(s)", file=sys.stderr)
        provenance = {"source": "iter_log", "logs": list(args.from_iter_log)}
        # A production log has no requested grid at all.
        expected_cells = frozenset()
    elif args.base_url:
        samples = samples_from_server(
            base_url=args.base_url.rstrip("/"),
            control_url=(args.control_url.rstrip("/")
                         if args.control_url else None),
            batch_sizes=batch_sizes,
            verify_lens=verify_lens,
            input_len=args.input_len,
            model=args.served_model_name,
            warmup_steps=args.warmup_steps,
            settle_s=float(args.server_settle_s),
            poll_s=float(args.server_poll_s),
            max_draft_len=args.max_draft_len,
            seed=args.seed,
            padded_batch_sizes=(_int_list(args.cuda_graph_batch_sizes)
                                if args.cuda_graph_batch_sizes else None),
            server_log=args.server_log,
            # The bs axis is per-rank under attention DP (see the collector's
            # docstring); the flags describing the deployment double as the
            # fan-out factor.
            dp_size=(int(args.tp_size) if args.enable_attention_dp else 1))
        if not samples:
            raise SystemExit(
                "the server sweep collected nothing. Check that /metrics "
                "returns iteration stats (enable_iter_perf_stats) and that "
                "/dspark/verify_len_pin exists on this build.")
        _write_samples(samples_out, samples)
        print(f"[dspark-sps] {len(samples)} steps from {args.base_url}",
              file=sys.stderr)
        provenance = {"source": "server_sweep", "base_url": args.base_url}
        # Which cells the sweep can PROMISE depends on who controls the batch
        # size. Without attention DP the server schedules one global batch, so
        # a steady client concurrency IS the row count and the requested grid
        # is enforceable. Under attention DP the router spreads the load and
        # each rank's occupancy is emergent -- 1024 requests over 8 ranks
        # oscillates around 128/rank and pads to whatever ladder entry the
        # drift lands on (observed: requested 128, measured at 192; only
        # the cap-saturated 256 landed exactly). The pin guarantees L; bs is
        # whatever the deployment did. Require nothing, keep what is thick,
        # and let the fit's grid-connectivity check catch a real hole.
        expected_cells = (frozenset(
            (int(b), int(l)) for b in batch_sizes for l in verify_lens)
                          if not args.enable_attention_dp else frozenset())
    else:
        if not args.model:
            raise SystemExit("--model is required unless --fit-only is used")
        extra: Dict[str, object] = {}
        if args.extra_llm_api_options:
            import yaml

            with open(args.extra_llm_api_options, encoding="utf-8") as handle:
                extra = yaml.safe_load(handle) or {}
        fracs = _float_list(args.fracs) if args.fracs else None
        if fracs:
            if args.verify_lens:
                raise SystemExit("--fracs and --verify-lens are mutually "
                                 "exclusive; the frac sweep's M is measured, "
                                 "not pinned")
            # The frac sweep runs the constant block with live windows; the
            # nominal verify_len only labels the samples and sizes the token
            # budget, and max_draft_len satisfies every geometry guard.
            verify_lens = [int(args.max_draft_len)]
        config = SweepConfig(
            model=args.model,
            speculative_model=args.speculative_model or args.model,
            batch_sizes=batch_sizes,
            verify_lens=verify_lens,
            fracs=fracs,
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            enable_attention_dp=args.enable_attention_dp,
            max_draft_len=args.max_draft_len,
            input_len=args.input_len,
            sts_path=args.sts_path,
            seed_table_path=args.seed_table,
            stats_poll_s=float(args.stats_poll_s),
            block_follows_verify_len=bool(args.block_follows_verify_len),
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            repeats=args.repeats,
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
            engine_tiers=(_int_list(args.engine_tiers)
                          if args.engine_tiers else None),
        )
        if fracs:
            samples = run_frac_sweep(config)
            # Which buckets a frac visits is the planner's choice, not a
            # promise; thin incidental buckets are dropped by summarize_cells.
            expected_cells = None
        else:
            samples = run_sweep(config)
            expected_cells = frozenset(
                (int(b), int(l)) for b in batch_sizes for l in verify_lens)
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
            # _prepare_environment pins COMPACT for both sweep flavours; the
            # pinned sweep's uniform shape comes from the pin, not the mode.
            "ragged_verify_mode": RaggedVerifyMode.COMPACT.value,
            **({"fracs": [float(f) for f in fracs]} if fracs else {}),
        }

    cells = summarize_cells(samples,
                            # Frac cells discard warmup inside _run_cell (per
                            # cell, by iteration); trimming again here would
                            # re-charge it against every (bs, M) bucket.
                            warmup_steps=(0 if args.fracs else args.warmup_steps),
                            min_samples=args.min_samples,
                            expected_cells=expected_cells)
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
            # The engine fingerprint the loader verifies (check_table_fingerprint).
            # Facts only, and only facts this process can VOUCH for. The loader
            # compares every key present in both dicts and refuses on mismatch,
            # so a key must be absent -- not defaulted -- when the fact is
            # unknown: recording ep as `args.ep_size or 0` fabricated ep=0
            # whenever the flag was unset, the loader's Mapping normalizes
            # unset to 1, and the profiler's own table was refused on the very
            # engine that produced it. The deployment-measuring sources
            # (--base-url, --from-iter-log) build no engine here at all, so
            # they attach no engine dict; the deployment's own config is the
            # fingerprint authority there, and an unfingerprinted table loads
            # with a warning rather than a false claim.
            **({"engine": {
                **({"tp": int(args.tp_size)}
                   if args.tp_size is not None else {}),
                **({"ep": int(args.ep_size)}
                   if args.ep_size else {}),
                "attention_dp": bool(args.enable_attention_dp),
                "block": block_size,
                "max_batch_size": max(_int_list(args.batch_sizes)),
                "geometry": ("block_follows_verify_len"
                             if getattr(args, "block_follows_verify_len", False)
                             else "constant_block"),
            }} if provenance.get("source") == "sweep" else {}),
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
