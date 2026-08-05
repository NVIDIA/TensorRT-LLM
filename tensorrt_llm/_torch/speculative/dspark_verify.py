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
"""Host-side coordinator turning DSpark confidence into a per-iteration draft length.

Three constraints shape this component, and all three point the same way:

1. **The decision must be on the host.** TensorRT-LLM picks a decode CUDA graph
   from ``(batch_size, draft_len, ...)`` -- every input to that choice is a host
   value. A device-resident budget would have to be copied back before the batch
   could be shaped, which is the synchronization we are trying to avoid.
2. **The step must land on a captured graph.** A shape with no captured graph
   does not raise; ``maybe_get_cuda_graph`` returns ``None`` and the step
   silently runs eager, which costs far more than the tokens saved. The drafted
   block is always ``max_tier`` long, so what varies is the *verified* token
   total, and the layout rounds that up to a captured verify bucket before the
   batch is shaped.
3. **Every rank must decide identically.** The graph key is not part of the
   attention-DP consistency allgather, so two ranks that shape the batch
   differently select different graphs -- one replays, one falls back to eager
   -- and their collectives diverge. The caller therefore reduces the ragged
   decision across ranks (a single allgather in ``py_executor``) before any rank
   acts on it; :meth:`decide_verify_lens` itself is rank-local by default.

Confidence for step ``t`` only becomes readable on the host at ``t+1`` (or ``t+2``
with the overlap scheduler), so the decision always runs on a lagged snapshot.
That is fine -- it is a throughput heuristic, not a correctness input -- but the
snapshot has to be *taken* without blocking, and a snapshot that is not ready yet
must degrade to verifying everything rather than to a stale guess.
"""

import os
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch

from ..._utils import prefer_pinned
from ...logger import logger
from .dspark_planner import SpsCostTable, compute_verify_token_budget
from .dspark_schedule import (NEUTRAL_CONFIDENCE_LOGIT, DSparkScheduleConfig,
                              compute_survival, schedule_verify_lens_topk)

#: Pin every request's verify window to this length, bypassing the planner.
#: Set by the SPS profiler to move M at a fixed draft length; also used to
#: hold the full block during STS collection, where a trimmed window would
#: clip the acceptance label and fit a temperature to the scheduler instead
#: of to the head. Not a serving knob.
FORCE_VERIFY_LEN_ENV = "TLLM_DSPARK_FORCE_VERIFY_LEN"

__all__ = ["DSparkVerifyPlanner"]


class DSparkVerifyPlanner:
    """Stages confidence off the device and turns it into a draft length.

    Args:
        cfg: verify-length bounds.
        cost_table: measured step cost. A flat table means "not profiled"; the
            planner then refuses to trim (see :meth:`decide_verify_lens`).
        tiers: draft lengths that have a captured CUDA graph. The drafted block
            is always ``max_tier``; these also bound the budget search.
        apply_calibration: maps raw confidence logits to probabilities. Normally
            ``confidence_head.apply_sts``; defaults to a plain sigmoid.
        all_rank_max: cross-rank reduction, returning the max of ``value`` over
            ranks. ``None`` for single-rank runs.
    """

    def __init__(
        self,
        *,
        cfg: DSparkScheduleConfig,
        cost_table: Optional[SpsCostTable] = None,
        tiers: Optional[Sequence[int]] = None,
        apply_calibration: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        all_rank_max: Optional[Callable[[int], int]] = None,
    ):
        self.cfg = cfg
        self.cost_table = cost_table
        self.tiers: List[int] = sorted({int(t) for t in (tiers or [cfg.resolved_max_verify_len])})
        self.apply_calibration = apply_calibration or torch.sigmoid
        self.all_rank_max = all_rank_max

        # Profiling pin. The SPS profiler has to hold M at a known value while it
        # times steps, and it cannot use the planner's own decision to do that:
        # the planner needs a cost table, and the cost table is what the profiler
        # is producing. SGLang solves this with `dspark_force_budget_frac` pushed
        # to a running server; TRT-LLM has no runtime control plane at
        # world_size > 1, so this is an env var read ONCE here -- constant per
        # process, zero cost on the step path, and forwarded to every worker at
        # spawn, so all ranks agree by construction rather than by a collective.
        #
        # Absolute length, not SGLang's fraction. `schedule_verify_lens_topk`
        # drops candidates below survival_eps, so a fractional budget yields
        # `granted <= budget` and `choose_ragged_capture_shape` then rounds the
        # total DOWN onto a smaller captured bucket -- the cell would be timed at
        # one M and filed under another. With an absolute L every request gets
        # exactly L, so M = bs * (L + 1) is the captured bucket by construction.
        self._forced_verify_len = self._read_forced_verify_len()

        self._host_buffer: Optional[torch.Tensor] = None
        self._copy_event: Optional[torch.cuda.Event] = None
        self._host_stamps: Optional[torch.Tensor] = None
        self._staged_seq: Optional[int] = None
        self._snapshot_valid = False
        # Observability: silent degradation is the main failure mode here, so
        # count every path that gives up on trimming.
        self.stats = {
            "decisions": 0,
            "fallback_no_snapshot": 0,
            "fallback_flat_cost": 0,
            "fallback_no_confidence": 0,
            "fallback_short_snapshot": 0,
            "fallback_no_gen_requests": 0,
            "forced_steps": 0,
        }

    @property
    def max_tier(self) -> int:
        return self.tiers[-1] if self.tiers else self.cfg.resolved_max_verify_len

    def stage_confidence(self, confidence_logits: torch.Tensor,
                         stamps: Optional[torch.Tensor] = None,
                         staged_seq: Optional[int] = None) -> None:
        """Start a non-blocking device->host copy of the confidence buffer.

        ``confidence_logits`` is the worker's whole slot-indexed buffer, staged
        in full so :meth:`decide_verify_lens` can look up any live request by
        slot. It is kilobytes (``max_num_requests x block`` fp32), so there is
        nothing to gain from staging a subset -- and a subset would need a row
        ordering to survive the one-iteration lag, which is exactly what slot
        keying exists to avoid.

        Never synchronizes: the copy is issued on the current stream and an event
        is recorded. :meth:`decide_verify_lens` polls that event and simply does
        not use the snapshot if it has not landed.
        """
        if confidence_logits is None or confidence_logits.shape[0] == 0:
            return
        if self._host_buffer is None or self._host_buffer.shape != confidence_logits.shape:
            self._host_buffer = torch.empty(
                confidence_logits.shape,
                dtype=torch.float32,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self._copy_event = torch.cuda.Event()
        self._host_buffer.copy_(confidence_logits, non_blocking=True)
        if stamps is not None:
            if (self._host_stamps is None
                    or self._host_stamps.shape != stamps.shape):
                self._host_stamps = torch.empty(stamps.shape,
                                                dtype=torch.int32,
                                                device="cpu",
                                                pin_memory=prefer_pinned())
            self._host_stamps.copy_(stamps, non_blocking=True)
            self._staged_seq = staged_seq
        self._copy_event.record()
        self._snapshot_valid = True

    def _ready_snapshot(self) -> Optional[torch.Tensor]:
        if not self._snapshot_valid or self._host_buffer is None:
            return None
        # query() is non-blocking; a not-yet-landed copy just means this step
        # decides without it.
        if self._copy_event is not None and not self._copy_event.query():
            return None
        return self._host_buffer

    _FORCE_ENV = FORCE_VERIFY_LEN_ENV

    def _read_forced_verify_len(self) -> Optional[int]:
        """Parse and validate the profiling pin, or None."""
        raw = os.environ.get(self._FORCE_ENV, "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(
                f"{self._FORCE_ENV}={raw!r} is not an integer") from None
        lo, hi = int(self.cfg.min_verify_len), int(self.cfg.resolved_max_verify_len)
        if not lo <= value <= hi:
            raise ValueError(
                f"{self._FORCE_ENV}={value} outside [{lo}, {hi}]")
        if value not in self.tiers:
            # A length with no captured graph makes every step run eager, and an
            # eager step is orders of magnitude off the graph-replay cost the
            # profiler is trying to measure -- the curve would be noise wearing
            # the shape of a cost model. Refuse rather than produce it.
            raise ValueError(
                f"{self._FORCE_ENV}={value} is not in the captured tier ladder "
                f"{self.tiers}; every step would fall out of CUDA-graph replay "
                f"and the measured times would not be step costs")
        logger.warning(
            f"DSpark: verify length PINNED to {value} by {self._FORCE_ENV}. "
            f"This bypasses the planner entirely and is a profiling mode, not a "
            f"serving configuration.")
        return value

    def _note_snapshot_stats(self, selected: torch.Tensor,
                             rows: Optional[Sequence[int]]) -> None:
        """Measure what the argmax is about to eat. Never changes behaviour.

        Two independent instruments:
        - stamp-lag histogram: (staged_seq - row stamp) per gathered row. A
          healthy relay shows one tight mode; mass at large lags is stale
          replay masquerading as data, and the clamp bucket collects
          never-drafted rows.
        - neutral-row count: rows still carrying the neutral fill on every
          position, i.e. "unknown" rows entering the argmax as certainties.
        Together they say whether the per-step snapshot content can be
        trusted; without them the planner's input was never measured at all.
        """
        try:
            n = int(selected.shape[0])
            self.stats["snap_rows"] = self.stats.get("snap_rows", 0) + n
            neutral = int((selected >= NEUTRAL_CONFIDENCE_LOGIT - 1.0).all(dim=1).sum())
            self.stats["snap_neutral_rows"] = (
                self.stats.get("snap_neutral_rows", 0) + neutral)
            if (self._host_stamps is not None and self._staged_seq is not None
                    and rows is not None):
                idx = torch.as_tensor(list(rows), dtype=torch.long)
                idx = idx.clamp_(0, self._host_stamps.shape[0] - 1)
                lags = (int(self._staged_seq)
                        - self._host_stamps[idx].to(torch.int64))
                hist = self.stats.setdefault("stamp_lag_hist", {})
                for lag, cnt in zip(*[t.tolist() for t in lags.unique(
                        return_counts=True)]):
                    key = int(max(-2, min(int(lag), 8)))
                    hist[key] = hist.get(key, 0) + int(cnt)
        except Exception as exc:  # noqa: BLE001 - instruments must not kill steps
            self.stats["snap_stats_errors"] = (
                self.stats.get("snap_stats_errors", 0) + 1)
            if self.stats["snap_stats_errors"] == 1:
                logger.warning(f"DSpark snapshot stats failed once: {exc}")

    def _gather_rows(
        self, *, num_gen_requests: int, rows: Optional[Sequence[int]]
    ) -> Optional[torch.Tensor]:
        """This step's confidence, one row per generation request, or None.

        ``rows`` is the per-request buffer row index (``DSparkWorker.
        confidence_row_for``). Supplying it is what makes the lagged snapshot
        correct: the batch is reshuffled between the step that wrote the
        confidence and the step that reads it, so the request at position ``i``
        is routinely a different one.

        ``rows=None`` keeps the positional reading for callers that own the
        ordering themselves (the unit tests stage a purpose-built buffer). That
        path refuses to run on a snapshot with fewer rows than the batch rather
        than silently returning a short answer -- a short answer becomes a
        partially-assigned batch downstream, where half the requests get a
        verify window and half do not.
        """
        snapshot = self._ready_snapshot()
        if snapshot is None:
            self.stats["fallback_no_snapshot"] += 1
            return None
        if rows is None:
            if snapshot.shape[0] < num_gen_requests:
                self.stats["fallback_short_snapshot"] += 1
                return None
            selected = snapshot[:num_gen_requests]
        else:
            if len(rows) != num_gen_requests:
                self.stats["fallback_short_snapshot"] += 1
                return None
            selected = snapshot[torch.as_tensor(list(rows), dtype=torch.long)]
        if selected.numel() == 0:
            self.stats["fallback_no_confidence"] += 1
            return None
        return selected

    def decide_verify_lens(
        self,
        *,
        num_gen_requests: int,
        rows: Optional[Sequence[int]] = None,
        all_rank_max: Optional[Callable[[int], int]] = None,
        reduce_across_ranks: bool = True,
    ) -> Optional[List[int]]:
        """Per-request verify lengths for a ragged step, or None to stay uniform.

        The budget still comes from the cost-model argmax -- that is what knows
        how many verified tokens the step can afford -- but instead of splitting
        it evenly it is handed to the global survival top-k, so a request whose
        draft is still alive at depth 5 can take positions from one whose draft
        died at depth 1.

        Returns None whenever the uniform path would have fallen back (no
        snapshot, flat cost table, empty batch): raggedness is a throughput
        heuristic and an untrustworthy input must degrade to verifying
        everything, not to a stale split.

        The batch-wide *maximum* is still reduced across ranks, because the
        drafted-token buffer width and the per-request padding are sized from
        it; the individual lengths stay local.

        The returned list always has exactly ``num_gen_requests`` entries, or is
        None. A partial list would leave the tail of the batch without a verify
        window, and the callers downstream disagree about what that means: the
        input layout is built per request (so it goes ragged) while the spec
        metadata sees a ``None`` and stays uniform.
        """
        # Counted before any decline, so the four fallback counters below have a
        # denominator. Without it they are absolute counts against an unknown
        # total, which is how a run reported `planner_declined: 2` alongside an
        # all-zero planner block: nothing said how many decisions there had
        # been, so nothing said whether zero fallbacks meant "healthy" or
        # "never asked".
        self.stats["decisions"] += 1

        if num_gen_requests <= 0:
            # Benign during ramp-up, but it has to be nameable: an unlabelled
            # `return None` here is indistinguishable in the counters from a
            # real decline, and this feature's failures are all silent.
            self.stats["fallback_no_gen_requests"] += 1
            return None

        if self._forced_verify_len is not None:
            # Ahead of the cost-table gate on purpose: the profiler runs without
            # a table (that is the artifact it is producing), so a pin placed
            # after this gate would never fire and the run would look entirely
            # normal while measuring the untrimmed block.
            #
            # It sets ``lens`` rather than returning, so the pinned path still
            # runs the cross-rank agreement and the length assert below. An
            # early return skipped both -- harmless while the pin equals the
            # block (every rank agrees trivially and the packed buffers are
            # full-width anyway), which is why arm B ran clean, but it left the
            # short-window case taking a different route through the same code
            # than any real decision ever does.
            self.stats["forced_steps"] += 1
            lens = [int(self._forced_verify_len)] * int(num_gen_requests)
        else:
            if self.cost_table is None or self.cost_table.is_flat:
                self.stats["fallback_flat_cost"] += 1
                return None
            selected = self._gather_rows(num_gen_requests=num_gen_requests, rows=rows)
            if selected is None:
                return None

            self._note_snapshot_stats(selected, rows)
            probs = self.apply_calibration(selected)
            survival = compute_survival(probs)
            # Scored with the tau the step will actually collect. The rung is
            # still chosen from the same ladder over the same cost grid -- the
            # token totals, the buckets and the captured graphs are unchanged --
            # but budget_argmax_over_uniform_lens scores rung L as if every
            # request took columns [min, L), while what runs is a top-k
            # redistribution of the same B = n*(L - min) tokens. Top-k takes the
            # B largest survivals, so the realised tau is >= the scored one at
            # an identical cost, for every rung. The gap is not constant across
            # rungs, so the argmax could sit on the wrong one; scoring what runs
            # removes the question.
            budget = compute_verify_token_budget(
                survival=survival.numpy().astype(np.float64),
                num_gen_requests=int(num_gen_requests),
                cost_table=self.cost_table,
                min_verify_len=self.cfg.min_verify_len,
                max_verify_len=self.cfg.resolved_max_verify_len,
                # Without this the answer is unrealisable: any non-tier total is
                # rounded up to a captured bucket, which can push a budget
                # chosen to sit below a cost riser back over it.
                allowed_lens=self.tiers,
            )
            # The price the argmax just paid for this step, kept so the run can
            # be reconciled against reality: hostStepTimeMS measures what the
            # step actually cost, and a systematic gap between the two is the
            # one number that catches a wrong table, a wrong lookup, and an
            # engine/table config mismatch alike (same cell measured 23% apart
            # across two engine configs). These keys ride along in self.stats,
            # which the [final] summary prints wholesale -- no extra plumbing.
            floor_tokens = int(num_gen_requests) * (self.cfg.min_verify_len + 1)
            predicted = float(self.cost_table.step_times(
                np.asarray([floor_tokens + int(budget)]),
                int(num_gen_requests))[0])
            self.last_predicted_step_ms = predicted
            self.stats["predicted_ms_sum"] = (
                self.stats.get("predicted_ms_sum", 0.0) + predicted)
            self.stats["predicted_steps"] = (
                self.stats.get("predicted_steps", 0) + 1)
            lens = schedule_verify_lens_topk(survival=survival, budget=budget, cfg=self.cfg).tolist()
            # What the argmax bought and what top-k handed out, recorded
            # SEPARATELY: the realized windows in verify_len_hist are read back
            # after the bucket fit widened them, so without these two counters
            # nothing says what this rank's own decision was -- and a fit-side
            # defect is indistinguishable from a planner-side one.
            rung = self.cfg.min_verify_len + int(budget) // max(int(num_gen_requests), 1)
            rhist = self.stats.setdefault("local_rung_hist", {})
            rhist[rung] = rhist.get(rung, 0) + 1
            lhist = self.stats.setdefault("local_len_hist", {})
            for v in lens:
                lhist[int(v)] = lhist.get(int(v), 0) + 1

        # `reduce_across_ranks=False` means the caller is doing the cross-rank
        # agreement itself. That is not just an optimization here: every early
        # return above is rank-local (an empty batch, an unprofiled cost table,
        # a confidence snapshot whose copy event has not landed yet -- that last
        # one is timing-dependent), so a collective issued *after* them is issued
        # by some ranks and not others. Deciding locally and reducing once,
        # unconditionally, at a point every rank reaches is the only shape of
        # this that cannot deadlock.
        if reduce_across_ranks:
            all_rank_max = all_rank_max or self.all_rank_max
            if all_rank_max is not None:
                agreed_max = int(all_rank_max(int(max(lens))))
                lens = [min(int(v), agreed_max) for v in lens]
        assert len(lens) == num_gen_requests, (
            f"internal: produced {len(lens)} verify lengths for {num_gen_requests} requests"
        )
        return [int(v) for v in lens]
