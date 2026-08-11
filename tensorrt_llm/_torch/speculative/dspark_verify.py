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

Contracts: the decision is made on the host (CUDA-graph selection is a host
value); the verified token total must round up to a captured verify bucket, or
the step silently runs eager; and every rank must shape the batch identically
-- the caller reduces the ragged decision across ranks before any rank acts on
it, while :meth:`decide_verify_lens` itself is rank-local by default.
Confidence is read from a lagged, non-blocking snapshot; a snapshot that is
not ready degrades to verifying everything, never to a stale guess.
"""

import os
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..._utils import prefer_pinned
from ...logger import logger
from .dspark_planner import SpsCostTable, compute_verify_token_budget
from .dspark_schedule import (NEUTRAL_CONFIDENCE_LOGIT, DSparkScheduleConfig,
                              compute_survival, schedule_verify_lens_topk)

#: Profiling-only pin (SPS profiling / STS collection): fixes every request's
#: verify window at this length, bypassing the planner. Not a serving knob.
FORCE_VERIFY_LEN_ENV = "TLLM_DSPARK_FORCE_VERIFY_LEN"

#: Profiling-only budget override (SPS profiling on the live ragged path):
#: replaces the cost-table argmax with ``frac`` of the maximum trimmable
#: budget, while windows still flow through the real confidence top-k. Unlike
#: the verify-length pin this keeps the executed shape production-faithful
#: (ragged windows, bucket fit); the profiler sweeps it to visit off-diagonal
#: (bs, M) cells. Not a serving knob.
FORCE_BUDGET_FRAC_ENV = "TLLM_DSPARK_FORCE_BUDGET_FRAC"

#: Wire encoding for the frac's cross-rank agreement: fractions travel as
#: fixed-point ints so the allgather payload stays homogeneous ints.
_FRAC_WIRE_SCALE = 10_000

#: Device-side window selection (the paper's dual-timescale split): the host
#: decides only the CAPACITY -- (padded_bs, bucket, budget) -- from a snapshot
#: one step older than today's (lag-2 by the block-relative count, matching
#: SGLang), and a pre-replay device prologue re-ranks the batch with the
#: verified block's own fresh confidence (lag-0). Constant per process, read
#: once at init on every rank.
DEVICE_WINDOWS_ENV = "TLLM_DSPARK_DEVICE_WINDOWS"

#: Experiment-only mixed-step guard: when set, a rank whose scheduled batch
#: carries context (prefill) requests declines to trim that step, and the
#: cross-rank all-or-nothing gate reverts the whole group to the full block.
#: The cost table is profiled on pure-decode steps and cannot price a trim
#: whose step time is dominated by prefill compute; this isolates that effect.
SKIP_MIXED_TRIM_ENV = "TLLM_DSPARK_SKIP_MIXED_TRIM"


def device_windows_enabled() -> bool:
    """Whether device-side window selection is on (process-constant env)."""
    return os.environ.get(DEVICE_WINDOWS_ENV, "0") == "1"


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

        # Profiling pin, read once at init: constant per process and forwarded
        # to every worker at spawn, so all ranks agree without a collective.
        # An absolute length (not a fraction) so M = bs * (L + 1) lands exactly
        # on a captured bucket.
        self._forced_verify_len = self._read_forced_verify_len()
        # Runtime pin queued but not yet in force, held as its wire value:
        # -1 nothing queued, 0 clear the pin, >0 pin to that length
        # (min_verify_len >= 1, so 0 is unambiguous). Adopted through the
        # step's cross-rank allgather so every rank starts pinning on the
        # same step; applying it locally would diverge the shape gate.
        self._pending_verify_len_pin: int = -1
        # Budget-fraction override (same lifecycle as the pin: env at init,
        # runtime requests adopted through the step's allgather). Wire values:
        # -1 nothing queued, 0 clear, >0 frac * _FRAC_WIRE_SCALE.
        self._forced_budget_frac = self._read_forced_budget_frac()
        self._pending_budget_frac: int = -1

        self._host_buffer: Optional[torch.Tensor] = None
        self._copy_event: Optional[torch.cuda.Event] = None
        self._host_stamps: Optional[torch.Tensor] = None
        self._staged_seq: Optional[int] = None
        self._snapshot_valid = False

        # Device-window mode: ranking happens on device with fresh confidence,
        # so the host snapshot serves ONLY the capacity argmax -- and that one
        # deliberately reads a snapshot one staging older (lag-2), per the
        # paper's design: capacity tolerates staleness, ranking does not.
        # Two pinned buffers alternate; the pair we are NOT writing this step
        # is the older, guaranteed-landed one.
        self.device_windows = os.environ.get(DEVICE_WINDOWS_ENV, "0") == "1"
        # Constant per process like the pin: every rank reads the same env at
        # spawn, so the per-step decline needs no extra collective.
        self.skip_mixed_trim = os.environ.get(SKIP_MIXED_TRIM_ENV, "0") == "1"
        if self.device_windows and self._forced_verify_len is not None:
            raise ValueError(
                f"{DEVICE_WINDOWS_ENV}=1 is incompatible with "
                f"{FORCE_VERIFY_LEN_ENV}: the pin fixes host windows that "
                f"device selection would silently override")
        self._prev_buffer: Optional[torch.Tensor] = None
        self._prev_event: Optional[torch.cuda.Event] = None
        self._prev_stamps: Optional[torch.Tensor] = None
        self._prev_seq: Optional[int] = None
        self._prev_valid = False
        # Count every path that declines to trim; degradation here is silent.
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
        slot across the one-iteration lag. Never synchronizes: the copy is
        issued on the current stream and an event is recorded;
        :meth:`decide_verify_lens` polls the event and skips a snapshot that
        has not landed.
        """
        if confidence_logits is None or confidence_logits.shape[0] == 0:
            return
        if self.device_windows:
            # Rotate before overwriting: what was current becomes the older
            # (lag-2) snapshot the budget argmax reads; the old older buffer's
            # storage is reused for this staging. Pure reference swaps.
            self._host_buffer, self._prev_buffer = (self._prev_buffer,
                                                    self._host_buffer)
            self._copy_event, self._prev_event = (self._prev_event,
                                                  self._copy_event)
            self._host_stamps, self._prev_stamps = (self._prev_stamps,
                                                    self._host_stamps)
            self._staged_seq, self._prev_seq = self._prev_seq, self._staged_seq
            self._prev_valid = self._snapshot_valid
            self._snapshot_valid = False
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
            raise ValueError(
                f"{self._FORCE_ENV}={value} is not in the captured tier ladder "
                f"{self.tiers}; every step would fall out of CUDA-graph replay "
                f"and the measured times would not be step costs")
        logger.warning(
            f"DSpark: verify length PINNED to {value} by {self._FORCE_ENV}. "
            f"This bypasses the planner entirely and is a profiling mode, not a "
            f"serving configuration.")
        return value

    def validate_verify_len_pin(self, value: Optional[int]) -> Optional[int]:
        """Check a pin against this planner's ladder; None clears the pin."""
        if value is None:
            return None
        value = int(value)
        lo, hi = int(self.cfg.min_verify_len), int(self.cfg.resolved_max_verify_len)
        if not lo <= value <= hi:
            raise ValueError(f"verify-length pin {value} outside [{lo}, {hi}]")
        if value not in self.tiers:
            raise ValueError(
                f"verify-length pin {value} is not in the captured tier ladder "
                f"{self.tiers}; every step would fall out of CUDA-graph replay "
                f"and the measured times would not be step costs")
        return value

    def _read_forced_budget_frac(self) -> Optional[float]:
        raw = os.environ.get(FORCE_BUDGET_FRAC_ENV, "").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(
                f"{FORCE_BUDGET_FRAC_ENV}={raw!r} is not a float") from None
        value = self.validate_budget_frac(value)
        logger.warning(
            f"DSpark: verify budget FORCED to {value} of the trimmable "
            f"maximum by {FORCE_BUDGET_FRAC_ENV}. Windows still follow "
            f"confidence; this is a profiling mode, not a serving "
            f"configuration.")
        return value

    def validate_budget_frac(self, value: Optional[float]) -> Optional[float]:
        """Check a budget fraction; None clears the override."""
        if value is None:
            return None
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"budget fraction {value} outside (0.0, 1.0]")
        return value

    @property
    def forced_budget_frac(self) -> Optional[float]:
        """The budget fraction currently in force, or None."""
        return self._forced_budget_frac

    @property
    def forced_verify_len(self) -> Optional[int]:
        """The verify-length pin currently in force, or None."""
        return self._forced_verify_len

    def request_budget_frac(self, value: Optional[float]) -> Optional[float]:
        """Queue a budget-fraction override (``None`` clears) for the next step.

        Validated here but NOT applied: the step's allgather carries it so all
        ranks adopt it together (see :meth:`adopt_budget_frac`). Returns the
        wire-quantized value that will actually be adopted, so the caller sees
        what applies rather than what it asked for.
        """
        value = self.validate_budget_frac(value)
        wire = (0 if value is None else
                max(1, int(round(value * _FRAC_WIRE_SCALE))))
        self._pending_budget_frac = wire
        return None if value is None else min(1.0, wire / _FRAC_WIRE_SCALE)

    def pending_budget_frac(self) -> int:
        """The queued request as a wire value: -1 none, 0 clear, >0 fixed-point."""
        return int(self._pending_budget_frac)

    def adopt_budget_frac(self, wire_value: int) -> None:
        """Apply the group's agreed budget fraction from the allgathered payload."""
        wire_value = int(wire_value)
        if wire_value < 0:
            return
        # Same non-atomic compare-and-clear as the pin above: an RPC-thread
        # request landing between the compare and the store is dropped. Both
        # knobs are single-operator profiling controls, so the race is
        # tolerated for both rather than fixed for one.
        if self._pending_budget_frac == wire_value:
            self._pending_budget_frac = -1
        new_frac = (None if wire_value == 0 else
                    min(1.0, wire_value / _FRAC_WIRE_SCALE))
        if new_frac == self._forced_budget_frac:
            return
        self._forced_budget_frac = new_frac
        logger.warning(
            f"DSpark: verify budget fraction set to {new_frac} at runtime. "
            f"This bypasses the cost table and is a profiling mode, not a "
            f"serving configuration.")

    def request_verify_len_pin(self, value: Optional[int]) -> Optional[int]:
        """Queue a pin (or ``None`` to clear) for the next decode step.

        Validated here but NOT applied: the step's allgather carries it so all
        ranks adopt it together (see ``adopt_verify_len_pin``).
        """
        value = self.validate_verify_len_pin(value)
        self._pending_verify_len_pin = 0 if value is None else int(value)
        return value

    def pending_verify_len_pin(self) -> int:
        """The queued request as a wire value: -1 none, 0 clear, >0 pin."""
        return int(self._pending_verify_len_pin)

    def adopt_verify_len_pin(self, wire_value: int) -> None:
        """Apply the group's agreed pin from the allgathered payload.

        Every rank applies the same payload, so the pin takes effect on the
        same step everywhere. ``-1`` leaves the current pin alone; ``0``
        clears it.
        """
        wire_value = int(wire_value)
        if wire_value < 0:
            return
        # Compare-and-clear: the RPC thread may queue a new request between
        # the payload being read and adopted; an unconditional reset would
        # drop it.
        if self._pending_verify_len_pin == wire_value:
            self._pending_verify_len_pin = -1
        new_pin = None if wire_value == 0 else wire_value
        if new_pin == self._forced_verify_len:
            return
        self._forced_verify_len = new_pin
        logger.warning(
            f"DSpark: verify length pin set to {new_pin} at runtime. This "
            f"bypasses the planner and is a profiling mode, not a serving "
            f"configuration.")

    def _note_snapshot_stats(self, selected: torch.Tensor,
                             rows: Optional[Sequence[int]],
                             stamps: Optional[torch.Tensor] = None,
                             staged_seq: Optional[int] = None) -> None:
        """Record snapshot-quality stats: stamp-lag histogram (staleness per
        gathered row) and neutral-row count ("unknown" rows entering the
        argmax as certainties). Observability only; never changes behaviour.
        ``stamps``/``staged_seq`` override the current snapshot's pair when the
        caller read the older (lag-2) one.
        """
        try:
            stamps = self._host_stamps if stamps is None else stamps
            staged_seq = self._staged_seq if staged_seq is None else staged_seq
            n = int(selected.shape[0])
            self.stats["snap_rows"] = self.stats.get("snap_rows", 0) + n
            neutral = int((selected >= NEUTRAL_CONFIDENCE_LOGIT - 1.0).all(dim=1).sum())
            self.stats["snap_neutral_rows"] = (
                self.stats.get("snap_neutral_rows", 0) + neutral)
            if (stamps is not None and staged_seq is not None
                    and rows is not None):
                idx = torch.as_tensor(list(rows), dtype=torch.long)
                idx = idx.clamp_(0, stamps.shape[0] - 1)
                lags = (int(staged_seq)
                        - stamps[idx].to(torch.int64))
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

    def _ready_older_snapshot(self) -> Optional[torch.Tensor]:
        """The lag-2 snapshot (device-window mode's capacity input), or None.

        By construction its copy was issued a full staging earlier, so its
        event has practically always landed; the poll stays for the same
        fail-closed reason as :meth:`_ready_snapshot`.
        """
        if not self._prev_valid or self._prev_buffer is None:
            return None
        if self._prev_event is not None and not self._prev_event.query():
            return None
        return self._prev_buffer

    def _gather_rows(
        self, *, num_gen_requests: int, rows: Optional[Sequence[int]],
        snapshot: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """This step's confidence, one row per generation request, or None.

        ``rows`` is the per-request buffer row index (``DSparkWorker.
        confidence_row_for``); the batch is reshuffled across the one-iteration
        lag, so positional reads (``rows=None``) are only correct for callers
        that own the ordering. Declines (None) rather than returning fewer
        rows than the batch, which would partially assign verify windows
        downstream. ``snapshot`` overrides the default (current) snapshot when
        the caller already picked one (the lag-2 budget read).
        """
        if snapshot is None:
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

    def decide_verify_budget(
        self,
        *,
        num_gen_requests: int,
        rows: Optional[Sequence[int]] = None,
    ) -> Optional[Tuple[int, List[int]]]:
        """Capacity only, for device-window mode: ``(budget, shape_lens)``.

        The budget (verify tokens above the per-request floor) comes from the
        cost-table argmax over the OLDER staged snapshot -- lag-2 by the
        block-relative count, the paper's prescription: capacity tolerates
        staleness, ranking does not. ``shape_lens`` is a canonical uniform
        spread of the budget used ONLY for the cross-rank shape agreement and
        bucket fit (both read totals, never the split); the true per-request
        windows are chosen on device from the verified block's own confidence.
        Fail-closed like :meth:`decide_verify_lens`: None means verify the
        full block.
        """
        self.stats["decisions"] += 1
        if num_gen_requests <= 0:
            self.stats["fallback_no_gen_requests"] += 1
            return None
        n = int(num_gen_requests)
        trimmable = self.cfg.resolved_max_verify_len - self.cfg.min_verify_len
        if self._forced_budget_frac is not None:
            # The frac sweep stays honest in device mode: it replaces only the
            # argmax; the device top-k still spends the budget for real.
            budget = int(round(self._forced_budget_frac * n * trimmable))
            self.stats["forced_budget_steps"] = (
                self.stats.get("forced_budget_steps", 0) + 1)
        else:
            if self.cost_table is None or self.cost_table.is_flat:
                self.stats["fallback_flat_cost"] += 1
                return None
            snapshot = self._ready_older_snapshot()
            if snapshot is None:
                self.stats["fallback_no_snapshot"] += 1
                return None
            selected = self._gather_rows(num_gen_requests=n, rows=rows,
                                         snapshot=snapshot)
            if selected is None:
                return None
            self._note_snapshot_stats(selected, rows,
                                      stamps=self._prev_stamps,
                                      staged_seq=self._prev_seq)
            survival = compute_survival(self.apply_calibration(selected))
            budget = compute_verify_token_budget(
                survival=survival.numpy().astype(np.float64),
                num_gen_requests=n,
                cost_table=self.cost_table,
                min_verify_len=self.cfg.min_verify_len,
                max_verify_len=self.cfg.resolved_max_verify_len,
                allowed_lens=self.tiers,
            )
            floor_tokens = n * (self.cfg.min_verify_len + 1)
            predicted = float(self.cost_table.step_times(
                np.asarray([floor_tokens + int(budget)]), n)[0])
            self.last_predicted_step_ms = predicted
            self.stats["predicted_ms_sum"] = (
                self.stats.get("predicted_ms_sum", 0.0) + predicted)
            self.stats["predicted_steps"] = (
                self.stats.get("predicted_steps", 0) + 1)
        budget = max(0, min(int(budget), n * trimmable))
        base, extra = divmod(budget, n)
        shape_lens = [
            min(self.cfg.min_verify_len + base + (1 if i < extra else 0),
                self.cfg.resolved_max_verify_len) for i in range(n)
        ]
        return budget, shape_lens

    def decide_verify_lens(
        self,
        *,
        num_gen_requests: int,
        rows: Optional[Sequence[int]] = None,
        all_rank_max: Optional[Callable[[int], int]] = None,
        reduce_across_ranks: bool = True,
        budget_override: Optional[int] = None,
    ) -> Optional[List[int]]:
        """Per-request verify lengths for a ragged step, or None to stay uniform.

        The cost-model argmax sets the budget; the global survival top-k splits
        it across requests. Fail-closed: any untrustworthy input (no snapshot,
        flat cost table, empty batch) returns None and the step verifies
        everything. The batch-wide maximum is reduced across ranks (it sizes
        the drafted-token buffer and per-request padding); individual lengths
        stay local. Returns exactly ``num_gen_requests`` entries or None,
        never a partial list.
        """
        # Counted before any decline so the fallback counters have a denominator.
        self.stats["decisions"] += 1

        if num_gen_requests <= 0:
            self.stats["fallback_no_gen_requests"] += 1
            return None

        if budget_override is not None:
            # Diagnostic: spend a caller-supplied budget (e.g. the lag-2
            # argmax's) through the normal CURRENT-snapshot top-k. Under
            # cap-accept this measures a schedule's true acceptance cost --
            # execution stays full-block, the window is only an accounting
            # cap -- so two budgets can be compared with identical compute.
            selected = self._gather_rows(num_gen_requests=num_gen_requests,
                                         rows=rows)
            if selected is None:
                return None
            self._note_snapshot_stats(selected, rows)
            survival = compute_survival(self.apply_calibration(selected))
            lens = schedule_verify_lens_topk(survival=survival,
                                             budget=int(budget_override),
                                             cfg=self.cfg).tolist()
        elif self._forced_verify_len is not None:
            # Must stay ahead of the cost-table gate (the profiler runs without
            # a table) and must set `lens` rather than return, so the pinned
            # path still runs the cross-rank agreement below.
            self.stats["forced_steps"] += 1
            lens = [int(self._forced_verify_len)] * int(num_gen_requests)
        elif self._forced_budget_frac is not None:
            # Also ahead of the cost-table gate (the frac sweep is how the
            # table gets built). Unlike the pin, the windows stay real: the
            # forced fraction only replaces the argmax'd budget, and the
            # confidence top-k spends it exactly as in production.
            selected = self._gather_rows(num_gen_requests=num_gen_requests,
                                         rows=rows)
            if selected is None:
                return None
            self._note_snapshot_stats(selected, rows)
            survival = compute_survival(self.apply_calibration(selected))
            trimmable = (self.cfg.resolved_max_verify_len
                         - self.cfg.min_verify_len)
            budget = int(round(self._forced_budget_frac
                               * int(num_gen_requests) * trimmable))
            self.stats["forced_budget_steps"] = (
                self.stats.get("forced_budget_steps", 0) + 1)
            lens = schedule_verify_lens_topk(survival=survival, budget=budget,
                                             cfg=self.cfg).tolist()
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
            # Scored as the top-k redistribution that actually runs, on the
            # same tier-aligned cost grid as the uniform argmax.
            budget = compute_verify_token_budget(
                survival=survival.numpy().astype(np.float64),
                num_gen_requests=int(num_gen_requests),
                cost_table=self.cost_table,
                min_verify_len=self.cfg.min_verify_len,
                max_verify_len=self.cfg.resolved_max_verify_len,
                # Required: any non-tier total is rounded up to a captured
                # bucket, pushing a budget chosen below a cost riser back over it.
                allowed_lens=self.tiers,
            )
            # Predicted step cost, recorded so runs can be reconciled against
            # measured hostStepTimeMS; a systematic gap flags a wrong or
            # mismatched cost table.
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
            # Local decision histograms, recorded before the bucket fit widens
            # the realized windows (verify_len_hist is post-fit).
            rung = self.cfg.min_verify_len + int(budget) // max(int(num_gen_requests), 1)
            rhist = self.stats.setdefault("local_rung_hist", {})
            rhist[rung] = rhist.get(rung, 0) + 1
            lhist = self.stats.setdefault("local_len_hist", {})
            for v in lens:
                lhist[int(v)] = lhist.get(int(v), 0) + 1

        # The early returns above are rank-local (some timing-dependent), so no
        # collective may run before this point; reduce once, unconditionally,
        # at a point every rank reaches. reduce_across_ranks=False means the
        # caller does that agreement itself.
        if reduce_across_ranks:
            all_rank_max = all_rank_max or self.all_rank_max
            if all_rank_max is not None:
                agreed_max = int(all_rank_max(int(max(lens))))
                lens = [min(int(v), agreed_max) for v in lens]
        assert len(lens) == num_gen_requests, (
            f"internal: produced {len(lens)} verify lengths for {num_gen_requests} requests"
        )
        return [int(v) for v in lens]
