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

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..._utils import prefer_pinned
from .dspark_planner import ExactSpsCostTable, SpsCostTable, compute_verify_token_budget
from .dspark_schedule import DSparkScheduleConfig, compute_survival, schedule_verify_lens_topk


@dataclass(frozen=True)
class ExactSpsLocalDecision:
    """One rank's pre-collective yield curve for exact ``(G,V)`` cells."""

    num_requests: int
    survival: torch.Tensor
    native_expected_yield: float
    compact_expected_yields: Tuple[float, ...]


def _exact_cell_geometry(
    *,
    num_real: int,
    graph_batch_size: int,
    verifier_budget: int,
    min_verify_len: int,
    max_verify_len: int,
) -> Optional[Tuple[int, int, int]]:
    """Return ``(pad_tokens, real_tokens, extra_budget)`` or infeasible.

    Real-containing ranks use one shared pad-row length. A zero-real rank may
    use the deterministic quotient/remainder split ``V = G*q + r`` backed by
    two cached dummy objects (low=q, high=q+1); the returned pad length is q.
    The same calculation is reused by yield modeling and execution so the
    policy cannot price one layout and run another.
    """
    n = int(num_real)
    graph_batch_size = int(graph_batch_size)
    verifier_budget = int(verifier_budget)
    if n < 0 or n > graph_batch_size or graph_batch_size <= 0:
        return None
    if n == 0:
        # Row zero is the scheduled ADP dummy; CUDA padding contributes the
        # remaining G-1 rows. Two cached CUDA dummy variants represent the
        # quotient/remainder split without allocating G distinct KV objects.
        pad_tokens, remainder = divmod(verifier_budget, graph_batch_size)
        if pad_tokens < 1 or pad_tokens + int(remainder > 0) > int(max_verify_len) + 1:
            return None
        return pad_tokens, 0, 0
    num_pad = graph_batch_size - n
    real_floor = n * (int(min_verify_len) + 1)
    real_capacity = n * (int(max_verify_len) + 1)
    if num_pad == 0:
        if not real_floor <= verifier_budget <= real_capacity:
            return None
        real_tokens = verifier_budget
        return 0, real_tokens, real_tokens - real_floor

    pad_capacity = int(max_verify_len) + 1
    pad_lo = max(1, -(-(verifier_budget - real_capacity) // num_pad))
    pad_hi = min(pad_capacity, (verifier_budget - real_floor) // num_pad)
    if pad_lo > pad_hi:
        return None
    pad_tokens = pad_lo
    real_tokens = verifier_budget - num_pad * pad_tokens
    if not real_floor <= real_tokens <= real_capacity:
        return None
    return pad_tokens, real_tokens, real_tokens - real_floor


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
        device_windows: use the lag-2 snapshot for capacity while a device
            prologue ranks the current requests.
        skip_mixed_trim: fail closed on mixed context/generation steps whose
            cost is not represented by a pure-decode table.
    """

    def __init__(
        self,
        *,
        cfg: DSparkScheduleConfig,
        cost_table: Optional[SpsCostTable | ExactSpsCostTable] = None,
        tiers: Optional[Sequence[int]] = None,
        apply_calibration: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        all_rank_max: Optional[Callable[[int], int]] = None,
        device_windows: bool = False,
        skip_mixed_trim: bool = False,
    ):
        self.cfg = cfg
        self.cost_table = cost_table
        self.tiers: List[int] = sorted({int(t) for t in (tiers or [cfg.resolved_max_verify_len])})
        self.apply_calibration = apply_calibration or torch.sigmoid
        self.all_rank_max = all_rank_max

        self._host_buffer: Optional[torch.Tensor] = None
        self._copy_event: Optional[torch.cuda.Event] = None
        self._snapshot_valid = False

        # Device-window mode: ranking happens on device with fresh confidence,
        # so the host snapshot serves ONLY the capacity argmax -- and that one
        # deliberately reads a snapshot one staging older (lag-2), per the
        # paper's design: capacity tolerates staleness, ranking does not.
        # Two pinned buffers alternate; the pair we are NOT writing this step
        # is the older, guaranteed-landed one.
        self.device_windows = bool(device_windows)
        self.skip_mixed_trim = bool(skip_mixed_trim)
        self._prev_buffer: Optional[torch.Tensor] = None
        self._prev_event: Optional[torch.cuda.Event] = None
        self._prev_valid = False
        # Count every path that declines to trim; degradation here is silent.
        self.stats = {
            "decisions": 0,
            "fallback_no_snapshot": 0,
            "fallback_flat_cost": 0,
            "fallback_no_confidence": 0,
            "fallback_short_snapshot": 0,
            "fallback_no_gen_requests": 0,
            "fallback_full_k": 0,
        }

    def install_runtime_cost_table(self, cost_table: SpsCostTable | ExactSpsCostTable) -> None:
        """Install ModelEngine's already validated runtime cost object."""
        if isinstance(cost_table, ExactSpsCostTable):
            if cost_table.max_draft_len != self.cfg.resolved_max_verify_len:
                raise ValueError(
                    "Exact SPS max_draft_len does not match planner: "
                    f"table={cost_table.max_draft_len}, "
                    f"planner={self.cfg.resolved_max_verify_len}"
                )
        if self.cost_table is not None and self.cost_table != cost_table:
            raise RuntimeError(
                "DSpark planner already holds a different SPS cost object; "
                "graph capture and runtime scheduling must share ModelEngine's "
                "single validated instance"
            )
        self.cost_table = cost_table

    @property
    def exact_cost_table(self) -> Optional[ExactSpsCostTable]:
        return self.cost_table if isinstance(self.cost_table, ExactSpsCostTable) else None

    def prepare_exact_sps_decision(
        self,
        *,
        num_gen_requests: int,
        rows: Optional[Sequence[int]] = None,
    ) -> Optional[ExactSpsLocalDecision]:
        """Price every measured V locally before the existing allgather.

        The payload includes the whole measured grid because the common G is
        not known until attention-DP ranks exchange their row counts. Pad rows
        have no expected output and initially consume one anchor token; any V
        above real-row capacity is feasible but adds no yield.
        """
        table = self.exact_cost_table
        if table is None:
            raise RuntimeError("prepare_exact_sps_decision requires an exact SPS table")
        n = int(num_gen_requests)
        if n < 0:
            return None

        if n == 0:
            # An idle ADP rank has no confidence rows and contributes no
            # expected output. It still advertises every pad-only cell that the
            # low/high dummy pair can represent; resource readiness is agreed
            # by the executor before a selected cell can execute.
            max_verify_len = int(self.cfg.resolved_max_verify_len)
            compact_expected_yields = tuple(
                0.0
                if _exact_cell_geometry(
                    num_real=0,
                    graph_batch_size=graph_batch_size,
                    verifier_budget=verifier_budget,
                    min_verify_len=self.cfg.min_verify_len,
                    max_verify_len=max_verify_len,
                )
                is not None
                else -1.0
                for graph_batch_size, verifier_budget in table.candidate_cells()
            )
            return ExactSpsLocalDecision(
                num_requests=0,
                survival=torch.empty((0, max_verify_len), dtype=torch.float32),
                native_expected_yield=0.0,
                compact_expected_yields=compact_expected_yields,
            )

        if self.device_windows:
            snapshot = self._ready_older_snapshot()
            if snapshot is None:
                return None
            selected = self._gather_rows(
                num_gen_requests=n,
                rows=rows,
                snapshot=snapshot,
            )
        else:
            selected = self._gather_rows(
                num_gen_requests=n,
                rows=rows,
            )
        if selected is None:
            return None
        survival = compute_survival(self.apply_calibration(selected))
        floor = int(self.cfg.min_verify_len)
        max_verify_len = int(self.cfg.resolved_max_verify_len)
        # Build every exact-cell yield from one float64 decomposition. Mixing
        # torch float32 native/base reductions with a NumPy float64 optional
        # prefix can make a saturated compact cell appear better than native
        # solely because of reduction order.
        survival_values = survival.numpy().astype(np.float64, copy=False)
        base_expected_yield = float(n) + float(survival_values[:, :floor].sum(dtype=np.float64))
        candidates = np.sort(survival_values[:, floor:max_verify_len].reshape(-1))[::-1]
        prefix = np.concatenate(([0.0], np.cumsum(candidates, dtype=np.float64)))
        native_expected_yield = (
            base_expected_yield
            + float(prefix[-1])
            + float(survival_values[:, max_verify_len:].sum(dtype=np.float64))
        )
        compact_expected_yields = []
        for graph_batch_size, verifier_budget in table.candidate_cells():
            geometry = _exact_cell_geometry(
                num_real=n,
                graph_batch_size=graph_batch_size,
                verifier_budget=verifier_budget,
                min_verify_len=floor,
                max_verify_len=max_verify_len,
            )
            if geometry is None:
                compact_expected_yields.append(-1.0)
                continue
            _, _, extra = geometry
            compact_expected_yields.append(base_expected_yield + float(prefix[extra]))
        return ExactSpsLocalDecision(
            num_requests=n,
            survival=survival,
            native_expected_yield=native_expected_yield,
            compact_expected_yields=tuple(compact_expected_yields),
        )

    def allocate_exact_sps_candidate(
        self,
        decision: ExactSpsLocalDecision,
        *,
        graph_batch_size: int,
        verifier_budget: int,
    ) -> Tuple[List[int], int, int]:
        """Allocate one globally selected exact V on this rank.

        Returns per-request draft windows, the real-row budget above the
        mandatory floor, and the shared pad-row token length. The bucket fitter
        receives all three parts of the exact plan and may only validate them.
        """
        table = self.exact_cost_table
        if table is None:
            raise RuntimeError("allocate_exact_sps_candidate requires an exact SPS table")
        n = int(decision.num_requests)
        graph_batch_size = int(graph_batch_size)
        verifier_budget = int(verifier_budget)
        if verifier_budget not in table.production_candidate_budgets(graph_batch_size):
            raise ValueError(f"Unmeasured exact SPS cell G={graph_batch_size}, V={verifier_budget}")
        floor = int(self.cfg.min_verify_len)
        max_verify_len = int(self.cfg.resolved_max_verify_len)
        geometry = _exact_cell_geometry(
            num_real=n,
            graph_batch_size=graph_batch_size,
            verifier_budget=verifier_budget,
            min_verify_len=floor,
            max_verify_len=max_verify_len,
        )
        if geometry is None:
            raise ValueError(
                f"Exact SPS cell G={graph_batch_size}, V={verifier_budget} "
                f"cannot fit {n} real requests with verify floor {floor}"
            )
        pad_tokens, real_tokens, budget = geometry
        if n == 0:
            if real_tokens != 0 or budget != 0:
                raise RuntimeError("zero-real exact SPS geometry assigned real-row work")
            return [], 0, int(pad_tokens)
        exact_cfg = replace(self.cfg, survival_eps=0.0)
        lens = schedule_verify_lens_topk(
            survival=decision.survival, budget=budget, cfg=exact_cfg
        ).tolist()
        if sum(int(value) + 1 for value in lens) != real_tokens:
            raise RuntimeError("Exact SPS allocation did not spend its real-row token target")
        return [int(value) for value in lens], int(budget), int(pad_tokens)

    @property
    def max_tier(self) -> int:
        return self.tiers[-1] if self.tiers else self.cfg.resolved_max_verify_len

    def stage_confidence(self, confidence_logits: torch.Tensor) -> None:
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
            self._host_buffer, self._prev_buffer = (self._prev_buffer, self._host_buffer)
            self._copy_event, self._prev_event = (self._prev_event, self._copy_event)
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
        self,
        *,
        num_gen_requests: int,
        rows: Optional[Sequence[int]],
        snapshot: Optional[torch.Tensor] = None,
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
        if self.cost_table is None or self.cost_table.is_flat:
            self.stats["fallback_flat_cost"] += 1
            return None
        snapshot = self._ready_older_snapshot()
        if snapshot is None:
            self.stats["fallback_no_snapshot"] += 1
            return None
        selected = self._gather_rows(num_gen_requests=n, rows=rows, snapshot=snapshot)
        if selected is None:
            return None
        survival = compute_survival(self.apply_calibration(selected))
        budget = compute_verify_token_budget(
            survival=survival.numpy().astype(np.float64),
            num_gen_requests=n,
            cost_table=self.cost_table,
            min_verify_len=self.cfg.min_verify_len,
            max_verify_len=self.cfg.resolved_max_verify_len,
            allowed_lens=self.tiers,
        )
        full_budget = n * trimmable
        if int(budget) >= full_budget:
            # The full tier is not a ragged schedule. Returning None preserves
            # the ordinary static-K executor path and its lower overhead.
            self.stats["fallback_full_k"] += 1
            return None
        floor_tokens = n * (self.cfg.min_verify_len + 1)
        predicted = float(
            self.cost_table.step_times(np.asarray([floor_tokens + int(budget)]), n)[0]
        )
        self.last_predicted_step_ms = predicted
        self.stats["predicted_ms_sum"] = self.stats.get("predicted_ms_sum", 0.0) + predicted
        self.stats["predicted_steps"] = self.stats.get("predicted_steps", 0) + 1
        budget = max(0, min(int(budget), n * trimmable))
        base, extra = divmod(budget, n)
        shape_lens = [
            min(
                self.cfg.min_verify_len + base + (1 if i < extra else 0),
                self.cfg.resolved_max_verify_len,
            )
            for i in range(n)
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
            # Device-window mode chooses capacity from a lag-2 snapshot, then
            # spends that budget against the current snapshot here.
            selected = self._gather_rows(num_gen_requests=num_gen_requests, rows=rows)
            if selected is None:
                return None
            survival = compute_survival(self.apply_calibration(selected))
            lens = schedule_verify_lens_topk(
                survival=survival, budget=int(budget_override), cfg=self.cfg
            ).tolist()
        else:
            if self.cost_table is None or self.cost_table.is_flat:
                self.stats["fallback_flat_cost"] += 1
                return None
            selected = self._gather_rows(num_gen_requests=num_gen_requests, rows=rows)
            if selected is None:
                return None

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
            full_budget = int(num_gen_requests) * (
                self.cfg.resolved_max_verify_len - self.cfg.min_verify_len
            )
            if int(budget) >= full_budget:
                # Full K uses the native uniform path and its lower overhead.
                self.stats["fallback_full_k"] += 1
                return None
            # Predicted step cost, recorded so runs can be reconciled against
            # measured hostStepTimeMS; a systematic gap flags a wrong or
            # mismatched cost table.
            floor_tokens = int(num_gen_requests) * (self.cfg.min_verify_len + 1)
            predicted = float(
                self.cost_table.step_times(
                    np.asarray([floor_tokens + int(budget)]), int(num_gen_requests)
                )[0]
            )
            self.last_predicted_step_ms = predicted
            self.stats["predicted_ms_sum"] = self.stats.get("predicted_ms_sum", 0.0) + predicted
            self.stats["predicted_steps"] = self.stats.get("predicted_steps", 0) + 1
            lens = schedule_verify_lens_topk(
                survival=survival, budget=budget, cfg=self.cfg
            ).tolist()

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
