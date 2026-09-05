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
"""Host-side coordinator for exact DSpark ``(G, V)`` policy decisions."""

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..._utils import prefer_pinned
from .dspark_planner import ExactSpsCostTable
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
    """Stages confidence and prepares exact ``(G,V)`` policy inputs.

    Args:
        cfg: verify-length bounds.
        cost_table: authenticated exact ``T(G,V)`` table installed by
            ``ModelEngine`` before the first decision.
        apply_calibration: maps raw confidence logits to probabilities. Normally
            ``confidence_head.apply_sts``; defaults to a plain sigmoid.
        device_windows: use the lag-2 snapshot for capacity while a device
            prologue ranks the current requests.
    """

    def __init__(
        self,
        *,
        cfg: DSparkScheduleConfig,
        cost_table: Optional[ExactSpsCostTable] = None,
        apply_calibration: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device_windows: bool = False,
    ):
        self.cfg = cfg
        self.cost_table = cost_table
        self.apply_calibration = apply_calibration or torch.sigmoid

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
        self._prev_buffer: Optional[torch.Tensor] = None
        self._prev_event: Optional[torch.cuda.Event] = None
        self._prev_valid = False
        # Count every path that declines to trim; degradation here is silent.
        self.stats = {
            "fallback_no_snapshot": 0,
            "fallback_no_confidence": 0,
            "fallback_short_snapshot": 0,
        }

    def install_runtime_cost_table(self, cost_table: ExactSpsCostTable) -> None:
        """Install ModelEngine's already validated runtime cost object."""
        if not isinstance(cost_table, ExactSpsCostTable):
            raise TypeError("DSpark confidence scheduling requires an exact SPS cost table")
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
        return self.cost_table

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
    def max_verify_len(self) -> int:
        return self.cfg.resolved_max_verify_len

    def stage_confidence(self, confidence_logits: torch.Tensor) -> None:
        """Start a non-blocking device->host copy of the confidence buffer.

        ``confidence_logits`` is the worker's whole slot-indexed buffer, staged
        in full so exact-policy preparation can look up any live request by
        slot across the one-iteration lag. Never synchronizes: the copy is
        issued on the current stream and an event is recorded; policy
        preparation polls the event and skips a snapshot that has not landed.
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
