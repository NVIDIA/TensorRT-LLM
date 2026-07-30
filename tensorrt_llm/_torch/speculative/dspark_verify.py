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
2. **The decision must land on a captured draft length.** A length with no
   captured graph does not raise; ``maybe_get_cuda_graph`` returns ``None`` and
   the step silently runs eager, which costs far more than the tokens saved.
   Hence :meth:`decide_draft_len` only ever returns a value from ``tiers``.
3. **Every rank must decide identically.** ``draft_len`` is part of the graph key
   but is *not* part of the attention-DP consistency allgather, so two ranks that
   pick different lengths select different graphs -- one replays, one falls back
   to eager -- and their collectives diverge. :meth:`decide_draft_len` therefore
   reduces across ranks before returning.

Confidence for step ``t`` only becomes readable on the host at ``t+1`` (or ``t+2``
with the overlap scheduler), so the decision always runs on a lagged snapshot.
That is fine -- it is a throughput heuristic, not a correctness input -- but the
snapshot has to be *taken* without blocking, and a snapshot that is not ready yet
must degrade to verifying everything rather than to a stale guess.
"""

from typing import Callable, List, Optional, Sequence

import numpy as np
import torch

from ..._utils import prefer_pinned
from .dspark_planner import SpsCostTable, budget_argmax_over_uniform_lens
from .dspark_schedule import DSparkScheduleConfig, compute_survival, schedule_verify_lens_topk

__all__ = ["DSparkVerifyPlanner"]


class DSparkVerifyPlanner:
    """Stages confidence off the device and turns it into a draft length.

    Args:
        cfg: verify-length bounds.
        cost_table: measured step cost. A flat table means "not profiled"; the
            planner then refuses to trim (see :meth:`decide_draft_len`).
        tiers: draft lengths that have a captured CUDA graph. The returned length
            is always one of these.
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

        self._host_buffer: Optional[torch.Tensor] = None
        self._copy_event: Optional[torch.cuda.Event] = None
        self._snapshot_valid = False
        # Observability: silent degradation is the main failure mode here, so
        # count every path that gives up on trimming.
        self.stats = {
            "decisions": 0,
            "fallback_no_snapshot": 0,
            "fallback_flat_cost": 0,
            "fallback_no_confidence": 0,
        }

    @property
    def max_tier(self) -> int:
        return self.tiers[-1] if self.tiers else self.cfg.resolved_max_verify_len

    def stage_confidence(self, confidence_logits: torch.Tensor, num_rows: int) -> None:
        """Start a non-blocking device->host copy of this step's confidence.

        Never synchronizes: the copy is issued on the current stream and an event
        is recorded. :meth:`decide_draft_len` polls that event and simply does not
        use the snapshot if it has not landed.
        """
        if confidence_logits is None or num_rows <= 0:
            return
        rows = confidence_logits[:num_rows]
        if self._host_buffer is None or self._host_buffer.shape != rows.shape:
            self._host_buffer = torch.empty(
                rows.shape, dtype=torch.float32, device="cpu", pin_memory=prefer_pinned()
            )
            self._copy_event = torch.cuda.Event()
        self._host_buffer.copy_(rows, non_blocking=True)
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

    def decide_draft_len(
        self,
        *,
        num_gen_requests: int,
        all_rank_max: Optional[Callable[[int], int]] = None,
    ) -> int:
        """Choose this iteration's draft length, agreed across all ranks.

        Degrades to the largest captured tier (i.e. verify everything, today's
        behavior) whenever the inputs are not trustworthy: no confidence
        snapshot yet, or an unprofiled (flat) cost model under which every extra
        verified token looks free.

        ``all_rank_max`` overrides the constructor's reduction; the caller that
        owns the distributed handle usually supplies it here.
        """
        self.stats["decisions"] += 1
        chosen = self._decide_local(num_gen_requests=num_gen_requests)
        all_rank_max = all_rank_max or self.all_rank_max
        if all_rank_max is not None:
            # Max, not min: a rank that wanted to trim more simply verifies a few
            # extra tokens. Disagreeing on the graph key is the unrecoverable
            # outcome, so agreement matters more than the saving.
            chosen = int(all_rank_max(int(chosen)))
            if chosen not in self.tiers:
                chosen = min((t for t in self.tiers if t >= chosen), default=self.max_tier)
        return int(chosen)

    def decide_verify_lens(
        self,
        *,
        num_gen_requests: int,
        all_rank_max: Optional[Callable[[int], int]] = None,
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
        """
        if num_gen_requests <= 0:
            return None
        if self.cost_table is None or self.cost_table.is_flat:
            self.stats["fallback_flat_cost"] += 1
            return None
        snapshot = self._ready_snapshot()
        if snapshot is None:
            self.stats["fallback_no_snapshot"] += 1
            return None
        rows = snapshot[:num_gen_requests]
        if rows.numel() == 0:
            self.stats["fallback_no_confidence"] += 1
            return None

        probs = self.apply_calibration(rows)
        survival = compute_survival(probs)
        uniform_len = budget_argmax_over_uniform_lens(
            survival=survival.numpy().astype(np.float64),
            num_gen_requests=int(num_gen_requests),
            cost_table=self.cost_table,
            allowed_lens=self.tiers,
            min_verify_len=self.cfg.min_verify_len,
        )
        # Same total the uniform decision would have spent, redistributed.
        budget = int(num_gen_requests) * (int(uniform_len) - self.cfg.min_verify_len)
        lens = schedule_verify_lens_topk(survival=survival, budget=budget, cfg=self.cfg).tolist()

        all_rank_max = all_rank_max or self.all_rank_max
        if all_rank_max is not None:
            agreed_max = int(all_rank_max(int(max(lens))))
            lens = [min(int(v), agreed_max) for v in lens]
        return [int(v) for v in lens]

    def _decide_local(self, *, num_gen_requests: int) -> int:
        if num_gen_requests <= 0:
            return self.max_tier
        if self.cost_table is None or self.cost_table.is_flat:
            self.stats["fallback_flat_cost"] += 1
            return self.max_tier
        snapshot = self._ready_snapshot()
        if snapshot is None:
            self.stats["fallback_no_snapshot"] += 1
            return self.max_tier
        rows = snapshot[:num_gen_requests]
        if rows.numel() == 0:
            self.stats["fallback_no_confidence"] += 1
            return self.max_tier
        probs = self.apply_calibration(rows)
        survival = compute_survival(probs).numpy().astype(np.float64)
        return budget_argmax_over_uniform_lens(
            survival=survival,
            num_gen_requests=int(num_gen_requests),
            cost_table=self.cost_table,
            allowed_lens=self.tiers,
            min_verify_len=self.cfg.min_verify_len,
        )
