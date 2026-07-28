# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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
"""Emission-assisted GVR top-k state for the DSA decode path.

Owns the persistent (graph-address-stable) buffers the emission tiers
ride on, the device-side closed-loop seed-row update (pure tensor ops:
CUDA-graph capturable, validated 11/11-step replay-exact against eager)
and the per-step routing decision. All of it is opt-in: without the
flag the DSA decode path is byte-identical to before.

Tier semantics (see gvr_routing):
  * this step's TOP-K consumes what the PREVIOUS step's indexer
    epilogue emitted;
  * this step's INDEXER emits what the routing planned for the NEXT
    step. N changes by at most one slot per step, so tier flapping is
    a non-issue.
"""

from typing import Optional

import torch

from ...cute_dsl_kernels.blackwell.top_k.gvr_routing import TopkRoute, pick_config, plan_emission

# Bucketed list geometry (validated defaults: B* = 8192 segment cap,
# 24576-entry C segment; see the f15/f17 sweeps).
LIST_SEG_A = 8192
LIST_CAP_C = 24576
LIST_WIDTH = 2 * LIST_SEG_A + LIST_CAP_C

# Closed-loop line derivation around the published k-th anchor: t1
# hugs the k-th value from below, t0/t2 guard by the (anchor - kth)
# span. Matches the graph_test.py-validated update.
GUARD_LO = 2.0
GUARD_HI = 0.5


class GvrExtState:
    """Per-attention-backend emission state (persistent buffers)."""

    def __init__(
        self, max_rows: int, top_k: int, device: torch.device, enable_list_tier: bool = True
    ):
        self.max_rows = max_rows
        self.top_k = top_k
        # packed seed row: lines at cols 0..2, counts (emission-filled)
        # at 3..5, adaptive-skip pass count at 6
        self.seed_row = torch.zeros((max_rows, 8), dtype=torch.float32, device=device)
        self.xstate = torch.zeros((max_rows, 8), dtype=torch.float32, device=device)
        self.cand_vals: Optional[torch.Tensor] = None
        self.cand_idx: Optional[torch.Tensor] = None
        self.cand_ctl: Optional[torch.Tensor] = None
        self.cand_cur: Optional[torch.Tensor] = None
        if enable_list_tier:
            self.cand_vals = torch.zeros((max_rows, LIST_WIDTH), dtype=torch.float32, device=device)
            self.cand_idx = torch.zeros((max_rows, LIST_WIDTH), dtype=torch.int32, device=device)
            self.cand_ctl = torch.zeros((max_rows, 4), dtype=torch.int32, device=device)
            self.cand_cur = torch.zeros((max_rows, 4), dtype=torch.int32, device=device)
        # GVR warm-start feedback: this layer's previous-step top-k
        # (same stable-address feedback-loop shape as
        # heuristic_prev_topk; zero-init -> first step's pre_idx points
        # at index 0, a valid benign candidate)
        self.prev_topk = torch.zeros((max_rows, top_k), dtype=torch.int32, device=device)
        # block_max prefix ([rows, nb_pad*4] fp32 warp-partials),
        # allocated lazily once max_seq_len is known
        self.block_max: Optional[torch.Tensor] = None
        # tier the PREVIOUS indexer call emitted (what this step's
        # top-k may consume); "rungs" until the first emission lands
        self.emitted_tier = "rungs"

    def ensure_block_max(self, max_seq_len: int) -> torch.Tensor:
        nb4 = ((max_seq_len + 255) // 256 * 256) // 128 * 4
        if self.block_max is None or self.block_max.shape[1] < nb4:
            self.block_max = torch.zeros(
                (self.max_rows, nb4), dtype=torch.float32, device=self.seed_row.device
            )
        return self.block_max

    def plan(self, batch: int, n_comp: int, num_sms: int) -> tuple[str, TopkRoute]:
        """Route this step: (tier to EMIT next, launch knobs to CONSUME
        what was emitted last step)."""
        emit_tier = plan_emission(batch, n_comp, self.top_k, have_epilogue=True)
        route = pick_config(self.emitted_tier, batch, n_comp, self.top_k, num_sms)
        return emit_tier, route

    def update_seed_rows(self, num_rows: int) -> None:
        """Device-side closed-loop line update from the last publish.

        Pure tensor ops (graph-capturable). Rows whose xstate is not
        valid (col 0 == 0, e.g. cold start) get non-finite lines, which
        the kernel's validity guard routes to the stock path - the
        closed loop never rides on host data quality.
        """
        s = self.seed_row[:num_rows]
        x = self.xstate[:num_rows]
        kth = x[:, 1]
        anch = torch.maximum(x[:, 2], kth + 1e-5)
        span = (anch - kth).clamp_min(1e-4)
        valid = x[:, 0] > 0
        inf = torch.full_like(kth, float("inf"))
        s[:, 0] = torch.where(valid, kth - GUARD_LO * span, inf)
        s[:, 1] = torch.where(valid, kth - 1e-6, inf)
        s[:, 2] = torch.where(valid, kth + GUARD_HI * span, inf)
        s[:, 3:8] = 0.0
        if self.cand_ctl is not None:
            self.cand_ctl[:num_rows].zero_()
            self.cand_cur[:num_rows].zero_()

    def indexer_emit_kwargs(self, emit_tier: str, num_rows: int) -> dict:
        """kwargs for CuteDSLFP4PagedMQALogitsRunner.forward covering the
        planned emission tier (caller merges into its call)."""
        kw: dict = {}
        if emit_tier in ("counts", "list"):
            kw.update(emit_seed_counts=True, seed_thr=self.seed_row[:num_rows])
        if emit_tier == "list":
            kw.update(
                emit_cand_bucketed=True,
                accept_cap=LIST_SEG_A,
                cand_out=self.cand_vals[:num_rows],
                cand_idx_out=self.cand_idx[:num_rows],
                cand_ctl_out=self.cand_ctl[:num_rows],
                cand_cur_out=self.cand_cur[:num_rows],
            )
        self.emitted_tier = emit_tier
        return kw

    def topk_ext_kwargs(
        self, route: TopkRoute, num_rows: int, block_max: Optional[torch.Tensor]
    ) -> dict:
        """kwargs for trtllm::cute_dsl_gvr_topk_decode consuming the
        PREVIOUS step's emission per the picked route."""
        kw: dict = {
            "xstate": self.xstate[:num_rows],
            "cluster_size": route.cluster_size,
        }
        if route.num_threads is not None:
            kw["num_threads"] = route.num_threads
        if route.tier in ("counts", "list"):
            kw["seed_thr"] = self.seed_row[:num_rows]
        # rungs tier (first step / no emission yet): pass no seed at
        # all - cold-start xstate is invalid so the lines would be
        # non-finite anyway; the plain stock path is the right fallback
        # (a [rows, 3] column view of the packed row is non-contiguous
        # and would trip the runner's contract assert)
        if route.tier == "list":
            kw.update(
                cand_vals=self.cand_vals[:num_rows],
                cand_idx=self.cand_idx[:num_rows],
                cand_ctl=self.cand_ctl[:num_rows],
            )
        if route.attach_block_max and block_max is not None:
            kw["block_max"] = block_max
        return kw
