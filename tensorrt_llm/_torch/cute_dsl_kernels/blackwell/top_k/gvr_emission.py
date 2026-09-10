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
ride on, the device-side closed-loop seed-row update (pure tensor ops,
CUDA-graph capturable) and the per-step routing decision. Opt-in:
without the flag the DSA decode path is unchanged.

Tier semantics (see gvr_routing):
  * this step's TOP-K consumes what the PREVIOUS step's indexer
    epilogue emitted;
  * this step's INDEXER emits what the routing planned for the NEXT
    step.
"""

import math
from typing import Optional

import torch

from .gvr_routing import (
    LIST_EMIT_MAX_B,
    LIST_EMIT_MIN_N,
    PRESCORE_LIST_MAX_B,
    PRESCORE_LIST_MIN_N,
    TopkRoute,
    pick_config,
    plan_emission,
)

# Bucketed candidate-list geometry: two tight segments of LIST_SEG_A
# entries plus a LIST_CAP_C-entry loose segment.
LIST_SEG_A = 8192
LIST_CAP_C = 24576
LIST_WIDTH = 2 * LIST_SEG_A + LIST_CAP_C

__all__ = [
    "GvrEmissionState",
    "LIST_EMIT_MIN_N",
    "LIST_PARK_LINE",
    "PRESCORE_LIST_MAX_B",
    "PRESCORE_LIST_MIN_N",
]

# Closed-loop line placement: fit the slope of log2(count) vs threshold
# from the previous step's (lines, counts) and place the new lines at
# these K-relative target counts (t0 loosest .. t2 tightest).
LINE_TARGETS = (8.0, 5.0, 2.0)
# prescore: newest compressed positions sampled next to the previous top-k
# (rounded per K so the row's tile count divides evenly)
PRESCORE_RECENT = 4096
LIST_T0_TARGET = 2.5  # list tier: single collect-line target (xK)
LIST_T0_COUNT_MAX = 6144.0  # keep n0 inside the [K, segA] admission band
SLOPE_MIN = 0.05
SLOPE_MAX = 64.0

# No-fit fallback: multiplicative guards around the published k-th value
# (t1 hugs it from below; t0/t2 guard by GUARD_LO/GUARD_HI spans).
FALLBACK_REL = 2.0**0.125 - 1.0
FALLBACK_ABS = 1e-3
GUARD_LO = 2.0
GUARD_HI = 0.5

# List tier only: park the two tight lines above any score so every
# admitted entry lands in the loosest segment. Any finite value above
# the score range works; the kernel's eligibility check only needs the
# three lines increasing and the loosest one finite.
LIST_PARK_LINE = 1.0e30

# Prescore tier: sample = prev top-k plus both neighbors. The sample must
# strictly exceed K (turnover between steps otherwise degrades the bound)
# and must be deduplicated before ranking (duplicates inflate the sample
# k-th above the true k-th, breaking soundness).
PRESCORE_NEIGHBORS = 3

# Intra-step scratch shared across layers: emission and consumption
# happen inside one layer's forward, and layers run sequentially on the
# stream, so ONE pool serves every layer (addresses stay stable for
# CUDA-graph capture; growth-only, fail-loud under capture).
_SHARED_SCRATCH: dict = {}


def _shared_scratch(kind: str, shape: tuple, dtype: torch.dtype, device: torch.device):
    key = (kind, device.index)
    need = math.prod(shape)
    t = _SHARED_SCRATCH.get(key)
    if t is None or t.numel() < need:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"gvr_emission shared scratch '{kind}': (re)allocation "
                "requested during CUDA graph capture; run a warmup step first"
            )
        t = torch.zeros(need, dtype=dtype, device=device)
        _SHARED_SCRATCH[key] = t
    return t[:need].view(shape)


class GvrEmissionState:
    """Per-attention-backend emission state (persistent buffers)."""

    def __init__(
        self,
        max_rows: int,
        top_k: int,
        device: torch.device,
        enable_list_tier: bool = True,
        own_prior: bool = True,
        cand_rows_cap: Optional[int] = None,
    ):
        self.max_rows = max_rows
        self.top_k = top_k
        self.cand_rows_cap = LIST_EMIT_MAX_B if cand_rows_cap is None else cand_rows_cap
        # packed seed row: lines at cols 0..2, counts (emission-filled)
        # at 3..5, adaptive-skip pass count at 6
        self.seed_row = torch.zeros((max_rows, 8), dtype=torch.float32, device=device)
        # contiguous alias of the three lines for the rungs tier (a
        # [rows, 3] column view of the packed row is non-contiguous)
        self.seed_rungs = torch.zeros((max_rows, 3), dtype=torch.float32, device=device)
        self.xstate = torch.zeros((max_rows, 8), dtype=torch.float32, device=device)
        self.cand_vals: Optional[torch.Tensor] = None
        self.cand_idx: Optional[torch.Tensor] = None
        self.cand_ctl: Optional[torch.Tensor] = None
        self.cand_cur: Optional[torch.Tensor] = None
        if enable_list_tier:
            # the routing only ever picks the list tier at
            # batch <= LIST_EMIT_MAX_B, so the wide candidate buffers
            # need that many rows, not max_rows (~0.33 MB/row/layer)
            cand_rows = min(max_rows, self.cand_rows_cap)
            self.cand_vals = _shared_scratch(
                "cand_vals", (cand_rows, LIST_WIDTH), torch.float32, device
            )
            self.cand_idx = _shared_scratch(
                "cand_idx", (cand_rows, LIST_WIDTH), torch.int32, device
            )
            self.cand_ctl = _shared_scratch("cand_ctl", (cand_rows, 4), torch.int32, device)
            self.cand_cur = _shared_scratch("cand_cur", (cand_rows, 4), torch.int32, device)
            # prescore histogram, sized once so captured graphs
            # never hold a freed address (prescore runs on list steps only)
            from .gvr_prescore_fp4 import HIST_WORDS

            self.ps_hist = _shared_scratch("ps_hist", (cand_rows, HIST_WORDS), torch.int32, device)
        # previous-step top-k feedback (address-stable; zero-init ->
        # first step's pre_idx points at index 0, a benign candidate).
        # own_prior=False when the caller already keeps this state (the
        # TopK module rides metadata's gvr_prior_indices).
        self.prev_topk = (
            torch.zeros((max_rows, top_k), dtype=torch.int32, device=device) if own_prior else None
        )
        # neighbour-duplicate flags per prev entry (bit0 p-1, bit1 p+1, bit2
        # p+2 in the row's top-k), written by the top-k kernel epilogue for
        # the prescore; 7 = every neighbour treated as duplicate (sound)
        self.prev_flags = torch.full((max_rows, top_k), 7, dtype=torch.uint8, device=device)
        # block_max prefix ([rows, nb_pad*4] fp32 warp-partials),
        # allocated lazily once max_seq_len is known
        self.block_max: Optional[torch.Tensor] = None
        # prescore-tier K-cache geometry (see ensure_prescore)
        self._mini_tpb = 0
        self._mini_rec = 0

    def ensure_prescore(self, tokens_per_block: int, record_bytes: int, num_sms: int) -> None:
        """Record the K-cache geometry the prescore kernel addresses with."""
        del num_sms
        sample = PRESCORE_NEIGHBORS * self.top_k
        assert sample % tokens_per_block == 0, (
            f"prescore sample {sample} must be a multiple of tokens_per_block {tokens_per_block}"
        )
        assert record_bytes == 4 * (record_bytes // 4)
        self._mini_tpb = tokens_per_block
        self._mini_rec = record_bytes

    def prescore_lines(
        self,
        num_rows: int,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        weights: torch.Tensor,
        block_table: torch.Tensor,
        kv_lens: torch.Tensor,
        head_dim: int,
        n_pad: int,
        q_sf: torch.Tensor,
        prev_topk: Optional[torch.Tensor] = None,
    ) -> None:
        """Prescore the previous step's top-k on the current query.
        One CuTe DSL launch re-scores prev_topk plus both neighbors
        (deduplicated by the top-k epilogue's flags) and the newest
        positions straight out of the paged FP4 K-cache into the row's
        score histogram ``ps_hist``; the FP4 scorer derives the seed lines
        from it (``indexer_emit_kwargs``): t0 is the lower edge of the bin
        holding the sample k-th minus a relative slack, a sound lower bound
        of the true k-th (the sample is a sub-multiset of the row's scores);
        rows without K finite samples get non-finite lines. The top-k
        resets the histogram (``topk_ext_kwargs``). Static state only:
        graph-capturable. ``prev_topk`` is the caller-owned prior when the
        state does not own one (own_prior=False).
        """
        from .gvr_prescore_fp4 import prescore_cute_fp4, recent_for

        prev = self.prev_topk if prev_topk is None else prev_topk
        assert prev is not None, "prescore_lines needs the previous top-k"
        assert num_rows <= self.cand_ctl.shape[0]
        hist = self.ps_hist[:num_rows]
        # the kernels address the paged pool through its base pointer only
        # (block table x record bytes, 64-bit), so hand over a flat view capped
        # to the int32 element range the FFI marshalling accepts; engine pools
        # run to several GiB
        pool_u8 = kv_pool.view(torch.uint8).reshape(-1)
        pool_u8 = pool_u8[: min(pool_u8.numel(), 2**31 - 4096)]
        # block-scaled kernel: q scales as packed UE8M0 per head, recent window
        # sampled as well
        prescore_cute_fp4(
            prev[:num_rows],
            kv_lens[:num_rows].contiguous(),
            self.prev_flags[:num_rows],
            block_table[:num_rows],
            pool_u8,
            q[:num_rows].contiguous().view(torch.uint8).reshape(-1),
            q_sf[:num_rows].contiguous().view(torch.int32).reshape(num_rows, -1),
            weights[:num_rows].contiguous(),
            hist,
            self.seed_row[:num_rows],
            self.cand_ctl[:num_rows],
            self.cand_cur[:num_rows],
            top_k=self.top_k,
            num_heads=q.shape[2],
            head_dim=head_dim,
            tokens_per_block=self._mini_tpb,
            record_bytes=self._mini_rec,
            recent=recent_for(self.top_k, PRESCORE_RECENT),
        )

    def ensure_block_max(self, max_seq_len: int) -> torch.Tensor:
        nb4 = ((max_seq_len + 255) // 256 * 256) // 128 * 4
        # exact width: the runner asserts shape == (rows, nrec), so a
        # wider reused buffer would trip it
        if self.block_max is None or self.block_max.shape[1] != nb4:
            # allocating during CUDA graph capture would bake a dangling
            # address into the graph, so fail loudly instead
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "GvrEmissionState.ensure_block_max: (re)allocation requested "
                    "during CUDA graph capture; the block_max buffer must be "
                    "created by a warmup step before capture"
                )
            self.block_max = torch.zeros(
                (self.max_rows, nb4), dtype=torch.float32, device=self.seed_row.device
            )
        return self.block_max

    def plan(
        self,
        batch: int,
        n_comp: int,
        num_sms: int,
        compress_ratio: int = 4,
        list_max_b: Optional[int] = None,
        prescore: bool = False,
        list_min_n: Optional[int] = None,
    ) -> tuple[str, TopkRoute]:
        """Route this step: (tier the epilogue emits, launch knobs the
        top-k consumes it with)."""
        emit_tier = plan_emission(
            batch,
            n_comp,
            self.top_k,
            have_epilogue=True,
            compress_ratio=compress_ratio,
            list_max_b=list_max_b,
            prescore=prescore,
            list_min_n=list_min_n,
        )
        if emit_tier == "list" and (self.cand_vals is None or batch > self.cand_vals.shape[0]):
            # constructed with enable_list_tier=False: no candidate
            # buffers to emit into, demote to the counts tier
            emit_tier = "counts"
        # emission and consumption happen inside the SAME forward (zero,
        # emit, consume), so the consumer routes on this step's tier
        route = pick_config(emit_tier, batch, n_comp, self.top_k, num_sms)
        return emit_tier, route

    def update_seed_rows(self, num_rows: int, emit_tier: str = "counts") -> None:
        """Device-side closed-loop line update from the last publish.

        Slope-fits log2(count) vs threshold from the previous step's
        (lines, counts) and places the new lines at K-relative target
        counts. Counts come from the packed row (counts/list emission)
        or from the kernel's rung-count publish in xstate cols 4..6
        (rungs tier). Rows without a usable fit get multiplicative
        guards around the published k-th value; rows with invalid
        xstate (col 0 == 0, e.g. cold start) get non-finite lines,
        which the kernel's validity guard routes to the stock path.
        Pure tensor ops (graph-capturable).
        """
        s = self.seed_row[:num_rows]
        x = self.xstate[:num_rows]
        valid = x[:, 0] > 0
        kth = x[:, 1]
        anchor = x[:, 2]
        t_prev0 = s[:, 0]
        t_prev2 = s[:, 2]
        cnts = x[:, 4:7] if emit_tier == "rungs" else s[:, 3:6]
        k = float(self.top_k)
        inf = torch.full_like(kth, float("inf"))
        d_fb = kth.abs() * FALLBACK_REL + FALLBACK_ABS
        if emit_tier == "list":
            # two-point fit (t0_prev, n0) / (kth, K): kth is the exact
            # k-th boundary on list rows
            n0 = cnts[:, 0].clamp_min(1.0)
            dthr = (kth - t_prev0).clamp_min(1e-3)
            slope = ((torch.log2(n0) - math.log2(k)) / dthr).clamp(SLOPE_MIN, SLOPE_MAX)
            tgt0 = min(LIST_T0_TARGET * k, LIST_T0_COUNT_MAX)
            t0 = kth - math.log2(tgt0 / k) / slope
            fit_ok = torch.isfinite(t_prev0) & (n0 > k)
            t0 = torch.where(fit_ok, t0, kth - GUARD_LO * d_fb)
            park = torch.full_like(kth, LIST_PARK_LINE)
            new0 = torch.where(valid, t0, inf)
            new1 = torch.where(valid, park, inf)
            new2 = torch.where(valid, park + park, inf)
        else:
            c0 = cnts[:, 0].clamp_min(1.0)
            c2 = cnts[:, 2].clamp_min(1.0)
            dthr = (t_prev2 - t_prev0).clamp_min(1e-3)
            slope = ((torch.log2(c0) - torch.log2(c2)) / dthr).clamp(SLOPE_MIN, SLOPE_MAX)
            # anchor count estimate: slide the anchor onto the prev line fit
            anch_c = (c2 * torch.exp2(-(anchor - t_prev2) * slope)).clamp(1.0, 1e6)
            t0 = anchor + torch.log2(anch_c / (LINE_TARGETS[0] * k)) / slope
            t1 = anchor + torch.log2(anch_c / (LINE_TARGETS[1] * k)) / slope
            t2 = anchor + torch.log2(anch_c / (LINE_TARGETS[2] * k)) / slope
            # t_prev2 < 1e29 also rejects a parked line left by a tier flip
            fit_ok = torch.isfinite(t_prev0) & (t_prev2 < 1e29) & (c0 > c2)
            t0 = torch.where(fit_ok, t0, kth - GUARD_LO * d_fb)
            t1 = torch.where(fit_ok, t1, kth - 1e-6)
            t2 = torch.where(fit_ok, t2, kth + GUARD_HI * d_fb)
            # strictly ascending (kernel line-validity contract)
            t1 = torch.maximum(t1, t0 + 1e-4)
            t2 = torch.maximum(t2, t1 + 1e-4)
            new0 = torch.where(valid, t0, inf)
            new1 = torch.where(valid, t1, inf)
            new2 = torch.where(valid, t2, inf)
        s[:, 0] = new0
        s[:, 1] = new1
        s[:, 2] = new2
        s[:, 3:8] = 0.0
        rungs = self.seed_rungs[:num_rows]
        rungs[:, 0] = new0
        rungs[:, 1] = new1
        rungs[:, 2] = new2
        if self.cand_ctl is not None:
            nc = min(num_rows, self.cand_ctl.shape[0])
            self.cand_ctl[:nc].zero_()
            self.cand_cur[:nc].zero_()

    def indexer_emit_kwargs(self, emit_tier: str, num_rows: int, single_band: bool = False) -> dict:
        """kwargs for the FP4 paged-MQA scoring runner covering
        the planned emission tier (caller merges into its call)."""
        kw: dict = {}
        if emit_tier in ("counts", "list"):
            kw["seed_thr"] = self.seed_row[:num_rows]
        if emit_tier == "list":
            kw.update(
                accept_cap=LIST_SEG_A,
                cand_out=self.cand_vals[:num_rows],
                cand_idx_out=self.cand_idx[:num_rows],
                cand_ctl_out=self.cand_ctl[:num_rows],
                cand_cur_out=self.cand_cur[:num_rows],
            )
            if single_band:
                # every admitted entry goes into the C claim window only
                # (no per-band exact-claim atomics); requires three REAL
                # ascending lines (prescore), not parked ones
                kw["cand_single_band"] = True
                # the lines come from the prescore histogram, scanned by the
                # scorer itself; the row's first CTA publishes them into the
                # seed row for the consumer
                kw["ps_hist"] = self.ps_hist[:num_rows]
                kw["lines_top_k"] = self.top_k
        return kw

    def topk_ext_kwargs(
        self,
        route: TopkRoute,
        num_rows: int,
        block_max: Optional[torch.Tensor],
        single_band: bool = False,
        prev_out: Optional[torch.Tensor] = None,
    ) -> dict:
        """kwargs for trtllm::cute_dsl_gvr_topk_decode consuming this
        step's emission per the picked route."""
        kw: dict = {
            "xstate": self.xstate[:num_rows],
            "cluster_size": route.cluster_size,
        }
        if route.num_threads is not None:
            kw["num_threads"] = route.num_threads
        if route.tier in ("counts", "list"):
            kw["seed_thr"] = self.seed_row[:num_rows]
        elif route.tier == "rungs":
            # [rows, 3] seed selects the op's ext_rungs variant
            kw["seed_thr"] = self.seed_rungs[:num_rows]
        if route.tier == "list":
            # accept_cap must match the emitter's segment geometry: the
            # buffers are laid out at bases 0 / LIST_SEG_A / 2*LIST_SEG_A,
            # and the consumer derives the C capacity from the tensor
            # width minus 2*accept_cap.
            kw.update(
                cand_vals=self.cand_vals[:num_rows],
                cand_idx=self.cand_idx[:num_rows],
                cand_ctl=self.cand_ctl[:num_rows],
                accept_cap=LIST_SEG_A,
            )
            if single_band:
                kw["cand_single_band"] = True
                # the consumer resets the prescore histogram after the
                # emitter (its producer) has read it
                kw["ps_hist"] = self.ps_hist[:num_rows]
        if route.attach_block_max and block_max is not None:
            kw["block_max"] = block_max
        if single_band and route.cluster_size == 1:
            # the top-k epilogue mirrors its indices into the prior and
            # emits the neighbour flags the next step's prescore needs
            prev = self.prev_topk if prev_out is None else prev_out
            assert prev is not None, "emit_prev needs the caller-owned prior"
            kw["prev_out"] = prev[:num_rows]
            kw["prev_flags"] = self.prev_flags[:num_rows]
        return kw
