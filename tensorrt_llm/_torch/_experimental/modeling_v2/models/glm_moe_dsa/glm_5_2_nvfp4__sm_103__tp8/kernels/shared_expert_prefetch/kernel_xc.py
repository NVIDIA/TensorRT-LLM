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
"""Routed-MoE kernel of package shared_expert_prefetch: hop_routing's expert-cluster fork that
stages the shared expert's weights in L2.

``SharedPrefetchExpertClusterKernel`` is ``HopRoutedExpertClusterKernel`` (routed_moe's
expert-cluster kernel reading the pre-MoE hop's routing, ``hop_routing/routed_moe.py``) with one
addition. Before its ``griddepcontrol.wait``, each CTA issues ``cp.async.bulk.prefetch.L2`` for
its ``1 / grid`` slice of the layer's BF16 shared-expert weights
(``gate_up`` ``[512, 6144]`` and ``down`` ``[6144, 256]``, one TP8 rank shard,
9,437,184 bytes). The lanes of the TMA warp each prefetch one chunk.

The routed kernel is launched with programmatic dependent launch during the
pre-MoE hop and waits there while DRAM is nearly idle. The prefetch uses
that window, so the shared expert, which runs beside the routed kernel on
the auxiliary stream right after the hop, reads its weights from L2 instead
of competing with the routed weight stream for DRAM.

``run`` and ``kernel`` are routed_moe's (``routed_moe/kernel_xc.py``, including package
cross_layer_expert_prefetch's speculative expert prefetch) with the two shared-expert addresses
added to their parameters (after cross_layer_expert_prefetch's ``spec_ids_addr``) and the
shared-expert prefetch added before the wait. Every other line, every helper method and
``_route_tokens`` (hop_routing's) are unchanged, so the schedule, the arithmetic and
the outputs are those of hop_routing's fork.

With package cross_layer_expert_prefetch (``spec_k`` > 0), the TMA warp runs one pre-wait prologue:
each lane issues its shared-expert chunks first (the shared expert reads all
of them right after the hop), then its chunks of cross_layer_expert_prefetch's predicted experts in
rank order (``_xc_spec_prefetch``). When cross_layer_expert_prefetch restricts its issuers to the
first ``spec_ctas`` CTAs (the CTAs resident before the dependency release),
the shared-expert chunks are spread over the same CTAs (``se_ctas``);
otherwise over the whole grid, as without cross_layer_expert_prefetch. With ``spec_self``, CTA 0's
MMA warp stores the call's distinct experts, ranked by row count, into cross_layer_expert_prefetch's
row after routing (``_xc_spec_store``), as in routed_moe's kernel.

Memory order and progress:

* Before ``griddepcontrol.wait`` the addition reads only the shared-expert
  weights, through the L2 prefetch. They are model parameters that nothing
  writes while serving. It writes nothing and touches no flag, buffer or
  counter.
* Everything a producer owns (activation, routing, the FP32 accumulator) is
  read and written only after the wait, exactly as in hop_routing's fork.
* The prefetch is an async-proxy read of memory that the generic proxy never
  writes while serving, so it needs no fence. Whether the lines are in L2 or
  not, the shared expert's own loads (ordered by its own graph edges) read the
  same bytes.
* No wait, spin or cross-CTA dependency is added.
* cross_layer_expert_prefetch's prologue reads its row of predicted expert IDs with relaxed loads
  before the wait. The row is written only by cross_layer_expert_prefetch (on another stream) or by
  CTA 0 of the same layer's previous call, after that call's wait; any value
  read only selects which immutable weight bytes are prefetched (each ID is
  masked into [0, 256)), never what the kernel reads or computes (the memory
  order of ``hop_routing/routed_moe.py`` and ``routed_moe/routed_moe.py``).
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32

from ..hop_routing import routed_moe as hop_routed_mod
from ..routed_moe import ptx
from ..routed_moe.kernel_xc import (
    A_STAGE_BYTES,
    B2_BAR,
    BAR_EPI,
    CLUSTER,
    DEBUG_ROUTING,
    DEBUG_TIMELINE,
    DELAY_ENTRY,
    EPI_WARPS,
    FC1_KB,
    FC2_MT,
    HIDDEN,
    HSTAGE_BYTES,
    KB_PER_CTA,
    KB_TX_BYTES,
    MAX_ROUNDS,
    MAX_SLOTS,
    MMA_WARP,
    NTOK,
    NUM_BAR,
    NUM_EXPERTS,
    NUM_THREADS,
    SFA_STAGE_BYTES,
    SFB2_BYTES,
    SFB_BYTES,
    TL_BYTE_BASE,
    TL_SLOTS,
    TMA_WARP,
    TMEM_COLS,
    TOP_K,
    X_BAR,
    X_ROW_BYTES,
    XS_ROW_BYTES,
    XT_ENTRY,
    XT_EXIT,
    XT_GT0,
    XT_PDL,
    XT_ROUTED,
    _stamp,
)

SE_INTER = 256  # per-rank shared intermediate size (2048 / TP8)
SE_W1_BYTES = 2 * SE_INTER * HIDDEN * 2  # gate_up [512, 6144] BF16
SE_W2_BYTES = HIDDEN * SE_INTER * 2  # down [6144, 256] BF16
PF_LANES = 32
# Debug-timeline slots of the pre-wait prologue (routed_moe's ``XT_*`` slots end at
# 30; ``XT_GT0`` is 47): SM clock at its start and after its prefetches are
# issued, the global timer at its start, and the global timer when the
# dependency wait returns (the release, next to ``XT_PDL``).
XZ_PRO_CLK0 = 40
XZ_PRO_CLK1 = 41
XZ_PRO_GT0 = 42
XZ_PDL_GT = 43


def _chunk(total: int, ctas: int) -> int:
    """Bytes per prefetching lane (a multiple of 16) so that ``ctas x 32`` lanes cover ``total``."""
    per = -(-total // (ctas * PF_LANES))
    return -(-per // 16) * 16


class SharedPrefetchExpertClusterKernel(hop_routed_mod._kernel_class()):
    """hop_routing's expert-cluster kernel plus the pre-wait L2 prefetch of the shared expert's weights."""

    def __init__(self, num_ctas: int, *args, se_pf: bool = True, **kwargs):
        super().__init__(num_ctas, *args, **kwargs)
        self.se_pf = se_pf
        # CTAs that issue the shared-expert prefetch: every CTA, or, when
        # cross_layer_expert_prefetch's prefetch is compiled in for its first ``spec_ctas`` CTAs
        # (the CTAs resident before the dependency release), the same CTAs.
        self.se_ctas = num_ctas
        if self.spec_k > 0 and self.spec_ctas > 0:
            self.se_ctas = min(num_ctas, self.spec_ctas)
        self.se_chunk1 = _chunk(SE_W1_BYTES, self.se_ctas)
        self.se_chunk2 = _chunk(SE_W2_BYTES, self.se_ctas)

    @cute.jit
    def _se_prefetch(self, bidx, lane, se_w1_addr: Int64, se_w2_addr: Int64):
        """Lane ``lane`` of CTA ``bidx`` prefetches chunk ``bidx * 32 + lane`` of ``gate_up`` and of ``down``.

        The chunks are sized for the first ``se_ctas`` CTAs; CTAs past them own no chunk.
        """
        i = Int64(bidx * PF_LANES + lane)
        s1 = i * Int64(self.se_chunk1)
        if s1 < Int64(SE_W1_BYTES):
            n1 = cutlass.min(Int64(self.se_chunk1), Int64(SE_W1_BYTES) - s1)
            ptx.prefetch_l2_bulk(se_w1_addr + s1, Int32(n1))
        s2 = i * Int64(self.se_chunk2)
        if s2 < Int64(SE_W2_BYTES):
            n2 = cutlass.min(Int64(self.se_chunk2), Int64(SE_W2_BYTES) - s2)
            ptx.prefetch_l2_bulk(se_w2_addr + s2, Int32(n2))

    @cute.jit
    def run(
        self,
        w13_ptr: cute.Pointer,
        w13_sf_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w2_sf_ptr: cute.Pointer,
        x_addr: Int64,
        logits_addr: Int64,
        bias_addr: Int64,
        alpha1_addr: Int64,
        scale_c1_addr: Int64,
        alpha2_addr: Int64,
        xgs_addr: Int64,
        rows_addr: Int64,
        raw_scale_addr: Int64,
        raw_idx_addr: Int64,
        out_addr: Int64,
        counter_addr: Int64,
        dbg_addr: Int64,
        w13_addr: Int64,
        w13_sf_addr: Int64,
        w2_addr: Int64,
        w2_sf_addr: Int64,
        spec_ids_addr: Int64,
        se_w1_addr: Int64,
        se_w2_addr: Int64,
        num_tokens: Int32,
        stream,
    ):
        self._setup()
        w13 = cute.make_tensor(
            w13_ptr, cute.make_ordered_layout((2 * 256, HIDDEN, NUM_EXPERTS), order=(1, 0, 2))
        )
        w2 = cute.make_tensor(
            w2_ptr, cute.make_ordered_layout((HIDDEN, 256, NUM_EXPERTS), order=(1, 0, 2))
        )
        w13_sf = cute.make_tensor(
            w13_sf_ptr, blockscaled_utils.tile_atom_to_shape_SF(w13.shape, self.sf_vec_size)
        )
        w2_sf = cute.make_tensor(
            w2_sf_ptr, blockscaled_utils.tile_atom_to_shape_SF(w2.shape, self.sf_vec_size)
        )
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), self.tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_a1, tma_a1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w13, a_smem_layout, self.mma_tiler, self.tiled_mma, self.cluster_layout_vmnk.shape
        )
        tma_atom_sfa1, tma_sfa1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w13_sf,
            sfa_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_a2, tma_a2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, w2, a_smem_layout, self.mma_tiler, self.tiled_mma, self.cluster_layout_vmnk.shape
        )
        tma_atom_sfa2, tma_sfa2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            w2_sf,
            sfa_smem_layout,
            self.mma_tiler,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        @cute.struct
        class SharedStorage:
            ring_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.acc_stages]
            b_mbar: cute.struct.MemRange[cutlass.Int64, NUM_BAR]
            tmem_holding: cutlass.Int32
            tl_buf: cute.struct.MemRange[cutlass.Int64, TL_SLOTS]
            hstage: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, HSTAGE_BYTES // 8], 16]
            misc: cute.struct.MemRange[cutlass.Int32, 16]
            closed: cute.struct.MemRange[cutlass.Int32, MAX_ROUNDS * FC2_MT]
            bitmap: cute.struct.MemRange[cutlass.Uint32, 8]
            topk_id: cute.struct.MemRange[cutlass.Int32, NTOK * TOP_K]
            topk_w: cute.struct.MemRange[cutlass.Float32, NTOK * TOP_K]
            slot_expert: cute.struct.MemRange[cutlass.Int32, MAX_SLOTS]
            slot_mask: cute.struct.MemRange[cutlass.Uint32, MAX_SLOTS]
            slot_w: cute.struct.MemRange[cutlass.Float32, MAX_SLOTS * NTOK]
            slot_k: cute.struct.MemRange[cutlass.Int32, MAX_SLOTS * NTOK]
            expert_slot: cute.struct.MemRange[cutlass.Int32, NUM_EXPERTS]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float4E2M1FN, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                1024,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E4M3FN, cute.cosize(self.sfa_smem_layout_staged)
                ],
                1024,
            ]
            # this CTA's share of the activation rows (NTOK x 6 k-blocks)
            sX: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, NTOK * XS_ROW_BYTES], 1024]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float4E2M1FN, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                1024,
            ]
            sSFB: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, SFB_BYTES], 1024]
            sB2: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float4E2M1FN, cute.cosize(self.b2_smem_layout_staged.outer)
                ],
                1024,
            ]
            sSFB2: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, SFB2_BYTES], 1024]

        self.shared_storage = SharedStorage

        self.kernel(
            self.tiled_mma,
            tma_atom_a1,
            tma_a1,
            tma_atom_sfa1,
            tma_sfa1,
            tma_atom_a2,
            tma_a2,
            tma_atom_sfa2,
            tma_sfa2,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.sfb_smem_layout_std,
            self.b2_smem_layout_staged,
            self.sfb2_smem_layout_staged,
            x_addr,
            logits_addr,
            bias_addr,
            alpha1_addr,
            scale_c1_addr,
            alpha2_addr,
            xgs_addr,
            rows_addr,
            raw_scale_addr,
            raw_idx_addr,
            out_addr,
            counter_addr,
            dbg_addr,
            w13_addr,
            w13_sf_addr,
            w2_addr,
            w2_sf_addr,
            spec_ids_addr,
            se_w1_addr,
            se_w2_addr,
            num_tokens,
        ).launch(
            grid=[self.num_ctas, 1, 1],
            block=[NUM_THREADS, 1, 1],
            cluster=[CLUSTER, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_atom_sfa1: cute.CopyAtom,
        mSFA1: cute.Tensor,
        tma_atom_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_atom_sfa2: cute.CopyAtom,
        mSFA2: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfb_smem_layout_staged: cute.Layout,
        sfb_smem_layout_std: cute.Layout,
        b2_smem_layout_staged: cute.ComposedLayout,
        sfb2_smem_layout_staged: cute.Layout,
        x_addr: Int64,
        logits_addr: Int64,
        bias_addr: Int64,
        alpha1_addr: Int64,
        scale_c1_addr: Int64,
        alpha2_addr: Int64,
        xgs_addr: Int64,
        rows_addr: Int64,
        raw_scale_addr: Int64,
        raw_idx_addr: Int64,
        out_addr: Int64,
        counter_addr: Int64,
        dbg_addr: Int64,
        w13_addr: Int64,
        w13_sf_addr: Int64,
        w2_addr: Int64,
        w2_sf_addr: Int64,
        spec_ids_addr: Int64,
        se_w1_addr: Int64,
        se_w2_addr: Int64,
        num_tokens: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        bidx, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        bidx = Int32(bidx)
        rk = ptx.cluster_ctarank()
        cl = bidx // CLUSTER
        ncl = Int32(grid) // CLUSTER

        # shared_expert_prefetch (+ cross_layer_expert_prefetch): one pre-wait prologue in the TMA
        # warp. Its lanes first start the L2 prefetch of this CTA's slice of the layer's
        # shared-expert weights, then (cross_layer_expert_prefetch) of their chunks of the
        # predicted routed experts in rank order; both are immutable model
        # parameters (see the module docstring). With cross_layer_expert_prefetch's issuing CTAs
        # set, both lists are spread over those CTAs only.
        # Debug timeline only: when the prologue starts (SM clock and global
        # timer) and when its prefetches are issued (SM clock), recorded by the
        # TMA warp's lane 0 and stored into slots XZ_PRO_* after the shared
        # memory set-up below.
        pro_on = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        if cutlass.const_expr(pro_on):
            pro_c0 = Int64(0)
            pro_g0 = Int64(0)
            pro_c1 = Int64(0)
        if warp_idx == TMA_WARP:
            if cutlass.const_expr(pro_on):
                pro_c0 = ptx.clock64()
                pro_g0 = ptx.globaltimer_u64()
            if cutlass.const_expr(self.se_pf):
                self._se_prefetch(bidx, lane, se_w1_addr, se_w2_addr)
            if cutlass.const_expr(self.spec_k > 0):
                self._xc_spec_prefetch(
                    bidx,
                    Int32(grid),
                    lane,
                    spec_ids_addr,
                    w13_addr,
                    w13_sf_addr,
                    w2_addr,
                    w2_sf_addr,
                )
            if cutlass.const_expr(pro_on):
                pro_c1 = ptx.clock64()

        if warp_idx == TMA_WARP:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa2)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        ring = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ring_mbar.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=A_STAGE_BYTES + SFA_STAGE_BYTES,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, EPI_WARPS),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        b_mbar = storage.b_mbar.data_ptr()
        if warp_idx == MMA_WARP:
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(FC1_KB):
                    cute.arch.mbarrier_init(b_mbar + kb, 1)
                cute.arch.mbarrier_init(b_mbar + X_BAR, 1)
                for rho in cutlass.range_constexpr(MAX_ROUNDS):
                    cute.arch.mbarrier_init(b_mbar + B2_BAR + rho, CLUSTER)
                # k-blocks quantized by a peer complete on its bulk copies
                for kb in cutlass.range_constexpr(FC1_KB):
                    if Int32(kb // KB_PER_CTA) != rk:
                        ptx.mbar_arrive_expect_tx(Int32((b_mbar + kb).toint()), Int32(KB_TX_BYTES))
        cute.arch.mbarrier_init_fence()
        # Peers arrive on this CTA's operand barriers: their initialization
        # must be visible cluster-wide first (waited before the first remote
        # arrive, below).
        cute.arch.cluster_arrive_relaxed()

        if warp_idx == 0:
            cute.arch.alloc_tmem(TMEM_COLS, storage.tmem_holding)

        s_misc = storage.misc.get_tensor(cute.make_layout(16))
        s_bitmap = storage.bitmap.get_tensor(cute.make_layout(8))
        s_topk_id = storage.topk_id.get_tensor(cute.make_layout(NTOK * TOP_K))
        s_topk_w = storage.topk_w.get_tensor(cute.make_layout(NTOK * TOP_K))
        s_slot_expert = storage.slot_expert.get_tensor(cute.make_layout(MAX_SLOTS))
        s_slot_mask = storage.slot_mask.get_tensor(cute.make_layout(MAX_SLOTS))
        s_slot_w = storage.slot_w.get_tensor(cute.make_layout(MAX_SLOTS * NTOK))
        s_slot_k = storage.slot_k.get_tensor(cute.make_layout(MAX_SLOTS * NTOK))
        s_expert_slot = storage.expert_slot.get_tensor(cute.make_layout(NUM_EXPERTS))

        s_expert_slot[tidx] = Int32(-1)
        s_slot_w[tidx] = Float32(0.0)
        s_slot_w[tidx + NUM_THREADS] = Float32(0.0)
        if tidx < MAX_SLOTS:
            s_slot_mask[tidx] = Uint32(0)
        if tidx < 16:
            s_misc[tidx] = Int32(0)
        if cutlass.const_expr(self.debug & DEBUG_TIMELINE):
            if tidx < TL_SLOTS:
                ptx.sts_u64(Int32(storage.tl_buf.data_ptr().toint()) + tidx * 8, Int64(0))
        if tidx < 8:
            s_bitmap[tidx] = Uint32(0)
        bias_r = cute.make_rmem_tensor((8,), Float32)
        for j in cutlass.range_constexpr(8):
            bias_r[j] = ptx.bf16x2_to_f32(ptx.ld_nc_u16(bias_addr + (lane + 32 * j) * 2), False)
        cute.arch.sync_threads()

        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        tl_s = Int32(storage.tl_buf.data_ptr().toint())
        if tidx == 0:
            _stamp(tl, tl_s, XT_ENTRY)
            if cutlass.const_expr(tl):
                ptx.sts_u64(tl_s + XT_GT0 * 8, ptx.globaltimer_u64())
        if cutlass.const_expr(pro_on):
            if warp_idx == TMA_WARP:
                if lane == 0:
                    ptx.sts_u64(tl_s + XZ_PRO_CLK0 * 8, pro_c0)
                    ptx.sts_u64(tl_s + XZ_PRO_GT0 * 8, pro_g0)
                    ptx.sts_u64(tl_s + XZ_PRO_CLK1 * 8, pro_c1)
        cute.arch.griddepcontrol_wait()
        if tidx == 0:
            _stamp(tl, tl_s, XT_PDL)
            if cutlass.const_expr(pro_on):
                ptx.sts_u64(tl_s + XZ_PDL_GT * 8, ptx.globaltimer_u64())
        if cutlass.const_expr(self.delay & DELAY_ENTRY):
            self._xc_delay(DELAY_ENTRY * 53, bidx)
        if warp_idx == TMA_WARP:
            with cute.arch.elect_one():
                xbar = Int32((b_mbar + X_BAR).toint())
                sx = Int32(storage.sX.data_ptr().toint())
                ptx.mbar_arrive_expect_tx(xbar, num_tokens * XS_ROW_BYTES)
                r = Int32(0)
                while r < num_tokens:
                    ptx.bulk_g2s(
                        sx + r * XS_ROW_BYTES,
                        x_addr + r * X_ROW_BYTES + rk * XS_ROW_BYTES,
                        Int32(XS_ROW_BYTES),
                        xbar,
                    )
                    r = r + 1
        # Every CTA of the cluster has initialized its barriers: remote
        # copies and arrivals may start.
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.trigger == "early"):
            cute.arch.griddepcontrol_launch_dependents()

        # -------- routing (warp t handles token t) -> slot tables --------
        self._route_tokens(
            warp_idx, lane, num_tokens, logits_addr, bias_r, s_topk_id, s_topk_w, storage
        )
        cute.arch.sync_threads()
        e = tidx
        bw = s_bitmap[e // 32]
        prefix = Int32(0)
        for w in cutlass.range_constexpr(8):
            word = s_bitmap[w]
            if w < e // 32:
                prefix = prefix + ptx.popc_u32(word)
        below = bw & ((Uint32(1) << Uint32(e % 32)) - Uint32(1))
        prefix = prefix + ptx.popc_u32(below)
        if ((bw >> Uint32(e % 32)) & Uint32(1)) != Uint32(0):
            s_slot_expert[prefix] = e
            s_expert_slot[e] = prefix
        if tidx == 0:
            total = Int32(0)
            for w in cutlass.range_constexpr(8):
                total = total + ptx.popc_u32(s_bitmap[w])
            s_misc[0] = total
        cute.arch.sync_threads()
        if tidx < NTOK * TOP_K:
            t = tidx // TOP_K
            k = tidx % TOP_K
            if t < num_tokens:
                e = s_topk_id[tidx]
                sl = s_expert_slot[e]
                s_slot_w[sl * NTOK + t] = s_topk_w[tidx]
                s_slot_k[sl * NTOK + t] = k
                ptx.atom_or_shared_u32(
                    Int32(storage.slot_mask.data_ptr().toint()) + sl * 4, Uint32(1) << Uint32(t)
                )
        cute.arch.sync_threads()
        num_slots = s_misc[0]
        if tidx == 0:
            _stamp(tl, tl_s, XT_ROUTED)

        # -------- expert rounds of this cluster --------
        # Round 0 spreads min(D, clusters) experts over all clusters (an
        # expert served by r clusters splits its FC2 tiles r ways); when
        # D > clusters, round 1 spreads the remaining D - clusters experts
        # over all clusters the same way. Every cluster then streams two FC1
        # tiles per CTA and an even share of the FC2 tiles.
        nact = ncl
        if cutlass.const_expr(self.min_clusters > 0):
            if num_slots <= ncl:
                nact = cutlass.max(num_slots, Int32(self.min_clusters))
                nact = cutlass.min(nact, ncl)
        if cutlass.const_expr(self.two_round_clusters > 0):
            if num_slots > ncl:
                if num_slots <= Int32(MAX_ROUNDS * self.two_round_clusters):
                    nact = cutlass.min(Int32(self.two_round_clusters), ncl)
        rounds = (num_slots + nact - 1) // nact
        if cl >= nact:
            rounds = Int32(0)
        slots = cute.make_rmem_tensor((MAX_ROUNDS,), Int32)
        shares = cute.make_rmem_tensor((MAX_ROUNDS,), Int32)
        firsts = cute.make_rmem_tensor((MAX_ROUNDS,), Int32)
        t_lo = cute.make_rmem_tensor((MAX_ROUNDS,), Int32)
        t_hi = cute.make_rmem_tensor((MAX_ROUNDS,), Int32)
        for rho in cutlass.range_constexpr(MAX_ROUNDS):
            base = Int32(0)
            count = cutlass.min(num_slots, nact)
            if cutlass.const_expr(rho == 1):
                base = nact
                count = cutlass.max(num_slots - nact, Int32(1))
            i = cl % count
            j = cl // count
            nshare = (nact - 1 - i) // count + 1
            a = (j * FC2_MT) // nshare
            n = ((j + 1) * FC2_MT) // nshare - a
            slots[rho] = base + i
            shares[rho] = nshare
            firsts[rho] = Int32(0)
            if j == 0:
                firsts[rho] = Int32(1)
            t_lo[rho] = a + (rk * n) // CLUSTER
            t_hi[rho] = a + ((rk + 1) * n) // CLUSTER

        if warp_idx == TMA_WARP:
            self._xc_tma_producer(
                tma_atom_a1,
                mA1,
                tma_atom_sfa1,
                mSFA1,
                tma_atom_a2,
                mA2,
                tma_atom_sfa2,
                mSFA2,
                a_smem_layout_staged,
                sfa_smem_layout_staged,
                tiled_mma,
                cluster_layout_vmnk,
                storage,
                ring,
                b_mbar,
                s_slot_expert,
                rk,
                rounds,
                slots,
                shares,
                firsts,
                t_lo,
                t_hi,
                w13_addr,
                w13_sf_addr,
                w2_addr,
                w2_sf_addr,
            )
        if cutlass.const_expr(self.spec_self):
            if bidx == 0:
                if warp_idx == MMA_WARP:
                    self._xc_spec_store(lane, num_slots, spec_ids_addr, storage, s_slot_expert)
        if warp_idx == MMA_WARP:
            self._xc_mma_issuer(
                tiled_mma,
                storage,
                ring,
                acc_pipe,
                b_mbar,
                a_smem_layout_staged,
                sfa_smem_layout_staged,
                b_smem_layout_staged,
                sfb_smem_layout_staged,
                sfb_smem_layout_std,
                b2_smem_layout_staged,
                sfb2_smem_layout_staged,
                rounds,
                t_lo,
                t_hi,
            )
        if (warp_idx != TMA_WARP) & (warp_idx != MMA_WARP):
            if rounds > 0:
                self._xc_quantize_x(
                    warp_idx, lane, xgs_addr, num_tokens, storage, b_mbar, dbg_addr, rk
                )
            else:
                # an idle cluster still lets its activation copy land
                cute.arch.mbarrier_wait(b_mbar + X_BAR, 0)

        if bidx == 0:
            if tidx < NTOK * TOP_K:
                t = tidx // TOP_K
                if t < num_tokens:
                    ptx.st_u16(raw_scale_addr + tidx * 2, ptx.bf16_bits(s_topk_w[tidx]))
                    ptx.st_u32(raw_idx_addr + tidx * 4, Uint32(tidx))
        if cutlass.const_expr(self.debug & DEBUG_ROUTING):
            if bidx == 0:
                if tidx < NTOK * TOP_K:
                    ptx.st_u32(dbg_addr + tidx * 4, Uint32(s_topk_id[tidx]))
                    ptx.st_f32(dbg_addr + 256 + tidx * 4, s_topk_w[tidx])
                if tidx < MAX_SLOTS:
                    ptx.st_u32(dbg_addr + 512 + tidx * 4, Uint32(s_slot_expert[tidx]))
                    ptx.st_u32(dbg_addr + 768 + tidx * 4, s_slot_mask[tidx])
                if tidx == 0:
                    ptx.st_u32(dbg_addr + 1024, Uint32(num_slots))

        if warp_idx < EPI_WARPS:
            self._xc_epilogue(
                tiled_mma,
                tidx,
                warp_idx,
                lane,
                storage,
                acc_pipe,
                b_mbar,
                s_slot_expert,
                s_slot_mask,
                s_slot_k,
                rk,
                rounds,
                slots,
                firsts,
                t_lo,
                t_hi,
                num_tokens,
                alpha1_addr,
                scale_c1_addr,
                alpha2_addr,
                rows_addr,
                out_addr,
                counter_addr,
                s_topk_w,
                num_slots,
                bidx,
                dbg_addr,
            )
        # No CTA leaves while a peer may still store into its shared memory.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp_idx < EPI_WARPS:
            if cutlass.const_expr(self.debug & DEBUG_TIMELINE):
                ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
                if tidx == 0:
                    _stamp(tl, tl_s, XT_EXIT)
                ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
                if tidx < TL_SLOTS:
                    ptx.st_u64(
                        dbg_addr + TL_BYTE_BASE + (bidx * TL_SLOTS + tidx) * 8,
                        ptx.lds_u64(tl_s + tidx * 8),
                    )
            if warp_idx == 0:
                tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding
                )
                cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)
