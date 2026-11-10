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
"""Expert-cluster schedule of the routed_moe routed NVFP4 MoE (CuTe DSL, SM100 family).

The grid is a set of 4-CTA thread block clusters. After every CTA has
recomputed the routing, each cluster owns one expert per round (one round when
the number of distinct experts D is at most the number of clusters, two
otherwise): cluster ``c`` of round ``rho`` serves expert slot ``v % D`` with
``v = rho * clusters + c``, so an expert is served by ``r`` clusters when there
are more clusters than experts, and each of those clusters computes the whole
FC1 of the expert (the repeated weight reads are L2 hits) and its share of the
48 FC2 output tiles.

Inside a cluster, CTA ``k`` stages and NVFP4-quantizes activation k-blocks
``[6k, 6k + 6)`` of every row and bulk-copies each finished k-block into the
three peers (distributed shared memory, completing on their k-block
mbarriers), so every CTA holds the whole FC1 B operand after a quarter of the
quantization work. CTA ``k`` then computes FC1 tile ``k`` (128 of the 512
stock gate/up rows, all 24 k-blocks, so no split-K) through an 8-stage TMA
ring, applies SwiGLU and the NVFP4 quantization of the intermediate, and
stores its 64 intermediate columns of every token straight into the FC2
operand stage of all four CTAs, then release-arrives on each CTA's operand
mbarrier. Every CTA then runs its FC2 tiles from that operand. Nothing crosses
a cluster: there are no global flags, split partials or epochs, and a cluster
that starts late (for example behind a co-running kernel) delays only its own
expert. The host sizes the grid so that every cluster is co-resident and
SMs remain for the concurrently running shared expert (on sm_103 32 clusters,
26 for the raw tuple consumed by the post-MoE hop when the router logits come
from the pre-MoE hop; see ``routed_moe.RoutedMoe._xc_ctas``); with
``min_clusters`` set, calls with
few distinct experts use only ``clamp(D, min_clusters, clusters)`` clusters
and the rest exit after routing.

The kernel writes the native unfinalized tuple (FC2 rows ``t * 8 + k``,
routing weights, identity permutation). The finalized output is produced by
``RoutedMoeFinalizeKernel``, launched right after with programmatic dependent
launch: the native finalize arithmetic (FP32 FMA over the top-k in order, one
BF16 rounding).
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32

from . import ptx
from .kernel import (
    A_STAGE_BYTES,
    BAR_EPI,
    BAR_QUANT,
    DEBUG_FC1,
    DEBUG_QUANT,
    DEBUG_ROUTING,
    DEBUG_TIMELINE,
    DELAY_ENTRY,
    DELAY_PRODUCER,
    EPI_WARPS,
    FC1_KB,
    FC1_MT,
    FC2_MT,
    HIDDEN,
    HSTAGE_BYTES,
    HSTAGE_SF,
    L2_EVICT_FIRST,
    MAX_SLOTS,
    MMA_WARP,
    NTOK,
    NUM_EXPERTS,
    NUM_THREADS,
    QUANT_WARPS,
    SFA_STAGE_BYTES,
    SFB_BYTES,
    SFB_CHUNK_STRIDE,
    SFB_STAGE_STRIDE,
    TL_BYTE_BASE,
    TL_SLOTS,
    TMA_WARP,
    TMEM_COLS,
    TMEM_SFA_COL,
    TMEM_SFB_COL,
    TOP_K,
    W2_EXPERT_BYTES,
    W2_SF_EXPERT_BYTES,
    W2_SF_TILE_BYTES,
    W2_TILE_BYTES,
    W13_EXPERT_BYTES,
    W13_ROW_BYTES,
    W13_SF_EXPERT_BYTES,
    W13_SF_TILE_BYTES,
    X_ROW_BYTES,
    RoutedMoeKernel,
    _stamp,
)

DELAY_FINALIZE = 16  # RoutedMoeFinalizeKernel CTAs, before their output stores
# L2 prefetch (``l2pf`` bits; 0 = off): the TMA warp issues
# ``cp.async.bulk.prefetch.L2`` for weight bytes its ring reaches later, so
# that those ring loads hit L2 instead of waiting for DRAM.
L2PF_FC1_R1 = 1  # round-1 FC1 tile (by the first cluster of the round-1 slot)
L2PF_FC2 = 2  # this CTA's FC2 tiles of both rounds
L2PF_EARLY = 4  # issue them before the first ring load (else after the first ring fill)
FC1_TILE_BYTES = 128 * W13_ROW_BYTES
PF_FC1_CHUNK = FC1_TILE_BYTES // 32  # one bulk prefetch per lane
CLUSTER = 4  # CTAs per cluster = FC1 tiles per expert
# Speculative prefetch of predicted experts (``spec_k`` > 0, package cross_layer_expert_prefetch):
# 16 KB chunks of one expert's FC1 weight, FC1 scales, FC2 weight and FC2
# scales, in that order, for ``SPEC_KMAX``-wide rows of ranked expert IDs.
SPEC_CHUNK = 16384
SPEC_C13 = W13_EXPERT_BYTES // SPEC_CHUNK
SPEC_C13SF = W13_SF_EXPERT_BYTES // SPEC_CHUNK
SPEC_C2 = W2_EXPERT_BYTES // SPEC_CHUNK
SPEC_C2SF = W2_SF_EXPERT_BYTES // SPEC_CHUNK
SPEC_CPE = SPEC_C13 + SPEC_C13SF + SPEC_C2 + SPEC_C2SF
SPEC_KMAX = 32
assert SPEC_CPE * SPEC_CHUNK == (
    W13_EXPERT_BYTES + W13_SF_EXPERT_BYTES + W2_EXPERT_BYTES + W2_SF_EXPERT_BYTES
)
MAX_ROUNDS = 2  # expert rounds per cluster (D <= 2 * clusters)
L2_EVICT_NORMAL = 0x1000000000000000
HIMG_B_BYTES = NTOK * 128  # one FC2 B operand stage (8 rows x 128 B FP4)
SFB2_BYTES = (MAX_ROUNDS - 1) * SFB_STAGE_STRIDE + 3 * SFB_CHUNK_STRIDE + 512

# mbarriers: one per quantized k-block, the staged activation rows, one FC2
# operand barrier per round (4 arrivals: one per cluster CTA)
X_BAR = FC1_KB
B2_BAR = X_BAR + 1
NUM_BAR = B2_BAR + MAX_ROUNDS
# Activation quantization is split over the cluster: CTA k stages and
# quantizes k-blocks [6k, 6k + 6) of every row and bulk-copies each finished
# k-block (1 KB B stage + 512 B compact SFB stage) into the three peers.
KB_PER_CTA = FC1_KB // CLUSTER
XS_ROW_BYTES = KB_PER_CTA * 512  # staged columns of one activation row
KB_TX_BYTES = 1024 + SFB_STAGE_STRIDE  # one k-block's B + SFB stage

# Timeline slots (debug builds only)
(
    XT_ENTRY,
    XT_PDL,
    XT_ROUTED,
    XT_Q_DONE,
    XT_TMA_FC1,
    XT_MMA_FC1,
    XT_EPI_FC1_ACC,
    XT_EPI_FC1_PUB,
    XT_B2_READY,
    XT_TMA_FC2,
    XT_FC2_FIRST_ACC,
    XT_MMA_FC2,
    XT_FC2_DONE,
    XT_EXIT,
    XT_MMA_FC1_R0,
    XT_EPI_FC1_PUB_R1,
    XT_B2_READY_R1,
    XT_FC2_R0_DONE,
    XT_TMA_FC1_R0,
    XT_PF_ISSUED,
    XT_EPI_FC1_ACC_R1,
    XT_EPI_FC1_SWIGLU,
    XT_EPI_FC1_SWIGLU_R1,
    XT_EPI_FC1_TMEM,  # 23, 24: accumulator loaded from TMEM (rounds 0, 1)
    XT_EPI_FC1_TMEM_R1,
    XT_EPI_FC1_SWQ,  # 25, 26: SwiGLU + quantization done, before the barrier
    XT_EPI_FC1_SWQ_R1,
    XT_EPI_FC1_SWQ_W_R1,  # 27..30: the same, round 1, epilogue warp w (lane 0)
) = range(28)
# slots 28..30 are XT_EPI_FC1_SWQ_W_R1 + 1..3
XT_GT0 = TL_SLOTS - 1


class RoutedMoeExpertClusterKernel(RoutedMoeKernel):
    """Expert-cluster kernel; raw output only (see module docstring)."""

    def __init__(
        self,
        num_ctas: int,
        num_stages: int = 8,
        debug: int = 0,
        delay: int = 0,
        delay_ns: int = 2000,
        finalize: bool = False,
        trigger: str = "late",
        min_clusters: int = 0,
        red_out: bool = False,
        l2pf: int = 0,
        two_round_clusters: int = 0,
        acc_stages: int = 4,
        spec_k: int = 0,
        spec_ctas: int = 0,
        spec_self: bool = False,
    ):
        assert num_ctas % CLUSTER == 0
        assert 0 <= spec_k <= SPEC_KMAX
        # Speculative L2 prefetch (cross_layer_expert_prefetch): before its dependency wait, the TMA
        # warp of each of the first ``spec_ctas`` CTAs (0 = all) issues
        # prefetches of the first ``spec_k`` experts listed at
        # ``spec_ids_addr`` (int32, ``SPEC_KMAX`` wide); 0 compiles it out.
        # ``spec_self``: after routing, CTA 0 also writes this call's distinct
        # experts there, ranked by row count, for the next call of the layer.
        self.spec_k = spec_k
        self.spec_ctas = spec_ctas
        self.spec_self = spec_self
        # 0: a two-round call (D > clusters) uses every cluster. Otherwise it
        # uses this many (the rest exit after routing and leave their SMs to
        # the shared expert), as long as two rounds of them cover D.
        self.two_round_clusters = two_round_clusters
        # L2 prefetch bits (L2PF_*); 0 compiles the prefetch out
        self.l2pf = l2pf
        # routed_moe_into_hop "fp32" entry: instead of the raw rows, every FC2 output adds
        # its weighted contribution w[t, k] * bf16(y) (exact in FP32) to an
        # FP32 [R, H] accumulator; the post-MoE hop rounds and clears it.
        self.red_out = red_out
        # programmatic-dependent-launch trigger: "late" after each CTA's last
        # FC2 tile, "early" right after the dependency wait (the successor's
        # CTAs may then wait on the free SMs), "none" (the successor launches
        # when the grid completes)
        assert trigger in ("late", "early", "none")
        self.trigger = trigger
        # 0: every cluster works. Otherwise, when D <= clusters, only
        # clamp(D, min_clusters, clusters) clusters work and the others exit
        # right after routing, leaving their SMs to concurrent kernels (the
        # shared expert on the aux stream).
        self.min_clusters = min_clusters
        # Two rounds cover at most 2 x clusters distinct experts; the host
        # launches this kernel only for calls with rows x 8 within that bound
        # (``RoutedMoe.schedule``).
        super().__init__(
            num_ctas,
            finalize=finalize,
            num_stages=num_stages,
            debug=debug,
            delay=delay,
            delay_ns=delay_ns,
            acc_stages=acc_stages,
        )

    def _setup(self):
        super()._setup()
        self.b2_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, cutlass.Float4E2M1FN, MAX_ROUNDS
        )
        sfb2_std = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler, self.sf_vec_size, MAX_ROUNDS
        )
        self.sfb2_smem_layout_staged = cute.make_layout(
            sfb2_std.shape,
            stride=((((16, 4), 0), (0, 1)), 0, SFB_CHUNK_STRIDE, SFB_STAGE_STRIDE),
        )

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

        if warp_idx == TMA_WARP:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_sfa2)
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
        cute.arch.griddepcontrol_wait()
        if tidx == 0:
            _stamp(tl, tl_s, XT_PDL)
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

    # ------------------------------------------------------------------
    @cute.jit
    def _xc_tma_producer(
        self,
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
    ):
        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        tl_s = Int32(storage.tl_buf.data_ptr().toint())
        lane0 = cute.arch.lane_idx() == 0
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        thr_mma = tiled_mma.get_slice(0)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        gA1 = cute.local_tile(mA1, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFA1 = cute.local_tile(
            mSFA1, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gA2 = cute.local_tile(mA2, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFA2 = cute.local_tile(
            mSFA2, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        tAsA, tAgA1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a1,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA1), 0, 3),
        )
        tAsSFA, tAgSFA1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa1,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gSFA1), 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA1 = cute.filter_zeros(tAgSFA1)
        _, tAgA2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a2,
            0,
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA2), 0, 3),
        )
        _, tAgSFA2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa2,
            0,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gSFA2), 0, 3),
        )
        tAgSFA2 = cute.filter_zeros(tAgSFA2)

        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
        if cutlass.const_expr(self.l2pf & L2PF_EARLY):
            self._xc_l2_prefetch(
                tl,
                tl_s,
                rk,
                rounds,
                slots,
                firsts,
                t_lo,
                t_hi,
                s_slot_expert,
                w13_addr,
                w13_sf_addr,
                w2_addr,
                w2_sf_addr,
            )
        # FC1: tile rk of each round's expert, all k-blocks. A weight read by
        # several clusters stays in L2 for the others (evict-normal).
        for rho in cutlass.range_constexpr(MAX_ROUNDS):
            if rho < rounds:
                expert = s_slot_expert[slots[rho]]
                policy = Int64(L2_EVICT_FIRST)
                if shares[rho] > 1:
                    policy = Int64(L2_EVICT_NORMAL)
                kb = Int32(0)
                while kb < FC1_KB:
                    if cutlass.const_expr(
                        rho == 0 and self.l2pf != 0 and not (self.l2pf & L2PF_EARLY)
                    ):
                        # after the first ring fill of round 0
                        if kb == Int32(self.num_stages):
                            self._xc_l2_prefetch(
                                tl,
                                tl_s,
                                rk,
                                rounds,
                                slots,
                                firsts,
                                t_lo,
                                t_hi,
                                s_slot_expert,
                                w13_addr,
                                w13_sf_addr,
                                w2_addr,
                                w2_sf_addr,
                            )
                    ring.producer_acquire(state)
                    bar = ring.producer_get_barrier(state)
                    cute.copy(
                        tma_atom_a1,
                        tAgA1[(None, rk, kb, expert)],
                        tAsA[(None, state.index)],
                        tma_bar_ptr=bar,
                        cache_policy=policy,
                    )
                    cute.copy(
                        tma_atom_sfa1,
                        tAgSFA1[(None, rk, kb, expert)],
                        tAsSFA[(None, state.index)],
                        tma_bar_ptr=bar,
                        cache_policy=policy,
                    )
                    state.advance()
                    kb = kb + 1
                if cutlass.const_expr(rho == 0):
                    if lane0:
                        _stamp(tl, tl_s, XT_TMA_FC1_R0)
        if lane0:
            _stamp(tl, tl_s, XT_TMA_FC1)
        for rho in cutlass.range_constexpr(MAX_ROUNDS):
            if rho < rounds:
                expert = s_slot_expert[slots[rho]]
                t = t_lo[rho]
                while t < t_hi[rho]:
                    ring.producer_acquire(state)
                    bar = ring.producer_get_barrier(state)
                    cute.copy(
                        tma_atom_a2,
                        tAgA2[(None, t, 0, expert)],
                        tAsA[(None, state.index)],
                        tma_bar_ptr=bar,
                        cache_policy=Int64(L2_EVICT_FIRST),
                    )
                    cute.copy(
                        tma_atom_sfa2,
                        tAgSFA2[(None, t, 0, expert)],
                        tAsSFA[(None, state.index)],
                        tma_bar_ptr=bar,
                        cache_policy=Int64(L2_EVICT_FIRST),
                    )
                    state.advance()
                    t = t + 1
        if lane0:
            _stamp(tl, tl_s, XT_TMA_FC2)
        ring.producer_tail(state)

    # ------------------------------------------------------------------
    @cute.jit
    def _xc_l2_prefetch(
        self,
        tl,
        tl_s,
        rk,
        rounds,
        slots,
        firsts,
        t_lo,
        t_hi,
        s_slot_expert,
        w13_addr,
        w13_sf_addr,
        w2_addr,
        w2_sf_addr,
    ):
        """Bulk L2 prefetch of weight bytes this CTA's ring reaches later.

        ``L2PF_FC1_R1``: FC1 tile ``rk`` of the round-1 expert, by the first
        cluster serving that slot only (the others read it from L2 anyway);
        one 12 KB chunk per lane, lane 0 also the tile's scales.
        ``L2PF_FC2``: this CTA's FC2 tiles of both rounds, one tile (16 KB +
        2 KB scales) per lane. Prefetches change no data, only where the
        later TMA loads find it.
        """
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(self.l2pf & L2PF_FC1_R1):
            if rounds > 1:
                if firsts[1] != 0:
                    e1 = Int64(s_slot_expert[slots[1]])
                    base = w13_addr + e1 * W13_EXPERT_BYTES + Int64(rk) * Int64(FC1_TILE_BYTES)
                    ptx.prefetch_l2_bulk(
                        base + Int64(lane) * Int64(PF_FC1_CHUNK), Int32(PF_FC1_CHUNK)
                    )
                    if lane == 0:
                        ptx.prefetch_l2_bulk(
                            w13_sf_addr
                            + e1 * W13_SF_EXPERT_BYTES
                            + Int64(rk) * Int64(W13_SF_TILE_BYTES),
                            Int32(W13_SF_TILE_BYTES),
                        )
        if cutlass.const_expr(self.l2pf & L2PF_FC2):
            for rho in cutlass.range_constexpr(MAX_ROUNDS):
                if rho < rounds:
                    e2 = Int64(s_slot_expert[slots[rho]])
                    t = t_lo[rho] + lane
                    if t < t_hi[rho]:
                        ptx.prefetch_l2_bulk(
                            w2_addr + e2 * W2_EXPERT_BYTES + Int64(t) * Int64(W2_TILE_BYTES),
                            Int32(W2_TILE_BYTES),
                        )
                        ptx.prefetch_l2_bulk(
                            w2_sf_addr
                            + e2 * W2_SF_EXPERT_BYTES
                            + Int64(t) * Int64(W2_SF_TILE_BYTES),
                            Int32(W2_SF_TILE_BYTES),
                        )
        if lane == 0:
            _stamp(tl, tl_s, XT_PF_ISSUED)

    # ------------------------------------------------------------------
    @cute.jit
    def _xc_spec_prefetch(
        self, bidx, grid, lane, ids_addr, w13_addr, w13_sf_addr, w2_addr, w2_sf_addr
    ):
        """Bulk L2 prefetch of the first ``spec_k`` experts at ``ids_addr``.

        Runs before the dependency wait: the IDs are written on another stream
        (cross_layer_expert_prefetch) and only choose which weight bytes are prefetched, so
        a stale or concurrently updated ID costs bandwidth, never correctness
        (each ID is masked into [0, 256)). Chunk ``c`` of the ``spec_k`` x
        ``SPEC_CPE`` chunks (expert-rank major) is issued by lane ``c %
        (32 * n)`` of the first ``n`` CTAs, so the best-ranked experts go
        first.
        """
        nsp = grid
        if cutlass.const_expr(self.spec_ctas > 0):
            nsp = cutlass.min(grid, Int32(self.spec_ctas))
        if bidx < nsp:
            c = bidx * 32 + lane
            while c < Int32(self.spec_k * SPEC_CPE):
                slot = c // SPEC_CPE
                off = c - slot * SPEC_CPE
                ex = Int64(
                    Int32(
                        ptx.ld_relaxed_gpu_u32(ids_addr + Int64(slot) * 4) & Uint32(NUM_EXPERTS - 1)
                    )
                )
                o = Int64(off) * SPEC_CHUNK
                addr = w13_addr + ex * W13_EXPERT_BYTES + o
                if off >= SPEC_C13:
                    addr = w13_sf_addr + ex * W13_SF_EXPERT_BYTES + o - SPEC_C13 * SPEC_CHUNK
                if off >= SPEC_C13 + SPEC_C13SF:
                    addr = w2_addr + ex * W2_EXPERT_BYTES + o - (SPEC_C13 + SPEC_C13SF) * SPEC_CHUNK
                if off >= SPEC_C13 + SPEC_C13SF + SPEC_C2:
                    addr = (
                        w2_sf_addr
                        + ex * W2_SF_EXPERT_BYTES
                        + o
                        - (SPEC_C13 + SPEC_C13SF + SPEC_C2) * SPEC_CHUNK
                    )
                ptx.prefetch_l2_bulk(addr, Int32(SPEC_CHUNK))
                c = c + nsp * 32

    # ------------------------------------------------------------------
    @cute.jit
    def _xc_spec_store(self, lane, num_slots, ids_addr, storage, s_slot_expert):
        """Write this call's distinct experts to ``ids_addr``, ranked by the
        number of rows that select them (ties: lower expert ID first); slots
        past ``num_slots`` keep their previous IDs."""
        mask_s = Int32(storage.slot_mask.data_ptr().toint())
        for h in cutlass.range_constexpr(MAX_SLOTS // 32):
            i = lane + 32 * h
            if i < num_slots:
                ci = ptx.popc_u32(ptx.lds_u32(mask_s + i * 4))
                rank = Int32(0)
                j = Int32(0)
                while j < num_slots:
                    cj = ptx.popc_u32(ptx.lds_u32(mask_s + j * 4))
                    if cj > ci:
                        rank = rank + 1
                    if cj == ci:
                        if j < i:
                            rank = rank + 1
                    j = j + 1
                if rank < SPEC_KMAX:
                    ptx.st_relaxed_gpu_u32(ids_addr + Int64(rank) * 4, Uint32(s_slot_expert[i]))

    # ------------------------------------------------------------------
    @cute.jit
    def _xc_mma_issuer(
        self,
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
    ):
        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        tl_s = Int32(storage.tl_buf.data_ptr().toint())
        lane0 = cute.arch.lane_idx() == 0
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sB2 = storage.sB2.get_tensor(
            b2_smem_layout_staged.outer, swizzle=b2_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sSFB2 = storage.sSFB2.get_tensor(sfb2_smem_layout_staged)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        tCrB2 = tiled_mma.make_fragment_B(sB2)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding
        )
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + TMEM_SFA_COL, dtype=cutlass.Float8E4M3FN)
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        sfb_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + TMEM_SFB_COL, dtype=cutlass.Float8E4M3FN)
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfb_smem_layout_std, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
        copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = self._s2t_partition(sSFA, tCtSFA)
        copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = self._s2t_partition(sSFB, tCtSFB)
        _, tCsSFB2_s2t, _ = self._s2t_partition(sSFB2, tCtSFB)

        ring_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        acc_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.acc_stages
        )
        num_kphases = cute.size(tCrA, mode=[2])

        for rho in cutlass.range_constexpr(MAX_ROUNDS):
            if rho < rounds:
                acc_pipe.producer_acquire(acc_state)
                tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                kb = Int32(0)
                while kb < FC1_KB:
                    cute.arch.mbarrier_wait(b_mbar + kb, 0)
                    ring.consumer_wait(ring_state)
                    cute.copy(
                        copy_s2t_sfa,
                        tCsSFA_s2t[(None, None, None, None, ring_state.index)],
                        tCtSFA_s2t,
                    )
                    cute.copy(copy_s2t_sfb, tCsSFB_s2t[(None, None, None, None, kb)], tCtSFB_s2t)
                    for kp in cutlass.range_constexpr(num_kphases):
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[(None, None, kp, ring_state.index)],
                            tCrB[(None, None, kp, kb)],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ring.consumer_release(ring_state)
                    ring_state.advance()
                    kb = kb + 1
                acc_pipe.producer_commit(acc_state)
                acc_state.advance()
                if cutlass.const_expr(rho == 0):
                    if lane0:
                        _stamp(tl, tl_s, XT_MMA_FC1_R0)
        if lane0:
            _stamp(tl, tl_s, XT_MMA_FC1)

        for rho in cutlass.range_constexpr(MAX_ROUNDS):
            if rho < rounds:
                # this round's FC2 operand: the four FC1 tiles of the cluster
                bar = Int32((b_mbar + B2_BAR + rho).toint())
                done = ptx.mbar_try_wait_cluster(bar, Int32(0))
                while done == 0:
                    done = ptx.mbar_try_wait_cluster(bar, Int32(0))
                ptx.fence_proxy_async_shared()
                if cutlass.const_expr(rho == 0):
                    if lane0:
                        _stamp(tl, tl_s, XT_B2_READY)
                else:
                    if lane0:
                        _stamp(tl, tl_s, XT_B2_READY_R1)
                t = t_lo[rho]
                while t < t_hi[rho]:
                    acc_pipe.producer_acquire(acc_state)
                    tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    ring.consumer_wait(ring_state)
                    cute.copy(
                        copy_s2t_sfa,
                        tCsSFA_s2t[(None, None, None, None, ring_state.index)],
                        tCtSFA_s2t,
                    )
                    cute.copy(copy_s2t_sfb, tCsSFB2_s2t[(None, None, None, None, rho)], tCtSFB_s2t)
                    for kp in cutlass.range_constexpr(num_kphases):
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[(None, None, kp, ring_state.index)],
                            tCrB2[(None, None, kp, rho)],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ring.consumer_release(ring_state)
                    ring_state.advance()
                    acc_pipe.producer_commit(acc_state)
                    acc_state.advance()
                    t = t + 1
        if lane0:
            _stamp(tl, tl_s, XT_MMA_FC2)
        acc_pipe.producer_tail(acc_state)

    # ------------------------------------------------------------------
    @cute.jit
    def _xc_quantize_x(self, warp_idx, lane, xgs_addr, num_tokens, storage, b_mbar, dbg_addr, rk):
        """NVFP4-quantize this CTA's six activation k-blocks (same arithmetic
        as the stock ``fp4_quantize``), one k-block per quantizing warp, into
        the FC1 B operand of every CTA of the cluster.

        The k-block is written to the local B / SFB stage and released on the
        local mbarrier, then bulk-copied into the same stage of the three
        peers, completing on their k-block mbarriers (which expect the bytes).
        """
        sb_base = Int32(storage.sB.data_ptr().toint())
        sfb_base = Int32(storage.sSFB.data_ptr().toint())
        gs = ptx.ld_nc_f32(xgs_addr)
        rgs = ptx.rcp_approx_ftz(gs)
        rcp6 = ptx.rcp_approx_ftz(Float32(6.0))
        qw = warp_idx
        if warp_idx > TMA_WARP:
            qw = warp_idx - 2
        sx = Int32(storage.sX.data_ptr().toint())
        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        tl_s = Int32(storage.tl_buf.data_ptr().toint())
        blk = lane // 2
        sf_off = (blk // 4) * SFB_CHUNK_STRIDE + blk % 4
        cute.arch.mbarrier_wait(b_mbar + X_BAR, 0)
        kb = rk * KB_PER_CTA + qw
        self._quant_rows_xs(
            0, kb, qw, lane, num_tokens, sx, sb_base, sfb_base, sf_off, gs, rgs, rcp6
        )
        if num_tokens > NTOK // 2:
            self._quant_rows_xs(
                1, kb, qw, lane, num_tokens, sx, sb_base, sfb_base, sf_off, gs, rgs, rcp6
            )
        else:
            if lane < NTOK // 2 * 4:
                n = NTOK // 2 + lane // 4
                ptx.sts_u32(
                    sfb_base + kb * SFB_STAGE_STRIDE + (lane % 4) * SFB_CHUNK_STRIDE + n * 16,
                    Uint32(0),
                )
        ptx.fence_proxy_async_shared()
        cute.arch.sync_warp()
        kbar = Int32((b_mbar + kb).toint())
        if lane == 0:
            cute.arch.mbarrier_arrive(b_mbar + kb)
        # lanes 1..6: (peer, B or SFB) bulk copies into the other CTAs
        if (lane >= 1) & (lane <= 2 * (CLUSTER - 1)):
            peer = (lane - 1) // 2
            if peer >= rk:
                peer = peer + 1
            src = sb_base + kb * 1024
            size = Int32(1024)
            if ((lane - 1) % 2) != 0:
                src = sfb_base + kb * SFB_STAGE_STRIDE
                size = Int32(SFB_STAGE_STRIDE)
            ptx.bulk_s2cluster(ptx.mapa_u32(src, peer), src, size, ptx.mapa_u32(kbar, peer))
        if (warp_idx == 0) & (lane == 0):
            _stamp(tl, tl_s, XT_Q_DONE)
        if cutlass.const_expr(self.debug & DEBUG_QUANT):
            # debug only: wait for every k-block (local and peers'), then CTA 0
            # dumps B / SFB
            for k2 in cutlass.range_constexpr(FC1_KB):
                cute.arch.mbarrier_wait(b_mbar + k2, 0)
            qt = warp_idx * 32 + lane
            if warp_idx > TMA_WARP:
                qt = qt - 64
            ptx.bar_sync(BAR_QUANT, 32 * QUANT_WARPS)
            cta_q, _, _ = cute.arch.block_idx()
            if cta_q == 0:
                w = qt
                while w < (FC1_KB * 1024 + SFB_BYTES) // 4:
                    if w < FC1_KB * 256:
                        ptx.st_u32(dbg_addr + 4096 + w * 4, ptx.lds_u32(sb_base + w * 4))
                    else:
                        r2 = w - FC1_KB * 256
                        ptx.st_u32(dbg_addr + 4096 + w * 4, ptx.lds_u32(sfb_base + r2 * 4))
                    w = w + 32 * QUANT_WARPS

    @cute.jit
    def _quant_rows_xs(
        self, half, kb, kb_local, lane, num_tokens, sx, sb_base, sfb_base, sf_off, gs, rgs, rcp6
    ):
        """Rows ``4*half .. 4*half+3`` of k-block ``kb``, read from this CTA's
        staged share of the activation rows (``kb_local`` within it); rows at
        or beyond ``num_tokens`` get zero data and zero scales."""
        rows = NTOK // 2
        d = cute.make_rmem_tensor((rows * 4,), Uint32)
        for r in cutlass.range_constexpr(rows):
            n = half * rows + r
            x4 = ptx.lds_v4_u32(sx + n * XS_ROW_BYTES + kb_local * 512 + lane * 16)
            for c in cutlass.range_constexpr(4):
                d[4 * r + c] = x4[c]
        for r in cutlass.range_constexpr(rows):
            n = half * rows + r
            vals = cute.make_rmem_tensor((8,), Float32)
            for i in cutlass.range_constexpr(4):
                vals[2 * i] = ptx.bf16x2_to_f32(d[4 * r + i], False)
                vals[2 * i + 1] = ptx.bf16x2_to_f32(d[4 * r + i], True)
            amax = Float32(0.0)
            for i in cutlass.range_constexpr(8):
                amax = cute.arch.fmax(amax, cute.math.absf(vals[i]))
            amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, 1))
            sf = ptx.cvt_e4m3_rn_sat(gs * (amax * rcp6))
            sf_f = ptx.e4m3_to_f32(sf)
            oscale = Float32(0.0)
            if amax != Float32(0.0):
                oscale = ptx.rcp_approx_ftz(sf_f * rgs)
            word = ptx.cvt_e2m1x8(
                vals[0] * oscale,
                vals[1] * oscale,
                vals[2] * oscale,
                vals[3] * oscale,
                vals[4] * oscale,
                vals[5] * oscale,
                vals[6] * oscale,
                vals[7] * oscale,
            )
            if n >= num_tokens:
                word = Uint32(0)
                sf = Uint32(0)
            ptx.sts_u32(
                sb_base + kb * 1024 + n * 128 + (((lane // 4) ^ n) * 16) + (lane % 4) * 4, word
            )
            if lane % 2 == 0:
                ptx.sts_u8(sfb_base + kb * SFB_STAGE_STRIDE + n * 16 + sf_off, sf)

    # ------------------------------------------------------------------
    @cute.jit
    def _xc_epilogue(
        self,
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
    ):
        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        tl_s = Int32(storage.tl_buf.data_ptr().toint())
        hst = Int32(storage.hstage.data_ptr().toint())
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        lane_off = (warp_idx * 32) << 16
        acc_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.acc_stages
        )
        v = cute.make_rmem_tensor((NTOK,), Float32)
        sb2 = Int32(storage.sB2.data_ptr().toint())
        ssfb2 = Int32(storage.sSFB2.data_ptr().toint())
        b2bar = Int32((b_mbar + B2_BAR).toint())

        # ---------------- FC1 tile epilogue -> cluster operand ----------------
        for rho in cutlass.range_constexpr(MAX_ROUNDS):
            if rho < rounds:
                slot = slots[rho]
                expert = s_slot_expert[slot]
                mask = s_slot_mask[slot]
                a_gate = ptx.ld_nc_f32(alpha1_addr + expert * 4)
                s_up = ptx.ld_nc_f32(scale_c1_addr + expert * 4)
                acc_pipe.consumer_wait(acc_state)
                if tidx == 0:
                    _stamp(tl, tl_s, XT_EPI_FC1_ACC if rho == 0 else XT_EPI_FC1_ACC_R1)
                vals = ptx.tmem_ld_32x32b_x8(
                    Int32(tCtAcc_base[(None, None, None, acc_state.index)].iterator.toint())
                    + lane_off
                )
                for n in cutlass.range_constexpr(NTOK):
                    v[n] = vals[n]
                cute.arch.fence_view_async_tmem_load()
                if tidx == 0:
                    _stamp(tl, tl_s, XT_EPI_FC1_TMEM if rho == 0 else XT_EPI_FC1_TMEM_R1)
                with cute.arch.elect_one():
                    acc_pipe.consumer_release(acc_state)
                acc_state.advance()
                if cutlass.const_expr(self.debug & DEBUG_FC1):
                    # the first cluster of a slot dumps its tile's accumulators
                    if firsts[rho] != 0:
                        for n in cutlass.range_constexpr(NTOK):
                            ptx.st_f32(
                                dbg_addr
                                + 4 * (65536 + ((slot * FC1_MT + rk) * 128 + tidx) * NTOK + n),
                                v[n],
                            )
                self._swiglu_quant_to_stage(v, warp_idx, lane, mask, a_gate, s_up, num_tokens, hst)
                if tidx == 0:
                    _stamp(tl, tl_s, XT_EPI_FC1_SWQ if rho == 0 else XT_EPI_FC1_SWQ_R1)
                if cutlass.const_expr(rho == 1):
                    if lane == 0:
                        _stamp(tl, tl_s, XT_EPI_FC1_SWQ_W_R1 + warp_idx)
                if cutlass.const_expr(self.delay & DELAY_PRODUCER):
                    self._xc_delay(DELAY_PRODUCER * 53, bidx)
                ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
                if tidx == 0:
                    _stamp(tl, tl_s, XT_EPI_FC1_SWIGLU if rho == 0 else XT_EPI_FC1_SWIGLU_R1)
                # 64 threads: 16-byte FP4 chunks (row r, half h) for CTA d;
                # 32 threads: the tile's four scale bytes of row r for CTA d.
                if tidx < 4 * 2 * NTOK:
                    d = tidx // (2 * NTOK)
                    q = tidx - d * (2 * NTOK)
                    r = q // 2
                    h = q - r * 2
                    c16 = 2 * rk + h
                    w = ptx.lds_v4_u32(hst + r * 32 + h * 16)
                    dst = ptx.mapa_u32(sb2 + rho * HIMG_B_BYTES + r * 128 + ((c16 ^ r) * 16), d)
                    ptx.st_cluster_v4_u32(dst, w[0], w[1], w[2], w[3])
                    ptx.fence_proxy_async_shared_cluster()
                if (tidx >= 4 * 2 * NTOK) & (tidx < 4 * 3 * NTOK):
                    q = tidx - 4 * 2 * NTOK
                    d = q // NTOK
                    r = q - d * NTOK
                    w1 = ptx.lds_u32(hst + HSTAGE_SF + r * 4)
                    dst = ptx.mapa_u32(
                        ssfb2 + rho * SFB_STAGE_STRIDE + rk * SFB_CHUNK_STRIDE + r * 16, d
                    )
                    ptx.st_cluster_u32(dst, w1)
                    ptx.fence_proxy_async_shared_cluster()
                # all operand stores (and staging reads) precede the arrivals;
                # the staging buffer is free for the next round afterwards
                ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
                if tidx < CLUSTER:
                    ptx.mbar_arrive_remote(ptx.mapa_u32(b2bar + rho * 8, tidx))
                if cutlass.const_expr(rho == 0):
                    if tidx == 0:
                        _stamp(tl, tl_s, XT_EPI_FC1_PUB)
                else:
                    if tidx == 0:
                        _stamp(tl, tl_s, XT_EPI_FC1_PUB_R1)

        # ---------------- FC2 epilogue: raw rows t * 8 + k ----------------
        hid = warp_idx * 32 + (lane % 8) * 4 + lane // 8
        first = Int32(1)
        for rho in cutlass.range_constexpr(MAX_ROUNDS):
            if rho < rounds:
                slot = slots[rho]
                expert = s_slot_expert[slot]
                mask = s_slot_mask[slot]
                a2 = ptx.ld_nc_f32(alpha2_addr + expert * 4)
                kk = cute.make_rmem_tensor((NTOK,), Int32)
                for n in cutlass.range_constexpr(NTOK):
                    kk[n] = s_slot_k[slot * NTOK + n]
                t = t_lo[rho]
                while t < t_hi[rho]:
                    acc_pipe.consumer_wait(acc_state)
                    if tidx == 0:
                        if first != 0:
                            _stamp(tl, tl_s, XT_FC2_FIRST_ACC)
                    first = Int32(0)
                    vals = ptx.tmem_ld_32x32b_x8(
                        Int32(tCtAcc_base[(None, None, None, acc_state.index)].iterator.toint())
                        + lane_off
                    )
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_pipe.consumer_release(acc_state)
                    acc_state.advance()
                    for n in cutlass.range_constexpr(NTOK):
                        if ((mask >> Uint32(n)) & Uint32(1)) != Uint32(0):
                            y = ptx.bf16_round(vals[n] * a2)
                            if cutlass.const_expr(self.red_out):
                                w = s_topk_w[n * TOP_K + kk[n]]
                                ptx.red_add_f32(out_addr + (n * HIDDEN + t * 128 + hid) * 4, w * y)
                            else:
                                row = n * TOP_K + kk[n]
                                ptx.st_u16(
                                    rows_addr + (row * HIDDEN + t * 128 + hid) * 2,
                                    ptx.bf16_bits(y),
                                )
                    t = t + 1
                if cutlass.const_expr(rho == 0):
                    if tidx == 0:
                        _stamp(tl, tl_s, XT_FC2_R0_DONE)
        if tidx == 0:
            _stamp(tl, tl_s, XT_FC2_DONE)
        if cutlass.const_expr(self.finalize):
            self._xc_finalize_last_arriver(
                tidx,
                rounds,
                t_lo,
                t_hi,
                num_slots,
                num_tokens,
                s_topk_w,
                rows_addr,
                out_addr,
                counter_addr,
                storage,
            )
        # the successor (finalize or the post-MoE hop) may start launching
        if cutlass.const_expr(self.trigger == "late"):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _xc_finalize_last_arriver(
        self,
        tidx,
        rounds,
        t_lo,
        t_hi,
        num_slots,
        num_tokens,
        s_topk_w,
        rows_addr,
        out_addr,
        counter_addr,
        storage,
    ):
        """In-kernel native finalize, by the last contributor of each tile.

        Every (expert, output tile) unit is computed by exactly one CTA, so
        an output tile is complete once ``num_slots`` units have counted in.
        After its raw rows are stored, a CTA releases them (one GPU-scope
        fence), counts each of its units into the tile's counter, and
        finalizes every tile for which it was the last arrival: FP32 FMA over
        the top-k in order and one BF16 rounding (the native finalize), then
        resets the counter for the next launch. Nobody waits for anyone.
        """
        # every thread releases its own row stores, then the units count in
        ptx.fence_acq_rel_gpu()
        ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
        closed_base = Int32(storage.closed.data_ptr().toint())
        u = Int32(0)
        for rho in cutlass.range_constexpr(MAX_ROUNDS):
            if rho < rounds:
                n = t_hi[rho] - t_lo[rho]
                if (tidx >= u) & (tidx < u + n):
                    t = t_lo[rho] + (tidx - u)
                    old = ptx.atom_add_relaxed_gpu_u32(counter_addr + t * 4, Uint32(1))
                    closed = Int32(0)
                    if old + Uint32(1) == Uint32(num_slots):
                        closed = Int32(1)
                        ptx.st_relaxed_gpu_u32(counter_addr + t * 4, Uint32(0))
                        ptx.fence_acq_rel_gpu()
                    ptx.sts_u32(closed_base + tidx * 4, Uint32(closed))
                u = u + n
        ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
        # one (token, 8 columns) task per thread: 16 x num_tokens <= 128
        n_tok = tidx // 16
        c8 = (tidx % 16) * 8
        u = Int32(0)
        for rho in cutlass.range_constexpr(MAX_ROUNDS):
            if rho < rounds:
                n = t_hi[rho] - t_lo[rho]
                i = Int32(0)
                while i < n:
                    if ptx.lds_u32(closed_base + (u + i) * 4) != Uint32(0):
                        if n_tok < num_tokens:
                            col = (t_lo[rho] + i) * 128 + c8
                            acc = cute.make_rmem_tensor((8,), Float32)
                            for j in cutlass.range_constexpr(8):
                                acc[j] = Float32(0.0)
                            for k in cutlass.range_constexpr(TOP_K):
                                w = s_topk_w[n_tok * TOP_K + k]
                                y = ptx.ld_ll2(rows_addr + ((n_tok * TOP_K + k) * HIDDEN + col) * 2)
                                for j in cutlass.range_constexpr(4):
                                    acc[2 * j] = ptx.fma_rn(
                                        w, ptx.bf16x2_to_f32(y[j], False), acc[2 * j]
                                    )
                                    acc[2 * j + 1] = ptx.fma_rn(
                                        w, ptx.bf16x2_to_f32(y[j], True), acc[2 * j + 1]
                                    )
                            ptx.st_v4_u32(
                                out_addr + (n_tok * HIDDEN + col) * 2,
                                ptx.pack_bf16x2(acc[0], acc[1]),
                                ptx.pack_bf16x2(acc[2], acc[3]),
                                ptx.pack_bf16x2(acc[4], acc[5]),
                                ptx.pack_bf16x2(acc[6], acc[7]),
                            )
                    i = i + 1
                u = u + n

    @cute.jit
    def _xc_delay(self, salt, bidx):
        """Stress testing: a pseudo-random third of the CTAs (re-drawn per
        launch from the global timer) spins 1..6 units of ``delay_ns``."""
        t0 = ptx.globaltimer_u64()
        seed = Uint32(Int32(t0 >> Int64(12)) & Int32(0xFFFF))
        h = (Uint32(bidx) * Uint32(37) + seed * Uint32(101) + Uint32(salt)) % Uint32(17)
        if h < Uint32(6):
            d = Int64(h + Uint32(1)) * Int64(self.delay_ns)
            t = t0
            while t - t0 < d:
                t = ptx.globaltimer_u64()

    @cute.jit
    def _swiglu_quant_to_stage(self, v, warp_idx, lane, mask, a_gate, s_up, num_tokens, hst):
        """SwiGLU + NVFP4 quantization of one 128-row FC1 tile into the
        staging buffer: 8 rows x 32 FP4 bytes, then 8 rows x 4 scale bytes.

        Row ``32w + q`` of the stock layout holds intermediate index
        ``64*mt + 16*w + j`` with ``j = 2*(q % 8) + (q // 16)`` as the up row
        when ``q % 16 < 8`` and its gate row at lane ``q + 8``, so one warp
        owns exactly one 16-element scale block of the intermediate. Tokens
        not routed to the expert get zero bytes and zero scales.
        """
        is_up = (lane % 16) < 8
        rcp6 = ptx.rcp_approx_ftz(Float32(6.0))
        for n in cutlass.range_constexpr(NTOK):
            gate_raw = cute.arch.shuffle_sync_down(v[n], 8)
            gr = gate_raw * a_gate
            up = v[n] * s_up
            sig = ptx.rcp_approx_ftz(
                Float32(1.0) + ptx.ex2_approx(gr * Float32(-1.4426950408889634))
            )
            h = Float32(0.0)
            if is_up:
                h = up * (gr * sig)
            am = cute.math.absf(h)
            am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, 1))
            am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, 2))
            am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, 4))
            am = cute.arch.fmax(am, cute.arch.shuffle_sync_bfly(am, 16))
            sf = ptx.cvt_e4m3_rn_sat(am * rcp6)
            sf_f = ptx.e4m3_to_f32(sf)
            osc = Float32(0.0)
            if am != Float32(0.0):
                osc = ptx.rcp_approx_ftz(sf_f)
            partner = cute.arch.shuffle_sync_down(h, 16)
            byte = ptx.cvt_e2m1x2(h * osc, partner * osc)
            routed = (((mask >> Uint32(n)) & Uint32(1)) != Uint32(0)) & (n < num_tokens)
            if not routed:
                byte = Uint32(0)
                sf = Uint32(0)
            if lane < 8:
                ptx.sts_u8(hst + n * 32 + warp_idx * 8 + lane, byte)
            if lane == 0:
                ptx.sts_u8(hst + HSTAGE_SF + n * 4 + warp_idx, sf)


class RoutedMoeFinalizeKernel:
    """Native finalize of the raw tuple: ``out[t] = bf16(sum_k fma(w[t, k],
    rows[t * 8 + k]))`` in top-k order with FP32 accumulation; each thread owns
    8 hidden columns of one token. Launched with programmatic dependent launch
    right after the routed kernel.

    ``delay_ns`` (stress testing only, 0 compiles it out): a pseudo-random
    third of the CTAs, re-drawn every launch from the global timer, spins 1..6
    units before storing, so a consumer that read the output before this
    grid completed would see stale rows.
    """

    THREADS = 128

    def __init__(self, delay_ns: int = 0):
        self.delay_ns = delay_ns

    @cute.jit
    def run(self, rows_addr: Int64, scale_addr: Int64, out_addr: Int64, num_tokens: Int32, stream):
        self.kernel(rows_addr, scale_addr, out_addr).launch(
            grid=[HIDDEN // (8 * self.THREADS), num_tokens, 1],
            block=[self.THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, rows_addr: Int64, scale_addr: Int64, out_addr: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        bx, t, _ = cute.arch.block_idx()
        col = (Int32(bx) * self.THREADS + Int32(tidx)) * 8
        t = Int32(t)
        cute.arch.griddepcontrol_wait()
        acc = cute.make_rmem_tensor((8,), Float32)
        for i in cutlass.range_constexpr(8):
            acc[i] = Float32(0.0)
        # written by the routed kernel: plain (coherent) loads after the wait
        ws = ptx.ld_v4_u32(scale_addr + t * (TOP_K * 2))
        for k in cutlass.range_constexpr(TOP_K):
            w = ptx.bf16x2_to_f32(ws[k // 2], k % 2 == 1)
            y = ptx.ld_v4_u32(rows_addr + ((t * TOP_K + k) * HIDDEN + col) * 2)
            for i in cutlass.range_constexpr(4):
                acc[2 * i] = ptx.fma_rn(w, ptx.bf16x2_to_f32(y[i], False), acc[2 * i])
                acc[2 * i + 1] = ptx.fma_rn(w, ptx.bf16x2_to_f32(y[i], True), acc[2 * i + 1])
        if cutlass.const_expr(self.delay_ns > 0):
            t0 = ptx.globaltimer_u64()
            seed = Uint32(Int32(t0 >> Int64(12)) & Int32(0xFFFF))
            h = (
                Uint32(bx + t * 8) * Uint32(37) + seed * Uint32(101) + Uint32(DELAY_FINALIZE)
            ) % Uint32(17)
            if h < Uint32(6):
                d = Int64(h + Uint32(1)) * Int64(self.delay_ns)
                tn = t0
                while tn - t0 < d:
                    tn = ptx.globaltimer_u64()
        ptx.st_v4_u32(
            out_addr + (t * HIDDEN + col) * 2,
            ptx.pack_bf16x2(acc[0], acc[1]),
            ptx.pack_bf16x2(acc[2], acc[3]),
            ptx.pack_bf16x2(acc[4], acc[5]),
            ptx.pack_bf16x2(acc[6], acc[7]),
        )
        cute.arch.griddepcontrol_launch_dependents()
