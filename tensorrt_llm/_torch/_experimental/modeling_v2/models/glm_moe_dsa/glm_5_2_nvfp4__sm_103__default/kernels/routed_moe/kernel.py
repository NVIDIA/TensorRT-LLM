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
"""Single-launch routed NVFP4 MoE for small-M decode (CuTe DSL, SM100 family).

One persistent kernel computes, for R <= 8 rows of one rank's TP shard of a
256-expert / top-8 / intermediate-256 / hidden-6144 NVFP4 MoE:

1. routing (sigmoid + bias top-8, scaled-sum normalization), recomputed by
   every CTA from the FP32 router logits, so no CTA waits for another;
2. NVFP4 quantization of the activation rows (bulk-copied to shared memory
   before any weight traffic) into the MMA B operand, all R rows: unused
   columns are masked in the epilogues;
3. FC1 (gate/up) as a split-K partition of all (expert, 128-row tile,
   256-element k-block) work over the grid: TMA streams the stock weight and
   scale buffers through a 4-stage ring straight into tcgen05 block-scaled
   MMAs (M=128 weight rows, N=8 rows of tokens, K=64 per instruction);
4. the FC1 epilogue (split-K reduction, SwiGLU, NVFP4 quantization of the
   intermediate) publishes a per-expert FC2 operand;
5. FC2 as a split partition of (128-row output tile, expert) units; its
   weight TMA starts as soon as the CTA's FC1 loads are issued;
6. the top-8 combine (finalize) in the FC2 epilogue, or the native
   unfinalized tuple.

Warp roles: 0-3 quantize, then run the epilogues; 4 issues MMAs; 5 issues
TMA; 6-7 quantize, then fetch the FC2 operands.

Cross-CTA hand-offs are kept to one round trip on the consumer side. A
finished FC1 tile is stored straight into its expert's FC2 operand image
(the exact shared-memory layout of one B / SFB stage) and published with an
epoch flag; each consumer CTA waits for the flags of its experts and fetches
every image with two bulk copies that complete on a per-expert mbarrier.
Split-tile partials move as LL words (8 bytes ``{payload, epoch}`` per
16-byte store, polled by the reader). A split tile is reduced by a fixed
owner (the CTA holding its first k-block or expert, which it processes last)
in CTA order, so results are deterministic. Each CTA keeps its own launch
counter as the epoch; all CTAs of a launch hold the same value without
synchronizing. The weight stream is loaded with an evict-first L2 policy.

The weight buffers are consumed in the stock TRT-LLM-Gen layout: gate/up rows
interleaved and shuffled in 32-row blocks, scale factors in the 128x4
interleaved layout. The epilogues undo the row shuffle with index arithmetic.
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

HIDDEN = 6144
INTER = 256
NUM_EXPERTS = 256
TOP_K = 8
NTOK = 8
KTILE = 256
FC1_KB = HIDDEN // KTILE  # 24 k-blocks per FC1 tile
FC1_MT = 2 * INTER // 128  # 4 FC1 tiles per expert
FC2_MT = HIDDEN // 128  # 48 FC2 tiles
MAX_SLOTS = NTOK * TOP_K  # 64 distinct experts at most
MAX_LOCAL_SLOTS = 24  # FC2 operand cache slots per CTA
ROUTED_SCALING = 2.5

# Stock weight geometry in bytes (L2 prefetch address arithmetic).
W13_ROW_BYTES = HIDDEN // 2
W13_EXPERT_BYTES = 2 * INTER * W13_ROW_BYTES
W13_SF_TILE_BYTES = 128 * HIDDEN // 16
W13_SF_EXPERT_BYTES = FC1_MT * W13_SF_TILE_BYTES
W13_SF_KB_BYTES = 128 * KTILE // 16
KB_ROW_BYTES = KTILE // 2
W2_TILE_BYTES = 128 * INTER // 2
W2_EXPERT_BYTES = HIDDEN * INTER // 2
W2_SF_TILE_BYTES = 128 * INTER // 16
W2_SF_EXPERT_BYTES = HIDDEN * INTER // 16

# Workspace: one launch-counter (epoch) word per CTA.
WS_WORDS = 1024
# LL exchange buffers (8-byte words {payload u32, epoch u32}).
LL_PART_BYTES = 128 * NTOK * 8  # one 128x8 FP32 partial
# Per-expert FC2 operand, stored as the exact shared-memory image of one B
# stage (8 rows x 128 B FP4, 16-byte chunks swizzled by row) followed by one
# compact SFB stage, so a consumer fetches it with two bulk copies.
HIMG_B_BYTES = NTOK * INTER // 2
HIMG_SF_BYTES = 4 * 128
HIMG_BYTES = HIMG_B_BYTES + HIMG_SF_BYTES
# LL form of the same operand (hand-off mode 1): 8-byte {4 payload bytes,
# epoch} words; words n*32 + j hold FP4 bytes 4j..4j+3 of row n, words
# 256 + n*4 + mt hold tile mt's four scale bytes of row n.
H_LL_DATA_WORDS = NTOK * INTER // 8
H_LL_WORDS = H_LL_DATA_WORDS + NTOK * FC1_MT
H_LL_BYTES = H_LL_WORDS * 8
H_LL_ITEMS = H_LL_BYTES // 16
H_LL_DATA_ITEMS = H_LL_DATA_WORDS // 2
HBUF_BYTES = max(HIMG_BYTES, H_LL_BYTES)  # per-expert hand-off buffer
# Shared staging of one FC1 tile's output: 8 rows x 32 FP4 bytes, then
# 8 rows x 4 scale bytes.
HSTAGE_SF = NTOK * 32
HSTAGE_BYTES = HSTAGE_SF + NTOK * 4
MAX_PEERS = 7  # other CTAs contributing to one split tile
GATHER_PEERS = 4  # peers' partials fetched per round trip

# Warp roles
EPI_WARPS = 4
MMA_WARP = 4
TMA_WARP = 5
NUM_WARPS = 8
NUM_THREADS = 32 * NUM_WARPS
QUANT_WARPS = 6  # warps 0-3 and 6-7 quantize
GATHER_WARP0 = 6  # warps 6-7 fetch the FC2 operands
GATHER_THREADS = 64

BAR_QUANT = 3  # quantizing warps (debug dump only)
BAR_EPI = 4  # warps 0-3
BAR_GATHER = 5  # warps 6-7

ACC_STAGES = 4
TMEM_COLS = 128
TMEM_SFA_COL = 64
TMEM_SFB_COL = 80
# Compact SFB: only token rows 0..7 of each 128-row scale chunk are read by
# the N=8 MMA, so consecutive 64-K chunks are stored 128 B apart (overlapping
# the unused rows) instead of 512 B.
SFB_CHUNK_STRIDE = 128
SFB_STAGE_STRIDE = 4 * SFB_CHUNK_STRIDE
SFB_BYTES = (MAX_LOCAL_SLOTS - 1) * SFB_STAGE_STRIDE + 3 * SFB_CHUNK_STRIDE + 512
B2_BAR = FC1_KB  # first of MAX_LOCAL_SLOTS FC2 operand barriers
X_BAR = B2_BAR + MAX_LOCAL_SLOTS  # the staged activation rows
FC1_DONE_BAR = X_BAR + 1  # all FC1 MMAs of this CTA have completed
STG_BAR = FC1_DONE_BAR + 1  # LL operand staged in sX (mode 1), per slot
NUM_BAR = STG_BAR + MAX_LOCAL_SLOTS
A_STAGE_BYTES = 128 * KTILE // 2
SFA_STAGE_BYTES = 128 * KTILE // 16
# L2 eviction policy for the weight stream (read once): evict first, so the
# kernel's code, activations and exchange buffers stay in L2.
L2_EVICT_FIRST = 0x12F0000000000000
X_ROW_BYTES = HIDDEN * 2

# L2 prefetch modes (bit mask, ``RoutedMoeKernel(prefetch=...)``; off in this target)
PF_FC1 = 1  # rest of this CTA's FC1 rows, once the ring is primed
PF_FC2_EARLY = 2  # this CTA's FC2 tiles, right after the FC1 prefetch
PF_FC2_LATE = 4  # this CTA's FC2 tiles, after the last FC1 TMA

# Shared misc words
MISC_NUM_SLOTS = 0

DEBUG_ROUTING = 1
DEBUG_QUANT = 2
DEBUG_FC1 = 4
DEBUG_TIMELINE = 8

# Stress-test delay injection sites (``delay`` bits; off in production). A
# pseudo-random third of the CTAs, re-drawn every launch from the epoch,
# spins 1..6 units of ``delay_ns`` on the global timer at each enabled site.
DELAY_ENTRY = 1  # whole CTA, right after the launch epoch is read
DELAY_PRODUCER = 2  # before publishing an FC1 operand tile or a split partial
DELAY_OWNER = 4  # split-tile owner, before it fetches its peers' partials
DELAY_CONSUMER = 8  # FC2 operand fetch, before polling the FC1 flags
TL_BYTE_BASE = 1600000
TL_SLOTS = 48
# Stamps are SM clock cycles; slot TL_GT0 holds the global timer read next
# to the entry stamp so the host can place every CTA on one time axis.
TL_GT0 = TL_SLOTS - 1
(
    TL_ENTRY,
    TL_PDL,
    TL_ROUTED,
    TL_Q_START,
    TL_Q_G0_COMPUTE,
    TL_Q_G0_FENCE,
    TL_Q_G1_FENCE,
    TL_Q_G2_FENCE,
    TL_Q_DONE,
    TL_MMA_FC1_START,
    TL_TMA_FC1,
    TL_PF_ISSUED,
    TL_MMA_FC1_DONE,
    TL_EPI_FIRST_ACC,
    TL_EPI_FC1_GATHERED,
    TL_EPI_FC1_DONE,
    TL_FC2_LL_ISSUED,
    TL_FC2_LL_READY,
    TL_B2_READY,
    TL_TMA_FC2,
    TL_MMA_FC2_DONE,
    TL_FC2_FIRST_ACC,
    TL_FC2_GATHERED,
    TL_FC2_DONE,
    TL_EPI_START,
    TL_FC1_MMA_COMPLETE,
    TL_FC2_SENTINELS,
    TL_EPI_T_COMPUTED,
    TL_EPI_T_STORED,
    TL_EPI_T_TMEM,
    TL_EPI_T_PARAMS,
    TL_G_STS_DONE,
    TL_G_FENCE_DONE,
    TL_G_BAR0_DONE,
    TL_FC2_LAST_SLOT,
) = range(35)


def _stamp(enabled, tl_s, idx):
    # Timeline stamps go to shared memory (a global store here would wait in
    # the same queues the stamps are meant to measure) and are flushed once
    # at the end of the kernel.
    if enabled:
        ptx.sts_u64(tl_s + idx * 8, ptx.clock64())


class RoutedMoeKernel:
    """CuTe DSL kernel object; one compiled instance per configuration."""

    def __init__(
        self,
        num_ctas: int,
        finalize: bool = True,
        num_stages: int = 4,
        prefetch: int = 0,
        min_fc1_kb: int = 12,
        acc_stages: int = 4,
        handoff: int = 0,
        delay: int = 0,
        delay_ns: int = 2000,
        debug: int = 0,
    ):
        assert num_ctas <= WS_WORDS
        # A split FC1 tile spans at most ceil(24 / min_fc1_kb) + 1 CTAs.
        assert (FC1_KB + min_fc1_kb - 1) // min_fc1_kb <= MAX_PEERS
        self.num_ctas = num_ctas
        self.finalize = finalize
        self.num_stages = num_stages
        self.prefetch = prefetch
        self.min_fc1_kb = min_fc1_kb
        # TMEM: acc_stages x 8 accumulator columns, then the scale columns
        assert acc_stages * NTOK <= TMEM_SFA_COL
        self.acc_stages = acc_stages
        # FC1 -> FC2 operand hand-off: 0 = operand image + per-tile release
        # flag, bulk-copied into place; 1 = LL words (no producer fence),
        # bulk-copied into the freed activation buffer and validated there
        self.handoff = handoff
        # Stress testing only: DELAY_* site bits and the delay unit
        self.delay = delay
        self.delay_ns = delay_ns
        self.debug = debug
        self.sf_vec_size = 16
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.mma_inst_mn = (128, NTOK)
        self.mma_tiler = (128, NTOK, KTILE)

    def _setup(self):
        self.tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float8E4M3FN,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_mn,
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (self.tiled_mma.thr_id.shape,)
        )
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler, cutlass.Float4E2M1FN, self.num_stages
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_stages
        )
        # The B operand is resident: one "stage" per FC1 k-block (FC1) and
        # one per local expert slot (FC2); both need 24.
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler, cutlass.Float4E2M1FN, MAX_LOCAL_SLOTS
        )
        # Standard SFB layout (defines the TMEM layout) and the compact SMEM
        # layout actually used for storage (see SFB_CHUNK_STRIDE).
        self.sfb_smem_layout_std = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler, self.sf_vec_size, MAX_LOCAL_SLOTS
        )
        self.sfb_smem_layout_staged = cute.make_layout(
            self.sfb_smem_layout_std.shape,
            stride=((((16, 4), 0), (0, 1)), 0, SFB_CHUNK_STRIDE, SFB_STAGE_STRIDE),
        )
        self.num_tma_load_bytes = A_STAGE_BYTES + SFA_STAGE_BYTES

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
        out_addr: Int64,
        raw_scale_addr: Int64,
        raw_idx_addr: Int64,
        ws_addr: Int64,
        part_addr: Int64,
        himg_addr: Int64,
        hflag_addr: Int64,
        dbg_addr: Int64,
        w13_addr: Int64,
        w13_sf_addr: Int64,
        w2_addr: Int64,
        w2_sf_addr: Int64,
        num_tokens: Int32,
        stream,
    ):
        self._setup()
        print(f"[routed_moe] a_smem {self.a_smem_layout_staged}")
        print(f"[routed_moe] b_smem {self.b_smem_layout_staged}")
        print(f"[routed_moe] sfa_smem {self.sfa_smem_layout_staged}")
        print(f"[routed_moe] sfb_smem {self.sfb_smem_layout_std} -> {self.sfb_smem_layout_staged}")
        # A operands: [rows, K, experts], K-major FP4.
        w13 = cute.make_tensor(
            w13_ptr, cute.make_ordered_layout((2 * INTER, HIDDEN, NUM_EXPERTS), order=(1, 0, 2))
        )
        w2 = cute.make_tensor(
            w2_ptr, cute.make_ordered_layout((HIDDEN, INTER, NUM_EXPERTS), order=(1, 0, 2))
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
            bitmap: cute.struct.MemRange[cutlass.Uint32, 8]
            topk_id: cute.struct.MemRange[cutlass.Int32, NTOK * TOP_K]
            topk_w: cute.struct.MemRange[cutlass.Float32, NTOK * TOP_K]
            slot_expert: cute.struct.MemRange[cutlass.Int32, MAX_SLOTS]
            slot_mask: cute.struct.MemRange[cutlass.Uint32, MAX_SLOTS]
            slot_w: cute.struct.MemRange[cutlass.Float32, MAX_SLOTS * NTOK]
            slot_k: cute.struct.MemRange[cutlass.Int32, MAX_SLOTS * NTOK]
            slot_alpha1: cute.struct.MemRange[cutlass.Float32, MAX_SLOTS]
            slot_scale_c1: cute.struct.MemRange[cutlass.Float32, MAX_SLOTS]
            slot_alpha2: cute.struct.MemRange[cutlass.Float32, MAX_SLOTS]
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
            sX: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, NTOK * X_ROW_BYTES], 1024]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float4E2M1FN, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                1024,
            ]
            sSFB: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, SFB_BYTES], 1024]

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
            x_addr,
            logits_addr,
            bias_addr,
            alpha1_addr,
            scale_c1_addr,
            alpha2_addr,
            xgs_addr,
            out_addr,
            raw_scale_addr,
            raw_idx_addr,
            ws_addr,
            part_addr,
            himg_addr,
            hflag_addr,
            dbg_addr,
            w13_addr,
            w13_sf_addr,
            w2_addr,
            w2_sf_addr,
            num_tokens,
        ).launch(
            grid=[self.num_ctas, 1, 1],
            block=[NUM_THREADS, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    # ------------------------------------------------------------------
    # Device code
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
        x_addr: Int64,
        logits_addr: Int64,
        bias_addr: Int64,
        alpha1_addr: Int64,
        scale_c1_addr: Int64,
        alpha2_addr: Int64,
        xgs_addr: Int64,
        out_addr: Int64,
        raw_scale_addr: Int64,
        raw_idx_addr: Int64,
        ws_addr: Int64,
        part_addr: Int64,
        himg_addr: Int64,
        hflag_addr: Int64,
        dbg_addr: Int64,
        w13_addr: Int64,
        w13_sf_addr: Int64,
        w2_addr: Int64,
        w2_sf_addr: Int64,
        num_tokens: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        cta, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        cta = Int32(cta)
        grid = Int32(grid)

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
            tx_count=self.num_tma_load_bytes,
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
                # one arrival per quantized k-block (all rows)
                for kb in cutlass.range_constexpr(FC1_KB):
                    cute.arch.mbarrier_init(b_mbar + kb, 1)
                for ls in cutlass.range_constexpr(MAX_LOCAL_SLOTS):
                    cute.arch.mbarrier_init(b_mbar + B2_BAR + ls, 1)
                cute.arch.mbarrier_init(b_mbar + X_BAR, 1)
                cute.arch.mbarrier_init(b_mbar + FC1_DONE_BAR, 1)
                for ls in cutlass.range_constexpr(MAX_LOCAL_SLOTS):
                    cute.arch.mbarrier_init(b_mbar + STG_BAR + ls, 1)

        cute.arch.mbarrier_init_fence()

        if warp_idx == 0:
            cute.arch.alloc_tmem(TMEM_COLS, storage.tmem_holding)

        # Shared routing tables, zeroed before routing.
        s_misc = storage.misc.get_tensor(cute.make_layout(16))
        s_bitmap = storage.bitmap.get_tensor(cute.make_layout(8))
        s_topk_id = storage.topk_id.get_tensor(cute.make_layout(NTOK * TOP_K))
        s_topk_w = storage.topk_w.get_tensor(cute.make_layout(NTOK * TOP_K))
        s_slot_expert = storage.slot_expert.get_tensor(cute.make_layout(MAX_SLOTS))
        s_slot_mask = storage.slot_mask.get_tensor(cute.make_layout(MAX_SLOTS))
        s_slot_w = storage.slot_w.get_tensor(cute.make_layout(MAX_SLOTS * NTOK))
        s_slot_k = storage.slot_k.get_tensor(cute.make_layout(MAX_SLOTS * NTOK))
        s_alpha1 = storage.slot_alpha1.get_tensor(cute.make_layout(MAX_SLOTS))
        s_scale_c1 = storage.slot_scale_c1.get_tensor(cute.make_layout(MAX_SLOTS))
        s_alpha2 = storage.slot_alpha2.get_tensor(cute.make_layout(MAX_SLOTS))
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
        # The router bias is a parameter, not an upstream output: read it
        # before waiting on the previous grid.
        bias_r = cute.make_rmem_tensor((8,), Float32)
        for j in cutlass.range_constexpr(8):
            bias_r[j] = ptx.bf16x2_to_f32(ptx.ld_nc_u16(bias_addr + (lane + 32 * j) * 2), False)
        cute.arch.sync_threads()

        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        tl_s = Int32(storage.tl_buf.data_ptr().toint())
        if tidx == 0:
            _stamp(tl, tl_s, TL_ENTRY)
            if cutlass.const_expr(tl):
                ptx.sts_u64(tl_s + TL_GT0 * 8, ptx.globaltimer_u64())
        # Everything below reads tensors produced by earlier kernels or
        # touches the workspace shared with the previous launch.
        cute.arch.griddepcontrol_wait()
        if tidx == 0:
            _stamp(tl, tl_s, TL_PDL)
        # Every CTA counts its own launches, so all CTAs of a launch read the
        # same value without synchronizing; this launch stamps LL words with
        # the incremented value (stale words carry older epochs).
        epoch_next = ptx.ld_relaxed_gpu_u32(ws_addr + cta * 4) + Uint32(1)
        if cutlass.const_expr(self.delay & DELAY_ENTRY):
            self._delay_spin(DELAY_ENTRY * 53, cta, epoch_next)
        # Stage the activation rows in shared memory with bulk copies before
        # any weight streaming starts, so they never queue behind it.
        if warp_idx == TMA_WARP:
            with cute.arch.elect_one():
                xbar = Int32((b_mbar + X_BAR).toint())
                sx = Int32(storage.sX.data_ptr().toint())
                ptx.mbar_arrive_expect_tx(xbar, num_tokens * X_ROW_BYTES)
                r = Int32(0)
                while r < num_tokens:
                    ptx.bulk_g2s(
                        sx + r * X_ROW_BYTES, x_addr + r * X_ROW_BYTES, Int32(X_ROW_BYTES), xbar
                    )
                    r = r + 1

        # -------- routing (warp t handles token t) -> slot tables --------
        self._route_tokens(
            warp_idx, lane, num_tokens, logits_addr, bias_r, s_topk_id, s_topk_w, storage
        )
        cute.arch.sync_threads()
        # Every thread has read the epoch word (barrier above): advance it.
        if tidx == 0:
            ptx.st_relaxed_gpu_u32(ws_addr + cta * 4, epoch_next)
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
            s_misc[MISC_NUM_SLOTS] = total
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
        num_slots = s_misc[MISC_NUM_SLOTS]
        if tidx == 0:
            _stamp(tl, tl_s, TL_ROUTED)

        # ---------------- static work partition ----------------
        # FC1 runs on p1 <= grid CTAs (at least min_fc1_kb k-blocks each);
        # FC2 runs on the whole grid.
        t1 = num_slots * (FC1_MT * FC1_KB)
        p1 = cutlass.min(grid, t1 // self.min_fc1_kb)
        g0 = t1
        g1 = t1
        if cta < p1:
            g0 = (cta * t1) // p1
            g1 = ((cta + 1) * t1) // p1
        t2 = num_slots * FC2_MT
        u0 = (cta * t2) // grid
        u1 = ((cta + 1) * t2) // grid

        if warp_idx == TMA_WARP:
            self._tma_producer(
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
                s_slot_expert,
                g0,
                g1,
                u0,
                u1,
                num_slots,
                w13_addr,
                w13_sf_addr,
                w2_addr,
                w2_sf_addr,
            )
        if warp_idx == MMA_WARP:
            self._mma_issuer(
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
                g0,
                g1,
                u0,
                u1,
                num_slots,
            )
        if (warp_idx != TMA_WARP) & (warp_idx != MMA_WARP):
            self._quantize_x(
                warp_idx, lane, xgs_addr, num_tokens, g0, g1, storage, dbg_addr, b_mbar
            )
        if warp_idx >= GATHER_WARP0:
            self._fetch_fc2_operands(
                tidx - 32 * GATHER_WARP0,
                u0,
                u1,
                num_slots,
                epoch_next,
                storage,
                b_mbar,
                himg_addr,
                hflag_addr,
            )

        if cutlass.const_expr(not self.finalize):
            if cta == 0:
                if tidx < NTOK * TOP_K:
                    t = tidx // TOP_K
                    if t < num_tokens:
                        ptx.st_u16(raw_scale_addr + tidx * 2, ptx.bf16_bits(s_topk_w[tidx]))
                        ptx.st_u32(raw_idx_addr + tidx * 4, Uint32(tidx))

        if cutlass.const_expr(self.debug & DEBUG_ROUTING):
            if cta == 0:
                if tidx < NTOK * TOP_K:
                    ptx.st_u32(dbg_addr + tidx * 4, Uint32(s_topk_id[tidx]))
                    ptx.st_f32(dbg_addr + 256 + tidx * 4, s_topk_w[tidx])
                if tidx < MAX_SLOTS:
                    ptx.st_u32(dbg_addr + 512 + tidx * 4, Uint32(s_slot_expert[tidx]))
                    ptx.st_u32(dbg_addr + 768 + tidx * 4, s_slot_mask[tidx])
                if tidx == 0:
                    ptx.st_u32(dbg_addr + 1024, Uint32(num_slots))

        if warp_idx < EPI_WARPS:
            if tidx < num_slots:
                e = s_slot_expert[tidx]
                s_alpha1[tidx] = ptx.ld_nc_f32(alpha1_addr + e * 4)
                s_scale_c1[tidx] = ptx.ld_nc_f32(scale_c1_addr + e * 4)
                s_alpha2[tidx] = ptx.ld_nc_f32(alpha2_addr + e * 4)
            ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
            if tidx == 0:
                _stamp(tl, tl_s, TL_EPI_START)
            self._epilogue(
                tiled_mma,
                tidx,
                warp_idx,
                lane,
                cta,
                grid,
                p1,
                storage,
                acc_pipe,
                b_mbar,
                s_slot_mask,
                s_slot_w,
                s_slot_k,
                s_alpha1,
                s_scale_c1,
                s_alpha2,
                num_slots,
                num_tokens,
                epoch_next,
                g0,
                g1,
                t1,
                u0,
                u1,
                t2,
                out_addr,
                part_addr,
                himg_addr,
                hflag_addr,
                dbg_addr,
            )
            ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
            if cutlass.const_expr(self.debug & DEBUG_TIMELINE):
                if tidx < TL_SLOTS:
                    ptx.st_u64(
                        dbg_addr + TL_BYTE_BASE + (cta * TL_SLOTS + tidx) * 8,
                        ptx.lds_u64(tl_s + tidx * 8),
                    )
            if warp_idx == 0:
                tmem_ptr = cute.arch.retrieve_tmem_ptr(
                    Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding
                )
                cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)

    # ------------------------------------------------------------------
    @cute.jit
    def _route_tokens(self, w, lane, num_tokens, logits_addr, bias_r, s_topk_id, s_topk_w, storage):
        """Native SigmoidBias + ScaledSumNormalize top-8 for token ``w``.

        Lane ``l`` owns experts ``l + 32 j``. Selection key: twiddled FP32
        ``sigmoid + bias`` bits above ``65535 - expert`` (ties -> lower id),
        as the native 64-bit key.
        """
        if w < num_tokens:
            lg = cute.make_rmem_tensor((8,), Float32)
            for j in cutlass.range_constexpr(8):
                lg[j] = ptx.ld_nc_f32(logits_addr + (w * NUM_EXPERTS + lane + 32 * j) * 4)
            self._route_one(w, lane, lg, 0, bias_r, s_topk_id, s_topk_w, storage)

    @cute.jit
    def _route_one(self, t, lane, lg, h, bias_r, s_topk_id, s_topk_w, storage):
        hi = cute.make_rmem_tensor((8,), Uint32)
        sg = cute.make_rmem_tensor((8,), Float32)
        for j in cutlass.range_constexpr(8):
            s = Float32(0.5) * cute.math.tanh(Float32(0.5) * lg[8 * h + j]) + Float32(0.5)
            score = s + bias_r[j]
            sg[j] = score - bias_r[j]
            bits = ptx.f32_bits(score)
            tw = bits ^ Uint32(0x80000000)
            if (bits >> Uint32(31)) != Uint32(0):
                tw = bits ^ Uint32(0xFFFFFFFF)
            hi[j] = tw
        taken = Uint32(0)
        sel_e = cute.make_rmem_tensor((8,), Int32)
        sig = cute.make_rmem_tensor((8,), Float32)
        for k in cutlass.range_constexpr(8):
            bh = Uint32(0)
            bl = Uint32(0)
            bs = Float32(0.0)
            for j in cutlass.range_constexpr(8):
                lo_j = Uint32(65535 - (lane + 32 * j))
                avail = ((taken >> Uint32(j)) & Uint32(1)) == Uint32(0)
                better = (hi[j] > bh) | ((hi[j] == bh) & (lo_j > bl))
                if avail & better:
                    bh = hi[j]
                    bl = lo_j
                    bs = sg[j]
            mh = ptx.redux_max_u32(bh)
            lo_c = Uint32(0)
            if bh == mh:
                lo_c = bl
            ml = ptx.redux_max_u32(lo_c)
            win = Int32(65535) - Int32(ml)
            if (win % 32) == lane:
                taken = taken | (Uint32(1) << Uint32(win // 32))
            sel_e[k] = win
            sig[k] = cute.arch.shuffle_sync(bs, win % 32)
        # Butterfly order of the native warp reduction over lanes 0..7.
        total = ((sig[0] + sig[4]) + (sig[2] + sig[6])) + ((sig[1] + sig[5]) + (sig[3] + sig[7]))
        if lane < TOP_K:
            my_sig = sig[0]
            my_e = sel_e[0]
            for k in cutlass.range_constexpr(1, 8):
                if lane == k:
                    my_sig = sig[k]
                    my_e = sel_e[k]
            wgt = ptx.bf16_round((my_sig * Float32(ROUTED_SCALING)) / total)
            s_topk_id[t * TOP_K + lane] = my_e
            s_topk_w[t * TOP_K + lane] = wgt
            ptx.atom_or_shared_u32(
                Int32(storage.bitmap.data_ptr().toint()) + (my_e // 32) * 4,
                Uint32(1) << Uint32(my_e % 32),
            )

    # ------------------------------------------------------------------
    @cute.jit
    def _prefetch_fc1(self, lane, g_begin, g1, s_slot_expert, w13_addr, w13_sf_addr):
        """L2 prefetch of FC1 units [g_begin, g1): per tile segment, each of
        the 128 rows' contiguous k-range and the segment's scale bytes."""
        g = g_begin
        while g < g1:
            tile = g // FC1_KB
            tile_base = tile * FC1_KB
            seg_end = cutlass.min(g1, tile_base + FC1_KB)
            slot = tile // FC1_MT
            mt = tile - slot * FC1_MT
            expert = Int64(s_slot_expert[slot])
            ka = g - tile_base
            nk = seg_end - g
            row0 = (
                w13_addr
                + expert * W13_EXPERT_BYTES
                + Int64(mt * 128) * W13_ROW_BYTES
                + Int64(ka * KB_ROW_BYTES)
            )
            for i in cutlass.range_constexpr(4):
                ptx.prefetch_l2_bulk(row0 + Int64(lane + 32 * i) * W13_ROW_BYTES, nk * KB_ROW_BYTES)
            if lane == 0:
                ptx.prefetch_l2_bulk(
                    w13_sf_addr
                    + expert * W13_SF_EXPERT_BYTES
                    + Int64(mt * W13_SF_TILE_BYTES + ka * W13_SF_KB_BYTES),
                    nk * W13_SF_KB_BYTES,
                )
            g = seg_end

    @cute.jit
    def _prefetch_fc2(self, lane, u0, u1, num_slots, s_slot_expert, w2_addr, w2_sf_addr):
        """L2 prefetch of FC2 units [u0, u1): 16 KB weights + 2 KB scales."""
        u = u0 + lane
        while u < u1:
            mt2 = u // num_slots
            slot = u - mt2 * num_slots
            expert = Int64(s_slot_expert[slot])
            ptx.prefetch_l2_bulk(
                w2_addr + expert * W2_EXPERT_BYTES + Int64(mt2 * W2_TILE_BYTES),
                Int32(W2_TILE_BYTES),
            )
            ptx.prefetch_l2_bulk(
                w2_sf_addr + expert * W2_SF_EXPERT_BYTES + Int64(mt2 * W2_SF_TILE_BYTES),
                Int32(W2_SF_TILE_BYTES),
            )
            u = u + 32

    @cute.jit
    def _tma_producer(
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
        s_slot_expert,
        g0,
        g1,
        u0,
        u1,
        num_slots,
        w13_addr,
        w13_sf_addr,
        w2_addr,
        w2_sf_addr,
    ):
        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        tl_s = Int32(storage.tl_buf.data_ptr().toint())
        lane = cute.arch.lane_idx()
        lane0 = lane == 0
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
        # Prime the ring first, then hand the rest of this CTA's weight
        # bytes to L2 prefetches so DRAM sees long sequential streams.
        n_prime = cutlass.min(Int32(self.num_stages), g1 - g0)
        g = g0
        while g < g1:
            if g - g0 == n_prime:
                self._issue_prefetches(
                    lane,
                    g0 + n_prime,
                    g1,
                    u0,
                    u1,
                    num_slots,
                    s_slot_expert,
                    w13_addr,
                    w13_sf_addr,
                    w2_addr,
                    w2_sf_addr,
                )
                if lane0:
                    _stamp(tl, tl_s, TL_PF_ISSUED)
            tile = g // FC1_KB
            kb = g - tile * FC1_KB
            slot = tile // FC1_MT
            mt = tile - slot * FC1_MT
            expert = s_slot_expert[slot]
            ring.producer_acquire(state)
            bar = ring.producer_get_barrier(state)
            cute.copy(
                tma_atom_a1,
                tAgA1[(None, mt, kb, expert)],
                tAsA[(None, state.index)],
                tma_bar_ptr=bar,
                cache_policy=Int64(L2_EVICT_FIRST),
            )
            cute.copy(
                tma_atom_sfa1,
                tAgSFA1[(None, mt, kb, expert)],
                tAsSFA[(None, state.index)],
                tma_bar_ptr=bar,
                cache_policy=Int64(L2_EVICT_FIRST),
            )
            state.advance()
            g = g + 1
        if g1 - g0 == n_prime:
            # Fewer FC1 units than ring stages: nothing left of FC1.
            self._issue_prefetches(
                lane,
                g1,
                g1,
                u0,
                u1,
                num_slots,
                s_slot_expert,
                w13_addr,
                w13_sf_addr,
                w2_addr,
                w2_sf_addr,
            )
            if lane0:
                _stamp(tl, tl_s, TL_PF_ISSUED)
        if lane0:
            _stamp(tl, tl_s, TL_TMA_FC1)
        if cutlass.const_expr(self.prefetch & PF_FC2_LATE):
            self._prefetch_fc2(lane, u0, u1, num_slots, s_slot_expert, w2_addr, w2_sf_addr)
        u = u0
        while u < u1:
            mt2 = u // num_slots
            slot = u - mt2 * num_slots
            expert = s_slot_expert[slot]
            ring.producer_acquire(state)
            bar = ring.producer_get_barrier(state)
            cute.copy(
                tma_atom_a2,
                tAgA2[(None, mt2, 0, expert)],
                tAsA[(None, state.index)],
                tma_bar_ptr=bar,
                cache_policy=Int64(L2_EVICT_FIRST),
            )
            cute.copy(
                tma_atom_sfa2,
                tAgSFA2[(None, mt2, 0, expert)],
                tAsSFA[(None, state.index)],
                tma_bar_ptr=bar,
                cache_policy=Int64(L2_EVICT_FIRST),
            )
            state.advance()
            u = u + 1
        if lane0:
            _stamp(tl, tl_s, TL_TMA_FC2)
        ring.producer_tail(state)

    @cute.jit
    def _issue_prefetches(
        self,
        lane,
        g_begin,
        g1,
        u0,
        u1,
        num_slots,
        s_slot_expert,
        w13_addr,
        w13_sf_addr,
        w2_addr,
        w2_sf_addr,
    ):
        if cutlass.const_expr(self.prefetch & PF_FC1):
            self._prefetch_fc1(lane, g_begin, g1, s_slot_expert, w13_addr, w13_sf_addr)
        if cutlass.const_expr(self.prefetch & PF_FC2_EARLY):
            self._prefetch_fc2(lane, u0, u1, num_slots, s_slot_expert, w2_addr, w2_sf_addr)

    # ------------------------------------------------------------------
    @cute.jit
    def _mma_issuer(
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
        g0,
        g1,
        u0,
        u1,
        num_slots,
    ):
        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        tl_s = Int32(storage.tl_buf.data_ptr().toint())
        lane0 = cute.arch.lane_idx() == 0
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
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

        ring_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        acc_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.acc_stages
        )
        num_kphases = cute.size(tCrA, mode=[2])

        # FC1: wait for the quantized activations, then one accumulator per
        # split-K segment.
        if lane0:
            _stamp(tl, tl_s, TL_MMA_FC1_START)
        g = g0
        while g < g1:
            tile = g // FC1_KB
            seg_end = cutlass.min(g1, (tile + 1) * FC1_KB)
            acc_pipe.producer_acquire(acc_state)
            tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            while g < seg_end:
                kb = g - (g // FC1_KB) * FC1_KB
                cute.arch.mbarrier_wait(b_mbar + kb, 0)
                ring.consumer_wait(ring_state)
                cute.copy(
                    copy_s2t_sfa, tCsSFA_s2t[(None, None, None, None, ring_state.index)], tCtSFA_s2t
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
                g = g + 1
            acc_pipe.producer_commit(acc_state)
            acc_state.advance()

        # The FC2 operands overwrite the FC1 B operand: tell the gathering
        # warps once every FC1 MMA (and scale copy) has completed.
        with cute.arch.elect_one():
            tcgen05.commit(b_mbar + FC1_DONE_BAR)
        # FC2: one accumulator per (output tile, expert) unit.
        if lane0:
            _stamp(tl, tl_s, TL_MMA_FC1_DONE)
        s_first = u0 - (u0 // num_slots) * num_slots
        u = u0
        while u < u1:
            mt2 = u // num_slots
            slot = u - mt2 * num_slots
            ls = slot - s_first
            if ls < 0:
                ls = ls + num_slots
            # operands of local slot ls are in shared memory
            cute.arch.mbarrier_wait(b_mbar + B2_BAR + ls, 0)
            acc_pipe.producer_acquire(acc_state)
            tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            ring_state = self._fc2_mma_ring(
                tiled_mma,
                ring,
                ring_state,
                tCtAcc,
                copy_s2t_sfa,
                tCsSFA_s2t,
                tCtSFA_s2t,
                copy_s2t_sfb,
                tCsSFB_s2t,
                tCtSFB_s2t,
                tCtSFA,
                tCtSFB,
                tCrA,
                tCrB,
                ls,
                num_kphases,
            )
            acc_pipe.producer_commit(acc_state)
            acc_state.advance()
            u = u + 1
        if lane0:
            _stamp(tl, tl_s, TL_MMA_FC2_DONE)
        acc_pipe.producer_tail(acc_state)

    @cute.jit
    def _fc2_mma_ring(
        self,
        tiled_mma,
        ring,
        ring_state,
        tCtAcc,
        copy_s2t_sfa,
        tCsSFA_s2t,
        tCtSFA_s2t,
        copy_s2t_sfb,
        tCsSFB_s2t,
        tCtSFB_s2t,
        tCtSFA,
        tCtSFB,
        tCrA,
        tCrB,
        ls,
        num_kphases,
    ):
        """One FC2 unit from the weight ring (A) and local slot ``ls`` (B)."""
        ring.consumer_wait(ring_state)
        cute.copy(copy_s2t_sfa, tCsSFA_s2t[(None, None, None, None, ring_state.index)], tCtSFA_s2t)
        cute.copy(copy_s2t_sfb, tCsSFB_s2t[(None, None, None, None, ls)], tCtSFB_s2t)
        for kp in cutlass.range_constexpr(num_kphases):
            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
            tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
            cute.gemm(
                tiled_mma,
                tCtAcc,
                tCrA[(None, None, kp, ring_state.index)],
                tCrB[(None, None, kp, ls)],
                tCtAcc,
            )
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        ring.consumer_release(ring_state)
        ring_state.advance()
        return ring_state

    def _s2t_partition(self, sSF, tSF):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), cutlass.Float8E4M3FN
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # ------------------------------------------------------------------
    @cute.jit
    def _quantize_x(self, warp_idx, lane, xgs_addr, num_tokens, g0, g1, storage, dbg_addr, b_mbar):
        """NVFP4-quantize the activation k-blocks this CTA's FC1 range uses.

        One warp task is one k-block for all rows: 256 BF16 elements per row,
        8 per lane, two lanes per 16-element scale block, read from the
        staged rows in shared memory; the rows are independent, so their
        dependency chains interleave. Same arithmetic as the stock
        ``fp4_quantize`` (E4M3 scale ``gs * amax * rcp(6)``, RN e2m1). Warps
        take k-blocks in the MMA's order and each finished k-block arrives on
        its mbarrier, so the first FC1 MMAs start while later k-blocks are
        still being quantized. Padded rows (>= num_tokens) are quantized from
        zeros: zero data and zero scales.
        """
        sb_base = Int32(storage.sB.data_ptr().toint())
        sfb_base = Int32(storage.sSFB.data_ptr().toint())
        gs = ptx.ld_nc_f32(xgs_addr)
        rgs = ptx.rcp_approx_ftz(gs)
        rcp6 = ptx.rcp_approx_ftz(Float32(6.0))
        nkb = cutlass.min(g1 - g0, FC1_KB)
        kb_first = g0 - (g0 // FC1_KB) * FC1_KB
        if cutlass.const_expr(self.debug & DEBUG_QUANT):
            nkb = Int32(FC1_KB)
            kb_first = Int32(0)
        qw = warp_idx
        if warp_idx > TMA_WARP:
            qw = warp_idx - 2
        sx = Int32(storage.sX.data_ptr().toint())
        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        tl_s = Int32(storage.tl_buf.data_ptr().toint())
        stamp_q = (warp_idx == 0) & (lane == 0)
        blk = lane // 2
        sf_off = (blk // 4) * SFB_CHUNK_STRIDE + blk % 4
        cute.arch.mbarrier_wait(b_mbar + X_BAR, 0)
        if stamp_q:
            _stamp(tl, tl_s, TL_Q_START)
        j = Int32(0)
        while qw + QUANT_WARPS * j < nkb:
            kb = kb_first + qw + QUANT_WARPS * j
            if kb >= FC1_KB:
                kb = kb - FC1_KB
            self._quant_rows(0, kb, lane, num_tokens, sx, sb_base, sfb_base, sf_off, gs, rgs, rcp6)
            if num_tokens > NTOK // 2:
                self._quant_rows(
                    1, kb, lane, num_tokens, sx, sb_base, sfb_base, sf_off, gs, rgs, rcp6
                )
            else:
                # rows 4..7 are padding: zero their scales
                if lane < NTOK // 2 * 4:
                    n = NTOK // 2 + lane // 4
                    ptx.sts_u32(
                        sfb_base + kb * SFB_STAGE_STRIDE + (lane % 4) * SFB_CHUNK_STRIDE + n * 16,
                        Uint32(0),
                    )
            if stamp_q:
                if j == 0:
                    _stamp(tl, tl_s, TL_Q_G0_COMPUTE)
            ptx.fence_proxy_async_shared()
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(b_mbar + kb)
            if stamp_q:
                if j < 3:
                    _stamp(tl, tl_s, TL_Q_G0_FENCE + j)
            j = j + 1
        if stamp_q:
            _stamp(tl, tl_s, TL_Q_DONE)
        if cutlass.const_expr(self.debug & DEBUG_QUANT):
            # debug only: wait for every k-block, then CTA 0 dumps B / SFB
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
    def _quant_rows(self, half, kb, lane, num_tokens, sx, sb_base, sfb_base, sf_off, gs, rgs, rcp6):
        """Quantize rows ``4*half .. 4*half+3`` of k-block ``kb``; rows at or
        beyond ``num_tokens`` get zero data and zero scales."""
        rows = NTOK // 2
        # All four rows' loads first (independent chains); rows past
        # num_tokens read stale shared memory and are masked to zero below.
        d = cute.make_rmem_tensor((rows * 4,), Uint32)
        for r in cutlass.range_constexpr(rows):
            n = half * rows + r
            x4 = ptx.lds_v4_u32(sx + n * X_ROW_BYTES + kb * 512 + lane * 16)
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
    def _fetch_fc2_operands(
        self, gt, u0, u1, num_slots, epoch_next, storage, b_mbar, himg_addr, hflag_addr
    ):
        """Warps 6-7: fetch this CTA's FC2 operands into shared memory.

        Thread ``ls`` (< local slots) waits for the four FC1 tile flags of
        expert slot ``s_first + ls``, then bulk-copies the slot's operand
        image into B stage ``ls`` / compact SFB stage ``ls``, completing on
        that slot's mbarrier, so the FC2 MMAs of early slots start while
        later ones are in flight. Shared memory is written only after every
        FC1 MMA of this CTA has completed.
        """
        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        tl_s = Int32(storage.tl_buf.data_ptr().toint())
        nls = cutlass.min(num_slots, u1 - u0)
        s_first = u0 - (u0 // num_slots) * num_slots
        cta, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.delay & DELAY_CONSUMER):
            self._delay_spin(DELAY_CONSUMER * 53, Int32(cta), epoch_next)
        if cutlass.const_expr(self.handoff == 1):
            self._fetch_fc2_operands_ll(
                gt, nls, s_first, num_slots, epoch_next, storage, b_mbar, himg_addr, tl, tl_s
            )
        if cutlass.const_expr(self.handoff == 0):
            if gt < nls:
                slot = s_first + gt
                if slot >= num_slots:
                    slot = slot - num_slots
                fa = hflag_addr + slot * (FC1_MT * 4)
                f0 = ptx.ld_acquire_gpu_u32(fa)
                f1 = ptx.ld_acquire_gpu_u32(fa + 4)
                f2 = ptx.ld_acquire_gpu_u32(fa + 8)
                f3 = ptx.ld_acquire_gpu_u32(fa + 12)
                while (
                    (f0 != epoch_next)
                    | (f1 != epoch_next)
                    | (f2 != epoch_next)
                    | (f3 != epoch_next)
                ):
                    f0 = ptx.ld_acquire_gpu_u32(fa)
                    f1 = ptx.ld_acquire_gpu_u32(fa + 4)
                    f2 = ptx.ld_acquire_gpu_u32(fa + 8)
                    f3 = ptx.ld_acquire_gpu_u32(fa + 12)
                if gt == 0:
                    _stamp(tl, tl_s, TL_FC2_SENTINELS)
                if gt == nls - 1:
                    _stamp(tl, tl_s, TL_FC2_LAST_SLOT)
                # the images were written by generic stores of other CTAs; the
                # bulk copy reads them through the async proxy
                ptx.fence_proxy_async_global()
                cute.arch.mbarrier_wait(b_mbar + FC1_DONE_BAR, 0)
                if gt == 0:
                    _stamp(tl, tl_s, TL_FC1_MMA_COMPLETE)
                bar = Int32((b_mbar + B2_BAR).toint()) + gt * 8
                ptx.mbar_arrive_expect_tx(bar, Int32(HIMG_BYTES))
                src = himg_addr + slot * HBUF_BYTES
                ptx.bulk_g2s(
                    Int32(storage.sB.data_ptr().toint()) + gt * HIMG_B_BYTES,
                    src,
                    Int32(HIMG_B_BYTES),
                    bar,
                )
                ptx.bulk_g2s(
                    Int32(storage.sSFB.data_ptr().toint()) + gt * SFB_STAGE_STRIDE,
                    src + HIMG_B_BYTES,
                    Int32(HIMG_SF_BYTES),
                    bar,
                )
                if gt == 0:
                    _stamp(tl, tl_s, TL_B2_READY)

    @cute.jit
    def _fetch_fc2_operands_ll(
        self, gt, nls, s_first, num_slots, epoch_next, storage, b_mbar, hll_addr, tl, tl_s
    ):
        """Hand-off mode 1: LL operands staged through the activation buffer.

        Thread ``ls`` waits for one LL word of each of its expert's four FC1
        tiles (relaxed polls, no fence), then bulk-copies the expert's LL
        words into sX; all 64 threads then validate every word (re-polling
        global memory for the rare stale one) and scatter the payload into
        B / compact SFB stage ``ls``, expert by expert, releasing each to the
        MMA warp with its own mbarrier.
        """
        sx = Int32(storage.sX.data_ptr().toint())
        if gt < nls:
            slot = s_first + gt
            if slot >= num_slots:
                slot = slot - num_slots
            base = hll_addr + slot * HBUF_BYTES
            # epoch half of the row-7 scale word of each tile
            sa = base + (H_LL_DATA_WORDS + (NTOK - 1) * FC1_MT) * 8 + 4
            f0 = ptx.ld_relaxed_gpu_u32(sa)
            f1 = ptx.ld_relaxed_gpu_u32(sa + 8)
            f2 = ptx.ld_relaxed_gpu_u32(sa + 16)
            f3 = ptx.ld_relaxed_gpu_u32(sa + 24)
            while (f0 != epoch_next) | (f1 != epoch_next) | (f2 != epoch_next) | (f3 != epoch_next):
                f0 = ptx.ld_relaxed_gpu_u32(sa)
                f1 = ptx.ld_relaxed_gpu_u32(sa + 8)
                f2 = ptx.ld_relaxed_gpu_u32(sa + 16)
                f3 = ptx.ld_relaxed_gpu_u32(sa + 24)
            if gt == 0:
                _stamp(tl, tl_s, TL_FC2_SENTINELS)
            if gt == nls - 1:
                _stamp(tl, tl_s, TL_FC2_LAST_SLOT)
            # sX is free: quantization is complete once all FC1 MMAs are
            cute.arch.mbarrier_wait(b_mbar + FC1_DONE_BAR, 0)
            bar = Int32((b_mbar + STG_BAR).toint()) + gt * 8
            ptx.mbar_arrive_expect_tx(bar, Int32(H_LL_BYTES))
            ptx.bulk_g2s(sx + gt * H_LL_BYTES, base, Int32(H_LL_BYTES), bar)
        sb_base = Int32(storage.sB.data_ptr().toint())
        sfb_base = Int32(storage.sSFB.data_ptr().toint())
        ls = Int32(0)
        while ls < nls:
            cute.arch.mbarrier_wait(b_mbar + STG_BAR + ls, 0)
            if gt == 0:
                if ls == 0:
                    _stamp(tl, tl_s, TL_FC1_MMA_COMPLETE)
            i = gt
            while i < H_LL_ITEMS:
                d = ptx.lds_v4_u32(sx + ls * H_LL_BYTES + i * 16)
                if (d[1] != epoch_next) | (d[3] != epoch_next):
                    slot = s_first + ls
                    if slot >= num_slots:
                        slot = slot - num_slots
                    a = hll_addr + slot * HBUF_BYTES + i * 16
                    d = ptx.ld_ll2(a)
                    while (d[1] != epoch_next) | (d[3] != epoch_next):
                        d = ptx.ld_ll2(a)
                if i < H_LL_DATA_ITEMS:
                    # words (n, w), (n, w + 1) of the FP4 row n
                    n = i // 16
                    w = 2 * (i - n * 16)
                    ptx.sts_v2_u32(
                        sb_base + ls * HIMG_B_BYTES + n * 128 + (((w // 4) ^ n) * 16) + (w % 4) * 4,
                        d[0],
                        d[2],
                    )
                else:
                    # scale words (n, mt), (n, mt + 1)
                    rr = i - H_LL_DATA_ITEMS
                    n = rr // 2
                    mt0 = 2 * (rr - n * 2)
                    ptx.sts_u32(
                        sfb_base + ls * SFB_STAGE_STRIDE + mt0 * SFB_CHUNK_STRIDE + n * 16, d[0]
                    )
                    ptx.sts_u32(
                        sfb_base + ls * SFB_STAGE_STRIDE + (mt0 + 1) * SFB_CHUNK_STRIDE + n * 16,
                        d[2],
                    )
                i = i + GATHER_THREADS
            ptx.fence_proxy_async_shared()
            ptx.bar_sync(BAR_GATHER, GATHER_THREADS)
            if gt == 0:
                cute.arch.mbarrier_arrive(b_mbar + B2_BAR + ls)
                if ls == 0:
                    _stamp(tl, tl_s, TL_B2_READY)
            ls = ls + 1

    # ------------------------------------------------------------------
    @cute.jit
    def _delay_spin(self, salt, cta, epoch):
        """Stress testing: spin on the global timer.

        The delayed CTAs and the delay length are a hash of (CTA, launch
        epoch, site salt), so every launch perturbs a different set of
        producers and consumers. Call sites are compiled only when their
        ``DELAY_*`` bit is set.
        """
        h = (
            Uint32(cta) * Uint32(37) + (epoch & Uint32(0xFFFF)) * Uint32(101) + Uint32(salt)
        ) % Uint32(17)
        if h < Uint32(6):
            d = Int64(h + Uint32(1)) * Int64(self.delay_ns)
            t0 = ptx.globaltimer_u64()
            t = t0
            while t - t0 < d:
                t = ptx.globaltimer_u64()

    @cute.jit
    def _cta_of(self, g, total, parts):
        """CTA whose range [c*total//parts, (c+1)*total//parts) holds g."""
        return ((g + 1) * parts - 1) // total

    @cute.jit
    def _ll_publish(self, v, tidx, cta, base, flag):
        """Fire-and-forget LL store of this thread's 8 partial values."""
        a = base + cta * LL_PART_BYTES + tidx * (NTOK * 8)
        for j in cutlass.range_constexpr(NTOK // 2):
            ptx.st_ll2(a + j * 16, ptx.f32_bits(v[2 * j]), ptx.f32_bits(v[2 * j + 1]), flag)

    @cute.jit
    def _ll_issue(self, buf, tidx, c_begin, c_end, base):
        """Start loading the LL partials of CTAs c_begin.. (at most
        GATHER_PEERS of them) into ``buf``; the owner issues this before its
        own last segment, so the round trip overlaps that work."""
        for p in cutlass.range_constexpr(GATHER_PEERS):
            if c_begin + p <= c_end:
                a = base + (c_begin + p) * LL_PART_BYTES + tidx * (NTOK * 8)
                for j in cutlass.range_constexpr(NTOK // 2):
                    d = ptx.ld_ll2(a + j * 16)
                    for k in cutlass.range_constexpr(4):
                        buf[p * 2 * NTOK + j * 4 + k] = d[k]

    @cute.jit
    def _ll_finish(self, v, buf, tidx, c_begin, c_end, base, flag):
        """v += the LL partials of CTAs c_begin..c_end, summed in CTA order.

        The first GATHER_PEERS peers' words are already in flight in
        ``buf``; stale words are re-polled in rounds. Further peers (only
        with very small splits) are fetched GATHER_PEERS at a time.
        """
        c0 = c_begin
        while c0 <= c_end:
            if c0 != c_begin:
                self._ll_issue(buf, tidx, c0, c_end, base)
            pending = Int32(1)
            while pending != 0:
                pending = Int32(0)
                for p in cutlass.range_constexpr(GATHER_PEERS):
                    if c0 + p <= c_end:
                        a = base + (c0 + p) * LL_PART_BYTES + tidx * (NTOK * 8)
                        for j in cutlass.range_constexpr(NTOK // 2):
                            o = p * 2 * NTOK + j * 4
                            if (buf[o + 1] != flag) | (buf[o + 3] != flag):
                                d = ptx.ld_ll2(a + j * 16)
                                for k in cutlass.range_constexpr(4):
                                    buf[o + k] = d[k]
                                pending = Int32(1)
            for p in cutlass.range_constexpr(GATHER_PEERS):
                if c0 + p <= c_end:
                    for j in cutlass.range_constexpr(NTOK // 2):
                        o = p * 2 * NTOK + j * 4
                        v[2 * j] = v[2 * j] + ptx.bits_f32(buf[o])
                        v[2 * j + 1] = v[2 * j + 1] + ptx.bits_f32(buf[o + 2])
            c0 = c0 + GATHER_PEERS

    @cute.jit
    def _epilogue(
        self,
        tiled_mma,
        tidx,
        warp_idx,
        lane,
        cta,
        grid,
        p1,
        storage,
        acc_pipe,
        b_mbar,
        s_slot_mask,
        s_slot_w,
        s_slot_k,
        s_alpha1,
        s_scale_c1,
        s_alpha2,
        num_slots,
        num_tokens,
        epoch_next,
        g0,
        g1,
        t1,
        u0,
        u1,
        t2,
        out_addr,
        part_addr,
        himg_addr,
        hflag_addr,
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
        part1 = part_addr
        part2 = part_addr + grid * LL_PART_BYTES

        # ---------------- FC1 epilogue ----------------
        # A tile split over CTAs is finished by the CTA holding its k-block
        # 0 (its last segment); the others publish their partial (their
        # first segment) as LL words.
        pbuf = cute.make_rmem_tensor((GATHER_PEERS * 2 * NTOK,), Uint32)
        g = g0
        while g < g1:
            tile = g // FC1_KB
            tile_base = tile * FC1_KB
            seg_end = cutlass.min(g1, tile_base + FC1_KB)
            split_head = (g == tile_base) & (seg_end != tile_base + FC1_KB)
            c_last = self._cta_of(tile_base + FC1_KB - 1, t1, p1)
            if split_head:
                if cutlass.const_expr(self.delay & DELAY_OWNER):
                    self._delay_spin(DELAY_OWNER * 53, cta, epoch_next)
                self._ll_issue(pbuf, tidx, cta + 1, c_last, part1)
            acc_pipe.consumer_wait(acc_state)
            if tidx == 0:
                if g == g0:
                    _stamp(tl, tl_s, TL_EPI_FIRST_ACC)
            vals = ptx.tmem_ld_32x32b_x8(
                Int32(tCtAcc_base[(None, None, None, acc_state.index)].iterator.toint()) + lane_off
            )
            for n in cutlass.range_constexpr(NTOK):
                v[n] = vals[n]
            cute.arch.fence_view_async_tmem_load()
            if tidx == 0:
                _stamp(tl, tl_s, TL_EPI_T_TMEM)
            with cute.arch.elect_one():
                acc_pipe.consumer_release(acc_state)
            acc_state.advance()
            if g == tile_base:
                if split_head:
                    self._ll_finish(v, pbuf, tidx, cta + 1, c_last, part1, epoch_next)
                    if tidx == 0:
                        _stamp(tl, tl_s, TL_EPI_FC1_GATHERED)
                self._fc1_tile_epilogue(
                    v,
                    tile,
                    warp_idx,
                    lane,
                    tidx,
                    s_slot_mask,
                    s_alpha1,
                    s_scale_c1,
                    num_tokens,
                    epoch_next,
                    himg_addr,
                    hflag_addr,
                    hst,
                    tl_s,
                    dbg_addr,
                )
            else:
                if cutlass.const_expr(self.delay & DELAY_PRODUCER):
                    self._delay_spin(DELAY_PRODUCER * 53, cta, epoch_next)
                self._ll_publish(v, tidx, cta, part1, epoch_next)
            g = seg_end
        if tidx == 0:
            _stamp(tl, tl_s, TL_EPI_FC1_DONE)

        # ---------------- FC2 epilogue ----------------
        o = cute.make_rmem_tensor((NTOK,), Float32)
        hid = warp_idx * 32 + (lane % 8) * 4 + lane // 8
        u = u0
        while u < u1:
            mt2 = u // num_slots
            seg_start = u
            seg_end = cutlass.min(u1, (mt2 + 1) * num_slots)
            tile_base = mt2 * num_slots
            c_last = self._cta_of(tile_base + num_slots - 1, t2, grid)
            split_head = (seg_start == tile_base) & (seg_end != tile_base + num_slots)
            if cutlass.const_expr(self.finalize):
                if split_head:
                    if cutlass.const_expr(self.delay & DELAY_OWNER):
                        self._delay_spin(DELAY_OWNER * 53, cta, epoch_next)
                    self._ll_issue(pbuf, tidx, cta + 1, c_last, part2)
            for n in cutlass.range_constexpr(NTOK):
                o[n] = Float32(0.0)
            while u < seg_end:
                slot = u - mt2 * num_slots
                mask = s_slot_mask[slot]
                a2 = s_alpha2[slot]
                acc_pipe.consumer_wait(acc_state)
                if tidx == 0:
                    if u == u0:
                        _stamp(tl, tl_s, TL_FC2_FIRST_ACC)
                vals = ptx.tmem_ld_32x32b_x8(
                    Int32(tCtAcc_base[(None, None, None, acc_state.index)].iterator.toint())
                    + lane_off
                )
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipe.consumer_release(acc_state)
                acc_state.advance()
                for n in cutlass.range_constexpr(NTOK):
                    y = ptx.bf16_round(vals[n] * a2)
                    if ((mask >> Uint32(n)) & Uint32(1)) != Uint32(0):
                        if cutlass.const_expr(self.finalize):
                            o[n] = o[n] + s_slot_w[slot * NTOK + n] * y
                        else:
                            k = s_slot_k[slot * NTOK + n]
                            row = n * TOP_K + k
                            ptx.st_u16(
                                out_addr + (row * HIDDEN + mt2 * 128 + hid) * 2, ptx.bf16_bits(y)
                            )
                u = u + 1
            if cutlass.const_expr(self.finalize):
                if seg_start == tile_base:
                    if split_head:
                        self._ll_finish(o, pbuf, tidx, cta + 1, c_last, part2, epoch_next)
                        if tidx == 0:
                            _stamp(tl, tl_s, TL_FC2_GATHERED)
                    for n in cutlass.range_constexpr(NTOK):
                        if n < num_tokens:
                            ptx.st_u16(
                                out_addr + (n * HIDDEN + mt2 * 128 + hid) * 2, ptx.bf16_bits(o[n])
                            )
                else:
                    if cutlass.const_expr(self.delay & DELAY_PRODUCER):
                        self._delay_spin(DELAY_PRODUCER * 53, cta, epoch_next)
                    self._ll_publish(o, tidx, cta, part2, epoch_next)
        if tidx == 0:
            _stamp(tl, tl_s, TL_FC2_DONE)

    @cute.jit
    def _fc1_tile_epilogue(
        self,
        v,
        tile,
        warp_idx,
        lane,
        tidx,
        s_slot_mask,
        s_alpha1,
        s_scale_c1,
        num_tokens,
        epoch_next,
        himg_addr,
        hflag_addr,
        hst,
        tl_s,
        dbg_addr,
    ):
        """SwiGLU + NVFP4 quantization of one 128-row FC1 tile.

        Row ``32w + q`` of the stock layout holds intermediate index
        ``64*mt + 16*w + j`` with ``j = 2*(q % 8) + (q // 16)`` as the up row
        when ``q % 16 < 8`` and its gate row at lane ``q + 8``, so one warp
        owns exactly one 16-element scale block of the intermediate. The
        tile's 256 FP4 bytes and 32 scale bytes are gathered in shared memory,
        stored into the expert's operand image and published with the
        tile's epoch flag.
        """
        slot = tile // FC1_MT
        mt = tile - slot * FC1_MT
        mask = s_slot_mask[slot]
        a_gate = s_alpha1[slot]
        s_up = s_scale_c1[slot]
        is_up = (lane % 16) < 8
        rcp6 = ptx.rcp_approx_ftz(Float32(6.0))
        tl = cutlass.const_expr(self.debug & DEBUG_TIMELINE)
        if tidx == 0:
            _stamp(tl, tl_s, TL_EPI_T_PARAMS)
        if cutlass.const_expr(self.debug & DEBUG_FC1):
            for n in cutlass.range_constexpr(NTOK):
                ptx.st_f32(
                    dbg_addr + 4 * (65536 + ((slot * FC1_MT + mt) * 128 + tidx) * NTOK + n), v[n]
                )
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
        if tidx == 0:
            _stamp(tl, tl_s, TL_EPI_T_COMPUTED)
        ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
        cta, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.delay & DELAY_PRODUCER):
            self._delay_spin(DELAY_PRODUCER * 53, Int32(cta), epoch_next)
        if cutlass.const_expr(self.handoff == 1):
            # LL words: payload and epoch in one store, no fence
            base = himg_addr + slot * HBUF_BYTES
            if tidx < 32:
                r = tidx // 4
                p = tidx - r * 4
                w = ptx.lds_v2_u32(hst + r * 32 + p * 8)
                ptx.st_ll2(base + (r * 32 + mt * 8 + 2 * p) * 8, w[0], w[1], epoch_next)
            if (tidx >= 32) & (tidx < 32 + NTOK):
                r = tidx - 32
                w1 = ptx.lds_u32(hst + HSTAGE_SF + r * 4)
                ptx.st_ll1(base + (H_LL_DATA_WORDS + r * FC1_MT + mt) * 8, w1, epoch_next)
            if tidx == 0:
                _stamp(tl, tl_s, TL_EPI_T_STORED)
            # the staging buffer is reused by the next tile
            ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
        else:
            img = himg_addr + slot * HBUF_BYTES
            if tidx < 2 * NTOK:
                # 16-byte chunk c16 = 2*mt + half of row r, swizzled as in sB
                r = tidx // 2
                half = tidx - r * 2
                c16 = 2 * mt + half
                d = ptx.lds_v4_u32(hst + r * 32 + half * 16)
                ptx.st_v4_u32(img + r * 128 + ((c16 ^ r) * 16), d[0], d[1], d[2], d[3])
                ptx.fence_proxy_async_global()
            if (tidx >= 2 * NTOK) & (tidx < 3 * NTOK):
                # the tile's four scale bytes of row r, compact SFB chunk mt
                r = tidx - 2 * NTOK
                w1 = ptx.lds_u32(hst + HSTAGE_SF + r * 4)
                ptx.st_u32(img + HIMG_B_BYTES + mt * SFB_CHUNK_STRIDE + r * 16, w1)
                ptx.fence_proxy_async_global()
            if tidx == 0:
                _stamp(tl, tl_s, TL_EPI_T_STORED)
            # all image stores (and staging reads) precede the flag; the staging
            # buffer is free for the next tile after this barrier
            ptx.bar_sync(BAR_EPI, 32 * EPI_WARPS)
            if tidx == 0:
                ptx.fence_acq_rel_gpu()
                ptx.st_relaxed_gpu_u32(hflag_addr + (slot * FC1_MT + mt) * 4, epoch_next)
