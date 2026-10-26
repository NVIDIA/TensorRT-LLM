# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Dense paged fp8 decode forward path.

This file owns the CUTE DSL entry point for decode attention via
``SparseDecodeAttentionForwardSm100`` — SM100 UTCMMA + persistent
scheduling, paged fp8 Q/K/V, BSA blk128-style intra-warp overlap pipeline.
Forward only.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.pipeline as cutlass_pipeline
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from quack import copy_utils, layout_utils

from src.common import pipeline
from src.common import blackwell_helpers as sm100_helpers
from src.common import mma_sm100_desc as sm100_desc
from src.common.cute_dsl_utils import assume_tensor_aligned, torch2cute_dtype_map
from src.common.named_barrier import NamedBarrierFwdSm100
from src.common.softmax import SoftmaxSm100
from src.common.mask import AttentionMask
from src.common.seqlen_info import SeqlenInfoQK
from src.common.pack_gqa import pack_gqa_layout
from src.common.tile_scheduler import SchedulingMode
from src.sm100.fwd_decode.tile_scheduler import (
    DecodeTileScheduler,
    DecodeTileSchedulerArguments,
)


class SparseDecodeAttentionForwardSm100:
    """SM100 dense paged fp8 decode forward attention (UTCMMA + CLC).

    Scope (Phase 1):
    - Dense decode, ``split_kv=False``, single q-tile per work item
      (``packed_q = seqlen_q * qhead_per_kv <= tile_m=128``).
    - Causal only.  KV reverse page loop; first reverse block applies
      causal/seqlen mask, the rest is unmasked.
    - fp8 Q/K/V, bf16 O, fp32 LSE.  P is quantized to fp8_e4m3fn before PV
      via ``SoftmaxSm100.apply_exp2_convert`` (mirror of prefill fp8 PV).
    - per-batch ``mSeqUsedK[b]`` heterogeneous; no uniform-length assumptions.

    Production scope reached at Phase 4+:
    - Multi q-tile (Phase 2), split-KV partial writeback (Phase 3),
      CLC persistent scheduling (Phase 4), TC SOL >= 90% (Phase 7).
    """

    # UTCMMA K-tile width (matches prefill SparseAttentionForwardSm100).
    k_tile = 64

    def __init__(
        self,
        head_dim: int = 128,
        qhead_per_kv: int = 16,
        m_block_size: int = 128,
        n_block_size: int = 128,
        page_size: int = 128,
        split_kv: bool = False,
        causal: bool = True,
        write_lse: bool = True,
        disable_softmax_exp2: bool = False,
    ):
        # --- structural constraints (Phase 1 scope) -------------------------
        if head_dim != 128:
            raise NotImplementedError(
                f"SparseDecodeAttentionForwardSm100 currently supports only D=128, "
                f"got D={head_dim}"
            )
        if m_block_size != 128:
            raise NotImplementedError(
                f"decode UMMA forward requires tile_m=128, got {m_block_size}"
            )
        if n_block_size != 128:
            raise NotImplementedError(
                f"decode UMMA forward requires n_block_size=128, got {n_block_size}"
            )
        if page_size != n_block_size:
            raise ValueError(
                f"page_size ({page_size}) must equal n_block_size ({n_block_size})"
            )
        if qhead_per_kv not in (16, 8, 4, 2, 1):
            raise ValueError(
                f"qhead_per_kv must be in {{1, 2, 4, 8, 16}}, got {qhead_per_kv}"
            )
        if not causal:
            raise NotImplementedError(
                "decode UMMA forward currently supports only causal=True"
            )

        self.head_dim = int(head_dim)
        self.qhead_per_kv = int(qhead_per_kv)
        self.m_block_size = int(m_block_size)
        self.n_block_size = int(n_block_size)
        self.page_size = int(page_size)
        self.tile_m = int(m_block_size)
        self.split_kv = bool(split_kv)
        self.causal = bool(causal)
        self.write_lse = bool(write_lse)
        self.disable_softmax_exp2 = bool(disable_softmax_exp2)
        # FA fp8 SM100 fwd uses a threshold of 4.0 to avoid rescaling O for
        # small row-max movements; correction receives acc_scale directly.
        self.rescale_threshold = 4.0

        # q tokens packed per (m_block_size) row group along M.
        self.q_tokens_per_group = self.m_block_size // self.qhead_per_kv

        self.mma_tiler_qk = (self.m_block_size, self.n_block_size, self.head_dim)
        self.mma_tiler_pv = (self.m_block_size, self.head_dim, self.n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32

        # --- pipeline ring stages (BSA blk128 q_stage=1, s_stage=2) ---
        self.q_stage = 1
        self.s_stage = 2
        self.o_stage = 2
        # Keep the fp8 decode KV ring deep enough to cover the K0/Q/K1/V0...
        # order.  This matches sage's fp8 setting and removes the underfed
        # two-stage KV pipeline seen in the q8/16K non-split case.
        self.kv_stage = 4
        self.k_stages = 2
        # Match prefill: PV is split at 3/4 of n_block_size for fp8.  The
        # producer (P store) must publish exactly 3N/4 fp8 columns at the
        # signal point; that requires the TMEM-store atom Repetition to be
        # ``8`` (one PV ``f8f6f4`` K=32 segment = 8 fp32 packed cols), so
        # ``shape[2]=4`` chunks and ``split_idx=3`` lands on the 3N/4
        # boundary exactly.  The previous N/2 cap was a workaround for
        # ``Repetition(16)`` whose coarser chunk boundary could not
        # represent 3N/4.
        self.split_P_arrive = self.n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size

        # --- warp layout (16 warps / 512 threads) — BSA-aligned (Phase 1.10.6b)
        # 0-3   softmax WG 0
        # 4-7   softmax WG 1
        # 8-11  correction WG  (acc_O rescale across pages + final epilogue
        #                       write-back; participates in TmemPtr barrier)
        # 12    MMA issue warp
        # 13    spare / future CLC scheduler
        # 14    load warp       (serial Q + K + V TMA loads)
        # 15    empty / register-budget reserve
        self.warps_per_group = 4
        self.softmax0_warp_base = 0
        self.softmax1_warp_base = self.softmax0_warp_base + self.warps_per_group
        self.correction_warp_base = (
            self.softmax1_warp_base + self.warps_per_group)
        self.mma_warp_id = self.correction_warp_base + self.warps_per_group
        self.spare_warp_id = self.mma_warp_id + 1
        self.load_warp_id = self.spare_warp_id + 1
        self.empty_warp_id = self.load_warp_id + 1
        self.total_warps = 16
        self.threads_per_cta = cute.arch.WARP_SIZE * self.total_warps

        # --- TMEM layout (fp8 P width-pack: 4 fp8 lanes per fp32 column) ---
        # S0/S1: [0:128], [128:256]
        # O0/O1: [256:384], [384:512] for head_dim_v=128
        # P (fp8) overlays the second half of each S tile via recast_ptr.
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_s_offset = 0
        self.tmem_stage_stride = self.n_block_size
        self.tmem_o_stage_stride = self.head_dim
        self.tmem_o_offset = self.s_stage * self.n_block_size
        # fp8 P occupies n_block_size * fp8_width / fp32_width = n/4 fp32 cols.
        # P offset is set in __call__ once q_dtype is known (defer to Phase 1.3).
        raw_tmem_total = self.tmem_o_offset + self.o_stage * self.tmem_o_stage_stride
        # SM100 TMEM allocation requires a power-of-two column count.
        self.tmem_total = 1 << (raw_tmem_total - 1).bit_length()

        # --- register budget per role (BSA hdim>=96 default) ---
        self.num_regs_softmax = 184
        self.num_regs_correction = 88
        self.num_regs_other = 56
        self.num_regs_mma = self.num_regs_other
        self.num_regs_load = self.num_regs_other
        self.num_regs_epilogue = self.num_regs_other
        self.num_regs_empty = self.num_regs_other

        # exp2 emulation for causal: matches prefill ex2_emu_freq=16.
        # disable_softmax_exp2 (Phase 7 SOL gate) bypasses both emulation and
        # native exp2 — the convert pass becomes a pure fp32 -> fp8 cast.
        self.ex2_emu_freq = 16 if (self.causal and not self.disable_softmax_exp2) else 0
        self.ex2_emu_start_frg = 1
        self.buffer_align_bytes = 1024

        # --- SM100 cluster config (single-CTA for decode, no 2-CTA pair) -
        self.use_2cta_instrs = False
        self.cta_group_size = 1
        self.cluster_shape_mn = (1, 1)
        self.cluster_shape_mnk = (1, 1, 1)
        self.use_clc_scheduler = True
        self.scheduling_mode = SchedulingMode.CLC
        self.sched_stages = 2
        self.clc_scheduler_warp_id = self.empty_warp_id

        self.arch = BaseDSL._get_dsl().get_arch_enum()

    # ------------------------------------------------------------------
    # Host-side: TMA descriptors, SMEM layout, launch
    # Phase 1.2+ fills in the body.  Phase 1.1 keeps signatures stable so
    # the rest of the codepath (run_decode_attention dispatch in 1.10)
    # can wire to this class without further churn.
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,                  # [B, Sq, Hq, D] fp8
        mK: cute.Tensor,                  # [num_pages, Hkv, page_size, D] fp8
        mV: cute.Tensor,                  # [num_pages, Hkv, page_size, D] fp8
        mPageTable: cute.Tensor,          # [B, max_pages] int32
        mSeqUsedK: cute.Tensor,           # [B] int32
        mRequestIndices: cute.Tensor,     # [work_capacity] int32
        mQoTileIndices: cute.Tensor,      # [work_capacity] int32
        mKvTileIndices: cute.Tensor,      # [work_capacity] int32
        mBlockValidMask: cute.Tensor,     # [work_capacity] int32
        mSplitCounts: cute.Tensor,        # [B] int32
        mOIndptr: cute.Tensor,            # [B + 1] int32
        mO: cute.Tensor,                  # [total_q, Hq, D] bf16
        mLSE: cute.Tensor,                # [total_q, Hq] fp32
        mO_partial: Optional[cute.Tensor],
        mLSE_partial: Optional[cute.Tensor],
        softmax_scale: Float32,
        seqlen_q: Int32,
        page_size: Int32,
        kv_chunk_size_pages: Int32,
        stream: cuda.CUstream = None,
    ):
        # --- dtype contract ------------------------------------------------
        if const_expr(mQ.element_type is not cutlass.Float8E4M3FN):
            raise TypeError("decode UMMA Q must be Float8E4M3FN")
        if const_expr(mK.element_type is not cutlass.Float8E4M3FN):
            raise TypeError("decode UMMA K must be Float8E4M3FN")
        if const_expr(mV.element_type is not cutlass.Float8E4M3FN):
            raise TypeError("decode UMMA V must be Float8E4M3FN")
        if const_expr(mO.element_type is not cutlass.BFloat16):
            raise TypeError("decode UMMA output O must be BFloat16")
        if const_expr(mLSE.element_type is not Float32):
            raise TypeError("decode UMMA output LSE must be Float32")
        if const_expr(self.split_kv):
            if const_expr(mO_partial is None or mO_partial.element_type is not Float32):
                raise TypeError("decode UMMA split path requires Float32 O_partial")
            if const_expr(mLSE_partial is None or mLSE_partial.element_type is not Float32):
                raise TypeError("decode UMMA split path requires Float32 LSE_partial")

        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = (
            mO_partial.element_type if const_expr(self.split_kv)
            else mO.element_type
        )
        # f8f6f4 MMA descriptor kind for fp8 Q/K/V.
        self.mma_kind = "f8f6f4"
        # fp8 P width-pack ratio: each fp32 TMEM column holds 4 fp8 P lanes.
        # Computed here so __init__ stays dtype-agnostic and the TMEM offsets
        # can later be derived from this ratio in Phase 1.3.
        elem_bytes = const_expr(self.q_dtype.width // 8)
        p_cols_as_fp32 = const_expr(
            self.n_block_size * self.q_dtype.width // Float32.width
        )
        self.tmem_s_to_p_offset = self.n_block_size - p_cols_as_fp32
        self.tmem_p_offset = self.tmem_s_offset + self.tmem_s_to_p_offset

        mQ, mK, mV, mO, mLSE = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mO, mLSE)
        ]
        if const_expr(mO_partial is not None):
            mO_partial = assume_tensor_aligned(mO_partial)
        if const_expr(mLSE_partial is not None):
            mLSE_partial = assume_tensor_aligned(mLSE_partial)
        mO_epilogue = mO_partial if const_expr(self.split_kv) else mO
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO_epilogue)
        self.epi_tile = (self.m_block_size, self.head_dim)

        # ------------------------------------------------------------------
        # UTCMMA TiledMma: QK^T + PV.  PV uses MN-major V operand (V already
        # transposed in the layout below) and a TMEM operand source for P.
        # Phase 1.4 builds tiled_mma_qk; Phase 1.5 adds tiled_mma_pv so sV
        # layout can derive the MN-major swizzle.
        # ------------------------------------------------------------------
        cta_group = tcgen05.CtaGroup.ONE
        tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            Float32, cta_group, self.mma_tiler_qk[:2])
        tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
            Float32, cta_group, self.mma_tiler_pv[:2], tcgen05.OperandSource.TMEM)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))

        # ------------------------------------------------------------------
        # Paged K/V tensor view permutation.
        # Input layout [num_pages, Hkv, page_size, D] (nhsd) is permuted to
        # [page_size, D, Hkv, num_pages] for the paged TMA descriptor (K).
        # V gets an additional (s,d) swap to become MN-major:
        # [D, page_size, Hkv, num_pages].
        # ------------------------------------------------------------------
        mK_paged = cute.make_tensor(
            mK.iterator, cute.select(mK.layout, mode=[2, 3, 1, 0])
        )
        mV_kv = cute.make_tensor(
            mV.iterator, cute.select(mV.layout, mode=[2, 3, 1, 0])
        )
        mV_paged = cute.make_tensor(
            mV_kv.iterator, cute.select(mV_kv.layout, mode=[1, 0, 2, 3])
        )

        # ------------------------------------------------------------------
        # Q SMEM layout + BSA/FA PackGQA full-tile TMA atom.
        #
        # Runtime Q is [B, Sq, Hq, D].  We transpose to [Sq, D, Hq, B], then
        # fold qhead_per_kv into the M dimension:
        #   ((qhead_per_kv, Sq), D, Hkv, B)
        # This lets one Q TMA load cover the whole packed (tile_m, D) tile
        # instead of issuing one TMA per q token.
        # ------------------------------------------------------------------
        total_q_stages = self.q_stage
        sQ_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, total_q_stages)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        mQ = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=[1, 3, 2, 0]))
        nheads_kv = mK.shape[1]
        mQ = pack_gqa_layout(mQ, self.qhead_per_kv, nheads_kv, head_idx=2)
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            cta_layout_vmnk.shape,
        )

        # ------------------------------------------------------------------
        # K / V SMEM layouts + TMA atoms (paged).
        # sK uses the QK MMA operand B swizzle; sV uses the PV MMA operand B
        # swizzle (MN-major).  tP_layout is the TMEM-side P descriptor — no
        # SMEM is actually allocated for P, it overlays the S region in TMEM
        # via cute.recast_ptr in Phase 1.7.
        # ------------------------------------------------------------------
        sK_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage)
        sV_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage)
        tP_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage)

        tma_atom_K, mK_paged = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mK_paged, cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk, tiled_mma_qk, cta_layout_vmnk.shape)
        tma_atom_V, mV_paged = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op, mV_paged, cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv, tiled_mma_pv, cta_layout_vmnk.shape)

        # ------------------------------------------------------------------
        # Phase 1.10.6b-B-2: TMA-store atom for the epilogue write-back.
        # Non-split writes bf16 final O; split-KV writes fp32 O_partial.
        # sO follows FA/BSA epilogue layout: one full m_block x D tile in
        # SMEM.  Both paths expose global O as a packed-GQA tensor view so the
        # final store is a full BSA-style m_block x D TMA tile.
        # ------------------------------------------------------------------
        sO_layout = sm100_utils.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.q_stage,
        )
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        num_heads_kv_tma = mK.shape[1]
        total_o_rows_tma = (
            mO_epilogue.shape[0]
            // (num_heads_kv_tma * self.qhead_per_kv)
        )
        head_stride_tma = self.head_dim
        o_row_stride_tma = (
            num_heads_kv_tma * self.qhead_per_kv * self.head_dim)
        kv_head_stride_tma = self.qhead_per_kv * self.head_dim
        mO_epilogue_tma = cute.make_tensor(
            mO_epilogue.iterator,
            cute.make_layout(
                ((self.qhead_per_kv, total_o_rows_tma), self.head_dim, num_heads_kv_tma),
                stride=((head_stride_tma, o_row_stride_tma), 1, kv_head_stride_tma),
            ),
        )
        tma_atom_O, mO_tma = cpasync.make_tiled_tma_atom(
            tma_store_op,
            mO_epilogue_tma,
            cute.select(sO_layout, mode=[0, 1]),
            self.epi_tile,
        )

        # Pre-multiply softmax scale by log2(e) so the inner exp2 path can
        # operate without re-scaling at every iteration.  Mirrors prefill.
        softmax_scale_log2 = softmax_scale * Float32(math.log2(math.e))

        work_capacity = mRequestIndices.shape[0]
        num_heads_kv = mK.shape[1]
        tile_sched_args = DecodeTileSchedulerArguments(
            Int32(work_capacity),
            Int32(num_heads_kv),
            cluster_shape_mn=self.cluster_shape_mn,
        )
        tile_sched_params = DecodeTileScheduler.to_underlying_arguments(
            tile_sched_args,
            scheduling_mode=self.scheduling_mode,
        )
        self.tile_scheduler_cls = DecodeTileScheduler
        grid = DecodeTileScheduler.get_grid_shape(tile_sched_params)

        # One CLC response record per scheduler stage.  A response is a 16B
        # opaque record that `cute.arch.clc_response` reads back with a
        # 128-bit `universal_copy`, so each slot must be 16B aligned.  cute
        # derives a MemRange's alignment from its element width, so the range
        # is typed Int128 rather than 4 x Int32 (which only guarantees 4B and
        # makes the copy fail the `cute.copy` verifier).
        clc_response_size = self.sched_stages if self.use_clc_scheduler else 0
        clc_mbar_size = self.sched_stages * 2 if self.use_clc_scheduler else 0

        # ------------------------------------------------------------------
        # SharedStorage mirrors BSA blk128's pipeline mesh for dense paged
        # decode: Q, shared K/V, S/P/O, P-lastsplit, O-acc, O-epilogue and
        # softmax stats mbarriers, plus the TMEM allocator state and SMEM
        # staging tensors.
        # ------------------------------------------------------------------
        @cute.struct
        class SharedStorage:
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[
                Int64, self.s_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[
                Int64, self.s_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_softmax_stats0: cute.struct.MemRange[Int64, 2]
            mbar_softmax_stats1: cute.struct.MemRange[Int64, 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.s_stage * 2]
            # Phase 1.10.6b-B-2: bf16 sO SMEM staging buffer for the TMA
            # store epilogue.  Sized for one full m_block_size × head_dim
            # tile (single stage; overlap with sQ left for later perf tune).
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(sO_layout)],
                self.buffer_align_bytes,
            ]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.MemRange[cutlass.Int128, clc_response_size]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(sV_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # ------------------------------------------------------------------
        # Launch — decode tasks are consumed from the
        # (work_idx, head_kv_idx) scheduler space.  In CLC mode grid is the
        # BSA-style hardware problem shape; in static mode it is capped to the
        # SM count and each CTA walks the flattened task stream.
        # ------------------------------------------------------------------
        # q_tma_bytes (and Phase 1.5+: kv_tma_bytes / q_subtile_bytes) are
        # recomputed inside the kernel from the constexpr SMEM layouts.
        # Passing them as Constexpr[int] kernel args ended up marshalling
        # to dynamic Int32 here, which then tripped MbarrierArray's
        # `if tx_count < 0` check inside PipelineTmaUmma.create.
        self.kernel(
            mQ, mK_paged, mV_paged,
            mPageTable, mSeqUsedK,
            mRequestIndices, mQoTileIndices, mKvTileIndices, mBlockValidMask,
            mSplitCounts, mOIndptr,
            mO, mO_tma, mLSE,
            mO_partial, mLSE_partial,
            softmax_scale_log2,
            sQ_layout, sK_layout, sV_layout, tP_layout, sO_layout,
            tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O,
            tiled_mma_qk, tiled_mma_pv,
            tile_sched_params,
            seqlen_q, page_size, kv_chunk_size_pages,
            Int32(num_heads_kv),
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(
                self.cluster_shape_mnk
                if cute.size(self.cluster_shape_mnk) > 1 else None
            ),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        # --- runtime tensors -------------------------------------------------
        mQ: cute.Tensor,                  # [((qhead_per_kv, Sq), D, Hkv, B)]
        mK_paged: cute.Tensor,            # [page_size, D, Hkv, num_pages] fp8
        mV_paged: cute.Tensor,            # [D, page_size, Hkv, num_pages] fp8
        mPageTable: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mRequestIndices: cute.Tensor,
        mQoTileIndices: cute.Tensor,
        mKvTileIndices: cute.Tensor,
        mBlockValidMask: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mOIndptr: cute.Tensor,
        mO: cute.Tensor,
        mO_tma: cute.Tensor,
        mLSE: cute.Tensor,
        mO_partial: Optional[cute.Tensor],
        mLSE_partial: Optional[cute.Tensor],
        # --- scalars ---------------------------------------------------------
        softmax_scale_log2: Float32,
        # --- SMEM layouts ----------------------------------------------------
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        # --- TMA atoms -------------------------------------------------------
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        # --- TiledMma --------------------------------------------------------
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: DecodeTileScheduler.Params,
        # --- Int32 iteration bounds ------------------------------------------
        seqlen_q: Int32,
        page_size: Int32,
        kv_chunk_size_pages: Int32,
        num_heads_kv: Int32,
    ):
        # ------------------------------------------------------------------
        # Thread / warp identity, work-item dispatch.
        # ------------------------------------------------------------------
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        if warp_idx == Int32(0):
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_O)

        # ------------------------------------------------------------------
        # SMEM allocation — same SharedStorage type was registered on the
        # class in __call__ (Phase 1.3).  Every warp materialises the same
        # storage view; later phases populate sQ/sK/sV/mbar contents.
        # ------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        # sQ is the MMA-operand layout and now also the Q TMA load target:
        # PackGQA makes the global Q view match the full BSA (tile_m, D) tile.
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        # ------------------------------------------------------------------
        # TMEM allocator — MMA warp performs the allocation, all softmax /
        # store / MMA warps participate in the TmemPtr named barrier that
        # broadcasts the allocator pointer.  Spare warp and KV-load warps
        # do not touch TMEM directly.
        # ------------------------------------------------------------------
        # TmemPtr participants: 2 softmax WGs (8 warps) + correction WG
        # (4 warps) + MMA warp = 13 warps × WARP_SIZE.  Load / spare /
        # empty warps don't touch TMEM and don't arrive on this barrier.
        tmem_alloc_warps: cutlass.Constexpr[int] = (
            self.warps_per_group * 3 + 1)
        tmem_alloc_threads = cute.arch.WARP_SIZE * tmem_alloc_warps
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=tmem_alloc_threads,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
        )
        tmem_cols = self.tmem_total

        # ------------------------------------------------------------------
        # Cluster layout + warp-specialized pipelines.
        # Mirrors prefill (src/sm100/fwd/atten_fwd.py:617-683): cta_layout_vmnk
        # is rebuilt in-kernel from tiled_mma_qk.thr_id.shape so its size is
        # constexpr (the `cute.size(cta_layout_vmnk) == 1` check inside
        # PipelineTmaUmma.create folds at compile time).  pipeline_q is
        # joined by the BSA S/P/O and shared K/V pipelines below.
        # ------------------------------------------------------------------
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,))

        ThreadCooperativeGroup = partial(
            cutlass_pipeline.CooperativeGroup, cutlass_pipeline.Agent.Thread)
        tma_thread = ThreadCooperativeGroup(1)
        mma_thread = ThreadCooperativeGroup(1)
        # One softmax WG participates per S/P/O stage; correction and the
        # epilogue warp handle O rescale and TMA write-back.
        softmax_warps = ThreadCooperativeGroup(self.warps_per_group)
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * self.warps_per_group)

        # Recompute TMA byte counts inside the kernel from the constexpr SMEM
        # layouts — see note in __call__ above the self.kernel(...) call for
        # why these can't be plumbed through as Constexpr[int] kernel args.
        q_tma_bytes = cute.size_in_bytes(
            self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2]))
        k_tma_bytes = cute.size_in_bytes(
            self.k_dtype, cute.select(sK_layout, mode=[0, 1, 2]))

        pipeline_q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_thread,
            consumer_group=mma_thread,
            tx_count=q_tma_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # Decode KV follows BSA's single K/V ring: K0 is primed before Q,
        # then K1, V0, K2, V1, ... share one PipelineTmaUmma state while
        # landing in separate sK/sV SMEM tensors.  For fp8 decode K/V TMA
        # tiles have the same byte count, so the shared barrier uses K's count.
        pipeline_kv = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=tma_thread,
            consumer_group=mma_thread,
            tx_count=k_tma_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # ------------------------------------------------------------------
        # BSA pipeline mesh.
        #   pipeline_s_p_o   — MMA→{softmax,correction} (8-warp cluster
        #                       consumer).  MMA producer_commit signals
        #                       "S ready"; consumer_release signals "P stored
        #                       and acc_O rescaled — MMA can issue next QK".
        #   pipeline_o_acc   — MMA→correction (acc_O updated by PV).
        #   pipeline_sm_stats0/1 — softmax→correction stage-local stats.
        #                       This avoids the per-warp NamedBarrier used by
        #                       the BSA reference while preserving the same
        #                       first/rescale/final signal sequence.
        #   pipeline_o_epi   — correction→epilogue warp 13 (final O ready).
        # ------------------------------------------------------------------
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE
            * (self.warps_per_group + self.warps_per_group)  # = 256
        )
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * self.warps_per_group  # = 128
        )
        epilogue_warp_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE  # warp 13 = 32 threads
        )

        pipeline_s_p_o = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_thread,
            consumer_group=softmax_correction_threads,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_warps,
            consumer_group=mma_thread,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_o_acc = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_thread,
            consumer_group=correction_threads,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_sm_stats0 = pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats0.data_ptr(),
            num_stages=1,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        pipeline_sm_stats1 = pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats1.data_ptr(),
            num_stages=1,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        pipeline_o_epi = pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_O_epi.data_ptr(),
            num_stages=self.s_stage,
            producer_group=correction_threads,
            consumer_group=epilogue_warp_threads,
            defer_sync=True,
        )

        # Fence mbar init across all regular pipelines.  CLC pipeline setup
        # follows the BSA ordering: arrive after mbar init, create scheduler
        # state, then wait before TMEM allocation and role dispatch.
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        if const_expr(self.use_clc_scheduler):
            clc_response_ptr = storage.clc_response.data_ptr()
            clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
            clc_pipeline_producer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread
            )
            num_clc_consumer_warps = (
                self.threads_per_cta // cute.arch.WARP_SIZE
            ) * self.cta_group_size
            clc_pipeline_consumer_group = cutlass_pipeline.CooperativeGroup(
                cutlass_pipeline.Agent.Thread,
                cute.arch.WARP_SIZE * num_clc_consumer_warps,
            )
            clc_pipeline = cutlass_pipeline.PipelineClcFetchAsync.create(
                barrier_storage=clc_mbar_ptr,
                num_stages=self.sched_stages,
                producer_group=clc_pipeline_producer_group,
                consumer_group=clc_pipeline_consumer_group,
                tx_count=16,
                cta_layout_vmnk=cta_layout_vmnk,
            )
            tile_scheduler = self.tile_scheduler_cls.create(
                tile_sched_params, clc_response_ptr=clc_response_ptr
            )
            clc_consumer_state = cutlass_pipeline.make_pipeline_state(
                cutlass_pipeline.PipelineUserType.Consumer,
                self.sched_stages,
            )
            tile_scheduler.set_clc_pipeline(
                clc_pipeline, clc_consumer_state)
        else:
            clc_pipeline = None
            tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)

        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # Single load warp issues Q + K + V TMA serially; no inter-warp
        # broadcast / Q-load WG barrier needed (the BSA-aligned layout
        # collapses the previous 4-warp Q-load fan-out into one warp).

        # ------------------------------------------------------------------
        # Phase 1.10.3: pre-dispatch TMEM partitions for softmax read/write.
        # Mirrors prefill softmax body setup
        # (src/sm100/fwd/atten_fwd.py:807-829, 1891-1921).  Built once across
        # all warps so each softmax WG can take its stage slice.
        # ------------------------------------------------------------------
        thr_mma_qk_pre = tiled_mma_qk.get_slice(0)
        qk_acc_shape_pre = thr_mma_qk_pre.partition_shape_C(
            self.mma_tiler_qk[:2])
        tStS_base_pre = thr_mma_qk_pre.make_fragment_C(qk_acc_shape_pre)
        tStS_pre = cute.make_tensor(
            tStS_base_pre.iterator,
            cute.append(
                tStS_base_pre.layout,
                cute.make_layout(
                    (self.s_stage,), stride=(self.tmem_stage_stride,)),
            ),
        )
        tScS_pre = thr_mma_qk_pre.partition_C(
            cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS_pre = tScS_pre[(None, None), 0, 0]
        # fp8 P occupies n_block_size * fp8_width / fp32_width fp32 cols.
        tilePlikeFP32 = const_expr(
            self.mma_tiler_qk[1] * self.q_dtype.width // Float32.width)
        tmem_load_atom_pre = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        # Repetition(8) gives ``tStP_r2t.shape[2] = tilePlikeFP32 / 8 = 4``
        # chunks for fp8 (tilePlikeFP32=32), with each chunk publishing
        # 8 fp32 cols = 32 fp8 cols = exactly one PV ``f8f6f4`` K=32
        # segment.  ``split_idx = 4 * 3N/4 / N = 3`` aligns the early
        # publish edge to the producer/consumer K boundary.  Larger
        # Repetition (e.g. 16) would coarsen shape[2] to 2 and force
        # split_idx to floor to 1, publishing only N/2 of P before MMA's
        # first three K=32 segments need cols 0..3N/4 — that mismatch is
        # the NaN source the workaround used to dodge with split=N/2.
        tmem_store_atom_pre = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)),
            Float32,
        )
        tmem_store_vec_atom_pre = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tmem_load_vec_atom_pre = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )

        # ------------------------------------------------------------------
        # Warp role dispatch.  Bodies are filled in Phase 1.3-1.9:
        #   softmax WG 0/1 (warps 0-3, 4-7) — softmax + P fp32->fp8 convert
        #   store / Q-load WG  (warps 8-11) — Q TMA gather + epilogue store
        #   MMA warp           (warp 12)    — UTCMMA QK + PV issue
        #   correction WG     (warps 8-11) — per-page acc_O rescale + epilogue
        #   MMA warp           (warp 12)    — UTCMMA QK + PV issue
        #   spare warp         (warp 13)    — empty / future CLC scheduler
        #   load warp          (warp 14)    — serial Q + K + V TMA loads
        #   empty warp         (warp 15)    — register-budget reserve
        # ------------------------------------------------------------------
        is_softmax0_warp = (
            warp_idx >= Int32(self.softmax0_warp_base)
            and warp_idx < Int32(self.softmax1_warp_base)
        )
        is_softmax1_warp = (
            warp_idx >= Int32(self.softmax1_warp_base)
            and warp_idx < Int32(self.correction_warp_base)
        )
        is_correction_warp = (
            warp_idx >= Int32(self.correction_warp_base)
            and warp_idx < Int32(self.mma_warp_id)
        )
        is_mma_warp = warp_idx == Int32(self.mma_warp_id)
        is_spare_warp = warp_idx == Int32(self.spare_warp_id)
        is_load_warp = warp_idx == Int32(self.load_warp_id)
        is_empty_warp = warp_idx == Int32(self.empty_warp_id)

        if const_expr(self.use_clc_scheduler):
            if warp_idx == Int32(self.clc_scheduler_warp_id):
                cute.arch.setmaxregister_decrease(self.num_regs_empty)
                self.clc_scheduler_warp(clc_pipeline, tile_scheduler)
                is_empty_warp = False

        if is_softmax0_warp:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr_wg0 = tmem.retrieve_ptr(self.qk_acc_dtype)
            _ = tmem_ptr_wg0
            self.softmax_loop(
                0,
                self.softmax0_warp_base,
                softmax_scale_log2,
                tStS_pre,
                tScS_pre,
                tilePlikeFP32,
                tmem_load_atom_pre,
                tmem_store_atom_pre,
                tmem_store_vec_atom_pre,
                thr_mma_qk_pre,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_sm_stats0,
                mRequestIndices,
                mQoTileIndices,
                mKvTileIndices,
                mSeqUsedK,
                mBlockValidMask,
                tile_scheduler,
                seqlen_q,
                page_size,
                kv_chunk_size_pages,
            )
            tmem_alloc_barrier.arrive()

        if is_softmax1_warp:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr_wg1 = tmem.retrieve_ptr(self.qk_acc_dtype)
            _ = tmem_ptr_wg1
            self.softmax_loop(
                1,
                self.softmax1_warp_base,
                softmax_scale_log2,
                tStS_pre,
                tScS_pre,
                tilePlikeFP32,
                tmem_load_atom_pre,
                tmem_store_atom_pre,
                tmem_store_vec_atom_pre,
                thr_mma_qk_pre,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_sm_stats1,
                mRequestIndices,
                mQoTileIndices,
                mKvTileIndices,
                mSeqUsedK,
                mBlockValidMask,
                tile_scheduler,
                seqlen_q,
                page_size,
                kv_chunk_size_pages,
            )
            tmem_alloc_barrier.arrive()

        if is_correction_warp:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # Participate in TmemPtr handshake so the MMA warp can free.
            tmem.wait_for_alloc()
            tmem_ptr_corr = tmem.retrieve_ptr(self.qk_acc_dtype)
            _ = tmem_ptr_corr

            self.correction_loop(
                tiled_mma_pv,
                tStS_pre,
                tScS_pre,
                tmem_load_vec_atom_pre,
                pipeline_s_p_o,
                pipeline_sm_stats0,
                pipeline_sm_stats1,
                pipeline_o_acc,
                pipeline_o_epi,
                sO,
                mRequestIndices,
                mQoTileIndices,
                mKvTileIndices,
                mSeqUsedK,
                mSplitCounts,
                mOIndptr,
                mLSE,
                mLSE_partial,
                mBlockValidMask,
                tile_scheduler,
                seqlen_q,
                page_size,
                kv_chunk_size_pages,
                num_heads_kv,
                softmax_scale_log2,
            )
            tmem_alloc_barrier.arrive()

        if is_spare_warp:
            cute.arch.setmaxregister_decrease(self.num_regs_epilogue)
            self.epilogue_s2g(
                mO_tma,
                sO,
                tma_atom_O,
                pipeline_o_epi,
                mRequestIndices,
                mQoTileIndices,
                mKvTileIndices,
                mOIndptr,
                mBlockValidMask,
                tile_scheduler,
                seqlen_q,
            )

        if is_load_warp:
            self.load(
                tiled_mma_qk,
                tiled_mma_pv,
                mQ,
                mK_paged,
                mV_paged,
                sQ,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                mRequestIndices,
                mQoTileIndices,
                mKvTileIndices,
                mSeqUsedK,
                mBlockValidMask,
                tile_scheduler,
                page_size,
                kv_chunk_size_pages,
            )

        if is_empty_warp:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        if is_mma_warp:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            # ----------------------------------------------------------------
            # MMA warp — Phase 1.6: QK fp8×fp8→fp32 UMMA.  Phase 1.10.1 now
            # wraps the body in the real TMEM allocator lifecycle:
            #   tmem.allocate(cols) -> wait_for_alloc -> retrieve_ptr
            #     -> ... QK work ...
            #   -> relinquish_alloc_permit -> tmem_alloc_barrier.arrive_and_wait
            #   -> free(ptr, cols)
            # Softmax WG 0/1 participate via wait_for_alloc + retrieve_ptr +
            # tmem_alloc_barrier.arrive (4+4+1 = 9 warps).
            # ----------------------------------------------------------------
            tmem.allocate(tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            _ = tmem_ptr  # consumed by gemm_pv via raw TMEM offsets

            self.mma(
                sQ,
                sK,
                sV,
                tP_layout,
                tiled_mma_qk,
                tiled_mma_pv,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                mRequestIndices,
                mKvTileIndices,
                mSeqUsedK,
                mBlockValidMask,
                tile_scheduler,
                page_size,
                kv_chunk_size_pages,
            )

            # Phase 1.10.1: TMEM allocator teardown.
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr, num_columns=tmem_cols)

    @cute.jit
    def clc_scheduler_warp(
        self,
        clc_pipeline: cutlass_pipeline.PipelineClcFetchAsync,
        tile_scheduler: DecodeTileScheduler,
    ) -> None:
        clc_producer_state = cutlass_pipeline.make_pipeline_state(
            cutlass_pipeline.PipelineUserType.Producer,
            self.sched_stages,
        )
        clc_consumer_state = cutlass_pipeline.make_pipeline_state(
            cutlass_pipeline.PipelineUserType.Consumer,
            self.sched_stages,
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            clc_pipeline.producer_acquire(clc_producer_state)
            mbarrier_addr = clc_pipeline.producer_get_barrier(
                clc_producer_state)
            tile_scheduler.advance_to_next_work(
                mbarrier_addr=mbarrier_addr,
                response_stage=clc_producer_state.index,
            )
            clc_producer_state.advance()

            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_scheduler.get_current_work(
                response_stage=clc_consumer_state.index)
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
        clc_pipeline.producer_tail(clc_producer_state)

    @cute.jit
    def correction_loop(
        self,
        tiled_mma_pv: cute.TiledMma,
        tStS_pre: cute.Tensor,
        tScS_pre: cute.Tensor,
        tmem_load_vec_atom_pre: cute.CopyAtom,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_sm_stats0: pipeline.PipelineAsync,
        pipeline_sm_stats1: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_o_epi: pipeline.PipelineAsync,
        sO: cute.Tensor,
        mRequestIndices: cute.Tensor,
        mQoTileIndices: cute.Tensor,
        mKvTileIndices: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mOIndptr: cute.Tensor,
        mLSE: cute.Tensor,
        mLSE_partial: Optional[cute.Tensor],
        mBlockValidMask: cute.Tensor,
        tile_scheduler: DecodeTileScheduler,
        seqlen_q: Int32,
        page_size: Int32,
        kv_chunk_size_pages: Int32,
        num_heads_kv: Int32,
        softmax_scale_log2: Float32,
    ) -> None:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        warp_idx_in_wg_corr = warp_idx - Int32(self.correction_warp_base)
        group_tidx_corr = (
            warp_idx_in_wg_corr * Int32(cute.arch.WARP_SIZE)
            + tidx % Int32(cute.arch.WARP_SIZE)
        )

        # First iter: no correction is required. Notify MMA that the
        # initial O slots are available, matching BSA's correction_loop.
        for stage_init in cutlass.range_constexpr(self.s_stage):
            pipeline_s_p_o.consumer_release_w_index(Int32(stage_init))

        o_corr_consumer_phase = Int32(0)
        sm_stats0_consumer_phase = Int32(0)
        sm_stats1_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        thr0_rs = tiled_mma_pv.get_slice(0)
        pv_acc_shape_rs_c = thr0_rs.partition_shape_C(
            self.mma_tiler_pv[:2])
        tOtO_base_rs_c = thr0_rs.make_fragment_C(pv_acc_shape_rs_c)
        tOtO_rs_c = cute.make_tensor(
            tOtO_base_rs_c.iterator + Int32(self.tmem_o_offset),
            cute.append(
                tOtO_base_rs_c.layout,
                cute.make_layout(
                    (self.s_stage,),
                    stride=(self.tmem_o_stage_stride,),
                ),
            ),
        )
        tScS_vec_layout_corr = cute.composition(
            tScS_pre.layout, cute.make_layout((self.m_block_size, 2)))
        tScS_vec_corr = cute.make_tensor(
            tScS_pre.iterator, tScS_vec_layout_corr)
        tSAcc_corr0 = tStS_pre[(None, None), 0, 0, 0]
        tSAcc_corr1 = tStS_pre[(None, None), 0, 0, 1]
        tStS_vec0_layout_corr = cute.composition(
            tSAcc_corr0.layout, cute.make_layout((self.m_block_size, 2)))
        tStS_vec1_layout_corr = cute.composition(
            tSAcc_corr1.layout, cute.make_layout((self.m_block_size, 2)))
        tStStats0_t2r_src = cute.make_tensor(
            tSAcc_corr0.iterator, tStS_vec0_layout_corr)
        tStStats1_t2r_src = cute.make_tensor(
            tSAcc_corr1.iterator, tStS_vec1_layout_corr)
        thr_tmem_load_vec = tcgen05.make_tmem_copy(
            tmem_load_vec_atom_pre,
            tStStats0_t2r_src,
        ).get_slice(group_tidx_corr)
        tStStats0_t2r = thr_tmem_load_vec.partition_S(tStStats0_t2r_src)
        tStStats1_t2r = thr_tmem_load_vec.partition_S(tStStats1_t2r_src)
        tScStats_t2r = thr_tmem_load_vec.partition_D(tScS_vec_corr)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_idx, head_kv_idx, _, _ = work_tile.tile_idx
            if mBlockValidMask[work_idx] != Int32(0):
                batch_idx_corr = mRequestIndices[work_idx]
                qo_tile_corr = mQoTileIndices[work_idx]
                seqused_k_corr = mSeqUsedK[batch_idx_corr]
                split_idx_corr = mKvTileIndices[work_idx]
                kv_pages_corr = (
                    seqused_k_corr + page_size - Int32(1)) // page_size
                kv_page_begin_corr = split_idx_corr * kv_chunk_size_pages
                kv_page_end_corr = cutlass.min(
                    kv_pages_corr,
                    kv_page_begin_corr + kv_chunk_size_pages,
                )
                page_count_corr = kv_page_end_corr - kv_page_begin_corr
                block_iter_count_corr = (
                    page_count_corr + Int32(1)) & ~Int32(1)
                stage0_count_corr = block_iter_count_corr // Int32(2)
                stage1_count_corr = block_iter_count_corr // Int32(2)

                if stage0_count_corr > Int32(0):
                    pipeline_sm_stats0.consumer_wait_w_index_phase(
                        Int32(0), sm_stats0_consumer_phase)
                    sm_stats0_consumer_phase = (
                        sm_stats0_consumer_phase ^ Int32(1))
                    pipeline_sm_stats0.consumer_release_w_index(Int32(0))
                if stage1_count_corr > Int32(0):
                    pipeline_sm_stats1.consumer_wait_w_index_phase(
                        Int32(0), sm_stats1_consumer_phase)
                    sm_stats1_consumer_phase = (
                        sm_stats1_consumer_phase ^ Int32(1))
                    pipeline_sm_stats1.consumer_release_w_index(Int32(0))

                for page_rel_corr in cutlass.range(
                    Int32(self.s_stage), block_iter_count_corr, unroll=1
                ):
                    # sm_stats[0] now holds the deferred-exp2 log2-delta:
                    # 0.0 means "no rescale needed", a negative value is the
                    # raw delta that needs exp2 to become a true scale factor.
                    if (page_rel_corr & Int32(1)) == Int32(0):
                        pipeline_sm_stats0.consumer_wait_w_index_phase(
                            Int32(0), sm_stats0_consumer_phase)
                        sm_stats0_consumer_phase = (
                            sm_stats0_consumer_phase ^ Int32(1))
                        tSrStats = cute.make_rmem_tensor(
                            tScStats_t2r.shape, self.qk_acc_dtype)
                        cute.copy(
                            thr_tmem_load_vec, tStStats0_t2r, tSrStats)
                        cute.arch.fence_view_async_tmem_load()
                        scale_corr_log2 = tSrStats[0]
                        pipeline_sm_stats0.consumer_release_w_index(Int32(0))
                        should_rescale = (
                            cute.arch.vote_ballot_sync(
                                scale_corr_log2 < Float32(0.0)) != 0
                        )
                        if should_rescale:
                            scale_corr = cute.math.exp2(
                                scale_corr_log2, fastmath=True)
                            self.correction_rescale(
                                tiled_mma_pv,
                                tOtO_rs_c[None, None, None, 0],
                                group_tidx_corr,
                                scale_corr,
                            )
                        pipeline_s_p_o.consumer_release_w_index(Int32(0))
                    else:
                        pipeline_sm_stats1.consumer_wait_w_index_phase(
                            Int32(0), sm_stats1_consumer_phase)
                        sm_stats1_consumer_phase = (
                            sm_stats1_consumer_phase ^ Int32(1))
                        tSrStats = cute.make_rmem_tensor(
                            tScStats_t2r.shape, self.qk_acc_dtype)
                        cute.copy(
                            thr_tmem_load_vec, tStStats1_t2r, tSrStats)
                        cute.arch.fence_view_async_tmem_load()
                        scale_corr_log2 = tSrStats[0]
                        pipeline_sm_stats1.consumer_release_w_index(Int32(0))
                        should_rescale = (
                            cute.arch.vote_ballot_sync(
                                scale_corr_log2 < Float32(0.0)) != 0
                        )
                        if should_rescale:
                            scale_corr = cute.math.exp2(
                                scale_corr_log2, fastmath=True)
                            self.correction_rescale(
                                tiled_mma_pv,
                                tOtO_rs_c[None, None, None, 1],
                                group_tidx_corr,
                                scale_corr,
                            )
                        pipeline_s_p_o.consumer_release_w_index(Int32(1))

                for stage_wait in cutlass.range_constexpr(self.s_stage):
                    stage_count_wait = (
                        stage0_count_corr
                        if const_expr(stage_wait == 0)
                        else stage1_count_corr
                    )
                    if stage_count_wait > Int32(0):
                        pipeline_o_acc.consumer_wait_w_index_phase(
                            Int32(stage_wait), o_corr_consumer_phase)

                row_sum0 = Float32(0.0)
                row_sum1 = Float32(0.0)
                row_max0 = -Float32.inf
                row_max1 = -Float32.inf
                for stage_final in cutlass.range_constexpr(self.s_stage):
                    if const_expr(stage_final == 0):
                        pipeline_sm_stats0.consumer_wait_w_index_phase(
                            Int32(0), sm_stats0_consumer_phase)
                        sm_stats0_consumer_phase = (
                            sm_stats0_consumer_phase ^ Int32(1))
                        tSrStats = cute.make_rmem_tensor(
                            tScStats_t2r.shape, self.qk_acc_dtype)
                        cute.copy(
                            thr_tmem_load_vec, tStStats0_t2r, tSrStats)
                        cute.arch.fence_view_async_tmem_load()
                        row_sum0 = tSrStats[0]
                        row_max0 = tSrStats[1]
                        pipeline_sm_stats0.consumer_release_w_index(Int32(0))
                    else:
                        pipeline_sm_stats1.consumer_wait_w_index_phase(
                            Int32(0), sm_stats1_consumer_phase)
                        sm_stats1_consumer_phase = (
                            sm_stats1_consumer_phase ^ Int32(1))
                        tSrStats = cute.make_rmem_tensor(
                            tScStats_t2r.shape, self.qk_acc_dtype)
                        cute.copy(
                            thr_tmem_load_vec, tStStats1_t2r, tSrStats)
                        cute.arch.fence_view_async_tmem_load()
                        row_sum1 = tSrStats[0]
                        row_max1 = tSrStats[1]
                        pipeline_sm_stats1.consumer_release_w_index(Int32(0))

                zero0 = row_sum0 == Float32(0.0) or row_sum0 != row_sum0
                zero1 = row_sum1 == Float32(0.0) or row_sum1 != row_sum1
                rm0 = -Float32.inf if zero0 else row_max0
                rm1 = -Float32.inf if zero1 else row_max1
                row_max_comb = cutlass.max(rm0, rm1)
                row_max_safe = (
                    Float32(0.0)
                    if row_max_comb == -Float32.inf
                    else row_max_comb
                )
                scale0 = (
                    Float32(0.0)
                    if zero0
                    else cute.math.exp2(
                        (rm0 - row_max_safe) * softmax_scale_log2,
                        fastmath=True,
                    )
                )
                scale1 = (
                    Float32(0.0)
                    if zero1
                    else cute.math.exp2(
                        (rm1 - row_max_safe) * softmax_scale_log2,
                        fastmath=True,
                    )
                )
                row_sum_comb = row_sum0 * scale0 + row_sum1 * scale1
                combined_zero_or_nan = (
                    row_sum_comb == Float32(0.0)
                    or row_sum_comb != row_sum_comb
                )
                inv_sum = cute.arch.rcp_approx(
                    Float32(1.0)
                    if combined_zero_or_nan else row_sum_comb)
                final_scale0 = scale0 * inv_sum
                final_scale1 = scale1 * inv_sum

                pipeline_o_epi.producer_acquire_w_index_phase(
                    Int32(0), corr_epi_producer_phase)
                self.correction_epilogue_combine(
                    tiled_mma_pv,
                    sO[None, None, 0],
                    group_tidx_corr,
                    final_scale0,
                    final_scale1,
                )

                if const_expr(self.write_lse or self.split_kv):
                    if group_tidx_corr < Int32(self.m_block_size):
                        is_bad_lse = (
                            row_sum_comb == Float32(0.0)
                            or row_sum_comb != row_sum_comb
                        )
                        LN2 = Float32(math.log(2.0))
                        lse_val = (
                            -Float32.inf if is_bad_lse
                            else (
                                row_max_safe * softmax_scale_log2
                                + cute.math.log2(row_sum_comb, fastmath=True)
                            ) * LN2
                        )
                        tok_lse = group_tidx_corr // Int32(self.qhead_per_kv)
                        if tok_lse < seqlen_q:
                            h_in_kv_lse = (
                                group_tidx_corr
                                - tok_lse * Int32(self.qhead_per_kv))
                            q_idx_lse = (
                                qo_tile_corr * Int32(self.q_tokens_per_group)
                                + tok_lse
                            )
                            h_abs_lse = (
                                head_kv_idx * Int32(self.qhead_per_kv)
                                + h_in_kv_lse
                            )
                            if const_expr(self.split_kv):
                                q_tokens_per_group = Int32(
                                    self.q_tokens_per_group)
                                q_stride_partial = (
                                    (seqlen_q + q_tokens_per_group - Int32(1))
                                    // q_tokens_per_group
                                ) * q_tokens_per_group
                                partial_row_lse = (
                                    mOIndptr[batch_idx_corr]
                                    + split_idx_corr * q_stride_partial
                                    + q_idx_lse
                                )
                                mLSE_partial[
                                    partial_row_lse, h_abs_lse] = lse_val
                            else:
                                q_abs_lse = (
                                    batch_idx_corr * seqlen_q + q_idx_lse)
                                mLSE[q_abs_lse, h_abs_lse] = lse_val

                for stage_release in cutlass.range_constexpr(self.s_stage):
                    stage_count_release = (
                        stage0_count_corr
                        if const_expr(stage_release == 0)
                        else stage1_count_corr
                    )
                    if stage_count_release > Int32(0):
                        pipeline_s_p_o.consumer_release_w_index(
                            Int32(stage_release))
                        pipeline_o_acc.consumer_release_w_index(
                            Int32(stage_release))
                if block_iter_count_corr > Int32(0):
                    o_corr_consumer_phase = (
                        o_corr_consumer_phase ^ Int32(1))

                pipeline_o_epi.producer_commit_w_index(Int32(0))
                corr_epi_producer_phase = (
                    corr_epi_producer_phase ^ Int32(1))

            work_tile = tile_scheduler.consumer_advance()

        pipeline_o_epi.producer_acquire_w_index_phase(
            Int32(self.q_stage - 1), corr_epi_producer_phase)

    @cute.jit
    def epilogue_s2g(
        self,
        mO_tma: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        pipeline_o_epi: pipeline.PipelineAsync,
        mRequestIndices: cute.Tensor,
        mQoTileIndices: cute.Tensor,
        mKvTileIndices: cute.Tensor,
        mOIndptr: cute.Tensor,
        mBlockValidMask: cute.Tensor,
        tile_scheduler: DecodeTileScheduler,
        seqlen_q: Int32,
    ) -> None:
        epi_consumer_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_idx, head_kv_idx, _, _ = work_tile.tile_idx
            if mBlockValidMask[work_idx] != Int32(0):
                batch_idx = mRequestIndices[work_idx]
                qo_tile = mQoTileIndices[work_idx]
                split_idx = mKvTileIndices[work_idx]

                pipeline_o_epi.consumer_wait_w_index_phase(
                    Int32(0), epi_consumer_phase)
                q_tokens_per_group = Int32(self.q_tokens_per_group)
                gO = cute.local_tile(
                    mO_tma[None, None, head_kv_idx],
                    self.epi_tile,
                    (None, 0),
                )
                store_O, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_O, 0, cute.make_layout(1), sO, gO)
                if const_expr(not self.split_kv):
                    q_abs = (
                        batch_idx * seqlen_q
                        + qo_tile * q_tokens_per_group
                    )
                    dst_idx = q_abs // q_tokens_per_group
                else:
                    q_stride_partial = (
                        (seqlen_q + q_tokens_per_group - Int32(1))
                        // q_tokens_per_group
                    ) * q_tokens_per_group
                    partial_row = (
                        mOIndptr[batch_idx]
                        + split_idx * q_stride_partial
                        + qo_tile * q_tokens_per_group
                    )
                    dst_idx = partial_row // q_tokens_per_group
                store_O(src_idx=Int32(0), dst_idx=dst_idx)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
                pipeline_o_epi.consumer_release_w_index(Int32(0))
                epi_consumer_phase = epi_consumer_phase ^ Int32(1)

            work_tile = tile_scheduler.consumer_advance()

    @cute.jit
    def correction_epilogue_combine(
        self,
        tiled_mma_pv: cute.TiledMma,
        sO: cute.Tensor,
        tidx: Int32,
        scale0: Float32,
        scale1: Float32,
    ) -> None:
        thr_mma = tiled_mma_pv.get_slice(0)
        pv_acc_shape = thr_mma.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO_base = thr_mma.make_fragment_C(pv_acc_shape)
        tOtO = cute.make_tensor(
            tOtO_base.iterator + Int32(self.tmem_o_offset),
            cute.append(
                tOtO_base.layout,
                cute.make_layout(
                    (self.s_stage,),
                    stride=(self.tmem_o_stage_stride,),
                ),
            ),
        )
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO_full = thr_mma.partition_C(
            cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size: cutlass.Constexpr[int] = (
            8 * 32 // self.o_dtype.width
        )
        tOsO_i = cute.logical_divide(
            tOsO,
            cute.make_layout((self.m_block_size, corr_tile_size)),
        )
        tOcO_i = cute.logical_divide(
            tOcO_full,
            cute.make_layout((self.m_block_size, corr_tile_size)),
        )
        tOtO0_i = cute.logical_divide(
            tOtO[None, None, None, 0],
            cute.make_layout((self.m_block_size, corr_tile_size)),
        )
        tOtO1_i = cute.logical_divide(
            tOtO[None, None, None, 1],
            cute.make_layout((self.m_block_size, corr_tile_size)),
        )
        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_load_atom = sm100_utils.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_load_atom, tOtO0_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load)
        tiled_smem_store = cute.make_tiled_copy_D(
            smem_copy_atom, tiled_tmem_load)
        tOtO0_t2r = thr_tmem_load.partition_S(
            tOtO0_i[(None, None), None])
        tOtO1_t2r = thr_tmem_load.partition_S(
            tOtO1_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(
            thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(
            tOcO_i[(None, None), None])

        for col_pass_idx in cutlass.range(
            self.head_dim // corr_tile_size, unroll_full=True):
            tOtO0_t2r_i = tOtO0_t2r[None, 0, 0, col_pass_idx]
            tOtO1_t2r_i = tOtO1_t2r[None, 0, 0, col_pass_idx]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, col_pass_idx]
            frg_shape = tOcO_t2r[None, 0, 0, col_pass_idx].shape
            tOrO0_frg = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            tOrO1_frg = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            is_zero_output = (
                scale0 == Float32(0.0) and scale1 == Float32(0.0)
            )
            if not is_zero_output:
                cute.copy(tiled_tmem_load, tOtO0_t2r_i, tOrO0_frg)
                cute.copy(tiled_tmem_load, tOtO1_t2r_i, tOrO1_frg)
                for j in cutlass.range(
                    0, cute.size(tOrO0_frg), 2, unroll_full=True
                ):
                    o0_a, o0_b = cute.arch.mul_packed_f32x2(
                        (tOrO0_frg[j], tOrO0_frg[j + 1]),
                        (scale0, scale0),
                    )
                    o1_a, o1_b = cute.arch.mul_packed_f32x2(
                        (tOrO1_frg[j], tOrO1_frg[j + 1]),
                        (scale1, scale1),
                    )
                    tOrO0_frg[j], tOrO0_frg[j + 1] = (
                        cute.arch.add_packed_f32x2(
                            (o0_a, o0_b), (o1_a, o1_b))
                    )
            else:
                tOrO0_frg.fill(Float32(0.0))
            copy_utils.cvt_copy(tiled_smem_store, tOrO0_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def correction_rescale(
        self,
        tiled_mma_pv: cute.TiledMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ) -> None:
        thr_mma = tiled_mma_pv.get_slice(0)
        tOcO = thr_mma.partition_C(
            cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size: cutlass.Constexpr[int] = 16
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(
                tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(
                tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        thr_tmem_load = tcgen05.make_tmem_copy(
            tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(
            tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count: cutlass.Constexpr[int] = self.head_dim // corr_tile_size
        for fi in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(
                tOtO_t2r.iterator + fi * corr_tile_size,
                tOtO_t2r.layout,
            )
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(
                0, cute.size(tOrO_frg), 2, unroll_full=True
            ):
                tOrO_frg[j], tOrO_frg[j + 1] = (
                    cute.arch.mul_packed_f32x2(
                        (tOrO_frg[j], tOrO_frg[j + 1]),
                        (scale, scale),
                    )
                )
            tOtO_r2t_i = cute.make_tensor(
                tOtO_r2t.iterator + fi * corr_tile_size,
                tOtO_r2t.layout,
            )
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def mma(
        self,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tP_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        mRequestIndices: cute.Tensor,
        mKvTileIndices: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mBlockValidMask: cute.Tensor,
        tile_scheduler: DecodeTileScheduler,
        page_size: Int32,
        kv_chunk_size_pages: Int32,
    ) -> None:
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tSrQ0_layout = tSrQ[None, None, None, 0].layout
        tSrK0_layout = tSrK[None, None, None, 0].layout
        qk_mma_op = tiled_mma_qk.op
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(
            sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(
            sK, sm100_desc.Major.K)
        q_smem_start = sm100_desc.make_smem_desc_start_addr(
            sQ[None, None, None, 0].iterator)
        sm100_helpers.declare_ptx_smem_desc(
            q_smem_start, q_smem_base, tSrQ0_layout,
            var_name_prefix="decode_q_smem_desc",
        )
        sm100_helpers.declare_ptx_idesc(
            qk_mma_op, var_name="decode_qk_idesc")
        gemm_qk = partial(
            sm100_helpers.gemm_ptx_precomputed_varname,
            smem_desc_base_b=k_smem_base,
            tCrB_layout=tSrK0_layout,
            smem_var_name_prefix="decode_q_smem_desc",
            idesc_var_name="decode_qk_idesc",
            smem_offset=0,
            zero_init=True,
            cta_group=self.cta_group_size,
            mma_kind=self.mma_kind,
        )

        thr_mma_pv = tiled_mma_pv.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_base = thr_mma_qk.make_fragment_C(qk_acc_shape)
        tStS = cute.make_tensor(
            tStS_base.iterator,
            cute.append(
                tStS_base.layout,
                cute.make_layout(
                    (self.s_stage,), stride=(self.tmem_stage_stride,)),
            ),
        )
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP_base = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        tP_width_ratio = const_expr(Float32.width // self.v_dtype.width)
        tP_stage_stride = const_expr(
            self.tmem_stage_stride * tP_width_ratio)
        tOrP = cute.make_tensor(
            tOrP_base.iterator + self.tmem_p_offset * tP_width_ratio,
            cute.append(
                tOrP_base.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        pv_mma_op = tiled_mma_pv.op
        sm100_helpers.declare_ptx_idesc(
            pv_mma_op, var_name="decode_pv_idesc")

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage)
        phase_s0 = Int32(0)
        phase_s1 = Int32(0)
        # pipeline_o_acc producer phases.  Producers start at phase 1 so the
        # very first acquire passes trivially (same convention as
        # ``corr_epi_producer_phase`` in correction_loop).
        o_acc_producer_phase0 = Int32(1)
        o_acc_producer_phase1 = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_idx, _, _, _ = work_tile.tile_idx
            if mBlockValidMask[work_idx] != Int32(0):
                batch_idx_mma = mRequestIndices[work_idx]
                split_idx_mma = mKvTileIndices[work_idx]
                seqused_k_mma = mSeqUsedK[batch_idx_mma]
                kv_pages_mma = (
                    seqused_k_mma + page_size - Int32(1)) // page_size
                kv_page_begin_mma = split_idx_mma * kv_chunk_size_pages
                kv_page_end_mma = cutlass.min(
                    kv_pages_mma,
                    kv_page_begin_mma + kv_chunk_size_pages,
                )
                page_count_mma = kv_page_end_mma - kv_page_begin_mma
                block_iter_count_mma = (
                    page_count_mma + Int32(1)) & ~Int32(1)

                # ---------------------------------------------------------
                # Tile-level O-accumulator acquire.
                #
                # The MMA is the pipeline_o_acc producer but used to
                # ``producer_commit`` without ever calling producer_acquire,
                # so nothing bounded how far ahead of the correction warp it
                # could run *across work tiles*.  That matters because the
                # softmax->correction stats mailbox (row_sum / row_max /
                # acc_scale) lives in the first two columns of the S
                # accumulator, TMEM S[stage][:, 0:2] (see the
                # ``tStS_vec_layout`` compositions in softmax_loop and
                # correction_loop).  The first QK GEMM of the *next* work
                # tile overwrites S[stage] wholesale and therefore destroys
                # the stats the correction warp is about to read.
                #
                # The only thing that previously ordered that overwrite was
                # the borrowed pipeline_s_p_o credit chain, whose depth is
                # ``block_iter_count // 2`` acquires per tile.  With
                # block_iter_count == 2 (per-split chunk of 1 or 2 pages)
                # that depth is 1, and the license to clobber S[stage] in
                # tile N+1 comes from tile N-1's correction release -- a
                # full tile stale.  The MMA then races the correction warp
                # for the tile-N stats and wins often enough to corrupt
                # ~50% of the partial rows (negative/garbage row_sum ->
                # log2(neg) -> NaN LSE_partial -> garbage after combine).
                #
                # pipeline_o_acc already has exactly the right semantics --
                # the correction warp releases it (correction_loop, right
                # after it has read the final stats and run the epilogue
                # combine) -- it was simply never waited on.  Acquiring it
                # here, before the first QK of the tile, makes the producer
                # side of pipeline_o_acc well formed (1 acquire + 1 commit
                # per stage per tile against 1 wait + 1 release on the
                # consumer side) and bounds the MMA to at most one tile
                # ahead of the correction warp for every chunk size.
                # ---------------------------------------------------------
                if block_iter_count_mma > Int32(0):
                    pipeline_o_acc.producer_acquire_w_index_phase(
                        Int32(0), o_acc_producer_phase0)
                    pipeline_o_acc.producer_acquire_w_index_phase(
                        Int32(1), o_acc_producer_phase1)
                    o_acc_producer_phase0 = (
                        o_acc_producer_phase0 ^ Int32(1))
                    o_acc_producer_phase1 = (
                        o_acc_producer_phase1 ^ Int32(1))

                pipeline_q.consumer_wait_w_index_phase(
                    Int32(0), mma_q_consumer_phase)
                mma_q_consumer_phase = mma_q_consumer_phase ^ Int32(1)
                if block_iter_count_mma > Int32(0):
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    k_smem_start = sm100_desc.make_smem_desc_start_addr(
                        sK[
                            None, None, None,
                            mma_kv_consumer_state.index,
                        ].iterator
                    )
                    gemm_qk(
                        Int32(self.tmem_s_offset),
                        smem_desc_start_b=k_smem_start,
                    )
                    pipeline_s_p_o.producer_commit_w_index(Int32(0))
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()

                if block_iter_count_mma > Int32(1):
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    k_smem_start = sm100_desc.make_smem_desc_start_addr(
                        sK[
                            None, None, None,
                            mma_kv_consumer_state.index,
                        ].iterator
                    )
                    gemm_qk(
                        Int32(self.tmem_s_offset)
                        + Int32(self.tmem_stage_stride),
                        smem_desc_start_b=k_smem_start,
                    )
                    pipeline_s_p_o.producer_commit_w_index(Int32(1))
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()

                if block_iter_count_mma > Int32(self.s_stage):
                    for page_rel_pv in cutlass.range(
                        Int32(0),
                        block_iter_count_mma - Int32(self.s_stage),
                        unroll=1,
                    ):
                        pv_slot = page_rel_pv & Int32(1)
                        pv_stage_iter = page_rel_pv // Int32(self.s_stage)
                        pv_phase = phase_s0
                        if pv_slot != Int32(0):
                            pv_phase = phase_s1
                        pipeline_s_p_o.producer_acquire_w_index_phase(
                            pv_slot, pv_phase)
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        v_idx = mma_kv_consumer_state.index
                        sm100_helpers.gemm_ptx_partial(
                            pv_mma_op,
                            Int32(self.tmem_o_offset)
                            + pv_slot * Int32(self.tmem_o_stage_stride),
                            tOrP[None, None, None, pv_slot],
                            tOrV[None, None, None, v_idx],
                            sA=None,
                            sB=sV[None, None, None, v_idx],
                            tA_addr=(
                                Int32(self.tmem_p_offset)
                                + pv_slot * Int32(self.tmem_stage_stride)
                            ),
                            zero_init=pv_stage_iter == Int32(0),
                            mbar_ptr=(
                                pipeline_p_lastsplit
                                .sync_object_full.get_barrier(pv_slot)
                                if self.split_P_arrive > 0 else None
                            ),
                            mbar_phase=(
                                pv_phase
                                if self.split_P_arrive > 0 else None),
                            split_arrive=(
                                self.split_P_arrive
                                if self.split_P_arrive > 0 else None
                            ),
                            cta_group=self.cta_group_size,
                            mma_kind=self.mma_kind,
                        )
                        if pv_slot == Int32(0):
                            phase_s0 = phase_s0 ^ Int32(1)
                        else:
                            phase_s1 = phase_s1 ^ Int32(1)
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()

                        page_rel_qk = page_rel_pv + Int32(self.s_stage)
                        qk_slot = page_rel_qk & Int32(1)
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        k_smem_start = sm100_desc.make_smem_desc_start_addr(
                            sK[
                                None, None, None,
                                mma_kv_consumer_state.index,
                            ].iterator
                        )
                        gemm_qk(
                            Int32(self.tmem_s_offset)
                            + qk_slot * Int32(self.tmem_stage_stride),
                            smem_desc_start_b=k_smem_start,
                        )
                        pipeline_s_p_o.producer_commit_w_index(qk_slot)
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()
                pipeline_q.consumer_release_w_index(Int32(0))

                if block_iter_count_mma > Int32(0):
                    page_rel_epi_begin = cutlass.max(
                        Int32(0),
                        block_iter_count_mma - Int32(self.s_stage),
                    )
                    for page_rel_epi in cutlass.range(
                        page_rel_epi_begin, block_iter_count_mma, unroll=1
                    ):
                        pv_slot = page_rel_epi & Int32(1)
                        pv_stage_iter = page_rel_epi // Int32(self.s_stage)
                        pv_phase = phase_s0
                        if pv_slot != Int32(0):
                            pv_phase = phase_s1
                        pipeline_s_p_o.producer_acquire_w_index_phase(
                            pv_slot, pv_phase)
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        v_idx = mma_kv_consumer_state.index
                        sm100_helpers.gemm_ptx_partial(
                            pv_mma_op,
                            Int32(self.tmem_o_offset)
                            + pv_slot * Int32(self.tmem_o_stage_stride),
                            tOrP[None, None, None, pv_slot],
                            tOrV[None, None, None, v_idx],
                            sA=None,
                            sB=sV[None, None, None, v_idx],
                            tA_addr=(
                                Int32(self.tmem_p_offset)
                                + pv_slot * Int32(self.tmem_stage_stride)
                            ),
                            zero_init=pv_stage_iter == Int32(0),
                            mbar_ptr=(
                                pipeline_p_lastsplit
                                .sync_object_full.get_barrier(pv_slot)
                                if self.split_P_arrive > 0 else None
                            ),
                            mbar_phase=(
                                pv_phase
                                if self.split_P_arrive > 0 else None),
                            split_arrive=(
                                self.split_P_arrive
                                if self.split_P_arrive > 0 else None
                            ),
                            cta_group=self.cta_group_size,
                            mma_kind=self.mma_kind,
                        )
                        pipeline_o_acc.producer_commit_w_index(pv_slot)
                        if pv_slot == Int32(0):
                            phase_s0 = phase_s0 ^ Int32(1)
                        else:
                            phase_s1 = phase_s1 ^ Int32(1)
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()

            work_tile = tile_scheduler.consumer_advance()

    @cute.jit
    def softmax_loop(
        self,
        stage: cutlass.Constexpr[int],
        warp_base: cutlass.Constexpr[int],
        softmax_scale_log2: Float32,
        tStS_pre: cute.Tensor,
        tScS_pre: cute.Tensor,
        tilePlikeFP32: cutlass.Constexpr[int],
        tmem_load_atom_pre: cute.CopyAtom,
        tmem_store_atom_pre: cute.CopyAtom,
        tmem_store_vec_atom_pre: cute.CopyAtom,
        thr_mma_qk_pre: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        mRequestIndices: cute.Tensor,
        mQoTileIndices: cute.Tensor,
        mKvTileIndices: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mBlockValidMask: cute.Tensor,
        tile_scheduler: DecodeTileScheduler,
        seqlen_q: Int32,
        page_size: Int32,
        kv_chunk_size_pages: Int32,
    ) -> None:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        warp_idx_in_wg = warp_idx - Int32(warp_base)
        group_tidx = (
            warp_idx_in_wg * Int32(cute.arch.WARP_SIZE)
            + tidx % Int32(cute.arch.WARP_SIZE)
        )
        stage_i32 = Int32(stage)

        tSAcc = tStS_pre[(None, None), 0, 0, stage]
        thr_tmem_load = tcgen05.make_tmem_copy(
            tmem_load_atom_pre, tSAcc).get_slice(group_tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)
        tScS_t2r = thr_tmem_load.partition_D(tScS_pre)
        tStP_layout = cute.composition(
            tSAcc.layout,
            cute.make_layout((self.m_block_size, tilePlikeFP32)),
        )
        tStP = cute.make_tensor(
            tSAcc.iterator + self.tmem_s_to_p_offset,
            tStP_layout,
        )
        thr_tmem_store = tcgen05.make_tmem_copy(
            tmem_store_atom_pre, tStP).get_slice(group_tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)
        tScS_vec_layout = cute.composition(
            tScS_pre.layout, cute.make_layout((self.m_block_size, 2)))
        tScS_vec = cute.make_tensor(tScS_pre.iterator, tScS_vec_layout)
        tStS_vec_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, 2)))
        tStStats_r2t_dst = cute.make_tensor(
            tSAcc.iterator, tStS_vec_layout)
        thr_tmem_store_vec = tcgen05.make_tmem_copy(
            tmem_store_vec_atom_pre,
            tStStats_r2t_dst,
        ).get_slice(group_tidx)
        tStStats_r2t = thr_tmem_store_vec.partition_D(tStStats_r2t_dst)
        tScStats_r2t = thr_tmem_store_vec.partition_S(tScS_vec)
        tScP_shape = (
            self.mma_tiler_qk[0] // thr_mma_qk_pre.thr_id.shape,
            tilePlikeFP32,
        )

        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(
                cute.make_identity_tensor(tScP_shape)).shape,
            Float32,
        )
        s_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_idx, _, _, _ = work_tile.tile_idx
            if mBlockValidMask[work_idx] != Int32(0):
                softmax = SoftmaxSm100.create(
                    softmax_scale_log2,
                    rescale_threshold=self.rescale_threshold,
                )
                softmax.reset()
                batch_idx = mRequestIndices[work_idx]
                qo_tile = mQoTileIndices[work_idx]
                seqused_k = mSeqUsedK[batch_idx]
                split_idx = mKvTileIndices[work_idx]
                kv_pages = (
                    seqused_k + page_size - Int32(1)) // page_size
                kv_page_begin = split_idx * kv_chunk_size_pages
                kv_page_end = cutlass.min(
                    kv_pages, kv_page_begin + kv_chunk_size_pages
                )
                page_count = kv_page_end - kv_page_begin
                block_iter_count = (page_count + Int32(1)) & ~Int32(1)
                if const_expr(stage == 0):
                    stage_page_count = block_iter_count // Int32(2)
                else:
                    stage_page_count = block_iter_count // Int32(2)

                seqlen_info = SeqlenInfoQK(
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    seqlen_q,
                    seqused_k,
                    False,
                    False,
                    False,
                    True,
                )
                mask = AttentionMask(
                    self.m_block_size,
                    self.n_block_size,
                    seqlen_info,
                    qhead_per_kvhead_packgqa=self.qhead_per_kv,
                )
                wg_count = stage_page_count
                if wg_count > Int32(0):
                    page_rel0 = stage_i32
                    page_rel0_clamped = cutlass.min(
                        page_rel0, page_count - Int32(1))
                    page_idx_global = kv_page_end - Int32(1) - page_rel0_clamped
                    kv_valid_cols = cutlass.min(
                        Int32(self.n_block_size),
                        seqused_k - page_idx_global * page_size,
                    )
                    if page_rel0 >= page_count:
                        kv_valid_cols = Int32(0)
                    sm_stats_producer_phase = self.softmax_step(
                        softmax,
                        mask,
                        stage_i32,
                        s_consumer_phase,
                        page_idx_global,
                        qo_tile,
                        kv_valid_cols,
                        tStS_t2r,
                        tScS_t2r,
                        tStP_r2t,
                        tSrP_r2t_f32,
                        thr_tmem_load,
                        thr_tmem_store,
                        thr_tmem_store_vec,
                        pipeline_s_p_o,
                        pipeline_p_lastsplit,
                        pipeline_sm_stats,
                        group_tidx,
                        warp_idx_in_wg,
                        tStStats_r2t,
                        tScStats_r2t,
                        sm_stats_producer_phase,
                        is_first=True,
                    )
                    s_consumer_phase = s_consumer_phase ^ Int32(1)

                    for stage_iter in cutlass.range(
                        Int32(1), wg_count, unroll=1
                    ):
                        page_rel = (
                            stage_iter * Int32(self.s_stage) + stage_i32)
                        page_rel_clamped = cutlass.min(
                            page_rel, page_count - Int32(1))
                        page_idx_global_n = (
                            kv_page_end - Int32(1) - page_rel_clamped)
                        kv_valid_cols_n = cutlass.min(
                            Int32(self.n_block_size),
                            seqused_k - page_idx_global_n * page_size,
                        )
                        # Dummy-iter analysis: with s_stage=2, the WG that
                        # handles stage_i32=0 only ever sees page_rel ≤
                        # block_iter_count - 2 < page_count → NEVER dummy.
                        # The WG with stage_i32=1 sees page_rel =
                        # block_iter_count - 1 at its last iter, which
                        # equals page_count iff page_count is odd → only
                        # WG1 may need the runtime mask_dummy_only guard.
                        # Pass None for WG0 so the const_expr branch in
                        # softmax_step eliminates the runtime check
                        # entirely (compile-time disappears).
                        if const_expr(stage == 0):
                            sm_stats_producer_phase = self.softmax_step(
                                softmax, mask, stage_i32, s_consumer_phase,
                                page_idx_global_n, qo_tile, kv_valid_cols_n,
                                tStS_t2r, tScS_t2r, tStP_r2t, tSrP_r2t_f32,
                                thr_tmem_load, thr_tmem_store, thr_tmem_store_vec,
                                pipeline_s_p_o, pipeline_p_lastsplit,
                                pipeline_sm_stats,
                                group_tidx, warp_idx_in_wg,
                                tStStats_r2t, tScStats_r2t,
                                sm_stats_producer_phase,
                                is_first=False,
                                apply_mask=False,
                                # mask_dummy_only=None → no runtime check
                            )
                        else:
                            is_dummy = page_rel >= page_count
                            if is_dummy:
                                kv_valid_cols_n = Int32(0)
                            sm_stats_producer_phase = self.softmax_step(
                                softmax, mask, stage_i32, s_consumer_phase,
                                page_idx_global_n, qo_tile, kv_valid_cols_n,
                                tStS_t2r, tScS_t2r, tStP_r2t, tSrP_r2t_f32,
                                thr_tmem_load, thr_tmem_store, thr_tmem_store_vec,
                                pipeline_s_p_o, pipeline_p_lastsplit,
                                pipeline_sm_stats,
                                group_tidx, warp_idx_in_wg,
                                tStStats_r2t, tScStats_r2t,
                                sm_stats_producer_phase,
                                is_first=False,
                                apply_mask=False,
                                mask_dummy_only=is_dummy,
                            )
                        s_consumer_phase = s_consumer_phase ^ Int32(1)

                    pipeline_sm_stats.producer_acquire_w_index_phase(
                        Int32(0), sm_stats_producer_phase)
                    tSrStats = cute.make_rmem_tensor(
                        tScStats_r2t.shape, self.qk_acc_dtype)
                    tSrStats[0] = softmax.row_sum[0]
                    tSrStats[1] = softmax.row_max[0]
                    cute.copy(thr_tmem_store_vec, tSrStats, tStStats_r2t)
                    cute.arch.fence_view_async_tmem_store()
                else:
                    pipeline_sm_stats.producer_acquire_w_index_phase(
                        Int32(0), sm_stats_producer_phase)
                    tSrStats = cute.make_rmem_tensor(
                        tScStats_r2t.shape, self.qk_acc_dtype)
                    tSrStats[0] = Float32(0.0)
                    tSrStats[1] = -Float32.inf
                    cute.copy(thr_tmem_store_vec, tSrStats, tStStats_r2t)
                    cute.arch.fence_view_async_tmem_store()
                pipeline_sm_stats.producer_commit_w_index(Int32(0))
                sm_stats_producer_phase = sm_stats_producer_phase ^ Int32(1)

            work_tile = tile_scheduler.consumer_advance()

        pipeline_sm_stats.producer_acquire_w_index_phase(
            Int32(0), sm_stats_producer_phase)

    @cute.jit
    def softmax_step(
        self,
        softmax: SoftmaxSm100,
        mask: AttentionMask,
        stage: Int32,
        s_phase: Int32,
        page_idx: Int32,
        qo_tile: Int32,
        kv_valid_cols: Int32,
        tStS_t2r: cute.Tensor,
        tScS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        tSrP_r2t_f32: cute.Tensor,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_vec: cute.CopyAtom,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        group_tidx: Int32,
        warp_idx_in_wg: Int32,
        tStStats_r2t: cute.Tensor,
        tScStats_r2t: cute.Tensor,
        sm_stats_producer_phase: Int32,
        is_first: cutlass.Constexpr[bool],
        apply_mask: cutlass.Constexpr[bool] = True,
        mask_dummy_only: Optional[cutlass.Boolean] = None,
    ) -> Int32:
        # apply_mask=False is the inner-page fast path: skip both the seqlen
        # bounds check and the causal-diagonal check, which together cost ~15
        # cyc per iter on the producer pre-publication critical path that
        # gates correction WG's consumer_wait (top long_scoreboard PC in NCU).
        # Callers must only set apply_mask=False when they can prove the tile
        # is fully unmasked (no partial-page seqlen tail, no causal diagonal
        # cut).
        #
        # mask_dummy_only (runtime bool, used only when apply_mask=False):
        # when True the iter is a "dummy" rounded-up iter that needs the
        # mask to zero out garbage S — runs the mask at runtime cost.  For
        # non-dummy iters it stays the fast no-mask path.
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, s_phase)
        sm_stats_try_acquire = (
            pipeline_sm_stats.producer_try_acquire_w_index_phase(
                Int32(0), sm_stats_producer_phase)
        )
        tSrS_t2r = cute.make_rmem_tensor(
            tScS_t2r.shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if const_expr(apply_mask):
            mask.apply_mask_sm100(
                tSrS_t2r,
                tScS_t2r,
                m_block=qo_tile,
                n_block=page_idx,
                mask_seqlen=True,
                mask_causal=self.causal,
                kv_valid_cols=kv_valid_cols,
            )
        elif const_expr(mask_dummy_only is not None):
            if mask_dummy_only:
                # Dummy iter — zero everything via mask (kv_valid_cols=0
                # makes mask_r2p_lambda set all positions to -inf).
                mask.apply_mask_sm100(
                    tSrS_t2r,
                    tScS_t2r,
                    m_block=qo_tile,
                    n_block=page_idx,
                    mask_seqlen=True,
                    mask_causal=self.causal,
                    kv_valid_cols=kv_valid_cols,
                )
        # Publish acc_scale in log2-domain (un-exp2'd); correction WG does
        # the exp2 only when an actual rescale fires.  Removes MUFU.EX2 from
        # the sm_stats publication critical path that gates correction's
        # consumer_wait (the dominant long_scoreboard hot PC in NCU).
        row_max, acc_scale_log2 = softmax.update_row_max_deferred_exp2(
            tSrS_t2r.load(), is_first)
        pipeline_sm_stats.producer_acquire_w_index_phase(
            Int32(0), sm_stats_producer_phase, sm_stats_try_acquire)
        tSrStats = cute.make_rmem_tensor(
            tScStats_r2t.shape, self.qk_acc_dtype)
        tSrStats[0] = acc_scale_log2
        tSrStats[1] = row_max
        cute.copy(thr_tmem_store_vec, tSrStats, tStStats_r2t)
        cute.arch.fence_view_async_tmem_store()
        pipeline_sm_stats.producer_commit_w_index(Int32(0))
        sm_stats_producer_phase = sm_stats_producer_phase ^ Int32(1)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype),
            tSrS_t2r.layout,
        )
        # exp2 for the internal row_sum carry happens AFTER producer_commit, so
        # it no longer extends correction's consumer-wait window.
        # acc_scale_log2 == 0.0 in the threshold/first-iter paths makes
        # exp2(0)=1.0, which is the no-rescale identity for the row_sum carry —
        # semantically equivalent to the original ``acc_scale=1.0`` branch.
        if const_expr(is_first):
            row_sum_init = Float32(0.0)
        else:
            acc_scale_mult = cute.math.exp2(acc_scale_log2, fastmath=True)
            row_sum_init = softmax.row_sum[0] * acc_scale_mult
        # Bulk EX2 emulation parameters.
        #
        #   ex2_emu_freq=16         emulate exp2 with FFMA2 polynomial on
        #                           15 of every 16 (j, k) positions; the
        #                           remaining 1/16 still issues MUFU.EX2.
        #                           This cuts the MUFU.EX2 throughput bottleneck
        #                           in the softmax inner loop (≈22k cyc
        #                           saved per stage at baseline).
        #   ex2_emu_res=3           degree-3 polynomial; res=4 broke
        #                           kv=1024 close-tolerance even with
        #                           poly_degree=5 — 3 is the most aggressive
        #                           setting that still passes cos_sim ≥ 0.99
        #                           against the reference for the fp8 PV path.
        #   ex2_emu_start_frg=1     skip the emulation for fragment index 0
        #                           (preserves accuracy on the first iter
        #                           where row_max is least settled).
        #
        # If you tune these, re-run the variable-kv self-consistency check
        # (split vs non-split must stay at cos_min ≥ 0.99).
        softmax.row_sum[0] = softmax.scale_apply_exp2_convert_sum(
            tSrS_t2r,
            row_max,
            tSrP_r2t,
            row_sum_init,
            ex2_emu_freq=16,
            ex2_emu_res=3,
            ex2_emu_start_frg=1,
        )
        for k in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store,
                tSrP_r2t_f32[None, None, k],
                tStP_r2t[None, None, k],
            )
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive
                    // self.n_block_size
                )
                if const_expr(k + 1 == split_P_arrive_idx):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        return sm_stats_producer_phase

    @cute.jit
    def load(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mQ: cute.Tensor,
        mK_paged: cute.Tensor,
        mV_paged: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        mRequestIndices: cute.Tensor,
        mQoTileIndices: cute.Tensor,
        mKvTileIndices: cute.Tensor,
        mSeqUsedK: cute.Tensor,
        mBlockValidMask: cute.Tensor,
        tile_scheduler: DecodeTileScheduler,
        page_size: Int32,
        kv_chunk_size_pages: Int32,
    ) -> None:
        cute.arch.setmaxregister_decrease(self.num_regs_load)
        thr_mma_qk_ld = tiled_mma_qk.get_slice(0)
        thr_mma_pv_ld = tiled_mma_pv.get_slice(0)
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            work_idx, head_kv_idx, _, _ = work_tile.tile_idx
            if mBlockValidMask[work_idx] != Int32(0):
                batch_idx_ld = mRequestIndices[work_idx]
                qo_tile_ld = mQoTileIndices[work_idx]
                split_idx_ld = mKvTileIndices[work_idx]
                seqused_k_ld = mSeqUsedK[batch_idx_ld]
                kv_pages_ld = (
                    seqused_k_ld + page_size - Int32(1)) // page_size
                kv_page_begin_ld = split_idx_ld * kv_chunk_size_pages
                kv_page_end_ld = cutlass.min(
                    kv_pages_ld, kv_page_begin_ld + kv_chunk_size_pages
                )
                page_count_ld = kv_page_end_ld - kv_page_begin_ld
                block_iter_count_ld = (
                    page_count_ld + Int32(1)) & ~Int32(1)
                physical_page_v0 = Int32(0)
                physical_page_v1 = Int32(0)

                mQ_cur_ld = mQ[None, None, None, batch_idx_ld][
                    None, None, head_kv_idx
                ]
                tiler_gQ_ld = (
                    (self.mma_tiler_qk[0] * self.q_stage),
                    self.head_dim,
                )
                gQ_ld = cute.local_tile(
                    mQ_cur_ld, tiler_gQ_ld, (qo_tile_ld, 0))
                gQ_ld = layout_utils.select(
                    cute.flat_divide(gQ_ld, (self.mma_tiler_qk[0],)),
                    mode=[0, 2, 1],
                )
                tSgQ_ld = thr_mma_qk_ld.partition_A(gQ_ld)
                load_Q_fn_full, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), tSgQ_ld, sQ
                )
                mK_cur_ld = mK_paged[None, None, head_kv_idx, None]
                gK_ld = cute.local_tile(
                    mK_cur_ld,
                    cute.select(self.mma_tiler_qk, mode=[1, 2]),
                    (None, 0, None),
                )
                tSgK_ld = thr_mma_qk_ld.partition_B(gK_ld)
                tKsK_ld, tKgK_ld = cpasync.tma_partition(
                    tma_atom_K, 0, cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK_ld, 0, 3),
                )
                mV_cur_ld = mV_paged[None, None, head_kv_idx, None]
                gV_ld = cute.local_tile(
                    mV_cur_ld,
                    cute.select(self.mma_tiler_pv, mode=[1, 2]),
                    (0, None, None),
                )
                tOgV_ld = thr_mma_pv_ld.partition_B(gV_ld)
                tVsV_ld, tVgV_ld = cpasync.tma_partition(
                    tma_atom_V, 0, cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tOgV_ld, 0, 3),
                )

                if block_iter_count_ld > Int32(0):
                    # Prime K0 before Q; then follow BSA order
                    # K1, V0, K2, V1, ...
                    page_idx_ld0 = kv_page_end_ld - Int32(1)
                    physical_page_v0 = mPageTable[batch_idx_ld, page_idx_ld0]
                    physical_page_v1 = physical_page_v0
                    self.load_KV_physical(
                        tma_atom_K,
                        tKgK_ld,
                        tKsK_ld,
                        physical_page_v0,
                        pipeline_kv,
                        kv_producer_state,
                    )
                    kv_producer_state.advance()

                self.load_Q(
                    load_Q_fn_full,
                    pipeline_q,
                    Int32(0),
                    q_producer_phase,
                )
                q_producer_phase = q_producer_phase ^ Int32(1)

                if block_iter_count_ld > Int32(0):
                    if block_iter_count_ld > Int32(1):
                        page_rel_k1 = cutlass.min(
                            Int32(1), page_count_ld - Int32(1))
                        page_idx_ld1 = kv_page_end_ld - Int32(1) - page_rel_k1
                        physical_page_v1 = mPageTable[
                            batch_idx_ld, page_idx_ld1]
                        self.load_KV_physical(
                            tma_atom_K,
                            tKgK_ld,
                            tKsK_ld,
                            physical_page_v1,
                            pipeline_kv,
                            kv_producer_state,
                        )
                        kv_producer_state.advance()

                    if block_iter_count_ld > Int32(2):
                        for page_rel in cutlass.range(
                            Int32(0),
                            block_iter_count_ld - Int32(2),
                            unroll=1,
                        ):
                            page_rel_v_ld = cutlass.min(
                                page_rel, page_count_ld - Int32(1))
                            physical_page_v_ld = physical_page_v0
                            if (page_rel & Int32(1)) != Int32(0):
                                physical_page_v_ld = physical_page_v1
                            self.load_KV_physical(
                                tma_atom_V,
                                tVgV_ld,
                                tVsV_ld,
                                physical_page_v_ld,
                                pipeline_kv,
                                kv_producer_state,
                            )
                            kv_producer_state.advance()
                            page_rel_k_ld = cutlass.min(
                                page_rel + Int32(2),
                                page_count_ld - Int32(1),
                            )
                            page_idx_k_ld = (
                                kv_page_end_ld - Int32(1) - page_rel_k_ld)
                            physical_page_k_ld = mPageTable[
                                batch_idx_ld, page_idx_k_ld]
                            self.load_KV_physical(
                                tma_atom_K,
                                tKgK_ld,
                                tKsK_ld,
                                physical_page_k_ld,
                                pipeline_kv,
                                kv_producer_state,
                            )
                            if (page_rel & Int32(1)) == Int32(0):
                                physical_page_v0 = physical_page_k_ld
                            else:
                                physical_page_v1 = physical_page_k_ld
                            kv_producer_state.advance()

                    page_rel_epi_begin_ld = cutlass.max(
                        Int32(0),
                        block_iter_count_ld - Int32(2),
                    )
                    for page_rel_epi in cutlass.range(
                        page_rel_epi_begin_ld,
                        block_iter_count_ld,
                        unroll=1,
                    ):
                        page_rel_v_ld = cutlass.min(
                            page_rel_epi, page_count_ld - Int32(1))
                        physical_page_v_ld = physical_page_v0
                        if (page_rel_epi & Int32(1)) != Int32(0):
                            physical_page_v_ld = physical_page_v1
                        self.load_KV_physical(
                            tma_atom_V,
                            tVgV_ld,
                            tVsV_ld,
                            physical_page_v_ld,
                            pipeline_kv,
                            kv_producer_state,
                        )
                        kv_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.consumer_advance()

        pipeline_kv.producer_tail(kv_producer_state)
        pipeline_q.producer_acquire_w_index_phase(
            Int32(self.q_stage - 1), q_producer_phase)

    @cute.jit
    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        stage: Int32,
        phase: Int32,
    ) -> None:
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(
            src_idx=Int32(0),
            dst_idx=stage,
            tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage),
        )

    @cute.jit
    def load_KV_physical(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        physical_page: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
    ) -> None:
        pipeline_kv.producer_acquire(producer_state)
        cute.copy(
            tma_atom,
            tXgX[(None, 0, physical_page)],
            tXsX[(None, producer_state.index)],
            tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
        )

_atten_compile_cache: dict[tuple[object, ...], object] = {}


def run_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    block_valid_mask: torch.Tensor,
    split_counts: torch.Tensor,
    o_indptr: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    O_partial: Optional[torch.Tensor],
    LSE_partial: Optional[torch.Tensor],
    *,
    softmax_scale: float,
    seqlen_q: int,
    page_size: int,
    kv_chunk_size_pages: int,
    split_kv: bool,
    causal: bool,
    return_lse: bool = True,
    disable_softmax_exp2: bool = False,
    O_partial_dummy: Optional[torch.Tensor] = None,
    LSE_partial_dummy: Optional[torch.Tensor] = None,
) -> None:
    """Launch the SM100 UMMA paged decode attention CUTE DSL kernel.

    qhead_per_kv is derived from input shapes (q.shape[1] // k.shape[1]).
    disable_softmax_exp2 toggles the sage-style host flag (decision §1.7);
    default False keeps full ex2 emulation.

    ``O_partial_dummy`` / ``LSE_partial_dummy`` let callers pre-allocate the
    placeholder buffers for the non-split path, avoiding ~5us of per-call
    ``torch.empty`` overhead in tight decoding loops.
    """

    q_dtype = torch2cute_dtype_map[q.dtype]
    o_dtype = torch2cute_dtype_map[out.dtype]
    qhead_per_kv = q.shape[1] // k.shape[1]
    q_tokens_per_group = 128 // int(qhead_per_kv)
    write_lse = bool(return_lse) or bool(split_kv)
    if int(seqlen_q) != q_tokens_per_group:
        raise NotImplementedError(
            "decode fp8 currently assumes one full packed-q tile: "
            f"seqlen_q must equal {q_tokens_per_group}, got {seqlen_q}"
        )
    key = (
        "decode_attention",
        q.shape[-1],
        q_dtype,
        o_dtype,
        bool(split_kv),
        bool(causal),
        int(qhead_per_kv),
        int(seqlen_q),
        bool(write_lse),
        bool(disable_softmax_exp2),
    )
    if key not in _atten_compile_cache:
        from quack.compile_utils import make_fake_tensor

        total_q = cute.sym_int64()
        head_q = cute.sym_int64()
        num_pages = cute.sym_int64()
        head_kv = cute.sym_int64()
        batch = cute.sym_int64()
        batch_plus_one = cute.sym_int64()
        max_pages = cute.sym_int64()
        work_capacity = cute.sym_int64()
        partial_rows = cute.sym_int64()
        partial_rows_flat = cute.sym_int64()
        head_dim = int(q.shape[-1])
        kernel = SparseDecodeAttentionForwardSm100(
            head_dim=head_dim,
            qhead_per_kv=int(qhead_per_kv),
            page_size=int(page_size),
            split_kv=bool(split_kv),
            causal=bool(causal),
            write_lse=bool(write_lse),
            disable_softmax_exp2=bool(disable_softmax_exp2),
        )
        # Always pass non-None fake tensors so the @cute.kernel positional
        # arg marshalling stays stable; the kernel only reads these when
        # split_kv=True (decision #10 epilogue branch).
        fake_O_partial = make_fake_tensor(
            Float32, (partial_rows_flat, head_dim), divisibility=4)
        fake_LSE_partial = make_fake_tensor(
            Float32, (partial_rows, head_q), divisibility=1, leading_dim=1)
        # Q is passed as a [B, Sq, Hq, D] view so the kernel can build the same
        # PackGQA TMA view used by FA/BSA and issue one full-tile Q TMA.
        # O still uses the compact 2D view for the packed-GQA TMA epilogue.
        total_q_flat = cute.sym_int64()
        _atten_compile_cache[key] = cute.compile(
            kernel,
            make_fake_tensor(
                q_dtype, (batch, int(seqlen_q), head_q, head_dim),
                divisibility=16),
            make_fake_tensor(q_dtype, (num_pages, head_kv, int(page_size), head_dim), divisibility=16),
            make_fake_tensor(q_dtype, (num_pages, head_kv, int(page_size), head_dim), divisibility=16),
            make_fake_tensor(Int32, (batch, max_pages), divisibility=1, leading_dim=1),
            make_fake_tensor(Int32, (batch,), divisibility=1, leading_dim=0),
            make_fake_tensor(Int32, (work_capacity,), divisibility=1, leading_dim=0),
            make_fake_tensor(Int32, (work_capacity,), divisibility=1, leading_dim=0),
            make_fake_tensor(Int32, (work_capacity,), divisibility=1, leading_dim=0),
            make_fake_tensor(Int32, (work_capacity,), divisibility=1, leading_dim=0),
            make_fake_tensor(Int32, (batch,), divisibility=1, leading_dim=0),
            make_fake_tensor(Int32, (batch_plus_one,), divisibility=1, leading_dim=0),
            make_fake_tensor(o_dtype, (total_q_flat, head_dim), divisibility=128 // o_dtype.width),
            make_fake_tensor(Float32, (total_q, head_q), divisibility=1, leading_dim=1),
            fake_O_partial,
            fake_LSE_partial,
            Float32(float(softmax_scale)),
            Int32(int(seqlen_q)),
            Int32(int(page_size)),
            Int32(int(kv_chunk_size_pages)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    q_4d = q.view(
        q.shape[0] // int(seqlen_q), int(seqlen_q), q.shape[1], q.shape[2])
    out_2d = out.view(out.shape[0] * out.shape[1], out.shape[2])
    # Compile keeps non-None fake partial buffers for positional stability
    # (see fake_O_partial / fake_LSE_partial above).  Runtime callers that
    # don't need them (split_kv=False) pass None; allocate small uninitialized
    # dummy buffers so the kernel signature still matches without launching
    # torch fill kernels.
    if O_partial is None:
        # Reuse caller-cached dummy when available (e.g. the
        # SparseDecodePagedAttentionWrapper plan() pre-allocation), else
        # allocate a small placeholder on the fly.
        O_partial_kernel = (
            O_partial_dummy
            if O_partial_dummy is not None
            else torch.empty(
                (1, q.shape[2]), dtype=torch.float32, device=q.device)
        )
    else:
        O_partial_kernel = O_partial.view(
            O_partial.shape[0] * O_partial.shape[1], O_partial.shape[2])
    if LSE_partial is None:
        LSE_partial = (
            LSE_partial_dummy
            if LSE_partial_dummy is not None
            else torch.empty(
                (1, q.shape[1]), dtype=torch.float32, device=q.device)
        )
    with torch.cuda.nvtx.range("Decode_Attention"):
        _atten_compile_cache[key](
            q_4d, k, v, page_table, seqused_k,
            request_indices, qo_tile_indices, kv_tile_indices, block_valid_mask,
            split_counts, o_indptr,
            out_2d, lse, O_partial_kernel, LSE_partial,
            softmax_scale, seqlen_q, page_size, kv_chunk_size_pages,
        )


__all__ = ["SparseDecodeAttentionForwardSm100", "run_decode_attention"]
