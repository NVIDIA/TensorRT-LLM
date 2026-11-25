# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Uniform-precision (bf16 or fp8-e4m3), head-dim-256 prefill FMHA for Blackwell (SM100) and newer.

Single 128-query tile per CTA, 2-CTA cluster (M=256 UMMA, flat (4,1) TMEM
layout, P in TMEM). Derived from mixed_input_fmha_prefill_d256.py with the
int8-KV dequant path removed: K/V TMA-load directly at the input precision into the MMA-B
SMEM, with pair-scoped (CtaGroup TWO, tx x2) K/V TMA feeding the leader-only
2-CTA UMMA barrier. Warp layout: softmax 0-3, correction 4-7, mma 8, load 9.

Optimizations carried:
- Softmax inner loop (fmha.py-style): the TMEM S load issues as 32-element
  chunks with the row-max interleaved between them, and the exp2/FMA/cvt loop
  is software-pipelined (scale-FMA runs ahead; cvt and row-sum trail behind).
- Arch-gated fused-reduction S load (tcgen05.ld.red MAX) on devices with
  cc >= 10.7; on Blackwell the plain chunked-load path compiles unchanged.
  TRTLLM_GLM52_FMHA_LDSTAT=auto|0|1 overrides the gate (debug).
- Per-chunk O accumulator handoff: pv_acc_stage == iterations_pv with the
  pipeline stage index equal to the dv-chunk index (mbar pairs only, no extra
  TMEM), so the correction warpgroup rescales/drains chunk i while the MMA
  accumulates chunk i+1, and the next work item's first PV is gated on chunk
  0's drain only.
- Sliding-window masks: the kv loop covers tiles [start, start+count) via
  FusedMask.get_masked_info (window start offset in the load warp and the
  softmax mask coordinates, plus two-sided edge masking), so windowed runtime
  scales with the window size. Causal/residual/no-mask reduce to start == 0.

Supported: no-mask / causal / residual / sliding-window (WINDOW_MASK_INFERENCE,
bottom-right alignment), GQA (h_q % h_k == 0), seqlen_q != seqlen_k, and
persistent tile scheduling. Exact math (no ex2 emulation / skip-correction).
"""

import os
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import testing
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.memory import SmemAllocator, TmemAllocator
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.tensor_utils import LayoutEnum

from . import fmha_helpers as fmha_utils


def _device_cc() -> Tuple[int, int]:
    """Host-side compute-capability probe: current context's device, else
    device 0. Returns (0, 0) if the probe fails so the caller falls back to
    the (always-correct) Blackwell load path."""
    if cuda.cuInit(0)[0] != cuda.CUresult.CUDA_SUCCESS:
        return (0, 0)
    err, dev = cuda.cuCtxGetDevice()
    if err != cuda.CUresult.CUDA_SUCCESS:
        err, dev = cuda.cuDeviceGet(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            return (0, 0)
    attr = cuda.CUdevice_attribute
    err0, major = cuda.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
    err1, minor = cuda.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
    if err0 != cuda.CUresult.CUDA_SUCCESS or err1 != cuda.CUresult.CUDA_SUCCESS:
        return (0, 0)
    return (major, minor)


@cute.jit
def load_qk_uniform(
    iterations: int,
    kv_step: cutlass.Int32,
    k_args: Tuple,
    q_args: Optional[Tuple] = None,
):
    """Uniform-precision variant of ``prefill_helpers.load_qk`` (no per-block scale).

    Issues ``iterations`` TMA loads of K (the head-dim split into 128-wide tiles)
    for ``kv_step`` directly into the MMA-B SMEM buffer, plus the Q tiles on the
    first call. Returns the advanced producer(s)."""
    if cutlass.const_expr(q_args is not None):
        tQgQ, tQsQ, tma_atom_q, load_q_producer = q_args
    else:
        tQgQ, tQsQ, tma_atom_q, load_q_producer = None, None, None, None
    tKgK, tKsK, tma_atom_k, load_k_producer = k_args
    for iter in cutlass.range(iterations, unroll=1):
        if cutlass.const_expr(q_args is not None):
            q_handle = load_q_producer.acquire_and_advance()
            cute.copy(
                tma_atom_q,
                tQgQ[None, iter],
                tQsQ[None, q_handle.index],
                tma_bar_ptr=q_handle.barrier,
            )
        k_handle = load_k_producer.acquire_and_advance()
        cute.copy(
            tma_atom_k,
            tKgK[None, kv_step, iter],
            tKsK[None, k_handle.index],
            tma_bar_ptr=k_handle.barrier,
        )
    if cutlass.const_expr(q_args is not None):
        return load_k_producer, load_q_producer
    return load_k_producer


@cute.jit
def load_v_uniform(
    iterations: int,
    kv_step: cutlass.Int32,
    v_args: Tuple,
):
    """Uniform-precision variant of ``prefill_helpers.load_v`` (no per-block scale)."""
    tVgV, tVsV, tma_atom_v, load_v_producer = v_args
    for iter in cutlass.range(iterations, unroll=1):
        v_handle = load_v_producer.acquire_and_advance()
        cute.copy(
            tma_atom_v,
            tVgV[None, iter, kv_step],
            tVsV[None, v_handle.index],
            tma_bar_ptr=v_handle.barrier,
        )
    return load_v_producer


@cute.jit
def mma_qk_uniform(
    iterations: int,
    qk_tiled_mma: cute.TiledMma,
    tensor_args: Tuple,
    pipeline_args: Tuple,
):
    """Uniform-precision variant of ``prefill_helpers.mma_qk``: K is consumed straight
    from the TMA->UMMA ``load_kv`` pipeline (no dequant stage)."""
    tStS, tSrQ, tSrK = tensor_args
    mma_s_producer, load_q_consumer, load_q_releaser, load_kv_consumer = pipeline_args
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    is_leader_cta = cta_rank_in_cluster % 2 == 0
    if is_leader_cta:
        s_handle = mma_s_producer.acquire_and_advance()
        tStS_slice = tStS[None, None, None, s_handle.index]
        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for iter in cutlass.range(iterations, unroll=1):
            if cutlass.const_expr(load_q_consumer is not None):
                load_q_consumer.wait_and_advance()
            tSrQ_slice = tSrQ[None, None, None, iter]
            k_handle = load_kv_consumer.wait_and_advance()
            tSrK_slice = tSrK[None, None, None, k_handle.index]
            num_kphases = cute.size(tSrQ_slice, mode=[2])
            for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                kphase_coord = (None, None, kphase_idx)
                cute.gemm(
                    qk_tiled_mma,
                    tStS_slice,
                    tSrQ_slice[kphase_coord],
                    tSrK_slice[kphase_coord],
                    tStS_slice,
                )
                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            k_handle.release()
            if cutlass.const_expr(load_q_releaser is not None):
                load_q_releaser.release()
                load_q_releaser.advance()
        s_handle.commit()
    return mma_s_producer, load_q_consumer, load_kv_consumer


@cute.jit
def _apply_mask_hoisted(
    mask_type: cutlass.Constexpr,
    acc_qk: cute.Tensor,
    index_qk: cute.Tensor,
    index_qk_static: cute.Tensor,
    seqlen_q: Int32,
    seqlen_k: Int32,
    window_size_left,
    window_size_right,
):
    """FlashInfer-style hoisted band mask.
    Row-invariant [lo,hi) bounds hoisted out of the element loop (one Q row
    per thread fragment); per-element K coordinates come from the UNSHIFTED
    identity partition (trace-time constants -> immediate operands; the
    tile's dynamic base folds into the hoisted bounds); masked writes use
    cutlass.select_ -- a traced `if` records an scf.if yielding the whole
    fragment and inflates register pressure."""
    has_l = cutlass.const_expr(window_size_left is not None)
    has_r = cutlass.const_expr(window_size_right is not None)
    is_residual = cutlass.const_expr(
        mask_type == fmha_utils.MaskEnum.RESIDUAL_MASK
        or mask_type == fmha_utils.MaskEnum.RESIDUAL_MASK_BWD
    )
    is_window_inf = cutlass.const_expr(
        mask_type is fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
        or mask_type is fmha_utils.MaskEnum.WINDOW_MASK_BWD_INFERENCE
    )
    if cutlass.const_expr(has_l or has_r or is_residual):
        base_k = index_qk[0][1] - index_qk_static[0][1]
        row_q_raw = index_qk[0][0]
        if cutlass.const_expr(is_window_inf):
            row_q = row_q_raw + (seqlen_k - seqlen_q)
        else:
            row_q = row_q_raw
        hi = seqlen_k
        if cutlass.const_expr(has_r):
            hi = cutlass.min(row_q + window_size_right + 1, seqlen_k)
        # residual q-side check: an out-of-range row masks all its elements
        hi = cutlass.select_(row_q_raw >= seqlen_q, Int32(0), hi)
        hi_rel = hi - base_k
        if cutlass.const_expr(has_l):
            lo_rel = row_q - window_size_left - base_k
            for i in range(cute.size(acc_qk)):
                k = index_qk_static[i][1]
                acc_qk[i] = cutlass.select_((k < lo_rel) | (k >= hi_rel), -Float32.inf, acc_qk[i])
        else:
            for i in range(cute.size(acc_qk)):
                k = index_qk_static[i][1]
                acc_qk[i] = cutlass.select_(k >= hi_rel, -Float32.inf, acc_qk[i])


class FusedMultiHeadAttentionPrefillD256:
    """Uniform-precision d=256 prefill FMHA (Choice A: single query tile, head-dim
    tiled into 2x128, 2-CTA, no TMEM swap). Forked from the int8 mixed-input
    d=256 kernel with the dequant/scale/transform path removed: K/V are loaded
    directly into the MMA-read SMEM buffer and consumed by the MMA warp."""

    def __init__(
        self,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        is_persistent: bool,
        mask_type: fmha_utils.MaskEnum,
    ):
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.cta_tiler = (128, 128, 256)
        self.qk_mma_tiler = (
            self.cta_tiler[0] * 2,  # default 2cta
            self.cta_tiler[1],
            min(self.cta_tiler[2], 128),  # avoid too large GemmK
        )
        self.pv_mma_tiler = self.qk_mma_tiler  # keep BMM1 & BMM2 at the same pace
        self.pv_block_tiler = (
            self.pv_mma_tiler[0] // 2,  # default 2cta
            self.pv_mma_tiler[1],
            self.pv_mma_tiler[2],
        )
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_pv = self.cta_tiler[2] // self.pv_mma_tiler[1]
        self.cluster_shape_mn = (2, 1)  # use 2x1 cluster by default
        self.tmem_warp_shape_mn = (4, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type
        # No transform warps needed for uniform precision (K/V are not dequantized).
        self.softmax_warp_ids = (0, 1, 2, 3)  # softmax
        self.correction_warp_ids = (4, 5, 6, 7)  # correction
        self.mma_warp_id = 8  # mma
        self.load_warp_id = 9  # load
        self.empty_warp_ids = (10, 11)  # empty
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        # Fused-reduction (tcgen05.ld.red MAX) S load, available past sm_100f.
        # TRTLLM_GLM52_FMHA_LDSTAT=auto (default) resolves from the device compute
        # capability so the same file runs on Blackwell (plain chunked load + FMAX
        # chains) and cc >= 10.7 (fused-max load); 0/1 forces the path.
        _ldstat = os.environ.get("TRTLLM_GLM52_FMHA_LDSTAT", "auto")
        if _ldstat == "auto":
            self.use_ldstat = _device_cc() >= (10, 7)
        else:
            self.use_ldstat = _ldstat == "1"
        self.tmem_alloc_sync_bar_id = 1
        self.tmem_dealloc_sync_bar_id = 2
        self.tmem_s_offset = 256
        self.tmem_p_offset = self.tmem_s_offset
        self.tmem_o_offset = 0
        self.num_regs_softmax = 256
        self.num_regs_correction = 112
        self.num_regs_other = 32
        self.buffer_align_bytes = 1024
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.softmax_warp_ids,
                *self.correction_warp_ids,
                self.load_warp_id,
                self.mma_warp_id,
                *self.empty_warp_ids,
            )
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.q_stage = self.iterations_qk
        # 8 slots x 16 KB: at d=256 a kv step occupies 4 ring slots (K+V),
        # so 4 slots covered one step of prefetch; 8 cover two. Total SMEM
        # ~194 KB -- inside the normal carveout on sm_100 and sm_107 alike.
        self.kv_stage = 8
        self.qk_acc_stage = 2
        # Per-chunk O handoff: stage index == dv-chunk index (iterations_pv).
        # NOT extra TMEM (the O regions are unchanged); each stage's mbar pair
        # guards one fixed 128-col chunk so the MMA, rescale, and epilogue hand
        # off O at chunk granularity instead of whole-buffer.
        self.pv_acc_stage = self.iterations_pv

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        o_iter: cute.Pointer,
        lse_iter: Optional[cute.Pointer],
        problem_shape: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
        scale_softmax_log2: Float32,
        scale_output: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        b, s_q, s_k, h_q, h_k, d = problem_shape
        h_r = h_q // h_k
        # (s, d, ((h_r, h_k), b))
        q_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b)),
            stride=(d, 1, ((d * s_q, d * s_q * h_r), h_r * h_k * s_q * d)),
        )
        q = cute.make_tensor(q_iter, q_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k_layout = cute.make_layout(
            (s_k, d, ((h_r, h_k), b)),
            stride=(d, 1, ((0, d * s_k), h_k * s_k * d)),
        )
        k = cute.make_tensor(k_iter, k_layout)
        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v_layout = cute.make_layout(
            (d, s_k, ((h_r, h_k), b)),
            stride=(1, d, ((0, d * s_k), h_k * s_k * d)),
        )
        v = cute.make_tensor(v_iter, v_layout)
        # (s, d, ((h_r, h_k), b))
        # set divby for better gmem store vectorization (placeholder)
        o_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b)),
            stride=(
                cute.assume(d, divby=256),
                1,
                (
                    (
                        cute.assume(d * s_q, divby=256),
                        cute.assume(d * s_q * h_r, divby=256),
                    ),
                    cute.assume(h_r * h_k * s_q * d, divby=256),
                ),
            ),
        )
        o = cute.make_tensor(o_iter, o_layout)

        # Optional log2-space LSE output: (s, ((h_r, h_k), b)) f32, contiguous
        # in s (a (B, H_q, S_q) torch tensor).
        lse = None
        if cutlass.const_expr(lse_iter is not None):
            lse_layout = cute.make_layout(
                (s_q, ((h_r, h_k), b)),
                stride=(1, ((s_q, s_q * h_r), h_r * h_k * s_q)),
            )
            lse = cute.make_tensor(lse_iter, lse_layout)

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.p_dtype = self.q_dtype  # pv should has the same dtype
        # FP8 P pre-scale (fmha.py-style): bias the exp2 exponent so P fills
        # E4M3's [0, 448] range (2^8 = 256 <= 448); the bias cancels in
        # O = P@V / row_sum and only shifts a log2-space LSE by -8.
        self.p_prescale_log2 = 8.0 if self.p_dtype.width == 8 else 0.0
        self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.p_dtype.width

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            o.shape,
            self.cta_tiler,
            self.is_persistent,
            is_2cta=True,
        )

        self.q_major_mode = LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = LayoutEnum.from_tensor(o)
        cta_group = tcgen05.CtaGroup.TWO
        p_major_mode = OperandMajorMode.K
        p_source = tcgen05.OperandSource.TMEM
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.pv_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        # Uniform precision: K/V are loaded directly into the MMA-B SMEM layout
        # (proper swizzle for the dtype) with no separate dequant/transform buffer.
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.kv_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.p_dtype,
            self.qk_acc_stage,
        )
        p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.kv_stage,
        )

        tma_load_q_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        # K/V synchronization workaround: K/V are consumed by
        # the 2-CTA UMMA via PipelineTmaUmma, where only the LEADER CTA arms
        # and waits the transaction barrier. The donor's CtaGroup.ONE op (per-
        # CTA sync, correct for its per-CTA transform-warp consumers) leaves
        # the PEER CTA's K/V arrival unsynchronized with the leader's MMA ->
        # the UMMA can read a stale K/V half. Use the pair-scoped TMA like Q.
        tma_load_kv_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_q_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_kv_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_kv_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout) * cute.size(
            qk_tiled_mma.thr_id.shape
        )
        # Companion to the K/V workaround above: with the pair-scoped K/V TMA,
        # the leader's transaction barrier receives BOTH CTAs' bytes per
        # stage (as Q already does).
        self.tma_copy_kv_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout) * cute.size(
            qk_tiled_mma.thr_id.shape
        )

        @cute.struct
        class SharedStorage:
            """Static SMEM plan consumed by the kernel via struct offsets:
            per-stage mbarrier pairs for each producer/consumer pipeline,
            the TMEM-allocation holding buffer and dealloc mbarrier, and
            the swizzle-aligned tensor SMEM regions."""

            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
            mma_o_mbar_ptr: cute.struct.MemRange[Int64, self.pv_acc_stage * 2]
            tmem_dealloc_mbar: Int64
            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            o,
            lse,
            scale_softmax_log2,
            scale_output,
            window_size_left,
            window_size_right,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            p_tmem_layout,
            v_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        mLSE_ql: Optional[cute.Tensor],
        scale_softmax_log2: Float32,
        scale_output: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma desc
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        # Alloc
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.load_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        # Uniform precision: K/V are TMA-loaded straight into the MMA-B buffer and
        # consumed by the MMA (UMMA) warp directly -- no dequant/scale stages.
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kv_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.load_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.softmax_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.softmax_warp_ids),
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.correction_warp_ids),
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.softmax_warp_ids),
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.correction_warp_ids),
            ),
            barrier_storage=storage.sum_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_o_producer, mma_o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.pv_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.correction_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_o_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=self.threads_per_warp
            * len((self.mma_warp_id, *self.softmax_warp_ids, *self.correction_warp_ids)),
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_dealloc_sync_bar_id,
            num_threads=self.threads_per_warp * len(self.correction_warp_ids),
        )
        # Tensor memory dealloc barrier init
        tmem = TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV_ptr = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)

        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )
        # Row-max rides the same softmax->correction handoff when LSE is on.
        sMax = None
        if cutlass.const_expr(mLSE_ql is not None):
            sMax = smem.allocate_tensor(
                element_type=self.qk_acc_dtype,
                layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
                byte_alignment=128,
            )

        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        qk_acc_shape = pv_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        # (atomV, restM, restN, accStage)
        tStS = qk_tiled_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        pv_acc_shape = pv_thr_mma.partition_shape_C(cute.select(self.pv_mma_tiler, mode=[0, 1]))
        # (atomV, restM, restN)
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.iterations_pv,
                stride=self.pv_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)
        # Local_tile partition global tensors
        q_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # (bM, bK, restM, restK, loopM, loopK, loopL)
        gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2]))
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            block_in_cluster_coord_vmnk[2],
            q_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )
        kv_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # (bN, bK, loopN, loopK, loopL)
        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            block_in_cluster_coord_vmnk[1],
            kv_cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK_kdl, 0, 3),
        )
        # (bN, bK, loopN, loopK, loopL)
        gV_dkl = cute.flat_divide(mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2]))
        tOgV_dkl = pv_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            block_in_cluster_coord_vmnk[1],
            kv_cta_layout,
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tOgV_dkl, 0, 3),
        )
        # (bM, bN, loopM, loopN, loopL)
        gO_qdl = cute.flat_divide(mO_qdl, cute.select(self.pv_block_tiler, mode=[0, 1]))
        cO_qdl = cute.flat_divide(
            cute.make_identity_tensor(mO_qdl.shape),
            cute.select(self.pv_block_tiler, mode=[0, 1]),
        )
        gLSE_ql = None
        if cutlass.const_expr(mLSE_ql is not None):
            # (bM, loopM, loopL)
            gLSE_ql = cute.flat_divide(mLSE_ql, (self.pv_block_tiler[0],))
        seqlen_q = mQ_qdl.shape[0]
        seqlen_k = mK_kdl.shape[0]
        tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                    self.mask_type,
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                # The kv loop covers tiles [start, start + count);
                # get_trip_count is start-relative.
                kv_tile_start = fmha_utils.FusedMask.get_trip_start(
                    self.mask_type,
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                # ((atom_v, rest_v), RestK)
                tQgQ = tQgQ_qdl[None, mma_block_coord[0], None, mma_block_coord[2]]
                # ((atom_v, rest_v), RestN, RestK)
                tKgK = tKgK_kdl[None, None, None, mma_block_coord[2]]
                # ((atom_v, rest_v), RestN, RestK)
                tVgV = tVgV_dkl[None, None, None, mma_block_coord[2]]
                load_kv_producer, load_q_producer = load_qk_uniform(  # Q & K0
                    self.iterations_qk,
                    kv_step=kv_tile_start,
                    k_args=(tKgK, tKsK, tma_atom_k, load_kv_producer),
                    q_args=(tQgQ, tQsQ, tma_atom_q, load_q_producer),
                )
                for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                    load_kv_producer = load_qk_uniform(  # Ki
                        self.iterations_qk,
                        kv_step=kv_tile_start + step,
                        k_args=(tKgK, tKsK, tma_atom_k, load_kv_producer),
                    )
                    load_kv_producer = load_v_uniform(  # Vi-1
                        self.iterations_pv,
                        kv_step=kv_tile_start + step - 1,
                        v_args=(tVgV, tVsV, tma_atom_v, load_kv_producer),
                    )
                load_kv_producer = load_v_uniform(  # Vend
                    self.iterations_pv,
                    kv_step=kv_tile_start + seqlen_kv_loop_steps - 1,
                    v_args=(tVgV, tVsV, tma_atom_v, load_kv_producer),
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            load_kv_producer.tail()
            load_q_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                    self.mask_type,
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                load_q_releaser = load_q_consumer.clone()
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                if seqlen_kv_loop_steps > 1:
                    mma_s_producer, load_q_consumer, load_kv_consumer = mma_qk_uniform(  # QK0
                        self.iterations_qk,
                        qk_tiled_mma,
                        (tStS, tSrQ, tSrK),
                        (
                            mma_s_producer,
                            load_q_consumer,
                            None,
                            load_kv_consumer,
                        ),
                    )
                    for i in cutlass.range(1, seqlen_kv_loop_steps - 1, 1, unroll=1):
                        mma_s_producer, _, load_kv_consumer = mma_qk_uniform(  # QKi
                            self.iterations_qk,
                            qk_tiled_mma,
                            (tStS, tSrQ, tSrK),
                            (mma_s_producer, None, None, load_kv_consumer),
                        )
                        (
                            pv_tiled_mma,
                            p_mma_consumer,
                            mma_o_producer,
                            load_kv_consumer,
                        ) = self.mma_pv(  # PVi
                            (pv_tiled_mma, pv_thr_mma),
                            (tOtO_staged, tStS, tOrV, p_tmem_layout),
                            (p_mma_consumer, mma_o_producer, load_kv_consumer),
                        )
                    mma_s_producer, _, load_kv_consumer = mma_qk_uniform(  # QKend (release Q)
                        self.iterations_qk,
                        qk_tiled_mma,
                        (tStS, tSrQ, tSrK),
                        (
                            mma_s_producer,
                            None,
                            load_q_releaser,
                            load_kv_consumer,
                        ),
                    )
                    (
                        pv_tiled_mma,
                        p_mma_consumer,
                        mma_o_producer,
                        load_kv_consumer,
                    ) = self.mma_pv(  # PVend-1
                        (pv_tiled_mma, pv_thr_mma),
                        (tOtO_staged, tStS, tOrV, p_tmem_layout),
                        (p_mma_consumer, mma_o_producer, load_kv_consumer),
                    )
                else:
                    mma_s_producer, load_q_consumer, load_kv_consumer = mma_qk_uniform(  # QK0
                        self.iterations_qk,
                        qk_tiled_mma,
                        (tStS, tSrQ, tSrK),
                        (
                            mma_s_producer,
                            load_q_consumer,
                            load_q_releaser,
                            load_kv_consumer,
                        ),
                    )
                pv_tiled_mma, p_mma_consumer, mma_o_producer, load_kv_consumer = (
                    self.mma_pv(  # PVend
                        (pv_tiled_mma, pv_thr_mma),
                        (tOtO_staged, tStS, tOrV, p_tmem_layout),
                        (p_mma_consumer, mma_o_producer, load_kv_consumer),
                    )
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mma_s_producer.tail()
            mma_o_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax_warp_ids[0]:
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                    self.mask_type,
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                # A left window clips the FIRST visited tile too, so the loop
                # is [ld_mask][unmask][tl_mask] (fmha.py's structure).
                (
                    kv_tile_start,
                    _end_cnt,
                    ld_mask_steps,
                    unmask_steps,
                    _tl_mask_steps,
                ) = fmha_utils.FusedMask.get_masked_info(
                    self.mask_type,
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                cS_base = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                cS = cute.domain_offset((mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base)
                tScS = qk_thr_mma.partition_C(cS)
                tScS_static = qk_thr_mma.partition_C(cS_base)
                row_max = -Float32.inf
                row_max_prev = -Float32.inf
                row_sum = 0.0
                for step in cutlass.range(seqlen_kv_loop_steps, unroll=1):
                    # Mask coordinates must be ABSOLUTE kv positions.
                    cS_iter = cute.domain_offset(
                        (0, (kv_tile_start + step) * self.qk_mma_tiler[1]), cS
                    )
                    tScS_iter = qk_thr_mma.partition_C(cS_iter)
                    # Mask unless step is inside the clean middle segment.
                    # Unsigned-wrap trick: (step - ld) as Uint32 is huge for
                    # step < ld, so one compare covers both masked edges.
                    need_apply_mask = cutlass.Uint32(step - ld_mask_steps) >= cutlass.Uint32(
                        unmask_steps
                    )
                    # Si -> Pi
                    (
                        row_max,
                        row_sum,
                        mma_s_consumer,
                        p_mma_producer,
                        s_corr_producer,
                    ) = self.softmax_step(
                        (need_apply_mask, window_size_left, window_size_right),
                        (
                            row_max_prev,
                            row_sum,
                            seqlen_q,
                            seqlen_k,
                            scale_softmax_log2,
                        ),
                        (tStS, tScS_iter, tScS_static),
                        (mma_s_consumer, p_mma_producer, s_corr_producer),
                    )
                    row_max_prev = row_max
                sum_producer = self.store_sum(row_sum, row_max_prev, sSum, sMax, sum_producer)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            p_mma_producer.tail()
            s_corr_producer.tail()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.mma_warp_id and warp_idx >= self.correction_warp_ids[0]:
            # With the reduced warp count the launch default regs/thread is higher
            # than num_regs_correction, so this is a decrease (cf. base fmha.py).
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            while work_tile.is_valid_tile:
                curr_block_coord = work_tile.tile_idx
                mma_block_coord = (
                    curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    curr_block_coord[1],
                    curr_block_coord[2],
                )
                seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
                    self.mask_type,
                    mma_block_coord,
                    self.qk_mma_tiler,
                    seqlen_q,
                    seqlen_k,
                    window_size_left,
                    window_size_right,
                )
                gO_staged = gO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                cO_staged = cO_qdl[None, None, curr_block_coord[0], None, curr_block_coord[2]]
                cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                tScS = qk_thr_mma.partition_C(cS)
                # Empty step as the first step is no need for correction
                stats_handle = s_corr_consumer.wait_and_advance()
                stats_handle.release()
                for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
                    # Oi-1 -> Oi
                    mma_o_consumer, s_corr_consumer = self.correction_rescale(
                        scale_softmax_log2,
                        (s_corr_consumer, tStS, tScS),
                        (mma_o_consumer, tOtO_staged, cO_staged),
                        epi_tile,
                    )
                # O_partial -> O_final
                gLSE_staged = None
                if cutlass.const_expr(gLSE_ql is not None):
                    gLSE_staged = gLSE_ql[None, curr_block_coord[0], curr_block_coord[2]]
                mma_o_consumer, sum_consumer = self.correction_epilog(
                    (seqlen_q, scale_output, scale_softmax_log2),
                    (sum_consumer, sSum, sMax),
                    (mma_o_consumer, gO_staged, cO_staged, tOtO_staged, gLSE_staged),
                    epi_tile,
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            tmem.relinquish_alloc_permit()
            # Synchronize before TMEM dealloc (done by the caller)
            tmem_dealloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Empty
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx > self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        return

    @cute.jit
    def mma_pv(
        self,
        mma_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ):
        pv_tiled_mma, pv_thr_mma = mma_args
        tOtO_staged, tStS, tOrV, p_tmem_layout = tensor_args
        p_mma_consumer, mma_o_producer, load_kv_consumer = pipeline_args
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_leader_cta = cta_rank_in_cluster % 2 == 0
        if is_leader_cta:
            p_handle = p_mma_consumer.wait_and_advance()
            pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
            # Per-chunk O handoff: acquire/commit each dv chunk around its own
            # gemms (pipeline stage index == iter). The commit is issued before
            # the next chunk's gemms, so its tcgen05-tracked arrival fires as
            # soon as THIS chunk's MMAs complete — the correction warpgroup can
            # rescale/drain chunk `iter` while the MMA works on chunk `iter+1`.
            for iter in cutlass.range(self.iterations_pv, unroll=1):
                v_handle = load_kv_consumer.wait_and_advance()
                o_handle = mma_o_producer.acquire_and_advance()
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                tOtO_slice = tOtO_staged[None, None, None, iter]
                tStS_slice = tStS[None, None, None, p_handle.index]
                tP = cute.make_tensor(tStS_slice.iterator, p_tmem_layout.outer)
                tOrP = pv_thr_mma.make_fragment_A(tP)
                tOrP_slice = cute.make_tensor(
                    cute.recast_ptr(tStS_slice.iterator, dtype=self.p_dtype),
                    tOrP.layout,
                )
                tOrV_slice = tOrV[None, None, None, v_handle.index]
                num_kphases = cute.size(tOrV_slice, mode=[2])
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    kphase_coord = (None, None, kphase_idx)
                    cute.gemm(
                        pv_tiled_mma,
                        tOtO_slice,
                        tOrP_slice[kphase_coord],
                        tOrV_slice[kphase_coord],
                        tOtO_slice,
                    )
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                o_handle.commit()
                v_handle.release()
            p_handle.release()
        return pv_tiled_mma, p_mma_consumer, mma_o_producer, load_kv_consumer

    @cute.jit
    def softmax_step(
        self,
        mask_args: Tuple,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        need_apply_mask, window_size_left, window_size_right = mask_args
        row_max, row_sum, seqlen_q, seqlen_k, scale_softmax_log2 = value_args
        tStS, tScS, tScS_static = tensor_args
        mma_s_consumer, p_mma_producer, s_corr_producer = pipeline_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tScS_static_slice = tScS_static[(None, None), 0, 0]
        if cutlass.const_expr(self.use_ldstat):
            # tcgen05.ld.red (>sm_100f): the load itself reduces (MAX) across the
            # loaded elements, delivering one partial max per atom call.
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(32), redOp=tcgen05.TmemLoadRedOp.MAX
                ),
                self.qk_acc_dtype,
            )
        else:
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
            )
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADcS_static = thr_load.partition_D(tScS_static_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        num_chunks = cute.size(tTMEM_LOADrS, mode=[2])
        old_row_max = row_max
        if need_apply_mask:
            # Masked steps (tile boundary / causal edge): load all of S, mask,
            # then reduce. Minority of steps; keep the simple schedule.
            if cutlass.const_expr(self.use_ldstat):
                # A red atom needs copy_atom_call with a (data, red) tuple
                # destination; the fused max is discarded here since masking
                # changes the values — the max is recomputed post-mask.
                tTMEM_LOADrMax = cute.make_rmem_tensor(
                    cute.make_layout((1, num_chunks)), self.qk_acc_dtype
                )
                for c in cutlass.range_constexpr(num_chunks):
                    cute.copy_atom_call(
                        tmem_tiled_load,
                        tTMEM_LOADtS[None, 0, c],
                        (tTMEM_LOADrS[None, 0, c], tTMEM_LOADrMax[None, c]),
                    )
            else:
                cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()
            _apply_mask_hoisted(
                self.mask_type,
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                tTMEM_LOADcS_static,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
            row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        elif cutlass.const_expr(self.use_ldstat):
            # Unmasked fused-reduction path: each chunk load delivers its own MAX,
            # so the register FMAX chains disappear entirely — only the
            # num_chunks partial maxes are reduced at the end (fmha.py's
            # >sm_100f softmax path).
            tTMEM_LOADrMax = cute.make_rmem_tensor(
                cute.make_layout((1, num_chunks)), self.qk_acc_dtype
            )
            for c in cutlass.range_constexpr(num_chunks):
                cute.copy_atom_call(
                    tmem_tiled_load,
                    tTMEM_LOADtS[None, 0, c],
                    (tTMEM_LOADrS[None, 0, c], tTMEM_LOADrMax[None, c]),
                )
            cute.arch.fence_view_async_tmem_load()
            tile_row_max = tTMEM_LOADrMax.load().reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            row_max = cute.arch.fmax(row_max, tile_row_max)
        else:
            # Unmasked steps: load S in four 32-element chunks and interleave
            # the row-max with the in-flight loads. tcgen05.ld results are
            # register-scoreboarded, so the FMAX on chunk i only waits for
            # chunk i's registers while later chunks stay in flight. Two max
            # chains, mirroring fmha.py's sm100 softmax schedule.
            cute.copy(
                tmem_tiled_load,
                tTMEM_LOADtS[None, None, 0],
                tTMEM_LOADrS[None, None, 0],
            )
            cute.copy(
                tmem_tiled_load,
                tTMEM_LOADtS[None, None, 1],
                tTMEM_LOADrS[None, None, 1],
            )
            row_max_a = row_max
            row_max_b = -cutlass.Float32.inf
            for j in cutlass.range_constexpr(0, 32, 4):
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j])
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j + 1])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 2])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 3])
            cute.copy(
                tmem_tiled_load,
                tTMEM_LOADtS[None, None, 2],
                tTMEM_LOADrS[None, None, 2],
            )
            for j in cutlass.range_constexpr(32, 64, 4):
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j])
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j + 1])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 2])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 3])
            cute.copy(
                tmem_tiled_load,
                tTMEM_LOADtS[None, None, 3],
                tTMEM_LOADrS[None, None, 3],
            )
            for j in cutlass.range_constexpr(64, 96, 4):
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j])
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j + 1])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 2])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 3])
            cute.arch.fence_view_async_tmem_load()
            for j in cutlass.range_constexpr(96, 128, 4):
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j])
                row_max_a = cute.arch.fmax(row_max_a, tTMEM_LOADrS[j + 1])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 2])
                row_max_b = cute.arch.fmax(row_max_b, tTMEM_LOADrS[j + 3])
            row_max = cute.arch.fmax(row_max_a, row_max_b)
        s_handle.release()
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0

        stats_handle = s_corr_producer.acquire_and_advance()
        stats_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
        )
        stats_c_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
        )
        tOtStats = cute.make_tensor(tStS_slice.iterator + self.tilePlikeFP32, stats_layout)
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_store_stats_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_stats = tcgen05.make_tmem_copy(tmem_store_stats_atom, tOtStats)
        thr_tmem_store_stats = tiled_tmem_store_stats.get_slice(thread_idx)
        tTMEM_STOREcStats = thr_tmem_store_stats.partition_S(tOcStats)
        tTMEM_STORErStats = cute.make_rmem_tensor(tTMEM_STOREcStats.shape, self.qk_acc_dtype)
        tTMEM_STORErStats[0] = old_row_max
        tTMEM_STORErStats[1] = row_max_safe
        tTMEM_STOREtStats = thr_tmem_store_stats.partition_D(tOtStats)
        cute.copy(tiled_tmem_store_stats, tTMEM_STORErStats, tTMEM_STOREtStats)
        cute.arch.fence_view_async_tmem_store()
        stats_handle.commit()

        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale
        if cutlass.const_expr(self.p_prescale_log2 != 0.0):
            minus_row_max_scale = minus_row_max_scale + self.p_prescale_log2
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.p_dtype)

        # Software-pipelined exp2 loop (fmha.py-style): the scale FMA runs
        # FMA_COUNT elements ahead of the exp2 pair, while the P cvt and
        # the row-sum trail CVT_COUNT behind, so FMA / XU(MUFU) / ALU work
        # interleaves instead of serializing into phases.
        FMA_COUNT = 8
        CVT_COUNT = 4
        CVT_PER_STEP = 2
        SIZE = cute.size(tTMEM_LOADrS)
        tTMEM_LOADrS_cvt = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(CVT_PER_STEP))
        tTMEM_STORErP_cvt = cute.logical_divide(tTMEM_STORErP, cute.make_layout(CVT_PER_STEP))
        for i in cutlass.range_constexpr(0, FMA_COUNT, 2):
            tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1] = cute.arch.fma_packed_f32x2(
                (tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1]),
                (scale, scale),
                (minus_row_max_scale, minus_row_max_scale),
            )
        acc_scale_ = scale * (old_row_max - row_max_safe)
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        row_sum *= acc_scale
        local_row_sum = (row_sum, row_sum)
        for i in cutlass.range_constexpr(0, SIZE, 2):
            if cutlass.const_expr(i >= CVT_COUNT):
                g = (i - CVT_COUNT) // CVT_PER_STEP
                s_vec = tTMEM_LOADrS_cvt[None, g].load()
                tTMEM_STORErP_cvt[None, g].store(s_vec.to(self.p_dtype))
                local_row_sum = cute.arch.add_packed_f32x2(
                    local_row_sum,
                    (
                        tTMEM_LOADrS[i - CVT_COUNT],
                        tTMEM_LOADrS[i - CVT_COUNT + 1],
                    ),
                )
            tTMEM_LOADrS[i] = cute.math.exp2(tTMEM_LOADrS[i], fastmath=True)
            if cutlass.const_expr(i + FMA_COUNT < SIZE):
                (
                    tTMEM_LOADrS[i + FMA_COUNT],
                    tTMEM_LOADrS[i + FMA_COUNT + 1],
                ) = cute.arch.fma_packed_f32x2(
                    (
                        tTMEM_LOADrS[i + FMA_COUNT],
                        tTMEM_LOADrS[i + FMA_COUNT + 1],
                    ),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                )
            tTMEM_LOADrS[i + 1] = cute.math.exp2(tTMEM_LOADrS[i + 1], fastmath=True)
        # Trailing cvt + row-sum for the last CVT_COUNT elements.
        for i in cutlass.range_constexpr(SIZE - CVT_COUNT, SIZE, 2):
            g = i // CVT_PER_STEP
            s_vec = tTMEM_LOADrS_cvt[None, g].load()
            tTMEM_STORErP_cvt[None, g].store(s_vec.to(self.p_dtype))
            local_row_sum = cute.arch.add_packed_f32x2(
                local_row_sum, (tTMEM_LOADrS[i], tTMEM_LOADrS[i + 1])
            )

        p_handle = p_mma_producer.acquire_and_advance()
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tilePlikeFP32 = tStS_slice.shape[1] // Float32.width * self.p_dtype.width
        tStS_P_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], tilePlikeFP32))
        )
        tStS_P = cute.make_tensor(tStS_slice.iterator, tStS_P_layout)
        tScS_P_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], tilePlikeFP32))
        )
        tScS_P = cute.make_tensor(tScS_slice.iterator, tScS_P_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtP = thr_store.partition_D(tStS_P)
        tTMEM_STOREcS = thr_store.partition_S(tScS_P)
        tTMEM_STORErP_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErP.iterator, dtype=self.qk_acc_dtype),
            tTMEM_STOREcS.shape,
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErP_, tTMEM_STOREtP)
        cute.arch.fence_view_async_tmem_store()

        p_handle.commit()
        row_sum = local_row_sum[0] + local_row_sum[1]
        return row_max, row_sum, mma_s_consumer, p_mma_producer, s_corr_producer

    @cute.jit
    def correction_rescale(
        self,
        scale_softmax_log2: Float32,
        stats_args: tuple,
        o_args: tuple,
        epi_tile: cute.Tile,
    ) -> pipeline.PipelineConsumer:
        (s_corr_consumer, tStS, tScS) = stats_args
        (mma_o_consumer, tOtO_staged, cO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))

        stats_handle = s_corr_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, stats_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        stats_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
        )
        stats_c_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
        )
        tOtStats = cute.make_tensor(tStS_slice.iterator + self.tilePlikeFP32, stats_layout)
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_load_stats_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load_stats = tcgen05.make_tmem_copy(tmem_load_stats_atom, tOtStats)
        thr_tmem_load_stats = tiled_tmem_load_stats.get_slice(thread_idx)
        tTMEM_LOADtStats = thr_tmem_load_stats.partition_S(tOtStats)
        tTMEM_LOADcStats = thr_tmem_load_stats.partition_D(tOcStats)
        tTMEM_LOADrStats = cute.make_rmem_tensor(tTMEM_LOADcStats.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load_stats, tTMEM_LOADtStats, tTMEM_LOADrStats)

        scale = scale_softmax_log2 * (tTMEM_LOADrStats[0] - tTMEM_LOADrStats[1])
        scale = cute.math.exp2(scale, fastmath=True)
        stats_handle.release()
        # Per-chunk O handoff: wait/rescale/release one dv chunk at a time so
        # chunk `iter` is back in the MMA's hands while chunk `iter+1` is still
        # being rescaled. The Rep16 double-buffered load/store pipeline inside
        # each chunk is unchanged.
        for iter in cutlass.range(self.iterations_pv, unroll_full=True):
            o_handle = mma_o_consumer.wait_and_advance()
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            cO = cO_staged[None, None, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
            thr_load = tmem_tiled_load.get_slice(thread_idx)
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_store_atom = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_epi)
            thr_store = tmem_store_atom.get_slice(thread_idx)
            tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
            tTMEM_LOADcO = thr_load.partition_D(cO_epi)
            tTMEM_STOREtO = thr_store.partition_D(tOtO_epi)
            tTMrO = cute.make_rmem_tensor_like(
                cute.append(
                    cute.make_layout(tTMEM_LOADcO[None, 0, 0].shape),
                    cute.make_layout(2, stride=cute.size(tTMEM_LOADcO[None, 0, 0].shape)),
                ),
                self.pv_acc_dtype,
            )
            tTMEM_LOADtO_0 = tTMEM_LOADtO[None, 0, 0]
            cute.copy(tmem_tiled_load, tTMEM_LOADtO_0, tTMrO[None, 0])
            iter_num = cute.size(tTMEM_LOADtO, mode=[1])
            for i in cutlass.range(1, iter_num, unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                cute.copy(tmem_tiled_load, tTMEM_LOADtO_i, tTMrO[None, i % 2])
                for j in cutlass.range(
                    cute.size(tTMrO, mode=[0]), unroll_full=True, vectorize=True
                ):
                    tTMrO[j, (i - 1) % 2] = tTMrO[j, (i - 1) % 2] * scale
                tTMEM_STOREtO_prev_i = tTMEM_STOREtO[None, i - 1, 0]
                cute.copy(tmem_store_atom, tTMrO[None, (i - 1) % 2], tTMEM_STOREtO_prev_i)

            for j in cutlass.range(cute.size(tTMrO, mode=[0]), unroll_full=True, vectorize=True):
                tTMrO[j, (iter_num - 1) % 2] = tTMrO[j, (iter_num - 1) % 2] * scale
            cute.copy(
                tmem_store_atom,
                tTMrO[None, (iter_num - 1) % 2],
                tTMEM_STOREtO[None, iter_num - 1, 0],
            )
            cute.arch.fence_view_async_tmem_store()
            o_handle.release()
        return mma_o_consumer, s_corr_consumer

    @cute.jit
    def correction_epilog(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        (seqlen_q, scale_output, scale_softmax_log2) = value_args
        (sum_consumer, sSum, sMax) = sum_args
        (mma_o_consumer, gO_staged, cO_staged, tOtO_staged, gLSE) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        row_max = Float32(0.0)
        if cutlass.const_expr(sMax is not None):
            row_max = sMax[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        scale = scale_output / row_sum
        if cutlass.const_expr(gLSE is not None):
            # log2-space LSE (flashinfer/trtllm convention); equals
            # (ln(row_sum) + scale_softmax * row_max) * log2(e).
            lse = cute.math.log2(row_sum, fastmath=True) + scale_softmax_log2 * row_max
            if cutlass.const_expr(self.p_prescale_log2 != 0.0):
                # row_sum was inflated by 2^prescale for the fp8 P conversion.
                lse = lse - self.p_prescale_log2
            if cute.elem_less(cO_staged[thread_idx, 0, 0][0], seqlen_q):
                gLSE[thread_idx] = lse
        # Per-chunk O handoff: wait/drain/release one dv chunk at a time so
        # the next work item's PV(iter 0) is gated only on chunk 0's drain,
        # not the whole epilogue.
        for iter in cutlass.range(self.iterations_pv):
            o_handle = mma_o_consumer.wait_and_advance()
            gO = gO_staged[None, None, iter]
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            gO_epi = cute.zipped_divide(gO, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADgO = thr_tmem_load.partition_D(gO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            for i in cutlass.range(cute.size(tTMEM_LOADtO, mode=[1]), unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                tTMrO = cute.make_rmem_tensor(tTMEM_LOADcO[None, 0, i].shape, self.pv_acc_dtype)
                cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                for j in cutlass.range(cute.size(tTMrO), unroll_full=True, vectorize=True):
                    tTMrO[j] = tTMrO[j] * scale
                tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                o_vec = tTMrO.load()
                tSMrO.store(o_vec.to(self.o_dtype))
                if cute.elem_less(tTMEM_LOADcO_i[0][0], seqlen_q):
                    cute.autovec_copy(tSMrO, tTMEM_LOADgO_i)
            # Release this chunk as soon as its slices are drained (the loaded
            # data is consumed via registers by the scale FMAs above, so the
            # register scoreboard orders the loads before this release — same
            # discipline as the original whole-buffer release).
            o_handle.release()
        return mma_o_consumer, sum_consumer

    def store_sum(self, row_sum, row_max, sSum, sMax, sum_producer):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_producer.acquire_and_advance()
        sSum[thread_idx] = row_sum
        if cutlass.const_expr(sMax is not None):
            sMax[thread_idx] = row_max
        cute.arch.fence_view_async_shared()
        sum_handle.commit()
        return sum_producer

    def can_implement(
        self,
        q_shape: Tuple[int, int, int, int],
        k_shape: Tuple[int, int, int, int],
        q_dtype: Type[cutlass.Numeric],
        kv_dtype: Type[cutlass.Numeric],
        o_dtype: Type[cutlass.Numeric],
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """Validate the uniform-precision d=256 configuration.

        :param q_shape: Shape of the query tensor (B, H_q, S_q, D).
        :param k_shape: Shape of the key tensor (B, H_k, S_k, D).
        :param q_dtype: Data type of Q (BFloat16 or Float8E4M3FN).
        :param kv_dtype: Data type of K/V (must equal q_dtype -- uniform precision).
        :param o_dtype: Data type of O (must equal q_dtype).
        :param qk_acc_dtype: QK accumulator dtype (Float32).
        :param pv_acc_dtype: PV accumulator dtype (Float32).
        :return: True if the kernel can be implemented, False otherwise.
        """
        try:
            if q_dtype not in {cutlass.BFloat16, cutlass.Float8E4M3FN}:
                raise testing.CantImplementError("Unsupported q_dtype (need bf16 or e4m3)")
            if kv_dtype is not q_dtype:
                raise testing.CantImplementError(
                    "Unsupported kv_dtype (uniform precision: must equal q_dtype)"
                )
            if o_dtype is not q_dtype:
                raise testing.CantImplementError(
                    "Unsupported o_dtype (uniform precision: must equal q_dtype)"
                )
            if qk_acc_dtype not in {cutlass.Float32}:
                raise testing.CantImplementError("Unsupported qk_acc_dtype")
            if pv_acc_dtype not in {cutlass.Float32}:
                raise testing.CantImplementError("Unsupported pv_acc_dtype")
            b, h_q, s_q, d = q_shape
            b_, h_k, s_k, d_ = k_shape
            if b <= 0 or h_q <= 0 or h_k <= 0 or s_q <= 0 or s_k <= 0:
                raise testing.CantImplementError("batch, head, and sequence sizes must be positive")
            if b != b_:
                raise testing.CantImplementError("q & k must share batch size")
            if d != d_ or d != 256:
                raise testing.CantImplementError("This kernel only supports d=256")
            if h_q % h_k != 0:
                raise testing.CantImplementError("h_q must be divisible by h_k")
            # Bottom-right aligned windows give the first q tiles no K/V tile once
            # s_q - s_k reaches a tile; the correction warp would then wait on a softmax
            # step that never runs.
            if self.mask_type is fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE and s_q > s_k:
                raise testing.CantImplementError("WINDOW_MASK_INFERENCE requires s_q <= s_k")
        except testing.CantImplementError:
            return False
        return True
