# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The absorbed MLA's per-head GEMMs of the sparse MLA prefill with their consumers in the epilogue (CuTe DSL).

Two batched GEMMs (batch = query head) of the context call, each with the pass that followed it folded in:

* ``wuk_fp8``: ``q_nope[:, h] @ W_UK[h]`` (``[t, 128] @ [128, 512]``) stored as the E4M3 latent columns of the FP8
  query the context FMHA reads ``[t, pad, 576]``: each FP32 sum rounded to BF16 (round to nearest even, the BMM's
  output rounding), widened, clamped to +-448 (``min.NaN`` / ``max.NaN``: a NaN stays a NaN) and converted
  (``cvt.rn.satfinite.e4m3x2``), the arithmetic of ``rubin_mla_ctx_q_fp8`` on the BMM's BF16 output. The CTAs of
  the padding heads ``heads <= h < pad`` store zeros. No BF16 query is written.
* ``wuv_gate``: ``lat[:, h] @ W_UV[h]`` (``[t, 512] @ [512, 128]``), each sum rounded to BF16 and gated with
  ``g[:, h * 128 : (h + 1) * 128]`` by ``gate_mul_pair`` (``k31_sigmoid_gate_mul``'s per-element PTX), stored as the
  ``o_proj`` input ``[t, heads * 128]``. Rows ``t >= rows_a`` (padding rows past the attention's rows) take a +0
  attention output, as the two-pass form zeroes them before the gate. The ungated rows are never written.

The GEMM: tiles of 128 rows x ``tile_n`` columns of one head (``tcgen05`` UMMA 128 x ``tile_n`` x 16, BF16 operands
from TMA-loaded 64-wide K chunks, FP32 accumulator in tensor memory). The K chunks are issued in order and each
one's four MMAs in order, so every output element is accumulated over K in order, 16 products per MMA, as cuBLAS's
(non split-K) tcgen05 kernels accumulate it; the module's tests check the bytes against the BMM + pass forms at the
production shapes. The kernel is persistent and warp specialized (:class:`HeadGemmKernel`): a TMA warp, an MMA warp
and epilogue warps on two tensor-memory accumulators, so a tile's epilogue runs under the next tile's loads and
MMAs. The epilogue stages each output tile in shared memory and stores it in coalesced 16-byte row chunks.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from ..activation.sigmoid_gate_mul import gate_mul_pair

#: Epilogues.
MODE_WUK_FP8 = "wuk_fp8"
MODE_WUV_GATE = "wuv_gate"

#: Accumulator rows per tile (UMMA M, one TMEM lane each) and the K extent of one TMA chunk.
TILE_M = 128
TILE_K = 64
#: Tensor-memory accumulators (the MMA warp fills one while the epilogue drains the other).
_ACC_BUFS = 2
#: Warp 0 loads, warp 1 issues the MMAs, the epilogue warps follow.
_EPI_WARP0 = 2
#: The epilogue warps' named barrier.
_EPI_BARRIER = 1
#: Bytes of one E4M3 / BF16 element.
_FP8_BYTES = 1
_BF16_BYTES = 2
#: Bytes of one output (and gate) tile row, its 16-byte chunks, and the tile rows of one coalesced pass.
_ROW_BYTES = 256
_CHUNKS = _ROW_BYTES // 16
#: Row pitch of the tile staged in shared memory: the 16 bytes of padding make a warp's per-row 16-byte
#: accesses (one row per lane) conflict-free.
_PITCH = _ROW_BYTES + 16
#: The clamp bounds of the E4M3 conversion (+-448) as FP32 bit patterns.
_E4M3_MAX_HEX = "0f43E00000"
_E4M3_MIN_HEX = "0fC3E00000"


class HeadGemmSpec(NamedTuple):
    """One compiled specialization: the epilogue, the GEMM extents and every stride (elements)."""

    mode: str
    heads: int  # batch: query heads with a GEMM
    pad: int  # heads of the grid (wuk_fp8: the padding heads' CTAs store zeros)
    n: int  # output columns per head
    k: int
    tile_n: int
    stages: int  # K chunks in the TMA ring
    epi_warps: int  # epilogue warps (a multiple of 4: each lane quadrant's columns split over epi_warps / 4)
    a_tok: int  # A [rows, heads, k]: row and head strides (unit K stride)
    a_head: int
    w_n: int  # W [heads, k, n] as B (n, k, head): strides
    w_k: int
    w_head: int
    o_tok: int  # output: row and head strides (unit column stride)
    o_head: int
    g_row: int  # wuv_gate: the gate's row stride (unit column stride); else 0


# ---------------------------------------------------------------------------
# Inline PTX
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, side_effects, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


_CLAMP = (
    "max.NaN.f32 f{i}, f{i}, "
    + _E4M3_MIN_HEX
    + ";\nmin.NaN.f32 f{i}, f{i}, "
    + _E4M3_MAX_HEX
    + ";\n"
)
# Four FP32 sums -> one word of four E4M3 codes (byte i = sum i): BF16 rounding, widening, clamp, conversion.
_E4M3X4_PTX = (
    "{\n.reg .b32 t0, t1;\n.reg .b16 l0, h0, l1, h1, e0, e1;\n.reg .f32 f0, f1, f2, f3;\n"
    "cvt.rn.bf16x2.f32 t0, $2, $1;\ncvt.rn.bf16x2.f32 t1, $4, $3;\n"
    "mov.b32 {l0, h0}, t0;\nmov.b32 {l1, h1}, t1;\n"
    "cvt.f32.bf16 f0, l0;\ncvt.f32.bf16 f1, h0;\ncvt.f32.bf16 f2, l1;\ncvt.f32.bf16 f3, h1;\n"
    + "".join(_CLAMP.format(i=i) for i in range(4))
    + "cvt.rn.satfinite.e4m3x2.f32 e0, f1, f0;\ncvt.rn.satfinite.e4m3x2.f32 e1, f3, f2;\n"
    "mov.b32 $0, {e0, e1};\n}"
)


@dsl_user_op
def _e4m3x4(a, b, c, d, *, loc=None, ip=None):
    """One word of four E4M3 codes from four FP32 sums (byte 0 from ``a``)."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(v, loc=loc, ip=ip) for v in (a, b, c, d)],
            _E4M3X4_PTX,
            "=r,f,f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2(a, b, *, loc=None, ip=None):
    """Two FP32 sums rounded to BF16 (nearest even) in one word, ``a`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_v4(addr, pred, *, loc=None, ip=None):
    """16 bytes at ``addr`` (zeros where ``pred`` is 0, without touching memory)."""
    ty = llvm.StructType.get_literal([T.i32()] * 4)
    res = _asm(
        ty,
        [_lv(addr, loc=loc, ip=ip), _iv(pred, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\n"
        "mov.b32 $2, 0;\nmov.b32 $3, 0;\n@p ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        "=r,=r,=r,=r,l,r",
        True,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _st_v4(addr, words, pred, *, loc=None, ip=None):
    """16 bytes to ``addr`` where ``pred`` is not 0."""
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)]
        + [_iv(w, loc=loc, ip=ip) for w in words]
        + [_iv(pred, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $5, 0;\n@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n}",
        "l,r,r,r,r,r",
        True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v4(addr, words, *, loc=None, ip=None):
    """16 bytes to shared memory."""
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_iv(w, loc=loc, ip=ip) for w in words],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_v4(addr, *, loc=None, ip=None):
    """16 bytes from shared memory."""
    ty = llvm.StructType.get_literal([T.i32()] * 4)
    res = _asm(
        ty,
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        True,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _tmem_ld32(taddr, *, loc=None, ip=None):
    """32 consecutive FP32 accumulator columns of this thread's TMEM lane (one wait)."""
    regs = ", ".join(f"${i}" for i in range(32))
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 32),
        [_iv(taddr, loc=loc, ip=ip)],
        "tcgen05.fence::after_thread_sync;\n"
        f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{regs}}}, [$32];\n"
        "tcgen05.wait::ld.sync.aligned;",
        ",".join(["=f"] * 32) + ",r",
        True,
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(32)]


@dsl_user_op
def _tmem_fence_before_sync(*, loc=None, ip=None):
    _asm(None, [], "tcgen05.fence::before_thread_sync;", "", True, loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class HeadGemmKernel:
    """The per-head GEMM with the ``spec.mode`` epilogue (see the module docstring).

    Persistent and warp specialized: CTA ``c`` of ``G`` takes tiles ``c, c + G, ...`` (tile order: wuk_fp8 the
    two column halves of one row tile and head, then the row tiles, then the heads, the padding heads last;
    wuv_gate the row tiles, then the heads). Warp 0 TMA-loads the K chunks of A and B into a ring of
    ``stages`` stages, warp 1 issues the MMAs into one of two tensor-memory accumulators, and the epilogue warps
    drain the other: tensor memory -> registers -> the staging tile in shared memory -> coalesced global
    stores (wuv_gate: the gate tile is loaded in coalesced row chunks before its accumulator is ready).
    """

    def __init__(self, spec: HeadGemmSpec):
        if spec.mode not in (MODE_WUK_FP8, MODE_WUV_GATE):
            raise ValueError(f"unknown epilogue {spec.mode!r}")
        if spec.k % TILE_K or spec.n % spec.tile_n or spec.tile_n % 32:
            raise ValueError(f"unsupported extents: {spec}")
        if spec.mode == MODE_WUV_GATE and spec.pad != spec.heads:
            raise ValueError("wuv_gate has no padding heads")
        self.spec = spec
        self.fp8 = spec.mode == MODE_WUK_FP8
        self.mma_tiler = (TILE_M, spec.tile_n, TILE_K)
        self.chunks = spec.k // TILE_K
        self.stages = spec.stages
        self.stage_bytes = (TILE_M + spec.tile_n) * TILE_K * _BF16_BYTES
        self.tmem_cols = _ACC_BUFS * spec.tile_n
        if self.tmem_cols & (self.tmem_cols - 1) or self.tmem_cols > 512:
            raise ValueError(f"the two accumulators need a power-of-two TMEM allocation: {spec}")
        self.out_bytes = _FP8_BYTES if self.fp8 else _BF16_BYTES
        if spec.tile_n * self.out_bytes != _ROW_BYTES:
            raise ValueError(f"an output tile row must be {_ROW_BYTES} bytes: {spec}")
        # Epilogue warps: one row (TMEM lane) per thread, the tile's columns split over ``col_split`` warps of
        # each lane quadrant.
        self.epi_warps = spec.epi_warps
        self.col_split = self.epi_warps // 4
        self.epi_threads = 32 * self.epi_warps
        self.threads = 32 * (_EPI_WARP0 + self.epi_warps)
        self.rpi = self.epi_threads // _CHUNKS
        if self.epi_warps % 4 or (spec.tile_n // self.col_split) % 16 or TILE_M % self.rpi:
            raise ValueError(f"unsupported epilogue split: {spec}")

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        rows_out: Int32,
        rows_a: Int32,
        out_addr: Int64,
        g_addr: Int64,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        s = self.spec
        mA = cute.make_tensor(
            a_ptr, cute.make_layout((rows_a, s.k, s.heads), stride=(s.a_tok, 1, s.a_head))
        )
        mB = cute.make_tensor(
            w_ptr, cute.make_layout((s.n, s.k, s.heads), stride=(s.w_n, s.w_k, s.w_head))
        )
        b_major = OperandMajorMode.K if s.w_k == 1 else OperandMajorMode.MN
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            OperandMajorMode.K,
            b_major,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.stages
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mA,
            cute.slice_(a_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mB,
            cute.slice_(b_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        @cute.struct
        class SharedStorage:
            full: cute.struct.MemRange[cutlass.Int64, self.stages]
            empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, _ACC_BUFS]
            acc_empty: cute.struct.MemRange[cutlass.Int64, _ACC_BUFS]
            tmem_ready: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]
            stage: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, TILE_M * _PITCH], 128]
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(b_layout.outer)], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            a_layout,
            b_layout,
            rows_out,
            rows_a,
            out_addr,
            g_addr,
        ).launch(
            grid=(ctas, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.jit
    def tile_coords(self, t: Int32, m_tiles: Int32):
        """(row tile, column tile, head) of tile ``t``."""
        n_tiles = self.spec.n // self.spec.tile_n
        m_tile = t % m_tiles
        n_tile = (t // m_tiles) % n_tiles
        head = t // (m_tiles * n_tiles)
        if cutlass.const_expr(self.fp8):
            n_tile = t % n_tiles
            m_tile = (t // n_tiles) % m_tiles
            head = t // (m_tiles * n_tiles)
        return m_tile, n_tile, head

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        rows_out: Int32,
        rows_a: Int32,
        out_addr: Int64,
        g_addr: Int64,
    ):
        s = self.spec
        stages = self.stages
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        m_tiles = (rows_out + TILE_M - 1) // TILE_M
        per_head = m_tiles * (s.n // s.tile_n)
        # This CTA's tiles with a GEMM (t < real) and with the padding heads' zeros.
        mine_real = (per_head * s.heads + ctas - 1 - cta) // ctas
        mine_all = (per_head * s.pad + ctas - 1 - cta) // ctas

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        full = storage.full.data_ptr()
        empty = storage.empty.data_ptr()
        acc_full = storage.acc_full.data_ptr()
        acc_empty = storage.acc_empty.data_ptr()
        tmem_ready = storage.tmem_ready.data_ptr()
        tmem_holding = storage.tmem_holding.data_ptr()
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(stages):
                    cute.arch.mbarrier_init(full + i, 1)
                    cute.arch.mbarrier_init(empty + i, 1)
                for i in cutlass.range_constexpr(_ACC_BUFS):
                    cute.arch.mbarrier_init(acc_full + i, 1)
                    cute.arch.mbarrier_init(acc_empty + i, self.epi_warps)
                cute.arch.mbarrier_init(tmem_ready, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
            gA = cute.local_tile(
                mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            gB = cute.local_tile(
                mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            thr_mma = tiled_mma.get_slice(0)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(thr_mma.partition_B(gB), 0, 3),
            )
            chunk = Int32(0)
            for i in cutlass.range(mine_real, unroll=1):
                m_tile, n_tile, head = self.tile_coords(cta + i * ctas, m_tiles)
                for c in cutlass.range_constexpr(self.chunks):
                    slot = chunk % stages
                    if chunk >= stages:
                        cute.arch.mbarrier_wait(empty + slot, ((chunk // stages) & 1) ^ 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(full + slot, self.stage_bytes)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, m_tile, c, head)],
                        tAsA[(None, slot)],
                        tma_bar_ptr=full + slot,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, n_tile, c, head)],
                        tBsB[(None, slot)],
                        tma_bar_ptr=full + slot,
                    )
                    chunk = chunk + 1
        elif warp == 1:
            cute.arch.alloc_tmem(self.tmem_cols, tmem_holding)
            cute.arch.relinquish_tmem_alloc_permit()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tmem_ready)
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding
            )
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_all = cute.make_tensor(
                acc_tmem_ptr,
                tiled_mma.make_fragment_C(cute.append(acc_shape, _ACC_BUFS)).layout,
            )
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            num_kphases = cute.size(tCrA, mode=[2])
            chunk = Int32(0)
            for i in cutlass.range(mine_real, unroll=1):
                buf = i % _ACC_BUFS
                if i >= _ACC_BUFS:
                    cute.arch.mbarrier_wait(acc_empty + buf, ((i // _ACC_BUFS) & 1) ^ 1)
                tCtAcc = tCtAcc_all[(None, None, None, buf)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for c in cutlass.range_constexpr(self.chunks):
                    slot = chunk % stages
                    cute.arch.mbarrier_wait(full + slot, (chunk // stages) & 1)
                    for kp in cutlass.range_constexpr(num_kphases):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[(None, None, kp, slot)],
                            tCrB[(None, None, kp, slot)],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(empty + slot)
                    chunk = chunk + 1
                with cute.arch.elect_one():
                    tcgen05.commit(acc_full + buf)
        else:
            self.epilogue(
                storage.stage.data_ptr().toint(),
                acc_full,
                acc_empty,
                tmem_ready,
                tmem_holding,
                warp,
                lane,
                tidx - 32 * _EPI_WARP0,
                cta,
                ctas,
                m_tiles,
                mine_real,
                mine_all,
                rows_out,
                rows_a,
                out_addr,
                g_addr,
            )

        # Every warp: the epilogue has read the last accumulator before warp 1 frees tensor memory.
        cute.arch.barrier()
        if warp == 1:
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(
                    cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding
                ),
                self.tmem_cols,
            )

    @cute.jit
    def epilogue(
        self,
        stage: Int32,
        acc_full,
        acc_empty,
        tmem_ready,
        tmem_holding,
        warp: Int32,
        lane: Int32,
        etid: Int32,
        cta: Int32,
        ctas: Int32,
        m_tiles: Int32,
        mine_real: Int32,
        mine_all: Int32,
        rows_out: Int32,
        rows_a: Int32,
        out_addr: Int64,
        g_addr: Int64,
    ):
        s = self.spec
        # Thread (warp w, lane l): accumulator row 32 (w % 4) + l (the TMEM lane quadrant of warp w), columns
        # [part * cols, (part + 1) * cols) of the tile.
        cols = s.tile_n // self.col_split
        part = (warp - _EPI_WARP0) // 4
        my_row = (warp % 4) * 32 + lane
        mine = stage + my_row * _PITCH + part * cols * self.out_bytes
        # Coalesced row-chunk mapping: thread e moves chunk e % CHUNKS of rows e // CHUNKS + rpi * it.
        c_chunk = etid % _CHUNKS
        c_row = etid // _CHUNKS
        o_row = s.o_tok * self.out_bytes
        cute.arch.mbarrier_wait(tmem_ready, 0)
        tmem_base = _lds_b32(tmem_holding.toint()) + (((warp % 4) * 32) << 16) + part * cols
        for i in cutlass.range(mine_real, unroll=1):
            m_tile, n_tile, head = self.tile_coords(cta + i * ctas, m_tiles)
            row0 = m_tile * TILE_M
            col0 = n_tile * s.tile_n
            tile_out = (
                out_addr
                + (Int64(row0) * s.o_tok + Int64(head) * s.o_head + Int64(col0)) * self.out_bytes
            )
            buf = i % _ACC_BUFS
            lane_base = tmem_base + buf * s.tile_n
            gate = []
            if cutlass.const_expr(not self.fp8):
                tile_g = (
                    g_addr + (Int64(row0) * s.g_row + Int64(head) * s.n + Int64(col0)) * _BF16_BYTES
                )
                for it in cutlass.range_constexpr(TILE_M // self.rpi):
                    r = c_row + it * self.rpi
                    gate.append(
                        _ld_v4(
                            tile_g + Int64(r) * (s.g_row * _BF16_BYTES) + c_chunk * 16,
                            Int32(row0 + r < rows_out),
                        )
                    )
            cute.arch.mbarrier_wait(acc_full + buf, (i // _ACC_BUFS) & 1)
            if cutlass.const_expr(self.fp8):
                # 32 columns per tensor-memory load: 32 E4M3 codes, two 16-byte chunks.
                for j in cutlass.range_constexpr(cols // 32):
                    v = _tmem_ld32(lane_base + j * 32)
                    words = []
                    for q in cutlass.range_constexpr(8):
                        words.append(_e4m3x4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]))
                    _sts_v4(mine + j * 32, words[:4])
                    _sts_v4(mine + j * 32 + 16, words[4:])
            else:
                for it in cutlass.range_constexpr(TILE_M // self.rpi):
                    _sts_v4(stage + (c_row + it * self.rpi) * _PITCH + c_chunk * 16, gate[it])
                cute.arch.barrier(barrier_id=_EPI_BARRIER, number_of_threads=self.epi_threads)
                # +0 attention output on the rows past the attention's rows.
                keep = Int32(0) - Int32(row0 + my_row < rows_a)
                # 32 columns per tensor-memory load: 16 BF16 pairs, four 16-byte chunks of the staged gate.
                for j in cutlass.range_constexpr(cols // 32):
                    v = _tmem_ld32(lane_base + j * 32)
                    g = []
                    for c in cutlass.range_constexpr(4):
                        g.append(_lds_v4(mine + j * 64 + c * 16))
                    res = []
                    for q in cutlass.range_constexpr(16):
                        x = _bf16x2(v[2 * q], v[2 * q + 1]) & keep
                        res.append(gate_mul_pair(x, g[q // 4][q % 4]))
                    for c in cutlass.range_constexpr(4):
                        _sts_v4(mine + j * 64 + c * 16, res[4 * c : 4 * c + 4])
            _tmem_fence_before_sync()
            cute.arch.barrier(barrier_id=_EPI_BARRIER, number_of_threads=self.epi_threads)
            # The accumulator is drained: the MMA warp may reuse it.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(acc_empty + buf)
            for it in cutlass.range_constexpr(TILE_M // self.rpi):
                r = c_row + it * self.rpi
                _st_v4(
                    tile_out + Int64(r) * o_row + c_chunk * 16,
                    _lds_v4(stage + r * _PITCH + c_chunk * 16),
                    Int32(row0 + r < rows_out),
                )
            # The staging tile is free for the next tile.
            cute.arch.barrier(barrier_id=_EPI_BARRIER, number_of_threads=self.epi_threads)
        # wuk_fp8: the padding heads' tiles, zeros.
        zero = [Int32(0)] * 4
        for i in cutlass.range(mine_real, mine_all, unroll=1):
            m_tile, n_tile, head = self.tile_coords(cta + i * ctas, m_tiles)
            row0 = m_tile * TILE_M
            tile_out = (
                out_addr
                + (Int64(row0) * s.o_tok + Int64(head) * s.o_head + Int64(n_tile * s.tile_n))
                * self.out_bytes
            )
            for it in cutlass.range_constexpr(TILE_M // self.rpi):
                r = c_row + it * self.rpi
                _st_v4(tile_out + Int64(r) * o_row + c_chunk * 16, zero, Int32(row0 + r < rows_out))


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _compiled(spec: HeadGemmSpec):
    kernel = HeadGemmKernel(spec)
    dummy = 1 << 20
    ptr = make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)
    addr = Int64(dummy)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel, ptr, ptr, Int32(TILE_M), Int32(TILE_M), addr, addr, Int32(1), stream
    )


@functools.cache
def _num_sms(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _launch(
    spec: HeadGemmSpec,
    a: torch.Tensor,
    w: torch.Tensor,
    rows_out: int,
    rows_a: int,
    out_ptr: int,
    g_ptr: int,
) -> None:
    tiles = -(-rows_out // TILE_M) * (spec.n // spec.tile_n) * spec.pad
    ctas = min(tiles, _num_sms(a.device.index))
    _compiled(spec)(
        make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int32(rows_out),
        Int32(rows_a),
        Int64(out_ptr),
        Int64(g_ptr),
        Int32(ctas),
        cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream),
    )


#: wuk_fp8 tiling: a 128 x 256 accumulator (half a head's 512 latent columns), a three-chunk ring (one and a
#: half tiles: the next tile's first chunk loads under this one's MMAs), sixteen epilogue warps (VR200, 16,384
#: rows: 173 us against 187 us with eight and 194-197 us with a four-chunk ring).
WUK_TILE_N = 256
WUK_STAGES = 3
WUK_EPI_WARPS = 16
#: wuv_gate tiling: a 128 x 128 accumulator (one head's values), a five-chunk ring over the 512-wide K, sixteen
#: epilogue warps: the gate's per-element arithmetic is the epilogue's cost (VR200, 16,384 rows: 235 us against
#: 316 us with eight; 239 / 259 us with a four / six-chunk ring).
WUV_TILE_N = 128
WUV_STAGES = 5
WUV_EPI_WARPS = 16


def wuk_fp8(dst: torch.Tensor, q_nope: torch.Tensor, w_uk: torch.Tensor) -> None:
    """Fill ``dst`` ``[n, pad, lora]`` (E4M3, unit last-dim stride, 16-byte aligned rows and heads): head
    ``h < heads`` with the converted ``q_nope[:, h] @ w_uk[h]``, the others with zeros. ``q_nope`` ``[n, heads,
    nope]`` and ``w_uk`` ``[heads, nope, lora]`` are BF16 with unit last-dim strides (layouts not re-checked)."""
    n, pad, lora = dst.shape
    heads, nope, _ = w_uk.shape
    if n == 0:
        return
    spec = HeadGemmSpec(
        mode=MODE_WUK_FP8,
        heads=heads,
        pad=pad,
        n=lora,
        k=nope,
        tile_n=WUK_TILE_N,
        stages=WUK_STAGES,
        epi_warps=WUK_EPI_WARPS,
        a_tok=q_nope.stride(0),
        a_head=q_nope.stride(1),
        w_n=w_uk.stride(2),
        w_k=w_uk.stride(1),
        w_head=w_uk.stride(0),
        o_tok=dst.stride(0),
        o_head=dst.stride(1),
        g_row=0,
    )
    _launch(spec, q_nope, w_uk, n, n, dst.data_ptr(), dst.data_ptr())


def wuv_gate(out: torch.Tensor, lat: torch.Tensor, w_uv: torch.Tensor, gate: torch.Tensor) -> None:
    """Fill ``out`` ``[rows, heads * v]`` (BF16, unit column stride) with ``gate(bf16(lat[:, h] @ w_uv[h]), g)`` per
    head, a +0 attention output on rows ``>= lat.shape[0]``. ``lat`` ``[rows_a, heads, lora]`` (unit last-dim
    stride, ``rows_a >= 1``), ``w_uv`` ``[heads, lora, v]`` (unit stride along ``lora`` or ``v``) and ``gate``
    ``[rows, heads * v]`` (unit column stride) are BF16 (layouts not re-checked)."""
    rows = out.shape[0]
    heads, lora, v = w_uv.shape
    if rows == 0:
        return
    spec = HeadGemmSpec(
        mode=MODE_WUV_GATE,
        heads=heads,
        pad=heads,
        n=v,
        k=lora,
        tile_n=WUV_TILE_N,
        stages=WUV_STAGES,
        epi_warps=WUV_EPI_WARPS,
        a_tok=lat.stride(0),
        a_head=lat.stride(1),
        w_n=w_uv.stride(2),
        w_k=w_uv.stride(1),
        w_head=w_uv.stride(0),
        o_tok=out.stride(0),
        o_head=v,
        g_row=gate.stride(0),
    )
    _launch(spec, lat, w_uv, rows, lat.shape[0], out.data_ptr(), gate.data_ptr())


__all__ = [
    "HeadGemmKernel",
    "HeadGemmSpec",
    "MODE_WUK_FP8",
    "MODE_WUV_GATE",
    "wuk_fp8",
    "wuv_gate",
]
