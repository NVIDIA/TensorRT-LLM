# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The 2-CTA QKV split-K GEMM with the multi-request decode prep folded into its epilogue (C3).

``ops/gemm/splitk_gemm_2cta`` (reduce-scatter epilogue) followed by
``ops/attention/multi_request_prep_append`` -- the per-head RMSNorm of the query and key heads,
the query pre-scale, the interleaved RoPE, the gate split, the BF16 ``qkv_out`` / ``gate_out``,
the E4M3 query, the FP8 K/V append into the paged cache, the FMHA scale words and the
cumulative query lengths -- in ONE launch, writing the same bytes as the two launches:

* The GEMM is the 2-CTA kernel's, statement for statement, with one change in who reduces what:
  the reduce-scatter hands each K-slice CTA whole 32-token blocks of its 64 output columns (one
  64-wide head: kernel M = the projection's columns) instead of a thread group's quarter of the
  tile. The epilogue's TMEM warp grid is 2 (columns) x 2 (token halves) and a subtile covers
  tokens ``32 s + [0, 32)`` and ``64 + 32 s + [0, 32)``, so token block ``b = n // 32`` lies in one
  subtile and one token half; block ``b`` goes to K slice ``b % split_k``. Every output element
  is still the sum of slice 0's partial, then slice 1's, ... in that order, rounded once to BF16:
  the raw ``out`` is the 2-CTA kernel's (and the served 64-row launch's) bits. The mailbox is the
  same size ((split_k - 1) / split_k of a tile).
* The reducing CTA writes its BF16 blocks to ``out`` and into a shared-memory staging tile
  ``[blocks][32 tokens][64 (+8 pad)]``; after a named barrier of the 128 epilogue threads it runs
  the packed prep's per-(row, head) item on them: 8 threads per item, 8 consecutive elements per
  thread, the squares summed with ``add.rn.ftz`` in element order and then over lane bits 0, 1, 2
  (``shfl.bfly``: each a single commutative add, as the item's three size-2 ``tl.sum``\\ s),
  ``div.full`` by 64, ``+ eps``, ``rsqrt.approx.ftz``, the BF16 rounding points of
  ``preattn_prep._normalize``, ``preattn_prep._rotate``'s ``fma.rn.ftz`` / ``mul.rn.ftz`` RoPE, and
  ``paged_decode_attention._to_cache_fp8`` (BF16 product with the BF16 scale, then E4M3 RN
  satfinite). A CTA's columns are one head, so its kind (query, gate, key, value) is uniform.
* CTA (0, 0) writes the scale words and ``cu_seqlens`` as the prep's program 0 does.
* No early ``griddepcontrol.launch_dependents``: like the prep it replaces, a dependent (the
  FMHA) launches when this grid ends. Every global read of the prep's operands follows the
  accumulator wait, so the TMA warp's ``griddepcontrol.wait``.

Served by ``tep8_wide64_splitk``'s PREP_FOLD (``kernels/dense_gemms/dispatcher._wide_2cta``).
Host entry: ``run_splitk_2cta_prep_fold``. Any compile or launch failure raises.
"""

import functools
from typing import NamedTuple, Optional

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch as _torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.custom_pipeline import (
    PipelineTmaUmma,
    PipelineUmmaAsync,
)

from . import splitk_gemm_2cta as _two

#: The head width the fold's items are written for (8 threads x 8 elements).
HEAD_DIM = 64
#: Tokens of one reduce-scatter block (one subtile's token half).
TOKEN_BLOCK = 32
#: BF16 elements per staged token row: 64 + 8 (16 bytes of pad).
STAGE_STRIDE = 72
#: Named barrier of the 128 epilogue threads before the prep reads the staging tile (barrier 2 is the
#: TMEM allocator's).
FOLD_BARRIER_ID = 3
#: ``log2(e)`` as the prep's float constant.
LOG2E = 1.4426950408889634
#: Rows per sequence the prep takes (one accepted token and up to 7 drafts).
MAX_ROWS_PER_SEQ = 8
#: Sequences one call takes.
MAX_SEQS = 128


class PrepFold(NamedTuple):
    """The compile-time shape of the folded prep."""

    num_q_heads: int
    num_kv_heads: int
    qkv_permuted: bool
    apply_rope: bool
    positions_int64: bool


# ---- one PTX instruction each: the prep's Triton statements as compiled -----------------------


def _asm(result_type, operands, text, constraints, loc, ip, side_effects=False):
    return llvm.inline_asm(
        result_type,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _f32(value, loc, ip):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _u32(value, loc, ip):
    return Uint32(value).ir_value(loc=loc, ip=ip)


def _i64(value, loc, ip):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _f32_op(text, operands, loc, ip):
    return Float32(
        _asm(
            T.f32(),
            [_f32(x, loc, ip) for x in operands],
            text,
            "=f" + ",f" * len(operands),
            loc,
            ip,
        )
    )


def _split(result, count, loc, ip):
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(count)
    )


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return _f32_op("mul.rn.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return _f32_op("add.rn.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def _add_rn_ftz(a, b, *, loc=None, ip=None):
    """``libdevice.add_rn`` as the prep's Triton build links it: ``add.rn.ftz.f32``."""
    return _f32_op("add.rn.ftz.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def _div_full(a, b, *, loc=None, ip=None):
    """Triton's ``x / 64.0``: ``div.full.f32``."""
    return _f32_op("div.full.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def _rsqrt_ftz(a, *, loc=None, ip=None):
    return _f32_op("rsqrt.approx.ftz.f32 $0, $1;", (a,), loc, ip)


@dsl_user_op
def _mul_ftz(a, b, *, loc=None, ip=None):
    return _f32_op("mul.rn.ftz.f32 $0, $1, $2;", (a, b), loc, ip)


@dsl_user_op
def _fma_ftz(a, b, c, *, loc=None, ip=None):
    return _f32_op("fma.rn.ftz.f32 $0, $1, $2, $3;", (a, b, c), loc, ip)


@dsl_user_op
def _neg(a, *, loc=None, ip=None):
    return _f32_op("neg.f32 $0, $1;", (a,), loc, ip)


@dsl_user_op
def _round_bf16(a, *, loc=None, ip=None):
    """``a`` rounded to BF16 (nearest even) and widened back, exactly."""
    return _f32_op("{\n.reg .b16 r;\ncvt.rn.bf16.f32 r, $1;\ncvt.f32.bf16 $0, r;\n}", (a,), loc, ip)


@dsl_user_op
def _shfl_bfly(value, lane_mask, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_f32(value, loc, ip), _u32(lane_mask, loc, ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _bits_float(word, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_u32(word, loc, ip)], "mov.b32 $0, $1;", "=f,r", loc, ip))


@dsl_user_op
def _float_bits(value, *, loc=None, ip=None):
    return Uint32(_asm(T.i32(), [_f32(value, loc, ip)], "mov.b32 $0, $1;", "=r,f", loc, ip))


@dsl_user_op
def _unpack_bf16x2(word, *, loc=None, ip=None):
    """FP32 values of the low and high BF16 halves of ``word`` (exact)."""
    result = _asm(
        ir.Type.parse("!llvm.struct<(f32,f32)>"),
        [_u32(word, loc, ip)],
        "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
        "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
        "=f,=f,r",
        loc,
        ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _pack_bf16x2(low, high, *, loc=None, ip=None):
    """Two FP32 values rounded to BF16 (nearest even), ``low`` in the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [_f32(high, loc, ip), _f32(low, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _e4m3x8(v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None):
    """Eight FP32 values as E4M3 (RN, satfinite), bytes 0..7 in two little-endian words."""
    result = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32)>"),
        [_f32(v, loc, ip) for v in (v0, v1, v2, v3, v4, v5, v6, v7)],
        "{\n.reg .b16 p0, p1, p2, p3;\n"
        "cvt.rn.satfinite.e4m3x2.f32 p0, $3, $2;\n"
        "cvt.rn.satfinite.e4m3x2.f32 p1, $5, $4;\n"
        "cvt.rn.satfinite.e4m3x2.f32 p2, $7, $6;\n"
        "cvt.rn.satfinite.e4m3x2.f32 p3, $9, $8;\n"
        "mov.b32 $0, {p0, p1};\nmov.b32 $1, {p2, p3};\n}",
        "=r,=r,f,f,f,f,f,f,f,f",
        loc,
        ip,
    )
    return _split(result, 2, loc, ip)


# Every memory access below is a side-effecting asm statement: none moves across the barriers.


@dsl_user_op
def _ld_shared_v4(smem_ptr, *, loc=None, ip=None):
    result = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [smem_ptr.toint(loc=loc, ip=ip).ir_value()],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side_effects=True,
    )
    return _split(result, 4, loc, ip)


@dsl_user_op
def _ld_b32(address, *, loc=None, ip=None):
    return Uint32(
        _asm(
            T.i32(),
            [_i64(address, loc, ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _ld_v4(address, *, loc=None, ip=None):
    result = _asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [_i64(address, loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side_effects=True,
    )
    return _split(result, 4, loc, ip)


@dsl_user_op
def _st_v4(address, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(address, loc, ip)] + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _st_v2(address, w0, w1, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(address, loc, ip), _u32(w0, loc, ip), _u32(w1, loc, ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _st_b32(address, w, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(address, loc, ip), _u32(w, loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


def _square_sum(x):
    """``_inverse_rms``'s in-lane sum: ``libdevice.add_rn`` (``add.rn.ftz.f32`` in the prep's PTX) of
    the squares (``mul.f32``) in element order."""
    s = _add_rn_ftz(_mul_rn(x[0], x[0]), _mul_rn(x[1], x[1]))
    for i in range(2, 8):
        s = _add_rn_ftz(s, _mul_rn(x[i], x[i]))
    return s


class SplitK2CtaPrepFoldKernel(_two.SplitK2CtaGemmKernel):
    """``SplitK2CtaGemmKernel`` (reduce-scatter, k_tile 64) with the prep in its epilogue; see the
    module docstring."""

    def __init__(self, split_k: int, use_pdl: bool, fold: PrepFold, prestage: bool = False):
        super().__init__(split_k, use_pdl, 0, True, True, 64, prestage)
        self.fold = fold
        #: Token blocks this CTA reduces and preps (4 blocks of 32 tokens per 128-token tile).
        self.blocks_per_cta = (_two.MAX_ROWS // TOKEN_BLOCK) // self.split_k
        self.stage_elems = self.blocks_per_cta * TOKEN_BLOCK * STAGE_STRIDE

    def _setup(self, a_dtype, a_major, b_major, c_layout, c_dtype):
        """``SplitK2CtaGemmKernel._setup`` with the staging tile beside the mailbox."""
        self.a_dtype = a_dtype
        self.c_dtype = c_dtype
        self.c_layout = c_layout
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, a_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        if self.k_tile % mma_inst_k:
            raise ValueError(
                f"k_tile {self.k_tile} is not whole MMA instructions of K {mma_inst_k}"
            )
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.k_tile)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.pair_layout_vmnk = cute.tiled_divide(
            cute.make_layout((2, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, True, c_layout, c_dtype
        )
        # The epilogue the token-block reduce-scatter is written for: 64 x 64 subtiles (two per CTA
        # tile, each thread 32 values of one 32-token block) on the (2, 2) TMEM warp grid.
        epi_mn = (cute.size(self.epi_tile[0]), cute.size(self.epi_tile[1]))
        if epi_mn != (HEAD_DIM, HEAD_DIM):
            raise ValueError(f"prep fold: epilogue subtiles of {epi_mn}, not 64 x 64")
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, a_dtype, 1)
        stage_bytes = cute.size_in_bytes(a_dtype, a_one) + cute.size_in_bytes(a_dtype, b_one)
        tile_elems = self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[1]
        if self.cta_tile_shape_mnk[0] != HEAD_DIM or self.cta_tile_shape_mnk[1] != _two.MAX_ROWS:
            raise ValueError(
                f"prep fold: a CTA tile of {self.cta_tile_shape_mnk[:2]}, not one 64-column head "
                f"x {_two.MAX_ROWS} tokens"
            )
        self.rs_group = self.epilog_threads // self.split_k
        self.mailbox_elems = (self.split_k - 1) * tile_elems // self.split_k
        fit = (
            _two._OPTIN_SMEM_BYTES - self.mailbox_elems * 4 - self.stage_elems * 2 - 2048
        ) // stage_bytes
        stages = min(fit, 12)
        if stages < 2:
            raise ValueError(f"2-CTA prep fold: only {fit} A/B stages fit beside the mailbox")
        self.num_ab_stage = stages
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, a_dtype, stages
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(
            tiled_mma.make_fragment_C(cute.append(acc_shape, 1)), arch="sm_100"
        )
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        qkv_out: Int64,
        gate_out: Int64,
        q8: Int64,
        pool: Int64,
        bt: Int64,
        bt_stride_seq: Int64,
        bt_stride_kv: Int64,
        seq_lens: Int64,
        positions: Int64,
        cos_sin: Int64,
        orig_quant: Int64,
        quant_orig: Int64,
        bmm1: Int64,
        bmm2: Int64,
        cu_seqlens: Int64,
        bmm1_host: Float32,
        o_scale: Float32,
        eps: Float32,
        q_scale: Float32,
        num_seqs: Int32,
        rows_per_seq: Int32,
        log2_p: Int32,
        stream: _cuda.CUstream,
    ):
        tiled_mma = self._setup(
            a.element_type,
            utils.LayoutEnum.from_tensor(a).mma_major_mode(),
            utils.LayoutEnum.from_tensor(b).mma_major_mode(),
            utils.LayoutEnum.from_tensor(c),
            c.element_type,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((2, 1), tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.pair_layout_vmnk.shape
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((2, 1), tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.pair_layout_vmnk.shape
        )
        a_copy = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy = cute.size_in_bytes(self.a_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy + b_copy) * atom_thr_size
        grid = (
            cute.ceil_div(c.layout.shape[0], self.cta_tile_shape_mnk[0]),
            cute.ceil_div(c.layout.shape[1], self.cta_tile_shape_mnk[1]) * self.split_k,
            1,
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
            qkv_out,
            gate_out,
            q8,
            pool,
            bt,
            bt_stride_seq,
            bt_stride_kv,
            seq_lens,
            positions,
            cos_sin,
            orig_quant,
            quant_orig,
            bmm1,
            bmm2,
            cu_seqlens,
            bmm1_host,
            o_scale,
            eps,
            q_scale,
            num_seqs,
            rows_per_seq,
            log2_p,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        qkv_out: Int64,
        gate_out: Int64,
        q8: Int64,
        pool: Int64,
        bt: Int64,
        bt_stride_seq: Int64,
        bt_stride_kv: Int64,
        seq_lens: Int64,
        positions: Int64,
        cos_sin: Int64,
        orig_quant: Int64,
        quant_orig: Int64,
        bmm1: Int64,
        bmm2: Int64,
        cu_seqlens: Int64,
        bmm1_host: Float32,
        o_scale: Float32,
        eps: Float32,
        q_scale: Float32,
        num_seqs: Int32,
        rows_per_seq: Int32,
        log2_p: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        bidx, bidy, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        k_rank = bidy % self.split_k
        tile_m = bidx // 2
        tile_n = bidy // self.split_k
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_dealloc_mbar: cutlass.Int64
            reduce_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline = PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id) * 2
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        reduce_mbar = storage.reduce_mbar.ptr
        if warp_idx == self.epilogue_warp_id[0]:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(reduce_mbar, 1)
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        mailbox = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(self.mailbox_elems),
            byte_alignment=128,
        )
        stage = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=cute.make_layout(self.stage_elems),
            byte_alignment=128,
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        cC_mnl = cute.local_tile(
            cute.make_identity_tensor(mC_mnl.shape),
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        k_tiles = k_tile_cnt // self.split_k
        k_begin = k_rank * k_tiles

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCcC = thr_mma.partition_C(cC_mnl)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )  # fmt: skip
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )  # fmt: skip
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # TMA warp: the 2-CTA kernel's, without the early dependent release.
        if warp_idx == self.tma_warp_id:
            tAgA_slice = tAgA[(None, tile_m, None, 0)]
            tBgB_slice = tBgB[(None, tile_n, None, 0)]
            ab_producer.reset()
            pre = Int32(0)
            if cutlass.const_expr(self.prestage):
                pre = cutlass.min(k_tiles, Int32(self.num_ab_stage))
                for kk in cutlass.range(0, pre, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, k_begin + handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )
            if cutlass.const_expr(self.prefetch_a):
                for k_tile in cutlass.range(pre, k_tiles, 1, unroll=1):
                    cute.prefetch(tma_atom_a, tAgA_slice[(None, k_begin + k_tile)])
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.prestage):
                for kk in cutlass.range(0, pre, 1, unroll=1):
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, k_begin + kk)],
                        tBsB[(None, kk)],
                        tma_bar_ptr=ab_pipeline.sync_object_full.get_barrier(kk),
                    )
            peek = ab_producer.try_acquire()
            for k_tile in cutlass.range(pre, k_tiles, 1, unroll=1):
                handle = ab_producer.acquire_and_advance(peek)
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k_begin + handle.count)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k_begin + handle.count)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                peek = cutlass.Boolean(1)
                if handle.count + 1 < k_tiles:
                    peek = ab_producer.try_acquire()
            ab_producer.tail()

        # MMA warp: the 2-CTA kernel's.
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[(None, None, None, 0)]
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            ab_consumer.reset()
            peek_full = cutlass.Boolean(1)
            if is_leader_cta:
                peek_full = ab_consumer.try_wait()
                acc_pipeline.producer_acquire(acc_state)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_tile in cutlass.range(0, k_tiles, 1, unroll=1):
                if is_leader_cta:
                    handle = ab_consumer.wait_and_advance(peek_full)
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        crd = (None, None, kblock_idx, handle.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    handle.release()
                    peek_full = cutlass.Boolean(1)
                    if handle.count + 1 < k_tiles:
                        peek_full = ab_consumer.try_wait()
            if is_leader_cta:
                acc_pipeline.producer_commit(acc_state)
            acc_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # Epilogue warps: the token-block reduce-scatter, then the prep of this CTA's blocks.
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, True
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)[(None, None, None, None, None, 0)]
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_gC = thr_copy_t2r.partition_D(gC_epi)[
                (None, None, None, None, None, tile_m, tile_n, 0)
            ]
            tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
            cC_epi = cute.flat_divide(tCcC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_cC = thr_copy_t2r.partition_D(cC_epi)[
                (None, None, None, None, None, tile_m, tile_n, 0)
            ]
            tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))
            tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0)].shape, self.acc_dtype)
            subtiles = cute.size(tTR_tAcc.shape, mode=[3])
            vals = cute.size(tTR_rAcc)
            groups = vals // 4
            nq = self.blocks_per_cta
            split_k = self.split_k
            fold = self.fold
            heads_q = fold.num_q_heads
            heads_kv = fold.num_kv_heads
            rows_n = mC_mnl.shape[1]
            n_base = tile_n * _two.MAX_ROWS

            # This thread's column half (wm) and its slot among the 64 threads of a token half;
            # this CTA's head (its 64 columns) and the head's kind: 0 query, 1 key, 2 value, 3 gate.
            lane = tidx % 32
            first = tTR_cC[(None, None, None, 0)][0]
            m_first = first[0]
            head = m_first // HEAD_DIM
            t64 = ((m_first % HEAD_DIM) // 32) * 32 + lane
            if cutlass.const_expr(fold.qkv_permuted):
                kind = cutlass.select_(
                    head < heads_q,
                    Int32(0),
                    cutlass.select_(
                        head < heads_q + heads_kv,
                        Int32(1),
                        cutlass.select_(head < heads_q + 2 * heads_kv, Int32(2), Int32(3)),
                    ),
                )
                hidx = cutlass.select_(
                    head < heads_q,
                    head,
                    cutlass.select_(
                        head < heads_q + heads_kv,
                        head - heads_q,
                        cutlass.select_(
                            head < heads_q + 2 * heads_kv,
                            head - heads_q - heads_kv,
                            head - heads_q - 2 * heads_kv,
                        ),
                    ),
                )
            else:
                kind = cutlass.select_(
                    head < 2 * heads_q,
                    cutlass.select_(head % 2 == 0, Int32(0), Int32(3)),
                    cutlass.select_(head < 2 * heads_q + heads_kv, Int32(1), Int32(2)),
                )
                hidx = cutlass.select_(
                    head < 2 * heads_q,
                    head // 2,
                    cutlass.select_(
                        head < 2 * heads_q + heads_kv,
                        head - 2 * heads_q,
                        head - 2 * heads_q - heads_kv,
                    ),
                )

            acc_pipeline.consumer_wait(acc_state)
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(reduce_mbar, self.mailbox_elems * 4)
            # Send every token block another K slice reduces.
            for s in cutlass.range_constexpr(subtiles):
                blk = (tTR_cC[(None, None, None, s)][0][1] - n_base) // TOKEN_BLOCK
                dest = blk % split_k
                if dest != k_rank:
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
                    j = k_rank - cutlass.select_(k_rank > dest, Int32(1), Int32(0))
                    base = (j * nq + blk // split_k) * groups
                    for gs in cutlass.range_constexpr(groups):
                        _two._store_remote_v4(
                            tTR_rAcc[4 * gs],
                            tTR_rAcc[4 * gs + 1],
                            tTR_rAcc[4 * gs + 2],
                            tTR_rAcc[4 * gs + 3],
                            mailbox.iterator + ((base + gs) * 64 + t64) * 4,
                            reduce_mbar,
                            cta_rank + 2 * (dest - k_rank),
                        )

            # The prep's operands of this CTA's tokens (after the accumulator wait, so after the
            # TMA warp's dependency wait), in flight while the partials arrive.
            lane8 = tidx % 8
            slot = tidx // 8
            p_size = Int32(1) << log2_p
            meta = []
            for it in cutlass.range_constexpr(nq * 2):
                tt = (it % 2) * 16 + slot
                row = n_base + ((it // 2) * split_k + k_rank) * TOKEN_BLOCK + tt
                row_c = cutlass.min(row, rows_n - 1)
                seq = row_c // rows_per_seq
                pos_c = (
                    Int32(_ld_b32(seq_lens + Int64(seq) * 4)) - rows_per_seq + row_c % rows_per_seq
                )
                table = bt + (Int64(seq) * bt_stride_seq + Int64(pos_c >> log2_p)) * 4
                page_k = Int32(_ld_b32(table))
                page_v = Int32(_ld_b32(table + bt_stride_kv * 4))
                rope = None
                if cutlass.const_expr(fold.apply_rope):
                    if cutlass.const_expr(fold.positions_int64):
                        rp = Int32(_ld_b32(positions + Int64(row_c) * 8))
                    else:
                        rp = Int32(_ld_b32(positions + Int64(row_c) * 4))
                    table_row = cos_sin + (Int64(rp) * HEAD_DIM + Int64(lane8 * 4)) * 4
                    rope = (_ld_v4(table_row), _ld_v4(table_row + Int64(128)))
                meta.append((tt, row, pos_c, page_k, page_v, rope))

            # Reduce this CTA's token blocks in slice order; BF16 into out and the staging tile.
            cute.arch.mbarrier_wait(reduce_mbar, 0)
            for s in cutlass.range_constexpr(subtiles):
                blk = (tTR_cC[(None, None, None, s)][0][1] - n_base) // TOKEN_BLOCK
                if blk % split_k == k_rank:
                    q = blk // split_k
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, s)], tTR_rAcc)
                    for gr in cutlass.range_constexpr(groups):
                        for ir in cutlass.range_constexpr(4):
                            own = tTR_rAcc[4 * gr + ir]
                            acc = own
                            for r in cutlass.range_constexpr(split_k):
                                jr = cutlass.min(
                                    Int32(r)
                                    - cutlass.select_(Int32(r) > k_rank, Int32(1), Int32(0)),
                                    Int32(split_k - 2),
                                )
                                off = (((jr * nq + q) * groups + gr) * 64 + t64) * 4 + ir
                                term = cutlass.select_(Int32(r) == k_rank, own, mailbox[off])
                                if cutlass.const_expr(r == 0):
                                    acc = term
                                else:
                                    acc = acc + term
                            tTR_rAcc[4 * gr + ir] = acc
                    tTR_gC_s = tTR_gC[(None, None, None, s)]
                    tTR_cC_s = tTR_cC[(None, None, None, s)]
                    for vr in cutlass.range_constexpr(vals):
                        value = tTR_rAcc[vr].to(self.c_dtype)
                        coord = tTR_cC_s[vr]
                        if cute.elem_less(coord, mC_mnl.shape):
                            tTR_gC_s[vr] = value
                        stage[
                            (q * TOKEN_BLOCK + (coord[1] - n_base) % TOKEN_BLOCK) * STAGE_STRIDE
                            + coord[0]
                            - head * HEAD_DIM
                        ] = value

            # The prep's program 0: the FMHA's scale words and the cumulative query lengths.
            if bidx == 0:
                if bidy == 0:
                    if tidx == 0:
                        u = _bits_float(_ld_b32(quant_orig))
                        word = _mul_rn(_mul_rn(u, u), bmm1_host)
                        _st_b32(bmm1, _float_bits(word))
                        _st_b32(bmm1 + Int64(4), _float_bits(_mul_rn(word, Float32(LOG2E))))
                        _st_b32(bmm2, _float_bits(_mul_rn(o_scale, u)))
                    if rows_per_seq > 1:
                        for c in cutlass.range_constexpr(2):
                            seq_i = tidx + 128 * c
                            if seq_i <= num_seqs:
                                _st_b32(cu_seqlens + Int64(seq_i) * 4, Uint32(seq_i * rows_per_seq))

            cute.arch.barrier(barrier_id=FOLD_BARRIER_ID, number_of_threads=self.epilog_threads)

            # The packed prep's item (row, head) per 8 threads, element for element.
            scale_bf16 = _round_bf16(_bits_float(_ld_b32(orig_quant)))
            out_width = (heads_q + 2 * heads_kv) * HEAD_DIM
            for it in cutlass.range_constexpr(nq * 2):
                tt, row, pos_c, page_k, page_v, rope = meta[it]
                valid = row < rows_n
                w = _ld_shared_v4(
                    stage.iterator + ((it // 2) * TOKEN_BLOCK + tt) * STAGE_STRIDE + lane8 * 8
                )
                x = []
                for kw in cutlass.range_constexpr(4):
                    low, high = _unpack_bf16x2(w[kw])
                    x.append(low)
                    x.append(high)
                cache_slot = Int64((hidx * p_size + (pos_c & (p_size - 1))) * HEAD_DIM + lane8 * 8)
                page_elems = Int64(heads_kv) * Int64(p_size) * HEAD_DIM
                # Every kind computes both paths (the kind is CTA-uniform; only the stores branch):
                # the query / key norm (+ RoPE) and the E4M3 copies of its output and of the input.
                total = _square_sum(x)
                total = _add_rn(total, _shfl_bfly(total, Uint32(1)))
                total = _add_rn(total, _shfl_bfly(total, Uint32(2)))
                total = _add_rn(total, _shfl_bfly(total, Uint32(4)))
                inverse_rms = _rsqrt_ftz(_add_rn(_div_full(total, Float32(64.0)), eps))
                scale = cutlass.select_(kind == 0, q_scale, Float32(1.0))
                y = [_round_bf16(_mul_rn(_round_bf16(_mul_rn(xn, inverse_rms)), scale)) for xn in x]
                if cutlass.const_expr(fold.apply_rope):
                    cos_w, sin_w = rope
                    for pr in cutlass.range_constexpr(4):
                        cos = _bits_float(cos_w[pr])
                        sin = _bits_float(sin_w[pr])
                        even, odd = y[2 * pr], y[2 * pr + 1]
                        y[2 * pr] = _fma_ftz(even, cos, _mul_ftz(_neg(odd), sin))
                        y[2 * pr + 1] = _fma_ftz(odd, cos, _mul_ftz(even, sin))
                normed = [_pack_bf16x2(y[2 * kp], y[2 * kp + 1]) for kp in range(4)]
                e_qk = [_round_bf16(_mul_rn(_round_bf16(yn), scale_bf16)) for yn in y]
                qk01, qk23 = _e4m3x8(*e_qk)
                e_v = [_round_bf16(_mul_rn(xv, scale_bf16)) for xv in x]
                v01, v23 = _e4m3x8(*e_v)
                words = [cutlass.select_(kind < 2, normed[kp], w[kp]) for kp in range(4)]
                # qkv_out: query / key heads normalized, value heads as projected; gate_out: the gate.
                out_head = cutlass.select_(
                    kind == 0,
                    hidx,
                    cutlass.select_(kind == 1, heads_q + hidx, heads_q + heads_kv + hidx),
                )
                if valid:
                    if kind < 3:
                        _st_v4(
                            qkv_out
                            + (Int64(row) * out_width + Int64(out_head * HEAD_DIM + lane8 * 8)) * 2,
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )
                    if kind == 0:
                        _st_v2(
                            q8 + (Int64(row) * heads_q + Int64(hidx)) * HEAD_DIM + Int64(lane8 * 8),
                            qk01,
                            qk23,
                        )
                    if kind == 1:
                        _st_v2(pool + Int64(page_k) * page_elems + cache_slot, qk01, qk23)
                    if kind == 2:
                        _st_v2(pool + Int64(page_v) * page_elems + cache_slot, v01, v23)
                    if kind == 3:
                        _st_v4(
                            gate_out
                            + (
                                Int64(row) * (heads_q * HEAD_DIM)
                                + Int64(hidx * HEAD_DIM + lane8 * 8)
                            )
                            * 2,
                            w[0],
                            w[1],
                            w[2],
                            w[3],
                        )
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_state)
            acc_state.advance()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def fold_wrapper(
        self,
        m: Int32,
        n: Int32,
        k: Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        qkv_out: Int64,
        gate_out: Int64,
        q8: Int64,
        pool: Int64,
        bt: Int64,
        bt_stride_seq: Int64,
        bt_stride_kv: Int64,
        seq_lens: Int64,
        positions: Int64,
        cos_sin: Int64,
        orig_quant: Int64,
        quant_orig: Int64,
        bmm1: Int64,
        bmm2: Int64,
        cu_seqlens: Int64,
        bmm1_host: Float32,
        o_scale: Float32,
        eps: Float32,
        q_scale: Float32,
        num_seqs: Int32,
        rows_per_seq: Int32,
        log2_p: Int32,
        stream: _cuda.CUstream,
    ):
        """A = w [m, k] (K-major), B = x [n, k] (K-major), C = out viewed as [m, n] M-major."""
        a = cute.make_tensor(a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2)))
        b = cute.make_tensor(b_ptr, layout=cute.make_ordered_layout((n, k, 1), order=(1, 0, 2)))
        c = cute.make_tensor(c_ptr, layout=cute.make_ordered_layout((m, n, 1), order=(0, 1, 2)))
        self(
            a,
            b,
            c,
            qkv_out,
            gate_out,
            q8,
            pool,
            bt,
            bt_stride_seq,
            bt_stride_kv,
            seq_lens,
            positions,
            cos_sin,
            orig_quant,
            quant_orig,
            bmm1,
            bmm2,
            cu_seqlens,
            bmm1_host,
            o_scale,
            eps,
            q_scale,
            num_seqs,
            rows_per_seq,
            log2_p,
            stream,
        )


@functools.cache
def _compiled(split_k: int, use_pdl: bool, prestage: bool, fold: PrepFold):
    kernel = SplitK2CtaPrepFoldKernel(split_k, use_pdl, fold, prestage)
    dummy = _torch.empty(16, dtype=_torch.bfloat16, device="cuda")
    ptr = lambda: make_ptr(  # noqa: E731
        cutlass.BFloat16, dummy.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel.fold_wrapper,
        2304,
        128,
        8192,
        ptr(),
        ptr(),
        ptr(),
        *([Int64(0)] * 15),
        *([Float32(0.0)] * 4),
        *([Int32(1)] * 3),
        stream,
    )


def check_fold(
    out: _torch.Tensor,
    qkv_out: _torch.Tensor,
    gate_out: _torch.Tensor,
    q_fp8: _torch.Tensor,
    kv_pool: _torch.Tensor,
    block_tables: _torch.Tensor,
    seq_lens: _torch.Tensor,
    positions: Optional[_torch.Tensor],
    cos_sin: Optional[_torch.Tensor],
    kv_scale_orig_quant: _torch.Tensor,
    kv_scale_quant_orig: _torch.Tensor,
    bmm1_scale: _torch.Tensor,
    bmm2_scale: _torch.Tensor,
    cu_seqlens: _torch.Tensor,
    num_seqs: int,
    num_q_heads: int,
    num_kv_heads: int,
) -> int:
    """The page size; raises ``ValueError`` for operands the folded prep does not take (the
    packed prep's preconditions, plus the fold's vector alignments)."""
    rows, width = out.shape
    heads = num_q_heads + num_kv_heads
    page = kv_pool.shape[2] if kv_pool.dim() == 4 else 0
    rows_per_seq = rows // num_seqs if num_seqs > 0 else 0
    problems = [
        why
        for failed, why in (
            (width != 2 * heads * HEAD_DIM, f"a projection of width {width}, not 2 (H + G) 64"),
            (not 2 <= num_seqs <= MAX_SEQS, f"{num_seqs} sequences"),
            (
                rows_per_seq * num_seqs != rows or not 1 <= rows_per_seq <= MAX_ROWS_PER_SEQ,
                f"{rows} rows over {num_seqs} sequences",
            ),
            (
                tuple(qkv_out.shape) != (rows, (num_q_heads + 2 * num_kv_heads) * HEAD_DIM)
                or not qkv_out.is_contiguous()
                or qkv_out.dtype != _torch.bfloat16,
                f"qkv_out {qkv_out.dtype} {tuple(qkv_out.shape)}",
            ),
            (
                tuple(gate_out.shape) != (rows, num_q_heads * HEAD_DIM)
                or not gate_out.is_contiguous()
                or gate_out.dtype != _torch.bfloat16,
                f"gate_out {gate_out.dtype} {tuple(gate_out.shape)}",
            ),
            (
                not q_fp8.is_contiguous()
                or q_fp8.numel() * q_fp8.element_size() < rows * num_q_heads * HEAD_DIM
                or q_fp8.data_ptr() % 8,
                f"a query buffer of {q_fp8.numel() * q_fp8.element_size()} bytes at "
                f"{q_fp8.data_ptr():#x}",
            ),
            (
                kv_pool.dtype != _torch.float8_e4m3fn
                or kv_pool.dim() != 4
                or tuple(kv_pool.shape[1:]) != (num_kv_heads, page, HEAD_DIM)
                or not kv_pool.is_contiguous()
                or page <= 0
                or page & (page - 1) != 0
                or kv_pool.data_ptr() % 8,
                f"pool {kv_pool.dtype} {tuple(kv_pool.shape)} at {kv_pool.data_ptr():#x}",
            ),
            (
                block_tables.dtype != _torch.int32
                or block_tables.dim() != 3
                or block_tables.shape[0] < num_seqs
                or block_tables.shape[1] != 2
                or block_tables.stride(-1) != 1,
                f"block table {block_tables.dtype} {tuple(block_tables.shape)}",
            ),
            (
                seq_lens.dtype != _torch.int32 or seq_lens.numel() < num_seqs,
                f"seq_lens {seq_lens.dtype}[{seq_lens.numel()}]",
            ),
            (
                cu_seqlens.dtype != _torch.int32 or cu_seqlens.numel() < num_seqs + 1,
                f"cu_seqlens {cu_seqlens.dtype}[{cu_seqlens.numel()}]",
            ),
            (
                kv_scale_orig_quant.dtype != _torch.float32
                or kv_scale_quant_orig.dtype != _torch.float32
                or bmm1_scale.dtype != _torch.float32
                or bmm1_scale.numel() < 2
                or bmm2_scale.dtype != _torch.float32
                or bmm2_scale.numel() < 1,
                "scales",
            ),
            (
                (cos_sin is None) != (positions is None),
                "a RoPE table without positions or positions without one",
            ),
            (
                cos_sin is not None
                and (
                    cos_sin.dtype != _torch.float32
                    or not cos_sin.is_contiguous()
                    or cos_sin.numel() % HEAD_DIM
                    or cos_sin.data_ptr() % 16
                ),
                "a RoPE table that is not a contiguous fp32 [positions, 2, 32] tensor",
            ),
            (
                positions is not None
                and (
                    positions.numel() != rows
                    or positions.dtype not in (_torch.int32, _torch.int64)
                    or not positions.is_contiguous()
                ),
                "positions that are not one contiguous int32 / int64 per row",
            ),
        )
        if failed
    ]
    if problems:
        raise ValueError("2-CTA prep fold: " + "; ".join(problems))
    return page


def run_splitk_2cta_prep_fold(
    x: _torch.Tensor,
    w: _torch.Tensor,
    out: _torch.Tensor,
    pdl: bool,
    split_k: int,
    prestage: bool,
    *,
    qkv_out: _torch.Tensor,
    gate_out: _torch.Tensor,
    cos_sin: Optional[_torch.Tensor],
    position_ids: Optional[_torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
    q_scale: float,
    qkv_permuted: bool,
    num_seqs: int,
    kv_pool: _torch.Tensor,
    block_tables: _torch.Tensor,
    seq_lens: _torch.Tensor,
    kv_scale_orig_quant: _torch.Tensor,
    kv_scale_quant_orig: _torch.Tensor,
    q_fp8: _torch.Tensor,
    bmm1_scale: _torch.Tensor,
    bmm2_scale: _torch.Tensor,
    cu_seqlens: _torch.Tensor,
    bmm1_host_scale: float,
    o_scale: float = 1.0,
) -> _torch.Tensor:
    """``out = x @ w.T`` (the 2-CTA kernel's bits, reduce-scatter, k_tile 64) and, from it, what
    ``multi_request_prep_append(out, cos_sin, position_ids, ...)`` writes -- ``qkv_out``,
    ``gate_out`` (both written here), the E4M3 query, the K/V pages, the scale words and
    ``cu_seqlens`` -- in one launch. Raises on operands either op refuses."""
    rows, n, k = _two.check(x, w, out, split_k, 64)
    positions = position_ids.reshape(-1) if position_ids is not None else None
    page = check_fold(
        out,
        qkv_out,
        gate_out,
        q_fp8,
        kv_pool,
        block_tables,
        seq_lens,
        positions,
        cos_sin,
        kv_scale_orig_quant,
        kv_scale_quant_orig,
        bmm1_scale,
        bmm2_scale,
        cu_seqlens,
        num_seqs,
        num_q_heads,
        num_kv_heads,
    )
    fold = PrepFold(
        int(num_q_heads),
        int(num_kv_heads),
        bool(qkv_permuted),
        cos_sin is not None,
        positions is not None and positions.dtype == _torch.int64,
    )
    compiled = _compiled(int(split_k), bool(pdl), bool(prestage), fold)
    mk = lambda t: make_ptr(  # noqa: E731
        cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(x.device).cuda_stream)
    compiled(
        n,
        rows,
        k,
        mk(w),
        mk(x),
        mk(out),
        qkv_out.data_ptr(),
        gate_out.data_ptr(),
        q_fp8.data_ptr(),
        kv_pool.data_ptr(),
        block_tables.data_ptr(),
        int(block_tables.stride(0)),
        int(block_tables.stride(1)),
        seq_lens.data_ptr(),
        positions.data_ptr() if positions is not None else 0,
        cos_sin.data_ptr() if cos_sin is not None else 0,
        kv_scale_orig_quant.data_ptr(),
        kv_scale_quant_orig.data_ptr(),
        bmm1_scale.data_ptr(),
        bmm2_scale.data_ptr(),
        cu_seqlens.data_ptr(),
        float(bmm1_host_scale),
        float(o_scale),
        float(eps),
        float(q_scale),
        int(num_seqs),
        rows // int(num_seqs),
        page.bit_length() - 1,
        stream,
    )
    return out
