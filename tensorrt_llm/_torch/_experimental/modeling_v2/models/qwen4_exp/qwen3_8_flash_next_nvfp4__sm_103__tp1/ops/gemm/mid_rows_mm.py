# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense projections of 17..256 decode rows on tcgen05: `y[M, N] = x[M, K] @ W^T`, bf16 out.

Two operand modes, one kernel:

- FP8: `W` is e4m3 codes `[N, K]` with one fp32 scale (or one per weight row); the bf16
  activation is quantized first by `fp8_row_quant` -- one launch, one e4m3 scale per row
  (`max |x[m, :]| / 448`) -- and the GEMM applies `scale_x[m] * scale_w` in its epilogue.
- bf16: `W` is the bf16 weight and `x` is read as it is.

CuTe DSL (tcgen05), swap-AB: the MMA's A operand is 128 weight rows (`BM`), its B the
activation rows (N = the call's rows rounded up to 32 / 64 / 128 / 256, rows past M
zero-filled by TMA), so the ring holds mostly weight bytes. One CTA a 128-row weight tile and
`K / KS` of K; the `KS` CTAs of one tile are a thread-block cluster. A TMA warp streams the
weight tile and the activation rows into a ring of shared-memory stages (a rolling window of
weight k-tiles past the ring is prefetched to L2), one MMA warp accumulates in tensor memory,
eight epilogue warps (two a TMEM lane quadrant, alternate column tiles) read the accumulator.
With `KS = 1` they scale it, round it to bf16 into shared memory transposed (over the drained
ring), and every warp stores the tile in 16-byte rows. With `KS > 1` the tile is cut into
units of 32 weight rows x 4 activation rows owned round-robin by the cluster's ranks: each
CTA pushes its fp32 partial of a unit into the owner's mailbox (`st.async`, counted on the
owner's mbarrier), and the owner sums the `KS` partials in rank order -- deterministic -- and
stores them. The mailbox is shared memory of its own, or the drained ring behind a cluster
barrier when it is large (256 rows).

Programmatic dependent launch: the weight stages that fit the ring are issued before
`griddepcontrol.wait` (the weight is static), the activation after it; dependents are released
at kernel start, and the CTA gives up its tensor-memory allocation permit as soon as it has
allocated, so a CTA of the next launch is scheduled beside it (two CTAs an SM) and fills its
own ring meanwhile. `fp8_row_quant` releases its dependents at its start, so the GEMM after it
launches and streams its weight while the quantization runs.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.kernels.blackwell.custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

#: Rows one launch takes: the largest MMA N.
MAX_ROWS = 256
E4M3_MAX = 448.0
#: Floor of a row's maximum: an all-zero row quantizes to zeros at this scale.
AMAX_FLOOR = 1e-12
_SMEM_CAP = 232448 - 1024


# ------------------------------------------------------------------------------------------
# inline PTX
# ------------------------------------------------------------------------------------------


def _asm(ret, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _unpack(res, n, ty, loc, ip):
    mlir_ty = T.i32() if ty is cutlass.Int32 else T.f32()
    return [ty(llvm.extractvalue(mlir_ty, res, [i], loc=loc, ip=ip)) for i in range(n)]


@dsl_user_op
def _ldg_v4(addr, *, loc=None, ip=None):
    """16 bytes from global as four words."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(addr)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _ldg_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(T.f32(), [cutlass.Int64(addr)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc, ip, True)
    )


@dsl_user_op
def _stg_f32(addr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Float32(v)],
        "{ st.global.f32 [$1], $2; mov.u32 $0, 0; }",
        "=r,l,f",
        loc,
        ip,
        True,
    )


@dsl_user_op
def _stg_bf16(addr, v, *, loc=None, ip=None):
    """`v` rounded to bf16, to global."""
    _asm(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Float32(v)],
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $2; st.global.b16 [$1], h; mov.u32 $0, 0; }",
        "=r,l,f",
        loc,
        ip,
        True,
    )


@dsl_user_op
def _stg_v2(addr, a, b, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Int32(a), cutlass.Int32(b)],
        "{ st.global.v2.b32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,r,r",
        loc,
        ip,
        True,
    )


@dsl_user_op
def _stg_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int64(addr)] + [cutlass.Int32(v) for v in (a, b, c, d)],
        "{ st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r",
        loc,
        ip,
        True,
    )


@dsl_user_op
def _bf16x2(lo, hi, *, loc=None, ip=None):
    """Two fp32 rounded to bf16, `lo` in the low half."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Float32(lo), cutlass.Float32(hi)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _sts_v4f(saddr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int32(saddr)] + [cutlass.Float32(v) for v in (a, b, c, d)],
        "{ st.shared.v4.f32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,r,f,f,f,f",
        loc,
        ip,
        True,
    )


@dsl_user_op
def _sts_f32(saddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int32(saddr), cutlass.Float32(v)],
        "{ st.shared.f32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,f",
        loc,
        ip,
        True,
    )


@dsl_user_op
def _sts_bf16(saddr, v, *, loc=None, ip=None):
    """`v` rounded to bf16, to shared memory."""
    _asm(
        T.i32(),
        [cutlass.Int32(saddr), cutlass.Float32(v)],
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $2; st.shared.b16 [$1], h; mov.u32 $0, 0; }",
        "=r,r,f",
        loc,
        ip,
        True,
    )


@dsl_user_op
def _lds_v4f(saddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [cutlass.Int32(saddr)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Float32, loc, ip)


@dsl_user_op
def _lds_v4(saddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int32(saddr)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side=True,
    )
    return _unpack(res, 4, cutlass.Int32, loc, ip)


@dsl_user_op
def _st_async_v4f(raddr, rbar, a, b, c, d, *, loc=None, ip=None):
    """Four fp32 into a cluster peer's shared memory (`raddr`, from `mapa`), crediting 16
    bytes to the peer's mbarrier `rbar`."""
    _asm(
        T.i32(),
        [cutlass.Int32(raddr), cutlass.Int32(rbar)] + [cutlass.Float32(v) for v in (a, b, c, d)],
        "{ .reg .b32 x, y, z, w; mov.b32 x, $3; mov.b32 y, $4; mov.b32 z, $5; mov.b32 w, $6; "
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$1], {x, y, z, w}, [$2]; "
        "mov.u32 $0, 0; }",
        "=r,r,r,f,f,f,f",
        loc,
        ip,
        True,
    )


@dsl_user_op
def _mbar_expect_tx(addr, nbytes, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int32(addr), cutlass.Int32(nbytes)],
        "{ mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        True,
    )


@dsl_user_op
def _mbar_wait(addr, parity, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [cutlass.Int32(addr), cutlass.Int32(parity)],
        "{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; "
        "@!p bra WAIT; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        True,
    )


@dsl_user_op
def _mapa(addr, rank, *, loc=None, ip=None):
    """This CTA's shared-memory address `addr` in cluster CTA `rank`'s window."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int32(addr), cutlass.Int32(rank)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _absmax_bf16x2(acc, w, *, loc=None, ip=None):
    """Per half, max(acc, |w|) over bf16x2 words (non-negative halves)."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int32(acc), cutlass.Int32(w)],
            "{ .reg .b32 a; and.b32 a, $2, 0x7fff7fff; max.bf16x2 $0, $1, a; }",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _row_amax(v, *, loc=None, ip=None):
    """The warp's largest |x| from per-lane bf16x2 maxima, as fp32: the two halves' larger
    one (as unsigned bits: the halves are non-negative), then `redux.sync.max.u32`."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(v)],
            "{ .reg .b32 l, h; and.b32 l, $1, 0xffff; shr.b32 h, $1, 16; max.u32 l, l, h; "
            "redux.sync.max.u32 l, l, 0xffffffff; shl.b32 l, l, 16; mov.b32 $0, l; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _e4m3x4(w0, w1, inv, *, loc=None, ip=None):
    """Four bf16 (two words) times `inv`, each rounded to e4m3 (saturating), packed in order."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int32(w0), cutlass.Int32(w1), cutlass.Float32(inv)],
            "{ .reg .b32 t0, t1, t2, t3; .reg .f32 f0, f1, f2, f3; .reg .b16 p, q; "
            "shl.b32 t0, $1, 16; and.b32 t1, $1, 0xffff0000; "
            "shl.b32 t2, $2, 16; and.b32 t3, $2, 0xffff0000; "
            "mov.b32 f0, t0; mov.b32 f1, t1; mov.b32 f2, t2; mov.b32 f3, t3; "
            "mul.rn.f32 f0, f0, $3; mul.rn.f32 f1, f1, $3; "
            "mul.rn.f32 f2, f2, $3; mul.rn.f32 f3, f3, $3; "
            "cvt.rn.satfinite.e4m3x2.f32 p, f1, f0; cvt.rn.satfinite.e4m3x2.f32 q, f3, f2; "
            "mov.b32 $0, {p, q}; }",
            "=r,r,r,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _div_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


# ------------------------------------------------------------------------------------------
# per-row activation quantization
# ------------------------------------------------------------------------------------------


class _RowQuant:
    """One warp a row: `scale[m] = max(max |x[m, :]|, AMAX_FLOOR) / 448`,
    `q[m, k] = e4m3_satfinite(x[m, k] * (448 / max))`. K a multiple of 256."""

    WARPS = 4

    def __init__(self, k: int, pdl: bool):
        assert k % 256 == 0
        self.k, self.pdl = k, pdl
        self.nv = k // 256  # 16-byte vectors a lane

    @cute.jit
    def __call__(
        self,
        m: cutlass.Int32,
        x_ptr: cute.Pointer,
        x_stride: cutlass.Int32,
        q_t: cute.Tensor,
        s_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        self.kernel(x_ptr.toint(), x_stride, q_t.iterator.toint(), s_ptr.toint(), m).launch(
            grid=[(m + self.WARPS - 1) // self.WARPS, 1, 1],
            block=[32 * self.WARPS, 1, 1],
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x_base: cutlass.Int64,
        x_stride: cutlass.Int32,
        q_base: cutlass.Int64,
        s_base: cutlass.Int64,
        m: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
        row = bidx * self.WARPS + warp
        if row < m:
            xr = (
                x_base + cutlass.Int64(row) * cutlass.Int64(x_stride) * 2 + cutlass.Int64(lane * 16)
            )
            vals = [_ldg_v4(xr + v * 512) for v in range(self.nv)]
            mx = cutlass.Int32(0)
            for v in cutlass.range_constexpr(self.nv):
                for j in cutlass.range_constexpr(4):
                    mx = _absmax_bf16x2(mx, vals[v][j])
            amax = cutlass.max(_row_amax(mx), cutlass.Float32(AMAX_FLOOR))
            inv = _div_rn(cutlass.Float32(E4M3_MAX), amax)
            qr = q_base + cutlass.Int64(row) * self.k + cutlass.Int64(lane * 8)
            for v in cutlass.range_constexpr(self.nv):
                _stg_v2(
                    qr + v * 256,
                    _e4m3x4(vals[v][0], vals[v][1], inv),
                    _e4m3x4(vals[v][2], vals[v][3], inv),
                )
            if lane == 0:
                _stg_f32(s_base + cutlass.Int64(row) * 4, _div_rn(amax, cutlass.Float32(E4M3_MAX)))


# ------------------------------------------------------------------------------------------
# GEMM
# ------------------------------------------------------------------------------------------

#: Weight rows one CTA computes: the MMA's M.
BM = 128


@dataclass(frozen=True)
class Plan:
    """`ks` K split (cluster size); `kpb` MMA k-blocks a ring stage (4: 128 bytes of K a row);
    `stages` ring depth; `l2` 1 to prefetch the CTA's weight slice past the ring into L2 at
    the start. A CTA computes `BM` weight rows for every activation row."""

    ks: int = 1
    kpb: int = 4
    stages: int = 4
    l2: int = 1
    epi: int = 16


def mma_rows(m: int) -> int:
    """The MMA N (activation rows, zero-padded by TMA) a call of `m` rows runs at."""
    for b in (32, 64, 128, 256):
        if m <= b:
            return b
    raise ValueError(f"{m} rows: at most {MAX_ROWS}")


class _MidMm:
    """See the module docstring. Swap-AB: the MMA's A operand is `BM` weight rows, its B the
    activation rows (N = `mpad`). Compile-time: N, K, the plan, `mpad`, the operand mode and
    the weight-scale mode (0 none / 1 one fp32 / 2 one a weight row). Runtime: the row count
    (<= mpad) and the pointers."""

    def __init__(self, *, n, k, plan: Plan, mpad, fp8, w_mode, pdl):
        epi_cols = min(plan.epi, mpad)
        assert mpad % 16 == 0 and 16 <= mpad <= 256 and mpad % epi_cols == 0
        self.n, self.k, self.plan, self.mpad, self.fp8, self.w_mode, self.pdl = (
            n,
            k,
            plan,
            mpad,
            fp8,
            w_mode,
            pdl,
        )
        self.ks = plan.ks
        self.io_dtype = cutlass.Float8E4M3FN if fp8 else cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.epi_cols = epi_cols
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        # Two warps a TMEM lane quadrant: warpgroup h reads the column tiles `sub % 2 == h`.
        self.epilogue_warp_id = tuple(range(8))
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.threads = 32 * 10
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.n_tiles = -(-n // BM)
        # The tile staged for the stores, `[mpad][BM]` bf16 (KS = 1, over the drained ring).
        self.stage_bytes = mpad * BM * 2 if plan.ks == 1 else 0
        # The split-K mailbox (KS > 1): the most units a rank owns, KS slots of 512 bytes each;
        # a large one reuses the drained ring (after a cluster barrier: every peer's mainloop is
        # done before any partial lands).
        mbox = -(-mpad // plan.ks) * plan.ks * 512 if plan.ks > 1 else 0
        self.alias_mbox = mbox > (64 << 10)
        self.mbox_bytes = 0 if self.alias_mbox else mbox
        self.mbox_need = mbox

    def _setup(self):
        k_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = utils.sm100.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            k_major,
            k_major,
            self.acc_dtype,
            self.cta_group,
            (BM, self.mpad),
        )
        inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (BM, self.mpad, inst_k * self.plan.kpb)
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = (BM, self.epi_cols)
        assert self.k % (self.mma_tiler[2] * self.ks) == 0, (self.k, self.mma_tiler, self.ks)
        self.k_tiles = self.k // self.mma_tiler[2] // self.ks
        self.stages = self._stages()
        self.n_pre = self.stages
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.io_dtype, self.stages
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.io_dtype, self.stages
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(acc_fake, arch="sm_100")
        return tiled_mma

    def _stages(self) -> int:
        """The plan's ring depth, at most the CTA's K tiles and what fits beside the mailbox."""
        bpe = 1 if self.fp8 else 2
        inst_k = 32 if self.fp8 else 16
        kb = inst_k * self.plan.kpb * bpe
        k_tiles = self.k // (inst_k * self.plan.kpb) // self.ks
        fit = (_SMEM_CAP - self.mbox_bytes - 4096) // ((BM + self.mpad) * kb)
        return max(1, min(self.plan.stages, k_tiles, fit))

    def smem_bytes(self) -> int:
        """Host-side estimate of the dynamic shared memory (ring or staged tile, mailbox)."""
        bpe = 1 if self.fp8 else 2
        inst_k = 32 if self.fp8 else 16
        ring = self._stages() * (BM + self.mpad) * inst_k * self.plan.kpb * bpe
        return max(ring, self.stage_bytes, self.mbox_need if self.alias_mbox else 0) + (
            self.mbox_bytes + 1024 + 2048
        )

    @cute.jit
    def __call__(
        self,
        m: cutlass.Int32,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        out_t: cute.Tensor,
        sx_ptr: cute.Pointer,
        sw_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        k = self.k
        a = cute.make_tensor(w_ptr, cute.make_layout((self.n, k, 1), stride=(k, 1, 0)))
        b = cute.make_tensor(x_ptr, cute.make_layout((m, k, 1), stride=(k, 1, 0)))
        tiled_mma = self._setup()
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.io_dtype, a_smem) + cute.size_in_bytes(
            self.io_dtype, b_smem
        )
        kern = self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            out_t.iterator.toint(),
            w_ptr.toint(),
            sx_ptr.toint(),
            sw_ptr.toint(),
            m,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
        )
        # All of the SM's L1 as shared memory: a CTA of the next launch then fits beside this
        # one (programmatic dependent launch) and streams its weight stages early.
        if const_expr(self.ks > 1):
            kern.launch(
                grid=[self.ks, self.n_tiles, 1],
                block=[self.threads, 1, 1],
                cluster=[self.ks, 1, 1],
                stream=stream,
                use_pdl=self.pdl,
                preferred_smem_carveout=100,
            )
        else:
            kern.launch(
                grid=[1, self.n_tiles, 1],
                block=[self.threads, 1, 1],
                stream=stream,
                use_pdl=self.pdl,
                preferred_smem_carveout=100,
            )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        out_base: cutlass.Int64,
        w_base: cutlass.Int64,
        sx_base: cutlass.Int64,
        sw_base: cutlass.Int64,
        m: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        kr = bidx  # K slice == rank in the (KS, 1, 1) cluster
        nt = bidy  # weight-row tile
        ks: cutlass.Constexpr = self.ks
        mpad: cutlass.Constexpr = self.mpad
        n0 = nt * BM

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_dealloc_mbar: cutlass.Int64
            red_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_producer, ab_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_layout_staged.inner,
        )
        ring_bytes = self.stages * (BM + mpad) * self.mma_tiler[2] * (1 if self.fp8 else 2)
        if const_expr(self.stage_bytes > ring_bytes):
            smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=cute.make_layout(((self.stage_bytes - ring_bytes) // 4 + 4,)),
                byte_alignment=16,
            )
        sXt = smem.allocate_tensor(
            element_type=cutlass.Float32, layout=cute.make_layout((256,)), byte_alignment=16
        )
        sx_s = cutlass.Int32(sXt.iterator.toint())
        red_bar = cutlass.Int32(storage.red_mbar.ptr.toint())
        mbox = cutlass.Int32(0)
        if const_expr(ks > 1):
            # The split-K mailbox: units of 32 weight rows x 4 activation rows (fp32, 512
            # bytes), unit u owned by rank u % KS; slot (u // KS, sender) of the owner.
            if const_expr(self.alias_mbox):
                ring_b = self.stages * (BM + mpad) * self.mma_tiler[2] * (1 if self.fp8 else 2)
                if const_expr(self.mbox_need > ring_b):
                    smem.allocate_tensor(
                        element_type=cutlass.Float32,
                        layout=cute.make_layout(((self.mbox_need - ring_b) // 4 + 4,)),
                        byte_alignment=16,
                    )
                mbox = cutlass.Int32(sA.iterator.toint())
            else:
                sMb = smem.allocate_tensor(
                    element_type=cutlass.Float32,
                    layout=cute.make_layout((self.mbox_bytes // 4,)),
                    byte_alignment=16,
                )
                mbox = cutlass.Int32(sMb.iterator.toint())
            if tidx == 0:
                cute.arch.mbarrier_init(storage.red_mbar.ptr, 1)
            cute.arch.mbarrier_init_fence()
            # Peers st.async into this mailbox only after this CTA's barrier is initialized.
            cute.arch.cluster_arrive_relaxed()
        # The output tile (bf16, KS = 1) or the fp32 partial (KS > 1) reuses the ring (drained
        # once the accumulator is complete): it starts at the A stages, the B stages follow.
        p_base = cutlass.Int32(sA.iterator.toint())

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        k0 = kr * self.k_tiles

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        if const_expr(ks > 1):
            cute.arch.cluster_wait()

        if warp_idx == self.tma_warp_id:
            tAgA_slice = tAgA[(None, nt, None, 0)]
            tBgB_slice = tBgB[(None, 0, None, 0)]
            ab_producer.reset()
            # The weight does not depend on the previous kernel: the ring's stages of it
            # stream before the grid-dependency wait, the activation rows after it.
            pre = []
            for kk in cutlass.range_constexpr(self.n_pre):
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k0 + kk)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                pre.append(handle)
            l2a: cutlass.Constexpr = min(self.plan.l2, self.k_tiles - self.n_pre)
            if const_expr(l2a > 0):
                # A rolling window of `l2` weight k-tiles past the ring prefetched towards L2
                # (TMA prefetch): the HBM stream does not wait for ring slots.
                for kk in cutlass.range_constexpr(self.n_pre, self.n_pre + l2a):
                    cute.prefetch(tma_atom_a, tAgA_slice[(None, k0 + kk)])
            if const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            for kk in cutlass.range_constexpr(self.n_pre):
                handle = pre[kk]
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k0 + kk)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            for k_tile in cutlass.range(self.n_pre, self.k_tiles, 1, unroll=1):
                handle = ab_producer.acquire_and_advance()
                if const_expr(l2a > 0):
                    if k_tile + l2a < self.k_tiles:
                        cute.prefetch(tma_atom_a, tAgA_slice[(None, k0 + k_tile + l2a)])
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k0 + k_tile)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k0 + k_tile)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            ab_producer.tail()
            if const_expr(ks > 1 and self.alias_mbox):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc = tCtAcc_base[(None, None, None, 0)]
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            ab_consumer.reset()
            peek = ab_consumer.try_wait()
            acc_pipeline.producer_acquire(acc_producer_state)
            num_kblocks = cute.size(tCrA, mode=[2])
            for k_tile in cutlass.range(0, self.k_tiles, 1, unroll=1):
                handle = ab_consumer.wait_and_advance(peek)
                for kb in cutlass.range_constexpr(num_kblocks):
                    if const_expr(kb == 0):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    else:
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kb, handle.index)],
                        tCrB[(None, None, kb, handle.index)],
                        tCtAcc,
                    )
                handle.release()
                peek = cutlass.Boolean(1)
                if handle.count + 1 < self.k_tiles:
                    peek = ab_consumer.try_wait()
            acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()
            if const_expr(ks > 1 and self.alias_mbox):
                cute.arch.cluster_arrive_relaxed()
            acc_pipeline.producer_tail(acc_producer_state)
            if const_expr(ks > 1 and self.alias_mbox):
                cute.arch.cluster_wait()

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            # This CTA allocates no more tensor memory: a CTA of the next launch may be
            # scheduled on this SM now (programmatic dependent launch).
            tmem.relinquish_alloc_permit()
            if const_expr(self.fp8):
                # The activation scales (the previous launch's output) into shared memory.
                if const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()
                if tidx < m:
                    _sts_f32(sx_s + tidx * 4, _ldg_f32(sx_base + cutlass.Int64(tidx) * 4))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                utils.LayoutEnum.ROW_MAJOR,
                self.acc_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx % BM)
            half = tidx // BM
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, 0)]
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            # This thread's accumulator row (a weight row) and first column (an activation
            # row): the 32dp32b load gives a thread one row's `epi_cols` consecutive columns.
            tCcC = thr_mma.partition_C(cute.make_identity_tensor((BM, mpad)))
            tTR_cC = thr_copy_t2r.partition_D(
                cute.flat_divide(tCcC[((None, None), 0, 0)], epi_tile)
            )
            frag_c = tTR_cC[(None, None, None, 0, 0)]
            crd0 = frag_c[0]
            r_loc = crd0[0]
            c_loc = crd0[1]
            tTR_rAcc = cute.make_rmem_tensor(frag_c.shape, self.acc_dtype)
            nfrag: cutlass.Constexpr = self.epi_cols
            n = n0 + r_loc
            sw = cutlass.Float32(1.0)
            if const_expr(ks == 1):
                if const_expr(self.w_mode == 1):
                    sw = _ldg_f32(sw_base)
                if const_expr(self.w_mode == 2):
                    if n < self.n:
                        sw = _ldg_f32(sw_base + cutlass.Int64(n) * 4)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id,
                number_of_threads=32 * len(self.epilogue_warp_id),
            )
            vc = (m + 3) // 4  # 4-row chunks of activation rows that hold real rows
            n_own = (4 * vc - kr + ks - 1) // ks  # units this rank owns
            if const_expr(ks > 1):
                if tidx == 0:
                    _mbar_expect_tx(red_bar, n_own * (ks - 1) * 512)
            acc_pipeline.consumer_wait(acc_consumer_state)
            if const_expr(ks > 1 and self.alias_mbox):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            # The tile goes to shared memory transposed, `[activation row][weight row]` (over
            # the drained ring): bf16 and scaled (KS = 1) or the fp32 partial (KS > 1).
            for sub in cutlass.range_constexpr(mpad // self.epi_cols):
                c0 = sub * self.epi_cols + c_loc
                if (c0 < m) & (half == sub % 2):
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, sub)], tTR_rAcc)
                    for q in cutlass.range_constexpr(nfrag // 4):
                        if const_expr(ks == 1):
                            if const_expr(self.fp8):
                                s4 = _lds_v4f(sx_s + (c0 + 4 * q) * 4)
                            for e in cutlass.range_constexpr(4):
                                i = 4 * q + e
                                v = tTR_rAcc[i] * sw
                                if const_expr(self.fp8):
                                    v = v * s4[e]
                                _sts_bf16(p_base + ((c0 + i) * BM + r_loc) * 2, v)
                        else:
                            cq = (c0 // 4) + q
                            if cq * 4 < m:
                                u = (r_loc // 32) * vc + cq
                                owner = u % ks
                                slot = mbox + (((u // ks) * ks + kr) * 32 + r_loc % 32) * 16
                                if owner == kr:
                                    _sts_v4f(
                                        slot,
                                        tTR_rAcc[4 * q],
                                        tTR_rAcc[4 * q + 1],
                                        tTR_rAcc[4 * q + 2],
                                        tTR_rAcc[4 * q + 3],
                                    )
                                else:
                                    _st_async_v4f(
                                        _mapa(slot, owner),
                                        _mapa(red_bar, owner),
                                        tTR_rAcc[4 * q],
                                        tTR_rAcc[4 * q + 1],
                                        tTR_rAcc[4 * q + 2],
                                        tTR_rAcc[4 * q + 3],
                                    )
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id,
                number_of_threads=32 * len(self.epilogue_warp_id),
            )
            tmem.free(tmem_ptr)
        # The tile's stores (KS = 1): an item is 8 weight rows of one activation row (16 bytes
        # out), consecutive threads take consecutive items (coalesced).
        g8: cutlass.Constexpr = BM // 8
        if const_expr(ks == 1):
            cute.arch.barrier()
            for f in cutlass.range(tidx, m * g8, self.threads, unroll=1):
                c = f // g8
                g = f - c * g8
                n2 = n0 + 8 * g
                if n2 < self.n:
                    w4 = _lds_v4(p_base + (c * BM + 8 * g) * 2)
                    _stg_v4(
                        out_base + (cutlass.Int64(c) * self.n + cutlass.Int64(n2)) * 2,
                        w4[0],
                        w4[1],
                        w4[2],
                        w4[3],
                    )
        else:
            # The owner sums its units' KS partials (its own and the peers', which arrive on
            # the mailbox barrier) in rank order and stores them: an item is one unit's weight
            # row (lane) and its 4 activation rows; consecutive threads take consecutive rows.
            if warp_idx < self.mma_warp_id:
                _mbar_wait(red_bar, 0)
                sw2 = cutlass.Float32(1.0)
                if const_expr(self.w_mode == 1):
                    sw2 = _ldg_f32(sw_base)
                vc2 = (m + 3) // 4
                n_own2 = (4 * vc2 - kr + ks - 1) // ks
                for f in cutlass.range(
                    tidx, n_own2 * 32, 32 * len(self.epilogue_warp_id), unroll=1
                ):
                    j = f // 32
                    ln = f - j * 32
                    u = kr + j * ks
                    w_ = u // vc2
                    cq = u - w_ * vc2
                    n2 = n0 + w_ * 32 + ln
                    if n2 < self.n:
                        base = mbox + ((j * ks) * 32 + ln) * 16
                        parts = [_lds_v4f(base + s_ * 512) for s_ in range(ks)]
                        acc = parts[0]
                        for s_ in cutlass.range_constexpr(1, ks):
                            acc = [acc[e] + parts[s_][e] for e in range(4)]
                        sc = sw2
                        if const_expr(self.w_mode == 2):
                            sc = _ldg_f32(sw_base + cutlass.Int64(n2) * 4)
                        if const_expr(self.fp8):
                            s4 = _lds_v4f(sx_s + cq * 16)
                        for e in cutlass.range_constexpr(4):
                            v = acc[e] * sc
                            if const_expr(self.fp8):
                                v = v * s4[e]
                            if cq * 4 + e < m:
                                _stg_bf16(
                                    out_base
                                    + (cutlass.Int64(cq * 4 + e) * self.n + cutlass.Int64(n2)) * 2,
                                    v,
                                )


# ------------------------------------------------------------------------------------------
# host side
# ------------------------------------------------------------------------------------------


def _env_pdl() -> bool:
    try:
        from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
    except ImportError:
        return True
    return bool(get_env_enable_pdl())


def _stream():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


@functools.lru_cache(maxsize=None)
def _compiled_quant(k: int, pdl: bool):
    kern = _RowQuant(k, pdl)
    dev = "cuda"
    x = torch.empty(4, k, dtype=torch.bfloat16, device=dev)
    q = torch.empty(4, k, dtype=torch.uint8, device=dev)
    s = torch.empty(4, dtype=torch.float32, device=dev)
    return cute.compile(
        kern,
        cutlass.Int32(4),
        make_ptr(cutlass.BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        cutlass.Int32(k),
        from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        make_ptr(cutlass.Float32, s.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        _stream(),
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.lru_cache(maxsize=None)
def _compiled_mm(n: int, k: int, plan: Plan, mpad: int, fp8: bool, w_mode: int, pdl: bool):
    kern = _MidMm(n=n, k=k, plan=plan, mpad=mpad, fp8=fp8, w_mode=w_mode, pdl=pdl)
    assert kern.smem_bytes() <= _SMEM_CAP, (plan, kern.smem_bytes())
    dev = "cuda"
    dt = cutlass.Float8E4M3FN if fp8 else cutlass.BFloat16
    x = torch.empty(16, dtype=torch.uint8, device=dev)
    out = torch.empty(4, n, dtype=torch.bfloat16, device=dev)
    s = torch.empty(4, dtype=torch.float32, device=dev)
    return cute.compile(
        kern,
        cutlass.Int32(4),
        make_ptr(dt, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(dt, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        make_ptr(cutlass.Float32, s.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, s.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        _stream(),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def fp8_row_quant(x: torch.Tensor, pdl: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """`(q [M, K] e4m3 contiguous, scale [M] fp32)` of a bf16 `[M, K]` activation (unit column
    stride, 16-byte aligned rows): `scale = max(max |x[m]|, AMAX_FLOOR) / 448`,
    `q = e4m3_satfinite(x * (448 / max))`."""
    m, k = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and k % 256 == 0
    assert x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    # The GEMM reads the scales of its whole MMA tile of rows (16-byte vectors): padded.
    s = torch.empty(mma_rows(m) if m <= MAX_ROWS else m, dtype=torch.float32, device=x.device)
    if m:
        _compiled_quant(k, bool(pdl and _env_pdl()))(
            m, x.data_ptr(), x.stride(0), q.view(torch.uint8), s.data_ptr()
        )
    return q, s[:m]


def mid_mm_raw(
    x: torch.Tensor,
    w: torch.Tensor,
    plan: Plan,
    sx: torch.Tensor | None = None,
    sw: torch.Tensor | None = None,
    pdl: bool = True,
) -> torch.Tensor:
    """The GEMM alone: `x [M, K]` (e4m3 with `sx [M]`, or bf16) contiguous, `w [N, K]` of the
    same dtype contiguous, `sw` None / fp32 [1] / fp32 [N]. Returns a fresh bf16 `[M, N]`."""
    m, k = x.shape
    n = w.shape[0]
    fp8 = x.dtype == torch.float8_e4m3fn
    assert w.dtype == x.dtype and w.shape[1] == k and x.is_contiguous() and w.is_contiguous()
    assert 1 <= m <= MAX_ROWS and n % 16 == 0
    assert fp8 == (sx is not None)
    w_mode = 0 if sw is None else (1 if sw.numel() == 1 else 2)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    fn = _compiled_mm(n, k, plan, mma_rows(m), fp8, w_mode, bool(pdl and _env_pdl()))
    dummy = out.data_ptr()
    fn(
        m,
        x.data_ptr(),
        w.data_ptr(),
        out,
        dummy if sx is None else sx.data_ptr(),
        dummy if sw is None else sw.data_ptr(),
    )
    return out


# ------------------------------------------------------------------------------------------
# plans, weights, entry
# ------------------------------------------------------------------------------------------


@dataclass
class MidWeight:
    """A projection weight as `mid_rows_mm` reads it: `w` `[N, K]` contiguous, e4m3 codes with
    `scale` (fp32 `[1]`, or `[N]` one a weight row) or bf16 with `scale` None."""

    w: torch.Tensor
    scale: torch.Tensor | None = None

    @property
    def n(self) -> int:
        return self.w.shape[0]

    @property
    def k(self) -> int:
        return self.w.shape[1]

    @property
    def fp8(self) -> bool:
        return self.w.dtype == torch.float8_e4m3fn


#: Measured plans (GB300, 152 SMs, CUDA-graph chain with PDL over HBM-cold weights; the
#: campaign's bench/gemm_mid) of the qwen3.8-flash-next-nvfp4 tp1 projections, by MMA rows:
#: (N, K, fp8) -> ((largest rows, Plan), ...). The ring depth is clamped to what fits.
_TABLE: dict[tuple[int, int, bool], tuple[tuple[int, Plan], ...]] = {
    # GDN in_proj (q|k|v|z|a|b)
    (16480, 2560, True): ((256, Plan(1, 4, 8, 2)),),
    # attention in-proj (q|gate|k|v|index)
    (14080, 2560, True): ((256, Plan(1, 4, 5, 2)),),
    # GDN out / attention o
    (2560, 6144, True): ((256, Plan(6, 4, 6, 2)),),
    # shared expert gate|up, down; the router (bf16)
    (1280, 2560, False): ((256, Plan(10, 4, 4, 0)),),
    (2560, 640, False): ((256, Plan(5, 4, 2, 0)),),
    (512, 2560, False): ((256, Plan(10, 4, 4, 0)),),
    # vocab head (per-row e4m3 weight)
    (248320, 2560, True): ((64, Plan(1, 4, 4, 0)), (256, Plan(1, 4, 6, 2))),
}


def default_plan(n: int, k: int, m: int, fp8: bool) -> Plan:
    """The table's plan for (N, K, operand mode) at `m` rows, else a heuristic: split over K
    until about one CTA an SM, as deep a ring as fits."""
    for bound, plan in _TABLE.get((n, k, fp8), ()):
        if m <= bound:
            return plan
    kpb = 4
    kt = k // ((32 if fp8 else 16) * kpb)
    tiles = -(-n // BM)
    ks = 1
    for d in range(1, 17):
        if kt % d == 0 and tiles * d <= 152:
            ks = d
    stage = (BM + mma_rows(m)) * 128
    return Plan(ks, kpb, max(2, min(10, (200 << 10) // stage)))


def mid_rows_mm(x: torch.Tensor, mw: MidWeight, plan: Plan | None = None, pdl: bool = True):
    """`x [M, K]` bf16 (unit column stride, 16-byte aligned rows) @ `mw`^T -> fresh bf16
    `[M, N]`, M = 1 .. MAX_ROWS. An e4m3 weight quantizes `x` per row first
    (`fp8_row_quant`)."""
    m = x.shape[0]
    if plan is None:
        plan = default_plan(mw.n, mw.k, m, mw.fp8)
    if mw.fp8:
        q, sx = fp8_row_quant(x, pdl)
        return mid_mm_raw(q, mw.w, plan, sx, mw.scale, pdl)
    if not x.is_contiguous():
        x = x.contiguous()
    return mid_mm_raw(x, mw.w, plan, None, None, pdl)


def precompile(mw: MidWeight, lo: int = 1, hi: int = MAX_ROWS, pdl: bool = True) -> int:
    """Compile every (plan, MMA rows) `mid_rows_mm` reaches over `lo`..`hi` rows of this
    weight's geometry now rather than at a first serving call; returns how many."""
    pdl = bool(pdl and _env_pdl())
    w_mode = 0 if mw.scale is None else (1 if mw.scale.numel() == 1 else 2)
    keys = {(default_plan(mw.n, mw.k, m, mw.fp8), mma_rows(m)) for m in range(lo, hi + 1)}
    for plan, mpad in sorted(keys, key=str):
        _compiled_mm(mw.n, mw.k, plan, mpad, mw.fp8, w_mode, pdl)
    if mw.fp8:
        _compiled_quant(mw.k, pdl)
    return len(keys)


# ------------------------------------------------------------------------------------------
# op wrappers
# ------------------------------------------------------------------------------------------


def _mm_cell(why: str, n: int, k: int, m: int, mode: str) -> Cell:
    """`mode`: "tensor" (e4m3, one weight scale), "row" (e4m3, one scale a weight row), or
    "bf16"."""
    return Cell(why=why, spec=dict(n=n, k=k, m=m, mode=mode))


class MidRowsMm(OpWrapper):
    """`mid_rows_mm(x, mw)`: `y[M, N] = x[M, K] @ W^T` in bf16 for 1..MAX_ROWS rows on tcgen05.
    With an e4m3 weight (`MidWeight(codes [N, K], scale)`, one fp32 scale or one a weight row)
    the activation is quantized to e4m3 with one scale a row (`fp8_row_quant`, its own launch)
    and the product is `(q(x) * scale_x[m]) @ (codes * scale_w)^T`; with a bf16 weight it is
    the plain product. fp32 accumulation (tensor memory; split-K partials summed in rank
    order), the scales applied in fp32, one bf16 rounding.

    Lossy (e4m3 operands) against the bf16 projection in the FP8 mode: ~3% relative error per
    output on random operands; a row's activation rounding depends only on its own maximum.

    Fusion boundary: activation quantization + GEMM (two launches) / the GEMM (one)."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _mm_cell("GDN in_proj, STP con32 (32 rows), e4m3", 16480, 2560, 32, "tensor"),
        _mm_cell(
            "attention in-proj, MTP3 con32 verify (128 rows), e4m3", 14080, 2560, 128, "tensor"
        ),
        _mm_cell("GDN in_proj, 256 rows (MMA N 256), e4m3", 16480, 2560, 256, "tensor"),
        _mm_cell("GDN out / attention o, 17 rows, split-K, e4m3", 2560, 6144, 17, "tensor"),
        _mm_cell(
            "GDN out / attention o, 200 rows, split-K, mailbox over the ring, e4m3",
            2560,
            6144,
            200,
            "tensor",
        ),
        _mm_cell("router, 128 rows, bf16", 512, 2560, 128, "bf16"),
        _mm_cell("shared expert gate|up, 64 rows, bf16", 1280, 2560, 64, "bf16"),
        _mm_cell("shared expert down, 256 rows, bf16", 2560, 640, 256, "bf16"),
        _mm_cell("vocab head, per-row e4m3 weight, 32 rows", 248320, 2560, 32, "row"),
    )

    note = """
    Tier 6 (CuTe DSL written for this target): at 17-256 decode rows cuBLAS(Lt) reaches
    ~0.4-0.6 of HBM bandwidth on these weights, and the per-tensor FP8 entry adds two
    quantization launches.
    Metadata consumed: none. Call sites: the decode target's dense projections in their
    `_MID_MM_ROWS` row ranges (`_dense_mm`), the router and shared expert, the FP8 vocab
    heads above `fp8_gemv`'s rows.
    Preconditions: `x` bf16 `[M, K]`, 1 <= M <= 256, unit column stride; the FP8 mode reads
    any row stride that is a multiple of 8 elements (16-byte aligned rows), the bf16 mode a
    contiguous `x` (others are copied). `mw.w` `[N, K]` contiguous, N a multiple of 16, K a
    multiple of 128 (e4m3) / 64 (bf16) times the plan's K split.
    Launch: grid (KS, ceil(N / 128)), cluster (KS, 1, 1) when KS > 1, 10 warps (8 epilogue,
    MMA, TMA), shared memory up to ~110 KB at 32 rows so two CTAs fit an SM.
    Programmatic dependent launch (TRTLLM_ENABLE_PDL=0 turns it off): the weight stages that
    fit the ring stream before `griddepcontrol.wait`, the activation after it; dependents are
    released at kernel start (the quantization launch releases the GEMM at its start).
    Graph-capturable (current stream, no host sync). Deterministic. Compiled per (N, K, plan,
    MMA rows (32 / 64 / 128 / 256), mode); `precompile` compiles every variant a weight's
    dispatch range reaches.
    """

    def __call__(self, x: torch.Tensor, mw: MidWeight) -> torch.Tensor:
        return mid_rows_mm(x, mw)

    def reference(self, x: torch.Tensor, mw: MidWeight) -> torch.Tensor:
        """fp32 `[M, N]`: the activation quantized as the op does (FP8 mode), both operands
        dequantized, a true fp32 matmul."""
        if mw.fp8:
            q, s = fp8_row_quant(x)
            xd = q.float() * s[:, None]
            wd = mw.w.float() * (mw.scale[:, None] if mw.scale.numel() > 1 else mw.scale)
        else:
            xd, wd = x.float(), mw.w.float()
        with true_fp32_matmul():
            return xd @ wd.t()

    def is_valid(self, x: torch.Tensor, mw: MidWeight) -> None:
        m, k = x.shape
        assert x.dtype == torch.bfloat16 and x.stride(1) == 1, "bf16 x with unit column stride"
        assert 1 <= m <= MAX_ROWS, f"{m} rows: 1..{MAX_ROWS}"
        assert mw.w.is_contiguous() and mw.k == k and mw.n % 16 == 0
        if mw.fp8:
            assert mw.scale is not None and mw.scale.dtype == torch.float32
            assert mw.scale.numel() in (1, mw.n), "one weight scale, or one a weight row"
            assert x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0, "16-byte aligned rows"
            assert k % 256 == 0, "K a multiple of 256 (the quantization's row vectors)"
        else:
            assert mw.w.dtype == torch.bfloat16 and mw.scale is None


class Fp8RowQuant(OpWrapper):
    """`fp8_row_quant(x)`: `(q [M, K] e4m3, scale [M] fp32)` with `scale = max(max |x[m]|,
    AMAX_FLOOR) / 448` and `q = e4m3_satfinite(x * (448 / max))` -- bitwise torch's cast of
    `x * (448 / max)`. One warp a row, one launch; releases its dependents at its start."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="in-projection input, 37 rows", spec=dict(m=37, k=2560)),
        Cell(
            why="output-projection input, 256 rows, row stride K + 64",
            spec=dict(m=256, k=6144, pad=64),
        ),
        Cell(why="one row", spec=dict(m=1, k=2560)),
    )

    note = """
    Call site: `mid_rows_mm`'s FP8 mode. Preconditions: `x` bf16 `[M, K]`, unit column
    stride, row stride a multiple of 8 elements, 16-byte aligned, K a multiple of 256.
    """

    def __call__(self, x: torch.Tensor):
        return fp8_row_quant(x)

    def reference(self, x: torch.Tensor):
        """Both quotients correctly rounded (float64, rounded once), as the kernel's
        `div.rn.f32`."""
        amax = x.float().abs().amax(1).clamp_min(AMAX_FLOOR).double()
        inv = (E4M3_MAX / amax).float()
        return (x.float() * inv[:, None]).to(torch.float8_e4m3fn), (amax / E4M3_MAX).float()


mid_rows_mm_op = MidRowsMm()
fp8_row_quant_op = Fp8RowQuant()
