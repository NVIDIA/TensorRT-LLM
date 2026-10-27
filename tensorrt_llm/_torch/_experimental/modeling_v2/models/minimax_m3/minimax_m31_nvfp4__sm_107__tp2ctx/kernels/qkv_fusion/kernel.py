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

# ruff: noqa: E501,E731,E741

"""SM107 persistent MXFP8 QKV GEMM fused with the MiniMax-M3 QKV-index producer.

Per output row m (token) and 128-wide head slot h of the packed
[q (Hq) | K (Hkv) | V (Hkv) | index-q (Hi) | index-K (1)] projection::

    x        = bf16_rn(alpha * (act[m, :] @ weight[h*128:(h+1)*128, :].T))
               # MXFP8 block-scaled GEMM (E4M3 operands, UE8M0 K32
               # scales), FP32 accumulation; x is re-read as fp32
    V        : no norm, no rope
    otherwise: RMSNorm with the per-slot-type bf16 weight w (1 + w
               scaling) over the 128 dims, then NeoX RoPE on dims 0..63
               with (cos, sin) = rotary_cos_sin[position_ids[m]]
    q        -> E4M3 q_out[m, h, :]
    index-q  -> E4M3 index_q_out[m, h - first index-q slot, :]
    index-K  -> E4M3 paged index-K cache row out_cache_loc[m]
    K, V     -> NVFP4 (16-element E4M3 block scales, second-level scale
               kv_quant_scale[1] / [2]) into the paged HND K/V caches

Every floating-point step after the GEMM is the op sequence that
``minimaxM3Nvfp4QKVIndexerNormRopeKVInsert{,Rows}Kernel`` compiles to, so
the outputs are bitwise those of running that producer on this kernel's
own bf16 GEMM output (``debug_gemm_out=True`` writes that bf16 tensor
instead of the producer outputs):

* sum of squares per 4-dim lane group l: fma(x0, x0, x1*x1) +
  fma(x2, x2, x3*x3), then the xor-butterfly 16, 8, 4, 2, 1 over the 32
  lane partials, replayed in-thread;
* inv_rms = rsqrt.approx.f32(fma(sum, 1/128, eps));
* e = x * (inv_rms * (1 + w));
* rope: out = fma(+-paired, sin, e * cos), paired = e[d ^ 32], negated for
  d < 32;
* E4M3 via cvt.rn.satfinite.e4m3x2.f32; NVFP4 via the bf16-rounded
  values, the 16-element abs-max, scale = sl * (max * 0x3E2AAAAB), its
  E4M3 encoding, output scale rcp.approx.ftz(e4m3(scale) *
  rcp.approx.ftz(sl)) (0 for an all-zero group) and
  cvt.rn.satfinite.e2m1x2.f32.

Schedule: a persistent warp-specialized SM107 block-scaled GEMM (CLC tile
scheduler; warps 0-3 epilogue, 4 MMA, 5 TMA, 6 scheduler, 7 idle).  One
work tile is (``mma_m`` rows, ``heads_per_tile`` head slots of 128
columns); ``mma_m=256`` runs the 2-CTA MMA with each CTA of the pair
owning 128 rows.  A tile wider than one MMA (N > ``mma_n``) issues its N
sets back to back per k-block with A held in the operand collector.  The
accumulator has as many TMEM stages as fit (``num_acc_stage``) so the
epilogue of one tile overlaps the mainloop of the next.  ``epi_wgs``
epilogue warpgroups split a tile's heads between them, so with a single
accumulator stage the MMA of the next tile waits for only one warpgroup's
share of producer math.  Each epilogue thread owns one tile row (one
token) and whole 128-column heads, so the row reduction needs no
shuffles.  Rows >= M and head slots past the last are
never stored; cache stores are skipped for out_cache_loc < 0 or
page >= num_pages.

Defaults (4 head slots per tile, 2-CTA M256, A reused across two N256
MMAs, one epilogue warpgroup) are the fastest measured on SM107 at
M=8192..16384: SMEM operand reads per SM, not L2, bound the mainloop, so
wider tiles win; a second epilogue warpgroup caps the epilogue at 208
registers and runs slower.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Tuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects import nvvm
from cutlass._mlir.extras import types as _T
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm import sm100 as gemm_sm100

from tensorrt_llm._torch import cute_aot

from .clc_tile_scheduler import (
    ClcDynamicPersistentTileScheduler,
    ClcDynamicPersistentTileSchedulerParams,
)

HEAD_DIM = 128
ROTARY_DIM = 64
PAGE_SIZE = 128
# float32(rcp.approx.ftz(6.0f)) as the producer's compiled FMUL immediate.
RCP_SIX_BITS = 0x3E2AAAAB


# ---------------------------------------------------------------------------
# Inline-PTX helpers.  Every float op is an explicitly rounded PTX op so that
# neither LLVM nor ptxas contracts or reassociates it.
# ---------------------------------------------------------------------------


def _asm(res_type, args, text, constraints, side_effects, loc, ip):
    return _llvm.inline_asm(
        res_type,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b), cutlass.Float32(c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _rsqrt_approx(x, *, loc=None, ip=None):
    """rsqrtf without fast-math: rsqrt.approx.f32 (denormal-preserving)."""
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(x)],
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _rcp_approx_ftz(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(x)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _absmax(acc, v, *, loc=None, ip=None):
    """fmaxf(acc, fabsf(v)) (max.f32 ignores a NaN operand like fmaxf)."""
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(acc), cutlass.Float32(v)],
            "{\n\t.reg .f32 a;\n\tabs.f32 a, $2;\n\tmax.f32 $0, $1, a;\n\t}",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16x2_rn(lo, hi, *, loc=None, ip=None):
    """bf16_rn(lo) | bf16_rn(hi) << 16."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(hi), cutlass.Float32(lo)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _e4m3x4(v0, v1, v2, v3, *, loc=None, ip=None):
    """Four f32 to E4M3 (RN, satfinite), v0 in the low byte."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(v) for v in (v0, v1, v2, v3)],
            "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}",
            "=r,f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _e4m3_byte(x, *, loc=None, ip=None):
    """__nv_fp8_e4m3(x).__x: cvt.rn.satfinite.e4m3x2.f32 of (0, x), low byte."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(x)],
            "{\n\t.reg .b16 h;\n\t.reg .f32 z;\n\tmov.f32 z, 0f00000000;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 h, z, $1;\n\tcvt.u32.u16 $0, h;\n\t"
            "and.b32 $0, $0, 255;\n\t}",
            "=r,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _e4m3_byte_to_f32(b, *, loc=None, ip=None):
    """static_cast<float>(__nv_fp8_e4m3) of the byte b (exact)."""
    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Int32(b)],
            "{\n\t.reg .b16 h, lo, hi;\n\t.reg .b32 r;\n\tcvt.u16.u32 h, $1;\n\t"
            "cvt.rn.f16x2.e4m3x2 r, h;\n\tmov.b32 {lo, hi}, r;\n\tcvt.f32.f16 $0, lo;\n\t}",
            "=f,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _e2m1x8(v, *, loc=None, ip=None):
    """fp32_vec_to_e2m1 of eight f32: element 2i in the low nibble of byte i."""
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(x) for x in v],
            "{\n\t.reg .b8 byte0, byte1, byte2, byte3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;\n\t"
            "mov.b32 $0, {byte0, byte1, byte2, byte3};\n\t}",
            "=r,f,f,f,f,f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ldg_v4_f32(addr, *, loc=None, ip=None):
    out = _asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [cutlass.Int64(addr)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        False,
        loc,
        ip,
    )
    return [
        cutlass.Float32(_llvm.extractvalue(_T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _ldg_s32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr)],
            "ld.global.nc.s32 $0, [$1];",
            "=r,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ldg_u16(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr)],
            "{\n\t.reg .b16 h;\n\tld.global.nc.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _stg_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr)] + [cutlass.Int32(w) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _stg_v2(addr, w0, w1, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(w0), cutlass.Int32(w1)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _stg_u8(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(w)],
        "st.global.u8 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _sts_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int32(addr), cutlass.Float32(v)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _lds_v4_f32(addr, *, loc=None, ip=None):
    out = _asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [cutlass.Int32(addr)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        True,
        loc,
        ip,
    )
    return [
        cutlass.Float32(_llvm.extractvalue(_T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    ]


def _lo_f32(w):
    """The bf16 in the low half of w, as fp32."""
    return (w << cutlass.Int32(16)).bitcast(cutlass.Float32)


def _hi_f32(w):
    """The bf16 in the high half of w, as fp32."""
    return (w & cutlass.Int32(-65536)).bitcast(cutlass.Float32)


class S2TCopyBundle(NamedTuple):
    """Tiled copy + partitioned tensors for one SMEM-to-TMEM SF copy."""

    tiled_copy: cute.TiledCopy
    sSF_compact: cute.Tensor
    tSF_compact: cute.Tensor


class M3QkvGemmNormRopeKvInsert:
    """Persistent SM107 MXFP8 QKV GEMM with the M3 producer as its epilogue."""

    def __init__(
        self,
        *,
        num_heads_q: int = 32,
        num_heads_kv: int = 2,
        num_heads_index: int = 2,
        heads_per_tile: int = 4,
        mma_m: int = 256,
        mma_n: int | None = None,
        num_acc_stage: int | None = None,
        cluster_shape_mn: Tuple[int, int] | None = None,
        swizzle_size: int = 1,
        raster_along_m: bool = True,
        debug_gemm_out: bool = False,
        enable_pdl: bool = True,
        epi_wgs: int = 1,
    ):
        if mma_m not in (128, 256):
            raise ValueError("mma_m must be 128 (1-CTA) or 256 (2-CTA pair)")
        if heads_per_tile not in (1, 2, 3, 4):
            raise ValueError("heads_per_tile must be 1..4")
        self.num_heads_q = int(num_heads_q)
        self.num_heads_kv = int(num_heads_kv)
        self.num_heads_index = int(num_heads_index)
        self.k_begin = self.num_heads_q
        self.v_begin = self.k_begin + self.num_heads_kv
        self.iq_begin = self.v_begin + self.num_heads_kv
        self.ik_slot = self.iq_begin + self.num_heads_index
        self.num_slots = self.ik_slot + 1
        self.n_total = self.num_slots * HEAD_DIM

        self.heads_per_tile = int(heads_per_tile)
        self.tile_n = HEAD_DIM * self.heads_per_tile
        inst_n = min(self.tile_n, 256) if mma_n is None else int(mma_n)
        if inst_n not in (128, 256) or self.tile_n % inst_n:
            raise ValueError("mma_n must be 128 or 256 and divide the tile N")
        self.n_sets = self.tile_n // inst_n
        self.mma_inst_shape = (mma_m, inst_n, 64)
        self.mma_tiler = (mma_m, self.tile_n, 128)
        self.use_2cta_instrs = mma_m == 256
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.atom_thr_size = 2 if self.use_2cta_instrs else 1

        if cluster_shape_mn is None:
            cluster_shape_mn = (self.atom_thr_size, 1)
        self.cluster_shape_mn = tuple(cluster_shape_mn)
        cm, cn = self.cluster_shape_mn
        if cm < 1 or cn < 1 or (cm & (cm - 1)) or (cn & (cn - 1)) or cm * cn > 16:
            raise ValueError("cluster dims must be powers of two, <= 16 CTAs")
        if cm % self.atom_thr_size:
            raise ValueError("a 2-CTA MMA needs an even cluster M")
        self.swizzle_size = int(swizzle_size)
        self.raster_along_m = bool(raster_along_m)
        self.debug_gemm_out = bool(debug_gemm_out)
        self.enable_pdl = bool(enable_pdl)
        if epi_wgs not in (1, 2) or self.heads_per_tile % epi_wgs:
            raise ValueError("epi_wgs must be 1 or 2 and divide heads_per_tile")
        self.epi_wgs = int(epi_wgs)
        self.heads_per_wg = self.heads_per_tile // self.epi_wgs

        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = 32
        self.occupancy = 1
        # Warpgroup g of the epilogue is warps 4g..4g+3 (TMEM lane quadrant
        # = warp % 4); the MMA / TMA / scheduler / idle warps follow.
        num_epi_warps = 4 * self.epi_wgs
        self.epilog_warp_id = tuple(range(num_epi_warps))
        self.mma_warp_id = num_epi_warps
        self.tma_warp_id = num_epi_warps + 1
        self.sched_warp_id = num_epi_warps + 2
        self.pad_warp_id = num_epi_warps + 3
        self.threads_per_warp = 32
        self.threads_per_cta = 32 * (num_epi_warps + 4)
        self.epilog_reg_count = 256 if self.epi_wgs == 1 else 208
        self.mainloop_reg_count = 88
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=32 * len(self.epilog_warp_id)
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * (1 + len(self.epilog_warp_id))
        )
        self.arch = "sm_107"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)
        self.requested_acc_stage = num_acc_stage
        self.epi_chunk = 32
        self.chunks_per_head = HEAD_DIM // self.epi_chunk

    # ------------------------------------------------------------------
    def _make_tiled_mmas(self):
        def make(a_op, b_op):
            mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=a_op,
                b_collector_op=b_op,
                atom_layout_mnk=(1, 1, 1),
                permutation_mnk=(1, 1, 1),
            )
            mma.set(tcgen05.Field.NEGATE_A, False)
            mma.set(tcgen05.Field.NEGATE_B, False)
            return mma

        # A stays in the collector across the tile's N sets of one k-block.
        if self.n_sets == 1:
            tiled_mma_first = make(CollectorOp.DISCARD, CollectorOp.DISCARD)
            tiled_mma_mid = tiled_mma_first
            tiled_mma_last = tiled_mma_first
        else:
            tiled_mma_first = make(CollectorOp.FILL, CollectorOp.DISCARD)
            tiled_mma_mid = make(CollectorOp.USE, CollectorOp.DISCARD)
            tiled_mma_last = make(CollectorOp.LASTUSE, CollectorOp.DISCARD)
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)
        return (tiled_mma_first, tiled_mma_mid, tiled_mma_last), tiled_mma_sfb

    def _cluster_layouts(self, tiled_mma, tiled_mma_sfb):
        vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        is_a_mcast = cute.size(vmnk.shape[2]) > 1
        is_b_mcast = cute.size(vmnk.shape[1]) > 1
        return vmnk, sfb_vmnk, is_a_mcast, is_b_mcast

    def _setup_attributes(self):
        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // self.atom_thr_size,
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        (tiled_mma, _, _), _ = self._make_tiled_mmas()
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, 1)
        sfa_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        sfb_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        stage_bytes = (
            cute.size_in_bytes(self.a_dtype, a_one)
            + cute.size_in_bytes(self.b_dtype, b_one)
            + cute.size_in_bytes(self.sf_dtype, sfa_one)
            + cute.size_in_bytes(self.sf_dtype, sfb_one)
        )
        # barriers, CLC response, (1 + w) table (4 x 128 fp32) and slack
        helpers_bytes = 1024 + 4 * HEAD_DIM * 4 + 1024
        self.num_ab_stage = (self.smem_capacity // self.occupancy - helpers_bytes) // stage_bytes

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )
        self.num_sfa_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFA_layout))
            & 0x0000FFFF
        )
        self.num_sfb_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFB_layout))
            & 0x0000FFFF
        )
        sf_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        max_acc_stage = (self.num_tmem_alloc_cols - sf_cols) // self.tile_n
        if self.requested_acc_stage is None:
            self.num_acc_stage = max(1, min(4, max_acc_stage))
        else:
            self.num_acc_stage = int(self.requested_acc_stage)
        if self.num_acc_stage < 1 or self.num_acc_stage > max_acc_stage:
            raise ValueError(
                f"num_acc_stage={self.num_acc_stage} does not fit TMEM "
                f"(max {max_acc_stage} for tile_n={self.tile_n})"
            )
        self.num_accumulator_tmem_cols = self.tile_n * self.num_acc_stage
        if os.environ.get("CUTE_DSL_DUMP_STAGES"):
            print(
                f"[stages] num_ab_stage={self.num_ab_stage} stage_bytes={stage_bytes} "
                f"num_acc_stage={self.num_acc_stage} sf_cols={sf_cols}",
                flush=True,
            )

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,  # (M, K) e4m3
        activation_sf: cute.Tensor,  # 1-D swizzled UE8M0 K32
        weight: cute.Tensor,  # (N, K) e4m3
        weight_sf: cute.Tensor,  # 1-D swizzled UE8M0 K32
        alpha: cute.Tensor,  # (1,) f32
        q_out: cute.Tensor,  # (M * Hq * 128,) u8 (E4M3 bytes)
        index_q_out: cute.Tensor,  # (M * Hi * 128,) u8
        kv_data_cache: cute.Tensor,  # u8, flat
        kv_scale_cache: cute.Tensor,  # u8, flat
        index_k_cache: cute.Tensor,  # u8 (E4M3 bytes), flat
        out_cache_loc: cute.Tensor,  # (>= M,) i32
        kv_quant_scale: cute.Tensor,  # (3,) f32
        rotary_cos_sin: cute.Tensor,  # (max_pos * 64,) f32
        position_ids: cute.Tensor,  # (M,) i32
        norm_weights: cute.Tensor,  # (4 * 128,) bf16: q, K, index-q, index-K
        gemm_out: cute.Tensor,  # (M * N,) bf16, debug only
        data_page_stride: cutlass.Int64,
        data_plane_stride: cutlass.Int64,
        data_head_stride: cutlass.Int64,
        data_token_stride: cutlass.Int64,
        scale_page_stride: cutlass.Int64,
        scale_plane_stride: cutlass.Int64,
        scale_head_stride: cutlass.Int64,
        index_page_stride: cutlass.Int64,
        index_token_stride: cutlass.Int64,
        num_pages: cutlass.Int64,
        eps: cutlass.Float32,
        stream: cuda_driver.CUstream,
    ):
        self.a_dtype = activation.element_type
        self.b_dtype = weight.element_type
        self.sf_dtype = activation_sf.element_type
        self.a_major_mode = OperandMajorMode.K
        self.b_major_mode = OperandMajorMode.K

        m = activation.shape[0]
        k = activation.shape[1]
        n = weight.shape[0]

        self._setup_attributes()
        tiled_mmas, tiled_mma_sfb = self._make_tiled_mmas()
        tiled_mma = tiled_mmas[0]

        a_tensor = cute.make_tensor(
            activation.iterator, cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        b_tensor = cute.make_tensor(
            weight.iterator, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2))
        )
        sfa_tensor = cute.make_tensor(
            activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size),
        )
        sfb_tensor = cute.make_tensor(
            weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size),
        )
        tma_atoms, tma_tensors = self._make_tma_atoms(
            tiled_mma, tiled_mma_sfb, a_tensor, b_tensor, sfa_tensor, sfb_tensor
        )

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        ) * self.atom_thr_size

        num_ctas_mnl = (
            cute.ceil_div(m, self.cta_tile_shape_mnk[0]),
            (self.num_slots + self.heads_per_tile - 1) // self.heads_per_tile,
            1,
        )
        tile_sched_params = ClcDynamicPersistentTileSchedulerParams(
            num_ctas_mnl,
            (*self.cluster_shape_mn, 1),
            self.swizzle_size,
            self.raster_along_m,
        )
        grid = ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_clc_stage * 4], 16
            ]
            sOnePlusW: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 4 * HEAD_DIM], 16]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mmas[0],
            tiled_mmas[1],
            tiled_mmas[2],
            tiled_mma_sfb,
            tma_atoms,
            tma_tensors,
            alpha,
            q_out,
            index_q_out,
            kv_data_cache,
            kv_scale_cache,
            index_k_cache,
            out_cache_loc,
            kv_quant_scale,
            rotary_cos_sin,
            position_ids,
            norm_weights,
            gemm_out,
            (
                data_page_stride,
                data_plane_stride,
                data_head_stride,
                data_token_stride,
                scale_page_stride,
                scale_plane_stride,
                scale_head_stride,
                index_page_stride,
                index_token_stride,
            ),
            num_pages,
            tile_sched_params,
            (m, n),
            eps,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
            smem_merge_branch_allocs=True,
        )
        return

    @cute.jit
    def _make_tma_atoms(self, tiled_mma, tiled_mma_sfb, a_tensor, b_tensor, sfa_tensor, sfb_tensor):
        vmnk, sfb_vmnk, _, _ = self._cluster_layouts(tiled_mma, tiled_mma_sfb)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id),
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            vmnk.shape,
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id),
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        return (
            (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
            (tma_tensor_a, tma_tensor_b, tma_tensor_sfa, tma_tensor_sfb),
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_mid: cute.TiledMma,
        tiled_mma_last: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atoms: Tuple,
        tma_tensors: Tuple,
        alpha: cute.Tensor,
        q_out: cute.Tensor,
        index_q_out: cute.Tensor,
        kv_data_cache: cute.Tensor,
        kv_scale_cache: cute.Tensor,
        index_k_cache: cute.Tensor,
        out_cache_loc: cute.Tensor,
        kv_quant_scale: cute.Tensor,
        rotary_cos_sin: cute.Tensor,
        position_ids: cute.Tensor,
        norm_weights: cute.Tensor,
        gemm_out: cute.Tensor,
        cache_strides: Tuple,
        num_pages: cutlass.Int64,
        tile_sched_params: ClcDynamicPersistentTileSchedulerParams,
        problem_mn: Tuple,
        eps: cutlass.Float32,
    ):
        self._setup_attributes()
        tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb = tma_atoms
        mA_mkl, mB_nkl, mSFA_mkl, mSFB_nkl = tma_tensors
        m, n = problem_mn
        m_i32 = cutlass.Int32(m)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        (
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            is_a_mcast,
            is_b_mcast,
        ) = self._cluster_layouts(tiled_mma, tiled_mma_sfb)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
        num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
        num_tma_producers = num_mcast_ctas_a + num_mcast_ctas_b - 1
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producers),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * len(self.epilog_warp_id) * self.atom_thr_size
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        cluster_size = cute.size(cluster_layout_vmnk.shape)
        num_clc_consumer_threads = 32 * (1 + cluster_size * (1 + len(self.epilog_warp_id) + 1))
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            num_stages=self.num_clc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_clc_consumer_threads
            ),
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(
            self.a_smem_layout_staged.outer, swizzle=self.a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            self.b_smem_layout_staged.outer, swizzle=self.b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(self.sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(self.sfb_smem_layout_staged)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(is_a_mcast or is_b_mcast or self.use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = a_full_mcast_mask
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        clc_response_ptr = storage.clc_response.data_ptr()
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_clc_stage
        )
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        tile_sched = ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
            physical_cluster_shape_mn=self.cluster_shape_mn,
        )
        work_tile = tile_sched.initial_work_tile_info()

        # ==============================================================
        # Epilogue warps
        # ==============================================================
        if warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_increase(self.epilog_reg_count)
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_wait()
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            s_w_addr = cutlass.Int32(storage.sOnePlusW.data_ptr().toint())
            if cutlass.const_expr(not self.debug_gemm_out):
                # (1 + w) per (slot type, dim), as the producer's FADD.
                nw_addr = norm_weights.iterator.toint().to(cutlass.Int64)
                for r in cutlass.range_constexpr(4 // self.epi_wgs):
                    j = tidx + cutlass.Int32(32 * len(self.epilog_warp_id) * r)
                    w_bits = _ldg_u16(nw_addr + j.to(cutlass.Int64) * cutlass.Int64(2))
                    w_f = (w_bits << cutlass.Int32(16)).bitcast(cutlass.Float32)
                    _sts_f32(s_w_addr + j * cutlass.Int32(4), _add_rn(w_f, cutlass.Float32(1.0)))
                self.epilog_sync_barrier.arrive_and_wait()

            alpha_v = alpha[0]
            sl_k = kv_quant_scale[1]
            sl_v = kv_quant_scale[2]
            rcp_six = cutlass.Int32(RCP_SIX_BITS).bitcast(cutlass.Float32)
            (
                data_page_stride,
                data_plane_stride,
                data_head_stride,
                data_token_stride,
                scale_page_stride,
                scale_plane_stride,
                scale_head_stride,
                index_page_stride,
                index_token_stride,
            ) = cache_strides
            q_addr = q_out.iterator.toint().to(cutlass.Int64)
            iq_addr = index_q_out.iterator.toint().to(cutlass.Int64)
            kvd_addr = kv_data_cache.iterator.toint().to(cutlass.Int64)
            kvs_addr = kv_scale_cache.iterator.toint().to(cutlass.Int64)
            ik_addr = index_k_cache.iterator.toint().to(cutlass.Int64)
            cs_addr = rotary_cos_sin.iterator.toint().to(cutlass.Int64)
            pos_addr = position_ids.iterator.toint().to(cutlass.Int64)
            loc_addr = out_cache_loc.iterator.toint().to(cutlass.Int64)
            g_addr = gemm_out.iterator.toint().to(cutlass.Int64)

            # TMEM -> register copy: thread t owns tile row t, 32 columns per load.
            tCtAcc_all = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_x = gemm_sm100.transform_partitioned_tensor_layout(tCtAcc_all)
            epi_tile = (self.cta_tile_shape_mnk[0], self.epi_chunk)
            tAcc_epi = cute.flat_divide(tCtAcc_x, epi_tile)
            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.epi_chunk)),
                self.acc_dtype,
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            row_in_cta = tidx % cutlass.Int32(128)
            epi_wg = cute.arch.make_warp_uniform(warp_idx // cutlass.Int32(4))
            thr_copy_t2r = tiled_copy_t2r.get_slice(row_in_cta)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            cC = cute.make_identity_tensor(epi_tile)
            tTR_cC = thr_copy_t2r.partition_D(cC)
            tTR_rAcc = cute.make_rmem_tensor(tTR_cC.shape, self.acc_dtype)
            tTR_rAcc_flat = cute.make_tensor(tTR_rAcc.iterator, cute.make_layout((self.epi_chunk,)))

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                row = cur_tile_coord[0] * cutlass.Int32(self.cta_tile_shape_mnk[0]) + row_in_cta
                head0 = cur_tile_coord[1] * cutlass.Int32(self.heads_per_tile)
                row_ok = row < m_i32
                row_c = cutlass.Int32(cutlass.select_(row_ok, row, m_i32 - cutlass.Int32(1)))
                row64 = row_c.to(cutlass.Int64)

                if cutlass.const_expr(not self.debug_gemm_out):
                    position = _ldg_s32(pos_addr + row64 * cutlass.Int64(4))
                    slot = _ldg_s32(loc_addr + row64 * cutlass.Int64(4))
                    cs_row = cs_addr + position.to(cutlass.Int64) * cutlass.Int64(ROTARY_DIM * 4)
                    cs = []
                    for qd in cutlass.range_constexpr(ROTARY_DIM // 4):
                        cs.extend(_ldg_v4_f32(cs_row + cutlass.Int64(16 * qd)))
                    slot_ok = slot >= cutlass.Int32(0)
                    slot_c = cutlass.Int32(cutlass.select_(slot_ok, slot, 0))
                    page = (slot_c // cutlass.Int32(PAGE_SIZE)).to(cutlass.Int64)
                    within = (slot_c & cutlass.Int32(PAGE_SIZE - 1)).to(cutlass.Int64)
                    cache_ok = row_ok & slot_ok & (page < num_pages)

                acc_pipeline.consumer_wait(acc_consumer_state)

                def load_head(h):
                    # TMEM -> registers, bf16_rn(alpha * acc) packed in pairs.
                    words = []
                    for c in cutlass.range_constexpr(self.chunks_per_head):
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[
                                (
                                    None,
                                    None,
                                    None,
                                    0,
                                    h * cutlass.Int32(self.chunks_per_head) + cutlass.Int32(c),
                                    acc_consumer_state.index,
                                )
                            ],
                            tTR_rAcc,
                        )
                        for p in cutlass.range_constexpr(self.epi_chunk // 2):
                            a0, a1 = cute.arch.mul_packed_f32x2(
                                (tTR_rAcc_flat[2 * p], tTR_rAcc_flat[2 * p + 1]),
                                (alpha_v, alpha_v),
                            )
                            words.append(_bf16x2_rn(a0, a1))
                    return words

                def release_acc():
                    cute.arch.fence_view_async_tmem_load()
                    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)
                    acc_pipeline.consumer_release(acc_consumer_state)

                for hl in cutlass.range_constexpr(self.heads_per_wg):
                    h = epi_wg * cutlass.Int32(self.heads_per_wg) + cutlass.Int32(hl)
                    head = head0 + h
                    words = load_head(h)
                    if cutlass.const_expr(hl == self.heads_per_wg - 1):
                        release_acc()

                    if cutlass.const_expr(self.debug_gemm_out):
                        if row_ok & (head < cutlass.Int32(self.num_slots)):
                            base = g_addr + (
                                row64 * cutlass.Int64(self.n_total)
                                + head.to(cutlass.Int64) * cutlass.Int64(HEAD_DIM)
                            ) * cutlass.Int64(2)
                            for v in cutlass.range_constexpr(HEAD_DIM // 8):
                                _stg_v4(
                                    base + cutlass.Int64(16 * v),
                                    words[4 * v],
                                    words[4 * v + 1],
                                    words[4 * v + 2],
                                    words[4 * v + 3],
                                )
                    else:
                        valid = head < cutlass.Int32(self.num_slots)
                        is_v = (head >= cutlass.Int32(self.v_begin)) & (
                            head < cutlass.Int32(self.iq_begin)
                        )
                        not_v = (head < cutlass.Int32(self.v_begin)) | (
                            head >= cutlass.Int32(self.iq_begin)
                        )
                        if valid & is_v:
                            dw, sb = self._nvfp4_pack(self._unpack(words), False, sl_v, rcp_six)
                            if cache_ok:
                                self._store_kv(
                                    dw,
                                    sb,
                                    False,
                                    kvd_addr,
                                    kvs_addr,
                                    (head - cutlass.Int32(self.v_begin)).to(cutlass.Int64),
                                    page,
                                    within,
                                    cache_strides,
                                )
                        if valid & not_v:
                            e = self._norm_rope(words, head, cs, s_w_addr, eps)
                            if (head >= cutlass.Int32(self.k_begin)) & (
                                head < cutlass.Int32(self.v_begin)
                            ):
                                dw, sb = self._nvfp4_pack(e, True, sl_k, rcp_six)
                                if cache_ok:
                                    self._store_kv(
                                        dw,
                                        sb,
                                        True,
                                        kvd_addr,
                                        kvs_addr,
                                        (head - cutlass.Int32(self.k_begin)).to(cutlass.Int64),
                                        page,
                                        within,
                                        cache_strides,
                                    )
                            else:
                                packed = []
                                for i in cutlass.range_constexpr(HEAD_DIM // 4):
                                    packed.append(
                                        _e4m3x4(e[4 * i], e[4 * i + 1], e[4 * i + 2], e[4 * i + 3])
                                    )
                                is_q = head < cutlass.Int32(self.k_begin)
                                is_iq = head < cutlass.Int32(self.ik_slot)
                                q_dst = q_addr + (
                                    row64 * cutlass.Int64(self.num_heads_q) + head.to(cutlass.Int64)
                                ) * cutlass.Int64(HEAD_DIM)
                                iq_dst = iq_addr + (
                                    row64 * cutlass.Int64(self.num_heads_index)
                                    + (head - cutlass.Int32(self.iq_begin)).to(cutlass.Int64)
                                ) * cutlass.Int64(HEAD_DIM)
                                ik_dst = (
                                    ik_addr + page * index_page_stride + within * index_token_stride
                                )
                                dst = cutlass.Int64(
                                    cutlass.select_(
                                        is_q,
                                        q_dst,
                                        cutlass.Int64(cutlass.select_(is_iq, iq_dst, ik_dst)),
                                    )
                                )
                                ok = cutlass.Boolean(cutlass.select_(is_iq, row_ok, cache_ok))
                                if ok:
                                    for v in cutlass.range_constexpr(HEAD_DIM // 16):
                                        _stg_v4(
                                            dst + cutlass.Int64(16 * v),
                                            packed[4 * v],
                                            packed[4 * v + 1],
                                            packed[4 * v + 2],
                                            packed[4 * v + 3],
                                        )

                acc_consumer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            tmem.relinquish_alloc_permit()
            # Every epilogue warp is done reading TMEM before the allocator frees it.
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

        # ==============================================================
        # TMA warp
        # ==============================================================
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(self.mainloop_reg_count)
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_wait()
            a_cache_policy = cutlass.Int64(0x14F0000000000000)  # L2 evict_last
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape)
                n_tile = cur_tile_coord[1]
                tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                tBgB_slice = tBgB[(None, n_tile, None, 0)]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_m, None, 0)]
                tBgSFB_slice = tBgSFB[(None, n_tile, None, 0)]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=a_full_mcast_mask,
                        cache_policy=a_cache_policy,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, ab_producer_state.count)],
                        tAsSFA[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfa_full_mcast_mask,
                        cache_policy=a_cache_policy,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, ab_producer_state.count)],
                        tBsSFB[(None, ab_producer_state.index)],
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            ab_pipeline.producer_tail(ab_producer_state)

        # ==============================================================
        # MMA warp
        # ==============================================================
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.mainloop_reg_count)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, self.tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, self.tCtSFB_layout)

            sfa_s2t = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            while work_tile.is_valid_tile:
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                tCtAcc_stage = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                        self._mainloop_s2t_copies(ab_consumer_state.index, sfa_s2t, sfb_s2t)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for k_block in cutlass.range(num_kblocks, unroll_full=True):
                            a_kblk_crd = (None, 0, k_block, ab_consumer_state.index)
                            sfa_kblk_crd = (None, 0, k_block)
                            for n_set in cutlass.range_constexpr(self.n_sets):
                                b_kblk_crd = (None, n_set, k_block, ab_consumer_state.index)
                                sfb_kblk_crd = (None, n_set, k_block)
                                tCtAcc_set = tCtAcc_stage[(None, 0, n_set)]
                                mma_op = (
                                    tiled_mma
                                    if n_set == 0
                                    else (
                                        tiled_mma_last
                                        if n_set == self.n_sets - 1
                                        else tiled_mma_mid
                                    )
                                )
                                mma_op.set(tcgen05.Field.ACCUMULATE, k_tile != 0 or k_block != 0)
                                cute.gemm(
                                    mma_op,
                                    tCtAcc_set,
                                    [tCrA[a_kblk_crd], tCtSFA[sfa_kblk_crd]],
                                    [tCrB[b_kblk_crd], tCtSFB[sfb_kblk_crd]],
                                    tCtAcc_set,
                                )
                        ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        # ==============================================================
        # CLC scheduler warp
        # ==============================================================
        if warp_idx == self.sched_warp_id:
            cute.arch.setmaxregister_decrease(self.mainloop_reg_count)
            if is_first_cta_in_cluster:
                clc_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
                )
                while work_tile.is_valid_tile:
                    clc_pipeline.producer_acquire(clc_producer_state)
                    mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                    tile_sched.advance_to_next_work(mbarrier_addr)
                    clc_producer_state.advance()
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                clc_pipeline.producer_tail(clc_producer_state)

        if warp_idx == self.pad_warp_id:
            cute.arch.setmaxregister_decrease(self.mainloop_reg_count)

    # ------------------------------------------------------------------
    # Straight-line epilogue helpers (plain Python: traced inline, no
    # dynamic control flow inside).
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack(words):
        x = []
        for w in words:
            x.append(_lo_f32(w))
            x.append(_hi_f32(w))
        return x

    def _norm_rope(self, words, head, cs, s_w_addr, eps):
        """RMSNorm (1 + w) + NeoX rope of one head row in the producer's op order."""
        x = self._unpack(words)
        # Lane partials: (0 + fma(x0, x0, x1*x1)) + fma(x2, x2, x3*x3); the
        # leading 0 + is exact (the partial is never -0).
        q = []
        for l in range(32):
            yy0, yy1 = cute.arch.mul_packed_f32x2(
                (x[4 * l + 1], x[4 * l + 3]), (x[4 * l + 1], x[4 * l + 3])
            )
            q.append(
                cute.arch.fma_packed_f32x2(
                    (x[4 * l], x[4 * l + 2]), (x[4 * l], x[4 * l + 2]), (yy0, yy1)
                )
            )
        lane = []
        for l in range(16):
            s0, s1 = cute.arch.add_packed_f32x2(
                (q[2 * l][0], q[2 * l + 1][0]), (q[2 * l][1], q[2 * l + 1][1])
            )
            lane.append(s0)
            lane.append(s1)
        # warpReduceSum (xor 16, 8, 4, 2, 1) as computed by lane 0.
        half = 16
        while half > 1:
            nxt = []
            for l in range(half // 2):
                a0, a1 = cute.arch.add_packed_f32x2(
                    (lane[2 * l], lane[2 * l + 1]), (lane[2 * l + half], lane[2 * l + 1 + half])
                )
                nxt.append(a0)
                nxt.append(a1)
            lane = nxt
            half //= 2
        total = _add_rn(lane[0], lane[1])
        inv_rms = _rsqrt_approx(_fma_rn(total, cutlass.Float32(1.0 / HEAD_DIM), eps))

        is_q = head < cutlass.Int32(self.k_begin)
        is_k = head < cutlass.Int32(self.v_begin)
        is_iq = head < cutlass.Int32(self.ik_slot)
        w_type = cutlass.Int32(
            cutlass.select_(
                is_q,
                cutlass.Int32(0),
                cutlass.Int32(
                    cutlass.select_(
                        is_k,
                        cutlass.Int32(1),
                        cutlass.Int32(cutlass.select_(is_iq, cutlass.Int32(2), cutlass.Int32(3))),
                    )
                ),
            )
        )
        w_base = s_w_addr + w_type * cutlass.Int32(HEAD_DIM * 4)
        e = []
        for qd in range(HEAD_DIM // 4):
            opw = _lds_v4_f32(w_base + cutlass.Int32(16 * qd))
            t0, t1 = cute.arch.mul_packed_f32x2((inv_rms, inv_rms), (opw[0], opw[1]))
            t2, t3 = cute.arch.mul_packed_f32x2((inv_rms, inv_rms), (opw[2], opw[3]))
            e0, e1 = cute.arch.mul_packed_f32x2((x[4 * qd], x[4 * qd + 1]), (t0, t1))
            e2, e3 = cute.arch.mul_packed_f32x2((x[4 * qd + 2], x[4 * qd + 3]), (t2, t3))
            e.extend([e0, e1, e2, e3])
        half_rot = ROTARY_DIM // 2
        out = list(e)
        for d in range(half_rot):
            cos_v = cs[d]
            sin_v = cs[half_rot + d]
            lo = e[d]
            hi = e[d + half_rot]
            m0, m1 = cute.arch.mul_packed_f32x2((lo, hi), (cos_v, cos_v))
            r0, r1 = cute.arch.fma_packed_f32x2((-hi, lo), (sin_v, sin_v), (m0, m1))
            out[d] = r0
            out[d + half_rot] = r1
        return out

    @staticmethod
    def _nvfp4_pack(vals, round_bf16, second_level, rcp_six):
        """quantizeMinimaxM3Fp4x4 over 8 groups of 16: (16 data words, 8 scale bytes)."""
        if round_bf16:
            v = []
            for i in range(HEAD_DIM // 2):
                w = _bf16x2_rn(vals[2 * i], vals[2 * i + 1])
                v.append(_lo_f32(w))
                v.append(_hi_f32(w))
        else:
            v = vals
        rcp_sl = _rcp_approx_ftz(second_level)
        data_words = []
        scale_bytes = []
        for g in range(HEAD_DIM // 16):
            amax = cutlass.Float32(0.0)
            for i in range(16):
                amax = _absmax(amax, v[16 * g + i])
            enc = _e4m3_byte(_mul_rn(second_level, _mul_rn(amax, rcp_six)))
            out_scale = _rcp_approx_ftz(_mul_rn(_e4m3_byte_to_f32(enc), rcp_sl))
            out_scale = cutlass.Float32(
                cutlass.select_(amax != cutlass.Float32(0.0), out_scale, cutlass.Float32(0.0))
            )
            qv = []
            for i in range(8):
                p0, p1 = cute.arch.mul_packed_f32x2(
                    (v[16 * g + 2 * i], v[16 * g + 2 * i + 1]), (out_scale, out_scale)
                )
                qv.append(p0)
                qv.append(p1)
            data_words.append(_e2m1x8(qv[0:8]))
            data_words.append(_e2m1x8(qv[8:16]))
            scale_bytes.append(enc)
        return data_words, scale_bytes

    @staticmethod
    def _store_kv(
        data_words, scale_bytes, is_k, kvd_addr, kvs_addr, head_kv, page, within, cache_strides
    ):
        (
            data_page_stride,
            data_plane_stride,
            data_head_stride,
            data_token_stride,
            scale_page_stride,
            scale_plane_stride,
            scale_head_stride,
            _,
            _,
        ) = cache_strides
        plane = cutlass.Int64(0 if is_k else 1)
        data_dst = (
            kvd_addr
            + page * data_page_stride
            + plane * data_plane_stride
            + head_kv * data_head_stride
            + within * data_token_stride
        )
        for i in range(4):
            _stg_v4(
                data_dst + cutlass.Int64(16 * i),
                data_words[4 * i],
                data_words[4 * i + 1],
                data_words[4 * i + 2],
                data_words[4 * i + 3],
            )
        scale_base = (
            kvs_addr
            + page * scale_page_stride
            + plane * scale_plane_stride
            + head_kv * scale_head_stride
        )
        if is_k:
            # K: the 8 group scales of a token are contiguous.
            words = []
            for wi in range(2):
                b = scale_bytes[4 * wi : 4 * wi + 4]
                words.append(
                    b[0]
                    | (b[1] << cutlass.Int32(8))
                    | (b[2] << cutlass.Int32(16))
                    | (b[3] << cutlass.Int32(24))
                )
            _stg_v2(scale_base + within * cutlass.Int64(8), words[0], words[1])
        else:
            # V: four tokens interleaved per 32-byte group block.
            vbase = (
                scale_base
                + (within // cutlass.Int64(4)) * cutlass.Int64(32)
                + (within % cutlass.Int64(4))
            )
            for g in range(HEAD_DIM // 16):
                _stg_u8(vbase + cutlass.Int64(4 * g), scale_bytes[g])

    # ------------------------------------------------------------------
    @cute.jit
    def _mainloop_s2t_copy_and_partition(self, sSF: cute.Tensor, tSF: cute.Tensor) -> S2TCopyBundle:
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        def append_mn_broadcast_mode(smem_layout: cute.Layout):
            mn_dim = cute.get(smem_layout, mode=[0, 0])
            mn_dim = cute.append(mn_dim, cute.make_layout((4), stride=(0)))
            layout = cute.append(cute.group_modes(mn_dim, 0), cute.get(smem_layout, mode=[0, 1]))
            layout = cute.append(cute.group_modes(layout, 0), cute.get(smem_layout, mode=[1]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[2]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[3]))
            return layout

        tCsSF_compact_bcast = cute.make_tensor(
            tCsSF_compact.iterator, append_mn_broadcast_mode(tCsSF_compact.layout)
        )
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact_bcast)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return S2TCopyBundle(tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    @cute.jit
    def _mainloop_s2t_copies(self, stage_idx, sfa_s2t_bundle, sfb_s2t_bundle):
        s2t_stage_coord = (None, None, None, None, stage_idx)
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfa_s2t_bundle.tSF_compact,
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfb_s2t_bundle.tSF_compact,
        )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_COMPILE_CACHE: dict = {}


def compile(
    *,
    num_heads_q: int = 32,
    num_heads_kv: int = 2,
    num_heads_index: int = 2,
    heads_per_tile: int = 4,
    mma_m: int = 256,
    mma_n: int | None = None,
    num_acc_stage: int | None = None,
    cluster_shape_mn: Tuple[int, int] | None = None,
    swizzle_size: int = 1,
    raster_along_m: bool = True,
    debug_gemm_out: bool = False,
    enable_pdl: bool = True,
    epi_wgs: int = 1,
):
    """Compile one handle serving every M (and any cache geometry)."""
    arch = os.environ.get("CUTE_DSL_ARCH")
    if arch is not None and arch != "sm_107a":
        raise RuntimeError("requires CUTE_DSL_ARCH=sm_107a")
    key = (
        num_heads_q,
        num_heads_kv,
        num_heads_index,
        heads_per_tile,
        mma_m,
        mma_n,
        num_acc_stage,
        None if cluster_shape_mn is None else tuple(cluster_shape_mn),
        swizzle_size,
        raster_along_m,
        debug_gemm_out,
        enable_pdl,
        epi_wgs,
    )
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]
    op = M3QkvGemmNormRopeKvInsert(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        num_heads_index=num_heads_index,
        heads_per_tile=heads_per_tile,
        mma_m=mma_m,
        mma_n=mma_n,
        num_acc_stage=num_acc_stage,
        cluster_shape_mn=cluster_shape_mn,
        swizzle_size=swizzle_size,
        raster_along_m=raster_along_m,
        debug_gemm_out=debug_gemm_out,
        enable_pdl=enable_pdl,
        epi_wgs=epi_wgs,
    )

    def compile_fn():
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

        def sym():
            return cute.sym_int64()

        sym_m, sym_k = sym(), sym()
        fake_act = make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (sym_m, sym_k), stride_order=(1, 0), assumed_align=16
        )
        fake_w = make_fake_compact_tensor(
            cutlass.Float8E4M3FN, (op.n_total, sym_k), stride_order=(1, 0), assumed_align=16
        )
        fake_asf = make_fake_compact_tensor(cutlass.Float8E8M0FNU, (sym(),), assumed_align=16)
        fake_wsf = make_fake_compact_tensor(cutlass.Float8E8M0FNU, (sym(),), assumed_align=16)
        fake_alpha = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        fake_q = make_fake_compact_tensor(cutlass.Uint8, (sym(),), assumed_align=16)
        fake_iq = make_fake_compact_tensor(cutlass.Uint8, (sym(),), assumed_align=16)
        fake_kvd = make_fake_compact_tensor(cutlass.Uint8, (sym(),), assumed_align=16)
        fake_kvs = make_fake_compact_tensor(cutlass.Uint8, (sym(),), assumed_align=8)
        fake_ik = make_fake_compact_tensor(cutlass.Uint8, (sym(),), assumed_align=16)
        fake_loc = make_fake_compact_tensor(cutlass.Int32, (sym(),), assumed_align=4)
        fake_kvq = make_fake_compact_tensor(cutlass.Float32, (sym(),), assumed_align=4)
        fake_cs = make_fake_compact_tensor(cutlass.Float32, (sym(),), assumed_align=16)
        fake_pos = make_fake_compact_tensor(cutlass.Int32, (sym(),), assumed_align=4)
        fake_nw = make_fake_compact_tensor(cutlass.BFloat16, (4 * HEAD_DIM,), assumed_align=16)
        fake_g = make_fake_compact_tensor(cutlass.BFloat16, (sym(),), assumed_align=16)
        return cute.compile(
            op,
            fake_act,
            fake_asf,
            fake_w,
            fake_wsf,
            fake_alpha,
            fake_q,
            fake_iq,
            fake_kvd,
            fake_kvs,
            fake_ik,
            fake_loc,
            fake_kvq,
            fake_cs,
            fake_pos,
            fake_nw,
            fake_g,
            *(cutlass.Int64(0) for _ in range(9)),
            cutlass.Int64(0),
            cutlass.Float32(1e-6),
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )

    # Shipped as an AOT object in the wheel (tensorrt_llm/_torch/cute_aot.py); compiled here only without one.
    compiled = cute_aot.compile_or_load(
        f"m31_tp2ctx_qkv_fusion {key!r} --enable-tvm-ffi",
        compile_fn,
        sources=cute_aot.py_files(os.path.dirname(os.path.abspath(__file__))),
        tvm_ffi=True,
    )
    _COMPILE_CACHE[key] = (op, compiled)
    return _COMPILE_CACHE[key]


def _cache_strides(kv_data_cache, kv_scale_cache, index_k_cache):
    """(data page, plane, head, token; scale page, plane, head; index page, token) strides in bytes."""
    return (
        int(kv_data_cache.stride(0)),
        int(kv_data_cache.stride(1)),
        int(kv_data_cache.stride(2)),
        int(kv_data_cache.stride(3)),
        int(kv_scale_cache.stride(0)),
        int(kv_scale_cache.stride(1)),
        int(kv_scale_cache.stride(2)),
        int(index_k_cache.stride(0)),
        int(index_k_cache.stride(2)),
    )


class M3QkvFusedRunner:
    """Torch-callable wrapper: builds the TVM-FFI dlpack arguments once per call.

    Tensors (all CUDA, contiguous unless noted):
      act e4m3 [M, K], act_sf / weight_sf 1-D swizzled UE8M0 K32 scales
      (``torch.ops.trtllm.mxfp8_quantize(x, True)``), weight e4m3 [N, K],
      alpha fp32 [1], q_out e4m3 [M, Hq, 128], index_q_out e4m3 [M, Hi, 128],
      kv_data_cache uint8 [P, 2, Hkv, 128, 64], kv_scale_cache uint8
      [P, 2, Hkv, 128, 8], index_k_cache e4m3 [P, 1, 128, 128] (strided
      pools allowed as long as rows stay contiguous and 16-byte aligned),
      out_cache_loc int32 [>= M], kv_quant_scale fp32 [3], rotary_cos_sin
      fp32 [max_pos, 2, 32], position_ids int32 [M], norm weights bf16 [128]
      each (or one pre-concatenated [4 * 128] q|K|index-q|index-K tensor).
    ``debug_gemm_out=True`` handles write only the bf16 GEMM output to
    ``gemm_out`` [M, N] and leave every producer output untouched.
    """

    def __init__(self, **cfg):
        self.op, self.compiled = compile(**cfg)
        self.debug = bool(cfg.get("debug_gemm_out", False))

    def make_args(
        self,
        act,
        act_sf,
        weight,
        weight_sf,
        alpha,
        q_out,
        index_q_out,
        kv_data_cache,
        kv_scale_cache,
        index_k_cache,
        out_cache_loc,
        kv_quant_scale,
        rotary_cos_sin,
        position_ids,
        norm_weights,
        eps,
        gemm_out=None,
    ):
        import torch
        from cutlass.cute.runtime import from_dlpack

        op = self.op
        assert act.dim() == 2 and weight.shape == (op.n_total, act.shape[1])
        assert act.shape[1] % 128 == 0
        m = act.shape[0]
        assert q_out.shape == (m, op.num_heads_q, HEAD_DIM)
        assert index_q_out.shape == (m, op.num_heads_index, HEAD_DIM)
        assert kv_data_cache.dim() == 5 and kv_data_cache.shape[1:] == (
            2,
            op.num_heads_kv,
            PAGE_SIZE,
            HEAD_DIM // 2,
        )
        assert kv_scale_cache.dim() == 5 and kv_scale_cache.shape[1:] == (
            2,
            op.num_heads_kv,
            PAGE_SIZE,
            HEAD_DIM // 16,
        )
        assert index_k_cache.dim() == 4 and index_k_cache.shape[1:] == (1, PAGE_SIZE, HEAD_DIM)
        assert kv_data_cache.stride(4) == 1 and kv_data_cache.stride(3) == HEAD_DIM // 2
        assert kv_scale_cache.stride(4) == 1 and kv_scale_cache.stride(3) == HEAD_DIM // 16
        assert index_k_cache.stride(3) == 1 and index_k_cache.stride(2) == HEAD_DIM
        strides = _cache_strides(kv_data_cache, kv_scale_cache, index_k_cache)
        assert all(s % 16 == 0 for s in strides[0:4] + strides[7:9]), strides
        assert all(s % 8 == 0 for s in strides[4:7]), strides
        for t, al in ((kv_data_cache, 16), (kv_scale_cache, 8), (index_k_cache, 16)):
            assert t.data_ptr() % al == 0
        if isinstance(norm_weights, (list, tuple)):
            norm_weights = torch.cat([w.reshape(-1) for w in norm_weights])
        assert norm_weights.dtype == torch.bfloat16 and norm_weights.numel() == 4 * HEAD_DIM
        if gemm_out is None:
            assert not self.debug
            gemm_out = torch.empty(8, dtype=torch.bfloat16, device=act.device)
        else:
            assert gemm_out.shape == (m, op.n_total) and gemm_out.dtype == torch.bfloat16

        def dl(t, al=16):
            return from_dlpack(t, assumed_align=al, enable_tvm_ffi=True)

        def flat_u8(t):
            return t.view(torch.uint8) if t.dtype != torch.uint8 else t

        # Caches are passed as flat byte views spanning their storage extent.
        def span_u8(t):
            base = flat_u8(t)
            extent = 1 + sum((s - 1) * st for s, st in zip(base.shape, base.stride()))
            return torch.as_strided(base, (extent,), (1,))

        return (
            dl(act),
            dl(act_sf.view(torch.float8_e8m0fnu).reshape(-1)),
            dl(weight),
            dl(weight_sf.view(torch.float8_e8m0fnu).reshape(-1)),
            dl(alpha, 4),
            dl(flat_u8(q_out).reshape(-1)),
            dl(flat_u8(index_q_out).reshape(-1)),
            dl(span_u8(kv_data_cache)),
            dl(span_u8(kv_scale_cache), 8),
            dl(span_u8(index_k_cache)),
            dl(out_cache_loc, 4),
            dl(kv_quant_scale, 4),
            dl(rotary_cos_sin.reshape(-1)),
            dl(position_ids, 4),
            dl(norm_weights.reshape(-1)),
            dl(gemm_out.reshape(-1)),
            *strides,
            int(kv_data_cache.shape[0]),
            float(eps),
        )

    def run_args(self, args, stream=None):
        import torch

        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        self.compiled(*args, cuda_driver.CUstream(stream))

    def __call__(self, *tensors, **kw):
        self.run_args(self.make_args(*tensors, **kw))
