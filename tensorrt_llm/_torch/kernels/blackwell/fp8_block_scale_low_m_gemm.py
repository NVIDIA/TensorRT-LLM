# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 block-scale weight GEMM for few-row, weight-bandwidth-bound projections (SM100/103/107).

Computes ``out[M, N] = x[M, K] @ (W[N, K] * S)^T`` where ``W`` is FP8 E4M3 with one
power-of-two (UE8M0) scale per 128x128 block and ``x`` is BF16 with ``M <= MAX_M``. It takes
the operands of ``fp8_swap_ab_gemm(..., disable_ue8m0_cast=True)``: the UE8M0-resmoothed
weight and its DeepGEMM-packed scale. ``N`` only needs to be a multiple of :data:`N_ALIGN`:
the weight TMA zero-fills the rows of a partial last 128-row tile past ``N`` and the
epilogues do not store them.

Low-M kernel (``M <= low_m_max_for(N, K) <= LOW_M_MAX``; kernel coordinates are swapped:
the weight's N rows are the UMMA M side):

* One CTA owns a 128-row weight tile and a contiguous K slice. The ``split_k`` CTAs that
  share a weight tile form a cluster and reduce-scatter their FP32 partials through DSMEM
  (one bulk copy per peer; each rank owns every ``split_k``-th 4-token column chunk).
* The MMA is ``tcgen05.mma.kind::mxf8f6f4.block_scale`` with 1x32 UE8M0 scale vectors.
  The activation is quantized inside the kernel to MXFP8 (E4M3 + one power-of-two scale
  per 1x32 group). Because a CTA's weight tile lies inside a single 128-row scale block,
  the weight block scale only depends on the K tile and is folded into the activation
  scale (``sfb = e_act + e_w - 127``); the weight scale operand is a constant 1.0. The
  weight therefore streams as raw FP8 codes with no expanded scale tensor.
* Warp roles: warp 0 streams the weight with TMA and never waits on the previous grid
  (PDL: with enough stages a CTA prefetches most of its weight slice while the previous
  kernel finishes), warp 1 issues the MMAs, warp 2 loads BF16 activation tiles with TMA
  after ``griddepcontrol.wait``, and ``quant_warps`` warps starting at warp 4 quantize
  activation tiles (round-robin over ``quant_groups`` groups) into the swizzled UMMA B
  layout. Warps 4-7 then run the epilogue (TMEM -> registers -> DSMEM reduce-scatter ->
  BF16 store).
* Shared memory is reused across phases: the activation ring becomes the reduction mailbox
  once every peer has finished quantizing, and the weight ring stages outgoing partials.

Mid-M path (``low_m_max_for(N, K) < M <= MAX_M``):

* A small kernel quantizes the activation to MXFP8 (FP8 codes + 1x32 UE8M0 scales already
  in the UMMA scale-factor layout). The GEMM is chained to it with PDL: its weight TMA warp
  starts streaming before the quantization finishes and only the activation producer waits.
* The GEMM uses the same swap-AB split-K cluster structure with UMMA N padded to a multiple
  of 32 tokens. The activation tile arrives by TMA (FP8) plus one bulk copy of its scales;
  the weight block exponent goes into the A scale operand, refreshed once per K tile.
* The epilogue double-buffers 32-column TMEM loads, reduce-scatters partials with
  ``st.async`` into the owner's DSMEM mailbox, and writes BF16 rows through a token-major
  shared staging buffer with 16-byte stores.

Both kernels launch one non-persistent grid (``N/128`` weight tiles x ``split_k`` cluster
ranks). Tactics, the low-/mid-M cut-over and the largest M for which this GEMM is selected
(:func:`max_m_for`, the measured cross-over against the architecture's default FP8
block-scale GEMM) come from per-architecture tables keyed by ``(N, K)``; shapes without an
entry use a heuristic split-K and a conservative M limit. On SM107, whose default GEMM reads
the weight scale in the CuTe MXFP8 layout, :func:`pack_block_scale_from_mxfp8_layout` derives
the DeepGEMM-packed scale this GEMM reads.
"""

from __future__ import annotations

import dataclasses
import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

#: K extent of one pipeline stage (one 128x128 FP8 weight block).
CTA_K = 128
#: Weight rows per CTA tile (UMMA M); equals the weight scale block height.
MMA_M = 128
#: MXFP8 scale vector length.
SF_VEC = 32
#: Granularity of N. The last 128-row weight tile may be partial: TMA zero-fills its missing
#: rows and the epilogue does not store them. A multiple of 16 keeps every BF16 output row
#: 16-byte aligned and the DeepGEMM-packed scale exactly N rows wide (TMA-aligned to 4).
N_ALIGN = 16
#: Supported token tiles (UMMA N) of the low-M kernel; public M is padded up to one of these.
SUPPORTED_MMA_N = (8, 16, 32, 64)
#: Largest public M served by the low-M kernel (in-kernel activation quantization).
LOW_M_MAX = 64
#: Largest public M the op accepts (larger M than the low-M kernel's runs the mid-M path).
MAX_M = 256

_TMA_WARP = 0
_MMA_WARP = 1
_X_WARP = 2
_QUANT_WARP0 = 4
_QUANT_BARRIER_ID = 1
_QUANT_DONE_BARRIER_ID = 2
_EPI_BARRIER_ID = 3
_TMEM_COLS = 128
_SF_TMEM_COLS = 16
#: L2 eviction hint for the streamed weight (CacheHintSm90::EVICT_FIRST).
_EVICT_FIRST = 0x12F0000000000000
#: BF16 bit pattern of ~1e-4; activation groups below this amax share its scale.
_AMAX_FLOOR_BF16 = 0x38D2

_SMEM_CAPACITY = utils.get_smem_capacity_in_bytes("sm_100")


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


@dataclasses.dataclass(frozen=True)
class LowMTactic:
    """One kernel specialization.

    ``mma_n`` is the padded token count, ``split_k`` the cluster size along K, and the
    stage counts size the weight ring (``a``), activation BF16 ring (``x``) and
    quantized activation ring (``b``).
    """

    mma_n: int
    split_k: int
    a_stages: int
    x_stages: int = 4
    b_stages: int = 4
    quant_warps: int = 4
    quant_groups: int = 4


def smem_bytes(n: int, k: int, tactic: LowMTactic) -> int:
    """Conservative host-side mirror of the low-M kernel's shared-memory footprint."""
    kt_max = _ceil_div(k // CTA_K, tactic.split_k)
    a_st = min(tactic.a_stages, kt_max)
    x_st = min(tactic.x_stages, kt_max)
    b_st = min(tactic.b_stages, kt_max)
    mbox = (tactic.split_k - 1) * _ceil_div(tactic.mma_n // 4, tactic.split_k) * MMA_M * 16
    total = 8 * (2 * (a_st + x_st + b_st) + 4) + 4 + 4 * kt_max
    # The weight ring doubles as the staging buffer for outgoing split-K partials.
    staging = tactic.split_k * _ceil_div(tactic.mma_n // 4, tactic.split_k) * MMA_M * 16
    total = _align_up(total, 1024) + max(a_st * MMA_M * CTA_K, staging if tactic.split_k > 1 else 0)
    total = _align_up(total, 1024) + b_st * tactic.mma_n * CTA_K
    # The reduction mailbox aliases the activation ring (dead once quantization is done).
    total = _align_up(total, 128) + max(x_st * tactic.mma_n * CTA_K * 2, mbox)
    total = _align_up(total, 128) + 512
    total = _align_up(total, 128) + b_st * 512
    return total + 1024  # allocator slack


def validate_tactic(n: int, k: int, tactic: LowMTactic) -> None:
    if tactic.mma_n not in SUPPORTED_MMA_N:
        raise ValueError(f"mma_n must be one of {SUPPORTED_MMA_N}, got {tactic.mma_n}")
    if n % N_ALIGN or k % CTA_K:
        raise ValueError(f"N={n} must be a multiple of {N_ALIGN} and K={k} of {CTA_K}")
    if not 1 <= tactic.split_k <= min(16, k // CTA_K):
        raise ValueError(f"split_k={tactic.split_k} invalid for K={k}")
    for name in ("a_stages", "x_stages", "b_stages"):
        if getattr(tactic, name) < 1:
            raise ValueError(f"{name} must be >= 1")
    groups, warps = tactic.quant_groups, tactic.quant_warps
    # The first four quantization warps also run the epilogue (32 TMEM lanes each).
    if warps < 4 or groups < 1 or warps % groups or (warps // groups) * 32 > tactic.mma_n * 4:
        raise ValueError(f"invalid quant_warps={warps} / quant_groups={groups}")
    if 32 * (4 + warps) > 1024:
        raise ValueError(f"quant_warps={warps} exceeds the CTA size limit")
    kt_max = _ceil_div(k // CTA_K, tactic.split_k)
    for name in ("x_stages", "b_stages"):
        # Each quantization group must own its ring slots, or a group could wait on a
        # slot two phases ahead (mbarrier parity aliasing).
        stages = getattr(tactic, name)
        if stages < kt_max and stages % groups:
            raise ValueError(f"{name}={stages} must be a multiple of quant_groups={groups}")
    need = smem_bytes(n, k, tactic)
    if need > _SMEM_CAPACITY:
        raise ValueError(f"tactic {tactic} needs {need} B of SMEM > {_SMEM_CAPACITY} B")


def max_a_stages(n: int, k: int, tactic: LowMTactic) -> int:
    """Largest weight-ring depth that fits next to the other buffers."""
    kt_max = _ceil_div(k // CTA_K, tactic.split_k)
    best = 1
    for a in range(1, kt_max + 1):
        if smem_bytes(n, k, dataclasses.replace(tactic, a_stages=a)) <= _SMEM_CAPACITY:
            best = a
    return best


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _lds_v4_b32(addr, *, loc=None, ip=None):
    """Load 16 bytes from shared memory as four 32-bit words."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _lds_v4_f32(addr, *, loc=None, ip=None):
    """Load four FP32 values from shared memory."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _sts_v2_b32(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(x, loc=loc, ip=ip) for x in (addr, v0, v1, v2, v3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _abs_max_bf16x32(words, *, loc=None, ip=None):
    """|x| max of 32 BF16 values (16 packed words) as two packed BF16 magnitudes.

    ``max.xorsign.abs`` returns ``max(|a|, |b|)`` with a garbage sign bit, which
    ``_ue8m0_of_amax`` masks off.
    """
    body = "".join(f"max.xorsign.abs.bf16x2 m{i}, ${2 * i + 1}, ${2 * i + 2};\n" for i in range(8))
    body += (
        "max.xorsign.abs.bf16x2 m0, m0, m1;\n"
        "max.xorsign.abs.bf16x2 m2, m2, m3;\n"
        "max.xorsign.abs.bf16x2 m4, m4, m5;\n"
        "max.xorsign.abs.bf16x2 m6, m6, m7;\n"
        "max.xorsign.abs.bf16x2 m0, m0, m2;\n"
        "max.xorsign.abs.bf16x2 m4, m4, m6;\n"
        "max.xorsign.abs.bf16x2 $0, m0, m4;\n"
    )
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in words],
            "{\n.reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n" + body + "}",
            "=r," + ",".join(["r"] * 16),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ue8m0_of_amax(packed_amax, *, loc=None, ip=None):
    """Biased exponent of ``2^ceil(log2(amax / 448))`` from two packed BF16 magnitudes.

    For ``amax = 2^E * (1 + f)``: ``amax / 448 <= 2^(E - 8)`` iff ``1 + f <= 1.75``,
    so the exponent is ``E - 8`` plus one when the 7-bit mantissa exceeds ``0x60``.
    """
    return Int32(
        _asm(
            T.i32(),
            [_iv(packed_amax, loc=loc, ip=ip)],
            "{\n"
            ".reg .b32 lo, hi, a, e, mant;\n"
            ".reg .pred p;\n"
            "and.b32 lo, $1, 0x7fff;\n"
            "shr.u32 hi, $1, 16;\n"
            "and.b32 hi, hi, 0x7fff;\n"
            "max.u32 a, lo, hi;\n"
            f"max.u32 a, a, {_AMAX_FLOOR_BF16};\n"
            "shr.u32 e, a, 7;\n"
            "and.b32 mant, a, 0x7f;\n"
            "setp.gt.u32 p, mant, 0x60;\n"
            "sub.u32 e, e, 8;\n"
            "@p add.u32 e, e, 1;\n"
            "mov.b32 $0, e;\n"
            "}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_inv_scale(ue8m0, *, loc=None, ip=None):
    """Packed BF16x2 ``2^(127 - ue8m0)``."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(ue8m0, loc=loc, ip=ip)],
            "{\n"
            ".reg .b32 s, t;\n"
            "sub.u32 s, 254, $1;\n"
            "shl.b32 s, s, 7;\n"
            "shl.b32 t, s, 16;\n"
            "or.b32 $0, s, t;\n"
            "}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _quant_bf16x8_to_e4m3(w0, w1, w2, w3, inv_scale, *, loc=None, ip=None):
    """Scale eight BF16 values by the packed BF16x2 ``inv_scale`` and round to E4M3.

    The scale is a power of two, so the BF16 multiply is exact and the only rounding
    is the satfinite round-to-nearest conversion to E4M3.
    """
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_iv(x, loc=loc, ip=ip) for x in (w0, w1, w2, w3, inv_scale)],
        "{\n"
        ".reg .b32 p0, p1, p2, p3;\n"
        ".reg .b16 h0, h1, h2, h3;\n"
        "mul.rn.bf16x2 p0, $2, $6;\n"
        "mul.rn.bf16x2 p1, $3, $6;\n"
        "mul.rn.bf16x2 p2, $4, $6;\n"
        "mul.rn.bf16x2 p3, $5, $6;\n"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h0, p0;\n"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h1, p1;\n"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h2, p2;\n"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h3, p3;\n"
        "mov.b32 $0, {h0, h1};\n"
        "mov.b32 $1, {h2, h3};\n"
        "}",
        "=r,=r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _sts_scale(addr, act_exp, w_exp, *, loc=None, ip=None):
    """Store the combined UE8M0 scale ``act_exp + w_exp - 127`` (clamped) as one byte."""
    _asm(
        None,
        [_iv(x, loc=loc, ip=ip) for x in (addr, act_exp, w_exp)],
        "{\n"
        ".reg .s32 v;\n"
        "add.s32 v, $1, $2;\n"
        "sub.s32 v, v, 127;\n"
        "max.s32 v, v, 0;\n"
        "min.s32 v, v, 254;\n"
        "st.shared.u8 [$0], v;\n"
        "}",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v4_f32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [
            _iv(addr, loc=loc, ip=ip),
            *(cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)),
        ],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_copy_to_peer(dst_local, src, nbytes, mbar_local, peer, *, loc=None, ip=None):
    """Bulk-copy ``nbytes`` of local SMEM into a peer CTA's SMEM, crediting its mbarrier.

    ``dst_local`` and ``mbar_local`` are offsets in this CTA's SMEM window; they are mapped
    to the peer's window, which has the same layout.
    """
    _asm(
        None,
        [_iv(x, loc=loc, ip=ip) for x in (dst_local, src, nbytes, mbar_local, peer)],
        "{\n"
        ".reg .b32 rd, rb;\n"
        "mapa.shared::cluster.u32 rd, $0, $4;\n"
        "mapa.shared::cluster.u32 rb, $3, $4;\n"
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
        "[rd], [$1], $2, [rb];\n"
        "}",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_commit(*, loc=None, ip=None):
    _asm(None, [], "cp.async.bulk.commit_group;", "", loc=loc, ip=ip)


@dsl_user_op
def _bulk_wait_read(*, loc=None, ip=None):
    """Wait until committed bulk copies have finished reading their SMEM sources."""
    _asm(None, [], "cp.async.bulk.wait_group.read 0;", "", loc=loc, ip=ip)


@dsl_user_op
def _mbarrier_arrive_remote(local_mbar, peer, *, loc=None, ip=None):
    """Relaxed arrive on the mbarrier at the same SMEM offset in a peer CTA.

    Only used to signal that reads of the local activation ring are done; those loads
    have already returned (their values were consumed), so no release fence is needed.
    """
    _asm(
        None,
        [_iv(local_mbar, loc=loc, ip=ip), _iv(peer, loc=loc, ip=ip)],
        "{\n"
        ".reg .b32 rb;\n"
        "mapa.shared::cluster.u32 rb, $0, $1;\n"
        "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [rb];\n"
        "}",
        "r,r",
        loc=loc,
        ip=ip,
    )


def _s2t_partition_sf(sSF: cute.Tensor, tSF: cute.Tensor):
    """S2T (SMEM -> TMEM) copy of a scale-factor operand, partitioned for one thread."""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E8M0FNU
    )
    tiled_copy = tcgen05.make_s2t_copy(copy_atom, tCtSF_compact)
    thr_copy = tiled_copy.get_slice(0)
    tCsSF_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy, thr_copy.partition_S(tCsSF_compact))
    tCtSF_s2t = thr_copy.partition_D(tCtSF_compact)
    return tiled_copy, tCsSF_s2t, tCtSF_s2t


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class Fp8BlockScaleLowMGemmKernel:
    """Swap-AB MXFP8 GEMM with in-kernel activation quantization and cluster split-K."""

    def __init__(self, n: int, k: int, tactic: LowMTactic, use_pdl: bool):
        validate_tactic(n, k, tactic)
        self.n = n
        self.k = k
        self.mma_n = tactic.mma_n
        self.split_k = tactic.split_k
        self.kt_total = k // CTA_K
        self.kt_max = _ceil_div(self.kt_total, self.split_k)
        self.a_stages = min(tactic.a_stages, self.kt_max)
        self.x_stages = min(tactic.x_stages, self.kt_max)
        self.b_stages = min(tactic.b_stages, self.kt_max)
        self.n_tiles = _ceil_div(n, MMA_M)
        self.use_pdl = use_pdl
        self.chunks_cap = _ceil_div(self.mma_n // 4, self.split_k)
        mailbox_bytes = (self.split_k - 1) * self.chunks_cap * MMA_M * 16
        self.xm_bytes = max(self.x_stages * self.mma_n * CTA_K * 2, mailbox_bytes)
        self.staging_bytes = self.split_k * self.chunks_cap * MMA_M * 16 if self.split_k > 1 else 0
        self.mma_tiler = (MMA_M, self.mma_n, CTA_K)
        self.a_stage_bytes = MMA_M * CTA_K
        self.b_stage_bytes = self.mma_n * CTA_K
        self.x_stage_bytes = self.mma_n * CTA_K * 2
        self.sf_stage_bytes = 512
        self.quant_warps = tactic.quant_warps
        self.quant_groups = tactic.quant_groups
        self.warps_per_group = tactic.quant_warps // tactic.quant_groups
        self.quant_threads = 32 * tactic.quant_warps
        self.threads = 32 * _QUANT_WARP0 + self.quant_threads

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
    ):
        n, k = self.n, self.k
        mW = cute.make_tensor(w_ptr, cute.make_layout((n, k, 1), stride=(k, 1, n * k)))
        mX = cute.make_tensor(x_ptr, cute.make_layout((m, k), stride=(k, 1)))
        # DeepGEMM packed UE8M0 weight scale: int32 [N, ceil(K/512)], MN-major; byte
        # (kb % 4) of word (n, kb // 4) holds the exponent of block (n // 128, kb).
        mWS = cute.make_tensor(
            ws_ptr,
            cute.make_layout(
                (_ceil_div(n, MMA_M), (4, _ceil_div(k // CTA_K, 4))),
                stride=(MMA_M * 4, (1, 4 * n)),
            ),
        )
        mOut = cute.make_tensor(out_ptr, cute.make_layout((m, n), stride=(n, 1)))

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            SF_VEC,
            tcgen05.CtaGroup.ONE,
            (MMA_M, self.mma_n),
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, self.a_stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, self.b_stages
        )
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, SF_VEC, 1)
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, self.b_stages
        )
        x_layout = cute.make_layout(
            (self.mma_n, CTA_K, self.x_stages), stride=(CTA_K, 1, self.mma_n * CTA_K)
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mW,
            cute.slice_(a_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_x, tma_tensor_x = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mX,
            cute.slice_(x_layout, (None, None, 0)),
            (self.mma_n, CTA_K),
        )

        @cute.struct
        class SharedStorage:
            a_full: cute.struct.MemRange[cutlass.Int64, self.a_stages]
            a_empty: cute.struct.MemRange[cutlass.Int64, self.a_stages]
            x_full: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            x_empty: cute.struct.MemRange[cutlass.Int64, self.x_stages]
            b_full: cute.struct.MemRange[cutlass.Int64, self.b_stages]
            b_empty: cute.struct.MemRange[cutlass.Int64, self.b_stages]
            acc_full: cutlass.Int64
            tmem_ready: cutlass.Int64
            tmem_free: cutlass.Int64
            mbox_full: cutlass.Int64
            peers_quantized: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            w_exp: cute.struct.MemRange[cutlass.Int32, self.kt_max]
            # Weight ring; reused to stage outgoing split-K partials after the mainloop.
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E4M3FN, max(cute.cosize(a_layout.outer), self.staging_bytes)
                ],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(b_layout.outer)], 1024
            ]
            # Activation ring; reused as the split-K reduction mailbox after quantization.
            sXM: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.xm_bytes], 128]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfa_layout)], 128
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfb_layout)], 128
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_x,
            tma_tensor_x,
            mWS,
            mOut,
            m,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            x_layout,
        ).launch(
            grid=(self.n_tiles * self.split_k, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.split_k, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mXt: cute.Tensor,
        mWS: cute.Tensor,
        mOut: cute.Tensor,
        m: Int32,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        x_layout: cute.Layout,
    ):
        S = self.split_k
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        rank = Int32(0)
        if cutlass.const_expr(S > 1):
            rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_tile = bidx // S

        kt_base = self.kt_total // S
        kt_rem = self.kt_total % S
        kt_count = Int32(kt_base)
        kt_start = rank * kt_base
        if cutlass.const_expr(kt_rem > 0):
            if rank < kt_rem:
                kt_count = kt_count + 1
                kt_start = kt_start + rank
            else:
                kt_start = kt_start + kt_rem

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        a_full = storage.a_full.data_ptr()
        a_empty = storage.a_empty.data_ptr()
        x_full = storage.x_full.data_ptr()
        x_empty = storage.x_empty.data_ptr()
        b_full = storage.b_full.data_ptr()
        b_empty = storage.b_empty.data_ptr()
        acc_full = storage.acc_full
        tmem_ready = storage.tmem_ready
        tmem_free = storage.tmem_free
        mbox_full = storage.mbox_full
        peers_quantized = storage.peers_quantized
        tmem_holding_buf = storage.tmem_holding_buf

        if warp_idx == _TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_x)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.a_stages):
                    cute.arch.mbarrier_init(a_full + i, 1)
                    cute.arch.mbarrier_init(a_empty + i, 1)
                for i in cutlass.range_constexpr(self.x_stages):
                    cute.arch.mbarrier_init(x_full + i, 1)
                    cute.arch.mbarrier_init(x_empty + i, self.warps_per_group)
                for i in cutlass.range_constexpr(self.b_stages):
                    cute.arch.mbarrier_init(b_full + i, self.warps_per_group)
                    cute.arch.mbarrier_init(b_empty + i, 1)
                cute.arch.mbarrier_init(acc_full, 1)
                cute.arch.mbarrier_init(tmem_ready, 1)
                cute.arch.mbarrier_init(tmem_free, 4)
                cute.arch.mbarrier_init(mbox_full, 1)
                cute.arch.mbarrier_init(peers_quantized, max(S - 1, 1))
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if cutlass.const_expr(S > 1):
            cute.arch.cluster_arrive_relaxed()

        if warp_idx == _TMA_WARP:
            # Weight producer: never waits on the previous grid.
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            gA = cute.local_tile(
                mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            tCgA = tiled_mma.get_slice(0).partition_A(gA)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            for kt in cutlass.range(kt_count, unroll=1):
                stage = kt % self.a_stages
                if kt >= self.a_stages:
                    cute.arch.mbarrier_wait(a_empty + stage, ((kt // self.a_stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, self.a_stage_bytes)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, n_tile, kt_start + kt, 0)],
                    tAsA[(None, stage)],
                    tma_bar_ptr=a_full + stage,
                    cache_policy=Int64(_EVICT_FIRST),
                )
        elif warp_idx == _X_WARP:
            # Activation producer: the only TMA that depends on the previous grid.
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_wait()
            sX = cute.make_tensor(
                cute.recast_ptr(storage.sXM.data_ptr(), dtype=cutlass.BFloat16), x_layout
            )
            gX = cute.local_tile(mXt, (self.mma_n, CTA_K), (0, None))
            tXsX, tXgX = cpasync.tma_partition(
                tma_atom_x,
                0,
                cute.make_layout(1),
                cute.group_modes(sX, 0, 2),
                cute.group_modes(gX, 0, 2),
            )
            for kt in cutlass.range(kt_count, unroll=1):
                stage = kt % self.x_stages
                if kt >= self.x_stages:
                    cute.arch.mbarrier_wait(x_empty + stage, ((kt // self.x_stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(x_full + stage, self.x_stage_bytes)
                cute.copy(
                    tma_atom_x,
                    tXgX[(None, kt_start + kt)],
                    tXsX[(None, stage)],
                    tma_bar_ptr=x_full + stage,
                )
        elif warp_idx == _MMA_WARP:
            self.mma_warp(
                tiled_mma,
                storage,
                a_layout,
                b_layout,
                sfa_layout,
                sfb_layout,
                a_full,
                a_empty,
                b_full,
                b_empty,
                acc_full,
                tmem_ready,
                tmem_free,
                tmem_holding_buf,
                kt_count,
            )
        elif warp_idx >= _QUANT_WARP0:
            q = tidx - _QUANT_WARP0 * 32
            self.quantize_activations(
                storage,
                mWS,
                q,
                n_tile,
                kt_start,
                kt_count,
                x_full,
                x_empty,
                b_full,
                b_empty,
            )
            if cutlass.const_expr(S > 1):
                # The activation ring becomes the split-K mailbox: tell every peer once
                # all quantization warps are done reading it.
                cute.arch.barrier(
                    barrier_id=_QUANT_DONE_BARRIER_ID, number_of_threads=self.quant_threads
                )
                if warp_idx < _QUANT_WARP0 + 4:
                    # Peers' mbarriers are initialized once the start-up cluster barrier
                    # completes (``barrier.cluster.wait`` is warp-aligned).
                    cute.arch.cluster_wait()
                    if q == 0:
                        for p in cutlass.range_constexpr(1, S):
                            _mbarrier_arrive_remote(peers_quantized.toint(), (rank + p) % S)
            if warp_idx < _QUANT_WARP0 + 4:
                self.epilogue(
                    tiled_mma,
                    storage,
                    mOut,
                    m,
                    q,
                    n_tile,
                    rank,
                    acc_full,
                    tmem_ready,
                    tmem_free,
                    mbox_full,
                    peers_quantized,
                    tmem_holding_buf,
                )

    @cute.jit
    def mma_warp(
        self,
        tiled_mma: cute.TiledMma,
        storage,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        a_full,
        a_empty,
        b_full,
        b_empty,
        acc_full,
        tmem_ready,
        tmem_free,
        tmem_holding_buf,
        kt_count: Int32,
    ):
        cute.arch.alloc_tmem(_TMEM_COLS, tmem_holding_buf)
        cute.arch.relinquish_tmem_alloc_permit()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(tmem_ready)

        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
        sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + self.mma_n, dtype=cutlass.Float8E8M0FNU)
        tCtSFA = cute.make_tensor(
            sfa_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfa_layout, (None, None, None, 0))
            ),
        )
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(acc_tmem_ptr + self.mma_n + _SF_TMEM_COLS, dtype=cutlass.Float8E8M0FNU),
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfb_layout, (None, None, None, 0))
            ),
        )
        s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = _s2t_partition_sf(
            storage.sSFA.get_tensor(sfa_layout), tCtSFA
        )
        s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = _s2t_partition_sf(
            storage.sSFB.get_tensor(sfb_layout), tCtSFB
        )

        # The weight scale is folded into SFB, so SFA is a constant 1.0 (UE8M0 127).
        lane = cute.arch.lane_idx()
        one4 = Int32(0x7F7F7F7F)
        _sts_v4_b32(storage.sSFA.data_ptr().toint() + lane * 16, one4, one4, one4, one4)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()
        cute.copy(s2t_sfa, tCsSFA_s2t[(None, None, None, None, 0)], tCtSFA_s2t)

        sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        num_kphases = cute.size(tCrA, mode=[2])

        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kt in cutlass.range(kt_count, unroll=1):
            a_stage = kt % self.a_stages
            b_stage = kt % self.b_stages
            cute.arch.mbarrier_wait(a_full + a_stage, (kt // self.a_stages) & 1)
            cute.arch.mbarrier_wait(b_full + b_stage, (kt // self.b_stages) & 1)
            cute.copy(s2t_sfb, tCsSFB_s2t[(None, None, None, None, b_stage)], tCtSFB_s2t)
            for kp in cutlass.range_constexpr(num_kphases):
                tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[(None, None, kp, a_stage)],
                    tCrB[(None, None, kp, b_stage)],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(a_empty + a_stage)
                tcgen05.commit(b_empty + b_stage)
        with cute.arch.elect_one():
            tcgen05.commit(acc_full)

        cute.arch.mbarrier_wait(tmem_free, 0)
        cute.arch.dealloc_tmem(acc_tmem_ptr, _TMEM_COLS)

    @cute.jit
    def quantize_activations(
        self,
        storage,
        mWS: cute.Tensor,
        q: Int32,
        n_tile: Int32,
        kt_start: Int32,
        kt_count: Int32,
        x_full,
        x_empty,
        b_full,
        b_empty,
    ):
        """Quantize BF16 activation tiles into the swizzled MXFP8 UMMA B layout.

        K tiles are dealt round-robin to ``quant_groups`` groups of warps, so several tiles
        are in flight at once; ring slot ``kt % stages`` is then always served by the same
        group. Within a tile, one work item is one (row, 32-element K group): four 16-byte
        loads, one UE8M0 scale, 32 E4M3 bytes, split evenly over the group's lanes.
        """
        nthr = self.quant_threads
        w_exp = storage.w_exp.get_tensor(cute.make_layout(self.kt_max))
        for kt in cutlass.range(q, kt_count, nthr, unroll=1):
            w_exp[kt] = Int32(mWS[n_tile, kt_start + kt])
        cute.arch.barrier(barrier_id=_QUANT_BARRIER_ID, number_of_threads=nthr)

        x_base = storage.sXM.data_ptr().toint()
        b_base = storage.sB.data_ptr().toint()
        sf_base = storage.sSFB.data_ptr().toint()
        group_threads = 32 * self.warps_per_group
        gq = q % group_threads
        for kt in cutlass.range(q // group_threads, kt_count, self.quant_groups, unroll=1):
            x_stage = kt % self.x_stages
            b_stage = kt % self.b_stages
            cute.arch.mbarrier_wait(x_full + x_stage, (kt // self.x_stages) & 1)
            if kt >= self.b_stages:
                cute.arch.mbarrier_wait(b_empty + b_stage, ((kt // self.b_stages) & 1) ^ 1)
            e_w = w_exp[kt]
            xs = x_base + x_stage * self.x_stage_bytes
            bs = b_base + b_stage * self.b_stage_bytes
            ss = sf_base + b_stage * self.sf_stage_bytes
            for j in cutlass.range_constexpr(self.mma_n * 4 // group_threads):
                item = gq + group_threads * j
                r = item >> 2
                g = item & 3
                # Rotate the 16-byte chunk order so a quarter-warp hits 8 distinct banks.
                rot = item >> 1
                xrow = xs + r * (CTA_K * 2) + g * 64
                ws = []
                for v in cutlass.range_constexpr(4):
                    ws.append(_lds_v4_b32(xrow + ((v + rot) & 3) * 16))
                amax = _abs_max_bf16x32([w for quad in ws for w in quad])
                e_a = _ue8m0_of_amax(amax)
                inv = _bf16x2_inv_scale(e_a)
                brow = bs + r * CTA_K
                swz = r & 7
                for v in cutlass.range_constexpr(4):
                    sub = (v + rot) & 3
                    p0, p1 = _quant_bf16x8_to_e4m3(*ws[v], inv)
                    unit = g * 2 + (sub >> 1)
                    _sts_v2_b32(brow + ((unit ^ swz) << 4) + (sub & 1) * 8, p0, p1)
                _sts_scale(ss + (r & 31) * 16 + (r >> 5) * 4 + g, e_a, e_w)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(x_empty + x_stage)
                cute.arch.mbarrier_arrive(b_full + b_stage)

    @cute.jit
    def epilogue(
        self,
        tiled_mma: cute.TiledMma,
        storage,
        mOut: cute.Tensor,
        m: Int32,
        q: Int32,
        n_tile: Int32,
        rank: Int32,
        acc_full,
        tmem_ready,
        tmem_free,
        mbox_full,
        peers_quantized,
        tmem_holding_buf,
    ):
        S = self.split_k
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
        cute.arch.mbarrier_wait(tmem_ready, 0)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
        tAcc = tCtAcc[((None, None), 0, 0)]
        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(self.mma_n)), cutlass.Float32
        )
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc)
        thr_t2r = tiled_t2r.get_slice(q)
        tTR_tAcc = thr_t2r.partition_S(tAcc)
        tTR_cAcc = thr_t2r.partition_D(cute.make_identity_tensor((MMA_M, self.mma_n)))
        rAcc = cute.make_rmem_tensor(tTR_cAcc.shape, cutlass.Float32)

        cute.arch.mbarrier_wait(acc_full, 0)
        cute.copy(tiled_t2r, tTR_tAcc, rAcc)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(tmem_free)

        n_base = n_tile * MMA_M
        nchunks = cute.size(rAcc) // 4
        # Every thread holds one accumulator row. Rows past N (the zero-filled tail of a
        # partial last weight tile) are reduced like the others but never stored.
        m_store = m
        if cutlass.const_expr(self.n % MMA_M != 0):
            row0, _ = tTR_cAcc[0]
            if n_base + row0 >= self.n:
                m_store = Int32(0)
        if cutlass.const_expr(S == 1):
            for i in cutlass.range_constexpr(nchunks * 4):
                row, col = tTR_cAcc[i]
                if col < m_store:
                    mOut[col, n_base + row] = rAcc[i].to(cutlass.BFloat16)
        else:
            # Reduce-scatter over 4-column chunks: chunk c is owned by rank c % S. Every
            # thread holds one accumulator row. Partials for each owner are staged in the
            # (now dead) weight ring as [chunk][row] and pushed with one bulk DSMEM copy per
            # owner into its mailbox (its dead activation ring), laid out [peer][chunk][row].
            mbox_base = storage.sXM.data_ptr().toint()
            stage_base = storage.sA.data_ptr().toint()
            mbar_addr = mbox_full.toint()
            slot_bytes = self.chunks_cap * MMA_M * 16
            my_chunks = Int32(0)
            for c in cutlass.range_constexpr(nchunks):
                _, col0 = tTR_cAcc[4 * c]
                if col0 < m:
                    if rank == c % S:
                        my_chunks = my_chunks + 1
                    else:
                        _sts_v4_f32(
                            stage_base + (c % S) * slot_bytes + ((c // S) * MMA_M + q) * 16,
                            rAcc[4 * c],
                            rAcc[4 * c + 1],
                            rAcc[4 * c + 2],
                            rAcc[4 * c + 3],
                        )
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=MMA_M)
            if q == 0:
                cute.arch.mbarrier_wait(peers_quantized, 0)
                for o in cutlass.range_constexpr(S):
                    if rank != o:
                        sent = Int32(0)
                        for c in cutlass.range_constexpr(o, nchunks, S):
                            _, col0 = tTR_cAcc[4 * c]
                            if col0 < m:
                                sent = sent + 1
                        if sent > 0:
                            slot = rank
                            if rank > o:
                                slot = rank - 1
                            _bulk_copy_to_peer(
                                mbox_base + slot * slot_bytes,
                                stage_base + o * slot_bytes,
                                sent * (MMA_M * 16),
                                mbar_addr,
                                o,
                            )
                _bulk_commit()
            if my_chunks > 0:
                if q == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbox_full, my_chunks * ((S - 1) * MMA_M * 16)
                    )
                cute.arch.mbarrier_wait(mbox_full, 0)
                for c in cutlass.range_constexpr(nchunks):
                    _, col0 = tTR_cAcc[4 * c]
                    if col0 < m:
                        if rank == c % S:
                            for p in cutlass.range_constexpr(S - 1):
                                v0, v1, v2, v3 = _lds_v4_f32(
                                    mbox_base + p * slot_bytes + ((c // S) * MMA_M + q) * 16
                                )
                                rAcc[4 * c] = rAcc[4 * c] + v0
                                rAcc[4 * c + 1] = rAcc[4 * c + 1] + v1
                                rAcc[4 * c + 2] = rAcc[4 * c + 2] + v2
                                rAcc[4 * c + 3] = rAcc[4 * c + 3] + v3
                            for j in cutlass.range_constexpr(4):
                                row, col = tTR_cAcc[4 * c + j]
                                if col < m_store:
                                    mOut[col, n_base + row] = rAcc[4 * c + j].to(cutlass.BFloat16)
            if q == 0:
                # The staging buffer must stay valid until the copies have read it.
                _bulk_wait_read()


# ---------------------------------------------------------------------------
# Mid-M path (M up to 256): standalone MXFP8 activation quantization + GEMM with a
# TMA-loaded FP8 activation operand.
# ---------------------------------------------------------------------------

#: Token tiles (UMMA N) of the mid-M GEMM: M rounds up to a multiple of 32 (the epilogue
#: drains TMEM in 32-column blocks); MMA time scales with the padded N.
SUPPORTED_MID_MMA_N = (64, 96, 128, 160, 192, 224, 256)
#: Leave room for the quantization kernel's CTAs (1 KB reserved SMEM each) next to a GEMM
#: CTA, so the GEMM can launch and stream weights while the quantization still runs.
_MID_SMEM_CAPACITY = _SMEM_CAPACITY - 2048


@dataclasses.dataclass(frozen=True)
class MidMTactic:
    """One mid-M specialization: ``mma_n`` padded tokens, cluster ``split_k``, ring depths."""

    mma_n: int
    split_k: int
    a_stages: int
    b_stages: int = 3


def _sf_blocks(mma_n: int) -> int:
    """128-row scale-factor atoms covering ``mma_n`` tokens."""
    return _ceil_div(mma_n, MMA_M)


def _mid_mailbox_bytes(tactic: MidMTactic) -> int:
    if tactic.split_k == 1:
        return 0
    return (tactic.split_k - 1) * _ceil_div(tactic.mma_n // 4, tactic.split_k) * MMA_M * 16


def mid_smem_bytes(n: int, k: int, tactic: MidMTactic) -> int:
    """Conservative host-side mirror of the mid-M kernel's shared-memory footprint."""
    kt_max = _ceil_div(k // CTA_K, tactic.split_k)
    a_st = min(tactic.a_stages, kt_max)
    b_st = min(tactic.b_stages, kt_max)
    total = 8 * (2 * (a_st + b_st) + 5) + 4
    # The split-K mailbox reuses the weight ring once the mainloop is done.
    total = _align_up(total, 1024) + max(a_st * MMA_M * CTA_K, _mid_mailbox_bytes(tactic))
    total = _align_up(total, 1024) + b_st * tactic.mma_n * CTA_K
    total = _align_up(total, 128) + kt_max * 512
    total = _align_up(total, 128) + b_st * _sf_blocks(tactic.mma_n) * 512
    return total + 1024  # allocator slack


def validate_mid_tactic(n: int, k: int, tactic: MidMTactic) -> None:
    if tactic.mma_n not in SUPPORTED_MID_MMA_N:
        raise ValueError(f"mma_n must be one of {SUPPORTED_MID_MMA_N}, got {tactic.mma_n}")
    if n % N_ALIGN or k % CTA_K:
        raise ValueError(f"N={n} must be a multiple of {N_ALIGN} and K={k} of {CTA_K}")
    if not 1 <= tactic.split_k <= min(16, k // CTA_K):
        raise ValueError(f"split_k={tactic.split_k} invalid for K={k}")
    if tactic.a_stages < 1 or tactic.b_stages < 1:
        raise ValueError("a_stages and b_stages must be >= 1")
    need = mid_smem_bytes(n, k, tactic)
    if need > _MID_SMEM_CAPACITY:
        raise ValueError(f"tactic {tactic} needs {need} B of SMEM > {_SMEM_CAPACITY} B")


def max_mid_a_stages(n: int, k: int, tactic: MidMTactic) -> int:
    kt_max = _ceil_div(k // CTA_K, tactic.split_k)
    best = 1
    for a in range(1, kt_max + 1):
        if mid_smem_bytes(n, k, dataclasses.replace(tactic, a_stages=a)) <= _MID_SMEM_CAPACITY:
            best = a
    return best


@dsl_user_op
def _ldg_v4_b32(addr, *, loc=None, ip=None):
    """Load 16 bytes from global memory as four 32-bit words."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _stg_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u8(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_load_g2s(dst_smem, src_gmem, nbytes, mbar, *, loc=None, ip=None):
    """Bulk-copy ``nbytes`` from global memory into local SMEM, crediting ``mbar``."""
    _asm(
        None,
        [
            _iv(dst_smem, loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            _iv(nbytes, loc=loc, ip=ip),
            _iv(mbar, loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_v4_f32(local_addr, v0, v1, v2, v3, local_mbar, peer, *, loc=None, ip=None):
    """Store four FP32 values into a peer CTA's SMEM, crediting its mbarrier."""
    _asm(
        None,
        [
            _iv(local_addr, loc=loc, ip=ip),
            *(cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)),
            _iv(local_mbar, loc=loc, ip=ip),
            _iv(peer, loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .b32 ra, rb;\n"
        "mapa.shared::cluster.u32 ra, $0, $6;\n"
        "mapa.shared::cluster.u32 rb, $5, $6;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [ra], {$1, $2, $3, $4}, [rb];\n"
        "}",
        "r,f,f,f,f,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_u16(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.u16 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _f32_to_bf16_bits(value, *, loc=None, ip=None):
    """Round-to-nearest-even FP32 -> BF16, returned in the low 16 bits of an Int32."""
    return Int32(
        _asm(
            T.i32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\ncvt.u32.u16 $0, h;\n}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _min_s32(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "min.s32 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


class MxFp8QuantKernel:
    """``x`` BF16 ``[M, K]`` -> E4M3 ``[M, K]`` + UE8M0 1x32 scales for the mid-M GEMM.

    One work item is one (row, 32-element group), quantized exactly like the low-M
    kernel's in-kernel path; each thread issues the loads of its ``_ITEMS`` items before
    converting any. CTAs are small so they fit next to a resident GEMM CTA (which
    launches early via PDL and streams its weights meanwhile).
    Scales are written in the tcgen05 scale-factor atom layout ``[K/128][sf_blocks][512 B]``
    with atom byte ``(r % 32) * 16 + (r // 32) * 4 + k32 % 4``, so the GEMM moves one
    contiguous ``sf_blocks * 512`` byte block per K tile. Rows past ``M`` (up to
    ``128 * sf_blocks``) get the neutral scale 1.0.
    """

    _THREADS = 128
    _ITEMS = 2

    def __init__(self, k: int, sf_blocks: int, use_pdl: bool):
        self.k = k
        self.sf_blocks = sf_blocks
        self.groups = k // SF_VEC
        self.items = MMA_M * sf_blocks * self.groups
        self.items_per_thread = self._ITEMS
        self.ctas = _ceil_div(self.items, self._THREADS * self._ITEMS)
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        xq_ptr: cute.Pointer,
        xs_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x_ptr, xq_ptr, xs_ptr, m).launch(
            grid=(self.ctas, 1, 1),
            block=(self._THREADS, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(self, x_ptr: cute.Pointer, xq_ptr: cute.Pointer, xs_ptr: cute.Pointer, m: Int32):
        if cutlass.const_expr(self.use_pdl):
            # The GEMM only reads this grid's results after its own griddepcontrol.wait;
            # let it launch right away so its weight stream overlaps this kernel.
            griddepcontrol_launch_dependents()
            griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        ipt = self.items_per_thread
        rX = cute.make_rmem_tensor((ipt * 16,), Int32)
        # Issue every load first; out-of-range items re-read a valid row and are not stored.
        for it in cutlass.range_constexpr(ipt):
            item = _min_s32(
                bidx * self._THREADS + tidx + it * (self.ctas * self._THREADS), self.items - 1
            )
            r = _min_s32(item // self.groups, m - 1)
            g = item - (item // self.groups) * self.groups
            src = x_ptr.toint() + Int64(r * self.k + g * SF_VEC) * 2
            for v in cutlass.range_constexpr(4):
                w0, w1, w2, w3 = _ldg_v4_b32(src + v * 16)
                rX[it * 16 + v * 4] = w0
                rX[it * 16 + v * 4 + 1] = w1
                rX[it * 16 + v * 4 + 2] = w2
                rX[it * 16 + v * 4 + 3] = w3
        for it in cutlass.range_constexpr(ipt):
            item = bidx * self._THREADS + tidx + it * (self.ctas * self._THREADS)
            if item < self.items:
                r = item // self.groups
                g = item - r * self.groups
                rr = r & (MMA_M - 1)
                sf_off = ((g >> 2) * self.sf_blocks + (r >> 7)) * 512
                sf_off = sf_off + (rr & 31) * 16 + (rr >> 5) * 4 + (g & 3)
                sf_addr = xs_ptr.toint() + Int64(sf_off)
                if r < m:
                    ws = [rX[it * 16 + i] for i in range(16)]
                    e_a = _ue8m0_of_amax(_abs_max_bf16x32(ws))
                    inv = _bf16x2_inv_scale(e_a)
                    packed = []
                    for v in cutlass.range_constexpr(4):
                        p0, p1 = _quant_bf16x8_to_e4m3(*ws[4 * v : 4 * v + 4], inv)
                        packed.extend((p0, p1))
                    dst = xq_ptr.toint() + Int64(r * self.k + g * SF_VEC)
                    _stg_v4_b32(dst, *packed[:4])
                    _stg_v4_b32(dst + 16, *packed[4:])
                    _stg_u8(sf_addr, e_a)
                else:
                    _stg_u8(sf_addr, Int32(127))


class Fp8BlockScaleMidMGemmKernel:
    """Swap-AB MXFP8 GEMM fed by pre-quantized FP8 activations (64 < M <= 256).

    Same weight path and split-K cluster as :class:`Fp8BlockScaleLowMGemmKernel`, but the
    activation operand is the output of :class:`MxFp8QuantKernel`: warp 2 loads the
    E4M3 tile with TMA (128B swizzle, i.e. directly in the UMMA B layout) and its scale
    block with a bulk copy. The weight block scale goes into SFA (one UE8M0 per K tile,
    replicated over the 128 rows), written once to SMEM before the mainloop. Warps 4-7 run
    the epilogue in 32-column TMEM blocks; with ``split_k > 1`` each rank owns every
    ``split_k``-th 4-token chunk and peers push partials with ``st.async`` into its
    weight ring once every CTA of the cluster has finished its mainloop.
    """

    _THREADS = 256
    _B_WARP = 2
    _EPI_WARP0 = 4

    def __init__(self, n: int, k: int, tactic: MidMTactic, use_pdl: bool):
        validate_mid_tactic(n, k, tactic)
        self.n = n
        self.k = k
        self.mma_n = tactic.mma_n
        self.split_k = tactic.split_k
        self.kt_total = k // CTA_K
        self.kt_max = _ceil_div(self.kt_total, self.split_k)
        self.a_stages = min(tactic.a_stages, self.kt_max)
        self.b_stages = min(tactic.b_stages, self.kt_max)
        self.n_tiles = _ceil_div(n, MMA_M)
        self.use_pdl = use_pdl
        self.mma_tiler = (MMA_M, self.mma_n, CTA_K)
        self.a_stage_bytes = MMA_M * CTA_K
        self.b_stage_bytes = self.mma_n * CTA_K
        self.sf_blocks = _sf_blocks(self.mma_n)
        self.sfb_stage_bytes = self.sf_blocks * 512
        self.chunks = self.mma_n // 4
        self.chunks_cap = _ceil_div(self.chunks, self.split_k)
        self.a_region_bytes = max(self.a_stages * self.a_stage_bytes, _mid_mailbox_bytes(tactic))
        self.out_stage_bytes = self.mma_n * MMA_M * 2
        # Accumulator columns plus SFA / SFB columns, rounded to a power of two.
        cols = self.mma_n + 2 * _SF_TMEM_COLS
        self.tmem_cols = 1 << (cols - 1).bit_length()

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        xq_ptr: cute.Pointer,
        xs_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
    ):
        n, k = self.n, self.k
        mW = cute.make_tensor(w_ptr, cute.make_layout((n, k, 1), stride=(k, 1, n * k)))
        mXq = cute.make_tensor(xq_ptr, cute.make_layout((m, k, 1), stride=(k, 1, m * k)))
        mWS = cute.make_tensor(
            ws_ptr,
            cute.make_layout(
                (_ceil_div(n, MMA_M), (4, _ceil_div(k // CTA_K, 4))),
                stride=(MMA_M * 4, (1, 4 * n)),
            ),
        )
        mOut = cute.make_tensor(out_ptr, cute.make_layout((m, n), stride=(n, 1)))

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            SF_VEC,
            tcgen05.CtaGroup.ONE,
            (MMA_M, self.mma_n),
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, self.a_stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, self.b_stages
        )
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, self.kt_max
        )
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, self.b_stages
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mW,
            cute.slice_(a_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mXq,
            cute.slice_(b_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        @cute.struct
        class SharedStorage:
            a_full: cute.struct.MemRange[cutlass.Int64, self.a_stages]
            a_empty: cute.struct.MemRange[cutlass.Int64, self.a_stages]
            b_full: cute.struct.MemRange[cutlass.Int64, self.b_stages]
            b_empty: cute.struct.MemRange[cutlass.Int64, self.b_stages]
            acc_full: cutlass.Int64
            tmem_ready: cutlass.Int64
            tmem_free: cutlass.Int64
            mbox_full: cutlass.Int64
            peers_done: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # Weight ring; reused as the split-K mailbox after the mainloop.
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, self.a_region_bytes], 1024
            ]
            # Activation ring; reused to stage the BF16 output tile token-major.
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float8E4M3FN, max(cute.cosize(b_layout.outer), self.out_stage_bytes)
                ],
                1024,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfa_layout)], 128
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfb_layout)], 128
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            xs_ptr,
            mWS,
            mOut,
            m,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
        ).launch(
            grid=(self.n_tiles * self.split_k, 1, 1),
            block=(self._THREADS, 1, 1),
            cluster=(self.split_k, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            # Caps registers at 128/thread so a quantization CTA can be co-resident.
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        xs_ptr: cute.Pointer,
        mWS: cute.Tensor,
        mOut: cute.Tensor,
        m: Int32,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
    ):
        S = self.split_k
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        rank = Int32(0)
        if cutlass.const_expr(S > 1):
            rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_tile = bidx // S

        kt_base = self.kt_total // S
        kt_rem = self.kt_total % S
        kt_count = Int32(kt_base)
        kt_start = rank * kt_base
        if cutlass.const_expr(kt_rem > 0):
            if rank < kt_rem:
                kt_count = kt_count + 1
                kt_start = kt_start + rank
            else:
                kt_start = kt_start + kt_rem

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        a_full = storage.a_full.data_ptr()
        a_empty = storage.a_empty.data_ptr()
        b_full = storage.b_full.data_ptr()
        b_empty = storage.b_empty.data_ptr()

        if warp_idx == _TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.a_stages):
                    cute.arch.mbarrier_init(a_full + i, 1)
                    cute.arch.mbarrier_init(a_empty + i, 1)
                for i in cutlass.range_constexpr(self.b_stages):
                    cute.arch.mbarrier_init(b_full + i, 1)
                    cute.arch.mbarrier_init(b_empty + i, 1)
                cute.arch.mbarrier_init(storage.acc_full, 1)
                cute.arch.mbarrier_init(storage.tmem_ready, 1)
                cute.arch.mbarrier_init(storage.tmem_free, 4)
                cute.arch.mbarrier_init(storage.mbox_full, 1)
                cute.arch.mbarrier_init(storage.peers_done, max(S - 1, 1))
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if cutlass.const_expr(S > 1):
            cute.arch.cluster_arrive_relaxed()

        if warp_idx == _TMA_WARP:
            # Weight producer: never waits on the previous grid.
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            gA = cute.local_tile(
                mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            tCgA = tiled_mma.get_slice(0).partition_A(gA)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            for kt in cutlass.range(kt_count, unroll=1):
                stage = kt % self.a_stages
                if kt >= self.a_stages:
                    cute.arch.mbarrier_wait(a_empty + stage, ((kt // self.a_stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, self.a_stage_bytes)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, n_tile, kt_start + kt, 0)],
                    tAsA[(None, stage)],
                    tma_bar_ptr=a_full + stage,
                    cache_policy=Int64(_EVICT_FIRST),
                )
        elif warp_idx == self._B_WARP:
            # Activation producer: E4M3 tile by TMA plus its scale block by a bulk copy.
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_wait()
            sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
            gB = cute.local_tile(
                mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            tCgB = tiled_mma.get_slice(0).partition_B(gB)
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )
            sfb_base = storage.sSFB.data_ptr().toint()
            for kt in cutlass.range(kt_count, unroll=1):
                stage = kt % self.b_stages
                if kt >= self.b_stages:
                    cute.arch.mbarrier_wait(b_empty + stage, ((kt // self.b_stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        b_full + stage, self.b_stage_bytes + self.sfb_stage_bytes
                    )
                    _bulk_load_g2s(
                        sfb_base + stage * self.sfb_stage_bytes,
                        xs_ptr.toint() + Int64((kt_start + kt) * self.sfb_stage_bytes),
                        self.sfb_stage_bytes,
                        (b_full + stage).toint(),
                    )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, kt_start + kt, 0)],
                    tBsB[(None, stage)],
                    tma_bar_ptr=b_full + stage,
                )
        elif warp_idx == _MMA_WARP:
            self.mma_warp(
                tiled_mma,
                storage,
                mWS,
                n_tile,
                kt_start,
                kt_count,
                a_layout,
                b_layout,
                sfa_layout,
                sfb_layout,
            )
        elif warp_idx >= self._EPI_WARP0:
            self.epilogue(
                tiled_mma,
                storage,
                mOut,
                m,
                tidx - 32 * self._EPI_WARP0,
                n_tile,
                rank,
            )

    @cute.jit
    def mma_warp(
        self,
        tiled_mma: cute.TiledMma,
        storage,
        mWS: cute.Tensor,
        n_tile: Int32,
        kt_start: Int32,
        kt_count: Int32,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
    ):
        a_full = storage.a_full.data_ptr()
        a_empty = storage.a_empty.data_ptr()
        b_full = storage.b_full.data_ptr()
        b_empty = storage.b_empty.data_ptr()
        lane = cute.arch.lane_idx()

        # SFA for every K tile: the weight block exponent replicated over 128 rows x 4
        # groups. Weights are static, so this needs no wait on the previous grid.
        sfa_base = storage.sSFA.data_ptr().toint()
        exps = []
        for j in cutlass.range_constexpr(_ceil_div(self.kt_max, 32)):
            e = Int32(127)
            if lane + 32 * j < kt_count:
                e = Int32(mWS[n_tile, kt_start + lane + 32 * j])
            exps.append(e)
        for kt in cutlass.range(kt_count, unroll=1):
            e = cute.arch.shuffle_sync(exps[0], kt & 31)
            for j in cutlass.range_constexpr(1, len(exps)):
                ej = cute.arch.shuffle_sync(exps[j], kt & 31)
                if (kt >> 5) == j:
                    e = ej
            word = e * Int32(0x01010101)
            _sts_v4_b32(sfa_base + kt * 512 + lane * 16, word, word, word, word)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()

        cute.arch.alloc_tmem(self.tmem_cols, storage.tmem_holding_buf)
        cute.arch.relinquish_tmem_alloc_permit()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(storage.tmem_ready)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
        tCtSFA = cute.make_tensor(
            cute.recast_ptr(acc_tmem_ptr + self.mma_n, dtype=cutlass.Float8E8M0FNU),
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfa_layout, (None, None, None, 0))
            ),
        )
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(acc_tmem_ptr + self.mma_n + _SF_TMEM_COLS, dtype=cutlass.Float8E8M0FNU),
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfb_layout, (None, None, None, 0))
            ),
        )
        s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = _s2t_partition_sf(
            storage.sSFA.get_tensor(sfa_layout), tCtSFA
        )
        s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = _s2t_partition_sf(
            storage.sSFB.get_tensor(sfb_layout), tCtSFB
        )
        sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        num_kphases = cute.size(tCrA, mode=[2])

        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kt in cutlass.range(kt_count, unroll=1):
            a_stage = kt % self.a_stages
            b_stage = kt % self.b_stages
            cute.arch.mbarrier_wait(a_full + a_stage, (kt // self.a_stages) & 1)
            cute.arch.mbarrier_wait(b_full + b_stage, (kt // self.b_stages) & 1)
            cute.copy(s2t_sfa, tCsSFA_s2t[(None, None, None, None, kt)], tCtSFA_s2t)
            cute.copy(s2t_sfb, tCsSFB_s2t[(None, None, None, None, b_stage)], tCtSFB_s2t)
            for kp in cutlass.range_constexpr(num_kphases):
                tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[(None, None, kp, a_stage)],
                    tCrB[(None, None, kp, b_stage)],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(a_empty + a_stage)
                tcgen05.commit(b_empty + b_stage)
        with cute.arch.elect_one():
            tcgen05.commit(storage.acc_full)

        cute.arch.mbarrier_wait(storage.tmem_free, 0)
        cute.arch.dealloc_tmem(acc_tmem_ptr, self.tmem_cols)

    @cute.jit
    def epilogue(
        self,
        tiled_mma: cute.TiledMma,
        storage,
        mOut: cute.Tensor,
        m: Int32,
        q: Int32,
        n_tile: Int32,
        rank: Int32,
    ):
        """TMEM -> registers (32-column blocks) -> token-major SMEM tile -> 16-byte stores.

        With ``split_k > 1`` rank ``r`` reduces every chunk ``c`` (4 tokens) with
        ``c % split_k == r``; peers push their FP32 partials with ``st.async`` into its
        dead weight ring once every CTA of the cluster has finished its mainloop.
        """
        S = self.split_k
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
        cute.arch.mbarrier_wait(storage.tmem_ready, 0)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
        tAcc = tCtAcc[((None, None), 0, 0)]
        cAcc = cute.make_identity_tensor((MMA_M, self.mma_n))
        blk = 32
        tiled_t2r = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(blk)), cutlass.Float32),
            cute.local_tile(tAcc, (MMA_M, blk), (0, 0)),
        )
        thr_t2r = tiled_t2r.get_slice(q)
        out_base = storage.sB.data_ptr().toint()
        out_gmem = mOut.iterator.toint() + Int64(n_tile * MMA_M * 2)
        row_stride = Int64(self.n * 2)
        # The output stores below give thread q the 8 weight rows starting at (q % 16) * 8
        # of the tile for every token; past N (partial last tile) it stores nothing.
        store_cols = Int32(16)
        if cutlass.const_expr(self.n % MMA_M != 0):
            if n_tile * MMA_M + (q & 15) * 8 >= self.n:
                store_cols = Int32(0)

        cute.arch.mbarrier_wait(storage.acc_full, 0)
        if cutlass.const_expr(S > 1):
            # Every CTA's rings are dead once its MMAs completed: the weight ring becomes
            # the mailbox. Tell the peers, then wait for theirs before pushing partials.
            cute.arch.cluster_wait()
            if q == 0:
                for p in cutlass.range_constexpr(1, S):
                    _mbarrier_arrive_remote(storage.peers_done.toint(), (rank + p) % S)
            cute.arch.mbarrier_wait(storage.peers_done, 0)

        mbox_base = storage.sA.data_ptr().toint()
        mbar_addr = storage.mbox_full.toint()
        slot_bytes = self.chunks_cap * MMA_M * 16
        # This rank's own partials of the chunks it reduces, indexed by c // S.
        rOwn = cute.make_rmem_tensor((self.chunks_cap * 4,), cutlass.Float32)
        nblk = self.mma_n // blk
        tTR_cAcc0 = thr_t2r.partition_D(cute.local_tile(cAcc, (MMA_M, blk), (0, 0)))
        # Double-buffered TMEM loads: block j + 1 is in flight while block j is consumed.
        rBuf = [cute.make_rmem_tensor(tTR_cAcc0.shape, cutlass.Float32) for _ in range(2)]
        cute.copy(
            tiled_t2r, thr_t2r.partition_S(cute.local_tile(tAcc, (MMA_M, blk), (0, 0))), rBuf[0]
        )
        for j in cutlass.range_constexpr(nblk):
            cute.arch.fence_view_async_tmem_load()
            rAcc = rBuf[j % 2]
            if cutlass.const_expr(j + 1 < nblk):
                cute.copy(
                    tiled_t2r,
                    thr_t2r.partition_S(cute.local_tile(tAcc, (MMA_M, blk), (0, j + 1))),
                    rBuf[(j + 1) % 2],
                )
            for c4 in cutlass.range_constexpr(blk // 4):
                c = j * (blk // 4) + c4
                if cutlass.const_expr(S == 1):
                    if 4 * c < m:
                        for v in cutlass.range_constexpr(4):
                            _sts_u16(
                                out_base + (4 * c + v) * (MMA_M * 2) + q * 2,
                                _f32_to_bf16_bits(rAcc[4 * c4 + v]),
                            )
                else:
                    if rank == c % S:
                        for v in cutlass.range_constexpr(4):
                            rOwn[(c // S) * 4 + v] = rAcc[4 * c4 + v]
                    else:
                        if 4 * c < m:
                            slot = rank
                            if rank > c % S:
                                slot = rank - 1
                            _st_async_v4_f32(
                                mbox_base + slot * slot_bytes + ((c // S) * MMA_M + q) * 16,
                                rAcc[4 * c4],
                                rAcc[4 * c4 + 1],
                                rAcc[4 * c4 + 2],
                                rAcc[4 * c4 + 3],
                                mbar_addr,
                                c % S,
                            )
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(storage.tmem_free)

        # Staged BF16 token rows (256 B) go out with 16-byte stores; 16 consecutive threads
        # cover one row, so every store is a full line.
        if cutlass.const_expr(S == 1):
            cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=MMA_M)
            for p in cutlass.range(q, m * store_cols, MMA_M, unroll=1):
                t = p >> 4
                col = (p & 15) * 16
                w0, w1, w2, w3 = _lds_v4_b32(out_base + t * (MMA_M * 2) + col)
                _stg_v4_b32(out_gmem + Int64(t) * row_stride + col, w0, w1, w2, w3)
        else:
            # Owned chunks c = rank + i * S; the column of element v of chunk c is 4c + v.
            active = (m + 3) >> 2
            my_chunks = Int32(0)
            if active > rank:
                my_chunks = (active - rank + S - 1) // S
            if my_chunks > 0:
                if q == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        storage.mbox_full, my_chunks * ((S - 1) * MMA_M * 16)
                    )
                cute.arch.mbarrier_wait(storage.mbox_full, 0)
                for i in cutlass.range_constexpr(self.chunks_cap):
                    if i < my_chunks:
                        acc = [rOwn[i * 4 + v] for v in range(4)]
                        for p in cutlass.range_constexpr(S - 1):
                            v0, v1, v2, v3 = _lds_v4_f32(
                                mbox_base + p * slot_bytes + (i * MMA_M + q) * 16
                            )
                            acc[0] = acc[0] + v0
                            acc[1] = acc[1] + v1
                            acc[2] = acc[2] + v2
                            acc[3] = acc[3] + v3
                        for v in cutlass.range_constexpr(4):
                            _sts_u16(
                                out_base + (4 * (rank + i * S) + v) * (MMA_M * 2) + q * 2,
                                _f32_to_bf16_bits(acc[v]),
                            )
                cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=MMA_M)
                # The r-th owned row is token 4 * (rank + (r // 4) * S) + r % 4.
                for p in cutlass.range(q, my_chunks * 4 * store_cols, MMA_M, unroll=1):
                    r = p >> 4
                    t = 4 * (rank + (r >> 2) * S) + (r & 3)
                    if t < m:
                        col = (p & 15) * 16
                        w0, w1, w2, w3 = _lds_v4_b32(out_base + t * (MMA_M * 2) + col)
                        _stg_v4_b32(out_gmem + Int64(t) * row_stride + col, w0, w1, w2, w3)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


@functools.cache
def _max_active_clusters(cluster_size: int) -> int:
    return utils.HardwareInfo().get_max_active_clusters(cluster_size)


def mma_n_for(m: int) -> int:
    """Token tile (UMMA N) of the low-M kernel for ``m`` tokens."""
    for mma_n in SUPPORTED_MMA_N:
        if m <= mma_n:
            return mma_n
    raise ValueError(f"M={m} exceeds {LOW_M_MAX}")


def mid_mma_n_for(m: int) -> int:
    """Token tile (UMMA N) of the mid-M kernel for ``m`` tokens."""
    for mma_n in SUPPORTED_MID_MMA_N:
        if m <= mma_n:
            return mma_n
    raise ValueError(f"M={m} exceeds {MAX_M}")


def _tile_n(n: int) -> int:
    """N rounded up to whole 128-row weight tiles: the key of the per-shape tables below.

    A partial last tile costs the same CTAs, cluster shape and per-CTA weight slice as a
    full one, so e.g. a 2112-row weight uses the 2176 entries.
    """
    return _align_up(n, MMA_M)


@dataclasses.dataclass(frozen=True)
class _ShapeTables:
    """Per-architecture tactics and selection limits, keyed by ``(N in whole tiles, K)``."""

    #: (N, K, mma_n) -> (split_k, x_stages, b_stages, quant_warps, quant_groups) of the
    #: low-M kernel; the weight ring takes the remaining shared memory.
    low_m_tuned: dict
    #: (N, K, mma_n) -> (split_k, b_stages) of the mid-M kernel; ``mid_m_default`` per (N, K)
    #: for buckets without an entry.
    mid_m_tuned: dict
    mid_m_default: dict
    #: (N, K) -> largest M for which :func:`can_run` selects this GEMM; ``default_max_m`` for
    #: other shapes.
    max_m: dict
    default_max_m: int
    #: (N, K) -> largest M of the low-M kernel (mid-M kernel above); ``LOW_M_MAX`` otherwise.
    low_m_cutover: dict


# SM100 / SM103, tuned on GB300 for the Kimi-K3 decode projections (CUDA graph, weights not
# L2-resident). The M limits are the largest M at which this GEMM (with its quantization) beat
# ``fp8_swap_ab_gemm`` (with its quantization), both standalone and behind a latency-bound
# kernel with PDL; every measured shape wins up to M = 128.
_SM100_TABLES = _ShapeTables(
    low_m_tuned={
        # KDA fused q/k/v/g (helix16 shard)
        (3072, 7168, 8): (5, 12, 4, 4, 4),
        (3072, 7168, 16): (4, 8, 4, 4, 2),
        (3072, 7168, 32): (5, 8, 4, 8, 4),
        (3072, 7168, 64): (5, 4, 4, 8, 4),
        # MoE latent down
        (3584, 7168, 8): (4, 14, 4, 4, 4),
        (3584, 7168, 16): (4, 8, 4, 4, 2),
        (3584, 7168, 32): (4, 8, 4, 8, 2),
        (3584, 7168, 64): (4, 4, 4, 8, 4),
        # MoE latent up
        (7168, 3584, 8): (2, 14, 4, 4, 4),
        (7168, 3584, 16): (2, 8, 4, 4, 2),
        (7168, 3584, 32): (2, 8, 4, 8, 2),
        (7168, 3584, 64): (2, 4, 4, 8, 4),
        # KDA / MLA o_proj (helix16 shard)
        (7168, 768, 8): (1, 6, 4, 4, 4),
        (7168, 768, 16): (1, 6, 4, 8, 4),
        (7168, 768, 32): (1, 6, 4, 8, 4),
        (7168, 768, 64): (1, 4, 4, 8, 2),
        # MLA fused q_a / kv_a (2112 rows)
        (2176, 7168, 8): (6, 10, 4, 4, 4),
        (2176, 7168, 16): (6, 8, 4, 8, 4),
        (2176, 7168, 32): (6, 8, 4, 8, 4),
        (2176, 7168, 64): (6, 4, 4, 8, 4),
        # MLA q_b
        (18432, 1536, 8): (1, 12, 4, 4, 4),
        (18432, 1536, 16): (1, 12, 4, 4, 2),
        (18432, 1536, 32): (1, 8, 4, 8, 2),
        (18432, 1536, 64): (1, 4, 4, 8, 2),
    },
    mid_m_tuned={
        (3072, 7168, 224): (4, 4),
        (3072, 7168, 256): (4, 3),
        (3584, 7168, 224): (3, 4),
        (3584, 7168, 256): (3, 3),
        (7168, 3584, 224): (2, 3),
        (7168, 3584, 256): (2, 3),
        (2176, 7168, 128): (6, 4),
        (2176, 7168, 192): (6, 4),
        (2176, 7168, 224): (4, 4),
        (2176, 7168, 256): (6, 3),
        (12288, 7168, 192): (1, 3),
        (12288, 7168, 224): (1, 3),
        (12288, 7168, 256): (1, 3),
        (7168, 6144, 256): (2, 3),
    },
    mid_m_default={
        (3072, 7168): (4, 4),
        (3584, 7168): (4, 4),
        (7168, 3584): (2, 4),
        (7168, 768): (1, 4),
        (2176, 7168): (5, 4),
        (18432, 1536): (1, 4),
        (12288, 7168): (1, 4),
        (7168, 6144): (2, 4),
    },
    max_m={
        (3072, 7168): 224,
        (3584, 7168): 224,
        (7168, 3584): 144,
        (7168, 768): 192,
        (2176, 7168): 224,
        (18432, 1536): 224,
        (12288, 7168): 128,
        (7168, 6144): 128,
    },
    default_max_m=128,
    # Large weights on fewer than all SMs, where the in-kernel quantization of a 64-token tile
    # is exposed: the mid-M kernel is faster above 32 tokens.
    low_m_cutover={
        (12288, 7168): 32,
        (7168, 6144): 32,
    },
)

# SM107, tuned on Rubin (212 SMs) for the Kimi-K3 decode projections with cold weights (rotated
# over 1.5 GiB): each tactic scores the mean CUDA-graph per-call time of back-to-back launches
# with and without PDL. Low-M kernel, per mma_n bucket timed at M = mma_n: a split_k sweep with
# the heuristic ring sizes, then quant warps / groups and x / b ring depths around the three
# best splits; mid-M kernel, per mma_n bucket: split_k x b_stages. With 212 SMs the best split
# is larger than on GB300 for the small-N weights. The M limits are the largest M up to which
# this GEMM ran at least 5% faster than ``fp8_quantize_1x128_cutedsl_ue8m0`` + the autotuned
# ``cute_dsl_mxfp8_gemm_rubin`` at every measured M (and, for the helix weights, not slower on
# the mean of the times with and without PDL); other shapes stop at 32 tokens.
_SM107_TABLES = _ShapeTables(
    low_m_tuned={
        # KDA fused q/k/v/g (dep16)
        (49152, 7168, 8): (1, 8, 4, 4, 4),
        (49152, 7168, 16): (1, 8, 4, 8, 4),
        (49152, 7168, 32): (1, 8, 4, 8, 4),
        (49152, 7168, 64): (1, 4, 4, 8, 4),
        # KDA / MLA o_proj (dep16)
        (7168, 12288, 8): (3, 8, 4, 4, 4),
        (7168, 12288, 16): (3, 8, 4, 8, 4),
        (7168, 12288, 32): (3, 8, 4, 8, 4),
        (7168, 12288, 64): (3, 4, 4, 8, 4),
        # MLA output gate / shared-expert gate_up (dep16)
        (12288, 7168, 8): (2, 8, 4, 4, 4),
        (12288, 7168, 16): (2, 8, 4, 8, 4),
        (12288, 7168, 32): (2, 8, 4, 8, 4),
        (12288, 7168, 64): (2, 4, 4, 8, 4),
        # shared-expert down (dep16)
        (7168, 6144, 8): (6, 8, 4, 4, 4),
        (7168, 6144, 16): (3, 8, 4, 8, 4),
        (7168, 6144, 32): (3, 8, 4, 8, 4),
        (7168, 6144, 64): (3, 4, 4, 8, 4),
        # MLA q_b
        (18432, 1536, 8): (2, 4, 4, 4, 4),
        (18432, 1536, 16): (2, 4, 4, 8, 4),
        (18432, 1536, 32): (1, 8, 4, 8, 4),
        (18432, 1536, 64): (1, 4, 4, 8, 4),
        # MoE latent down
        (3584, 7168, 8): (8, 7, 4, 4, 4),
        (3584, 7168, 16): (8, 4, 4, 8, 4),
        (3584, 7168, 32): (5, 8, 4, 8, 4),
        (3584, 7168, 64): (6, 4, 4, 8, 2),
        # MoE latent up
        (7168, 3584, 8): (7, 4, 4, 4, 4),
        (7168, 3584, 16): (6, 4, 4, 8, 4),
        (7168, 3584, 32): (6, 5, 4, 8, 4),
        (7168, 3584, 64): (3, 4, 4, 8, 4),
        # KDA fused q/k/v/g (helix16 shard)
        (3072, 7168, 8): (7, 8, 4, 4, 4),
        (3072, 7168, 16): (7, 4, 4, 8, 4),
        (3072, 7168, 32): (6, 8, 4, 8, 4),
        (3072, 7168, 64): (6, 4, 4, 8, 2),
        # MLA fused q_a / kv_a (2112 rows)
        (2176, 7168, 8): (10, 4, 4, 4, 4),
        (2176, 7168, 16): (7, 8, 4, 8, 4),
        (2176, 7168, 32): (8, 4, 4, 8, 4),
        (2176, 7168, 64): (8, 7, 4, 8, 4),
        # MLA output gate (helix16 shard)
        (768, 7168, 8): (14, 4, 4, 4, 4),
        (768, 7168, 16): (14, 4, 4, 8, 4),
        (768, 7168, 32): (10, 6, 4, 8, 4),
        (768, 7168, 64): (16, 4, 4, 8, 4),
        # KDA / MLA o_proj (helix16 shard)
        (7168, 768, 8): (3, 2, 2, 4, 4),
        (7168, 768, 16): (3, 2, 2, 8, 4),
        (7168, 768, 32): (1, 6, 4, 8, 4),
        (7168, 768, 64): (2, 3, 3, 8, 4),
        # KDA fused q/k/v/g (helix8 shard)
        (6144, 7168, 8): (3, 4, 4, 4, 4),
        (6144, 7168, 16): (3, 8, 4, 8, 4),
        (6144, 7168, 32): (3, 8, 4, 8, 4),
        (6144, 7168, 64): (3, 4, 4, 8, 4),
        # KDA / MLA o_proj (helix8 shard)
        (7168, 1536, 8): (4, 3, 3, 4, 4),
        (7168, 1536, 16): (4, 3, 3, 8, 4),
        (7168, 1536, 32): (2, 6, 4, 8, 4),
        (7168, 1536, 64): (2, 4, 4, 8, 4),
        # MLA output gate (helix8 shard)
        (1536, 7168, 8): (12, 5, 4, 4, 4),
        (1536, 7168, 16): (7, 8, 4, 8, 4),
        (1536, 7168, 32): (8, 7, 4, 8, 4),
        (1536, 7168, 64): (12, 5, 4, 8, 2),
    },
    mid_m_tuned={
        (49152, 7168, 64): (1, 4),
        (49152, 7168, 96): (1, 4),
        (49152, 7168, 128): (1, 4),
        (7168, 12288, 64): (3, 4),
        (7168, 12288, 96): (3, 4),
        (7168, 12288, 128): (3, 4),
        (12288, 7168, 64): (2, 4),
        (12288, 7168, 96): (2, 4),
        (12288, 7168, 128): (2, 4),
        (7168, 6144, 64): (2, 4),
        (7168, 6144, 96): (2, 4),
        (7168, 6144, 128): (2, 4),
        (18432, 1536, 64): (1, 4),
        (18432, 1536, 96): (1, 4),
        (18432, 1536, 128): (1, 4),
        (18432, 1536, 160): (1, 4),
        (3584, 7168, 64): (5, 4),
        (3584, 7168, 96): (5, 4),
        (3584, 7168, 128): (4, 4),
        (3584, 7168, 160): (4, 4),
        (7168, 3584, 64): (2, 4),
        (7168, 3584, 96): (2, 4),
        (7168, 3584, 128): (2, 4),
        (3072, 7168, 64): (6, 4),
        (3072, 7168, 96): (6, 4),
        (3072, 7168, 128): (5, 4),
        (3072, 7168, 160): (6, 4),
        (2176, 7168, 64): (7, 4),
        (2176, 7168, 96): (7, 4),
        (2176, 7168, 128): (7, 4),
        (2176, 7168, 160): (8, 4),
        (2176, 7168, 192): (8, 4),
        (768, 7168, 64): (12, 4),
        (768, 7168, 96): (8, 4),
        (768, 7168, 128): (8, 4),
        (768, 7168, 160): (10, 4),
        (768, 7168, 192): (8, 4),
        (768, 7168, 224): (7, 4),
        (768, 7168, 256): (8, 3),
        (7168, 768, 64): (3, 3),
        (7168, 768, 96): (1, 4),
        (7168, 768, 128): (1, 4),
        (6144, 7168, 64): (3, 4),
        (6144, 7168, 96): (3, 4),
        (6144, 7168, 128): (3, 4),
        (6144, 7168, 160): (3, 4),
        (7168, 1536, 64): (4, 3),
        (7168, 1536, 96): (1, 4),
        (7168, 1536, 128): (1, 4),
        (1536, 7168, 64): (8, 4),
        (1536, 7168, 96): (7, 4),
        (1536, 7168, 128): (8, 4),
        (1536, 7168, 160): (10, 3),
        (1536, 7168, 192): (10, 3),
        (1536, 7168, 224): (10, 3),
    },
    mid_m_default={
        (49152, 7168): (1, 4),
        (7168, 12288): (3, 4),
        (12288, 7168): (2, 4),
        (7168, 6144): (2, 4),
        (18432, 1536): (1, 4),
        (3584, 7168): (4, 4),
        (7168, 3584): (2, 4),
        (3072, 7168): (5, 4),
        (2176, 7168): (7, 4),
        (768, 7168): (8, 4),
        (7168, 768): (1, 4),
        (6144, 7168): (3, 4),
        (7168, 1536): (1, 4),
        (1536, 7168): (8, 4),
    },
    max_m={
        (49152, 7168): 32,
        (7168, 12288): 96,
        (12288, 7168): 96,
        (7168, 6144): 128,
        (18432, 1536): 160,
        (3584, 7168): 160,
        (7168, 3584): 96,
        (3072, 7168): 160,
        (2176, 7168): 192,
        (768, 7168): 256,
        (7168, 768): 128,
        (6144, 7168): 160,
        (7168, 1536): 96,
        (1536, 7168): 224,
    },
    default_max_m=0,
    low_m_cutover={
        (49152, 7168): 32,
        (7168, 12288): 32,
        (12288, 7168): 32,
        (7168, 6144): 32,
        (18432, 1536): 32,
        (3584, 7168): LOW_M_MAX,
        (7168, 3584): 32,
        (3072, 7168): 32,
        (2176, 7168): 32,
        (6144, 7168): 32,
    },
)

#: Minimum weight-ring depth the heuristic keeps when growing the activation ring.
_MIN_A_STAGES = 6
#: Rule for the cluster split-K of shapes without measured tactics: the largest split whose
#: grid stays within this fraction of the SMs, with at least this many K tiles per CTA (low-M
#: kernel / mid-M kernel; the mid-M kernel's wider token tiles want longer K slices).
_LOW_M_GRID_FRACTION, _LOW_M_MIN_K_TILES = 0.8, 6
_MID_M_GRID_FRACTION, _MID_M_MIN_K_TILES = 0.75, 8


@functools.cache
def _device_capability(device_index: int) -> tuple:
    return torch.cuda.get_device_capability(device_index)


def _is_supported_device(device_index: int) -> bool:
    """SM100 / SM103; SM107 only with a Rubin-capable CuTe DSL."""
    from ...cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    capability = _device_capability(device_index)
    return capability in ((10, 0), (10, 3)) or (
        capability == (10, 7) and IS_CUTLASS_DSL_RUBIN_AVAILABLE
    )


def _tables() -> _ShapeTables:
    """Tables of the current device: SM107 or SM100 / SM103."""
    if _device_capability(torch.cuda.current_device()) == (10, 7):
        return _SM107_TABLES
    return _SM100_TABLES


#: The low-M kernel serves up to ``LOW_M_MAX`` tokens when each CTA streams at most this many K
#: tiles: up to there its in-kernel quantization of a 64-token tile costs at most ~10% against
#: the mid-M path, less than its gated / sigmoid / prequantized variants save by folding in a
#: separate kernel. With longer K slices the mid-M path takes over above 32 tokens.
_LOW_M_FULL_RANGE_K_TILES = 12
#: Shapes without a measured limit run the mid-M path only while its CTAs stream at most this
#: many K tiles: its K loop is unrolled, and with longer K slices it is not faster than the
#: default GEMM and takes seconds to compile.
_MID_M_MAX_K_TILES = 96


def max_m_for(n: int, k: int) -> int:
    """Largest M for which this GEMM is selected for an ``[N, K]`` weight on the current device.

    0 where it never is: unsupported architecture, a weight shape the kernels do not take, or a
    shape without a measured limit whose weight tiles do not fit in one wave. Such shapes whose
    mid-M K slices are longer than ``_MID_M_MAX_K_TILES`` are selected for the low-M kernel only.
    """
    if not _is_supported_device(torch.cuda.current_device()) or n % N_ALIGN or k % CTA_K:
        return 0
    tables = _tables()
    max_m = tables.max_m.get((_tile_n(n), k))
    if max_m is None:
        max_m = 0
        if _ceil_div(n, MMA_M) <= _num_sms():
            max_m = tables.default_max_m
            if _ceil_div(k // CTA_K, _rule_split(n, k, mid=True)) > _MID_M_MAX_K_TILES:
                max_m = min(max_m, low_m_max_for(n, k))
    return min(MAX_M, max_m)


def low_m_max_for(n: int, k: int) -> int:
    """Largest M served by the low-M kernel for an ``[N, K]`` weight (mid-M kernel above)."""
    cutover = _tables().low_m_cutover.get((_tile_n(n), k))
    if cutover is not None:
        return cutover
    k_tiles = _ceil_div(k // CTA_K, _rule_split(n, k, mid=False))
    return LOW_M_MAX if k_tiles <= _LOW_M_FULL_RANGE_K_TILES else 32


def kernel_max_m_for(n: int, k: int) -> int:
    """Largest M for which the selected GEMM runs the low-M kernel for an ``[N, K]`` weight on the
    current device: where its gated / sigmoid / prequantized variants match it bitwise."""
    return min(max_m_for(n, k), low_m_max_for(n, k))


@functools.cache
def _rule_split(n: int, k: int, mid: bool) -> int:
    """Cluster split-K of the low-M (``mid``: mid-M) kernel for an ``[N, K]`` weight without
    measured tactics: the largest split whose clusters are all resident, whose grid stays
    within the kernel's fraction of the SMs and whose CTAs keep its minimum K slice."""
    kt_total = k // CTA_K
    n_tiles = _ceil_div(n, MMA_M)
    fraction, min_k_tiles = (
        (_MID_M_GRID_FRACTION, _MID_M_MIN_K_TILES)
        if mid
        else (_LOW_M_GRID_FRACTION, _LOW_M_MIN_K_TILES)
    )
    split = 1
    for s in range(2, min(16, kt_total) + 1):
        if n_tiles * s > fraction * _num_sms() or kt_total / s < min_k_tiles:
            continue
        if _max_active_clusters(s) < n_tiles:
            continue
        split = s
    return split


@functools.cache
def default_tactic(m: int, n: int, k: int) -> LowMTactic:
    """Deterministic low-M tactic for ``(M, N, K)``: measured table first, the rule otherwise."""
    mma_n = mma_n_for(m)
    tuned = _tables().low_m_tuned.get((_tile_n(n), k, mma_n))
    if tuned is not None:
        split, x_stages, b_stages, quant_warps, quant_groups = tuned
        base = LowMTactic(
            mma_n=mma_n,
            split_k=split,
            a_stages=1,
            x_stages=x_stages,
            b_stages=b_stages,
            quant_warps=quant_warps,
            quant_groups=quant_groups,
        )
        return dataclasses.replace(base, a_stages=max_a_stages(n, k, base))
    split = _rule_split(n, k, mid=False)
    kt_max = _ceil_div(k // CTA_K, split)
    quant_warps = 4 if mma_n <= 8 else 8
    tactic = None
    # The activation ring holds every K tile at 8 tokens, 8 tiles up to 32 tokens and 4 at 64
    # tokens (quantized by two groups of warps), if the weight ring stays deep enough;
    # shallower otherwise.
    first = kt_max if mma_n <= 8 else (8 if mma_n <= 32 else 4)
    for x_stages in [x for x in (kt_max, 8, 4) if x <= first]:
        base = LowMTactic(
            mma_n=mma_n,
            split_k=split,
            a_stages=1,
            x_stages=min(x_stages, kt_max),
            b_stages=min(4, kt_max),
            quant_warps=quant_warps,
            quant_groups=2 if mma_n == 64 else 4,
        )
        try:
            validate_tactic(n, k, base)
        except ValueError:
            continue
        tactic = dataclasses.replace(base, a_stages=max_a_stages(n, k, base))
        if tactic.a_stages >= min(_MIN_A_STAGES, kt_max):
            break
    if tactic is None:
        raise ValueError(f"no valid tactic for M={m}, N={n}, K={k}")
    return tactic


@functools.cache
def default_mid_tactic(m: int, n: int, k: int) -> MidMTactic:
    """Deterministic mid-M tactic: tuned table first, heuristic otherwise."""
    mma_n = mid_mma_n_for(m)
    tables = _tables()
    tuned = tables.mid_m_tuned.get((_tile_n(n), k, mma_n)) or tables.mid_m_default.get(
        (_tile_n(n), k)
    )
    split, b_stages = tuned if tuned is not None else (_rule_split(n, k, mid=True), 4)
    base = MidMTactic(mma_n=mma_n, split_k=split, a_stages=1, b_stages=b_stages)
    tactic = dataclasses.replace(base, a_stages=max_mid_a_stages(n, k, base))
    # A per-shape default may not fit shared memory with the wider token tiles: drop
    # activation stages, then split-K ranks.
    while mid_smem_bytes(n, k, tactic) > _MID_SMEM_CAPACITY and (
        tactic.b_stages > 2 or tactic.split_k > 1
    ):
        if tactic.b_stages > 2:
            base = dataclasses.replace(base, b_stages=base.b_stages - 1)
        else:
            base = dataclasses.replace(base, split_k=base.split_k - 1)
        tactic = dataclasses.replace(base, a_stages=max_mid_a_stages(n, k, base))
    return tactic


def _dummy_ptr(dtype):
    return make_ptr(dtype, 1 << 20, cute.AddressSpace.gmem, assumed_align=16)


def _gmem_ptr(dtype, tensor: torch.Tensor):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _current_stream(device: torch.device) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


@functools.cache
def _compiled_kernel(n: int, k: int, tactic: LowMTactic, use_pdl: bool):
    kernel = Fp8BlockScaleLowMGemmKernel(n, k, tactic, use_pdl)
    return cute.compile(
        kernel,
        _dummy_ptr(cutlass.Float8E4M3FN),
        _dummy_ptr(cutlass.BFloat16),
        _dummy_ptr(cutlass.Uint8),
        _dummy_ptr(cutlass.BFloat16),
        Int32(1),
        _current_stream(torch.device("cuda")),
    )


@functools.cache
def _compiled_quant(k: int, sf_blocks: int, use_pdl: bool):
    kernel = MxFp8QuantKernel(k, sf_blocks, use_pdl)
    return cute.compile(
        kernel,
        _dummy_ptr(cutlass.BFloat16),
        _dummy_ptr(cutlass.Float8E4M3FN),
        _dummy_ptr(cutlass.Uint8),
        Int32(1),
        _current_stream(torch.device("cuda")),
    )


@functools.cache
def _compiled_mid_kernel(n: int, k: int, tactic: MidMTactic, use_pdl: bool):
    kernel = Fp8BlockScaleMidMGemmKernel(n, k, tactic, use_pdl)
    return cute.compile(
        kernel,
        _dummy_ptr(cutlass.Float8E4M3FN),
        _dummy_ptr(cutlass.Float8E4M3FN),
        _dummy_ptr(cutlass.Uint8),
        _dummy_ptr(cutlass.Uint8),
        _dummy_ptr(cutlass.BFloat16),
        Int32(1),
        _current_stream(torch.device("cuda")),
    )


def _run_mid(x, w, w_scale, out, tactic: MidMTactic, use_pdl: bool) -> None:
    m, k = x.shape
    n = w.shape[0]
    sf_blocks = _sf_blocks(tactic.mma_n)
    xq = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty((k // CTA_K) * sf_blocks * 512, dtype=torch.uint8, device=x.device)
    stream = _current_stream(x.device)
    _compiled_quant(k, sf_blocks, use_pdl)(
        _gmem_ptr(cutlass.BFloat16, x),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        Int32(m),
        stream,
    )
    _compiled_mid_kernel(n, k, tactic, use_pdl)(
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        _gmem_ptr(cutlass.Uint8, w_scale),
        _gmem_ptr(cutlass.BFloat16, out),
        Int32(m),
        stream,
    )


def run_low_m_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    tactic: LowMTactic | MidMTactic | None = None,
    use_pdl: bool = True,
) -> torch.Tensor:
    """Launch ``out = x @ dequant(w, w_scale)^T`` on the current stream.

    ``M <= low_m_max_for(N, K)`` runs the low-M kernel (activation quantized inside the
    GEMM); larger M runs a small MXFP8 quantization kernel followed by the mid-M GEMM, chained
    with PDL so the GEMM's weight stream overlaps the quantization.

    Args:
        x: ``[M, K]`` contiguous BF16 activation, ``1 <= M <= MAX_M``.
        w: ``[N, K]`` contiguous FP8 E4M3 weight with UE8M0 block scales.
        w_scale: DeepGEMM-packed UE8M0 scale, int32 ``[N, ceil(K / 512)]`` with strides
            ``(1, N)``.
        out: ``[M, N]`` contiguous BF16 output.
        tactic: kernel specialization; the per-architecture default if None.
        use_pdl: launch with programmatic dependent launch.
    """
    m, k = x.shape
    n = w.shape[0]
    if not 1 <= m <= MAX_M:
        raise ValueError(f"M={m} must be in [1, {MAX_M}]")
    if tactic is None:
        if m <= low_m_max_for(n, k):
            tactic = default_tactic(m, n, k)
        else:
            tactic = default_mid_tactic(m, n, k)
    if m > tactic.mma_n:
        raise ValueError(f"M={m} exceeds tactic.mma_n={tactic.mma_n}")
    if isinstance(tactic, MidMTactic):
        _run_mid(x, w, w_scale, out, tactic, use_pdl)
        return out
    _compiled_kernel(n, k, tactic, use_pdl)(
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        _gmem_ptr(cutlass.BFloat16, x),
        _gmem_ptr(cutlass.Uint8, w_scale),
        _gmem_ptr(cutlass.BFloat16, out),
        Int32(m),
        _current_stream(x.device),
    )
    return out


def pack_block_scale_from_mxfp8_layout(sf: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """DeepGEMM-packed UE8M0 scale of an ``[N, K]`` weight from its CuTe MXFP8 GEMM scale.

    ``sf`` is the R128c4 K32 UE8M0 tensor ``transform_k128_scales_to_cutedsl_mxfp8_layout``
    builds from 128x128 block scales: each block exponent replicated over its 128 rows and 4
    K32 slots, so the first byte of every 512-byte atom is the exponent of block
    ``(r // 128, c // 4)``. The result equals ``transform_sf_into_required_layout`` of the
    original FP32 block scales: one UE8M0 byte per (row, K block), four K blocks per int32
    word, MN-major ``int32 [N, ceil(K / 512)]`` with strides ``(1, N)``.
    """
    if k % CTA_K:
        raise ValueError(f"K={k} must be a multiple of {CTA_K}")
    nb, kb = _ceil_div(n, MMA_M), k // CTA_K
    flat = sf.reshape(-1)
    if sf.dtype != torch.uint8 or flat.numel() != nb * kb * 512:
        raise ValueError(f"sf must be the uint8 R128c4 scale of an [{n}, {k}] weight")
    exponents = flat.view(nb, kb, 512)[:, :, 0]
    rows = exponents.repeat_interleave(MMA_M, dim=0)[:n]
    padded = torch.zeros((n, _ceil_div(kb, 4) * 4), dtype=torch.uint8, device=sf.device)
    padded[:, :kb] = rows
    words = padded.view(torch.int32)
    packed = torch.empty((words.shape[1], n), dtype=torch.int32, device=sf.device).t()
    packed.copy_(words)
    return packed


def check_inputs(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor) -> None:
    """Raise ``ValueError`` unless :func:`run_low_m_gemm` accepts these operands."""
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2D bfloat16 tensor")
    if w.dtype != torch.float8_e4m3fn or w.dim() != 2 or not w.is_contiguous():
        raise ValueError("w must be a contiguous 2D float8_e4m3fn tensor")
    n, k = w.shape
    if x.shape[1] != k:
        raise ValueError(f"x has K={x.shape[1]}, weight has K={k}")
    if n % N_ALIGN or k % CTA_K:
        raise ValueError(f"N={n} must be a multiple of {N_ALIGN} and K={k} of {CTA_K}")
    words = _ceil_div(k // CTA_K, 4)
    if (
        w_scale.dtype != torch.int32
        or tuple(w_scale.shape) != (n, words)
        or w_scale.stride(0) != 1
        or (words > 1 and w_scale.stride(1) != n)
    ):
        raise ValueError(
            f"w_scale must be DeepGEMM-packed UE8M0 int32 [{n}, {words}] with strides (1, {n})"
        )
    if x.data_ptr() % 16 or w.data_ptr() % 16:
        raise ValueError("x and w must be 16-byte aligned")


__all__ = [
    "LOW_M_MAX",
    "MAX_M",
    "N_ALIGN",
    "LowMTactic",
    "MidMTactic",
    "check_inputs",
    "default_mid_tactic",
    "default_tactic",
    "kernel_max_m_for",
    "low_m_max_for",
    "max_m_for",
    "pack_block_scale_from_mxfp8_layout",
    "run_low_m_gemm",
]
