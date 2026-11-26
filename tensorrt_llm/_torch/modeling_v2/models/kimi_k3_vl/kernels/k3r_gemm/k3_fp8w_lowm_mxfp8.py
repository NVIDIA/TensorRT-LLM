# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M FP8-weight GEMM fed by an MXFP8 activation made upstream (SM100/SM103).

``out[M, N] = x[M, K] @ (W[N, K] * S)^T`` like :mod:`k3_fp8w_lowm_gemm`'s low-M kernel, but the
producer of ``x`` hands over its MXFP8 form instead of BF16:

* ``xq``: E4M3 ``[M, K]`` row-major, and
* ``xs``: UE8M0 ``[M, K / 32]`` row-major, one exponent per 1x32 group,

computed from the BF16 ``x`` exactly like the low-M kernel's in-kernel quantization
(:func:`mxfp8_quantize_reference` documents the arithmetic; producers call the same PTX helpers
``_abs_max_bf16x32`` / ``_ue8m0_of_amax`` / ``_bf16x2_inv_scale`` / ``_quant_bf16x8_to_e4m3``).
With the low-M kernel's tiling (same ``mma_n`` and split-K), K-tile order, scale folding
(``SFB = e_act + e_w - 127``, ``SFA = 1``) and split-K reduction order, the output is bitwise the
low-M kernel's on the BF16 ``x``.

Why: at decode batch sizes the low-M kernel's tail after its ``griddepcontrol.wait`` is
dominated by the activation path (BF16 tiles arrive by TMA one per ~0.1-0.2 us per quantization
group and take ~0.25 us each to quantize), not by the weights, which a programmatic head start
already streams into the ring. Here the E4M3 tiles go by TMA (128B swizzle) straight into the
UMMA B ring (every K tile of the CTA in flight at once) and one warp folds the weight block
exponents into the activation exponents.

Warp roles: warp 0 streams the weight ring (and releases the dependent grid at once, before its
own wait: it only reads weights), warp 1 issues the MMAs, warp 2 loads the E4M3 tiles after
``griddepcontrol.wait``, warp 3 loads and folds the scales after it, warps 4-7 run the epilogue
(TMEM -> registers in 32-column blocks -> split-K partials pushed with ``st.async`` into the
owner's dead weight ring once every CTA of the cluster is done with its MMAs -> token-major
staging -> 16-byte BF16 stores).

Tensor-memory weight staging (``K3LowMMxTactic.tmem_tiles`` = ``t > 0``, :func:`staged_mx_tactic`):
the MMA warp copies the first ``t`` weight K tiles of its slice from the ring into tensor memory
(``tcgen05.cp`` 128x256b, four per tile) as they land and releases their ring stages at once, so
the weight warp refills them with the next tiles before ``griddepcontrol.wait``. With ``t`` =
K tiles minus ring stages the CTA's whole weight slice is on chip when the activation lands. The
MMAs of those tiles read A from tensor memory (``kind::mxf8f6f4.block_scale`` with A in TMEM),
in the same K-tile and K-block order as the ring MMAs, so the output is bitwise the unstaged
kernel's. The CTA then allocates 256 or 512 tensor-memory columns instead of 128.
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
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from ..k3r_comm import k3_lamport_ar
from . import k3_fp8w_lowm_gemm as lowm
from .k3_fp8w_lowm_gemm import (
    _EVICT_FIRST,
    _SF_TMEM_COLS,
    _TMEM_COLS,
    CTA_K,
    MMA_M,
    N_ALIGN,
    SF_VEC,
    SUPPORTED_MMA_N,
    _abs_max_bf16x32,
    _add_bf16x2,
    _asm,
    _bf16x2_inv_scale,
    _ceil_div,
    _f32_to_bf16_bits,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _iv,
    _ldg_v4_b32,
    _lds_v4_b32,
    _lds_v4_f32,
    _mbarrier_arrive_remote,
    _quant_bf16x8_to_e4m3,
    _s2t_partition_sf,
    _st_async_v4_f32,
    _stg_v4_b32,
    _sts_u16,
    _sts_v4_b32,
    _ue8m0_of_amax,
)

#: Largest M of this kernel (the low-M kernel's range).
MAX_M = lowm.LOW_M_MAX

_TMA_WARP = 0
_MMA_WARP = 1
_X_WARP = 2
_SF_WARP = 3
_EPI_WARP0 = 4
_THREADS = 256
_EPI_BARRIER_ID = 3

__all__ = [
    "MAX_M",
    "K3LowMMxTactic",
    "can_run_mxfp8",
    "default_mx_tactic",
    "can_stage_weights",
    "ring_holds_weights",
    "weights_on_chip",
    "staged_matches_mid_m",
    "staged_mx_tactic",
    "mxfp8_quantize_reference",
    "run_k3_fp8w_lowm_gemm_mxfp8",
    "run_k3_fp8w_lowm_gemm_mxfp8_ag_push",
    "run_k3_fp8w_lowm_gemm_mxfp8_ar_push",
    "run_k3_fp8w_lowm_gemm_mxfp8_slice_ar_push",
]


#: First tensor-memory column of the staged weight tiles (the accumulator and the scale-factor
#: slots use the columns below it).
_A_TMEM_COL0 = _TMEM_COLS
#: Tensor-memory columns of one staged 128 x 128 E4M3 weight tile (four codes per column).
_A_TMEM_TILE_COLS = CTA_K // 4
#: Largest tensor-memory allocation of one CTA.
_TMEM_MAX_COLS = 512
#: Most weight K tiles that fit in tensor memory next to the accumulator and scale factors.
MAX_TMEM_TILES = (_TMEM_MAX_COLS - _A_TMEM_COL0) // _A_TMEM_TILE_COLS
#: Upper 32 bits of the shared-memory matrix descriptor of a 128B-swizzled K-major weight stage
#: (stride between 8-row groups 1024 B, descriptor version 1, 128-byte swizzle), the source of
#: ``tcgen05.cp``; the lower word is the 16-byte start address with a leading offset of one.
_A_DESC_HI = 0x40004040
_A_DESC_LO_LBO = 0x10000


@dataclasses.dataclass(frozen=True)
class K3LowMMxTactic:
    """``mma_n`` padded tokens, cluster ``split_k``, weight ring ``a_stages`` and activation
    ring ``b_stages`` (``b_stages`` = the K tiles of a CTA keeps every activation load in
    flight); ``tmem_tiles``: the first weight K tiles of the slice staged in tensor memory
    (0: none)."""

    mma_n: int
    split_k: int
    a_stages: int
    b_stages: int
    tmem_tiles: int = 0


def _mailbox_bytes(t: K3LowMMxTactic) -> int:
    if t.split_k == 1:
        return 0
    return (t.split_k - 1) * _ceil_div(t.mma_n // 4, t.split_k) * MMA_M * 16


def smem_bytes(n: int, k: int, t: K3LowMMxTactic) -> int:
    """Conservative host-side mirror of the kernel's shared-memory footprint."""
    kt_max = _ceil_div(k // CTA_K, t.split_k)
    a_st = min(t.a_stages, kt_max)
    b_st = min(t.b_stages, kt_max)
    total = 8 * (2 * (a_st + b_st) + 5) + 4
    total = lowm._align_up(total, 1024) + max(a_st * MMA_M * CTA_K, _mailbox_bytes(t))
    total = lowm._align_up(total, 1024) + max(b_st * t.mma_n * CTA_K, t.mma_n * MMA_M * 2)
    total = lowm._align_up(total, 128) + 512  # SFA (one stage, constant 1.0)
    total = lowm._align_up(total, 128) + b_st * 512
    return total + 1024


def validate_tactic(n: int, k: int, t: K3LowMMxTactic) -> None:
    if t.mma_n not in SUPPORTED_MMA_N:
        raise ValueError(f"mma_n must be one of {SUPPORTED_MMA_N}, got {t.mma_n}")
    if n % N_ALIGN or k % CTA_K:
        raise ValueError(f"N={n} must be a multiple of {N_ALIGN} and K={k} of {CTA_K}")
    if not 1 <= t.split_k <= min(16, k // CTA_K):
        raise ValueError(f"split_k={t.split_k} invalid for K={k}")
    if t.a_stages < 1 or t.b_stages < 1:
        raise ValueError("a_stages and b_stages must be >= 1")
    # Every rank of the cluster owns at least k // CTA_K // split_k K tiles; the staged tiles
    # must be K tiles of every rank.
    if not 0 <= t.tmem_tiles <= min(MAX_TMEM_TILES, (k // CTA_K) // t.split_k):
        raise ValueError(f"tmem_tiles={t.tmem_tiles} invalid for K={k}, split_k={t.split_k}")
    need = smem_bytes(n, k, t)
    if need > lowm._SMEM_CAPACITY:
        raise ValueError(f"tactic {t} needs {need} B of SMEM > {lowm._SMEM_CAPACITY} B")


def tmem_alloc_cols(t: K3LowMMxTactic) -> int:
    """Tensor-memory columns a CTA of tactic ``t`` allocates (a power of two)."""
    need = _A_TMEM_COL0 + t.tmem_tiles * _A_TMEM_TILE_COLS
    cols = _TMEM_COLS
    while cols < need:
        cols *= 2
    return cols


#: Weight-ring depth :func:`default_mx_tactic` keeps before it shortens the activation ring.
_MIN_A_STAGES = 6


@functools.cache
def default_mx_tactic(m: int, n: int, k: int) -> K3LowMMxTactic:
    """The low-M kernel's ``mma_n`` and split-K for ``(M, N, K)`` (same bits) and the deepest
    weight ring that fits next to the activation ring. The activation ring holds every K tile
    of the CTA unless that leaves fewer than ``_MIN_A_STAGES`` weight stages (long K slices at
    ``mma_n >= 32``); then it is the deepest ring that does."""
    base = lowm.default_tactic(m, n, k)
    kt_max = _ceil_div(k // CTA_K, base.split_k)
    best = None
    for b_stages in range(kt_max, 0, -1):
        t = K3LowMMxTactic(mma_n=base.mma_n, split_k=base.split_k, a_stages=1, b_stages=b_stages)
        fits = [
            a
            for a in range(1, kt_max + 1)
            if smem_bytes(n, k, dataclasses.replace(t, a_stages=a)) <= lowm._SMEM_CAPACITY
        ]
        if fits:
            t = dataclasses.replace(t, a_stages=max(fits))
            best = best or t
            if t.a_stages >= min(_MIN_A_STAGES, kt_max):
                return t
    if best is None:
        raise ValueError(f"no MXFP8 tactic fits for M={m}, N={n}, K={k}")
    return best


#: Shared memory of a staged CTA (:func:`staged_mx_tactic`). With the predecessor's CTAs at the
#: largest carve-out (``k3_attn_res`` ``entry_trigger``), a CTA of up to 225 KB lands on an SM
#: that runs a 1 KB CTA (measured on Rubin; 226 KB does not); 220 KB leaves room for an attn_res
#: CTA's own few KB.
STAGED_SMEM_BYTES = 220 * 1024


@functools.cache
def staged_mx_tactic(m: int, n: int, k: int) -> K3LowMMxTactic:
    """:func:`default_mx_tactic`'s ``mma_n``, split-K and activation ring (so the same bits)
    with the weight ring cut to :data:`STAGED_SMEM_BYTES` of shared memory and as many weight K
    tiles staged in tensor memory as the ring cannot hold: the CTA's whole weight slice is on
    chip before ``griddepcontrol.wait`` where tensor memory allows."""
    base = default_mx_tactic(m, n, k)
    kt_max = _ceil_div(k // CTA_K, base.split_k)
    budget = min(STAGED_SMEM_BYTES, lowm._SMEM_CAPACITY)
    a_stages = base.a_stages
    while a_stages > 1 and smem_bytes(n, k, dataclasses.replace(base, a_stages=a_stages)) > budget:
        a_stages -= 1
    tiles = min(max(kt_max - a_stages, 0), MAX_TMEM_TILES, (k // CTA_K) // base.split_k)
    tactic = dataclasses.replace(base, a_stages=a_stages, tmem_tiles=tiles)
    validate_tactic(n, k, tactic)
    return tactic


@functools.cache
def can_stage_weights(m: int, n: int, k: int) -> bool:
    """Whether :func:`staged_mx_tactic` has a tactic with staged tiles for ``(M, N, K)``."""
    try:
        return staged_mx_tactic(m, n, k).tmem_tiles > 0
    except ValueError:
        return False


@functools.cache
def ring_holds_weights(m: int, n: int, k: int) -> bool:
    """Whether :func:`staged_mx_tactic` keeps the CTA's whole weight K slice in its shared-memory
    ring (at most :data:`STAGED_SMEM_BYTES`), with no tiles in tensor memory (e.g. the KDA QKVG
    projection at TP16: 8 K tiles per CTA of the 7-way split). The staged tactic only sizes the
    ring: the same bits."""
    try:
        tactic = staged_mx_tactic(m, n, k)
    except ValueError:
        return False
    return tactic.tmem_tiles == 0 and tactic.a_stages >= _ceil_div(k // CTA_K, tactic.split_k)


def weights_on_chip(m: int, n: int, k: int) -> bool:
    """Whether :func:`staged_mx_tactic` has the CTA's weight slice on chip before
    ``griddepcontrol.wait`` (:func:`can_stage_weights` or :func:`ring_holds_weights`), so
    that the GEMM may run as the entry-released dependent of a ``k3_attn_res``
    (``run_k3_fp8w_lowm_gemm_mxfp8(..., stage_weights=True)``)."""
    return can_stage_weights(m, n, k) or ring_holds_weights(m, n, k)


@functools.cache
def staged_matches_mid_m(m: int, n: int, k: int) -> bool:
    """Whether the staged kernel on ``M`` rows above the low-M cut-over (``low_m_max_for(N, K)
    < M <= MAX_M``, whole 128-row weight tiles) computes the mid-M GEMM's bits: both run the
    same split-K over the same K-tile order, the mid-M kernel on the same MXFP8 quantization of
    the activation (checked bitwise in the tests for the KDA QKVG weight)."""
    if not lowm.low_m_max_for(n, k) < m <= MAX_M or n % MMA_M:
        return False
    try:
        mid = lowm.default_mid_tactic(m, n, k)
    except ValueError:
        return False
    return can_stage_weights(m, n, k) and mid.split_k == staged_mx_tactic(m, n, k).split_k


@dsl_user_op
def _ldg_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fold_scales_x4(act4, w_exp, *, loc=None, ip=None):
    """Four bytes ``clamp(e_act_i + w_exp - 127, 0, 254)`` of a packed word of four UE8M0
    activation exponents (``_sts_scale`` of the low-M kernel, four at a time)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(act4, loc=loc, ip=ip), _iv(w_exp, loc=loc, ip=ip)],
            "{\n"
            ".reg .s32 a, v, r;\n"
            "mov.b32 r, 0;\n"
            + "".join(
                f"bfe.u32 a, $1, {8 * i}, 8;\n"
                "add.s32 v, a, $2;\n"
                "sub.s32 v, v, 127;\n"
                "max.s32 v, v, 0;\n"
                "min.s32 v, v, 254;\n"
                f"bfi.b32 r, v, r, {8 * i}, 8;\n"
                for i in range(4)
            )
            + "mov.b32 $0, r;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_b32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tmem_cp_128x256b(taddr, smem_addr, *, loc=None, ip=None):
    """``tcgen05.cp`` of 128 rows x 32 bytes (one K block of E4M3 codes) of the 128B-swizzled
    K-major tile at shared address ``smem_addr`` into eight tensor-memory columns at
    ``taddr`` (row r -> lane r)."""
    _asm(
        None,
        [_iv(taddr, loc=loc, ip=ip), _iv(smem_addr, loc=loc, ip=ip)],
        "{\n"
        ".reg .b32 lo, hi;\n"
        ".reg .b64 desc;\n"
        "shr.u32 lo, $1, 4;\n"
        "and.b32 lo, lo, 0x3FFF;\n"
        f"or.b32 lo, lo, {_A_DESC_LO_LBO};\n"
        f"mov.b32 hi, {_A_DESC_HI};\n"
        "mov.b64 desc, {lo, hi};\n"
        "tcgen05.cp.cta_group::1.128x256b [$0], desc;\n"
        "}",
        "r,r",
        loc=loc,
        ip=ip,
    )


class K3Fp8wLowMMxGemmKernel:
    """Swap-AB MXFP8 GEMM on an upstream-quantized activation (``M <= 64``), cluster split-K.

    ``ar_push`` (``AR_PUSH_TWO_SHOT`` / ``AR_PUSH_ONE_SHOT`` of :mod:`k3_fp8w_lowm_gemm`,
    ``ar_world`` ranks): instead of the BF16 store, every 16-byte output vector goes into the
    Lamport slots of the next all-reduce on a ``k3_lamport_ar`` workspace (``push_slot`` /
    ``push_v4``, as the low-M kernel's push epilogue). The dependent grid is then released from
    the activation warp after its ``griddepcontrol.wait`` (the low-M push variant's order), since
    the consumer reads the workspace's rotation state as soon as it is resident.
    """

    def __init__(
        self,
        n: int,
        k: int,
        tactic: K3LowMMxTactic,
        use_pdl: bool,
        ar_push: int = 0,
        ar_world: int = 0,
        strides: tuple[int, int, int] | None = None,
        push_addend: bool = False,
        ag_hidden: int = 0,
    ):
        validate_tactic(n, k, tactic)
        if ar_push and (
            ar_push not in (lowm.AR_PUSH_TWO_SHOT, lowm.AR_PUSH_ONE_SHOT) or ar_world < 2
        ):
            raise ValueError(f"invalid all-reduce push {ar_push} over {ar_world} ranks")
        # ag_hidden > 0: the output is these ranks' columns of the call's all-gather rows of width
        # ag_hidden (k3_lamport_ar "ag": after the call's M x ar_world slots, one-shot transport),
        # starting at the column the launch passes in place of the rank.
        if ag_hidden and (ar_push != lowm.AR_PUSH_ONE_SHOT or push_addend or ag_hidden < n):
            raise ValueError("the all-gather push is a one-shot push without addend")
        self.ag_hidden = ag_hidden
        if push_addend and not ar_push:
            raise ValueError("push_addend goes with the all-reduce push")
        self.ar_push = ar_push
        self.ar_world = ar_world
        # Row strides (elements) of the weight, the E4M3 activation and its UE8M0 scales: a
        # column slice of wider operands (the row-parallel MoE tail reads its latent_up columns
        # and its slice of the latent all-reduce's MXFP8 output in place).
        self.w_ld, self.x_ld, self.xs_ld = strides if strides is not None else (k, k, k // SF_VEC)
        if (
            self.w_ld < k
            or self.x_ld < k
            or self.xs_ld < k // SF_VEC
            or self.w_ld % 16
            or self.x_ld % 16
        ):
            raise ValueError(f"invalid row strides {strides} for K={k}")
        # Every pushed BF16 value y becomes bf16(y + a), a from a BF16 [M, N] addend read after
        # griddepcontrol.wait (the PyTorch BF16 add; k3_fp8w_lowm_gemm's ``push_addend``).
        self.push_addend = push_addend
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
        self.chunks = self.mma_n // 4
        self.chunks_cap = _ceil_div(self.chunks, self.split_k)
        self.a_region_bytes = max(self.a_stages * self.a_stage_bytes, _mailbox_bytes(tactic))
        self.out_stage_bytes = self.mma_n * MMA_M * 2
        self.groups = k // SF_VEC
        self.blk = min(32, self.mma_n)
        self.tmem_tiles = tactic.tmem_tiles
        self.tmem_cols = tmem_alloc_cols(tactic)

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
        self._launch(
            w_ptr,
            xq_ptr,
            xs_ptr,
            ws_ptr,
            out_ptr,
            m,
            stream,
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(0),
        )

    @cute.jit
    def call_ar_push(
        self,
        w_ptr: cute.Pointer,
        xq_ptr: cute.Pointer,
        xs_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        m: Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        """All-reduce push launch (``ar_push``): the output goes into the Lamport slots of the
        next call on the workspace (``k3_lamport_ar.LamportWorkspace`` addresses: peer pointer
        table, multicast base, state flags; this is rank ``rank``). Nothing is stored through
        ``out_ptr``."""
        self._launch(
            w_ptr, xq_ptr, xs_ptr, ws_ptr, out_ptr, m, stream, peer_ptrs, mc_base, flags, rank
        )

    @cute.jit
    def _launch(
        self,
        w_ptr: cute.Pointer,
        xq_ptr: cute.Pointer,
        xs_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
        ar_peers: Int64,
        ar_mc: Int64,
        ar_flags: Int64,
        ar_rank: Int32,
    ):
        n, k = self.n, self.k
        w_ld, x_ld = self.w_ld, self.x_ld
        mW = cute.make_tensor(w_ptr, cute.make_layout((n, k, 1), stride=(w_ld, 1, n * w_ld)))
        mXq = cute.make_tensor(xq_ptr, cute.make_layout((m, k, 1), stride=(x_ld, 1, m * x_ld)))
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
        # The same MMA with A in tensor memory for the staged weight tiles (unused without them).
        tiled_mma_ts = tiled_mma
        a_tmem_layout = a_layout
        if cutlass.const_expr(self.tmem_tiles > 0):
            tiled_mma_ts = sm100_utils.make_blockscaled_trivial_tiled_mma(
                cutlass.Float8E4M3FN,
                cutlass.Float8E4M3FN,
                OperandMajorMode.K,
                OperandMajorMode.K,
                cutlass.Float8E8M0FNU,
                SF_VEC,
                tcgen05.CtaGroup.ONE,
                (MMA_M, self.mma_n),
                tcgen05.OperandSource.TMEM,
            )
            a_tmem_layout = sm100_utils.make_smem_layout_a(
                tiled_mma_ts, self.mma_tiler, cutlass.Float8E4M3FN, 1
            )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, self.b_stages
        )
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, SF_VEC, 1)
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
            # Weight ring; the split-K mailbox once every CTA of the cluster is done with it.
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, self.a_region_bytes], 1024
            ]
            # Activation ring; stages the BF16 output tile token-major after the mainloop.
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
            tiled_mma_ts,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            xs_ptr,
            mWS,
            mOut,
            m,
            a_layout,
            a_tmem_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            ar_peers,
            ar_mc,
            ar_flags,
            ar_rank,
        ).launch(
            grid=(self.n_tiles * self.split_k, 1, 1),
            block=(_THREADS, 1, 1),
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
        tiled_mma_ts: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        xs_ptr: cute.Pointer,
        mWS: cute.Tensor,
        mOut: cute.Tensor,
        m: Int32,
        a_layout: cute.ComposedLayout,
        a_tmem_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        ar_peers: Int64,
        ar_mc: Int64,
        ar_flags: Int64,
        ar_rank: Int32,
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
                    # the E4M3 tile (TMA, expect_tx) and its folded scales (warp 3)
                    cute.arch.mbarrier_init(b_full + i, 2)
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
            # Weight producer: never waits on the previous grid (weights are static), so the
            # dependent grid may launch at once (with the push: after the activation warp's wait).
            if cutlass.const_expr(self.use_pdl and not self.ar_push):
                _griddepcontrol_launch_dependents()
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
            # Activation producer: the E4M3 tiles of every K tile (128B swizzle = UMMA B layout).
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_wait()
                if cutlass.const_expr(self.ar_push != 0):
                    _griddepcontrol_launch_dependents()
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
            for kt in cutlass.range(kt_count, unroll=1):
                stage = kt % self.b_stages
                if kt >= self.b_stages:
                    cute.arch.mbarrier_wait(b_empty + stage, ((kt // self.b_stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(b_full + stage, self.b_stage_bytes)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, kt_start + kt, 0)],
                    tBsB[(None, stage)],
                    tma_bar_ptr=b_full + stage,
                )
        elif warp_idx == _SF_WARP:
            self.scale_warp(storage, xs_ptr, mWS, m, n_tile, kt_start, kt_count)
        elif warp_idx == _MMA_WARP:
            self.mma_warp(
                tiled_mma, tiled_mma_ts, storage, kt_count, a_layout, a_tmem_layout, b_layout,
                sfa_layout, sfb_layout,
            )  # fmt: skip
        elif warp_idx >= _EPI_WARP0:
            self.epilogue(
                tiled_mma, storage, mOut, m, tidx - 32 * _EPI_WARP0, n_tile, rank,
                ar_peers, ar_mc, ar_flags, ar_rank,
            )  # fmt: skip

    @cute.jit
    def scale_warp(
        self,
        storage,
        xs_ptr: cute.Pointer,
        mWS: cute.Tensor,
        m: Int32,
        n_tile: Int32,
        kt_start: Int32,
        kt_count: Int32,
    ):
        """SFB of every K tile: lane r (and r + 32) writes token row r's four folded exponents
        ``clamp(e_act + e_w - 127, 0, 254)`` at ``(r % 32) * 16 + (r // 32) * 4``; rows past M
        (zero codes) get ``e_act = 127``."""
        lane = cute.arch.lane_idx()
        b_full = storage.b_full.data_ptr()
        b_empty = storage.b_empty.data_ptr()
        sf_base = storage.sSFB.data_ptr().toint()
        # Weight block exponents of this CTA's K tiles (static: before the wait).
        w_exps = []
        for j in cutlass.range_constexpr(_ceil_div(self.kt_max, 32)):
            e = Int32(127)
            if lane + 32 * j < kt_count:
                e = Int32(mWS[n_tile, kt_start + lane + 32 * j])
            w_exps.append(e)
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        rows_per_lane = _ceil_div(self.mma_n, 32)
        # Issue every activation exponent load first (one word = the 4 groups of a K tile).
        acts = []
        for i in cutlass.range_constexpr(rows_per_lane):
            r = lane + 32 * i
            row_words = []
            for kt in cutlass.range_constexpr(self.kt_max):
                v = Int32(0x7F7F7F7F)
                if (r < m) & (kt < kt_count) & (r < self.mma_n):
                    v = _ldg_b32(xs_ptr.toint() + Int64(r * self.xs_ld + (kt_start + kt) * 4))
                row_words.append(v)
            acts.append(row_words)
        if cutlass.const_expr(self.b_stages >= self.kt_max):
            # Every K tile has its own stage: write all scales, fence once, release all tiles.
            for kt in cutlass.range_constexpr(self.kt_max):
                if kt < kt_count:
                    e_w = cute.arch.shuffle_sync(w_exps[kt // 32], kt % 32)
                    for i in cutlass.range_constexpr(rows_per_lane):
                        r = lane + 32 * i
                        if r < self.mma_n:
                            _sts_b32(
                                sf_base + kt * 512 + (r & 31) * 16 + (r >> 5) * 4,
                                _fold_scales_x4(acts[i][kt], e_w),
                            )
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            for j in cutlass.range_constexpr(_ceil_div(self.kt_max, 32)):
                if lane + 32 * j < kt_count:
                    cute.arch.mbarrier_arrive(b_full + lane + 32 * j)
        else:
            for kt in cutlass.range_constexpr(self.kt_max):
                if kt < kt_count:
                    stage = kt % self.b_stages
                    if cutlass.const_expr(kt >= self.b_stages):
                        cute.arch.mbarrier_wait(b_empty + stage, ((kt // self.b_stages) & 1) ^ 1)
                    e_w = cute.arch.shuffle_sync(w_exps[kt // 32], kt % 32)
                    for i in cutlass.range_constexpr(rows_per_lane):
                        r = lane + 32 * i
                        if r < self.mma_n:
                            _sts_b32(
                                sf_base + stage * 512 + (r & 31) * 16 + (r >> 5) * 4,
                                _fold_scales_x4(acts[i][kt], e_w),
                            )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(b_full + stage)

    @cute.jit
    def mma_warp(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_ts: cute.TiledMma,
        storage,
        kt_count: Int32,
        a_layout: cute.ComposedLayout,
        a_tmem_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
    ):
        a_full = storage.a_full.data_ptr()
        a_empty = storage.a_empty.data_ptr()
        b_full = storage.b_full.data_ptr()
        b_empty = storage.b_empty.data_ptr()
        cute.arch.alloc_tmem(self.tmem_cols, storage.tmem_holding_buf)
        cute.arch.relinquish_tmem_alloc_permit()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(storage.tmem_ready)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
        )
        staged = self.tmem_tiles
        if cutlass.const_expr(staged > 0):
            # Weight K tiles 0 .. staged - 1 into tensor memory as they land (tile j at columns
            # _A_TMEM_COL0 + 32 j, one tcgen05.cp per 32-byte K block); their ring stages are
            # released at once, so the weight warp refills them before griddepcontrol.wait.
            # tcgen05.commit tracks the copies like MMAs, and the MMAs below that read the tiles
            # are issued after them by the same thread.
            tmem_base = storage.tmem_holding_buf.load()
            sa_base = storage.sA.data_ptr().toint()
            for j in cutlass.range_constexpr(staged):
                stage = j % self.a_stages
                cute.arch.mbarrier_wait(a_full + stage, (j // self.a_stages) & 1)
                with cute.arch.elect_one():
                    for kb in cutlass.range_constexpr(CTA_K // SF_VEC):
                        _tmem_cp_128x256b(
                            tmem_base + _A_TMEM_COL0 + j * _A_TMEM_TILE_COLS + kb * 8,
                            sa_base + stage * self.a_stage_bytes + kb * SF_VEC,
                        )
                    if cutlass.const_expr(j + self.a_stages < self.kt_max):
                        tcgen05.commit(a_empty + stage)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
        tCtSFA = cute.make_tensor(
            cute.recast_ptr(acc_tmem_ptr + self.mma_n, dtype=cutlass.Float8E8M0FNU),
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfa_layout, (None, None, None, 0))
            ),
        )
        s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = _s2t_partition_sf(
            storage.sSFA.get_tensor(sfa_layout), tCtSFA
        )
        # SFB: TMEM columns [mma_n + 16, +16).
        sSFB = storage.sSFB.get_tensor(sfb_layout)
        tmem_sfb = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfb_layout, (None, None, None, 0))
        )
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(
                acc_tmem_ptr + self.mma_n + _SF_TMEM_COLS,
                dtype=cutlass.Float8E8M0FNU,
            ),
            tmem_sfb,
        )
        s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = _s2t_partition_sf(sSFB, tCtSFB)
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
        # A operands of the staged tiles: tile j's fragment at its tensor-memory columns.
        tCtA_tiles = []
        if cutlass.const_expr(staged > 0):
            tCtA_frag = tiled_mma_ts.make_fragment_A(
                cute.make_tensor(acc_tmem_ptr, a_tmem_layout.outer)
            )[None, None, None, 0]
            for j in cutlass.range_constexpr(staged):
                tCtA_tiles.append(
                    cute.make_tensor(
                        cute.recast_ptr(
                            acc_tmem_ptr + _A_TMEM_COL0 + j * _A_TMEM_TILE_COLS,
                            dtype=tCtA_frag.element_type,
                        ),
                        tCtA_frag.layout,
                    )
                )

        tiled_mma.set(tcgen05.Field.ACCUMULATE, staged > 0)
        tiled_mma_ts.set(tcgen05.Field.ACCUMULATE, False)
        # Fully unrolled over the largest K-tile count; the MMA descriptor fields are set outside
        # the per-tile predicate (a rank with one K tile fewer skips the last tile's work). The
        # staged tiles are K tiles of every rank (``validate_tactic``).
        for kt in cutlass.range_constexpr(self.kt_max):
            a_stage = kt % self.a_stages
            b_stage = kt % self.b_stages
            live = kt < kt_count
            if live:
                if cutlass.const_expr(kt >= staged):
                    cute.arch.mbarrier_wait(a_full + a_stage, (kt // self.a_stages) & 1)
                cute.arch.mbarrier_wait(b_full + b_stage, (kt // self.b_stages) & 1)
                cute.copy(s2t_sfb, tCsSFB_s2t[(None, None, None, None, b_stage)], tCtSFB_s2t)
            for kp in cutlass.range_constexpr(num_kphases):
                if cutlass.const_expr(kt < staged):
                    tiled_mma_ts.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                    tiled_mma_ts.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                    if live:
                        cute.gemm(
                            tiled_mma_ts,
                            tCtAcc,
                            tCtA_tiles[kt][(None, None, kp)],
                            tCrB[(None, None, kp, b_stage)],
                            tCtAcc,
                        )
                    tiled_mma_ts.set(tcgen05.Field.ACCUMULATE, True)
                else:
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                    if live:
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[(None, None, kp, a_stage)],
                            tCrB[(None, None, kp, b_stage)],
                            tCtAcc,
                        )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            if live:
                with cute.arch.elect_one():
                    if cutlass.const_expr(kt >= staged and kt + self.a_stages < self.kt_max):
                        tcgen05.commit(a_empty + a_stage)
                    if cutlass.const_expr(kt + self.b_stages < self.kt_max):
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
        ar_peers: Int64,
        ar_mc: Int64,
        ar_flags: Int64,
        ar_rank: Int32,
    ):
        """TMEM -> registers -> token-major SMEM tile -> 16-byte stores (``ar_push``: into the
        all-reduce's Lamport slots); with ``split_k > 1`` rank ``r`` reduces every 4-token chunk
        ``c`` with ``c % split_k == r`` (own partial, then the peers' in rank order: the low-M
        kernel's order)."""
        S = self.split_k
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        # The previous call on the workspace has completed: its rotation state is final.
        ar_cur, ar_nbytes = Int32(0), Int32(0)
        if cutlass.const_expr(self.ar_push != 0):
            ar_cur, _, ar_nbytes, _, _, _, _, _ = k3_lamport_ar.lamport_state(ar_flags)
        cute.arch.mbarrier_wait(storage.tmem_ready, 0)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
        tAcc = tCtAcc[((None, None), 0, 0)]
        cAcc = cute.make_identity_tensor((MMA_M, self.mma_n))
        blk = self.blk
        tiled_t2r = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(blk)), cutlass.Float32),
            cute.local_tile(tAcc, (MMA_M, blk), (0, 0)),
        )
        thr_t2r = tiled_t2r.get_slice(q)
        out_base = storage.sB.data_ptr().toint()
        out_gmem = mOut.iterator.toint() + Int64(n_tile * MMA_M * 2)
        row_stride = Int64(self.n * 2)
        store_cols = Int32(16)
        if cutlass.const_expr(self.n % MMA_M != 0):
            if n_tile * MMA_M + (q & 15) * 8 >= self.n:
                store_cols = Int32(0)

        cute.arch.mbarrier_wait(storage.acc_full, 0)
        if cutlass.const_expr(S > 1):
            # Every CTA's weight ring is dead once its MMAs completed: it becomes the mailbox.
            cute.arch.cluster_wait()
            if q == 0:
                for p in cutlass.range_constexpr(1, S):
                    _mbarrier_arrive_remote(storage.peers_done.toint(), (rank + p) % S)
            cute.arch.mbarrier_wait(storage.peers_done, 0)

        mbox_base = storage.sA.data_ptr().toint()
        mbar_addr = storage.mbox_full.toint()
        slot_bytes = self.chunks_cap * MMA_M * 16
        rOwn = cute.make_rmem_tensor((self.chunks_cap * 4,), cutlass.Float32)
        nblk = self.mma_n // blk
        tTR_cAcc0 = thr_t2r.partition_D(cute.local_tile(cAcc, (MMA_M, blk), (0, 0)))
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

        if cutlass.const_expr(S == 1):
            cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=MMA_M)
            for p in cutlass.range(q, m * store_cols, MMA_M, unroll=1):
                t = p >> 4
                col = (p & 15) * 16
                w0, w1, w2, w3 = _lds_v4_b32(out_base + t * (MMA_M * 2) + col)
                if cutlass.const_expr(self.ar_push != 0):
                    self._push(
                        (w0, w1, w2, w3), t, n_tile * MMA_M + (p & 15) * 8,
                        ar_peers, ar_mc, ar_cur, ar_nbytes, ar_rank, mOut.iterator.toint(), m,
                    )  # fmt: skip
                else:
                    _stg_v4_b32(out_gmem + Int64(t) * row_stride + col, w0, w1, w2, w3)
        else:
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
                for p in cutlass.range(q, my_chunks * 4 * store_cols, MMA_M, unroll=1):
                    r = p >> 4
                    t = 4 * (rank + (r >> 2) * S) + (r & 3)
                    if t < m:
                        col = (p & 15) * 16
                        w0, w1, w2, w3 = _lds_v4_b32(out_base + t * (MMA_M * 2) + col)
                        if cutlass.const_expr(self.ar_push != 0):
                            self._push(
                                (w0, w1, w2, w3), t, n_tile * MMA_M + (p & 15) * 8,
                                ar_peers, ar_mc, ar_cur, ar_nbytes, ar_rank, mOut.iterator.toint(),
                                m,
                            )  # fmt: skip
                        else:
                            _stg_v4_b32(out_gmem + Int64(t) * row_stride + col, w0, w1, w2, w3)

    @cute.jit
    def _push(
        self,
        w,
        t: Int32,
        elem: Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        cur: Int32,
        nbytes: Int32,
        ar_rank: Int32,
        add_base: Int64,
        m: Int32,
    ):
        """One 16-byte output vector (features ``[elem, elem + 8)`` of token ``t``) into its
        Lamport slot of the next all-reduce (``-0`` sent as ``+0``); with ``push_addend`` after
        the BF16 add of the addend at ``add_base`` (BF16 ``[M, N]``, read after the wait). With
        ``ag_hidden`` into columns ``ar_rank + elem`` (the launch's column offset) of the call's
        all-gather row ``t`` instead."""
        if cutlass.const_expr(self.ag_hidden > 0):
            dst = (
                k3_lamport_ar.ag_row_addr(mc_base, cur, nbytes, m, t, self.ar_world, self.ag_hidden)
                + Int64(ar_rank + elem) * 2
            )
            k3_lamport_ar.push_v4(dst, w[0], w[1], w[2], w[3], True)
        else:
            if cutlass.const_expr(self.push_addend):
                a = _ldg_v4_b32(add_base + (Int64(t) * self.n + Int64(elem)) * 2)
                w = tuple(_add_bf16x2(w[i], a[i]) for i in range(4))
            one_shot = self.ar_push == lowm.AR_PUSH_ONE_SHOT
            dst = k3_lamport_ar.push_slot(
                peer_ptrs, mc_base, cur, nbytes, t, ar_rank, elem, self.ar_world, self.n, one_shot
            )
            k3_lamport_ar.push_v4(dst, w[0], w[1], w[2], w[3], one_shot)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _compiled_kernel(n: int, k: int, tactic: K3LowMMxTactic, use_pdl: bool):
    kernel = K3Fp8wLowMMxGemmKernel(n, k, tactic, use_pdl)
    dummy = 1 << 20
    ptrs = [
        make_ptr(dtype, dummy, cute.AddressSpace.gmem, assumed_align=16)
        for dtype in (
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            cutlass.Uint8,
            cutlass.Uint8,
            cutlass.BFloat16,
        )
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(kernel, *ptrs, Int32(1), stream)


@functools.cache
def _compiled_ar_push_kernel(
    n: int,
    k: int,
    tactic: K3LowMMxTactic,
    use_pdl: bool,
    ar_push: int,
    ar_world: int,
    strides: tuple[int, int, int] | None = None,
    push_addend: bool = False,
    ag_hidden: int = 0,
):
    kernel = K3Fp8wLowMMxGemmKernel(
        n,
        k,
        tactic,
        use_pdl,
        ar_push=ar_push,
        ar_world=ar_world,
        strides=strides,
        push_addend=push_addend,
        ag_hidden=ag_hidden,
    )
    dummy = 1 << 20
    # A column slice of the scales starts at a 4-byte boundary (K / 32 of whole 128-blocks).
    xs_align = 4 if strides is not None else 16
    ptrs = [
        make_ptr(dtype, dummy, cute.AddressSpace.gmem, assumed_align=align)
        for dtype, align in (
            (cutlass.Float8E4M3FN, 16),
            (cutlass.Float8E4M3FN, 16),
            (cutlass.Uint8, xs_align),
            (cutlass.Uint8, 16),
            (cutlass.BFloat16, 16),
        )
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    addr = Int64(dummy)
    return cute.compile(kernel.call_ar_push, *ptrs, Int32(1), addr, addr, addr, Int32(0), stream)


def _gmem_ptr(dtype, tensor: torch.Tensor):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def can_run_mxfp8(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    allow_partial_n_tile: bool = False,
    above_cutover: bool = False,
) -> bool:
    """Whether :func:`run_k3_fp8w_lowm_gemm_mxfp8` accepts these operands: the low-M kernel's
    weight contract (UE8M0-resmoothed E4M3 ``w [N, K]``, DeepGEMM-packed ``w_scale``, N a
    multiple of 128 unless ``allow_partial_n_tile``), ``1 <= M <= low_m_max_for(N, K)`` on
    SM100/SM103 (``above_cutover``: up to :data:`MAX_M`), and ``xq`` E4M3 ``[M, K]`` / ``xs``
    uint8 ``[M, K / 32]``, contiguous, 16- and 4-byte aligned."""
    if xq.dim() != 2 or xs.dim() != 2 or w.dim() != 2 or not xq.is_cuda:
        return False
    if not lowm._is_supported_device(xq.device.index):
        return False
    m, k = xq.shape
    n = w.shape[0]
    if w.shape[1] != k or n % N_ALIGN or k % CTA_K or (n % MMA_M and not allow_partial_n_tile):
        return False
    if not 1 <= m <= (MAX_M if above_cutover else min(MAX_M, lowm.low_m_max_for(n, k))):
        return False
    if xq.dtype != torch.float8_e4m3fn or xs.dtype != torch.uint8:
        return False
    if tuple(xs.shape) != (m, k // SF_VEC) or not xq.is_contiguous() or not xs.is_contiguous():
        return False
    if w.dtype != torch.float8_e4m3fn or not w.is_contiguous():
        return False
    try:
        validate_tactic(n, k, default_mx_tactic(m, n, k))
    except ValueError:
        return False
    words = _ceil_div(k // CTA_K, 4)
    if (
        w_scale.dtype != torch.int32
        or tuple(w_scale.shape) != (n, words)
        or w_scale.stride(0) != 1
        or (words > 1 and w_scale.stride(1) != n)
    ):
        return False
    return xq.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0 and xs.data_ptr() % 4 == 0


def run_k3_fp8w_lowm_gemm_mxfp8(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    tactic: K3LowMMxTactic | None = None,
    use_pdl: bool = True,
    stage_weights: bool = False,
) -> torch.Tensor:
    """``out = dequant(xq, xs) @ dequant(w, w_scale)^T`` into ``out`` (bf16 ``[M, N]``) on the
    current stream, bitwise the low-M kernel on the BF16 activation that ``(xq, xs)`` quantizes
    (inputs are not re-validated; see :func:`can_run_mxfp8`). ``stage_weights`` (no ``tactic``):
    :func:`staged_mx_tactic`, the same bits."""
    m, k = xq.shape
    n = w.shape[0]
    if tactic is None:
        tactic = staged_mx_tactic(m, n, k) if stage_weights else default_mx_tactic(m, n, k)
    if m > tactic.mma_n:
        raise ValueError(f"M={m} exceeds tactic.mma_n={tactic.mma_n}")
    compiled = _compiled_kernel(n, k, tactic, use_pdl)
    compiled(
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        _gmem_ptr(cutlass.Uint8, w_scale),
        _gmem_ptr(cutlass.BFloat16, out),
        Int32(m),
        cuda.CUstream(torch.cuda.current_stream(xq.device).cuda_stream),
    )
    return out


def run_k3_fp8w_lowm_gemm_mxfp8_ar_push(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    ws: "k3_lamport_ar.LamportWorkspace",
    one_shot: bool,
    *,
    tactic: K3LowMMxTactic | None = None,
    use_pdl: bool = True,
) -> None:
    """:func:`run_k3_fp8w_lowm_gemm_mxfp8` with the all-reduce push epilogue: stores the BF16
    output into the Lamport slots of the next all-reduce on ``ws``;
    ``k3_lamport_ar.run_lamport_finish`` (or a consumer's fused finish) with the same
    ``one_shot`` completes it. Same tactic and rounding as the plain MXFP8 GEMM, so the pair is
    bitwise that GEMM (and the gated low-M GEMM whose quantization ``(xq, xs)`` reproduces)
    followed by the MNNVL all-reduce of its output."""
    m, k = xq.shape
    n = w.shape[0]
    if tactic is None:
        tactic = default_mx_tactic(m, n, k)
    if m > tactic.mma_n:
        raise ValueError(f"M={m} exceeds tactic.mma_n={tactic.mma_n}")
    need = k3_lamport_ar.required_bytes(m, n, ws.world, one_shot)
    if need > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B < {need} B")
    mode = lowm.AR_PUSH_ONE_SHOT if one_shot else lowm.AR_PUSH_TWO_SHOT
    compiled = _compiled_ar_push_kernel(n, k, tactic, use_pdl, mode, ws.world)
    compiled(
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        _gmem_ptr(cutlass.Uint8, w_scale),
        make_ptr(cutlass.BFloat16, xq.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int32(m),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(ws.rank),
        cuda.CUstream(torch.cuda.current_stream(xq.device).cuda_stream),
    )


def run_k3_fp8w_lowm_gemm_mxfp8_slice_ar_push(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    k0: int,
    k1: int,
    ws: "k3_lamport_ar.LamportWorkspace",
    one_shot: bool,
    *,
    addend: torch.Tensor | None = None,
    tactic: K3LowMMxTactic | None = None,
    use_pdl: bool = True,
) -> None:
    """:func:`run_k3_fp8w_lowm_gemm_mxfp8_ar_push` on the K columns ``[k0, k0 + k1)`` of wider
    operands, read in place: E4M3 ``xq [M, K]`` and UE8M0 ``xs [M, K / 32]`` (e.g. the MXFP8
    output of ``k3_latent_ar_norm``) and the E4M3 weight ``w [N, K]``; ``w_scale`` is the
    DeepGEMM-packed scale of the ``[N, k1]`` slice (``k3_fp8w_lowm_gemm.pack_ue8m0_exponents``).
    With ``addend`` (contiguous BF16 ``[M, N]``, read after the kernel's wait) every pushed value
    is ``bf16(y + addend)``. Same tactic, K-tile order and rounding as
    ``k3_fp8w_lowm_gemm.run_k3_fp8w_lowm_gemm_split_ar_push`` on the BF16 rows that ``(xq, xs)``
    quantize, so the pushed bits are that GEMM's (the row-parallel MoE tail)."""
    m = xq.shape[0]
    n = w.shape[0]
    if k0 % CTA_K or k1 % CTA_K or k0 + k1 > xq.shape[1] or xq.shape[1] != w.shape[1]:
        raise ValueError(f"invalid K slice [{k0}, {k0 + k1}) of K={xq.shape[1]}")
    if xs.shape != (m, xq.shape[1] // SF_VEC) or xs.stride(1) != 1 or xq.stride(1) != 1:
        raise ValueError("xs must be the [M, K / 32] scales of xq, rows contiguous")
    if addend is not None and (
        addend.dtype != torch.bfloat16
        or tuple(addend.shape) != (m, n)
        or not addend.is_contiguous()
        or addend.data_ptr() % 16
    ):
        raise ValueError("addend must be a contiguous, 16-byte aligned bf16 [M, N] tensor")
    if tactic is None:
        tactic = default_mx_tactic(m, n, k1)
    if m > tactic.mma_n:
        raise ValueError(f"M={m} exceeds tactic.mma_n={tactic.mma_n}")
    need = k3_lamport_ar.required_bytes(m, n, ws.world, one_shot)
    if need > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B < {need} B")
    mode = lowm.AR_PUSH_ONE_SHOT if one_shot else lowm.AR_PUSH_TWO_SHOT
    strides = (w.stride(0), xq.stride(0), xs.stride(0))
    compiled = _compiled_ar_push_kernel(
        n, k1, tactic, use_pdl, mode, ws.world, strides, addend is not None
    )

    def ptr(dtype, t, off):
        return make_ptr(dtype, t.data_ptr() + off, cute.AddressSpace.gmem, assumed_align=16)

    args = [
        ptr(cutlass.Float8E4M3FN, w, k0),
        ptr(cutlass.Float8E4M3FN, xq, k0),
        make_ptr(
            cutlass.Uint8, xs.data_ptr() + k0 // SF_VEC, cute.AddressSpace.gmem, assumed_align=4
        ),
        _gmem_ptr(cutlass.Uint8, w_scale),
        _gmem_ptr(cutlass.BFloat16, addend if addend is not None else xq),
        Int32(m),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(ws.rank),
    ]
    compiled(*args, cuda.CUstream(torch.cuda.current_stream(xq.device).cuda_stream))


def run_k3_fp8w_lowm_gemm_mxfp8_ag_push(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    ws: "k3_lamport_ar.LamportWorkspace",
    col0: int,
    hidden: int,
    *,
    tactic: K3LowMMxTactic | None = None,
    use_pdl: bool = True,
) -> None:
    """:func:`run_k3_fp8w_lowm_gemm_mxfp8` whose BF16 output (``[M, N]``, ``N`` = rows of ``w``)
    goes into columns ``[col0, col0 + N)`` of the all-gather rows (width ``hidden``) of the next
    one-shot call on ``ws`` (``k3_lamport_ar`` "ag": every rank pushes its columns, the call's
    finish adds them to the reduced slots). ``tactic``: e.g. the tactic of the full ``[hidden, K]``
    projection whose output columns ``w`` holds, so the pushed values are bitwise that
    projection's (every 128-row output tile is computed alone, with the same split-K)."""
    m, k = xq.shape
    n = w.shape[0]
    if tactic is None:
        tactic = default_mx_tactic(m, n, k)
    if m > tactic.mma_n:
        raise ValueError(f"M={m} exceeds tactic.mma_n={tactic.mma_n}")
    if col0 % 8 or col0 + n > hidden:
        raise ValueError(f"invalid column range [{col0}, {col0 + n}) of {hidden}")
    need = k3_lamport_ar.required_bytes(m, hidden, ws.world, True, ag=True)
    if need > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B < {need} B")
    compiled = _compiled_ar_push_kernel(
        n, k, tactic, use_pdl, lowm.AR_PUSH_ONE_SHOT, ws.world, ag_hidden=hidden
    )
    args = [
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        _gmem_ptr(cutlass.Uint8, w_scale),
        make_ptr(cutlass.BFloat16, xq.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int32(m),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(col0),
    ]
    compiled(*args, cuda.CUstream(torch.cuda.current_stream(xq.device).cuda_stream))


def mxfp8_quantize_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(xq, xs)`` of a BF16 ``[M, K]`` activation as the low-M kernel quantizes it (torch
    reference for tests): per 1x32 group, ``e = biased exponent of 2^ceil(log2(amax / 448))``
    with ``amax`` floored at ~1e-4 (BF16 ``0x38D2``), codes ``satfinite_rn_e4m3(x * 2^(127 - e))``
    (the power-of-two multiply is exact in BF16)."""
    m, k = x.shape
    g = x.view(m, k // SF_VEC, SF_VEC)
    bits = g.abs().view(torch.int16).to(torch.int32) & 0x7FFF
    amax = bits.amax(dim=-1).clamp_min(lowm._AMAX_FLOOR_BF16)
    e = (amax >> 7) - 8 + ((amax & 0x7F) > 0x60).to(torch.int32)
    inv = torch.pow(2.0, (127 - e).float()).to(torch.bfloat16)
    scaled = (g * inv.unsqueeze(-1)).float()
    xq = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(m, k)
    return xq, e.to(torch.uint8)


@dsl_user_op
def _amax_bf16x8_words(w0, w1, w2, w3, *, loc=None, ip=None):
    """|x| max of eight BF16 values (four packed words) as a packed BF16 pair (garbage sign)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            "{\n.reg .b32 m0, m1;\n"
            "max.xorsign.abs.bf16x2 m0, $1, $2;\n"
            "max.xorsign.abs.bf16x2 m1, $3, $4;\n"
            "max.xorsign.abs.bf16x2 $0, m0, m1;\n}",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _amax_pair(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "max.xorsign.abs.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


# Producer-side helpers (bitwise the low-M kernel's in-kernel quantization).
amax_bf16x8_words = _amax_bf16x8_words
amax_pair = _amax_pair
ue8m0_of_amax = _ue8m0_of_amax
bf16x2_inv_scale = _bf16x2_inv_scale
quant_bf16x8_to_e4m3 = _quant_bf16x8_to_e4m3
abs_max_bf16x32 = _abs_max_bf16x32
