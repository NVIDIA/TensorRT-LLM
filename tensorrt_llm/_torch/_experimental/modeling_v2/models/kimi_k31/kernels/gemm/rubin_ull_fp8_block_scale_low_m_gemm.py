# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 block-scale weight GEMM for few-row, weight-bandwidth-bound projections (SM107).

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
(:func:`max_m_for`, the measured cross-over against SM107's default FP8 block-scale GEMM, the
CuTe MXFP8 GEMM) come from the SM107 tables keyed by ``(N, K)``; shapes without an entry use a
heuristic split-K and are not selected. The default GEMM reads the weight scale in the CuTe
MXFP8 layout, so :func:`pack_block_scale_from_mxfp8_layout` derives the DeepGEMM-packed scale
this GEMM reads.

:func:`run_k3_fp8w_lowm_gemm_fused` runs either kernel with a fused epilogue instead of the BF16
store (whole 128-row weight tiles only): the owner of each token chunk stages its BF16 rows
token-major in the dead activation ring, then writes the NVFP4 quantization of them
(``fp4_quantize`` arithmetic); the results equal the BF16 GEMM at the same tactic followed by
``fp4_quantize``. The fused Kimi-K3 latent_down projection (``k3_latent_fp8_gemm``) runs it on
SM107.

All-reduce push epilogue (:func:`run_k3_fp8w_lowm_gemm_ar_push`, low-M kernel): instead of
storing its BF16 output, the kernel stores it into the Lamport slots of the next
MNNVL all-reduce on a workspace (:mod:`k3_lamport_ar`): each token row goes to its owner's
scatter slot (two-shot) or to every rank's slot (one-shot, multicast), with ``-0`` sent as
``+0`` as the C++ all-reduce sends it. The epilogue stages the BF16 tile token-major in the dead
weight ring and pushes 16-byte vectors; with split-K (SM107 only, see :func:`ar_push_tactic`) each
cluster rank stages and pushes the token rows it owns after the reduction, in the dead
quantized-activation ring like the fused NVFP4 epilogue. :func:`k3_lamport_ar.run_lamport_finish`
completes the
all-reduce, so the pair is bitwise equal to this GEMM followed by the MNNVL all-reduce of its
output. PDL: this variant releases its dependent (the finish kernel) from the activation warp
right after its own ``griddepcontrol.wait`` instead of from the weight warp at kernel start. The
finish kernel reads the workspace's rotation state as soon as it is resident, which is then
safe whatever precedes this GEMM: once this GEMM has passed its wait, every earlier grid has
completed (each waits for its predecessor before it completes), the previous all-reduce on the
workspace included. The weight warp still streams before the wait.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional, Tuple

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

from tensorrt_llm._torch.kernels.blackwell.fp8_block_scale_low_m_gemm import (
    _abs_max_bf16x32,
    _align_up,
    _asm,
    _bf16x2_inv_scale,
    _bulk_commit,
    _bulk_copy_to_peer,
    _bulk_load_g2s,
    _bulk_wait_read,
    _ceil_div,
    _current_stream,
    _device_capability,
    _dummy_ptr,
    _f32_to_bf16_bits,
    _gmem_ptr,
    _iv,
    _ldg_v4_b32,
    _lds_v4_b32,
    _lds_v4_f32,
    _max_active_clusters,
    _mbarrier_arrive_remote,
    _mid_mailbox_bytes,
    _min_s32,
    _num_sms,
    _quant_bf16x8_to_e4m3,
    _s2t_partition_sf,
    _sf_blocks,
    _st_async_v4_f32,
    _stg_u8,
    _stg_v4_b32,
    _sts_scale,
    _sts_u16,
    _sts_v2_b32,
    _sts_v4_b32,
    _sts_v4_f32,
    _tile_n,
    _ue8m0_of_amax,
    check_inputs,
    max_mid_a_stages,
    mid_mma_n_for,
    mid_smem_bytes,
    mma_n_for,
    pack_block_scale_from_mxfp8_layout,
    validate_mid_tactic,
)
from tensorrt_llm._torch.kernels.blackwell.utils import (
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)

from ..comm import rubin_ull_k3_lamport_ar as k3_lamport_ar
from .rubin_ull_latent_fp8_gemm import (
    _abs_max_f32x16,
    _e2m1x8,
    _ldg_f32,
    _nvfp4_scales_x2,
    _rcp_approx_ftz,
)

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
#: ``ar_push`` modes of the all-reduce push epilogue (the C++ kernel the call stands in for).
AR_PUSH_TWO_SHOT = 1
AR_PUSH_ONE_SHOT = 2

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

_SMEM_CAPACITY = utils.get_smem_capacity_in_bytes("sm_100")


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


def smem_bytes(k: int, tactic: LowMTactic) -> int:
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
    need = smem_bytes(k, tactic)
    if need > _SMEM_CAPACITY:
        raise ValueError(f"tactic {tactic} needs {need} B of SMEM > {_SMEM_CAPACITY} B")


def max_a_stages(k: int, tactic: LowMTactic) -> int:
    """Largest weight-ring depth that fits next to the other buffers."""
    kt_max = _ceil_div(k // CTA_K, tactic.split_k)
    best = 1
    for a in range(1, kt_max + 1):
        if smem_bytes(k, dataclasses.replace(tactic, a_stages=a)) <= _SMEM_CAPACITY:
            best = a
    return best


# ---------------------------------------------------------------------------
# Output epilogues
# ---------------------------------------------------------------------------

#: BF16 ``out [M, N]`` (the GEMM itself).
EPI_BF16 = "bf16"
#: NVFP4 of the BF16 result, as ``fp4_quantize(out, global_sf, 16)`` with the linear scale
#: layout: E2M1 codes ``[M, N / 2]`` (low nibble first) and E4M3 scales ``[M, N / 16]``.
EPI_NVFP4 = "nvfp4"
_EPILOGUES = (EPI_BF16, EPI_NVFP4)


@dsl_user_op
def _bf16x2_to_f32x2(word, *, loc=None, ip=None):
    """The two BF16 halves of ``word`` (low half first) widened to FP32 (exact)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(word, loc=loc, ip=ip)],
        "{\n"
        ".reg .b32 lo, hi;\n"
        "shl.b32 lo, $2, 16;\n"
        "and.b32 hi, $2, 0xffff0000;\n"
        "mov.b32 $0, lo;\n"
        "mov.b32 $1, hi;\n"
        "}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2)
    )


@dsl_user_op
def _stg_u16(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u16 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _add_bf16x2(a, b, *, loc=None, ip=None):
    """Lane-wise ``bf16_rn(float(a) + float(b))`` of two BF16 pairs (an FP32 round-to-nearest add,
    as PyTorch adds BF16 tensors)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "{\n.reg .b32 al, ah, bl, bh;\n.reg .f32 fal, fah, fbl, fbh, sl, sh;\n"
            "shl.b32 al, $1, 16;\nand.b32 ah, $1, 0xffff0000;\n"
            "shl.b32 bl, $2, 16;\nand.b32 bh, $2, 0xffff0000;\n"
            "mov.b32 fal, al;\nmov.b32 fah, ah;\nmov.b32 fbl, bl;\nmov.b32 fbh, bh;\n"
            "add.rn.f32 sl, fal, fbl;\nadd.rn.f32 sh, fah, fbh;\n"
            "cvt.rn.bf16x2.f32 $0, sh, sl;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


class _FusedStoreMixin:
    """Final stores of the NVFP4 epilogue from a token-major SMEM tile.

    Both GEMMs stage the BF16 result of their 128-row weight tile as token rows of 256 bytes
    (``stage_base + t * 256``, column ``c`` of the tile at byte ``2 c``) before this runs.
    Row ``r < nrows`` of the work list is token ``r`` (``split_k == 1``) or the ``r``-th
    token this cluster rank owns (chunk ``rank + (r // 4) * split_k``, 4 tokens per chunk).
    """

    @cute.jit
    def _owned_token(self, r, rank):
        t = r
        if cutlass.const_expr(self.split_k > 1):
            t = 4 * (rank + (r >> 2) * self.split_k) + (r & 3)
        return t

    @cute.jit
    def fused_store(self, stage_base, nrows, rank, m, n_tile, q, out0, out1, aux0):
        """``out0`` codes, ``out1`` scales, ``aux0`` global scale (FP32)."""
        n = self.n
        col0 = n_tile * MMA_M
        gsf = _ldg_f32(aux0.toint())
        rgsf = _rcp_approx_ftz(gsf)
        rcp6 = _rcp_approx_ftz(cutlass.Float32(6.0))
        # One item = 32 columns of one token: two 16-element NVFP4 groups.
        for p in cutlass.range(q, nrows * 4, MMA_M, unroll=1):
            t = self._owned_token(p >> 2, rank)
            g = p & 3
            if t < m:
                src = stage_base + t * (MMA_M * 2) + g * 64
                vals = []
                for v in cutlass.range_constexpr(4):
                    words = _lds_v4_b32(src + v * 16)
                    for i in cutlass.range_constexpr(4):
                        lo, hi = _bf16x2_to_f32x2(words[i])
                        vals.extend((lo, hi))
                amax0 = _abs_max_f32x16(vals[:16])
                amax1 = _abs_max_f32x16(vals[16:])
                sf2, os0, os1 = _nvfp4_scales_x2(amax0, amax1, gsf, rgsf, rcp6)
                col = col0 + g * 32
                _stg_v4_b32(
                    out0.toint() + Int64(t) * (n // 2) + Int64(col // 2),
                    _e2m1x8(vals[0:8], os0),
                    _e2m1x8(vals[8:16], os0),
                    _e2m1x8(vals[16:24], os1),
                    _e2m1x8(vals[24:32], os1),
                )
                _stg_u16(out1.toint() + Int64(t) * (n // 16) + Int64(col // 16), sf2)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class Fp8BlockScaleLowMGemmKernel(_FusedStoreMixin):
    """Swap-AB MXFP8 GEMM with in-kernel activation quantization and cluster split-K.

    ``epilogue`` selects the output (:data:`EPI_BF16` or :data:`EPI_NVFP4`); the NVFP4 one stages
    the BF16 tile token-major in the dead quantized-activation ring and needs whole 128-row weight
    tiles.

    ``ar_push`` (``AR_PUSH_TWO_SHOT`` / ``AR_PUSH_ONE_SHOT``, ``ar_world`` ranks) replaces the
    BF16 store by the all-reduce push (:meth:`call_ar_push`, ``_push_lamport``, or with split-K
    ``_push_owned_rows``); it needs the BF16 epilogue.
    ``split_src = (kt1, ldw1, ldx1)`` reads the K tiles from two operand pairs
    (:meth:`call_split`, :meth:`call_split_ar_push`): K tiles ``[0, kt1)`` from a weight with row
    stride ``ldw1`` and an activation with row stride ``ldx1``, the others from a second
    contiguous weight / activation pair, as if both were concatenated along K (with ``kt1`` = all
    K tiles, only the strided pair). It needs the BF16 epilogue (or the push).
    ``push_addend`` (with the push): every pushed BF16 value ``y`` becomes ``bf16(y + a)`` with
    ``a`` from a BF16 ``[M, N]`` tensor (the PyTorch BF16 add).
    """

    def __init__(
        self,
        n: int,
        k: int,
        tactic: LowMTactic,
        use_pdl: bool,
        epilogue: str = EPI_BF16,
        ar_push: int = 0,
        ar_world: int = 0,
        split_src: Optional[Tuple[int, int, int]] = None,
        push_addend: bool = False,
        ar_slot_n: int = 0,
    ):
        validate_tactic(n, k, tactic)
        _validate_epilogue(n, epilogue)
        if ar_push:
            if ar_push not in (AR_PUSH_TWO_SHOT, AR_PUSH_ONE_SHOT) or ar_world < 2:
                raise ValueError(f"invalid all-reduce push {ar_push} over {ar_world} ranks")
            if epilogue != EPI_BF16:
                raise ValueError("the all-reduce push needs the BF16 epilogue")
        # Row width (features) of one Lamport slot of the push: N, or more where ranks with
        # different N push into slots of one size (the latent all-gather of k3_ull_front).
        if ar_slot_n and (ar_slot_n < n or ar_slot_n % 8):
            raise ValueError(f"ar_slot_n={ar_slot_n} must be a multiple of 8 and >= N={n}")
        self.ar_slot_n = ar_slot_n or n
        if split_src is not None:
            kt1, ldw1, ldx1 = split_src
            if not 1 <= kt1 <= k // CTA_K:
                raise ValueError(f"split_src: kt1={kt1} not in [1, {k // CTA_K}]")
            if ldw1 < kt1 * CTA_K or ldw1 % 16 or ldx1 < kt1 * CTA_K or ldx1 % 8:
                raise ValueError(f"split_src: invalid row strides ldw1={ldw1} ldx1={ldx1}")
            if epilogue != EPI_BF16:
                raise ValueError("split_src goes with the BF16 epilogue or the push only")
        if push_addend and not ar_push:
            raise ValueError("push_addend goes with the all-reduce push")
        self.split_src = split_src
        # Two operand pairs (split_src with K tiles left for the second pair).
        self.two_src = split_src is not None and split_src[0] < k // CTA_K
        self.push_addend = push_addend
        self.epilogue_kind = epilogue
        self.ar_push = ar_push
        self.ar_world = ar_world
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
        if epilogue != EPI_BF16 and self.b_stages * self.b_stage_bytes < self.mma_n * MMA_M * 2:
            raise ValueError("the NVFP4 epilogue stages the output tile in >= 2 b_stages")
        staged = self.b_stages * self.b_stage_bytes >= self.mma_n * MMA_M * 2
        if ar_push and self.split_k > 1 and not staged:
            raise ValueError("the split-K all-reduce push stages its rows in >= 2 b_stages")

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
    ):
        """``out_ptr``: the BF16 output, or the NVFP4 codes (``out1_ptr`` scales, ``aux0_ptr``
        global scale)."""
        self._launch(w_ptr, x_ptr, ws_ptr, out_ptr, out1_ptr, aux0_ptr, m, stream)

    @cute.jit
    def call_ar_push(
        self,
        w_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        m: Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        """All-reduce push launch: the output ``x @ W^T`` goes into the Lamport slots of the
        next call on the workspace (:class:`k3_lamport_ar.LamportWorkspace` addresses: peer
        pointer table, multicast base, state flags; this is rank ``rank`` of ``ar_world``).
        Nothing is stored through the output pointers, so ``x_ptr`` stands in for them."""
        self._launch(
            w_ptr,
            x_ptr,
            ws_ptr,
            x_ptr,
            x_ptr,
            x_ptr,
            m,
            stream,
            ar_peers=peer_ptrs,
            ar_mc=mc_base,
            ar_flags=flags,
            ar_rank=rank,
        )

    @cute.jit
    def call_split(
        self,
        w_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        x2_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
    ):
        """Two-source launch (``split_src``): ``out = [x | x2] @ [w | w2]^T`` with ``w`` / ``x``
        the strided first sources and ``w2`` / ``x2`` the contiguous second ones."""
        self._launch(
            w_ptr,
            x_ptr,
            ws_ptr,
            out_ptr,
            out_ptr,
            out_ptr,
            m,
            stream,
            w2_ptr=w2_ptr,
            x2_ptr=x2_ptr,
        )

    @cute.jit
    def call_split_ar_push(
        self,
        w_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        x2_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        add_ptr: cute.Pointer,
        m: Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        """:meth:`call_split` with the all-reduce push epilogue (see :meth:`call_ar_push`);
        ``add_ptr``: the BF16 ``[M, N]`` addend of ``push_addend``."""
        self._launch(
            w_ptr,
            x_ptr,
            ws_ptr,
            x_ptr,
            x_ptr,
            add_ptr,
            m,
            stream,
            ar_peers=peer_ptrs,
            ar_mc=mc_base,
            ar_flags=flags,
            ar_rank=rank,
            w2_ptr=w2_ptr,
            x2_ptr=x2_ptr,
        )

    @cute.jit
    def _launch(
        self,
        w_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
        ar_peers=None,
        ar_mc=None,
        ar_flags=None,
        ar_rank=None,
        w2_ptr=None,
        x2_ptr=None,
    ):
        args = self.kernel_args(
            w_ptr,
            x_ptr,
            ws_ptr,
            out_ptr,
            out1_ptr,
            aux0_ptr,
            m,
            ar_peers,
            ar_mc,
            ar_flags,
            ar_rank,
            w2_ptr,
            x2_ptr,
        )
        self.kernel(*args).launch(
            grid=(self.n_tiles * self.split_k, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.split_k, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def kernel_args(
        self,
        w_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        m: Int32,
        ar_peers=None,
        ar_mc=None,
        ar_flags=None,
        ar_rank=None,
        w2_ptr=None,
        x2_ptr=None,
    ) -> tuple:
        """The arguments of :meth:`kernel` (TMA atoms, layouts, tensors) for one launch, traced
        inside a ``cute.jit`` caller; also sets ``self.shared_storage``. A kernel that runs this
        GEMM's CTAs next to other roles (``k3_ull_front``) passes them to :meth:`cta`."""
        n, k = self.n, self.k
        mW2, mX2 = None, None
        if cutlass.const_expr(self.split_src is not None):
            kt1, ldw1, ldx1 = self.split_src
            k1 = kt1 * CTA_K
            k2 = k - k1
            mW = cute.make_tensor(w_ptr, cute.make_layout((n, k1, 1), stride=(ldw1, 1, n * ldw1)))
            mX = cute.make_tensor(x_ptr, cute.make_layout((m, k1), stride=(ldx1, 1)))
            if cutlass.const_expr(self.two_src):
                mW2 = cute.make_tensor(w2_ptr, cute.make_layout((n, k2, 1), stride=(k2, 1, n * k2)))
                mX2 = cute.make_tensor(x2_ptr, cute.make_layout((m, k2), stride=(k2, 1)))
        else:
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
        # Second weight / activation sources of the two-source variant (``split_src``).
        tma_atom_a2, tma_tensor_a2, tma_atom_x2, tma_tensor_x2 = None, None, None, None
        if cutlass.const_expr(self.two_src):
            tma_atom_a2, tma_tensor_a2 = cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                mW2,
                cute.slice_(a_layout, (None, None, None, 0)),
                self.mma_tiler,
                tiled_mma,
                cluster_layout_vmnk.shape,
            )
            tma_atom_x2, tma_tensor_x2 = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mX2,
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

        return (
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_x,
            tma_tensor_x,
            mWS,
            mOut,
            out1_ptr,
            aux0_ptr,
            m,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            x_layout,
            ar_peers,
            ar_mc,
            ar_flags,
            ar_rank,
            tma_atom_a2,
            tma_tensor_a2,
            tma_atom_x2,
            tma_tensor_x2,
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
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        m: Int32,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        x_layout: cute.Layout,
        ar_peers: Optional[Int64],
        ar_mc: Optional[Int64],
        ar_flags: Optional[Int64],
        ar_rank: Optional[Int32],
        tma_atom_a2: Optional[cute.CopyAtom],
        mA2: Optional[cute.Tensor],
        tma_atom_x2: Optional[cute.CopyAtom],
        mXt2: Optional[cute.Tensor],
    ):
        self.cta(
            tiled_mma,
            tma_atom_a,
            mA,
            tma_atom_x,
            mXt,
            mWS,
            mOut,
            out1_ptr,
            aux0_ptr,
            m,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            x_layout,
            ar_peers,
            ar_mc,
            ar_flags,
            ar_rank,
            tma_atom_a2,
            mA2,
            tma_atom_x2,
            mXt2,
        )

    @cute.jit
    def cta(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mXt: cute.Tensor,
        mWS: cute.Tensor,
        mOut: cute.Tensor,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        m: Int32,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        x_layout: cute.Layout,
        ar_peers: Optional[Int64],
        ar_mc: Optional[Int64],
        ar_flags: Optional[Int64],
        ar_rank: Optional[Int32],
        tma_atom_a2: Optional[cute.CopyAtom],
        mA2: Optional[cute.Tensor],
        tma_atom_x2: Optional[cute.CopyAtom],
        mXt2: Optional[cute.Tensor],
        storage=None,
    ):
        """One CTA of the GEMM grid (block ``n_tile * split_k + rank``), the body of
        :meth:`kernel`. ``storage``: the ``shared_storage`` instance if the caller allocated it
        (a kernel with other roles shares one allocation between them)."""
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

        if cutlass.const_expr(storage is None):
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
            if cutlass.const_expr(self.two_src):
                cpasync.prefetch_descriptor(tma_atom_a2)
                cpasync.prefetch_descriptor(tma_atom_x2)
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
            # Weight producer: never waits on the previous grid. The all-reduce push variant
            # releases its dependent after its own wait (below).
            if cutlass.const_expr(self.use_pdl and not self.ar_push):
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
            tAgA2 = None
            if cutlass.const_expr(self.two_src):
                gA2 = cute.local_tile(
                    mA2, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
                )
                _, tAgA2 = cpasync.tma_partition(
                    tma_atom_a2,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(tiled_mma.get_slice(0).partition_A(gA2), 0, 3),
                )
            for kt in cutlass.range(kt_count, unroll=1):
                stage = kt % self.a_stages
                if kt >= self.a_stages:
                    cute.arch.mbarrier_wait(a_empty + stage, ((kt // self.a_stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, self.a_stage_bytes)
                if cutlass.const_expr(self.two_src):
                    # K tiles [0, kt1) from the first weight, the rest from the second.
                    gk = kt_start + kt
                    kt1 = self.split_src[0]
                    if gk < kt1:
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, n_tile, gk, 0)],
                            tAsA[(None, stage)],
                            tma_bar_ptr=a_full + stage,
                            cache_policy=Int64(_EVICT_FIRST),
                        )
                    else:
                        cute.copy(
                            tma_atom_a2,
                            tAgA2[(None, n_tile, gk - kt1, 0)],
                            tAsA[(None, stage)],
                            tma_bar_ptr=a_full + stage,
                            cache_policy=Int64(_EVICT_FIRST),
                        )
                else:
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
                if cutlass.const_expr(self.ar_push != 0):
                    # The all-reduce finish kernel reads the Lamport rotation state as soon as
                    # it is resident (k3_lamport_ar): release it only once every earlier grid
                    # (the previous all-reduce on the workspace included) has completed.
                    griddepcontrol_launch_dependents()
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
            tXgX2 = None
            if cutlass.const_expr(self.two_src):
                _, tXgX2 = cpasync.tma_partition(
                    tma_atom_x2,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sX, 0, 2),
                    cute.group_modes(cute.local_tile(mXt2, (self.mma_n, CTA_K), (0, None)), 0, 2),
                )
            for kt in cutlass.range(kt_count, unroll=1):
                stage = kt % self.x_stages
                if kt >= self.x_stages:
                    cute.arch.mbarrier_wait(x_empty + stage, ((kt // self.x_stages) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(x_full + stage, self.x_stage_bytes)
                if cutlass.const_expr(self.two_src):
                    gk = kt_start + kt
                    kt1 = self.split_src[0]
                    if gk < kt1:
                        cute.copy(
                            tma_atom_x,
                            tXgX[(None, gk)],
                            tXsX[(None, stage)],
                            tma_bar_ptr=x_full + stage,
                        )
                    else:
                        cute.copy(
                            tma_atom_x2,
                            tXgX2[(None, gk - kt1)],
                            tXsX[(None, stage)],
                            tma_bar_ptr=x_full + stage,
                        )
                else:
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
                    out1_ptr,
                    aux0_ptr,
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
                    ar_peers,
                    ar_mc,
                    ar_flags,
                    ar_rank,
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
        sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfb_layout, (None, None, None, 0))
        )
        sSFA = storage.sSFA.get_tensor(sfa_layout)
        sSFB = storage.sSFB.get_tensor(sfb_layout)
        s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = _s2t_partition_sf(sSFA, tCtSFA)
        # SFB: TMEM columns [mma_n + 16, +16). The scale copy of a K tile waits for the MMAs of
        # the previous tile to read the same columns.
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(
                acc_tmem_ptr + self.mma_n + _SF_TMEM_COLS,
                dtype=cutlass.Float8E8M0FNU,
            ),
            sfb_tmem_layout,
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

    def _out_value(self, acc):
        """The stored BF16 value of one FP32 accumulator."""
        return acc.to(cutlass.BFloat16)

    @cute.jit
    def epilogue(
        self,
        tiled_mma: cute.TiledMma,
        storage,
        mOut: cute.Tensor,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
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
        ar_peers: Optional[Int64] = None,
        ar_mc: Optional[Int64] = None,
        ar_flags: Optional[Int64] = None,
        ar_rank: Optional[Int32] = None,
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
        # Fused NVFP4 epilogue: BF16 token rows staged in the dead quantized-activation ring.
        stage_out = storage.sB.data_ptr().toint()
        if cutlass.const_expr(self.ar_push != 0 and S == 1):
            self._push_lamport(
                storage,
                rAcc,
                tTR_cAcc,
                m,
                q,
                n_base,
                ar_peers,
                ar_mc,
                ar_flags,
                ar_rank,
                aux0_ptr,
            )
        elif cutlass.const_expr(S == 1):
            if cutlass.const_expr(self.epilogue_kind == EPI_BF16):
                for i in cutlass.range_constexpr(nchunks * 4):
                    row, col = tTR_cAcc[i]
                    if col < m_store:
                        mOut[col, n_base + row] = self._out_value(rAcc[i])
            else:
                for i in cutlass.range_constexpr(nchunks * 4):
                    row, col = tTR_cAcc[i]
                    if col < m:
                        _sts_u16(
                            stage_out + col * (MMA_M * 2) + row * 2, _f32_to_bf16_bits(rAcc[i])
                        )
                cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=MMA_M)
                self.fused_store(
                    stage_out, m, rank, m, n_tile, q, mOut.iterator, out1_ptr, aux0_ptr
                )
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
                                if cutlass.const_expr(
                                    self.epilogue_kind == EPI_BF16 and self.ar_push == 0
                                ):
                                    if col < m_store:
                                        mOut[col, n_base + row] = self._out_value(rAcc[4 * c + j])
                                else:
                                    if col < m:
                                        _sts_u16(
                                            stage_out + col * (MMA_M * 2) + row * 2,
                                            _f32_to_bf16_bits(rAcc[4 * c + j]),
                                        )
                if cutlass.const_expr(self.ar_push != 0):
                    cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=MMA_M)
                    self._push_owned_rows(
                        stage_out,
                        my_chunks * 4,
                        rank,
                        m,
                        n_base,
                        q,
                        ar_peers,
                        ar_mc,
                        ar_flags,
                        ar_rank,
                        aux0_ptr,
                    )
                elif cutlass.const_expr(self.epilogue_kind != EPI_BF16):
                    cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=MMA_M)
                    self.fused_store(
                        stage_out,
                        my_chunks * 4,
                        rank,
                        m,
                        n_tile,
                        q,
                        mOut.iterator,
                        out1_ptr,
                        aux0_ptr,
                    )
            if q == 0:
                # The staging buffer must stay valid until the copies have read it.
                _bulk_wait_read()

    @cute.jit
    def _add_addend(self, w, addend: cute.Pointer, t, col):
        """``bf16(w + a)`` per BF16 lane of the 16-byte vector ``w`` (features ``[col, col + 8)``
        of token ``t``), ``a`` from the BF16 ``[M, N]`` addend (the PyTorch BF16 add)."""
        a = _ldg_v4_b32(addend.toint() + (Int64(t) * self.n + Int64(col)) * 2)
        return tuple(_add_bf16x2(w[i], a[i]) for i in range(4))

    @cute.jit
    def _push_lamport(
        self,
        storage,
        rAcc: cute.Tensor,
        tTR_cAcc: cute.Tensor,
        m: Int32,
        q: Int32,
        n_base: Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        ar_rank: Int32,
        addend: cute.Pointer,
    ):
        """All-reduce push of this CTA's output tile (``split_k == 1``): round every value to
        BF16 (as the BF16 store would), stage the tile token-major in the dead weight ring, then
        store 16-byte vectors (8 features of one token) into the Lamport slots of the next call
        on the workspace (``k3_lamport_ar.push_slot``; ``-0`` is sent as ``+0``), with
        ``push_addend`` after the BF16 add of ``addend``. The epilogue has passed
        ``griddepcontrol.wait``: the previous call on the workspace has completed, so its
        rotation state is final, and the addend is complete."""
        stage_base = storage.sA.data_ptr().toint()
        tile_row = MMA_M * 2
        for i in cutlass.range_constexpr(cute.size(rAcc)):
            row, col = tTR_cAcc[i]
            _sts_u16(stage_base + col * tile_row + row * 2, _f32_to_bf16_bits(rAcc[i]))
        cute.arch.barrier(barrier_id=_EPI_BARRIER_ID, number_of_threads=MMA_M)
        cur, _, nbytes, _, _, _, _, _ = k3_lamport_ar.lamport_state(flags)
        one_shot = self.ar_push == AR_PUSH_ONE_SHOT
        vecs = MMA_M // 8
        for p in cutlass.range_constexpr(self.mma_n * vecs // MMA_M):
            j = q + p * MMA_M
            t = j // vecs
            c = j - t * vecs
            if t < m and n_base + c * 8 < self.n:
                w = _lds_v4_b32(stage_base + t * tile_row + c * 16)
                if cutlass.const_expr(self.push_addend):
                    w = self._add_addend(w, addend, t, n_base + c * 8)
                dst = k3_lamport_ar.push_slot(
                    peer_ptrs,
                    mc_base,
                    cur,
                    nbytes,
                    t,
                    ar_rank,
                    n_base + c * 8,
                    self.ar_world,
                    self.ar_slot_n,
                    one_shot,
                )
                k3_lamport_ar.push_v4(dst, *w, one_shot)

    @cute.jit
    def _push_owned_rows(
        self,
        stage_base,
        nrows: Int32,
        rank: Int32,
        m: Int32,
        n_base: Int32,
        q: Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        ar_rank: Int32,
        addend: cute.Pointer,
    ):
        """All-reduce push after the split-K reduction: the ``nrows`` token rows this cluster
        rank owns are staged as BF16 at ``stage_base + t * 256`` (as for the NVFP4 epilogue;
        the ``r``-th owned row is token ``_owned_token(r, rank)``) and go out in 16-byte
        vectors like :meth:`_push_lamport`'s, so every (token, feature) vector is pushed by
        exactly one CTA of the cluster."""
        cur, _, nbytes, _, _, _, _, _ = k3_lamport_ar.lamport_state(flags)
        one_shot = self.ar_push == AR_PUSH_ONE_SHOT
        # One item = 8 features (16 bytes) of one owned row; MMA_M // 8 = 16 items per row.
        for p in cutlass.range(q, nrows * 16, MMA_M, unroll=1):
            t = self._owned_token(p >> 4, rank)
            c = p & 15
            if t < m and n_base + c * 8 < self.n:
                w = _lds_v4_b32(stage_base + t * (MMA_M * 2) + c * 16)
                if cutlass.const_expr(self.push_addend):
                    w = self._add_addend(w, addend, t, n_base + c * 8)
                dst = k3_lamport_ar.push_slot(
                    peer_ptrs,
                    mc_base,
                    cur,
                    nbytes,
                    t,
                    ar_rank,
                    n_base + c * 8,
                    self.ar_world,
                    self.ar_slot_n,
                    one_shot,
                )
                k3_lamport_ar.push_v4(dst, *w, one_shot)


# ---------------------------------------------------------------------------
# Mid-M path (M up to 256): standalone MXFP8 activation quantization + GEMM with a
# TMA-loaded FP8 activation operand.
# ---------------------------------------------------------------------------

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


class Fp8BlockScaleMidMGemmKernel(_FusedStoreMixin):
    """Swap-AB MXFP8 GEMM fed by pre-quantized FP8 activations (64 < M <= 256).

    Same weight path and split-K cluster as :class:`Fp8BlockScaleLowMGemmKernel`, but the
    activation operand is the output of :class:`MxFp8QuantKernel`: warp 2 loads the
    E4M3 tile with TMA (128B swizzle, i.e. directly in the UMMA B layout) and its scale
    block with a bulk copy. The weight block scale goes into SFA (one UE8M0 per K tile,
    replicated over the 128 rows), written once to SMEM before the mainloop. Warps 4-7 run
    the epilogue in 32-column TMEM blocks; with ``split_k > 1`` each rank owns every
    ``split_k``-th 4-token chunk and peers push partials with ``st.async`` into its
    weight ring once every CTA of the cluster has finished its mainloop. ``epilogue`` selects
    the output as in :class:`Fp8BlockScaleLowMGemmKernel`.

    ``ar_push`` (``AR_PUSH_TWO_SHOT`` / ``AR_PUSH_ONE_SHOT``, ``ar_world`` ranks;
    :meth:`call_ar_push`): the staged BF16 rows go into the Lamport slots of the next all-reduce
    on a workspace instead of the output (:meth:`_push_staged_rows`, the low-M kernel's push),
    and the dependent grid is released from the activation warp after its
    ``griddepcontrol.wait``.
    """

    _THREADS = 256
    _B_WARP = 2
    _EPI_WARP0 = 4

    def __init__(
        self,
        n: int,
        k: int,
        tactic: MidMTactic,
        use_pdl: bool,
        epilogue: str = EPI_BF16,
        ar_push: int = 0,
        ar_world: int = 0,
    ):
        validate_mid_tactic(n, k, tactic)
        _validate_epilogue(n, epilogue)
        if ar_push:
            if ar_push not in (AR_PUSH_TWO_SHOT, AR_PUSH_ONE_SHOT) or ar_world < 2:
                raise ValueError(f"invalid all-reduce push {ar_push} over {ar_world} ranks")
            if epilogue != EPI_BF16:
                raise ValueError("the all-reduce push needs the BF16 epilogue")
        self.ar_push = ar_push
        self.ar_world = ar_world
        self.epilogue_kind = epilogue
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
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
    ):
        self._launch(w_ptr, xq_ptr, xs_ptr, ws_ptr, out_ptr, out1_ptr, aux0_ptr, m, stream)

    @cute.jit
    def call_ar_push(
        self,
        w_ptr: cute.Pointer,
        xq_ptr: cute.Pointer,
        xs_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        m: Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        """All-reduce push launch (see :meth:`Fp8BlockScaleLowMGemmKernel.call_ar_push`). Nothing is
        stored through the output pointers, so ``xs_ptr`` stands in for them."""
        self._launch(
            w_ptr,
            xq_ptr,
            xs_ptr,
            ws_ptr,
            xs_ptr,
            xs_ptr,
            xs_ptr,
            m,
            stream,
            ar_peers=peer_ptrs,
            ar_mc=mc_base,
            ar_flags=flags,
            ar_rank=rank,
        )

    @cute.jit
    def _launch(
        self,
        w_ptr: cute.Pointer,
        xq_ptr: cute.Pointer,
        xs_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
        ar_peers=None,
        ar_mc=None,
        ar_flags=None,
        ar_rank=None,
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
            out1_ptr,
            aux0_ptr,
            m,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            ar_peers,
            ar_mc,
            ar_flags,
            ar_rank,
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
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        m: Int32,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        ar_peers: Optional[Int64],
        ar_mc: Optional[Int64],
        ar_flags: Optional[Int64],
        ar_rank: Optional[Int32],
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
            # Weight producer: never waits on the previous grid. The all-reduce push variant
            # releases its dependent after its own wait (below).
            if cutlass.const_expr(self.use_pdl and not self.ar_push):
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
                if cutlass.const_expr(self.ar_push != 0):
                    # A finish kernel reads the Lamport rotation state as soon as it is
                    # resident (k3_lamport_ar): release it only once every earlier grid (the
                    # previous all-reduce on the workspace included) has completed.
                    griddepcontrol_launch_dependents()
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
                out1_ptr,
                aux0_ptr,
                m,
                tidx - 32 * self._EPI_WARP0,
                n_tile,
                rank,
                ar_peers,
                ar_mc,
                ar_flags,
                ar_rank,
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
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        m: Int32,
        q: Int32,
        n_tile: Int32,
        rank: Int32,
        ar_peers: Optional[Int64],
        ar_mc: Optional[Int64],
        ar_flags: Optional[Int64],
        ar_rank: Optional[Int32],
    ):
        """TMEM -> registers (32-column blocks) -> token-major SMEM tile -> 16-byte stores
        (BF16), :meth:`fused_store` or :meth:`_push_staged_rows` (``ar_push``).

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
            if cutlass.const_expr(self.ar_push != 0):
                self._push_staged_rows(
                    out_base, m, rank, m, n_tile, q, store_cols, ar_peers, ar_mc, ar_flags, ar_rank
                )
            elif cutlass.const_expr(self.epilogue_kind == EPI_BF16):
                for p in cutlass.range(q, m * store_cols, MMA_M, unroll=1):
                    t = p >> 4
                    col = (p & 15) * 16
                    w0, w1, w2, w3 = _lds_v4_b32(out_base + t * (MMA_M * 2) + col)
                    _stg_v4_b32(out_gmem + Int64(t) * row_stride + col, w0, w1, w2, w3)
            else:
                self.fused_store(out_base, m, rank, m, n_tile, q, mOut.iterator, out1_ptr, aux0_ptr)
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
                if cutlass.const_expr(self.ar_push != 0):
                    self._push_staged_rows(
                        out_base,
                        my_chunks * 4,
                        rank,
                        m,
                        n_tile,
                        q,
                        store_cols,
                        ar_peers,
                        ar_mc,
                        ar_flags,
                        ar_rank,
                    )
                elif cutlass.const_expr(self.epilogue_kind == EPI_BF16):
                    # The r-th owned row is token 4 * (rank + (r // 4) * S) + r % 4.
                    for p in cutlass.range(q, my_chunks * 4 * store_cols, MMA_M, unroll=1):
                        r = p >> 4
                        t = 4 * (rank + (r >> 2) * S) + (r & 3)
                        if t < m:
                            col = (p & 15) * 16
                            w0, w1, w2, w3 = _lds_v4_b32(out_base + t * (MMA_M * 2) + col)
                            _stg_v4_b32(out_gmem + Int64(t) * row_stride + col, w0, w1, w2, w3)
                else:
                    self.fused_store(
                        out_base,
                        my_chunks * 4,
                        rank,
                        m,
                        n_tile,
                        q,
                        mOut.iterator,
                        out1_ptr,
                        aux0_ptr,
                    )

    @cute.jit
    def _push_staged_rows(
        self,
        stage_base,
        nrows: Int32,
        rank: Int32,
        m: Int32,
        n_tile: Int32,
        q: Int32,
        store_cols: Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        ar_rank: Int32,
    ):
        """All-reduce push of the staged BF16 rows (token ``t`` at ``stage_base + t * 256``):
        rows ``r < nrows``, i.e. token ``r`` or with split-K this rank's ``r``-th owned token, go
        out as the BF16 store's 16-byte vectors through ``k3_lamport_ar.push_slot`` /
        ``push_v4`` (``-0`` sent as ``+0``), so every (token, feature) vector is pushed once.
        The epilogue has passed ``griddepcontrol.wait``: the previous call on the workspace has
        completed and its rotation state is final."""
        cur, _, nbytes, _, _, _, _, _ = k3_lamport_ar.lamport_state(flags)
        one_shot = self.ar_push == AR_PUSH_ONE_SHOT
        for p in cutlass.range(q, nrows * store_cols, MMA_M, unroll=1):
            r = p >> 4
            t = r
            if cutlass.const_expr(self.split_k > 1):
                t = 4 * (rank + (r >> 2) * self.split_k) + (r & 3)
            if t < m:
                c = p & 15
                w0, w1, w2, w3 = _lds_v4_b32(stage_base + t * (MMA_M * 2) + c * 16)
                dst = k3_lamport_ar.push_slot(
                    peer_ptrs,
                    mc_base,
                    cur,
                    nbytes,
                    t,
                    ar_rank,
                    n_tile * MMA_M + c * 8,
                    self.ar_world,
                    self.n,
                    one_shot,
                )
                k3_lamport_ar.push_v4(dst, w0, w1, w2, w3, one_shot)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ShapeTables:
    """Tactics and selection limits, keyed by ``(N in whole tiles, K)``."""

    #: (N, K, mma_n) -> (split_k, x_stages, b_stages, quant_warps, quant_groups) of the
    #: low-M kernel; the weight ring takes the remaining shared memory.
    low_m_tuned: dict
    #: (N, K, mma_n) -> (split_k, b_stages) of the mid-M kernel; ``mid_m_default`` per (N, K)
    #: for buckets without an entry.
    mid_m_tuned: dict
    mid_m_default: dict
    #: (N, K) -> largest M for which :func:`can_run` selects this GEMM; other shapes are not
    #: selected.
    max_m: dict
    #: (N, K) -> largest M of the low-M kernel (mid-M kernel above); ``LOW_M_MAX`` otherwise.
    low_m_cutover: dict


# SM107, tuned on Rubin (212 SMs) for the Kimi-K3 decode projections with cold weights (rotated
# over 1.5 GiB): each tactic scores the mean CUDA-graph per-call time of back-to-back launches
# with and without PDL. Low-M kernel, per mma_n bucket timed at M = mma_n: a split_k sweep with
# the heuristic ring sizes, then quant warps / groups and x / b ring depths around the three
# best splits; mid-M kernel, per mma_n bucket: split_k x b_stages. The M limits are the largest
# M up to which this GEMM ran at least 5% faster than ``fp8_quantize_1x128_cutedsl_ue8m0`` + the
# autotuned ``cute_dsl_mxfp8_gemm_rubin`` at every measured M (and, for the helix weights, not
# slower on the mean of the times with and without PDL); other shapes stop at 32 tokens.
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


def _is_supported_device(device_index: int) -> bool:
    """SM107 with a Rubin-capable CuTe DSL."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return _device_capability(device_index) == (10, 7) and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _tables() -> _ShapeTables:
    """The tables of the one device this GEMM serves, SM107."""
    return _SM107_TABLES


#: The low-M kernel serves up to ``LOW_M_MAX`` tokens when each CTA streams at most this many K
#: tiles: up to there its in-kernel quantization of a 64-token tile costs less against the mid-M
#: path than its prequantized variants save by folding in a separate kernel.
#: With longer K slices the mid-M path takes over above 32 tokens.
_LOW_M_FULL_RANGE_K_TILES = 12


def max_m_for(n: int, k: int) -> int:
    """Largest M for which this GEMM is selected for an ``[N, K]`` weight on the current device.

    0 where it never is: unsupported architecture, a weight shape the kernels do not take, or a
    shape without a measured limit.
    """
    if not _is_supported_device(torch.cuda.current_device()) or n % N_ALIGN or k % CTA_K:
        return 0
    return min(MAX_M, _tables().max_m.get((_tile_n(n), k), 0))


def low_m_max_for(n: int, k: int) -> int:
    """Largest M served by the low-M kernel for an ``[N, K]`` weight (mid-M kernel above)."""
    cutover = _tables().low_m_cutover.get((_tile_n(n), k))
    if cutover is not None:
        return cutover
    k_tiles = _ceil_div(k // CTA_K, _rule_split(n, k, mid=False))
    return LOW_M_MAX if k_tiles <= _LOW_M_FULL_RANGE_K_TILES else 32


def kernel_max_m_for(n: int, k: int) -> int:
    """Largest M for which the selected GEMM runs the low-M kernel for an ``[N, K]`` weight on the
    current device: where its prequantized variants match it bitwise."""
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
        return dataclasses.replace(base, a_stages=max_a_stages(k, base))
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
        tactic = dataclasses.replace(base, a_stages=max_a_stages(k, base))
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


@functools.cache
def split_tactic(m: int, n: int, k: int, split_k: int, mid: bool) -> LowMTactic | MidMTactic:
    """The default tactic for M tokens with ``split_k`` cluster ranks instead of the tuned count:
    the low-M kernel's (its ring depths rounded down to whole quantization groups) or, with
    ``mid``, the mid-M kernel's (4 activation stages, fewer if they do not fit)."""
    if mid:
        for b_stages in (4, 3, 2):
            base = MidMTactic(
                mma_n=mid_mma_n_for(m), split_k=split_k, a_stages=1, b_stages=b_stages
            )
            tactic = dataclasses.replace(base, a_stages=max_mid_a_stages(n, k, base))
            try:
                validate_mid_tactic(n, k, tactic)
            except ValueError:
                continue
            return tactic
        raise ValueError(f"no mid-M tactic with split_k={split_k} for M={m}, N={n}, K={k}")
    base = default_tactic(m, n, k)
    kt_max = _ceil_div(k // CTA_K, split_k)
    groups = base.quant_groups

    def whole_groups(stages):
        return max(groups, min(stages, kt_max) // groups * groups)

    tactic = dataclasses.replace(
        base,
        split_k=split_k,
        a_stages=1,
        x_stages=whole_groups(base.x_stages),
        b_stages=max(2, whole_groups(base.b_stages)),
    )
    tactic = dataclasses.replace(tactic, a_stages=max_a_stages(k, tactic))
    validate_tactic(n, k, tactic)
    return tactic


#: Largest M per (N, K) for which this kernel on the BF16 activation (quantized inside the
#: kernel) beats the CuTe MXFP8 GEMM alone on an activation its producer already quantized
#: (``_Fp8BlockScaleWeightReadLinear.forward_prequantized``, the KDA MTP verify o_proj), by the
#: helix gate rule of the SM107 tables. Other shapes always take the prequantized operand.
_SM107_BF16_OVER_PREQUANTIZED_MAX_M = {
    (7168, 1536): 32,
    (7168, 768): 32,
}


def bf16_over_prequantized_max_m(n: int, k: int) -> int:
    """Largest M up to which a producer that can emit the SM107 MXFP8 operand of an ``[N, K]``
    weight should rather hand its BF16 output to this kernel (0: never; always 0 off SM107,
    where :func:`max_m_for` is 0)."""
    table = _SM107_BF16_OVER_PREQUANTIZED_MAX_M.get((_tile_n(n), k), 0)
    return min(max_m_for(n, k), table)


@functools.cache
def _compiled_kernel(
    n: int,
    k: int,
    tactic: LowMTactic,
    use_pdl: bool,
    epilogue: str = EPI_BF16,
):
    kernel = Fp8BlockScaleLowMGemmKernel(n, k, tactic, use_pdl, epilogue)
    return cute.compile(
        kernel,
        _dummy_ptr(cutlass.Float8E4M3FN),
        _dummy_ptr(cutlass.BFloat16),
        _dummy_ptr(cutlass.Uint8),
        _dummy_ptr(cutlass.BFloat16),
        *(_dummy_ptr(cutlass.Uint8) for _ in range(2)),
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
def _compiled_mid_kernel(
    n: int, k: int, tactic: MidMTactic, use_pdl: bool, epilogue: str = EPI_BF16
):
    kernel = Fp8BlockScaleMidMGemmKernel(n, k, tactic, use_pdl, epilogue)
    return cute.compile(
        kernel,
        _dummy_ptr(cutlass.Float8E4M3FN),
        _dummy_ptr(cutlass.Float8E4M3FN),
        _dummy_ptr(cutlass.Uint8),
        _dummy_ptr(cutlass.Uint8),
        _dummy_ptr(cutlass.BFloat16),
        *(_dummy_ptr(cutlass.Uint8) for _ in range(2)),
        Int32(1),
        _current_stream(torch.device("cuda")),
    )


def _validate_epilogue(n: int, epilogue: str) -> None:
    if epilogue not in _EPILOGUES:
        raise ValueError(f"epilogue must be one of {_EPILOGUES}, got {epilogue!r}")
    if epilogue != EPI_BF16 and n % MMA_M:
        raise ValueError(f"the {epilogue} epilogue needs N={n} to be a multiple of {MMA_M}")


def _run_mid(
    x, w, w_scale, out, tactic: MidMTactic, use_pdl: bool, epilogue=EPI_BF16, extra=None
) -> None:
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
    _compiled_mid_kernel(n, k, tactic, use_pdl, epilogue)(
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        _gmem_ptr(cutlass.Uint8, w_scale),
        _gmem_ptr(cutlass.BFloat16, out),
        *(_gmem_ptr(cutlass.Uint8, t) for t in (extra or (out, out))),
        Int32(m),
        stream,
    )


def mid_prequant_plan(m: int, n: int, k: int) -> Optional[Tuple[MidMTactic, int]]:
    """``(tactic, sf_blocks)`` of the mid-M path :func:`run_low_m_gemm` runs for ``m``
    rows of an ``[N, K]`` weight (its default tactic), or None where it runs the low-M kernel.

    A producer that stores the MXFP8 form of those rows exactly as :class:`MxFp8QuantKernel`
    does (E4M3 ``[M, K]``; UE8M0 1x32 exponents in ``[K / 128][sf_blocks][512 B]`` tcgen05 atoms,
    byte ``(r % 32) * 16 + (r % 128 // 32) * 4 + k32 % 4``) can hand them to
    :func:`run_mid_prequantized` instead: bitwise the same output.
    """
    if not low_m_max_for(n, k) < m <= MAX_M:
        return None
    tactic = default_mid_tactic(m, n, k)
    if m > tactic.mma_n:
        return None
    return tactic, _sf_blocks(tactic.mma_n)


def run_mid_prequantized(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    tactic: MidMTactic,
    use_pdl: bool = True,
) -> torch.Tensor:
    """:func:`_run_mid` without its quantization kernel: the mid-M GEMM (BF16 epilogue into
    ``out [M, N]``) on the MXFP8 operand ``xq`` (E4M3 ``[M, K]``) / ``xs`` (UE8M0,
    ``(K / 128) * sf_blocks * 512`` bytes, see :func:`mid_prequant_plan`). With ``use_pdl`` it
    launches as a programmatic dependent of the producer and streams its weights before its
    ``griddepcontrol.wait``."""
    m, k = xq.shape
    n = w.shape[0]
    stream = _current_stream(xq.device)
    _compiled_mid_kernel(n, k, tactic, use_pdl, EPI_BF16)(
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        _gmem_ptr(cutlass.Uint8, w_scale),
        _gmem_ptr(cutlass.BFloat16, out),
        *(_gmem_ptr(cutlass.Uint8, t) for t in (out, out)),
        Int32(m),
        stream,
    )
    return out


def quantize_mid_operand(
    x: torch.Tensor, tactic: MidMTactic, use_pdl: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The MXFP8 operand of the mid-M GEMM (``tactic``) for the BF16 rows ``x [M, K]``, as
    :func:`run_low_m_gemm` quantizes them (:class:`MxFp8QuantKernel`): E4M3 ``[M, K]`` and
    UE8M0 scales ``[(K / 128) * sf_blocks * 512]`` in that GEMM's layout, for
    :func:`run_mid_prequantized` (``_ar_push``). With one scale block (``tactic.mma_n <= 128``)
    the scales are also the R128c4 layout of the CuTe MXFP8 GEMM."""
    m, k = x.shape
    sf_blocks = _sf_blocks(tactic.mma_n)
    xq = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty((k // CTA_K) * sf_blocks * 512, dtype=torch.uint8, device=x.device)
    _compiled_quant(k, sf_blocks, use_pdl)(
        _gmem_ptr(cutlass.BFloat16, x),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        Int32(m),
        _current_stream(x.device),
    )
    return xq, xs


def mid_ar_push_tactic(m: int, n: int, k: int) -> Optional[MidMTactic]:
    """The mid-M tactic :func:`run_low_m_gemm` runs for ``m`` rows of an ``[N, K]``
    weight (:func:`mid_prequant_plan`), which the mid-M all-reduce push epilogue runs with the
    same bits; None where the low-M kernel serves ``m`` (see :func:`ar_push_tactic`) or above
    the mid-M range."""
    plan = mid_prequant_plan(m, n, k)
    if plan is None or n % 8:
        return None
    return plan[0]


@functools.cache
def _compiled_mid_ar_push_kernel(
    n: int, k: int, tactic: MidMTactic, use_pdl: bool, ar_push: int, ar_world: int
):
    kernel = Fp8BlockScaleMidMGemmKernel(n, k, tactic, use_pdl, ar_push=ar_push, ar_world=ar_world)
    dummy = 1 << 20
    ptrs = [
        make_ptr(dtype, dummy, cute.AddressSpace.gmem, assumed_align=16)
        for dtype in (cutlass.Float8E4M3FN, cutlass.Float8E4M3FN, cutlass.Uint8, cutlass.Uint8)
    ]
    addr = Int64(dummy)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(kernel.call_ar_push, *ptrs, Int32(1), addr, addr, addr, Int32(0), stream)


def run_mid_prequantized_ar_push(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    ws: "k3_lamport_ar.LamportWorkspace",
    one_shot: bool,
    tactic: MidMTactic,
    use_pdl: bool = True,
) -> None:
    """:func:`run_mid_prequantized` with the all-reduce push epilogue: ``bf16(xq @ W^T)``
    goes into the Lamport slots of the next all-reduce on ``ws`` instead of an output tensor;
    :func:`k3_lamport_ar.run_lamport_finish` with the same ``one_shot`` (or a consumer that
    finishes it) completes it, bitwise the GEMM followed by the MNNVL all-reduce of its
    output."""
    m, k = xq.shape
    n = w.shape[0]
    if m > tactic.mma_n:
        raise ValueError(f"M={m} exceeds tactic.mma_n={tactic.mma_n}")
    need = k3_lamport_ar.required_bytes(m, n, ws.world, one_shot)
    if need > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B < {need} B")
    mode = AR_PUSH_ONE_SHOT if one_shot else AR_PUSH_TWO_SHOT
    compiled = _compiled_mid_ar_push_kernel(n, k, tactic, use_pdl, mode, ws.world)
    stream = _current_stream(xq.device)
    compiled(
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        _gmem_ptr(cutlass.Uint8, w_scale),
        Int32(m),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(ws.rank),
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
        tactic: kernel specialization; the default tactic if None.
        use_pdl: launch with programmatic dependent launch.
    """
    _launch(x, w, w_scale, out, None, EPI_BF16, tactic, use_pdl)
    return out


def run_k3_fp8w_lowm_gemm_fused(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    epilogue: str,
    out0: torch.Tensor,
    out1: torch.Tensor,
    aux0: torch.Tensor,
    *,
    tactic: LowMTactic | MidMTactic | None = None,
    use_pdl: bool = True,
) -> None:
    """The GEMM of :func:`run_low_m_gemm` (same kernels, tactics and numerics) with a
    fused epilogue on the BF16 result ``y``:

    * :data:`EPI_NVFP4`: ``out0`` / ``out1`` = ``fp4_quantize(y, aux0, 16)`` codes ``uint8
      [M, N / 2]`` / linear E4M3 scales ``uint8 [M, N / 16]``; ``aux0`` is the FP32 global
      scale.

    N must be a multiple of 128 and the low-M tactic must stage the output tile in its
    quantized-activation ring (``b_stages >= 2``, true for every default tactic).
    """
    if epilogue == EPI_BF16:
        raise ValueError("use run_low_m_gemm for the BF16 output")
    _launch(x, w, w_scale, out0, (out1, aux0), epilogue, tactic, use_pdl)


def _launch(x, w, w_scale, out, extra, epilogue, tactic, use_pdl) -> None:
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
        _run_mid(x, w, w_scale, out, tactic, use_pdl, epilogue, extra)
        return
    _compiled_kernel(n, k, tactic, use_pdl, epilogue)(
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        _gmem_ptr(cutlass.BFloat16, x),
        _gmem_ptr(cutlass.Uint8, w_scale),
        _gmem_ptr(cutlass.BFloat16, out),
        *(_gmem_ptr(cutlass.Uint8, t) for t in (extra or (out, out))),
        Int32(m),
        _current_stream(x.device),
    )


@functools.cache
def _compiled_ar_push_kernel(
    n: int, k: int, tactic: LowMTactic, use_pdl: bool, ar_push: int, ar_world: int
):
    kernel = Fp8BlockScaleLowMGemmKernel(n, k, tactic, use_pdl, ar_push=ar_push, ar_world=ar_world)
    w_ptr = _dummy_ptr(cutlass.Float8E4M3FN)
    x_ptr = _dummy_ptr(cutlass.BFloat16)
    ws_ptr = _dummy_ptr(cutlass.Uint8)
    stream = _current_stream(torch.device("cuda"))
    addr = Int64(1 << 20)
    return cute.compile(
        kernel.call_ar_push, w_ptr, x_ptr, ws_ptr, Int32(1), addr, addr, addr, Int32(0), stream
    )


def ar_push_tactic(m: int, n: int, k: int) -> Optional[LowMTactic]:
    """The tactic the low-M GEMM uses for this shape, if the all-reduce push epilogue can run it
    (low-M kernel; ``split_k == 1``, or on SM107 also split-K); else None.

    The SM107 tables split K for most Kimi-K3 decode projections (212 SMs), so on Rubin the
    split-K owners push their rows (:meth:`Fp8BlockScaleLowMGemmKernel._push_owned_rows`) and the
    push keeps the tactic, and with it the bits, of the plain GEMM. Whether a pushed split-K
    GEMM beats the GEMM plus the C++ all-reduce at each shape and token count still has to be
    measured on Rubin.
    """
    if not 1 <= m <= low_m_max_for(n, k) or n % 8:
        return None
    tactic = default_tactic(m, n, k)
    if tactic.split_k == 1:
        return tactic
    kt_max = _ceil_div(k // CTA_K, tactic.split_k)
    if min(tactic.b_stages, kt_max) * tactic.mma_n * CTA_K < tactic.mma_n * MMA_M * 2:
        return None
    return tactic


def run_k3_fp8w_lowm_gemm_ar_push(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    ws: "k3_lamport_ar.LamportWorkspace",
    one_shot: bool,
    *,
    tactic: LowMTactic | MidMTactic | None = None,
    use_pdl: bool = True,
) -> None:
    """Low-M GEMM with the all-reduce push epilogue, on the current stream.

    Stores ``bf16(x @ dequant(w)^T)`` into the Lamport slots of the next all-reduce on ``ws``
    instead of an output tensor; :func:`k3_lamport_ar.run_lamport_finish` with the same
    ``one_shot`` completes it. Same tactic and rounding as :func:`run_low_m_gemm`, so the pair is
    bitwise that GEMM followed by the MNNVL all-reduce of its output.

    Above the low-M range the mid-M path runs, its quantization kernel then the mid-M GEMM with
    the push epilogue (:func:`mid_ar_push_tactic`).
    """
    m, k = x.shape
    n = w.shape[0]
    if tactic is None:
        tactic = ar_push_tactic(m, n, k)
        if tactic is None:
            tactic = mid_ar_push_tactic(m, n, k)
        if tactic is None:
            raise ValueError(f"no all-reduce push tactic for M={m}, N={n}, K={k}")
    if isinstance(tactic, MidMTactic):
        if m > tactic.mma_n:
            raise ValueError(f"M={m} exceeds tactic.mma_n={tactic.mma_n}")
        xq, xs = quantize_mid_operand(x, tactic, use_pdl)
        run_mid_prequantized_ar_push(xq, xs, w, w_scale, ws, one_shot, tactic, use_pdl)
        return
    if m > tactic.mma_n:
        raise ValueError(f"M={m} exceeds tactic.mma_n={tactic.mma_n}")
    need = k3_lamport_ar.required_bytes(m, n, ws.world, one_shot)
    if need > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B < {need} B")
    mode = AR_PUSH_ONE_SHOT if one_shot else AR_PUSH_TWO_SHOT
    compiled = _compiled_ar_push_kernel(n, k, tactic, use_pdl, mode, ws.world)
    ptrs = (_gmem_ptr(cutlass.Float8E4M3FN, w), _gmem_ptr(cutlass.BFloat16, x))
    ws_ptr = _gmem_ptr(cutlass.Uint8, w_scale)
    tail = (
        Int32(m),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(ws.rank),
        _current_stream(x.device),
    )
    compiled(*ptrs, ws_ptr, *tail)


@functools.cache
def _compiled_split_kernel(
    n: int,
    k: int,
    tactic: LowMTactic,
    use_pdl: bool,
    split_src: Tuple[int, int, int],
    ar_push: int,
    ar_world: int,
    push_addend: bool = False,
):
    kernel = Fp8BlockScaleLowMGemmKernel(
        n,
        k,
        tactic,
        use_pdl,
        ar_push=ar_push,
        ar_world=ar_world,
        split_src=split_src,
        push_addend=push_addend,
    )
    dummy = 1 << 20
    w_ptrs = [make_ptr(cutlass.Float8E4M3FN, dummy, cute.AddressSpace.gmem, assumed_align=16)] * 2
    x_ptrs = [make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)] * 2
    ws_ptr = make_ptr(cutlass.Uint8, dummy, cute.AddressSpace.gmem, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if ar_push:
        addr = Int64(dummy)
        add_ptr = make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)
        return cute.compile(
            kernel.call_split_ar_push,
            *w_ptrs,
            *x_ptrs,
            ws_ptr,
            add_ptr,
            Int32(1),
            addr,
            addr,
            addr,
            Int32(0),
            stream,
        )
    out_ptr = make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(kernel.call_split, *w_ptrs, *x_ptrs, ws_ptr, out_ptr, Int32(1), stream)


#: Largest cluster split-K and minimum K tiles per CTA of :func:`_split_source_split`.
_SPLIT_SOURCE_MAX_SPLIT, _SPLIT_SOURCE_MIN_K_TILES = 8, 3


@functools.cache
def _split_source_split(n: int, k: int) -> int:
    """Cluster split-K of the two-source GEMM for an ``[N, K]`` weight without measured tactics:
    the split with the most CTAs whose clusters all run in one wave on the whole GPU, each CTA
    keeping at least ``_SPLIT_SOURCE_MIN_K_TILES`` K tiles. The row-parallel MoE tail's K slices
    are short (a few hundred columns per rank), so they split where :func:`_rule_split` keeps
    one CTA per weight tile."""
    kt_total = k // CTA_K
    n_tiles = _ceil_div(n, MMA_M)
    best, split = 0, 1
    for s in range(1, min(_SPLIT_SOURCE_MAX_SPLIT, kt_total) + 1):
        if n_tiles > _max_active_clusters(s) or n_tiles * s > _num_sms():
            continue
        if s > 1 and kt_total // s < _SPLIT_SOURCE_MIN_K_TILES:
            continue
        if n_tiles * s > best:
            best, split = n_tiles * s, s
    return split


@functools.cache
def split_source_tactic(m: int, n: int, k: int) -> Optional[LowMTactic]:
    """The tactic of the two-source GEMM (:func:`run_k3_fp8w_lowm_gemm_split`) for ``M`` tokens
    of an ``[N, K]`` weight (``K`` the total of both sources), where its all-reduce push
    epilogue can run it; else None. Shapes with measured low-M tactics or cut-over take
    :func:`ar_push_tactic`; the others are served by the low-M kernel up to ``LOW_M_MAX`` tokens
    (the two-source GEMM has no mid-M path) with the split-K of :func:`_split_source_split`."""
    if not 1 <= m <= LOW_M_MAX or n % 8:
        return None
    tables = _tables()
    if (_tile_n(n), k) in tables.low_m_cutover or (
        (_tile_n(n), k, mma_n_for(m)) in tables.low_m_tuned
    ):
        return ar_push_tactic(m, n, k)
    split = _split_source_split(n, k)
    tactic = default_tactic(m, n, k)
    if split != tactic.split_k:
        try:
            tactic = split_tactic(m, n, k, split, mid=False)
        except ValueError:
            return ar_push_tactic(m, n, k)
    if tactic.split_k == 1:
        return tactic
    kt_max = _ceil_div(k // CTA_K, tactic.split_k)
    if min(tactic.b_stages, kt_max) * tactic.mma_n * CTA_K < tactic.mma_n * MMA_M * 2:
        return ar_push_tactic(m, n, k)
    return tactic


def _check_split_inputs(x1, x2, w1, w2, w_scale) -> Tuple[int, int, int, int, int]:
    """``(m, n, k1, k2, kt1)`` of valid two-source operands (see
    :func:`run_k3_fp8w_lowm_gemm_split`; ``x2`` / ``w2`` may both be None, ``k2`` is then 0);
    raises ValueError otherwise."""
    if (x2 is None) != (w2 is None):
        raise ValueError("pass both or neither of x2 and w2")
    pairs = [(x1, "x1", torch.bfloat16), (w1, "w1", torch.float8_e4m3fn)]
    if x2 is not None:
        pairs += [(x2, "x2", torch.bfloat16), (w2, "w2", torch.float8_e4m3fn)]
    for t, name, dtype in pairs:
        if t.dtype != dtype or t.dim() != 2 or t.stride(1) != 1 or t.data_ptr() % 16:
            raise ValueError(f"{name} must be a 16-byte aligned 2D {dtype} tensor, unit col stride")
    m, k1 = x1.shape
    n = w1.shape[0]
    k2 = 0 if x2 is None else x2.shape[1]
    if tuple(w1.shape) != (n, k1):
        raise ValueError("operand shapes do not match")
    if x2 is not None:
        if x2.shape[0] != m or tuple(w2.shape) != (n, k2):
            raise ValueError("operand shapes do not match")
        if not (x2.is_contiguous() and w2.is_contiguous()):
            raise ValueError("x2 and w2 must be contiguous")
    if k1 % CTA_K or k2 % CTA_K or n % N_ALIGN:
        raise ValueError(f"K1={k1} and K2={k2} must be multiples of {CTA_K}, N={n} of {N_ALIGN}")
    if x1.stride(0) % 8 or w1.stride(0) % 16:
        raise ValueError("x1 / w1 row strides must be 16-byte multiples")
    words = _ceil_div((k1 + k2) // CTA_K, 4)
    if (
        w_scale.dtype != torch.int32
        or tuple(w_scale.shape) != (n, words)
        or w_scale.stride(0) != 1
        or (words > 1 and w_scale.stride(1) != n)
    ):
        raise ValueError(f"w_scale must be DeepGEMM-packed UE8M0 int32 [{n}, {words}]")
    return m, n, k1, k2, k1 // CTA_K


def run_k3_fp8w_lowm_gemm_split(
    x1: torch.Tensor,
    x2: Optional[torch.Tensor],
    w1: torch.Tensor,
    w2: Optional[torch.Tensor],
    w_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    tactic: LowMTactic | None = None,
    use_pdl: bool = True,
) -> torch.Tensor:
    """``out = [x1 | x2] @ dequant([w1 | w2], w_scale)^T`` (BF16 ``[M, N]``) with the K tiles of
    both operand pairs read in place, low-M kernel only.

    ``x1`` (BF16 ``[M, K1]``) and ``w1`` (E4M3 ``[N, K1]``) may be column slices of wider tensors
    (unit column stride, 16-byte aligned rows); ``x2`` / ``w2`` (``[M, K2]`` / ``[N, K2]``) are
    contiguous, or both None (only the first pair); ``w_scale`` is the DeepGEMM-packed UE8M0 scale
    of the concatenated ``[N, K1 + K2]`` weight. Bitwise equal to :func:`run_k3_fp8w_lowm_gemm` on
    the concatenated operands with the same tactic.
    """
    m, n, k1, k2, kt1 = _check_split_inputs(x1, x2, w1, w2, w_scale)
    k = k1 + k2
    if tactic is None:
        if not 1 <= m <= LOW_M_MAX:
            raise ValueError(f"M={m} must be in [1, {LOW_M_MAX}] for the two-source GEMM")
        tactic = split_source_tactic(m, n, k) or default_tactic(m, n, k)
    if m > tactic.mma_n:
        raise ValueError(f"M={m} exceeds tactic.mma_n={tactic.mma_n}")
    if out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n) or not out.is_contiguous():
        raise ValueError("out must be a contiguous bf16 [M, N] tensor")
    split_src = (kt1, w1.stride(0), x1.stride(0))
    compiled = _compiled_split_kernel(n, k, tactic, use_pdl, split_src, 0, 0)
    compiled(
        _gmem_ptr(cutlass.Float8E4M3FN, w1),
        _gmem_ptr(cutlass.Float8E4M3FN, w1 if w2 is None else w2),
        _gmem_ptr(cutlass.BFloat16, x1),
        _gmem_ptr(cutlass.BFloat16, x1 if x2 is None else x2),
        _gmem_ptr(cutlass.Uint8, w_scale),
        _gmem_ptr(cutlass.BFloat16, out),
        Int32(m),
        cuda.CUstream(torch.cuda.current_stream(x1.device).cuda_stream),
    )
    return out


def run_k3_fp8w_lowm_gemm_split_ar_push(
    x1: torch.Tensor,
    x2: Optional[torch.Tensor],
    w1: torch.Tensor,
    w2: Optional[torch.Tensor],
    w_scale: torch.Tensor,
    ws: "k3_lamport_ar.LamportWorkspace",
    one_shot: bool,
    *,
    addend: Optional[torch.Tensor] = None,
    tactic: LowMTactic | None = None,
    use_pdl: bool = True,
) -> None:
    """:func:`run_k3_fp8w_lowm_gemm_split` with the all-reduce push epilogue of
    :func:`run_k3_fp8w_lowm_gemm_ar_push`: the BF16 ``[M, N]`` result goes into the Lamport slots
    of the next all-reduce on ``ws`` (``one_shot`` or two-shot transport). With ``addend`` (a
    contiguous BF16 ``[M, N]`` tensor, read after the kernel's ``griddepcontrol.wait``) the pushed
    values are ``result + addend`` as the PyTorch BF16 add computes it."""
    m, n, k1, k2, kt1 = _check_split_inputs(x1, x2, w1, w2, w_scale)
    k = k1 + k2
    if addend is not None and (
        addend.dtype != torch.bfloat16
        or tuple(addend.shape) != (m, n)
        or not addend.is_contiguous()
        or addend.data_ptr() % 16
    ):
        raise ValueError("addend must be a contiguous, 16-byte aligned bf16 [M, N] tensor")
    if tactic is None:
        tactic = split_source_tactic(m, n, k)
        if tactic is None:
            raise ValueError(f"no all-reduce push tactic for M={m}, N={n}, K={k}")
    if m > tactic.mma_n:
        raise ValueError(f"M={m} exceeds tactic.mma_n={tactic.mma_n}")
    need = k3_lamport_ar.required_bytes(m, n, ws.world, one_shot)
    if need > ws.bytes_per_buffer:
        raise ValueError(f"Lamport buffer of {ws.bytes_per_buffer} B < {need} B")
    mode = AR_PUSH_ONE_SHOT if one_shot else AR_PUSH_TWO_SHOT
    split_src = (kt1, w1.stride(0), x1.stride(0))
    compiled = _compiled_split_kernel(
        n, k, tactic, use_pdl, split_src, mode, ws.world, addend is not None
    )
    compiled(
        _gmem_ptr(cutlass.Float8E4M3FN, w1),
        _gmem_ptr(cutlass.Float8E4M3FN, w1 if w2 is None else w2),
        _gmem_ptr(cutlass.BFloat16, x1),
        _gmem_ptr(cutlass.BFloat16, x1 if x2 is None else x2),
        _gmem_ptr(cutlass.Uint8, w_scale),
        _gmem_ptr(cutlass.BFloat16, x1 if addend is None else addend),
        Int32(m),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(ws.rank),
        cuda.CUstream(torch.cuda.current_stream(x1.device).cuda_stream),
    )


def pack_ue8m0_block_scale(w_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Pack power-of-two ``[ceil(N/128), K/128]`` FP32 block scales into DeepGEMM's layout.

    Produces the same tensor as ``transform_sf_into_required_layout(w_scale, mn=N, k=K,
    recipe=(1, 128, 128), is_sfa=False)``: UE8M0 bytes replicated per row, four K blocks
    per int32 word, MN-major ``int32 [N, ceil(K/512)]`` with strides ``(1, N)``.
    """
    ue8m0 = ((w_scale.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return pack_ue8m0_exponents(ue8m0, n, k)


def pack_ue8m0_exponents(ue8m0: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """UE8M0 block exponents ``uint8 [ceil(N/128), K/128]`` -> the DeepGEMM-packed int32 scale
    ``[N, ceil(K/512)]`` (strides ``(1, N)``) the low-M kernels read."""
    kb = k // CTA_K
    rows = ue8m0.repeat_interleave(MMA_M, dim=0)[:n]
    padded = torch.zeros((n, _ceil_div(kb, 4) * 4), dtype=torch.uint8, device=ue8m0.device)
    padded[:, :kb] = rows
    words = padded.view(torch.int32)
    packed = torch.empty((words.shape[1], n), dtype=torch.int32, device=ue8m0.device).t()
    packed.copy_(words)
    return packed


def block_exponents_of(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The UE8M0 exponents ``uint8 [ceil(N/128), K/128]`` of an ``[N, K]`` weight's 128x128
    blocks, from either layout ``_Fp8BlockScaleWeightReadLinear`` holds: the Rubin MXFP8 R128c4
    ``uint8`` scale or the DeepGEMM-packed ``int32 [N, ceil(K/512)]`` scale."""
    nb, kb = _ceil_div(n, MMA_M), k // CTA_K
    if k % CTA_K:
        raise ValueError(f"K={k} must be a multiple of {CTA_K}")
    if scale.dtype == torch.uint8:
        flat = scale.reshape(-1)
        if flat.numel() != nb * kb * 512:
            raise ValueError(f"not the R128c4 scale of an [{n}, {k}] weight")
        return flat.view(nb, kb, 512)[:, :, 0].contiguous()
    words = _ceil_div(kb, 4)
    if scale.dtype != torch.int32 or tuple(scale.shape) != (n, words):
        raise ValueError(f"not the packed int32 scale of an [{n}, {k}] weight")
    rows = scale[::MMA_M].contiguous()  # [nb, words], one row per 128-row block
    return rows.view(torch.uint8).view(nb, words * 4)[:, :kb].contiguous()


__all__ = [
    "EPI_BF16",
    "EPI_NVFP4",
    "LOW_M_MAX",
    "MAX_M",
    "N_ALIGN",
    "LowMTactic",
    "MidMTactic",
    "bf16_over_prequantized_max_m",
    "block_exponents_of",
    "check_inputs",
    "default_mid_tactic",
    "default_tactic",
    "kernel_max_m_for",
    "low_m_max_for",
    "max_m_for",
    "pack_block_scale_from_mxfp8_layout",
    "pack_ue8m0_block_scale",
    "pack_ue8m0_exponents",
    "run_k3_fp8w_lowm_gemm_fused",
    "run_k3_fp8w_lowm_gemm_split",
    "run_k3_fp8w_lowm_gemm_split_ar_push",
    "run_low_m_gemm",
    "split_source_tactic",
    "split_tactic",
]
