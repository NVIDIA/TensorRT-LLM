# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 MoE back half in one kernel for at most 8 rows (``TRTLLM_K3_ULL_TAIL1``).

After the routed experts, the MoE block of a decode step needs, per token ``t``,
``y = latent_up(rmsnorm(sum_r p_r))`` where ``p_r`` is rank ``r``'s routed latent partial
``[M, 3584]`` (BF16), and the next layer's pre-attention mix adds ``y`` and the all-reduced shared
expert to the residual. :class:`K3MoeTail1Kernel` computes this rank's ``HIDDEN / W`` columns of
``y`` (rows ``[c0, c0 + HIDDEN / W)`` of latent_up) straight from the partials the routed experts
push, and multicasts them into the all-gather rows of the tail's Lamport call, which the next
pre-attention mix finishes (``k3_attn_res`` ``finish_ag``) together with the shared expert's slots:

* Input. Each rank's BF16 latent partial rows go into slot ``(token, rank)`` of every rank's
  current buffer of the latent Lamport workspace (one-shot layout ``[M][W][3584]``, sentinel
  ``0x80000000`` per 32-bit word, ``-0`` sent as ``+0``), stored by the routed experts
  (``k3_moe_ull`` with the all-reduce push) from their combine; the kernel polls those slots and
  waits for the experts' grid only at its end.
* Grid. ``G`` clusters of 16 CTAs x 256 threads. Cluster ``g`` owns ``RG = 112`` output rows
  (:data:`ROWS_PER_CLUSTER`, ``G = (HIDDEN / W) / RG``: 4 clusters at 16 ranks); CTA ``q`` of the
  cluster owns latent columns ``[224 q, 224 q + 224)``. Before the poll each lane loads its FP8
  latent_up fragments (the 128x128 UE8M0 block scales applied, BF16 exact) into registers.
* Poll and sum. Warp ``t`` polls token ``t``'s 224 columns of all ``W`` slots (lane ``l < 28``:
  16 bytes = 8 columns per slot; every lane reloads all slots each pass and the warp leaves the
  loop by a vote). The slots are summed in FP32 in rank order, in chunks of eight ranks (the C++
  one-shot all-reduce's order) and rounded to BF16 (bitwise the latent all-reduce's sum). Each lane
  adds its eight squares (FP32) and stores ``bf16(x * gamma)`` into shared memory.
* RMSNorm late. A token's partial sum of squares goes to its two owner CTAs (``st.async``); the up
  projection runs on ``bf16(x * gamma)`` and the owners scale the summed products by
  ``rsqrt(sum / 3584 + eps)`` before rounding, so the squares exchange overlaps the MMAs.
* Up projection. ``mma.sync`` m16n8k16 BF16 with FP32 accumulation; warp ``w`` takes the 16-row
  m-tiles ``w, w + 8, ..`` over the CTA's 224 columns (14 K steps). The MMA's K index is permuted
  (:func:`_k_col`): lane group ``t``'s four columns of a K step are contiguous and those of four
  consecutive steps are one 16-byte FP8 load per weight row (steps 12-13: 8 bytes); its B fragment
  of a step is the 8 bytes of the token row at the same columns. Every weight load of a lane is
  issued before any conversion, every polling pass issues all its loads before any check.
* Owners. Output line (token ``t``, row half ``h``) is owned by CTA ``t + 8 h`` of the cluster:
  every m-tile's 4-row partials go there by ``st.async``; the owner sums the 16 CTAs' partials in
  CTA order, scales, rounds to BF16, replaces ``-0`` by ``+0`` and multicasts the row segment into
  the all-gather rows of the tail call.
* Lamport state of the latent workspace (the role of the latent all-reduce): the kernel reads the
  rotation state at entry (final: the previous call completed before the experts started) and
  counts every CTA there; it re-arms the previous call's buffer at entry. The CTA that counts last
  rotates the buffers and records this call's ``M x W`` slots as the next call's bytes to clear,
  after its ``griddepcontrol.wait`` (the experts, which read the state, have completed). The tail
  call's state is only read (``cur``, bytes per buffer); the pre-attention mix advances it.

PDL: the dependent (the pre-attention mix) is released at entry. Weights, gamma, the block
exponents and both rotation states are read before any wait; the kernel stores nothing a
predecessor reads.

Numerics (not bitwise the column-parallel tail): the latent sum is the all-reduce's; the RMS uses
FP32 squares of it; the projection's activation is ``bf16(x * gamma)`` in BF16 instead of the
MXFP8 ``bf16((x * r) * gamma)``, and ``r`` multiplies the FP32 product sum before its rounding.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .k3_lamport_ar import (
    _asm,
    _fv,
    _iv,
    _lv,
    _unpack,
    acc_bf16x2,
    add_f32,
    atom_add_acq_rel_u32,
    clean4,
    clear_dirty_stages,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    ld_volatile_v4,
    multimem_st_v4,
    pack_bf16x2,
    sanitize_bf16x2,
    st_u32,
    st_v4,
)

__all__ = [
    "HIDDEN",
    "LATENT",
    "MAX_M",
    "ROWS_PER_CLUSTER",
    "K3MoeTail1Kernel",
    "can_run",
    "rank_rows",
    "run_k3_moe_tail1",
]

#: Latent (latent_up's K) and hidden (latent_up's N) widths.
LATENT = 3584
HIDDEN = 7168
#: CTAs per cluster; each owns ``LATENT / CLUSTER`` latent columns.
CLUSTER = 16
KS = LATENT // CLUSTER
#: 16-byte column chunks of a token's K slice (one polling lane each).
CHUNKS = KS // 8
#: Threads per CTA (warp ``t`` polls token ``t``).
THREADS = 256
WARPS = THREADS // 32
#: Largest row count (one warp per token).
MAX_M = 8
#: MMA K steps (16) over a CTA's K slice.
K_STEPS = KS // 16
#: 16-row m-tiles a warp may hold in registers (56 registers each).
MAX_MTILES_PER_WARP = 2
#: Output rows of one 16-CTA cluster (four clusters at 16 ranks).
ROWS_PER_CLUSTER = 112


def _k_col(j: int):
    """``(base, lane stride)`` of K step ``j``'s first column for lane group ``t`` (column
    ``base + stride * t``, four consecutive columns per step): steps ``4 i .. 4 i + 3`` (i < 3) of
    lane group ``t`` are the 16 columns ``64 i + 16 t ..``, steps 12-13 the 8 columns
    ``192 + 8 t ..``; a bijection of the 224 columns onto (step, lane group, element)."""
    if j < 12:
        return 64 * (j // 4) + 4 * (j % 4), 16
    return 192 + 4 * (j - 12), 8


#: Bytes of a token row of ``bf16(x * gamma)`` in shared memory (16 bytes of padding).
XS_PITCH = KS * 2 + 16
#: FP32 words per token row of a warp's staged MMA partial (16 rows + padding: conflict-free).
STAGE_PITCH = 20
STAGE_WARP_BYTES = MAX_M * STAGE_PITCH * 4
#: Lamport buffers of a workspace and the word of its CTA arrival counter.
NUM_BUFFERS = 3
_FLAG_ARRIVALS = 8
#: Ranks per FP32 partial sum of the latent (the C++ one-shot all-reduce's chunking).
RANK_CHUNK = 8

assert KS % 8 == 0 and KS == 224 and 56 * 4 == KS


@dsl_user_op
def _ld_nc_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _ld_nc_u8(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )  # fmt: skip
    )


@dsl_user_op
def _ld_nc_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _deq_bf16x2(word, scale, high: bool, *, loc=None, ip=None):
    """Two E4M3 codes (the low or high 16 bits of ``word``, low byte first) times ``scale`` (a power
    of two) as a BF16 pair, low value in the low half. Exact for latent_up's weights: E4M3 -> F16 ->
    F32 are exact and the power-of-two scale keeps the 4 significant bits."""
    half = "h" if high else "l"
    return Int32(
        _asm(
            T.i32(),
            [_iv(word, loc=loc, ip=ip), _fv(scale, loc=loc, ip=ip)],
            "{\n.reg .b16 l, h, p0, p1;\n.reg .b32 p;\n.reg .f32 f0, f1;\n"
            "mov.b32 {l, h}, $1;\n"
            f"cvt.rn.f16x2.e4m3x2 p, {half};\n"
            "mov.b32 {p0, p1}, p;\n"
            "cvt.f32.f16 f0, p0;\ncvt.f32.f16 f1, p1;\n"
            "mul.rn.f32 f0, f0, $2;\nmul.rn.f32 f1, f1, $2;\n"
            "cvt.rn.bf16x2.f32 $0, f1, f0;\n}",
            "=r,r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _exp2_of_ue8m0(e, *, loc=None, ip=None):
    """``2^(e - 127)`` of a UE8M0 exponent ``1 <= e <= 254`` (the FP32 word ``e << 23``)."""
    return Float32(
        _asm(
            T.f32(),
            [_iv(e, loc=loc, ip=ip)],
            "{\n.reg .b32 b;\nshl.b32 b, $1, 23;\nmov.b32 $0, b;\n}",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_f32(word, *, loc=None, ip=None):
    """(low, high) BF16 of a pair as FP32."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
        "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip), _fv(c, loc=loc, ip=ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmul(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rms_scale(total, eps, *, loc=None, ip=None):
    """``rsqrt(total / 3584 + eps)`` (IEEE division, then add, approximate reciprocal root)."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(total, loc=loc, ip=ip), _fv(eps, loc=loc, ip=ip)],
            "{\n.reg .f32 q;\ndiv.rn.f32 q, $1, 0f45600000;\nadd.rn.f32 q, q, $2;\n"
            "rsqrt.approx.f32 $0, q;\n}",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_bfly_f32(value, offset: int, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 31, -1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    """``d[16x8] = a[16x16] @ b[16x8] + c`` (BF16 inputs, FP32 accumulation)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(x, loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [_fv(x, loc=loc, ip=ip) for x in (c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _lds_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _lds_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_iv(addr, loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            loc=loc,
            ip=ip,
        )  # fmt: skip
    )


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            loc=loc,
            ip=ip,
        )  # fmt: skip
    )


@dsl_user_op
def _sts_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_init(addr, count: int, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.init.shared::cta.b64 [$0], {count};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_expect_tx(addr, nbytes: int, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {nbytes};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_mbar_init(*, loc=None, ip=None):
    _asm(None, [], "fence.mbarrier_init.release.cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def _mbar_try_wait_cluster(addr, *, loc=None, ip=None):
    """1 once phase 0 of the local mbarrier at ``addr`` (completed by peers' ``st.async``) has
    completed, else 0 (cluster-scope acquire)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], 0;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_async_f32(local_addr, peer, local_mbar, value, *, loc=None, ip=None):
    """One FP32 into CTA ``peer``'s copy of ``local_addr``, counted by its copy of
    ``local_mbar``."""
    _asm(
        None,
        [
            _iv(local_addr, loc=loc, ip=ip),
            _iv(peer, loc=loc, ip=ip),
            _iv(local_mbar, loc=loc, ip=ip),
            _fv(value, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 ra, rm;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "mapa.shared::cluster.u32 rm, $2, $1;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [ra], $3, [rm];\n}",
        "r,r,r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_v4_f32(local_addr, peer, local_mbar, v0, v1, v2, v3, *, loc=None, ip=None):
    """16 bytes into CTA ``peer``'s copy of ``local_addr``, counted by its copy of
    ``local_mbar``."""
    _asm(
        None,
        [_iv(x, loc=loc, ip=ip) for x in (local_addr, peer, local_mbar)]
        + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "{\n.reg .b32 ra, rm;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "mapa.shared::cluster.u32 rm, $2, $1;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
        "[ra], {$3, $4, $5, $6}, [rm];\n}",
        "r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s(dst, src, nbytes: int, mbar, *, loc=None, ip=None):
    """Bulk copy of ``nbytes`` from global ``src`` to shared ``dst``, completing on ``mbar``."""
    _asm(
        None,
        [_iv(dst, loc=loc, ip=ip), _lv(src, loc=loc, ip=ip), _iv(mbar, loc=loc, ip=ip)],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        f"[$0], [$1], {nbytes}, [$2];",
        "r,l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.wait.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _bar_sync(barrier_id: int, count: int, *, loc=None, ip=None):
    _asm(None, [], f"bar.sync {barrier_id}, {count};", "", loc=loc, ip=ip)


class K3MoeTail1Kernel:
    """The MoE back half of ``m <= 8`` rows on one rank of ``world`` (see the module docstring).

    :param world: Ranks of the latent / tail workspaces (``HIDDEN / world`` output rows per rank,
        in ``HIDDEN / world / ROWS_PER_CLUSTER`` clusters; warp ``w`` multiplies the 16-row m-tiles
        ``w, w + 8, ..`` of its cluster's rows).
    :param m: Tokens of the call.
    :param mc_out: Store the output through the multicast mapping (``multimem.st``); False stores
        to the given address (single-GPU tests).
    """

    def __init__(self, world: int, m: int, mc_out: bool = True):
        rows = HIDDEN // world
        if HIDDEN % world or rows % ROWS_PER_CLUSTER:
            raise ValueError(
                f"{rows} rows per rank do not split into {ROWS_PER_CLUSTER}-row clusters"
            )
        if not 1 <= m <= MAX_M:
            raise ValueError(f"m={m} rows: 1..{MAX_M}")
        self.world = world
        self.m = m
        self.rg = ROWS_PER_CLUSTER
        self.groups = rows // ROWS_PER_CLUSTER
        self.nctas = self.groups * CLUSTER
        self.mtiles = ROWS_PER_CLUSTER // 16
        self.mpw = (self.mtiles + WARPS - 1) // WARPS
        assert self.mpw <= MAX_MTILES_PER_WARP
        # Two owners per token (row halves of the cluster's rows), 8-row lines per lane.
        self.ro = ROWS_PER_CLUSTER // 2
        self.mc_out = mc_out
        self.part_words = CLUSTER * self.ro
        self.rank_chunk = min(RANK_CHUNK, world)

    @cute.jit
    def __call__(
        self,
        lat_uc: Int64,
        lat_flags: Int64,
        w_addr: Int64,
        wexp_addr: Int64,
        gamma_addr: Int64,
        ag_base: Int64,
        ag_flags: Int64,
        c0: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            bars: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 16]
            sq: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, CLUSTER], 16]
            xs: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, MAX_M * XS_PITCH], 16]
            stage: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, WARPS * STAGE_WARP_BYTES], 16
            ]
            part: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.part_words], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            lat_uc,
            lat_flags,
            w_addr,
            wexp_addr,
            gamma_addr,
            ag_base,
            ag_flags,
            c0,
            eps,
        ).launch(  # fmt: skip
            grid=(self.nctas, 1, 1),
            block=(THREADS, 1, 1),
            cluster=(CLUSTER, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    def rotate(self, lat_flags, cur, nbytes):
        """The latent workspace's next state: buffer ``cur`` becomes the dirty one with this call's
        ``m x W`` slots to re-arm; the arrival counter back at zero (one thread)."""
        nxt = cutlass.select_(cur + 1 == NUM_BUFFERS, Int32(0), cur + 1)
        st_v4(lat_flags, nxt, cur, nbytes, Int32(1))
        call_bytes = self.m * self.world * LATENT * 2
        st_v4(lat_flags + Int64(16), Int32(call_bytes), Int32(0), Int32(0), Int32(0))
        st_u32(lat_flags + Int64(_FLAG_ARRIVALS * 4), Int32(0))

    @cute.kernel
    def kernel(
        self,
        lat_uc: Int64,
        lat_flags: Int64,
        w_addr: Int64,
        wexp_addr: Int64,
        gamma_addr: Int64,
        ag_base: Int64,
        ag_flags: Int64,
        c0: Int32,
        eps: Float32,
    ):
        W = self.world
        m = self.m
        KJ = K_STEPS
        RO = self.ro
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        q = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        gcl = bidx // CLUSTER
        gq = lane // 4
        tq = lane % 4

        griddepcontrol_launch_dependents()

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bars = storage.bars.data_ptr().toint()
        bar_sq = bars
        bar_part = bars + 8
        sq_s = storage.sq.data_ptr().toint()
        xs_s = storage.xs.data_ptr().toint()
        stage_s = storage.stage.data_ptr().toint()
        part_s = storage.part.data_ptr().toint()
        k0 = q * KS
        rbase = c0 + gcl * self.rg

        # ---- latent_up loads first (before any wait; their conversions follow the setup) ----
        # Per m-tile and weight row: three 16-byte loads (K steps 0-11) and one 8-byte load (steps
        # 12-13) of FP8 codes, and the UE8M0 exponents of the K slice's (up to) three 128-column
        # scale blocks.
        b0 = k0 // 128
        raw = []
        exps = []
        for s in cutlass.range_constexpr(self.mpw):
            mt = cutlass.min(warp + WARPS * s, self.mtiles - 1)
            r0 = rbase + mt * 16 + gq
            for h in cutlass.range_constexpr(2):
                w_row = w_addr + Int64(r0 + 8 * h) * LATENT + Int64(k0)
                for i in cutlass.range_constexpr(3):
                    raw += list(_ld_nc_v4(w_row + Int64(64 * i + 16 * tq)))
                raw += list(_ld_nc_v2(w_row + Int64(192 + 8 * tq)))
                e_row = wexp_addr + Int64(((r0 + 8 * h) // 128) * (LATENT // 128))
                for bi in cutlass.range_constexpr(3):
                    blk = cutlass.min(b0 + bi, LATENT // 128 - 1)
                    exps.append(_ld_nc_u8(e_row + Int64(blk)))
        lat_lane = (warp < m) & (lane < CHUNKS)
        gcol = k0 + cutlass.select_(lane < CHUNKS, lane, Int32(0)) * 8
        gw = _ld_nc_v4(gamma_addr + Int64(gcol) * 2)

        # ---- setup: barriers, rotation states (every thread reads them), the arrival count ----
        # Owner role: line (token q % 8, row half q // 8) of the cluster's rows.
        own_t = q % MAX_M
        own_h = q // MAX_M
        # Rotation states: final, the previous calls on both workspaces completed before the
        # experts (whose launch released this grid) passed their own wait.
        f = ld_volatile_v4(lat_flags)
        fc = ld_volatile_v4(lat_flags + Int64(16))
        a = ld_volatile_v4(ag_flags)
        cur, dirty, nbytes, dstages = f[0], f[1], f[2], f[3]
        ag_cur, ag_bytes = a[0], a[2]
        if tidx == 0:
            _mbar_init(bar_sq, 1)
            _mbar_init(bar_part, 1)
            if own_t < m:
                _mbar_expect_tx(bar_sq, CLUSTER * 4)
                _mbar_expect_tx(bar_part, CLUSTER * RO * 4)
            _fence_mbar_init()
        # Every thread of the CTA has read the state before its count.
        _bar_sync(1, THREADS)
        # The CTAs of a cluster start together; the wait (every peer's barriers initialized) comes
        # before the first remote store.
        _cluster_arrive_relaxed()
        arr_old = Int32(0)
        if tidx == 0:
            # Count this CTA: the last one rotates the buffers once the experts (which read the
            # state) completed.
            arr_old = atom_add_acq_rel_u32(lat_flags + Int64(_FLAG_ARRIVALS * 4), Int32(1))

        # ---- latent_up fragments: BF16 of the FP8 codes times the block scale ----
        afr = cute.make_rmem_tensor((self.mpw * KJ * 4,), Int32)
        scales = [_exp2_of_ue8m0(e) for e in exps]
        for s in cutlass.range_constexpr(self.mpw):
            for j in cutlass.range_constexpr(KJ):
                cb, cs = _k_col(j)
                bi = (k0 + cb + cs * tq) // 128 - b0
                sc = []
                for h in cutlass.range_constexpr(2):
                    e3 = (s * 2 + h) * 3
                    hi = cutlass.select_(bi == 1, scales[e3 + 1], scales[e3 + 2])
                    sc.append(cutlass.select_(bi == 0, scales[e3], hi))
                w0 = raw[(s * 2) * KJ + j]
                w1 = raw[(s * 2 + 1) * KJ + j]
                base = (s * KJ + j) * 4
                afr[base + 0] = _deq_bf16x2(w0, sc[0], False)
                afr[base + 1] = _deq_bf16x2(w1, sc[1], False)
                afr[base + 2] = _deq_bf16x2(w0, sc[0], True)
                afr[base + 3] = _deq_bf16x2(w1, sc[1], True)

        # The previous call's buffer (complete before this grid launched; nobody reads it).
        clear_dirty_stages(
            lat_uc, dirty, nbytes, dstages, fc[0], fc[1], fc[2], fc[3],
            bidx * THREADS + tidx, self.nctas * THREADS,
        )  # fmt: skip

        # ---- poll token `warp`'s slots of every rank (lanes < 28: 8 columns each) ----
        words = cute.make_rmem_tensor((W * 4,), Int32)
        row0 = lat_uc + Int64(cur) * Int64(nbytes) + Int64((warp * W) * LATENT + gcol) * 2
        pending = Int32(0)
        for e in cutlass.range_constexpr(W * 4):
            words[e] = Int32(0)
        # Every pass issues all W loads before it checks any (they are in flight together).
        if lat_lane:
            loaded = []
            for s in cutlass.range_constexpr(W):
                loaded.append(list(ld_volatile_v4(row0 + Int64(s * LATENT * 2))))
            for s in cutlass.range_constexpr(W):
                for e in cutlass.range_constexpr(4):
                    words[s * 4 + e] = loaded[s][e]
                clean = clean4(loaded[s][0], loaded[s][1], loaded[s][2], loaded[s][3])
                pending = pending | ((Int32(1) - clean) << s)
        while cute.arch.vote_any_sync(pending != 0):
            if pending != 0:
                loaded = []
                for s in cutlass.range_constexpr(W):
                    loaded.append(list(ld_volatile_v4(row0 + Int64(s * LATENT * 2))))
                pending = Int32(0)
                for s in cutlass.range_constexpr(W):
                    for e in cutlass.range_constexpr(4):
                        words[s * 4 + e] = loaded[s][e]
                    clean = clean4(loaded[s][0], loaded[s][1], loaded[s][2], loaded[s][3])
                    pending = pending | ((Int32(1) - clean) << s)

        # Rank-order sum (chunks of RANK_CHUNK ranks), BF16; squares; bf16(x * gamma) to smem.
        acc = [Float32(0.0)] * 8
        for cb in cutlass.range_constexpr(0, W, self.rank_chunk):
            part = [Float32(0.0)] * 8
            for s in cutlass.range_constexpr(cb, cb + self.rank_chunk):
                for j in cutlass.range_constexpr(4):
                    part[2 * j], part[2 * j + 1] = acc_bf16x2(
                        words[s * 4 + j], part[2 * j], part[2 * j + 1]
                    )
            for i in cutlass.range_constexpr(8):
                acc[i] = add_f32(acc[i], part[i])
        s2 = Float32(0.0)
        xg = []
        for j in cutlass.range_constexpr(4):
            red = pack_bf16x2(acc[2 * j], acc[2 * j + 1])
            x_lo, x_hi = _bf16x2_f32(red)
            g_lo, g_hi = _bf16x2_f32(gw[j])
            s2 = _fma(x_lo, x_lo, s2)
            s2 = _fma(x_hi, x_hi, s2)
            xg.append(pack_bf16x2(_fmul(x_lo, g_lo), _fmul(x_hi, g_hi)))
        if lane < CHUNKS:
            if warp < m:
                _sts_v4(xs_s + warp * XS_PITCH + lane * 16, xg[0], xg[1], xg[2], xg[3])
            else:
                _sts_v4(xs_s + warp * XS_PITCH + lane * 16, Int32(0), Int32(0), Int32(0), Int32(0))
        s2 = cutlass.select_(lat_lane, s2, Float32(0.0))
        for off in (16, 8, 4, 2, 1):
            s2 = s2 + _shfl_bfly_f32(s2, off)
        _cluster_wait()
        if warp < m:
            if lane < 2:
                # This CTA's sum of squares of token `warp` to its owners (row halves 0 and 1).
                _st_async_f32(sq_s + q * 4, warp + lane * MAX_M, bar_sq, s2)
        _bar_sync(1, THREADS)

        # ---- up projection: m-tiles w, w + 8, ..; 4-row partials to the line owners ----
        xrow = xs_s + gq * XS_PITCH
        bfr = []
        for j in cutlass.range_constexpr(KJ):
            cb, cs = _k_col(j)
            bfr += list(_lds_v2(xrow + (cb + cs * tq) * 2))
        stg = stage_s + warp * STAGE_WARP_BYTES
        for s in cutlass.range_constexpr(self.mpw):
            mt = warp + WARPS * s
            if mt < self.mtiles:
                # Two accumulator chains (even / odd K steps), added at the end.
                acc2 = [[Float32(0.0)] * 4, [Float32(0.0)] * 4]
                for j in cutlass.range_constexpr(KJ):
                    base = (s * KJ + j) * 4
                    ch = acc2[j % 2]
                    acc2[j % 2] = list(_mma_bf16(
                        afr[base], afr[base + 1], afr[base + 2], afr[base + 3],
                        bfr[2 * j], bfr[2 * j + 1], ch[0], ch[1], ch[2], ch[3],
                    ))  # fmt: skip
                d0 = add_f32(acc2[0][0], acc2[1][0])
                d1 = add_f32(acc2[0][1], acc2[1][1])
                d2 = add_f32(acc2[0][2], acc2[1][2])
                d3 = add_f32(acc2[0][3], acc2[1][3])
                # d0 (row gq, token 2 tq), d1 (row gq, token 2 tq + 1), d2 / d3 (row gq + 8).
                _sts_f32(stg + ((2 * tq) * STAGE_PITCH + gq) * 4, d0)
                _sts_f32(stg + ((2 * tq + 1) * STAGE_PITCH + gq) * 4, d1)
                _sts_f32(stg + ((2 * tq) * STAGE_PITCH + gq + 8) * 4, d2)
                _sts_f32(stg + ((2 * tq + 1) * STAGE_PITCH + gq + 8) * 4, d3)
                cute.arch.sync_warp()
                tk = lane // 4
                r4 = (lane % 4) * 4
                if tk < m:
                    v0, v1, v2, v3 = _lds_v4_f32(stg + (tk * STAGE_PITCH + r4) * 4)
                    rloc = mt * 16 + r4
                    h = rloc // RO
                    dst = part_s + (q * RO + rloc - h * RO) * 4
                    _st_async_v4_f32(dst, tk + h * MAX_M, bar_part, v0, v1, v2, v3)
                cute.arch.sync_warp()

        # ---- owner: sum the CTAs' partials, scale by the token's RMS reciprocal, multicast ----
        if own_t < m:
            if warp == 0:
                done = _mbar_try_wait_cluster(bar_sq)
                while done == 0:
                    done = _mbar_try_wait_cluster(bar_sq)
                done = _mbar_try_wait_cluster(bar_part)
                while done == 0:
                    done = _mbar_try_wait_cluster(bar_part)
                if lane < RO // 8:
                    total = Float32(0.0)
                    for r in cutlass.range_constexpr(CLUSTER):
                        total = add_f32(total, _lds_f32(sq_s + r * 4))
                    inv = _rms_scale(total, eps)
                    parts = []
                    for sl in cutlass.range_constexpr(CLUSTER):
                        parts.append(_lds_v4_f32(part_s + (sl * RO + lane * 8) * 4))
                        parts.append(_lds_v4_f32(part_s + (sl * RO + lane * 8 + 4) * 4))
                    y = [Float32(0.0)] * 8
                    for sl in cutlass.range_constexpr(CLUSTER):
                        for i in cutlass.range_constexpr(4):
                            y[i] = add_f32(y[i], parts[2 * sl][i])
                            y[4 + i] = add_f32(y[4 + i], parts[2 * sl + 1][i])
                    ow = [
                        sanitize_bf16x2(pack_bf16x2(_fmul(y[2 * j], inv), _fmul(y[2 * j + 1], inv)))
                        for j in range(4)
                    ]
                    col = c0 + gcl * self.rg + own_h * RO + lane * 8
                    dst = (
                        ag_base
                        + Int64(ag_cur) * Int64(ag_bytes)
                        + (Int64((m * W + own_t) * HIDDEN) + Int64(col)) * 2
                    )
                    if cutlass.const_expr(self.mc_out):
                        multimem_st_v4(dst, ow[0], ow[1], ow[2], ow[3])
                    else:
                        st_v4(dst, ow[0], ow[1], ow[2], ow[3])

        # ---- rotate once the experts (which read the state) completed ----
        griddepcontrol_wait()
        if tidx == 0:
            if arr_old == self.nctas - 1:
                self.rotate(lat_flags, cur, nbytes)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def rank_rows(world: int, rank: int):
    """``(first row, rows)`` of latent_up's output this rank computes: ``HIDDEN / world`` each."""
    n = HIDDEN // world
    return rank * n, n


def can_run(m: int, world: int) -> bool:
    """Whether the kernel serves ``m`` rows on ``world`` ranks."""
    return 1 <= m <= MAX_M and HIDDEN % world == 0 and (HIDDEN // world) % ROWS_PER_CLUSTER == 0


def _set_kernel_attributes(compiled, device_index: int) -> None:
    """Clusters of 16 (non-portable size) and the maximum shared-memory carveout (the next
    kernels' CTAs may land beside this grid's) on every kernel of ``compiled``."""
    from . import k3_launch_attrs as attrs

    status, cu_device = cuda.cuDeviceGet(device_index)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet: {status}")
    attr = cuda.CUfunction_attribute
    for _, kernel in attrs._kernels(compiled, device_index):
        for a, v in (
            (attr.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1),
            (attr.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, attrs.MAX_SHARED_CARVEOUT),
        ):
            (status,) = cuda.cuKernelSetAttribute(a, v, kernel, cu_device)
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuKernelSetAttribute({a}): {status}")


@functools.cache
def _compiled(world: int, m: int, mc_out: bool, device_index: int):
    kernel = K3MoeTail1Kernel(world, m, mc_out)
    z = Int64(0)
    with torch.cuda.device(device_index):
        compiled = cute.compile(
            kernel, z, z, z, z, z, z, z, Int32(0), Float32(1e-6),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )  # fmt: skip
    _set_kernel_attributes(compiled, device_index)
    return compiled


def run_k3_moe_tail1(
    *,
    m: int,
    world: int,
    lat_uc: int,
    lat_flags: torch.Tensor,
    weight: torch.Tensor,
    wexp: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    ag_base: int,
    ag_flags: torch.Tensor,
    c0: int,
    mc_out: bool = True,
) -> None:
    """Launch :class:`K3MoeTail1Kernel` on the current stream.

    Args:
        m: Tokens (1..8).
        world: Ranks of both workspaces.
        lat_uc / lat_flags: The latent Lamport workspace (local unicast base, uint32 [9] rotation
            state) the routed experts push their partials into.
        weight: latent_up ``[HIDDEN, LATENT]`` FP8 E4M3 (all rows; the kernel reads rows
            ``[c0, c0 + HIDDEN / world)``).
        wexp: latent_up's UE8M0 block exponents ``uint8 [HIDDEN / 128, LATENT / 128]`` (1..254).
        gamma: The routed latent RMSNorm weight ``[LATENT]`` BF16.
        ag_base / ag_flags: The tail call's workspace (multicast base, or with ``mc_out=False``
            a local base) and its rotation state; the kernel writes the call's all-gather rows.
        c0: This rank's first output row (:func:`rank_rows`).
    """
    if not can_run(m, world):
        raise ValueError(f"k3_moe_tail1 does not serve m={m} world={world}")
    if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != (HIDDEN, LATENT):
        raise ValueError("weight must be the [7168, 3584] FP8 E4M3 latent_up")
    if wexp.dtype != torch.uint8 or tuple(wexp.shape) != (HIDDEN // 128, LATENT // 128):
        raise ValueError("wexp must be uint8 [56, 28]")
    if gamma.dtype != torch.bfloat16 or gamma.numel() != LATENT:
        raise ValueError("gamma must be BF16 [3584]")
    device = weight.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    _compiled(world, m, mc_out, index)(
        Int64(lat_uc),
        Int64(lat_flags.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(wexp.data_ptr()),
        Int64(gamma.data_ptr()),
        Int64(ag_base),
        Int64(ag_flags.data_ptr()),
        Int32(c0),
        Float32(eps),
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
