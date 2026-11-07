# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ungrouped ``noaux_tc`` top-16 routing + ``moe_sort`` in one launch (SM100/SM103/SM107).

:class:`NoauxTcRouteSortKernel` replaces ``noaux_tc_op`` (one group) -> ``moe_sort`` for decode
batches by one launch on a cluster of eight small CTAs, so the routing occupies few SMs next to
the side-stream kernels it runs with.

* Input: FP32 router logits ``[M, E]``, ``1 <= M <= 256``, ``E`` a multiple of 128 up to 1024
  (compile-time parameter; Kimi-K3 routes 896 experts).
* Top-k, bit-identical to ``noaux_tc_op`` on the same logits (selection, order, weights). A
  team of ``G`` warps routes one token, lane ``g`` of the team owning experts
  ``j * 32 G + g``. Each lane forms the order-preserving bits of ``sigmoid(logit) + bias`` (the
  high word of ``noaux_tc_op``'s 64-bit key, whose low word ``65535 - expert`` breaks ties
  toward smaller experts). In each warp the 16th largest lane maximum is a lower bound of the
  token's 16th largest score; the team keeps the largest of these bounds. Every expert at or
  above it (usually about 20 per token) is a candidate: warps compact theirs into shared
  memory and rank them by their full keys. If a warp has more than 32 (64 for one-warp teams)
  candidates (heavy ties), its team's first warp instead runs 16 exact arg-max rounds over all
  ``E`` keys. The weights are the ``sigmoid`` scores renormalized like ``noaux_tc_op``.
  ``top_k = 16`` is part of the algorithm (the 16th-largest lane bound), not a parameter.
* Sort (``moe_sort`` metadata of one expert-parallel rank, 128-row tiles): CTA ``c`` of the
  cluster owns a contiguous block of at most 32 tokens and marks, per local expert, a token
  bitmask in shared memory, then stores it into every CTA of the cluster (distributed shared
  memory). After a cluster barrier each CTA scans the per-expert tile counts
  (``ceil(count / 128)`` tiles in local-expert order) and writes the rows of its own tokens:
  rows of an expert follow the token order, which is what ``moe_sort``'s single-CTA kernels
  produce (its cluster kernel orders rows by atomics). CTA 0 writes the tile arrays.

The kernel is launched with programmatic dependent launch (``use_pdl``, defaulting to
``TRTLLM_ENABLE_PDL``, like the ops it replaces): it stages the bias (a weight) in shared
memory before ``griddepcontrol.wait`` and releases its dependents once the top-k is done. It
keeps no state between launches, so a captured CUDA graph replays it as is.
"""

from __future__ import annotations

import functools
import struct

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ....kernels.blackwell.utils import TRTLLM_ENABLE_PDL
from . import raw_ptx as ptx

#: Experts selected per token.
TOP_K = 16
#: Largest token count served.
MAX_M = 256
#: Rows per tile of the ``moe_sort`` outputs (``tile_tokens_dim``).
TILE_SIZE = 128
#: CTAs of the cluster (portable cluster size).
CLUSTER_SIZE = 8
#: Largest expert count: the exact fallback keeps a lane's ``E / 128 * 4`` slots in a 32-bit mask.
MAX_EXPERTS = 1024
#: Tokens per CTA are capped by the 32-bit token mask of the sort.
_MAX_TOKENS_PER_CTA = 32
#: Bytes of one compacted candidate: ``{65535 - expert, key high word, sigmoid bits, 0}``.
_CAND_BYTES = 16
#: Local-expert capacity of the common build (up to 224 local experts); more use ``E``.
_SMALL_LOCAL_CAP = 224
#: ``1e-20`` as a PTX double literal (the renormalization epsilon of ``noaux_tc_op``).
_EPS_F64 = "0d" + struct.pack(">d", 1e-20).hex().upper()

assert TILE_SIZE == 128  # the sort computes ceil(count / TILE_SIZE) with a shift by 7
assert MAX_M <= CLUSTER_SIZE * _MAX_TOKENS_PER_CTA


# ---------------------------------------------------------------------------
# Integer helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _low_bits_mask(n, *, loc=None, ip=None):
    """``(1 << n) - 1`` for ``0 <= n < 32``."""
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [ptx.i32(n, loc=loc, ip=ip)],
            "{\n.reg .b32 t;\nshl.b32 t, 1, $1;\nsub.u32 $0, t, 1;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bit(n, *, loc=None, ip=None):
    """``1 << n`` for ``0 <= n < 32``."""
    return Int32(
        ptx.inline_asm(
            T.i32(), [ptx.i32(n, loc=loc, ip=ip)], "shl.b32 $0, 1, $1;", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _max_u32(a, b, *, loc=None, ip=None):
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [ptx.i32(a, loc=loc, ip=ip), ptx.i32(b, loc=loc, ip=ip)],
            "max.u32 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _warp_inclusive_sum(value, *, loc=None, ip=None):
    body = "".join(
        f"shfl.sync.up.b32 t|p, x, {off}, 0, 0xffffffff;\n@p add.s32 x, x, t;\n"
        for off in (1, 2, 4, 8, 16)
    )
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [ptx.i32(value, loc=loc, ip=ip)],
            "{\n.reg .b32 x, t;\n.reg .pred p;\nmov.b32 x, $1;\n" + body + "mov.b32 $0, x;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Top-k helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _score_bits(score, *, loc=None, ip=None):
    """Order-preserving bits of an FP32 score (the high word of ``noaux_tc``'s key)."""
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [ptx.f32(score, loc=loc, ip=ip)],
            "{\n.reg .b32 b, m;\nmov.b32 b, $1;\nshr.s32 m, b, 31;\n"
            "or.b32 m, m, 0x80000000;\nxor.b32 $0, b, m;\n}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _warp_16th_largest_u32(value, lane, *, loc=None, ip=None):
    """16th largest of the 32 lane values (unsigned), in every lane.

    Bitonic-sorts the lower 16 lanes descending and the upper 16 ascending; the pairwise maxima
    of lanes ``i`` and ``i + 16`` are then the 16 largest values, whose minimum is the answer.
    """
    body = "{\n.reg .b32 x, o, mx, mn, t, u;\n.reg .pred pj, pk, km;\nmov.b32 x, $1;\n"
    k = 2
    while k <= 16:
        j = k // 2
        while j >= 1:
            body += f"shfl.sync.bfly.b32 o, x, {j}, 0x1f, 0xffffffff;\n"
            body += "max.u32 mx, x, o;\nmin.u32 mn, x, o;\n"
            # Keep the maximum iff the lane is the lower one of its pair in a descending block.
            body += f"and.b32 t, $2, {j};\nsetp.eq.u32 pj, t, 0;\n"
            body += f"and.b32 u, $2, {k};\nsetp.eq.u32 pk, u, 0;\nxor.pred km, pj, pk;\n"
            body += "selp.b32 x, mn, mx, km;\n"
            j //= 2
        k *= 2
    body += "shfl.sync.bfly.b32 o, x, 16, 0x1f, 0xffffffff;\nmax.u32 x, x, o;\n"
    body += "redux.sync.min.u32 $0, x, 0xffffffff;\n}"
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [ptx.i32(value, loc=loc, ip=ip), ptx.i32(lane, loc=loc, ip=ip)],
            body,
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _count_ge_u32(values, threshold, *, loc=None, ip=None):
    """Number of ``values`` (unsigned) at or above ``threshold``."""
    n = len(values)
    body = "{\n.reg .pred p;\nmov.u32 $0, 0;\n"
    for i in range(n):
        body += f"setp.ge.u32 p, ${2 + i}, $1;\n@p add.u32 $0, $0, 1;\n"
    body += "}"
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [ptx.i32(threshold, loc=loc, ip=ip)] + [ptx.i32(v, loc=loc, ip=ip) for v in values],
            body,
            "=&r,r" + ",r" * n,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_candidates(addr, his, los, sigs, threshold, *, loc=None, ip=None):
    """Append the slots with ``hi >= threshold`` as 16-byte records at ``addr``; returns the end.

    Record: ``{65535 - expert, hi, sigmoid bits, 0}`` (``ld.shared.b64`` of its first eight
    bytes is ``noaux_tc``'s key).
    """
    n = len(his)
    body = "{\n.reg .pred p;\n.reg .b32 z;\nmov.b32 z, 0;\nmov.u32 $0, $1;\n"
    for i in range(n):
        h, lo, s = 3 + i, 3 + n + i, 3 + 2 * n + i
        body += (
            f"setp.ge.u32 p, ${h}, $2;\n"
            f"@p st.shared.v4.b32 [$0], {{${lo}, ${h}, ${s}, z}};\n"
            "@p add.u32 $0, $0, 16;\n"
        )
    body += "}"
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [ptx.i32(addr, loc=loc, ip=ip), ptx.i32(threshold, loc=loc, ip=ip)]
            + [ptx.i32(v, loc=loc, ip=ip) for v in his]
            + [ptx.i32(v, loc=loc, ip=ip) for v in los]
            + [ptx.f32(v, loc=loc, ip=ip) for v in sigs],
            body,
            "=&r,r,r" + ",r" * (2 * n) + ",f" * n,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _count_greater4(acc, addr, key, *, loc=None, ip=None):
    """``acc`` plus how many of the four candidate keys at ``addr`` (16-byte records) exceed
    ``key``."""
    body = "{\n.reg .b64 a, b, c, d;\n.reg .pred p, q, r, s;\n"
    body += "ld.shared.b64 a, [$2];\nld.shared.b64 b, [$2+16];\n"
    body += "ld.shared.b64 c, [$2+32];\nld.shared.b64 d, [$2+48];\n"
    body += "setp.gt.u64 p, a, $3;\nsetp.gt.u64 q, b, $3;\nsetp.gt.u64 r, c, $3;\nsetp.gt.u64 s, d, $3;\n"
    body += "mov.u32 $0, $1;\n@p add.u32 $0, $0, 1;\n@q add.u32 $0, $0, 1;\n"
    body += "@r add.u32 $0, $0, 1;\n@s add.u32 $0, $0, 1;\n}"
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [
                ptx.i32(acc, loc=loc, ip=ip),
                ptx.i32(addr, loc=loc, ip=ip),
                ptx.i64(key, loc=loc, ip=ip),
            ],
            body,
            "=&r,r,r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _argmax_scan4(
    best_hi, best_slot, best_sig, removed, his, sigs, slot0: int, *, loc=None, ip=None
):
    """Fold four slots (``slot0..slot0+3``) into a lane's running best ``(hi, slot, sigmoid)``:
    the largest ``hi`` whose bit in ``removed`` is clear, the first slot on ties (slots are in
    expert order, so the first is the smallest expert)."""
    body = "{\n.reg .pred p;\n.reg .b32 h, b;\nmov.u32 $0, $3;\nmov.u32 $1, $4;\nmov.b32 $2, $5;\n"
    for i in range(4):
        body += (
            f"bfe.u32 b, $6, {slot0 + i}, 1;\n"
            "setp.eq.u32 p, b, 0;\n"
            f"selp.b32 h, ${7 + i}, 0, p;\n"
            "setp.gt.u32 p, h, $0;\n"
            "@p mov.u32 $0, h;\n"
            f"@p mov.u32 $1, {slot0 + i};\n"
            f"@p mov.b32 $2, ${11 + i};\n"
        )
    body += "}"
    res = ptx.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.f32()]),
        [
            ptx.i32(best_hi, loc=loc, ip=ip),
            ptx.i32(best_slot, loc=loc, ip=ip),
            ptx.f32(best_sig, loc=loc, ip=ip),
            ptx.i32(removed, loc=loc, ip=ip),
        ]
        + [ptx.i32(v, loc=loc, ip=ip) for v in his]
        + [ptx.f32(v, loc=loc, ip=ip) for v in sigs],
        body,
        "=&r,=&r,=&f,r,r,f,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def _warp_sum_f32(value, *, loc=None, ip=None):
    """Butterfly warp sum, same order as ``cooperative_groups::reduce(plus)``."""
    body = "".join(
        f"shfl.sync.bfly.b32 t, x, {off}, 0x1f, 0xffffffff;\nadd.rn.f32 x, x, t;\n"
        for off in (16, 8, 4, 2, 1)
    )
    return Float32(
        ptx.inline_asm(
            T.f32(),
            [ptx.f32(value, loc=loc, ip=ip)],
            "{\n.reg .f32 x, t;\nmov.b32 x, $1;\n" + body + "mov.b32 $0, x;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _renormalize(score, total, scale_bits, *, loc=None, ip=None):
    """``float(double(score) * scale / (double(total) + 1e-20))`` like ``noaux_tc_op``."""
    return Float32(
        ptx.inline_asm(
            T.f32(),
            [
                ptx.f32(score, loc=loc, ip=ip),
                ptx.f32(total, loc=loc, ip=ip),
                ptx.i64(scale_bits, loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .f64 a, b, s;\n"
            "mov.b64 s, $3;\n"
            "cvt.f64.f32 a, $1;\n"
            "mul.rn.f64 a, a, s;\n"
            "cvt.f64.f32 b, $2;\n"
            f"add.rn.f64 b, b, {_EPS_F64};\n"
            "div.rn.f64 a, a, b;\n"
            "cvt.rn.f32.f64 $0, a;\n"
            "}",
            "=f,f,f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _unit_scale_fast_path(scale_bits, total, *, loc=None, ip=None):
    """1 if ``scale == 1.0`` and ``total >= 2^-12``: the FP32 division ``score / total`` then
    equals :func:`_renormalize` exactly (the epsilon is below half an ulp of ``total``)."""
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [ptx.i64(scale_bits, loc=loc, ip=ip), ptx.f32(total, loc=loc, ip=ip)],
            "{\n.reg .pred p, q;\n"
            "setp.eq.u64 p, $1, 0x3FF0000000000000;\n"
            "setp.ge.f32 q, $2, 0f39800000;\n"
            "and.pred p, p, q;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,l,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_rn_f32(a, b, *, loc=None, ip=None):
    return Float32(
        ptx.inline_asm(
            T.f32(),
            [ptx.f32(a, loc=loc, ip=ip), ptx.f32(b, loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class NoauxTcRouteSortKernel:
    """Top-16 + ``moe_sort`` of up to 256 tokens on one cluster of ``CLUSTER_SIZE`` CTAs (the
    host side guarantees at most 32 tokens per CTA).

    The warps of a CTA form teams of ``warps_per_token`` warps; a team routes one token at a
    time, lane ``g`` of the team (``g < 32 * warps_per_token``) owning experts
    ``j * 32 * warps_per_token + g``.

    Args:
        num_experts: Routed experts ``E`` (see :func:`supports_num_experts`).
        num_warps: Warps per CTA.
        warps_per_token: Warps routing one token (1 or 2).
        local_cap: Largest number of local experts served.
    """

    def __init__(
        self,
        num_experts: int,
        num_warps: int,
        warps_per_token: int,
        local_cap: int,
        use_pdl: bool = TRTLLM_ENABLE_PDL,
    ) -> None:
        if not supports_num_experts(num_experts):
            raise ValueError(f"num_experts={num_experts} must be a multiple of 128 up to 1024")
        if warps_per_token not in (1, 2) or num_warps % warps_per_token:
            raise ValueError(f"warps_per_token={warps_per_token} must be 1 or 2")
        if num_warps // warps_per_token > 15 and warps_per_token > 1:
            raise ValueError("at most 15 multi-warp teams (named barriers 1..15)")
        self.num_experts = num_experts
        self.num_ctas = CLUSTER_SIZE
        # Expert slots per lane of the exact fallback: E / 128 blocks x 4 consecutive experts.
        self.blocks = num_experts // 128
        self.fallback_slots = self.blocks * 4
        self.num_warps = num_warps
        self.threads = 32 * num_warps
        self.team_warps = warps_per_token
        self.team_lanes = 32 * warps_per_token
        self.teams = num_warps // warps_per_token
        self.slots = num_experts // self.team_lanes
        # Candidate records per warp (plus zero padding to a multiple of 4); more fall back to
        # the exact arg-max rounds.
        self.region_cap = 64 if warps_per_token == 1 else 32
        self.region_slots = self.region_cap + 4
        self.local_cap = local_cap
        self.use_pdl = use_pdl
        # Local experts handled per thread in the sort scan (contiguous).
        self.experts_per_thread = (local_cap + self.threads - 1) // self.threads
        self.max_tiles_per_expert = (MAX_M + TILE_SIZE - 1) // TILE_SIZE

    @cute.jit
    def __call__(
        self,
        m: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        scale_bits: Int64,
        stream: cuda.CUstream,
    ):
        teams = self.teams

        @cute.struct
        class SharedStorage:
            bias: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.num_experts], 16]
            mask: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.local_cap], 16]
            # Every CTA's masks, broadcast over distributed shared memory.
            masks: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_ctas * self.local_cap], 16
            ]
            base: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.local_cap], 16]
            ids: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _MAX_TOKENS_PER_CTA * TOP_K], 16
            ]
            cand: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_warps * self.region_slots * 4], 16
            ]
            top: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, teams * TOP_K * 2], 16]
            # Per team and warp: threshold, then candidate count.
            xchg: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, teams * 16], 16]
            warp_tot: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.num_warps], 16]

        self.shared_storage = SharedStorage

        kernel = self.kernel(
            m,
            logits_addr,
            bias_addr,
            ids_addr,
            vals_addr,
            e2p_addr,
            p2e_addr,
            tile_expert_addr,
            tile_limit_addr,
            total_addr,
            num_tiles_addr,
            local_offset,
            local_num,
            scale_bits,
        )
        kernel.launch(
            grid=(self.num_ctas, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.num_ctas, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        m: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        scale_bits: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        rank, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bias_smem = storage.bias.data_ptr().toint()
        mask_smem = storage.mask.data_ptr().toint()
        masks_smem = storage.masks.data_ptr().toint()
        base_smem = storage.base.data_ptr().toint()
        ids_smem = storage.ids.data_ptr().toint()
        team = warp // self.team_warps
        team_warp = warp - team * self.team_warps
        cand_smem = storage.cand.data_ptr().toint() + team * (
            self.team_warps * self.region_slots * _CAND_BYTES
        )
        top_smem = storage.top.data_ptr().toint() + team * (TOP_K * 8)
        xchg_smem = storage.xchg.data_ptr().toint() + team * 64
        warp_tot_smem = storage.warp_tot.data_ptr().toint()

        per_cta = (m + self.num_ctas - 1) // self.num_ctas
        t_begin = rank * per_cta
        t_end = t_begin + per_cta
        if t_end > m:
            t_end = m

        for j in cutlass.range(tidx, local_num, self.threads, unroll=1):
            ptx.st_shared_b32(mask_smem + j * 4, Int32(0))
        # The bias is a weight: stage it in shared memory before waiting on the previous grid.
        for e4 in cutlass.range(tidx, self.num_experts // 4, self.threads, unroll=1):
            v = ptx.ld_global_nc_v4_f32(bias_addr + Int64(e4 * 16))
            ptx.st_shared_v4_f32(bias_smem + e4 * 16, v[0], v[1], v[2], v[3])
        # Every CTA of the cluster must be running before a peer writes its shared memory.
        cute.arch.cluster_arrive_relaxed()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        cute.arch.barrier()

        for token in cutlass.range(t_begin + team, t_end, self.teams, unroll=1):
            self.topk_token(
                token,
                token - t_begin,
                team,
                team_warp,
                lane,
                logits_addr,
                bias_smem,
                ids_addr,
                vals_addr,
                scale_bits,
                cand_smem,
                top_smem,
                xchg_smem,
                ids_smem,
                mask_smem,
                local_offset,
                local_num,
            )
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        cute.arch.barrier()
        cute.arch.cluster_wait()
        # Broadcast this CTA's token masks into every CTA of the cluster.
        for j in cutlass.range(tidx, local_num, self.threads, unroll=1):
            word = ptx.ld_shared_b32(mask_smem + j * 4)
            dst = masks_smem + (rank * self.local_cap + j) * 4
            for c in cutlass.range_constexpr(self.num_ctas):
                ptx.st_dsmem_b32(dst, Int32(c), word)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        self.sort(
            tidx,
            warp,
            lane,
            rank,
            t_begin,
            t_end,
            mask_smem,
            masks_smem,
            base_smem,
            ids_smem,
            warp_tot_smem,
            e2p_addr,
            p2e_addr,
            tile_expert_addr,
            tile_limit_addr,
            total_addr,
            num_tiles_addr,
            local_offset,
            local_num,
        )

    def load_logits(self, row: Int64, g: Int32, stride: int, n: int):
        """Logits ``s * stride + g`` (``s < n``) of the token whose logits start at ``row``. A
        subclass may read them differently (the Kimi-K3 MoE front polls them)."""
        return [ptx.ld_global_f32(row + Int64((s * stride + g) * 4)) for s in range(n)]

    def load_logits4(self, row: Int64, lane: Int32, n: int):
        """Logits ``128 j + 4 lane .. + 3`` (``j < n``) of the token whose logits start at
        ``row``, concatenated (a plain Python loop, unrolled at trace time)."""
        logit = []
        for blk in range(n):
            logit.extend(ptx.ld_global_v4_f32(row + Int64((blk * 128 + lane * 4) * 4)))
        return logit

    @cute.jit
    def team_sync(self, team: Int32):
        if cutlass.const_expr(self.team_warps > 1):
            cute.arch.barrier(barrier_id=team + 1, number_of_threads=self.team_lanes)
        else:
            cute.arch.sync_warp()

    @cute.jit
    def topk_token(
        self,
        token: Int32,
        token_local: Int32,
        team: Int32,
        team_warp: Int32,
        lane: Int32,
        logits_addr: Int64,
        bias_smem: Int32,
        ids_addr: Int64,
        vals_addr: Int64,
        scale_bits: Int64,
        cand_smem: Int32,
        top_smem: Int32,
        xchg_smem: Int32,
        ids_smem: Int32,
        mask_smem: Int32,
        local_offset: Int32,
        local_num: Int32,
    ):
        """Top-16 of one token by one team; its first warp writes ids/weights and mask bits."""
        expert, weight = self.select_token(
            token,
            team,
            team_warp,
            lane,
            logits_addr,
            bias_smem,
            scale_bits,
            cand_smem,
            top_smem,
            xchg_smem,
        )
        if team_warp == 0:
            if lane < TOP_K:
                out = (token * TOP_K + lane) * 4
                ptx.st_global_b32(ids_addr + Int64(out), expert)
                ptx.st_global_f32(vals_addr + Int64(out), weight)
                ptx.st_shared_b32(ids_smem + (token_local * TOP_K + lane) * 4, expert)
                local = expert - local_offset
                if local >= 0 and local < local_num:
                    ptx.red_shared_or_b32(mask_smem + local * 4, _bit(token_local))

    @cute.jit
    def select_token(
        self,
        token: Int32,
        team: Int32,
        team_warp: Int32,
        lane: Int32,
        logits_addr: Int64,
        bias_smem: Int32,
        scale_bits: Int64,
        cand_smem: Int32,
        top_smem: Int32,
        xchg_smem: Int32,
    ):
        """Top-16 of one token by one team: returns ``(expert, weight)``, the ``lane``-th
        selection in lanes ``0..15`` of the team's first warp."""
        n_slots = self.slots
        stride = self.team_lanes
        g = team_warp * 32 + lane
        row = logits_addr + Int64(token) * Int64(self.num_experts * 4)
        logit = self.load_logits(row, g, stride, n_slots)
        bias = [ptx.ld_shared_f32(bias_smem + (s * stride + g) * 4) for s in range(n_slots)]
        sig = [
            cute.math.tanh(logit[s] * Float32(0.5)) * Float32(0.5) + Float32(0.5)
            for s in range(n_slots)
        ]
        his = [_score_bits(sig[s] + bias[s]) for s in range(n_slots)]
        lane_max = his[0]
        for s in cutlass.range_constexpr(1, n_slots):
            lane_max = _max_u32(lane_max, his[s])
        # A lower bound of the 16th largest score: the 16th largest lane maximum of a warp (16
        # of the warp's experts score at least that much); the team takes the largest.
        threshold = _warp_16th_largest_u32(lane_max, lane)
        if cutlass.const_expr(self.team_warps > 1):
            if lane == 0:
                ptx.st_shared_b32(xchg_smem + team_warp * 4, threshold)
            self.team_sync(team)
            for w in cutlass.range_constexpr(self.team_warps):
                threshold = _max_u32(threshold, ptx.ld_shared_b32(xchg_smem + w * 4))
        count = Int32(0)
        for c4 in cutlass.range_constexpr(0, n_slots, 4):
            count = count + _count_ge_u32(his[c4 : min(c4 + 4, n_slots)], threshold)
        inclusive = _warp_inclusive_sum(count)
        num_mine = cute.arch.shuffle_sync(inclusive, 31)
        cap = self.region_cap
        region_bytes = self.region_slots * _CAND_BYTES
        region = cand_smem + team_warp * region_bytes
        if num_mine <= cap:
            los = [Int32(65535 - s * stride) - g for s in range(n_slots)]
            dst = region + (inclusive - count) * _CAND_BYTES
            for c4 in cutlass.range_constexpr(0, n_slots, 4):
                q = slice(c4, min(c4 + 4, n_slots))
                dst = _store_candidates(dst, his[q], los[q], sig[q], threshold)
            # Zero keys after the last candidate, so the ranking reads groups of four.
            if lane < 4:
                ptx.st_shared_zero_v4(region + (num_mine + lane) * _CAND_BYTES)
        if cutlass.const_expr(self.team_warps > 1):
            if lane == 0:
                ptx.st_shared_b32(xchg_smem + 32 + team_warp * 4, num_mine)
        self.team_sync(team)
        counts = [num_mine]
        overflow = Int32(0)
        if num_mine > cap:
            overflow = Int32(1)
        if cutlass.const_expr(self.team_warps > 1):
            counts = [ptx.ld_shared_b32(xchg_smem + 32 + w * 4) for w in range(self.team_warps)]
            for w in cutlass.range_constexpr(self.team_warps):
                if counts[w] > cap:
                    overflow = Int32(1)

        expert = Int32(0)
        score = Float32(0.0)
        if overflow == 0:
            # Rank this warp's candidates against all candidates of the team.
            for c0 in cutlass.range_constexpr(cap // 32):
                c = lane + c0 * 32
                valid = Int32(0)
                if c < num_mine:
                    valid = Int32(1)
                key = ptx.ld_shared_b64_or_zero(valid, region + c * _CAND_BYTES)
                r = Int32(0)
                for w in cutlass.range_constexpr(self.team_warps):
                    other = cand_smem + w * region_bytes
                    for o in cutlass.range(0, counts[w], 4, unroll=1):
                        r = _count_greater4(r, other + o * _CAND_BYTES, key)
                if valid != 0 and r < TOP_K:
                    ptx.st_shared_b32(top_smem + r * 8, 65535 - (Int32(key) & 0xFFFF))
                    ptx.st_shared_b32(
                        top_smem + r * 8 + 4, ptx.ld_shared_b32(region + c * _CAND_BYTES + 8)
                    )
            self.team_sync(team)
            if team_warp == 0 and lane < TOP_K:
                expert = ptx.ld_shared_b32(top_smem + lane * 8)
                score = ptx.ld_shared_f32(top_smem + lane * 8 + 4)
        else:
            if team_warp == 0:
                expert, score = self.topk_rounds(row, bias_smem, lane)

        weight = Float32(0.0)
        if team_warp == 0:
            total = _warp_sum_f32(score)
            # Exact FP32 fast path of noaux_tc's FP64 expression.
            if _unit_scale_fast_path(scale_bits, total) != 0:
                weight = _div_rn_f32(score, total)
            else:
                weight = _renormalize(score, total, scale_bits)
        return expert, weight

    @cute.jit
    def topk_rounds(self, row: Int64, bias_smem: Int32, lane: Int32):
        """Exact fallback by one warp: 16 arg-max rounds over all ``E`` keys of the token (lane
        ``l`` owns experts ``128 j + 4 l + i``); lane ``k`` gets the ``k``-th expert."""
        logit = self.load_logits4(row, lane, self.blocks)
        sig = [
            cute.math.tanh(logit[s] * Float32(0.5)) * Float32(0.5) + Float32(0.5)
            for s in range(self.fallback_slots)
        ]
        his = []
        for blk in cutlass.range_constexpr(self.blocks):
            b = ptx.ld_shared_v4_f32(bias_smem + (blk * 128 + lane * 4) * 4)
            for i in cutlass.range_constexpr(4):
                his.append(_score_bits(sig[blk * 4 + i] + b[i]))
        removed = Int32(0)
        expert = Int32(0)
        score = Float32(0.0)
        for r in cutlass.range_constexpr(TOP_K):
            best_hi = Int32(0)
            best_slot = Int32(0)
            best_sig = Float32(0.0)
            for blk in cutlass.range_constexpr(self.blocks):
                q = slice(blk * 4, blk * 4 + 4)
                best_hi, best_slot, best_sig = _argmax_scan4(
                    best_hi, best_slot, best_sig, removed, his[q], sig[q], blk * 4
                )
            top_hi = cute.arch.warp_redux_sync(best_hi, "umax")
            # Slot s of this lane is expert 128 (s / 4) + 4 lane + s % 4.
            cand = Int32(0xFFFF)
            if best_hi == top_hi:
                cand = (best_slot >> 2) * 128 + lane * 4 + (best_slot & 3)
            winner = cute.arch.warp_redux_sync(cand, "umin")
            owner = (winner & 127) >> 2
            if lane == owner:
                removed = removed | _bit(best_slot)
            win_sig = cute.arch.shuffle_sync(best_sig, owner)
            if lane == r:
                expert = winner
                score = win_sig
        return expert, score

    @cute.jit
    def sort(
        self,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        rank: Int32,
        t_begin: Int32,
        t_end: Int32,
        mask_smem: Int32,
        masks_smem: Int32,
        base_smem: Int32,
        ids_smem: Int32,
        warp_tot_smem: Int32,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
    ):
        """``moe_sort`` metadata from the token masks of all CTAs of the cluster."""
        ept = self.experts_per_thread
        j0 = tidx * ept
        counts = []
        before = []
        for q in cutlass.range_constexpr(ept):
            j = j0 + q
            jj = j
            if j >= local_num:
                jj = Int32(0)
            words = []
            for c in cutlass.range_constexpr(self.num_ctas):
                words.append(ptx.ld_shared_b32(masks_smem + (c * self.local_cap + jj) * 4))
            cnt = Int32(0)
            pre = Int32(0)
            for c in cutlass.range_constexpr(self.num_ctas):
                n = cute.arch.popc(words[c])
                cnt = cnt + n
                if c < rank:
                    pre = pre + n
            if j >= local_num:
                cnt = Int32(0)
            counts.append(cnt)
            before.append(pre)
        tiles = [(counts[q] + (TILE_SIZE - 1)) >> 7 for q in range(ept)]
        thread_tiles = tiles[0]
        for q in cutlass.range_constexpr(1, ept):
            thread_tiles = thread_tiles + tiles[q]
        inclusive = _warp_inclusive_sum(thread_tiles)
        if lane == 31:
            ptx.st_shared_b32(warp_tot_smem + warp * 4, inclusive)
        cute.arch.barrier()
        tile = inclusive - thread_tiles
        num_tiles = Int32(0)
        for w in cutlass.range_constexpr(self.num_warps):
            cnt_w = ptx.ld_shared_b32(warp_tot_smem + w * 4)
            if w < warp:
                tile = tile + cnt_w
            num_tiles = num_tiles + cnt_w
        for q in cutlass.range_constexpr(ept):
            j = j0 + q
            if tiles[q] != 0:
                ptx.st_shared_b32(base_smem + j * 4, tile * TILE_SIZE + before[q])
                if rank == 0:
                    for t in cutlass.range_constexpr(self.max_tiles_per_expert):
                        if t < tiles[q]:
                            limit = tile * TILE_SIZE + counts[q]
                            if limit > (tile + t + 1) * TILE_SIZE:
                                limit = (tile + t + 1) * TILE_SIZE
                            ptx.st_global_b32(tile_expert_addr + Int64((tile + t) * 4), j)
                            ptx.st_global_b32(tile_limit_addr + Int64((tile + t) * 4), limit)
            tile = tile + tiles[q]
        if rank == 0 and tidx == 0:
            ptx.st_global_b32(total_addr, num_tiles * TILE_SIZE)
            ptx.st_global_b32(num_tiles_addr, num_tiles)
        cute.arch.barrier()

        n_entries = (t_end - t_begin) * TOP_K
        for e in cutlass.range(tidx, n_entries, self.threads, unroll=1):
            token_local = e >> 4
            expanded = t_begin * TOP_K + e
            local = ptx.ld_shared_b32(ids_smem + e * 4) - local_offset
            perm = Int32(-1)
            if local >= 0 and local < local_num:
                word = ptx.ld_shared_b32(mask_smem + local * 4)
                perm = ptx.ld_shared_b32(base_smem + local * 4) + cute.arch.popc(
                    word & _low_bits_mask(token_local)
                )
                ptx.st_global_b32(p2e_addr + Int64(perm * 4), expanded)
            ptx.st_global_b32(e2p_addr + Int64(expanded * 4), perm)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _device_capability(device_index: int) -> tuple:
    return torch.cuda.get_device_capability(device_index)


def _is_supported_device(device_index: int) -> bool:
    """SM100 / SM103; SM107 only with a Rubin-capable CuTe DSL."""
    from ....cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    capability = _device_capability(device_index)
    return capability in ((10, 0), (10, 3)) or (
        capability == (10, 7) and IS_CUTLASS_DSL_RUBIN_AVAILABLE
    )


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def supports_device(device: torch.device) -> bool:
    """Whether the kernel runs on ``device`` (see :func:`_is_supported_device`)."""
    return device.type == "cuda" and _is_supported_device(_device_index(device))


def supports_num_experts(num_experts: int) -> bool:
    """Whether the kernel routes over ``num_experts`` experts: a multiple of 128 up to 1024."""
    return num_experts % 128 == 0 and 128 <= num_experts <= MAX_EXPERTS


def serves(
    num_tokens: int,
    num_experts: int,
    local_expert_offset: int,
    local_num_experts: int,
    device: torch.device,
) -> bool:
    """Whether :func:`run_route_sort` serves ``num_tokens`` tokens of ``num_experts``-expert
    routing for experts ``[local_expert_offset, local_expert_offset + local_num_experts)`` on
    ``device``; :func:`can_run` adds the tensor checks."""
    return (
        1 <= num_tokens <= MAX_M
        and supports_num_experts(num_experts)
        and 1 <= local_num_experts <= num_experts
        and 0 <= local_expert_offset <= num_experts - local_num_experts
        and supports_device(device)
    )


def _config(m: int) -> tuple[int, int]:
    """``(warps_per_token, num_warps)`` for ``m`` tokens on the eight CTAs (at most 32 tokens per
    CTA): 8-warp CTAs with two-warp teams up to 32 tokens, one-warp teams up to 64; 16-warp CTAs
    up to 16 tokens per CTA, 32-warp CTAs above (each CTA routes its tokens in one round)."""
    if m <= 32:
        return 2, 8
    if m <= 64:
        return 1, 8
    if m <= 16 * CLUSTER_SIZE:
        return 1, 16
    return 1, 32


@functools.cache
def _compiled_kernel(
    num_experts: int,
    num_warps: int,
    warps_per_token: int,
    local_cap: int,
    use_pdl: bool = TRTLLM_ENABLE_PDL,
) -> object:
    kernel = NoauxTcRouteSortKernel(
        num_experts, num_warps, warps_per_token, local_cap, use_pdl=use_pdl
    )
    dummy = Int64(1 << 20)
    return cute.compile(
        kernel,
        Int32(CLUSTER_SIZE),
        *([dummy] * 10),
        Int32(0),
        Int32(num_experts),
        Int64(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def sort_buffer_sizes(num_tokens: int, local_num_experts: int) -> tuple[int, int]:
    """``(max_num_tiles, max_num_permuted_tokens)`` exactly as ``moe_sort`` allocates them for
    128-row tiles."""
    expanded = num_tokens * TOP_K
    filled = min(local_num_experts, expanded)
    max_tiles = filled + (expanded - filled) // TILE_SIZE
    return max_tiles, max_tiles * TILE_SIZE


def sort_outputs(
    num_tokens: int, local_num_experts: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Fresh ``moe_sort`` outputs for ``num_tokens`` tokens, with ``moe_sort``'s shapes and dtypes.

    ``(tile_idx_to_expert_idx [T], tile_idx_to_mn_limit [T], expanded_idx_to_permuted_idx
    [M, 16], permuted_idx_to_expanded_idx [T * 128], total_num_padded_tokens [1],
    num_non_exiting_tiles [1])``, all ``int32``, ``T`` from :func:`sort_buffer_sizes`.
    """
    max_tiles, max_permuted = sort_buffer_sizes(num_tokens, local_num_experts)
    i32 = dict(dtype=torch.int32, device=device)
    return (
        torch.empty((max_tiles,), **i32),
        torch.empty((max_tiles,), **i32),
        torch.empty((num_tokens, TOP_K), **i32),
        torch.empty((max_permuted,), **i32),
        torch.empty((1,), **i32),
        torch.empty((1,), **i32),
    )


def precompile(num_experts: int, use_pdl: bool = TRTLLM_ENABLE_PDL) -> int:
    """Compile every (``num_warps``, ``warps_per_token``, ``local_cap``) variant ``_config`` can
    land on, ahead of serving, so no served step compiles them; returns the compiles done (0
    when the current device is one the kernel does not serve).

    ``_config`` token bands: 8-warp CTAs with two-warp teams up to 32 tokens, one-warp teams
    up to 64, 16-warp CTAs above; ``local_cap`` is ``_SMALL_LOCAL_CAP`` up to that many local
    experts, ``num_experts`` above.
    """
    if not supports_device(torch.cuda.current_device()):
        return 0
    compiled = 0
    for warps_per_token, num_warps in ((2, 8), (1, 8), (1, 16)):
        local_caps = {num_experts}
        if num_experts > _SMALL_LOCAL_CAP:
            local_caps.add(_SMALL_LOCAL_CAP)
        for local_cap in sorted(local_caps):
            _compiled_kernel(num_experts, num_warps, warps_per_token, local_cap, use_pdl=use_pdl)
            compiled += 1
    return compiled


def can_run(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    local_expert_offset: int,
    local_num_experts: int,
) -> bool:
    """Whether :func:`run_route_sort` serves these inputs (non-raising check).

    ``logits``: contiguous FP32 ``[M, E]`` on a CUDA device where :func:`serves` holds;
    ``e_score_correction_bias``: contiguous FP32 ``[E]`` on the same device; both 16-byte aligned.
    """
    return (
        logits.dim() == 2
        and logits.is_cuda
        and serves(
            logits.shape[0],
            logits.shape[1],
            local_expert_offset,
            local_num_experts,
            logits.device,
        )
        and logits.dtype == torch.float32
        and logits.is_contiguous()
        and e_score_correction_bias.dtype == torch.float32
        and e_score_correction_bias.numel() == logits.shape[1]
        and e_score_correction_bias.is_contiguous()
        and e_score_correction_bias.device == logits.device
        and logits.data_ptr() % 16 == 0
        and e_score_correction_bias.data_ptr() % 16 == 0
    )


def run_route_sort(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    local_expert_offset: int,
    local_num_experts: int,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sort_buffers: tuple[torch.Tensor, ...],
) -> None:
    """Launch the routing on the current stream (inputs as :func:`can_run` accepts them).

    Args:
        logits: FP32 ``[M, E]`` router logits.
        e_score_correction_bias: FP32 ``[E]``.
        routed_scaling_factor: Multiplier of the renormalized weights.
        local_expert_offset: First expert of this rank.
        local_num_experts: Experts of this rank.
        topk_ids: Output ``int32 [M, 16]``, bit-identical to ``noaux_tc_op``'s ids.
        topk_weights: Output ``fp32 [M, 16]``, bit-identical to ``noaux_tc_op``'s weights.
        sort_buffers: ``moe_sort`` outputs for this rank and 128-row tiles, as
            :func:`sort_outputs` allocates them. Only valid entries are written (as in
            ``moe_sort``); the rows of an expert are in ascending token order.
    """
    m, num_experts = logits.shape
    warps_per_token, num_warps = _config(m)
    local_cap = _SMALL_LOCAL_CAP if local_num_experts <= _SMALL_LOCAL_CAP else num_experts
    compiled = _compiled_kernel(
        num_experts, num_warps, warps_per_token, local_cap, use_pdl=TRTLLM_ENABLE_PDL
    )
    tile_expert, tile_limit, e2p, p2e, total, num_tiles = (t.data_ptr() for t in sort_buffers)
    scale_bits = struct.unpack("<q", struct.pack("<d", float(routed_scaling_factor)))[0]
    compiled(
        Int32(m),
        Int64(logits.data_ptr()),
        Int64(e_score_correction_bias.data_ptr()),
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
        Int64(e2p),
        Int64(p2e),
        Int64(tile_expert),
        Int64(tile_limit),
        Int64(total),
        Int64(num_tiles),
        Int32(local_expert_offset),
        Int32(local_num_experts),
        Int64(scale_bits),
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )
