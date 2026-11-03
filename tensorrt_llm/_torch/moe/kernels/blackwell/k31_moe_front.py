# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Kimi K3.1 sharded MoE front for decode, one CTM (prims/cute) kernel per MoE layer, M <= 8 tokens:
#
#   head   = x @ [latent-down slice; router slice]^T       (fp32; this rank's 3584/W latent columns, 896/W logits)
#   gather = every rank's head slice, through the multicast mapping of the head workspace (Lamport buffers)
#   route  = top-16 of sigmoid(logits) + bias per token; MXFP8 latent + UE8M0 scales per token
#   shared = SiTU(bf16(x @ gate^T)) * SiTU_lin(bf16(x @ up^T))  (the shared experts' gate_up and activation)
#
# One weight W [N, K = 7168] bf16 in 128-row tiles: the head rows (latent slice, then router slice, then zero rows
# up to a whole tile), then the shared gate_up rows re-ordered so that every 32-row slice holds 16 gate rows and
# the 16 up rows of the same columns (`front_weight`).
#
# Geometry (the long-K CTM decode GEMV): clusters of SPLIT = 8 CTAs; a GEMV cluster owns one 128-row tile per round
# (at most three rounds), rank r streaming k-tiles r, r + 8, ... through a RING-stage ring filled, with every later
# k-tile of every round prefetched into L2, before any wait; x's k-tiles of the rank stay resident. tcgen05 MMA
# M 128, N 8, K 16 into a TMEM accumulator per round. Split-K: rank w < 4 owns rows [32 w, 32 w + 32) of the tile;
# the other ranks write their fp32 partials of those rows into slot [rank] of the owner's round mailbox by st.async,
# which completes the mailbox barrier by bytes; the owner, which expected the bytes at setup, spins on test_wait (a
# warp suspended in try_wait is woken late by remote completions) and adds the 8 partials in rank order. Then, by
# tile kind:
#   head latent rows -> bf16 pairs (cvt.rn.bf16x2.f32, -0.0 halves -> +0.0), 16-byte vectors pushed into slot
#       [buffer][token][rank] of every rank's head buffer (the head all-gather's packing: k31_route_quant);
#   head router rows skip the owner: every rank of the cluster pushes its own fp32 partial (-0.0 -> +0.0) into
#       [buffer][token][rank][q = cluster rank] of the partials region after the buffers, and the route CTAs add the
#       8 partials in rank order from +0.0 (the owner's sum, bit for bit); no fence after the pushes, the readers poll
#       the words themselves;
#   shared rows -> gate and up of one column in lanes l and l + 16: bf16(SiTU(bf16 gate) * SiTU_lin(bf16 up)).
# Role clusters, the first two of the grid (resident before any GEMV cluster): CTA t < M routes token t, CTA M + t
# quantizes it from the gathered slots (the GEMV epilogues push). Every CTA that reads the
# buffer index (the 2 M role CTAs and every GEMV CTA) signs in on flags[3] right after the read; thread 0 of token 0's
# quantization CTA waits for all of them and flips the index for the next call, long before the data it polls arrives,
# so no count sits on the role CTAs' output path. With `publish`, per-token ready words as route_quant_ag's.
#
# Warps of a GEMV CTA: 0 weight TMA (+ early dependent trigger), 1 grid-dependency wait, head workspace flags,
# activation TMA, 2 TMEM allocation + MMA, 3 idle, 4-7 epilogue, 8-13 exit after setup. Role CTAs: all 14 warps
# (the quantization CTA holds one 8-element vector per thread).
# =============================================================================
"""Kimi K3.1 MoE front at decode size (head GEMV + all-gather + routing + MXFP8, shared gate_up) as one kernel."""

from __future__ import annotations

import functools
import os
import threading
from typing import Dict, Optional, Tuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass import dsl_user_op
from cutlass.experimental import primitives as prims

from . import k31_route_quant as _ag
from . import k31_route_quant as _rq

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN_SIZE = 3584  # the routed latent
MODEL_HIDDEN = 7168  # K
SF_VEC_SIZE = 32
M_MAX = 8  # rows of an 8-slot launch
M_CHUNK = 32  # rows of one launch at most (16 MMA token slots, two token halves)
M_WS = _ag.M_MAX  # token slots of the head workspace layout
EMPTY_WORD = _ag.EMPTY_WORD
# The runner sort (do_sort): smem words ([0, 256) token masks, [256, 512) tiles, [512, 528) warp totals), local experts.
SORT_SMEM_WORDS = 528
SORT_MAX_LOCAL = 256
# Rows the runner sort covers (token masks are 32-bit) and the entries per thread of a 448-thread CTA.
SORT_MAX_TOKENS = 32
SORT_ENTRIES_PER_THREAD = -(-SORT_MAX_TOKENS * TOP_K // (HIDDEN_SIZE // 8))
MAX_ROWS = SORT_MAX_TOKENS

CTA_M = 128
HALF_M = 64  # a head half-tile's rows (one round, e.g. TP16: every head k-tile of the CTA on chip before the wait)
MAX_HALF_TILES = 8  # k-tiles a head half-tile CTA can hold (its full-barrier count)
MMA_N = 8  # token slots (MMA N) of the smallest launch; 16-row launches use 16
CTA_K = 128
MMA_K = 16
TMA_K_BOX = 64
TMA_COPY_ITERS = CTA_K // TMA_K_BOX
K_BLOCKS_PER_HALF = TMA_K_BOX // MMA_K
SPLIT = 8
assert SPLIT == _ag.FRONT_SPLIT  # the router partials region is sized by the all-gather's layout
OWNERS = 4  # ranks owning 32-row slices of a tile (epilogue warps 4-7 hold TMEM lanes 32 w ..)
ROWS_PER_WARP = 32
MAX_ROUNDS = 3
ROLE_CLUSTERS = 2  # role clusters of an 8-slot launch


def role_clusters_of(slots: int) -> int:
    """Clusters of the role CTAs of a ``slots``-row launch: a route CTA and a quantization CTA per row, or above 16 rows
    (two token halves) one CTA per row that routes and then quantizes it."""
    return (slots if slots > 16 else 2 * slots) // SPLIT


def mma_n_of(num_tokens: int) -> int:
    """MMA token slots of a launch on ``num_tokens <= M_CHUNK`` rows."""
    return 8 if num_tokens <= 8 else 16


THREADS = HIDDEN_SIZE // 8  # 448: one 8-element latent vector per thread in a quantization CTA
ELEM_BYTES = 2
EVICT_FIRST = 0x12F0000000000000

LEADING = 16
STRIDE = 8 * TMA_K_BOX * ELEM_BYTES
A_HALF_ELEMS = CTA_M * TMA_K_BOX
STEP = (MMA_K * ELEM_BYTES) >> 4
A_BOX = A_HALF_ELEMS >> 3
STAGE_A = (CTA_M * CTA_K * ELEM_BYTES) >> 4
STAGE_AH = (HALF_M * CTA_K * ELEM_BYTES) >> 4  # a head half-tile's k-tile (16 KB)
A_BOX_H = (HALF_M * TMA_K_BOX) >> 3  # its 64-element K box (8 KB)

io_dtype = cutlass.BFloat16


def router_row_base(world: int) -> int:
    """First router row of a rank's head: its latent_down rows rounded up to whole 32-row epilogue slices (a slice is
    either latent or router rows); the rows in between are zero."""
    return -(-(HIDDEN_SIZE // world) // ROWS_PER_WARP) * ROWS_PER_WARP


def head_rows(world: int) -> int:
    return router_row_base(world) + NUM_EXPERTS // world


def head_tiles(world: int) -> int:
    return (head_rows(world) + CTA_M - 1) // CTA_M


def geometry(
    world: int, shared_cols: int, max_clusters: int, mma_n: int = MMA_N
) -> tuple[int, int, int]:
    """(head tiles, shared tiles, GEMV clusters) for this TP world and shared activation width.

    ``max_clusters``: how many clusters of SPLIT CTAs (one CTA per SM) the GPU holds at once. The role clusters and
    every GEMV cluster must fit together: the role clusters stay until the routing is done, so a GEMV cluster that
    does not fit waits for them (a 152-SM GB200 holds 15, not the 16 its 8 GPCs suggest).
    """
    ht = head_tiles(world)
    st = shared_cols // 64
    tiles = ht + st
    clusters = min(tiles, max(1, max_clusters - role_clusters_of(mma_n)))
    return ht, st, clusters


def half_geometry(
    world: int,
    shared_cols: int,
    max_clusters: int,
    k_in: int,
    ring: int,
    mma_n: int = MMA_N,
    halves: int = 1,
):
    """(head half-tiles, shared tiles, GEMV clusters) when the head fits as 64-row half-tiles in one round next to the
    shared tiles, each head CTA holding all its k-tiles in shared memory (the ring's space); else None."""
    hh = (head_rows(world) + HALF_M - 1) // HALF_M
    st = shared_cols // 64
    my_tiles = k_in // CTA_K // SPLIT
    fits = (
        (hh + st) * halves <= max_clusters - role_clusters_of(mma_n * halves)
        and my_tiles * HALF_M * CTA_K + HALF_M * TMA_K_BOX <= ring * CTA_M * CTA_K
        and my_tiles <= MAX_HALF_TILES
    )
    return (hh, st, hh + st) if fits else None


def _weight_rows(world: int, n_head_tiles: int, n_tiles: int, head_half: bool) -> int:
    """Rows of the front weight: the head padded to 128-row tiles, then the shared tiles."""
    if not head_half:
        return n_tiles * CTA_M
    return (head_tiles(world) + n_tiles - n_head_tiles) * CTA_M


def _half_plan(
    head_half: bool, world: int, n_head_tiles: int, ring: int, my_tiles: int
) -> tuple[int, int]:
    """(weight-row shift of the shared tiles, weight full/empty barriers) for the kernel's tile mode."""
    if not head_half:
        return 0, ring
    return (head_tiles(world) - n_head_tiles) * CTA_M, max(ring, my_tiles)


def supports(
    world: int, shared_cols: int, max_clusters: int, k_in: int, ring: int, mma_n: int = 16
) -> bool:
    ht, st, clusters = geometry(world, shared_cols, max_clusters, mma_n)
    k_tiles = k_in // CTA_K
    return (
        world in (4, 8, 16, 32)
        and k_in % (CTA_K * SPLIT) == 0
        and shared_cols % 64 == 0
        and ht + st <= MAX_ROUNDS * clusters
        and 1 <= ring <= k_tiles // SPLIT
        and ring * CTA_M * CTA_K * ELEM_BYTES + (k_tiles // SPLIT) * mma_n * CTA_K * ELEM_BYTES
        <= 176 * 1024
    )


def _situ(gate, up, gate_cap: float, linear_cap: float):
    """SiTU(gate) * SiTU_lin(up) in fp32 (the fused MoE's device functions)."""
    log2e = 1.4426950408889634

    def tanh_f32(v):
        e = cute.math.exp2(cute.math.abs(v) * cutlass.Float32(-2.0 * log2e), fastmath=True)
        t = (cutlass.Float32(1.0) - e) * cute.arch.rcp_approx(cutlass.Float32(1.0) + e)
        return cutlass.select_(v < cutlass.Float32(0.0), -t, t)

    sig = cute.arch.rcp_approx(
        cutlass.Float32(1.0) + cute.math.exp2(gate * cutlass.Float32(-log2e), fastmath=True)
    )
    g = cutlass.Float32(gate_cap) * tanh_f32(gate * cutlass.Float32(1.0 / gate_cap)) * sig
    u = cutlass.Float32(linear_cap) * tanh_f32(up * cutlass.Float32(1.0 / linear_cap))
    return g * u


@dsl_user_op
def _mapa_u32(smem_ptr, peer, *, loc=None, ip=None):
    """The shared::cluster address of this CTA's shared-memory location in cluster CTA ``peer``."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [smem_ptr.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_async_v4_f32(dst, a, b, c, d, mbar, *, loc=None, ip=None):
    """st.async of four fp32 to a shared::cluster address, completing ``mbar`` (shared::cluster) by 16 bytes."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip), cutlass.Float32(c).ir_value(loc=loc, ip=ip),
         cutlass.Float32(d).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [$0], {$1, $2, $3, $4}, [$5];", "r,f,f,f,f,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _st_async_b32(dst, value, mbar, *, loc=None, ip=None):
    """st.async of one 32-bit word to a shared::cluster address, completing ``mbar`` (shared::cluster) by 4 bytes."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Int32(value).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [$0], $1, [$2];", "r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@cute.jit
def _poll_logit_parts(arr, idx, stride: cutlass.Constexpr[int]):
    """Spin (no back-off) until none of the SPLIT 4-word vectors at idx + q * stride is empty, every sweep loading all
    of them at once. Returns the 4 logits, each the fp32 sum of its partials in rank order from +0.0 (the owner's
    order; a -0.0 partial pushed as +0.0 cannot change a sum that starts at +0.0), -0.0 as +0.0 (the owner's push)."""
    empty = cutlass.Int32(EMPTY_WORD)
    s0 = empty
    s1 = empty
    s2 = empty
    s3 = empty
    pending = cutlass.Boolean(True)
    while pending:
        vecs = []
        for q in cutlass.range_constexpr(SPLIT):
            vecs.append(
                arr.load(
                    idx=idx + cutlass.Int32(q * stride),
                    vector_size=4,
                    alignment=16,
                    is_volatile=True,
                )
            )
        missing = cutlass.Boolean(False)
        totals = [cutlass.Float32(0.0)] * 4
        for q in cutlass.range_constexpr(SPLIT):
            for j in cutlass.range_constexpr(4):
                word = cutlass.Int32(vecs[q][j])
                missing = missing | (word == empty)
                totals[j] = totals[j] + word.bitcast(cutlass.Float32)
        s0 = _ag._sanitize_f32(totals[0].bitcast(cutlass.Int32))
        s1 = _ag._sanitize_f32(totals[1].bitcast(cutlass.Int32))
        s2 = _ag._sanitize_f32(totals[2].bitcast(cutlass.Int32))
        s3 = _ag._sanitize_f32(totals[3].bitcast(cutlass.Int32))
        pending = missing
    return s0, s1, s2, s3


def _bf16_rn(v):
    """fp32 -> bf16 precision (round to nearest even), kept as fp32, in the integer domain."""
    u = v.bitcast(cutlass.Int32)
    u = u + (((u >> 16) & 1) + 0x7FFF)
    u = (u >> 16) << 16
    return u.bitcast(cutlass.Float32)


@cute.kernel
def k31_moe_front_kernel(
    tma_desc_w: cutlass.GridConstant[cuda.TensorMap],  # W [N, K] bf16, 5-D, one call per k-tile
    tma_desc_w64: cutlass.GridConstant[
        cuda.TensorMap
    ],  # the same W with a 64-row box (head half-tiles)
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],  # x [M, K] bf16, box 64 x 8
    bias: cutlass.Array,  # fp32 [896]
    buf_uc: cutlass.Array,  # int32 words of this rank's head buffers
    buf_mc: cutlass.Array,  # int32 words of their multicast mapping
    flags: cutlass.Array,  # int32 [0] buffer of this call, [1] unused (0), [2] epoch, [3] CTAs that read [0]
    ready: cutlass.Array,  # int32 [16] per-token ready words (publish)
    topk_ids: cutlass.Array,  # int32 [M * 16]
    topk_weight_bits: cutlass.Array,  # int16 view of bf16 [M * 16]
    quant_words: cutlass.Array,  # int32 view of e4m3 [M, 3584]
    scales: cutlass.Array,  # uint8 [M * 112]
    shared_bits: cutlass.Array,  # int16 view of bf16 [M, shared_cols]
    sort_ids: cutlass.Array,  # int32 [sort_tokens * 16] every row's top-16 (this call's rows are the last ones)
    sort_e2p: cutlass.Array,  # int32 [sort_tokens * 16] expanded_idx_to_permuted_idx (do_sort)
    sort_p2t: cutlass.Array,  # int32 permuted_idx_to_token_idx
    sort_total: cutlass.Array,  # int32 [1] total_num_padded_tokens
    sort_tile_expert: cutlass.Array,  # int32 cta_idx_xy_to_batch_idx
    sort_tile_limit: cutlass.Array,  # int32 cta_idx_xy_to_mn_limit
    sort_num_tiles: cutlass.Array,  # int32 [1] num_non_exiting_ctas
    num_tokens: cutlass.Int32,
    tp_rank: cutlass.Int32,
    routed_scaling_factor: cutlass.Float64,
    local_offset: cutlass.Int32,
    local_num: cutlass.Int32,
    tile_log2: cutlass.Int32,
    sort_tokens: cutlass.Int32,
    world: cutlass.Constexpr[int],
    shared_cols: cutlass.Constexpr[int],
    n_head_tiles: cutlass.Constexpr[int],
    n_tiles: cutlass.Constexpr[int],
    gemv_clusters: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    gate_cap: cutlass.Constexpr[float],
    linear_cap: cutlass.Constexpr[float],
    publish: cutlass.Constexpr[bool],
    head_half: cutlass.Constexpr[bool],
    do_sort: cutlass.Constexpr[bool],
    mma_n: cutlass.Constexpr[int],
    halves: cutlass.Constexpr[int],
):
    WL = HIDDEN_SIZE // world
    # The first router row: WL unless WL is not a multiple of 32 rows (32 ranks: 112 -> 128).
    RB = router_row_base(world)
    WE = NUM_EXPERTS // world
    LV = WL // 8
    EV = WE // 4
    SLOT = (LV + EV) * 4
    BUF = M_WS * world * SLOT
    PBASE = (
        _ag.BUFFERS * BUF
    )  # the router logits' split-K partials, after the buffers: [buffer][token][rank][q][WE]
    k_tiles = k_in // CTA_K
    my_tiles = k_tiles // SPLIT
    rounds_max = (n_tiles + gemv_clusters - 1) // gemv_clusters
    # Token slots of this launch (the MMA N): 8 or 16; the route + quantization CTAs of 2 * mma_n roles come first.
    role_clusters = role_clusters_of(mma_n * halves)
    tmem_cols = 32 if rounds_max * mma_n <= 32 else 64
    b_half_elems = mma_n * TMA_K_BOX
    b_box = b_half_elems >> 3
    stage_b = (mma_n * CTA_K * ELEM_BYTES) >> 4
    # head_half (one round): clusters < n_head_tiles each own a 64-row head half-tile and hold all its k-tiles; the
    # shared tiles' weight rows start after the 128-row-padded head, hence the row shift for tiles >= n_head_tiles.
    row_shift, nbar = _half_plan(head_half, world, n_head_tiles, ring, my_tiles)

    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    lane = tx % 32
    crank = cute.arch.block_idx_in_cluster()
    # The role clusters come first, so they are resident before any GEMV cluster (they spin; the GEMV CTAs never
    # wait on them).
    role_or_gemv = bx // cutlass.Int32(SPLIT)
    is_gemv = role_or_gemv >= cutlass.Int32(role_clusters)
    cluster = role_or_gemv - cutlass.Int32(role_clusters)
    # halves = 2: GEMV clusters [0, gemv_clusters) serve rows [0, mma_n), the next gemv_clusters rows [mma_n, 2 mma_n)
    # (the same weight tiles, the second set reading them from L2).
    tok0 = cutlass.Int32(0)
    if cutlass.const_expr(halves > 1):
        tok0 = (cluster // cutlass.Int32(gemv_clusters)) * cutlass.Int32(mma_n)
        cluster = cluster % cutlass.Int32(gemv_clusters)
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_w64 = tma_desc_w64.get_ptr()
    # A GEMV CTA of a head half-tile (head_half: one round, the head in 64-row tiles on the first clusters).
    head_cta = cutlass.Boolean(False)
    stream_cta = cutlass.Boolean(True)  # its k-tiles stream through the ring
    if cutlass.const_expr(head_half):
        head_cta = cluster < cutlass.Int32(n_head_tiles)
        stream_cta = cluster >= cutlass.Int32(n_head_tiles)
    tma_ptr_x = tma_desc_x.get_ptr()

    smem_a = cutlass.Array(
        io_dtype, ring * CTA_M * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        io_dtype, my_tiles * mma_n * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    tma_full = cutlass.Array(cutlass.Int64, nbar, space=cutlass.AddressSpace.smem, alignment=8)
    mma_done = cutlass.Array(cutlass.Int64, nbar, space=cutlass.AddressSpace.smem, alignment=8)
    act_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    flags_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(
        cutlass.Int64, MAX_ROUNDS, space=cutlass.AddressSpace.smem, alignment=8
    )
    mail_full = cutlass.Array(
        cutlass.Int64, MAX_ROUNDS, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # [round][source rank][row of the owned slice][token] fp32 partials.
    mailbox = cutlass.Array(
        cutlass.Float32,
        rounds_max * SPLIT * ROWS_PER_WARP * mma_n,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    s_flags = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    # The owner warp's reduced rows [row][token], for the vector pushes.
    s_rows = cutlass.Array(
        cutlass.Float32,
        OWNERS * ROWS_PER_WARP * mma_n,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    s_key = cutlass.Array(cutlass.Int32, NUM_EXPERTS, space=cutlass.AddressSpace.smem, alignment=16)
    s_sigmoid = cutlass.Array(
        cutlass.Float32, NUM_EXPERTS, space=cutlass.AddressSpace.smem, alignment=16
    )
    # do_sort: the runner sort's token masks, tiles and warp totals (_runner_sort).
    s_sort = cutlass.Array(
        cutlass.Int32, SORT_SMEM_WORDS, space=cutlass.AddressSpace.smem, alignment=16
    )
    # cluster_sort (an 8-slot launch: the route CTAs are role cluster 0, token t's on cluster rank t): with every row
    # of the sort in this launch, the route CTAs st.async their top-16 into token 0's route CTA's s_sort_ids, which
    # waits on sort_bar for the bytes instead of polling a global count and reloading the top-16.
    cluster_sort = do_sort and mma_n * halves == SPLIT
    s_sort_ids = cutlass.Array(
        cutlass.Int32, M_MAX * TOP_K, space=cutlass.AddressSpace.smem, alignment=16
    )
    sort_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    sort_in_smem = cutlass.Boolean(False)
    if cutlass.const_expr(cluster_sort):
        sort_in_smem = sort_tokens == num_tokens

    if cutlass.const_expr(cluster_sort):
        if not is_gemv:
            if bx == cutlass.Int32(0):
                if tx == cutlass.Int32(0):
                    prims.mbarrier_init(sort_bar, 1)
                    if sort_in_smem:
                        if num_tokens > cutlass.Int32(1):
                            prims.mbarrier_arrive_expect_tx(
                                sort_bar, (num_tokens - cutlass.Int32(1)) * cutlass.Int32(TOP_K * 4)
                            )
    if is_gemv:
        if warp_id == 0:
            prims.prefetch_tensormap(tma_ptr_w)
            if cutlass.const_expr(head_half):
                prims.prefetch_tensormap(tma_ptr_w64)
            prims.prefetch_tensormap(tma_ptr_x)
            if prims.elect_sync():
                for s in cutlass.range_constexpr(nbar):
                    prims.mbarrier_init(tma_full.subview(s), 1)
                    prims.mbarrier_init(mma_done.subview(s), 1)
                prims.mbarrier_init(act_full, 1)
                prims.mbarrier_init(flags_ready, 1)
                for r in cutlass.range_constexpr(MAX_ROUNDS):
                    prims.mbarrier_init(acc_done.subview(r), 1)
                    # Owner ranks (< 4): the other 7 ranks' partials of the owned 32 rows x 8 tokens arrive by
                    # st.async; the phase completes on their bytes (expected here, before the cluster forms).
                    prims.mbarrier_init(mail_full.subview(r), 1)
                    prims.mbarrier_arrive_expect_tx(
                        mail_full.subview(r), (SPLIT - 1) * ROWS_PER_WARP * mma_n * 4
                    )
        if warp_id == 2:
            prims.tcgen05_alloc(tmem_ptr_i32, tmem_cols)
            prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    # Cluster formation: the peers' shared memory and barriers are addressable from here on.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    if is_gemv:
        tmem_ptr = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)
        if warp_id == 0:
            # =================================================================
            # Weight TMA: fill the ring, prefetch every later k-tile of both
            # rounds into L2, all before any wait; then refill as MMAs finish.
            # =================================================================
            if prims.elect_sync():
                if head_cta:
                    # A head half-tile: every k-tile of the rank, 64-row boxes 16 KB apart; nothing loads later.
                    mh = cluster * cutlass.Int32(HALF_M)
                    for hi in cutlass.range_constexpr(my_tiles):
                        kh = crank + cutlass.Int32(hi * SPLIT)
                        prims.mbarrier_arrive_expect_tx(
                            tma_full.subview(hi), HALF_M * CTA_K * ELEM_BYTES
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_a.subview(hi * HALF_M * CTA_K),
                            tma_ptr_w64,
                            (cutlass.Int32(0), mh, kh * cutlass.Int32(TMA_COPY_ITERS), cutlass.Int32(0),
                             cutlass.Int32(0)),
                            tma_full.subview(hi),
                            l2_cache_hint=EVICT_FIRST,
                        )  # fmt: skip
                else:
                    m0 = cluster * cutlass.Int32(CTA_M) + cutlass.Int32(row_shift)
                    for i in cutlass.range_constexpr(ring):
                        k = crank + cutlass.Int32(i * SPLIT)
                        prims.mbarrier_arrive_expect_tx(
                            tma_full.subview(i), CTA_M * CTA_K * ELEM_BYTES
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_a.subview(i * CTA_M * CTA_K),
                            tma_ptr_w,
                            (
                                cutlass.Int32(0),
                                m0,
                                k * cutlass.Int32(TMA_COPY_ITERS),
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                            ),
                            tma_full.subview(i),
                            l2_cache_hint=EVICT_FIRST,
                        )
                    for rd in cutlass.range_constexpr(rounds_max):
                        tile = cluster + cutlass.Int32(rd * gemv_clusters)
                        if tile < cutlass.Int32(n_tiles):
                            first = ring if rd == 0 else 0
                            for i in range(first, my_tiles):
                                k = crank + i * cutlass.Int32(SPLIT)
                                prims.cp_async_bulk_tensor_prefetch(
                                    tma_ptr_w,
                                    [cutlass.Int32(0), tile * cutlass.Int32(CTA_M) + cutlass.Int32(row_shift),
                                     k * cutlass.Int32(TMA_COPY_ITERS), cutlass.Int32(0), cutlass.Int32(0)],
                                    [],
                                )  # fmt: skip
            # Dependents may launch now; they wait for this whole grid before reading its outputs.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
            if prims.elect_sync():
                if stream_cta:
                    stage = cutlass.Int32(0)
                    phase = cutlass.Int32(0)
                    seq = cutlass.Int32(0)  # k-tiles issued so far, over the rounds
                    for rd in cutlass.range_constexpr(rounds_max):
                        tile = cluster + cutlass.Int32(rd * gemv_clusters)
                        if tile < cutlass.Int32(n_tiles):
                            for i in range(my_tiles):
                                if seq >= cutlass.Int32(ring):
                                    while not cute.arch.mbarrier_try_wait(
                                        mma_done.subview(stage).data_ptr(), phase
                                    ):
                                        pass
                                    k = crank + i * cutlass.Int32(SPLIT)
                                    prims.mbarrier_arrive_expect_tx(
                                        tma_full.subview(stage), CTA_M * CTA_K * ELEM_BYTES
                                    )
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        smem_a.subview(stage * cutlass.Int32(CTA_M * CTA_K)),
                                        tma_ptr_w,
                                        (cutlass.Int32(0), tile * cutlass.Int32(CTA_M) + cutlass.Int32(row_shift),
                                         k * cutlass.Int32(TMA_COPY_ITERS),
                                         cutlass.Int32(0), cutlass.Int32(0)),
                                        tma_full.subview(stage),
                                        l2_cache_hint=EVICT_FIRST,
                                    )  # fmt: skip
                                seq = seq + cutlass.Int32(1)
                                stage = stage + cutlass.Int32(1)
                                if stage == cutlass.Int32(ring):
                                    stage = cutlass.Int32(0)
                                    if seq > cutlass.Int32(ring):
                                        phase = phase ^ cutlass.Int32(1)
        elif warp_id == 1:
            # =================================================================
            # Grid dependency, the head workspace's flags, then x's k-tiles.
            # =================================================================
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(act_full, my_tiles * mma_n * CTA_K * ELEM_BYTES)
                for i in range(my_tiles):
                    k = crank + i * cutlass.Int32(SPLIT)
                    for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_b.subview(
                                i * cutlass.Int32(mma_n * CTA_K)
                                + cutlass.Int32(half * b_half_elems)
                            ),
                            tma_ptr_x,
                            (
                                k * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX),
                                tok0,
                            ),
                            act_full,
                        )
                # The buffer index is for the epilogue's pushes, microseconds later: read it once x's loads are
                # issued, so they do not wait for its round trip.
                s_flags.store(flags.load(idx=0, is_volatile=True), idx=0)
                _red_add_release(flags.data_ptr(3).toint(), cutlass.Int32(1))
                prims.mbarrier_arrive(flags_ready)
        elif warp_id == 2:
            # =================================================================
            # MMA: per round, the rank's k-tiles into that round's accumulator.
            # =================================================================
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=cutlass.Float32,
                a_dtype=io_dtype,
                b_dtype=io_dtype,
                n_dim=mma_n,
                m_dim=CTA_M,
            )
            desc_a_base = prims.Tcgen05SmemDesc.build(
                start_address=smem_a, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )  # fmt: skip
            desc_b_base = prims.Tcgen05SmemDesc.build(
                start_address=smem_b, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )  # fmt: skip
            while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                pass
            if head_cta:
                # A head half-tile: all its k-tiles are in (16 KB apart, 8 KB per 64-element K box). The MMA stays
                # M 128: rows 64-127 read the next box (or the pad) and land in accumulator rows nobody reads.
                acc_h = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)
                for hi in range(my_tiles):
                    while not cute.arch.mbarrier_try_wait(tma_full.subview(hi).data_ptr(), 0):
                        pass
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    for kh in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                        desc_ah = desc_a_base + (
                            hi * cutlass.Int32(STAGE_AH)
                            + cutlass.Int32(
                                (kh // K_BLOCKS_PER_HALF) * A_BOX_H
                                + (kh % K_BLOCKS_PER_HALF) * STEP
                            )
                        )
                        desc_bh = desc_b_base + (
                            hi * cutlass.Int32(stage_b)
                            + cutlass.Int32(
                                (kh // K_BLOCKS_PER_HALF) * b_box + (kh % K_BLOCKS_PER_HALF) * STEP
                            )
                        )
                        accumulate_h = cutlass.Boolean(True)
                        if cutlass.const_expr(kh == 0):
                            accumulate_h = hi > cutlass.Int32(0)
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, acc_h, desc_ah, desc_bh, idesc,
                                accumulate_h,
                            )  # fmt: skip
                if prims.elect_sync():
                    prims.tcgen05_commit(acc_done.subview(0))
            else:
                stage = cutlass.Int32(0)
                phase = cutlass.Int32(0)
                for rd in cutlass.range_constexpr(rounds_max):
                    tile = cluster + cutlass.Int32(rd * gemv_clusters)
                    if tile < cutlass.Int32(n_tiles):
                        acc = cutlass.inttoptr(
                            tmem_ptr_i32.load() + cutlass.Int32(rd * mma_n), 6, cutlass.Int32
                        )
                        for i in range(my_tiles):
                            while not cute.arch.mbarrier_try_wait(
                                tma_full.subview(stage).data_ptr(), phase
                            ):
                                pass
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                                box = kb // K_BLOCKS_PER_HALF
                                within = kb % K_BLOCKS_PER_HALF
                                desc_a = desc_a_base + (
                                    stage * cutlass.Int32(STAGE_A)
                                    + cutlass.Int32(box * A_BOX + within * STEP)
                                )
                                desc_b = desc_b_base + (
                                    i * cutlass.Int32(stage_b)
                                    + cutlass.Int32(box * b_box + within * STEP)
                                )
                                accumulate = cutlass.Boolean(True)
                                if cutlass.const_expr(kb == 0):
                                    accumulate = i > cutlass.Int32(0)
                                if prims.elect_sync():
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, acc, desc_a, desc_b, idesc,
                                        accumulate,
                                    )  # fmt: skip
                            if prims.elect_sync():
                                prims.tcgen05_commit(mma_done.subview(stage))
                            stage = stage + cutlass.Int32(1)
                            if stage == cutlass.Int32(ring):
                                stage = cutlass.Int32(0)
                                phase = phase ^ cutlass.Int32(1)
                        if prims.elect_sync():
                            prims.tcgen05_commit(acc_done.subview(rd))
        elif warp_id >= 4 and warp_id < 8:
            # =================================================================
            # Epilogue: per round, TMEM -> registers, push to / reduce at the
            # row owner; the owner pushes the head slice or writes the shared
            # activation.
            # =================================================================
            w = warp_id - 4  # TMEM lanes 32 w ..: tile rows owned by rank w
            while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                pass
            # flags_ready orders warp 1's store of the buffer index before this read.
            while not cute.arch.mbarrier_try_wait(flags_ready.data_ptr(), 0):
                pass
            b = s_flags.load(idx=0)
            slot_row = lane * cutlass.Int32(mma_n)
            # A head half-tile's rows are TMEM lanes 0-63 (warps 4-5); warps 6-7 hold garbage rows and sit out.
            live = cutlass.Boolean(True)
            if cutlass.const_expr(head_half):
                live = stream_cta | (w < cutlass.Int32(2))
            for rd in cutlass.range_constexpr(rounds_max):
                tile = cluster + cutlass.Int32(rd * gemv_clusters)
                if (tile < cutlass.Int32(n_tiles)) & live:
                    while not cute.arch.mbarrier_try_wait(acc_done.subview(rd).data_ptr(), 0):
                        pass
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    acc = prims.tcgen05_ld(
                        "32x32b",
                        cutlass.inttoptr(
                            tmem_ptr_i32.load() + cutlass.Int32(rd * mma_n), 6, cutlass.Float32
                        ),
                        num=mma_n,
                    )
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    box_base = cutlass.Int32(rd * SPLIT * ROWS_PER_WARP * mma_n)
                    tile_row = cutlass.select_(
                        head_cta, tile * cutlass.Int32(HALF_M), tile * cutlass.Int32(CTA_M)
                    )
                    row0 = cutlass.Int32(tile_row) + w * cutlass.Int32(ROWS_PER_WARP)
                    srow0 = w * cutlass.Int32(ROWS_PER_WARP)
                    if (tile < cutlass.Int32(n_head_tiles)) & (row0 >= cutlass.Int32(RB)):
                        # Router rows: every rank of the cluster pushes its own split-K partial (fp32, -0.0 as +0.0)
                        # into slot [crank] of the partials region; the route CTAs add the 8 in rank order.
                        pe0 = row0 - cutlass.Int32(RB)
                        if pe0 < cutlass.Int32(WE):
                            for tt in cutlass.range_constexpr(mma_n):
                                s_rows.store(
                                    cutlass.Float32(acc[tt]),
                                    idx=(srow0 + lane) * cutlass.Int32(mma_n) + cutlass.Int32(tt),
                                )
                            cute.arch.sync_warp()
                            # 8 router vectors (4 logits each) per token: mma_n / 4 (token, vector) pairs per lane.
                            for ph in cutlass.range_constexpr(mma_n // 4):
                                ppair = lane + cutlass.Int32(ph * 32)
                                ptok = ppair // cutlass.Int32(8)
                                pvec = ppair % cutlass.Int32(8)
                                pe = pe0 + pvec * cutlass.Int32(4)
                                if ptok + tok0 < num_tokens:
                                    if pe < cutlass.Int32(WE):
                                        pwords = []
                                        for pp in cutlass.range_constexpr(4):
                                            pval = cutlass.Float32(
                                                s_rows.load(
                                                    idx=(
                                                        srow0
                                                        + pvec * cutlass.Int32(4)
                                                        + cutlass.Int32(pp)
                                                    )
                                                    * cutlass.Int32(mma_n)
                                                    + ptok
                                                )
                                            )
                                            pwords.append(
                                                _ag._sanitize_f32(pval.bitcast(cutlass.Int32))
                                            )
                                        pdst = (
                                            cutlass.Int32(PBASE)
                                            + (
                                                (b * cutlass.Int32(M_WS) + ptok + tok0)
                                                * cutlass.Int32(world)
                                                + tp_rank
                                            )
                                            * cutlass.Int32(SPLIT * WE)
                                            + crank * cutlass.Int32(WE)
                                            + pe
                                        )
                                        buf_mc.store(
                                            (pwords[0], pwords[1], pwords[2], pwords[3]),
                                            idx=pdst,
                                            alignment=16,
                                        )
                    elif w != crank:
                        base = box_base + crank * cutlass.Int32(ROWS_PER_WARP * mma_n) + slot_row
                        bar = _mapa_u32(mail_full.subview(rd).data_ptr(), w)
                        for h in cutlass.range_constexpr(mma_n // 4):
                            _st_async_v4_f32(
                                _mapa_u32(mailbox.subview(base + cutlass.Int32(4 * h)).data_ptr(), w),
                                cutlass.Float32(acc[4 * h]), cutlass.Float32(acc[4 * h + 1]),
                                cutlass.Float32(acc[4 * h + 2]), cutlass.Float32(acc[4 * h + 3]), bar,
                            )  # fmt: skip
                    else:
                        while not cute.arch.mbarrier_test_wait(mail_full.subview(rd).data_ptr(), 0):
                            pass
                        tot = [cutlass.Float32(0.0)] * mma_n
                        for t in cutlass.range_constexpr(mma_n):
                            total = cutlass.Float32(0.0)
                            for q in cutlass.range_constexpr(SPLIT):
                                part = mailbox.load(
                                    idx=box_base
                                    + cutlass.Int32(q * ROWS_PER_WARP * mma_n)
                                    + slot_row
                                    + cutlass.Int32(t)
                                )
                                total = total + cutlass.Float32(
                                    cutlass.select_(
                                        crank == cutlass.Int32(q), cutlass.Float32(acc[t]), part
                                    )
                                )
                            tot[t] = total
                        if tile < cutlass.Int32(n_head_tiles):
                            # Head: stage the warp's 32 rows x 8 tokens, then 16-byte vectors to every rank.
                            for t in cutlass.range_constexpr(mma_n):
                                s_rows.store(
                                    tot[t],
                                    idx=(w * cutlass.Int32(ROWS_PER_WARP) + lane)
                                    * cutlass.Int32(mma_n)
                                    + cutlass.Int32(t),
                                )
                            cute.arch.sync_warp()
                            # Latent rows (router rows push their partials above).
                            # 4 latent vectors (8 columns each) per token: pair = token * 4 + vector, mma_n / 8
                            # pairs per lane.
                            for lp in cutlass.range_constexpr(mma_n // 8):
                                pair = lane + cutlass.Int32(lp * 32)
                                t = pair // cutlass.Int32(4)
                                v = pair % cutlass.Int32(4)
                                # The zero rows between the latent and the router rows are not pushed.
                                if (t + tok0 < num_tokens) & (
                                    row0 // cutlass.Int32(8) + v < cutlass.Int32(LV)
                                ):
                                    words = []
                                    for p in cutlass.range_constexpr(4):
                                        c0 = srow0 + v * cutlass.Int32(8) + cutlass.Int32(2 * p)
                                        lo = cutlass.Float32(
                                            s_rows.load(idx=c0 * cutlass.Int32(mma_n) + t)
                                        )
                                        hi = cutlass.Float32(
                                            s_rows.load(
                                                idx=(c0 + cutlass.Int32(1)) * cutlass.Int32(mma_n)
                                                + t
                                            )
                                        )
                                        words.append(_ag._sanitize_bf16x2(_ag._pack_bf16x2(hi, lo)))
                                    dst = b * cutlass.Int32(BUF) + (
                                        (t + tok0) * cutlass.Int32(world) + tp_rank
                                    ) * cutlass.Int32(SLOT)
                                    dst = dst + ((row0 // cutlass.Int32(8)) + v) * cutlass.Int32(4)
                                    buf_mc.store(
                                        (words[0], words[1], words[2], words[3]),
                                        idx=dst,
                                        alignment=16,
                                    )
                        else:
                            # Shared: lane l < 16 holds gate column j, lane l + 16 its up row.
                            col = (
                                (tile - cutlass.Int32(n_head_tiles)) * cutlass.Int32(64)
                                + w * cutlass.Int32(16)
                                + lane
                            )
                            for t in cutlass.range_constexpr(mma_n):
                                up = cutlass.Float32(cute.arch.shuffle_sync_bfly(tot[t], offset=16))
                                if lane < cutlass.Int32(16):
                                    if cutlass.Int32(t) + tok0 < num_tokens:
                                        out = _bf16_rn(
                                            _situ(
                                                _bf16_rn(tot[t]), _bf16_rn(up), gate_cap, linear_cap
                                            )
                                        )
                                        shared_bits.store(
                                            cutlass.Int16(out.bitcast(cutlass.Int32) >> 16),
                                            idx=(cutlass.Int32(t) + tok0)
                                            * cutlass.Int32(shared_cols)
                                            + col,
                                        )
            prims.barrier_cta_sync(1, thread_count=128)
            if warp_id == 4:
                prims.tcgen05_dealloc(tmem_ptr, tmem_cols)
    else:
        # =====================================================================
        # Role CTAs: route (CTA t < M) or quantize (CTA M + t) token t from
        # every rank's pushed slot.
        # =====================================================================
        role = role_or_gemv * cutlass.Int32(SPLIT) + crank
        # halves = 2 (fused roles): CTA t routes token t, then quantizes it (fewer clusters next to the GEMV ones).
        fused_roles = halves > 1
        n_roles = num_tokens * cutlass.Int32(1 if fused_roles else 2)
        if role < n_roles:
            routes = role < num_tokens
            tok = cutlass.select_(routes, role, role - num_tokens)
            polls_logits = tx < cutlass.Int32(NUM_EXPERTS // 4)
            e0 = cutlass.select_(polls_logits, tx * cutlass.Int32(4), cutlass.Int32(0))
            bias4 = bias.load(idx=e0, vector_size=4, alignment=16)
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
            if tx == 0:
                s_flags.store(flags.load(idx=0, is_volatile=True), idx=0)
                s_flags.store(flags.load(idx=2, is_volatile=True), idx=1)
            cute.arch.barrier()
            b = s_flags.load(idx=0)
            epoch = s_flags.load(idx=1)
            # A role CTA signs in as soon as it has read the buffer index (it reads nothing of flags after this).
            # Token 0's quantization CTA hands the buffer over once every reader is in; its polls wait for the pushes,
            # which come microseconds later.
            if tx == 0:
                _red_add_release(flags.data_ptr(3).toint(), cutlass.Int32(1))
                # Fused roles: token 0's CTA; else token 0's quantization CTA.
                if role == num_tokens * cutlass.Int32(0 if fused_roles else 1):
                    _flip(flags, b, n_roles + cutlass.Int32(gemv_clusters * halves * SPLIT))
            tok_base = b * cutlass.Int32(BUF) + tok * cutlass.Int32(world * SLOT)
            empty = cutlass.Int32(EMPTY_WORD)
            if routes:
                # The keys and the selection run once per layer, so their code is cold in the SM's instruction cache.
                # Pass 0 runs the same code (one copy: a dynamic loop) on synthetic logits while the pushes are still
                # on their way, its outputs discarded; pass 1 polls the pushes and runs it for real, from warm code.
                for p in cutlass.range(2, unroll=1):
                    real = p == cutlass.Int32(1)
                    if polls_logits:
                        # Pass 0's logits: distinct values in [-4.5, 4.5).
                        dry0 = cutlass.Float32(cutlass.Int32(e0)) * cutlass.Float32(
                            0.01
                        ) - cutlass.Float32(4.5)
                        w0 = dry0.bitcast(cutlass.Int32)
                        w1 = (dry0 + cutlass.Float32(0.01)).bitcast(cutlass.Int32)
                        w2 = (dry0 + cutlass.Float32(0.02)).bitcast(cutlass.Int32)
                        w3 = (dry0 + cutlass.Float32(0.03)).bitcast(cutlass.Int32)
                        if real:
                            # The 8 split-K partials of these 4 logits, pushed by the pushing rank's GEMV CTAs.
                            addr = (
                                cutlass.Int32(PBASE)
                                + (
                                    (b * cutlass.Int32(M_WS) + tok) * cutlass.Int32(world)
                                    + tx // cutlass.Int32(EV)
                                )
                                * cutlass.Int32(SPLIT * WE)
                                + (tx % cutlass.Int32(EV)) * cutlass.Int32(4)
                            )
                            w0, w1, w2, w3 = _poll_logit_parts(buf_uc, addr, WE)
                            for q in cutlass.range_constexpr(SPLIT):
                                buf_uc.store(
                                    (empty, empty, empty, empty),
                                    idx=addr + cutlass.Int32(q * WE),
                                    alignment=16,
                                )
                        words = [w0, w1, w2, w3]
                        for q in cutlass.range_constexpr(4):
                            sig = _rq.sigmoid_accurate(words[q].bitcast(cutlass.Float32))
                            s_sigmoid.store(sig, idx=e0 + q)
                            s_key.store(
                                _rq.selection_key(sig + cutlass.Float32(bias4[q])), idx=e0 + q
                            )
                    cute.arch.barrier()
                    if tx < cutlass.Int32(32):
                        expert, weight_bits = _rq.top16_warp(
                            s_key, s_sigmoid, tx, routed_scaling_factor
                        )
                        if real:
                            if tx < cutlass.Int32(TOP_K):
                                out = tok * cutlass.Int32(TOP_K) + tx
                                topk_ids.store(expert, idx=out)
                                topk_weight_bits.store(weight_bits, idx=out)
                                if cutlass.const_expr(publish):
                                    cute.arch.fence_acq_rel_gpu()
                            if cutlass.const_expr(publish):
                                cute.arch.sync_warp()
                                if tx == cutlass.Int32(0):
                                    _ag._store_release(
                                        ready.data_ptr(tok).toint(), epoch + cutlass.Int32(1)
                                    )
                            if cutlass.const_expr(do_sort):
                                # Token 0's route CTA builds the runner's routing from every token's top-16.
                                if sort_in_smem:
                                    # cluster_sort: into its s_sort_ids (this CTA is cluster rank tok).
                                    if tx < cutlass.Int32(TOP_K):
                                        slot = tok * cutlass.Int32(TOP_K) + tx
                                        if tok == cutlass.Int32(0):
                                            s_sort_ids.store(expert, idx=slot)
                                        else:
                                            _st_async_b32(
                                                _mapa_u32(s_sort_ids.subview(slot).data_ptr(), 0),
                                                expert,
                                                _mapa_u32(sort_bar.data_ptr(), 0),
                                            )
                                else:
                                    if tx < cutlass.Int32(TOP_K):
                                        cute.arch.fence_acq_rel_gpu()
                                    cute.arch.sync_warp()
                                    if tx == cutlass.Int32(0):
                                        _red_add_release(
                                            flags.data_ptr(1).toint(), cutlass.Int32(1)
                                        )
                    # Pass 0's readers of the keys (warp 0's selection) are done before pass 1 rewrites them.
                    cute.arch.barrier()
                if cutlass.const_expr(do_sort):
                    if role == cutlass.Int32(0):
                        _runner_sort(
                            tx, flags, sort_ids, s_sort, sort_e2p, sort_p2t, sort_total, sort_tile_expert,
                            sort_tile_limit, sort_num_tiles, num_tokens, sort_tokens, local_offset, local_num,
                            tile_log2, sort_in_smem, s_sort_ids, sort_bar,
                        )  # fmt: skip
            if cutlass.const_expr(fused_roles):
                cute.arch.barrier()
                _quant_token(
                    tx,
                    buf_uc,
                    tok_base,
                    quant_words,
                    scales,
                    ready,
                    tok,
                    epoch,
                    LV,
                    SLOT,
                    M_WS,
                    publish,
                )
            else:
                if not routes:
                    _quant_token(
                        tx,
                        buf_uc,
                        tok_base,
                        quant_words,
                        scales,
                        ready,
                        tok,
                        epoch,
                        LV,
                        SLOT,
                        M_WS,
                        publish,
                    )
            cute.arch.barrier()
        else:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.jit
def _quant_token(
    tx, buf_uc, tok_base, quant_words, scales, ready, tok, epoch,
    LV: cutlass.Constexpr[int], SLOT: cutlass.Constexpr[int], M_WS: cutlass.Constexpr[int],
    publish: cutlass.Constexpr[bool],
):  # fmt: skip
    """MXFP8 of token ``tok``'s gathered latent: thread ``tx`` polls its 8-element vector of every rank's slot,
    re-arms it and stores its E4M3 bytes (and its group's UE8M0 scale)."""
    empty = cutlass.Int32(EMPTY_WORD)
    addr = (
        tok_base
        + (tx // cutlass.Int32(LV)) * cutlass.Int32(SLOT)
        + (tx % cutlass.Int32(LV)) * cutlass.Int32(4)
    )
    w0, w1, w2, w3 = _ag._poll4(buf_uc, addr)
    buf_uc.store((empty, empty, empty, empty), idx=addr, alignment=16)
    q_lo, q_hi, sf_byte = _rq.mxfp8_quant_vec8([w0, w1, w2, w3])
    quant_words.store(
        (q_lo, q_hi),
        idx=tok * cutlass.Int32(HIDDEN_SIZE // 4) + tx * cutlass.Int32(2),
        alignment=8,
    )
    if tx % cutlass.Int32(SF_VEC_SIZE // 8) == cutlass.Int32(0):
        scales.store(
            cutlass.Uint8(sf_byte),
            idx=tok * cutlass.Int32(HIDDEN_SIZE // SF_VEC_SIZE)
            + tx // cutlass.Int32(SF_VEC_SIZE // 8),
        )
    if cutlass.const_expr(publish):
        prims.fence_proxy("async_global")
        cute.arch.fence_acq_rel_gpu()
        cute.arch.barrier()
        if tx == cutlass.Int32(0):
            _ag._store_release(
                ready.data_ptr(tok + cutlass.Int32(M_WS)).toint(), epoch + cutlass.Int32(1)
            )


@dsl_user_op
def _red_or_shared(smem_ptr, value, *, loc=None, ip=None):
    """red.shared.or.b32 at a shared-memory location."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None, [smem_ptr.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;", "r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _popc(value, *, loc=None, ip=None):
    """popc.b32."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Int32(value).ir_value(loc=loc, ip=ip)], "popc.b32 $0, $1;", "=r,r",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@cute.jit
def _runner_sort(
    tx, flags, topk_ids, s_sort, e2p, p2t, total, tile_expert, tile_limit, num_tiles_out, num_tokens,
    sort_tokens, local_offset, local_num, tile_log2, in_smem, s_ids, ids_bar,
):  # fmt: skip
    """The TRTLLM-Gen MXFP4 runner's routing of this rank's local experts ``[local_offset, local_offset +
    local_num)`` from the top-16 of all ``sort_tokens <= 32`` rows in ``topk_ids`` (``k3_route_tail``'s runner sort
    with ``p2_token``: rows of an expert in ascending token order, tiles of ``2^tile_log2`` rows). This call's
    ``num_tokens`` rows are the last ones (earlier calls of the same forward wrote the others). Runs on token 0's
    route CTA once every route CTA of this call counted itself in ``flags[1]`` after storing its top-16; re-arms
    ``flags[1]``. With ``in_smem`` (every row in this launch, the route CTAs one cluster) the top-16 come from
    ``s_ids`` instead, once ``ids_bar`` has received the other route CTAs' st.async bytes. Smem: ``[0, 256)`` per
    local expert the mask of the rows that chose it, ``[256, 512)`` its first tile, ``[512, 528)`` warp totals."""
    n_e = sort_tokens * cutlass.Int32(TOP_K)
    if tx < local_num:
        s_sort.store(cutlass.Int32(0), idx=tx)
    if tx == cutlass.Int32(0):
        if in_smem:
            if num_tokens > cutlass.Int32(1):
                # test_wait spin: a warp suspended in try_wait is woken late by remote completions.
                while not cute.arch.mbarrier_test_wait(ids_bar.data_ptr(), 0):
                    pass
        else:
            while _load_acquire(flags.data_ptr(1).toint()) < num_tokens:
                pass
    cute.arch.barrier()
    lx = []
    for q in cutlass.range_constexpr(SORT_ENTRIES_PER_THREAD):
        e = tx + cutlass.Int32(q * THREADS)
        valid = e < n_e
        idx = cutlass.select_(valid, e, cutlass.Int32(0))
        raw = cutlass.Int32(0)
        if in_smem:
            raw = cutlass.Int32(s_ids.load(idx=idx))
        else:
            raw = cutlass.Int32(topk_ids.load(idx=idx, is_volatile=True))
        v = raw - local_offset
        v = cutlass.Int32(
            cutlass.select_(valid & (v >= cutlass.Int32(0)) & (v < local_num), v, cutlass.Int32(-1))
        )
        lx.append(v)
        if v >= cutlass.Int32(0):
            _red_or_shared(
                s_sort.subview(v).data_ptr(), cutlass.Int32(1) << (e // cutlass.Int32(TOP_K))
            )
    cute.arch.barrier()
    cnt = cutlass.Int32(0)
    if tx < local_num:
        cnt = _popc(s_sort.load(idx=tx))
    tiles = (cnt + (cutlass.Int32(1) << tile_log2) - cutlass.Int32(1)) >> tile_log2
    warp = tx // cutlass.Int32(32)
    lane = tx % cutlass.Int32(32)
    inclusive = _ag.warp_inclusive_sum(tiles)
    if lane == cutlass.Int32(31):
        s_sort.store(inclusive, idx=cutlass.Int32(512) + warp)
    cute.arch.barrier()
    tile = inclusive - tiles
    num_tiles = cutlass.Int32(0)
    for w in cutlass.range_constexpr(SORT_MAX_LOCAL // 32):
        tot = cutlass.Int32(s_sort.load(idx=512 + w))
        if cutlass.Int32(w * 32) < local_num:
            if cutlass.Int32(w) < warp:
                tile = tile + tot
            num_tiles = num_tiles + tot
    if tx < local_num:
        s_sort.store(tile, idx=cutlass.Int32(256) + tx)
        end = (tile << tile_log2) + cnt
        for t in cutlass.range(tiles, unroll=1):
            limit = (tile + t + cutlass.Int32(1)) << tile_log2
            limit = cutlass.Int32(cutlass.select_(end < limit, end, limit))
            tile_expert.store(tx, idx=tile + t)
            tile_limit.store(limit, idx=tile + t)
    cute.arch.barrier()
    for q in cutlass.range_constexpr(SORT_ENTRIES_PER_THREAD):
        e = tx + cutlass.Int32(q * THREADS)
        if e < n_e:
            perm = cutlass.Int32(-1)
            v = lx[q]
            if v >= cutlass.Int32(0):
                t = e // cutlass.Int32(TOP_K)
                below = (cutlass.Int32(1) << t) - cutlass.Int32(1)
                perm = (
                    cutlass.Int32(s_sort.load(idx=cutlass.Int32(256) + v)) << tile_log2
                ) + _popc(cutlass.Int32(s_sort.load(idx=v)) & below)
                p2t.store(t, idx=perm)
            e2p.store(perm, idx=e)
    if tx == cutlass.Int32(0):
        total.store(num_tiles << tile_log2, idx=0)
        num_tiles_out.store(num_tiles, idx=0)
        flags.store(cutlass.Int32(0), idx=1)


@dsl_user_op
def _red_add_release(addr_i64, val, *, loc=None, ip=None):
    """red.release.gpu.global.add.u32 (no round trip)."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None, [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _load_acquire(addr_i64, *, loc=None, ip=None):
    """ld.acquire.gpu.global.u32."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip)], "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l",
            has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@cute.jit
def _flip(flags, b, readers):
    """Hands the other buffer to the next call once all ``readers`` CTAs of this call that read the buffer index have
    signed in (flags[3], each with a release right after its read), so none reads it any more. The next call reads the
    index only after this grid has completed."""
    while _load_acquire(flags.data_ptr(3).toint()) < readers:
        pass
    flags.store(cutlass.Int32(0), idx=3)
    flags.store(b ^ cutlass.Int32(1), idx=0)


def _weight_tensor_map(w, n_out, k_in, box_rows=CTA_M):
    """W as five TMA dimensions (64-element column chunk, row, 64-element chunk index, 1, 1), box_rows rows a call."""
    return cuda.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[TMA_K_BOX, n_out, k_in // TMA_K_BOX, 1, 1],
        global_strides=[
            (k_in * ELEM_BYTES) // 16,
            (TMA_K_BOX * ELEM_BYTES) // 16,
            (n_out * k_in * ELEM_BYTES) // 16,
            (n_out * k_in * ELEM_BYTES) // 16,
        ],
        box_dims=[TMA_K_BOX, box_rows, TMA_COPY_ITERS, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def _activation_tensor_map(x, cols, num_tokens, mma_n):
    """x [M, cols] as (cols, M) with an mma_n-row box: rows past num_tokens arrive as zeros."""
    return cuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[cols, num_tokens],
        global_strides=[(cols * ELEM_BYTES) // 16],
        box_dims=[TMA_K_BOX, mma_n],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def k31_moe_front_jit(
    w: cute.Tensor,  # [N, K] bf16
    x: cute.Tensor,  # [M, K] bf16
    bias: cute.Tensor,
    buf_uc: cute.Tensor,
    buf_mc: cute.Tensor,
    flags: cute.Tensor,
    ready: cute.Tensor,
    topk_ids: cute.Tensor,
    topk_weight_bits: cute.Tensor,
    quant_words: cute.Tensor,
    scales: cute.Tensor,
    shared_bits: cute.Tensor,
    sort_ids: cute.Tensor,
    sort_e2p: cute.Tensor,
    sort_p2t: cute.Tensor,
    sort_total: cute.Tensor,
    sort_tile_expert: cute.Tensor,
    sort_tile_limit: cute.Tensor,
    sort_num_tiles: cute.Tensor,
    num_tokens: cutlass.Int32,
    tp_rank: cutlass.Int32,
    routed_scaling_factor: cutlass.Float64,
    local_offset: cutlass.Int32,
    local_num: cutlass.Int32,
    tile_log2: cutlass.Int32,
    sort_tokens: cutlass.Int32,
    world: cutlass.Constexpr[int],
    shared_cols: cutlass.Constexpr[int],
    n_head_tiles: cutlass.Constexpr[int],
    n_tiles: cutlass.Constexpr[int],
    gemv_clusters: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    gate_cap: cutlass.Constexpr[float],
    linear_cap: cutlass.Constexpr[float],
    publish: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    head_half: cutlass.Constexpr[bool],
    do_sort: cutlass.Constexpr[bool],
    mma_n: cutlass.Constexpr[int],
    halves: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
) -> None:
    rows = _weight_rows(world, n_head_tiles, n_tiles, head_half)
    tma_desc_w = _weight_tensor_map(w, rows, k_in)
    tma_desc_w64 = _weight_tensor_map(w, rows, k_in, box_rows=HALF_M)
    tma_desc_x = _activation_tensor_map(x, k_in, num_tokens, mma_n)
    k31_moe_front_kernel(
        tma_desc_w, tma_desc_w64, tma_desc_x, bias, buf_uc, buf_mc, flags, ready, topk_ids, topk_weight_bits,
        quant_words, scales, shared_bits, sort_ids, sort_e2p, sort_p2t, sort_total, sort_tile_expert,
        sort_tile_limit, sort_num_tiles, num_tokens, tp_rank, routed_scaling_factor, local_offset, local_num,
        tile_log2, sort_tokens, world,
        shared_cols, n_head_tiles, n_tiles, gemv_clusters, k_in, ring, gate_cap, linear_cap, publish, head_half,
        do_sort, mma_n, halves,
    ).launch(
        grid=((gemv_clusters * halves + role_clusters_of(mma_n * halves)) * SPLIT, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(SPLIT, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
        # One CTA per SM (shared memory): without it ptxas may cap registers for two and spill (bias4 did).
        min_blocks_per_mp=1,
    )  # fmt: skip


# =============================================================================
# Host side: operands, head workspace, compile + launch
# =============================================================================
DEFAULT_RING = 4

_compile_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_workspaces: Dict[object, dict] = {}


def head_weight(
    latent_down: torch.Tensor, router: torch.Tensor, rank: int, world: int
) -> torch.Tensor:
    """This rank's head rows: latent_down rows ``[r * 3584 / W, (r + 1) * 3584 / W)``, zero rows up to
    :func:`router_row_base`, then router rows ``[r * 896 / W, (r + 1) * 896 / W)`` (both BF16 ``[rows, K]``)."""
    wl, we = HIDDEN_SIZE // world, NUM_EXPERTS // world
    pad = latent_down.new_zeros(router_row_base(world) - wl, latent_down.shape[1])
    return torch.cat(
        [latent_down[rank * wl : (rank + 1) * wl], pad, router[rank * we : (rank + 1) * we]]
    )


def front_weight(head: torch.Tensor, gate_up: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The front's one weight: ``head`` (:func:`head_weight`) zero-padded to whole 128-row tiles, then ``gate_up``
    (``[2 I, K]``: gate rows, then up rows) with every 32 rows holding 16 gate rows and the 16 up rows of the same
    columns. Without ``gate_up`` the front computes the head alone (``shared_cols`` 0)."""
    rows, k_in = head.shape
    if gate_up is None:
        gate_up = head[:0]
    inter = gate_up.shape[0] // 2
    if gate_up.shape[1] != k_in or inter % 64 != 0 or head.dtype != gate_up.dtype:
        raise ValueError(
            f"k31_moe_front weight: head {tuple(head.shape)}, gate_up {tuple(gate_up.shape)} "
            "(needs equal K and dtype and a multiple of 64 shared columns)"
        )
    padded = -(-rows // CTA_M) * CTA_M
    out = torch.zeros(padded, k_in, dtype=head.dtype, device=head.device)
    out[:rows].copy_(head)
    gate = gate_up[:inter].reshape(inter // 16, 16, k_in)
    up = gate_up[inter:].reshape(inter // 16, 16, k_in)
    shared = torch.stack([gate, up], dim=1).reshape(2 * inter, k_in)
    return torch.cat([out, shared]).contiguous()


@functools.lru_cache(maxsize=None)
def _max_clusters_of(device_index: int) -> int:
    from cutlass.utils.hardware_info import HardwareInfo

    return HardwareInfo(device_index).get_max_active_clusters(SPLIT)


def max_clusters(device: torch.device) -> int:
    """Clusters of SPLIT CTAs, one CTA per SM, that ``device`` holds at once."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _max_clusters_of(index)


def weight_supported(
    world: int, shared_cols: int, k_in: int, device: torch.device, ring: int = DEFAULT_RING
) -> bool:
    """Whether the front serves this TP world, shared activation width and hidden size (any M <= 8)."""
    return supports(world, shared_cols, max_clusters(device), k_in, ring)


def can_run(
    x: torch.Tensor, w_front: torch.Tensor, world: int, shared_cols: int, ring: int = DEFAULT_RING
) -> bool:
    return (
        x.dim() == 2
        and 0 < x.shape[0] <= MAX_ROWS
        and x.dtype == torch.bfloat16
        and w_front.dtype == torch.bfloat16
        and x.shape[1] == w_front.shape[1]
        and w_front.shape[0] >= head_tiles(world) * CTA_M + 2 * shared_cols
        and supports(world, shared_cols, max_clusters(x.device), x.shape[1], ring)
    )


def head_workspace(mapping) -> dict:
    """The head workspace of ``mapping``'s TP group (MNNVL multicast, every word empty), created on first use:
    collective, every rank of the group calls it at the same point (outside CUDA-graph capture)."""
    ws = _workspaces.get(mapping)
    if ws is not None:
        return ws
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    world = mapping.tp_size
    words = _ag.workspace_words(world)
    comm = _get_mnnvl_workspace_comm(mapping)
    use_fabric_handle = (
        os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
    )
    error: Optional[Exception] = None
    try:
        handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, use_fabric_handle)
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        with torch.inference_mode():
            uc.fill_(EMPTY_WORD)
            flags = torch.zeros(4, dtype=torch.int32, device=uc.device)
        torch.cuda.synchronize()
        ws = dict(
            handle=handle, comm=comm, uc=uc, mc=mc, flags=flags, rank=mapping.tp_rank, world=world
        )
    except (RuntimeError, ValueError) as exc:
        error = exc
    # Also the barrier that keeps any rank from pushing into a peer's buffer before the peer has emptied it.
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError("k31_moe_front head workspace failed on at least one rank") from error
    _workspaces[mapping] = ws
    return ws


def _launch_config(
    num_tokens: int, k_in: int, world: int, shared_cols: int, ring: int, device: torch.device
):
    """The launch of ``num_tokens <= _chunk_rows(...)`` rows: (world, shared_cols, head tiles, tiles, GEMV clusters per
    token half, k_in, ring, PDL, head_half, MMA token slots, token halves)."""
    mma_n = mma_n_of(min(num_tokens, 16))
    halves = 2 if num_tokens > 16 else 1
    capacity = max_clusters(device)
    # One round with the head in 64-row half-tiles where it fits (TP16): every head k-tile on chip before the wait.
    half = half_geometry(world, shared_cols, capacity, k_in, ring, mma_n, halves)
    head_half = half is not None
    ht, st, clusters = half if head_half else geometry(world, shared_cols, capacity, mma_n)
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    return (
        world,
        shared_cols,
        ht,
        ht + st,
        clusters,
        k_in,
        ring,
        use_pdl,
        head_half,
        mma_n,
        halves,
    )


@functools.lru_cache(maxsize=None)
def _chunk_rows(world: int, shared_cols: int, k_in: int, ring: int, device_index: int) -> int:
    """Rows of one launch: 32 where two token halves of GEMV clusters and the role clusters of 32 rows fit on the GPU
    at once (head half-tiles), else 16."""
    capacity = _max_clusters_of(device_index)
    two = half_geometry(world, shared_cols, capacity, k_in, ring, 16, 2)
    return M_CHUNK if two is not None else 16


def _chunks(num_tokens: int, rows: int = M_CHUNK):
    """Row ranges of the launches of one call: ``rows`` rows each, the last one shorter."""
    return [(r0, min(num_tokens, r0 + rows)) for r0 in range(0, num_tokens, rows)]


def is_compiled(num_tokens: int, k_in: int, world: int, shared_cols: int, device: torch.device,
                gate_cap: float = 1.0, linear_cap: float = 1.0, ring: int = DEFAULT_RING,
                do_sort: bool = False) -> bool:  # fmt: skip
    """Whether every kernel variant a :func:`run_k31_moe_front` call on ``num_tokens`` rows launches is compiled."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    chunks = _chunks(num_tokens, _chunk_rows(world, shared_cols, k_in, ring, index))
    need = [(r1 - r0, do_sort and i == len(chunks) - 1) for i, (r0, r1) in enumerate(chunks)]
    return all(
        (
            _launch_config(n, k_in, world, shared_cols, ring, device),
            float(gate_cap),
            float(linear_cap),
            d,
        )
        in _compiled
        for n, d in need
    )


def runner_sort_fits(num_tokens: int, local_num: int, tile: int) -> bool:
    """Whether the front can write the runner's routing for ``num_tokens`` rows (tiles of ``tile``, a power of two)."""
    return (
        0 < num_tokens <= SORT_MAX_TOKENS
        and 0 < local_num <= SORT_MAX_LOCAL
        and tile >= 1
        and tile & (tile - 1) == 0
    )


def run_k31_moe_front(
    x: torch.Tensor,
    w_front: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    ws: dict,
    shared_cols: int = 0,
    gate_cap: float = 1.0,
    linear_cap: float = 1.0,
    ring: int = DEFAULT_RING,
    sort: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list]
]:
    """``x`` BF16 ``[M <= 32, 7168]`` (the MoE input, the same on every rank of the group), ``w_front`` from
    :func:`front_weight`, ``bias`` the routing bias FP32 ``[896]``, ``ws`` from :func:`head_workspace`. Returns
    ``(topk_ids int32 [M, 16], topk_weights bf16 [M, 16], quantized e4m3 [M, 3584], scales uint8 [M, 112],
    shared bf16 [M, shared_cols] or None)``: the top-16 and routing weights of the gathered router logits and the
    MXFP8 quantization (linear scale layout) of the gathered latent, as ``kimi_k3_noaux_tc_mxfp8_quant`` gives for
    them. With ``sort = (tile, local_expert_offset, local_num_experts, num_experts)`` (:func:`runner_sort_fits`) the
    sixth element is the TRTLLM-Gen MXFP4 runner's routing of the local experts (``[expanded_idx_to_permuted_idx,
    permuted_idx_to_token_idx, total_num_padded_tokens, cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit,
    num_non_exiting_ctas]``, ``k3_route_tail.run_k3_route_runner``'s layout and sizes), else ``None``. One launch
    per 32 rows (8 or 16 MMA token slots, one or two token halves of GEMV clusters; 16 rows per launch where two
    halves do not fit the GPU); the last one also writes the routing. The kernel compiles
    on the first call for each configuration, which must not be under graph capture."""
    world = ws["world"]
    if not can_run(x, w_front, world, shared_cols, ring):
        raise ValueError(
            f"k31_moe_front: unsupported call x {tuple(x.shape)} {x.dtype}, w_front {tuple(w_front.shape)} "
            f"{w_front.dtype}, world {world}, shared_cols {shared_cols}, ring {ring}"
        )
    num_tokens = x.shape[0]
    device = x.device
    if not x.is_contiguous():
        x = x.contiguous()
    topk_ids = torch.empty(num_tokens, TOP_K, dtype=torch.int32, device=device)
    topk_weights = torch.empty(num_tokens, TOP_K, dtype=torch.bfloat16, device=device)
    quantized = torch.empty(num_tokens, HIDDEN_SIZE, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(num_tokens, HIDDEN_SIZE // SF_VEC_SIZE, dtype=torch.uint8, device=device)
    shared = (
        torch.empty(num_tokens, shared_cols, dtype=torch.bfloat16, device=device)
        if shared_cols
        else None
    )
    prerouted = None
    if sort is not None:
        tile, _, local_num, num_experts = sort
        if not runner_sort_fits(num_tokens, local_num, tile):
            raise ValueError(
                f"k31_moe_front: no runner sort for M {num_tokens}, {local_num} experts, tile {tile}"
            )
        expanded = num_tokens * TOP_K
        filled = min(num_experts, expanded)
        max_ctas = filled + (expanded - filled) // tile
        i32 = dict(dtype=torch.int32, device=device)
        prerouted = [
            torch.empty((expanded,), **i32),
            torch.empty((max_ctas * tile + 1,), **i32),
            torch.empty((1,), **i32),
            torch.empty((max_ctas,), **i32),
            torch.empty((max_ctas,), **i32),
            torch.empty((1,), **i32),
        ]
    index = device.index if device.index is not None else torch.cuda.current_device()
    for r0, r1 in _chunks(num_tokens, _chunk_rows(world, shared_cols, x.shape[1], ring, index)):
        last = r1 == num_tokens
        _launch(
            x[r0:r1], w_front, bias, routed_scaling_factor, ws, shared_cols, gate_cap, linear_cap, ring,
            (topk_ids[r0:r1], topk_weights[r0:r1], quantized[r0:r1], scales[r0:r1],
             shared[r0:r1] if shared is not None else None),
            sort if last else None, prerouted if last else None, topk_ids, num_tokens,
        )  # fmt: skip
    return topk_ids, topk_weights, quantized, scales, shared, prerouted


def _launch(x, w_front, bias, routed_scaling_factor, ws, shared_cols, gate_cap, linear_cap, ring, outs, sort,
            prerouted, sort_ids, sort_tokens) -> None:  # fmt: skip
    """One front launch on ``x`` (``<= _chunk_rows`` rows) into the row views ``outs``; with ``sort`` it also writes the
    runner's routing ``prerouted`` of all ``sort_tokens`` rows of ``sort_ids`` (this launch's rows last)."""
    import cuda.bindings.driver as cuda_driver
    from cutlass.cute.runtime import from_dlpack

    world, rank = ws["world"], ws["rank"]
    num_tokens, k_in = x.shape
    device = x.device
    topk_ids, topk_weights, quantized, scales, shared = outs
    flags = ws["flags"]
    # A head-only front still hands the kernel an aligned shared-output argument it never writes.
    shared_arg = shared if shared is not None else flags.view(torch.int16).view(-1)
    tile_log2 = local_offset = local_num = 0
    if sort is not None:
        tile, local_offset, local_num, _ = sort
        tile_log2 = tile.bit_length() - 1
        sort_args = [sort_ids] + prerouted
    else:
        sort_args = [flags] * 7

    def arg2(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)

    def arg(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=0)

    args = (
        arg2(w_front),
        arg2(x),
        arg(bias.contiguous().view(-1)),
        arg(ws["uc"].view(-1)),
        arg(ws["mc"].view(-1)),
        arg(flags.view(-1)),
        arg(flags.view(-1)),  # per-token ready words: unused (no publish)
        arg(topk_ids.reshape(-1)),
        arg(topk_weights.reshape(-1).view(torch.int16)),
        arg(quantized.reshape(-1).view(torch.int32)),
        arg(scales.reshape(-1)),
        arg(shared_arg.reshape(-1).view(torch.int16)),
        *(arg(t.reshape(-1)) for t in sort_args),
    )
    cfg = _launch_config(num_tokens, k_in, world, shared_cols, ring, device)
    _, _, ht, n_tiles, clusters, _, _, use_pdl, head_half, mma_n, halves = cfg
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    do_sort = sort is not None
    key = (cfg, float(gate_cap), float(linear_cap), do_sort)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "k31_moe_front must run once per configuration outside CUDA-graph capture first"
            )
        with _compile_lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    k31_moe_front_jit, *args, num_tokens, rank, float(routed_scaling_factor), local_offset,
                    local_num, tile_log2, sort_tokens, world, shared_cols, ht, n_tiles, clusters, k_in, ring,
                    float(gate_cap), float(linear_cap), False, use_pdl, head_half, do_sort, mma_n, halves, stream,
                )  # fmt: skip
    fn(
        *args, num_tokens, rank, float(routed_scaling_factor), local_offset, local_num, tile_log2, sort_tokens,
        stream,
    )  # fmt: skip
