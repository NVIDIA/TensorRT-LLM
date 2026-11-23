# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 decode TopK + routing maps with replicated hot experts.

Decode MoE under EP4 is skewed: per layer the rank that activates the most distinct
local experts finishes its grouped GEMMs last and every other rank waits for it in
the post-MoE all-reduce (about 12 us per layer on B300 at 64 decode rows; the last
rank is the one with the most grouped-GEMM work in 82 % of layers). Decode FC1/FC2
stream one expert's weights per active expert, so a rank's cost is its count of
distinct active experts, not its token count, and work moves between ranks only a
whole expert at a time.

This kernel keeps ``moe_topk_routing_push``'s contract (selection, weights, PDL roles,
push-based cluster hand-off) and adds ``NREP`` hot experts per layer that every rank
holds a copy of in local slots ``64 .. 64 + NREP - 1`` (``hot_ids[h]`` = the global id
held in slot ``64 + h``). Every rank computes the same TopK over all tokens, so every
rank derives the same assignment without communication:

* CTAs claim slots in a 256-bin *global* histogram and push it to cluster rank 0;
* rank 0 counts, per EP rank, the active experts that are not hot (two warps cover one
  rank's 64 experts: ballot + popc), then assigns each active hot expert, in ``hot_ids``
  order, to the EP rank with the fewest active experts so far (ties: lowest rank);
* this EP rank's tiles are its active home experts that are not hot (local slot
  ``e - offset``) plus the hot experts assigned to it (slot ``64 + h``, whatever their
  home), in slot order (a bitmask over the 64 + NREP slots, prefix by popc);
* rank 0 pushes every CTA a 256-entry base (``-1`` = not this EP rank's), and each
  winner maps through it.

Every (token, k) pair is therefore mapped on exactly one EP rank, and the EP all-reduce
that sums the ranks' FC2 partials still sums every winner exactly once; only which rank
computes a hot expert's partial changes (bf16 partial grouping, not bitwise against the
kernel without replicas). ``expert_hist`` (the decode MoE L2 prefetch's counts) keeps its
meaning: the step's token count of each of this rank's 64 home experts. ``hot_ids``
entries ``< 0`` are unused slots (no expert), so one compiled kernel also serves layers
without replicas (the MTP layer).

Tile capacity becomes ``min(64 + NREP, 8 T)`` (``allocate_outputs``); FC1/FC2 must be
built for ``64 + NREP`` experts with the replica weights in those slots. Push cluster
path only (9..64 rows at 256 threads); smaller T keeps the home mapping
(``moe_topk_routing``'s block path), which every rank also takes for the same T.
"""

from typing import Dict, Optional

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48
from . import moe_topk_routing as _base
from . import moe_topk_routing_push as _push

_BLOCK = _base._BLOCK
_WARP = _base._WARP
_WARPS = _base._WARPS
_FULL_MASK = _base._FULL_MASK
_EXPERTS = _base._EXPERTS
_TOP_K = _base._TOP_K
_TILE = _base._TILE
_LOCAL_EXPERTS = _base._LOCAL_EXPERTS
_VEC = _push._VEC
_ONE_REDUX = _push._ONE_REDUX
OUTPUT_ORDER = _base.OUTPUT_ORDER
MAX_NREP = 64  # slot bitmask: 4 x 32 bits over 64 + NREP slots
cluster_ctas_for_tokens = _base.cluster_ctas_for_tokens

assert _BLOCK >= _EXPERTS, "rank 0 maps one global expert per thread"


@cute.jit
def _select_topk_global(
    logits: cute.Tensor,
    logit_splits: cutlass.Constexpr,
    one_redux: cutlass.Constexpr,
    bias_vals,
    topk_weights: cute.Tensor,
    topk_indices: cute.Tensor,
    expert_count,
    token: cutlass.Int32,
    lane: cutlass.Int32,
    valid_token,
    local_expert_offset: cutlass.Int32,
    routed_scaling_factor: cutlass.Float32,
):
    """``moe_topk_routing_push._select_topk_lanevec`` (bit-identical selection and weights) with the
    slot claimed in a 256-bin global histogram: returns this lane's (global expert or -1, slot)."""
    keys = cutlass.Array(cutlass.Uint64, _VEC)
    raw_scores = cutlass.Array(cutlass.Float32, _VEC)
    my_raw = cutlass.Float32(0.0)
    my_selected_idx = cutlass.Int32(-1)
    for v in cutlass.range_constexpr(_VEC):
        keys[v] = cutlass.Uint64(0)
        raw_scores[v] = cutlass.Float32(0.0)
    if valid_token:
        lptr = logits.iterator.raw_ptr()
        row = token * cutlass.Int32(_EXPERTS * logit_splits) + lane * cutlass.Int32(_VEC)
        parts = []
        for split in cutlass.range_constexpr(logit_splits):
            lo = prims.load_ext(
                lptr + row + cutlass.Int32(split * _EXPERTS),
                dtype=cutlass.Float32,
                count=4,
            )
            hi = prims.load_ext(
                lptr + row + cutlass.Int32(split * _EXPERTS + 4),
                dtype=cutlass.Float32,
                count=4,
            )
            parts.append((lo, hi))
        for v in cutlass.range_constexpr(_VEC):
            half = 0 if v < 4 else 1
            idx = v if v < 4 else v - 4
            logit = cutlass.Float32(parts[0][half][idx])
            for split in cutlass.range_constexpr(1, logit_splits):
                logit = logit + cutlass.Float32(parts[split][half][idx])
            raw = cutlass.Float32(0.5) * cute.math.tanh(
                cutlass.Float32(0.5) * logit
            ) + cutlass.Float32(0.5)
            raw_scores[v] = raw
            keys[v] = _base._pack_score_index(
                raw + bias_vals[v], lane * cutlass.Int32(_VEC) + cutlass.Int32(v)
            )
        _base._sort8_key_desc(keys)
        if cutlass.const_expr(one_redux):
            head_idx = cutlass.Int32(65535) - cutlass.Int32(keys[0] & cutlass.Uint64(0xFFFF))
            for k in cutlass.range_constexpr(_TOP_K):
                score_key = cutlass.Uint32(keys[0] >> cutlass.Uint64(32))
                warp_score_key = cutlass.Uint32(
                    prims.redux_sync(cutlass.Int32(score_key), prims.ReductionKind.UMAX, _FULL_MASK)
                )
                is_max = score_key == warp_score_key
                ball = cutlass.Uint32(prims.vote_sync(_FULL_MASK, is_max, prims.VoteSync.BALLOT))
                owner = cutlass.Int32(cute.arch.popc(ball - cutlass.Uint32(1)))
                if cutlass.Int32(cute.arch.popc(ball)) != cutlass.Int32(1):
                    # Exact score tie: the served tie-break, smallest expert index (largest inverse index).
                    inverse_idx = cutlass.Uint32(0)
                    if is_max:
                        inverse_idx = cutlass.Uint32(keys[0] & cutlass.Uint64(0xFFFF))
                    winner_inverse_idx = cutlass.Uint32(
                        prims.redux_sync(
                            cutlass.Int32(inverse_idx),
                            prims.ReductionKind.UMAX,
                            _FULL_MASK,
                        )
                    )
                    cand = cutlass.Int32(0)
                    if is_max and inverse_idx == winner_inverse_idx:
                        cand = cutlass.Int32(1)
                    ball2 = cutlass.Uint32(
                        prims.vote_sync(_FULL_MASK, cand != cutlass.Int32(0), prims.VoteSync.BALLOT)
                    )
                    owner = cutlass.Int32(cute.arch.popc(ball2 - cutlass.Uint32(1)))
                head_raw = cutlass.Float32(0.0)
                for v in cutlass.range_constexpr(_VEC):
                    if lane * cutlass.Int32(_VEC) + cutlass.Int32(v) == head_idx:
                        head_raw = raw_scores[v]
                winner_idx = cute.arch.shuffle_sync(head_idx, owner)
                winner_raw = cute.arch.shuffle_sync(head_raw, owner)
                if lane == owner:
                    for v in cutlass.range_constexpr(_VEC - 1):
                        keys[v] = keys[v + 1]
                    keys[_VEC - 1] = cutlass.Uint64(0)
                    head_idx = cutlass.Int32(65535) - cutlass.Int32(
                        keys[0] & cutlass.Uint64(0xFFFF)
                    )
                if lane == cutlass.Int32(k):
                    my_raw = winner_raw
                    my_selected_idx = winner_idx
        else:
            for k in cutlass.range_constexpr(_TOP_K):
                score_key = cutlass.Uint32(keys[0] >> cutlass.Uint64(32))
                warp_score_key = cutlass.Uint32(
                    prims.redux_sync(cutlass.Int32(score_key), prims.ReductionKind.UMAX, _FULL_MASK)
                )
                inverse_idx = cutlass.Uint32(0)
                if score_key == warp_score_key:
                    inverse_idx = cutlass.Uint32(keys[0] & cutlass.Uint64(0xFFFF))
                winner_inverse_idx = cutlass.Uint32(
                    prims.redux_sync(
                        cutlass.Int32(inverse_idx), prims.ReductionKind.UMAX, _FULL_MASK
                    )
                )
                winner_idx = cutlass.Int32(65535) - cutlass.Int32(winner_inverse_idx)
                winner_raw_local = cutlass.Float32(0.0)
                if score_key == warp_score_key and inverse_idx == winner_inverse_idx:
                    for v in cutlass.range_constexpr(_VEC):
                        if lane * cutlass.Int32(_VEC) + cutlass.Int32(v) == winner_idx:
                            winner_raw_local = raw_scores[v]
                    for v in cutlass.range_constexpr(_VEC - 1):
                        keys[v] = keys[v + 1]
                    keys[_VEC - 1] = cutlass.Uint64(0)
                winner_raw = cute.arch.shuffle_sync(
                    winner_raw_local, winner_idx >> cutlass.Int32(3)
                )
                if lane == cutlass.Int32(k):
                    my_raw = winner_raw
                    my_selected_idx = winner_idx
    weight_sum = _base._warp_sum_f32(my_raw)
    my_local = cutlass.Int32(-1)
    my_slot = cutlass.Int32(0)
    if valid_token and lane < cutlass.Int32(_TOP_K):
        expert = my_selected_idx
        expanded = token * cutlass.Int32(_TOP_K) + lane
        topk_indices[expanded] = expert
        topk_weights[expanded] = (
            my_raw * routed_scaling_factor / (weight_sum + cutlass.Float32(1.0e-20))
        )
        local = expert - local_expert_offset
        if local >= cutlass.Int32(0) and local < cutlass.Int32(_EXPERTS):
            my_local = local
            my_slot = prims.atomicrmw("add", expert_count.subview(local), 1)
    return my_local, my_slot


@cute.kernel
def fused_topk_routing_rep_kernel(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    hot_ids: cute.Tensor,
    topk_weights: cute.Tensor,
    topk_indices: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    num_tokens: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    routed_scaling_factor: cutlass.Float32,
    CLUSTER_CTAS: cutlass.Constexpr,
    LOGIT_SPLITS: cutlass.Constexpr,
    WAIT_FOR_ROUTER_PDL: cutlass.Constexpr,
    SIGNAL_GG1_PDL: cutlass.Constexpr,
    ONE_REDUX: cutlass.Constexpr,
    NREP: cutlass.Constexpr,
    expert_hist: Optional[cute.Tensor] = None,
):
    """Push cluster path with ``NREP`` replicated hot experts (module doc)."""
    tidx, _, _ = cute.arch.thread_idx()
    rank, _, _ = cute.arch.block_idx()
    lane = tidx & cutlass.Int32(_WARP - 1)
    warp = tidx >> cutlass.Int32(5)
    count = cutlass.Array(cutlass.Int32, _EXPERTS, space=cutlass.AddressSpace.smem, alignment=128)
    all_counts = cutlass.Array(
        cutlass.Int32,
        CLUSTER_CTAS * _EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    base = cutlass.Array(cutlass.Int32, _EXPERTS, space=cutlass.AddressSpace.smem, alignment=128)
    hot = cutlass.Array(cutlass.Int32, NREP, space=cutlass.AddressSpace.smem, alignment=16)
    hot_active = cutlass.Array(cutlass.Int32, NREP, space=cutlass.AddressSpace.smem, alignment=16)
    assign = cutlass.Array(cutlass.Int32, NREP, space=cutlass.AddressSpace.smem, alignment=16)
    warp_cold = cutlass.Array(cutlass.Int32, 8, space=cutlass.AddressSpace.smem, alignment=16)
    slot_bits = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    count_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    base_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    if tidx < cutlass.Int32(_EXPERTS):
        count.subview(tidx).store(cutlass.Int32(0))
    if tidx < cutlass.Int32(NREP):
        # Router-independent layer constants, read before the PDL wait.
        hot.subview(tidx).store(cutlass.Int32(hot_ids[tidx]))
        hot_active.subview(tidx).store(cutlass.Int32(0))
    if tidx < cutlass.Int32(4):
        slot_bits.subview(tidx).store(cutlass.Int32(0))
    if tidx == cutlass.Int32(0):
        prims.mbarrier_init(count_bar.subview(0), 1)
        prims.mbarrier_init(base_bar.subview(0), 1)
        prims.fence_mbarrier_init()
        if rank == cutlass.Int32(0):
            prims.mbarrier_arrive_expect_tx(count_bar.subview(0), CLUSTER_CTAS * _EXPERTS * 4)
        prims.mbarrier_arrive_expect_tx(base_bar.subview(0), _EXPERTS * 4)
    bias_vals = cutlass.Array(cutlass.Float32, _VEC)
    for v in cutlass.range_constexpr(_VEC):
        bias_vals[v] = cutlass.Float32(routing_bias[lane * cutlass.Int32(_VEC) + cutlass.Int32(v)])
    prims.barrier_cta_sync(0)
    prims.barrier_cluster_arrive()
    token = rank * cutlass.Int32(_WARPS) + warp
    valid = token < num_tokens
    if cutlass.const_expr(WAIT_FOR_ROUTER_PDL):
        cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(SIGNAL_GG1_PDL):
        cute.arch.griddepcontrol_launch_dependents()
    my_expert, my_slot = _select_topk_global(
        logits,
        LOGIT_SPLITS,
        ONE_REDUX,
        bias_vals,
        topk_weights,
        topk_indices,
        count,
        token,
        lane,
        valid,
        cutlass.Int32(0),
        routed_scaling_factor,
    )
    prims.barrier_cta_sync(0)  # this CTA's 256-bin histogram atomics are complete
    prims.barrier_cluster_wait()
    if tidx < cutlass.Int32(_EXPERTS):
        _compat48.store_async(
            prims.mapa(all_counts, cutlass.Int32(0)).subview(rank * cutlass.Int32(_EXPERTS) + tidx),
            count.subview(tidx).load(),
            prims.mapa(count_bar.subview(0), cutlass.Int32(0)),
        )
    if rank == cutlass.Int32(0) and tidx < cutlass.Int32(_EXPERTS):
        # One global expert e = tidx per thread; warps 2r, 2r+1 cover EP rank r's 64 experts.
        while not prims.mbarrier_try_wait_parity(count_bar.subview(0), 0):
            pass
        rank_prefix = cutlass.Array(cutlass.Int32, CLUSTER_CTAS)
        total = cutlass.Int32(0)
        for r in cutlass.range_constexpr(CLUSTER_CTAS):
            rank_prefix[r] = total
            total = total + all_counts.subview(cutlass.Int32(r * _EXPERTS) + tidx).load()
        active = total != cutlass.Int32(0)
        my_hot = cutlass.Int32(-1)
        for h in cutlass.range_constexpr(NREP):
            if hot.subview(h).load() == tidx:
                my_hot = cutlass.Int32(h)
        cold_i = cutlass.Int32(0)
        if active:
            if my_hot < cutlass.Int32(0):
                cold_i = cutlass.Int32(1)
        cold_ball = cutlass.Uint32(
            prims.vote_sync(_FULL_MASK, cold_i != cutlass.Int32(0), prims.VoteSync.BALLOT)
        )
        if lane == cutlass.Int32(0):
            warp_cold.subview(warp).store(cutlass.Int32(cute.arch.popc(cold_ball)))
        if active and my_hot >= cutlass.Int32(0):
            hot_active.subview(my_hot).store(cutlass.Int32(1))
        prims.barrier_cta_sync(1, thread_count=_EXPERTS)
        if tidx == cutlass.Int32(0):
            # Deterministic greedy, identical on every EP rank (same TopK, same totals).
            n0 = warp_cold.subview(0).load() + warp_cold.subview(1).load()
            n1 = warp_cold.subview(2).load() + warp_cold.subview(3).load()
            n2 = warp_cold.subview(4).load() + warp_cold.subview(5).load()
            n3 = warp_cold.subview(6).load() + warp_cold.subview(7).load()
            for h in cutlass.range_constexpr(NREP):
                a = cutlass.Int32(-1)
                if hot_active.subview(h).load() != cutlass.Int32(0):
                    a = cutlass.Int32(0)
                    best = n0
                    if n1 < best:
                        a = cutlass.Int32(1)
                        best = n1
                    if n2 < best:
                        a = cutlass.Int32(2)
                        best = n2
                    if n3 < best:
                        a = cutlass.Int32(3)
                        best = n3
                    if a == cutlass.Int32(0):
                        n0 = n0 + cutlass.Int32(1)
                    if a == cutlass.Int32(1):
                        n1 = n1 + cutlass.Int32(1)
                    if a == cutlass.Int32(2):
                        n2 = n2 + cutlass.Int32(1)
                    if a == cutlass.Int32(3):
                        n3 = n3 + cutlass.Int32(1)
                assign.subview(h).store(a)
        prims.barrier_cta_sync(1, thread_count=_EXPERTS)
        my_ep = local_expert_offset >> cutlass.Int32(6)
        ext = cutlass.Int32(-1)
        if active:
            if my_hot >= cutlass.Int32(0):
                if assign.subview(my_hot).load() == my_ep:
                    ext = cutlass.Int32(_LOCAL_EXPERTS) + my_hot
            elif (tidx >> cutlass.Int32(6)) == my_ep:
                ext = tidx - local_expert_offset
        if ext >= cutlass.Int32(0):
            prims.atomicrmw(
                "or",
                slot_bits.subview(ext >> cutlass.Int32(5)),
                cutlass.Int32(1) << (ext & cutlass.Int32(31)),
            )
        prims.barrier_cta_sync(1, thread_count=_EXPERTS)
        w0 = cutlass.Uint32(slot_bits.subview(0).load())
        w1 = cutlass.Uint32(slot_bits.subview(1).load())
        w2 = cutlass.Uint32(slot_bits.subview(2).load())
        w3 = cutlass.Uint32(slot_bits.subview(3).load())
        tile = cutlass.Int32(0)
        expert_base = cutlass.Int32(-1)
        if ext >= cutlass.Int32(0):
            # tile = number of set slot bits below `ext` (word-wise masks: full / partial / none)
            word = ext >> cutlass.Int32(5)
            below = (cutlass.Uint32(1) << cutlass.Uint32(ext & cutlass.Int32(31))) - cutlass.Uint32(
                1
            )
            m0 = cutlass.Uint32(0xFFFFFFFF)
            if word == cutlass.Int32(0):
                m0 = below
            m1 = cutlass.Uint32(0)
            if word > cutlass.Int32(1):
                m1 = cutlass.Uint32(0xFFFFFFFF)
            if word == cutlass.Int32(1):
                m1 = below
            m2 = cutlass.Uint32(0)
            if word > cutlass.Int32(2):
                m2 = cutlass.Uint32(0xFFFFFFFF)
            if word == cutlass.Int32(2):
                m2 = below
            m3 = cutlass.Uint32(0)
            if word == cutlass.Int32(3):
                m3 = below
            tile = (
                cutlass.Int32(cute.arch.popc(w0 & m0))
                + cutlass.Int32(cute.arch.popc(w1 & m1))
                + cutlass.Int32(cute.arch.popc(w2 & m2))
                + cutlass.Int32(cute.arch.popc(w3 & m3))
            )
            expert_base = tile * cutlass.Int32(_TILE)
        for r in cutlass.range_constexpr(CLUSTER_CTAS):
            value = cutlass.Int32(-1)
            if ext >= cutlass.Int32(0):
                value = expert_base + rank_prefix[r]
            _compat48.store_async(
                prims.mapa(base, cutlass.Int32(r)).subview(tidx),
                value,
                prims.mapa(base_bar.subview(0), cutlass.Int32(r)),
            )
        if ext >= cutlass.Int32(0):
            tile_idx_to_expert_idx[tile] = ext
            tile_idx_to_mn_limit[tile] = expert_base + total
        if cutlass.const_expr(expert_hist is not None):
            if (tidx >> cutlass.Int32(6)) == my_ep:
                expert_hist[tidx - local_expert_offset] = total
        if tidx == cutlass.Int32(0):
            active_tiles = (
                cutlass.Int32(cute.arch.popc(w0))
                + cutlass.Int32(cute.arch.popc(w1))
                + cutlass.Int32(cute.arch.popc(w2))
                + cutlass.Int32(cute.arch.popc(w3))
            )
            num_non_exiting_tiles[0] = active_tiles
            total_num_padded_tokens[0] = active_tiles * cutlass.Int32(_TILE)
    if (valid and lane < cutlass.Int32(_TOP_K)) or tidx == cutlass.Int32(0):
        while not prims.mbarrier_try_wait_parity(base_bar.subview(0), 0):
            pass
    if valid and lane < cutlass.Int32(_TOP_K):
        expanded = token * cutlass.Int32(_TOP_K) + lane
        permuted = cutlass.Int32(-1)
        if my_expert >= cutlass.Int32(0):
            b = base.subview(my_expert).load()
            if b >= cutlass.Int32(0):
                permuted = b + my_slot
                permuted_idx_to_expanded_idx[permuted] = expanded
        expanded_idx_to_permuted_idx[expanded] = permuted


fused_topk_routing_rep_kernel.set_name_prefix(
    "glm52_moe_topk_routing_rep", remove_cutlass_symbol=True
)


@cute.jit
def launch_fused_topk_routing_rep(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    hot_ids: cute.Tensor,
    topk_weights: cute.Tensor,
    topk_indices: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    num_tokens: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    routed_scaling_factor: cutlass.Float32,
    CLUSTER_CTAS: cutlass.Constexpr,
    LOGIT_SPLITS: cutlass.Constexpr,
    WAIT_FOR_ROUTER_PDL: cutlass.Constexpr,
    SIGNAL_GG1_PDL: cutlass.Constexpr,
    ONE_REDUX: cutlass.Constexpr,
    NREP: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
    expert_hist: Optional[cute.Tensor] = None,
):
    fused_topk_routing_rep_kernel(
        logits,
        routing_bias,
        hot_ids,
        topk_weights,
        topk_indices,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
        num_tokens,
        local_expert_offset,
        routed_scaling_factor,
        CLUSTER_CTAS,
        LOGIT_SPLITS,
        WAIT_FOR_ROUTER_PDL,
        SIGNAL_GG1_PDL,
        ONE_REDUX,
        NREP,
        expert_hist,
    ).launch(
        grid=(CLUSTER_CTAS, 1, 1),
        block=(_BLOCK, 1, 1),
        cluster=(CLUSTER_CTAS, 1, 1),
        use_pdl=WAIT_FOR_ROUTER_PDL,
        stream=stream,
    )


def max_tiles(num_tokens: int, nrep: int) -> int:
    return min(_LOCAL_EXPERTS + nrep, num_tokens * _TOP_K)


def allocate_outputs(num_tokens: int, nrep: int, device: str = "cuda") -> Dict[str, "torch.Tensor"]:
    """``moe_topk_routing.allocate_outputs`` with ``64 + nrep`` tile capacity."""
    cluster_ctas_for_tokens(num_tokens)
    tiles = max_tiles(num_tokens, nrep)
    i32 = dict(dtype=torch.int32, device=device)
    return {
        "topk_weights": torch.empty(num_tokens * _TOP_K, dtype=torch.float32, device=device),
        "topk_indices": torch.empty(num_tokens * _TOP_K, **i32),
        "tile_idx_to_expert_idx": torch.empty(tiles, **i32),
        "tile_idx_to_mn_limit": torch.empty(tiles, **i32),
        "expanded_idx_to_permuted_idx": torch.empty(num_tokens * _TOP_K, **i32),
        "permuted_idx_to_expanded_idx": torch.empty(tiles * _TILE, **i32),
        "total_num_padded_tokens": torch.empty(1, **i32),
        "num_non_exiting_tiles": torch.empty(1, **i32),
    }


def uses_rep_path(num_tokens: int) -> bool:
    """The replica kernel serves the push cluster path's rows (9..64)."""
    return _push.uses_push_path(num_tokens)


_CACHE: Dict[tuple, object] = {}


def _validate(logits, routing_bias, hot_ids, outputs, nrep, expert_hist):
    if not 0 < nrep <= MAX_NREP:
        raise ValueError(f"nrep must be in [1, {MAX_NREP}], got {nrep}")
    if logits.ndim == 3:
        num_tokens, logit_splits, experts = logits.shape
        if logit_splits not in (2, 4, 8):
            raise ValueError("split logits must have 2, 4, or 8 K partitions")
    elif logits.ndim == 2:
        num_tokens, experts = logits.shape
        logit_splits = 1
    else:
        raise ValueError("logits must be [T,256] or [T,2|4|8,256]")
    if not uses_rep_path(int(num_tokens)):
        raise ValueError(f"the replica path serves 9..64 rows, got {int(num_tokens)}")
    if experts != _EXPERTS or logits.dtype != torch.float32 or not logits.is_contiguous():
        raise ValueError("logits must be contiguous FP32 over 256 experts")
    if logits.data_ptr() % 16:
        raise ValueError("logits must be 16-byte aligned")
    if tuple(routing_bias.shape) != (_EXPERTS,) or routing_bias.dtype != torch.float32:
        raise ValueError("routing_bias must be FP32 [256]")
    if (
        tuple(hot_ids.shape) != (nrep,)
        or hot_ids.dtype != torch.int32
        or not hot_ids.is_contiguous()
    ):
        raise ValueError(f"hot_ids must be contiguous int32 [{nrep}]")
    tiles = max_tiles(int(num_tokens), nrep)
    expect = {
        "topk_weights": (int(num_tokens) * _TOP_K,),
        "topk_indices": (int(num_tokens) * _TOP_K,),
        "tile_idx_to_expert_idx": (tiles,),
        "tile_idx_to_mn_limit": (tiles,),
        "expanded_idx_to_permuted_idx": (int(num_tokens) * _TOP_K,),
        "permuted_idx_to_expanded_idx": (tiles * _TILE,),
        "total_num_padded_tokens": (1,),
        "num_non_exiting_tiles": (1,),
    }
    for name in OUTPUT_ORDER:
        if tuple(outputs[name].shape) != expect[name] or not outputs[name].is_contiguous():
            raise ValueError(
                f"{name} must be contiguous {expect[name]}, got {tuple(outputs[name].shape)}"
            )
    _push._check_hist(expert_hist, logits.device)
    return int(num_tokens), int(logit_splits)


def compile_fused_topk_routing_rep(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    hot_ids: "torch.Tensor",
    outputs: Dict[str, "torch.Tensor"],
    *,
    nrep: int,
    wait_for_router_pdl: bool = False,
    signal_gg1_pdl: bool = False,
    expert_hist: Optional["torch.Tensor"] = None,
):
    """Compile (cached per T / splits / PDL roles / nrep / hist form). The handle takes
    ``(logits, bias, hot_ids, *outputs, num_tokens, local_expert_offset,
    routed_scaling_factor, stream[, expert_hist])``."""
    num_tokens, logit_splits = _validate(logits, routing_bias, hot_ids, outputs, nrep, expert_hist)
    device = logits.device.index if logits.device.index is not None else torch.cuda.current_device()
    key = (
        device,
        num_tokens,
        logit_splits,
        wait_for_router_pdl,
        signal_gg1_pdl,
        _ONE_REDUX,
        nrep,
        expert_hist is not None,
    )
    if key not in _CACHE:
        cutlass.cuda.initialize_cuda_context(device)
        _CACHE[key] = cute.compile(
            launch_fused_topk_routing_rep,
            _base._as_tensor(logits.reshape(-1), assumed_align=16),
            _base._as_tensor(routing_bias),
            _base._as_tensor(hot_ids, assumed_align=16),
            *(_base._as_tensor(outputs[name]) for name in OUTPUT_ORDER),
            cutlass.Int32(num_tokens),
            cutlass.Int32(0),
            cutlass.Float32(1.0),
            cluster_ctas_for_tokens(num_tokens),
            logit_splits,
            wait_for_router_pdl,
            signal_gg1_pdl,
            _ONE_REDUX,
            nrep,
            cutlass_torch.current_stream(),
            *(
                (_base._as_tensor(expert_hist, assumed_align=16),)
                if expert_hist is not None
                else ()
            ),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    return _CACHE[key]


def run_fused_topk_routing_rep(
    logits,
    routing_bias,
    hot_ids,
    outputs,
    *,
    nrep,
    local_expert_offset,
    routed_scaling_factor,
    wait_for_router_pdl=False,
    signal_gg1_pdl=False,
    expert_hist=None,
) -> None:
    if local_expert_offset not in (0, 64, 128, 192):
        raise ValueError(
            f"local_expert_offset must select one EP4 shard, got {local_expert_offset}"
        )
    compiled = compile_fused_topk_routing_rep(
        logits,
        routing_bias,
        hot_ids,
        outputs,
        nrep=nrep,
        wait_for_router_pdl=wait_for_router_pdl,
        signal_gg1_pdl=signal_gg1_pdl,
        expert_hist=expert_hist,
    )
    compiled(
        _base._as_tensor(logits.reshape(-1), assumed_align=16),
        _base._as_tensor(routing_bias),
        _base._as_tensor(hot_ids, assumed_align=16),
        *(_base._as_tensor(outputs[name]) for name in OUTPUT_ORDER),
        cutlass.Int32(int(logits.shape[0])),
        cutlass.Int32(local_expert_offset),
        cutlass.Float32(routed_scaling_factor),
        cutlass_torch.current_stream(),
        *((_base._as_tensor(expert_hist, assumed_align=16),) if expert_hist is not None else ()),
    )


def reference_assignment(topk_indices: "torch.Tensor", hot_ids: list) -> list:
    """Torch/host definition of the greedy: EP rank serving each hot slot (-1 = inactive/unused)."""
    act = set(int(e) for e in topk_indices.reshape(-1).tolist())
    hot = [int(h) for h in hot_ids]
    n = [0, 0, 0, 0]
    for e in act:
        if e not in hot:
            n[e // 64] += 1
    out = []
    for h in hot:
        if h >= 0 and h in act:
            a = min(range(4), key=lambda r: (n[r], r))
            n[a] += 1
            out.append(a)
        else:
            out.append(-1)
    return out


def check_rank(
    outputs: dict,
    topk_indices: "torch.Tensor",
    hot_ids: list,
    assign: list,
    offset: int,
) -> list:
    """Failures of one EP rank's maps against the replica contract (row sets, slots, limits)."""
    fail = []
    ep = offset // 64
    ids = topk_indices.reshape(-1).tolist()
    hot = [int(h) for h in hot_ids]
    want = {}  # ext slot -> sorted expanded rows
    for x, e in enumerate(ids):
        if e in hot:
            h = hot.index(e)
            if assign[h] == ep:
                want.setdefault(64 + h, []).append(x)
        elif e // 64 == ep:
            want.setdefault(e - offset, []).append(x)
    slots = sorted(want)
    tiles = int(outputs["num_non_exiting_tiles"][0])
    if tiles != len(slots):
        return [f"rank {ep}: {tiles} tiles, want {len(slots)}"]
    if int(outputs["total_num_padded_tokens"][0]) != tiles * _TILE:
        fail.append(f"rank {ep}: total_num_padded_tokens")
    t2e = outputs["tile_idx_to_expert_idx"][:tiles].tolist()
    lim = outputs["tile_idx_to_mn_limit"][:tiles].tolist()
    p2e = outputs["permuted_idx_to_expanded_idx"].tolist()
    e2p = outputs["expanded_idx_to_permuted_idx"].tolist()
    if t2e != slots:
        fail.append(f"rank {ep}: tile slots {t2e} != {slots}")
        return fail
    for t, s in enumerate(slots):
        rows = p2e[t * _TILE : lim[t]]
        if lim[t] - t * _TILE != len(want[s]) or sorted(rows) != want[s]:
            fail.append(f"rank {ep}: slot {s} rows differ")
        for k, x in enumerate(rows):
            if e2p[x] != t * _TILE + k:
                fail.append(f"rank {ep}: inverse map at {x}")
                break
    mine = set(x for v in want.values() for x in v)
    for x in range(len(ids)):
        if x not in mine and e2p[x] != -1:
            fail.append(f"rank {ep}: expanded {x} mapped but not this rank's")
            break
    return fail
