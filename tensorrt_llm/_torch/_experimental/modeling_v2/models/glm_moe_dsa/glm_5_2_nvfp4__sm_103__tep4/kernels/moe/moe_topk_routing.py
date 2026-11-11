# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused GLM-5.2 sigmoid TopK and MoE routing metadata generation.

This kernel specializes the production ``deepseek_v3_topk`` followed by
``routingIndices{Block,Cluster}Kernel`` path to the GLM-5.2 inference contract:

* 256 global experts, TopK=8, EP=4 and 64 contiguous experts per rank;
* no group-limited routing; selection uses ``sigmoid(logit) + bias``;
* output weights use the unbiased sigmoid values, renormalized across TopK and
  multiplied by ``routed_scaling_factor``;
* at most 64 tokens and a grouped-GEMM M tile of 128, hence each active local
  expert contributes exactly one grouped-GEMM tile.

The important fusion boundary is the register-resident TopK result.  A warp owns
one token, reduces its eight winners, and immediately contributes those winners
to the local-expert histogram.  The unfused path materializes TopK indices to
global memory and the routing kernel reads them back.

The logits input may also contain two, four, or eight contiguous split-K partials
per token.
In that mode each lane adds all split-K partials for its experts in registers
before sigmoid, folding the router reduction into this kernel.  This removes
the standalone reduction launch and the reduced-logits global-memory
write/read round trip.

There are two launch shapes:

* ``fused_topk_routing_block_kernel`` serves T<=16 (small decode batches);
* ``fused_topk_routing_cluster_kernel`` serves 16*CLUSTER_CTAS tokens, with
  CLUSTER_CTAS in {2, 4}; T=64 therefore uses a 4-CTA cluster.

Within either path, two warps map the 64 local experts.  Since an expert has at
most T<=64 rows and TILE=128, active-expert prefixing is a ballot plus popcount,
not a generic block scan.  In the cluster path CTA 0 gathers just 64 histogram
bins from distributed shared memory, computes the global prefix once, and
writes each CTA's expert base back through DSMEM.

The row order within one expert is intentionally unspecified: shared-memory
atomic adds claim its slots, matching TensorRT-LLM's routing contract.  Tests
therefore compare row sets and inverse maps rather than atomic claim order.

The optional output zeroing uses 128-bit stores.  Its public wrapper therefore
requires a 16-byte-aligned ``zero_output`` base pointer; every vector and row
offset preserves that alignment because a row contains 6144 BF16 elements.

``permuted_idx_to_expanded_idx`` is defined only for real rows in each expert
tile: ``[tile * TILE, tile_idx_to_mn_limit[tile])``.  Padding rows are
intentionally left untouched because FC1 and FC2 bound their reads by the
per-tile limit.  ``expanded_idx_to_permuted_idx`` still guarantees ``-1`` for
winners owned by another EP rank.

The two PDL roles are independent.  ``WAIT_FOR_ROUTER_PDL`` makes this kernel a
programmatic secondary of router GEMM and waits immediately before the first
logits load.  ``SIGNAL_GG1_PDL`` makes it a producer for GG1 and signals after
the optional router wait.  The signal conveys launch readiness, not
routing-metadata readiness: GG1 must execute its own PDL wait before reading any
output listed above.
"""

import math
from typing import Callable, Dict, Optional, Tuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental import primitives as prims

_BLOCK = 512
_WARP = 32
_WARPS = _BLOCK // _WARP
_FULL_MASK = 0xFFFFFFFF
_EXPERTS = 256
_TOP_K = 8
_LOCAL_EXPERTS = 64
_MAX_TOKENS = 64
_TILE = 128
_VEC_SIZE = _EXPERTS // _WARP
_HIDDEN = 6144
_BF16_PER_ZERO_STORE = 8
_ZERO_STORE_BYTES = 16
_ZERO_VECTORS_PER_ROW = _HIDDEN // _BF16_PER_ZERO_STORE

TopKRoutingFn = Callable[..., None]
_CACHE: Dict[Tuple[int, int, int, bool, bool, bool], TopKRoutingFn] = {}


@cute.jit
def _ballot_exclusive(active):
    """Return the exclusive active-lane prefix and active count for a warp."""
    mask = cutlass.Uint32(prims.vote_sync(_FULL_MASK, active, prims.VoteSync.BALLOT))
    prefix = cutlass.Int32(cute.arch.popc(mask & cute.arch.lanemask_lt()))
    return prefix, cutlass.Int32(cute.arch.popc(mask))


@cute.jit
def _warp_sum_f32(value):
    """Return a float32 sum to every lane.

    PTX ``redux.sync.add`` has only an integer form.  Using its ``ADD`` enum on
    float32 would reinterpret the bits, so the floating-point normalization sum
    deliberately uses the five-stage butterfly sequence.
    """
    value = value + cute.arch.shuffle_sync_bfly(value, 16)
    value = value + cute.arch.shuffle_sync_bfly(value, 8)
    value = value + cute.arch.shuffle_sync_bfly(value, 4)
    value = value + cute.arch.shuffle_sync_bfly(value, 2)
    value = value + cute.arch.shuffle_sync_bfly(value, 1)
    return value


@cute.jit
def _pack_score_index(score, index):
    """Pack ``(score, -index)`` into the production unsigned ordering key."""
    bits = cutlass.Float32(score).bitcast(cutlass.Uint32)
    mask = cutlass.Uint32(0x80000000)
    if (bits & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0):
        mask = cutlass.Uint32(0xFFFFFFFF)
    ordered = bits ^ mask
    return (cutlass.Uint64(ordered) << cutlass.Uint64(32)) | cutlass.Uint64(
        cutlass.Int32(65535) - index
    )


@cute.jit
def _compare_swap_key_desc(keys, hi: cutlass.Constexpr, lo: cutlass.Constexpr):
    """Put the larger packed score/index key in slot hi."""
    if keys[lo] > keys[hi]:
        temporary = keys[hi]
        keys[hi] = keys[lo]
        keys[lo] = temporary


@cute.jit
def _sort8_key_desc(keys):
    """Nineteen-comparator network over packed production TopK keys."""
    _compare_swap_key_desc(keys, 0, 1)
    _compare_swap_key_desc(keys, 2, 3)
    _compare_swap_key_desc(keys, 4, 5)
    _compare_swap_key_desc(keys, 6, 7)
    _compare_swap_key_desc(keys, 0, 2)
    _compare_swap_key_desc(keys, 1, 3)
    _compare_swap_key_desc(keys, 4, 6)
    _compare_swap_key_desc(keys, 5, 7)
    _compare_swap_key_desc(keys, 1, 2)
    _compare_swap_key_desc(keys, 5, 6)
    _compare_swap_key_desc(keys, 0, 4)
    _compare_swap_key_desc(keys, 3, 7)
    _compare_swap_key_desc(keys, 1, 5)
    _compare_swap_key_desc(keys, 2, 6)
    _compare_swap_key_desc(keys, 1, 4)
    _compare_swap_key_desc(keys, 3, 6)
    _compare_swap_key_desc(keys, 2, 4)
    _compare_swap_key_desc(keys, 3, 5)
    _compare_swap_key_desc(keys, 3, 4)


@cute.jit
def _zero_owned_token_rows(
    zero_output: cute.Tensor,
    num_tokens: cutlass.Int32,
    first_token: cutlass.Int32,
    tidx: cutlass.Int32,
):
    """Clear this CTA's at-most-16 BF16 output rows with 128-bit stores."""
    zero_ptr = zero_output.iterator.raw_ptr()
    zero_128b = cutlass.Vector.from_elements(
        (cutlass.Int64(0), cutlass.Int64(0)), dtype=cutlass.Int64
    )
    local_vector = tidx
    while local_vector < cutlass.Int32(_WARPS * _ZERO_VECTORS_PER_ROW):
        token_in_cta = local_vector // cutlass.Int32(_ZERO_VECTORS_PER_ROW)
        token = first_token + token_in_cta
        if token < num_tokens:
            vector_in_row = local_vector - token_in_cta * cutlass.Int32(_ZERO_VECTORS_PER_ROW)
            offset = token * cutlass.Int32(_HIDDEN) + vector_in_row * cutlass.Int32(
                _BF16_PER_ZERO_STORE
            )
            prims.store_ext(zero_128b, zero_ptr + offset)
        local_vector += cutlass.Int32(_BLOCK)


@cute.jit
def _select_topk_and_count(
    logits: cute.Tensor,
    logit_splits: cutlass.Constexpr,
    routing_bias: cute.Tensor,
    topk_weights: cute.Tensor,
    topk_indices: cute.Tensor,
    expert_count,
    token: cutlass.Int32,
    lane: cutlass.Int32,
    valid_token,
    local_expert_offset: cutlass.Int32,
    routed_scaling_factor: cutlass.Float32,
):
    """Select one token's TopK and return this lane's routing claim.

    Every lane owns experts ``lane + 32*v``.  ``redux.sync`` performs the
    warp-wide score max and index min, and one indexed shuffle broadcasts the
    unbiased sigmoid value from the winning expert's owner lane.  Lanes 0..7
    then write the public TopK outputs and claim local-expert histogram slots.
    """
    keys = cutlass.Array(cutlass.Uint64, _VEC_SIZE)
    raw_scores = cutlass.Array(cutlass.Float32, _VEC_SIZE)
    my_raw = cutlass.Float32(0.0)
    my_selected_idx = cutlass.Int32(-1)

    for v in cutlass.range_constexpr(_VEC_SIZE):
        keys[v] = cutlass.Uint64(0)
        raw_scores[v] = cutlass.Float32(0.0)
    if valid_token:
        row = cutlass.Int64(token) * cutlass.Int64(_EXPERTS) * cutlass.Int64(logit_splits)
        for v in cutlass.range_constexpr(_VEC_SIZE):
            expert = lane + cutlass.Int32(v * _WARP)
            logit = cutlass.Float32(logits[row + cutlass.Int64(expert)])
            for split in cutlass.range_constexpr(1, logit_splits):
                logit = logit + cutlass.Float32(
                    logits[row + cutlass.Int64(split * _EXPERTS) + cutlass.Int64(expert)]
                )
            raw = cutlass.Float32(0.5) * cute.math.tanh(
                cutlass.Float32(0.5) * logit
            ) + cutlass.Float32(0.5)
            raw_scores[v] = raw
            keys[v] = _pack_score_index(raw + cutlass.Float32(routing_bias[expert]), expert)

        # Production sorts each lane's local candidates once, then shifts the head
        # only when that lane wins.  This avoids recompiling an eight-way rescan into
        # every one of the eight global TopK rounds.
        _sort8_key_desc(keys)

        for k in cutlass.range_constexpr(_TOP_K):
            score_key = cutlass.Uint32(keys[0] >> cutlass.Uint64(32))
            warp_score_key = cutlass.Uint32(
                prims.redux_sync(cutlass.Int32(score_key), prims.ReductionKind.UMAX, _FULL_MASK)
            )
            inverse_idx = cutlass.Uint32(0)
            if score_key == warp_score_key:
                inverse_idx = cutlass.Uint32(keys[0] & cutlass.Uint64(0xFFFF))
            winner_inverse_idx = cutlass.Uint32(
                prims.redux_sync(cutlass.Int32(inverse_idx), prims.ReductionKind.UMAX, _FULL_MASK)
            )
            winner_idx = cutlass.Int32(65535) - cutlass.Int32(winner_inverse_idx)

            winner_raw_local = cutlass.Float32(0.0)
            if score_key == warp_score_key and inverse_idx == winner_inverse_idx:
                for v in cutlass.range_constexpr(_VEC_SIZE):
                    if lane + cutlass.Int32(v * _WARP) == winner_idx:
                        winner_raw_local = raw_scores[v]
                for v in cutlass.range_constexpr(_VEC_SIZE - 1):
                    keys[v] = keys[v + 1]
                keys[_VEC_SIZE - 1] = cutlass.Uint64(0)
            winner_raw = cute.arch.shuffle_sync(
                winner_raw_local, winner_idx & cutlass.Int32(_WARP - 1)
            )
            # Lane k retains winner k.  This avoids a dynamic register-array index
            # after selection and halves the live TopK register footprint.
            if lane == cutlass.Int32(k):
                my_raw = winner_raw
                my_selected_idx = winner_idx

    weight_sum = _warp_sum_f32(my_raw)

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
        if local >= cutlass.Int32(0) and local < cutlass.Int32(_LOCAL_EXPERTS):
            my_local = local
            my_slot = prims.atomicrmw("add", expert_count.subview(local), 1)
    return my_local, my_slot


@cute.kernel
def fused_topk_routing_block_kernel(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    zero_output: cute.Tensor,
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
    LOGIT_SPLITS: cutlass.Constexpr,
    WAIT_FOR_ROUTER_PDL: cutlass.Constexpr,
    SIGNAL_GG1_PDL: cutlass.Constexpr,
    FUSE_ZERO: cutlass.Constexpr,
):
    """Single-CTA fused path for 1..16 tokens."""
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx & cutlass.Int32(_WARP - 1)
    warp = tidx >> cutlass.Int32(5)

    if cutlass.const_expr(FUSE_ZERO):
        _zero_owned_token_rows(zero_output, num_tokens, cutlass.Int32(0), tidx)

    count = cutlass.Array(
        cutlass.Int32,
        _LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    base = cutlass.Array(
        cutlass.Int32,
        _LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    warp_totals = cutlass.Array(
        cutlass.Int32,
        2,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )

    if tidx < cutlass.Int32(_LOCAL_EXPERTS):
        count.subview(tidx).store(cutlass.Int32(0))
    prims.barrier_cta_sync(0)

    token = warp
    valid = token < num_tokens
    if cutlass.const_expr(WAIT_FOR_ROUTER_PDL):
        # Shared initialization and token indexing are independent of router
        # logits, so perform them while this secondary grid is admitted early.
        # Wait immediately before the first dependent load.
        cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(SIGNAL_GG1_PDL):
        # Admit GG1 as soon as any upstream dependency is resolved.  GG1's own
        # wait protects every routing-metadata read.
        cute.arch.griddepcontrol_launch_dependents()
    my_local, my_slot = _select_topk_and_count(
        logits,
        LOGIT_SPLITS,
        routing_bias,
        topk_weights,
        topk_indices,
        count,
        token,
        lane,
        valid,
        local_expert_offset,
        routed_scaling_factor,
    )
    prims.barrier_cta_sync(0)

    if warp < cutlass.Int32(2):
        expert_count = count.subview(tidx).load()
        active = expert_count != cutlass.Int32(0)
        local_prefix, local_total = _ballot_exclusive(active)
        if lane == cutlass.Int32(_WARP - 1):
            warp_totals.subview(warp).store(local_total)
        prims.barrier_cta_sync(1, thread_count=_LOCAL_EXPERTS)

        first_warp_total = warp_totals.subview(0).load()
        tile = local_prefix
        if warp == cutlass.Int32(1):
            tile = tile + first_warp_total
        base.subview(tidx).store(tile * cutlass.Int32(_TILE))
        if active:
            tile_idx_to_expert_idx[tile] = tidx
            tile_idx_to_mn_limit[tile] = tile * cutlass.Int32(_TILE) + expert_count
        if tidx == cutlass.Int32(0):
            active_experts = first_warp_total + warp_totals.subview(1).load()
            num_non_exiting_tiles[0] = active_experts
            total_num_padded_tokens[0] = active_experts * cutlass.Int32(_TILE)
    prims.barrier_cta_sync(0)

    if valid and lane < cutlass.Int32(_TOP_K):
        expanded = token * cutlass.Int32(_TOP_K) + lane
        if my_local >= cutlass.Int32(0):
            permuted = base.subview(my_local).load() + my_slot
            expanded_idx_to_permuted_idx[expanded] = permuted
            permuted_idx_to_expanded_idx[permuted] = expanded
        else:
            expanded_idx_to_permuted_idx[expanded] = cutlass.Int32(-1)


@cute.kernel
def fused_topk_routing_cluster_kernel(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    zero_output: cute.Tensor,
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
    FUSE_ZERO: cutlass.Constexpr,
):
    """Cluster fused path; one CTA owns 16 tokens and one warp owns a token."""
    tidx, _, _ = cute.arch.thread_idx()
    rank, _, _ = cute.arch.block_idx()
    lane = tidx & cutlass.Int32(_WARP - 1)
    warp = tidx >> cutlass.Int32(5)

    if cutlass.const_expr(FUSE_ZERO):
        _zero_owned_token_rows(
            zero_output,
            num_tokens,
            rank * cutlass.Int32(_WARPS),
            tidx,
        )

    count = cutlass.Array(
        cutlass.Int32,
        _LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    base = cutlass.Array(
        cutlass.Int32,
        _LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    warp_totals = cutlass.Array(
        cutlass.Int32,
        2,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )

    if tidx < cutlass.Int32(_LOCAL_EXPERTS):
        count.subview(tidx).store(cutlass.Int32(0))
    prims.barrier_cta_sync(0)

    token = rank * cutlass.Int32(_WARPS) + warp
    valid = token < num_tokens
    if cutlass.const_expr(WAIT_FOR_ROUTER_PDL):
        # Match the block path: overlap independent setup with router GEMM and
        # wait only at the first logits-dependent operation.
        cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(SIGNAL_GG1_PDL):
        # Match the block path's early admission of GG1.
        cute.arch.griddepcontrol_launch_dependents()
    my_local, my_slot = _select_topk_and_count(
        logits,
        LOGIT_SPLITS,
        routing_bias,
        topk_weights,
        topk_indices,
        count,
        token,
        lane,
        valid,
        local_expert_offset,
        routed_scaling_factor,
    )

    # Ordered cluster barrier publishes every CTA's histogram to DSMEM.
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()

    if rank == cutlass.Int32(0) and tidx < cutlass.Int32(_LOCAL_EXPERTS):
        rank_prefix = cutlass.Array(cutlass.Int32, CLUSTER_CTAS)
        total = cutlass.Int32(0)
        for remote_rank in cutlass.range_constexpr(CLUSTER_CTAS):
            rank_prefix[remote_rank] = total
            total = total + prims.mapa(count.subview(tidx), cutlass.Int32(remote_rank)).load()

        active = total != cutlass.Int32(0)
        local_prefix, local_total = _ballot_exclusive(active)
        if lane == cutlass.Int32(_WARP - 1):
            warp_totals.subview(warp).store(local_total)
        prims.barrier_cta_sync(1, thread_count=_LOCAL_EXPERTS)

        first_warp_total = warp_totals.subview(0).load()
        tile = local_prefix
        if warp == cutlass.Int32(1):
            tile = tile + first_warp_total
        if active:
            tile_idx_to_expert_idx[tile] = tidx
            tile_idx_to_mn_limit[tile] = tile * cutlass.Int32(_TILE) + total

        expert_base = tile * cutlass.Int32(_TILE)
        for remote_rank in cutlass.range_constexpr(CLUSTER_CTAS):
            prims.mapa(base.subview(tidx), cutlass.Int32(remote_rank)).store(
                expert_base + rank_prefix[remote_rank]
            )

        if tidx == cutlass.Int32(0):
            active_experts = first_warp_total + warp_totals.subview(1).load()
            num_non_exiting_tiles[0] = active_experts
            total_num_padded_tokens[0] = active_experts * cutlass.Int32(_TILE)

    # Publish controller-written per-rank bases before the winner lanes consume them.
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()

    if valid and lane < cutlass.Int32(_TOP_K):
        expanded = token * cutlass.Int32(_TOP_K) + lane
        if my_local >= cutlass.Int32(0):
            permuted = base.subview(my_local).load() + my_slot
            expanded_idx_to_permuted_idx[expanded] = permuted
            permuted_idx_to_expanded_idx[permuted] = expanded
        else:
            expanded_idx_to_permuted_idx[expanded] = cutlass.Int32(-1)


fused_topk_routing_block_kernel.set_name_prefix(
    "glm52_moe_topk_routing_block", remove_cutlass_symbol=True
)
fused_topk_routing_cluster_kernel.set_name_prefix(
    "glm52_moe_topk_routing_cluster", remove_cutlass_symbol=True
)


@cute.jit
def launch_fused_topk_routing(
    logits: cute.Tensor,
    routing_bias: cute.Tensor,
    zero_output: cute.Tensor,
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
    TOKEN_CTA_MODE: cutlass.Constexpr,
    LOGIT_SPLITS: cutlass.Constexpr,
    WAIT_FOR_ROUTER_PDL: cutlass.Constexpr,
    SIGNAL_GG1_PDL: cutlass.Constexpr,
    FUSE_ZERO: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    """Launch the block or cluster specialization selected at compile time."""
    if cutlass.const_expr(TOKEN_CTA_MODE):
        fused_topk_routing_block_kernel(
            logits,
            routing_bias,
            zero_output,
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
            LOGIT_SPLITS,
            WAIT_FOR_ROUTER_PDL,
            SIGNAL_GG1_PDL,
            FUSE_ZERO,
        ).launch(
            grid=(1, 1, 1),
            block=(_BLOCK, 1, 1),
            use_pdl=WAIT_FOR_ROUTER_PDL,
            stream=stream,
        )
    else:
        fused_topk_routing_cluster_kernel(
            logits,
            routing_bias,
            zero_output,
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
            FUSE_ZERO,
        ).launch(
            grid=(CLUSTER_CTAS, 1, 1),
            block=(_BLOCK, 1, 1),
            cluster=(CLUSTER_CTAS, 1, 1),
            use_pdl=WAIT_FOR_ROUTER_PDL,
            stream=stream,
        )


def cluster_ctas_for_tokens(num_tokens: int) -> int:
    """Return the smallest supported launch shape for ``num_tokens``."""
    if not 0 < num_tokens <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [1, {_MAX_TOKENS}], got {num_tokens}")
    if num_tokens <= _WARPS:
        return 1
    return 2 ** math.ceil(math.log2(math.ceil(num_tokens / _WARPS)))


def _max_tiles(num_tokens: int) -> int:
    return min(_LOCAL_EXPERTS, num_tokens * _TOP_K)


OUTPUT_ORDER = (
    "topk_weights",
    "topk_indices",
    "tile_idx_to_expert_idx",
    "tile_idx_to_mn_limit",
    "expanded_idx_to_permuted_idx",
    "permuted_idx_to_expanded_idx",
    "total_num_padded_tokens",
    "num_non_exiting_tiles",
)


def allocate_outputs(num_tokens: int, device: str = "cuda") -> Dict[str, "torch.Tensor"]:
    """Allocate all public outputs, including padded permutation capacity."""
    # Validate the supported token range before allocating any output storage.
    cluster_ctas_for_tokens(num_tokens)
    tiles = _max_tiles(num_tokens)
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


def _device_index(tensor: "torch.Tensor") -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


def _as_tensor(tensor: "torch.Tensor", *, assumed_align: int = 4) -> cute.Tensor:
    return from_dlpack(
        tensor, assumed_align=assumed_align, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=0)


def _validate_zero_output(zero_output: Optional["torch.Tensor"], num_tokens: int) -> None:
    """Validate storage used by the fused 128-bit BF16 zero stores."""
    if zero_output is None or (
        tuple(zero_output.shape) != (num_tokens, _HIDDEN)
        or zero_output.dtype != torch.bfloat16
        or not zero_output.is_cuda
        or not zero_output.is_contiguous()
    ):
        raise ValueError(
            f"zero_output must be contiguous CUDA BF16 [{num_tokens},{_HIDDEN}] when fuse_zero=True"
        )
    if zero_output.data_ptr() % _ZERO_STORE_BYTES:
        raise ValueError(
            f"zero_output data pointer must be {_ZERO_STORE_BYTES}-byte aligned "
            "for fused 128-bit stores"
        )


def _validate_serving_inputs(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    zero_output: Optional["torch.Tensor"],
    outputs: Dict[str, "torch.Tensor"],
    fuse_zero: bool,
) -> Tuple[int, int, int]:
    """Validate the exact GLM decode ABI shared by compile and run."""
    if logits.ndim == 2:
        num_tokens, experts = logits.shape
        logit_splits = 1
    elif logits.ndim == 3:
        num_tokens, logit_splits, experts = logits.shape
        if logit_splits not in (2, 4, 8):
            raise ValueError("split logits must have 2, 4, or 8 K partitions")
    else:
        raise ValueError("logits must be [T,256] or [T,2|4|8,256]")
    cluster_ctas_for_tokens(int(num_tokens))
    if experts != _EXPERTS:
        raise ValueError(f"logits must contain {_EXPERTS} experts, got {experts}")
    if logits.dtype != torch.float32 or not logits.is_cuda or not logits.is_contiguous():
        raise ValueError("logits must be contiguous CUDA FP32")
    if (
        tuple(routing_bias.shape) != (_EXPERTS,)
        or routing_bias.dtype != torch.float32
        or not routing_bias.is_cuda
        or not routing_bias.is_contiguous()
    ):
        raise ValueError(f"routing_bias must be contiguous CUDA FP32 [{_EXPERTS}]")
    if fuse_zero:
        _validate_zero_output(zero_output, int(num_tokens))

    tiles = _max_tiles(int(num_tokens))
    expected = {
        "topk_weights": ((num_tokens * _TOP_K,), torch.float32),
        "topk_indices": ((num_tokens * _TOP_K,), torch.int32),
        "tile_idx_to_expert_idx": ((tiles,), torch.int32),
        "tile_idx_to_mn_limit": ((tiles,), torch.int32),
        "expanded_idx_to_permuted_idx": ((num_tokens * _TOP_K,), torch.int32),
        "permuted_idx_to_expanded_idx": ((tiles * _TILE,), torch.int32),
        "total_num_padded_tokens": ((1,), torch.int32),
        "num_non_exiting_tiles": ((1,), torch.int32),
    }
    missing = [name for name in OUTPUT_ORDER if name not in outputs]
    if missing:
        raise ValueError(f"missing routing outputs: {missing}")
    device = _device_index(logits)
    for name in OUTPUT_ORDER:
        tensor = outputs[name]
        shape, dtype = expected[name]
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA {dtype} {shape}, got "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"device={tensor.device} contiguous={tensor.is_contiguous()}"
            )
        if _device_index(tensor) != device:
            raise ValueError("all routing tensors must share one CUDA device")
    if _device_index(routing_bias) != device:
        raise ValueError("all routing tensors must share one CUDA device")
    if fuse_zero and zero_output is not None and _device_index(zero_output) != device:
        raise ValueError("all routing tensors must share one CUDA device")
    return device, int(num_tokens), int(logit_splits)


def compile_fused_topk_routing(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    zero_output: Optional["torch.Tensor"],
    outputs: Dict[str, "torch.Tensor"],
    *,
    wait_for_router_pdl: bool = False,
    signal_gg1_pdl: bool = False,
    fuse_zero: bool = False,
) -> TopKRoutingFn:
    """Compile or retrieve one exact-T fused TopK/routing variant.

    ``wait_for_router_pdl`` enables both programmatic secondary launch and the
    in-kernel dependency wait. An explicit producer signal is optional for
    correctness because CUDA triggers dependents implicitly after the producer
    grid exits, but the preceding router must signal explicitly to enable early
    overlap. ``signal_gg1_pdl`` independently controls this kernel's producer
    role for the following grouped GEMM.
    """
    device, num_tokens, logit_splits = _validate_serving_inputs(
        logits, routing_bias, zero_output, outputs, fuse_zero
    )
    cluster_ctas = cluster_ctas_for_tokens(num_tokens)
    key = (
        device,
        num_tokens,
        logit_splits,
        wait_for_router_pdl,
        signal_gg1_pdl,
        fuse_zero,
    )
    if key not in _CACHE:
        cutlass.cuda.initialize_cuda_context(device)
        zero_arg = zero_output if fuse_zero else logits
        assert zero_arg is not None
        zero_arg_align = _ZERO_STORE_BYTES if fuse_zero else 4
        tensor_args = (
            _as_tensor(logits.reshape(-1)),
            _as_tensor(routing_bias),
            _as_tensor(zero_arg.reshape(-1), assumed_align=zero_arg_align),
            *(_as_tensor(outputs[name]) for name in OUTPUT_ORDER),
        )
        _CACHE[key] = cute.compile(
            launch_fused_topk_routing,
            *tensor_args,
            cutlass.Int32(num_tokens),
            cutlass.Int32(0),
            cutlass.Float32(1.0),
            cluster_ctas,
            cluster_ctas == 1,
            logit_splits,
            wait_for_router_pdl,
            signal_gg1_pdl,
            fuse_zero,
            cutlass_torch.current_stream(),
            options="--opt-level 2 --enable-tvm-ffi",
        )
    return _CACHE[key]


def run_fused_topk_routing(
    logits: "torch.Tensor",
    routing_bias: "torch.Tensor",
    zero_output: Optional["torch.Tensor"],
    outputs: Dict[str, "torch.Tensor"],
    *,
    local_expert_offset: int,
    routed_scaling_factor: float,
    wait_for_router_pdl: bool = False,
    signal_gg1_pdl: bool = False,
    fuse_zero: bool = False,
) -> None:
    """Launch the fused kernel on the current stream without allocating.

    For early router-to-TopK overlap, pass the same PDL mode used by the
    preceding router as ``wait_for_router_pdl``. A false value retains normal
    same-stream serialization. ``signal_gg1_pdl`` is an independent dependency
    edge for the following grouped GEMM.
    """
    if local_expert_offset not in (0, 64, 128, 192):
        raise ValueError(
            f"local_expert_offset must select one contiguous EP4 shard, got {local_expert_offset}"
        )
    compiled = compile_fused_topk_routing(
        logits,
        routing_bias,
        zero_output,
        outputs,
        wait_for_router_pdl=wait_for_router_pdl,
        signal_gg1_pdl=signal_gg1_pdl,
        fuse_zero=fuse_zero,
    )
    num_tokens = int(logits.shape[0])
    zero_arg = zero_output if fuse_zero else logits
    assert zero_arg is not None
    zero_arg_align = _ZERO_STORE_BYTES if fuse_zero else 4
    compiled(
        _as_tensor(logits.reshape(-1)),
        _as_tensor(routing_bias),
        _as_tensor(zero_arg.reshape(-1), assumed_align=zero_arg_align),
        *(_as_tensor(outputs[name]) for name in OUTPUT_ORDER),
        cutlass.Int32(num_tokens),
        cutlass.Int32(local_expert_offset),
        cutlass.Float32(routed_scaling_factor),
        cutlass_torch.current_stream(),
    )
