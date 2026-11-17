# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route GLM MTP post-topK expert IDs for grouped-GEMM dispatch.

This kernel replaces the post-topK path of ``torch.ops.trtllm.moe_sort``. TopK
selection is already complete: the input is a contiguous ``int32[T, 8]`` tensor of
global expert IDs. The kernel keeps IDs in
``[local_expert_offset, local_expert_offset + 64)`` and produces TRT-LLM's six
routing tensors:

* ``tile_idx_to_expert_idx``: local expert ID for each populated 128-row tile.
* ``tile_idx_to_mn_limit``: exclusive valid-row limit for each populated tile.
* ``expanded_idx_to_permuted_idx``: grouped row for every expanded input, or ``-1``
  when the selected expert belongs to another EP rank.
* ``permuted_idx_to_expanded_idx``: inverse map over valid grouped rows.
* ``total_num_padded_tokens``: number of grouped rows including tile padding.
* ``num_non_exiting_tiles``: number of populated grouped-GEMM tiles.

The semantic contract matches the selected-ID overload of TRT-LLM routing. The
``token_final_scales`` tensor accepted by ``moe_sort`` is intentionally absent because
these six outputs depend only on selected IDs; scales remain inputs to downstream MoE
compute. Expert order within one expert is nondeterministic in both implementations
because slot assignment uses atomics. Tests therefore compare expert row sets, tile
limits, and the inverse map instead of requiring one atomic claim order.

The GLM configuration is specialized at compile time to topK 8, 256 global experts,
64 local experts, and 128 rows per grouped-GEMM tile. ``num_expanded`` and
``local_expert_offset`` are runtime values. ``PER_THREAD`` is a compile-time capacity
and schedule key. Integrations should obtain it from ``routing_schedule_bucket()`` so
different runtime token counts that lower to identical code reuse one compiled callable.
The dispatch is gap-free:

* ``T <= 128``: one-CTA register cache and ballot prefix.
* ``128 < T <= 512``: one larger CTA and a two-warp integer prefix.
* ``512 < T <= 16384``: int16-SMEM cache kernels on clusters of 8, 10 or 16 CTAs,
  with 256-bit topK8 global loads.
* ``16384 < T <= 24576``: direct cooperative routing on 96 CTAs.
* ``24576 < T <= 286720``: direct cooperative routing on 140 CTAs, up to the exact
  resident-grid capacity boundary.
* ``286720 < T <= 524288``: three-stage multipass routing on 768 CTAs.
* ``T > 524288``: three-stage multipass routing on 1024 CTAs with runtime grid-stride
  loops.

The speedup comes from specializing TRT-LLM's generic 256-expert routing algorithm to
the local EP4 contract. Only 64 histogram bins are initialized, reduced, and scanned;
topK8 IDs use one 256-bit load; the prefix scan uses two warps instead of a 1024-wide
CUB block scan; and the small paths eliminate TRT-LLM's separate count initialization
launch. NCU at ``T=20480`` reports 32 registers/thread and about 205K executed SM
instructions for the CuTe DSL direct kernel versus 164 registers/thread and about 398K
instructions for TRT-LLM's cooperative kernel. The large-T multipass path keeps the
same three-launch topology as TRT-LLM but fuses offsets, scatter, and metadata around
the narrower local-expert state.

The direct path uses one init kernel and one fused histogram/scan/permutation/metadata
kernel. The multipass path uses scratch init, histogram, and fused
offsets/scatter/metadata kernels. All launches enable PDL. Two caller-owned scratch
tensors are required: ``expert_counts_workspace: int32[64]`` and
``barrier_workspace: int32[65]``. They are internal resources, not semantic outputs.

Integration guidance:

1. Flatten ``token_selected_experts`` without copying when it is already contiguous.
2. Cache the two scratch tensors per concurrent stream or CUDA graph; do not allocate
   them in the timed hot path.
3. Compile ``routing_host_with_workspace`` with the six outputs, both scratch tensors,
   runtime ``num_expanded`` and ``local_expert_offset``, constants ``64`` and ``128``,
   and the result of ``routing_schedule_bucket(num_expanded)``.
4. Cache compiled callables by device, dtype, constants, and schedule bucket. Tensor
   leading dimensions are dynamic; do not include exact ``T`` or runtime
   ``local_expert_offset`` in the compile key. Do not reject long prefill at the former
   ``T * topK <= 512`` integration guard.
5. Keep the scratch tensors and all six outputs alive through CUDA graph replay.

Integrations use ``routing_host_with_workspace``, which takes the two scratch tensors
in addition to the semantic input and all six public outputs. Padding
entries in ``permuted_idx_to_expanded_idx`` are unspecified; consumers must use
``tile_idx_to_mn_limit`` exactly as TRT-LLM FC1/FC2 do.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

_BLOCK = 256  # threads; also the upper bound on histogram width, since
# zeroing is one bin per thread
_WARP = 32
_FULL_MASK = 0xFFFFFFFF
_GLM_LOCAL_EXPERTS = 64
_GLM_TOP_K = 8
_GLM_TILE = 128


def routing_schedule_bucket(num_expanded: int) -> int:
    """Return the finite compile-time schedule key for a runtime routing shape.

    Small single-CTA kernels keep exact capacities because ``PER_THREAD`` controls
    register arrays and unrolled loops. The cached-SMEM schedules use fixed launch
    constants within each dispatch tier. The 140-CTA cooperative tier is bucketed by its
    derived ``TOKENS_PER_THREAD`` value. Multipass kernels use runtime grid-stride loops;
    ``16385`` selects the unbounded 1024-CTA multipass schedule. Together these
    rules produce at most 32 compile keys for every positive runtime shape.
    """
    if num_expanded <= 0:
        raise ValueError(f"num_expanded must be positive, got {num_expanded}")

    per_thread = (num_expanded + _BLOCK - 1) // _BLOCK
    if per_thread <= 16:
        return per_thread
    for capacity in (32, 64, 128, 256, 512, 768):
        if per_thread <= capacity:
            return capacity
    if per_thread <= 8960:
        return ((per_thread + 1119) // 1120) * 1120
    if per_thread <= 16384:
        return 16384
    return 16385


@cute.jit
def _ballot_exclusive(active):
    """Return this lane's exclusive active-lane prefix and the warp total."""
    mask = cutlass.Uint32(prims.vote_sync(_FULL_MASK, active, prims.VoteSync.BALLOT))
    prefix = cutlass.Int32(cute.arch.popc(mask & cute.arch.lanemask_lt()))
    total = cutlass.Int32(cute.arch.popc(mask))
    return prefix, total


@cute.jit
def _intra_warp_inclusive_scan_i32(value, lane):
    """Return the inclusive sum of ``value`` over lanes ``0..lane``."""
    n1 = cute.arch.shuffle_sync_up(value, 1)
    value = value + n1 * cute.math.min(lane, cutlass.Int32(1))
    n2 = cute.arch.shuffle_sync_up(value, 2)
    value = value + n2 * cute.math.min(lane >> cutlass.Int32(1), cutlass.Int32(1))
    n4 = cute.arch.shuffle_sync_up(value, 4)
    value = value + n4 * cute.math.min(lane >> cutlass.Int32(2), cutlass.Int32(1))
    n8 = cute.arch.shuffle_sync_up(value, 8)
    value = value + n8 * cute.math.min(lane >> cutlass.Int32(3), cutlass.Int32(1))
    n16 = cute.arch.shuffle_sync_up(value, 16)
    value = value + n16 * cute.math.min(lane >> cutlass.Int32(4), cutlass.Int32(1))
    return value


@cute.jit
def _two_warp_exclusive_scan_i32(value, lane, warp, warp_totals):
    """Exclusive scan over the 64 local experts and its total tile count."""
    original = value
    inclusive = _intra_warp_inclusive_scan_i32(value, lane)
    if lane == cutlass.Int32(_WARP - 1):
        warp_totals.subview(warp).store(inclusive)
    prims.barrier_cta_sync(1, thread_count=_GLM_LOCAL_EXPERTS)

    first_total = warp_totals.subview(0).load()
    second_total = warp_totals.subview(1).load()
    previous_warps = cutlass.Int32(0)
    if warp == cutlass.Int32(1):
        previous_warps = first_total
    return previous_warps + inclusive - original, first_total + second_total


@cute.kernel
def routing_kernel(
    token_selected_experts: cute.Tensor,  # (T*top_k,) int32, global expert slots
    tile_idx_to_expert_idx: cute.Tensor,  # (max_ctas,) int32       out
    tile_idx_to_mn_limit: cute.Tensor,  # (max_ctas,) int32       out
    expanded_idx_to_permuted_idx: cute.Tensor,  # (T*top_k,) int32        out
    permuted_idx_to_expanded_idx: cute.Tensor,  # (max_padded,) int32     out
    total_num_padded_tokens: cute.Tensor,  # (1,) int32              out
    num_non_exiting_tiles: cute.Tensor,  # (1,) int32              out
    num_expanded: cutlass.Int32,  # T * top_k, runtime
    local_expert_offset: cutlass.Int32,
    LOCAL_EXPERTS: cutlass.Constexpr,  # 64
    TILE: cutlass.Constexpr,  # 128
    PER_THREAD: cutlass.Constexpr,  # ceil(max_expanded / _BLOCK)
):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % _WARP
    warp = tidx // _WARP

    cnt = cutlass.Array(
        cutlass.Int32, LOCAL_EXPERTS, space=cutlass.AddressSpace.smem, alignment=128
    )
    base = cutlass.Array(
        cutlass.Int32, LOCAL_EXPERTS, space=cutlass.AddressSpace.smem, alignment=128
    )
    warp_tot = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=128)

    # ---- 1. zero the histogram ------------------------------------------------------
    if tidx < LOCAL_EXPERTS:
        cnt.subview(tidx).store(cutlass.Int32(0))
    prims.barrier_cta_sync(0)

    # ---- 2. histogram; each thread claims a slot per expanded index it owns ----------
    # `my_local` holds the local expert (-1 when the expert lives on another rank) and
    # `my_slot` the row within it, both kept in registers so phase 5 needs no second read.
    my_local = cutlass.Array(cutlass.Int32, PER_THREAD)
    my_slot = cutlass.Array(cutlass.Int32, PER_THREAD)
    for ii in cutlass.range_constexpr(PER_THREAD):
        my_local[ii] = cutlass.Int32(-1)
        my_slot[ii] = cutlass.Int32(0)

    for ii in cutlass.range_constexpr(PER_THREAD):
        e = tidx + cutlass.Int32(ii * _BLOCK)
        if e < num_expanded:
            loc = token_selected_experts[e] - local_expert_offset
            if loc >= cutlass.Int32(0) and loc < cutlass.Int32(LOCAL_EXPERTS):
                my_local[ii] = loc
                my_slot[ii] = prims.atomicrmw("add", cnt.subview(loc), 1)
    prims.barrier_cta_sync(0)

    # ---- 3. prefix and tile records; only the two expert warps participate ----------
    if warp < cutlass.Int32(2):
        c = cnt.subview(tidx).load()
        active = c != cutlass.Int32(0)
        local_prefix, local_total = _ballot_exclusive(active)
        if lane == cutlass.Int32(_WARP - 1):
            warp_tot.subview(warp).store(local_total)
        prims.barrier_cta_sync(1, thread_count=_GLM_LOCAL_EXPERTS)

        total0 = warp_tot.subview(0).load()
        total1 = warp_tot.subview(1).load()
        previous = cutlass.Int32(0)
        if warp == cutlass.Int32(1):
            previous = total0
        cta_offset = previous + local_prefix

        if active:
            tile_idx_to_expert_idx[cta_offset] = tidx
            tile_idx_to_mn_limit[cta_offset] = cta_offset * cutlass.Int32(TILE) + c
        base.subview(tidx).store(cta_offset * cutlass.Int32(TILE))

        if tidx == cutlass.Int32(0):
            grand = total0 + total1
            num_non_exiting_tiles[0] = grand
            total_num_padded_tokens[0] = grand * cutlass.Int32(TILE)
    prims.barrier_cta_sync(0)

    # ---- 4. write the permutation ----------------------------------------------------
    for ii in cutlass.range_constexpr(PER_THREAD):
        e = tidx + cutlass.Int32(ii * _BLOCK)
        if e < num_expanded:
            loc = my_local[ii]
            if loc >= cutlass.Int32(0):
                p = base.subview(loc).load() + my_slot[ii]
                expanded_idx_to_permuted_idx[e] = p
                permuted_idx_to_expanded_idx[p] = e
            else:
                # The reference writes -1 for experts owned by another rank and downstream
                # consumers rely on that, rather than on the entry being left untouched.
                expanded_idx_to_permuted_idx[e] = cutlass.Int32(-1)

    prims.griddepcontrol("launch_dependents")


@cute.kernel
def routing_single_cta_kernel(
    token_selected_experts: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    num_expanded: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    LOCAL_EXPERTS: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
    BLOCK_THREADS: cutlass.Constexpr,
    ITEMS_PER_THREAD: cutlass.Constexpr,
):
    """General one-CTA path supporting multiple GEMM tiles per expert."""
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % _WARP
    warp = tidx // _WARP

    cnt = cutlass.Array(
        cutlass.Int32,
        LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    base = cutlass.Array(
        cutlass.Int32,
        LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    warp_totals = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=128)

    if tidx < LOCAL_EXPERTS:
        cnt.subview(tidx).store(cutlass.Int32(0))
    prims.barrier_cta_sync(0)

    my_local = cutlass.Array(cutlass.Int32, ITEMS_PER_THREAD)
    my_slot = cutlass.Array(cutlass.Int32, ITEMS_PER_THREAD)
    for ii in cutlass.range_constexpr(ITEMS_PER_THREAD):
        my_local[ii] = cutlass.Int32(-1)
        my_slot[ii] = cutlass.Int32(0)

    for ii in cutlass.range_constexpr(ITEMS_PER_THREAD):
        expanded = tidx + cutlass.Int32(ii * BLOCK_THREADS)
        if expanded < num_expanded:
            local = token_selected_experts[expanded] - local_expert_offset
            if local >= cutlass.Int32(0) and local < cutlass.Int32(LOCAL_EXPERTS):
                my_local[ii] = local
                my_slot[ii] = prims.atomicrmw("add", cnt.subview(local), 1)
    prims.barrier_cta_sync(0)

    if tidx < LOCAL_EXPERTS:
        count = cnt.subview(tidx).load()
        num_tiles = (count + cutlass.Int32(TILE - 1)) // cutlass.Int32(TILE)
        tile_offset, total_tiles = _two_warp_exclusive_scan_i32(num_tiles, lane, warp, warp_totals)

        tile_in_expert = cutlass.Int32(0)
        while tile_in_expert < num_tiles:
            output_tile = tile_offset + tile_in_expert
            tile_idx_to_expert_idx[output_tile] = tidx
            tile_end = (output_tile + cutlass.Int32(1)) * cutlass.Int32(TILE)
            expert_end = tile_offset * cutlass.Int32(TILE) + count
            tile_idx_to_mn_limit[output_tile] = cute.math.min(tile_end, expert_end)
            tile_in_expert = tile_in_expert + cutlass.Int32(1)

        base.subview(tidx).store(tile_offset * cutlass.Int32(TILE))
        if tidx == cutlass.Int32(0):
            num_non_exiting_tiles[0] = total_tiles
            total_num_padded_tokens[0] = total_tiles * cutlass.Int32(TILE)
    prims.barrier_cta_sync(0)

    for ii in cutlass.range_constexpr(ITEMS_PER_THREAD):
        expanded = tidx + cutlass.Int32(ii * BLOCK_THREADS)
        if expanded < num_expanded:
            local = my_local[ii]
            if local >= cutlass.Int32(0):
                permuted = base.subview(local).load() + my_slot[ii]
                expanded_idx_to_permuted_idx[expanded] = permuted
                permuted_idx_to_expanded_idx[permuted] = expanded
            else:
                expanded_idx_to_permuted_idx[expanded] = cutlass.Int32(-1)

    prims.griddepcontrol("launch_dependents")


@cute.kernel
def routing_smem_cached_cluster_kernel(
    token_selected_experts: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    num_expanded: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    LOCAL_EXPERTS: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
    BLOCK_THREADS: cutlass.Constexpr,
    CLUSTER_SIZE: cutlass.Constexpr,
    TOKENS_PER_THREAD: cutlass.Constexpr,
):
    """Cache topK8 local expert IDs in SMEM using 256b GMEM vectors."""
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % _WARP
    warp = tidx // _WARP
    cluster_rank = cute.arch.block_idx_in_cluster()
    cluster_thread = cluster_rank * cutlass.Int32(BLOCK_THREADS) + tidx
    total_threads = BLOCK_THREADS * CLUSTER_SIZE
    num_tokens = num_expanded // cutlass.Int32(_GLM_TOP_K)

    cnt = cutlass.Array(
        cutlass.Int32,
        LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    base = cutlass.Array(
        cutlass.Int32,
        LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    warp_totals = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=128)
    cached_local_experts = cutlass.Array(
        cutlass.Int16,
        BLOCK_THREADS * TOKENS_PER_THREAD * _GLM_TOP_K,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )

    if tidx < LOCAL_EXPERTS:
        cnt.subview(tidx).store(cutlass.Int32(0))
    prims.barrier_cta_sync(0)

    input_ptr = token_selected_experts.iterator.raw_ptr()
    for token_item in cutlass.range_constexpr(TOKENS_PER_THREAD):
        token = cluster_thread + cutlass.Int32(token_item * total_threads)
        if token < num_tokens:
            expanded_base = token * cutlass.Int32(_GLM_TOP_K)
            expert_ids = (input_ptr + cutlass.Int64(expanded_base)).load(
                count=_GLM_TOP_K, alignment=32
            )
            local_ids = cutlass.Array(cutlass.Int16, _GLM_TOP_K, alignment=16)
            for topk_slot in cutlass.range_constexpr(_GLM_TOP_K):
                local = expert_ids[topk_slot] - local_expert_offset
                local_ids[topk_slot] = cutlass.Int16(-1)
                if local >= cutlass.Int32(0) and local < cutlass.Int32(LOCAL_EXPERTS):
                    local_ids[topk_slot] = cutlass.Int16(local)
                    prims.atomicrmw("add", cnt.subview(local), 1)

            cache_base = (tidx * TOKENS_PER_THREAD + token_item) * _GLM_TOP_K
            cache_ptr = cached_local_experts.data_ptr() + cache_base
            cache_ptr.store(local_ids[0:_GLM_TOP_K], alignment=16)

    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()

    if tidx < LOCAL_EXPERTS:
        total_count = cutlass.Array(cutlass.Int32, 1)
        rank_prefix = cutlass.Array(cutlass.Int32, 1)
        total_count[0] = cutlass.Int32(0)
        rank_prefix[0] = cutlass.Int32(0)
        for rank in cutlass.range_constexpr(CLUSTER_SIZE):
            rank_count = prims.mapa(cnt.subview(tidx), cutlass.Int32(rank)).load()
            if cutlass.Int32(rank) == cluster_rank:
                rank_prefix[0] = total_count[0]
            total_count[0] = total_count[0] + rank_count

        count = total_count[0]
        num_tiles = (count + cutlass.Int32(TILE - 1)) // cutlass.Int32(TILE)
        tile_offset, total_tiles = _two_warp_exclusive_scan_i32(num_tiles, lane, warp, warp_totals)

        tile_in_expert = cluster_rank
        while tile_in_expert < num_tiles:
            output_tile = tile_offset + tile_in_expert
            tile_idx_to_expert_idx[output_tile] = tidx
            tile_end = (output_tile + cutlass.Int32(1)) * cutlass.Int32(TILE)
            expert_end = tile_offset * cutlass.Int32(TILE) + count
            tile_idx_to_mn_limit[output_tile] = cute.math.min(tile_end, expert_end)
            tile_in_expert = tile_in_expert + cutlass.Int32(CLUSTER_SIZE)

        base.subview(tidx).store(tile_offset * cutlass.Int32(TILE) + rank_prefix[0])
        if cluster_rank == cutlass.Int32(0) and tidx == cutlass.Int32(0):
            num_non_exiting_tiles[0] = total_tiles
            total_num_padded_tokens[0] = total_tiles * cutlass.Int32(TILE)

    prims.barrier_cta_sync(0)
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()
    if tidx < LOCAL_EXPERTS:
        cnt.subview(tidx).store(cutlass.Int32(0))
    prims.barrier_cta_sync(0)

    e2p_ptr = expanded_idx_to_permuted_idx.iterator.raw_ptr()
    for token_item in cutlass.range_constexpr(TOKENS_PER_THREAD):
        token = cluster_thread + cutlass.Int32(token_item * total_threads)
        if token < num_tokens:
            expanded_base = token * cutlass.Int32(_GLM_TOP_K)
            cache_base = (tidx * TOKENS_PER_THREAD + token_item) * _GLM_TOP_K
            cache_ptr = cached_local_experts.data_ptr() + cache_base
            local_ids = cache_ptr.load(count=_GLM_TOP_K, alignment=16)
            output_indices = cutlass.Array(cutlass.Int32, _GLM_TOP_K, alignment=32)

            for topk_slot in cutlass.range_constexpr(_GLM_TOP_K):
                local = cutlass.Int32(local_ids[topk_slot])
                output_indices[topk_slot] = cutlass.Int32(-1)
                if local >= cutlass.Int32(0):
                    local_slot = prims.atomicrmw("add", cnt.subview(local), 1)
                    permuted = base.subview(local).load() + local_slot
                    output_indices[topk_slot] = permuted
                    permuted_idx_to_expanded_idx[permuted] = expanded_base + cutlass.Int32(
                        topk_slot
                    )

            (e2p_ptr + cutlass.Int64(expanded_base)).store(
                output_indices[0:_GLM_TOP_K], alignment=32
            )

    prims.griddepcontrol("launch_dependents")


@cute.jit
def _launch_smem_cached_cluster(
    token_selected_experts: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    num_expanded: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    LOCAL_EXPERTS: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
    BLOCK_THREADS: cutlass.Constexpr,
    CLUSTER_SIZE: cutlass.Constexpr,
    TOKENS_PER_THREAD: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
) -> None:
    """Launch helper for the vectorized SMEM-cache family."""
    routing_smem_cached_cluster_kernel(
        token_selected_experts,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
        num_expanded,
        local_expert_offset,
        LOCAL_EXPERTS,
        TILE,
        BLOCK_THREADS,
        CLUSTER_SIZE,
        TOKENS_PER_THREAD,
    ).launch(
        grid=(CLUSTER_SIZE, 1, 1),
        block=(BLOCK_THREADS, 1, 1),
        cluster=(CLUSTER_SIZE, 1, 1),
        stream=stream,
        use_pdl=True,
    )


@cute.kernel
def routing_cooperative_init_kernel(
    expert_counts: cute.Tensor,
    barrier_storage: cute.Tensor,
    LOCAL_EXPERTS: cutlass.Constexpr,
):
    """Reset the local-64 histogram and cooperative barrier word."""
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < LOCAL_EXPERTS:
        expert_counts[tidx] = cutlass.Int32(0)
    if tidx == cutlass.Int32(0):
        barrier_storage[0] = cutlass.Int32(0)
    prims.griddepcontrol("launch_dependents")


@cute.jit
def _cooperative_cluster_leader_barrier(
    barrier_word,
    cluster_rank,
    NUM_CLUSTERS: cutlass.Constexpr,
):
    """Synchronize a freshly initialized cooperative grid."""
    tidx, _, _ = cute.arch.thread_idx()
    prims.barrier_cta_sync(0)
    if tidx == cutlass.Int32(0):
        if cluster_rank == cutlass.Int32(0):
            prims.atomicrmw(
                prims.AtomicOp.ADD,
                barrier_word,
                cutlass.Uint32(1),
                mem_order=prims.MemOrder.RELEASE,
                syncscope=prims.MemScope.GPU,
            )

        value = prims.load_ext(
            barrier_word,
            dtype=cutlass.Uint32,
            order=prims.MemOrder.ACQUIRE,
            scope=prims.MemScope.GPU,
        )
        while value != cutlass.Uint32(NUM_CLUSTERS):
            value = prims.load_ext(
                barrier_word,
                dtype=cutlass.Uint32,
                order=prims.MemOrder.ACQUIRE,
                scope=prims.MemScope.GPU,
            )
    prims.barrier_cta_sync(0)


@cute.kernel
def routing_hierarchical_dsmem_cooperative_kernel(
    token_selected_experts: cute.Tensor,
    expert_counts: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    barrier_storage: cute.Tensor,
    num_expanded: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    LOCAL_EXPERTS: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
    BLOCK_THREADS: cutlass.Constexpr,
    NUM_BLOCKS: cutlass.Constexpr,
    CLUSTER_SIZE: cutlass.Constexpr,
    TOKENS_PER_THREAD: cutlass.Constexpr,
    FUSE_METADATA: cutlass.Constexpr,
):
    """Full-GPU routing with DSMEM histogram reduction per cluster."""
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % _WARP
    warp = tidx // _WARP
    block_idx = cute.arch.block_idx()[0]
    cluster_rank = cute.arch.block_idx_in_cluster()
    global_thread = block_idx * cutlass.Int32(BLOCK_THREADS) + tidx
    total_threads = cutlass.Int32(BLOCK_THREADS * NUM_BLOCKS)
    num_tokens = num_expanded // cutlass.Int32(_GLM_TOP_K)

    local_counts = cutlass.Array(
        cutlass.Int32,
        LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    expert_base = cutlass.Array(
        cutlass.Int32,
        LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    warp_totals = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=128)

    if tidx < LOCAL_EXPERTS:
        local_counts.subview(tidx).store(cutlass.Int32(0))
    prims.barrier_cta_sync(0)
    prims.griddepcontrol("wait")

    item_count: cutlass.Constexpr = TOKENS_PER_THREAD * _GLM_TOP_K
    my_local = cutlass.Array(cutlass.Int32, item_count, alignment=32)
    my_slot = cutlass.Array(cutlass.Int32, item_count, alignment=32)
    for item in cutlass.range_constexpr(item_count):
        my_local[item] = cutlass.Int32(-1)
        my_slot[item] = cutlass.Int32(0)

    input_ptr = token_selected_experts.iterator.raw_ptr()
    for token_item in cutlass.range_constexpr(TOKENS_PER_THREAD):
        token = global_thread + cutlass.Int32(token_item) * total_threads
        if token < num_tokens:
            expanded_base = token * cutlass.Int32(_GLM_TOP_K)
            expert_ids = (input_ptr + cutlass.Int64(expanded_base)).load(
                count=_GLM_TOP_K, alignment=32
            )
            for topk_slot in cutlass.range_constexpr(_GLM_TOP_K):
                item = token_item * _GLM_TOP_K + topk_slot
                local = expert_ids[topk_slot] - local_expert_offset
                if local >= cutlass.Int32(0) and local < cutlass.Int32(LOCAL_EXPERTS):
                    my_local[item] = local
                    my_slot[item] = prims.atomicrmw("add", local_counts.subview(local), 1)

    if cutlass.const_expr(CLUSTER_SIZE == 1):
        # The direct-global tier retains token-wise 256-bit I/O but avoids
        # self-DSMEM traffic and cluster barriers when no peer CTA exists.
        prims.barrier_cta_sync(0)
        if tidx < LOCAL_EXPERTS:
            block_offset = prims.atomicrmw(
                "add",
                expert_counts.iterator.raw_ptr() + cutlass.Int64(tidx),
                local_counts.subview(tidx).load(),
            )
            expert_base.subview(tidx).store(block_offset)
    else:
        # Split the 64 expert bins across cluster ranks. The owner of one bin
        # pulls that bin from every peer, contributes one aggregate to global
        # counts, and writes every CTA's prefix back through DSMEM.
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
        experts_per_rank: cutlass.Constexpr = (LOCAL_EXPERTS + CLUSTER_SIZE - 1) // CLUSTER_SIZE
        expert = cluster_rank * cutlass.Int32(experts_per_rank) + tidx
        if tidx < cutlass.Int32(experts_per_rank) and expert < LOCAL_EXPERTS:
            rank_counts = cutlass.Array(cutlass.Int32, CLUSTER_SIZE)
            cluster_count = cutlass.Int32(0)
            for rank in cutlass.range_constexpr(CLUSTER_SIZE):
                rank_count = prims.mapa(local_counts.subview(expert), cutlass.Int32(rank)).load()
                rank_counts[rank] = rank_count
                cluster_count = cluster_count + rank_count

            cluster_offset = prims.atomicrmw(
                "add",
                expert_counts.iterator.raw_ptr() + cutlass.Int64(expert),
                cluster_count,
            )
            rank_prefix = cluster_offset
            for rank in cutlass.range_constexpr(CLUSTER_SIZE):
                prims.mapa(expert_base.subview(expert), cutlass.Int32(rank)).store(rank_prefix)
                rank_prefix = rank_prefix + rank_counts[rank]

        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

    barrier_word = barrier_storage.iterator.raw_ptr()
    _cooperative_cluster_leader_barrier(
        barrier_word,
        cluster_rank,
        NUM_BLOCKS // CLUSTER_SIZE,
    )

    count = cutlass.Int32(0)
    num_tiles = cutlass.Int32(0)
    tile_offset = cutlass.Int32(0)
    total_tiles = cutlass.Int32(0)
    if tidx < LOCAL_EXPERTS:
        count = expert_counts[tidx]
        num_tiles = (count + cutlass.Int32(TILE - 1)) // cutlass.Int32(TILE)
        tile_offset, total_tiles = _two_warp_exclusive_scan_i32(num_tiles, lane, warp, warp_totals)
        expert_base.subview(tidx).store(
            tile_offset * cutlass.Int32(TILE) + expert_base.subview(tidx).load()
        )
    prims.barrier_cta_sync(0)

    e2p_ptr = expanded_idx_to_permuted_idx.iterator.raw_ptr()
    for token_item in cutlass.range_constexpr(TOKENS_PER_THREAD):
        token = global_thread + cutlass.Int32(token_item) * total_threads
        if token < num_tokens:
            expanded_base = token * cutlass.Int32(_GLM_TOP_K)
            output_indices = cutlass.Array(cutlass.Int32, _GLM_TOP_K, alignment=32)
            for topk_slot in cutlass.range_constexpr(_GLM_TOP_K):
                item = token_item * _GLM_TOP_K + topk_slot
                local = my_local[item]
                output_indices[topk_slot] = cutlass.Int32(-1)
                if local >= cutlass.Int32(0):
                    permuted = expert_base.subview(local).load() + my_slot[item]
                    output_indices[topk_slot] = permuted
                    permuted_idx_to_expanded_idx[permuted] = expanded_base + cutlass.Int32(
                        topk_slot
                    )
            (e2p_ptr + cutlass.Int64(expanded_base)).store(
                output_indices[0:_GLM_TOP_K], alignment=32
            )

    if cutlass.const_expr(FUSE_METADATA):
        # Dedicated counts remain live while the final output metadata is
        # published, so this path needs no second grid phase or epilogue kernel.
        if tidx < LOCAL_EXPERTS:
            tile_in_expert = block_idx
            while tile_in_expert < num_tiles:
                output_tile = tile_offset + tile_in_expert
                tile_idx_to_expert_idx[output_tile] = tidx
                tile_end = (output_tile + cutlass.Int32(1)) * cutlass.Int32(TILE)
                expert_end = tile_offset * cutlass.Int32(TILE) + count
                tile_idx_to_mn_limit[output_tile] = cute.math.min(tile_end, expert_end)
                tile_in_expert = tile_in_expert + cutlass.Int32(NUM_BLOCKS)

            if block_idx == cutlass.Int32(0) and tidx == cutlass.Int32(0):
                num_non_exiting_tiles[0] = total_tiles
                total_num_padded_tokens[0] = total_tiles * cutlass.Int32(TILE)

    prims.griddepcontrol("launch_dependents")


@cute.jit
def _launch_hierarchical_workspace(
    token_selected_experts: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    expert_counts: cute.Tensor,
    barrier_storage: cute.Tensor,
    num_expanded: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    LOCAL_EXPERTS: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
    BLOCK_THREADS: cutlass.Constexpr,
    NUM_BLOCKS: cutlass.Constexpr,
    CLUSTER_SIZE: cutlass.Constexpr,
    TOKENS_PER_THREAD: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
) -> None:
    """Launch fused hierarchical routing with 65 int32 scratch elements."""
    routing_cooperative_init_kernel(expert_counts, barrier_storage, LOCAL_EXPERTS).launch(
        grid=(1, 1, 1),
        block=(128, 1, 1),
        stream=stream,
        use_pdl=True,
    )
    if cutlass.const_expr(CLUSTER_SIZE == 1):
        routing_hierarchical_dsmem_cooperative_kernel(
            token_selected_experts,
            expert_counts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            barrier_storage,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            BLOCK_THREADS,
            NUM_BLOCKS,
            CLUSTER_SIZE,
            TOKENS_PER_THREAD,
            True,
        ).launch(
            grid=(NUM_BLOCKS, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            cooperative=True,
            stream=stream,
            use_pdl=True,
        )
    else:
        routing_hierarchical_dsmem_cooperative_kernel(
            token_selected_experts,
            expert_counts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            barrier_storage,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            BLOCK_THREADS,
            NUM_BLOCKS,
            CLUSTER_SIZE,
            TOKENS_PER_THREAD,
            True,
        ).launch(
            grid=(NUM_BLOCKS, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            cluster=(CLUSTER_SIZE, 1, 1),
            cooperative=True,
            stream=stream,
            use_pdl=True,
        )


@cute.kernel
def routing_multipass_histogram_kernel(
    token_selected_experts: cute.Tensor,
    expert_counts: cute.Tensor,
    num_expanded: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    LOCAL_EXPERTS: cutlass.Constexpr,
    BLOCK_THREADS: cutlass.Constexpr,
    NUM_BLOCKS: cutlass.Constexpr,
    ITEMS_PER_THREAD: cutlass.Constexpr,
):
    """Accumulate local-64 counts using many independent CTAs."""
    prims.griddepcontrol("wait")
    tidx, _, _ = cute.arch.thread_idx()
    block_idx = cute.arch.block_idx()[0]
    local_counts = cutlass.Array(
        cutlass.Int32,
        LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )

    if tidx < LOCAL_EXPERTS:
        local_counts.subview(tidx).store(cutlass.Int32(0))
    prims.barrier_cta_sync(0)

    num_tokens = num_expanded // cutlass.Int32(_GLM_TOP_K)
    token_tile = block_idx * cutlass.Int32(BLOCK_THREADS)
    grid_stride = cutlass.Int32(NUM_BLOCKS * BLOCK_THREADS)
    input_ptr = token_selected_experts.iterator.raw_ptr()
    while token_tile < num_tokens:
        token = token_tile + tidx
        if token < num_tokens:
            expanded_base = token * cutlass.Int32(_GLM_TOP_K)
            expert_ids = (input_ptr + cutlass.Int64(expanded_base)).load(
                count=_GLM_TOP_K, alignment=32
            )
            for topk_slot in cutlass.range_constexpr(_GLM_TOP_K):
                local = expert_ids[topk_slot] - local_expert_offset
                if local >= cutlass.Int32(0) and local < cutlass.Int32(LOCAL_EXPERTS):
                    prims.atomicrmw("add", local_counts.subview(local), 1)
        token_tile = token_tile + grid_stride

    prims.barrier_cta_sync(0)
    if tidx < LOCAL_EXPERTS:
        count = local_counts.subview(tidx).load()
        prims.atomicrmw("add", expert_counts.iterator.raw_ptr() + cutlass.Int64(tidx), count)
    prims.griddepcontrol("launch_dependents")


@cute.kernel
def routing_multipass_workspace_init_kernel(
    expert_counts: cute.Tensor,
    barrier_and_claims: cute.Tensor,
    LOCAL_EXPERTS: cutlass.Constexpr,
):
    """Reset dedicated histogram and claim scratch for multipass routing."""
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < LOCAL_EXPERTS:
        expert_counts[tidx] = cutlass.Int32(0)
        barrier_and_claims[tidx + cutlass.Int32(1)] = cutlass.Int32(0)
    prims.griddepcontrol("launch_dependents")


@cute.kernel
def routing_multipass_offsets_kernel(
    token_selected_experts: cute.Tensor,
    expert_counts: cute.Tensor,
    barrier_and_claims: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    num_expanded: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    LOCAL_EXPERTS: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
    BLOCK_THREADS: cutlass.Constexpr,
    NUM_BLOCKS: cutlass.Constexpr,
    ITEMS_PER_THREAD: cutlass.Constexpr,
):
    """Scan counts, scatter both maps, and publish grouped-GEMM metadata."""
    prims.griddepcontrol("wait")
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % _WARP
    warp = tidx // _WARP
    block_idx = cute.arch.block_idx()[0]

    local_counts = cutlass.Array(
        cutlass.Int32,
        LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    block_base = cutlass.Array(
        cutlass.Int32,
        LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    expert_base = cutlass.Array(
        cutlass.Int32,
        LOCAL_EXPERTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    warp_totals = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=128)

    if tidx < LOCAL_EXPERTS:
        count = expert_counts[tidx]
        num_tiles = (count + cutlass.Int32(TILE - 1)) // cutlass.Int32(TILE)
        tile_offset, total_tiles = _two_warp_exclusive_scan_i32(num_tiles, lane, warp, warp_totals)
        expert_base.subview(tidx).store(tile_offset * cutlass.Int32(TILE))

        tile_in_expert = block_idx
        while tile_in_expert < num_tiles:
            output_tile = tile_offset + tile_in_expert
            tile_idx_to_expert_idx[output_tile] = tidx
            tile_end = (output_tile + cutlass.Int32(1)) * cutlass.Int32(TILE)
            expert_end = tile_offset * cutlass.Int32(TILE) + count
            tile_idx_to_mn_limit[output_tile] = cute.math.min(tile_end, expert_end)
            tile_in_expert = tile_in_expert + cutlass.Int32(NUM_BLOCKS)

        if block_idx == cutlass.Int32(0) and tidx == cutlass.Int32(0):
            num_non_exiting_tiles[0] = total_tiles
            total_num_padded_tokens[0] = total_tiles * cutlass.Int32(TILE)
    prims.barrier_cta_sync(0)

    claims_ptr = barrier_and_claims.iterator.raw_ptr() + cutlass.Int64(1)
    num_tokens = num_expanded // cutlass.Int32(_GLM_TOP_K)
    token_tile = block_idx * cutlass.Int32(BLOCK_THREADS)
    grid_stride = cutlass.Int32(NUM_BLOCKS * BLOCK_THREADS)
    input_ptr = token_selected_experts.iterator.raw_ptr()
    e2p_ptr = expanded_idx_to_permuted_idx.iterator.raw_ptr()
    while token_tile < num_tokens:
        if tidx < LOCAL_EXPERTS:
            local_counts.subview(tidx).store(cutlass.Int32(0))
        prims.barrier_cta_sync(0)

        token = token_tile + tidx
        my_local = cutlass.Array(cutlass.Int32, _GLM_TOP_K, alignment=32)
        my_slot = cutlass.Array(cutlass.Int32, _GLM_TOP_K, alignment=32)
        for topk_slot in cutlass.range_constexpr(_GLM_TOP_K):
            my_local[topk_slot] = cutlass.Int32(-1)
            my_slot[topk_slot] = cutlass.Int32(0)
        if token < num_tokens:
            expanded_base = token * cutlass.Int32(_GLM_TOP_K)
            expert_ids = (input_ptr + cutlass.Int64(expanded_base)).load(
                count=_GLM_TOP_K, alignment=32
            )
            for topk_slot in cutlass.range_constexpr(_GLM_TOP_K):
                local = expert_ids[topk_slot] - local_expert_offset
                if local >= cutlass.Int32(0) and local < cutlass.Int32(LOCAL_EXPERTS):
                    my_local[topk_slot] = local
                    my_slot[topk_slot] = prims.atomicrmw("add", local_counts.subview(local), 1)
        prims.barrier_cta_sync(0)

        if tidx < LOCAL_EXPERTS:
            count = local_counts.subview(tidx).load()
            claim = prims.atomicrmw("add", claims_ptr + cutlass.Int64(tidx), count)
            block_base.subview(tidx).store(expert_base.subview(tidx).load() + claim)
        prims.barrier_cta_sync(0)

        if token < num_tokens:
            expanded_base = token * cutlass.Int32(_GLM_TOP_K)
            output_indices = cutlass.Array(cutlass.Int32, _GLM_TOP_K, alignment=32)
            for topk_slot in cutlass.range_constexpr(_GLM_TOP_K):
                local = my_local[topk_slot]
                output_indices[topk_slot] = cutlass.Int32(-1)
                if local >= cutlass.Int32(0):
                    permuted = block_base.subview(local).load() + my_slot[topk_slot]
                    expanded = expanded_base + cutlass.Int32(topk_slot)
                    output_indices[topk_slot] = permuted
                    permuted_idx_to_expanded_idx[permuted] = expanded
            (e2p_ptr + cutlass.Int64(expanded_base)).store(
                output_indices[0:_GLM_TOP_K], alignment=32
            )
        prims.barrier_cta_sync(0)
        token_tile = token_tile + grid_stride

    prims.griddepcontrol("launch_dependents")


@cute.jit
def _launch_multipass_workspace(
    token_selected_experts: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    expert_counts: cute.Tensor,
    barrier_and_claims: cute.Tensor,
    num_expanded: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    LOCAL_EXPERTS: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
    BLOCK_THREADS: cutlass.Constexpr,
    NUM_BLOCKS: cutlass.Constexpr,
    ITEMS_PER_THREAD: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
) -> None:
    """Launch TRT-style init, histogram, and fused offsets/scatter stages."""
    routing_multipass_workspace_init_kernel(
        expert_counts, barrier_and_claims, LOCAL_EXPERTS
    ).launch(
        grid=(1, 1, 1),
        block=(128, 1, 1),
        stream=stream,
        use_pdl=True,
    )
    routing_multipass_histogram_kernel(
        token_selected_experts,
        expert_counts,
        num_expanded,
        local_expert_offset,
        LOCAL_EXPERTS,
        BLOCK_THREADS,
        NUM_BLOCKS,
        ITEMS_PER_THREAD,
    ).launch(
        grid=(NUM_BLOCKS, 1, 1),
        block=(BLOCK_THREADS, 1, 1),
        stream=stream,
        use_pdl=True,
    )
    routing_multipass_offsets_kernel(
        token_selected_experts,
        expert_counts,
        barrier_and_claims,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
        num_expanded,
        local_expert_offset,
        LOCAL_EXPERTS,
        TILE,
        BLOCK_THREADS,
        NUM_BLOCKS,
        ITEMS_PER_THREAD,
    ).launch(
        grid=(NUM_BLOCKS, 1, 1),
        block=(BLOCK_THREADS, 1, 1),
        stream=stream,
        use_pdl=True,
    )


@cute.jit
def _check_constexpr(LOCAL_EXPERTS: int, TILE: int, PER_THREAD: int) -> None:
    """Reject bad compile-time constants for both public routing entry points.

    ``PER_THREAD`` is the value returned by ``routing_schedule_bucket()``. It is a
    compile-time dispatch key, not a functional rejection boundary. Direct schedules use
    it as a capacity; the unbounded multipass schedule uses ``16385`` as a sentinel and
    traverses ``num_expanded`` with runtime grid-stride loops.
    """
    if cutlass.const_expr(LOCAL_EXPERTS != _GLM_LOCAL_EXPERTS):
        raise ValueError(
            f"GLM routing requires local_experts={_GLM_LOCAL_EXPERTS}, got {LOCAL_EXPERTS}"
        )
    if cutlass.const_expr(TILE != _GLM_TILE):
        raise ValueError(f"GLM routing requires tile={_GLM_TILE}, got {TILE}")
    if cutlass.const_expr(PER_THREAD <= 0):
        raise ValueError(f"GLM routing requires PER_THREAD >= 1, got {PER_THREAD}")


@cute.jit
def routing_host_with_workspace(
    token_selected_experts: cute.Tensor,
    tile_idx_to_expert_idx: cute.Tensor,
    tile_idx_to_mn_limit: cute.Tensor,
    expanded_idx_to_permuted_idx: cute.Tensor,
    permuted_idx_to_expanded_idx: cute.Tensor,
    total_num_padded_tokens: cute.Tensor,
    num_non_exiting_tiles: cute.Tensor,
    expert_counts_workspace: cute.Tensor,
    barrier_workspace: cute.Tensor,
    num_expanded: cutlass.Int32,
    local_expert_offset: cutlass.Int32,
    LOCAL_EXPERTS: cutlass.Constexpr,
    TILE: cutlass.Constexpr,
    PER_THREAD: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
) -> None:
    """Dispatch the specialized post-topK routing implementation.

    Tensor order is the selected-ID input, TRT-LLM's six semantic outputs, then the
    ``int32[64]`` count workspace and ``int32[65]`` barrier/claim workspace. The two
    scalar inputs are runtime values. ``LOCAL_EXPERTS``, ``TILE``, and ``PER_THREAD``
    are compile-time values; obtain ``PER_THREAD`` from ``routing_schedule_bucket()``.
    """
    _check_constexpr(LOCAL_EXPERTS, TILE, PER_THREAD)
    if cutlass.const_expr(PER_THREAD <= 4):
        routing_kernel(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            PER_THREAD,
        ).launch(
            grid=(1, 1, 1),
            block=(_BLOCK, 1, 1),
            stream=stream,
            use_pdl=True,
        )
    elif cutlass.const_expr(PER_THREAD <= 8):
        routing_single_cta_kernel(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            256,
            PER_THREAD,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream, use_pdl=True)
    elif cutlass.const_expr(PER_THREAD <= 16):
        routing_single_cta_kernel(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            512,
            (PER_THREAD + 1) // 2,
        ).launch(grid=(1, 1, 1), block=(512, 1, 1), stream=stream, use_pdl=True)
    elif cutlass.const_expr(PER_THREAD <= 32):
        _launch_smem_cached_cluster(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            256,
            8,
            1,
            stream,
        )
    elif cutlass.const_expr(PER_THREAD <= 64):
        _launch_smem_cached_cluster(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            256,
            10,
            1,
            stream,
        )
    elif cutlass.const_expr(PER_THREAD <= 128):
        _launch_smem_cached_cluster(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            256,
            16,
            1,
            stream,
        )
    elif cutlass.const_expr(PER_THREAD <= 256):
        _launch_smem_cached_cluster(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            512,
            16,
            1,
            stream,
        )
    elif cutlass.const_expr(PER_THREAD <= 512):
        _launch_smem_cached_cluster(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            1024,
            16,
            1,
            stream,
        )
    elif cutlass.const_expr(PER_THREAD <= 768):
        _launch_hierarchical_workspace(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            expert_counts_workspace,
            barrier_workspace,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            256,
            96,
            1,
            (PER_THREAD + 767) // 768,
            stream,
        )
    elif cutlass.const_expr(PER_THREAD <= 8960):
        _launch_hierarchical_workspace(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            expert_counts_workspace,
            barrier_workspace,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            256,
            140,
            1,
            (PER_THREAD + 1119) // 1120,
            stream,
        )
    elif cutlass.const_expr(PER_THREAD <= 16384):
        _launch_multipass_workspace(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            expert_counts_workspace,
            barrier_workspace,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            256,
            768,
            8,
            stream,
        )
    else:
        _launch_multipass_workspace(
            token_selected_experts,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_expanded_idx,
            total_num_padded_tokens,
            num_non_exiting_tiles,
            expert_counts_workspace,
            barrier_workspace,
            num_expanded,
            local_expert_offset,
            LOCAL_EXPERTS,
            TILE,
            256,
            1024,
            8,
            stream,
        )
