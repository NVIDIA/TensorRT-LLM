# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 context TopK + routing maps with replicated hot experts split across EP ranks.

Context MoE under EP4 is skewed: per layer one rank's grouped GEMMs (GG1 + GG2) run ~8.5 % longer
than the rank mean (28.7 of 339 us per layer at 7-8k rows on B300, ~2.2 ms per context step) and
every other rank waits for it in the post-MoE all-reduce. The slowest rank changes from layer to
layer, so no static placement removes it. ``moe_decode_topk_routing_rep`` gives every rank a copy
of ``NREP`` hot experts per layer in local slots ``64 .. 64 + NREP - 1``. On context steps an
expert has hundreds of rows, so a hot expert's rows can be *split* across ranks; this module
balances the ranks' row loads with them.

Every rank computes the same TopK over all T rows, so every rank derives the same assignment
without communication. Rows inside one expert are claimed in shared-memory atomic order, which
differs between ranks, so the unit of assignment is not a row index but a **producer CTA**: token
``t`` is selected by CTA ``(t // warps) % ctas`` (fixed by T), and the per-CTA per-expert counts
are the same on every rank.

* histogram: the noraw selection (same ids and weights as ``moe_prefill_topk_routing_noraw``);
  each CTA counts and claims every winner in a 256-bin histogram (``ctas x 256`` slab);
* prefix (one CTA): per-expert totals; each EP rank's load = its home experts that are not hot,
  in 128-row tiles; a water fill gives each rank its share of the hot experts' rows; the hot rows
  are laid out in ``hot_ids`` order and, within one expert, producer-CTA order, and that sequence
  is cut into four consecutive segments (rank 0 first); a CTA's rows of a hot expert go to the
  rank whose segment holds their first row. This rank's slots are its home experts that are not
  hot (slot ``e - offset``) and the hot experts' owned CTA chunks (slot ``64 + h``, whatever the
  expert's home; the home slot of a hot expert gets no rows). Per-slot per-CTA offsets, bases and
  tile maps as in the plain prefix;
* scatter: each (token, k) maps through its CTA's offset for its slot, or ``-1`` when another
  rank owns it.

Every (token, k) pair is therefore mapped on exactly one EP rank and the EP all-reduce sums each
winner once; only which rank computes a hot expert's rows changes (bf16 partial regrouping, not
bitwise against the home mapping). ``hot_ids`` entries ``< 0`` are unused slots. FC1 / FC2 must
run on the ``64 + NREP``-deep stacks with the replica weights in slots ``64 + h``.
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass.experimental import primitives as prims

from .moe_topk_routing import (
    _EXPERTS,
    _FULL_MASK,
    _LOCAL_EXPERTS,
    _TOP_K,
    _VEC_SIZE,
    _WARP,
    _pack_score_index,
    _sort8_key_desc,
    _warp_sum_f32,
)

_PRODUCER_CTAS = 148
_TILE = 128
_EP_RANKS = _EXPERTS // _LOCAL_EXPERTS


@cute.jit
def _select_noraw_claim_all(
    logits,
    logit_splits: cutlass.Constexpr,
    routing_bias,
    topk_weights,
    topk_indices,
    counts,
    token,
    lane,
    valid_token,
    routed_scaling_factor,
):
    """``moe_prefill_topk_routing_noraw._select_noraw_and_count`` (same arithmetic, same ids and
    weights) with every winner claiming a slot in the CTA's 256-bin histogram."""
    keys = cutlass.Array(cutlass.Uint64, _VEC_SIZE)
    my_raw = cutlass.Float32(0.0)
    my_selected_idx = cutlass.Int32(-1)
    for v in cutlass.range_constexpr(_VEC_SIZE):
        keys[v] = cutlass.Uint64(0)
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
            keys[v] = _pack_score_index(raw + cutlass.Float32(routing_bias[expert]), expert)
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
            if score_key == warp_score_key and inverse_idx == winner_inverse_idx:
                for v in cutlass.range_constexpr(_VEC_SIZE - 1):
                    keys[v] = keys[v + 1]
                keys[_VEC_SIZE - 1] = cutlass.Uint64(0)
            if lane == cutlass.Int32(k):
                my_selected_idx = cutlass.Int32(65535) - cutlass.Int32(winner_inverse_idx)
        if lane < cutlass.Int32(_TOP_K):
            logit = cutlass.Float32(logits[row + cutlass.Int64(my_selected_idx)])
            for split in cutlass.range_constexpr(1, logit_splits):
                logit = logit + cutlass.Float32(
                    logits[row + cutlass.Int64(split * _EXPERTS) + cutlass.Int64(my_selected_idx)]
                )
            my_raw = cutlass.Float32(0.5) * cute.math.tanh(
                cutlass.Float32(0.5) * logit
            ) + cutlass.Float32(0.5)
    weight_sum = _warp_sum_f32(my_raw)
    my_slot = cutlass.Int32(-1)
    if valid_token and lane < cutlass.Int32(_TOP_K):
        expanded = token * cutlass.Int32(_TOP_K) + lane
        topk_indices[expanded] = my_selected_idx
        topk_weights[expanded] = (
            my_raw * routed_scaling_factor / (weight_sum + cutlass.Float32(1.0e-20))
        )
        my_slot = prims.atomicrmw("add", counts.subview(my_selected_idx), 1)
    return my_slot


@cute.kernel
def rep_histogram_kernel(
    logits: cute.Tensor,
    bias: cute.Tensor,
    weights: cute.Tensor,
    selected: cute.Tensor,
    claims: cute.Tensor,
    partial_counts: cute.Tensor,
    tokens: cutlass.Int32,
    routed_scale: cutlass.Float32,
    warps: cutlass.Constexpr,
    logit_splits: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
):
    """Noraw selection; every winner claims a slot in this CTA's 256-bin histogram."""
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    lane = tid & cutlass.Int32(31)
    warp = tid >> cutlass.Int32(5)
    counts = cutlass.Array(cutlass.Int32, _EXPERTS, space=cutlass.AddressSpace.smem, alignment=128)
    if tid < cutlass.Int32(_EXPERTS):
        counts.subview(tid).store(cutlass.Int32(0))
    prims.barrier_cta_sync(0)
    if cutlass.const_expr(use_pdl):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    stride = blocks * cutlass.Int32(warps)
    base = block * cutlass.Int32(warps)
    while base < tokens:
        token = base + warp
        slot = _select_noraw_claim_all(
            logits,
            logit_splits,
            bias,
            weights,
            selected,
            counts,
            token,
            lane,
            token < tokens,
            routed_scale,
        )
        if token < tokens and lane < cutlass.Int32(_TOP_K):
            claims[token * cutlass.Int32(_TOP_K) + lane] = slot
        base += stride
    prims.barrier_cta_sync(0)
    if tid < cutlass.Int32(_EXPERTS):
        partial_counts[block * cutlass.Int32(_EXPERTS) + tid] = counts.subview(tid).load()
    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if tid == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.jit
def _load_hot_index(hot_index, hot_ids, tid, nrep: cutlass.Constexpr):
    """``hot_index[e]`` = h for ``hot_ids[h] == e``, else -1 (256 threads)."""
    hot_index.subview(tid).store(cutlass.Int32(-1))
    prims.barrier_cta_sync(0)
    if tid < cutlass.Int32(nrep):
        e = cutlass.Int32(hot_ids[tid])
        if e >= cutlass.Int32(0):
            hot_index.subview(e).store(tid)
    prims.barrier_cta_sync(0)


@cute.kernel
def rep_prefix_kernel(
    partial_counts: cute.Tensor,
    hot_ids: cute.Tensor,
    slot_offsets: cute.Tensor,
    expert_bases: cute.Tensor,
    tile_expert: cute.Tensor,
    tile_limit: cute.Tensor,
    padded_rows: cute.Tensor,
    actual_tiles: cute.Tensor,
    producer_ctas: cutlass.Int32,
    local_offset: cutlass.Int32,
    nrep: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
):
    """Totals, the four ranks' water fill over the hot rows, this rank's slot offsets and tile maps
    (256 threads, one CTA)."""
    slots = _LOCAL_EXPERTS + nrep
    tid, _, _ = cute.arch.thread_idx()
    hot_index = cutlass.Array(
        cutlass.Int32, _EXPERTS, space=cutlass.AddressSpace.smem, alignment=128
    )
    totals = cutlass.Array(cutlass.Int32, _EXPERTS, space=cutlass.AddressSpace.smem, alignment=128)
    # [0..3] rank loads (rows), [4] hot rows, [5] segment lo, [6] segment hi
    scalars = cutlass.Array(cutlass.Int32, 8, space=cutlass.AddressSpace.smem, alignment=128)
    hot_base = cutlass.Array(cutlass.Int32, nrep, space=cutlass.AddressSpace.smem, alignment=128)
    slot_tiles = cutlass.Array(cutlass.Int32, slots, space=cutlass.AddressSpace.smem, alignment=128)
    slot_first = cutlass.Array(cutlass.Int32, slots, space=cutlass.AddressSpace.smem, alignment=128)
    if tid < cutlass.Int32(8):
        scalars.subview(tid).store(cutlass.Int32(0))
    _load_hot_index(hot_index, hot_ids, tid, nrep)
    if cutlass.const_expr(use_pdl):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    # per-expert totals over the producer CTAs
    total = cutlass.Int32(0)
    for block in range(producer_ctas):
        total += cutlass.Int32(partial_counts[block * cutlass.Int32(_EXPERTS) + tid])
    totals.subview(tid).store(total)
    if hot_index.subview(tid).load() >= cutlass.Int32(0):
        prims.atomicrmw("add", scalars.subview(_EP_RANKS), total)
    else:
        tile_rows = (
            (total + cutlass.Int32(_TILE - 1)) // cutlass.Int32(_TILE) * cutlass.Int32(_TILE)
        )
        prims.atomicrmw("add", scalars.subview(tid // cutlass.Int32(_LOCAL_EXPERTS)), tile_rows)
    prims.barrier_cta_sync(0)
    if tid == cutlass.Int32(0):
        hot_rows = scalars.subview(_EP_RANKS).load()
        loads = cutlass.Array(cutlass.Int32, _EP_RANKS)
        top = cutlass.Int32(0)
        for r in cutlass.range_constexpr(_EP_RANKS):
            loads[r] = scalars.subview(r).load()
            top = cute.math.max(top, loads[r])
        # water level: the least level whose fill reaches the hot rows
        lo = cutlass.Int32(0)
        hi = top + hot_rows + cutlass.Int32(1)
        while lo < hi:
            mid = (lo + hi) // cutlass.Int32(2)
            fill = cutlass.Int32(0)
            for r in cutlass.range_constexpr(_EP_RANKS):
                fill += cute.math.max(cutlass.Int32(0), mid - loads[r])
            if fill >= hot_rows:
                hi = mid
            else:
                lo = mid + cutlass.Int32(1)
        need = cutlass.Array(cutlass.Int32, _EP_RANKS)
        given = cutlass.Int32(0)
        for r in cutlass.range_constexpr(_EP_RANKS):
            need[r] = cute.math.max(cutlass.Int32(0), lo - cutlass.Int32(1) - loads[r])
            given += need[r]
        rest = hot_rows - given
        for r in cutlass.range_constexpr(_EP_RANKS):
            if rest > cutlass.Int32(0) and loads[r] < lo:
                need[r] = need[r] + cutlass.Int32(1)
                rest -= cutlass.Int32(1)
        me = local_offset // cutlass.Int32(_LOCAL_EXPERTS)
        start = cutlass.Int32(0)
        for r in cutlass.range_constexpr(_EP_RANKS):
            if me == cutlass.Int32(r):
                scalars.subview(5).store(start)
                scalars.subview(6).store(start + need[r])
            start += need[r]
        run = cutlass.Int32(0)
        for h in cutlass.range_constexpr(nrep):
            hot_base.subview(h).store(run)
            e = cutlass.Int32(hot_ids[h])
            if e >= cutlass.Int32(0):
                run += totals.subview(e).load()
    prims.barrier_cta_sync(0)
    # this rank's slots: per-CTA offsets (-1: another rank owns that CTA's rows of a hot expert)
    if tid < cutlass.Int32(slots):
        count = cutlass.Int32(0)
        if tid < cutlass.Int32(_LOCAL_EXPERTS):
            e = local_offset + tid
            if hot_index.subview(e).load() < cutlass.Int32(0):
                for block in range(producer_ctas):
                    p = cutlass.Int32(partial_counts[block * cutlass.Int32(_EXPERTS) + e])
                    slot_offsets[block * cutlass.Int32(slots) + tid] = count
                    count += p
        else:
            h = tid - cutlass.Int32(_LOCAL_EXPERTS)
            e = cutlass.Int32(hot_ids[h])
            if e >= cutlass.Int32(0):
                seg_lo = scalars.subview(5).load()
                seg_hi = scalars.subview(6).load()
                position = hot_base.subview(h).load()
                for block in range(producer_ctas):
                    p = cutlass.Int32(partial_counts[block * cutlass.Int32(_EXPERTS) + e])
                    own = cutlass.Int32(-1)
                    if position >= seg_lo and position < seg_hi:
                        own = count
                        count += p
                    slot_offsets[block * cutlass.Int32(slots) + tid] = own
                    position += p
        slot_tiles.subview(tid).store((count + cutlass.Int32(_TILE - 1)) // cutlass.Int32(_TILE))
        slot_first.subview(tid).store(count)
    prims.barrier_cta_sync(0)
    if tid == cutlass.Int32(0):
        run = cutlass.Int32(0)
        for s in cutlass.range_constexpr(slots):
            t = slot_tiles.subview(s).load()
            slot_tiles.subview(s).store(run)
            run += t
        padded_rows[0] = run * cutlass.Int32(_TILE)
        actual_tiles[0] = run
    prims.barrier_cta_sync(0)
    if tid < cutlass.Int32(slots):
        count = slot_first.subview(tid).load()
        first = slot_tiles.subview(tid).load()
        tiles = (count + cutlass.Int32(_TILE - 1)) // cutlass.Int32(_TILE)
        base = first * cutlass.Int32(_TILE)
        expert_bases[tid] = base
        for tile in range(tiles):
            q = first + tile
            tile_expert[q] = tid
            tile_limit[q] = cute.math.min(
                (q + cutlass.Int32(1)) * cutlass.Int32(_TILE), base + count
            )
    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if tid == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.kernel
def rep_scatter_kernel(
    selected: cute.Tensor,
    hot_ids: cute.Tensor,
    slot_offsets: cute.Tensor,
    expert_bases: cute.Tensor,
    inverse: cute.Tensor,
    permutation: cute.Tensor,
    tokens: cutlass.Int32,
    local_offset: cutlass.Int32,
    producer_ctas: cutlass.Int32,
    warps: cutlass.Constexpr,
    nrep: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
):
    """Consume each claim once; overwrite it with this rank's row, or -1."""
    slots = _LOCAL_EXPERTS + nrep
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    hot_index = cutlass.Array(
        cutlass.Int32, _EXPERTS, space=cutlass.AddressSpace.smem, alignment=128
    )
    _load_hot_index(hot_index, hot_ids, tid, nrep)
    if cutlass.const_expr(use_pdl):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    expanded = block * cutlass.Int32(256) + tid
    if expanded < tokens * cutlass.Int32(_TOP_K):
        e = cutlass.Int32(selected[expanded])
        token = expanded // cutlass.Int32(_TOP_K)
        owner = (token // cutlass.Int32(warps)) % producer_ctas
        h = hot_index.subview(e).load()
        slot = cutlass.Int32(-1)
        if h >= cutlass.Int32(0):
            slot = cutlass.Int32(_LOCAL_EXPERTS) + h
        else:
            local = e - local_offset
            if local >= cutlass.Int32(0) and local < cutlass.Int32(_LOCAL_EXPERTS):
                slot = local
        row = cutlass.Int32(-1)
        if slot >= cutlass.Int32(0):
            off = cutlass.Int32(slot_offsets[owner * cutlass.Int32(slots) + slot])
            if off >= cutlass.Int32(0):
                row = cutlass.Int32(expert_bases[slot]) + off + cutlass.Int32(inverse[expanded])
                permutation[row] = expanded
        inverse[expanded] = row
    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if tid == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


rep_histogram_kernel.set_name_prefix("glm52_prefill_topk_rep_histogram", remove_cutlass_symbol=True)
rep_prefix_kernel.set_name_prefix("glm52_prefill_topk_rep_prefix", remove_cutlass_symbol=True)
rep_scatter_kernel.set_name_prefix("glm52_prefill_topk_rep_scatter", remove_cutlass_symbol=True)


@cute.jit
def launch_topk_routing_rep(
    logits: cute.Tensor,
    bias: cute.Tensor,
    hot_ids: cute.Tensor,
    weights: cute.Tensor,
    selected: cute.Tensor,
    partial_counts: cute.Tensor,
    slot_offsets: cute.Tensor,
    expert_bases: cute.Tensor,
    tile_expert: cute.Tensor,
    tile_limit: cute.Tensor,
    inverse: cute.Tensor,
    permutation: cute.Tensor,
    padded_rows: cute.Tensor,
    actual_tiles: cute.Tensor,
    tokens: cutlass.Int32,
    local_offset: cutlass.Int32,
    routed_scale: cutlass.Float32,
    warps: cutlass.Constexpr,
    maximum_ctas: cutlass.Constexpr,
    logit_splits: cutlass.Constexpr,
    nrep: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    """Histogram (``ctas`` x ``warps`` warps), prefix (one CTA, 256 threads), scatter."""
    if cutlass.const_expr(warps not in (8, 16, 32)):
        raise ValueError("warps must be 8, 16 or 32")
    if cutlass.const_expr(nrep < 1 or nrep > 16):
        raise ValueError("nrep must be 1..16")
    blocks = cute.math.min(cute.ceil_div(tokens, warps), cutlass.Int32(maximum_ctas))
    rep_histogram_kernel(
        logits,
        bias,
        weights,
        selected,
        inverse,
        partial_counts,
        tokens,
        routed_scale,
        warps,
        logit_splits,
        use_pdl,
    ).launch(
        grid=(blocks, 1, 1),
        block=(warps * 32, 1, 1),
        min_blocks_per_mp=1,
        stream=stream,
        use_pdl=use_pdl,
    )
    rep_prefix_kernel(
        partial_counts,
        hot_ids,
        slot_offsets,
        expert_bases,
        tile_expert,
        tile_limit,
        padded_rows,
        actual_tiles,
        blocks,
        local_offset,
        nrep,
        use_pdl,
    ).launch(grid=(1, 1, 1), block=(_EXPERTS, 1, 1), stream=stream, use_pdl=use_pdl)
    rep_scatter_kernel(
        selected,
        hot_ids,
        slot_offsets,
        expert_bases,
        inverse,
        permutation,
        tokens,
        local_offset,
        blocks,
        warps,
        nrep,
        use_pdl,
    ).launch(
        grid=(cute.ceil_div(tokens * _TOP_K, 256), 1, 1),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def max_tiles(tokens: int, nrep: int) -> int:
    """Tile records the maps may use: every slot can end in one partial tile."""
    return (tokens * _TOP_K + _TILE - 1) // _TILE + _LOCAL_EXPERTS + nrep - 1


def compile_topk_routing_rep(
    *, nrep: int, logit_splits: int = 1, warps: int = 16, use_pdl: bool = False
) -> Callable[..., None]:
    """17-argument handle: ``(logits, bias, hot_ids, weights, selected, partial_counts [ctas*256],
    slot_offsets [ctas*(64+nrep)], expert_bases [64+nrep], tile_expert, tile_limit, inverse,
    permutation, padded_rows, actual_tiles, T, local_offset, routed_scale)`` (148 producer CTAs,
    symbolic T)."""
    if type(nrep) is not int or not 1 <= nrep <= 16:
        raise ValueError("nrep must be an int in 1..16")
    if type(logit_splits) is not int or logit_splits not in (1, 2, 3, 4, 5):
        raise ValueError("native router split count must be one of 1..5")
    if type(warps) is not int or warps not in (8, 16, 32):
        raise ValueError("histogram width must be 8, 16 or 32 warps")
    if type(use_pdl) is not bool:
        raise TypeError("use_pdl must be an explicit bool")
    tokens, tiles = cute.sym_int64(), cute.sym_int64()

    def fake(dtype, size):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (size,), stride_order=(0,), assumed_align=16
        )

    slots = _LOCAL_EXPERTS + nrep
    return cute.compile(
        launch_topk_routing_rep,
        fake(cutlass.Float32, tokens * logit_splits * _EXPERTS),
        fake(cutlass.Float32, _EXPERTS),
        fake(cutlass.Int32, nrep),
        fake(cutlass.Float32, tokens * _TOP_K),
        fake(cutlass.Int32, tokens * _TOP_K),
        fake(cutlass.Int32, _PRODUCER_CTAS * _EXPERTS),
        fake(cutlass.Int32, _PRODUCER_CTAS * slots),
        fake(cutlass.Int32, slots),
        fake(cutlass.Int32, tiles),
        fake(cutlass.Int32, tiles),
        fake(cutlass.Int32, tokens * _TOP_K),
        fake(cutlass.Int32, tiles * _TILE),
        fake(cutlass.Int32, 1),
        fake(cutlass.Int32, 1),
        cutlass.Int32(64),
        cutlass.Int32(0),
        cutlass.Float32(2.5),
        warps,
        _PRODUCER_CTAS,
        logit_splits,
        nrep,
        use_pdl,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --generate-line-info",
    )


def buffers(T: int, nrep: int, dev):
    """The handle's twelve output / workspace tensors, poisoned."""
    n = T * _TOP_K
    q = max_tiles(T, nrep)
    slots = _LOCAL_EXPERTS + nrep
    i32 = dict(dtype=torch.int32, device=dev)
    return [
        torch.zeros(n, dtype=torch.float32, device=dev),  # weights
        torch.full((n,), -7, **i32),  # selected ids
        torch.full((_PRODUCER_CTAS * _EXPERTS,), -7, **i32),  # count slab
        torch.full((_PRODUCER_CTAS * slots,), -7, **i32),  # slot offsets
        torch.full((slots,), -7, **i32),  # slot bases
        torch.full((q,), -7, **i32),  # tile -> slot
        torch.full((q,), -7, **i32),  # tile -> mn limit
        torch.full((n,), -7, **i32),  # inverse
        torch.full((q * _TILE,), -7, **i32),  # permutation
        torch.full((1,), -7, **i32),  # padded rows
        torch.full((1,), -7, **i32),  # actual tiles
    ]


def reference_owner(ids, T: int, warps: int, hot: list) -> "torch.Tensor":
    """Host definition of the assignment: the EP rank that maps each (token, k) pair, ``[T, 8]``."""
    ids = ids.view(T, _TOP_K).long().cpu()
    ctas = min((T + warps - 1) // warps, _PRODUCER_CTAS)
    cta = (torch.arange(T) // warps) % ctas
    slab = torch.zeros(ctas, _EXPERTS, dtype=torch.long)
    slab.index_put_(
        (cta.repeat_interleave(_TOP_K), ids.flatten()),
        torch.ones(T * _TOP_K, dtype=torch.long),
        accumulate=True,
    )
    total = slab.sum(0)
    hot_index = {int(e): h for h, e in enumerate(hot) if int(e) >= 0}
    loads = [0] * _EP_RANKS
    hot_rows = 0
    for e in range(_EXPERTS):
        if e in hot_index:
            hot_rows += int(total[e])
        else:
            loads[e // _LOCAL_EXPERTS] += (int(total[e]) + _TILE - 1) // _TILE * _TILE
    lo, hi = 0, max(loads) + hot_rows + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(max(0, mid - x) for x in loads) >= hot_rows:
            hi = mid
        else:
            lo = mid + 1
    need = [max(0, lo - 1 - x) for x in loads]
    rest = hot_rows - sum(need)
    for r in range(_EP_RANKS):
        if rest > 0 and loads[r] < lo:
            need[r] += 1
            rest -= 1
    seg = []
    start = 0
    for r in range(_EP_RANKS):
        seg.append((start, start + need[r]))
        start += need[r]
    owner_hot = {}
    run = 0
    for h, e in enumerate(hot):
        e = int(e)
        if e < 0:
            continue
        pos = run
        for c in range(ctas):
            p = int(slab[c, e])
            owner_hot[(e, c)] = next((r for r, (a, b) in enumerate(seg) if a <= pos < b), -1)
            pos += p
        run += int(total[e])
    own = torch.empty(T, _TOP_K, dtype=torch.long)
    for t in range(T):
        for k in range(_TOP_K):
            e = int(ids[t, k])
            own[t, k] = owner_hot[(e, int(cta[t]))] if e in hot_index else e // _LOCAL_EXPERTS
    return own
