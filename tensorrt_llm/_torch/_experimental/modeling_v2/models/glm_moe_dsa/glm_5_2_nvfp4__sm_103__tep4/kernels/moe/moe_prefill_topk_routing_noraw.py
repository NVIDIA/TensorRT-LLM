# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large-T GLM prefill TopK routing with a cheaper histogram stage (noraw top-k).

``moe_prefill_topk_routing`` (served) selects one token per warp: each lane keeps eight packed
(score, index) keys and the eight unbiased sigmoids, and each of the eight selection rounds scans those
sigmoids for the winner and shuffles the value to lane k. At T=7310 the stage issues 6.16M warp
instructions (~842 per token) at 0.58 IPC per scheduler -- it is bound by issued instructions, so more
resident warps (32 per CTA: 11.4 -> 10.6 us) or two tokens per warp (11.4 -> 11.0 us, 96 registers) buy
little, and in the MoE layer region both lose to the shared expert's down projection they crowd out.

``noraw`` removes the per-round lookup: lane k recomputes its own winner's sigmoid once, after the rounds,
from the same logit with the same expression -- the same float -- and keeps no sigmoid array: 4.33M
instructions, 11.4 -> 9.2 us at T=7310 alone, ~2-3 us per MoE layer in the region.

Everything else is the served kernel's: the arithmetic, the 148-CTA per-CTA count slab, the token to
producer-CTA ownership ``(token // warps) % ctas`` and the served prefix and scatter kernels. TopK ids and
weights, counts, expert bases, tile maps and the padded / actual tile scalars are bit-identical; as in the
served kernel the row order inside one expert is shared-memory atomic-claim order, so a comparison takes
those maps as per-expert row sets plus the inverse map.

`cute_dsl_moe_topk_routing` builds its 16-warp prefill width (above 512 rows) from this module.
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

from .moe_prefill_topk_routing import partial_prefix_kernel, partial_scatter_kernel
from .moe_topk_routing import (
    _EXPERTS,
    _FULL_MASK,
    _LOCAL_EXPERTS,
    _TOP_K,
    _VEC_SIZE,
    _WARP,
    _pack_score_index,
    _select_topk_and_count,
    _sort8_key_desc,
    _warp_sum_f32,
)

_PRODUCER_CTAS = 148


@cute.jit
def _publish(
    topk_weights,
    topk_indices,
    expert_count,
    token,
    lane,
    valid,
    my_raw,
    my_idx,
    weight_sum,
    local_expert_offset,
    routed_scaling_factor,
):
    """Lanes 0..7 write TopK and claim a CTA-local slot, exactly as the served helper."""
    my_local = cutlass.Int32(-1)
    my_slot = cutlass.Int32(0)
    if valid and lane < cutlass.Int32(_TOP_K):
        expanded = token * cutlass.Int32(_TOP_K) + lane
        topk_indices[expanded] = my_idx
        topk_weights[expanded] = (
            my_raw * routed_scaling_factor / (weight_sum + cutlass.Float32(1.0e-20))
        )
        local = my_idx - local_expert_offset
        if local >= cutlass.Int32(0) and local < cutlass.Int32(_LOCAL_EXPERTS):
            my_local = local
            my_slot = prims.atomicrmw("add", expert_count.subview(local), 1)
    return my_local, my_slot


@cute.jit
def _select_noraw_and_count(
    logits,
    logit_splits: cutlass.Constexpr,
    routing_bias,
    topk_weights,
    topk_indices,
    expert_count,
    token,
    lane,
    valid_token,
    local_expert_offset,
    routed_scaling_factor,
):
    """The served selection without the per-round winner-sigmoid lookup.

    The served helper keeps all eight unbiased sigmoids per lane and, every round, scans them for the
    winner and shuffles the value to lane k. Here lane k (< 8) recomputes its own winner's sigmoid once,
    after the eight rounds, from the same logit with the same expression -- the same float -- so ids,
    weights (same lane-k-holds-winner-k butterfly sum) and claims are unchanged, with ~150 fewer issued
    instructions per token and no 8-entry sigmoid array.
    """
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
    return _publish(
        topk_weights,
        topk_indices,
        expert_count,
        token,
        lane,
        valid_token,
        my_raw,
        my_selected_idx,
        weight_sum,
        local_expert_offset,
        routed_scaling_factor,
    )


@cute.kernel
def partial_histogram_noraw_kernel(
    logits: cute.Tensor,
    bias: cute.Tensor,
    weights: cute.Tensor,
    selected: cute.Tensor,
    claims: cute.Tensor,
    partial_counts: cute.Tensor,
    tokens: cutlass.Int32,
    local_offset: cutlass.Int32,
    routed_scale: cutlass.Float32,
    warps: cutlass.Constexpr,
    logit_splits: cutlass.Constexpr,
    noraw: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
):
    """The served histogram stage with `warps` up to 32, optionally with the winner sigmoid recomputed once
    per token instead of looked up every round (`noraw`)."""
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    lane = tid & cutlass.Int32(31)
    warp = tid >> cutlass.Int32(5)
    counts = cutlass.Array(cutlass.Int32, 64, space=cutlass.AddressSpace.smem, alignment=128)
    if tid < cutlass.Int32(64):
        counts.subview(tid).store(cutlass.Int32(0))
    prims.barrier_cta_sync(0)
    if cutlass.const_expr(use_pdl):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    stride = blocks * cutlass.Int32(warps)
    base = block * cutlass.Int32(warps)
    while base < tokens:
        token = base + warp
        if cutlass.const_expr(noraw):
            local, slot = _select_noraw_and_count(
                logits,
                logit_splits,
                bias,
                weights,
                selected,
                counts,
                token,
                lane,
                token < tokens,
                local_offset,
                routed_scale,
            )
        else:
            local, slot = _select_topk_and_count(
                logits,
                logit_splits,
                bias,
                weights,
                selected,
                counts,
                token,
                lane,
                token < tokens,
                local_offset,
                routed_scale,
            )
        if token < tokens and lane < cutlass.Int32(8):
            claim = cutlass.Int32(-1)
            if local >= cutlass.Int32(0):
                claim = slot
            claims[token * cutlass.Int32(8) + lane] = claim
        base += stride
    prims.barrier_cta_sync(0)
    if tid < cutlass.Int32(64):
        partial_counts[block * cutlass.Int32(64) + tid] = counts.subview(tid).load()
    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if tid == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


partial_histogram_noraw_kernel.set_name_prefix(
    "glm52_prefill_topk_partial_histogram_noraw", remove_cutlass_symbol=True
)


@cute.jit
def launch_topk_routing_noraw(
    logits: cute.Tensor,
    bias: cute.Tensor,
    weights: cute.Tensor,
    selected: cute.Tensor,
    partial_counts: cute.Tensor,
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
    noraw: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    """`moe_prefill_topk_routing.launch_topk_routing`'s ABI; only the histogram stage differs."""
    if cutlass.const_expr(warps not in (8, 16, 32)):
        raise ValueError("warps must be 8, 16 or 32")
    blocks = cute.math.min(cute.ceil_div(tokens, warps), cutlass.Int32(maximum_ctas))
    partial_histogram_noraw_kernel(
        logits,
        bias,
        weights,
        selected,
        inverse,
        partial_counts,
        tokens,
        local_offset,
        routed_scale,
        warps,
        logit_splits,
        noraw,
        use_pdl,
    ).launch(
        grid=(blocks, 1, 1),
        block=(warps * 32, 1, 1),
        min_blocks_per_mp=1,
        stream=stream,
        use_pdl=use_pdl,
    )
    partial_prefix_kernel(
        partial_counts,
        expert_bases,
        tile_expert,
        tile_limit,
        padded_rows,
        actual_tiles,
        blocks,
        use_pdl,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream, use_pdl=use_pdl)
    partial_scatter_kernel(
        selected,
        partial_counts,
        expert_bases,
        inverse,
        permutation,
        tokens,
        local_offset,
        blocks,
        warps,
        use_pdl,
    ).launch(
        grid=(cute.ceil_div(tokens * 8, 256), 1, 1),
        block=(256, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def compile_topk_routing_noraw(
    *, logit_splits: int, warps: int = 16, noraw: bool = True, use_pdl: bool = False
) -> Callable[..., None]:
    """Same 15-argument handle as `compile_topk_routing` (148 producer CTAs, symbolic T)."""
    if type(logit_splits) is not int or logit_splits not in (1, 2, 3, 4, 5):
        raise ValueError("native router split count must be one of 1..5")
    if type(warps) is not int or warps not in (8, 16, 32):
        raise ValueError("histogram width must be 8, 16 or 32 warps")
    if type(noraw) is not bool or type(use_pdl) is not bool:
        raise TypeError("noraw / use_pdl must be explicit bools")
    tokens, tiles = cute.sym_int64(), cute.sym_int64()

    def fake(dtype, size):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (size,), stride_order=(0,), assumed_align=16
        )

    return cute.compile(
        launch_topk_routing_noraw,
        fake(cutlass.Float32, tokens * logit_splits * 256),
        fake(cutlass.Float32, 256),
        fake(cutlass.Float32, tokens * 8),
        fake(cutlass.Int32, tokens * 8),
        fake(cutlass.Int32, _PRODUCER_CTAS * 64),
        fake(cutlass.Int32, 64),
        fake(cutlass.Int32, tiles),
        fake(cutlass.Int32, tiles),
        fake(cutlass.Int32, tokens * 8),
        fake(cutlass.Int32, tiles * 128),
        fake(cutlass.Int32, 1),
        fake(cutlass.Int32, 1),
        cutlass.Int32(64),
        cutlass.Int32(0),
        cutlass.Float32(2.5),
        warps,
        _PRODUCER_CTAS,
        logit_splits,
        noraw,
        use_pdl,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --generate-line-info",
    )
