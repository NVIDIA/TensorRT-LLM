# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental large-T GLM TopK and compact-M128 routing in three stages.

The decoder's exact per-token selection arithmetic is retained. Each bounded
producer CTA accumulates a private shared histogram and publishes its64 counts,
while each local TopK lane retains its CTA-local slot in the inverse workspace.
One two-warp prefix stage converts all CTA counts into offsets, constructs the
compact expert tiles, and publishes the runtime tile count. Parallel scatter
combines expert base, CTA offset and local slot to form the final maps. This
avoids a separate global histogram reset and global slot-claim atomics.

Every invocation overwrites the active partial-count domain before reading it;
there is no persistent counter state. Unused capacity and padded permutation
rows are unspecified and must never be consumed past absolute mn_limit. A
permutation entry is token_id*8+topk_slot, not token_id. Experts may span multiple
M128 tiles; an empty local rank emits zero tiles.

PDL waits precede every input read and have matching launch attributes. Signals
are emitted only after CTA output stores. Consumers still require their wait;
launch order alone is not a memory fence. With PDL disabled, ordinary stream
ordering separates all three stages. Quantization, output-zero, shared-expert
and GG event dependencies remain caller-owned and are not removed here.

This module is a kernel-level experiment: no serving selector is enabled and no
performance benefit is asserted. The caller owns disjoint workspace per live
invocation/graph and retains all tensors through replay completion.
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

from .moe_topk_routing import _select_topk_and_count
from .mtp_moe_routing import _two_warp_exclusive_scan_i32


@cute.kernel
def partial_histogram_kernel(
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
    use_pdl: cutlass.Constexpr,
):
    """Write TopK, per-CTA local claims and a fully initialized count slab."""
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
    base = block * cutlass.Int32(warps)
    while base < tokens:
        token = base + warp
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
        base += blocks * cutlass.Int32(warps)
    prims.barrier_cta_sync(0)
    if tid < cutlass.Int32(64):
        partial_counts[block * cutlass.Int32(64) + tid] = counts.subview(tid).load()
    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if tid == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.kernel
def partial_prefix_kernel(
    partial_counts: cute.Tensor,
    expert_bases: cute.Tensor,
    tile_expert: cute.Tensor,
    tile_limit: cute.Tensor,
    padded_rows: cute.Tensor,
    actual_tiles: cute.Tensor,
    producer_ctas: cutlass.Int32,
    use_pdl: cutlass.Constexpr,
):
    """Convert count slabs to exclusive per-CTA offsets and compact tile maps."""
    expert, _, _ = cute.arch.thread_idx()
    lane = expert & cutlass.Int32(31)
    warp = expert >> cutlass.Int32(5)
    warp_totals = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=128)
    if cutlass.const_expr(use_pdl):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    count = cutlass.Int32(0)
    for block in range(producer_ctas):
        index = block * cutlass.Int32(64) + expert
        partial = cutlass.Int32(partial_counts[index])
        partial_counts[index] = count
        count += partial
    tiles = (count + cutlass.Int32(127)) // cutlass.Int32(128)
    offset, total = _two_warp_exclusive_scan_i32(tiles, lane, warp, warp_totals)
    base = offset * cutlass.Int32(128)
    expert_bases[expert] = base
    for tile in range(tiles):
        q = offset + tile
        tile_expert[q] = expert
        tile_limit[q] = cute.math.min((q + cutlass.Int32(1)) * cutlass.Int32(128), base + count)
    if expert == cutlass.Int32(0):
        padded_rows[0] = total * cutlass.Int32(128)
        actual_tiles[0] = total
    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if expert == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.kernel
def partial_scatter_kernel(
    selected: cute.Tensor,
    partial_offsets: cute.Tensor,
    expert_bases: cute.Tensor,
    inverse: cute.Tensor,
    permutation: cute.Tensor,
    tokens: cutlass.Int32,
    local_offset: cutlass.Int32,
    producer_ctas: cutlass.Int32,
    warps: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr,
):
    """Consume each private claim once, overwrite it with the public inverse."""
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(use_pdl):
        prims.griddepcontrol(prims.GridDepAction.WAIT)
    expanded = block * cutlass.Int32(256) + tid
    if expanded < tokens * cutlass.Int32(8):
        local = cutlass.Int32(selected[expanded]) - local_offset
        row = cutlass.Int32(-1)
        if local >= cutlass.Int32(0) and local < cutlass.Int32(64):
            token = expanded // cutlass.Int32(8)
            owner = (token // cutlass.Int32(warps)) % producer_ctas
            row = (
                cutlass.Int32(expert_bases[local])
                + cutlass.Int32(partial_offsets[owner * cutlass.Int32(64) + local])
                + cutlass.Int32(inverse[expanded])
            )
            permutation[row] = expanded
        inverse[expanded] = row
    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if tid == cutlass.Int32(0):
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


partial_histogram_kernel.set_name_prefix(
    "glm52_prefill_topk_partial_histogram", remove_cutlass_symbol=True
)
partial_prefix_kernel.set_name_prefix(
    "glm52_prefill_topk_partial_prefix", remove_cutlass_symbol=True
)
partial_scatter_kernel.set_name_prefix(
    "glm52_prefill_topk_partial_scatter", remove_cutlass_symbol=True
)


@cute.jit
def launch_topk_routing(
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
    use_pdl: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    """Low-level T1..8192 ABI; validate tensors/capacity before compiling.

    The first12 arguments are contiguous1-D tensor views. Partial-count capacity
    is maximum_ctas*64 int32 entries, bases64 entries, both public TopK tensors
    and inverse8*T entries. Allocate at least ceil(8*T/128)+63 tile records and
    that capacity*128 permutation entries. Last two scalars contain padded rows
    and actual M tiles. All data/writable views are disjoint except inverse is
    deliberately its own stage-local claims workspace.
    """
    if cutlass.const_expr(warps not in (4, 8, 16)):
        raise ValueError("warps must be 4, 8 or 16")
    if cutlass.const_expr(maximum_ctas < 1):
        raise ValueError("maximum_ctas must be positive")
    if cutlass.const_expr(logit_splits not in (1, 2, 3, 4, 5, 8)):
        raise ValueError("logit_splits must be 1, 2, 3, 4, 5 or 8")
    blocks = cute.math.min(cute.ceil_div(tokens, warps), cutlass.Int32(maximum_ctas))
    partial_histogram_kernel(
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
        use_pdl,
    ).launch(grid=(blocks, 1, 1), block=(warps * 32, 1, 1), stream=stream, use_pdl=use_pdl)
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


def compile_topk_routing(
    *, logit_splits: int, warps: int, use_pdl: bool = False
) -> Callable[..., None]:
    """Compile the selected 148-producer-CTA, symbolic-T packed tensor ABI.

    The returned handle accepts the first 15 arguments of
    ``launch_topk_routing``: 12 flat tensors, then T, *expert offset* (rank*64,
    not rank), and positive routing scale. The current Torch CUDA stream is
    supplied by TVM-FFI. The caller validates T1..8192, rank0..3, capacities,
    dtypes, device, disjoint storage and at least 16-byte alignment before
    launching, and retains each graph's workspace until replay completion.

    ``logit_splits`` is the separately verified numerical router association;
    it is not inferred from token count or this geometry. This convenience
    boundary covers the observed native families 1..5. The low-level JIT entry
    additionally accepts an explicitly prepared eight-partial input.
    """
    if type(logit_splits) is not int or logit_splits not in (1, 2, 3, 4, 5):
        raise ValueError("native router split count must be one of 1..5")
    if type(warps) is not int or warps not in (4, 8, 16):
        raise ValueError("qualified histogram width must be 4, 8 or 16 warps")
    if type(use_pdl) is not bool:
        raise TypeError("consumer PDL contract must be an explicit bool")
    tokens, tiles = cute.sym_int64(), cute.sym_int64()

    def fake(dtype, size):
        return cute.runtime.make_fake_compact_tensor(
            dtype, (size,), stride_order=(0,), assumed_align=16
        )

    return cute.compile(
        launch_topk_routing,
        fake(cutlass.Float32, tokens * logit_splits * 256),
        fake(cutlass.Float32, 256),
        fake(cutlass.Float32, tokens * 8),
        fake(cutlass.Int32, tokens * 8),
        fake(cutlass.Int32, 148 * 64),
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
        148,
        logit_splits,
        use_pdl,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --generate-line-info",
    )
