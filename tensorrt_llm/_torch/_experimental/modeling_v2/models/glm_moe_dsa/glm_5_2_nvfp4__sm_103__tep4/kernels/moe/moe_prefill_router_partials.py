# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic-T GLM BF16 router with explicit persistent pipeline slot/phase state.

Each producer/consumer stage index and parity survives every persistent work
item. The empty barrier starts being waited on only after the first ring fill;
its parity advances one ring behind the full barrier. This replaces per-K-step
signed division/modulo without changing barrier ownership, MMA order or output.
Stage descriptors are hoisted outside each group of four ascending K16 MMAs.
The N tile changes concurrency, not arithmetic partition or reduction order.
Every TMEM drain and output vector width follows that actual N tile.

A six-warp persistent CTA has a TMA producer, an MMA issuer and four TMEM
draining warps. Two accumulator slots overlap output stores with subsequent
tiles. The caller chooses the K partition as an explicit arithmetic contract:
changing it changes rounding and can change TopK. There is no automatic
production-policy selector in this module. The companion
``moe_prefill_frontend_schedule.propose_schedule`` selects geometry only.

Input A is contiguous BF16 [T,6144]; weights are BF16 [256,6144]; output is
contiguous FP32 [T,split_k,256]. The next TopK stage must sum split planes in
increasing split order in FP32. No atomic reduction or global initialization
is needed. Symbolic T permits one compile per schedule, not per arriving T.
For a non-dividing K partition (split_k=5), TMA zero-fills the final four
64-element tiles: both descriptors retain the true K extent of 6144 and use
the default zero OOB fill. Do not enlarge their global extent to the padded
partition size; the extra MMAs must consume zeros, not adjacent allocation data.

With PDL, immutable weights must be ready before launch. The producer can load
one weight tile before WAIT; A is first read after WAIT. The end-of-producer
signal admits dependents early but does not publish output readiness: their
WAIT must precede every partial-output read. Other-stream inputs still require
explicit event dependencies. This module does not install any serving path.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cuda.bindings import driver as cuda
from cutlass.base_dsl.array import L1EvictKind
from cutlass.experimental import primitives as prims


@cute.kernel
def projection_kernel(
    a_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    b_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    output: cute.Tensor,
    tokens: cutlass.Int32,
    columns: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    split_k: cutlass.Constexpr[int],
):
    tile_k: cutlass.Constexpr[int] = 64
    k_steps: cutlass.Constexpr[int] = (k_total + tile_k * split_k - 1) // (tile_k * split_k)
    n_tiles: cutlass.Constexpr[int] = cute.ceil_div(columns, tile_n)
    total_tiles = cute.ceil_div(tokens, tile_m) * n_tiles * split_k
    output_n: cutlass.Constexpr[int] = columns * split_k
    tmem_columns: cutlass.Constexpr[int] = tile_n * 2
    drain_width: cutlass.Constexpr[int] = min(tile_n, 64)
    a_stage: cutlass.Constexpr[int] = tile_m * tile_k
    b_stage: cutlass.Constexpr[int] = tile_n * tile_k
    tma_bytes: cutlass.Constexpr[int] = (a_stage + b_stage) * 2
    smem_a = cutlass.Array(
        cutlass.BFloat16,
        stages * a_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    smem_b = cutlass.Array(
        cutlass.BFloat16,
        stages * b_stage,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    full = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    empty = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    acc_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    acc_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_location = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    warp = cute.arch.warp_idx()
    lane = tid % 32
    if warp == 0:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(stages):
                prims.mbarrier_init(full.subview(stage), 1)
                prims.mbarrier_init(empty.subview(stage), 1)
            for slot in cutlass.range_constexpr(2):
                prims.mbarrier_init(acc_full.subview(slot), 1)
                prims.mbarrier_init(acc_empty.subview(slot), 128)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if warp == 0:
        prims.tcgen05_alloc(tmem_location, tmem_columns, group="cta_1")
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_location.load()
    tmem = cutlass.inttoptr(tmem_raw, 6, cutlass.Float32)
    prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    swizzle: cutlass.Constexpr = a_desc.swizzle.to(prims.Tcgen05SmemSwizzle)
    desc_a = prims.Tcgen05SmemDesc.build(
        start_address=smem_a,
        leading_byte_offset=128,
        stride_byte_offset=1024,
        layout=swizzle,
    )
    desc_b = prims.Tcgen05SmemDesc.build(
        start_address=smem_b,
        leading_byte_offset=128,
        stride_byte_offset=1024,
        layout=swizzle,
    )
    instruction = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        m_dim=tile_m,
        n_dim=tile_n,
    )

    if warp == 0:
        ordinal = cutlass.Int32(0)
        stage = cutlass.Int32(0)
        empty_phase = cutlass.Int32(0)
        for work in cutlass.range(block, total_tiles, grid):
            coord_m = ((work // split_k) // n_tiles) * tile_m
            coord_n = ((work // split_k) % n_tiles) * tile_n
            for k in cutlass.range(k_steps):
                if ordinal >= stages:
                    while not prims.mbarrier_try_wait_parity(empty.subview(stage), empty_phase):
                        pass
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(full.subview(stage), tma_bytes)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(stage * b_stage),
                        b_desc.get_ptr(),
                        ((work % split_k) * k_steps * tile_k + k * tile_k, coord_n),
                        full.subview(stage),
                    )
                    if cutlass.const_expr(use_pdl):
                        if ordinal == 0:
                            cute.arch.griddepcontrol_wait()
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(stage * a_stage),
                        a_desc.get_ptr(),
                        ((work % split_k) * k_steps * tile_k + k * tile_k, coord_m),
                        full.subview(stage),
                    )
                ordinal += 1
                stage += 1
                if stage == stages:
                    stage = cutlass.Int32(0)
                    if ordinal > stages:
                        empty_phase = empty_phase ^ 1

    if warp == 1:
        stage = cutlass.Int32(0)
        full_phase = cutlass.Int32(0)
        tile_iteration = cutlass.Int32(0)
        for work in cutlass.range(block, total_tiles, grid):
            slot = tile_iteration % 2
            if tile_iteration >= 2:
                phase = ((tile_iteration - 2) // 2) % 2
                while not prims.mbarrier_try_wait_parity(acc_empty.subview(slot), phase):
                    pass
            acc_pointer = prims.TmemAddr.from_row_col(
                prims.TmemAddr(tmem_raw).row_id,
                prims.TmemAddr(tmem_raw).col_id + slot * tile_n,
            ).as_ptr(cutlass.Float32)
            for k in cutlass.range(k_steps):
                while not prims.mbarrier_try_wait_parity(full.subview(stage), full_phase):
                    pass
                stage_desc_a = desc_a.advance_start_address(stage * a_stage * 2)
                stage_desc_b = desc_b.advance_start_address(stage * b_stage * 2)
                for kb in cutlass.range_constexpr(4):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_1,
                            acc_pointer,
                            stage_desc_a.advance_start_address(kb * 32),
                            stage_desc_b.advance_start_address(kb * 32),
                            instruction,
                            k > 0 if kb == 0 else True,
                        )
                if prims.elect_sync():
                    prims.tcgen05_commit(empty.subview(stage), group=prims.CTAGroup.CTA_1)
                stage += 1
                if stage == stages:
                    stage = cutlass.Int32(0)
                    full_phase = full_phase ^ 1
            if prims.elect_sync():
                prims.tcgen05_commit(acc_full.subview(slot), group=prims.CTAGroup.CTA_1)
            tile_iteration += 1

    if warp >= 2:
        tile_iteration = cutlass.Int32(0)
        row_in_tile = (warp % 4) * (tile_m // 4) + lane
        owns_row = lane < tile_m // 4
        global_output = output.iterator.raw_ptr()
        for work in cutlass.range(block, total_tiles, grid):
            slot = tile_iteration % 2
            phase = (tile_iteration // 2) % 2
            while not prims.mbarrier_try_wait_parity(acc_full.subview(slot), phase):
                pass
            row = ((work // split_k) // n_tiles) * tile_m + row_in_tile
            coord_n = ((work // split_k) % n_tiles) * tile_n
            for chunk in cutlass.range_constexpr(cute.ceil_div(tile_n, 64)):
                address = prims.TmemAddr.from_row_col(
                    prims.TmemAddr(tmem_raw).row_id,
                    prims.TmemAddr(tmem_raw).col_id + slot * tile_n + chunk * 64,
                ).as_ptr(cutlass.Float32)
                values = prims.tcgen05_ld("32x32b", address, num=drain_width)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                if cutlass.const_expr(chunk == cute.ceil_div(tile_n, 64) - 1):
                    # All TMEM reads precede this arrival. The next writer may
                    # reuse the slot while this warp completes register stores.
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.mbarrier_arrive(acc_empty.subview(slot))
                if owns_row and row < tokens:
                    registers = cutlass.Array(cutlass.Float32, drain_width, alignment=32)
                    registers.store(values)
                    for piece in cutlass.range_constexpr(drain_width // 8):
                        col = (work % split_k) * columns + coord_n + chunk * 64 + piece * 8
                        vector = registers.load(piece * 8, vector_size=8, alignment=32)
                        prims.store_ext(
                            vector.bitcast(cutlass.Int32),
                            global_output + row * output_n + col,
                            evict=L1EvictKind.NOALLOCATE,
                        )
            tile_iteration += 1
        prims.barrier_cta_sync(1, thread_count=128)
        if warp == 2:
            prims.tcgen05_dealloc(tmem, tmem_columns, group="cta_1")
    if cutlass.const_expr(use_pdl):
        # This is an early scheduling signal, not data publication. A consumer
        # must WAIT before reading partials; that waits for every CTA to finish.
        if tid == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def launch(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    tokens: cutlass.Int32,
    stream: cuda.CUstream,
    columns: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    grid_ctas: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    split_k: cutlass.Constexpr[int],
):
    a_desc = cuda_exp.create_tensor_map_tiled(
        global_address=a.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[k_total, tokens],
        global_strides=[k_total * 2 // 16],
        box_dims=[64, tile_m],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    b_desc = cuda_exp.create_tensor_map_tiled(
        global_address=b.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[k_total, columns],
        global_strides=[k_total * 2 // 16],
        box_dims=[64, tile_n],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    projection_kernel(
        a_desc,
        b_desc,
        c,
        tokens,
        columns,
        k_total,
        tile_m,
        tile_n,
        stages,
        use_pdl,
        split_k,
    ).launch(
        grid=(
            cute.math.min(cute.ceil_div(tokens, tile_m) * (columns // tile_n) * split_k, grid_ctas),
            1,
            1,
        ),
        block=(192, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


projection_kernel.set_name_prefix("glm52_prefill_router_partials", remove_cutlass_symbol=True)


def validate_schedule(split_k, physical_m, tile_n, stages, maximum_ctas, use_pdl):
    """Reject unsupported arithmetic, geometry and SMEM contracts before JIT."""
    for name, value in (
        ("split_k", split_k),
        ("physical_m", physical_m),
        ("tile_n", tile_n),
        ("stages", stages),
        ("maximum_ctas", maximum_ctas),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")
    if split_k not in (1, 2, 3, 4, 5, 8):
        raise ValueError("split_k must be 1, 2, 3, 4, 5 or 8")
    if physical_m not in (64, 128) or tile_n not in (16, 32, 64, 128, 256):
        raise ValueError("router requires M64/M128 and N16/N32/N64/N128/N256")
    if stages < 1 or maximum_ctas < 1:
        raise ValueError("stages and maximum_ctas must be positive")
    if type(use_pdl) is not bool:
        raise TypeError("use_pdl must be bool")
    smem_bytes = stages * (physical_m + tile_n) * 128 + 16 * stages + 36
    if smem_bytes > 227 * 1024:
        raise ValueError("router schedule exceeds B300 SMEM capacity")


def compile_router(
    *, split_k, physical_m=128, tile_n=128, stages=3, maximum_ctas=148, use_pdl=False
):
    """Compile a caller-owned handle(a, weights, partials, T) for T1..8192.

    The caller validates actual dtype, contiguous shape, device and alignment;
    all three views must be disjoint and retained until stream completion.
    This low-level boundary intentionally does not silently choose K splitting.
    A and weights are at least 16-byte aligned. Output must be 32-byte aligned
    because the epilogue issues256-bit stores. T must equal both tensor extents.
    Call validate_inputs before invoking this low-level handle on new views.
    """
    validate_schedule(split_k, physical_m, tile_n, stages, maximum_ctas, use_pdl)
    tokens = cute.sym_int64()
    fake = cute.runtime.make_fake_compact_tensor
    a = fake(cutlass.BFloat16, (tokens, 6144), stride_order=(1, 0), assumed_align=16)
    b = fake(cutlass.BFloat16, (256, 6144), stride_order=(1, 0), assumed_align=16)
    c = fake(
        cutlass.Float32,
        (tokens, split_k, 256),
        stride_order=(2, 1, 0),
        assumed_align=32,
    )
    return cute.compile(
        launch,
        a,
        b,
        c,
        cutlass.Int32(64),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        columns=256,
        k_total=6144,
        tile_m=physical_m,
        tile_n=tile_n,
        stages=stages,
        grid_ctas=maximum_ctas,
        use_pdl=use_pdl,
        split_k=split_k,
        options="--enable-tvm-ffi --generate-line-info",
    )


def validate_inputs(a, weights, partials, split_k):
    """Validate caller-owned views before raw-handle execution, without CUDA work.

    Returns the device index and runtime T. The views must remain alive through
    completion; validation does not establish stream readiness or choose a
    production-compatible arithmetic partition.
    """
    validate_schedule(split_k, 128, 128, 3, 148, False)
    values = (a, weights, partials)
    for name, value, dtype, alignment in zip(
        ("input", "weights", "partials"),
        values,
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (16, 16, 32),
    ):
        if not isinstance(value, torch.Tensor) or value.dtype != dtype:
            raise TypeError(f"{name} has the wrong tensor type or dtype")
        if not value.is_cuda or not value.is_contiguous():
            raise ValueError(f"{name} must be a contiguous CUDA tensor")
        if value.device != a.device:
            raise ValueError("router tensors must share a device")
        if value.data_ptr() % alignment:
            raise ValueError(f"{name} requires {alignment}-byte alignment")
    if a.ndim != 2 or a.shape[1] != 6144 or not 1 <= a.shape[0] <= 8192:
        raise ValueError("input must have shape [T,6144], T in 1..8192")
    tokens = a.shape[0]
    if tuple(weights.shape) != (256, 6144):
        raise ValueError("weights must have shape [256,6144]")
    if tuple(partials.shape) != (tokens, split_k, 256):
        raise ValueError("partials must have shape [T,split_k,256]")
    ranges = [
        (value.data_ptr(), value.data_ptr() + value.numel() * value.element_size())
        for value in values
    ]
    for index, (start, end) in enumerate(ranges):
        for other_start, other_end in ranges[index + 1 :]:
            if start < other_end and other_start < end:
                raise ValueError("router tensor views must not overlap")
    return a.device.index, tokens
