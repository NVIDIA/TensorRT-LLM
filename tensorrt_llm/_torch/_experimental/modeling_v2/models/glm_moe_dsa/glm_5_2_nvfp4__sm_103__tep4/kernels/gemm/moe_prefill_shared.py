# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 shared-expert projections for GLM prefill, T=65..8192.

Both NVFP4 transformer layers and the unquantized MTP layer use BF16 shared
experts. Up computes the interleaved gate/up projection, rounds the accumulators
through BF16 (the native GEMM boundary), applies SwiGLU, and emits BF16. Down
computes the linear projection and emits BF16. Local routed/shared addition is
deliberately separate so the same kernels support native serial and parallel
shared-expert execution with explicit event joins.

Each two-CTA cluster computes 256 rows using TMA and tcgen05 primitives. One
warp loads operands, one issues MMA, four drain alternating accumulator slots,
and a dedicated warp stores alternating epilogue buffers. All consumers signal
before either accumulator or shared-memory slot is reused. Actual tensor
extents clip the final tile; there is no padded-token input or output access.

The runtime callable does not specialize on T and rejects the decode domain.
Weights must be immutable for the invocation: their first TMA can precede the
PDL wait; activation reads cannot. PDL signals occur after all output stores.
These are candidate schedules, not automatic serving dispatch or performance
claims. Callers must preserve producer streams, event joins and buffer lifetime.
"""

from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cuda.bindings import driver as cuda
from cutlass.experimental import primitives as prims


@cute.kernel
def shared_projection_kernel(
    a_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    b_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    c_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    tokens: cutlass.Int32,
    columns: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    fused_swiglu: cutlass.Constexpr[bool],
    out_fp32: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
):
    tile_k: cutlass.Constexpr[int] = 64
    k_steps: cutlass.Constexpr[int] = k_total // tile_k
    n_tiles: cutlass.Constexpr[int] = cute.ceil_div(columns, tile_n)
    total_tiles = cute.ceil_div(tokens, tile_m * 2) * n_tiles
    # Swizzled TMA requires the contiguous box extent to be <=128 bytes.
    # FP32 and BF16 linear retain their original fragment sizes. SwiGLU now
    # reads 128 FP32 accumulators and emits 64 BF16 columns per transaction.
    # The two-slot output scratch doubles, without changing the AB stage count
    # in the tested N128/stages8 and N256/stages6 schedules.
    chunk_in: cutlass.Constexpr[int] = 32 if out_fp32 else 128 if fused_swiglu else 64
    chunk_out: cutlass.Constexpr[int] = 32 if out_fp32 else 64
    output_bytes: cutlass.Constexpr[int] = 4 if out_fp32 else 2
    c_row_bytes: cutlass.Constexpr[int] = chunk_out * output_bytes
    c_swizzle: cutlass.Constexpr = cutlass.Swizzle(3, 4, 3)
    smem_c = cutlass.Array(
        cutlass.Int8,
        2 * tile_m * c_row_bytes,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    tmem_columns: cutlass.Constexpr[int] = tile_n * 2
    a_stage: cutlass.Constexpr[int] = tile_m * tile_k
    b_stage: cutlass.Constexpr[int] = (tile_n // 2) * tile_k
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
    c_full = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    c_empty = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_location = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    rank = cute.arch.block_idx_in_cluster()
    group_id = block // 2
    groups = grid // 2
    warp = cute.arch.warp_idx()
    lane = tid % 32
    if warp == 0:
        if prims.elect_sync():
            for stage in cutlass.range_constexpr(stages):
                prims.mbarrier_init(full.subview(stage), 1)
                prims.mbarrier_init(empty.subview(stage), 1)
            for slot in cutlass.range_constexpr(2):
                prims.mbarrier_init(acc_full.subview(slot), 1)
                prims.mbarrier_init(acc_empty.subview(slot), 256)
                prims.mbarrier_init(c_full.subview(slot), 128)
                prims.mbarrier_init(c_empty.subview(slot), 1)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    if warp == 0:
        prims.tcgen05_alloc(tmem_location, tmem_columns, group="cta_2")
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_location.load()
    tmem = cutlass.inttoptr(tmem_raw, 6, cutlass.Float32)
    if warp == 0:
        prims.tcgen05_relinquish_alloc_permit(group="cta_2")
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
        m_dim=tile_m * 2,
        n_dim=tile_n,
    )

    if warp == 0:
        ordinal = cutlass.Int32(0)
        for work in cutlass.range(group_id, total_tiles, groups):
            coord_m = (work // n_tiles) * tile_m * 2 + rank * tile_m
            coord_n = (work % n_tiles) * tile_n
            for k in cutlass.range(k_steps):
                stage = ordinal % stages
                if ordinal >= stages:
                    phase = ((ordinal - stages) // stages) % 2
                    while not prims.mbarrier_try_wait_parity(empty.subview(stage), phase):
                        pass
                if prims.elect_sync():
                    if rank == 0:
                        prims.mbarrier_arrive_expect_tx(full.subview(stage), tma_bytes * 2)
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        smem_b.subview(stage * b_stage),
                        b_desc.get_ptr(),
                        (k * tile_k, coord_n + rank * (tile_n // 2)),
                        full.subview(stage),
                        [],
                        group=prims.CTAGroup.CTA_2,
                    )
                    if cutlass.const_expr(use_pdl):
                        if ordinal == 0:
                            cute.arch.griddepcontrol_wait()
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        smem_a.subview(stage * a_stage),
                        a_desc.get_ptr(),
                        (k * tile_k, coord_m),
                        full.subview(stage),
                        [],
                        group=prims.CTAGroup.CTA_2,
                    )
                ordinal += 1

    if warp == 1 and rank == 0:
        ordinal = cutlass.Int32(0)
        tile_iteration = cutlass.Int32(0)
        for work in cutlass.range(group_id, total_tiles, groups):
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
                stage = ordinal % stages
                phase = (ordinal // stages) % 2
                while not prims.mbarrier_try_wait_parity(full.subview(stage), phase):
                    pass
                for kb in cutlass.range_constexpr(4):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16,
                            prims.CTAGroup.CTA_2,
                            acc_pointer,
                            desc_a.advance_start_address(stage * a_stage * 2 + kb * 32),
                            desc_b.advance_start_address(stage * b_stage * 2 + kb * 32),
                            instruction,
                            k > 0 if kb == 0 else True,
                        )
                if prims.elect_sync():
                    prims.tcgen05_commit(
                        empty.subview(stage),
                        multicast_mask=3,
                        group=prims.CTAGroup.CTA_2,
                    )
                ordinal += 1
            if prims.elect_sync():
                prims.tcgen05_commit(
                    acc_full.subview(slot), multicast_mask=3, group=prims.CTAGroup.CTA_2
                )
            tile_iteration += 1

    if warp >= 2 and warp < 6:
        tile_iteration = cutlass.Int32(0)
        c_ordinal = cutlass.Int32(0)
        row_in_tile = (warp % 4) * (tile_m // 4) + lane
        for work in cutlass.range(group_id, total_tiles, groups):
            slot = tile_iteration % 2
            phase = (tile_iteration // 2) % 2
            while not prims.mbarrier_try_wait_parity(acc_full.subview(slot), phase):
                pass
            coord_n = (work % n_tiles) * tile_n
            for chunk in cutlass.range_constexpr(tile_n // chunk_in):
                c_slot = c_ordinal % 2
                if c_ordinal >= 2:
                    c_phase = ((c_ordinal - 2) // 2) % 2
                    while not prims.mbarrier_try_wait_parity(c_empty.subview(c_slot), c_phase):
                        pass
                address = prims.TmemAddr.from_row_col(
                    prims.TmemAddr(tmem_raw).row_id,
                    prims.TmemAddr(tmem_raw).col_id + slot * tile_n + chunk * chunk_in,
                ).as_ptr(cutlass.Float32)
                values = prims.tcgen05_ld("32x32b", address, num=chunk_in)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                if cutlass.const_expr(chunk == tile_n // chunk_in - 1):
                    # All TMEM reads precede this arrival. The next writer may
                    # reuse the slot while this warp completes register stores.
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.mbarrier_arrive(prims.mapa(acc_empty.subview(slot), cutlass.Int32(0)))
                if cutlass.const_expr(fused_swiglu):
                    activated = cutlass.Array(cutlass.Float32, chunk_out, alignment=32)
                    for i in cutlass.range_constexpr(chunk_out):
                        gate = values[2 * i].to(cutlass.BFloat16).to(cutlass.Float32)
                        up = values[2 * i + 1].to(cutlass.BFloat16).to(cutlass.Float32)
                        activated[i] = (
                            gate
                            * cute.math.rcp(
                                1.0 + cute.math.exp(-gate, fastmath=True), fastmath=True
                            )
                            * up
                        )
                    packed = (
                        activated.load(vector_size=chunk_out)
                        .to(cutlass.BFloat16)
                        .bitcast(cutlass.Int8)
                    )
                else:
                    if cutlass.const_expr(out_fp32):
                        packed = values.bitcast(cutlass.Int8)
                    else:
                        packed = values.to(cutlass.BFloat16).bitcast(cutlass.Int8)
                # Every epilogue thread owns one complete row. TMA clips the
                # final M tile against the descriptor's true token dimension.
                smem_c.subview(
                    (c_slot * tile_m + row_in_tile) * c_row_bytes
                ).data_ptr().store_swizzled(packed, alignment=c_row_bytes, swizzle=c_swizzle)
                prims.fence_proxy(kind=prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.mbarrier_arrive(c_full.subview(c_slot))
                c_ordinal += 1
            tile_iteration += 1
        prims.barrier_cta_sync(1, thread_count=128)
    if warp == 6:
        c_ordinal = cutlass.Int32(0)
        for work in cutlass.range(group_id, total_tiles, groups):
            coord_n = (work % n_tiles) * tile_n
            for chunk in cutlass.range_constexpr(tile_n // chunk_in):
                c_slot = c_ordinal % 2
                c_phase = (c_ordinal // 2) % 2
                while not prims.mbarrier_try_wait_parity(c_full.subview(c_slot), c_phase):
                    pass
                if prims.elect_sync():
                    output_col = (
                        coord_n // 2 + chunk * chunk_out
                        if fused_swiglu
                        else coord_n + chunk * chunk_out
                    )
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        c_desc.get_ptr(),
                        smem_c.subview(c_slot * tile_m * c_row_bytes),
                        (output_col, (work // n_tiles) * tile_m * 2 + rank * tile_m),
                    )
                    prims.cp_async_bulk_commit_group()
                    prims.cp_async_bulk_wait_group(0, read=True)
                    prims.mbarrier_arrive(c_empty.subview(c_slot))
                c_ordinal += 1
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    if warp == 2:
        prims.tcgen05_dealloc(tmem, tmem_columns, group="cta_2")
    if cutlass.const_expr(use_pdl):
        if tid == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def launch(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
    columns: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    grid_ctas: cutlass.Constexpr[int],
    fused_swiglu: cutlass.Constexpr[bool],
    out_fp32: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
):
    tokens = a.shape[0]
    total_tiles = cute.ceil_div(tokens, tile_m * 2) * cute.ceil_div(columns, tile_n)
    active_grid = cutlass.Int32(grid_ctas)
    if 2 * total_tiles < active_grid:
        active_grid = 2 * total_tiles
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
        box_dims=[64, tile_n // 2],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    out_columns = columns // 2 if fused_swiglu else columns
    out_bytes = 4 if out_fp32 else 2
    chunk_columns = 32 if out_fp32 else 64
    c_desc = cuda_exp.create_tensor_map_tiled(
        global_address=c.iterator.toint(),
        dtype=cutlass.Float32 if out_fp32 else cutlass.BFloat16,
        global_dims=[out_columns, tokens],
        global_strides=[out_columns * out_bytes // 16],
        box_dims=[chunk_columns, tile_m],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    shared_projection_kernel(
        a_desc,
        b_desc,
        c_desc,
        tokens,
        columns,
        k_total,
        tile_m,
        tile_n,
        stages,
        fused_swiglu,
        out_fp32,
        use_pdl,
    ).launch(
        grid=(active_grid, 1, 1),
        block=(256, 1, 1),
        cluster=(2, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def _validate_arguments(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    columns: int,
    reduction: int,
    fused_swiglu: bool,
    output_alignment: int,
) -> None:
    """Validate true runtime extents without reading device data or synchronizing."""
    tokens = a.shape[0] if a.ndim == 2 else 0
    if not 65 <= tokens <= 8192:
        raise ValueError("shared prefill requires 65..8192 token rows")
    output_columns = columns // 2 if fused_swiglu else columns
    shapes = ((tokens, reduction), (columns, reduction), (tokens, output_columns))
    for tensor, shape, alignment in zip((a, b, c), shapes, (128, 128, output_alignment)):
        if tuple(tensor.shape) != shape:
            raise ValueError("shared projection tensor shape mismatch")
        if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("shared projection requires contiguous BF16 tensors")
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("shared projection tensors must share a CUDA device")
        if tensor.data_ptr() % alignment:
            raise ValueError("shared projection pointer alignment mismatch")
    output_start = c.data_ptr()
    output_end = output_start + c.numel() * c.element_size()
    for tensor in (a, b):
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        if max(start, output_start) < min(end, output_end):
            raise ValueError("shared projection output must not overlap input storage")


def _compile_projection(
    *,
    fused_swiglu: bool,
    tile_n: int,
    stages: int,
    grid_ctas: int,
    use_pdl: bool,
    output_alignment: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]:
    if torch.cuda.get_device_capability() != (10, 3):
        raise ValueError("shared prefill projection requires an SM103 device")
    if type(use_pdl) is not bool:
        raise ValueError("use_pdl must be a bool")
    if type(tile_n) is not int or tile_n not in (128, 256):
        raise ValueError("tile_n must be 128 or 256")
    if type(stages) is not int or stages < 1:
        raise ValueError("stages must be positive")
    if type(grid_ctas) is not int or grid_ctas < 2 or grid_ctas % 2:
        raise ValueError("grid_ctas must be a positive multiple of two")
    if type(output_alignment) is not int or output_alignment not in (16, 32, 64, 128):
        raise ValueError("output_alignment must be 16, 32, 64 or 128 bytes")
    # Two BF16 operand rings, barrier arrays, and two 128-row output slots.
    smem_bytes = stages * (128 + tile_n // 2) * 128 + 16 * stages + 68 + 32768
    if smem_bytes > 227 * 1024:
        raise ValueError("shared projection exceeds the B300 shared-memory budget")
    columns, reduction = (1024, 6144) if fused_swiglu else (6144, 512)
    output_columns = columns // 2 if fused_swiglu else columns
    tokens = cute.sym_int32()
    fake = cute.runtime.make_fake_compact_tensor
    a = fake(cutlass.BFloat16, (tokens, reduction), stride_order=(1, 0), assumed_align=128)
    b = fake(cutlass.BFloat16, (columns, reduction), stride_order=(1, 0), assumed_align=128)
    c = fake(
        cutlass.BFloat16,
        (tokens, output_columns),
        stride_order=(1, 0),
        assumed_align=output_alignment,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    shared_projection_kernel.set_name_prefix(
        "glm52_moe_prefill_shared_" + ("up" if fused_swiglu else "down"),
        remove_cutlass_symbol=True,
    )
    raw = cute.compile(
        launch,
        a,
        b,
        c,
        stream,
        columns=columns,
        k_total=reduction,
        tile_m=128,
        tile_n=tile_n,
        stages=stages,
        grid_ctas=grid_ctas,
        fused_swiglu=fused_swiglu,
        out_fp32=False,
        use_pdl=use_pdl,
        options="--enable-tvm-ffi --generate-line-info",
    )

    def compiled(a_tensor: torch.Tensor, b_tensor: torch.Tensor, c_tensor: torch.Tensor):
        _validate_arguments(
            a_tensor,
            b_tensor,
            c_tensor,
            columns=columns,
            reduction=reduction,
            fused_swiglu=fused_swiglu,
            output_alignment=output_alignment,
        )
        return raw(a_tensor, b_tensor, c_tensor)

    return compiled


def compile_up(
    *,
    grid_ctas: int,
    use_pdl: bool,
    tile_n: int = 256,
    stages: int = 6,
    output_alignment: int = 128,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]:
    """Compile [T,6144] @ [1024,6144] -> BF16 SwiGLU [T,512].

    Weight rows must be interleaved gate,up,gate,up,... once at model loading,
    not packed during invocation. Native BF16 GEMM rounding precedes activation.
    """
    return _compile_projection(
        fused_swiglu=True,
        tile_n=tile_n,
        stages=stages,
        grid_ctas=grid_ctas,
        use_pdl=use_pdl,
        output_alignment=output_alignment,
    )


def compile_down(
    *,
    grid_ctas: int,
    use_pdl: bool,
    tile_n: int = 256,
    stages: int = 6,
    output_alignment: int = 128,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]:
    """Compile [T,512] @ [6144,512] -> BF16 [T,6144], without local merge."""
    return _compile_projection(
        fused_swiglu=False,
        tile_n=tile_n,
        stages=stages,
        grid_ctas=grid_ctas,
        use_pdl=use_pdl,
        output_alignment=output_alignment,
    )
