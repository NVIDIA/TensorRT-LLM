# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed M3 H6144 gather/raw residual/RMS, reusing pinned K3 norm kernel.

128 threads x 8 BF16 x 6 iterations covers the row. Unlike the original
K3 wrapper this target does not promise FlashInfer's bit-identical reduction.
"""

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

NORM_THREADS = 128
NORM_VEC = 8
NORM_WARPS = 4  # capacity: fixed: the CTA's warps


@cute.kernel
def m3_embed_norm_kernel(
    ids: cute.Tensor,  # int32 or int64 [N]
    table: cute.Tensor,  # bf16 [V, H]
    weight: cute.Tensor,  # bf16 [H]
    raw: cute.Tensor,  # bf16 [N, H], out: the rows
    out: cute.Tensor,  # bf16 [N, H], out: the normed rows
    vocab: cutlass.Int64,
    eps: cutlass.Float32,
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
    hidden: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    prims.griddepcontrol(prims.GridDepAction.WAIT)

    smem = cutlass.utils.SmemAllocator()
    reduction_buffer = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((1, NORM_WARPS)), byte_alignment=4
    )

    tok = cutlass.Int64(ids[bidx])
    valid = (tok >= cutlass.Int64(0)) & (tok < vocab)
    row = cutlass.Int64(cutlass.select_(valid, tok, cutlass.Int64(0)))

    gX = cute.local_tile(table, tiler_mn, (row, 0))
    gR = cute.local_tile(raw, tiler_mn, (bidx, 0))
    gY = cute.local_tile(out, tiler_mn, (bidx, 0))
    w_layout = cute.prepend(weight.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
    gW = cute.local_tile(cute.make_tensor(weight.iterator, w_layout), tiler_mn, (0, 0))

    copy_atom_load = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), table.element_type, num_bits_per_copy=128
    )
    copy_atom_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), out.element_type, num_bits_per_copy=128
    )
    thr_copy_X = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_W = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_O = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn).get_slice(tidx)

    tXgX = thr_copy_X.partition_S(gX)
    tXrX = cute.make_fragment_like(tXgX)
    tWgW = thr_copy_W.partition_S(gW)
    tWrW = cute.make_fragment_like(tWgW)
    tXrW = thr_copy_X.retile(tWrW)
    tXgR = thr_copy_O.partition_D(gR)
    tXgO = thr_copy_O.partition_D(gY)
    tXrO = cute.make_fragment_like(tXgO)

    # The row (zero for an id outside [0, V)), straight to the snapshot slot; the weight.
    tXrX.store(cute.zeros_like(tXrX, dtype=table.element_type))
    if valid:
        cute.copy(copy_atom_load, tXgX, tXrX)
    cute.copy(copy_atom_load, tWgW, tWrW)
    cute.copy(copy_atom_store, tXrX, tXgR)

    # flashinfer's RMSNormKernel arithmetic, in its order (see the header).
    x = tXrX.load().to(cutlass.Float32)
    x_sq = x * x
    sum_sq = x_sq.reduce(cute.ReductionOp.ADD, init_val=cutlass.Float32(0.0), reduction_profile=0)
    for i in cutlass.range_constexpr(5):
        sum_sq = sum_sq + cute.arch.shuffle_sync_bfly(sum_sq, offset=1 << i)
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    if lane == 0:
        reduction_buffer[0, warp] = sum_sq
    cute.arch.barrier()
    total = cutlass.Float32(0.0)
    if lane < NORM_WARPS:
        total = reduction_buffer[0, lane]
    for i in cutlass.range_constexpr(5):
        total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << i)
    mean_sq = total / cutlass.Float32(hidden)
    rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
    w = tXrW.load().to(cutlass.Float32)
    y = x * rstd * (w + cutlass.Float32(0.0))
    tXrO.store(y.to(out.element_type))
    cute.copy(copy_atom_store, tXrO, tXgO)


@cute.jit
def m3_embed_norm(
    ids: cute.Tensor,
    table: cute.Tensor,
    weight: cute.Tensor,
    raw: cute.Tensor,
    out: cute.Tensor,
    vocab: cutlass.Int64,
    eps: cutlass.Float32,
    n_tokens: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    # flashinfer's thread-value layout for one row: thread t, value (v, k) -> column 8 t + v + 1024 k.
    blocks = hidden // (NORM_VEC * NORM_THREADS)
    tv_layout = cute.make_layout(((NORM_THREADS, 1), (NORM_VEC, blocks)),
                                 stride=((NORM_VEC, 1), (1, NORM_VEC * NORM_THREADS)))  # fmt: skip
    m3_embed_norm_kernel(
        ids, table, weight, raw, out, vocab, eps, tv_layout, (1, hidden), hidden
    ).launch(
        grid=[n_tokens, 1, 1],
        block=[NORM_THREADS, 1, 1],
        smem=NORM_WARPS * 4,
        stream=stream,
        use_pdl=use_pdl,
    )
