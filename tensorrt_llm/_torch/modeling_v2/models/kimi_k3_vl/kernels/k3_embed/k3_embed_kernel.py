# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# Decode-size embedding row gather -- CTM (prims/cute) kernel
# =============================================================================
#
#   out[t, :] = table[ids[t], :]        t < N (a decode step's tokens), bf16 rows of H elements
#
# One 16-byte vector per thread: the grid covers N x H / 8 vectors, so every row is read by one wave of loads (one
# round trip after the grid dependency). The ids are the predecessor's output, read after griddepcontrol.wait; the
# dependents are launched at entry (they wait for this grid before reading out). An id outside [0, V) gives a zero
# row (the table is not read out of bounds).
#
# Norm mode (k3_embed_norm): the rows and the first layer's input RMSNorm in one launch
#
#   raw[t, :] = table[ids[t], :]                                      (the attention-residual snapshot bank's slot 0)
#   out[t, :] = raw[t, :] * rsqrt(mean(raw[t, :]^2) + eps) * weight
#
# Bit-identical to k3_embed followed by flashinfer.norm.rmsnorm, whose GB200 kernel is the CuTe DSL RMSNormKernel
# (flashinfer/norm/kernels/rmsnorm.py). For a row of 6144 < H <= 16384 elements that kernel runs one 128-thread CTA
# per row, and thread t holds columns 8 t + v + 1024 k. This kernel keeps that geometry, the same tiled copies and
# fragments, and the same DSL expressions in the same order:
#   - x * x, then the fragment's TensorSSA sum;
#   - a butterfly over offsets 1 .. 16, the 4 warp sums through shared memory, a butterfly again;
#   - sum / H, rsqrt(mean + eps) (fast-math), x * rstd * (w + 0).
# One CTA per token; no CTA reads another's writes.
# =============================================================================
"""CTM decode embedding: the rows of a replicated embedding table for a step's token ids, one launch (optionally
with the first layer's input RMSNorm)."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

THREADS = 256
VEC_WORDS = 4  # int32 words per 16-byte vector (8 bf16)
NORM_THREADS = 128  # flashinfer's threads per row for 6144 < H <= 16384 (one row per CTA)
NORM_VEC = 8  # bf16 per 16-byte copy
NORM_WARPS = NORM_THREADS // 32


def norm_supports_hidden(hidden: int) -> bool:
    """Rows this norm mode reproduces: flashinfer's one-CTA 128-thread geometry, and a tile that covers the row."""
    return 6144 < hidden <= 16384 and hidden % (NORM_VEC * NORM_THREADS) == 0


@cute.kernel
def k3_embed_kernel(
    ids: cutlass.Array,  # int32 or int64 [N]
    table: cutlass.Array,  # int32 words of the bf16 table [V * H / 2]
    out: cutlass.Array,  # int32 words of the bf16 output [N * H / 2]
    vocab: cutlass.Int32,
    n_tokens: cutlass.Constexpr[int],
    row_vecs: cutlass.Constexpr[int],  # H / 8
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    prims.griddepcontrol(prims.GridDepAction.WAIT)
    v = bx * cutlass.Int32(THREADS) + tx
    if v < cutlass.Int32(n_tokens * row_vecs):
        t = v // cutlass.Int32(row_vecs)
        c = v - t * cutlass.Int32(row_vecs)
        tok = cutlass.Int64(ids.load(idx=t))
        valid = (tok >= cutlass.Int64(0)) & (tok < cutlass.Int64(vocab))
        row = cutlass.Int64(cutlass.select_(valid, tok, cutlass.Int64(0)))
        e = table.load(idx=(row * cutlass.Int64(row_vecs) + cutlass.Int64(c)) * cutlass.Int64(VEC_WORDS),
                       vector_size=VEC_WORDS, alignment=16)  # fmt: skip
        zero = cutlass.Int32(0)
        out.store(
            (cutlass.Int32(cutlass.select_(valid, cutlass.Int32(e[0]), zero)),
             cutlass.Int32(cutlass.select_(valid, cutlass.Int32(e[1]), zero)),
             cutlass.Int32(cutlass.select_(valid, cutlass.Int32(e[2]), zero)),
             cutlass.Int32(cutlass.select_(valid, cutlass.Int32(e[3]), zero))),
            idx=v * cutlass.Int32(VEC_WORDS),
            alignment=16,
        )  # fmt: skip


@cute.jit
def k3_embed(
    ids: cute.Tensor,
    table: cute.Tensor,
    out: cute.Tensor,
    vocab: cutlass.Int32,
    n_tokens: cutlass.Constexpr[int],
    row_vecs: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    grid = (n_tokens * row_vecs + THREADS - 1) // THREADS
    k3_embed_kernel(ids, table, out, vocab, n_tokens, row_vecs).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )


@cute.kernel
def k3_embed_norm_kernel(
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
def k3_embed_norm(
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
    k3_embed_norm_kernel(
        ids, table, weight, raw, out, vocab, eps, tv_layout, (1, hidden), hidden
    ).launch(
        grid=[n_tokens, 1, 1],
        block=[NORM_THREADS, 1, 1],
        smem=NORM_WARPS * 4,
        stream=stream,
        use_pdl=use_pdl,
    )
