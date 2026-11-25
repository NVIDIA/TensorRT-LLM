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
"""MTP draft-step input glue in one kernel.

``DeepseekV3MTP`` feeds ``eh_proj`` with columns of
``cat([enorm(embed(input_ids)), hnorm(hidden_states)], -1)``. ``DraftGlueRMSNormKernel``
computes both RMSNorms, optionally gathers the embedding rows itself, and stores only the
requested columns of the concatenated rows.

Each row is normalized exactly as FlashInfer's CuTe ``RMSNormKernel`` does, the kernel
``flashinfer.norm.rmsnorm`` runs for 2D BF16 inputs: the thread geometry is taken from an
``RMSNormKernel`` instance (64 threads per row for H = 6144, 8-element vectors, the same
column order per thread), the reduction is FlashInfer's ``row_reduce_sum_multirow``, the
arithmetic statements are FlashInfer's, and the kernel is compiled with the same options.
The per-row code mirrors FlashInfer's ``QKRMSNormKernel``, which gives every row its own
tile so a row can come from any tensor.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import make_tv_layout, predicate_k, row_reduce_sum_multirow

_IDS_DTYPES = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}


@cute.jit
def _predicate_cols(
    tXcX: cute.Tensor, lo: cutlass.Constexpr[int], hi: cutlass.Constexpr[int]
) -> cute.Tensor:
    """Copy predicate that is true where a vector's first column lies in [lo, hi)."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            col = tXcX[(0, rest_v), 0, rest_k][1]
            in_window = cute.elem_less(col, hi)
            if cutlass.const_expr(lo > 0):
                in_window = in_window & cute.elem_less(lo - 1, col)
            tXpX[rest_v, 0, rest_k] = in_window
    return tXpX


class DraftGlueRMSNormKernel:
    """RMSNorm of the embedding rows (half 0) and of the hidden rows (half 1) of the MTP draft step.

    Column ``k`` of half ``h`` is column ``h * H + k`` of the concatenated row. The kernel computes
    the halves overlapping the column window ``[col_lo, col_hi)`` and stores the window's columns
    at ``out[row, col - out_col_shift]``. Each group of ``threads_per_row`` threads normalizes one
    (row, half) pair; with both halves a CTA covers one row.

    Args:
        hidden_size: H, the length of each normalized row.
        half_begin: First half overlapping the window.
        num_halves: Number of halves overlapping the window (1 or 2).
        gather: Half 0 reads ``table[input_ids[row]]`` (zero row for ids outside the table)
            instead of row ``row`` of a dense embedding tensor.
        out_cols: Row length of the output buffer.
        out_col_shift: Concatenated column stored at output column 0.
        col_lo: First concatenated column stored.
        col_hi: One past the last concatenated column stored.
    """

    def __init__(
        self,
        hidden_size: int,
        half_begin: int,
        num_halves: int,
        gather: bool,
        out_cols: int,
        out_col_shift: int,
        col_lo: int,
        col_hi: int,
    ):
        reference = RMSNormKernel(cutlass.BFloat16, hidden_size, weight_bias=0.0)
        if reference.cluster_n != 1:
            raise ValueError(
                f"hidden size {hidden_size} needs a CTA cluster per row in RMSNormKernel; "
                "the draft glue kernel covers single-CTA rows only"
            )
        self.H = hidden_size
        self.weight_bias = reference.weight_bias
        self.copy_bits = reference.copy_bits
        self.vec_size = reference.vec_size
        self.threads_per_row = reference.threads_per_row
        self.num_threads = reference.num_threads
        self.rows_per_block = reference.rows_per_block
        self.warps_per_row = reference.warps_per_row
        self.num_vec_blocks = reference.num_vec_blocks
        self.cols_per_tile = reference.cols_per_tile
        self.use_async_copy = reference.use_async_copy

        self.half_begin = half_begin
        self.num_halves = num_halves
        self.gather = gather
        self.out_cols = out_cols
        self.out_col_shift = out_col_shift
        # Stored columns [lo, hi) of each half, in that half's column coordinates.
        windows = []
        for h in range(2):
            lo = min(max(col_lo - h * hidden_size, 0), hidden_size)
            hi = max(min(col_hi - h * hidden_size, hidden_size), lo)
            windows.append((lo, hi))
        self.windows = tuple(windows)

    def _smem_size_in_bytes(self) -> int:
        tile_bytes = 0
        if self.use_async_copy:
            tile_bytes = self.rows_per_block * self.cols_per_tile * (cutlass.BFloat16.width // 8)
        return tile_bytes + self.rows_per_block * self.warps_per_row * 4

    @cute.jit
    def __call__(
        self,
        mSrc0: cute.Tensor,
        mIds: cute.Tensor,
        mSrc1: cute.Tensor,
        mW0: cute.Tensor,
        mW1: cute.Tensor,
        mOut: cute.Tensor,
        M: Int64,
        vocab: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = make_tv_layout(
            self.threads_per_row, self.vec_size, self.num_vec_blocks
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_1d = (1, self.cols_per_tile)

        self.kernel(
            mSrc0, mIds, mSrc1, mW0, mW1, mOut, M, vocab, eps, enable_pdl, tv_layout, tiler_1d
        ).launch(
            grid=[cute.ceil_div(M * self.num_halves, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mSrc0: cute.Tensor,
        mIds: cute.Tensor,
        mSrc1: cute.Tensor,
        mW0: cute.Tensor,
        mW1: cute.Tensor,
        mOut: cute.Tensor,
        M: Int64,
        vocab: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_1d: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: the ids and the hidden rows may come from the preceding kernel, so wait before
        # any load. The dependent (eh_proj) may launch right away; it waits before it reads
        # this kernel's output.
        if enable_pdl:
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        H = self.H
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = self.threads_per_row
        rows_per_block = self.rows_per_block
        warps_per_row = self.warps_per_row
        num_halves = self.num_halves

        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row
        slot = bidx * rows_per_block + row_in_block
        if cutlass.const_expr(num_halves == 1):
            row = slot
            is_half0 = self.half_begin == 0
        else:
            row = slot // num_halves
            is_half0 = slot % num_halves == 0
        row_in_bounds = row < M

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mSrc1.element_type,
                cute.make_ordered_layout((rows_per_block, self.cols_per_tile), order=(1, 0)),
                byte_alignment=16,
            )

        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )
        mbar_ptr = None
        cluster_n = 1

        # ===== Source rows =====
        # Half 0 reads the embedding row of the token, half 1 the hidden row.
        valid0 = row_in_bounds
        src_row0 = row
        if cutlass.const_expr(self.gather and self.half_begin == 0):
            # Same guard as the Embedding module: an id outside [0, vocab), such as the -1
            # padding of rejected draft tokens, gives a zero embedding row.
            token = Int64(-1)
            if row_in_bounds:
                token = Int64(mIds[row])
            valid0 = (token >= Int64(0)) & (token < vocab)
            src_row0 = cutlass.select_(valid0, token, Int64(0))
        gX0 = cute.local_tile(mSrc0, tiler_1d, (src_row0, 0))
        gX1 = cute.local_tile(mSrc1, tiler_1d, (row, 0))

        # ===== Output: the window columns of each half =====
        # Column k of half h is stored at out column h * H + k - out_col_shift.
        out_layout = cute.make_layout((M, H), stride=(self.out_cols, 1))
        mO0 = cute.make_tensor(mOut.iterator + (0 - self.out_col_shift), out_layout)
        mO1 = cute.make_tensor(mOut.iterator + (H - self.out_col_shift), out_layout)
        gO0 = cute.local_tile(mO0, tiler_1d, (row, 0))
        gO1 = cute.local_tile(mO1, tiler_1d, (row, 0))

        mW0_2d = cute.prepend_ones(mW0, up_to_rank=2)
        mW1_2d = cute.prepend_ones(mW1, up_to_rank=2)

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mSrc1.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mOut.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mSrc1.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_1d)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_1d)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_1d)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_1d)

        thr_copy_X = tiled_copy_load.get_slice(lane_in_row)
        thr_copy_W = tiled_copy_W.get_slice(lane_in_row)
        thr_copy_O = tiled_copy_store.get_slice(lane_in_row)

        # ===== Partition input =====
        tXgX0 = thr_copy_X.partition_S(gX0)
        tXgX1 = thr_copy_X.partition_S(gX1)
        tXrX = cute.make_fragment_like(tXgX1)

        if cutlass.const_expr(self.use_async_copy):
            sX_row = cute.local_tile(sX, tiler_1d, (row_in_block, 0))
            tXsX = thr_copy_X.partition_D(sX_row)

        # ===== Partition weight (sync, separate tiled copy) =====
        tWgW0 = thr_copy_W.partition_S(mW0_2d)
        tWgW1 = thr_copy_W.partition_S(mW1_2d)
        tWrW = cute.make_fragment_like(tWgW0)
        tXrW = thr_copy_X.retile(tWrW)

        # ===== Partition output =====
        tXgO0 = thr_copy_O.partition_D(gO0)
        tXgO1 = thr_copy_O.partition_D(gO1)
        tXrO = cute.make_fragment_like(tXgO0)

        # ===== Bounds checking =====
        id1d = cute.make_identity_tensor(tiler_1d)
        tXcX = thr_copy_X.partition_S(id1d)
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(id1d), limit=H)
        lo0, hi0 = self.windows[0]
        lo1, hi1 = self.windows[1]
        if cutlass.const_expr(lo0 == 0 and hi0 == H):
            tXpO0 = tXpX
        else:
            tXpO0 = _predicate_cols(tXcX, lo0, hi0)
        if cutlass.const_expr(lo1 == 0 and hi1 == H):
            tXpO1 = tXpX
        else:
            tXpO1 = _predicate_cols(tXcX, lo1, hi1)

        # ===== Pass 1: Load input + compute sum of squares =====
        if cutlass.const_expr(self.use_async_copy):
            if is_half0:
                if valid0:
                    cute.copy(copy_atom_async, tXgX0, tXsX, pred=tXpX)
                else:
                    tXrX.store(cute.zeros_like(tXrX, dtype=mSrc1.element_type))
                    cute.autovec_copy(tXrX, tXsX)
            else:
                if row_in_bounds:
                    cute.copy(copy_atom_async, tXgX1, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            if is_half0:
                cute.copy(copy_atom_sync, tWgW0, tWrW, pred=tWpW)
            else:
                cute.copy(copy_atom_sync, tWgW1, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mSrc1.element_type))
            if is_half0:
                if valid0:
                    cute.copy(copy_atom_sync, tXgX0, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tWgW0, tWrW, pred=tWpW)
            else:
                if row_in_bounds:
                    cute.copy(copy_atom_sync, tXgX1, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tWgW1, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        # Re-load x from shared memory to relieve register pressure.
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mOut.element_type))

        if row_in_bounds:
            if is_half0:
                cute.copy(copy_atom_store, tXrO, tXgO0, pred=tXpO0)
            else:
                cute.copy(copy_atom_store, tXrO, tXgO1, pred=tXpO1)


def draft_glue_matches_flashinfer(device: torch.device, hidden_size: int) -> bool:
    """Whether ``flashinfer.norm.rmsnorm`` runs ``RMSNormKernel`` with one CTA per row here.

    The draft glue kernel reproduces that kernel's numerics; FlashInfer's CUDA JIT norm and
    the cluster variant of ``RMSNormKernel`` (very large hidden sizes) are not covered.
    """
    import flashinfer.norm

    use_cuda_norm = getattr(flashinfer.norm, "_use_cuda_norm", None)
    if use_cuda_norm is None or use_cuda_norm(device):
        return False
    with torch.cuda.device(device):
        reference = RMSNormKernel(cutlass.BFloat16, hidden_size, weight_bias=0.0)
    return reference.cluster_n == 1


@functools.cache
def _compile_draft_glue_kernel(
    hidden_size: int,
    half_begin: int,
    num_halves: int,
    gather: bool,
    ids_dtype: torch.dtype,
    out_cols: int,
    out_col_shift: int,
    col_lo: int,
    col_hi: int,
    enable_pdl: bool,
):
    kernel_obj = DraftGlueRMSNormKernel(
        hidden_size, half_begin, num_halves, gather, out_cols, out_col_shift, col_lo, col_hi
    )

    sym_m = cute.sym_int(64)
    src0_rows = cute.sym_int(64) if gather else sym_m
    src0_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (src0_rows, hidden_size),
        (cute.sym_int64(divisibility=kernel_obj.vec_size), 1),
        assumed_align=16,
    )
    ids_fake = cute.runtime.make_fake_compact_tensor(_IDS_DTYPES[ids_dtype], (sym_m,))
    src1_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (sym_m, hidden_size),
        (cute.sym_int64(divisibility=kernel_obj.vec_size), 1),
        assumed_align=16,
    )
    w0_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (hidden_size,), assumed_align=16
    )
    w1_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (hidden_size,), assumed_align=16
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, out_cols), stride_order=(1, 0), assumed_align=16
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Same compile options as FlashInfer's RMSNorm kernels.
    return cute.compile(
        kernel_obj,
        src0_fake,
        ids_fake,
        src1_fake,
        w0_fake,
        w1_fake,
        out_fake,
        Int64(1),
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def draft_glue_rmsnorm_cat(
    embed_src: torch.Tensor,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
    *,
    gather: bool,
    col_lo: int,
    col_hi: int,
    contiguous_window: bool,
    enable_pdl: bool,
) -> torch.Tensor:
    """Columns ``[col_lo, col_hi)`` of ``cat([rmsnorm(e), rmsnorm(h)], -1)`` in one kernel.

    ``e`` is normalized with ``enorm_weight`` and ``h`` with ``hnorm_weight``. The result is
    bit-identical to FlashInfer ``rmsnorm`` on each half followed by ``torch.cat`` and the column
    slice. Only the halves overlapping the window are normalized.

    Args:
        embed_src: With ``gather``, the embedding table [vocab, H], and ``e`` is
            ``embed_src[input_ids]`` with zero rows for ids outside [0, vocab). Otherwise the
            embedding rows ``e`` [M, H].
            BF16, unit column stride. Unused when the window lies in the hidden half.
        input_ids: Token ids [M], int32 or int64, contiguous. Read only with ``gather``.
        hidden_states: Hidden rows ``h`` [M, H], BF16, unit column stride.
        enorm_weight: RMSNorm weight of the embedding half [H], BF16, contiguous.
        hnorm_weight: RMSNorm weight of the hidden half [H], BF16, contiguous.
        eps: RMSNorm epsilon of both halves.
        gather: Gather the embedding rows from the table in the kernel.
        col_lo: First column of the window, a multiple of 8.
        col_hi: One past the last column of the window, a multiple of 8.
        contiguous_window: Return the window as a contiguous [M, col_hi - col_lo] tensor. Otherwise
            return the window as a column slice of a [M, 2H] buffer, the strides ``torch.cat``
            followed by the slice would have.
        enable_pdl: Launch with programmatic dependent launch.

    Returns:
        The window [M, col_hi - col_lo], BF16.
    """
    num_rows, hidden_size = hidden_states.shape
    half_begin = col_lo // hidden_size
    num_halves = (col_hi - 1) // hidden_size - half_begin + 1
    gather = gather and half_begin == 0
    if half_begin != 0:
        # The embedding half is not stored; any [M, H] tensor fills the unused argument.
        embed_src = hidden_states

    if contiguous_window:
        out = hidden_states.new_empty((num_rows, col_hi - col_lo))
        out_col_shift = col_lo
    else:
        out = hidden_states.new_empty((num_rows, 2 * hidden_size))
        out_col_shift = 0

    if num_rows > 0:
        kernel = _compile_draft_glue_kernel(
            hidden_size,
            half_begin,
            num_halves,
            gather,
            input_ids.dtype,
            out.shape[1],
            out_col_shift,
            col_lo,
            col_hi,
            enable_pdl,
        )
        kernel(
            embed_src,
            input_ids,
            hidden_states,
            enorm_weight,
            hnorm_weight,
            out,
            num_rows,
            embed_src.shape[0],
            eps,
        )

    if contiguous_window or (col_lo == 0 and col_hi == 2 * hidden_size):
        return out
    return out[:, col_lo:col_hi]


__all__ = [
    "DraftGlueRMSNormKernel",
    "draft_glue_matches_flashinfer",
    "draft_glue_rmsnorm_cat",
]
