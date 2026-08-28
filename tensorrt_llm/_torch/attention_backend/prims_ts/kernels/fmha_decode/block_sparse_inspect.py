# Copyright (c) 2026 by FlashInfer team.
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

"""Validate canonical BSR and reduce its maximum semantic row width.

A BSR Q-block row is one ``block_indptr`` row keyed by
``(batch, kv_head, q_block)``. The maximum row width gives the one-shot API the
same semantic ``max_blocks_per_row`` bound that reusable plans receive from
their caller. Token-mask contents belong to the run-time prepare kernel and
are not read here.

Four warps validate four BSR Q-block rows per CTA and publish one validation
status plus the maximum row width in one Int64 summary. No per-route payload
is constructed.
"""

import functools
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda_drv

_WARPS_PER_CTA = 4
_WARP_SIZE = 32
_THREADS_PER_CTA = _WARPS_PER_CTA * _WARP_SIZE
_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"

# ``summary`` is a zero-initialized Int64[2] shared with the host wrapper.
_SUMMARY_ERROR_CODE = 0
_SUMMARY_MAX_BSR_BLOCK_COUNT = 1
_SUMMARY_FIELDS = 2

_BSR_ERROR_NONE = 0
_BSR_ERROR_NOT_STRICTLY_INCREASING = 1
_BSR_ERROR_INDEX_OUT_OF_RANGE = 2
_BSR_ERROR_INVALID_INDPTR = 3


@cute.jit
def _validate_bsr_row_lane(
    block_indices: cute.Tensor,
    bsr_row_begin: cutlass.Int32,
    bsr_row_end: cutlass.Int32,
    lane_idx: cutlass.Int32,
    num_kv_blocks: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Validate one lane stripe of a canonical ordered BSR row."""

    error_code = cutlass.Int32(_BSR_ERROR_NONE)
    selected_kv_block_count = bsr_row_end - bsr_row_begin
    bsr_entry_offset = lane_idx
    while bsr_entry_offset < selected_kv_block_count:
        entry_position = bsr_row_begin + bsr_entry_offset
        block_id = cutlass.Int32(block_indices[entry_position])
        in_range = block_id >= 0 and block_id < num_kv_blocks
        if not in_range:
            error_code = cutlass.Int32(_BSR_ERROR_INDEX_OUT_OF_RANGE)
        else:
            if entry_position > bsr_row_begin:
                previous_block_id = cutlass.Int32(block_indices[entry_position - 1])
                if (
                    block_id <= previous_block_id
                    and error_code < _BSR_ERROR_NOT_STRICTLY_INCREASING
                ):
                    error_code = cutlass.Int32(_BSR_ERROR_NOT_STRICTLY_INCREASING)
        bsr_entry_offset += _WARP_SIZE
    return error_code


class _InspectBlockSparseBsr:
    """Validate canonical BSR and reduce the facts needed by ``plan()``.

    The runtime inputs are compact tensors with the following ABI:

    * ``block_indptr``: Int32 ``[B, Hkv, num_q_block_rows + 1]`` offsets;
    * ``block_indices``: Int32 ``[nnz]`` semantic KV-block IDs;
    * ``summary``: zero-initialized Int64 ``[2]`` output.

    One 128-thread CTA contains four independent warps, and each warp handles
    one flattened ``(batch, kv_head, q_block_row)``. Lanes validate the row's
    index stripe, then lane zero contributes its planning bound:

    * ``[0]`` highest validation error code (zero means canonical BSR);
    * ``[1]`` maximum semantic BSR-block count of any row.

    No per-route metadata is produced here; the run-time prepare kernel
    consumes the caller's BSR and current token mask before attention.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        q_block_size: int,
        kv_block_size: int,
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
        self.num_kv_blocks = (seq_len_kv + kv_block_size - 1) // kv_block_size
        self.total_bsr_row_count = batch_size * num_kv_heads * self.num_q_block_rows

    @cute.jit
    def __call__(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        summary: cute.Tensor,
        stream: cuda_drv.CUstream,
    ) -> None:
        """Launch four BSR-row inspectors per CTA on ``stream``."""

        self.kernel(
            block_indptr,
            block_indices,
            summary,
        ).launch(
            grid=[
                (self.total_bsr_row_count + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA,
                1,
                1,
            ],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        summary: cute.Tensor,
    ) -> None:
        """Validate rows and atomically reduce their plan-time statistics."""

        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = thread_idx // _WARP_SIZE
        lane_idx = thread_idx % _WARP_SIZE

        # Q-block row is the fastest-moving dimension, then head and batch.
        linear_bsr_row_idx = block_idx * _WARPS_PER_CTA + warp_idx
        bsr_row_is_valid = linear_bsr_row_idx < self.total_bsr_row_count
        safe_linear_bsr_row_idx = (
            linear_bsr_row_idx if bsr_row_is_valid else cutlass.Int32(0)
        )
        q_block_row_idx = safe_linear_bsr_row_idx % self.num_q_block_rows
        linear_batch_head_idx = safe_linear_bsr_row_idx // self.num_q_block_rows
        kv_head_idx = linear_batch_head_idx % self.num_kv_heads
        batch_idx = linear_batch_head_idx // self.num_kv_heads

        bsr_row_begin = cutlass.Int32(0)
        bsr_row_end = cutlass.Int32(0)
        bsr_row_range_is_valid = cutlass.Boolean(False)
        num_indices = cutlass.Int32(cute.size(block_indices))
        error_code = cutlass.Int32(_BSR_ERROR_NONE)

        if bsr_row_is_valid:
            bsr_row_begin = cutlass.Int32(
                block_indptr[batch_idx, kv_head_idx, q_block_row_idx]
            )
            bsr_row_end = cutlass.Int32(
                block_indptr[batch_idx, kv_head_idx, q_block_row_idx + 1]
            )
            bsr_row_range_is_valid = cutlass.Boolean(
                bsr_row_begin >= 0
                and bsr_row_begin <= bsr_row_end
                and bsr_row_end <= num_indices
            )
            if not bsr_row_range_is_valid:
                error_code = cutlass.Int32(_BSR_ERROR_INVALID_INDPTR)
            else:
                error_code = _validate_bsr_row_lane(
                    block_indices,
                    bsr_row_begin,
                    bsr_row_end,
                    lane_idx,
                    self.num_kv_blocks,
                )

        # Numeric error codes encode reporting priority. Both this warp
        # reduction and summary[0]'s atomic max retain the strongest error.
        bsr_row_error_code = cutlass.Int32(cute.arch.warp_redux_sync(error_code, "max"))

        if lane_idx == 0 and bsr_row_is_valid:
            if bsr_row_error_code != _BSR_ERROR_NONE:
                cute.arch.atomic_max(
                    summary.iterator + _SUMMARY_ERROR_CODE,
                    cutlass.Int64(bsr_row_error_code),
                    sem="relaxed",
                    scope="gpu",
                )
            else:
                selected_kv_block_count = bsr_row_end - bsr_row_begin
                cute.arch.atomic_max(
                    summary.iterator + _SUMMARY_MAX_BSR_BLOCK_COUNT,
                    cutlass.Int64(selected_kv_block_count),
                    sem="relaxed",
                    scope="gpu",
                )


def _fake_compact(
    dtype: type,
    shape: tuple[object, ...],
    *,
    alignment: int,
) -> cute.Tensor:
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=alignment,
    )


@functools.cache
def compile_block_sparse_inspection(
    *,
    device_index: int,
    batch_size: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    q_block_size: int,
    kv_block_size: int,
) -> Callable[..., None]:
    """Compile one geometry specialization while keeping ``indices[nnz]`` dynamic.

    Tensor ranks, dtypes, compact strides, and all attention geometry are part
    of the specialization. Only the flat ``block_indices`` extent is symbolic;
    tensor contents may of course vary between calls to the cached function.
    """

    num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
    logical_nnz = cute.sym_int()
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = _InspectBlockSparseBsr(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )
    with torch.cuda.device(device_index):
        return cute.compile(
            kernel,
            _fake_compact(
                cutlass.Int32,
                (batch_size, num_kv_heads, num_q_block_rows + 1),
                alignment=4,
            ),
            _fake_compact(cutlass.Int32, (logical_nnz,), alignment=4),
            _fake_compact(cutlass.Int64, (_SUMMARY_FIELDS,), alignment=8),
            stream,
            options=_COMPILE_OPTIONS,
        )


__all__ = ["compile_block_sparse_inspection"]
