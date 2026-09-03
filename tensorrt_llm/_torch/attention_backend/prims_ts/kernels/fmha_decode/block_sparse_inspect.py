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

"""Validate paged request metadata and canonical BSR for one-shot planning.

A BSR Q-block row is one ``block_indptr`` row keyed by
``(batch, kv_head, q_block)``. The maximum row width gives the one-shot API the
same semantic ``max_blocks_per_row`` bound that reusable plans receive from
their caller. Token-mask contents belong to the run-time prepare kernel and
are not read here.

Paged inspection first validates per-run sequence lengths and page rows, then
four warps validate four BSR Q-block rows per CTA. Both publish one validation
status plus the maximum row width in one Int64 summary; no route payload is
constructed.
"""

import functools
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda_drv

from .fmha_decode_resources.helpers_common import _warp_broadcast_i32

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
_ERROR_INVALID_SEQ_LEN = 4
_ERROR_INVALID_PAGE_INDPTR = 5
_ERROR_INSUFFICIENT_PAGE_CAPACITY = 6
_ERROR_INVALID_PHYSICAL_PAGE_ID = 7


@cute.jit
def _validate_bsr_row_lane(
    block_indices: cute.Tensor,
    bsr_row_begin: cutlass.Int32,
    bsr_row_end: cutlass.Int32,
    lane_idx: cutlass.Int32,
    num_kv_blocks: cutlass.Int32,
) -> cutlass.Int32:
    """Validate one lane stripe of a canonical ordered BSR row."""

    error_code = cutlass.Int32(_BSR_ERROR_NONE)
    selected_kv_block_count = cutlass.Int64(bsr_row_end) - cutlass.Int64(bsr_row_begin)
    bsr_entry_offset = cutlass.Int64(lane_idx)
    while bsr_entry_offset < selected_kv_block_count:
        entry_position = cutlass.Int64(bsr_row_begin) + bsr_entry_offset
        block_id = cutlass.Int32(block_indices[entry_position])
        in_range = block_id >= 0 and block_id < num_kv_blocks
        if not in_range:
            error_code = cutlass.Int32(_BSR_ERROR_INDEX_OUT_OF_RANGE)
        else:
            if entry_position > cutlass.Int64(bsr_row_begin):
                previous_block_id = cutlass.Int32(
                    block_indices[entry_position - cutlass.Int64(1)]
                )
                if (
                    block_id <= previous_block_id
                    and error_code < _BSR_ERROR_NOT_STRICTLY_INCREASING
                ):
                    error_code = cutlass.Int32(_BSR_ERROR_NOT_STRICTLY_INCREASING)
        bsr_entry_offset += cutlass.Int64(_WARP_SIZE)
    return error_code


@cute.jit
def _inspect_bsr_row(
    block_indptr: cute.Tensor,
    block_indices: cute.Tensor,
    batch_idx: cutlass.Int32,
    kv_head_idx: cutlass.Int32,
    q_block_row_idx: cutlass.Int32,
    lane_idx: cutlass.Int32,
    bsr_row_is_valid: cutlass.Boolean,
    num_kv_blocks: cutlass.Int32,
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Validate one BSR row against a static or per-run Int32 upper bound."""

    bsr_row_begin = cutlass.Int32(0)
    bsr_row_end = cutlass.Int32(0)
    row_range_is_valid = cutlass.Int32(0)
    if lane_idx == cutlass.Int32(0) and bsr_row_is_valid:
        bsr_row_begin = cutlass.Int32(
            block_indptr[batch_idx, kv_head_idx, q_block_row_idx]
        )
        bsr_row_end = cutlass.Int32(
            block_indptr[batch_idx, kv_head_idx, q_block_row_idx + 1]
        )
        row_range_is_valid = cutlass.Int32(
            bsr_row_begin >= cutlass.Int32(0)
            and bsr_row_begin <= bsr_row_end
            and bsr_row_end <= cutlass.Int32(cute.size(block_indices))
        )
    bsr_row_begin = _warp_broadcast_i32(bsr_row_begin, 0)
    bsr_row_end = _warp_broadcast_i32(bsr_row_end, 0)
    row_range_is_valid = _warp_broadcast_i32(row_range_is_valid, 0)

    error_code = cutlass.Int32(_BSR_ERROR_NONE)
    if bsr_row_is_valid:
        if row_range_is_valid == cutlass.Int32(0):
            error_code = cutlass.Int32(_BSR_ERROR_INVALID_INDPTR)
        else:
            error_code = _validate_bsr_row_lane(
                block_indices,
                bsr_row_begin,
                bsr_row_end,
                lane_idx,
                num_kv_blocks,
            )
    error_code = cutlass.Int32(cute.arch.warp_redux_sync(error_code, "max"))
    return error_code, bsr_row_end - bsr_row_begin


@cute.jit
def _inspect_runtime_paged_bsr_row(
    block_indptr: cute.Tensor,
    block_indices: cute.Tensor,
    seq_lens_kv: cute.Tensor,
    batch_idx: cutlass.Int32,
    kv_head_idx: cutlass.Int32,
    q_block_row_idx: cutlass.Int32,
    lane_idx: cutlass.Int32,
    bsr_row_is_valid: cutlass.Boolean,
    kv_block_size: cutlass.Constexpr[int],
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Inspect one BSR row against the current request sequence length."""

    error_code = cutlass.Int32(_BSR_ERROR_NONE)
    selected_kv_block_count = cutlass.Int32(0)
    num_active_kv_blocks = cutlass.Int32(0)
    if lane_idx == cutlass.Int32(0) and bsr_row_is_valid:
        runtime_seq_len_kv = cutlass.Int32(seq_lens_kv[batch_idx])
        num_active_kv_blocks = (runtime_seq_len_kv - cutlass.Int32(1)) // cutlass.Int32(
            kv_block_size
        ) + cutlass.Int32(1)
    num_active_kv_blocks = _warp_broadcast_i32(num_active_kv_blocks, 0)
    error_code, selected_kv_block_count = _inspect_bsr_row(
        block_indptr,
        block_indices,
        batch_idx,
        kv_head_idx,
        q_block_row_idx,
        lane_idx,
        bsr_row_is_valid,
        num_active_kv_blocks,
    )
    return error_code, selected_kv_block_count


class _InspectBlockSparseBsr:
    """Validate canonical BSR against static or per-run request metadata."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        seq_len_q: int,
        seq_len_kv: int | None,
        q_block_size: int,
        kv_block_size: int,
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
        self.num_kv_blocks = (
            0
            if seq_len_kv is None
            else (seq_len_kv + kv_block_size - 1) // kv_block_size
        )
        self.kv_block_size = kv_block_size
        self.total_bsr_row_count = batch_size * num_kv_heads * self.num_q_block_rows

    @cute.jit
    def _launch(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        seq_lens_kv: cute.Tensor | None,
        summary: cute.Tensor,
        stream: cuda_drv.CUstream,
    ) -> None:
        self.kernel(
            block_indptr,
            block_indices,
            seq_lens_kv,
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
        seq_lens_kv: cute.Tensor | None,
        summary: cute.Tensor,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = thread_idx // _WARP_SIZE
        lane_idx = thread_idx % _WARP_SIZE

        linear_bsr_row_idx = block_idx * _WARPS_PER_CTA + warp_idx
        bsr_row_is_valid = linear_bsr_row_idx < self.total_bsr_row_count
        safe_linear_bsr_row_idx = (
            linear_bsr_row_idx if bsr_row_is_valid else cutlass.Int32(0)
        )
        q_block_row_idx = safe_linear_bsr_row_idx % self.num_q_block_rows
        linear_batch_head_idx = safe_linear_bsr_row_idx // self.num_q_block_rows
        kv_head_idx = linear_batch_head_idx % self.num_kv_heads
        batch_idx = linear_batch_head_idx // self.num_kv_heads

        if cutlass.const_expr(seq_lens_kv is None):
            error_code, selected_kv_block_count = _inspect_bsr_row(
                block_indptr,
                block_indices,
                batch_idx,
                kv_head_idx,
                q_block_row_idx,
                lane_idx,
                bsr_row_is_valid,
                cutlass.Int32(self.num_kv_blocks),
            )
        else:
            error_code, selected_kv_block_count = _inspect_runtime_paged_bsr_row(
                block_indptr,
                block_indices,
                seq_lens_kv,
                batch_idx,
                kv_head_idx,
                q_block_row_idx,
                lane_idx,
                bsr_row_is_valid,
                self.kv_block_size,
            )

        if lane_idx == cutlass.Int32(0) and bsr_row_is_valid:
            if error_code != cutlass.Int32(_BSR_ERROR_NONE):
                cute.arch.atomic_max(
                    summary.iterator + _SUMMARY_ERROR_CODE,
                    cutlass.Int64(error_code),
                    sem="relaxed",
                    scope="gpu",
                )
            else:
                cute.arch.atomic_max(
                    summary.iterator + _SUMMARY_MAX_BSR_BLOCK_COUNT,
                    cutlass.Int64(selected_kv_block_count),
                    sem="relaxed",
                    scope="gpu",
                )

    @cute.jit
    def __call__(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        seq_lens_kv: cute.Tensor | None,
        summary: cute.Tensor,
        stream: cuda_drv.CUstream,
    ) -> None:
        self._launch(block_indptr, block_indices, seq_lens_kv, summary, stream)


class _InspectPagedKvMetadata:
    """Validate per-run lengths and page rows with one warp per request."""

    def __init__(
        self,
        *,
        batch_size: int,
        minimum_seq_len_kv: int,
        max_seq_len_kv: int,
        page_size: int,
    ) -> None:
        self.batch_size = batch_size
        self.minimum_seq_len_kv = minimum_seq_len_kv
        self.max_seq_len_kv = max_seq_len_kv
        self.page_size = page_size

    @cute.jit
    def __call__(
        self,
        paged_kv_indptr: cute.Tensor,
        paged_kv_indices: cute.Tensor,
        seq_lens_kv: cute.Tensor,
        num_physical_kv_pages: cutlass.Int64,
        summary: cute.Tensor,
        stream: cuda_drv.CUstream,
    ) -> None:
        self.kernel(
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens_kv,
            num_physical_kv_pages,
            summary,
        ).launch(
            grid=[
                (self.batch_size + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA,
                1,
                1,
            ],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        paged_kv_indptr: cute.Tensor,
        paged_kv_indices: cute.Tensor,
        seq_lens_kv: cute.Tensor,
        num_physical_kv_pages: cutlass.Int64,
        summary: cute.Tensor,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = thread_idx // _WARP_SIZE
        lane_idx = thread_idx % _WARP_SIZE
        batch_idx = block_idx * _WARPS_PER_CTA + warp_idx
        request_is_valid = batch_idx < self.batch_size

        request_begin = cutlass.Int32(0)
        request_end = cutlass.Int32(0)
        request_range_is_valid = cutlass.Int32(0)
        error_code = cutlass.Int32(_BSR_ERROR_NONE)
        if lane_idx == 0 and request_is_valid:
            seq_len_kv = cutlass.Int32(seq_lens_kv[batch_idx])
            seq_len_is_valid = cutlass.Boolean(
                seq_len_kv >= cutlass.Int32(self.minimum_seq_len_kv)
                and seq_len_kv <= cutlass.Int32(self.max_seq_len_kv)
            )
            if not seq_len_is_valid:
                error_code = cutlass.Int32(_ERROR_INVALID_SEQ_LEN)

            request_begin = cutlass.Int32(paged_kv_indptr[batch_idx])
            request_end = cutlass.Int32(paged_kv_indptr[batch_idx + 1])
            num_page_indices = cutlass.Int32(cute.size(paged_kv_indices))
            request_range_is_valid = cutlass.Int32(
                paged_kv_indptr[cutlass.Int32(0)] == cutlass.Int32(0)
                and request_begin >= cutlass.Int32(0)
                and request_begin <= request_end
                and request_end <= num_page_indices
            )
            if request_range_is_valid == cutlass.Int32(0):
                error_code = cutlass.Int32(_ERROR_INVALID_PAGE_INDPTR)
            elif seq_len_is_valid:
                required_pages = (seq_len_kv - cutlass.Int32(1)) // cutlass.Int32(
                    self.page_size
                ) + cutlass.Int32(1)
                if request_end - request_begin < required_pages:
                    error_code = cutlass.Int32(_ERROR_INSUFFICIENT_PAGE_CAPACITY)

        request_begin = _warp_broadcast_i32(request_begin, 0)
        request_end = _warp_broadcast_i32(request_end, 0)
        request_range_is_valid = _warp_broadcast_i32(request_range_is_valid, 0)
        if request_is_valid and request_range_is_valid != cutlass.Int32(0):
            page_offset = cutlass.Int64(lane_idx)
            request_page_count = cutlass.Int64(request_end) - cutlass.Int64(
                request_begin
            )
            while page_offset < request_page_count:
                page_position = cutlass.Int64(request_begin) + page_offset
                physical_page_id = cutlass.Int32(paged_kv_indices[page_position])
                if (
                    physical_page_id < cutlass.Int32(0)
                    or cutlass.Int64(physical_page_id) >= num_physical_kv_pages
                ):
                    error_code = cutlass.Int32(_ERROR_INVALID_PHYSICAL_PAGE_ID)
                page_offset += cutlass.Int64(_WARP_SIZE)

        error_code = cutlass.Int32(cute.arch.warp_redux_sync(error_code, "max"))
        if (
            lane_idx == cutlass.Int32(0)
            and request_is_valid
            and error_code != cutlass.Int32(_BSR_ERROR_NONE)
        ):
            cute.arch.atomic_max(
                summary.iterator + _SUMMARY_ERROR_CODE,
                cutlass.Int64(error_code),
                sem="relaxed",
                scope="gpu",
            )


class _InspectPagedBlockSparseMetadata:
    """Launch request and BSR inspection into one summary."""

    def __init__(
        self,
        *,
        inspect_requests: Callable[..., None],
        inspect_bsr: Callable[..., None],
    ) -> None:
        self.inspect_requests = inspect_requests
        self.inspect_bsr = inspect_bsr

    @cute.jit
    def __call__(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        paged_kv_indptr: cute.Tensor,
        paged_kv_indices: cute.Tensor,
        seq_lens_kv: cute.Tensor,
        num_physical_kv_pages: cutlass.Int64,
        summary: cute.Tensor,
        stream: cuda_drv.CUstream,
    ) -> None:
        self.inspect_requests(
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens_kv,
            num_physical_kv_pages,
            summary,
            stream,
        )
        self.inspect_bsr(
            block_indptr,
            block_indices,
            seq_lens_kv,
            summary,
            stream,
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
            None,
            _fake_compact(cutlass.Int64, (_SUMMARY_FIELDS,), alignment=8),
            stream,
            options=_COMPILE_OPTIONS,
        )


@functools.cache
def compile_paged_block_sparse_metadata_inspection(
    *,
    device_index: int,
    batch_size: int,
    num_kv_heads: int,
    seq_len_q: int,
    minimum_seq_len_kv: int,
    max_seq_len_kv: int,
    q_block_size: int,
    kv_block_size: int,
    page_size: int,
) -> Callable[..., None]:
    """Compile one paged metadata entry that launches request then BSR checks."""

    num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
    logical_page_capacity = cute.sym_int()
    logical_nnz = cute.sym_int()
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    inspect_requests = _InspectPagedKvMetadata(
        batch_size=batch_size,
        minimum_seq_len_kv=minimum_seq_len_kv,
        max_seq_len_kv=max_seq_len_kv,
        page_size=page_size,
    )
    inspect_bsr = _InspectBlockSparseBsr(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=None,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
    )
    inspect_paged_block_sparse_metadata = _InspectPagedBlockSparseMetadata(
        inspect_requests=inspect_requests,
        inspect_bsr=inspect_bsr,
    )

    with torch.cuda.device(device_index):
        return cute.compile(
            inspect_paged_block_sparse_metadata,
            _fake_compact(
                cutlass.Int32,
                (batch_size, num_kv_heads, num_q_block_rows + 1),
                alignment=4,
            ),
            _fake_compact(cutlass.Int32, (logical_nnz,), alignment=4),
            _fake_compact(cutlass.Int32, (batch_size + 1,), alignment=4),
            _fake_compact(cutlass.Int32, (logical_page_capacity,), alignment=4),
            _fake_compact(cutlass.Int32, (batch_size,), alignment=4),
            cutlass.Int64(1),
            _fake_compact(cutlass.Int64, (_SUMMARY_FIELDS,), alignment=8),
            stream,
            options=_COMPILE_OPTIONS,
        )


__all__ = [
    "compile_block_sparse_inspection",
    "compile_paged_block_sparse_metadata_inspection",
]
