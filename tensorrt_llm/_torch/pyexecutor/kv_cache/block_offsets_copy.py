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
"""Batch page-table copy for KVCacheManagerV2 limited to the columns a batch can use
(Kimi K3.1 generation servers without attention DP, ``KVCacheManagerV2._batch_page_table_columns``).

Same values as ``copy_batch_block_offsets_to_device`` (kvCacheManagerV2Utils.cu) on the copied columns: for every
pool ``p`` and batch row ``s``, ``dst[p, s, 0, c] = scale[p] * src[p, copy_idx[s], 0, c]`` and
``dst[p, s, 1, c] = dst[p, s, 0, c] + kv_offset[p]``, both 0 where the source holds BAD_PAGE_INDEX. The host page
table, the copy index and the per-pool constants are pinned host memory read in place by the kernel.
"""

import torch
import triton
import triton.language as tl

_BAD_PAGE_INDEX = -1
_BLOCK = 1024


class _MappedHost:
    """Pass a pinned host tensor to a Triton kernel by address (the driver maps it to its device pointer)."""

    def __init__(self, tensor: torch.Tensor):
        assert tensor.is_pinned()
        self._tensor = tensor
        self.dtype = tensor.dtype

    def data_ptr(self) -> int:
        return self._tensor.data_ptr()


@triton.jit
def _copy_block_offsets_kernel(
    src_ptr,
    dst_ptr,
    copy_idx_ptr,
    scales_ptr,
    kv_offset_ptr,
    src_rows,
    dst_rows,
    width,
    num_cols,
    BAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pool = tl.program_id(0)
    row = tl.program_id(1)
    c = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    m = c < num_cols
    src_row = (pool * src_rows + tl.load(copy_idx_ptr + row)).to(tl.int64) * 2 * width
    page = tl.load(src_ptr + src_row + c, mask=m, other=BAD)
    scale = tl.load(scales_ptr + pool)
    kv_offset = tl.load(kv_offset_ptr + pool)
    bad = page == BAD
    key = tl.where(bad, 0, scale * page)
    dst_row = (pool * dst_rows + row).to(tl.int64) * 2 * width
    tl.store(dst_ptr + dst_row + c, key, mask=m)
    tl.store(dst_ptr + dst_row + width + c, tl.where(bad, 0, key + kv_offset), mask=m)


def copy_batch_block_offsets_columns(
    host_block_offsets: torch.Tensor,
    dst: torch.Tensor,
    copy_idx: torch.Tensor,
    index_scales: torch.Tensor,
    kv_offset: torch.Tensor,
    num_cols: int,
    stream: torch.cuda.Stream,
) -> bool:
    """Copy page-table columns ``[0, num_cols)`` of the batch rows ``copy_idx`` into ``dst`` on ``stream``.

    ``host_block_offsets`` is the pinned ``[pools, slots, 2, width]`` host table, ``dst`` the device
    ``[>= pools, >= rows, 2, width]`` table. Columns at or past ``num_cols`` of ``dst`` are left unchanged.
    Returns False (nothing launched) when the tensors do not have that layout, so the caller runs the
    full-width copy instead.
    """
    if host_block_offsets.dim() != 4 or dst.dim() != 4:
        return False
    num_pools, src_rows, kv_factor, width = host_block_offsets.shape
    num_seqs = copy_idx.shape[0]
    host_tensors = (host_block_offsets, index_scales, kv_offset) + (
        () if copy_idx.is_cuda else (copy_idx,)
    )
    if not (
        kv_factor == 2
        and dst.shape[0] >= num_pools
        and dst.shape[1] >= num_seqs
        and dst.shape[2] == 2
        and dst.shape[3] == width
        and dst.is_cuda
        and dst.is_contiguous()
        and 0 < num_cols <= width
        and dst.dtype == torch.int32
        and copy_idx.dtype == torch.int32
        and copy_idx.is_contiguous()
        and all(
            t.dtype == torch.int32 and t.is_pinned() and t.is_contiguous() for t in host_tensors
        )
        and index_scales.numel() >= num_pools
        and kv_offset.numel() >= num_pools
    ):
        return False
    if num_seqs == 0:
        return True
    with torch.cuda.stream(stream):
        _copy_block_offsets_kernel[(num_pools, num_seqs, triton.cdiv(num_cols, _BLOCK))](
            _MappedHost(host_block_offsets),
            dst,
            _MappedHost(copy_idx) if not copy_idx.is_cuda else copy_idx,
            _MappedHost(index_scales),
            _MappedHost(kv_offset),
            src_rows,
            dst.shape[1],
            width,
            num_cols,
            BAD=_BAD_PAGE_INDEX,
            BLOCK=_BLOCK,
            num_warps=4,
        )
    return True
