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
"""cuDNN paged attention over a ``CausalKVCacheManager``.

cuDNN's paged SDPA reads each sequence's pages whole, in table order, and cuts only at the
sequence's length. A causal block's keys, the fixed region ``[0, fixed)`` and its window
``[start, start + length)``, start and end mid-page in general, so each block gets its own
table row:

    [ whole pages of the fixed region | whole pages of the window | private pages ]

The private pages hold copies of the block's visible tokens on its partial pages -- where the
fixed region ends, where the window starts and where the block ends, at most
``3 * (tokens_per_page - 1)`` tokens -- packed from the first slot, so the row's only partial
page is its last and the length cuts it. Every query of a block sees all of the block's keys,
so their order in the row does not matter.

One Triton kernel per layer and forward, after the staged K/V are written, copies the
partial-page tokens into the private pages and writes the rows and their lengths. The private
pages live in the cache's pool, in the ``extra_pages`` it never hands out; block ``i`` owns
pages ``num_pool_pages + i * pages_per_block`` onwards.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Dict, Tuple

import cudnn
import torch
import triton
import triton.language as tl

from tensorrt_llm.math_utils import ceil_div

if TYPE_CHECKING:
    from ...cache import CausalKVCacheManager


@triton.jit
def _copy_piece(
    pool_ptr,
    page_table_ptr,
    first,
    count,
    offset,
    private_page,
    plane_head,
    TOKENS_PER_PAGE: tl.constexpr,
    PLANE_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Copy logical positions ``[first, first + count)``, all on one page, of one plane and head
    to private slots ``offset`` onwards of the pages from ``private_page``."""
    t = tl.arange(0, TOKENS_PER_PAGE)
    d = tl.arange(0, HEAD_DIM)
    mask = (t < count)[:, None]
    src_page = tl.load(page_table_ptr + first // TOKENS_PER_PAGE).to(tl.int64)
    src_slot = first % TOKENS_PER_PAGE + t
    src = ((src_page * PLANE_HEADS + plane_head) * TOKENS_PER_PAGE + src_slot) * HEAD_DIM
    values = tl.load(pool_ptr + src[:, None] + d[None, :], mask=mask)
    dst_page = (private_page + (offset + t) // TOKENS_PER_PAGE).to(tl.int64)
    dst_slot = (offset + t) % TOKENS_PER_PAGE
    dst = ((dst_page * PLANE_HEADS + plane_head) * TOKENS_PER_PAGE + dst_slot) * HEAD_DIM
    tl.store(pool_ptr + dst[:, None] + d[None, :], values, mask=mask)


@triton.jit
def _causal_rows_kernel(
    pool_ptr,
    page_table_ptr,
    start_ptr,
    length_ptr,
    rows_ptr,
    seq_len_kv_ptr,
    fixed,
    row_len,
    private_base,
    TOKENS_PER_PAGE: tl.constexpr,
    PRIVATE_PAGES: tl.constexpr,
    PLANE_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROW_CHUNK: tl.constexpr,
):
    """Program ``(i, plane_head)`` fills causal block ``i``'s private pages for one plane and
    head; program ``(i, 0)`` also writes the block's row and length."""
    i = tl.program_id(0)
    plane_head = tl.program_id(1)
    start = tl.load(start_ptr + i)
    end = start + tl.load(length_ptr + i)
    # A window that reaches the fixed region makes the keys one span [0, end).
    merged = start <= fixed
    fixed_end = tl.where(merged, 0, fixed)
    lo = tl.where(merged, 0, start)
    # Pieces on partial pages: the fixed region's last page, the window's first and last pages.
    fixed_pages = fixed_end // TOKENS_PER_PAGE
    first0 = fixed_pages * TOKENS_PER_PAGE
    count0 = fixed_end - first0
    lo_ceil = (lo + TOKENS_PER_PAGE - 1) // TOKENS_PER_PAGE * TOKENS_PER_PAGE
    hi_floor = end // TOKENS_PER_PAGE * TOKENS_PER_PAGE
    count1 = tl.minimum(end, lo_ceil) - lo
    first2 = tl.maximum(hi_floor, lo_ceil)
    count2 = tl.maximum(end - first2, 0)
    window_first_page = lo_ceil // TOKENS_PER_PAGE
    window_pages = tl.maximum(hi_floor // TOKENS_PER_PAGE - window_first_page, 0)
    private_page = private_base + i * PRIVATE_PAGES

    _copy_piece(
        pool_ptr, page_table_ptr, first0, count0, 0, private_page, plane_head,
        TOKENS_PER_PAGE, PLANE_HEADS, HEAD_DIM,
    )  # fmt: skip
    _copy_piece(
        pool_ptr, page_table_ptr, lo, count1, count0, private_page, plane_head,
        TOKENS_PER_PAGE, PLANE_HEADS, HEAD_DIM,
    )  # fmt: skip
    _copy_piece(
        pool_ptr, page_table_ptr, first2, count2, count0 + count1, private_page, plane_head,
        TOKENS_PER_PAGE, PLANE_HEADS, HEAD_DIM,
    )  # fmt: skip

    if plane_head == 0:
        private_tokens = count0 + count1 + count2
        whole = fixed_pages + window_pages
        used = whole + (private_tokens + TOKENS_PER_PAGE - 1) // TOKENS_PER_PAGE
        tl.store(seq_len_kv_ptr + i, whole * TOKENS_PER_PAGE + private_tokens)
        for chunk in range(0, row_len, ROW_CHUNK):
            k = chunk + tl.arange(0, ROW_CHUNK)
            in_fixed = k < fixed_pages
            in_window = (k >= fixed_pages) & (k < whole)
            logical = tl.where(in_fixed, k, window_first_page + k - fixed_pages)
            pool_page = tl.load(page_table_ptr + logical, mask=in_fixed | in_window, other=0)
            entry = tl.where(k < whole, pool_page, private_page + k - whole)
            entry = tl.where(k < used, entry, 0)
            tl.store(rows_ptr + i * row_len + k, entry, mask=k < row_len)


class CuDNNCausalKVPages:
    """Per-block page-table rows and private pages that let cuDNN's paged SDPA read a
    ``CausalKVCacheManager``.

    The cache must be built with ``extra_pages`` of at least
    ``private_pages_for(tokens_per_page, max_causal_blocks)``; that budget bounds the causal
    blocks of one forward. One instance per cache, from ``for_cache``.
    """

    _instances: "weakref.WeakKeyDictionary[CausalKVCacheManager, CuDNNCausalKVPages]" = (
        weakref.WeakKeyDictionary()
    )

    @staticmethod
    def pages_per_block(tokens_per_page: int) -> int:
        """Private pages of one causal block: up to three partial pages' visible tokens."""
        return ceil_div(3 * (tokens_per_page - 1), tokens_per_page)

    @classmethod
    def private_pages_for(cls, tokens_per_page: int, max_causal_blocks: int) -> int:
        """``extra_pages`` a cache needs for forwards of up to ``max_causal_blocks`` blocks."""
        return max_causal_blocks * cls.pages_per_block(tokens_per_page)

    @classmethod
    def for_cache(cls, cache: "CausalKVCacheManager") -> "CuDNNCausalKVPages":
        pages = cls._instances.get(cache)
        if pages is None:
            pages = cls(cache)
            cls._instances[cache] = pages
        return pages

    def __init__(self, cache: "CausalKVCacheManager") -> None:
        tpb = cache.tokens_per_page
        self.cache = cache
        self.private_pages = self.pages_per_block(tpb)
        self.max_causal_blocks = cache.extra_pages // self.private_pages
        if self.max_causal_blocks == 0:
            raise ValueError(
                f"cuDNN attention over the K/V cache needs private pages: build the cache with "
                f"extra_pages=CuDNNCausalKVPages.private_pages_for({tpb}, max_causal_blocks)"
            )
        # The longest row: the fixed region, a window plus the largest block, the private pages.
        span = cache.window_tokens + cache.pin_tokens + cache.max_staged_tokens
        self.row_len = ceil_div(cache.pin_tokens, tpb) + ceil_div(span, tpb) + self.private_pages
        device = cache.page_table.device
        self._rows = torch.zeros(
            self.max_causal_blocks, self.row_len, dtype=torch.int32, device=device
        )
        self._seq_len_kv = torch.zeros(self.max_causal_blocks, dtype=torch.int32, device=device)
        self._seq_len_q: Dict[Tuple[int, int], torch.Tensor] = {}

    def prepare(
        self, layer_idx: int, num_blocks: int, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fill layer ``layer_idx``'s private pages for this forward's ``num_blocks`` causal
        blocks of ``block_size`` and return ``(rows, seq_len_q, seq_len_kv)``: ``[num_blocks,
        row_len]`` pool pages and two ``[num_blocks]`` lengths, int32. Call after the staged K/V
        are written."""
        cache = self.cache
        if num_blocks > self.max_causal_blocks:
            raise ValueError(
                f"{num_blocks} causal blocks exceed the {self.max_causal_blocks} the cache's "
                f"{cache.extra_pages} extra pages hold"
            )
        start, length = cache.kv_ranges(num_blocks, block_size)
        pool = cache.kv_buffer(layer_idx)
        rows, seq_len_kv = self._rows[:num_blocks], self._seq_len_kv[:num_blocks]
        _causal_rows_kernel[(num_blocks, 2 * cache.num_kv_heads)](
            pool,
            cache.page_table,
            start,
            length,
            rows,
            seq_len_kv,
            cache.fixed_tokens,
            self.row_len,
            cache.num_pool_pages,
            TOKENS_PER_PAGE=cache.tokens_per_page,
            PRIVATE_PAGES=self.private_pages,
            PLANE_HEADS=2 * cache.num_kv_heads,
            HEAD_DIM=cache.head_dim,
            ROW_CHUNK=128,
        )
        key = (num_blocks, block_size)
        seq_len_q = self._seq_len_q.get(key)
        if seq_len_q is None:
            seq_len_q = torch.full((num_blocks,), block_size, dtype=torch.int32, device=rows.device)
            self._seq_len_q[key] = seq_len_q
        return rows, seq_len_q, seq_len_kv


_graphs: Dict[Tuple, Tuple] = {}


def _paged_graph(key: Tuple) -> Tuple:
    """Build (or reuse) the paged SDPA graph of ``key``; returns ``(graph, workspace bytes,
    tensors by name)``."""
    entry = _graphs.get(key)
    if entry is not None:
        return entry
    (_, dtype, b, h_q, h_kv, s_q, d, num_pages, tpb, row_len, sm_scale) = key
    io = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    i32 = cudnn.data_type.INT32
    graph = cudnn.pygraph(
        io_data_type=io,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        name="visual_gen_sdpa_causal_kv",
    )
    # The batch axis is the causal block axis. Q and O are NHD rows viewed as [b, H, s_q, D];
    # K and V are the pool's planes, [pages, H_kv, tokens_per_page, D] with the pool's strides.
    q_dims, q_strides = [b, h_q, s_q, d], [s_q * h_q * d, d, h_q * d, 1]
    kv_dims, kv_strides = [num_pages, h_kv, tpb, d], [2 * h_kv * tpb * d, tpb * d, d, 1]
    tensors = {
        "q": graph.tensor(name="q", dim=q_dims, stride=q_strides, data_type=io),
        "k": graph.tensor(name="k", dim=kv_dims, stride=kv_strides, data_type=io),
        "v": graph.tensor(name="v", dim=kv_dims, stride=kv_strides, data_type=io),
        # One page id per (block, row entry), shared by every K/V head.
        "rows": graph.tensor(
            name="rows", dim=[b, 1, row_len, 1], stride=[row_len, row_len, 1, 1], data_type=i32
        ),
        "seq_len_q": graph.tensor(
            name="seq_len_q", dim=[b, 1, 1, 1], stride=[1] * 4, data_type=i32
        ),
        "seq_len_kv": graph.tensor(
            name="seq_len_kv", dim=[b, 1, 1, 1], stride=[1] * 4, data_type=i32
        ),
    }
    # A page table needs the padding mask and both lengths; seq_len_kv bounds each row's read.
    o, _ = graph.sdpa(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        is_inference=True,
        attn_scale=sm_scale,
        use_padding_mask=True,
        seq_len_q=tensors["seq_len_q"],
        seq_len_kv=tensors["seq_len_kv"],
        paged_attention_k_table=tensors["rows"],
        paged_attention_v_table=tensors["rows"],
        paged_attention_max_seq_len_kv=row_len * tpb,
    )
    o.set_output(True).set_dim(q_dims).set_stride(q_strides).set_data_type(io)
    tensors["o"] = o
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    entry = (graph, graph.get_workspace_size(), tensors)
    _graphs[key] = entry
    return entry


def cudnn_causal_kv_attention(
    handle,
    layer_idx: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cache: "CausalKVCacheManager",
    block_size: int,
    sm_scale: float,
) -> None:
    """Stage ``k``/``v`` (``[T, H_kv, D]``) in ``cache`` and write to ``out`` the attention of
    ``q`` (``[T, H, D]``, NHD rows) in causal blocks of ``block_size`` over the cache.
    ``handle`` is a cuDNN handle on ``q``'s device, bound to the current stream."""
    num_tokens, num_heads, head_dim = q.shape
    num_blocks = num_tokens // block_size
    cache.write_staged(layer_idx, k, v)
    pages = CuDNNCausalKVPages.for_cache(cache)
    rows, seq_len_q, seq_len_kv = pages.prepare(layer_idx, num_blocks, block_size)
    pool = cache.kv_buffer(layer_idx)
    key = (
        q.device.index,
        q.dtype,
        num_blocks,
        num_heads,
        cache.num_kv_heads,
        block_size,
        head_dim,
        pool.shape[0],
        cache.tokens_per_page,
        pages.row_len,
        sm_scale,
    )
    graph, workspace_size, tensors = _paged_graph(key)

    def blocks(x: torch.Tensor) -> torch.Tensor:
        return x.view(num_blocks, block_size, num_heads, head_dim).transpose(1, 2)

    pack = {
        tensors["q"]: blocks(q),
        tensors["k"]: pool[:, 0],
        tensors["v"]: pool[:, 1],
        tensors["rows"]: rows,
        tensors["seq_len_q"]: seq_len_q,
        tensors["seq_len_kv"]: seq_len_kv,
        tensors["o"]: blocks(out),
    }
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=q.device)
    graph.execute(pack, workspace, handle=handle)
