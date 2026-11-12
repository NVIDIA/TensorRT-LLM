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
"""FlashAttention-4 shared-prefix attention for generation (SM90).

Rows of a generation step that attend the same leading KV pages form a group. One layer of a step
runs four launches:

* ``_gather_q_kernel`` copies the queries into grouped order (the rows of a group contiguous);
* ``Fa4Prefix`` attends every group's shared pages once for all of its queries (non-causal);
* ``Fa4Tail`` attends each row's private pages (causal, bottom-right aligned);
* ``_merge_fa4_tail_kernel`` joins the two normalized states by their log-sum-exp and writes
  the output in batch order.

The groups are described by ``PrefixGroups``; how rows are grouped is up to the caller (the
n-sample library groups rows whose block tables start with the same pages). A caller with a
different tail passes any object with ``Fa4Tail``'s ``q_len``, ``run``, ``out`` and ``lse``.

Every table is a fixed-address device tensor refreshed before the step, so the sequence can be
captured in a CUDA graph.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl

from . import flash_attn4_kernels
from .flash_attn4_kernels import (
    FA4_NUM_THREADS,
    FA4_TILE_MN,
    allocate_split_workspace,
    fa4_forward,
    fa4_paged_generation_split,
    split_tables,
    write_kv_paged,
)

_NEG = tl.constexpr(-1.0e30)


@dataclass
class PrefixGroups:
    """The groups of a step as the prefix call reads them, in grouped (``perm``) order.

    Group ``g`` holds the query tokens ``[qo_indptr[g], qo_indptr[g + 1])`` and attends the pages
    ``prefix_indices[prefix_indptr[g]:prefix_indptr[g + 1]]``; its last page holds
    ``prefix_last_page_len[g]`` tokens, so a group may share a partly filled last page.
    """

    num_groups: int
    """Groups handed to the prefix call, placeholder and padding groups included."""
    qo_indptr: np.ndarray
    """[num_groups + 1] int32: query-token range of each group."""
    prefix_indptr: np.ndarray
    """[num_groups + 1] int32: page range of each group in ``prefix_indices``."""
    prefix_indices: np.ndarray
    """int32: the groups' shared pages."""
    prefix_last_page_len: np.ndarray
    """[num_groups] int32: tokens in each group's last prefix page."""


def prefix_request_tables(
    groups: PrefixGroups,
    tokens_per_block: int,
    chunk_tokens: int,
    num_requests: int,
    max_pages: int,
    splits: int = 1,
) -> tuple:
    """Request tables of the prefix call (``Fa4Prefix``).

    Every group with query rows becomes one request per ``chunk_tokens`` query tokens, each
    over the group's shared pages; the rest of the ``num_requests`` entries are empty. With
    ``splits`` > 1 the queries are replicated ``splits`` times (copy ``s`` at query offset
    ``s * tokens``) and copy ``s`` of a request attends only the ``s``-th of ``splits``
    near-equal runs of the group's pages (possibly empty).
    Returns ``(block_tables, seq_lens, cum_q, cum_kv)`` as int32 arrays.
    """
    qo = groups.qo_indptr
    pages_ptr = groups.prefix_indptr
    pages = groups.prefix_indices
    last = groups.prefix_last_page_len
    block_tables = np.zeros((num_requests, max_pages), dtype=np.int32)
    seq_lens = np.zeros((num_requests,), dtype=np.int32)
    cum_q = np.zeros((num_requests + 1,), dtype=np.int32)
    cum_kv = np.zeros((num_requests + 1,), dtype=np.int32)
    request = 0
    tokens = int(qo[groups.num_groups])
    for split in range(splits):
        for group in range(groups.num_groups):
            q0, q1 = int(qo[group]), int(qo[group + 1])
            if q1 == q0:
                continue
            first, stop = int(pages_ptr[group]), int(pages_ptr[group + 1])
            count = stop - first
            kv_len = (count - 1) * tokens_per_block + int(last[group]) if count else 0
            if splits > 1:
                # Runs of at least MIN_SPLIT_TOKENS: a short prompt uses fewer (non-empty) runs.
                min_split_tokens = flash_attn4_kernels.MIN_SPLIT_TOKENS
                per = max(-(-count // splits), -(-min_split_tokens // tokens_per_block))
                lo = min(split * per, count)
                hi = min(lo + per, count)
                first, stop = first + lo, first + hi
                # Only the group's last page may be partly filled.
                kv_len = max(min(kv_len - lo * tokens_per_block, (hi - lo) * tokens_per_block), 0)
                count = hi - lo
            for start in range(q0, q1, chunk_tokens):
                if request >= num_requests:
                    raise RuntimeError(
                        f"shared-prefix attention: the step needs more prefix requests than the "
                        f"{num_requests} its buffers hold."
                    )
                block_tables[request, :count] = pages[first:stop]
                seq_lens[request] = kv_len
                cum_q[request + 1] = split * tokens + min(start + chunk_tokens, q1)
                cum_kv[request + 1] = cum_kv[request] + kv_len
                request += 1
    cum_q[request + 1 :] = cum_q[request]
    cum_kv[request + 1 :] = cum_kv[request]
    return block_tables, seq_lens, cum_q, cum_kv


@triton.jit
def _gather_q_kernel(
    q_ptr,
    q_sb,
    q_sh,
    perm_ptr,
    out_ptr,
    H: tl.constexpr,
    HP: tl.constexpr,
    D: tl.constexpr,
    QL: tl.constexpr,
    HAS_PERM: tl.constexpr,
    SPLITS: tl.constexpr = 1,
):
    # Grouped token t = position * QL + j reads the batch row perm[position]; it is written
    # SPLITS times, copy s at token s * ntok + t (ntok = the grid size).
    t = tl.program_id(0)
    ntok = tl.num_programs(0)
    pos = t // QL
    j = t % QL
    if HAS_PERM:
        row = tl.load(perm_ptr + pos).to(tl.int64)
    else:
        row = pos.to(tl.int64)
    heads = tl.arange(0, HP)
    dims = tl.arange(0, D)
    mask = (heads < H)[:, None]
    src = q_ptr + (row * QL + j) * q_sb + heads[:, None] * q_sh + dims[None, :]
    dst = out_ptr + t.to(tl.int64) * (H * D) + heads[:, None] * D + dims[None, :]
    x = tl.load(src, mask=mask, other=0.0)
    for split in tl.static_range(SPLITS):
        tl.store(dst + split * ntok.to(tl.int64) * (H * D), x, mask=mask)


class Fa4Prefix:
    """Shared-prefix attention through FlashAttention-4 (CuTe DSL), non-causal.

    The queries of every group (grouped order, ``q_len`` tokens per row) attend the group's
    shared pages: the group is a request whose KV is its page list. FA4 sizes its grid by the
    request count times ``max_seqlen_q``, so a group is cut into chunks of at most
    ``MAX_CHUNK_QUERIES`` query tokens, each a request over the same pages. The tables are
    fixed-address device tensors, rewritten when the groups change.

    With ``splits`` > 1 the queries are replicated ``splits`` times and copy ``s`` attends the
    ``s``-th run of the group's pages; ``_merge_fa4_tail_kernel`` joins the copies.
    """

    MAX_CHUNK_QUERIES = 128
    TILE_MN = (128, 128)
    NUM_THREADS = 384

    def __init__(
        self,
        tokens: int,
        num_heads: int,
        head_dim: int,
        q_len: int,
        num_requests: int,
        max_pages: int,
        dtype: torch.dtype,
        device,
        tokens_per_block: int,
        splits: int = 1,
    ):
        self.num_heads = num_heads
        self.q_len = q_len
        self.tokens = tokens
        self.splits = splits
        self.tokens_per_block = tokens_per_block
        self.chunk_tokens = max(q_len, self.MAX_CHUNK_QUERIES // q_len * q_len)
        self.num_requests = num_requests
        self.q = torch.empty((tokens * splits, num_heads, head_dim), dtype=dtype, device=device)
        self.out = torch.empty_like(self.q)
        # FA4's LSE layout: [heads, splits * tokens], natural log.
        self.lse = torch.empty((num_heads, tokens * splits), dtype=torch.float32, device=device)
        self.block_tables = torch.zeros((num_requests, max_pages), dtype=torch.int32, device=device)
        self.seq_lens = torch.zeros((num_requests,), dtype=torch.int32, device=device)
        self.cum_q = torch.zeros((num_requests + 1,), dtype=torch.int32, device=device)
        self.cum_kv = torch.zeros((num_requests + 1,), dtype=torch.int32, device=device)

    @classmethod
    def num_requests_for(cls, num_groups: int, tokens: int, q_len: int) -> int:
        """Upper bound of the (group, chunk) requests of a batch of ``tokens`` query tokens."""
        chunk = max(q_len, cls.MAX_CHUNK_QUERIES // q_len * q_len)
        return num_groups + tokens // chunk + 1

    def table_arrays(self):
        """Device tables, in the order ``prefix_request_tables`` returns their host copies."""
        return self.block_tables, self.seq_lens, self.cum_q, self.cum_kv

    def run(self, pool: torch.Tensor, sm_scale: float) -> None:
        fa4_forward()(
            self.q,
            pool[:, 0].transpose(1, 2),
            pool[:, 1].transpose(1, 2),
            cu_seqlens_q=self.cum_q,
            max_seqlen_q=self.chunk_tokens,
            seqused_k=self.seq_lens,
            page_table=self.block_tables,
            softmax_scale=sm_scale,
            causal=False,
            tile_mn=self.TILE_MN,
            num_threads=self.NUM_THREADS,
            num_splits=1,
            pack_gqa=None,
            out=self.out,
            return_lse=True,
            lse=self.lse,
        )


@triton.jit
def _fa4_tail_table_kernel(
    offsets_ptr,
    offsets_row_stride,
    perm_ptr,
    shared_ptr,
    kv_lens_ptr,
    table_ptr,
    table_row_stride,
    tail_lens_ptr,
    width,
    page_div,
    TPB: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    # Program (i, c): grouped position i, page chunk c. Row b = perm[i] keeps its pages from
    # its group's shared run onward: table[b, p] = offsets[b, shared + p] // page_div.
    i = tl.program_id(0)
    c = tl.program_id(1)
    if HAS_PERM:
        b = tl.load(perm_ptr + i).to(tl.int64)
    else:
        b = i.to(tl.int64)
    shared = tl.load(shared_ptr + i).to(tl.int64)
    if c == 0:
        kv_len = tl.load(kv_lens_ptr + b)
        tl.store(tail_lens_ptr + b, (kv_len - shared * TPB).to(tl.int32))
    p = c * BLOCK + tl.arange(0, BLOCK)
    src = shared + p
    entry = tl.load(offsets_ptr + b * offsets_row_stride + src, mask=src < width, other=0)
    tl.store(table_ptr + b * table_row_stride + p, (entry // page_div).to(tl.int32), mask=p < width)


@triton.jit
def _merge_fa4_tail_kernel(
    pre_out_ptr,
    pre_lse_ptr,
    tail_out_ptr,
    tail_lse_ptr,
    perm_ptr,
    pempty_ptr,
    out_ptr,
    out_sb,
    ntok,
    H: tl.constexpr,
    HP: tl.constexpr,
    D: tl.constexpr,
    QL: tl.constexpr,
    HAS_PERM: tl.constexpr,
    HAS_EMPTY: tl.constexpr,
    PREFIX_SPLITS: tl.constexpr = 1,
):
    # One program per grouped token t: the prefix states (normalized outputs, grouped order,
    # copy s of token t at s * ntok + t, LSE [heads, PREFIX_SPLITS * ntok]) join the tail
    # state (normalized output, batch order, LSE [rows, heads, QL]); all LSEs are FA4's
    # natural log (an empty prefix split has LSE -inf).
    LOG2E: tl.constexpr = 1.4426950408889634
    t = tl.program_id(0)
    pos = t // QL
    j = t % QL
    if HAS_PERM:
        row = tl.load(perm_ptr + pos).to(tl.int64)
    else:
        row = pos.to(tl.int64)
    tok = row * QL + j
    heads = tl.arange(0, HP)
    h_ok = heads < H
    dims = tl.arange(0, D)
    stride_l: tl.constexpr = PREFIX_SPLITS
    empty = False
    if HAS_EMPTY:
        empty = tl.load(pempty_ptr + pos) != 0
    lse_t = tl.load(tail_lse_ptr + (row * H + heads) * QL + j, mask=h_ok, other=_NEG) * LOG2E
    peak = lse_t
    for sp in tl.static_range(PREFIX_SPLITS):
        lse_p = tl.load(
            pre_lse_ptr + heads * (stride_l * ntok) + sp * ntok + t, mask=h_ok, other=_NEG
        )
        lse_p = tl.where(empty, _NEG, tl.maximum(lse_p * LOG2E, _NEG))
        peak = tl.maximum(peak, lse_p)
    w_t = tl.exp2(lse_t - peak)
    l_tot = w_t
    res = w_t[:, None] * tl.load(
        tail_out_ptr + (tok * H + heads[:, None]) * D + dims[None, :],
        mask=h_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    for sp in tl.static_range(PREFIX_SPLITS):
        lse_p = tl.load(
            pre_lse_ptr + heads * (stride_l * ntok) + sp * ntok + t, mask=h_ok, other=_NEG
        )
        lse_p = tl.where(empty, _NEG, tl.maximum(lse_p * LOG2E, _NEG))
        w_p = tl.exp2(lse_p - peak)
        pre = tl.load(
            pre_out_ptr + ((sp * ntok + t).to(tl.int64) * H + heads[:, None]) * D + dims[None, :],
            mask=h_ok[:, None] & (w_p > 0)[:, None],
            other=0.0,
        ).to(tl.float32)
        l_tot += w_p
        res += w_p[:, None] * pre
    res = res / l_tot[:, None]
    tl.store(
        out_ptr + tok * out_sb + heads[:, None] * D + dims[None, :],
        res.to(out_ptr.dtype.element_ty),
        mask=h_ok[:, None],
    )


class Fa4Tail:
    """The rows' private pages through FlashAttention-4, paired with ``Fa4Prefix``.

    Every row's pages from its group's shared run onward form the row's own page table
    (``table``, batch order, rebuilt once per step on the device from the attention op's
    block offsets), and its KV length past the shared run (``tail_lens``). The step's K/V is
    written into those pages, then FA4's paged decode (causal, bottom-right aligned) attends
    them and returns the normalized output and LSE that ``_merge_fa4_tail_kernel`` joins with
    the prefix state. The tables are fixed-address, so the layer can be captured in a graph.
    """

    def __init__(
        self,
        rows: int,
        q_len: int,
        num_heads: int,
        head_dim: int,
        max_pages: int,
        dtype,
        device,
        splits: int = 1,
    ):
        # ``splits`` > 1: the private pages are attended split-KV (``fa4_paged_generation_split``).
        self.rows = rows
        self.q_len = q_len
        self.splits = splits
        self.split_tables = None
        self.split_workspace = (
            allocate_split_workspace(rows, splits, q_len, num_heads, head_dim, dtype, device)
            if splits > 1
            else None
        )
        self.table = torch.zeros((rows, max_pages), dtype=torch.int32, device=device)
        self.tail_lens = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.out = torch.empty((rows * q_len, num_heads, head_dim), dtype=dtype, device=device)
        self.lse = torch.empty((rows, num_heads, q_len), dtype=torch.float32, device=device)

    def build_tables(
        self,
        offsets: torch.Tensor,
        perm: Optional[torch.Tensor],
        shared_pages: torch.Tensor,
        kv_lens: torch.Tensor,
        page_div: int,
        tokens_per_block: int,
    ) -> None:
        """Once per step: the rows' tail page tables and lengths.

        Args:
            offsets: [rows, max_blocks] int32 device block offsets of the rows (batch order).
            perm: [rows] batch row of each grouped position (None: identity).
            shared_pages: [rows] int32, grouped order: shared pages of the row's group.
            kv_lens: [rows] int32 device KV lengths including the step's tokens (batch order).
            page_div: offset entries per pool-view page (the KV factor).
        """
        assert offsets.stride(-1) == 1 and offsets.shape[0] >= self.rows
        width = min(int(offsets.shape[-1]), int(self.table.shape[1]))
        block = 256
        _fa4_tail_table_kernel[(self.rows, triton.cdiv(width, block))](
            offsets, offsets.stride(0), perm if perm is not None else shared_pages,
            shared_pages, kv_lens, self.table, self.table.stride(0), self.tail_lens,
            width, page_div, TPB=tokens_per_block, BLOCK=block, HAS_PERM=perm is not None,
        )  # fmt: skip
        if self.splits > 1:
            self.split_tables = split_tables(
                self.table, self.tail_lens, self.q_len, self.splits, tokens_per_block
            )

    def run(
        self,
        pool: torch.Tensor,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        sm_scale: float,
    ) -> None:
        """Attend the private pages; ``k`` None: the step's K/V is already in them."""
        tokens, num_heads, head_dim = q.shape
        if k is not None:
            write_kv_paged(pool, k, v, self.table, self.tail_lens, self.q_len)
        if self.splits > 1:
            fa4_paged_generation_split(
                pool,
                q,
                self.out,
                self.split_tables,
                self.q_len,
                self.splits,
                sm_scale,
                self.split_workspace,
                lse=self.lse,
            )
            return
        fa4_forward()(
            q.view(self.rows, self.q_len, num_heads, head_dim),
            pool[:, 0].transpose(1, 2),
            pool[:, 1].transpose(1, 2),
            seqused_k=self.tail_lens,
            page_table=self.table,
            softmax_scale=sm_scale,
            causal=True,
            tile_mn=FA4_TILE_MN,
            num_threads=FA4_NUM_THREADS,
            num_splits=1,
            pack_gqa=None,
            out=self.out.view(self.rows, self.q_len, num_heads, head_dim),
            return_lse=True,
            lse=self.lse,
        )


FA4_PREFIX_MAX_SPLITS = 8


def fa4_prefix_splits(
    rows: int, q_len: int, num_heads: int, num_kv_heads: int, num_sms: int
) -> int:
    """KV splits of the FA4 prefix call of a batch size.

    The prefix call launches about one CTA per (request, KV head, 128-row tile of the
    request's packed GQA queries) and walks the group's whole prompt in it; a batch of a few
    rows (a single group) then runs a handful of CTAs. When the rows' query tiles fill less
    than half the SMs, the group's pages are cut into up to one wave's worth of runs (at most
    ``FA4_PREFIX_MAX_SPLITS``), each attended by a copy of the queries.
    """
    gqa = max(num_heads // max(num_kv_heads, 1), 1)
    tiles = max(1, -(-(rows * q_len * gqa) // Fa4Prefix.TILE_MN[0]))
    ctas = tiles * (num_kv_heads if gqa > 1 else num_heads)
    if 2 * ctas > num_sms:
        return 1
    return max(1, min(FA4_PREFIX_MAX_SPLITS, -(-num_sms // ctas)))


def fa4_supported(head_dim: int, dtype: torch.dtype, device=None) -> bool:
    """Whether FlashAttention-4 can serve shared-prefix attention (SM90, bf16, d64/d128)."""
    if dtype != torch.bfloat16 or head_dim not in (64, 128):
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != (9, 0):
        return False
    return fa4_forward() is not None


def fa4_shared_prefix_attention(
    pool: torch.Tensor,
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    perm: Optional[torch.Tensor],
    prefix_empty: Optional[torch.Tensor],
    prefix: Fa4Prefix,
    tail: Fa4Tail,
    sm_scale: float,
) -> None:
    """One layer: attend the shared and private pages, write ``out`` in batch order.

    Every row carries ``q_len`` query tokens at positions ``kv_len - q_len + j`` (``kv_len``
    the device KV length including them): they all see the group's shared pages, the row's
    private pages, and the row's own tokens up to themselves.

    Args:
        pool: [pages, 2, kv_heads, tokens_per_block, head_dim] layer pool (HND).
        q: [rows * q_len, heads, head_dim], batch order (the tokens of a row consecutive).
        k, v: [rows * q_len, kv_heads, head_dim], batch order, or None when the step's K/V is
            already in the private pages (``rope_write_kv_paged``).
        out: [rows * q_len, heads, head_dim], batch order.
        perm: [rows] batch row of each grouped position (None: identity).
        prefix_empty: [rows] bool, grouped order: rows that take the tail alone.
        prefix, tail: the step's tables (``Fa4Prefix`` tables of the groups, ``Fa4Tail``
            tables built for this step).
    """
    ntok, num_heads, head_dim = q.shape
    q_len = tail.q_len
    assert ntok % q_len == 0 and out.stride(2) == 1 and pool.stride(4) == 1
    assert prefix.splits == 1 or prefix.q.shape[0] == prefix.splits * ntok
    _gather_q_kernel[(ntok,)](
        q, q.stride(0), q.stride(1), perm if perm is not None else q, prefix.q,
        H=num_heads, HP=triton.next_power_of_2(num_heads), D=head_dim, QL=q_len,
        HAS_PERM=perm is not None, SPLITS=prefix.splits, num_warps=4,
    )  # fmt: skip
    prefix.run(pool, sm_scale)
    tail.run(pool, q, k, v, sm_scale)
    _merge_fa4_tail_kernel[(ntok,)](
        prefix.out, prefix.lse, tail.out, tail.lse,
        perm if perm is not None else q,
        prefix_empty if prefix_empty is not None else q, out, out.stride(0), ntok,
        H=num_heads, HP=triton.next_power_of_2(num_heads), D=head_dim, QL=q_len,
        HAS_PERM=perm is not None, HAS_EMPTY=prefix_empty is not None,
        PREFIX_SPLITS=prefix.splits, num_warps=4,
    )  # fmt: skip
