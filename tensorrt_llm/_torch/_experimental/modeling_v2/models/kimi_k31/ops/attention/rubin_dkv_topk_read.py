# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 cep under DKV: the latent rows this rank's sparse MLA attends, read from their owners.

Under DKV (page_split) most of a request's history lives on other ranks. The sparse MLA reads its
Top-K rows from local memory: the history rows the step's Top-K selects are copied, once each, into
a local buffer behind the step's new tokens, and the Top-K becomes rows of that buffer.

``rubin_dkv_topk_read`` takes the Top-K already remapped over the new tokens' pages and returns it
with the history entries replaced; ``rubin_dkv_topk_read_fused`` does the remap, the history
entries, the padding (to row 0) and the zeroing of the buffer's page 0 in one CuTe DSL launch.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention.rubin_dkv_topk_read import gather_rows_into, topk_rows_into
from .rubin_dsa_topk_remap import rubin_dsa_topk_remap


def _copy_history(
    kv: torch.Tensor,
    new_rows: int,
    topk: torch.Tensor,
    req_idx: torch.Tensor,
    hist_start: torch.Tensor,
    hist_len: torch.Tensor,
    hist_src: torch.Tensor,
    src: torch.Tensor,
    row_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy the history rows ``topk`` selects into ``kv`` from row ``new_rows``, once each in
    ascending order (``hist_src`` not empty, ``topk`` at least one row). Returns ``(hist, flat,
    rank)``: which entries are history, each entry's history token (``tokens`` for the others),
    and ``rank[f]``, the selected history tokens up to ``f``: token ``f`` lands in row
    ``new_rows + rank[f] - 1``."""
    tokens = hist_src.numel()
    start = hist_start.index_select(0, req_idx).unsqueeze(1)
    limit = hist_len.index_select(0, req_idx).unsqueeze(1)
    hist = (topk >= 0) & (topk < limit)
    flat = torch.where(hist, topk + start, tokens)
    picked = torch.zeros(tokens + 1, dtype=torch.uint8, device=topk.device)
    # A device value: a Python scalar would be copied from the host, which waits for it.
    picked.index_put_((flat.view(-1),), torch.ones((), dtype=torch.uint8, device=topk.device))
    # rank[f]: the selected history tokens up to f; slot tokens is the non-history sink.
    rank = torch.cumsum(picked, 0, dtype=torch.int32)
    pos = torch.nonzero_static(
        picked[:tokens], size=min(kv.shape[0] - new_rows, tokens), fill_value=-1
    ).view(-1)
    gather_rows_into(
        kv, new_rows, src.data_ptr(), hist_src, pos, rank[tokens - 1 : tokens], row_offset
    )
    return hist, flat, rank


def _cell(why: str, **spec) -> Cell:
    """``cached`` / ``new``: each context request's cached and new tokens; ``rows``: the requests
    whose rows this rank holds, with how many of their new rows (from their last); a request not
    listed has history length 0 in the plan."""
    base = dict(row_bytes=576, topk=2048, seed=0)
    base.update(spec)
    return Cell(why=why, spec=base)


class RubinDkvTopkRead(OpWrapper):
    """Copy the history rows ``topk`` selects into ``kv`` and return ``topk`` as rows of ``kv``.

    ``kv`` uint8 ``[new_rows + bound, row_bytes]``: its first ``new_rows`` rows hold the step's new
    tokens (the caller's append fills them; ``new_idx`` addresses them); the selected history rows
    land after them, once each, in ascending (request, position) order. ``bound`` must be at least
    ``min(history tokens, topk.numel())``.

    ``topk`` int32 ``[rows, K]``: each row's request-local positions (negative: none); ``req_idx``
    int32 ``[rows]``: its context request. A request's history is its first ``hist_len[r]``
    positions (0 for a request whose history this rank does not read); its token ``p`` is history
    token ``hist_start[r] + p`` (int32 ``[requests]`` each), whose latent row is row
    ``hist_src[hist_start[r] + p] + row_offset`` of ``src`` (uint8 ``[rows, row_bytes]``: every
    rank's latent pool through the peer mapping; ``hist_src`` int64, the layer-0 row of each history
    token; ``row_offset`` the layer's).

    ``new_idx`` int32 ``[rows, K]``: the Top-K remapped over the new tokens' pages (-1 where
    ``topk`` is negative); returns it with every history entry replaced by the ``kv`` row of the
    copied row.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _cell(
            "a 2048-row chunk of one request over 30k cached tokens",
            cached=[30000],
            new=[2048],
            rows=[(0, 2048)],
        ),
        _cell(
            "three requests with rows here, one a first chunk (no history), short rows -1 padded",
            cached=[700, 0, 5000],
            new=[300, 900, 100],
            rows=[(0, 300), (1, 900), (2, 100)],
        ),
        _cell(
            "a request without rows here (no history read) beside one with rows",
            cached=[4000, 9000],
            new=[200, 50],
            rows=[(1, 50)],
        ),
        _cell(
            "a history shorter than the Top-K: every history row selected",
            cached=[1200],
            new=[300],
            rows=[(0, 300)],
        ),
        _cell(
            "one row over a long history: the selection bound is the Top-K, not the history",
            cached=[100000],
            new=[1],
            rows=[(0, 1)],
        ),
        _cell(
            "no history at all: the indices are the new tokens' rows",
            cached=[0, 0],
            new=[64, 10],
            rows=[(0, 64), (1, 10)],
        ),
    )

    note = """
    The selection is deduplicated over the rank's rows (a flag per history token, a prefix sum, the
    flags' positions in order), so a row the Top-K of many query rows shares is copied once. The
    copy runs on the SMs (one CuTe DSL kernel, 16-byte loads from the peer mapping); the count of
    selected rows stays on the device, so nothing waits for the host. The rows a selection does not
    fill (``bound`` past the count) are left as they were.
    """

    def __call__(
        self,
        kv: torch.Tensor,
        new_rows: int,
        topk: torch.Tensor,
        req_idx: torch.Tensor,
        new_idx: torch.Tensor,
        hist_start: torch.Tensor,
        hist_len: torch.Tensor,
        hist_src: torch.Tensor,
        src: torch.Tensor,
        row_offset: int,
    ) -> torch.Tensor:
        rows, k = topk.shape
        tokens = hist_src.numel()
        if tokens == 0 or rows == 0:
            return new_idx
        hist, flat, rank = _copy_history(
            kv, new_rows, topk, req_idx, hist_start, hist_len, hist_src, src, row_offset
        )
        rows_of = rank.index_select(0, flat.view(-1)).view(rows, k) + (new_rows - 1)
        return torch.where(hist, rows_of, new_idx)

    def reference(
        self,
        kv: torch.Tensor,
        new_rows: int,
        topk: torch.Tensor,
        req_idx: torch.Tensor,
        new_idx: torch.Tensor,
        hist_start: torch.Tensor,
        hist_len: torch.Tensor,
        hist_src: torch.Tensor,
        src: torch.Tensor,
        row_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(kv after the copy, indices)`` as new tensors."""
        out = kv.clone()
        tokens = hist_src.numel()
        if tokens == 0 or topk.shape[0] == 0:
            return out, new_idx.clone()
        req = req_idx.long()
        top = topk.long()
        hist = (top >= 0) & (top < hist_len.long()[req].unsqueeze(1))
        flat = top + hist_start.long()[req].unsqueeze(1)
        chosen = torch.unique(flat[hist])  # ascending
        out[new_rows : new_rows + chosen.numel()] = src[hist_src[chosen] + row_offset]
        slot = torch.full((tokens,), -1, dtype=torch.long, device=topk.device)
        slot[chosen] = torch.arange(chosen.numel(), device=topk.device)
        indices = new_idx.long().clone()
        indices[hist] = new_rows + slot[flat[hist]]
        return out, indices.to(torch.int32)

    def is_valid(
        self,
        kv: torch.Tensor,
        new_rows: int,
        topk: torch.Tensor,
        req_idx: torch.Tensor,
        new_idx: torch.Tensor,
        hist_start: torch.Tensor,
        hist_len: torch.Tensor,
        hist_src: torch.Tensor,
        src: torch.Tensor,
        row_offset: int,
    ) -> None:
        # The kernel moves 16-byte vectors of contiguous rows and reads row addresses as int64.
        row_bytes = kv.shape[1]
        assert kv.dtype == torch.uint8 and kv.dim() == 2 and kv.is_contiguous()
        assert row_bytes % 16 == 0, "rows must be 16-byte multiples"
        assert src.dtype == torch.uint8 and src.dim() == 2 and src.shape[1] == row_bytes
        assert src.stride(1) == 1 and src.stride(0) == row_bytes, "src rows must be contiguous"
        assert topk.dtype == torch.int32 and topk.dim() == 2 and topk.is_contiguous()
        assert new_idx.dtype == torch.int32 and new_idx.shape == topk.shape
        assert req_idx.dtype == torch.int32 and req_idx.shape == (topk.shape[0],)
        assert hist_start.dtype == torch.int32 and hist_len.dtype == torch.int32
        assert hist_start.shape == hist_len.shape and hist_start.dim() == 1
        assert hist_src.dtype == torch.int64 and hist_src.dim() == 1 and hist_src.is_contiguous()
        assert 0 <= new_rows <= kv.shape[0]
        bound = kv.shape[0] - new_rows
        assert hist_src.numel() == 0 or bound >= min(hist_src.numel(), topk.numel()), (
            "kv must hold every row the selection can copy"
        )


rubin_dkv_topk_read = RubinDkvTopkRead()


def _fused_cell(why: str, **spec) -> Cell:
    """As ``_cell``, with the new-token table (``new_token_table`` of at least ``width`` columns,
    ``block_size``-token pages); ``holes``: some table entries -1 and the table cut short of the
    last pages; ``pad_rows``: that many rows of all -1 (a chunk without keys)."""
    base = dict(row_bytes=576, topk=2048, block_size=64, width=1, seed=0)
    base.update(spec)
    return Cell(why=why, spec=base)


class RubinDkvTopkReadFused(OpWrapper):
    """The DKV read with its indices made in one CuTe DSL launch: copy the history rows ``topk``
    selects into ``kv`` (as ``rubin_dkv_topk_read``), and return the sparse MLA's indices over
    ``kv`` with the padding pointed at row 0, which this zeroes.

    The indices are, bit for bit, the plain steps: the Top-K remapped over the new tokens' pages
    (``rubin_dsa_topk_remap``'s rule over ``block_table`` with ``block_size``, ``stride_factor``
    and ``layer_id``; -1 for a negative entry, a page past the table or a negative table entry),
    every history entry replaced by its copied row (``rubin_dkv_topk_read``), then ``clamp(min=0)``.
    The buffer's page 0 (its first ``block_size`` rows) is zeroed as ``kv[:block_size].zero_()``
    would; no other byte of ``kv`` changes beyond the copied history rows.

    Arguments as ``rubin_dkv_topk_read``'s, with the new-token table (int32 ``[requests, width]``)
    and its remap constants in place of ``new_idx``. Returns a new int32 ``[rows, K]``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _fused_cell(
            "the cep CTX chunk with history: 16384 rows over 100k cached tokens, no -1",
            cached=[100000],
            new=[16384],
            rows=[(0, 16384)],
        ),
        _fused_cell(
            "a fresh 16k prompt (no history): its first 2047 rows -1 padded",
            cached=[0],
            new=[16384],
            rows=[(0, 16384)],
        ),
        _fused_cell(
            "three requests, one a first chunk, one cached off the page grid (700)",
            cached=[700, 0, 5000],
            new=[300, 900, 100],
            rows=[(0, 300), (1, 900), (2, 100)],
        ),
        _fused_cell(
            "a request without rows here (no history read) beside one with rows",
            cached=[4000, 9000],
            new=[200, 50],
            rows=[(1, 50)],
        ),
        _fused_cell(
            "a history shorter than the Top-K: every history row selected",
            cached=[1200],
            new=[300],
            rows=[(0, 300)],
        ),
        _fused_cell(
            "rows whose Top-K is all -1 (a chunk without keys) beside full rows",
            cached=[3000],
            new=[512],
            rows=[(0, 512)],
            pad_rows=200,
        ),
        _fused_cell(
            "one row of a one-token prompt: the smallest step",
            cached=[0],
            new=[1],
            rows=[(0, 1)],
        ),
        _fused_cell(
            "one row over a long history",
            cached=[100000],
            new=[1],
            rows=[(0, 1)],
        ),
        _fused_cell(
            "24 rows (the plain path's C++ remap side)",
            cached=[2000],
            new=[24],
            rows=[(0, 24)],
        ),
        _fused_cell(
            "255 rows over a history (the last row count before the Triton remap)",
            cached=[5000],
            new=[255],
            rows=[(0, 255)],
        ),
        _fused_cell(
            "256 rows of a fresh prompt (the first Triton remap row count)",
            cached=[0],
            new=[256],
            rows=[(0, 256)],
        ),
        _fused_cell(
            "table holes and pages past the table (the remap's other -1 rules)",
            cached=[600, 1000],
            new=[700, 300],
            rows=[(0, 700), (1, 300)],
            holes=True,
        ),
        _fused_cell(
            "128-token pages and a wide table (the plan's fixed width)",
            cached=[3000, 0],
            new=[400, 200],
            rows=[(0, 400), (1, 200)],
            block_size=128,
            width=2048,
        ),
    )

    note = """
    One CuTe DSL kernel (DkvTopkRowsKernel) replaces the plain path's remap (Triton from 256
    rows, C++ below), its index_select / add / where tail, the clamp and page 0's zero_: CTAs of
    128 threads over 512, 1024 or 2048 entries of a Top-K row (by the row count and history) read
    the row once, each entry's rank or table entry once, and write the row once; the first CTAs
    also store page 0's zeros (after the PDL wait: the buffer's memory may still be in use by the
    kernel before). A history entry is exactly 0 <= t < hist_len[req] and
    its row is at least new_rows (> 0), so the clamp only changes the remap's -1s: the kernel
    writes 0 there. The history copy and its selection (dedup, prefix sum) are the plain op's.
    Without history tokens a variant that reads no hist_* or rank runs. Bit-exact against the
    reference and against the plain GPU path on every cell; nothing waits for the host.
    """

    def __call__(
        self,
        kv: torch.Tensor,
        new_rows: int,
        topk: torch.Tensor,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        stride_factor: int,
        layer_id: int,
        hist_start: torch.Tensor,
        hist_len: torch.Tensor,
        hist_src: torch.Tensor,
        src: torch.Tensor,
        row_offset: int,
    ) -> torch.Tensor:
        rows = topk.shape[0]
        if rows == 0:
            kv[:block_size].zero_()
            return torch.empty_like(topk)
        rank = None
        if hist_src.numel():
            _, _, rank = _copy_history(
                kv, new_rows, topk, req_idx, hist_start, hist_len, hist_src, src, row_offset
            )
        return topk_rows_into(
            torch.empty_like(topk),
            kv,
            block_size,
            topk,
            req_idx,
            block_table,
            block_size,
            stride_factor,
            layer_id,
            hist_start,
            hist_len,
            rank,
            new_rows - 1,
        )

    def reference(
        self,
        kv: torch.Tensor,
        new_rows: int,
        topk: torch.Tensor,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        stride_factor: int,
        layer_id: int,
        hist_start: torch.Tensor,
        hist_len: torch.Tensor,
        hist_src: torch.Tensor,
        src: torch.Tensor,
        row_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(kv after the read, indices)`` as new tensors, by the plain steps' references: page 0
        zeroed, the remap, the history read, the clamp."""
        out = kv.clone()
        out[:block_size] = 0
        new_idx = rubin_dsa_topk_remap.reference(
            req_idx, block_table, topk, block_size, topk.shape[1], stride_factor, layer_id
        )
        out, indices = rubin_dkv_topk_read.reference(
            out, new_rows, topk, req_idx, new_idx, hist_start, hist_len, hist_src, src, row_offset
        )
        return out, indices.clamp(min=0)

    def is_valid(
        self,
        kv: torch.Tensor,
        new_rows: int,
        topk: torch.Tensor,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        stride_factor: int,
        layer_id: int,
        hist_start: torch.Tensor,
        hist_len: torch.Tensor,
        hist_src: torch.Tensor,
        src: torch.Tensor,
        row_offset: int,
    ) -> None:
        assert kv.dtype == torch.uint8 and kv.dim() == 2 and kv.is_contiguous()
        assert kv.shape[1] % 16 == 0, "rows must be 16-byte multiples"
        assert block_size > 0 and block_size & (block_size - 1) == 0, "pages of 2^n tokens"
        assert block_size <= new_rows <= kv.shape[0], "page 0 lies inside the new tokens' pages"
        assert src.dtype == torch.uint8 and src.dim() == 2 and src.shape[1] == kv.shape[1]
        assert src.stride(1) == 1 and src.stride(0) == kv.shape[1], "src rows must be contiguous"
        # The kernel: CTAs of 128 threads over 512, 1024 or 2048 entries of a row, 16-byte loads
        # and stores.
        assert topk.dtype == torch.int32 and topk.dim() == 2 and topk.is_contiguous()
        assert topk.shape[1] % 512 == 0, "a Top-K width of a multiple of 512"
        assert topk.data_ptr() % 16 == 0, "16-byte aligned Top-K rows"
        assert req_idx.dtype == torch.int32 and req_idx.shape == (topk.shape[0],)
        assert req_idx.is_contiguous()
        assert block_table.dtype == torch.int32 and block_table.dim() == 2
        assert block_table.is_contiguous() and block_table.shape[1] >= 1
        assert hist_start.dtype == torch.int32 and hist_len.dtype == torch.int32
        assert hist_start.shape == hist_len.shape and hist_start.dim() == 1
        assert hist_start.is_contiguous() and hist_len.is_contiguous()
        assert hist_src.dtype == torch.int64 and hist_src.dim() == 1 and hist_src.is_contiguous()
        if req_idx.numel():
            # A request index past the table or the history reads another allocation's memory.
            top = int(req_idx.max())
            assert int(req_idx.min()) >= 0 and top < block_table.shape[0], (
                "a request index past the block table"
            )
            assert top < hist_len.shape[0], "a request index past the history"
        # The rows are int32 (the remap's and the kernel's arithmetic).
        top_page = int(block_table.max()) if block_table.numel() else 0
        largest = max(top_page, 0) * stride_factor + (layer_id + 1) * block_size
        assert largest < 2**31 and kv.shape[0] < 2**31, "rows overflow int32"
        bound = kv.shape[0] - new_rows
        assert hist_src.numel() == 0 or bound >= min(hist_src.numel(), topk.numel()), (
            "kv must hold every row the selection can copy"
        )


rubin_dkv_topk_read_fused = RubinDkvTopkReadFused()


class RubinDkvRowsRead(OpWrapper):
    """Copy rows ``src[src_rows[k] + row_offset]`` into ``dst[k]`` for ``k < count[0]``, from
    the peer mapping (``src``: every rank's latent pool as uint8 token rows); returns ``dst``.

    ``pos`` int64 ``[n]`` is ``arange(n)`` and ``count`` int32 ``[1]`` holds ``n`` (on the device,
    so the launch reads no host value). Under owner writes it moves the step's new rows of the
    requests a rank reads from their owners into its read buffer, 16-byte vectors per thread.
    """

    ARCHS = frozenset()
    note = """
    The same CuTe DSL row gather as rubin_dkv_topk_read (gather_rows_into) with
    an identity selection. A copy: bit-exact against the reference. Driven by
    the cep target's verify mode (trtllm_cep_new_rows: verify), which compares
    every copied row with the all-gather's append; no single-GPU cells.
    """

    def __call__(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        src_rows: torch.Tensor,
        pos: torch.Tensor,
        count: torch.Tensor,
        row_offset: int,
    ) -> torch.Tensor:
        gather_rows_into(dst, 0, src.data_ptr(), src_rows, pos, count, row_offset)
        return dst

    def reference(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        src_rows: torch.Tensor,
        pos: torch.Tensor,
        count: torch.Tensor,
        row_offset: int,
    ) -> torch.Tensor:
        """``dst`` with its first ``count[0]`` rows copied."""
        n = int(count[0])
        out = dst.clone()
        out[:n] = src[src_rows[pos[:n]] + row_offset]
        return out

    def is_valid(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        src_rows: torch.Tensor,
        pos: torch.Tensor,
        count: torch.Tensor,
        row_offset: int,
    ) -> None:
        assert dst.dtype == torch.uint8 and dst.dim() == 2 and dst.is_contiguous()
        assert dst.shape[1] % 16 == 0, "rows must be 16-byte multiples"
        assert src.dtype == torch.uint8 and src.shape[1] == dst.shape[1] and src.stride(1) == 1
        assert src_rows.dtype == torch.int64 and pos.dtype == torch.int64
        assert count.dtype == torch.int32 and count.numel() == 1
        assert int(count[0]) <= min(pos.numel(), dst.shape[0])


rubin_dkv_rows_read = RubinDkvRowsRead()
