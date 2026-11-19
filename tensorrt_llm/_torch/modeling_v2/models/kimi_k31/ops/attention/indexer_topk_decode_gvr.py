# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA decode top-k: the deterministic self-sampling GVR top-k (CuTe DSL)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention import (
    gvr_topk_decode_host as _gvr_host,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.dsa_topk_order_remap import (  # noqa: E501
    dsa_topk_order_remap_into,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.gvr_topk_decode_host import (  # noqa: E501
    run_varlen as gvr_topk_run_varlen,
)


def _spec(batch, next_n, context_lens, **overrides) -> dict:
    """``context_lens``: (low, high) of the per-request key counts, drawn per cell."""
    spec = dict(
        batch=batch,
        next_n=next_n,
        context_lens=context_lens,
        topk=2048,
        max_seq_len=1 << 20,
        row_pad=0,
        data="real",
    )
    spec.update(overrides)
    return spec


class IndexerTopkDecodeGvr(OpWrapper):
    """Row ``r`` of ``logits`` (fp32, ``num_rows = batch * next_n``) has ``n_r = kv_lens[r //
    next_n] - next_n + r % next_n + 1`` valid columns. ``indices[r]`` receives the ``K =
    indices.shape[1]`` columns of largest score among them -- ties at the K-th score go to the
    smaller column -- or, when ``n_r <= K``, columns ``0 .. n_r - 1`` and ``-1`` after them. The
    order within a row is fixed for given inputs and launch shape but is not ascending; the
    caller orders the rows (``dsa_topk_order_remap``). Writes ``indices`` in place.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="ll B1: 8 rows at 190k", spec=_spec(1, 8, (189_000, 191_000))),
        Cell(why="ll B2: 16 rows at 190k", spec=_spec(2, 8, (185_000, 191_000))),
        Cell(why="ll B4: 32 rows at 190k", spec=_spec(4, 8, (180_000, 191_000))),
        Cell(why="mid B1: 6 rows at 150k", spec=_spec(1, 6, (149_000, 151_000))),
        Cell(why="mid B4: 24 rows at 126k-170k", spec=_spec(4, 6, (126_000, 170_000))),
        Cell(why="M2 B12: 72 rows at 126k-170k", spec=_spec(12, 6, (126_000, 170_000))),
        Cell(why="TEP16 B6 w8: 48 rows at 110k-130k", spec=_spec(6, 8, (110_000, 130_000))),
        Cell(why="B7 w5: 35 rows at 60k-170k", spec=_spec(7, 5, (60_000, 170_000))),
        Cell(why="DEP16 B14: 98 rows at 130k", spec=_spec(14, 7, (120_000, 140_000))),
        Cell(why="DEP16 B32: 224 rows at 130k", spec=_spec(32, 7, (120_000, 140_000))),
        Cell(
            why="short requests mixed in: rows with n_r <= K take the identity + -1 pad",
            spec=_spec(4, 8, (1_000, 40_000)),
        ),
        Cell(
            why="threshold ties: integer-valued scores, the smaller columns win the ties",
            spec=_spec(2, 8, (60_000, 80_000), data="ties"),
        ),
        Cell(
            why="a logits arena row stride wider than the envelope's columns",
            spec=_spec(1, 8, (100_000, 110_000), max_seq_len=200_000, row_pad=256),
        ),
        Cell(why="next_n 1: one row per request", spec=_spec(8, 1, (50_000, 190_000))),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).
    Every K3.1 target on SM 10.3 calls this in the DSA indexer's decode branch in place of
    indexer_topk_decode (the CUDA radix; the perf targets take the pool-row variant below when the
    backend arms the remap): logits the CuTe DSL paged-MQA logits view
    [rows, indexer_max_seq_len] (row stride rounded to whole splits), kv_lens the generation
    requests' kv_lens_cuda_runtime (int32, uncompressed key counts), indices the top-k buffer
    rows, next_n the uniform verify width, max_seq_len the indexer's max_seq_len (a capture-stable
    envelope). Uniform next_n only: ragged verify rows (row_kv_lens) keep the radix. The
    selected set equals the radix's on rows without a tie at the K-th score; with ties the
    radix's pick is atomic-order dependent and this one takes the smallest columns. The kernels
    are the family's deterministic GVR (kernels/attention/gvr_topk_decode*):
    the same selection and output order as the engine's deterministic GVR, with the bitmap emit
    compiled in for the 1M decode envelope. Compiles once per (rows, row stride, K, envelope,
    next_n) outside CUDA-graph capture (warmup); the host module keeps a per-device workspace.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        max_seq_len: int,
    ) -> None:
        gvr_topk_run_varlen(
            logits,
            kv_lens,
            indices,
            next_n=next_n,
            compress_ratio=1,
            max_seq_len=max_seq_len,
        )

    def reference(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        max_seq_len: int,
    ) -> None:
        """Per row: a stable descending sort of the valid prefix, first K columns (row order is
        the sort's, i.e. descending score)."""
        k = indices.shape[1]
        rows = logits.shape[0]
        lens = kv_lens.long().repeat_interleave(next_n)
        offs = torch.arange(rows, device=logits.device) % next_n
        n_rows = (lens - next_n + offs + 1).clamp_min(0).tolist()
        indices.fill_(-1)
        for r, n in enumerate(n_rows):
            if n <= k:
                indices[r, :n] = torch.arange(n, dtype=torch.int32, device=logits.device)
                continue
            order = torch.sort(logits[r, :n], descending=True, stable=True).indices
            indices[r] = order[:k].int()

    def is_valid(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        max_seq_len: int,
    ) -> None:
        assert logits.dtype == torch.float32 and logits.stride(1) == 1
        assert indices.dtype == torch.int32 and indices.is_contiguous()
        assert kv_lens.dtype == torch.int32 and kv_lens.dim() == 1
        rows = logits.shape[0]
        assert rows % next_n == 0 and kv_lens.shape[0] == rows // next_n, (
            "uniform next_n rows: one kv length per next_n rows"
        )
        assert indices.shape[0] == rows
        assert int(kv_lens.max()) <= min(max_seq_len, logits.shape[1]), (
            "a row's valid columns past the logits"
        )
        # NaN scores order implementation-specifically (the set can differ from the radix's).
        lens = kv_lens.long().repeat_interleave(next_n)
        offs = torch.arange(rows, device=logits.device) % next_n
        n_rows = (lens - next_n + offs + 1).clamp_min(0)
        cols = torch.arange(logits.shape[1], device=logits.device)[None, :]
        valid = cols < n_rows[:, None]
        assert not bool(torch.isnan(logits).logical_and(valid).any()), "NaN in a valid column"


indexer_topk_decode_gvr = IndexerTopkDecodeGvr()


def _split_launch(logits: torch.Tensor, top_k: int, max_seq_len: int) -> bool:
    """Whether the deterministic GVR launch for these logits is a SPLIT launch (rows end ascending)."""
    rows = logits.shape[0]
    npad = logits.stride(0) if rows > 1 else logits.shape[1]
    n_env = min(max(int(max_seq_len), 1), npad)
    sms, sm_version = _gvr_host._unpack_device_profile(
        _gvr_host._device_profile_key(logits.get_device())
    )
    return _gvr_host.varlen_launch_is_split(rows, npad, top_k, n_env, sms, sm_version)


class IndexerTopkDecodeGvrPaged(OpWrapper):
    """The decode top-k rows as KV-pool rows, ready for the sparse MLA: row ``r``'s
    ``indexer_topk_decode_gvr`` selection in ascending column order, ``-1`` slots last, each column
    ``tok`` mapped to ``block_table[r // next_n, tok // block_size] * stride_factor + layer_id *
    block_size + tok % block_size`` (``-1`` past the table row or for a negative entry) -- i.e.
    ``dsa_topk_order_remap`` of the GVR rows with request ``r // next_n``. Writes ``indices`` in
    place.

    A SPLIT GVR launch (several CTAs per row; every decode launch of up to 49 rows) already ends
    each row in ascending order, so it writes the pool rows from its own epilogue; other launches
    write the columns and ``dsa_topk_order_remap`` orders and maps them in place.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=c.why + " -> pool rows", spec=dict(c.spec, pool=True))
        for c in IndexerTopkDecodeGvr.CELLS
    ) + (
        Cell(
            why="table entries past the table row and negative entries map to -1 in place",
            spec=_spec(2, 8, (180_000, 191_000), pool=True, holes=True),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).
    The K3.1 perf targets call this in the DSA indexer's decode branch for uniform next_n rows
    (not ragged verify, compress ratio 1, fp32 logits) in place of the top-k + the generation
    rows' deterministic_topk_order + convert_req_index_to_global: block_table the metadata's
    _cached_block_table_gen (row-major, contiguous rows), block_size its tokens per block,
    stride_factor / layer_id the MLA pool's page_index_scale * tokens_per_block and the layer's
    offset (get_primary_pool_page_index_params). The rows it writes are bitwise the current
    path's (GVR set = radix set on tie-free rows; smaller columns on ties). The SPLIT decision
    is a host function of the launch shape (cached), so a CUDA graph captures one path.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        max_seq_len: int,
        block_table: torch.Tensor,
        block_size: int,
        stride_factor: int,
        layer_id: int,
    ) -> None:
        remap = (block_table, block_size, stride_factor, layer_id)
        if _split_launch(logits, indices.shape[1], max_seq_len) and gvr_topk_run_varlen(
            logits,
            kv_lens,
            indices,
            next_n=next_n,
            compress_ratio=1,
            max_seq_len=max_seq_len,
            remap=remap,
        ):
            return
        gvr_topk_run_varlen(
            logits,
            kv_lens,
            indices,
            next_n=next_n,
            compress_ratio=1,
            max_seq_len=max_seq_len,
        )
        dsa_topk_order_remap_into(
            indices,
            indices,
            None,
            block_table,
            block_size,
            stride_factor,
            layer_id,
            rows_per_req=next_n,
        )

    def reference(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        max_seq_len: int,
        block_table: torch.Tensor,
        block_size: int,
        stride_factor: int,
        layer_id: int,
    ) -> None:
        """The GVR reference rows, sorted ascending (-1 last), mapped in int64 and cast to int32."""
        IndexerTopkDecodeGvr.reference(self, logits, kv_lens, indices, next_n, max_seq_len)
        unused = torch.iinfo(torch.int32).max
        tok = torch.where(indices < 0, unused, indices).sort(dim=-1).values
        tok = torch.where(tok == unused, -1, tok).long()
        page = tok.clamp_min(0) // block_size
        req = (torch.arange(tok.shape[0], device=tok.device) // next_n)[:, None].expand_as(tok)
        valid_page = page < block_table.shape[1]
        base = block_table[req, page.clamp(max=block_table.shape[1] - 1)].long()
        rows = base * stride_factor + tok.clamp_min(0) % block_size + layer_id * block_size
        valid = (tok >= 0) & valid_page & (base >= 0)
        indices.copy_(torch.where(valid, rows, torch.full_like(rows, -1)).int())

    def is_valid(
        self,
        logits: torch.Tensor,
        kv_lens: torch.Tensor,
        indices: torch.Tensor,
        next_n: int,
        max_seq_len: int,
        block_table: torch.Tensor,
        block_size: int,
        stride_factor: int,
        layer_id: int,
    ) -> None:
        IndexerTopkDecodeGvr.is_valid(self, logits, kv_lens, indices, next_n, max_seq_len)
        assert block_table.dtype == torch.int32 and block_table.dim() == 2
        assert block_table.stride(1) == 1 and block_table.stride(0) == block_table.shape[1], (
            "a row-major block table with contiguous rows"
        )
        assert block_table.shape[0] >= logits.shape[0] // next_n, "one table row per request"
        assert block_size > 0 and block_size & (block_size - 1) == 0, "power-of-two pages"
        assert indices.shape[1] % 128 == 0 and indices.data_ptr() % 16 == 0
        largest = int(block_table.max()) * stride_factor + (layer_id + 1) * block_size
        assert largest < 2**31, "pool rows overflow int32"


indexer_topk_decode_gvr_paged = IndexerTopkDecodeGvrPaged()


def enable_gvr_early_release(mapping) -> None:
    """Attention-TP 8 generation layouts on SM 10.3: the decode GVR launches that write the pool rows from their own
    epilogue (the SPLIT launches of ``indexer_topk_decode_gvr_paged``) release their dependent right after their
    grid-dependency wait, so the sparse MLA's launch and prologue overlap the GVR merge CTAs. The TRTLLM-Gen sparse
    MLA reads only the sequence lengths before its own wait and the pool rows after it, so no result changes.
    Process-wide; call before warmup and CUDA graph capture."""
    if mapping.enable_attention_dp or mapping.tp_size != 8:
        return
    if torch.cuda.get_device_capability() != (10, 3):
        return
    _gvr_host.enable_early_release()
