# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA: top-k rows in ascending key order, remapped to rows of the MLA KV pool (CuTe DSL, one launch)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.attention.dsa_topk_order_remap import (  # noqa: E501
    dsa_topk_order_remap_into,
)


def _spec(rows, **overrides) -> dict:
    spec = dict(
        rows=rows,
        requests=rows,
        topk=2048,
        tokens_per_block=64,
        kv_len=40_000,
        table_width=1_024,
        pool_pages=4_096,
        page_index_scale=24,
        layer=7,
        invalid=0,
        table_layout="rows",
        order="shuffled",
    )
    spec.update(overrides)
    return spec


class DsaTopkOrderRemap(OpWrapper):
    """``out[t]`` = row ``t`` of ``token_indices`` with its key positions (``>= 0``) in ascending
    order and its unused slots (``< 0``) last as ``-1``, each position ``tok`` then mapped to the
    KV-pool row of request ``req_id[t]``: ``block_table[req, tok // block_size] * stride_factor +
    layer_id * block_size + tok % block_size``; ``-1`` where ``tok // block_size`` is past the
    table row or the table entry is negative. Returns a fresh contiguous int32 ``[rows,
    num_topk_tokens]`` tensor. This is ``convert_req_index_to_global`` applied to the rows sorted
    ascending with their ``-1`` slots last.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row", spec=_spec(1)),
        Cell(
            why="a short request: the -1 top-k slots stay -1 and go last",
            spec=_spec(1, kv_len=1_500, invalid=548),
        ),
        Cell(
            why="ll B1: 8 rows of one request, keys of a 190k context",
            spec=_spec(8, requests=1, kv_len=190_000, table_width=4_096, pool_pages=40_000),
        ),
        Cell(
            why="ll B4 / mid: 32 rows of 4 requests at 190k",
            spec=_spec(32, requests=4, kv_len=190_000, table_width=4_096, pool_pages=40_000),
        ),
        Cell(
            why="M2: 72 rows of 12 requests at 150k",
            spec=_spec(72, requests=12, kv_len=150_000, table_width=4_096, pool_pages=40_000),
        ),
        Cell(
            why="DEP16: 224 rows of 32 requests at 130k",
            spec=_spec(224, requests=32, kv_len=130_000, table_width=4_096, pool_pages=40_000),
        ),
        Cell(
            why="DFlash2 w5 verify of 4 requests: 5 rows per request",
            spec=_spec(20, requests=4),
        ),
        Cell(
            why="the generation requests' rows of a mixed batch's table",
            spec=_spec(16, requests=16, table_layout="gen_rows"),
        ),
        Cell(why="32-token pages", spec=_spec(8, tokens_per_block=32, table_width=2048)),
        Cell(why="128-token pages", spec=_spec(8, tokens_per_block=128, table_width=512)),
        Cell(
            why="rows already ascending (the kernel's input order does not matter)",
            spec=_spec(8, order="ascending"),
        ),
        Cell(
            why="keys clustered in a few pages: long runs of set bits in one bitmap word run",
            spec=_spec(8, kv_len=190_000, table_width=4_096, pool_pages=40_000, order="clustered"),
        ),
        Cell(
            why="a long context: 700k keys on the last MLA layer's offset (three bitmap segments)",
            spec=_spec(4, kv_len=700_000, table_width=11_000, pool_pages=12_000, layer=23),
        ),
        Cell(
            why="table entries past the table row and negative entries map to -1 in place",
            spec=_spec(8, kv_len=190_000, table_width=1_000, pool_pages=4_000, holes=True),
        ),
        Cell(
            why="a prefill chunk: 3000 rows of one request",
            spec=_spec(3000, requests=1, kv_len=6000, invalid=0),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).
    The K3.1 perf targets call this once per DSA layer per phase in place of
    deterministic_topk_order + convert_req_index_to_global (transform_local_topk_and_prepare_pool_view):
    token_indices the phase's top-k rows in any order, req_id the metadata's per-token request
    index (the generation part rebased to the first generation request), block_table the matching
    rows of the MLA block table, block_size the page size (a power of two: 32, 64 or 128),
    num_topk_tokens the top-k width (2048), stride_factor / layer_id the MLA pool's
    page_index_scale * tokens_per_block and the layer's offset in the pool. Bitwise equal to the
    torch sort followed by convert_req_index_to_global for rows without duplicate positions (top-k
    rows have none): the order is ascending position, a function of the row's set only. One CTA
    of 512 threads per row; launched with programmatic dependent launch. Positions must be below
    2^31 - 1 and rows 16-byte aligned with a unit inner stride.
    """

    def __call__(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> torch.Tensor:
        out = torch.empty(
            token_indices.shape[0], num_topk_tokens, dtype=torch.int32, device=token_indices.device
        )
        return dsa_topk_order_remap_into(
            token_indices, out, req_id, block_table, block_size, stride_factor, layer_id
        )

    def reference(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> torch.Tensor:
        """torch sort (``-1`` slots last), then the pool rows in int64 arithmetic cast to int32."""
        unused = torch.iinfo(torch.int32).max
        keyed = torch.where(token_indices < 0, unused, token_indices)
        ordered = keyed.sort(dim=-1).values
        ordered = torch.where(ordered == unused, -1, ordered)
        tok = ordered[:, :num_topk_tokens].long()
        page = tok.clamp_min(0) // block_size
        valid_page = page < block_table.shape[1]
        req = req_id.long()[:, None].expand_as(tok)
        base = block_table[req, page.clamp(max=block_table.shape[1] - 1)].long()
        rows = base * stride_factor + tok.clamp_min(0) % block_size + layer_id * block_size
        valid = (tok >= 0) & valid_page & (base >= 0)
        return torch.where(valid, rows, torch.full_like(rows, -1)).int()

    def is_valid(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> None:
        assert token_indices.dtype == torch.int32 and token_indices.dim() == 2
        assert num_topk_tokens == token_indices.shape[1], "the kernel writes whole rows"
        k = num_topk_tokens
        assert k % 128 == 0 and 128 <= k <= 4096, f"top-k width {k} is not served"
        # 16-byte row loads.
        assert token_indices.stride(1) == 1 and token_indices.data_ptr() % 16 == 0
        assert token_indices.shape[0] <= 1 or token_indices.stride(0) % 4 == 0
        assert block_size > 0 and block_size & (block_size - 1) == 0, "power-of-two pages"
        assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
        assert req_id.dtype == torch.int32 and req_id.shape[0] == token_indices.shape[0], (
            "one request index per top-k row"
        )
        if req_id.numel() == 0:
            return
        assert int(req_id.max()) < block_table.shape[0] and int(req_id.min()) >= 0, (
            "a request index past the block table"
        )
        # The bitmap sort emits a repeated position once: rows must be duplicate-free.
        ordered = torch.where(token_indices < 0, -1, token_indices).sort(dim=-1).values
        dup = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)
        assert not bool(dup.any()), "a top-k row repeats a key position"
        assert int(token_indices.max()) < torch.iinfo(torch.int32).max
        largest = int(block_table.max()) * stride_factor + (layer_id + 1) * block_size
        assert largest < 2**31, "pool rows overflow int32"


dsa_topk_order_remap = DsaTopkOrderRemap()
