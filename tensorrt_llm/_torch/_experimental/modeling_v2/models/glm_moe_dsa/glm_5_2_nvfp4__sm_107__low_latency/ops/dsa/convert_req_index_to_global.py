# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-local TopK token indices to rows of the MLA KV pool, via the trtllm convert op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _spec(rows, **overrides) -> dict:
    spec = dict(
        rows=rows,
        topk=2048,
        tokens_per_block=64,
        kv_len=530_000,
        table_width=8_288,
        pool_pages=9_000,
        page_index_scale=79,
        layer=78,
        invalid=0,
        table_layout="rows",
    )
    spec.update(overrides)
    return spec


class GlmConvertReqIndexToGlobal(OpWrapper):
    """``out[t, c]`` = the KV-pool row of request-local token ``tok = token_indices[t, c]`` of
    request ``req_id[t]``: ``block_table[req, tok // block_size] * stride_factor + layer_id *
    block_size + tok % block_size``; -1 where ``tok < 0``, ``tok // block_size`` is past the table
    row, or the table entry is negative. Columns ``[0, num_topk_tokens)``; returns a fresh tensor
    shaped like ``token_indices``.

    For the MLA pool of layer ``l`` the target passes ``stride_factor = page_index_scale *
    tokens_per_block`` and ``layer_id = layer_offset`` from the KV cache manager's
    ``get_primary_pool_page_index_params(l)``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1 draft step 1: the reused TopK row of one request, MTP layer 78",
            spec=_spec(1),
        ),
        Cell(
            why="-1 TopK slots of a short request stay -1",
            spec=_spec(1, kv_len=1_500, invalid=548),
        ),
        Cell(
            why="four requests, table rows of the generation requests only",
            spec=_spec(4, kv_len=70_000, table_width=1_200, table_layout="gen_rows"),
        ),
    )

    note = """
    Metadata consumed: none. At c1 draft step 1 the target passes the request map of the draft
    step (``req_idx_per_token`` of the generation tokens minus the context count), the generation
    requests' rows of the MLA block table, the stashed TopK rows ([num_requests, 2048] int32),
    the page size, 2048, and the MTP layer's stride factor and layer offset. A plain launch.

    The op reads ``token_indices`` and the table through ``.contiguous()`` copies when they are
    strided.
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
        return torch.ops.trtllm.convert_req_index_to_global(
            req_id, block_table, token_indices, block_size, num_topk_tokens, stride_factor, layer_id
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
        """The pool rows in int64 arithmetic, cast back to int32."""
        tok = token_indices[:, :num_topk_tokens].long()
        page = tok.clamp_min(0) // block_size
        valid_page = page < block_table.shape[1]
        req = req_id.long()[:, None].expand_as(tok)
        base = block_table[req, page.clamp(max=block_table.shape[1] - 1)].long()
        rows = base * stride_factor + tok.clamp_min(0) % block_size + layer_id * block_size
        valid = (tok >= 0) & valid_page & (base >= 0)
        out = token_indices.clone()
        out[:, :num_topk_tokens] = torch.where(valid, rows, torch.full_like(rows, -1)).int()
        return out

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
        # Columns past num_topk_tokens are left unwritten in the fresh output.
        assert num_topk_tokens == token_indices.shape[1], (
            f"num_topk_tokens={num_topk_tokens} leaves columns of the "
            f"{token_indices.shape[1]}-wide output unwritten"
        )
        assert req_id.shape[0] == token_indices.shape[0], "one request index per TopK row"
        # A request index past the table reads another allocation's memory.
        assert int(req_id.max()) < block_table.shape[0] and int(req_id.min()) >= 0, (
            "a request index past the block table"
        )
        # The pool row is computed in int32.
        largest = int(block_table.max()) * stride_factor + (layer_id + 1) * block_size
        assert largest < 2**31, "pool rows overflow int32"


convert_req_index_to_global = GlmConvertReqIndexToGlobal()
