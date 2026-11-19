# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA: request-local top-k key positions to rows of the MLA KV pool (CUDA)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


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
    )
    spec.update(overrides)
    return spec


class ConvertReqIndexToGlobal(OpWrapper):
    """``out[t, c]`` = the KV-pool row of request-local key ``tok = token_indices[t, c]`` of request
    ``req_id[t]``: ``block_table[req, tok // block_size] * stride_factor + layer_id * block_size +
    tok % block_size``; -1 where ``tok < 0``, ``tok // block_size`` is past the table row, or the
    table entry is negative. Columns ``[0, num_topk_tokens)``; returns a fresh int32 tensor shaped
    like ``token_indices``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row", spec=_spec(1)),
        Cell(
            why="a short request: the -1 top-k slots stay -1",
            spec=_spec(1, kv_len=1_500, invalid=548),
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
            why="a prefill chunk: 3000 rows of one request",
            spec=_spec(3000, requests=1, kv_len=6000, invalid=0),
        ),
        Cell(
            why="a long context: 500k keys on the last MLA layer's offset",
            spec=_spec(4, kv_len=500_000, table_width=8_000, pool_pages=9_000, layer=23),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The K3.1 general target calls this once per DSA layer per phase, in
    transform_local_topk_and_prepare_pool_view: token_indices is the phase's top-k rows after
    each row is sorted ascending with its -1 slots last (one plain torch sort, kept as torch and
    not an entry: it only fixes the order in which the sparse MLA accumulates the selected keys,
    so the attention output does not depend on the top-k kernels' atomic order); req_id the
    metadata's per-token request index (the generation part rebased to the first generation
    request), block_table the matching rows of the MLA block table, block_size the page size
    (32, 64 or 128), num_topk_tokens the top-k width (2048), and stride_factor / layer_id the
    MLA pool's page_index_scale * tokens_per_block and the layer's offset in the pool, from the
    KV cache manager. Exact and deterministic: one thread per output element, no reduction. The
    op makes contiguous copies of strided inputs (an allocation per call); the target's inputs
    are contiguous.
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
        # The launch takes its row count from req_id: more rows read past token_indices, fewer
        # leave the output's last rows unwritten.
        assert req_id.shape[0] == token_indices.shape[0], "one request index per top-k row"
        if req_id.numel() == 0:
            return
        # A request index past the table reads another allocation's memory.
        assert int(req_id.max()) < block_table.shape[0] and int(req_id.min()) >= 0, (
            "a request index past the block table"
        )
        # The pool row is computed in int32.
        largest = int(block_table.max()) * stride_factor + (layer_id + 1) * block_size
        assert largest < 2**31, "pool rows overflow int32"


convert_req_index_to_global = ConvertReqIndexToGlobal()
