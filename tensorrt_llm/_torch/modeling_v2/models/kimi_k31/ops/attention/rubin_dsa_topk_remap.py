# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA Top-K remap for prefill-sized batches: request-local token indices to latent-pool rows (Triton).

The same integers as ``trtllm::convert_req_index_to_global`` (``convert_req_index_to_global``),
computed in 1024-entry column tiles instead of one thread per entry, which is faster for
prefill-sized batches. The targets take it from 256 rows on (their ``_IDX_REMAP_TRITON_ROWS``)
and the C++ op below that.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention.rubin_dsa_kernels import triton_convert_req_index_to_global_index

#: Column tile of the Triton kernel; the Top-K width must be a multiple of it.
BLOCK_N = 1024


class RubinDsaTopkRemap(OpWrapper):
    """``out[t, j]`` = ``block_table[req_id[t], tok // block_size] * stride_factor + layer_id *
    block_size + tok % block_size`` for ``tok = token_indices[t, j]``, or ``-1`` where ``tok`` is
    negative, its page lies past the table or the table entry is negative.

    ``req_id`` int32 ``[T]``, ``block_table`` int32 ``[requests, pages]``, ``token_indices`` int32
    ``[T, num_topk_tokens]``; returns a new int32 ``[T, num_topk_tokens]``. ``stride_factor`` is the
    pool's rows per page (page index scale x ``block_size``), ``layer_id`` the layer's offset in a
    page.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the smallest prefill batch the indexer remaps here: 256 rows",
            spec=dict(rows=256, requests=1, pages=64, block_size=64, topk=2048, layers=24),
        ),
        Cell(
            why="a 16k-row prefill chunk of 4 requests, -1 padding in the short rows",
            spec=dict(rows=16384, requests=4, pages=300, block_size=64, topk=2048, layers=24),
        ),
        Cell(
            why="unmapped pages (-1 in the table) and indices past the table",
            spec=dict(
                rows=512, requests=2, pages=16, block_size=64, topk=2048, layers=1, holes=True
            ),
        ),
    )
    note = """
    The family copy of the engine's Triton remap with the C++ op's -1 rules (a negative index, a
    page past the table or a negative table entry). Bit for bit the C++ op on every cell.
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
        rows = token_indices.shape[0]
        return triton_convert_req_index_to_global_index(
            req_id,
            block_table,
            token_indices,
            block_size,
            num_topk_tokens,
            BLOCK_N=BLOCK_N,
            stride_factor=stride_factor,
            layer_id=layer_id,
        ).view(rows, num_topk_tokens)

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
        bad = (tok < 0) | ~valid_page | (base < 0)
        return torch.where(bad, -1, rows).to(torch.int32)

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
        # The kernel tiles the columns by BLOCK_N and never masks a tail: a ragged width reads
        # and writes past each row.
        assert num_topk_tokens % BLOCK_N == 0, f"the Top-K width must be a multiple of {BLOCK_N}"
        assert token_indices.shape[1] == num_topk_tokens
        assert req_id.shape[0] == token_indices.shape[0], "one request index per row"
        # A request index past the table reads another allocation's row.
        assert int(req_id.max()) < block_table.shape[0], "a request index past the block table"


rubin_dsa_topk_remap = RubinDsaTopkRemap()
