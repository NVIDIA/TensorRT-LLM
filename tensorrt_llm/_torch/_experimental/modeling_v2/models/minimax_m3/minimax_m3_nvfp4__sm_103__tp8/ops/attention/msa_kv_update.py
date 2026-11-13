# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MSA KV-length correction in one launch (this target's ``MsaKvUpdate`` hook)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The MSA pool's page: one sparse block is one 128-token KV page.
TOKENS_PER_BLOCK = 128


class MsaKvUpdate(OpWrapper):
    """Re-derive the MSA decode state after the overlap scheduler moved ``kv_lens``.

    ``MiniMaxM3MsaSparseAttentionMetadata.on_update_kv_lens`` calls this with its
    own buffers at the head of every captured speculative decode step. One
    single-CTA launch writes, for ``batch`` requests and ``total_q`` query
    tokens (all int32):

    * ``msa_seq_lens[b] = min(kv_lens[b], kv_lens_staged[b])`` and, when given,
      ``cu_kv_lens`` as its exclusive prefix sum (``batch + 1`` entries);
    * ``out_cache_loc[t]``, the KV write slot of the position token ``t``
      attends up to, through ``block_table``;
    * ``n_valid_blocks[t]``, when given, the block count top-k selection is
      bounded by.

    It returns None; the cells read the four buffers.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode without speculation at one request: qo = 1, every buffer written",
            spec=dict(batch=1, query_len=1, accept="full", cu_kv=True, n_valid=True),
        ),
        Cell(
            why="decode without speculation at the captured batch 8",
            spec=dict(batch=8, query_len=1, accept="full", cu_kv=True, n_valid=True),
        ),
        Cell(
            why="Eagle3 verify (draft 3): qo = 4 per request, partial acceptance shortens kv_lens below staged",
            spec=dict(batch=8, query_len=4, accept="partial", cu_kv=True, n_valid=True),
        ),
        Cell(
            why="the metadata without cu_kv_lens (no extend_ctx promotion): the optional prefix sum is skipped",
            spec=dict(batch=4, query_len=4, accept="partial", cu_kv=False, n_valid=True),
        ),
        Cell(
            why="CUDA-graph padding rows (kv_lens 0, pos -1) and no n_valid_blocks buffer",
            spec=dict(batch=8, query_len=4, accept="padded", cu_kv=True, n_valid=False),
        ),
        Cell(
            why="positions crossing a page boundary: the slot moves to the previous page of the request",
            spec=dict(batch=3, query_len=4, accept="page_edge", cu_kv=True, n_valid=True),
        ),
    )

    note = """
    The kernel covers any batch up to the block size's loop (it strides over
    256 threads), and any page size; only `tokens_per_block = 128`, the MSA
    pool's page, is certified.

    The op does not bounds-check the request index a token reads
    (`q_batch_row`) or the block-table column it derives: both come from the
    metadata's own staging, which keeps them in range by construction, and an
    out-of-range value reads whatever memory sits there. `is_valid` states both.
    """

    def __call__(
        self,
        kv_lens: torch.Tensor,
        kv_lens_staged: torch.Tensor,
        seq_lens: torch.Tensor,
        q_batch_row: torch.Tensor,
        q_intra: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        batch: int,
        total_q: int,
        msa_seq_lens: torch.Tensor,
        cu_kv_lens: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor,
        n_valid_blocks: Optional[torch.Tensor],
    ) -> None:
        torch.ops.trtllm.minimax_m3_msa_kv_update(
            kv_lens,
            kv_lens_staged,
            seq_lens,
            q_batch_row,
            q_intra,
            block_table,
            tokens_per_block,
            batch,
            total_q,
            msa_seq_lens,
            cu_kv_lens,
            out_cache_loc,
            n_valid_blocks,
        )

    def reference(
        self,
        kv_lens: torch.Tensor,
        kv_lens_staged: torch.Tensor,
        seq_lens: torch.Tensor,
        q_batch_row: torch.Tensor,
        q_intra: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        batch: int,
        total_q: int,
        msa_seq_lens: torch.Tensor,
        cu_kv_lens: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor,
        n_valid_blocks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """``(msa_seq_lens[:batch], cu_kv_lens[:batch + 1], out_cache_loc[:total_q],
        n_valid_blocks[:total_q])`` as the op is supposed to write them.

        The in-tree metadata's own torch re-derivation
        (``MiniMaxM3MsaSparseAttentionMetadata.on_update_kv_lens`` past its hook
        call), statement for statement, returning new tensors rather than
        writing the buffers. The buffer arguments only say which outputs exist:
        a ``None`` buffer gives a ``None`` result.
        """
        page = int(tokens_per_block)
        kv_true = torch.minimum(kv_lens[:batch], kv_lens_staged[:batch])
        cu_kv = None
        if cu_kv_lens is not None:
            cu_kv = torch.zeros(batch + 1, dtype=torch.int32, device=kv_lens.device)
            cu_kv[1:] = torch.cumsum(kv_true, 0)
        qbr = q_batch_row[:total_q].to(torch.long)
        qo_dev = seq_lens[:batch]
        pos = kv_true[qbr] - qo_dev[qbr] + q_intra[:total_q]
        pos_slot = pos.clamp_min(0)
        block_col = torch.div(pos_slot, page, rounding_mode="floor").clamp(
            max=int(block_table.shape[1]) - 1
        )
        slots = block_table[qbr, block_col.to(torch.long)] * page + torch.remainder(pos_slot, page)
        n_valid = None
        if n_valid_blocks is not None:
            n_valid = torch.div(
                (pos + 1).clamp_min(0) + (page - 1), page, rounding_mode="floor"
            ).to(torch.int32)
        return kv_true.to(torch.int32), cu_kv, slots.to(torch.int32), n_valid

    def is_valid(
        self,
        kv_lens: torch.Tensor,
        kv_lens_staged: torch.Tensor,
        seq_lens: torch.Tensor,
        q_batch_row: torch.Tensor,
        q_intra: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        batch: int,
        total_q: int,
        msa_seq_lens: torch.Tensor,
        cu_kv_lens: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor,
        n_valid_blocks: Optional[torch.Tensor],
    ) -> None:
        """The inputs this op takes and answers wrongly.

        Every operand but the block table is read as a dense run from its first
        element, so a strided view reads the wrong elements. The request index of
        each token and the table row it selects are not bounds-checked. Reads
        ``q_batch_row`` back to the host: a test-only cost.
        """
        for name, t in (
            ("kv_lens", kv_lens),
            ("kv_lens_staged", kv_lens_staged),
            ("seq_lens", seq_lens),
            ("q_batch_row", q_batch_row),
            ("q_intra", q_intra),
            ("msa_seq_lens", msa_seq_lens),
            ("out_cache_loc", out_cache_loc),
            ("cu_kv_lens", cu_kv_lens),
            ("n_valid_blocks", n_valid_blocks),
        ):
            assert t is None or t.is_contiguous(), (
                f"{name} must be contiguous: the kernel indexes it densely from data_ptr and ignores strides"
            )
        rows = q_batch_row[:total_q]
        assert total_q == 0 or (
            int(rows.min()) >= 0 and int(rows.max()) < min(batch, block_table.shape[0])
        ), (
            "q_batch_row must index a request below batch (and a block-table row): the op reads the "
            "request's lengths and table row unchecked"
        )


msa_kv_update = MsaKvUpdate()
