# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's block-sparse decode attention (Triton, with the one-launch long-context route)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..triton_sparse_decode import minimax_m3_sparse_attn_decode
from .m3_sparse_decode_lc8 import PAGE, sparse_attention_reference


class M3SparseDecode(OpWrapper):
    """``minimax_m3_sparse_attn_decode`` as ``M3Attention._decode_sparse_core`` calls it.

    Attention of each query token over the blocks ``topk_idx`` selected for it
    (``[num_kv_heads, T, topk]``, ascending ids with a -1 tail), each block
    truncated at the token's causal extent, written into ``output`` in place.
    The target passes FP8 q, the packed NVFP4 sparse-layer cache with its block
    and per-tensor scales, and BF16 ``output`` [T, 8, 128]. Returns ``output``.

    Which kernel runs is decided inside, per step: a selected (non-identity)
    table on the TP8 geometry goes to the one-launch ``m3_sparse_decode_lc8``
    kernel; an identity table (every request within 16 blocks, staged by the
    metadata) runs the Triton producer, or from 17 query rows the multi-query
    producer, then the LSE merge. Every route also publishes the MXFP8 o_proj
    hand-off to ``m3_prequant.SLOT``; this entry certifies the attention output.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="identity selection, decode without speculation at batch 8: the Triton producer + merge",
            spec=dict(
                seq_lens=(1, 128, 129, 700, 2048, 1500, 64, 0), decode_query_len=1, identity=True
            ),
        ),
        Cell(
            why="identity selection, Eagle3 verify at one request (T=4): one block per producer program",
            spec=dict(seq_lens=(1900,), decode_query_len=4, identity=True),
        ),
        Cell(
            why="identity selection, Eagle3 verify at batch 4 (T=16): the last row count below the MQ producer",
            spec=dict(seq_lens=(300, 2048, 4, 1000), decode_query_len=4, identity=True),
        ),
        Cell(
            why="identity selection, Eagle3 verify at batch 8 (T=32): the multi-query producer, padding rows",
            spec=dict(
                seq_lens=(2048, 130, 5, 0, 1777, 900, 2000, 0), decode_query_len=4, identity=True
            ),
        ),
        Cell(
            why="selected blocks, Eagle3 verify at batch 2: routed to the one-launch long-context kernel",
            spec=dict(seq_lens=(9000, 40000), decode_query_len=4, identity=False),
        ),
        Cell(
            why="selected blocks, decode without speculation at batch 8: the long-context route at T=8",
            spec=dict(
                seq_lens=(3000, 2100, 9999, 2048, 2049, 4500, 16000, 2200),
                decode_query_len=1,
                identity=False,
            ),
        ),
    )

    note = """
    Certified only as the target calls it: one KV head of 8 query heads,
    head_dim 128, FP8 q, packed NVFP4 cache, BF16 output, `num_topk_chunks`
    left to the dispatcher. The function also takes BF16 / FP8 caches, other
    head geometries and an explicit split-K factor (the in-tree tests drive
    those); none is certified here.

    The MXFP8 hand-off each route publishes is certified on the long-context
    kernel's own entry (`m3_sparse_decode_lc8`); the Triton merge writes it with
    the same recipe (`_store_mxfp8_head`).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k_paged: torch.Tensor,
        v_paged: torch.Tensor,
        topk_idx: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
        decode_query_len: int,
        kv_scale: Optional[torch.Tensor] = None,
        k_block_scale: Optional[torch.Tensor] = None,
        v_block_scale: Optional[torch.Tensor] = None,
        k_global_scale: Optional[torch.Tensor] = None,
        v_global_scale: Optional[torch.Tensor] = None,
        num_topk_chunks: Optional[int] = None,
        identity_blocks: bool = False,
    ) -> torch.Tensor:
        minimax_m3_sparse_attn_decode(
            q,
            k_paged,
            v_paged,
            topk_idx,
            block_table,
            seq_lens,
            sm_scale=sm_scale,
            output=output,
            decode_query_len=decode_query_len,
            kv_scale=kv_scale,
            k_block_scale=k_block_scale,
            v_block_scale=v_block_scale,
            k_global_scale=k_global_scale,
            v_global_scale=v_global_scale,
            num_topk_chunks=num_topk_chunks,
            identity_blocks=identity_blocks,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        k_paged: torch.Tensor,
        v_paged: torch.Tensor,
        topk_idx: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
        decode_query_len: int,
        kv_scale: Optional[torch.Tensor] = None,
        k_block_scale: Optional[torch.Tensor] = None,
        v_block_scale: Optional[torch.Tensor] = None,
        k_global_scale: Optional[torch.Tensor] = None,
        v_global_scale: Optional[torch.Tensor] = None,
        num_topk_chunks: Optional[int] = None,
        identity_blocks: bool = False,
    ) -> torch.Tensor:
        """``output`` as the call is supposed to write it: float64 block-sparse attention over the
        dequantized NVFP4 blocks (``sparse_attention_reference``), rounded once to ``output``'s dtype.

        Only the packed NVFP4 cache with one KV head is mirrored -- the target's
        only configuration. ``kv_scale``, ``num_topk_chunks`` and
        ``identity_blocks`` do not change what is computed.
        """
        assert k_block_scale is not None and k_paged.shape[1] == 1, (
            "mirrors the one-KV-head NVFP4 cache only"
        )
        attn = sparse_attention_reference(
            q,
            k_paged.view(torch.uint8),
            v_paged.view(torch.uint8),
            k_block_scale.view(torch.uint8),
            v_block_scale.view(torch.uint8),
            k_global_scale,
            v_global_scale,
            topk_idx[0],
            block_table,
            seq_lens,
            decode_query_len,
            sm_scale,
        )
        return attn.to(output.dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        k_paged: torch.Tensor,
        v_paged: torch.Tensor,
        topk_idx: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
        decode_query_len: int,
        kv_scale: Optional[torch.Tensor] = None,
        k_block_scale: Optional[torch.Tensor] = None,
        v_block_scale: Optional[torch.Tensor] = None,
        k_global_scale: Optional[torch.Tensor] = None,
        v_global_scale: Optional[torch.Tensor] = None,
        num_topk_chunks: Optional[int] = None,
        identity_blocks: bool = False,
    ) -> None:
        """The inputs this call takes and answers wrongly.

        Every producer walks a token's first ``min(topk, ceil(kv_len / 128))``
        table entries and indexes the block table with them unchecked, so they
        must be real blocks of the request. ``identity_blocks`` is a promise the
        multi-query producer relies on: every token of a request reads the same
        blocks, all of them. Reads ``topk_idx`` and ``seq_lens`` back to the
        host: a test-only cost.
        """
        total_q = q.shape[0]
        seq = seq_lens.tolist()
        max_topk = topk_idx.shape[-1]
        for tok in range(total_q):
            req, offset = divmod(tok, decode_query_len)
            kv_len = max(seq[req] - decode_query_len + offset + 1, 0)
            nblocks = (kv_len + PAGE - 1) // PAGE
            ids = topk_idx[:, tok, : min(max_topk, nblocks)]
            assert bool(((ids >= 0) & (ids < min(nblocks, block_table.shape[1]))).all()), (
                f"token {tok}: the first min(topk, block count) selected ids must be blocks of the request; "
                "the block table is indexed with them unchecked"
            )
            if identity_blocks:
                assert nblocks <= max_topk and bool(
                    (ids == torch.arange(nblocks, device=ids.device, dtype=ids.dtype)).all()
                ), (
                    f"token {tok}: identity_blocks promises the table holds every block of the request in order"
                )


m3_sparse_decode = M3SparseDecode()
