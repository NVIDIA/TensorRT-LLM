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
    The target passes FP8 q, the packed NVFP4 sparse-layer cache (one KV head) with its block and per-tensor
    scales, and BF16 ``output`` [T, 8, 128]. Returns ``output``.

    Which kernel runs is decided inside, per step: a selected (non-identity) table on the TP8 geometry goes to the
    one-launch ``m3_sparse_decode_lc8`` kernel while modeling's ``SPARSE_DECODE_LC8`` covers it; an identity table
    (every request within 16 blocks, staged by the metadata), or a selected table with that range killed, runs the
    Triton per-row producer (the multi-query producer for identity steps of at least 17 rows), then the LSE merge.
    Every route also publishes the MXFP8 o_proj hand-off to ``m3_prequant.SLOT``, bit for bit
    ``torch.ops.trtllm.mxfp8_quantize(output, True)``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="identity selection, decode without speculation at batch 8: the Triton producer + merge",
            spec=dict(
                seq_lens=(1, 128, 129, 700, 2048, 1500, 64, 0), decode_query_len=1, identity=True
            ),
        ),
        Cell(
            why="identity selection, Eagle3 verify at one request (T=4): the Triton producer + merge",
            spec=dict(seq_lens=(1900,), decode_query_len=4, identity=True),
        ),
        Cell(
            why="identity selection, Eagle3 verify at batch 2 (T=8), a padding request",
            spec=dict(seq_lens=(2048, 0), decode_query_len=4, identity=True),
        ),
        Cell(
            why="identity selection, Eagle3 verify at batch 8 (T=32): the multi-query producer + merge",
            spec=dict(
                seq_lens=(1900, 2048, 700, 1500, 1200, 2000, 300, 1024),
                decode_query_len=4,
                identity=True,
            ),
        ),
        Cell(
            why="selected blocks, Eagle3 verify at one request (T=4): the one-launch long-context kernel",
            spec=dict(seq_lens=(40000,), decode_query_len=4, identity=False),
        ),
        Cell(
            why="selected blocks, Eagle3 verify at batch 2 (T=8): the one-launch long-context kernel",
            spec=dict(seq_lens=(9000, 40000), decode_query_len=4, identity=False),
        ),
    )

    note = """
    Certified only as the target calls it: one KV head of 8 query heads, head_dim 128, FP8 q, packed NVFP4
    cache, BF16 output, `num_topk_chunks` left to the dispatcher. The function also takes BF16 / FP8 caches, other
    head geometries and an explicit split-K factor (the in-tree tests drive those); none is certified here.

    The MXFP8 hand-off of the long-context route is certified on its own entry (`m3_sparse_decode_lc8`); the Triton
    merge's (`_store_mxfp8_head`) is held to `torch.ops.trtllm.mxfp8_quantize` by this entry's test.
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

        Only the packed NVFP4 cache is mirrored -- the target's only configuration -- one KV head at a time: KV head
        ``h`` serves query heads ``[h * group, (h + 1) * group)`` with its own top-k row. ``kv_scale``,
        ``num_topk_chunks`` and ``identity_blocks`` do not change what is computed.
        """
        assert k_block_scale is not None, "mirrors the NVFP4 cache only"
        kv_heads = k_paged.shape[1]
        assert kv_heads == 1, "mirrors the TP8 cache (one KV head per rank) only"
        group = q.shape[1] // kv_heads
        heads = []
        for h in range(kv_heads):
            heads.append(
                sparse_attention_reference(
                    q[:, h * group : (h + 1) * group],
                    k_paged.view(torch.uint8),
                    v_paged.view(torch.uint8),
                    k_block_scale.view(torch.uint8),
                    v_block_scale.view(torch.uint8),
                    k_global_scale,
                    v_global_scale,
                    topk_idx[h],
                    block_table,
                    seq_lens,
                    decode_query_len,
                    sm_scale,
                )
            )
        return torch.cat(heads, dim=1).to(output.dtype)

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
