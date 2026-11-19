# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA decode-step metadata in one launch: request index per token, index-key slot mappings and
the generation indptrs (the engine's Triton kernel)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.kernels import (
    fused_dsa_decode_metadata as _engine_fused_dsa_decode_metadata,
)
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _spec(batch, next_n, kv_lens, tokens_per_block=64, **overrides) -> dict:
    """``batch`` generation requests of ``next_n`` tokens, KV lengths drawn in ``kv_lens``."""
    spec = dict(
        batch=batch,
        next_n=next_n,
        max_query_len=next_n,
        kv_lens=tuple(kv_lens),
        tokens_per_block=tokens_per_block,
        table_slack=3,
        pool_pages=40_000,
    )
    spec.update(overrides)
    return spec


class FusedDsaDecodeMetadata(OpWrapper):
    """For a generation-only step of ``S`` requests, request ``r`` holding ``seq_lens[r]`` query
    tokens at KV positions ``kv_lens[r] - seq_lens[r] ..  kv_lens[r] - 1``, writes in place:

    * ``req_idx_per_token[t]`` (int32) = the request of flattened token ``t``;
    * ``slot_mapping_fp8[t]`` / ``slot_mapping_scale[t]`` (int64) = the byte offsets of token
      ``t``'s index key and its scale in the index-key pool: with ``p`` its KV position, ``b =
      block_offsets[r, p // P]`` (P = ``tokens_per_block``), ``b * P * (D + 4) + (p % P) * D``
      and ``b * P * (D + 4) + P * D + (p % P) * 4`` (D = ``data_bytes_per_token``, 4 bytes of
      scale per ``quant_block_size`` values of ``index_head_dim``);
    * ``gen_kv_indptr`` / ``gen_cached_token_indptr`` (int64 [S + 1]) = 0 then the running sums
      of ``kv_lens`` and of ``kv_lens - seq_lens``.

    Every output must be sliced to its live extent (``[:num_tokens]``, ``[:S + 1]``); nothing past
    it is written. Returns None.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="plain decode, one request", spec=_spec(1, 1, (5000, 5000))),
        Cell(
            why="plain decode, 64 requests, 1-token to 30k contexts on 64-token pages",
            spec=_spec(64, 1, (1, 30000)),
        ),
        Cell(why="plain decode, 32-token pages", spec=_spec(16, 1, (100, 20000), 32)),
        Cell(why="plain decode, 128-token pages", spec=_spec(16, 1, (100, 20000), 128)),
        Cell(
            why="plain decode, 256 requests: the header's widest cumsum",
            spec=_spec(256, 1, (1, 16000)),
        ),
        Cell(why="DFlash2 w5 verify, 8 requests", spec=_spec(8, 5, (5, 9000))),
        Cell(
            why="DFlash2 w8 verify, 64 requests, tokens straddling page edges",
            spec=_spec(64, 8, (8, 4100), 32),
        ),
        Cell(
            why="DFlash2 w8 verify, 256 requests on 128-token pages",
            spec=_spec(256, 8, (2000, 60000), 128),
        ),
        Cell(
            why="fewer tokens than lanes: one-token rows under an 8-token lane count",
            spec=_spec(32, 1, (10, 9000), max_query_len=8),
        ),
        Cell(
            why="a 256k context: the last page of a 4096-entry table row",
            spec=_spec(2, 8, (262000, 262144), table_slack=0),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The K3.1 general target calls this from the DSA metadata's on_update_kv_lens; it runs on every
    generation-only step (no context request), replacing the eager chain it states in
    `reference`: seq_lens / kv_lens the step's [:S] device lengths, block_offsets the index-key
    cache's block offsets [:S], the four outputs the metadata's persistent buffers sliced to the
    step, num_tokens the step's token count, max_query_len 1 + max_draft_tokens,
    tokens_per_block the KV page size, index_head_dim 128, quant_block_size 128 and
    data_bytes_per_token 128 (FP8 index keys). Integer arithmetic only: bit-identical to the
    eager chain. One Triton compile per (lane count, request-count power of two, page size, table
    width); the launch captures in the decode CUDA graph.

    A KV position past the block-table row is clamped to its last entry, as the eager chain's
    CUDA path does; the target's tables always cover the context.
    """

    def __call__(
        self,
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        block_offsets: torch.Tensor,
        req_idx_per_token: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        gen_kv_indptr: torch.Tensor,
        gen_cached_token_indptr: torch.Tensor,
        num_tokens: int,
        max_query_len: int,
        tokens_per_block: int,
        index_head_dim: int,
        quant_block_size: int,
        data_bytes_per_token: Optional[int],
    ) -> None:
        _engine_fused_dsa_decode_metadata(
            seq_lens,
            kv_lens,
            block_offsets,
            req_idx_per_token,
            slot_mapping_fp8,
            slot_mapping_scale,
            gen_kv_indptr,
            gen_cached_token_indptr,
            num_tokens=num_tokens,
            max_query_len=max_query_len,
            tokens_per_block=tokens_per_block,
            index_head_dim=index_head_dim,
            quant_block_size=quant_block_size,
            data_bytes_per_token=data_bytes_per_token,
        )

    def reference(
        self,
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        block_offsets: torch.Tensor,
        req_idx_per_token: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        gen_kv_indptr: torch.Tensor,
        gen_cached_token_indptr: torch.Tensor,
        num_tokens: int,
        max_query_len: int,
        tokens_per_block: int,
        index_head_dim: int,
        quant_block_size: int,
        data_bytes_per_token: Optional[int],
    ) -> None:
        """The eager chain the kernel replaces (searchsorted request map, the slot-mapping
        arithmetic with the table clamp, two int64 cumsums), into the same outputs.
        ``max_query_len`` only sizes the kernel's lanes and is ignored."""
        device = seq_lens.device
        data_bytes = index_head_dim if data_bytes_per_token is None else data_bytes_per_token
        scale_size = index_head_dim // quant_block_size * 4
        block_stride = tokens_per_block * (data_bytes + scale_size)
        scale_base = tokens_per_block * data_bytes

        cu_seq = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        token = torch.arange(num_tokens, device=device, dtype=torch.int32)
        req = torch.searchsorted(cu_seq, token, right=True)
        req_idx_per_token[:num_tokens] = req.to(req_idx_per_token.dtype)

        req64 = req.long()
        seq64 = seq_lens.long()
        starts = torch.cumsum(seq64, dim=0) - seq64
        offset = torch.arange(num_tokens, device=device, dtype=torch.int64) - starts[req64]
        position = (kv_lens.long() - seq64)[req64] + offset
        page = (position // tokens_per_block).clamp(0, block_offsets.shape[1] - 1)
        in_page = position % tokens_per_block
        block = block_offsets[req64, page].long()
        slot_mapping_fp8[:num_tokens] = block * block_stride + in_page * data_bytes
        slot_mapping_scale[:num_tokens] = block * block_stride + scale_base + in_page * scale_size

        num_seqs = seq_lens.shape[0]
        gen_kv_indptr[0] = 0
        gen_cached_token_indptr[0] = 0
        torch.cumsum(kv_lens.long(), dim=0, out=gen_kv_indptr[1 : num_seqs + 1])
        torch.cumsum(kv_lens.long() - seq64, dim=0, out=gen_cached_token_indptr[1 : num_seqs + 1])

    def is_valid(
        self,
        seq_lens: torch.Tensor,
        kv_lens: torch.Tensor,
        block_offsets: torch.Tensor,
        req_idx_per_token: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        gen_kv_indptr: torch.Tensor,
        gen_cached_token_indptr: torch.Tensor,
        num_tokens: int,
        max_query_len: int,
        tokens_per_block: int,
        index_head_dim: int,
        quant_block_size: int,
        data_bytes_per_token: Optional[int],
    ) -> None:
        num_seqs = seq_lens.shape[0]
        # Rows past num_tokens are dropped, and tokens past sum(seq_lens) are never written.
        assert int(seq_lens.sum()) == num_tokens, (
            f"num_tokens={num_tokens} is not the step's {int(seq_lens.sum())} tokens"
        )
        # A request's tokens past the lane count are never written.
        assert int(seq_lens.max()) <= max_query_len, (
            f"a request holds more than max_query_len={max_query_len} tokens"
        )
        # Positions before 0 or past the table row are clamped to a wrong page.
        assert bool((kv_lens >= seq_lens).all()), "a request's KV length is below its tokens"
        pages = -(-int(kv_lens.max()) // tokens_per_block)
        assert pages <= block_offsets.shape[1], (
            f"a context needs {pages} pages; the block table row holds {block_offsets.shape[1]}"
        )
        # The kernel writes through the views' base pointers by index: a view shorter than the
        # live extent is written past its end.
        assert (
            min(req_idx_per_token.numel(), slot_mapping_fp8.numel(), slot_mapping_scale.numel())
            >= num_tokens
        ), "a per-token output is too short"
        assert min(gen_kv_indptr.numel(), gen_cached_token_indptr.numel()) >= num_seqs + 1, (
            "an indptr output is shorter than num_seqs + 1"
        )
        assert all(
            t.is_contiguous()
            for t in (
                req_idx_per_token,
                slot_mapping_fp8,
                slot_mapping_scale,
                gen_kv_indptr,
                gen_cached_token_indptr,
                seq_lens,
                kv_lens,
            )
        ), "the kernel indexes its 1D inputs and outputs as packed arrays"


fused_dsa_decode_metadata = FusedDsaDecodeMetadata()
