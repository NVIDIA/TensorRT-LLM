# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer decode / verify logits over the paged FP4 index-key cache (DeepGEMM).

The FP4 counterpart of ``fp8_paged_mqa_logits``: DeepGEMM's paged MQA logits with the 2-D context
lengths, over the FP4 cache layout and with MXFP4 queries (``rubin_idx_fp4_quantize``'s format).
DeepGEMM is looked up on the first call, so the entry imports on a host without the built
extension.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .rubin_idx_fp4_quantize import mxfp4_values

#: Bytes of one index key in a page: 64 packed E2M1 bytes, then (after all P keys) its int32 scales.
DATA_BYTES = 64
TOKEN_BYTES = DATA_BYTES + 4


@functools.cache
def _deep_gemm():
    """``tensorrt_llm.deep_gemm``, on the first call."""
    from tensorrt_llm import deep_gemm

    return deep_gemm


def _spec(batch, next_n, context_lens, **overrides) -> dict:
    """``context_lens``: (low, high) of the per-request key counts; every ``zero_every``-th
    request then has none (a helix8 rank without keys of it)."""
    spec = dict(
        batch=batch,
        next_n=next_n,
        num_heads=32,
        head_dim=128,
        tokens_per_block=64,
        context_lens=tuple(context_lens),
        page_stride_pad=0,
        max_context_len=32768,
        data="int",
        zero_every=None,
    )
    spec.update(overrides)
    return spec


class RubinFp4PagedMqaLogitsDeepgemm(OpWrapper):
    """``logits[b * n + t, j] = sum_h weights[b * n + t, h] * relu(q[b, t, h] . k[j])`` for the keys
    ``j <= context_lens[b, t] - n + t`` of request ``b``.

    The cache and query layout of ``rubin_fp4_paged_mqa_logits``: ``q`` ``[B, n, 32, 64]`` packed
    E2M1 pairs (any 1-byte dtype) with ``sf_q`` int32 ``[B, n, 32]``, ``kv_cache`` uint8
    ``[pages, P, 1, 68]``, ``weights`` fp32 ``[B * n, 32]`` without the query scale.
    ``context_lens`` is int32 ``[B, n]`` (the request's key count repeated over its n rows),
    ``schedule_meta`` the DeepGEMM schedule built from those 2-D lengths with block_kv 64. Returns a
    new fp32 ``[B * n, max_context_len]`` tensor; columns past a row's causal end hold unspecified
    values.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="plain decode, one request at the top-k edge",
            spec=_spec(1, 1, (2049, 2049)),
        ),
        Cell(
            why="plain decode, 8 requests of 1.5k-10k keys",
            spec=_spec(8, 1, (1500, 10000)),
        ),
        Cell(
            why="helix8 verify rows expanded to one row per request: b16 k5, 96 rows",
            spec=_spec(96, 1, (1000, 6000)),
        ),
        Cell(
            why="helix8 rows of requests this rank holds no key of (zero lengths) among others",
            spec=_spec(96, 1, (1000, 6000), zero_every=4),
        ),
        Cell(
            why="verify rows n=2 (a native DeepGEMM next_n)",
            spec=_spec(4, 2, (3000, 9000)),
        ),
        Cell(
            why="verify rows n=4 (DeepGEMM's widest native next_n on sm_100f)",
            spec=_spec(4, 4, (3000, 9000)),
        ),
        Cell(
            why="padded page stride: the V2 manager's per-layer view of a wider page",
            spec=_spec(4, 1, (3000, 5000), page_stride_pad=512),
        ),
        Cell(
            why="real-valued MXFP4 keys, queries and weights: accumulation order visible",
            spec=_spec(4, 1, (3000, 6000), data="randn"),
        ),
    )
    note = """
    The Rubin default target calls this on helix8 with an FP4 index-key cache where the
    specialized helix8 fallback calls ``fp8_paged_mqa_logits``: an eager decode step that the
    key-shard path does not take and whose rows carry per-token bounds, which the CuTe DSL
    schedule cannot take. q is [B, n, 32, 64] with context_lens [B, n], or the rows expanded to
    one per request ([rows, 1, 32, 64], [rows, 1]). The output is a fresh allocation per call.
    Deterministic for equal inputs.
    """

    def __call__(
        self,
        q: torch.Tensor,
        sf_q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
    ) -> torch.Tensor:
        return _deep_gemm().fp8_fp4_paged_mqa_logits(
            (q.view(torch.int8), sf_q),
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            int(max_context_len),
        )

    def reference(
        self,
        q: torch.Tensor,
        sf_q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
    ) -> torch.Tensor:
        """fp64 evaluation over each row's causal keys, -inf elsewhere (``schedule_meta`` is
        ignored)."""
        batch, next_n = q.shape[:2]
        page = kv_cache.shape[1]
        queries = mxfp4_values(q, sf_q).double()
        logits = torch.full(
            (batch * next_n, max_context_len), float("-inf"), dtype=torch.float32, device=q.device
        )
        for b in range(batch):
            for t in range(next_n):
                ctx = int(context_lens[b, t])
                end = ctx - next_n + t + 1
                if end <= 0:  # no causal key (a request without keys on this rank included)
                    continue
                pages = block_table[b, : -(-ctx // page)].long()
                raw = kv_cache[pages].reshape(pages.shape[0], -1)
                keys = raw[:, : page * DATA_BYTES].reshape(-1, DATA_BYTES)
                scales = raw[:, page * DATA_BYTES : page * TOKEN_BYTES].reshape(-1, 4)
                k = mxfp4_values(keys, scales)[:end].double()
                score = (queries[b, t] @ k.t()).relu()
                row = b * next_n + t
                logits[row, :end] = (score * weights[row].double()[:, None]).sum(0).float()
        return logits

    def is_valid(
        self,
        q: torch.Tensor,
        sf_q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
    ) -> None:
        # The bytes are read as E2M1 pairs, 64 per 128-element head: 128-byte rows are FP8 data.
        assert q.element_size() == 1 and q.dim() == 4 and q.shape[3] == DATA_BYTES, (
            f"q must be packed E2M1 pairs [B, n, heads, 64], got {q.dtype} {tuple(q.shape)}"
        )
        # An fp32 per-head scale is an FP8 query's: its bits would be read as four exponents.
        assert sf_q.dtype == torch.int32 and sf_q.shape == q.shape[:3], (
            f"sf_q must be int32 [B, n, heads], got {sf_q.dtype} {tuple(sf_q.shape)}"
        )
        # A page is P * 64 key bytes, then P int32 scales: an FP8 page (132-byte tokens) puts the
        # keys and scales elsewhere.
        assert kv_cache.shape[2:] == (1, TOKEN_BYTES), (
            f"kv_cache must be [pages, P, 1, 68], got {tuple(kv_cache.shape)}"
        )
        assert context_lens.shape == q.shape[:2], (
            f"context_lens must be [B, n], got {tuple(context_lens.shape)}"
        )
        max_ctx = int(context_lens.max())
        assert max_ctx <= max_context_len, (
            "a context longer than max_context_len writes past the logits row"
        )
        # Key j of request b is on page block_table[b, j // P], and the column is never
        # bounded: a narrower table is read past its row.
        pages = -(-max_ctx // kv_cache.shape[1])
        assert block_table.shape[1] >= pages, (
            f"block_table must have >= ceil(max context / P) = {pages} columns, "
            f"got {block_table.shape[1]}"
        )
        assert weights.shape == (q.shape[0] * q.shape[1], q.shape[2]), (
            "weights must hold one row per query row"
        )


rubin_fp4_paged_mqa_logits_deepgemm = RubinFp4PagedMqaLogitsDeepgemm()
