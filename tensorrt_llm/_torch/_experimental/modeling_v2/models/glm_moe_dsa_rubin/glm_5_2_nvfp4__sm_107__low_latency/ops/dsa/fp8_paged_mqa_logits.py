# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexer FP8 MQA logits over paged keys (CuTe DSL), the whole verify window as one query atom."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._fp8_paged_mqa_logits import SPLIT_KV, num_sms, run_fp8_paged_mqa_logits

#: The indexer K cache's page: 64 tokens of 128 FP8 bytes, then their 64 fp32 scales.
_PAGE = 64
#: Columns of the c1 shard logits: the largest shard of the c1 gen server's indexer capacity
#: (1,048,596 keys) on 16 ranks.
_SHARD_CAPACITY = 65_664


def _spec(context_lens, max_context_len=_SHARD_CAPACITY, **overrides) -> dict:
    spec = dict(
        next_n=8,
        num_heads=32,
        head_dim=128,
        tokens_per_block=_PAGE,
        context_lens=tuple(context_lens),
        max_context_len=max_context_len,
        early_kv=True,
        data="int",
        pool_layout="contiguous",
    )
    spec.update(overrides)
    return spec


class GlmFp8PagedMqaLogits(OpWrapper):
    """``logits[b * next_n + t, j] = k_scale[j] * sum_h weights[b * next_n + t, h] *
    relu(q[b, t, h] . k[j])`` for the keys ``j < context_lens[b]`` of request ``b``.

    The keys live in the indexer K cache: key ``j`` of request ``b`` is token ``j % page`` of page
    ``block_table[b, j // page]``, whose bytes hold the page's FP8 rows and then their fp32 scales.
    ``weights`` already carry the Q scales and the softmax scale (the indexer prologue's or the
    wq_b epilogue's output). ``schedule_meta`` partitions the requests' 256-key splits over the
    device's SMs (``paged_mqa_logits_schedule``); the kernel runs one persistent CTA per SM.

    A request's ``next_n`` query rows are one MMA atom of ``next_n * num_heads`` columns. With
    ``early_kv`` (an even ``next_n`` whose half-atom is a whole number of 16-column UMMA atoms) the
    N-split variant runs, launched with PDL: it loads the KV blocks that end before each request's
    newest ``next_n`` keys before ``griddepcontrol.wait`` and everything else after it. Both
    variants compute every element in the same order.

    ``logits`` is the caller's [B * next_n, >= round_up(max_context_len, 256)] fp32 buffer: the
    kernel writes whole 256-key splits, so columns from ``context_lens[b]`` to the split's end hold
    unspecified values. Returns the view ``logits[:, :max_context_len]``.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1 key shard, 8 rows x 32 heads, early KV: a full 33,152-key shard",
            spec=_spec([33_152]),
        ),
        Cell(
            why="the request's last shard: 32,720 keys, a partial final page",
            spec=_spec([32_720]),
        ),
        Cell(
            why="an empty shard scores one key",
            spec=_spec([1]),
        ),
        Cell(
            why="the default kernel (early KV off) on the c1 shard",
            spec=_spec([33_152], early_kv=False),
        ),
        Cell(
            why="UE8M0-scaled keys and real-valued weights (accumulation order visible)",
            spec=_spec([20_000], data="randn"),
        ),
        Cell(
            why="the whole context on every rank (no key sharding): 200,000 keys of 1M",
            spec=_spec([200_000], max_context_len=1_048_576),
        ),
        Cell(
            why="cache pages interleaved with other layers' pages (strided page dim)",
            spec=_spec([5_000], pool_layout="interleaved"),
        ),
        Cell(
            why="four requests of 8 rows each (a larger batch's verify step)",
            spec=_spec([3_000, 70_000, 129, 40_000], max_context_len=131_072),
        ),
        Cell(
            why="TP4 three-request verify, full six-query atom and unequal contexts",
            spec=_spec([3000, 33000, 129], max_context_len=65536, next_n=6),
        ),
        Cell(
            why="TP4 partial one-request verify, full six-query atom",
            spec=_spec([33000], max_context_len=65536, next_n=6),
        ),
        Cell(
            why="TP4 three-request draft, one-query atoms",
            spec=_spec([3000, 33000, 129], max_context_len=65536, next_n=1, early_kv=False),
        ),
    )

    note = """
    Metadata consumed: none; every input is an argument. At c1 the target passes the step's Q
    ([1, 8, 32, 128] E4M3) and scaled weights ([8, 32] fp32), the layer's indexer K-cache view
    ([pages, 64, 1, 132] uint8), and, with the key-sharded top-k, the shard's context length,
    block table and schedule (``hist_shard_meta``, ``paged_mqa_logits_schedule``) with
    ``max_context_len`` = the shard capacity; without it, the request's indexer KV length, its
    INDEX_KEY block-table row and the full schedule.

    The compiled kernel is cached per (page size, heads, head dim, next_n, SMs, weight-register
    cache, early KV); the first call of a shape compiles (seconds), so it must run before CUDA
    graph capture.

    PDL: only the early-KV variant is launched with the PDL attribute. It requires that the
    kernels it waits for write no key older than each request's newest ``next_n`` and no block
    table, context length or schedule entry.

    Not certified: FP16 epilogues, next_n other than 1, 6 or 8, head counts other than 32, pages other
    than 64 tokens.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        logits: torch.Tensor,
        max_context_len: int,
        early_kv: bool,
    ) -> torch.Tensor:
        return run_fp8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            logits,
            max_context_len,
            early_kv,
        )

    def reference(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        logits: torch.Tensor,
        max_context_len: int,
        early_kv: bool,
    ) -> torch.Tensor:
        """[B * next_n, max_context_len] fp32, fp64-accumulated and rounded once; columns past a
        request's context are 0. The schedule, the output buffer and the variant change where and
        when the values are computed, not what they are."""
        batch, next_n, num_heads, head_dim = q.shape
        page = kv_cache.shape[1]
        out = torch.zeros(batch * next_n, max_context_len, dtype=torch.float32, device=q.device)
        page_bytes = kv_cache.reshape(kv_cache.shape[0], -1)
        for b, ctx in enumerate(context_lens.tolist()):
            num_pages = -(-ctx // page)
            pages = page_bytes[block_table[b, :num_pages].long()]
            keys = pages[:, : page * head_dim].contiguous().view(torch.float8_e4m3fn)
            keys = keys.reshape(num_pages * page, head_dim)[:ctx].float().double()
            scales = pages[:, page * head_dim :].contiguous().view(torch.float32)
            scales = scales.reshape(num_pages * page)[:ctx].double()
            qb = q[b].reshape(next_n * num_heads, head_dim).float().double()
            wb = weights[b * next_n : (b + 1) * next_n].reshape(next_n * num_heads, 1).double()
            for start in range(0, ctx, 16_384):
                stop = min(start + 16_384, ctx)
                scores = torch.relu(qb @ keys[start:stop].t()) * wb
                summed = scores.reshape(next_n, num_heads, stop - start).sum(dim=1)
                out[b * next_n : (b + 1) * next_n, start:stop] = (
                    summed * scales[start:stop]
                ).float()
        return out

    def is_valid(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        logits: torch.Tensor,
        max_context_len: int,
        early_kv: bool,
    ) -> None:
        assert q.dim() == 4 and q.dtype == torch.float8_e4m3fn, (
            f"q must be [B, next_n, H, D] E4M3, got {tuple(q.shape)} {q.dtype}"
        )
        batch, next_n, num_heads, head_dim = q.shape
        # The kernel takes Q and the weights through compact layouts; a strided view would be read
        # as if it were packed.
        assert q.is_contiguous(), "q must be contiguous"
        assert head_dim == 128, f"head_dim={head_dim}"
        assert weights.shape == (batch * next_n, num_heads) and weights.is_contiguous(), (
            f"weights must be a contiguous [B * next_n, H] = [{batch * next_n}, {num_heads}] "
            f"tensor, got {tuple(weights.shape)}"
        )
        assert weights.dtype == torch.float32, weights.dtype
        # A page is read as [page * D FP8 bytes][page * 4 scale bytes]: any other inner layout
        # reads keys and scales from the wrong bytes.
        assert kv_cache.dim() == 4 and kv_cache.dtype == torch.uint8, (
            "kv_cache must be the [pages, page, 1, D + 4] uint8 view of the indexer K cache"
        )
        page = kv_cache.shape[1]
        assert page in (32, 64, 128) and kv_cache.shape[2:] == (1, head_dim + 4), (
            f"kv_cache pages must be [{page}, 1, {head_dim + 4}] with page in (32, 64, 128)"
        )
        assert kv_cache[0].is_contiguous(), "the bytes of one page must be contiguous"
        assert context_lens.shape == (batch,) and context_lens.dtype == torch.int32, (
            "context_lens must be [B] int32"
        )
        assert block_table.dim() == 2 and block_table.shape[0] == batch, (
            "block_table must hold one row per request"
        )
        assert block_table.dtype == torch.int32 and block_table.is_contiguous(), (
            "block_table must be contiguous int32 rows"
        )
        lens = context_lens.tolist()
        longest = max(lens) if lens else 0
        # The kernel reads whole 128-key compute blocks of block-table entries.
        needed = -(-longest // 128) * (128 // page)
        assert block_table.shape[1] >= needed, (
            f"block_table rows of {block_table.shape[1]} pages cannot cover {longest} keys "
            f"({needed} page entries are read)"
        )
        if needed:
            assert int(block_table[:, :needed].max()) < kv_cache.shape[0], (
                "a block-table entry points past the cache"
            )
        assert longest <= max_context_len, (
            f"context length {longest} exceeds max_context_len={max_context_len}"
        )
        sms = num_sms(q.device.index if q.device.index is not None else 0)
        assert schedule_meta.shape == (sms + 1, 2) and schedule_meta.dtype == torch.int32, (
            f"schedule_meta must be the [{sms + 1}, 2] int32 schedule of this device's {sms} SMs"
        )
        assert logits.dim() == 2 and logits.dtype == torch.float32 and logits.stride(1) == 1, (
            "logits must be an fp32 matrix with a unit column stride"
        )
        assert logits.shape[0] == batch * next_n, (
            f"logits must hold B * next_n = {batch * next_n} rows, got {logits.shape[0]}"
        )
        assert logits.shape[1] >= -(-max_context_len // SPLIT_KV) * SPLIT_KV, (
            f"logits rows of {logits.shape[1]} columns cannot take the whole {SPLIT_KV}-key "
            f"splits of max_context_len={max_context_len}"
        )


fp8_paged_mqa_logits = GlmFp8PagedMqaLogits()
