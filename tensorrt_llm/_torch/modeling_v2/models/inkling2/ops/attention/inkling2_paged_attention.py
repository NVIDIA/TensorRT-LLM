# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 attention on a paged KV cache: append the step's keys and values, then attend with
the relative position bias, the attention sink and the output gate.

Inkling 2 has no rotary embedding. Position enters the logits only through a bias that each query
token mixes per head from a bank of 16 bias-vs-distance profiles, `rel_logits[t, h, d] = r[t, h]
. proj[:, d]` for backward distances `0 <= d < rel_extent` (1024), and zero beyond. A sliding
layer attends the last 1024 positions (`p - j < window`); a global layer the whole prefix. Each
query token and head also has a sink logit, from its own projection, that absorbs probability
mass without contributing a value. The attention output is then gated per element by
`sigmoid(gate)`. See `kernels/paged_attention.py` for the formula.

The KV cache holds the keys and values after their short convolution, clipping and normalization
(bf16). The page table maps each request's token positions to pages of the layer's pool view.
"""

from __future__ import annotations

import math

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

from ...kernels.paged_attention import append_kv, attention

#: `1 / sqrt(head_dim)` for Inkling 2's 128-wide heads.
SM_SCALE = 1.0 / math.sqrt(128)

#: Requests of the cells, as (new tokens, cached tokens before the step).
_PREFILL = ((100, 0),)
_DECODE_PAST_WINDOW = ((1, 1500), (1, 7), (1, 1024), (1, 1023))
_MIXED = ((70, 1000), (1, 1500), (37, 0), (1, 60))


class Inkling2PagedAttention(OpWrapper):
    """Append `k`, `v` `[T, kv heads, 128]` to `kv_pool` and attend with `q` `[T, heads, 128]`.

    `kv_pool` is the layer's HND pool view `[pages, 2, kv heads, page size, 128]`; `page_table`
    `[R, W]` int32 maps request `r`'s token blocks to its pages. `query_start` `[R + 1]` and
    `kv_len` `[R]` (cached plus new tokens) give each request's new tokens and positions;
    `token_request` `[T]` maps tokens to requests. `tile_request`, `tile_start` (int32, one entry
    per program) and `block_t` are the step's tile plan (`kernels.paged_attention.tile_plan`).
    `rel_logits` `[T, heads, rel_extent]` fp32, `sink` `[T, heads]` and `gate` `[T, heads, 128]`
    are per query token; `window` is 1024 on a sliding layer and 0 on a global one. Returns
    `[T, heads, 128]` in `q`'s dtype.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="sliding layer at TP4 (12 q / 2 kv heads), one 100-token prefill",
            spec=dict(heads=12, kv_heads=2, window=1024, requests=_PREFILL),
        ),
        Cell(
            why="sliding layer at TP4, decodes before, at and past the window edge",
            spec=dict(heads=12, kv_heads=2, window=1024, requests=_DECODE_PAST_WINDOW),
        ),
        Cell(
            why="sliding layer at TP4, a mixed step with a chunk that crosses the window",
            spec=dict(heads=12, kv_heads=2, window=1024, requests=_MIXED),
        ),
        Cell(
            why="global layer at TP4, decodes past the bias extent",
            spec=dict(heads=12, kv_heads=2, window=0, requests=_DECODE_PAST_WINDOW),
        ),
        Cell(
            why="global layer at TP1 (48 q / 8 kv heads), a mixed step",
            spec=dict(heads=48, kv_heads=8, window=0, requests=_MIXED),
        ),
    )

    note = """
    The bias is indexed by backward distance and is zero from rel_extent on, so a global
    layer's keys past 1023 tokens carry no bias while a sliding layer's window ends exactly
    where its bias does. The sink is a logit, compared with the logsumexp of the scaled and
    biased scores; it is not scaled by sm_scale. Only decodes and chunks at positions past
    1023 tell the window, the bias extent and an off-by-one in either apart, which is what the
    decode cells' positions 1023, 1024 and 1500 are for.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_pool: torch.Tensor,
        page_table: torch.Tensor,
        query_start: torch.Tensor,
        kv_len: torch.Tensor,
        token_request: torch.Tensor,
        tile_request: torch.Tensor,
        tile_start: torch.Tensor,
        block_t: int,
        rel_logits: torch.Tensor,
        sink: torch.Tensor,
        gate: torch.Tensor,
        window: int,
        sm_scale: float,
    ) -> torch.Tensor:
        append_kv(k, v, kv_pool, page_table, query_start, kv_len, token_request)
        return attention(
            q,
            kv_pool,
            page_table,
            query_start,
            kv_len,
            tile_request,
            tile_start,
            block_t,
            rel_logits,
            sink,
            gate,
            window,
            sm_scale,
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_pool: torch.Tensor,
        page_table: torch.Tensor,
        query_start: torch.Tensor,
        kv_len: torch.Tensor,
        token_request: torch.Tensor,
        tile_request: torch.Tensor,
        tile_start: torch.Tensor,
        block_t: int,
        rel_logits: torch.Tensor,
        sink: torch.Tensor,
        gate: torch.Tensor,
        window: int,
        sm_scale: float,
    ) -> torch.Tensor:
        """Per request and head, dense fp32 attention over the pages it reads; writes `k`, `v`
        to `kv_pool` as the op does. The tile plan is not needed."""
        page_size = kv_pool.shape[3]
        num_heads = q.shape[1]
        group = num_heads // kv_pool.shape[2]
        rel_extent = rel_logits.shape[2]
        bounds = query_start.tolist()
        lengths = kv_len.tolist()
        out = torch.empty_like(q)
        for r in range(len(lengths)):
            start, end = bounds[r], bounds[r + 1]
            n, length = end - start, lengths[r]
            positions = torch.arange(length - n, length, device=q.device)
            pages = page_table[r, positions // page_size].long()
            kv_pool[pages, 0, :, positions % page_size] = k[start:end]
            kv_pool[pages, 1, :, positions % page_size] = v[start:end]
            lo = max(0, length - n - window + 1) if window > 0 else 0
            keys_pos = torch.arange(lo, length, device=q.device)
            key_pages = page_table[r, keys_pos // page_size].long()
            keys = kv_pool[key_pages, 0, :, keys_pos % page_size].float()  # [L, kv heads, D]
            values = kv_pool[key_pages, 1, :, keys_pos % page_size].float()
            dist = positions[:, None] - keys_pos[None, :]
            allowed = dist >= 0
            if window > 0:
                allowed &= dist < window
            in_extent = (dist >= 0) & (dist < rel_extent)
            for h in range(num_heads):
                with true_fp32_matmul():
                    logits = q[start:end, h].float() @ keys[:, h // group].t() * sm_scale
                bias = rel_logits[start:end, h].gather(1, dist.clamp(0, rel_extent - 1))
                logits = logits + torch.where(in_extent, bias, torch.zeros_like(bias))
                logits = logits.masked_fill(~allowed, float("-inf"))
                lse = torch.logsumexp(logits, dim=1)
                with true_fp32_matmul():
                    o = torch.softmax(logits, dim=1) @ values[:, h // group]
                keep = torch.sigmoid(lse - sink[start:end, h].float())
                o = o * keep[:, None] * torch.sigmoid(gate[start:end, h].float())
                out[start:end, h] = o.to(q.dtype)
        return out

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_pool: torch.Tensor,
        page_table: torch.Tensor,
        query_start: torch.Tensor,
        kv_len: torch.Tensor,
        token_request: torch.Tensor,
        tile_request: torch.Tensor,
        tile_start: torch.Tensor,
        block_t: int,
        rel_logits: torch.Tensor,
        sink: torch.Tensor,
        gate: torch.Tensor,
        window: int,
        sm_scale: float,
    ) -> None:
        num_tokens, num_heads, head_dim = q.shape
        assert kv_pool.dim() == 5 and kv_pool.shape[1] == 2 and kv_pool.shape[4] == head_dim, (
            "kv_pool must be the HND view [pages, 2, kv heads, page size, head dim]"
        )
        assert kv_pool.stride(4) == 1 and q.stride(2) == 1 and gate.stride(2) == 1, (
            "head dims must be contiguous"
        )
        num_kv_heads = kv_pool.shape[2]
        assert num_heads % num_kv_heads == 0, "query heads must group evenly over kv heads"
        assert k.shape == v.shape == (num_tokens, num_kv_heads, head_dim), (
            "k and v must be [T, kv heads, head dim]"
        )
        assert k.stride(2) == 1 and v.stride(2) == 1, "k and v must have contiguous head dims"
        assert k.dtype == v.dtype == q.dtype == kv_pool.dtype, "q, k, v and the pool share a dtype"
        assert rel_logits.dtype == torch.float32 and rel_logits.shape[:2] == (num_tokens, num_heads)
        assert rel_logits.stride(2) == 1, "rel_logits must be contiguous over distances"
        assert sink.shape == (num_tokens, num_heads) and sink.stride(1) == 1, (
            "sink must be [T, heads]"
        )
        assert gate.shape == q.shape, "gate must be [T, heads, head dim]"
        assert window >= 0, "window is the sliding window length, or 0 for full attention"
        if window > 0:
            # A key past the bias extent inside the window would have no bias; Inkling 2's
            # sliding layers have window == rel_extent.
            assert window <= rel_logits.shape[2], "a sliding window may not exceed rel_extent"
        bounds = query_start.tolist()
        lengths = kv_len.tolist()
        new = [b - a for a, b in zip(bounds, bounds[1:])]
        assert bounds[0] == 0 and bounds[-1] == num_tokens and all(n > 0 for n in new)
        assert all(length >= n for length, n in zip(lengths, new)), "kv_len counts the new tokens"
        page_size = kv_pool.shape[3]
        assert all(-(-length // page_size) <= page_table.shape[1] for length in lengths), (
            "page_table must cover every request's tokens"
        )
        plan_request, plan_start = [], []
        for r, n in enumerate(new):
            for first in range(bounds[r], bounds[r] + n, block_t):
                plan_request.append(r)
                plan_start.append(first)
        assert tile_request.tolist() == plan_request and tile_start.tolist() == plan_start, (
            "the tile plan must cover each request's tokens in blocks of block_t"
        )
        assert torch.equal(
            token_request.cpu(),
            torch.repeat_interleave(torch.arange(len(new), dtype=torch.int32), torch.tensor(new)),
        ), "token_request must map each token to its request"


inkling2_paged_attention = Inkling2PagedAttention()
