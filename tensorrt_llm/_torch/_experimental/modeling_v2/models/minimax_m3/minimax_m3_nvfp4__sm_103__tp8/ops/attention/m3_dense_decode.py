# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dense layers' decode attention over the full page table with valid sequence-length bounds."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.trtllm_gen_dense_decode import (  # noqa: E501
    minimax_m3_trtllm_gen_dense_decode,
)

__all__ = ["m3_dense_decode"]

#: The TP8 dense layers' geometry: 8 query heads and one KV head of dim 128 per rank, an FP8 KV pool.
HEADS = 8
KV_HEADS = 1
HEAD_DIM = 128
#: Valid upper bounds used to compare short-context and model-maximum dispatch.
SHORT_BOUND = 16384
MODEL_MAX = 1 << 20


def dense_attention_reference(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    sm_scale: float,
) -> torch.Tensor:
    """fp32 causal decode attention: query row ``b * qlen + i`` of request b attends its first
    ``seq_lens[b] - qlen + i + 1`` keys. ``keys`` / ``values``: [batch, kv_heads, max_len, head_dim] (the request's
    tokens in order). Returns fp32 [total_q, heads, head_dim]."""
    batch, kv_heads, max_len, _ = keys.shape
    heads = q.shape[1]
    group = heads // kv_heads
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    pos = torch.arange(max_len, device=q.device)
    with true_fp32_matmul():
        for b in range(batch):
            full = int(seq_lens[b])
            rows = q[b * decode_query_len : (b + 1) * decode_query_len].float()  # [qlen, heads, d]
            limit = full - decode_query_len + 1 + torch.arange(decode_query_len, device=q.device)
            valid = pos[None, :] < limit[:, None]  # [qlen, max_len]
            for h in range(kv_heads):
                qh = rows[:, h * group : (h + 1) * group]  # [qlen, group, d]
                logits = torch.einsum("qgd,kd->qgk", qh, keys[b, h].float()) * sm_scale
                logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                out[
                    b * decode_query_len : (b + 1) * decode_query_len, h * group : (h + 1) * group
                ] = torch.einsum("qgk,kd->qgd", probs, values[b, h].float())
    return out


def gather_paged_kv(
    kv_cache_manager, layer_idx: int, block_table: torch.Tensor, seq_lens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The K and V of each request in token order, [batch, kv_heads, max_len, head_dim], from the flat sub-page pool
    (``get_kv_subpage_pool``: K of slot s at sub-page s * subpages_per_slot, V one sub-page later)."""
    pool, subpages = kv_cache_manager.get_kv_subpage_pool(layer_idx, "HND")
    page = pool.shape[2]
    pages = (int(seq_lens.max()) + page - 1) // page
    slots = block_table[:, :pages].long()
    k = pool[slots * subpages]  # [batch, pages, kv_heads, page, d]
    v = pool[slots * subpages + 1]
    batch, _, kv_heads, _, d = k.shape

    def order(t):
        return t.permute(0, 2, 1, 3, 4).reshape(batch, kv_heads, pages * page, d)

    return order(k), order(v)


class M3DenseDecode(OpWrapper):
    """Decode over each request's full causal context in the paged FP8 pool.

    The sequence-length bound covers every request and the BF16 output is written in place.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request verifying 1 + 3 rows over a 20000-token context, its full sequence-length bound",
            spec=dict(kv_lens=(20000,), qlen=4, max_seq_len=20000, tight_bound=True),
        ),
        Cell(
            why="one request, one row, over a 65536-token context with its full sequence-length bound",
            spec=dict(kv_lens=(65536,), qlen=1, max_seq_len=65536, tight_bound=True),
        ),
        Cell(
            why="one request verifying 1 + 3 rows over a 3000-token context: an upper bound above the length",
            spec=dict(kv_lens=(3000,), qlen=4, max_seq_len=SHORT_BOUND, tight_bound=True),
        ),
        Cell(
            why="two requests with the model maximum as the upper bound",
            spec=dict(kv_lens=(20000, 5000), qlen=4, max_seq_len=MODEL_MAX, tight_bound=False),
        ),
    )

    note = """
    Outputs at two valid sequence-length bounds agree to a few BF16 ulps and both match the fp32
    reference within the FP8-KV tolerance of the in-tree dense decode test.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache_manager,
        layer_idx: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
        decode_query_len: int,
        max_seq_len: int,
        max_num_requests: int,
    ) -> torch.Tensor:
        minimax_m3_trtllm_gen_dense_decode(
            q,
            kv_cache_manager,
            layer_idx,
            block_table,
            seq_lens,
            sm_scale=sm_scale,
            output=output,
            decode_query_len=decode_query_len,
            max_seq_len=max_seq_len,
            max_num_requests=max_num_requests,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        kv_cache_manager,
        layer_idx: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
        decode_query_len: int,
        max_seq_len: int,
        max_num_requests: int,
    ) -> torch.Tensor:
        """fp32 attention over the pool's K / V (q rounded to the pool's dtype first, as the kernel runs it).

        ``output``, ``max_seq_len`` and ``max_num_requests`` do not change the result."""
        keys, values = gather_paged_kv(kv_cache_manager, layer_idx, block_table, seq_lens)
        if keys.dtype == torch.float8_e4m3fn:
            q = q.to(torch.float8_e4m3fn)
        return dense_attention_reference(q, keys, values, seq_lens, decode_query_len, sm_scale)

    def is_valid(
        self,
        q: torch.Tensor,
        kv_cache_manager,
        layer_idx: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
        decode_query_len: int,
        max_seq_len: int,
        max_num_requests: int,
    ) -> None:
        # The multi-CTA KV counters are sized for max_num_requests: more requests overrun them.
        assert seq_lens.shape[0] <= max_num_requests, "more requests than the counters cover"
        assert bool((seq_lens <= max_seq_len).all()), "sequence length exceeds its bound"
        # Each verify row attends a causal prefix of at least one key.
        assert bool((seq_lens >= decode_query_len).all()), "a request shorter than its query rows"


m3_dense_decode = M3DenseDecode()
