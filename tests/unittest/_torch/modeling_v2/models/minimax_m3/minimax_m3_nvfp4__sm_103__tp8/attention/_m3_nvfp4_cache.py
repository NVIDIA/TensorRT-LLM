# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A MiniMax-M3 TP8 sparse-layer decode step over a packed NVFP4 cache, for the attention entries' tests.

The cache is written by the in-tree scatter (``fused_write_layer_caches_nvfp4``,
which quantizes with ``torch.ops.trtllm.fp4_quantize``), into one
``[pages, 2 (K, V), 1, 128, .]`` pool per role exactly as ``msa_paged_kv`` and
``get_block_scale_buffers`` hand the K and V views over. Pages are shuffled so
a kernel that ignored the block table would read the wrong ones.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_scatter import (
    fused_write_layer_caches_nvfp4,
)

#: Query heads per rank at TP8 (on one KV head).
NUM_HEADS = 8
HEAD_DIM = 128
PAGE = 128
TOPK = 16


def token_kv_len(seq_len: int, decode_query_len: int, offset: int) -> int:
    """The KV extent query token ``offset`` of a request attends, as every decode kernel derives it."""
    return max(seq_len - decode_query_len + offset + 1, 0)


def build_step(
    seq_lens: Sequence[int], decode_query_len: int, *, identity: bool, seed: int
) -> SimpleNamespace:
    """One decode step: E4M3 q [T, 8, 128], the packed NVFP4 K/V views and their scales, a [1, T, 16]
    head-major top-k table (the identity table, or a random ascending selection with a -1 tail), the block
    table and per-request lengths."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    total_q = batch * decode_query_len
    max_blocks = max(1, max((s + PAGE - 1) // PAGE for s in seq_lens))
    num_pages = batch * max_blocks
    num_slots = num_pages * PAGE
    block_table = (
        torch.randperm(num_pages, device="cuda", generator=g)
        .to(torch.int32)
        .reshape(batch, max_blocks)
    )
    data_pool = torch.zeros(num_pages, 2, 1, PAGE, HEAD_DIM // 2, dtype=torch.uint8, device="cuda")
    scale_pool = torch.zeros(
        num_pages, 2, 1, PAGE, HEAD_DIM // 16, dtype=torch.uint8, device="cuda"
    )
    quant = torch.ones(3, dtype=torch.float32, device="cuda")
    values = []
    for role in (0, 1):
        x = torch.randn(num_slots, HEAD_DIM, device="cuda", generator=g, dtype=torch.float32).to(
            torch.bfloat16
        )
        quant[role + 1] = (448.0 * 6.0) / x.abs().max().float().item()
        values.append(x)
    assert fused_write_layer_caches_nvfp4(
        data_pool[:, 0],
        data_pool[:, 1],
        scale_pool[:, 0],
        scale_pool[:, 1],
        None,
        torch.arange(num_slots, device="cuda", dtype=torch.int32),
        values[0],
        values[1],
        None,
        quant,
    )
    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, device="cuda", generator=g).to(
        torch.float8_e4m3fn
    )
    topk = torch.full((1, total_q, TOPK), -1, dtype=torch.int32, device="cuda")
    for tok in range(total_q):
        req, offset = divmod(tok, decode_query_len)
        nblocks = (token_kv_len(seq_lens[req], decode_query_len, offset) + PAGE - 1) // PAGE
        if identity:
            assert nblocks <= TOPK, "an identity step has every request within 16 blocks"
            chosen = torch.arange(nblocks, device="cuda")
        else:
            chosen = torch.randperm(nblocks, device="cuda", generator=g)[:TOPK].sort().values
        topk[0, tok, : chosen.numel()] = chosen.to(torch.int32)
    return SimpleNamespace(
        q=q,
        k_paged=data_pool[:, 0],
        v_paged=data_pool[:, 1],
        k_block_scale=scale_pool[:, 0],
        v_block_scale=scale_pool[:, 1],
        k_global_scale=(1.0 / quant[1:2]).contiguous(),
        v_global_scale=(1.0 / quant[2:3]).contiguous(),
        topk=topk,
        block_table=block_table,
        seq_lens=torch.tensor(list(seq_lens), dtype=torch.int32, device="cuda"),
        decode_query_len=decode_query_len,
        sm_scale=HEAD_DIM**-0.5,
        source=values,
        quant=quant,
        num_slots=num_slots,
    )
