# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A MiniMax-M3.1-Flash sm_107 sparse-layer decode step over a packed NVFP4 cache, for the targets' attention entries'
tests, at a target's per-rank geometry (tp2: 32 query heads over 2 KV heads; tp4: 16 over 1; tp8: 8 over 1).

The cache is written by the in-tree scatter (``fused_write_layer_caches_nvfp4``,
which quantizes with ``torch.ops.trtllm.fp4_quantize``), into one
``[pages, 2 (K, V), KV heads, 128, .]`` pool per role exactly as ``msa_paged_kv`` and
``get_block_scale_buffers`` hand the K and V views over. Pages are shuffled so
a kernel that ignored the block table would read the wrong ones.

Also the calls the targets' ``m3_sparse_decode`` tests share: the entry's call on a step, the stock in-tree decode on
it, and the identity selection staged from the lengths.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_scatter import (
    fused_write_layer_caches_nvfp4,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode import (
    minimax_m3_sparse_attn_decode as stock_decode,
)

HEAD_DIM = 128
PAGE = 128
TOPK = 16
# Target -> (query heads, KV heads) per rank.
GEOMETRY = {
    "minimax_m3_1_flash_nvfp4__sm_107__tp2": (32, 2),
    "minimax_m3_1_flash_nvfp4__sm_107__tp4": (16, 1),
    "minimax_m3_1_flash_nvfp4__sm_107__tp8": (8, 1),
}
# Each crosses a 128-token edge within its four verify tokens or between the stale and the true lengths.
EDGE_LENS = (1284, 1287, 1281, 129, 132, 2048, 1028, 5, 643, 1920, 256, 1)


def token_kv_len(seq_len: int, decode_query_len: int, offset: int) -> int:
    """The KV extent query token ``offset`` of a request attends, as every decode kernel derives it."""
    return max(seq_len - decode_query_len + offset + 1, 0)


def build_step(
    seq_lens: Sequence[int],
    decode_query_len: int,
    *,
    identity: bool,
    seed: int,
    target: str,
) -> SimpleNamespace:
    """One decode step at ``target``'s geometry: E4M3 q [T, heads, 128], the packed NVFP4 K/V views and their scales,
    a [KV heads, T, 16] head-major top-k table (the identity table, or a random ascending selection with a -1 tail per
    KV head), the block table and per-request lengths."""
    num_heads, num_kv_heads = GEOMETRY[target]
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
    data_pool = torch.zeros(
        num_pages, 2, num_kv_heads, PAGE, HEAD_DIM // 2, dtype=torch.uint8, device="cuda"
    )
    scale_pool = torch.zeros(
        num_pages, 2, num_kv_heads, PAGE, HEAD_DIM // 16, dtype=torch.uint8, device="cuda"
    )
    quant = torch.ones(3, dtype=torch.float32, device="cuda")
    values = []
    for role in (0, 1):
        x = torch.randn(
            num_slots, num_kv_heads * HEAD_DIM, device="cuda", generator=g, dtype=torch.float32
        ).to(torch.bfloat16)
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
    q = torch.randn(total_q, num_heads, HEAD_DIM, device="cuda", generator=g).to(
        torch.float8_e4m3fn
    )
    topk = torch.full((num_kv_heads, total_q, TOPK), -1, dtype=torch.int32, device="cuda")
    for tok in range(total_q):
        req, offset = divmod(tok, decode_query_len)
        nblocks = (token_kv_len(seq_lens[req], decode_query_len, offset) + PAGE - 1) // PAGE
        for head in range(num_kv_heads):
            if identity:
                assert nblocks <= TOPK, "an identity step has every request within 16 blocks"
                chosen = torch.arange(nblocks, device="cuda")
            else:
                chosen = torch.randperm(nblocks, device="cuda", generator=g)[:TOPK].sort().values
            topk[head, tok, : chosen.numel()] = chosen.to(torch.int32)
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


def empty_output(case) -> torch.Tensor:
    """A [T, heads, 128] BF16 output for ``case``."""
    return torch.empty(*case.q.shape[:2], HEAD_DIM, dtype=torch.bfloat16, device="cuda")


def sparse_decode_args(case, output, identity: bool, topk=None) -> tuple:
    """The m3_sparse_decode entry's (and its reference's) arguments for ``case`` (its own table unless ``topk``)."""
    return (
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk if topk is None else topk,
        case.block_table,
        case.seq_lens,
        case.sm_scale,
        output,
        case.decode_query_len,
        None,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        None,
        identity,
    )


def stock(case, output) -> None:
    """The stock in-tree MSA decode on ``case``, into ``output``."""
    stock_decode(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk,
        case.block_table,
        case.seq_lens,
        sm_scale=case.sm_scale,
        output=output,
        decode_query_len=case.decode_query_len,
        k_block_scale=case.k_block_scale,
        v_block_scale=case.v_block_scale,
        k_global_scale=case.k_global_scale,
        v_global_scale=case.v_global_scale,
    )


def identity_table(
    seq_lens: torch.Tensor, decode_query_len: int, heads: int, lag: int = 0
) -> torch.Tensor:
    """The identity selection of every verify token, staged from lengths ``lag`` tokens behind seq_lens."""
    seq = (seq_lens.cpu().to(torch.int64) - lag).clamp(min=0).repeat_interleave(decode_query_len)
    intra = torch.arange(seq.shape[0]) % decode_query_len
    n_valid = ((seq - decode_query_len + intra + 1).clamp(min=0) + 127) // 128
    ar = torch.arange(16)
    table = torch.where(ar[None, :] < n_valid[:, None], ar[None, :], -1).to(torch.int32)
    return table.unsqueeze(0).expand(heads, -1, -1).contiguous().cuda()


def staged_table(case, fault: str, seed: int) -> torch.Tensor:
    """A staged identity table that is wrong in the way ``fault`` names: one verify step behind the lengths
    (``stale_one_step``), all -1 (``all_minus_one``) or garbage ids (``random_ids``)."""
    heads, dql = case.topk.shape[0], case.decode_query_len
    if fault == "stale_one_step":
        return identity_table(case.seq_lens, dql, heads, lag=dql)
    true_table = identity_table(case.seq_lens, dql, heads)
    if fault == "all_minus_one":
        return torch.full_like(true_table, -1)
    assert fault == "random_ids", fault
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(-1, case.block_table.shape[1], true_table.shape, generator=g)
        .to(torch.int32)
        .cuda()
    )
