# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A non-causal DFlash window on TRTLLM: causal paged decode + block_upper_triangle_fixup."""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dflash import block_upper_triangle_fixup

BLOCK = 8


def _window_attention(q, k, v, ctx_lens, window):
    """fp64 reference: each block query attends to the keys within ``window`` of it."""
    outs = []
    num_kv_heads, group = k.shape[2], q.shape[2] // k.shape[2]
    for b, ctx in enumerate(ctx_lens):
        qb = q[b].double().unflatten(1, (num_kv_heads, group))
        kb, vb = k[b, : ctx + BLOCK].double(), v[b, : ctx + BLOCK].double()
        scores = torch.einsum("ihgd,khd->ihgk", qb, kb) * q.shape[-1] ** -0.5
        pos = torch.arange(ctx + BLOCK, device=q.device)
        allowed = (pos - pos[ctx:, None]).abs() <= window
        scores = scores.masked_fill(~allowed[:, None, None], float("-inf"))
        outs.append(torch.einsum("ihgk,khd->ihgd", scores.softmax(-1), vb))
    return torch.stack(outs)


def _case(ctx_lens, num_heads, num_kv_heads, head_dim):
    gen = torch.Generator(device="cuda").manual_seed(sum(ctx_lens))
    longest = max(ctx_lens) + BLOCK
    k, v = (
        torch.randn(len(ctx_lens), longest, num_kv_heads, head_dim, generator=gen, device="cuda")
        for _ in range(2)
    )
    q = 2 * torch.randn(len(ctx_lens), BLOCK, num_heads, head_dim, generator=gen, device="cuda")
    idx = torch.tensor(ctx_lens, device="cuda")[:, None] + torch.arange(BLOCK, device="cuda")
    rows = torch.arange(len(ctx_lens), device="cuda")[:, None]
    return q, k, v, k[rows, idx], v[rows, idx]


def _trtllm_gen_unavailable():
    if not torch.cuda.is_available():
        return "needs CUDA"
    from tensorrt_llm._torch.speculative.dflash_attention import (
        dflash_trtllm_gen_unavailability_reason,
    )

    return dflash_trtllm_gen_unavailability_reason()


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim",
    [(16, 2, 128), (32, 4, 64)],
    ids=["radixark_tp4", "nvidia_tp4"],
)
def test_trtllm_window_matches_flash_attn(num_heads, num_kv_heads, head_dim):
    """Muse Spark Open drafter geometry, window 2048: TRTLLM == VANILLA's flash-attn."""
    if (reason := _trtllm_gen_unavailable()) is not None:
        pytest.skip(reason)
    from flash_attn import flash_attn_with_kvcache

    from tensorrt_llm._torch.speculative.dflash_attention import get_dflash_trtllm_gen_ops

    left, page_size = 2047, 32
    ctx_lens = [1, 100, 2047, 2048, 5000]
    q, k, v, k_block, v_block = _case(ctx_lens, num_heads, num_kv_heads, head_dim)
    q, k, v, k_block, v_block = (t.bfloat16() for t in (q, k, v, k_block, v_block))
    seq_lens = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda") + BLOCK
    # The paged cache, with each request's pages scattered through the pool.
    pages_per_seq = k.shape[1] // page_size + 1
    num_pages = len(ctx_lens) * pages_per_seq
    table = torch.randperm(num_pages, device="cuda").view(len(ctx_lens), -1).int()
    cache = torch.randn(num_pages, 2, num_kv_heads, page_size, head_dim, device="cuda").bfloat16()
    pos = torch.arange(k.shape[1], device="cuda")
    for half, src in enumerate((k, v)):
        cache[:, half].transpose(1, 2)[table[:, pos // page_size].long(), pos % page_size] = src

    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:3], dtype=torch.float32, device="cuda")
    get_dflash_trtllm_gen_ops().batch_decode_with_kv_cache(
        query=q.flatten(0, 1),
        kv_cache=(cache[:, 0], cache[:, 1]),
        workspace_buffer=torch.zeros(256 << 20, dtype=torch.uint8, device="cuda"),
        block_tables=table,
        seq_lens=seq_lens,
        max_seq_len=pages_per_seq * page_size,
        bmm1_scale=head_dim**-0.5,
        window_left=left,
        out=out.flatten(0, 1),
        backend="trtllm-gen",
        q_len_per_req=BLOCK,
        lse=lse.flatten(0, 1),
        multi_ctas_kv_counter_buffer=torch.zeros(1 << 20, dtype=torch.uint8, device="cuda"),
    )
    grouped = (num_kv_heads, num_heads // num_kv_heads)
    later = torch.ones(BLOCK, BLOCK, dtype=torch.bool, device="cuda").triu(1)
    got = block_upper_triangle_fixup(
        out.unflatten(2, grouped),
        lse.unflatten(2, grouped),
        q.unflatten(2, grouped),
        k_block,
        v_block,
        head_dim**-0.5,
        later,
    ).flatten(2, 3)

    vanilla = flash_attn_with_kvcache(
        q, k, v, cache_seqlens=seq_lens, causal=False, window_size=(left, left)
    )
    want = _window_attention(q, k, v, ctx_lens, left).flatten(2, 3)
    trtllm_err = (got.double() - want).abs().max().item()
    vanilla_err = (vanilla.double() - want).abs().max().item()
    # bf16 outputs: both sit within a few ULPs of the fp64 reference.
    assert trtllm_err < 2 * vanilla_err + 4e-3, (trtllm_err, vanilla_err)
