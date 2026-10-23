# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical and phase regressions for fixed-reduction paged attention."""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.deterministic_attention import deterministic_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def test_attention_ignores_cuda_graph_padding_pages():
    keys = torch.randn(3, 1, 32, 64, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    query = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
    pages = torch.tensor([[-1, -1, -1], [1, 0, 2]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 65], device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)
    deterministic_attention(
        query=query,
        key_cache=keys,
        value_cache=values,
        block_tables=pages,
        seq_lens=lengths,
        cu_query_lens=None,
        fixed_query_len=1,
        max_query_len=1,
        scale=0.125,
        shared_page_indices=True,
        output=output,
    )
    expected = torch.empty_like(query[1:])
    deterministic_attention(
        query=query[1:],
        key_cache=keys,
        value_cache=values,
        block_tables=pages[1:],
        seq_lens=lengths[1:],
        cu_query_lens=None,
        fixed_query_len=1,
        max_query_len=1,
        scale=0.125,
        shared_page_indices=True,
        output=expected,
    )
    assert torch.isfinite(output).all()
    assert not output[0].count_nonzero()
    torch.testing.assert_close(output[1:], expected, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("length", [33, 129, 513])
@pytest.mark.parametrize("shared", [True, False])
def test_attention_prefill_decode_and_ragged_batch(dim, length, shared):
    torch.manual_seed(11)
    heads, kv_heads, page = 6, 2, 32
    total = length + 8
    pages = math.ceil(total / page)
    keys = torch.randn(3 * pages, kv_heads, page, dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    table = torch.randperm(pages, device="cuda", dtype=torch.int32)
    v_table = table if shared else table.flip(0)
    tables = table[None] if shared else torch.stack((table, v_table))[None]
    queries = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)

    def run(q, kv_lengths, page_tables, cu_q=None, fixed_q=0, max_q=1):
        out = torch.empty_like(q)
        deterministic_attention(
            query=q,
            key_cache=keys,
            value_cache=values,
            block_tables=page_tables,
            seq_lens=kv_lengths,
            cu_query_lens=cu_q,
            fixed_query_len=fixed_q,
            max_query_len=max_q,
            scale=dim**-0.5,
            shared_page_indices=shared,
            output=out,
        )
        return out

    full = run(
        queries,
        torch.tensor([total], device="cuda", dtype=torch.int32),
        tables,
        torch.tensor([0, total], device="cuda", dtype=torch.int32),
        max_q=total,
    )
    target_q = queries[length - 1 : length].contiguous()
    lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
    decoded = run(target_q, lengths, tables, fixed_q=1)
    torch.testing.assert_close(decoded[0], full[length - 1], rtol=0, atol=0)

    # FP64 reference uses dense logical K/V, independently of paging and tiles.
    k = keys[table.long()].transpose(1, 2).reshape(-1, kv_heads, dim)[:length].double()
    v = values[v_table.long()].transpose(1, 2).reshape(-1, kv_heads, dim)[:length].double()
    k = k.repeat_interleave(heads // kv_heads, dim=1)
    v = v.repeat_interleave(heads // kv_heads, dim=1)
    logits = torch.einsum("hd,thd->ht", target_q[0].double(), k) * dim**-0.5
    reference = torch.einsum("ht,thd->hd", logits.softmax(-1), v).bfloat16()
    torch.testing.assert_close(decoded[0].float(), reference.float(), rtol=0.008, atol=0.0001)

    batched_q = torch.cat((target_q, queries[:5])).contiguous()
    batch_tables = tables.expand(3, *tables.shape[1:]).contiguous()
    ragged = run(
        batched_q,
        torch.tensor([length, 31, 17], device="cuda", dtype=torch.int32),
        batch_tables,
        torch.tensor([0, 1, 4, 6], device="cuda", dtype=torch.int32),
        max_q=3,
    )
    torch.testing.assert_close(ragged[0], decoded[0], rtol=0, atol=0)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(target_q, lengths, tables, fixed_q=1)
    target_q.mul_(0.5)
    expected = run(target_q, lengths, tables, fixed_q=1)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
