# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical and phase regressions for fixed-reduction paged attention."""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.deterministic_attention import (
    deterministic_attention,
    split_workspace_bytes,
)

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
    # The probabilities are rounded to BF16 before the value product, as in
    # other Tensor Core attention kernels, so the error is that of BF16
    # attention rather than of one final rounding of an FP32 reduction.
    error = (decoded[0].float() - reference.float()).abs()
    assert error.mean() < 2e-3
    torch.testing.assert_close(decoded[0].float(), reference.float(), rtol=0.008, atol=0.02)

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


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("shared", [True, False])
def test_attention_all_rows_unaligned_chunks_and_graph_metadata(dim, shared):
    torch.manual_seed(812)
    length, heads, kh, page = 257, 6, 2, 32
    capacity = length + 64
    pages = math.ceil(capacity / page)
    k = torch.randn(pages, kh, page, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # Deliberately non-contiguous rows and heads, as in fused QKV views.
    q = torch.randn(2 * length, heads + 2, dim, device="cuda", dtype=torch.bfloat16)[::2, 1:-1]
    kp = torch.randperm(pages, device="cuda", dtype=torch.int32)
    table = kp[None] if shared else torch.stack([kp, kp.flip(0)])[None]

    def run(query, lens, tables, cu, fixed=0, maxq=length):
        result = torch.empty(query.shape, device="cuda", dtype=torch.bfloat16)
        deterministic_attention(
            query=query,
            key_cache=k,
            value_cache=v,
            block_tables=tables,
            seq_lens=lens,
            cu_query_lens=cu,
            fixed_query_len=fixed,
            max_query_len=maxq,
            scale=dim**-0.5,
            shared_page_indices=shared,
            output=result,
        )
        return result

    lens = torch.tensor([length], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    full = run(q, lens, table, cu)
    # Place every token at the first query row of a separate decode request.
    decoded = run(
        q,
        torch.arange(1, length + 1, device="cuda", dtype=torch.int32),
        table.expand(length, *table.shape[1:]),
        None,
        fixed=1,
        maxq=1,
    )
    torch.testing.assert_close(decoded, full, rtol=0, atol=0)
    start = 0
    for stop in (1, 3, 16, 33, 64, 97, 130, length):
        chunk = run(
            q[start:stop],
            torch.tensor([stop], device="cuda", dtype=torch.int32),
            table,
            torch.tensor([0, stop - start], device="cuda", dtype=torch.int32),
            maxq=stop - start,
        )
        torch.testing.assert_close(chunk, full[start:stop], rtol=0, atol=0)
        start = stop
    run(q, lens, table, cu)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(q, lens, table, cu)
    q.mul_(0.75)
    lens.add_(17)
    table.copy_(table.roll(1, dims=-1))
    graph.replay()
    torch.testing.assert_close(captured, run(q, lens, table, cu), rtol=0, atol=0)


# 32 query heads per KV head is the largest supported group; decode then runs
# 32 rows per program on eight warps instead of four.
@pytest.mark.parametrize("heads_kv", [(6, 2), (32, 1)], ids=["6x2", "32x1"])
@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("shared", [True, False])
def test_attention_key_segments_and_parallel_decode(dim, shared, heads_kv):
    """Rows whose keys span several segments agree in prefill, serial and parallel decode."""
    torch.manual_seed(5)
    (heads, kv_heads), length, page = heads_kv, 2500, 32
    pages = math.ceil(length / page)
    keys = torch.randn(pages, kv_heads, page, dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    queries = torch.randn(length, heads, dim, device="cuda", dtype=torch.bfloat16)
    kp = torch.randperm(pages, device="cuda", dtype=torch.int32)
    table = kp[None] if shared else torch.stack([kp, kp.flip(0)])[None]

    def run(q, lens, tables, cu=None, fixed=0, maxq=1, workspace=None):
        out = torch.empty_like(q)
        deterministic_attention(
            query=q,
            key_cache=keys,
            value_cache=values,
            block_tables=tables,
            seq_lens=lens,
            cu_query_lens=cu,
            fixed_query_len=fixed,
            max_query_len=maxq,
            scale=dim**-0.5,
            shared_page_indices=shared,
            output=out,
            workspace=workspace,
        )
        return out

    full = run(
        queries,
        torch.tensor([length], device="cuda", dtype=torch.int32),
        table,
        torch.tensor([0, length], device="cuda", dtype=torch.int32),
        maxq=length,
    )
    # First, last and boundary positions of each 1024-key segment.
    positions = torch.tensor([0, 1022, 1023, 1024, 1025, 2047, 2048, 2499], device="cuda")
    decode_q = queries[positions].contiguous()
    lens = (positions + 1).int()
    tables = table.expand(len(positions), *table.shape[1:]).contiguous()
    serial = run(decode_q, lens, tables, fixed=1)
    torch.testing.assert_close(serial, full[positions], rtol=0, atol=0)

    workspace = torch.empty(
        split_workspace_bytes(len(positions), heads, kv_heads, dim, pages * page),
        device="cuda",
        dtype=torch.uint8,
    )
    parallel = run(decode_q, lens, tables, fixed=1, workspace=workspace)
    torch.testing.assert_close(parallel, serial, rtol=0, atol=0)
    # A workspace that is too small keeps the serial layout.
    small = run(decode_q, lens, tables, fixed=1, workspace=workspace[:64])
    torch.testing.assert_close(small, serial, rtol=0, atol=0)

    # A captured parallel decode follows lengths that grow across a segment boundary.
    lens = torch.tensor([1020], device="cuda", dtype=torch.int32)
    one_q = queries[1019:1020].contiguous()
    run(one_q, lens, table, fixed=1, workspace=workspace)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(one_q, lens, table, fixed=1, workspace=workspace)
    for position in (1023, 1024, 2499):
        one_q.copy_(queries[position : position + 1])
        lens.fill_(position + 1)
        graph.replay()
        torch.testing.assert_close(captured[0], full[position], rtol=0, atol=0)
