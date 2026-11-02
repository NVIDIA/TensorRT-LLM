# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare GLM k-pool kernels against PyTorch references using synthetic paged caches."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.kernels import (
    append_paged_state_rows,
    compact_sparse_rows,
    kpool_expand,
    kpool_score,
    kpool_score_fp8,
    kpool_update,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd

    HAS_FLASH_MLA = flash_mla_sparse_fwd is not None
except ImportError:
    HAS_FLASH_MLA = False

HD = 128
KPOOL = 4
TPB = 8
FP32_MIN = torch.finfo(torch.float32).min


def _strided_index_pool(slots: int, gen: torch.Generator) -> torch.Tensor:
    """[slots, TPB, 3 * HD] bf16 view with a padded row stride, like the
    coalesced V2 pool the backend reads (the kernels take explicit strides)."""
    storage = torch.randn(slots, TPB, 3 * HD + 64, generator=gen, device="cuda")
    return storage.to(torch.bfloat16)[..., : 3 * HD]


def _paged_tables(num_tables: int, pages: int, slots: int, gen: torch.Generator) -> torch.Tensor:
    """Disjoint, shuffled slot ids per table (gaps between a request's pages)."""
    perm = torch.randperm(slots, generator=gen, device="cuda")[: num_tables * pages]
    return perm.view(num_tables, pages).to(torch.int64)


def _gather_rows(pool: torch.Tensor, table: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    page = positions // TPB
    return pool[table[page], positions - page * TPB]


def _reference_pool_key(pool, table, pos, ape):
    """build_pools numerics for the pool containing pos: fp32 softmax
    over gate + ape with invisible members masked, bf16 probabilities and
    products, fp32 sum, bf16 result."""
    start = pos // KPOOL * KPOOL
    members = start + torch.arange(KPOOL, device="cuda")
    valid = members <= pos
    rows = _gather_rows(pool, table, members)
    k, g = rows[:, :HD].float(), rows[:, HD : 2 * HD].float()
    logits = (g + ape.float()).masked_fill(~valid[:, None], float("-inf"))
    probs = logits.softmax(dim=0).to(torch.bfloat16).float()
    prod = (probs * k).to(torch.bfloat16).float()
    return prod.sum(dim=0).to(torch.bfloat16)


@pytest.mark.parametrize("packed_rows", [False, True], ids=["one_row_per_table", "request_ids"])
def test_pool_key_refresh_matches_build_pools_numerics(packed_rows):
    gen = torch.Generator(device="cuda").manual_seed(0)
    slots, pages = 24, 3
    pool = _strided_index_pool(slots, gen)
    before = pool.clone()
    ape = (torch.randn(KPOOL, HD, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    if packed_rows:
        # Two requests, several rows each: rows of one request share a table.
        tables = _paged_tables(2, pages, slots, gen)
        request_ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device="cuda")
        # Distinct pools per request (the backend's contract: one program per
        # pool, so no two rows race on one pool key); mid-pool rows exercise
        # the invisible-member mask, 6 / 22 the page-crossing addressing.
        positions = torch.tensor([1, 7, 9, 6, 22], dtype=torch.int64, device="cuda")
    else:
        tables = _paged_tables(5, pages, slots, gen)
        request_ids = None
        positions = torch.tensor([0, 3, 10, 15, 23], dtype=torch.int64, device="cuda")

    kpool_update(
        pool, tables, positions, ape, TPB, head_dim=HD, kpool=KPOOL, request_ids=request_ids
    )
    torch.cuda.synchronize()

    touched = []
    for row, pos in enumerate(positions.tolist()):
        table = tables[request_ids[row].item() if packed_rows else row]
        start = pos // KPOOL * KPOOL
        slot, off = table[start // TPB].item(), start % TPB
        touched.append((slot, off))
        got = pool[slot, off, 2 * HD :]
        want = _reference_pool_key(before, table, pos, ape)
        torch.testing.assert_close(got.float(), want.float(), atol=1e-2, rtol=1e-2)
        # Only the pool-key columns of the first member's row are written.
        assert torch.equal(pool[slot, off, : 2 * HD], before[slot, off, : 2 * HD])
    mask = torch.ones(slots, TPB, dtype=torch.bool, device="cuda")
    for slot, off in touched:
        mask[slot, off] = False
    assert torch.equal(pool[mask], before[mask])


def _reference_scores(q, w, pool, tables, kv_lens, *, num_pools_max, q_scale, w_scale):
    n = q.shape[0]
    out = torch.full((n, num_pools_max), FP32_MIN, device="cuda")
    for r in range(n):
        num_pools = kv_lens[r].item() // KPOOL
        if num_pools == 0:
            continue
        first = torch.arange(num_pools, device="cuda") * KPOOL
        keys = _gather_rows(pool, tables[r], first)[:, 2 * HD :].float()  # [P, HD]
        scores = torch.relu(q[r].float() @ keys.T * q_scale)  # [H, P]
        out[r, :num_pools] = (scores * (w[r].float() * w_scale)[:, None]).sum(0)
    return out


@pytest.mark.parametrize("precision", ["ieee", "tf32", "bf16"])
def test_pool_scores_generation_rows(precision):
    gen = torch.Generator(device="cuda").manual_seed(1)
    n, heads, slots, pages = 6, 4, 64, 8
    pool = _strided_index_pool(slots, gen)
    tables = _paged_tables(n, pages, slots, gen)
    q = (torch.randn(n, heads, HD, generator=gen, device="cuda")).to(torch.bfloat16)
    w = torch.rand(n, heads, generator=gen, device="cuda").to(torch.bfloat16)
    # 0 complete pools, partial, exactly full pages, and the capacity edge.
    kv_lens = torch.tensor([2, 5, 16, 33, 60, 64], dtype=torch.int64, device="cuda")
    num_pools_max = 100  # > 64: two BP blocks, second one entirely invisible
    kwargs = dict(num_pools_max=num_pools_max, q_scale=HD**-0.5, w_scale=0.25)

    got = kpool_score(
        q, w, pool, tables, kv_lens, TPB, head_dim=HD, kpool=KPOOL, precision=precision, **kwargs
    )
    want = _reference_scores(q, w, pool, tables, kv_lens, **kwargs)
    visible = want > FP32_MIN
    assert torch.equal(got <= FP32_MIN, ~visible)
    # bf16 x bf16 products are exact in fp32, so every precision is an fp32
    # accumulation of exact products.
    torch.testing.assert_close(got[visible], want[visible], atol=2e-3, rtol=2e-3)
    if precision == "bf16":
        # FP32 queries are not exact in bf16, so they keep the tf32 dot.
        q32 = torch.randn(n, heads, HD, generator=gen, device="cuda")
        args = (q32, w, pool, tables, kv_lens, TPB)
        assert torch.equal(
            kpool_score(*args, head_dim=HD, kpool=KPOOL, precision="bf16", **kwargs),
            kpool_score(*args, head_dim=HD, kpool=KPOOL, precision="tf32", **kwargs),
        )


@pytest.mark.parametrize("layout", ["generation", "prefill_shared", "prefill_boundary"])
@pytest.mark.parametrize("weight_head_stride", [1, 2])
def test_pool_scores_strided_head_weights(layout, weight_head_stride):
    gen = torch.Generator(device="cuda").manual_seed(23)
    n, heads, num_pools_max = 4, 32, 520
    pages = num_pools_max * KPOOL // TPB
    num_tables = n if layout == "generation" else 2 if layout == "prefill_boundary" else 1
    slots = num_tables * pages + 8
    pool = _strided_index_pool(slots, gen)
    tables = _paged_tables(num_tables, pages, slots, gen)
    request_ids = None
    if layout != "generation":
        request_ids = torch.tensor(
            [0, 0, 1, 1] if layout == "prefill_boundary" else [0, 0, 0, 0],
            device="cuda",
            dtype=torch.int32,
        )
    q = torch.randn(n, heads, HD, generator=gen, device="cuda").to(torch.bfloat16)
    # Match project_state's [key | gate | head weights] slice: [T, 32]
    # with row stride 288. Also exercise a non-unit head stride.
    projection = torch.rand(
        n, 2 * HD + heads * weight_head_stride, generator=gen, device="cuda"
    ).to(torch.bfloat16)
    weights = projection[:, 2 * HD :: weight_head_stride]
    assert weights.stride() == (2 * HD + heads * weight_head_stride, weight_head_stride)
    assert not weights.is_contiguous()
    # More than 512 complete pools forces top-k to discard candidates.
    kv_lens = torch.tensor([2052, 2053, 2056, 2057], device="cuda", dtype=torch.int64)
    kwargs = dict(num_pools_max=num_pools_max, q_scale=HD**-0.5, w_scale=heads**-0.5)
    got = kpool_score(
        q,
        weights,
        pool,
        tables,
        kv_lens,
        TPB,
        head_dim=HD,
        kpool=KPOOL,
        rows_per_program=1 if request_ids is None else 16,
        request_ids=request_ids,
        precision="tf32",
        **kwargs,
    )
    reference_tables = tables if request_ids is None else tables[request_ids.long()]
    want = _reference_scores(q, weights, pool, reference_tables, kv_lens, **kwargs)
    visible = want > FP32_MIN
    assert torch.equal(got <= FP32_MIN, ~visible)
    torch.testing.assert_close(got[visible], want[visible], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        got.topk(512, dim=-1).indices.sort(dim=-1).values,
        want.topk(512, dim=-1).indices.sort(dim=-1).values,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("bounded", [False, True], ids=["full_capacity", "max_kv_len"])
def test_pool_scores_packed_context_rows_share_tables(bounded, monkeypatch):
    """rows_per_program=16 with request_ids: the packed query tokens
    of several requests, in position order, including a group straddling two
    requests (per-row gather branch). With max_kv_len, only pools some row can
    see are launched; columns past the bound are left unwritten."""
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import kernels

    gen = torch.Generator(device="cuda").manual_seed(2)
    heads, num_pools_max = 3, 520
    pages = num_pools_max * KPOOL // TPB
    pool = _strided_index_pool(2 * pages + 8, gen)
    tables = _paged_tables(2, pages, 2 * pages + 8, gen)
    # Request 0: 21 query tokens after 250 cached ones, request 1: 19 (0..18).
    lens, cached = [21, 19], [250, 0]
    request_ids = torch.cat(
        [torch.full((length,), i, dtype=torch.int32) for i, length in enumerate(lens)]
    ).cuda()
    kv_lens = torch.cat([torch.arange(c + 1, c + n + 1) for c, n in zip(cached, lens)]).cuda()
    n = kv_lens.shape[0]
    q = torch.randn(n, heads, HD, generator=gen, device="cuda").to(torch.bfloat16)
    w = torch.rand(n, heads, generator=gen, device="cuda").to(torch.bfloat16)
    kwargs = dict(num_pools_max=num_pools_max, q_scale=HD**-0.5, w_scale=1.0)

    def score(rows, max_kv_len):
        return kpool_score(
            q[rows],
            w[rows],
            pool,
            tables,
            kv_lens[rows],
            TPB,
            head_dim=HD,
            kpool=KPOOL,
            rows_per_program=16,
            request_ids=request_ids[rows],
            max_kv_len=max_kv_len,
            **kwargs,
        )

    got = score(slice(None), int(kv_lens.max()) if bounded else None)
    want = _reference_scores(q, w, pool, tables[request_ids.long()], kv_lens, **kwargs)
    visible = want > FP32_MIN
    torch.testing.assert_close(got[visible], want[visible], atol=2e-3, rtol=2e-3)
    if not bounded:
        assert torch.equal(got <= FP32_MIN, ~visible)
        return
    # The indexer's pool top-k scans kv_len // kpool columns, so poisoning the
    # rest changes no selection (rows without a complete pool select nothing).
    indexer, num_cand = _meta_indexer(), (kv_lens // KPOOL).to(torch.int32)

    def select(scores):
        out = torch.empty(n, indexer.select_k, dtype=torch.int32, device="cuda")
        indexer.pool_top_k(
            scores, out, is_prefill=False, sequence_lengths=num_cand, scan_lengths=num_cand
        )
        return out.sort(dim=-1).values

    selected = select(got.masked_fill(~visible, float("inf")))
    assert torch.equal(selected, select(want))
    assert (selected[num_cand == 0] == -1).all()
    # The first rows of request 1 complete no pool, so nothing is launched.
    monkeypatch.setattr(kernels, "_kpool_score_kernel", None)
    score(slice(lens[0], lens[0] + KPOOL - 1), KPOOL - 1)


def test_verify_rows_index_tables_like_repeated_tables():
    """Verification rows addressing their tables by request id score and expand
    exactly like rows reading per-row copies of the generation tables."""
    gen = torch.Generator(device="cuda").manual_seed(4)
    requests, tokens, heads, pages = 3, 4, 8, 16
    slots = (requests + 1) * pages + 8
    pool = _strided_index_pool(slots, gen)
    # Row 0 is a context request's table; the generation tables follow it.
    tables = _paged_tables(requests + 1, pages, slots, gen)
    repeated = tables[1:].repeat_interleave(tokens, dim=0)
    request_ids = torch.arange(1, requests + 1, dtype=torch.int32, device="cuda")
    request_ids = request_ids.repeat_interleave(tokens)
    first = torch.randint(KPOOL, pages * TPB - tokens, (requests, 1), generator=gen, device="cuda")
    visible = (first + torch.arange(1, tokens + 1, device="cuda")).reshape(-1)
    n = requests * tokens
    q = torch.randn(n, heads, HD, generator=gen, device="cuda").to(torch.bfloat16)
    w = torch.rand(n, heads, generator=gen, device="cuda").to(torch.bfloat16)
    kwargs = dict(
        num_pools_max=pages * TPB // KPOOL, head_dim=HD, kpool=KPOOL, q_scale=HD**-0.5, w_scale=1.0
    )
    want = kpool_score(q, w, pool, repeated, visible, TPB, precision="bf16", **kwargs)
    got = kpool_score(
        q,
        w,
        pool,
        tables,
        visible,
        TPB,
        precision="bf16",
        request_ids=request_ids,
        **kwargs,
    )
    torch.testing.assert_close(got, want, atol=0, rtol=0)

    selected = torch.randint(-1, 32, (n, 8), generator=gen, device="cuda", dtype=torch.int32)
    expand = dict(base_row=2, rows_per_slot=TPB + 4, kpool=KPOOL, out_width=64)
    torch.testing.assert_close(
        kpool_expand(selected, visible, tables, TPB, request_ids=request_ids, **expand),
        kpool_expand(selected, visible, repeated, TPB, **expand),
        atol=0,
        rtol=0,
    )


def test_pool_scores_broadcast_table_single_context_request():
    """One context request: a stride-0 broadcast table with rows_per_program=16."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    heads, slots, pages, n = 4, 32, 4, 30
    pool = _strided_index_pool(slots, gen)
    table = _paged_tables(1, pages, slots, gen)
    tables = table.expand(n, pages)
    assert tables.stride(0) == 0
    kv_lens = torch.arange(1, n + 1, device="cuda")
    q = torch.randn(n, heads, HD, generator=gen, device="cuda").to(torch.bfloat16)
    w = torch.rand(n, heads, generator=gen, device="cuda").to(torch.bfloat16)
    kwargs = dict(num_pools_max=8, q_scale=HD**-0.5, w_scale=1.0)

    got = kpool_score(
        q, w, pool, tables, kv_lens, TPB, head_dim=HD, kpool=KPOOL, rows_per_program=16, **kwargs
    )
    want = _reference_scores(q, w, pool, tables, kv_lens, **kwargs)
    visible = want > FP32_MIN
    assert torch.equal(got <= FP32_MIN, ~visible)
    torch.testing.assert_close(got[visible], want[visible], atol=2e-3, rtol=2e-3)

    with pytest.raises(ValueError, match="rows_per_program"):
        kpool_score(
            q,
            w,
            pool,
            table.repeat(n, 1),
            kv_lens,
            TPB,
            head_dim=HD,
            kpool=KPOOL,
            rows_per_program=16,
            **kwargs,
        )


@pytest.mark.skipif(
    get_sm_version() not in (100, 103), reason="DeepGEMM MQA logits require SM100/103"
)
@pytest.mark.parametrize("single_request", [True, False], ids=["one_table", "two_tables"])
def test_fp8_pool_scores_track_bf16_selection(single_request):
    """FP8 context scoring keeps the BF16 scorer's top pools on packed context rows."""
    gen = torch.Generator(device="cuda").manual_seed(10)
    heads, tpb, slots = 32, 32, 400
    lens = [6000] if single_request else [6000, 2500]
    pages = [(length + tpb - 1) // tpb for length in lens]
    pool = torch.randn(slots, 2, tpb, 3 * HD, generator=gen, device="cuda").to(torch.bfloat16)[:, 1]
    perm = torch.randperm(slots, generator=gen, device="cuda")
    tables = torch.zeros(len(lens), max(pages), dtype=torch.int64, device="cuda")
    start = 0
    for i, count in enumerate(pages):
        tables[i, :count] = perm[start : start + count]
        start += count
    request_ids = torch.cat(
        [torch.full((length,), i, dtype=torch.int32) for i, length in enumerate(lens)]
    ).cuda()
    kv_lens = torch.cat([torch.arange(1, length + 1, dtype=torch.int32) for length in lens]).cuda()
    n = kv_lens.shape[0]
    q = torch.randn(n, heads, HD, generator=gen, device="cuda").to(torch.bfloat16)
    w = torch.randn(n, heads, generator=gen, device="cuda").to(torch.bfloat16)
    kwargs = dict(
        head_dim=HD,
        kpool=KPOOL,
        q_scale=HD**-0.5,
        w_scale=heads**-0.5,
        request_ids=request_ids,
        max_kv_len=max(lens),
    )
    want = kpool_score(
        q,
        w,
        pool,
        tables,
        kv_lens,
        tpb,
        num_pools_max=max(pages) * tpb // KPOOL,
        precision="bf16",
        rows_per_program=16,
        **kwargs,
    )
    got = kpool_score_fp8(q, w, pool, tables, kv_lens, tpb, **kwargs)
    assert got is not None and got.stride(1) == 1 and got.stride(0) % 4 == 0
    overlaps = []
    for row in torch.randint(0, n, (64,), generator=gen, device="cuda").tolist():
        visible = int(kv_lens[row]) // KPOOL
        if visible < 64:
            continue
        k = visible // 2
        a = set(want[row, :visible].topk(k).indices.tolist())
        b = set(got[row, :visible].topk(k).indices.tolist())
        overlaps.append(len(a & b) / k)
    assert overlaps and sum(overlaps) / len(overlaps) > 0.97 and min(overlaps) > 0.9


@pytest.mark.skipif(
    get_sm_version() not in (100, 103), reason="DeepGEMM MQA logits require SM100/103"
)
@pytest.mark.parametrize("lens", [[16384], [16000, 12000, 8000]], ids=["one-table", "three-tables"])
def test_fp8_pool_score_peak_memory_fits_the_planned_cost(lens):
    """kpool_score_fp8's peak allocation beyond its row-sized buffers fits the planner's cost."""
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import GlmKpoolSparseAttention

    gen = torch.Generator(device="cuda").manual_seed(13)
    heads, tpb, rows_per_table = 32, 32, 256
    pages = [(length + tpb - 1) // tpb for length in lens]
    pool = torch.randn(sum(pages) + 8, 2, tpb, 3 * HD, generator=gen, device="cuda")
    pool = pool.to(torch.bfloat16)[:, 1]
    tables = torch.zeros(len(lens), max(pages), dtype=torch.int64, device="cuda")
    start = 0
    for i, count in enumerate(pages):
        tables[i, :count] = torch.arange(start, start + count, device="cuda")
        start += count
    # The last rows_per_table queries of each prompt, so most pools are visible.
    request_ids = torch.cat(
        [torch.full((rows_per_table,), i, dtype=torch.int32) for i in range(len(lens))]
    ).cuda()
    kv_lens = torch.cat(
        [
            torch.arange(length - rows_per_table + 1, length + 1, dtype=torch.int32)
            for length in lens
        ]
    ).cuda()
    n = kv_lens.shape[0]
    q = torch.randn(n, heads, HD, generator=gen, device="cuda").to(torch.bfloat16)
    w = torch.randn(n, heads, generator=gen, device="cuda").to(torch.bfloat16)
    kwargs = dict(head_dim=HD, kpool=KPOOL, q_scale=HD**-0.5, w_scale=heads**-0.5)
    args = (q, w, pool, tables, kv_lens, tpb)
    kpool_score_fp8(*args, request_ids=request_ids, max_kv_len=max(lens), **kwargs)  # JIT warmup
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    scores = kpool_score_fp8(*args, request_ids=request_ids, max_kv_len=max(lens), **kwargs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del scores
    cost = GlmKpoolSparseAttention._context_score_cost(
        n, len(lens), max(lens) // KPOOL, fp8=True, head_dim=HD, capacity_pools=0
    )
    # Quantized queries (with the quantizer's scratch), head weights and per-row
    # indices grow only with rows; startup profiling measures them.
    row_bytes = n * heads * (2 * HD + 16) + 64 * n
    assert peak <= cost * 4 + row_bytes + (64 << 10)


@pytest.mark.skipif(
    get_sm_version() not in (100, 103), reason="DeepGEMM MQA logits require SM100/103"
)
def test_fp8_pool_scores_of_a_table_range_match_the_packed_call():
    """Scoring one request against its own table equals its rows of a multi-table call."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    heads, tpb, slots = 32, 32, 400
    lens = [3000, 2500, 1200]
    pages = [(length + tpb - 1) // tpb for length in lens]
    pool = torch.randn(slots, 2, tpb, 3 * HD, generator=gen, device="cuda").to(torch.bfloat16)[:, 1]
    perm = torch.randperm(slots, generator=gen, device="cuda")
    tables = torch.zeros(len(lens), max(pages), dtype=torch.int64, device="cuda")
    start = 0
    for i, count in enumerate(pages):
        tables[i, :count] = perm[start : start + count]
        start += count
    request_ids = torch.cat(
        [torch.full((length,), i, dtype=torch.int32) for i, length in enumerate(lens)]
    ).cuda()
    kv_lens = torch.cat([torch.arange(1, length + 1, dtype=torch.int32) for length in lens]).cuda()
    q = torch.randn(kv_lens.shape[0], heads, HD, generator=gen, device="cuda").to(torch.bfloat16)
    w = torch.randn(kv_lens.shape[0], heads, generator=gen, device="cuda").to(torch.bfloat16)
    kwargs = dict(head_dim=HD, kpool=KPOOL, q_scale=HD**-0.5, w_scale=heads**-0.5)
    packed = kpool_score_fp8(
        q, w, pool, tables, kv_lens, tpb, request_ids=request_ids, max_kv_len=max(lens), **kwargs
    )
    first = lens[0]
    rows = slice(first, first + lens[1])
    alone = kpool_score_fp8(
        q[rows],
        w[rows],
        pool,
        tables[1:2],
        kv_lens[rows],
        tpb,
        request_ids=request_ids[rows] - 1,
        max_kv_len=lens[1],
        **kwargs,
    )
    assert packed is not None and alone is not None
    for row in range(0, lens[1], 97):
        visible = (row + 1) // KPOOL
        torch.testing.assert_close(alone[row, :visible], packed[first + row, :visible])


def test_pool_expansion_graph_replay_tracks_request_tables_and_tails():
    # Requests occupy disjoint slots, with gaps between slots from coalesced buffers.
    tables = torch.tensor([[3, 1], [5, 2]], dtype=torch.int64, device="cuda")
    requests = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([3, 8, 9, 15], dtype=torch.int64, device="cuda")
    selected = torch.tensor([[0, -1], [1, 0], [1, 0], [2, 1]], dtype=torch.int32, device="cuda")

    def expand():
        return kpool_expand(
            selected,
            lengths,
            tables,
            8,
            base_row=2,
            rows_per_slot=24,
            kpool=4,
            out_width=64,
            request_ids=requests,
        )

    def reference():
        result = torch.full((4, 64), -1, dtype=torch.int32)
        for row, (request, length, pools) in enumerate(
            zip(requests.cpu().tolist(), lengths.cpu().tolist(), selected.cpu().tolist())
        ):
            positions = [
                pool * 4 + member if 0 <= pool < length // 4 else -1
                for pool in pools
                for member in range(4)
            ]
            positions += list(range(length // 4 * 4, length))
            for col, position in enumerate(positions):
                if position >= 0:
                    result[row, col] = 2 + int(tables[request, position // 8]) * 24 + position % 8
        return result.cuda()

    expand()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = expand()
    graph.replay()
    assert torch.equal(output, reference())
    lengths.copy_(torch.tensor([4, 7, 12, 16], device="cuda"))
    tables.copy_(tables.flip(0))
    graph.replay()
    assert torch.equal(output, reference())


@pytest.mark.skipif(not HAS_FLASH_MLA, reason="FlashMLA not available")
@pytest.mark.skipif(get_sm_version() < 90, reason="FlashMLA requires SM90 (Hopper) or later")
@pytest.mark.parametrize("heads", [16, 64])
def test_fp8_sparse_core_matches_dequantized_cache_and_replays_graph(heads):
    """Exercise the selected-row core used by production prefill, decode and verify."""
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import (
        GlmKpoolSparseAttention,
        latent_pool_rows,
    )

    backend = object.__new__(GlmKpoolSparseAttention)
    backend.kv_lora_rank = 512
    backend.softmax_scale = 256**-0.5
    backend.kv_scale_quant_orig = torch.tensor([0.5], device="cuda")
    backend._FP8_QUERY_CHUNK_SIZE = 2  # Exercise multiple chunks and a short final chunk.
    generator = torch.Generator(device="cuda").manual_seed(71)
    storage = torch.randn(12, 4, 512, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    pool = storage[1::3]  # Nonzero offset and coalesced page stride.
    rows, base, stride = latent_pool_rows(pool)
    q = torch.randn(5, heads, 512, device="cuda", generator=generator).to(torch.bfloat16)
    positions = torch.randint(0, 16, (5, 67), device="cuda", generator=generator)
    indices = (base + positions // 4 * stride + positions % 4).int()
    indices[:, -2] = -1
    indices[:, -1] = rows.shape[0]  # Preserve FlashMLA's out-of-range sentinel semantics.

    def reference():
        decoded = (rows.float() * backend.kv_scale_quant_orig).to(q.dtype)
        return backend._dispatch_sparse_core(q, decoded, indices)

    expected = reference()
    actual = backend._dispatch_sparse_core(q, rows, indices)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        backend._dispatch_sparse_core(q, rows, indices)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend._dispatch_sparse_core(q, rows, indices)
    storage.copy_((storage.float() * 0.5).to(storage.dtype))
    graph.replay()
    torch.testing.assert_close(captured, reference(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("context", [False, True])
def test_fp8_cache_storage_helpers_preserve_coalesced_pages(context):
    """Write FP8 payloads without disturbing coalesced pages or indexer pool keys."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import GlmKpoolSparseAttention

    backend = object.__new__(GlmKpoolSparseAttention)
    backend.kv_scale_orig_quant = torch.tensor([2.0], device="cuda")
    backend.kv_scale_quant_orig = torch.tensor([0.5], device="cuda")
    backend.sparse_params = SimpleNamespace(packed_state_dim=4)
    storage = torch.zeros(9, 4, 8, device="cuda", dtype=torch.float8_e4m3fn)
    latent_pool = storage[1::3]
    index_pool = torch.zeros(9, 4, 6, device="cuda", dtype=torch.bfloat16)[2::3]
    state = SimpleNamespace(
        latent_pool=latent_pool,
        index_pool=index_pool,
        tokens_per_block=4,
        num_contexts=1 if context else 0,
        block_tables=torch.tensor([[2, 0]], device="cuda"),
    )
    backend._cache_state = Mock(return_value=state)
    latent = torch.tensor([[0.0, 0.125, -0.4, 1.1, 5.0, -5.0, 500.0, -500.0]], device="cuda")
    packed = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda")
    positions = torch.tensor([0] if context else [[0]], device="cuda")
    backend.append_paged_state(
        latent,
        packed,
        positions,
        object(),
        request_ids=torch.zeros(1, dtype=torch.int32, device="cuda") if context else None,
    )
    decoded = latent_pool[2, 0].float() * backend.kv_scale_quant_orig
    actual_packed = index_pool[2, 0, :4]
    expected = (latent * 2).clamp(-448, 448).to(torch.float8_e4m3fn).float() * 0.5
    torch.testing.assert_close(decoded.unsqueeze(0), expected)
    torch.testing.assert_close(actual_packed.float().unsqueeze(0), packed)
    assert torch.count_nonzero(storage.float()[[0, 2, 3, 5, 6, 8]]) == 0
    assert torch.count_nonzero(index_pool[:, :, 4:]) == 0


@pytest.mark.skipif(
    get_sm_version() not in (100, 103), reason="native sparse MLA requires SM100/103"
)
@pytest.mark.parametrize(
    "rows,width,heads,cache_dtype",
    [
        (1, 64, 16, torch.bfloat16),
        (8, 2052, 16, torch.bfloat16),
        (8, 2052, 16, torch.float8_e4m3fn),
        (256, 128, 16, torch.float8_e4m3fn),
        (8, 2052, 32, torch.float8_e4m3fn),
        (64, 256, 64, torch.float8_e4m3fn),
    ],
)
def test_native_sparse_decode_coalesced_rows_and_graph_replay(
    rows: int, width: int, heads: int, cache_dtype: torch.dtype
) -> None:
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.backend import latent_pool_rows
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.native_decode import (
        GlmKpoolNativeDecode,
    )

    generator = torch.Generator(device="cuda").manual_seed(6810832)
    # Only the middle subpage belongs to this layer. Other subpages contain NaN
    # to expose wrong physical addressing instead of accidentally valid values.
    storage = torch.full((32, 3, 32, 512), float("nan"), dtype=cache_dtype, device="cuda")
    latent = storage[:, 1]
    latent.copy_(
        torch.randn(latent.shape, dtype=torch.bfloat16, device=latent.device, generator=generator)
    )
    kv_rows, base, stride = latent_pool_rows(latent)
    q_storage = torch.randn(
        rows, heads * 2, 512, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    q = q_storage[:, ::2]
    indices_storage = torch.empty(rows, width + 7, dtype=torch.int32, device="cuda")
    indices = indices_storage[:, :width]
    logical = torch.randint(0, 1024, (rows, width), device="cuda", generator=generator)
    physical = base + (logical // 32) * stride + logical % 32
    indices.copy_(physical)
    indices[:, ::5] = -1
    runner = GlmKpoolNativeDecode()
    metadata = SimpleNamespace(effective_workspace=torch.empty(0, dtype=torch.int8, device="cuda"))
    runner.prepare_workspace(q, metadata, max_query_rows=max(rows, 8), query_dtype=cache_dtype)
    scale = 256**-0.5
    fp8 = cache_dtype == torch.float8_e4m3fn
    kv_scale = torch.tensor([0.75], device="cuda") if fp8 else 1.0
    runner(q, kv_rows, indices, scale, metadata, kv_scale=kv_scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = runner(q, kv_rows, indices, scale, metadata, kv_scale=kv_scale)
    for replay in range(3):
        q.copy_(torch.randn(q.shape, dtype=q.dtype, device=q.device, generator=generator))
        if fp8 and replay == 2:
            q[:, 0, 0] = 1024
            q[:, 0, 1] = -1024
        indices.copy_(physical.roll(replay, dims=1))
        indices[:, replay::5] = -1
        if replay == 1:
            indices[-1] = -1
        graph.replay()
        valid = indices >= 0
        selected = kv_rows.float()[indices.clamp_min(0).long(), 0] * kv_scale
        selected.masked_fill_(~valid[:, :, None], 0)
        query = q.clamp(-448, 448).to(cache_dtype).float() if fp8 else q.float()
        logits = torch.einsum("rhd,rkd->rhk", query, selected) * scale
        logits.masked_fill_(~valid[:, None, :], float("-inf"))
        probabilities = torch.nan_to_num(logits.softmax(dim=-1))
        expected = torch.einsum("rhk,rkd->rhd", probabilities, selected).to(q.dtype)
        if fp8:
            # Match the FP8 sparse-MLA tests' energy-normalized error metric.
            a, b = actual.float(), expected.float()
            diff = (a - b).square().sum() / (a.square().sum() + b.square().sum())
            assert diff < 1e-3
            torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
            if replay < 2:
                # The reference above quantizes the query as well; also bound the
                # FP8 query's own error against BF16-query attention.
                logits = torch.einsum("rhd,rkd->rhk", q.float(), selected) * scale
                logits.masked_fill_(~valid[:, None, :], float("-inf"))
                c = torch.einsum("rhk,rkd->rhd", torch.nan_to_num(logits.softmax(dim=-1)), selected)
                assert (a - c).square().sum() / (a.square().sum() + c.square().sum()) < 1e-3
        else:
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


def test_pool_scores_offsets_beyond_int32() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.kernels import _kpool_score_kernel

    # A 1M-token capacity has 2**18 pools. Row 8192 starts at element 2**31.
    # Only launch the first pool tile: exercise real pointer arithmetic without
    # scoring every pool or touching the entire 8 GiB allocation.
    rows, capacity, rows_per_program = 8208, 1 << 18, 16
    required_bytes = rows * capacity * 4
    if torch.cuda.mem_get_info()[0] < required_bytes + (1 << 30):
        pytest.skip("large score-offset regression requires 9 GiB of free device memory")
    out = torch.empty(rows, capacity, dtype=torch.float32, device="cuda")
    q = torch.ones(rows, 1, HD, dtype=torch.bfloat16, device="cuda")
    weights = torch.ones(rows, 1, dtype=torch.bfloat16, device="cuda")
    pool = torch.ones(2, TPB, 3 * HD, dtype=torch.bfloat16, device="cuda")
    pool[1].fill_(2)
    tables = torch.tensor([[0], [1]], dtype=torch.int64, device="cuda")
    lengths = torch.zeros(rows, dtype=torch.int64, device="cuda")
    requests = torch.zeros(rows, dtype=torch.int32, device="cuda")

    for branch in ("invisible", "shared", "request_boundary"):
        if branch != "invisible":
            lengths.fill_(KPOOL)
        if branch == "request_boundary":
            requests[-8:] = 1
        _kpool_score_kernel[(rows // rows_per_program, 1)](
            q,
            weights,
            pool,
            tables,
            lengths,
            requests,
            out,
            rows,
            weights.stride(0),
            weights.stride(1),
            pool.stride(0),
            pool.stride(1),
            tables.stride(0),
            TPB,
            capacity,
            1.0 / HD,
            1.0,
            FP32_MIN,
            HD=HD,
            KPOOL=KPOOL,
            H=1,
            HP=16,
            BP=64,
            ROWS=rows_per_program,
            PRECISION="tf32",
            HAS_REQ=True,
            num_warps=4,
        )
        expected = torch.full((32, 64), FP32_MIN, device="cuda")
        if branch != "invisible":
            expected[:, 0] = 1
        if branch == "request_boundary":
            expected[-8:, 0] = 2
        torch.testing.assert_close(out[-32:, :64], expected, rtol=0, atol=0)


def _meta_indexer():
    """A GLM-5.3-Flash indexer (meta weights) exposing its pool top-k."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextIndexer

    config = SimpleNamespace(
        hidden_size=256,
        q_lora_rank=128,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_kpool=4,
        index_kpool_always_select_tail=True,
    )
    with torch.device("meta"):
        return Glm5NextIndexer(config, layer_idx=0)


def test_pool_topk_without_cute_dsl_replays_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch import cute_dsl_utils

    monkeypatch.setattr(cute_dsl_utils, "IS_CUTLASS_DSL_AVAILABLE", False)
    indexer = _meta_indexer()
    scores = torch.arange(520, dtype=torch.float32, device="cuda").repeat(3, 1)
    lengths = torch.tensor([0, 3, 513], dtype=torch.int32, device="cuda")
    selected = torch.empty(3, 512, dtype=torch.int32, device="cuda")

    def select() -> torch.Tensor:
        return indexer.pool_top_k(
            scores, selected, is_prefill=False, sequence_lengths=lengths, scan_lengths=lengths
        )

    select()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select()
    for bounds in ([0, 3, 513], [520, 1, 0]):
        lengths.copy_(torch.tensor(bounds, dtype=torch.int32, device="cuda"))
        graph.replay()
        for row, count in enumerate(bounds):
            actual = selected[row].cpu()
            valid = actual[actual >= 0].sort().values
            expected = torch.arange(max(0, count - 512), count, dtype=torch.int32)
            assert torch.equal(valid, expected)
            assert int((actual == -1).sum()) == 512 - min(count, 512)


@pytest.mark.parametrize("latent_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize("tokens_per_request", [1, 4])
def test_append_paged_state_rows_matches_indexed_writes(latent_dtype, tokens_per_request):
    """One launch writes the latent (quantized for FP8) and packed state of each
    generation row into its paged slot, bit-exact with indexed writes."""
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import paged_slot_indices

    gen = torch.Generator(device="cuda").manual_seed(7)
    requests, pages, slots, latent_dim = 6, 6, 40, 512
    table = _paged_tables(requests, pages, slots, gen)
    first = torch.randint(
        0, pages * TPB - tokens_per_request, (requests, 1), generator=gen, device="cuda"
    )
    positions = first + torch.arange(tokens_per_request, device="cuda")
    # Strided rows, like slices of a fused projection output.
    latent = (
        torch.randn(requests, tokens_per_request, latent_dim + 64, generator=gen, device="cuda")
        * 50
    ).to(torch.bfloat16)[..., :latent_dim]
    packed = torch.randn(requests, tokens_per_request, 2 * HD, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    scale = torch.tensor([0.7], device="cuda")

    def pools():
        # Coalesced layer pages: foreign buffers sit between this layer's slots.
        latent_pool = torch.zeros(slots, 2, TPB, latent_dim, device="cuda").to(latent_dtype)[:, 0]
        index_pool = torch.zeros(slots, 3, TPB, 3 * HD, device="cuda", dtype=torch.bfloat16)[:, 1]
        return latent_pool, index_pool

    want_latent, want_index = pools()
    page, offset = paged_slot_indices(table, positions, TPB)
    if latent_dtype == torch.float8_e4m3fn:
        quantized = (latent.float() * scale).clamp(-448, 448).to(latent_dtype)
        want_latent.view(torch.uint8)[page, offset] = quantized.view(torch.uint8)
    else:
        want_latent[page, offset] = latent
    want_index[page, offset, : 2 * HD] = packed

    got_latent, got_index = pools()
    append_paged_state_rows(latent, packed, positions, table, got_latent, got_index, TPB, scale)
    assert torch.equal(got_latent.view(torch.uint8), want_latent.view(torch.uint8))
    assert torch.equal(got_index, want_index)


def test_generation_append_uses_generation_tables_in_mixed_batch():
    """Generation rows of a mixed batch write through the generation slice of the block
    tables, leaving the context requests' pages untouched."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import (
        GlmKpoolSparseAttention,
        paged_slot_indices,
    )

    gen = torch.Generator(device="cuda").manual_seed(12)
    num_contexts, num_generations, pages, slots, latent_dim = 2, 3, 4, 40, 512
    tokens_per_request = 4
    tables = _paged_tables(num_contexts + num_generations, pages, slots, gen)
    first = torch.randint(
        0, pages * TPB - tokens_per_request, (num_generations, 1), generator=gen, device="cuda"
    )
    positions = first + torch.arange(tokens_per_request, device="cuda")
    latent = (
        torch.randn(num_generations, tokens_per_request, latent_dim, generator=gen, device="cuda")
        * 50
    ).to(torch.bfloat16)
    packed = torch.randn(
        num_generations, tokens_per_request, 2 * HD, generator=gen, device="cuda"
    ).to(torch.bfloat16)
    scale = torch.tensor([0.7], device="cuda")

    def pools():
        latent_pool = torch.zeros(slots, TPB, latent_dim, device="cuda", dtype=torch.bfloat16)
        index_pool = torch.zeros(slots, TPB, 3 * HD, device="cuda", dtype=torch.bfloat16)
        return latent_pool, index_pool

    want_latent, want_index = pools()
    page, offset = paged_slot_indices(tables[num_contexts:], positions, TPB)
    want_latent[page, offset] = latent
    want_index[page, offset, : 2 * HD] = packed

    got_latent, got_index = pools()
    backend = object.__new__(GlmKpoolSparseAttention)
    backend.kv_scale_orig_quant = scale
    backend._cache_state = Mock(
        return_value=SimpleNamespace(
            latent_pool=got_latent,
            index_pool=got_index,
            tokens_per_block=TPB,
            num_contexts=num_contexts,
            block_tables=tables,
        )
    )
    backend.append_paged_state(latent, packed, positions, object())

    assert torch.equal(got_latent, want_latent)
    assert torch.equal(got_index, want_index)
    context_slots = tables[:num_contexts].flatten()
    assert not got_latent[context_slots].any()
    assert not got_index[context_slots].any()


@pytest.mark.parametrize("tokens_per_request", [2, 4])
def test_pool_key_refresh_draft_positions_match_sequential_updates(tokens_per_request):
    """[N, T] draft positions in one launch equal T ordered single-position updates."""
    gen = torch.Generator(device="cuda").manual_seed(8)
    requests, pages, slots = 5, 6, 40
    pool = _strided_index_pool(slots, gen)
    table = _paged_tables(requests, pages, slots, gen)
    first = torch.randint(
        0, pages * TPB - tokens_per_request, (requests, 1), generator=gen, device="cuda"
    )
    positions = first + torch.arange(tokens_per_request, device="cuda")
    ape = torch.randn(KPOOL, HD, generator=gen, device="cuda").to(torch.bfloat16)

    sequential = pool.clone()
    for step in range(tokens_per_request):
        kpool_update(
            sequential, table, positions[:, step].contiguous(), ape, TPB, head_dim=HD, kpool=KPOOL
        )
    batched = pool.clone()
    kpool_update(batched, table, positions, ape, TPB, head_dim=HD, kpool=KPOOL)
    assert torch.equal(batched, sequential)


@pytest.mark.parametrize("width", [5, 64, 2052])
def test_compact_sparse_rows_tail_and_empty_rows(width):
    """Valid indices move to the front with a -1 tail; empty rows get one dummy index."""
    gen = torch.Generator(device="cuda").manual_seed(9)
    rows = 7
    indices = torch.randint(
        -1, 1000, (rows, width), generator=gen, device="cuda", dtype=torch.int32
    )
    indices[3] = -1
    packed, lengths, nonempty = compact_sparse_rows(indices)
    for row in range(rows):
        values = indices[row][indices[row] >= 0]
        expected = torch.full((packed.shape[1],), -1, dtype=torch.int32, device="cuda")
        expected[: values.numel()] = values
        if values.numel() == 0:
            expected[0] = 0
        assert torch.equal(packed[row], expected)
        assert int(lengths[row]) == max(values.numel(), 1)
        assert bool(nonempty[row]) == (values.numel() > 0)
