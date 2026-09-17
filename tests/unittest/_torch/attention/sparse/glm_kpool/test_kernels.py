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
    kpool_expand,
    kpool_score,
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


@pytest.mark.parametrize("precision", ["ieee", "tf32"])
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
    # bf16 inputs are exact in tf32, so both precisions are fp32 accumulations
    # of exact products.
    torch.testing.assert_close(got[visible], want[visible], atol=2e-3, rtol=2e-3)


def test_pool_scores_packed_context_rows_share_tables():
    """rows_per_program=16 with request_ids: the packed query tokens
    of several requests, in position order, including a group straddling two
    requests (per-row gather branch)."""
    gen = torch.Generator(device="cuda").manual_seed(2)
    heads, slots, pages = 3, 64, 8
    pool = _strided_index_pool(slots, gen)
    tables = _paged_tables(2, pages, slots, gen)
    # Request 0: 21 query tokens (positions 0..20), request 1: 19 (0..18).
    lens = [21, 19]
    request_ids = torch.cat(
        [torch.full((length,), i, dtype=torch.int32) for i, length in enumerate(lens)]
    ).cuda()
    kv_lens = torch.cat([torch.arange(1, length + 1) for length in lens]).cuda()
    n = kv_lens.shape[0]
    q = torch.randn(n, heads, HD, generator=gen, device="cuda").to(torch.bfloat16)
    w = torch.rand(n, heads, generator=gen, device="cuda").to(torch.bfloat16)
    kwargs = dict(num_pools_max=8, q_scale=HD**-0.5, w_scale=1.0)

    got = kpool_score(
        q,
        w,
        pool,
        tables,
        kv_lens,
        TPB,
        head_dim=HD,
        kpool=KPOOL,
        rows_per_program=16,
        request_ids=request_ids,
        **kwargs,
    )
    want = _reference_scores(q, w, pool, tables[request_ids.long()], kv_lens, **kwargs)
    visible = want > FP32_MIN
    assert torch.equal(got <= FP32_MIN, ~visible)
    torch.testing.assert_close(got[visible], want[visible], atol=2e-3, rtol=2e-3)


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
