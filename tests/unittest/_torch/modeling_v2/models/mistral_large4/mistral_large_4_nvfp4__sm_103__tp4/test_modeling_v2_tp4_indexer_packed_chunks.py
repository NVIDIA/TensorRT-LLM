# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp4 target's context indexer at the 1M config (max_seq_len 1048576, chunked prefill in
steps of max_num_tokens 32768) on prefill chunks the engine packs past max_seq_len keys.

Under chunked prefill the engine plans the indexer's prefill chunks with
`plan_mla_chunked_prefill_chunks`: consecutive context requests share one chunk while (summed
new tokens) x (summed cached + new keys) stays within the packed-logits budget, 2^28 at
max_num_tokens 32768 and max_seq_len 1048576. A short last chunk of a near-1M request beside a
short prompt is therefore one chunk of more than 1048576 keys, past the key scorer's
(`fp8_mqa_logits`) and the prefill top-k's cells. The target cuts a chunk past max_seq_len keys,
and only such a chunk, at its request boundaries (`_indexer_blocks`) and scores each request's
rows against its own keys (`PrefillTarget._select_ctx`, in tiles under the engine's
`_prefill_logits_budget`, 1 << 28 here).

On one GPU, each case's chunk is the one the engine's own planner and chunk builder
(`Indexer.prepare_one_prefill_chunk`) make at that config, over an indexer key pool the
case's full-KV gather mappings address through a page table (the engine's
`_compute_slot_mappings`, each request's pages drawn from one permutation of the pool), with
the step's offsets through the target's `_ctx_kv_offsets` and `_ctx_host_offsets`. Checked
per case:

* the planner packs every request into one chunk, of the expected key count;
* the target's blocks: one per request where the chunk holds more than max_seq_len keys, the
  chunk itself where it does not, each row covered once;
* no key gather, scorer or top-k call sees more than 1048576 keys, and every top-k call's
  selection is an exact top-k of the scores it was given (rewritten canonically: ascending,
  `-1` last, a tie at the k-th score given to the lowest-index keys, since the top-k promises
  neither order nor which tied keys it keeps);
* every row is written, and each request's selections are exactly those of the same request
  planned and selected alone.

Skipped off sm_103 or without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tp4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as engine
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 103,
    reason="the mistral_large4 modeling_v2 targets are certified on sm_103 only",
)

MAX_SEQ_LEN = 1 << 20
MAX_NUM_TOKENS = 32768
TOKENS_PER_BLOCK = 32
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOPK = 2048
#: The step's widest block table at max_seq_len 1048576 (V2: ceil((L + 1) / 32), to 4).
TABLE_WIDTH = 32772
SENTINEL = -(2**31)
#: Elements the exact-top-k check holds at once.
_CHECK_ELEMENTS = 1 << 25

#: (name, ((cached, new) per context request, in batch order), the packed chunk's keys).
CASES = (
    (
        "a 1048559-token prompt's 100-token tail beside a 150-token prompt",
        ((1048459, 100), (0, 150)),
        1048709,
    ),
    (
        "two 5-token tails over 983040 cached keys",
        ((983040, 5), (983040, 5)),
        1966090,
    ),
    (
        "a 64-token tail of 1015872, a 100-token prompt and a 50-token tail of 32818",
        ((1015808, 64), (0, 100), (32768, 50)),
        1048790,
    ),
    (
        "a 128-token tail and a 128-token prompt, exactly max_seq_len keys",
        ((1048320, 128), (0, 128)),
        1048576,
    ),
    (
        "16 tails of 16 tokens over 61440 cached keys",
        tuple((61440, 16) for _ in range(16)),
        983296,
    ),
)


def _md(requests):
    """The metadata the engine's planner, its chunk builder and the target's offsets read, for
    a step whose context requests are `requests` (no decode rows)."""
    cached = [c for c, _ in requests]
    new = [n for _, n in requests]
    kv = [c + n for c, n in requests]

    def indptr(values):
        return torch.tensor([0, *torch.tensor(values).cumsum(0).tolist()], dtype=torch.int64)

    return SimpleNamespace(
        compress_ratios=[1],
        cu_seqlen_ks=torch.empty(0, dtype=torch.int32, device="cuda"),
        max_num_tokens=MAX_NUM_TOKENS,
        kv_cache_manager=SimpleNamespace(max_seq_len=MAX_SEQ_LEN),
        enable_context_mla_with_cached_kv=True,
        host_ctx_cached_token_indptr=indptr(cached),
        host_ctx_uncached_token_indptr=indptr(new),
        host_ctx_kv_indptr=indptr(kv),
        ctx_cached_token_indptr=indptr(cached).cuda(),
        ctx_kv_indptr=indptr(kv).cuda(),
        max_ctx_seq_len=max(new),
        num_ctx_cached_tokens=sum(cached),
    )


def _chunks(md, requests):
    """The engine's indexer prefill chunks for the step, unsplit (fewer rows than
    q_split_threshold)."""
    q_lens = [n for _, n in requests]
    kv_lens = [c + n for c, n in requests]
    budget = engine._packed_logits_elem_budget(md)
    assert budget == 1 << 28, budget
    return [
        engine.Indexer.prepare_one_prefill_chunk(md, specs)
        for specs in engine.plan_mla_chunked_prefill_chunks(q_lens, kv_lens, budget)
    ]


def _step(md, requests, chunks, pool):
    """The target's step dict for the indexer's context selection, through its own offsets and
    block plan."""
    tc = sum(n for _, n in requests)
    step = dict(
        num_ctx_tokens=tc,
        num_ctx_kv_tokens=sum(c + n for c, n in requests),
        cu_seqlen_ks=None,
        cu_seqlen_ke=None,
        max_seq_len=MAX_SEQ_LEN,
        slot_mapping_fp8_fullkv=pool.fp8,
        slot_mapping_scale_fullkv=pool.scale,
        **M._ctx_kv_offsets(md, len(requests)),
        **M._ctx_host_offsets(md, len(requests)),
    )
    step["indexer_blocks"] = M._indexer_blocks(chunks, step)
    return step


def _pool(requests, gen, cuda_gen):
    """An indexer key pool holding every request's keys at random e4m3 values and positive
    scales, and the step's full-KV gather mappings (every request's cached and new keys,
    request after request), each request's pages drawn from one permutation of the pool."""
    kv = [c + n for c, n in requests]
    pages = [-(-k // TOKENS_PER_BLOCK) for k in kv]
    num_pages = sum(pages) + 7
    order = torch.randperm(num_pages, generator=gen)
    table = torch.zeros(len(requests), TABLE_WIDTH, dtype=torch.int32)
    at = 0
    for r, p in enumerate(pages):
        table[r, :p] = order[at : at + p].to(torch.int32)
        at += p
    req = torch.repeat_interleave(torch.arange(len(requests)), torch.tensor(kv))
    starts = torch.tensor([0, *torch.tensor(kv).cumsum(0).tolist()[:-1]])
    positions = torch.arange(sum(kv)) - starts.repeat_interleave(torch.tensor(kv))
    fp8, scale = engine._compute_slot_mappings(
        positions, table, req, INDEX_DIM, TOKENS_PER_BLOCK, INDEX_DIM
    )
    row = TOKENS_PER_BLOCK * (INDEX_DIM + 4)
    flat = torch.empty(num_pages, row, dtype=torch.uint8, device="cuda")
    data = torch.randn(num_pages, TOKENS_PER_BLOCK * INDEX_DIM, generator=cuda_gen, device="cuda")
    flat[:, : TOKENS_PER_BLOCK * INDEX_DIM] = data.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = 0.5 + torch.rand(num_pages, TOKENS_PER_BLOCK, generator=cuda_gen, device="cuda")
    flat[:, TOKENS_PER_BLOCK * INDEX_DIM :] = scales.view(torch.uint8)
    return SimpleNamespace(
        k_cache=flat.view(num_pages, TOKENS_PER_BLOCK, 1, INDEX_DIM + 4),
        fp8=fp8.to(torch.int64).cuda(),
        scale=scale.to(torch.int64).cuda(),
        kv=kv,
    )


def _canonicalize(values, ks, ke, out, top_k: int) -> None:
    """Check that `out` is an exact top-k of each row's window `[ks, ke)` of `values`
    (window-local indices; a window within top_k kept whole, then `-1`) and rewrite it
    canonically: ascending, `-1` last, a tie at the k-th score given to the lowest-index
    keys."""
    n, width = values.shape
    dev = values.device
    cols = torch.arange(width, device=dev, dtype=torch.int64)
    ident = torch.arange(top_k, device=dev, dtype=torch.int64)
    big = torch.iinfo(torch.int64).max
    step = max(1, _CHECK_ELEMENTS // max(width, 1))
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        s = ks[r0:r1].long()[:, None]
        e = ke[r0:r1].long()[:, None]
        length = (e - s)[:, 0]
        assert int(e.max()) <= width, (int(e.max()), width)
        raw = out[r0:r1].long()
        raw_sorted = torch.where(raw < 0, big, raw).sort(dim=1).values
        raw_sorted = torch.where(raw_sorted == big, -1, raw_sorted)
        long_row = length > top_k
        short = torch.where(ident < length[:, None], ident, -1)
        bad_short = (~long_row) & (raw_sorted != short).any(1)
        assert not bad_short.any(), f"{int(bad_short.sum())} short windows not kept whole"
        canon = short
        if long_row.any():
            sc = torch.where((cols >= s) & (cols < e), values[r0:r1].float(), float("-inf"))
            kth = sc.topk(top_k, dim=1).values[:, -1:]
            above = (sc > kth).sum(1)
            at = sc == kth
            inside = (raw_sorted >= 0) & (raw_sorted < length[:, None])
            distinct = (raw_sorted[:, 1:] != raw_sorted[:, :-1]).all(1)
            picked = sc.gather(1, (s + raw_sorted.clamp(min=0)).clamp(max=width - 1))
            exact = (
                inside.all(1)
                & distinct
                & (picked.min(1).values == kth[:, 0])
                & ((picked > kth).sum(1) == above)
            )
            bad = long_row & ~exact
            assert not bad.any(), f"{int(bad.sum())} rows are not an exact top-{top_k}"
            take = (sc > kth) | (at & (at.int().cumsum(1) <= (top_k - above)[:, None]))
            best = torch.where(take, cols - s, big).sort(dim=1).values[:, :top_k]
            canon = torch.where(long_row[:, None], best, short)
        out[r0:r1].copy_(canon.to(out.dtype))


@pytest.fixture
def calls(monkeypatch):
    """Every key gather's, scorer's and top-k's key count, through the module names the target
    calls them by; each top-k selection checked exact and made canonical where it is made."""
    seen = SimpleNamespace(gather=[], logits=[], topk=[])
    gather, logits = M.indexer_k_cache_gather_op, M.fp8_mqa_logits
    topk = M.cute_dsl_gvr_topk_prefill

    def gather_(k_cache, fp8, scale, k_token_start, num_tokens, head_dim):
        seen.gather.append((k_token_start, num_tokens))
        return gather(k_cache, fp8, scale, k_token_start, num_tokens, head_dim)

    def logits_(q, kv, w, ks, ke, *args, **kwargs):
        seen.logits.append((q.shape[0], kv[0].shape[0]))
        return logits(q, kv, w, ks, ke, *args, **kwargs)

    def topk_(values, ks, ke, out, top_k, *args, **kwargs):
        seen.topk.append((values.shape[0], values.shape[1]))
        result = topk(values, ks, ke, out, top_k, *args, **kwargs)
        _canonicalize(values, ks, ke, out, top_k)
        return result

    monkeypatch.setattr(M, "indexer_k_cache_gather_op", gather_)
    monkeypatch.setattr(M, "fp8_mqa_logits", logits_)
    monkeypatch.setattr(M, "cute_dsl_gvr_topk_prefill", topk_)
    return seen


def _select(requests, chunks, md, pool, q_fp8, weights):
    """The target's context selection over one step, off a core sized for this config (the
    logits budget `_derive_first_step` sets from max_num_tokens and max_seq_len); the selection
    buffer it fills."""
    step = _step(md, requests, chunks, pool)
    tc = step["num_ctx_tokens"]
    topk_buf = torch.full([tc, INDEX_TOPK], SENTINEL, dtype=torch.int32, device="cuda")
    core = SimpleNamespace(
        prefill_logits_budget=M._prefill_logits_budget(MAX_NUM_TOKENS, MAX_SEQ_LEN),
        idx_dim=INDEX_DIM,
        idx_topk=INDEX_TOPK,
        _max_ctx_len=MAX_SEQ_LEN,
    )
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = core
    M.PrefillTarget._select_ctx(target, step, pool.k_cache, q_fp8, weights, topk_buf)
    torch.cuda.synchronize()
    return step, topk_buf


@pytest.mark.parametrize("name,requests,packed_keys", CASES, ids=[c[0] for c in CASES])
def test_packed_chunk_selects_as_each_request_alone(name, requests, packed_keys, calls):
    seed = len(name) * 7919 + packed_keys
    gen = torch.Generator().manual_seed(seed)
    cuda_gen = torch.Generator(device="cuda").manual_seed(seed)
    md = _md(requests)
    chunks = _chunks(md, requests)
    # The engine packs every request into one chunk over all their keys.
    tc = sum(n for _, n in requests)
    assert len(chunks) == 1, [(c.token_start, c.token_end) for c in chunks]
    (chunk,) = chunks
    assert (chunk.token_start, chunk.token_end) == (0, tc)
    assert chunk.k_token_end - chunk.k_token_start == packed_keys

    pool = _pool(requests, gen, cuda_gen)
    q = torch.randn(tc, INDEX_HEADS, INDEX_DIM, generator=cuda_gen, device="cuda")
    q_fp8 = q.to(torch.float8_e4m3fn)
    weights = torch.randn(tc, INDEX_HEADS, generator=cuda_gen, device="cuda")
    step, packed = _select(requests, chunks, md, pool, q_fp8, weights)

    # The blocks: the chunk itself within max_seq_len keys, else one per request.
    blocks = step["indexer_blocks"]
    cu_new, cu_kv = step["host_cu_ctx_new"], step["host_cu_ctx_kv"]
    if packed_keys <= MAX_SEQ_LEN:
        assert len(blocks) == 1 and blocks[0] is chunk, blocks
    else:
        expect = [(cu_new[r], cu_new[r + 1], cu_kv[r], cu_kv[r + 1]) for r in range(len(requests))]
        got = [(b.token_start, b.token_end, b.k_token_start, b.k_token_end) for b in blocks]
        assert got == expect, (got, expect)
        for r, b in enumerate(blocks):
            cached = requests[r][0]
            assert torch.equal(b.cu_seqlen_ks.cpu(), torch.zeros(requests[r][1], dtype=torch.int32))
            ends = torch.arange(cached + 1, cached + requests[r][1] + 1, dtype=torch.int32)
            assert torch.equal(b.cu_seqlen_ke.cpu(), ends)
    # No call sees more than max_seq_len keys.
    widest = max(n for _, n in calls.gather)
    assert widest <= MAX_SEQ_LEN, calls.gather
    assert max(n for _, n in calls.logits) <= MAX_SEQ_LEN, calls.logits
    assert max(w for _, w in calls.topk) <= MAX_SEQ_LEN, calls.topk
    assert int((packed == SENTINEL).any(dim=1).sum()) == 0, "rows left unwritten"

    # Each request alone: its own chunk over its own keys (the same pool slots).
    for r, (cached, new) in enumerate(requests):
        alone_req = ((cached, new),)
        alone_md = _md(alone_req)
        alone_chunks = _chunks(alone_md, alone_req)
        lo, hi = cu_kv[r], cu_kv[r + 1]
        alone_pool = SimpleNamespace(
            k_cache=pool.k_cache,
            fp8=pool.fp8[lo:hi].contiguous(),
            scale=pool.scale[lo:hi].contiguous(),
        )
        rows = slice(cu_new[r], cu_new[r + 1])
        _, alone = _select(
            alone_req, alone_chunks, alone_md, alone_pool, q_fp8[rows], weights[rows]
        )
        differ = int((packed[rows] != alone).any(dim=1).sum())
        assert differ == 0, f"{name}: request {r}, {differ} of {new} rows select otherwise alone"
    print(
        f"PACKED {name}: chunk keys {packed_keys}, blocks "
        f"{[(b.token_start, b.token_end, b.k_token_end - b.k_token_start) for b in blocks]}, "
        f"widest gather {widest}, top-k calls (rows, width) {sorted(set(calls.topk))}"
    )
