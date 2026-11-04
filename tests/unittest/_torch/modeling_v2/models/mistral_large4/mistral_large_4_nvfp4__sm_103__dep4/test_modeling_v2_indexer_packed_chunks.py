# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep4 target's indexer selection over packed prefill chunks past max_seq_len.

Under chunked prefill the engine plans the indexer's prefill chunks with
`plan_mla_chunked_prefill_chunks` over each context request's whole KV (cached + new), packing
requests into one chunk while (summed new tokens) x (summed KV) stays within its packed-logits
budget (1 << 28). A few short last chunks over long prefixes on one rank therefore form one
chunk with more keys than one request can hold: two 5-token chunks over 983040 cached keys each
plan as (token_start, token_end, k_token_start, k_token_end) = (0, 10, 0, 1966090).
`PrefillTarget._select_ctx` cuts such a chunk at its request boundaries (`_prefill_blocks`), at
any engine: each request's rows are scored against its own keys only, gathered at that
request's `k_token_start` in the step's full-KV mappings, its rows' windows rebased to it, in
query tiles under the engine's logits budget (`_prefill_logits_budget(max_num_tokens,
max_seq_len)`, the ceiling 2**26 at max_num_tokens 32768 and max_seq_len 65536 or 1048576). A
chunk of at most max_seq_len keys is scored whole.

Each case builds its step as the engine does: `Indexer.prepare_for_update_k_cache` and
`Indexer.prepare_for_chunked_prefill` on DSA metadata holding what they read (each context
request's cached and new tokens, its pages in an indexer key pool of random e4m3 keys and fp32
scales), and the target's own `_ctx_kv_offsets` and `_ctx_host_offsets` projections of it. The
target's `_select_ctx` then runs on it, with every gather, logits and top-k call recorded, and
on each request's own step, that request alone. Every call must stay at or below max_seq_len
keys, and each row's selection must be the one it gets alone, compared as sets: keys tied at
the 2048th score may be kept differently by two calls, so a key in one selection and not the
other must score exactly the row's 2048th.

The cases: case (d) itself; a ~1M request's 100-token last chunk packed with a fresh 155-token
prompt (1048655 keys); a scaled analogue, four requests whose summed 110330 keys pass a
max_seq_len of 65536; and, not cut, sixteen tails packed to exactly 1048576 keys.

One GB300 (sm_103, where the indexer kernels are certified); `_prefill_blocks` alone on CPU.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__dep4 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    Indexer,
    _packed_logits_elem_budget,
)
from tensorrt_llm._utils import get_sm_version

MAX_SEQ_LEN = 1048576
MAX_NUM_TOKENS = 32768
TOP_K = 2048
HEADS = 32
HEAD_DIM = 128
TOKENS_PER_BLOCK = 32
# One indexer key page: 32 keys of 128 e4m3 codes, then their 32 fp32 scales.
PAGE_BYTES = TOKENS_PER_BLOCK * (HEAD_DIM + 4)
# A selection slot no call wrote.
UNWRITTEN = -7

gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 103,
    reason="the mistral_large4 modeling_v2 targets are certified on sm_103 only",
)

#: name -> (max_seq_len, [(cached, new)] per context request, the engine's chunks, the blocks
#: `_select_ctx` scores, as (token_start, token_end, k_token_start, n_k)).
CASES = {
    "case_d": (
        MAX_SEQ_LEN,
        [(983_040, 5), (983_040, 5)],
        [(0, 10, 0, 1_966_090)],
        [(0, 5, 0, 983_045), (5, 10, 983_045, 983_045)],
    ),
    "tail_and_fresh": (
        MAX_SEQ_LEN,
        [(1_048_400, 100), (0, 155)],
        [(0, 255, 0, 1_048_655)],
        [(0, 100, 0, 1_048_500), (100, 255, 1_048_500, 155)],
    ),
    "scaled": (
        65536,
        [(40_000, 7), (50_000, 3), (0, 300), (20_000, 20)],
        [(0, 330, 0, 110_330)],
        [
            (0, 7, 0, 40_007),
            (7, 10, 40_007, 50_003),
            (10, 310, 90_010, 300),
            (310, 330, 90_310, 20_020),
        ],
    ),  # fmt: skip
    "sixteen_tails_at_max": (
        MAX_SEQ_LEN,
        [(65_520, 16)] * 16,
        [(0, 256, 0, MAX_SEQ_LEN)],
        [(0, 256, 0, MAX_SEQ_LEN)],
    ),
}


def _pool(pages: int, gen: torch.Generator) -> torch.Tensor:
    """An indexer key pool `[pages, 32, 1, 132]` (the V2 view's layout) of random keys."""
    pool = torch.empty(pages, PAGE_BYTES, dtype=torch.uint8, device="cuda")
    codes_end = TOKENS_PER_BLOCK * HEAD_DIM
    for lo in range(0, pages, 8192):
        hi = min(lo + 8192, pages)
        codes = torch.randn(hi - lo, codes_end, generator=gen, device="cuda")
        pool[lo:hi, :codes_end] = codes.to(torch.float8_e4m3fn).view(torch.uint8)
        scales = torch.rand(hi - lo, TOKENS_PER_BLOCK, generator=gen, device="cuda") + 0.5
        pool[lo:hi, codes_end:] = scales.view(torch.uint8)
    return pool.view(pages, TOKENS_PER_BLOCK, 1, HEAD_DIM + 4)


def _step(contexts, table, max_seq_len: int) -> dict:
    """`Indexer.prepare_for_update_k_cache` and `prepare_for_chunked_prefill` on one dep4 rank's
    step of context requests `contexts` ((cached, new) each, its pages on its row of `table`),
    then the target's step state over it, as `PrefillTarget.step_args` hands it over: the
    engine's chunks and full-KV mappings (the step's own slot mappings when no request has a
    cached prefix), `_ctx_kv_offsets` and `_ctx_host_offsets`."""
    num_contexts = len(contexts)
    new = [n for _, n in contexts]
    cached = [c for c, _ in contexts]
    cu_new = [0, *itertools.accumulate(new)]
    cu_cached = [0, *itertools.accumulate(cached)]
    cu_kv = [0, *itertools.accumulate(c + n for c, n in contexts)]
    i64 = dict(dtype=torch.int64)
    md = SimpleNamespace(
        mapping=SimpleNamespace(enable_attention_dp=True, tp_size=4, tp_rank=0),
        enable_context_mla_with_cached_kv=True,
        runtime_features=SimpleNamespace(chunked_prefill=True),
        sparse_metadata_params=SimpleNamespace(q_split_threshold=8192),
        compress_ratios=[1],
        max_num_tokens=MAX_NUM_TOKENS,
        kv_cache_manager=SimpleNamespace(
            max_seq_len=max_seq_len, index_head_dim=HEAD_DIM, tokens_per_block=TOKENS_PER_BLOCK
        ),
        indexer_head_dim=HEAD_DIM,
        indexer_quant_block_size=HEAD_DIM,
        _tokens_per_block=TOKENS_PER_BLOCK,
        num_contexts=num_contexts,
        num_generations=0,
        num_ctx_tokens=sum(new),
        request_ids=list(range(num_contexts)),
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=cached),
        seq_lens=torch.tensor(new, dtype=torch.int32),
        host_ctx_cached_token_indptr=torch.tensor(cu_cached, **i64),
        host_ctx_uncached_token_indptr=torch.tensor(cu_new, **i64),
        host_ctx_kv_indptr=torch.tensor(cu_kv, **i64),
        ctx_cached_token_indptr=torch.tensor(cu_cached, **i64, device="cuda"),
        ctx_kv_indptr=torch.tensor(cu_kv, **i64, device="cuda"),
        max_ctx_seq_len=max(new),
        num_ctx_cached_tokens=sum(cached),
        cu_seqlen_ks=torch.empty(MAX_NUM_TOKENS, dtype=torch.int32, device="cuda"),
        skip_indexer_for_ctx_reqs=False,
        slot_mapping_fp8=torch.empty(MAX_NUM_TOKENS, **i64, device="cuda"),
        slot_mapping_scale=torch.empty(MAX_NUM_TOKENS, **i64, device="cuda"),
        host_slot_mapping_fp8=torch.empty(MAX_NUM_TOKENS, **i64),
        host_slot_mapping_scale=torch.empty(MAX_NUM_TOKENS, **i64),
        host_indexer_k_cache_block_offsets=table,
        indexer_prefill_q_split="unset",
        indexer_prefill_chunks=None,
    )
    params = Indexer.build_indexer_params(md)
    Indexer.prepare_for_update_k_cache(md, params)
    Indexer.prepare_for_chunked_prefill(md, params)
    assert md.indexer_prefill_q_split is None
    assert _packed_logits_elem_budget(md) == 1 << 28
    return dict(
        indexer_prefill_chunks=md.indexer_prefill_chunks,
        slot_mapping_fp8_fullkv=md.slot_mapping_fp8_fullkv,
        slot_mapping_scale_fullkv=md.slot_mapping_scale_fullkv,
        num_ctx_kv_tokens=cu_kv[-1],
        cu_seqlen_ks=None,
        cu_seqlen_ke=None,
        **M._ctx_kv_offsets(md, num_contexts),
        **M._ctx_host_offsets(md, num_contexts),
    )


def _budget(max_seq_len: int) -> int:
    """The logits budget `_derive_first_step` gives the core at max_num_tokens 32768 and
    `max_seq_len`."""
    return M._prefill_logits_budget(MAX_NUM_TOKENS, max_seq_len)


def _select(step, max_seq_len: int, pool, q, weights) -> torch.Tensor:
    """The target's `_select_ctx` over every context row of `step`, on a prefill target holding
    just what it reads off its core at max_num_tokens 32768 and `max_seq_len`."""
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = SimpleNamespace(
        idx_dim=HEAD_DIM,
        idx_topk=TOP_K,
        _max_ctx_len=max_seq_len,
        prefill_logits_budget=_budget(max_seq_len),
    )
    rows = q.shape[0]
    out = torch.full([rows, TOP_K], UNWRITTEN, dtype=torch.int32, device="cuda")
    target._select_ctx(step, pool, q, weights, out, rows)
    return out


def _recorded(monkeypatch) -> list[tuple]:
    """Record every gather, logits and top-k call the target makes, then make it."""
    calls: list[tuple] = []
    gather = M.indexer_k_cache_gather_op
    logits = M.fp8_mqa_logits
    topk = M.cute_dsl_gvr_topk_prefill

    def gather_rec(k_pool, fp8, scale, k_token_start, num_tokens, head_dim):
        calls.append(("gather", k_token_start, num_tokens))
        return gather(k_pool, fp8, scale, k_token_start, num_tokens, head_dim)

    def logits_rec(q, kv, weights, ks, ke, schedule_meta=None):
        assert schedule_meta is not None, "a tile scored without its split-KV schedule"
        calls.append(("logits", q.shape[0], kv[0].shape[0]))
        return logits(q, kv, weights, ks, ke, schedule_meta=schedule_meta)

    def topk_rec(values, ks, ke, out, top_k, workspace=None):
        calls.append(("topk", values.shape[0], values.shape[1]))
        return topk(values, ks, ke, out, top_k, workspace)

    monkeypatch.setattr(M, "indexer_k_cache_gather_op", gather_rec)
    monkeypatch.setattr(M, "fp8_mqa_logits", logits_rec)
    monkeypatch.setattr(M, "cute_dsl_gvr_topk_prefill", topk_rec)
    return calls


def _alone(context, row, max_seq_len: int, pool, q, weights):
    """One request scored in a step of its own: its selection, and its rows' logits and
    windows to judge a tie by."""
    step = _step([context], row, max_seq_len)
    sel = _select(step, max_seq_len, pool, q, weights)
    (chunk,) = step["indexer_prefill_chunks"]
    keys, scales = M.indexer_k_cache_gather_op(
        pool, step["slot_mapping_fp8_fullkv"], step["slot_mapping_scale_fullkv"], 0,
        chunk.k_token_end, HEAD_DIM,
    )  # fmt: skip
    ks, ke = chunk.cu_seqlen_ks, chunk.cu_seqlen_ke
    scores = M.fp8_mqa_logits(q, (keys, torch.reshape(scales, [-1])), weights, ks, ke)
    return sel, scores, ks.tolist(), ke.tolist()


def _compare(where, got, sel, scores, ks, ke) -> tuple[int, int]:
    """A request's rows of the packed step's selection `got` against its own `sel`, as sets; a
    key in one and not the other must score exactly the row's 2048th. Returns (rows the same
    set, rows that differ on such a tie)."""
    same = tied = 0
    for i in range(got.shape[0]):
        a, b = set(got[i].tolist()), set(sel[i].tolist())
        assert UNWRITTEN not in a and UNWRITTEN not in b, (where, i, "a row was not written")
        if a == b:
            same += 1
            continue
        width = ke[i] - ks[i]
        assert width > TOP_K, (where, i, width, sorted(a ^ b)[:8])
        row = scores[i, ks[i] : ke[i]]
        kth = torch.topk(row, TOP_K).values[-1]
        off = [p for p in a ^ b if p < 0 or p >= width or row[p] != kth]
        assert not off, (where, i, "differs off the 2048th score", off[:8], float(kth))
        tied += 1
    return same, tied


def test_prefill_blocks_cut_only_past_max_seq_len():
    """`_prefill_blocks` on host values: a chunk within max_seq_len is one block, itself; past
    it, one block per request, each its own rows over its own keys."""
    packed = SimpleNamespace(token_start=0, token_end=10, k_token_start=0, k_token_end=1_966_090)
    cu_new, cu_kv = [0, 5, 10], [0, 983_045, 1_966_090]
    assert M._prefill_blocks(packed, cu_new, cu_kv, MAX_SEQ_LEN) == [
        (0, 5, 0, 983_045),
        (5, 10, 983_045, 983_045),
    ]
    assert M._prefill_blocks(packed, cu_new, cu_kv, 1_966_090) == [(0, 10, 0, 1_966_090)]
    # A packed chunk behind an earlier context request: its rows and keys start past that one's.
    behind = SimpleNamespace(
        token_start=32_000, token_end=32_010, k_token_start=500_000, k_token_end=2_466_090
    )
    cu_new, cu_kv = [0, 32_000, 32_005, 32_010], [0, 500_000, 1_483_045, 2_466_090]
    assert M._prefill_blocks(behind, cu_new, cu_kv, MAX_SEQ_LEN) == [
        (32_000, 32_005, 500_000, 983_045),
        (32_005, 32_010, 1_483_045, 983_045),
    ]
    # One request's 32768-row last chunk over its whole 1M KV is never cut.
    last = SimpleNamespace(token_start=0, token_end=32768, k_token_start=0, k_token_end=MAX_SEQ_LEN)
    assert M._prefill_blocks(last, [0, 32768], [0, MAX_SEQ_LEN], MAX_SEQ_LEN) == [
        (0, 32768, 0, MAX_SEQ_LEN)
    ]


@gpu
@pytest.mark.parametrize("name", list(CASES))
def test_packed_chunk_selections_are_each_requests_own(name, monkeypatch):
    """The engine's chunk plan of the case, the blocks the target scores, every call at or below
    max_seq_len keys, and every request's rows selecting as that request alone does."""
    max_seq_len, contexts, chunks, blocks = CASES[name]
    gen = torch.Generator(device="cuda").manual_seed(len(name))
    kv = [c + n for c, n in contexts]
    pages = [-(-k // TOKENS_PER_BLOCK) for k in kv]
    # Each request's pages, scattered over the pool.
    order = torch.randperm(sum(pages), generator=torch.Generator().manual_seed(7))
    table = torch.zeros(len(contexts), max(pages), dtype=torch.int32)
    for r, (lo, n) in enumerate(zip(itertools.accumulate([0, *pages]), pages)):
        table[r, :n] = order[lo : lo + n].to(torch.int32)
    pool = _pool(sum(pages), gen)
    rows = sum(n for _, n in contexts)
    q = torch.randn(rows, HEADS, HEAD_DIM, generator=gen, device="cuda") * 0.5
    q = q.to(torch.float8_e4m3fn)
    weights = torch.randn(rows, HEADS, generator=gen, device="cuda")

    step = _step(contexts, table, max_seq_len)
    planned = [
        (c.token_start, c.token_end, c.k_token_start, c.k_token_end)
        for c in step["indexer_prefill_chunks"]
    ]
    assert planned == chunks, planned
    assert step["host_cu_ctx_kv"][-1] == step["slot_mapping_fp8_fullkv"].shape[0] == sum(kv)

    calls = _recorded(monkeypatch)
    got = _select(step, max_seq_len, pool, q, weights)
    torch.cuda.synchronize()
    monkeypatch.undo()
    print(f"PACKED {name}: max_seq_len {max_seq_len}, contexts {contexts}, chunks {planned}")
    print(f"PACKED {name}: calls {calls}")

    # One gather per block at its own keys; then its rows in tiles of the engine's logits
    # budget.
    expected = []
    for lo, hi, k_start, n_k in blocks:
        expected.append(("gather", k_start, n_k))
        tile = max(1, _budget(max_seq_len) // n_k)
        for t in range(lo, hi, tile):
            expected += [("logits", min(tile, hi - t), n_k), ("topk", min(tile, hi - t), n_k)]
    assert calls == expected, calls
    assert all(n_k <= max_seq_len for _, _, n_k in calls), calls

    cu_new = step["host_cu_ctx_new"]
    for r, context in enumerate(contexts):
        mine = slice(cu_new[r], cu_new[r + 1])
        sel, scores, ks, ke = _alone(context, table[r : r + 1], max_seq_len, pool, q[mine],
                                     weights[mine])  # fmt: skip
        same, tied = _compare((name, r), got[mine], sel, scores, ks, ke)
        print(
            f"PACKED {name} request {r} {context}: {same} of {len(ks)} rows the same set as "
            f"alone, {tied} differ on a tie at the 2048th score; bitwise equal "
            f"{torch.equal(got[mine], sel)}"
        )
