# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-request hash-chain reuse (TRTLLM_MOONCAKE_STORE_CHAIN_MEMO_MB).

A fresh BlockHashChain takes the hashes of the longest byte-identical block prefix of a recent prompt (same seed,
same block size) and digests only the rest. The chain must equal the one computed from scratch in every case:
shared prefixes cut mid-block or on a block edge, unrelated prompts, different reuse scopes, other block sizes,
eviction under a small budget, short prompts, and later incremental extension. CPU only, stdlib.
"""

import random

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import keys

pytestmark = pytest.mark.cpu_only


def _chain(tokens, tpb=32, scope=None, memo=None):
    saved = keys._CHAIN_MEMO
    keys._CHAIN_MEMO = memo
    try:
        chain = keys.BlockHashChain(tpb, scope=scope)
        return list(chain.extend(tokens)), chain
    finally:
        keys._CHAIN_MEMO = saved


def _rand(rng, n):
    return [rng.randrange(0, 202048) for _ in range(n)]


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(keys.CHAIN_MEMO_ENV, raising=False)
    assert keys.CHAIN_MEMO_ENV == "TRTLLM_MOONCAKE_STORE_CHAIN_MEMO_MB"
    assert keys._chain_memo_from_env() is None
    monkeypatch.setenv(keys.CHAIN_MEMO_ENV, "0")
    assert keys._chain_memo_from_env() is None
    monkeypatch.setenv(keys.CHAIN_MEMO_ENV, "junk")
    assert keys._chain_memo_from_env() is None
    monkeypatch.setenv(keys.CHAIN_MEMO_ENV, "64")
    memo = keys._chain_memo_from_env()
    assert memo is not None and memo._budget == 64 << 20


def test_agentic_turns_reuse_and_match_from_scratch():
    rng = random.Random(7)
    memo = keys._ChainMemo(256 << 20)
    prompt = _rand(rng, 32 * 300 + 17)
    for turn in range(8):
        got, _ = _chain(prompt, memo=memo)
        ref, _ = _chain(prompt, memo=None)
        assert got == ref
        if turn:
            # the previous turn's whole prompt is a prefix of this one
            assert memo.reused >= prev_reused + (prev_len // 32)
        prev_len, prev_reused = len(prompt), memo.reused
        prompt = prompt + _rand(rng, rng.randrange(1, 3000))
    assert memo.chains == 8 and memo.hits == 7


@pytest.mark.parametrize("seed", range(6))
def test_fuzz_equivalence(seed):
    rng = random.Random(seed)
    memo = keys._ChainMemo(rng.choice([1 << 20, 4 << 20, 256 << 20]))
    pool = [_rand(rng, rng.randrange(1, 32 * 500))]
    scopes = [None, keys.ReuseScope(cache_salt="a"), keys.ReuseScope(cache_salt="b")]
    for _ in range(40):
        base = rng.choice(pool)
        kind = rng.random()
        if kind < 0.5:  # continue a session, cut anywhere (mid-block or edge)
            cut = rng.randrange(0, len(base) + 1)
            tokens = base[:cut] + _rand(rng, rng.randrange(0, 4000))
        elif kind < 0.7:  # same prompt again
            tokens = list(base)
        elif kind < 0.85:  # diverge inside the first blocks
            tokens = list(base)
            if tokens:
                tokens[rng.randrange(0, min(len(tokens), 100))] ^= 1
        else:  # unrelated
            tokens = _rand(rng, rng.randrange(1, 32 * 400))
        tpb = rng.choice([32, 32, 64])
        scope = rng.choice(scopes)
        got, _ = _chain(tokens, tpb, scope, memo)
        ref, _ = _chain(tokens, tpb, scope, None)
        assert got == ref
        pool.append(tokens)
    # the index never points at an evicted entry
    for ids in memo._index.values():
        assert ids and all(i in memo._entries for i in ids)
    assert memo._bytes <= memo._budget


def test_scopes_and_block_sizes_do_not_share():
    rng = random.Random(3)
    tokens = _rand(rng, 32 * 200)
    memo = keys._ChainMemo(64 << 20)
    _chain(tokens, 32, keys.ReuseScope(cache_salt="x"), memo)
    _chain(tokens, 32, keys.ReuseScope(cache_salt="y"), memo)
    _chain(tokens, 64, keys.ReuseScope(cache_salt="x"), memo)
    assert memo.hits == 0
    got, _ = _chain(tokens, 32, keys.ReuseScope(cache_salt="x"), memo)
    assert memo.hits == 1 and memo.reused == 200
    assert got == _chain(tokens, 32, keys.ReuseScope(cache_salt="x"), None)[0]


def test_short_prompts_bypass_and_incremental_extension():
    rng = random.Random(5)
    memo = keys._ChainMemo(64 << 20)
    short = _rand(rng, 32 * (keys.CHAIN_MEMO_MIN_BLOCKS - 1) + 5)
    _chain(short, memo=memo)
    assert memo.chains == 0
    long = _rand(rng, 32 * 150)
    _chain(long, memo=memo)
    got_hashes, chain = _chain(long + _rand(rng, 10), memo=memo)
    assert memo.hits == 1
    more = long + _rand(rng, 10) + _rand(rng, 32 * 7)
    saved = keys._CHAIN_MEMO
    keys._CHAIN_MEMO = memo
    try:
        extended = list(chain.extend(more))
    finally:
        keys._CHAIN_MEMO = saved
    assert extended == _chain(more, memo=None)[0]


def test_budget_evicts_lru():
    rng = random.Random(11)
    a, b = _rand(rng, 32 * 100), _rand(rng, 32 * 100)
    memo = keys._ChainMemo(8 * 32 * 100 + 10)  # room for one 100-block prompt
    _chain(a, memo=memo)
    _chain(b, memo=memo)
    assert len(memo._entries) == 1
    _chain(a, memo=memo)
    assert memo.hits == 0  # a was evicted by b
    _chain(a, memo=memo)
    assert memo.hits == 1


def test_memo_honours_the_packed_block_hash(monkeypatch, capsys):
    rng = random.Random(13)
    a = _rand(rng, 32 * 120)
    b = a + _rand(rng, 32 * 9 + 3)
    ref_a, ref_b = _chain(a, memo=None)[0], _chain(b, memo=None)[0]
    monkeypatch.setattr(keys, "_PACK_TOKENS", True)
    monkeypatch.setattr(keys, "_packed_receipt_sent", False)
    memo = keys._ChainMemo(64 << 20)
    assert _chain(a, memo=memo)[0] == ref_a
    assert _chain(b, memo=memo)[0] == ref_b
    assert memo.hits == 1 and memo.reused == 120
    assert keys.PACKED_HASH_RECEIPT in capsys.readouterr().err
