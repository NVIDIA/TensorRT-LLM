# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M3.1 packed block hash port (TRTLLM_MOONCAKE_STORE_PACKED_BLOCK_HASH): byte-identical hashes and keys.

Every hash and every store key of the packed path must equal the default `struct.pack` path's, on random token
arrays, across incremental extends, for tokens_per_block 32 (MSO) and 128 (M3.1), and for both of MSO's layer
groups (full attention + sliding window, different bytes per page) on every TP rank.
"""

import os
import random
import struct
import time

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import keys as keys_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import (
    BlockHashChain,
    KeyNamespace,
    ReuseScope,
)

pytestmark = pytest.mark.cpu_only


def _namespaces(tokens_per_block, world_size=4):
    # MSO: layer group 0 full attention, 1 sliding window, distinct page sizes; one namespace per rank.
    return [
        KeyNamespace("trtllm", "mso", rank, world_size, group, tokens_per_block, bytes_per_page)
        for group, bytes_per_page in ((0, 1327104), (1, 589824))
        for rank in range(world_size)
    ]


def _chain_and_keys(monkeypatch, packed, tokens, cuts, tokens_per_block, scope):
    monkeypatch.setattr(keys_module, "_PACK_TOKENS", packed)
    chain = BlockHashChain(tokens_per_block, scope=scope)
    snapshots = []
    for end in cuts:
        snapshots.append(list(chain.extend(tokens[:end])))
    hashes = list(chain.hashes)
    keys = [ns.key(h) for h in hashes for ns in _namespaces(tokens_per_block)]
    return snapshots, hashes, keys


def test_off_by_default():
    assert keys_module.PACKED_BLOCK_HASH_ENV == "TRTLLM_MOONCAKE_STORE_PACKED_BLOCK_HASH"
    if os.environ.get(keys_module.PACKED_BLOCK_HASH_ENV) is None:
        assert keys_module._PACK_TOKENS is False


@pytest.mark.parametrize("tokens_per_block", [32, 128])
@pytest.mark.parametrize("seed", range(6))
def test_packed_hashes_and_keys_are_identical(monkeypatch, tokens_per_block, seed):
    rng = random.Random(seed)
    num_tokens = rng.randrange(0, 60 * tokens_per_block)
    # Vocabulary ids, plus a few extreme int64 values (negative ids and both bounds).
    tokens = [rng.randrange(0, 200064) for _ in range(num_tokens)]
    for _ in range(min(4, num_tokens)):
        tokens[rng.randrange(num_tokens)] = rng.choice([-(2**63), 2**63 - 1, -1, 0])
    cuts = sorted(rng.randrange(0, num_tokens + 1) for _ in range(4)) + [num_tokens]
    scope = ReuseScope(cache_salt=rng.choice([None, "salt"]), multimodal_digests=())
    stock = _chain_and_keys(monkeypatch, False, tokens, cuts, tokens_per_block, scope)
    packed = _chain_and_keys(monkeypatch, True, tokens, cuts, tokens_per_block, scope)
    assert packed == stock
    assert len(stock[1]) == num_tokens // tokens_per_block


def test_packed_tokens_match_struct_bytes():
    rng = random.Random(1)
    tokens = [rng.randrange(-(2**63), 2**63) for _ in range(1000)]
    assert bytes(keys_module._pack_tokens_array(tokens)) == struct.pack(f"<{len(tokens)}q", *tokens)


def test_packed_rejects_out_of_range_tokens_like_stock(monkeypatch):
    for packed in (False, True):
        monkeypatch.setattr(keys_module, "_PACK_TOKENS", packed)
        with pytest.raises((OverflowError, struct.error)):
            BlockHashChain(4).extend([2**63, 1, 2, 3])


def test_packed_path_prints_its_receipt_once(monkeypatch, capsys):
    monkeypatch.setattr(keys_module, "_packed_receipt_sent", False)
    monkeypatch.setattr(keys_module, "_PACK_TOKENS", False)
    BlockHashChain(4).extend(list(range(16)))
    assert keys_module.PACKED_HASH_RECEIPT not in capsys.readouterr().err
    monkeypatch.setattr(keys_module, "_PACK_TOKENS", True)
    chain = BlockHashChain(4)
    chain.extend(list(range(16)))
    chain.extend(list(range(32)))
    BlockHashChain(4).extend(list(range(8)))
    assert capsys.readouterr().err.count(keys_module.PACKED_HASH_RECEIPT + " pid=") == 1


def test_report_cost(monkeypatch, capsys):
    """Not a bar: prints the per-prompt cost of both paths (MSO-sized prompt, 108k tokens, tpb 32)."""
    rng = random.Random(3)
    tokens = [rng.randrange(0, 200064) for _ in range(108000)]
    cost = {}
    for packed in (False, True):
        monkeypatch.setattr(keys_module, "_PACK_TOKENS", packed)
        best = float("inf")
        for _ in range(5):
            started = time.perf_counter()
            BlockHashChain(32).extend(tokens)
            best = min(best, time.perf_counter() - started)
        cost[packed] = best
    with capsys.disabled():
        print(
            f"\n[packed-hash] 108k tokens tpb 32: struct {cost[False] * 1e3:.2f} ms, "
            f"packed {cost[True] * 1e3:.2f} ms"
        )
