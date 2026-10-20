# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate of the drafter last-row context FMHA's K/V tail zeroing (drafter_last_row._zero_kv_tails)."""

import types

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import drafter_last_row

PAGE, HEADS, D = 128, 4, 128


def _pool_and_tables(kv_lens, seed=0):
    """A flat FP8 sub-block pool filled with NaN-coded and random bytes, and a shuffled [batch, 2, pages] table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [(n + PAGE - 1) // PAGE for n in kv_lens]
    num_blocks = 2 * sum(pages) + 6
    raw = torch.randint(
        0, 256, (num_blocks, HEADS, PAGE, D), dtype=torch.uint8, device="cuda", generator=g
    )
    raw[::3] = 0x7F
    perm = torch.randperm(num_blocks, device="cuda", generator=g)
    tables = torch.zeros(len(kv_lens), 2, max(pages), dtype=torch.int32, device="cuda")
    used = 0
    for b, n in enumerate(pages):
        tables[b, 0, :n] = perm[used : used + n]
        tables[b, 1, :n] = perm[used + n : used + 2 * n]
        used += 2 * n
    return raw.view(torch.float8_e4m3fn), tables


@pytest.mark.parametrize(
    "kv_lens",
    [[16390], [16], [32, 128, 129], [1, 255, 256, 257], [16384, 6]],
)
def test_only_the_partial_last_pages_tails_are_zeroed(kv_lens):
    pool, tables = _pool_and_tables(kv_lens)
    before = pool.view(torch.uint8).clone()
    lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    drafter_last_row._zero_kv_tails(pool, tables, lens, len(kv_lens))
    after = pool.view(torch.uint8)

    expected = before.clone()
    for b, n in enumerate(kv_lens):
        if n % PAGE:
            for half in (0, 1):
                expected[int(tables[b, half, (n - 1) // PAGE]), :, n % PAGE :, :] = 0
    assert torch.equal(after, expected)
    for b, n in enumerate(kv_lens):
        if n % PAGE:
            for half in (0, 1):
                tail = after[int(tables[b, half, (n - 1) // PAGE]), :, n % PAGE :, :]
                assert int(((tail & 0x7F) == 0x7F).sum()) == 0


def test_rows_past_batch_are_untouched():
    pool, tables = _pool_and_tables([100, 200])
    before = pool.view(torch.uint8).clone()
    lens = torch.tensor([100, 200], dtype=torch.int32, device="cuda")
    drafter_last_row._zero_kv_tails(pool, tables, lens, 1)
    after = pool.view(torch.uint8)
    second = [int(tables[1, h, 1]) for h in (0, 1)]
    for blk in second:
        assert torch.equal(after[blk], before[blk])


@pytest.mark.parametrize(
    "manager,safe",
    [
        (types.SimpleNamespace(enable_block_reuse=False, enable_partial_reuse=True), True),
        (types.SimpleNamespace(enable_block_reuse=True, enable_partial_reuse=False), True),
        (types.SimpleNamespace(enable_block_reuse=True, enable_partial_reuse=True), False),
        (
            types.SimpleNamespace(
                enable_block_reuse=True, enable_partial_reuse=True, copy_on_partial_reuse=True
            ),
            True,
        ),
    ],
)
def test_tail_zeroing_needs_private_partial_pages(manager, safe):
    assert drafter_last_row._v_tail_zero_safe(manager) is safe


def _library(monkeypatch, enabled=True):
    monkeypatch.setattr(drafter_last_row, "_KV_TAIL_ZERO", enabled)
    lib = drafter_last_row.DrafterLastRowContextFmha.__new__(
        drafter_last_row.DrafterLastRowContextFmha
    )
    lib._kv_tail_safe = None
    return lib


def _zero_if_allowed(lib, manager, pool, tables, lens):
    """The attention path's own gate: zero the tails only when the library allows it for this KV cache."""
    if lib._kv_tail_zero(types.SimpleNamespace(kv_cache_manager=manager)):
        drafter_last_row._zero_kv_tails(pool, tables, lens, lens.shape[0])


def test_a_shared_partial_page_declines_and_changes_no_byte(monkeypatch):
    # Block and partial reuse on, no copy on partial reuse: the partial last page may belong to another request.
    shared = types.SimpleNamespace(enable_block_reuse=True, enable_partial_reuse=True)
    lib = _library(monkeypatch)
    pool, tables = _pool_and_tables([16390, 16])
    before = pool.view(torch.uint8).clone()
    lens = torch.tensor([16390, 16], dtype=torch.int32, device="cuda")
    _zero_if_allowed(lib, shared, pool, tables, lens)
    assert lib._kv_tail_safe is False
    assert torch.equal(pool.view(torch.uint8), before)


def test_the_kill_switch_changes_no_byte(monkeypatch):
    private = types.SimpleNamespace(enable_block_reuse=False, enable_partial_reuse=False)
    lib = _library(monkeypatch, enabled=False)
    pool, tables = _pool_and_tables([16390])
    before = pool.view(torch.uint8).clone()
    _zero_if_allowed(
        lib, private, pool, tables, torch.tensor([16390], dtype=torch.int32, device="cuda")
    )
    assert torch.equal(pool.view(torch.uint8), before)


def test_a_private_cache_zeroes_through_the_gate(monkeypatch):
    private = types.SimpleNamespace(enable_block_reuse=False, enable_partial_reuse=False)
    lib = _library(monkeypatch)
    pool, tables = _pool_and_tables([16390])
    _zero_if_allowed(
        lib, private, pool, tables, torch.tensor([16390], dtype=torch.int32, device="cuda")
    )
    for half in (0, 1):
        assert (
            int(
                pool.view(torch.uint8)[
                    int(tables[0, half, 16389 // PAGE]), :, 16390 % PAGE :
                ].count_nonzero()
            )
            == 0
        )
