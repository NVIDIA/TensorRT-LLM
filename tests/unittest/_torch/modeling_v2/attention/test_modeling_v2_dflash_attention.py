# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA tests of the DFlash draft block's non-causal windowed attention over a paged pool.

``modeling_dflash.dflash_window_gather`` / ``dflash_windowed_block_attention`` serve a drafter
whose layers attend non-causally within a two-sided window (the DFlash 2 Beam drafter: a
4096-token window, eight-row blocks, 10 query / 2 KV heads of 128 per TP4 rank) on the TRTLLM
draft-attention backend, whose TRTLLM-Gen kernels have no such mask. Query ``j`` of a block
sits at slot-local position ``ctx + j`` and must see exactly the request's keys within
``window - 1`` positions on either side: its committed context and the current block, nothing
at or past ``ctx + block_size``, nothing of another request.

What is checked, against independent integer and FP32 references (never the helper's own
mask):

* every query's effective key-index set equals the integer rule at contexts on both sides of
  the window, including the 4095/4096/4097 boundary and short contexts;
* boundary sentinels with one-hot values: a key at distance ``window - 1`` contributes, one at
  distance ``window`` contributes nothing, so a missing or extra key changes the output by a
  whole one-hot column rather than by an average's rounding;
* stale and foreign pages: high-scoring poisoned keys past the block and in other requests'
  pages contribute nothing; a rewound block (smaller context after a rejection) excludes the
  tail it no longer owns;
* the attention output against explicit FP32 masked softmax (reported, BF16 tolerance).

Needs one CUDA device; no model files.
"""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dflash import (
    dflash_window_gather,
    dflash_windowed_block_attention,
)

WINDOW = 4096
BLOCK = 8
PAGE = 32
HQ, HKV, D = 10, 2, 128
SCALE = D**-0.5

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _oracle(ctx: int, j: int) -> set:
    """Key positions query ``j`` of a block at context ``ctx`` must see (integer rule)."""
    q = ctx + j
    return {k for k in range(0, ctx + BLOCK) if abs(q - k) <= WINDOW - 1}


def _pool(ctx_lens, *, poison: float = 0.0):
    """A paged HND pool with each request on its own pages and an identity-free block table.

    Pages are dealt out interleaved across requests (request r owns pages r, r + R, ...), so a
    request's pages are not contiguous and a wrong row of the table reads another request.
    Returns ``(pool [pages, 2, HKV, PAGE, D], block_tables [R, pages_per_row])``.
    """
    rows = len(ctx_lens)
    pages_per_row = (max(ctx_lens) + BLOCK + PAGE - 1) // PAGE + 1
    total = rows * pages_per_row
    pool = torch.full((total, 2, HKV, PAGE, D), poison, dtype=torch.bfloat16, device="cuda")
    tables = torch.arange(total, dtype=torch.int32, device="cuda").view(pages_per_row, rows).t()
    return pool, tables.contiguous()


def _write(pool, tables, row, position, k=None, v=None):
    page = int(tables[row, position // PAGE])
    if k is not None:
        pool[page, 0, :, position % PAGE] = k
    if v is not None:
        pool[page, 1, :, position % PAGE] = v


@pytest.mark.parametrize(
    "ctx_lens",
    [
        (1, 5, 52, 513),
        (4087, 4088, 4089, 4095),
        (4096, 4097, 4100, 5000),
        (0, 4095, 4096, 4097),
    ],
    ids=["short", "below-window", "past-window", "boundary"],
)
def test_key_index_sets_equal_integer_rule(ctx_lens):
    """Every query's gathered, unmasked key positions are exactly the integer rule's set."""
    _, tables = _pool(ctx_lens)
    num_ctx = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
    gather = dflash_window_gather(tables, num_ctx, BLOCK, WINDOW, PAGE)
    assert gather.positions.shape == (len(ctx_lens), WINDOW + BLOCK - 1)
    assert gather.mask.shape == (len(ctx_lens), 1, BLOCK, WINDOW + BLOCK - 1)
    positions = gather.positions.cpu()
    mask = gather.mask[:, 0].cpu()
    valid = gather.valid.cpu()
    for row, ctx in enumerate(ctx_lens):
        for j in range(BLOCK):
            seen = set(positions[row][mask[row, j]].tolist())
            assert seen == _oracle(ctx, j), (ctx, j)
        # Valid keys are the request's own: committed context and the current block.
        assert set(positions[row][valid[row]].tolist()) == set(range(0, ctx + BLOCK)) & set(
            positions[row].tolist()
        )


def test_invalid_positions_address_the_blocks_own_page():
    """A position outside the request's keys is redirected to its block's first slot."""
    ctx_lens = (3, 4200)
    _, tables = _pool(ctx_lens)
    num_ctx = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
    gather = dflash_window_gather(tables, num_ctx, BLOCK, WINDOW, PAGE)
    for row, ctx in enumerate(ctx_lens):
        own_page = int(tables[row, ctx // PAGE])
        invalid = ~gather.valid[row]
        assert bool((gather.pages[row][invalid] == own_page).all())
        assert bool((gather.offsets[row][invalid] == ctx % PAGE).all())
        owned = set(tables[row].tolist())
        assert set(gather.pages[row].tolist()) <= owned


def _one_hot(column: int) -> torch.Tensor:
    v = torch.zeros(HKV, D, dtype=torch.bfloat16, device="cuda")
    v[:, column] = 1.0
    return v


def _sentinel_query(rows: int) -> torch.Tensor:
    """Queries along e_0 scaled so that a key ``e_0`` scores 40 and a zero key scores 0."""
    q = torch.zeros(rows, BLOCK, HQ, D, dtype=torch.float32, device="cuda")
    q[..., 0] = 40.0 / SCALE
    return q.to(torch.bfloat16)


def _place_sentinels(pool, tables, row: int, ctx: int, *, stale: int = 0) -> None:
    """Keys of request ``row`` at ``ctx - 4096 + m`` (m = 0..8) score 40 with value ``e_m``;
    ``stale`` keys past its block (a rejected tail it no longer owns) score 80 with ``e_9``."""
    key = _one_hot(0)
    for m in range(BLOCK + 1):
        position = ctx - WINDOW + m
        if position >= 0:
            _write(pool, tables, row, position, k=key, v=_one_hot(m))
    for s in range(stale):
        _write(pool, tables, row, ctx + BLOCK + s, k=2 * key, v=_one_hot(9))


def _boundary_inputs(ctx: int):
    pool, tables = _pool([ctx])
    _place_sentinels(pool, tables, 0, ctx)
    return _sentinel_query(1), pool, tables


@pytest.mark.parametrize("ctx", [4095, 4096, 4097, 4103, 6000])
def test_window_boundary_sentinels(ctx):
    """Query j sees the sentinel keys at distance <= 4095 and none at distance 4096.

    The visible sentinels share the softmax almost evenly (each scores ~40 against ~4000
    zero-score keys), so column m of query j's output is ~1/(#visible) exactly for the
    visible m and ~0 for the others; one extra or missing boundary key moves a whole column.
    """
    q, pool, tables = _boundary_inputs(ctx)
    num_ctx = torch.tensor([ctx], dtype=torch.int32, device="cuda")
    gather = dflash_window_gather(tables, num_ctx, BLOCK, WINDOW, PAGE)
    out = dflash_windowed_block_attention(q, pool, gather, SCALE).float()[0]
    for j in range(BLOCK):
        visible = [m for m in range(BLOCK + 1) if ctx - WINDOW + m >= 0 and m >= j + 1]
        expected = torch.zeros(BLOCK + 1, device="cuda")
        if visible:
            expected[visible] = 1.0 / len(visible)
        got = out[j, :, : BLOCK + 1]
        assert torch.allclose(got, expected.expand_as(got), atol=2e-2), (ctx, j, got[0])


@pytest.mark.parametrize("stale", [1, 8])
def test_stale_tail_and_foreign_pages_contribute_nothing(stale):
    """Poisoned keys past request 0's block (a rewound tail still in its pages) and every slot
    of request 1's interleaved pages outscore all valid keys, so any leak would dominate:
    request 0's output must carry none of their value columns."""
    ctx = 4100
    pool, tables = _pool([ctx + stale, ctx + stale])
    _place_sentinels(pool, tables, 0, ctx, stale=stale)
    for page in tables[1].tolist():
        pool[page, 0, :, :, 0] = 2.0
        pool[page, 1] = _one_hot(10).unsqueeze(1).expand(HKV, PAGE, D)
    num_ctx = torch.tensor([ctx, ctx], dtype=torch.int32, device="cuda")
    gather = dflash_window_gather(tables, num_ctx, BLOCK, WINDOW, PAGE)
    out = dflash_windowed_block_attention(_sentinel_query(2), pool, gather, SCALE).float()
    assert float(out[0, :, :, 9].abs().max()) < 1e-3, "a key past the block leaked"
    assert float(out[0, :, :, 10].abs().max()) < 1e-3, "another request's page leaked"
    for j in range(BLOCK):
        assert abs(float(out[0, j, 0, j + 1 : BLOCK + 1].sum()) - 1.0) < 2e-2, j
    # Request 1's keys within its window are all poison: it sees only them.
    assert float((out[1, :, :, 10] - 1.0).abs().max()) < 2e-2


def _rewound_block(num_ctx: int):
    """A block appended at 4092..4099, then a rejection that committed context 4095, plus a
    far stale row at 4103..4107 (key scoring 80, value e_9); the next block is gathered at
    context ``num_ctx``. Returns the block's output and gather."""
    pool, tables = _pool([4100])
    _place_sentinels(pool, tables, 0, 4095)
    key = _one_hot(0)
    for position in range(4103, 4108):
        _write(pool, tables, 0, position, k=2 * key, v=_one_hot(9))
    gather = dflash_window_gather(
        tables, torch.tensor([num_ctx], dtype=torch.int32, device="cuda"), BLOCK, WINDOW, PAGE
    )
    out = dflash_windowed_block_attention(_sentinel_query(1), pool, gather, SCALE).float()
    return out, gather


def _check_rewound(out, gather, committed: int) -> None:
    """The block after the rejection reads no stale value and its keys end at its own rows."""
    assert float(out[0, :, :, 9].abs().max()) < 1e-3, "a stale row past the block was read"
    assert int(gather.positions[0][gather.valid[0]].max()) == committed + BLOCK - 1, (
        "the gathered keys do not end at the block's own last row"
    )


def test_rewound_block_excludes_the_tail_it_no_longer_owns():
    """The next block at the committed context 4095 must neither see the superseded rows past
    4102 nor miss a key."""
    _check_rewound(*_rewound_block(4095), committed=4095)


def test_unrewound_draft_context_is_detected():
    """Fault control: a drafter that keeps the superseded block as committed context (context
    4100 where the rejection committed 4095) gathers the stale rows; the same check fails."""
    with pytest.raises(AssertionError, match="stale row"):
        _check_rewound(*_rewound_block(4100), committed=4095)


def _fp32_reference(q, pool, tables, ctx_lens):
    out = torch.empty(q.shape, dtype=torch.float32, device="cuda")
    for row, ctx in enumerate(ctx_lens):
        keys = list(range(0, ctx + BLOCK))
        pages = tables[row, torch.tensor(keys, device="cuda") // PAGE].long()
        offsets = torch.tensor(keys, device="cuda") % PAGE
        k = pool[pages, 0, :, offsets].float().repeat_interleave(HQ // HKV, dim=1)
        v = pool[pages, 1, :, offsets].float().repeat_interleave(HQ // HKV, dim=1)
        kpos = torch.tensor(keys, device="cuda")
        for j in range(BLOCK):
            allowed = (ctx + j - kpos).abs() <= WINDOW - 1
            s = torch.einsum("hd,khd->hk", q[row, j].float(), k) * SCALE
            s = s.masked_fill(~allowed[None], float("-inf"))
            out[row, j] = torch.einsum("hk,khd->hd", s.softmax(-1), v)
    return out


def test_matches_fp32_masked_softmax():
    """Random data at four requests (the TP4 geometry): BF16 SDPA vs explicit FP32."""
    torch.manual_seed(0)
    ctx_lens = (52, 513, 4095, 4097)
    pool, tables = _pool(ctx_lens)
    pool.normal_()
    q = torch.randn(len(ctx_lens), BLOCK, HQ, D, dtype=torch.bfloat16, device="cuda")
    num_ctx = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
    gather = dflash_window_gather(tables, num_ctx, BLOCK, WINDOW, PAGE)
    out = dflash_windowed_block_attention(q, pool, gather, SCALE).float()
    ref = _fp32_reference(q, pool, tables, ctx_lens)
    rel = float((out - ref).abs().max() / ref.abs().max())
    assert rel < 1e-2, rel
