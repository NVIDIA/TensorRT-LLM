# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Kimi K3.1 DSA indexer prefill share across attention-DP ranks (dsa_prefill_share).

Host tests: the count exchange is unchanged off the DEP perf target and on a disaggregated generation server; on
it, every rank derives the same plan from the exchanged counts, takes only an imbalanced rank's long chunks and splits
each chunk's rows once.
GPU test (sm_103): the rows a rank scores for a shared chunk are bit for bit the rows the owner writes alone.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (
    dsa_prefill_share as share,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_dsa_inputs as inputs  # noqa: E402


def _chunk(rows: int, keys: int, token_start: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        token_start=token_start, token_end=token_start + rows, k_token_start=0, k_token_end=keys
    )


@pytest.fixture(autouse=True)
def _aggregated_server(monkeypatch):
    """No disaggregated worker role, wherever the test runs."""
    monkeypatch.delenv(share._DISAGG_ROLE_ENV, raising=False)


def _metadata(
    rank: int, world: int, chunks, attention_dp: bool = True, k31_perf: bool = True
) -> SimpleNamespace:
    mapping = SimpleNamespace(
        enable_attention_dp=attention_dp, tp_size=world, tp_rank=rank, has_cp_helix=lambda: False
    )
    num_ctx_tokens = sum(c.token_end - c.token_start for c in chunks)
    return SimpleNamespace(
        mapping=mapping,
        indexer_prefill_chunks=list(chunks),
        num_contexts=len(chunks),
        num_ctx_tokens=num_ctx_tokens,
        skip_indexer_for_ctx_reqs=False,
        indexer_prefill_q_split=None,
        is_cuda_graph=False,
        indexer_share_plan="stale",
        _k31_perf=k31_perf,
    )


def _exchange(metas, base):
    """The attention-DP exchange: every rank's counts gathered, as columns (count i of every rank)."""
    rows = [list(b) + share.offer_counts(m) for b, m in zip(base, metas)]
    return [list(col) for col in zip(*rows)]


def test_off_without_attention_dp():
    m = _metadata(0, 4, [_chunk(8192, 400_000)], attention_dp=False)
    assert share.offer_counts(m) == []
    assert m.indexer_share_plan is None
    columns = [[8192, 8192, 8192, 8192]]
    assert share.take_plan(m, columns) is columns
    assert m.indexer_share_plan is None


@pytest.mark.parametrize(
    "k31_perf, role, shares",
    [
        (False, None, False),
        (True, "generation", False),
        (True, "context", True),
        (True, None, True),
    ],
    ids=["general-target", "disagg-generation", "disagg-context", "aggregated"],
)
def test_on_for_the_dep_perf_target_on_servers_that_run_context_chunks(
    monkeypatch, k31_perf, role, shares
):
    """The share runs on the DEP perf target only, and not on a disaggregated generation server."""
    if role is not None:
        monkeypatch.setenv(share._DISAGG_ROLE_ENV, role)
    m = _metadata(0, 4, [_chunk(8192, 400_000)], k31_perf=k31_perf)
    assert (share.offer_counts(m) != []) == shares
    columns = [[8192, 8192, 8192, 8192]]
    if not shares:
        assert share.take_plan(m, columns) is columns
        assert m.indexer_share_plan is None


def test_offers_the_largest_long_chunks():
    chunks = [
        _chunk(2048, 300_000, 0),  # long
        _chunk(512, 900_000, 2048),  # too few rows
        _chunk(4096, 250_000, 2560),  # long
        _chunk(1536, 600_000, 6656),  # long, the largest
        _chunk(1024, 100_000, 8192),  # too few keys
    ]
    desc, local = share.local_descriptors(_metadata(0, 4, chunks))
    work = sum((c.token_end - c.token_start) * c.k_token_end for c in chunks)
    # Two slots: the two largest offers (rows x keys), in batch order.
    assert local == [2, 3]
    assert desc == [1, -(-work // (1 << 20)), 4096, 250_000, 1536, 600_000]


def test_ineligible_rank_stops_the_step():
    metas = [_metadata(r, 4, [_chunk(8192, 500_000 if r == 0 else 50_000)]) for r in range(4)]
    metas[3].skip_indexer_for_ctx_reqs = True
    columns = _exchange(metas, [[8192]] * 4)
    for m in metas:
        assert share.take_plan(m, columns) == [[8192] * 4]
        assert m.indexer_share_plan is None


def test_every_rank_derives_one_plan():
    world = 4
    per_rank = [
        [_chunk(8192, 600_000)],  # heavy: offered and taken
        [_chunk(4096, 210_000), _chunk(4096, 30_000, 4096)],  # long chunk, but near the mean: kept
        [_chunk(8192, 20_000)],
        [_chunk(8192, 10_000)],
    ]
    metas = [_metadata(r, world, c) for r, c in enumerate(per_rank)]
    base = [[sum(c.token_end - c.token_start for c in cs)] for cs in per_rank]
    columns = _exchange(metas, base)
    stripped = [share.take_plan(m, columns) for m in metas]
    assert all(s == [[b[0] for b in base]] for s in stripped)
    plans = [m.indexer_share_plan for m in metas]
    assert all(p is not None for p in plans)
    assert all(p.entries == plans[0].entries for p in plans)
    assert [(e.owner, e.rows, e.keys) for e in plans[0].entries] == [(0, 8192, 600_000)]
    assert plans[0].local_chunks == [0] and all(p.local_chunks == [] for p in plans[1:])
    assert all(p.pieces == plans[0].pieces for p in plans)
    # The rows of the taken chunk are cut once and spread.
    pieces = sorted((p.lo, p.hi) for p in plans[0].pieces)
    assert pieces[0][0] == 0 and pieces[-1][1] == 8192
    assert all(a[1] == b[0] for a, b in zip(pieces, pieces[1:]))
    assert len({p.rank for p in plans[0].pieces}) > 1


def test_a_balanced_step_shares_nothing():
    metas = [_metadata(r, 2, [_chunk(8192, 300_000)]) for r in range(2)]
    columns = _exchange(metas, [[8192]] * 2)
    for m in metas:
        assert share.take_plan(m, columns) == [[8192] * 2]
        assert m.indexer_share_plan is None


def _budget(monkeypatch, elems):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (
        dsa_indexer,
    )

    monkeypatch.setattr(dsa_indexer, "_INDEXER_MQA_LOGITS_ELEM_BUDGET", elems)


def test_pieces_stay_inside_the_owner_tiles(monkeypatch):
    # 1000 rows over 10 keys, 300-row owner tiles (budget 3000): 128-row pieces from each tile's start.
    _budget(monkeypatch, 3000)
    assert share.chunk_pieces(1000, 10, 128) == [
        (0, 128, 300), (128, 256, 300), (256, 300, 300),
        (300, 428, 300), (428, 556, 300), (556, 600, 300),
        (600, 728, 300), (728, 856, 300), (856, 900, 300),
        (900, 1000, 100),
    ]  # fmt: skip


def _loads(plan, work, unit=1 << 20):
    """Per-rank indexer work after the plan: own chunks kept plus the pieces scored."""
    load = [w * unit for w in work]
    for e in plan.entries:
        load[e.owner] -= e.rows * e.keys
    for p in plan.pieces:
        load[p.rank] += (p.hi - p.lo) * plan.entries[p.entry].keys
    return load


def test_one_long_rank_spreads_over_the_spare_ranks(monkeypatch):
    """skew1: one rank at 8192 x 663k, fifteen at 8192 x 139k. The pieces fill the light ranks and the owner itself
    (its own load without the chunk is zero); every rank ends far below the owner's unshared work."""
    _budget(monkeypatch, 1 << 31)
    long_c, short_c = (8192, 663_552), (8192, 139_264)
    work = [-(-long_c[0] * long_c[1] // (1 << 20))] + [
        -(-short_c[0] * short_c[1] // (1 << 20))
    ] * 15
    columns = [[1] * 16, work, [long_c[0]] + [0] * 15, [long_c[1]] + [0] * 15, [0] * 16, [0] * 16]
    plan = share.build_plan(columns, 0, [0])
    assert [(e.owner, e.rows) for e in plan.entries] == [(0, 8192)]
    assert all(p.hi - p.lo <= 512 for p in plan.pieces) and len({p.rank for p in plan.pieces}) >= 8
    load = _loads(plan, work)
    assert max(load) < 0.4 * long_c[0] * long_c[1]
    # The owner keeps the rows of the pieces it scores; only the other ranks' rows travel back.
    own = sum(p.hi - p.lo for p in plan.pieces if p.rank == 0)
    assert own > 0 and plan.rows_sent_by(0) == 0
    assert sum(plan.rows_sent_by(r) for r in range(16)) == long_c[0] - own


def test_a_loaded_rank_that_keeps_its_chunk_takes_no_piece(monkeypatch):
    """ramp: the gate takes the top ranks; a rank above the mean that kept its own chunk gets no piece."""
    _budget(monkeypatch, 1 << 31)
    keys = [131_072 + 34_816 * r for r in range(16)]
    work = [-(-8192 * k // (1 << 20)) for k in keys]
    columns = [[1] * 16, work, [8192] * 16, keys, [0] * 16, [0] * 16]
    plan = share.build_plan(columns, 3, [0])
    owners = {e.owner for e in plan.entries}
    mean = sum(work) / 16
    kept_above_mean = [r for r in range(16) if r not in owners and work[r] > mean]
    assert owners and kept_above_mean
    assert not {p.rank for p in plan.pieces} & set(kept_above_mean)
    load = _loads(plan, work)
    assert max(load) <= max(work[r] * (1 << 20) for r in kept_above_mean)


@inputs.on_sm103
@pytest.mark.parametrize(
    "world,tile",
    [(16, 300), (3, 140), (5, 200), (7, 1024)],
    ids=[
        "tier2-tiles-128-row-pieces",
        "tier0-tiles-64-row-pieces",
        "tier1-tiles-whole",
        "one-tile-whole",
    ],
)
def test_shared_rows_match_the_owner(monkeypatch, world, tile):
    """Every piece of a long-prefix chunk, scored alone with its owner tile's Top-K plan, equals the owner's own
    rows bit for bit (FP8 index keys, the perf targets' GVR prefill Top-K)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (
        dsa_indexer,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_params import (
        K31DSAParams,
    )

    piece_rows = {300: 128, 140: 64, 200: 512, 1024: 1024}[tile]
    params = K31DSAParams(
        index_n_heads=32, index_head_dim=128, index_topk=2048, indexer_k_dtype="fp8"
    )
    # forward_prepared runs on prequantized inputs: the projection weights stay unused.
    mla = SimpleNamespace(hidden_size=128, q_lora_rank=128)
    indexer = dsa_indexer.K31Indexer(mla, params, torch.bfloat16)
    indexer._k31_perf = True

    rows, cached = 1024, 40_000
    gen = torch.Generator().manual_seed(1000 * world + tile)

    ks, ke, num_keys = inputs.chunk_windows([(cached, rows)])
    q = inputs.queries(rows, "randn", gen)
    k, k_scale = inputs.keys(num_keys, "randn", gen)
    k, k_scale = k.cuda(), k_scale.cuda()
    weights = inputs.weights(rows, "randn", gen)
    monkeypatch.setattr(dsa_indexer, "_INDEXER_MQA_LOGITS_ELEM_BUDGET", tile * num_keys)
    assert indexer.prefill_q_tile(rows, num_keys) == min(tile, rows)

    owner = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    for c0 in range(0, rows, tile):
        c1 = min(c0 + tile, rows)
        indexer.forward_prepared(
            q[c0:c1], k, k_scale, weights[c0:c1], ks[c0:c1], ke[c0:c1], owner[c0:c1]
        )
    # Every piece on its own (as some rank scores it), written back at its rows.
    shared = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    for lo, hi, tier in share.chunk_pieces(rows, num_keys, piece_rows):
        out = torch.full((hi - lo, 2048), -7, dtype=torch.int32, device="cuda")
        share.score_piece(indexer, q, k, k_scale, weights, ks, ke, out, lo, hi, tier)
        shared[lo:hi] = out
    assert torch.equal(shared, owner)
