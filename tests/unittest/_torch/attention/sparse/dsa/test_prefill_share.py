# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Planning of the attention-DP sharing of long DSA indexer prefill chunks."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse.dsa import prefill_share as ps


def _chunk(t0, t1, k0, k1, row_starts_zero=False):
    return SimpleNamespace(
        token_start=t0,
        token_end=t1,
        k_token_start=k0,
        k_token_end=k1,
        row_starts_zero=row_starts_zero,
    )


def _metadata(chunks, **kw):
    md = SimpleNamespace(
        indexer_prefill_chunks=chunks,
        num_contexts=len(chunks),
        num_ctx_tokens=sum(c.token_end - c.token_start for c in chunks),
        skip_indexer_for_ctx_reqs=False,
        indexer_prefill_q_split=None,
    )
    for k, v in kw.items():
        setattr(md, k, v)
    return md


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("TRTLLM_DSA_PREFILL_SHARE_MAX", "2")
    monkeypatch.setenv("TRTLLM_DSA_PREFILL_SHARE_MIN_KEYS", "100000")
    monkeypatch.setenv("TRTLLM_DSA_PREFILL_SHARE_MIN_ROWS", "1024")
    monkeypatch.setenv("TRTLLM_DSA_PREFILL_SHARE_IMBALANCE", "1.25")


def test_local_descriptors_offer_the_largest_eligible_chunks_in_batch_order():
    chunks = [
        _chunk(0, 4096, 0, 50000),  # too few keys
        _chunk(4096, 12288, 50000, 450000),  # 8192 x 400k
        _chunk(12288, 12800, 450000, 950000),  # too few rows
        _chunk(12800, 16384, 950000, 1250000),  # 3584 x 300k
        _chunk(
            16384, 32768, 1250000, 1450000, row_starts_zero=True
        ),  # 16384 x 200k (largest), one request
    ]
    desc, ids = ps.local_descriptors(_metadata(chunks))
    assert ids == [1, 4]
    work = sum((c.token_end - c.token_start) * (c.k_token_end - c.k_token_start) for c in chunks)
    # slot 1 (chunk 4) has every row start at its first key
    assert desc == [1, -(-work // (1 << 20)), 8192, 400000, 16384, 200000, 0b10]
    assert ps.num_descriptor_counts() == len(desc)


def test_ineligible_ranks_offer_nothing_and_block_the_plan():
    chunks = [_chunk(0, 8192, 0, 300000)]
    desc, ids = ps.local_descriptors(_metadata(chunks, skip_indexer_for_ctx_reqs=True))
    assert desc == [0, 0, 0, 0, 0, 0, 0] and ids == []
    other, _ = ps.local_descriptors(_metadata(chunks))
    columns = [list(c) for c in zip(desc, other)]
    assert ps.build_plan(columns, 0, []) is None


def test_plan_rows_cover_every_chunk_once():
    # rank 0 offers two chunks, rank 2 one, ranks 1 / 3 none; ranks 0 and 2 carry most of the work.
    per_rank = [
        [1, 6000, 8192, 400000, 5000, 120000, 0b10],
        [1, 100, 0, 0, 0, 0, 0],
        [1, 10000, 16384, 650000, 0, 0, 0b01],
        [1, 100, 0, 0, 0, 0, 0],
    ]
    columns = [list(c) for c in zip(*per_rank)]
    plan = ps.build_plan(columns, 2, [7])
    assert [(e.owner, e.slot, e.rows, e.keys, e.row_starts_zero) for e in plan.entries] == [
        (0, 0, 8192, 400000, False),
        (0, 1, 5000, 120000, True),
        (2, 0, 16384, 650000, True),
    ]
    for e in plan.entries:
        spans = [plan.row_range(e, r) for r in range(plan.world)]
        assert spans[0][0] == 0 and spans[-1][1] == e.rows
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert sum(plan.rows_of_rank(r) for r in range(4)) == 8192 + 5000 + 16384
    assert plan.local_chunks == [7]


def test_entry_layout_is_aligned():
    offs, size = ps.entry_layout(5000, 120001, 128, 4096, 128)
    assert size % 256 == 0 and all(o % 256 == 0 for o in offs.values())
    assert offs["s"] >= 120001 * 128 and offs["q"] >= offs["s"] + 120001 * 4


def test_balanced_steps_share_nothing():
    # Every rank near the mean: the gate (1.25 x mean) rejects all offers.
    per_rank = [
        [1, 1000, 8192, 300000, 0, 0, 0],
        [1, 900, 8192, 280000, 0, 0, 0],
        [1, 1100, 0, 0, 0, 0, 0],
    ]
    columns = [list(c) for c in zip(*per_rank)]
    assert ps.build_plan(columns, 0, [3]) is None


def test_gate_keeps_only_the_heavy_ranks(monkeypatch):
    per_rank = [
        [1, 5000, 8192, 600000, 0, 0, 0],
        [1, 1000, 8192, 220000, 0, 0, 0],
        [1, 100, 0, 0, 0, 0, 0],
    ]
    columns = [list(c) for c in zip(*per_rank)]
    plan = ps.build_plan(columns, 1, [2])
    assert [(e.owner, e.keys) for e in plan.entries] == [(0, 600000)]
    assert plan.local_chunks == []  # rank 1's offer was not taken
    monkeypatch.setenv("TRTLLM_DSA_PREFILL_SHARE_IMBALANCE", "0")
    plan = ps.build_plan(columns, 1, [2])
    assert [e.owner for e in plan.entries] == [0, 1] and plan.local_chunks == [2]


@pytest.mark.parametrize(
    "rows,tile,world", [(8192, 3484, 16), (4096, 4096, 16), (1030, 512, 3), (16384, 2048, 7)]
)
def test_tile_pieces_follow_the_owner_tiles(rows, tile, world):
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer

    plan = ps.SharePlan(world, 0, [ps.ShareEntry(0, 0, rows, 600000)])
    covered = []
    for r in range(world):
        lo, hi = plan.row_range(plan.entries[0], r)
        for t0, t1, tier_rows in Indexer.prefill_tile_pieces(lo, hi, rows, tile):
            tile_lo = t0 // tile * tile
            assert tile_lo <= t0 < t1 <= min(tile_lo + tile, rows)  # inside one owner tile
            assert tier_rows == min(tile_lo + tile, rows) - tile_lo  # that tile's row count
            covered += range(t0, t1)
    assert covered == list(range(rows))
