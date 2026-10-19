# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Muse Spark Open tep8 ``ring_buddy_assign`` entry (``replicated_experts``).

The entry is integer bookkeeping, so every cell is gated exactly: against the
entry's own ``reference`` on every rank, on the partition property the
finalize all-reduce relies on (each (token, slot) pair executed by exactly one
rank, through the stacked id that names the same global expert), and on the
placement being the ring optimum.
"""

import random
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
    BLOCK,
    EXPERTS,
    RANKS,
    TOP_K,
    balance_ring,
    block_of,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
    ring_buddy_assign as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "ring_buddy_assign requires a CUDA device"


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _routing(rows: int, kind: str, rng: random.Random, block: int = BLOCK) -> list[list[int]]:
    """``rows`` top-8 rows of distinct global expert ids under one routing model.

    ``one_block`` draws from block 0 on the tep8 ring, and from the last block's
    upper half on the tep4 ring, so a 64-bit presence word's high half is exercised.
    """
    out = []
    prev = None
    for _ in range(rows):
        if kind == "one_block":
            pool = range(block) if block == BLOCK else range(EXPERTS - block // 2, EXPERTS)
            row = rng.sample(pool, TOP_K)
        elif kind == "keep5" and prev is not None:
            kept = rng.sample(prev, 5)
            row = kept + rng.sample([e for e in range(EXPERTS) if e not in kept], TOP_K - 5)
        else:
            row = rng.sample(range(EXPERTS), TOP_K)
        out.append(row)
        prev = row
    return out


def _inputs(rows: int, kind: str, seed: int, block: int = BLOCK):
    rng = random.Random(seed)
    ids = _routing(rows, kind, rng, block)
    selected = torch.tensor(ids, dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(seed)
    weights = torch.rand((rows, TOP_K), generator=g, device="cuda", dtype=torch.float32)
    return selected, weights


def _executed(local_ids: torch.Tensor, rank: int, ranks: int = RANKS) -> dict[int, int]:
    """slot -> global expert, for the slots ``rank`` executes."""
    block = block_of(ranks)
    out = {}
    for slot, local in enumerate(local_ids.reshape(-1).tolist()):
        if local < block:
            out[slot] = block * rank + local
        elif local < 2 * block:
            out[slot] = block * ((rank - 1) % ranks) + local - block
    return out


def _check_partition(
    selected: torch.Tensor, per_rank: list[torch.Tensor], ranks: int = RANKS
) -> list[int]:
    """Every slot runs on exactly one rank, as the expert it selected; returns per-rank expert counts."""
    block = block_of(ranks)
    flat = selected.reshape(-1).tolist()
    owner = {}
    experts_on = []
    for rank, local_ids in enumerate(per_rank):
        executed = _executed(local_ids, rank, ranks)
        for slot, expert in executed.items():
            assert slot not in owner, f"slot {slot} executed by ranks {owner[slot]} and {rank}"
            assert expert == flat[slot], (
                f"rank {rank} runs expert {expert} for slot {slot}, which selected {flat[slot]}"
            )
            assert expert // block in (rank, (rank - 1) % ranks), (
                f"rank {rank} holds no copy of expert {expert}"
            )
            owner[slot] = rank
        experts_on.append(len(set(executed.values())))
    assert sorted(owner) == list(range(len(flat))), "some slot is executed by no rank"
    return experts_on


@pytest.mark.parametrize("balance", [True, False], ids=["balanced", "owner_only"])
@pytest.mark.parametrize("cell", op.CELLS, ids=[str(c.spec) for c in op.CELLS])
def test_cells(cell, balance, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    ranks = spec.get("ranks", RANKS)
    block = block_of(ranks)
    for trial in range(4):
        selected, weights = _inputs(spec["rows"], spec["routing"], _seed(f"{spec}-{trial}"), block)
        per_rank = []
        stats_seen = None
        for rank in range(ranks):
            local_ids, weights_bf16, stats = op(selected, weights, rank, balance, ranks=ranks)
            ref_ids, ref_w, ref_stats = op.reference(selected, weights, rank, balance, ranks=ranks)
            assert torch.equal(local_ids, ref_ids), f"rank {rank}: ids differ from the reference"
            assert torch.equal(weights_bf16, ref_w)
            assert torch.equal(stats, ref_stats), f"rank {rank}: stats {stats} vs {ref_stats}"
            # Every rank must derive the same placement from the same routing.
            if stats_seen is None:
                stats_seen = stats
            assert torch.equal(stats, stats_seen)
            per_rank.append(local_ids)
        experts_on = _check_partition(selected, per_rank, ranks)
        before, after = stats_seen[:ranks].tolist(), stats_seen[ranks:].tolist()
        assert experts_on == after
        assert sum(after) == sum(before) == len(set(selected.reshape(-1).tolist()))
        if balance:
            assert max(after) == balance_ring(before)[0]
            assert max(after) <= max(before)
        else:
            assert after == before


def _optimal_flows(load: list[int]) -> tuple[int, list[tuple[int, ...]]]:
    """Smallest max load and every flow vector reaching it (exhaustive; small loads only)."""
    import itertools

    best, flows = max(load), []
    for x in itertools.product(*[range(v + 1) for v in load]):
        peak = max(load[r] - x[r] + x[(r - 1) % RANKS] for r in range(RANKS))
        if peak < best:
            best, flows = peak, []
        if peak == best:
            flows.append(x)
    return best, flows


def test_ring_balance_is_optimal():
    """The closed form reaches the min-max ring flow with the pointwise smallest moves."""
    rng = random.Random(3)
    for _ in range(25):
        load = [rng.randint(0, 3) for _ in range(RANKS)]
        best, flows = _optimal_flows(load)
        t, x = balance_ring(load)
        assert t == best, load
        assert tuple(x) in flows, (load, x)
        assert all(all(a <= b for a, b in zip(x, f)) for f in flows), (load, x)


def test_graph_replay_follows_inputs():
    """Captured once, replayed on new routing: a pure function of its inputs."""
    selected, weights = _inputs(8, "keep5", 11)
    with validating(op):
        op(selected, weights, 3, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(selected, weights, 3, True)
    for seed in range(5):
        new_sel, new_w = _inputs(8, "uniform", 100 + seed)
        selected.copy_(new_sel)
        weights.copy_(new_w)
        graph.replay()
        ref = op.reference(selected, weights, 3, True)
        for got, want in zip(out, ref):
            assert torch.equal(got, want)


@pytest.mark.parametrize(
    "bad, match",
    [
        (dict(selected_dtype=torch.int64), "int32"),
        (dict(rows=17), "at most"),
        (dict(rank=8), "rank must be"),
        (dict(weights_dtype=torch.bfloat16), "float32"),
    ],
)
def test_is_valid_rejects(bad, match):
    rows = bad.get("rows", 8)
    selected, weights = _inputs(min(rows, 16), "uniform", 5)
    if rows > 16:
        selected = torch.cat([selected, selected[:1]])
        weights = torch.cat([weights, weights[:1]])
    selected = selected.to(bad.get("selected_dtype", torch.int32))
    weights = weights.to(bad.get("weights_dtype", torch.float32))
    with pytest.raises(AssertionError, match=match):
        op.is_valid(selected, weights, bad.get("rank", 0), True)


@pytest.mark.parametrize("rows", [1, 5, 8, 16])
def test_static_placement(rows):
    """The static placement: local experts 28..31 of every block on the successor, at every step."""
    for trial in range(4):
        selected, weights = _inputs(rows, "keep5", _seed(f"static-{rows}-{trial}"))
        per_rank = []
        for rank in range(RANKS):
            got = op(selected, weights, rank, False, 28)
            want = op.reference(selected, weights, rank, False, 28)
            for a, b in zip(got, want):
                assert torch.equal(a, b), f"rank {rank}: kernel differs from the reference"
            per_rank.append(got[0])
        _check_partition(selected, per_rank)
        for rank, local_ids in enumerate(per_rank):
            for slot, expert in _executed(local_ids, rank).items():
                moved = expert % BLOCK >= 28
                assert expert // BLOCK == ((rank - 1) % RANKS if moved else rank), (slot, expert)


@pytest.mark.parametrize("rows", [1, 8, 16])
def test_tep4_owner_only_ids_address_the_own_block_window(rows):
    """tep4 owner-only: a rank's own experts are ``g - 64 * rank`` (0..63), every other id is
    past the 128-expert two-window range, so a 64-expert own-block call (``ring_buddy``'s
    ``windows=1``) and a 128-expert two-window call execute the same slots."""
    ranks, block = 4, block_of(4)
    for trial in range(4):
        selected, weights = _inputs(rows, "keep5", _seed(f"tep4-owner-{rows}-{trial}"), block)
        for rank in range(ranks):
            ids, _, _ = op(selected, weights, rank, False, ranks=ranks)
            for g, local in zip(selected.reshape(-1).tolist(), ids.reshape(-1).tolist()):
                if g // block == rank:
                    assert local == g - block * rank
                else:
                    assert 2 * block <= local < EXPERTS


@pytest.mark.parametrize("rows", [1, 8, 16])
def test_tep4_static_placement(rows):
    """The static placement on the tep4 ring: local experts 56..63 of every block on the successor."""
    ranks, block = 4, block_of(4)
    for trial in range(4):
        selected, weights = _inputs(rows, "keep5", _seed(f"tep4-static-{rows}-{trial}"), block)
        per_rank = []
        for rank in range(ranks):
            got = op(selected, weights, rank, False, 56, ranks=ranks)
            want = op.reference(selected, weights, rank, False, 56, ranks=ranks)
            for a, b in zip(got, want):
                assert torch.equal(a, b), f"rank {rank}: kernel differs from the reference"
            per_rank.append(got[0])
        _check_partition(selected, per_rank, ranks)


def test_tep4_ring_balance_is_optimal():
    """The closed form on the 4-rank ring reaches the min-max flow."""
    import itertools

    rng = random.Random(4)
    for _ in range(25):
        load = [rng.randint(0, 6) for _ in range(4)]
        best = min(
            max(load[r] - x[r] + x[(r - 1) % 4] for r in range(4))
            for x in itertools.product(*[range(v + 1) for v in load])
        )
        assert balance_ring(load)[0] == best, load
