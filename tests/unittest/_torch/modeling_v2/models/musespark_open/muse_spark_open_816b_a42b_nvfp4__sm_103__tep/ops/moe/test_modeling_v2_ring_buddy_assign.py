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
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
    ring_buddy_assign as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "ring_buddy_assign requires a CUDA device"


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _routing(rows: int, kind: str, rng: random.Random) -> list[list[int]]:
    """``rows`` top-8 rows of distinct global expert ids under one routing model."""
    out = []
    prev = None
    for _ in range(rows):
        if kind == "one_block":
            row = rng.sample(range(BLOCK), TOP_K)
        elif kind == "keep5" and prev is not None:
            kept = rng.sample(prev, 5)
            row = kept + rng.sample([e for e in range(EXPERTS) if e not in kept], TOP_K - 5)
        else:
            row = rng.sample(range(EXPERTS), TOP_K)
        out.append(row)
        prev = row
    return out


def _inputs(rows: int, kind: str, seed: int):
    rng = random.Random(seed)
    ids = _routing(rows, kind, rng)
    selected = torch.tensor(ids, dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(seed)
    weights = torch.rand((rows, TOP_K), generator=g, device="cuda", dtype=torch.float32)
    return selected, weights


def _executed(local_ids: torch.Tensor, rank: int) -> dict[int, int]:
    """slot -> global expert, for the slots ``rank`` executes."""
    out = {}
    for slot, local in enumerate(local_ids.reshape(-1).tolist()):
        if local < BLOCK:
            out[slot] = BLOCK * rank + local
        elif local < 2 * BLOCK:
            out[slot] = BLOCK * ((rank - 1) % RANKS) + local - BLOCK
    return out


def _check_partition(selected: torch.Tensor, per_rank: list[torch.Tensor]) -> list[int]:
    """Every slot runs on exactly one rank, as the expert it selected; returns per-rank expert counts."""
    flat = selected.reshape(-1).tolist()
    owner = {}
    experts_on = []
    for rank, local_ids in enumerate(per_rank):
        executed = _executed(local_ids, rank)
        for slot, expert in executed.items():
            assert slot not in owner, f"slot {slot} executed by ranks {owner[slot]} and {rank}"
            assert expert == flat[slot], (
                f"rank {rank} runs expert {expert} for slot {slot}, which selected {flat[slot]}"
            )
            assert expert // BLOCK in (rank, (rank - 1) % RANKS), (
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
    for trial in range(4):
        selected, weights = _inputs(spec["rows"], spec["routing"], _seed(f"{spec}-{trial}"))
        per_rank = []
        stats_seen = None
        for rank in range(RANKS):
            local_ids, weights_bf16, stats = op(selected, weights, rank, balance)
            ref_ids, ref_w, ref_stats = op.reference(selected, weights, rank, balance)
            assert torch.equal(local_ids, ref_ids), f"rank {rank}: ids differ from the reference"
            assert torch.equal(weights_bf16, ref_w)
            assert torch.equal(stats, ref_stats), f"rank {rank}: stats {stats} vs {ref_stats}"
            # Every rank must derive the same placement from the same routing.
            if stats_seen is None:
                stats_seen = stats
            assert torch.equal(stats, stats_seen)
            per_rank.append(local_ids)
        experts_on = _check_partition(selected, per_rank)
        before, after = stats_seen[:RANKS].tolist(), stats_seen[RANKS:].tolist()
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
