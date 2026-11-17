# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certify `comm/moe_alltoall`: the round plan on CPU, and the dispatch and combine cells on
4 GPUs of one NVLink domain (a GB200 or GB300 tray).

Every rank builds every rank's rows from one seed convention, so each rank states on its own
what it should receive and what its combine should return, without a second collective. A
received row is matched to the token it came from by its routing weights, which are unique
per token: where a row lands within its source's block is the transport's business. Payloads
are compared by value, not by byte: for small rounds the ranks agree on a sentinel transport,
whose senders may rewrite a payload word into another encoding of the same value (a NaN into
another NaN, an e2m1 -0 into +0).

The combine check runs a stand-in expert: a rank's finalized row is the token's base vector
times the token's routing weights on the slots this rank holds, so a token's combined row is
its base vector times all its weights. Rows that hold no token for the rank are filled with a
large value the combine must never read.
"""

import pickle
import sys
from typing import Iterator

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from tensorrt_llm._torch.modeling_v2.catalog.comm.moe_alltoall import (
    INVALID_SLOT,
    MoeAlltoallDispatch,
    alltoall_rounds,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)

WORLD = 4
HIDDEN = 7168
EPLB_EXPERTS = 256
#: What a rank writes into the rows it received that hold no token for it.
_NOT_A_TOKEN = 3.0e4


# --- the round plan (CPU) -----------------------------------------------------------------


@pytest.mark.cpu_only
def test_every_rank_runs_every_round_with_its_rows_or_none():
    assert alltoall_rounds([64, 3, 0, 17], 16) == [
        [16, 3, 0, 16],
        [16, 0, 0, 1],
        [16, 0, 0, 0],
        [16, 0, 0, 0],
    ]
    assert alltoall_rounds([5, 5], 8) == [[5, 5]]
    assert alltoall_rounds([8, 8], 8) == [[8, 8]]
    assert alltoall_rounds([9, 0], 8) == [[8, 0], [1, 0]]


@pytest.mark.cpu_only
def test_a_forward_without_rows_still_runs_one_round():
    assert alltoall_rounds([0, 0, 0], 4) == [[0, 0, 0]]


@pytest.mark.cpu_only
def test_the_rounds_cover_every_row_once():
    counts = [1000, 1, 513, 512]
    rounds = alltoall_rounds(counts, 512)
    assert [sum(r[i] for r in rounds) for i in range(len(counts))] == counts
    assert all(max(r) <= 512 for r in rounds)


# --- the cells (4 GPUs) -------------------------------------------------------------------


def _rows(payload: str, rows: int, slots: int, top_k: int, rank: int, seed: int) -> dict:
    """Rank `rank`'s rows of a cell, on CPU. The last `pad` rows of rank 1 go to no rank."""
    g = torch.Generator().manual_seed(seed * 1009 + rank)
    ids = torch.empty((rows, top_k), dtype=torch.int32)
    for r in range(rows):
        ids[r] = torch.randperm(slots, generator=g)[:top_k].to(torch.int32)
    base = (torch.rand(rows, HIDDEN, generator=g) - 0.5).to(torch.bfloat16)
    if payload == "NVFP4":
        hidden = torch.randint(0, 256, (rows, HIDDEN // 2), generator=g, dtype=torch.uint8)
        sf = torch.randint(0, 256, (rows, HIDDEN // 16), generator=g, dtype=torch.uint8)
        weights = torch.rand(rows, top_k, generator=g).to(torch.bfloat16)
    else:
        hidden, sf = base, None
        weights = torch.rand(rows, top_k, generator=g)
    pad = 2 if rank == 1 and rows >= 2 else 0
    if pad:
        ids[-pad:] = INVALID_SLOT
    return dict(ids=ids, base=base, hidden=hidden, sf=sf, weights=weights, pad=pad)


def _e2m1(codes: torch.Tensor) -> torch.Tensor:
    """NVFP4 codes `[..., hidden // 2]` uint8 as their e2m1 nibbles, -0 read as +0."""
    nibbles = torch.stack([codes & 0xF, codes >> 4], -1).flatten(-2)
    return torch.where(nibbles == 8, torch.zeros_like(nibbles), nibbles)


def _same_value(a: torch.Tensor, b: torch.Tensor, payload: str) -> bool:
    """Whether two payload rows hold the same values."""
    if payload == "NVFP4":
        return torch.equal(_e2m1(a), _e2m1(b))
    return torch.allclose(a.float(), b.float(), rtol=0, atol=0, equal_nan=True)


def _same_scales(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two rows of e4m3 block scales hold the same values."""
    fa, fb = (t.view(torch.float8_e4m3fn).float() for t in (a, b))
    return torch.allclose(fa, fb, rtol=0, atol=0, equal_nan=True)


def _worker(case: dict) -> dict:
    """One rank of a cell: dispatch, check what arrived, combine, check what came back."""
    import torch

    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    from tensorrt_llm._torch.modeling_v2.catalog.comm.moe_alltoall import (
        moe_alltoall_combine,
        moe_alltoall_dispatch,
        moe_alltoall_gathered_statistic,
        moe_alltoall_workspace,
    )
    from tensorrt_llm.mapping import Mapping

    rank = MPI.COMM_WORLD.Get_rank()
    torch.cuda.set_device(rank)
    MnnvlMemory.initialize()
    if not all(MPI.COMM_WORLD.allgather(MnnvlMemory.supports_mnnvl())):
        return {"rank": rank, "skip": "the alltoall needs multi-node NVLink on every rank"}

    spec = case["spec"]
    payload, slots, top_k = spec["payload"], spec["slots"], spec["top_k"]
    counts = list(spec["rows"])
    local_slots = slots // WORLD
    every = [_rows(payload, n, slots, top_k, r, case["seed"]) for r, n in enumerate(counts)]
    mine = every[rank]
    mapping = Mapping(rank=rank, world_size=WORLD, tp_size=WORLD, moe_ep_size=WORLD)
    statistic = None
    if case["eplb"]:
        g = torch.Generator().manual_seed(case["seed"] + 77 * rank)
        statistic = torch.randint(0, 50, (EPLB_EXPERTS,), generator=g, dtype=torch.int32)
    a2a = moe_alltoall_workspace(
        mapping,
        num_slots=slots,
        top_k=top_k,
        max_tokens_per_rank=max(max(counts), 1),
        hidden_size=HIDDEN,
        payload=payload,
        eplb_num_experts=EPLB_EXPERTS if case["eplb"] else None,
    )
    dev = torch.device("cuda", rank)
    hq, hsf, rids, rw = moe_alltoall_dispatch(
        a2a,
        mine["hidden"].to(dev),
        None if mine["sf"] is None else mine["sf"].to(dev),
        mine["ids"].to(dev),
        mine["weights"].to(dev),
        counts,
        None if statistic is None else statistic.to(dev),
    )
    gathered = moe_alltoall_gathered_statistic(a2a) if case["eplb"] else None

    # Which source token each received row holds, by its routing weights.
    hq, rids, rw = hq.cpu(), rids.cpu(), rw.cpu()
    hsf = None if hsf is None else hsf.cpu()
    index = {
        row.float().numpy().tobytes(): (src, t)
        for src, d in enumerate(every)
        for t, row in enumerate(d["weights"])
    }
    expected = MoeAlltoallDispatch().reference(rank, [d["ids"] for d in every], local_slots)
    received: dict[int, list[int]] = {src: [] for src in range(WORLD)}
    partial = torch.full((hq.shape[0], HIDDEN), _NOT_A_TOKEN, dtype=torch.bfloat16)
    lo, hi = rank * local_slots, (rank + 1) * local_slots
    errors = []
    for i in range(hq.shape[0]):
        if bool((rids[i] == INVALID_SLOT).all()):
            continue
        key = rw[i].float().numpy().tobytes()
        if key not in index:
            errors.append(f"row {i}: weights {rw[i].tolist()[:3]}... are no token's")
            continue
        src, t = index[key]
        received[src].append(t)
        d = every[src]
        if not torch.equal(rids[i], d["ids"][t]):
            errors.append(f"row {i}: slots {rids[i].tolist()} != {d['ids'][t].tolist()}")
        if not _same_value(hq[i], d["hidden"][t], payload):
            errors.append(f"row {i}: payload differs from rank {src} token {t}'s")
        if hsf is not None and not _same_scales(hsf[i], d["sf"][t]):
            errors.append(f"row {i}: block scales differ from rank {src} token {t}'s")
        on_mine = (d["ids"][t] >= lo) & (d["ids"][t] < hi)
        share = (d["weights"][t].float() * on_mine).sum()
        partial[i] = (d["base"][t].float() * share).to(torch.bfloat16)
    for src in range(WORLD):
        if sorted(received[src]) != expected[src]:
            errors.append(
                f"from rank {src} received tokens {sorted(received[src])}, expected {expected[src]}"
            )

    combined = moe_alltoall_combine(a2a, partial.to(dev)).cpu().float()
    want = mine["base"].float() * mine["weights"].float().sum(-1, keepdim=True)
    if mine["pad"]:
        want[-mine["pad"] :] = 0
    err = (combined - want).abs().max().item() if combined.numel() else 0.0
    del a2a
    return {
        "rank": rank,
        "errors": errors,
        "combine_err": err,
        "combine_scale": want.abs().max().item() if want.numel() else 1.0,
        "gathered": None if gathered is None else gathered.cpu(),
        "statistic": statistic,
    }


@pytest.fixture(scope="module")
def pool() -> Iterator[MPIPoolExecutor]:
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"needs {WORLD} GPUs")
    executor = MPIPoolExecutor(
        WORLD,
        env={"TRTLLM_MOE_COMM_TIMEOUT_SEC": "60", "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC": "120"},
    )
    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _run(pool: MPIPoolExecutor, case: dict) -> list[dict]:
    results = sorted(pool.map(_worker, [case] * WORLD), key=lambda r: r["rank"])
    skipped = [r["skip"] for r in results if "skip" in r]
    if skipped:
        assert len(skipped) == WORLD, "platform support differs across ranks"
        pytest.skip(skipped[0])
    return results


@pytest.mark.no_xdist
@pytest.mark.parametrize("cell", MoeAlltoallDispatch.CELLS, ids=lambda c: c.why[:40])
def test_certified_cells(pool, cell) -> None:
    """Every rank receives exactly the tokens with a slot in its window, payload, scales,
    slots and weights intact; every token comes back with all its weights summed, and a row
    sent to no rank comes back as zeros."""
    for r in _run(pool, dict(spec=dict(cell.spec), seed=11, eplb=False)):
        assert not r["errors"], f"rank {r['rank']}: " + "; ".join(r["errors"][:5])
        # bf16 rows, summed over at most top_k ranks: a few ulps of the largest value.
        assert r["combine_err"] <= 2**-6 * max(r["combine_scale"], 1.0), (r["rank"], r)


@pytest.mark.no_xdist
def test_a_dispatch_carries_every_ranks_histogram(pool) -> None:
    """Under online EPLB the forward's last dispatch carries every rank's histogram of the
    router's experts; each rank gets all of them back, in rank order."""
    spec = dict(payload="NVFP4", rows=(8, 0, 3, 8), slots=288, top_k=8)
    results = _run(pool, dict(spec=spec, seed=5, eplb=True))
    every = torch.stack([r["statistic"] for r in results])
    for r in results:
        assert not r["errors"], r["errors"][:5]
        assert torch.equal(r["gathered"].reshape(WORLD, EPLB_EXPERTS), every), r["rank"]
