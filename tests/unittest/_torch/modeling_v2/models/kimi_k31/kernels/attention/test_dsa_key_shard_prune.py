# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 key-shard pruned exchange (kernels/attention/dsa_key_shard_prune).

Every rank's local scores are written straight into its workspace (the score battery below), each rank takes its
local Top-L with the CuTe DSL decode Top-K, and the three launches run. Every rank's rows must hold exactly the
row's global Top-2048 by (score desc, token asc) over the full row (a torch stable sort), every exchanged word must
be empty again after each call, and no poll may time out.

* emulated: ``W`` ranks on one GPU with plain buffers (each launch of every rank before the next launch of any);
* tray: four MPI ranks of one tray over a real multicast workspace (MPI-spawning: run under ``mpirun -n 1`` without
  the Slurm/PMIx environment);
* group: launched directly with 16 or 32 ranks (``srun --mpi=pmix -N 4|8 --ntasks-per-node=4 python -m pytest
  <this file> -k group``); skipped otherwise.
"""

import os
import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.dsa_key_shard_prune import (  # noqa: E501
    SENTINEL,
    TOKENS_PER_BLOCK,
    KeyShardPrunedExchange,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
        reason="certified on GB300 (sm_103)",
    ),
]

K = 2048
NEXT_N = 5
#: Score kinds of the battery.
KINDS = ("randn", "ties", "allequal", "pm0", "onerank", "cluster")


def _local_count(n: int, w: int, world: int) -> int:
    full = n // TOKENS_PER_BLOCK
    owned = max((full - w + world - 1) // world, 0)
    return owned * TOKENS_PER_BLOCK + (n % TOKENS_PER_BLOCK if full % world == w else 0)


def _kv_lens(batch: int, ctx: int, kind: str, gen: torch.Generator) -> list:
    if kind == "mixed":
        # rows of at most K keys, rows just above K and long rows in one call
        base = [1_500, 2_050, ctx, 2_000, ctx // 3, ctx]
        return [
            base[b % len(base)] + int(torch.randint(0, 64, (1,), generator=gen))
            for b in range(batch)
        ]
    return [ctx + int(torch.randint(0, 64, (1,), generator=gen)) for _ in range(batch)]


def _scores(kind: str, rows: int, n: int, world: int, seed: int) -> torch.Tensor:
    """[rows, n] fp32 full-row scores (the same on every rank)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "ties":
        s = (torch.randn(rows, n, device="cuda", generator=g) * 2).round()
    elif kind == "allequal":
        s = torch.full((rows, n), 0.5, device="cuda")
    elif kind == "pm0":
        s = torch.relu(torch.randn(rows, n, device="cuda", generator=g) - 2.5)
        flip = torch.rand(rows, n, device="cuda", generator=g) < 0.5
        s = torch.where((s == 0) & flip, torch.full_like(s, -0.0), s)
    else:
        s = torch.randn(rows, n, device="cuda", generator=g)
    blk = torch.arange(n, device="cuda") // TOKENS_PER_BLOCK
    if kind == "onerank":
        s[:, blk % world == 1] += 50.0
    if kind == "cluster":
        starts = torch.randint(
            0, max(int(blk[-1]) - 8, 1), (24,), generator=g, device="cuda"
        ).tolist()
        for b0 in starts:
            s[:, b0 * TOKENS_PER_BLOCK : (b0 + 8) * TOKENS_PER_BLOCK] += 3.0
    return s


def _row_lens(kv: list) -> list:
    return [kv[r // NEXT_N] - NEXT_N + 1 + r % NEXT_N for r in range(len(kv) * NEXT_N)]


def _reference(full: torch.Tensor, ns: list) -> list:
    """Each row's Top-K token set (sorted), by (score desc, token asc) over its first n tokens."""
    bits = full.contiguous().view(torch.int32).long()
    ordered = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    out = []
    for r, n in enumerate(ns):
        if n <= K:
            out.append(torch.arange(n, device="cuda", dtype=torch.int32))
            continue
        key = (ordered[r, :n] << 32) | (4294967295 - torch.arange(n, device="cuda"))
        out.append(torch.topk(key, K).indices.sort().values.int())
    return out


def _local(full: torch.Tensor, ns: list, w: int, world: int, width: int):
    """Rank ``w``'s [rows, width] local scores (its 64-token blocks, compacted) and its row lengths."""
    rows, n = full.shape
    p = torch.arange(n, device="cuda")
    own = ((p // TOKENS_PER_BLOCK) % world) == w
    col = ((p // TOKENS_PER_BLOCK) // world) * TOKENS_PER_BLOCK + p % TOKENS_PER_BLOCK
    out = torch.full((rows, width), float("-inf"), device="cuda")
    out[:, col[own]] = full[:, own]
    lens = torch.tensor([_local_count(x, w, world) for x in ns], dtype=torch.int32, device="cuda")
    return out, lens


def _bad_rows(out: torch.Tensor, ref: list) -> int:
    bad = 0
    for r, rr in enumerate(ref):
        o = out[r][out[r] >= 0].sort().values
        bad += int(o.numel() != rr.numel() or not torch.equal(o, rr))
    return bad


def _case(kind: str, world: int, batch: int, ctx: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    kv = _kv_lens(batch, ctx, kind, g)
    ns = _row_lens(kv)
    full = _scores("randn" if kind == "mixed" else kind, len(ns), max(ns), world, seed)
    return torch.tensor(kv, dtype=torch.int32, device="cuda"), ns, full


def _local_width(ctx: int, world: int) -> int:
    return -(-(_local_count(ctx + 64, 0, world) + TOKENS_PER_BLOCK) // 256) * 256


# ---------------------------------------------------------------------------
# Emulated ranks on one GPU
# ---------------------------------------------------------------------------


def _emulated(world: int, local_k: int, rows_max: int, width: int):
    need = KeyShardPrunedExchange.bytes_needed(rows_max, world, local_k, width)
    bufs = [torch.zeros(need + 4096, dtype=torch.uint8, device="cuda") for _ in range(world)]
    exs = [KeyShardPrunedExchange(bufs, w, rows_max, K, width, local_k) for w in range(world)]
    for ex in exs:
        ex.rearm()
    return exs


def _emulated_call(exs, kv, ns, full):
    world = len(exs)
    rows = len(ns)
    outs = []
    for w, ex in enumerate(exs):
        loc, lens = _local(full, ns, w, world, ex.local_width)
        ex.local_logits[:rows].copy_(loc)
        topk = torch.empty(rows, ex.local_k, dtype=torch.int32, device="cuda")
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(
            ex.local_logits[:rows], lens, topk, ex.local_k, 1
        )
        ex.ladder(topk, kv, NEXT_N)
        outs.append(torch.full((rows, K), -7, dtype=torch.int32, device="cuda"))
    for ex in exs:
        ex.push(rows, kv, NEXT_N)
    for ex, out in zip(exs, outs):
        ex.merge(out, kv, NEXT_N)
    torch.cuda.synchronize()
    return outs


@pytest.mark.parametrize("world,local_k", [(16, 512), (32, 256), (8, 512)])
@pytest.mark.parametrize("kind", KINDS + ("mixed",))
def test_emulated_battery(kind, world, local_k):
    """Each score kind at 16 / 32 / 8 ranks: every rank's rows = the reference set; fallback where required."""
    ctx = 40_000
    exs = _emulated(world, local_k, 64, _local_width(ctx, world))
    kv, ns, full = _case(
        kind, world, 8, ctx, 1000 + world + KINDS.index(kind) if kind in KINDS else 77
    )
    ref = _reference(full, ns)
    outs = _emulated_call(exs, kv, ns, full)
    for w, (ex, out) in enumerate(zip(exs, outs)):
        assert _bad_rows(out, ref) == 0, f"rank {w}"
        st = ex.stats()
        assert st[0] == 1 and st[3] == 0, st
        assert bool((ex.lamport == SENTINEL).all()), f"rank {w}: workspace not re-armed"
    if kind == "onerank":
        assert exs[0].stats()[2] > 0, "one rank holding the Top-K must take the exact re-selection"
    if kind == "randn":
        assert exs[0].stats()[2] == 0, "random scores never need the re-selection"


def test_emulated_back_to_back_rearm():
    """Calls of 40, 10, 40 and 15 rows back to back at 16 ranks: rows exact, workspace empty after each call."""
    world, local_k, ctx = 16, 512, 30_000
    exs = _emulated(world, local_k, 64, _local_width(ctx, world))
    for call, (batch, kind) in enumerate(((8, "randn"), (2, "ties"), (8, "cluster"), (3, "mixed"))):
        kv, ns, full = _case(kind, world, batch, ctx, 500 + call)
        ref = _reference(full, ns)
        outs = _emulated_call(exs, kv, ns, full)
        for w, (ex, out) in enumerate(zip(exs, outs)):
            assert _bad_rows(out, ref) == 0, f"call {call} rank {w}"
            assert ex.stats()[0] == call + 1
            assert bool((ex.lamport == SENTINEL).all()), f"call {call} rank {w}: not re-armed"


# ---------------------------------------------------------------------------
# Real ranks over a multicast workspace
# ---------------------------------------------------------------------------


def _rank_check(world: int, local_k: int, rank: int, kinds: tuple, ctx: int) -> dict:
    """Collective on ``world`` ranks: every kind, eager twice and under CUDA-graph replay."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.dsa_key_shard_prune import (  # noqa: E501
        get_key_shard_pruned_exchange,
    )
    from tensorrt_llm.mapping import Mapping

    # Multicast over the fabric handles on one tray too (the production layouts span trays).
    os.environ["TRTLLM_FORCE_MNNVL_AR"] = "1"
    torch.cuda.set_device(rank % 4)
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=4, tp_size=world)
    width = _local_width(ctx, world)
    ex = get_key_shard_pruned_exchange(mapping, 64, K, width, local_k)
    ex.max_spin = 1 << 22  # a broken exchange fails the test instead of hanging it
    comm = MPI.COMM_WORLD
    result = dict(rank=rank, bad=0, timeouts=0, rearm=True, fallback=0)
    for i, kind in enumerate(kinds):
        kv, ns, full = _case(kind, world, 8, ctx, 2000 + i)
        ref = _reference(full, ns)
        rows = len(ns)
        loc, lens = _local(full, ns, rank, world, ex.local_width)
        ex.local_logits[:rows].copy_(loc)
        topk = torch.empty(rows, local_k, dtype=torch.int32, device="cuda")
        out = torch.full((rows, K), -7, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        comm.Barrier()

        def call():
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                ex.local_logits[:rows], lens, topk, local_k, 1
            )
            ex.exchange(topk, kv, NEXT_N, out)

        for _ in range(2):
            call()
            torch.cuda.synchronize()
            result["bad"] += _bad_rows(out, ref)
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
            call()
        comm.Barrier()
        out.fill_(-7)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        result["bad"] += _bad_rows(out, ref)
        comm.Barrier()
        result["rearm"] &= bool((ex.lamport == SENTINEL).all())
        del graph
    st = ex.stats()
    result["timeouts"] = st[3]
    result["fallback"] = st[2]
    return result


TRAY = 4


def _tray_rank(kinds: tuple) -> dict:
    import tensorrt_llm

    return _rank_check(TRAY, 1024, tensorrt_llm.mpi_rank(), kinds, 20_000)


@pytest.mark.skipif(torch.cuda.device_count() < TRAY, reason="four GPUs of one tray")
@pytest.mark.parametrize("mpi_pool_executor", [TRAY], indirect=True)
def test_tray_ranks(mpi_pool_executor):
    """Four real ranks of one tray (multicast workspace): every kind, repeated and under graph replay."""
    results = list(mpi_pool_executor.map(_tray_rank, *zip(*[(KINDS,)] * TRAY)))
    for r in results:
        assert r["bad"] == 0 and r["timeouts"] == 0 and r["rearm"], r


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() not in (16, 32),
    reason="launched directly with 16 or 32 ranks (srun --mpi=pmix ... python -m pytest -k group)",
)
def test_group_ranks():
    """16 / 32 real ranks over MNNVL (the TP16 / TP32 layouts): every kind, repeated and under graph replay."""
    world = MPI.COMM_WORLD.Get_size()
    res = _rank_check(world, 512 if world == 16 else 256, MPI.COMM_WORLD.Get_rank(), KINDS, 120_000)
    everyone = MPI.COMM_WORLD.allgather(res)
    if int(os.environ.get("K31_TEST_VERBOSE", "0")):
        print(everyone[0])
    for r in everyone:
        assert r["bad"] == 0 and r["timeouts"] == 0 and r["rearm"], r
