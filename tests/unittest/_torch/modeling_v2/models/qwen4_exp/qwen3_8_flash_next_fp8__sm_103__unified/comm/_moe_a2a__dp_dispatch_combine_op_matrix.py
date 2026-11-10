# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certification matrix for the FP8 target's MoE all-to-all round trip (`moe_a2a_dispatch` and
`moe_a2a_combine` on the stock NVLink one-sided all-to-all) at dep2 and dep4 under mpirun, eagerly
and graph-replayed. Passes without checking where the platform has no MNNVL."""

import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import torch

assert torch.cuda.is_available(), "the MoE all-to-all requires CUDA devices"

_MPI_WORKER_FLAG = "--mpi-rank-worker"
DEADLINE_S = 900

#: The attention-DP group sizes this target serves (dep2, dep4).
CERTIFIED_WORLDS: Tuple[int, ...] = (2, 4)

HIDDEN = 2560
NUM_EXPERTS = 512
TOPK = 10

_MODELING_V2_TESTS = Path(__file__).resolve().parents[4]
_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp."
    "qwen3_8_flash_next_fp8__sm_103__unified"
)

# Bound inside the rank body: importing the entry pulls in tensorrt_llm, which calls MPI_Init at
# import, and the launcher must stay free of MPI and CUDA state.
entry: Any = None
validating: Any = None
MPI_RANK: Any = None
STRATEGY: Any = None
RANK = 0
WORLD = 1


def _gen(*key: int) -> torch.Generator:
    """Seeded alike on every rank for a key, so each rank can rebuild what any other rank holds."""
    seed = 0
    for k in key:
        seed = seed * 1000003 + int(k)
    return torch.Generator(device="cuda").manual_seed(seed % (2**62))


def _inputs(rows: int, src: int, salt: int) -> tuple:
    """Rank `src`'s rows and routing: bf16 rows, ten distinct experts a row, softmax weights."""
    g = _gen(salt, src)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    ids = torch.rand(rows, NUM_EXPERTS, generator=g, device="cuda").topk(TOPK, dim=1).indices
    w = torch.softmax(torch.randn(rows, TOPK, generator=g, device="cuda"), dim=1)
    return x, ids.to(torch.int32), w.to(torch.bfloat16)


def _window(rows, ids, w, rank: int) -> torch.Tensor:
    """Rank `rank`'s stand-in expert window: each row scaled by its in-window experts' weights times
    their ids, in fp32, rounded once; a row with no expert there (an empty slot too) is zero."""
    epr = NUM_EXPERTS // WORLD
    keep = (ids >= rank * epr) & (ids < (rank + 1) * epr)
    scale = (w.float() * keep * (ids.float() + 1) / NUM_EXPERTS).sum(-1, keepdim=True)
    out = scale * rows.float()
    return torch.where(keep.any(-1, keepdim=True), out, torch.zeros_like(out)).to(torch.bfloat16)


def _cells() -> list:
    return [c for c in entry.moe_a2a_dispatch.CELLS if c.spec["world"] == WORLD]


def _check_received(got, everyone, counts, what: str) -> None:
    """Received token slots equal the reference rows as a set: the kernel orders a block itself."""
    want = entry.moe_a2a_dispatch.reference(everyone, counts, NUM_EXPERTS // WORLD, rank=RANK)
    by_key = {}
    for x, ids, w in want:
        for i in range(ids.shape[0]):
            by_key[tuple(ids[i].tolist())] = (x[i], w[i])
    valid = (got.topk_ids != entry.NO_EXPERT).any(1)
    assert got.rows.shape[0] == WORLD * max(counts), (what, got.rows.shape, counts)
    assert int(valid.sum()) == len(by_key), (what, int(valid.sum()), len(by_key))
    for i in torch.nonzero(valid).flatten().tolist():
        key = tuple(got.topk_ids[i].tolist())
        assert key in by_key, f"{what}: slot {i} holds a row no source routed here"
        x, w = by_key[key]
        assert torch.equal(got.rows[i], x) and torch.equal(got.topk_weights[i], w), (what, i)


def _check_combined(out, mine, what: str) -> None:
    x, ids, w = mine
    want = entry.moe_a2a_combine.reference([_window(x, ids, w, r) for r in range(WORLD)])
    assert out.shape == want.shape, (what, out.shape, want.shape)
    torch.testing.assert_close(out.float(), want.float(), rtol=2 * 2**-8, atol=1e-3, msg=what)


def _round_trip(x, ids, w, counts):
    got = entry.moe_a2a_dispatch(STRATEGY, x, ids, w, counts)
    partials = _window(got.rows, got.topk_ids, got.topk_weights, RANK)
    return got, entry.moe_a2a_combine(STRATEGY, partials)


def check_cells() -> None:
    """Every cell of this world size: dispatch as a set against its reference, then the round
    trip through the stand-in window against the combine reference."""
    cells = _cells()
    assert cells, f"no cell at world {WORLD}"
    for i, cell in enumerate(cells):
        counts = list(cell.spec["counts"])
        everyone = [_inputs(n, r, 10 + i) for r, n in enumerate(counts)]
        with validating(entry.moe_a2a_dispatch):
            got, out = _round_trip(*everyone[RANK], counts)
        torch.cuda.synchronize()
        _check_received(got, everyone, counts, cell.why)
        _check_combined(out, everyone[RANK], cell.why)
        MPI_RANK.barrier()


def check_graph() -> None:
    """The round trip captured once at a uniform cell and replayed with fresh rows and routing."""
    cell = max(
        (c for c in _cells() if len(set(c.spec["counts"])) == 1), key=lambda c: c.spec["counts"][0]
    )
    counts = list(cell.spec["counts"])
    static = [t.clone() for t in _inputs(counts[RANK], RANK, 50)]
    _round_trip(*static, counts)  # eager first: the graph replays what the warm call set up
    torch.cuda.synchronize()
    MPI_RANK.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got, out = _round_trip(*static, counts)
    for salt in (51, 52):
        everyone = [_inputs(n, r, salt) for r, n in enumerate(counts)]
        for dst, src in zip(static, everyone[RANK]):
            dst.copy_(src)
        MPI_RANK.barrier()
        graph.replay()
        torch.cuda.synchronize()
        _check_received(got, everyone, counts, f"replay {salt}")
        _check_combined(out, everyone[RANK], f"replay {salt}")
        MPI_RANK.barrier()
    del graph
    torch.cuda.synchronize()
    MPI_RANK.barrier()


CHECKS = (check_cells, check_graph)


def _run_one_mpi_rank() -> int:
    """One MPI rank's checks; a failing check aborts the MPI job, which ends its siblings."""
    global entry, validating, MPI_RANK, STRATEGY, RANK, WORLD
    os.environ.pop("TLLM_DISABLE_MPI", None)
    sys.path.insert(0, str(_MODELING_V2_TESTS))
    sys.path.insert(0, str(_MODELING_V2_TESTS / "comm"))
    import importlib

    import _lockstep
    from _validating import validating as _validating_cm

    MPI_RANK = _lockstep.Rank(_lockstep.parse_args(sys.argv[1:]))
    from tensorrt_llm.mapping import Mapping

    entry = importlib.import_module(f"{_TARGET}.ops.comm.moe_a2a__dp_dispatch_combine")
    validating = _validating_cm
    RANK, WORLD = MPI_RANK.rank, MPI_RANK.world
    mapping = Mapping(
        world_size=WORLD,
        rank=RANK,
        gpus_per_node=WORLD,
        tp_size=WORLD,
        moe_tp_size=1,
        moe_ep_size=WORLD,
        enable_attention_dp=True,
    )
    max_rows = max(max(c.spec["counts"]) for c in entry.moe_a2a_dispatch.CELLS)
    STRATEGY = entry.build_strategy(mapping, max_rows)
    if STRATEGY is None:
        print(f"[rank {RANK}] no MNNVL on this platform: nothing to certify", flush=True)
        MPI_RANK.MPI.Finalize()
        os._exit(0)
    _lockstep.run_checks(MPI_RANK, list(CHECKS))
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed over the MPI session", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    MPI_RANK.MPI.Finalize()
    # Skip static destructors: the symmetric workspace's teardown adds nothing to the verdict.
    os._exit(0)


def _world_size(argv: List[str]) -> int:
    if "--world-size" in argv:
        return int(argv[argv.index("--world-size") + 1])
    return len([d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()])


if __name__ == "__main__":
    if _MPI_WORKER_FLAG in sys.argv:
        sys.exit(_run_one_mpi_rank())
    import _launch  # beside this file

    WORLD_SIZE = _world_size(sys.argv)
    assert WORLD_SIZE in CERTIFIED_WORLDS, (WORLD_SIZE, CERTIFIED_WORLDS)
    _launch.mpirun_ranks(
        __file__, WORLD_SIZE, [_MPI_WORKER_FLAG, "--world-size", str(WORLD_SIZE)], DEADLINE_S
    )
    print(f"OK: {WORLD_SIZE} ranks over the MPI session", flush=True)
