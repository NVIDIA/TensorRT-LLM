# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certification matrix for the FP8 target's `mnnvl_allreduce__tp_sum` under mpirun: each sum is
bitwise the IPC op's and within `_compare`'s band of the fp32 sum, eagerly and graph-replayed.
The MPI-disabled launch (`TLLM_DISABLE_MPI=1`) is not certified here."""

import argparse
import functools
import os
import sys
from pathlib import Path
from typing import Any

import torch

_MPI_WORKER_FLAG = "--mpi-rank-worker"
DEADLINE_S = 900

#: tests/unittest/_torch/modeling_v2: `_validating` and the shared `comm/` helpers.
_MODELING_V2_TESTS = Path(__file__).resolve().parents[4]

# Bound inside the rank body: module scope imports only torch and the standard library, since
# `_rank_job` executes it to read `DEADLINE_S`.
entry: Any = None
ipc: Any = None
mnnvl: Any = None
modeling: Any = None
validating: Any = None
WORKSPACE: dict = {}
MPI_RANK: Any = None
RANK = 0
WORLD = 1

#: Row counts a captured `tp_sum` meets inside the budget: decode batches 1..8 and the MTP-3
#: verify steps of 1..8 requests (4..32 rows).
GRAPH_ROWS = (1, 2, 3, 4, 8, 12, 16, 24, 32)


def _compare(out, ref) -> None:
    """`catalog/comm/allreduce`'s band: 5 ulp per element against the row scale, 1.25 ulp RMS."""
    from _validating import assert_within_ulp

    assert_within_ulp(out, ref, element_ulp=5.0, rms_ulp=1.25)


def _payload(rank: int, rows: int, seed: int) -> torch.Tensor:
    """One rank's partial: every rank can regenerate every other rank's from the seed."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    return torch.randn(rows, entry.HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)


def _reference(rows: int, seed: int) -> torch.Tensor:
    return entry.mnnvl_tp_sum.reference([_payload(r, rows, seed) for r in range(WORLD)])


def _exact(out, ref) -> float:
    return (out == ref).float().mean().item()


def check_certified_cells() -> None:
    cells = [c for c in entry.mnnvl_tp_sum.CELLS if c.spec["world"] == WORLD]
    assert cells, f"no cell is certified at world size {WORLD}"
    for i, cell in enumerate(cells):
        rows = cell.spec["rows"]
        for s in range(3):
            seed = 4000 + 31 * i + 1000 * s
            x = _payload(RANK, rows, seed)
            with validating(entry.mnnvl_tp_sum):
                out = mnnvl(x)
            twin = ipc(x)
            ref = _reference(rows, seed)
            torch.cuda.synchronize()
            assert out.shape == x.shape and out.dtype == x.dtype, (out.shape, out.dtype)
            assert torch.equal(out, twin), f"{cell.why}: not the IPC op's bits (seed {seed})"
            _compare(out, ref)
            if RANK == 0 and s == 0:
                print(
                    f"[world {WORLD}] {cell.why}: bitwise the IPC op; "
                    f"{100 * _exact(out, ref):.4f}% bit-exact to the fp32 sum rounded once",
                    flush=True,
                )
            MPI_RANK.barrier()


def check_cuda_graph_replay_matches_eager() -> None:
    """Captured once per row count, refilled with fresh partials and replayed: bitwise the
    eager call and the IPC op on those partials (the Lamport buffers rotate per call)."""
    for rows in GRAPH_ROWS:
        static = _payload(RANK, rows, 900 + rows)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                mnnvl(static)
        torch.cuda.current_stream().wait_stream(side)
        MPI_RANK.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mnnvl(static)
        MPI_RANK.barrier()
        for step in range(3):
            fresh = _payload(RANK, rows, 1900 + 10 * rows + step)
            static.copy_(fresh)
            graph.replay()
            eager = mnnvl(fresh)
            twin = ipc(fresh)
            torch.cuda.synchronize()
            assert torch.equal(captured, eager), f"{rows} rows step {step}: replay != eager"
            assert torch.equal(captured, twin), f"{rows} rows step {step}: replay != IPC op"
            MPI_RANK.barrier()
        if RANK == 0:
            print(f"[world {WORLD}] captured {rows} rows: replay == eager == IPC op", flush=True)
        del graph


CHECKS = (check_certified_cells, check_cuda_graph_replay_matches_eager)


def _run_one_mpi_rank() -> int:
    global entry, ipc, mnnvl, modeling, validating, WORKSPACE, MPI_RANK, RANK, WORLD
    os.environ.pop("TLLM_DISABLE_MPI", None)
    sys.path.insert(0, str(_MODELING_V2_TESTS))
    sys.path.insert(0, str(_MODELING_V2_TESTS / "comm"))
    import _lockstep
    from _validating import validating as _validating_cm

    MPI_RANK = _lockstep.Rank(_lockstep.parse_args(sys.argv[1:]))
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import (
        allreduce as allreduce_mod,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
        modeling as _modeling,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.comm import (  # noqa: E501
        mnnvl_allreduce__tp_sum as _entry,
    )
    from tensorrt_llm.functional import AllReduceFusionOp

    entry, modeling, validating = _entry, _modeling, _validating_cm
    RANK, WORLD = MPI_RANK.rank, MPI_RANK.world
    group = allreduce_mod.allreduce_group_args(MPI_RANK.mapping)
    ipc = functools.partial(
        allreduce_mod.allreduce,
        residual=None,
        norm_weight=None,
        scale=None,
        bias=None,
        op=int(AllReduceFusionOp.NONE),
        eps=1e-6,
        trigger_completion_at_end=False,
        **group,
    )
    WORKSPACE = entry.mnnvl_workspace_args(
        MPI_RANK.mapping, modeling._TP_SUM_MNNVL_MAX_ROWS, entry.HIDDEN, torch.bfloat16
    )
    mnnvl = functools.partial(entry.mnnvl_tp_sum, **WORKSPACE)

    code = _lockstep.run_checks(MPI_RANK, list(CHECKS))
    torch.cuda.synchronize()
    sys.stdout.flush()
    sys.stderr.flush()
    MPI_RANK.MPI.Finalize()
    os._exit(code)


if __name__ == "__main__":
    if _MPI_WORKER_FLAG in sys.argv:
        sys.exit(_run_one_mpi_rank())
    import _launch  # beside this file

    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=None)
    world_size = parser.parse_args().world_size or len(_launch.visible_devices())
    _launch.mpirun_ranks(
        __file__, world_size, [_MPI_WORKER_FLAG, "--world-size", str(world_size)], DEADLINE_S
    )
    print("OK: MPI session (mnnvl_fusion_allreduce, fusion NONE)", flush=True)
    print("OK")
