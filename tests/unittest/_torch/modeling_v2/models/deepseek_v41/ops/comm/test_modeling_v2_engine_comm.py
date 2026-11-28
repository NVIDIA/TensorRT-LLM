# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-rank test of the targets' engine collectives (`ops/comm/engine_comm.py`): `engine_allgather` bit for bit
against the concatenation of every rank's rows (ragged `sizes` included), `engine_allreduce` against the float64 sum
of every rank's input (the engine's kernels sum in fp32 or bf16 and round to bf16: a bf16-sized band),
`engine_alltoallv_tokens` bit for bit against each member's own token rows of every member's input. The test runs
the module on its NCCL strategy, eager (the AUTO strategy's symmetric windows belong to the engine's
capture). Needs 4 visible GPUs of one tray; run under the repo's `mpi_pool_executor` fixture."""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed.ops import AllReduceStrategy
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.comm.engine_comm import (  # noqa: E501
    engine_allgather,
    engine_allreduce,
    engine_allreduce_create,
    engine_alltoallv_tokens,
)
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 4
        or torch.cuda.get_device_capability() != (10, 3),
        reason="needs 4 SM103 GPUs",
    ),
]


_ALLREDUCE: dict = {}


def _input(rank, rows, width, salt, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(1000 * salt + rank)
    return torch.randn(rows, width, generator=g, device="cuda").to(dtype)


def _rank_body(world, kind, spec):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        mapping = Mapping(world_size=world, tp_size=world, rank=rank)
        if kind == "allgather":
            sizes = spec["sizes"]
            rows = [sizes[r] if sizes else spec["rows"] for r in range(world)]
            xs = [_input(r, rows[r], spec["width"], 1).to(torch.int64) for r in range(world)]
            engine_allgather.is_valid(xs[rank], mapping, 0, list(sizes) if sizes else None)
            got = engine_allgather(xs[rank], mapping, 0, list(sizes) if sizes else None)
            mapping.reference_inputs = xs
            assert torch.equal(got, engine_allgather.reference(xs[rank], mapping, 0, sizes))
        elif kind == "alltoallv":
            counts = list(spec["counts"])
            group = list(range(world))
            xs = [
                _input(r, sum(counts), spec["width"], 2) for r in range(world)
            ]  # every member: all the group's tokens
            engine_alltoallv_tokens.is_valid(xs[rank], counts, group, rank)
            got = engine_alltoallv_tokens(xs[rank], counts, group, rank)
            xs[rank].reference_inputs = xs
            assert torch.equal(
                got, engine_alltoallv_tokens.reference(xs[rank], counts, group, rank)
            )
            torch.cuda.synchronize()
        else:
            ar = _ALLREDUCE.get(world)
            if ar is None:  # one module per process, as a target builds it at load time
                # NCCL: the AUTO strategy's symmetric-memory windows are leases the engine's own graph capture and
                # teardown manage; a test process releasing them through tensor destructors aborts at exit.
                ar = _ALLREDUCE[world] = engine_allreduce_create(mapping, AllReduceStrategy.NCCL)
            for salt in range(3):
                xs = [_input(r, spec["rows"], spec["width"], salt) for r in range(world)]
                ar.reference_inputs = xs
                got = engine_allreduce(ar, xs[rank])
                torch.testing.assert_close(
                    got.float(),
                    engine_allreduce.reference(ar, xs[rank]).float(),
                    rtol=2e-2,
                    atol=3e-2,
                )
            torch.cuda.synchronize()
    except Exception:
        traceback.print_exc()
        raise
    return True


_CELLS = (
    [("allgather", c) for c in engine_allgather.CELLS]
    + [("allreduce", c) for c in engine_allreduce.CELLS]
    + [("alltoallv", c) for c in engine_alltoallv_tokens.CELLS]
)
assert {c.spec["world"] for _, c in _CELLS} == {4}


@pytest.mark.parametrize("kind,cell", _CELLS, ids=[f"{k}-{c.why[:36]}" for k, c in _CELLS])
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_certified_cells(mpi_pool_executor, kind, cell):
    """Every cell of the entries (`engine_allgather`, `engine_allreduce`, `engine_alltoallv_tokens`) on four ranks."""
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_rank_body, *zip(*[(world, kind, cell.spec)] * world))
    for r in results:
        assert r is True
