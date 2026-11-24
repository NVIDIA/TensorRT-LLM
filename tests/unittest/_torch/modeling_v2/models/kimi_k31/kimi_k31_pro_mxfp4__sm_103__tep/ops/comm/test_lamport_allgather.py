# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 lamport_allgather entry on one GB300 tray (four MPI ranks).

Every rank builds the engine `AllReduce` module of the TP group (bf16, MNNVL forced: one tray),
regenerates every rank's input from the cell's seed and gathers its own through the entry: the result
must equal the reference bit for bit on every rank (the concatenation with -0.0 as +0.0), repeated,
interleaved with engine all-reduces on the same workspace, and under CUDA-graph replay.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import pickle
import sys
import types
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the all-gather is a CuTe DSL kernel", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.lamport_allgather import (  # noqa: E402, E501
    lamport_allgather,
)

__extra_import_path__ = ["../../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 3), (10, 7))
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs (sm_107 driven separately)",
    ),
]


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    xs = [torch.randn(spec["shape"], generator=gen, device="cuda") for _ in range(WORLD)]
    if spec["special"]:
        flat = xs[1].view(-1)
        flat[0], flat[1], flat[2], flat[3] = -0.0, float("inf"), float("-inf"), float("nan")
        flat[4] = 0.0
        xs[2].view(-1)[5] = -0.0
    return xs


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm import (
        lamport_allgather as agm,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    reg = vars(sys.modules.setdefault("_k31_lamport_ag", types.ModuleType("_k31_lamport_ag")))
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    if "ar" not in reg:
        reg["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    ar = reg["ar"]
    xs = _inputs(spec, seed)
    x = xs[rank]
    max_bytes = spec.get("max_bytes", agm.MAX_BYTES)
    ref = lamport_allgather.reference(xs, mapping, ar, max_bytes)
    probe = torch.randn(8, 7168, device="cuda").to(torch.bfloat16)
    with torch.inference_mode():
        with validating(lamport_allgather):
            out = lamport_allgather(x, mapping, ar, max_bytes)
        ar(probe)  # an engine all-reduce on the same workspace in between
        again = lamport_allgather(x, mapping, ar, max_bytes)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = lamport_allgather(x, mapping, ar, max_bytes)
            ar(probe)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()

    def same(a, b):
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))

    served = agm.lamport_allgather_serves(x, mapping, ar, max_bytes)
    return dict(
        rank=rank,
        served=served,
        shape=tuple(out.shape),
        equal=same(out, ref) if served else None,
        close=bool(torch.equal(out.nan_to_num(), ref.nan_to_num())),
        repeat=same(out, again),
        graph=same(out, graph_out),
    )


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", lamport_allgather.CELLS, ids=[c.why[4:60] for c in lamport_allgather.CELLS]
)
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = list(mpi_pool_executor.map(_rank_cell, *zip(*[(spec, seed)] * WORLD)))
    for r in results:
        assert r["shape"] == (*spec["shape"][:-1], WORLD * spec["shape"][-1]), r
        assert r["close"] and r["repeat"] and r["graph"], r
        if r["served"]:
            assert r["equal"], r
    # Served exactly where the gathered message fits the cell's cap (the default unless it raises it).
    cap = spec.get("max_bytes", 1 << 20)
    gathered = WORLD * 4 * torch.Size(spec["shape"]).numel()
    assert [r["served"] for r in results] == [gathered <= cap] * WORLD, results
