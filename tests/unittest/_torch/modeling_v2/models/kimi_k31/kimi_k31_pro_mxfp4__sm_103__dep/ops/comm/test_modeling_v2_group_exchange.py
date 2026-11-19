# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 lamport_group_exchange entry on one GB300 tray (four MPI ranks).

On a sample of the entry's cells (``sample_cells``): a cell of a two-rank group runs on both pairs of the tray at
once (ranks 0-1 and 2-3), a four-rank cell on the whole tray. Every rank regenerates every group rank's input from
the cell's seed and exchanges its own through the entry: the result must equal the reference bit for bit (the other
ranks' -0.0 words as +0.0), eagerly and under CUDA-graph replay interleaved with another exchange on the same
workspace.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the group exchange is a CuTe DSL kernel", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.comm.group_exchange import (  # noqa: E402, E501
    lamport_group_exchange,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    on_ranks,
    pickle_by_value,
    run_per_rank,
    sample_cells,
)
from _validating import validating  # noqa: E402 — needs the path declared above

pickle_by_value(__name__)

WORLD = 4
pytestmark = on_ranks(WORLD, 103, 107)


def _inputs(spec: dict, seed: int, group: int):
    rows = spec["rows"] * (1 if spec["bcast"] else spec["world"])
    gen = torch.Generator(device="cuda").manual_seed(seed + 7919 * group)
    xs = [
        torch.randn(rows, spec["row"], generator=gen, device="cuda").to(torch.bfloat16)
        for _ in range(spec["world"])
    ]
    if spec["special"]:
        for x in xs:
            words = x.view(torch.float32).view(-1)
            words[0], words[1], words[2] = -0.0, float("inf"), float("nan")
            words[3] = 0.0
    return xs


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.comm.group_exchange import (  # noqa: E501
        group_workspace,
    )
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    world = spec["world"]
    group = Mapping(
        world_size=WORLD, rank=rank, gpus_per_node=4, tp_size=world, pp_size=WORLD // world
    )
    ws = group_workspace(group)
    xs = _inputs(spec, seed, rank // world)
    x = xs[group.tp_rank]
    ref = lamport_group_exchange.reference(xs, ws, spec["sections"], spec["bcast"])
    other = torch.randn(3, 64, device="cuda")
    with torch.inference_mode():
        with validating(lamport_group_exchange):
            out = lamport_group_exchange(x, ws, spec["sections"], spec["bcast"])
        # Replayed from a CUDA graph with another exchange on the same workspace (the call counter is on the device).
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = lamport_group_exchange(x, ws, spec["sections"], spec["bcast"])
            lamport_group_exchange(other, ws, 1, True)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()

    def same(a, b):
        return a.shape == b.shape and bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))

    return dict(rank=rank, shape=tuple(out.shape), equal=same(out, ref), graph=same(graph_out, ref))


CELLS = sample_cells(lamport_group_exchange.CELLS, size_key="rows")[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:60] for c in CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = run_per_rank(mpi_pool_executor, _rank_cell, spec, seed, world=WORLD)
    for r in results:
        assert r["shape"] == (spec["rows"], spec["world"] * spec["row"]), r
        assert r["equal"] and r["graph"], r
