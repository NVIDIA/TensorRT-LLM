# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 rows_alltoall entry (ShardQ head / token exchange) on one GB300 tray.

Each cell runs on four MPI ranks: every rank regenerates every rank's pieces from the cell's seed, calls the entry
with its own pieces and checks the meaningful rows of every received block bit for bit against the reference.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import pickle
import sys
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.rows_alltoall import (  # noqa: E501
    rows_alltoall,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 3)
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs",
    ),
]


def _pieces(spec: dict, seed: int) -> list:
    """Every rank's pieces (``pieces[i][j]`` = what rank i sends to rank j), identical on every rank. Rank i sends
    ``sizes[j]`` rows to rank j, as views into one buffer (the target slices one tensor)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    sizes, tail = spec["sizes"], spec["tail"]
    out = []
    for _ in range(spec["world"]):
        buf = torch.randn(sum(sizes), *tail, generator=gen, device="cuda").to(torch.bfloat16)
        out.append(list(buf.split(list(sizes), dim=0)))
    return out


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=spec["world"], tp_size=spec["world"], rank=rank)
    pieces = _pieces(spec, seed)
    with torch.inference_mode():
        out = rows_alltoall(pieces[rank], spec["rows"], mapping)
        ref = rows_alltoall.reference(pieces, spec["rows"], mapping)
        torch.cuda.synchronize()
        n = spec["sizes"][rank]
        exact = bool(torch.equal(out[:, :n], ref[:, :n]))
    return dict(rank=rank, shape=tuple(out.shape), exact=exact)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", rows_alltoall.CELLS, ids=[c.why[:56] for c in rows_alltoall.CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = mpi_pool_executor.map(_rank_cell, *zip(*[(spec, seed)] * WORLD))
    for r in results:
        assert r["shape"] == (WORLD, spec["rows"]) + tuple(spec["tail"]), r
        assert r["exact"], f"rank {r['rank']}: received blocks differ from what the peers sent"


def test_one_rank_group_pads_the_piece() -> None:
    from tensorrt_llm.mapping import Mapping

    x = torch.randn(3, 8, device="cuda")
    out = rows_alltoall([x], 5, Mapping(world_size=1, tp_size=1, rank=0))
    assert out.shape == (1, 5, 8) and torch.equal(out[0, :3], x)
