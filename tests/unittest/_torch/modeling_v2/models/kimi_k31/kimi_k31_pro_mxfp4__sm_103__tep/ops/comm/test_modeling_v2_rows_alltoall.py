# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 rows_alltoall entry (ShardQ head / token exchange) on one GB300 tray.

Each cell runs on four MPI ranks: every rank regenerates every rank's pieces from the cell's seed, calls the entry
with its own pieces and checks the meaningful rows of every received block bit for bit against the reference.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.rows_alltoall import (  # noqa: E501
    rows_alltoall,
)

__extra_import_path__ = ["../../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    on_ranks,
    pickle_by_value,
    run_per_rank,
)

pickle_by_value(__name__)

WORLD = 4
pytestmark = on_ranks(WORLD, 103)


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


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", rows_alltoall.CELLS, ids=[c.why[:56] for c in rows_alltoall.CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = run_per_rank(mpi_pool_executor, _rank_cell, spec, seed, world=WORLD)
    for r in results:
        assert r["shape"] == (WORLD, spec["rows"]) + tuple(spec["tail"]), r
        assert r["exact"], f"rank {r['rank']}: received blocks differ from what the peers sent"
