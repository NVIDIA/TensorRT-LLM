# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 allgather_module entry: the engine allgather on one GB300 tray.

Each cell runs on four MPI ranks (one tray), the way a served engine runs: every rank regenerates
every rank's input from the cell's seed, calls the entry on its own slice with its TP mapping and
checks the result bit for bit against the entry's reference over all four inputs. Ragged cells
hand each rank a row slice of a larger buffer, as the indexer does.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allgather_module import (
    allgather_module,
)

__extra_import_path__ = ["../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    on_ranks,
    pickle_by_value,
    run_per_rank,
    sample_cells,
)

pickle_by_value(__name__)

WORLD = 4
pytestmark = on_ranks(WORLD, 103, 107)


def _inputs(spec: dict, seed: int) -> list:
    """Every rank's input, in rank order, identical on every rank.

    Ragged cells (`sizes` given) split one `shape`-sized buffer along `dim` by `sizes` and hand each
    rank its slice, a view at a nonzero offset into the buffer for every rank but the first.
    Even cells give every rank its own `shape`-sized tensor.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype, sizes = spec["shape"], spec["dtype"], spec["sizes"]

    def make(*s):
        if dtype == torch.int32:
            # Index-shaped values, -1 (no index) included.
            return torch.randint(-1, 1 << 20, s, generator=gen, device="cuda", dtype=torch.int32)
        return torch.randn(*s, generator=gen, device="cuda").to(dtype)

    if sizes is None:
        return [make(*shape) for _ in range(spec["world"])]
    assert sum(sizes) == shape[spec["dim"]] and len(sizes) == spec["world"]
    return list(make(*shape).split(list(sizes), dim=spec["dim"]))


def _rank_cell(spec: dict, seed: int) -> dict:
    """One rank's half of a cell; runs inside an MPI pool worker."""
    import tensorrt_llm
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=spec["world"], tp_size=spec["world"], rank=rank)
    inputs = _inputs(spec, seed)
    sizes = None if spec["sizes"] is None else list(spec["sizes"])
    args = (inputs[rank], mapping, spec["dim"], sizes)
    with torch.inference_mode():
        allgather_module.is_valid(*args)
        out = allgather_module(*args)
        exact = bool(torch.equal(out, allgather_module.reference(inputs, spec["dim"])))
    return dict(rank=rank, exact=exact)


CELLS = sample_cells(allgather_module.CELLS)[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[4:64] for c in CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    seed = zlib.crc32(cell.why.encode()) % 2**31
    for r in run_per_rank(mpi_pool_executor, _rank_cell, cell.spec, seed, world=WORLD):
        assert r["exact"], f"rank {r['rank']}: the gather is not the rank-order concatenation"
