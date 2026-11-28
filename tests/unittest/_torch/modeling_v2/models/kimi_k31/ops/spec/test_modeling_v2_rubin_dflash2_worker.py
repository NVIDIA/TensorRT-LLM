# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 worker's draft-token agreement entry (``ops/spec/rubin_dflash2_worker.py``).

Four GPUs (one tray): a sample of the ``rubin_draft_token_agreement`` cells, each on a TP 4 group of
MPI ranks with the engine all-gather; every rank must get rank 0's drafts bit for bit.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_worker import (
    rubin_draft_token_agreement,
)
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above  # noqa: E402 — needs the path declared above
    on_sm,
    pickle_by_value,
    run_per_rank,
    sample_cells,
)

pickle_by_value(__name__)

WORLD = 4
CELLS = sample_cells(rubin_draft_token_agreement.CELLS, size_key="rows")[:2]


def _rank_agreement_cell(spec: dict, seed: int) -> dict:
    """One rank's half of an agreement cell; runs inside an MPI pool worker."""
    import tensorrt_llm

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    drafts = [
        torch.randint(0, 163840, (spec["rows"], spec["k"]), generator=gen, device="cuda").int()
        for _ in range(WORLD)
    ]
    mapping = Mapping(world_size=WORLD, rank=rank, tp_size=WORLD)
    with torch.inference_mode():
        rubin_draft_token_agreement.is_valid(drafts[rank], mapping)
        out = rubin_draft_token_agreement(drafts[rank], mapping)
        exact = bool(torch.equal(out, rubin_draft_token_agreement.reference(drafts)))
    return dict(rank=rank, exact=exact)


@on_sm(103, 107, gpus=WORLD)
@pytest.mark.threadleak(enabled=False)
@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why for c in CELLS])
def test_agreement_cells_take_rank0_drafts(mpi_pool_executor, cell):
    """Each cell on a TP 4 group with the engine all-gather: every rank gets rank 0's drafts (each
    rank's own differ), as the entry's reference states."""
    results = run_per_rank(
        mpi_pool_executor,
        _rank_agreement_cell,
        cell.spec,
        zlib.crc32(cell.why.encode()),
        world=WORLD,
    )
    assert all(r["exact"] for r in results), results
