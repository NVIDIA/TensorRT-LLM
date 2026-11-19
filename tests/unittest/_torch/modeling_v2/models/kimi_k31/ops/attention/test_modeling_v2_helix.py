# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family Helix target combine (``ops/attention/helix.py``) on one tray of four GPUs.

Each cell runs on four MPI ranks of a Helix CP 4 group, as a Helix generation server does: every rank regenerates
every rank's padded partials and softmax statistics from one seed, with queries that see no keys on some ranks and
NaN in the discarded heads and the empty rows, and checks its head shard of the exchanged combine against the
entry's reference over all four ranks.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.helix import helix_target_combine

__extra_import_path__ = ["../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    on_ranks,  # noqa: E402 — needs the path declared above
    pickle_by_value,
    run_per_rank,
)

pickle_by_value(__name__)

WORLD = 4
_ROWS, _LATENT = 5, 512
pytestmark = on_ranks(WORLD, 103, 107)


def _rank_cell(spec: dict, seed: int) -> dict:
    """One rank's half of a cell; runs inside an MPI pool worker."""
    import tensorrt_llm
    from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    world, heads, padded = spec["cp"], spec["heads"], spec["padded"]
    mapping = Mapping(
        world_size=world,
        rank=rank,
        tp_size=1,
        cp_size=world,
        cp_config={"cp_type": "HELIX", "use_nccl_for_alltoall": True},
        gpus_per_node=world,
    )
    gen = torch.Generator(device="cuda").manual_seed(seed)
    partials = torch.randn(world, _ROWS, padded, _LATENT, generator=gen, device="cuda").bfloat16()
    stats = torch.randn(world, _ROWS, padded, 2, generator=gen, device="cuda")
    stats[..., 1] = stats[..., 1].abs() + 1
    # Some queries see no keys on a rank; the discarded heads carry poison too.
    empty = torch.zeros(world, _ROWS, device="cuda", dtype=torch.bool)
    empty[1:, 0] = True
    empty[2:, 1] = True
    partials[empty] = float("nan")
    partials[:, :, heads:] = float("nan")
    metadata = SimpleNamespace(helix_kv_bounds=(~empty[rank]).int(), _helix_spec_tokens_valid=True)
    reference_stats = stats[:, :, :heads].clone()
    reference_stats[..., 0].masked_fill_(empty[:, :, None], -torch.inf)
    reference_stats[..., 1].masked_fill_(empty[:, :, None], 0)
    with torch.inference_mode():
        expected = helix_target_combine.reference(partials[:, :, :heads], reference_stats, rank)
        with nccl_window_tensor_scope(partials):
            actual = helix_target_combine(
                partials[rank].flatten(1),
                stats[rank],
                mapping,
                heads,
                _LATENT,
                metadata,
                None,
                None,
            )
        close = bool(
            torch.allclose(actual.view_as(expected).float(), expected.float(), atol=0.02, rtol=0.02)
        )
    return dict(rank=rank, close=close)


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", helix_target_combine.CELLS, ids=[c.why[:48] for c in helix_target_combine.CELLS]
)
def test_combine_cells_over_four_cp_ranks(mpi_pool_executor, cell):
    assert cell.spec["cp"] == WORLD
    results = run_per_rank(
        mpi_pool_executor, _rank_cell, cell.spec, zlib.crc32(cell.why.encode()), world=WORLD
    )
    assert all(r["close"] for r in results), results
