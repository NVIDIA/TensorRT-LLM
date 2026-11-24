# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 sequence-parallel row collectives on one GB300 tray.

Each cell runs on four MPI ranks: every rank regenerates every rank's inputs from the cell's seed, calls the entry
on its own list with its TP mapping and checks the result against the entry's reference over all four ranks'
inputs: bit for bit for the gather, within a BF16 band for the reduce-scatter (NCCL sums in BF16 in ring order).

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import pickle
import sys
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.seq_parallel_rows import (  # noqa: E501
    sp_gather_rows,
    sp_reduce_scatter_rows,
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


def _inputs(spec: dict, seed: int) -> list:
    """Every rank's list of inputs, in rank order, identical on every rank."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return [
        [
            torch.randn(*shape, generator=gen, device="cuda").to(dtype)
            for shape, dtype in zip(spec["shapes"], spec["dtypes"])
        ]
        for _ in range(spec["world"])
    ]


def _rank_cell(op_name: str, spec: dict, seed: int) -> dict:
    """One rank's half of a cell; runs inside an MPI pool worker."""
    import tensorrt_llm
    from tensorrt_llm.mapping import Mapping

    op = {"gather": sp_gather_rows, "reduce_scatter": sp_reduce_scatter_rows}[op_name]
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=spec["world"], tp_size=spec["world"], rank=rank)
    inputs = _inputs(spec, seed)
    mine = inputs[rank]
    before = [x.clone() for x in mine]
    with torch.inference_mode():
        op.is_valid(mine, mapping)
        out = op(mine, mapping)
        again = op(mine, mapping)
        ref = op.reference(inputs, mapping)
        torch.cuda.synchronize()
        shapes = [tuple(o.shape) for o in out]
        exact = all(torch.equal(o, r) for o, r in zip(out, ref))
        # BF16 sums of 4 terms in another order: each rounding is at most 2^-9 of the running partial, which is
        # bounded by the sum of the magnitudes, so the error is gated relative to that sum.
        mag = op.reference([[x.abs() for x in r] for r in inputs], mapping)
        err = max(
            float(((o.float() - r.float()).abs() / (m.float() + 1e-6)).max())
            for o, r, m in zip(out, ref, mag)
        )
        repeat_equal = all(torch.equal(o, a) for o, a in zip(out, again))
    return dict(
        rank=rank,
        shapes=shapes,
        exact=exact,
        rel_err=err,
        repeat_equal=repeat_equal,
        input_untouched=all(torch.equal(x, b) for x, b in zip(mine, before)),
    )


def _run(op_name, cell, mpi_pool_executor):
    spec = cell.spec
    assert spec["world"] == WORLD
    seed = zlib.crc32(cell.why.encode()) % 2**31
    return sorted(
        mpi_pool_executor.map(_rank_cell, *zip(*[(op_name, spec, seed)] * WORLD)),
        key=lambda r: r["rank"],
    )


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", sp_gather_rows.CELLS, ids=[c.why[4:60] for c in sp_gather_rows.CELLS]
)
def test_gather_cells(cell, mpi_pool_executor) -> None:
    for r in _run("gather", cell, mpi_pool_executor):
        assert r["shapes"] == [(s[0] * WORLD,) + tuple(s[1:]) for s in cell.spec["shapes"]], r
        assert r["input_untouched"] and r["repeat_equal"], r
        assert r["exact"], f"rank {r['rank']}: the gather is not the rank-order concatenation"


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", sp_reduce_scatter_rows.CELLS, ids=[c.why[4:60] for c in sp_reduce_scatter_rows.CELLS]
)
def test_reduce_scatter_cells(cell, mpi_pool_executor) -> None:
    for r in _run("reduce_scatter", cell, mpi_pool_executor):
        assert r["shapes"] == [(s[0] // WORLD,) + tuple(s[1:]) for s in cell.spec["shapes"]], r
        assert r["input_untouched"] and r["repeat_equal"], r
        assert r["rel_err"] < 2**-6, (
            f"rank {r['rank']}: reduce-scatter off the fp32 sum by {r['rel_err']}"
        )


def test_one_rank_group_returns_the_inputs() -> None:
    from tensorrt_llm.mapping import Mapping

    x = [torch.randn(4, 16, device="cuda")]
    one = Mapping(world_size=1, tp_size=1, rank=0)
    assert sp_gather_rows(x, one)[0] is x[0]
    assert sp_reduce_scatter_rows(x, one)[0] is x[0]
