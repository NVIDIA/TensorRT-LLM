# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 sequence-parallel collectives on NVLink peer mappings, one GB300 tray (4 ranks).

Every rank regenerates every rank's inputs from the cell's seed and calls the entry on its own: the all-gather must
equal the rank-order concatenation bit for bit, the reduce-scatter the rank-order FP32 sum rounded once, on every
rank. Each cell runs the entry three times interleaved with the other entry (the calls share the workspace and its
call counter), and a tensor the kernels do not serve takes the NCCL op.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import pickle
import sys
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the sequence-parallel kernels are CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.seq_parallel_rows_mnnvl import (  # noqa: E402, E501
    sp_gather_rows_mnnvl,
    sp_reduce_scatter_rows_mnnvl,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        or torch.cuda.device_count() < WORLD,
        reason="one GB300 tray: sm_100 / sm_103 and four GPUs",
    ),
]


def _inputs(spec: dict, seed: int):
    """Every rank's list of tensors, in rank order."""
    out = []
    for r in range(spec["world"]):
        g = torch.Generator(device="cuda").manual_seed(seed * 100 + r)
        tensors = []
        for shape, dtype in zip(spec["shapes"], spec["dtypes"]):
            t = torch.randn(shape, generator=g, device="cuda").to(dtype)
            t.view(-1)[:16] = 0.0
            t.view(-1)[16:32] = -0.0
            tensors.append(t)
        out.append(tensors)
    return out


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _rank_cell(gather_spec: dict, rs_spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    ag_in = _inputs(gather_spec, seed)
    rs_in = _inputs(rs_spec, seed + 1)
    ag_ref = sp_gather_rows_mnnvl.reference(ag_in, mapping)
    rs_ref = sp_reduce_scatter_rows_mnnvl.reference(rs_in, mapping)
    fails = []
    for it in range(3):
        ag = sp_gather_rows_mnnvl(ag_in[rank], mapping)
        rs = sp_reduce_scatter_rows_mnnvl(rs_in[rank], mapping)
        torch.cuda.synchronize()
        for i, (got, ref) in enumerate(zip(ag, ag_ref)):
            if got.shape != ref.shape or not torch.equal(_bits(got), _bits(ref)):
                fails.append(("gather", it, i))
        for i, (got, ref) in enumerate(zip(rs, rs_ref)):
            if got.shape != ref.shape or not torch.equal(_bits(got), _bits(ref)):
                fails.append(("reduce_scatter", it, i))
    # A tensor the kernels do not serve (rows of 12 bytes) takes the NCCL op: still the concatenation.
    odd = [torch.full((5, 3), float(rank), device="cuda")]
    got = sp_gather_rows_mnnvl(odd, mapping)[0]
    want = torch.cat([torch.full((5, 3), float(r), device="cuda") for r in range(WORLD)])
    if not torch.equal(got, want):
        fails.append(("fallback",))
    return dict(rank=rank, fails=fails)


# The entries' certified cells, paired by index (gather, reduce-scatter), plus rows below one chunk.
CASES = [
    (g.spec, r.spec) for g, r in zip(sp_gather_rows_mnnvl.CELLS, sp_reduce_scatter_rows_mnnvl.CELLS)
] + [
    (
        dict(world=WORLD, shapes=((3, 7168),), dtypes=(torch.bfloat16,)),
        dict(world=WORLD, shapes=((5 * WORLD, 3584),), dtypes=(torch.bfloat16,)),
    )
]


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("case", range(len(CASES)))
def test_gather_and_reduce_scatter(case, mpi_pool_executor) -> None:
    gspec, rspec = CASES[case]
    seed = zlib.crc32(repr((gspec["shapes"], rspec["shapes"])).encode()) % 100000
    results = list(mpi_pool_executor.map(_rank_cell, *zip(*[(gspec, rspec, seed)] * WORLD)))
    for r in results:
        assert not r["fails"], r
