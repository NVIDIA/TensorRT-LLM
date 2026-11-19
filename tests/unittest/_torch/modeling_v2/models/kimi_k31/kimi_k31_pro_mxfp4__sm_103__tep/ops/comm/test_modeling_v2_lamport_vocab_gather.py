# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 lamport_vocab_gather and lamport_allgather entries on one GB300 tray (four MPI ranks).

Every rank builds the engine `AllReduce` module of the TP group (bf16, MNNVL forced: one tray), whose
workspace the gathers use, regenerates every rank's BF16 shard from the cell's seed and gathers its own
through the entry: the result must equal the reference bit for bit on every rank (the FP32
concatenation, with the sentinel word's -0.0 as +0.0). Each entry runs a sample of its cells
(``sample_cells``).

The FP32 Lamport all-gather is checked the same way on its own cells: bit for bit the reference (the
concatenation with -0.0 as +0.0) on every rank, served exactly where the gathered message fits the cell's
cap.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the gather is a CuTe DSL kernel", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.lamport_allgather import (  # noqa: E402, E501
    lamport_allgather,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.lamport_vocab_gather import (  # noqa: E402, E501
    lamport_vocab_gather,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    on_ranks,
    pickle_by_value,
    run_per_rank,
    sample_cells,
    tray_allreduce,
)
from _validating import validating  # noqa: E402 — needs the path declared above

pickle_by_value(__name__)

WORLD = 4
pytestmark = on_ranks(WORLD, 103, 107)


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (spec["rows"], spec["shard"])
    xs = [
        (torch.randn(shape, generator=gen, device="cuda") * 8).to(torch.bfloat16)
        for _ in range(WORLD)
    ]
    if spec["special"]:
        flat = xs[1].view(-1)
        flat[0], flat[1], flat[2], flat[3] = 0.0, -0.0, float("inf"), float("-inf")
        flat[4], flat[5] = float("nan"), -0.0  # -0.0 after a non-zero even column
        flat[6], flat[7] = -0.0, 0.0
        xs[2].view(-1)[8], xs[2].view(-1)[9] = 0.0, -0.0
        xs[3].view(-1)[-2], xs[3].view(-1)[-1] = 0.0, -0.0
    return xs


def _same(a, b) -> bool:
    return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping, ar = tray_allreduce(rank, WORLD, gpus_per_node=4)
    xs = _inputs(spec, seed)
    ref = lamport_vocab_gather.reference(xs, mapping, ar)
    with torch.inference_mode(), validating(lamport_vocab_gather):
        out = lamport_vocab_gather(xs[rank], mapping, ar)
    return dict(rank=rank, shape=tuple(out.shape), equal=_same(out, ref))


CELLS = sample_cells(lamport_vocab_gather.CELLS, size_key="rows")[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[4:70] for c in CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    for r in run_per_rank(mpi_pool_executor, _rank_cell, spec, seed, world=WORLD):
        assert r["shape"] == (spec["rows"], WORLD * spec["shard"]) and r["equal"], r


def _allgather_inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    xs = [torch.randn(spec["shape"], generator=gen, device="cuda") for _ in range(WORLD)]
    if spec["special"]:
        flat = xs[1].view(-1)
        flat[0], flat[1], flat[2], flat[3] = -0.0, float("inf"), float("-inf"), float("nan")
        flat[4] = 0.0
        xs[2].view(-1)[5] = -0.0
    return xs


def _rank_allgather_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm import (
        lamport_allgather as agm,
    )

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping, ar = tray_allreduce(rank, WORLD, gpus_per_node=4)
    xs = _allgather_inputs(spec, seed)
    max_bytes = spec.get("max_bytes", agm.MAX_BYTES)
    ref = lamport_allgather.reference(xs, mapping, ar, max_bytes)
    with torch.inference_mode(), validating(lamport_allgather):
        out = lamport_allgather(xs[rank], mapping, ar, max_bytes)
    served = agm.lamport_allgather_serves(xs[rank], mapping, ar, max_bytes)
    # The engine all-gather fallback keeps NaN payloads; the Lamport path's bits are the reference's.
    equal = _same(out, ref) if served else bool(torch.equal(out.nan_to_num(), ref.nan_to_num()))
    return dict(rank=rank, served=served, shape=tuple(out.shape), equal=equal)


ALLGATHER_CELLS = sample_cells(lamport_allgather.CELLS)[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", ALLGATHER_CELLS, ids=[c.why[4:60] for c in ALLGATHER_CELLS])
def test_allgather_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = run_per_rank(mpi_pool_executor, _rank_allgather_cell, spec, seed, world=WORLD)
    for r in results:
        assert r["shape"] == (*spec["shape"][:-1], WORLD * spec["shape"][-1]) and r["equal"], r
    # Served exactly where the gathered message fits the cell's cap (the default unless it raises it).
    cap = spec.get("max_bytes", 1 << 20)
    gathered = WORLD * 4 * torch.Size(spec["shape"]).numel()
    assert [r["served"] for r in results] == [gathered <= cap] * WORLD, results
