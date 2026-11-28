# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 sharded MoE head on one GB300 tray (four MPI ranks).

Every rank builds the engine `AllReduce` module of the TP group (bf16, MNNVL forced: one tray), whose workspace the
head's gather uses, regenerates the cell's inputs and runs the sharded head on its slice of the head weight: the
gathered router logits and routed latent must equal the replicated head's (``reference``) bit for bit on every rank,
for random inputs at two scales, rows of one repeated token, large spikes and exact cancellation (K's second half
repeats the first with the weight's sign flipped, so every output is the summation order's own rounding) and all-zero
rows. The routed variant (the MoE prologue folded into the all-gather) must give the bits of the sharded head followed
by the MoE prologue with the runner's weights, for three correction biases (with the zero bias every score of an
all-zero row ties, so routing takes its exact top-k fallback).

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the sharded head is CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe.moe_head_shard import (  # noqa: E402, E501
    gather_bytes,
    k31_moe_head_shard,
    k31_moe_head_shard_routed,
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
HIDDEN, ROUTER, LATENT = 7168, 896, 3584
pytestmark = on_ranks(WORLD, 103)


def _battery(rows: int, seed: int):
    """``(x, head weight)`` pairs, identical on every rank."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = (torch.randn(ROUTER + LATENT, HIDDEN, generator=gen, device="cuda") * 0.02).to(
        torch.bfloat16
    )
    cancel = weight.clone()
    cancel[:, HIDDEN // 2 :] = -cancel[:, : HIDDEN // 2]
    pairs = [
        ((torch.randn(rows, HIDDEN, generator=gen, device="cuda") * s).to(torch.bfloat16), weight)
        for s in (0.5, 4.0)
    ]
    one = (torch.randn(1, HIDDEN, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    pairs.append((one.repeat(rows, 1), weight))
    spikes = torch.randn(rows, HIDDEN, generator=gen, device="cuda") * 0.01
    spikes[:, ::97] = 30.0
    pairs.append((spikes.to(torch.bfloat16), weight))
    for _ in range(2):
        x = torch.randn(rows, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
        x[:, HIDDEN // 2 :] = x[:, : HIDDEN // 2]
        pairs.append((x, cancel))
    pairs.append((torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda"), weight))
    return pairs


def _rank_workspace(spec: dict):
    """This rank's index and the head's gather workspace on the TP group's engine all-reduce, for the cell's rows."""
    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.oproj_allreduce_push import (  # noqa: E501
        lamport_workspace,
    )

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    _, ar = tray_allreduce(rank, WORLD, gpus_per_node=4)
    width = 2 * gather_bytes(1, WORLD) // (4 * WORLD)
    return rank, lamport_workspace(ar, spec["rows"], width, one_shot=True)


def _same(a, b) -> bool:
    return all(
        bool(torch.equal(u.contiguous().view(torch.uint8), v.contiguous().view(torch.uint8)))
        for u, v in zip(a, b)
    )


def _rank_cell(spec: dict, seed: int) -> dict:
    rank, ws = _rank_workspace(spec)
    equal = True
    with torch.inference_mode():
        for x, weight in _battery(spec["rows"], seed):
            ref = k31_moe_head_shard.reference(x, weight, rank, WORLD, None)
            with validating(k31_moe_head_shard):
                out = k31_moe_head_shard(x, weight, rank, WORLD, ws)
            equal &= _same(out, ref)
    return dict(rank=rank, equal=equal)


CELLS = sample_cells(k31_moe_head_shard.CELLS, size_key="rows")[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[30:70] for c in CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    seed = zlib.crc32(cell.why.encode()) % 2**31
    for r in run_per_rank(mpi_pool_executor, _rank_cell, cell.spec, seed, world=WORLD):
        assert r["equal"], r


def _rank_cell_routed(spec: dict, seed: int) -> dict:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.moe_prologue import (
        k31_moe_prologue,
    )

    rank, ws = _rank_workspace(spec)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    biases = [torch.zeros(ROUTER, device="cuda")] + [
        torch.randn(ROUTER, generator=gen, device="cuda") * s for s in (0.05, 2.0)
    ]
    equal = True
    with torch.inference_mode():
        for x, weight in _battery(spec["rows"], seed):
            for bias in biases:
                logits, latent = k31_moe_head_shard(x, weight, rank, WORLD, ws)
                unfused = k31_moe_prologue(logits, bias, 1.0, latent, True)
                with validating(k31_moe_head_shard_routed):
                    out = k31_moe_head_shard_routed(x, weight, bias, rank, WORLD, ws)
                equal &= _same(out, unfused)
    return dict(rank=rank, equal=equal)


ROUTED_CELLS = sample_cells(k31_moe_head_shard_routed.CELLS, size_key="rows")[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell",
    ROUTED_CELLS,
    ids=[c.why[30:70] for c in ROUTED_CELLS],
)
def test_routed_cells(cell, mpi_pool_executor) -> None:
    seed = zlib.crc32(cell.why.encode()) % 2**31
    for r in run_per_rank(mpi_pool_executor, _rank_cell_routed, cell.spec, seed, world=WORLD):
        assert r["equal"], r
