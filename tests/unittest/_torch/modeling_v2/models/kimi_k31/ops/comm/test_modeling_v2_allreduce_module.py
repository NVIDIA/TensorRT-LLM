# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 allreduce_module entry: the engine AllReduce module on one GB300 tray.

Each cell runs on four MPI ranks (one tray), the way a served engine runs: every rank builds the
same `AllReduce` module the target builds (bf16, the cell's strategy), regenerates every rank's
input from the cell's seed, calls the entry on its own input and gates the result against the
entry's reference over all four inputs.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import sys
import types
import zlib

import pytest
import torch

from tensorrt_llm._torch.distributed import AllReduceStrategy
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (
    allreduce_module,
    tp_allreduce,
    uses_mnnvl,
)

__extra_import_path__ = ["../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    bf16_deviation,
    on_ranks,
    pickle_by_value,
    run_per_rank,
    sample_cells,
)

pickle_by_value(__name__)

WORLD = 4
pytestmark = on_ranks(WORLD, 103, 107)

# Gate in bf16 ulps: per element against its row's largest magnitude, and the relative RMS.
# Measured on GB300 over every cell: the one-shot, two-shot and MNNVL NONE sums are bit-exact
# against the fp32 sum rounded once; NCCL (bf16 reduction steps) reaches 2.3 ulp element / 0.76
# ulp RMS; every RMS_NORM path, which rounds the normalized value as well as the product, reaches
# 2.8 ulp / 0.91 ulp (NCCL plus the separate norm kernel). A reduction that drops one rank's
# partial misses both bands by orders of magnitude.
ELEMENT_ULP, RMS_ULP = 4.0, 1.5


def _inputs(spec: dict, seed: int):
    """Every rank's input, in rank order, plus the norm weight: identical on every rank."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, hidden = spec["rows"], spec["hidden"]
    inputs = [
        torch.randn(rows, hidden, generator=gen, device="cuda").to(torch.bfloat16)
        for _ in range(spec["world"])
    ]
    norm_weight = (1 + 0.1 * torch.randn(hidden, generator=gen, device="cuda")).to(torch.bfloat16)
    return inputs, norm_weight


def _rank_cell(spec: dict, seed: int) -> dict:
    """One rank's half of a cell; runs inside an MPI pool worker."""
    import tensorrt_llm
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=spec["world"], tp_size=spec["world"], rank=rank)
    # This function is pickled by value, so its own globals are fresh on every call; keep the
    # modules (and with them their workspaces) in a worker-process registry instead.
    modules = vars(
        sys.modules.setdefault("_k31_allreduce_modules", types.ModuleType("_k31_allreduce_modules"))
    ).setdefault("by_key", {})
    key = (spec["world"], spec["strategy"])
    if key not in modules:
        AutoTuner.get().setup_distributed_state(mapping)
        # What the target builds: tp_allreduce with the configured strategy (its int value, as the
        # cell carries it) and the default bf16.
        modules[key] = tp_allreduce(mapping, spec["strategy"])
    module = modules[key]
    if spec["strategy"] == int(AllReduceStrategy.MNNVL):
        assert uses_mnnvl(module), "the MNNVL cell did not build the MNNVL path"

    inputs, norm_weight = _inputs(spec, seed)
    fusion_op, eps = spec["fusion_op"], spec["eps"]
    args = (inputs[rank], module, fusion_op, norm_weight, eps)
    # Under MPI the AUTO path can hand back an NCCL-window tensor, which must be released inside
    # a window-tensor scope -- the target's decoder layers are such scopes. Everything that reads
    # the output stays inside it.
    with torch.inference_mode(), discard_nccl_window_tensor_outputs((inputs, norm_weight)):
        allreduce_module.is_valid(*args)
        if spec["strategy"] == int(AllReduceStrategy.AUTO):
            # A served engine tunes AUTO's kernel choice at warm-up; certify the tuned choice.
            with autotune():
                allreduce_module(*args)
        out = allreduce_module(*args)
        ref = allreduce_module.reference(inputs, fusion_op, norm_weight, eps)
        element, rms = bf16_deviation(out, ref)
        del out
    return dict(rank=rank, element_ulp=element, rms_ulp=rms)


CELLS = sample_cells(allreduce_module.CELLS, size_key="rows")[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[4:64] for c in CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    seed = zlib.crc32(cell.why.encode()) % 2**31
    for r in run_per_rank(mpi_pool_executor, _rank_cell, cell.spec, seed, world=WORLD):
        assert r["element_ulp"] <= ELEMENT_ULP and r["rms_ulp"] <= RMS_ULP, (
            f"rank {r['rank']}: element {r['element_ulp']:.2f} ulp, rms {r['rms_ulp']:.2f} ulp"
        )
