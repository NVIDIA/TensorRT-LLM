# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 oproj_allreduce_push entry on one GB300 tray (four MPI ranks).

Every rank builds the engine `AllReduce` module the target builds for the attention TP group (bf16,
MNNVL forced: one tray).

On a sample of its cells (``sample_cells``; sm_103, sm_107), every rank regenerates every rank's
o_proj input and weight shard from the cell's seed, calls the entry and gates it against the
reference over all four ranks. Where the push does not serve the call, the result is the current
target path (`F.linear` + `allreduce_module`) bit for bit.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the push chain is a CuTe DSL kernel pair", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.oproj_allreduce_push import (  # noqa: E402, E501
    oproj_allreduce_push,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    bf16_deviation,
    on_ranks,
    pickle_by_value,
    run_per_rank,
    sample_cells,
    tray_allreduce,
)
from _validating import validating  # noqa: E402 — needs the path declared above

pickle_by_value(__name__)

WORLD = 4


def _marks(marks):
    """Apply a list of marks (``on_ranks``) to one test."""

    def apply(fn):
        for mark in marks:
            fn = mark(fn)
        return fn

    return apply


# Gate in bf16 ulps against the fp32 sum of the fp32 products rounded once. The op rounds each
# rank's partial to bf16 before the sum (as the target's o_proj + all-reduce does): up to half an
# ulp of each partial, which is at most the row's scale, plus the final rounding.
ELEMENT_ULP, RMS_ULP = 0.5 * WORLD + 1.0, 1.0


def _inputs(spec: dict, seed: int):
    """Every rank's input and weight shard, in rank order: identical on every rank."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, k, hidden = spec["rows"], spec["k"], spec["hidden"]
    xs = [
        torch.randn(rows, k, generator=gen, device="cuda").to(torch.bfloat16) for _ in range(WORLD)
    ]
    ws = [
        (torch.randn(hidden, k, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
        for _ in range(WORLD)
    ]
    return xs, ws


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm import (
        oproj_allreduce_push as opm,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (
        allreduce_module,
    )

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    _, ar = tray_allreduce(rank, WORLD, gpus_per_node=4)
    assert ar.mnnvl_allreduce is not None, "MNNVL was not built on this tray"

    xs, ws = _inputs(spec, seed)
    x, w = xs[rank], ws[rank]
    with torch.inference_mode():
        with validating(oproj_allreduce_push):
            out = oproj_allreduce_push(x, w, ar)
        current = allreduce_module(torch.nn.functional.linear(x, w), ar, 0, None, None)
        ref = oproj_allreduce_push.reference(xs, ws, ar)
        element, rms = bf16_deviation(out, ref)
    return dict(
        rank=rank,
        shape=tuple(out.shape),
        served=opm.push_serves(x, w, ar),
        element_ulp=element,
        rms_ulp=rms,
        current_equal=bool(torch.equal(out.view(torch.int16), current.view(torch.int16))),
    )


CELLS = sample_cells(oproj_allreduce_push.CELLS, size_key="rows")[:2]


@_marks(on_ranks(WORLD, 103, 107))
@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell",
    CELLS,
    ids=[c.why[4:72] for c in CELLS],
)
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    assert spec["world"] == WORLD
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = run_per_rank(mpi_pool_executor, _rank_cell, spec, seed, world=WORLD)
    for r in results:
        assert r["shape"] == (spec["rows"], spec["hidden"]), r
        assert r["element_ulp"] <= ELEMENT_ULP and r["rms_ulp"] <= RMS_ULP, r
        # The fallback is the current target path itself.
        assert r["served"] or r["current_equal"], r
