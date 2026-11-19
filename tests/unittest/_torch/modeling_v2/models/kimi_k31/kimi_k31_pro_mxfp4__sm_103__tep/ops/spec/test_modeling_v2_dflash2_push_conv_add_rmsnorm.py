# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 k31_dflash2_push_conv_add_rmsnorm entry on one GB300 tray (4 ranks).

On a sample of the entry's cells (``sample_cells``), every rank builds the engine AllReduce module (MNNVL forced,
as the drafter's TP16 / TP32 groups resolve it), regenerates every rank's projection input / weight shard and the
epilogue operands from the cell's seed and calls the entry, the transport threshold set to 0 so the cells under the
fused cap take the fused two-shot path (at four ranks the drafter's small blocks are under the engine's one-shot
threshold). Gates: the output and the updated
residual bitwise equal to the two-kernel chain (push GEMM + finish kernel via ``oproj_allreduce_push``, then
``k31_dflash2_conv_add_rmsnorm``); the result within a few BF16 steps of the FP32 reference in a small fraction.

MPI-spawning: run under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.dflash2_push_conv_add_rmsnorm import (  # noqa: E501
    k31_dflash2_push_conv_add_rmsnorm as op,
)

__extra_import_path__ = ["../../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    on_ranks,
    pickle_by_value,
    run_per_rank,
    sample_cells,
    tray_allreduce,
)

pickle_by_value(__name__)

WORLD = 4
H = 7168
TAPS = 2
GROUP = 16
EPS = 1e-6
pytestmark = on_ranks(WORLD, 103, cutedsl=True)


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, k = spec["rows"], spec["k"]

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    inputs = [r(rows, k) for _ in range(WORLD)]
    weights = [r(H, k, scale=k**-0.5) for _ in range(WORLD)]
    # The convolution's dynamic coefficients: the strided finish-side view of a [rows, 2, taps, groups] projection.
    projected = r(rows, 2, TAPS, H // GROUP, scale=0.1)
    delta = projected[:, 1]
    base = r(TAPS, H, scale=0.5)
    residual = r(rows, H, scale=2.0)
    norm_w = (1 + r(H, scale=0.1).float()).bfloat16()
    return inputs, weights, delta, base, residual, norm_w


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.oproj_allreduce_push import (  # noqa: E501
        oproj_allreduce_push,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec import (
        dflash2_push_conv_add_rmsnorm as mod,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.dflash2_conv_add_rmsnorm import (  # noqa: E501
        dflash2_conv_add_rmsnorm,
    )

    one_shot_max_bytes = mod.ONE_SHOT_MAX_BYTES
    mod.ONE_SHOT_MAX_BYTES = 0
    try:
        rank_op = mod.k31_dflash2_push_conv_add_rmsnorm

        rank = tensorrt_llm.mpi_rank()
        torch.cuda.set_device(rank)
        _, ar = tray_allreduce(rank, WORLD)
        inputs, weights, delta, base, residual, norm_w = _inputs(spec, seed)
        block = spec["block"]

        def call(res):
            return rank_op(
                inputs[rank], weights[rank], ar, delta, base, block, GROUP, res, norm_w, EPS
            )

        with torch.inference_mode():
            rank_op.is_valid(
                inputs[rank], weights[rank], ar, delta, base, block, GROUP, residual, norm_w, EPS
            )
            oproj_allreduce_push(inputs[rank], weights[rank], ar)  # grow the workspace eagerly
            r1 = residual.clone()
            out, _ = call(r1)
            x = oproj_allreduce_push(inputs[rank], weights[rank], ar)
            rw = residual.clone()
            want, _ = dflash2_conv_add_rmsnorm(x, delta, base, block, GROUP, rw, norm_w, EPS)
            ref_out, ref_res = rank_op.reference(
                inputs, weights, delta, base, block, GROUP, residual, norm_w, EPS
            )
            return dict(
                rank=rank,
                out_bitwise=bool(torch.equal(out.view(torch.int16), want.view(torch.int16))),
                residual_bitwise=bool(torch.equal(r1.view(torch.int16), rw.view(torch.int16))),
                res_vs_ref_frac=float(
                    (r1.view(torch.int16) != ref_res.view(torch.int16)).float().mean()
                ),
                out_vs_ref_frac=float(
                    (out.view(torch.int16) != ref_out.view(torch.int16)).float().mean()
                ),
            )
    finally:
        mod.ONE_SHOT_MAX_BYTES = one_shot_max_bytes


CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = run_per_rank(mpi_pool_executor, _rank_cell, cell.spec, seed, world=WORLD)
    for r in results:
        assert r["out_bitwise"] and r["residual_bitwise"], r
        # The push GEMM rounds each rank's partial (cuBLAS-class error), so the summed rows are a few BF16 steps
        # from the FP32 reference in a fraction of the elements, which the epilogue carries on.
        assert r["res_vs_ref_frac"] < 0.5 and r["out_vs_ref_frac"] < 0.5, r
