# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 k31_dflash2_push_conv_add_rmsnorm entry on one GB300 tray (4 ranks).

Every rank builds the engine AllReduce module (MNNVL forced, as the drafter's TP16 / TP32 groups resolve it),
regenerates every rank's projection input / weight shard and the epilogue operands from the cell's seed and calls
the entry, the transport threshold set to 0 so the cells under the fused cap take the fused two-shot path (at four
ranks the drafter's small blocks are under the engine's one-shot threshold). Gates: the output and the updated
residual bitwise equal to the two-kernel chain (push GEMM + finish kernel via ``oproj_allreduce_push``, then
``k31_dflash2_conv_add_rmsnorm``); every rank the same bits; a repeat and a CUDA-graph replay the same bits; the
result within a few BF16 steps of the FP32 reference in a small fraction.

MPI-spawning: run under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import pickle
import sys
import types
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.dflash2_push_conv_add_rmsnorm import (  # noqa: E501
    k31_dflash2_push_conv_add_rmsnorm as op,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
H = 7168
TAPS = 2
GROUP = 16
EPS = 1e-6
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not IS_CUTLASS_DSL_AVAILABLE
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 3),)
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs",
    ),
]


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
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.oproj_allreduce_push import (  # noqa: E501
        oproj_allreduce_push,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec import (
        dflash2_push_conv_add_rmsnorm as mod,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.dflash2_conv_add_rmsnorm import (  # noqa: E501
        dflash2_conv_add_rmsnorm,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy

    mod.ONE_SHOT_MAX_BYTES = 0
    rank_op = mod.k31_dflash2_push_conv_add_rmsnorm
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mods = vars(sys.modules.setdefault("_k31_push_conv", types.ModuleType("_k31_push_conv")))
    if "ar" not in mods:
        mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank)
        mods["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    ar = mods["ar"]
    inputs, weights, delta, base, residual, norm_w = _inputs(spec, seed)
    block = spec["block"]

    def call(res):
        return rank_op(inputs[rank], weights[rank], ar, delta, base, block, GROUP, res, norm_w, EPS)

    with torch.inference_mode():
        rank_op.is_valid(
            inputs[rank], weights[rank], ar, delta, base, block, GROUP, residual, norm_w, EPS
        )
        oproj_allreduce_push(inputs[rank], weights[rank], ar)  # grow the workspace eagerly
        r1 = residual.clone()
        out, _ = call(r1)
        r2 = residual.clone()
        again, _ = call(r2)
        rg = residual.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_out, _ = call(rg)
        rg.copy_(residual)
        graph.replay()
        x = oproj_allreduce_push(inputs[rank], weights[rank], ar)
        rw = residual.clone()
        want, _ = dflash2_conv_add_rmsnorm(x, delta, base, block, GROUP, rw, norm_w, EPS)
        ref_out, ref_res = rank_op.reference(
            inputs, weights, delta, base, block, GROUP, residual, norm_w, EPS
        )
        torch.cuda.synchronize()
        bits = out.view(torch.int16).to(torch.int64).flatten()
        result = dict(
            rank=rank,
            out_bitwise=bool(torch.equal(out.view(torch.int16), want.view(torch.int16))),
            residual_bitwise=bool(torch.equal(r1.view(torch.int16), rw.view(torch.int16))),
            repeat=bool(torch.equal(out, again) and torch.equal(r1, r2)),
            graph=bool(torch.equal(out, g_out) and torch.equal(r1, rg)),
            res_vs_ref_frac=float(
                (r1.view(torch.int16) != ref_res.view(torch.int16)).float().mean()
            ),
            out_vs_ref_frac=float(
                (out.view(torch.int16) != ref_out.view(torch.int16)).float().mean()
            ),
            hash=int((bits * (torch.arange(bits.numel(), device="cuda") % 65521 + 1)).sum().item()),
        )
    return result


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = list(mpi_pool_executor.map(_rank_cell, *zip(*[(cell.spec, seed)] * WORLD)))
    print(f"{cell.why}: {results[0]}")
    for r in results:
        assert r["out_bitwise"] and r["residual_bitwise"], r
        assert r["repeat"] and r["graph"], r
        # The push GEMM rounds each rank's partial (cuBLAS-class error), so the summed rows are a few BF16 steps
        # from the FP32 reference in a fraction of the elements, which the epilogue carries on.
        assert r["res_vs_ref_frac"] < 0.5 and r["out_vs_ref_frac"] < 0.5, r
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"
