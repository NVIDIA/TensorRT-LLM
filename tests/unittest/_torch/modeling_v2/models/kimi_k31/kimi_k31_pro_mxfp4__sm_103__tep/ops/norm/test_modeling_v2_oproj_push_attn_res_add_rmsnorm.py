# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 oproj_push_attn_res_add_rmsnorm entry on one GB300 tray (4 ranks).

On a sample of the entry's cells (``sample_cells``), every rank builds the engine AllReduce module
(MNNVL forced, as the attention TP16 / TP32 groups resolve it), regenerates every rank's o_proj
input / weight shard from the cell's seed and calls the entry. Gates: against the reference within
the attn_res band; the updated residual bitwise equal to ``prefix + oproj_allreduce_push(...)``
(push GEMM + finish kernel) and the output bitwise equal to ``k31_attn_res_add_rmsnorm`` on that
sum (the finishing mix reproduces the finish kernel).

MPI-spawning: run under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.norm.oproj_push_attn_res_add_rmsnorm import (  # noqa: E501
    k31_oproj_push_attn_res_add_rmsnorm as op,
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
EPS = 1e-5
pytestmark = on_ranks(WORLD, 103, cutedsl=True)


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, k, n = spec["rows"], spec["k"], spec["candidates"]

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    inputs = [r(rows, k) for _ in range(WORLD)]
    weights = [r(H, k, scale=k**-0.5) for _ in range(WORLD)]
    prefix = r(rows, H, scale=4.0)
    block = r(n - 1, rows, H, scale=4.0)
    mix = (r(H, scale=0.05), (1 + r(H, scale=0.1).float()).bfloat16(), r(H))
    return inputs, weights, prefix, block, mix


def _one_step(got: torch.Tensor, want: torch.Tensor, out_w: torch.Tensor) -> bool:
    g, w = got.float(), want.float()
    weight = out_w.float().abs()

    def ulp(x):
        return torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(2.0**-126))) - 7)

    normed = (w.abs() / weight.clamp_min(2.0**-20)).clamp_min(1.0)
    tol = 2 * ulp(normed) * weight + ulp(torch.maximum(g.abs(), w.abs()))
    diff = (g - w).abs()
    return bool(torch.all(diff <= tol * 1.0001)) and (diff > 0).float().mean().item() < 0.005


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.norm.attn_res import (  # noqa: E501
        k31_attn_res_add_rmsnorm,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.oproj_allreduce_push import (  # noqa: E501
        oproj_allreduce_push,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm import _attn_res_common as _c

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    _, ar = tray_allreduce(rank, WORLD)
    inputs, weights, prefix, block, mix = _inputs(spec, seed)
    args = (inputs[rank], weights[rank], ar, prefix, block, *mix, EPS, EPS)
    with torch.inference_mode():
        op.is_valid(*args)
        oproj_allreduce_push(inputs[rank], weights[rank], ar)  # grow the workspace eagerly
        updated, out = op(*args)
        two_kernel = oproj_allreduce_push(inputs[rank], weights[rank], ar)
        _, want_o = k31_attn_res_add_rmsnorm(prefix, two_kernel, block, *mix, EPS, EPS)
        # The mix + norm band is the attn_res kernel's, on the residual the kernel produced.
        ref_o = _c.reference(updated, block, *mix, EPS, EPS)
        return dict(
            rank=rank,
            updated_bitwise=bool(
                torch.equal(updated.view(torch.int16), (prefix + two_kernel).view(torch.int16))
            ),
            out_bitwise=bool(torch.equal(out.view(torch.int16), want_o.view(torch.int16))),
            out_vs_ref=_one_step(out, ref_o, mix[2]),
        )


CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    seed = zlib.crc32(cell.why.encode()) % 2**31
    for r in run_per_rank(mpi_pool_executor, _rank_cell, cell.spec, seed, world=WORLD):
        assert r["updated_bitwise"] and r["out_bitwise"] and r["out_vs_ref"], r
