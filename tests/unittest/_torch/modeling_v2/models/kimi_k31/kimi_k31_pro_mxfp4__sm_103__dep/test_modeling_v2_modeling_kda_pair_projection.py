# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep target's pair-sharded KDA [q | k | v | g] projection is bitwise the per-rank projection.

Under attention DP each rank of a pair keeps one head half of the [q | k | v | g] weight, multiplies both ranks'
rows by it and hands every row's halves back to its owner (``_K31KdaPair.project``). Each rank's rows must come out
bit for bit as the full GEMM the rank would run alone: the DEP16 low-M plan's kernel for its generation row count.
The two-rank test runs the projection end to end over the pair's group exchange, with the eager row-count exchange,
for one width-5 verify request on one rank and four on the other.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

__extra_import_path__ = [".."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    on_sm,
    pickle_by_value,
    run_per_rank,
)

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip(
        "the low-M kernels and the group exchange are CuTe DSL kernels", allow_module_level=True
    )

pickle_by_value(__name__)

_MODELING = (
    "tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.modeling"
)
_HIDDEN, _PROJ = 7168, 12288

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    on_sm(103, reason="the DEP16 plan is certified on sm_103"),
]


def _weights(seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(4 * _PROJ, _HIDDEN, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)


def _half(w: torch.Tensor, r: int):
    """Pair rank r's half of ``w`` and its full-width view (the half followed by zeros), as the target pools them."""
    half = _PROJ // 2
    pool = torch.zeros(2 * 4 * half, _HIDDEN, dtype=w.dtype, device=w.device)
    pool[: 4 * half] = torch.cat(
        [w[s * _PROJ + r * half : s * _PROJ + (r + 1) * half] for s in range(4)]
    )
    return pool[: 4 * half], pool


def _rows(n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(n, _HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)


def _full(modeling, x: torch.Tensor, w: torch.Tensor, generation: bool, plan) -> torch.Tensor:
    """What a rank projects alone: the plan's GEMM for a generation part, cuBLAS for a context part."""
    if generation:
        return modeling.k31_low_m_linear(x, w, plan, True)
    return F.linear(x.unsqueeze(0), w).squeeze(0)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def _rank_projection(case: tuple) -> dict:
    import tensorrt_llm
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    modeling = importlib.import_module(_MODELING)
    generation, rows = case
    mapping = Mapping(world_size=2, rank=rank, gpus_per_node=4, tp_size=2, enable_attention_dp=True)
    group = modeling._k31_kda_pair_group(mapping)
    pair = modeling._K31KdaPair.of(group, layer_idx=0)
    plan = modeling.k31_low_m_plan(16, True)
    w = _weights(11)
    x = _rows(rows[rank], 100 + rank)
    kinds = modeling._k31_kda_pair_kinds(plan, rows, 4 * _PROJ, _HIDDEN, generation)
    own = (0, rows[rank]) if generation else (rows[rank], 0)
    meta = SimpleNamespace(num_ctx_tokens=own[0], num_tokens=own[0] + own[1])
    with torch.inference_mode():
        counts = pair.step_counts(meta, 0)
        out = pair.project(x, rows, _half(w, rank), kinds, not generation)
        ref = _full(modeling, x, w, generation, plan)
    torch.cuda.synchronize()
    expected = ((0, rows[0]), (0, rows[1])) if generation else ((rows[0], 0), (rows[1], 0))
    return dict(rank=rank, counts=counts == expected, equal=_same(out, ref))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs of a tray")
@pytest.mark.gpu2
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_pair_projection_end_to_end(mpi_pool_executor) -> None:
    results = run_per_rank(mpi_pool_executor, _rank_projection, (True, (5, 20)), world=2)
    for r in results:
        assert r["counts"] and r["equal"], r
