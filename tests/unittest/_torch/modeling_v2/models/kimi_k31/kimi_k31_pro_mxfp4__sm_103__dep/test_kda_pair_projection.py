# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep target's pair-sharded KDA [q | k | v | g] projection is bitwise the per-rank projection.

Under attention DP each rank of a pair keeps one head half of the [q | k | v | g] weight, multiplies both ranks'
rows by it and hands every row's halves back to its owner (``_K31KdaPair.project``). Each rank's rows must come out
bit for bit as the full GEMM the rank would run alone: the DEP16 low-M plan's kernel for its generation row count,
cuBLAS for a context part. The single-GPU test checks the half GEMMs over equal and unequal row counts of the pair;
the two-rank test runs the projection end to end over the pair's group exchange, eager and replayed from a CUDA
graph, with the eager row-count exchange.

MPI-spawning (the two-rank test): under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import importlib
import pickle
import sys
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
import torch.nn.functional as F
from mpi4py import MPI

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip(
        "the low-M kernels and the group exchange are CuTe DSL kernels", allow_module_level=True
    )

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_MODELING = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.modeling"
_HIDDEN, _PROJ = 7168, 12288

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
        reason="the DEP16 plan is certified on sm_103",
    ),
]

# (generation or context part, pair rank 0's rows, pair rank 1's rows)
_GENERATION = [
    (1, 1),  # one-token decode: chunked TGV, cuBLAS's 7-way split
    (2, 2),
    (4, 4),  # top of the chunked TGV row
    (3, 7),  # the ranks' rows on different plan rows
    (5, 5),  # one width-5 verify request: the chunked split-K row
    (8, 8),
    (16, 16),  # TGV, top of its first plan row
    (20, 20),  # TGV, second plan row
    (32, 32),
    (40, 40),  # cuBLAS from here
    (80, 80),
    (100, 100),  # c512 rows per rank
    (120, 120),
    (160, 160),
    (5, 20),  # an eager step: the ranks' rows on different plan rows
    (10, 0),  # one rank without generation rows
    (0, 35),
    (100, 35),
]
# context: cuBLAS splits K by the output width at 5-8 rows, so those run through the full-width view
_CONTEXT = [
    (1, 1),
    (6, 6),
    (7, 33),
    (8, 100),
    (64, 65),
    (512, 512),
    (1000, 2954),
    (4096, 0),
    (8192, 8192),
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


@pytest.mark.parametrize(
    "generation,rows",
    [(True, r) for r in _GENERATION] + [(False, r) for r in _CONTEXT],
    ids=[f"gen{r}" for r in _GENERATION] + [f"ctx{r}" for r in _CONTEXT],
)
def test_half_gemms_bitwise(generation, rows) -> None:
    """Both ranks' half GEMMs over the pair's interleaved rows equal the full GEMM's columns of every rank's rows."""
    modeling = importlib.import_module(_MODELING)
    plan = modeling._k31_decode_plan(SimpleNamespace(tp_size=16, enable_attention_dp=True))
    w = _weights(11)
    xs = [_rows(n, 100 + r) for r, n in enumerate(rows)]
    m = max(rows)
    padded = [torch.cat([x, x.new_zeros((m - x.shape[0], _HIDDEN))]) for x in xs]
    gathered = torch.stack(padded, dim=1).reshape(2 * m, _HIDDEN)  # rank r's row t at t * 2 + r
    kinds = modeling._k31_kda_pair_kinds(plan, rows, 4 * _PROJ, _HIDDEN, generation)
    with torch.inference_mode():
        halves = [
            modeling._k31_kda_pair_halves(gathered, *_half(w, h), kinds, rows, not generation)
            for h in range(2)
        ]
        for r, x in enumerate(xs):
            if x.shape[0] == 0:
                continue
            full = _full(modeling, x, w, generation, plan).view(x.shape[0], 4, 2, _PROJ // 2)
            for h in range(2):
                got = halves[h].view(m, 2, 4, _PROJ // 2)[: x.shape[0], r]
                assert _same(got, full[:, :, h]), (rows, r, h)


def test_decode_plan() -> None:
    """The attention-DP decode plan is DEP16's plan."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import (
        K31_LOW_M_PLANS,
    )

    modeling = importlib.import_module(_MODELING)
    plan = modeling._k31_decode_plan(SimpleNamespace(tp_size=16, enable_attention_dp=True))
    assert plan is K31_LOW_M_PLANS["dep16"]


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
        again = pair.project(x, rows, _half(w, rank), kinds, not generation)
        graph_ok = None
        if generation and rows[0] == rows[1]:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = pair.project(x, rows, _half(w, rank), kinds, not generation)
            graph.replay()
            graph.replay()
            torch.cuda.synchronize()
            graph_ok = _same(graph_out, out)
        ref = _full(modeling, x, w, generation, plan)
    torch.cuda.synchronize()
    expected = ((0, rows[0]), (0, rows[1])) if generation else ((rows[0], 0), (rows[1], 0))
    return dict(
        rank=rank,
        counts=counts == expected,
        equal=_same(out, ref),
        repeat=_same(out, again),
        graph=graph_ok,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs of a tray")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@pytest.mark.parametrize(
    "case",
    [
        (True, (100, 100)),
        (True, (5, 5)),
        (True, (5, 20)),
        (True, (0, 35)),
        (False, (512, 1000)),
        (False, (6, 0)),
    ],
    ids=["gen100", "gen5", "gen5-20", "gen0-35", "ctx512-1000", "ctx6-0"],
)
def test_pair_projection_end_to_end(case, mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(_rank_projection, [case] * 2))
    for r in results:
        assert r["counts"] and r["equal"] and r["repeat"], r
        assert r["graph"] in (None, True), r
