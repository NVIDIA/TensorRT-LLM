# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 oproj_push_attn_res_add_rmsnorm entry on one GB300 tray (4 ranks).

Every rank builds the engine AllReduce module (MNNVL forced, as the attention TP16 / TP32 groups
resolve it), regenerates every rank's o_proj input / weight shard from the cell's seed and calls
the entry. Gates: against the reference within the attn_res band; the updated residual bitwise
equal to ``prefix + oproj_allreduce_push(...)`` (push GEMM + finish kernel) and the output bitwise
equal to ``k31_attn_res_add_rmsnorm`` on that sum (the finishing mix reproduces the finish kernel);
every rank the same bits; a repeat and a CUDA-graph replay the same bits.

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

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.norm.oproj_push_attn_res_add_rmsnorm import (  # noqa: E501
    k31_oproj_push_attn_res_add_rmsnorm as op,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
H = 7168
EPS = 1e-5
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
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.norm.attn_res import (  # noqa: E501
        k31_attn_res_add_rmsnorm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.oproj_allreduce_push import (  # noqa: E501
        oproj_allreduce_push,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.norm.oproj_push_attn_res_add_rmsnorm import (  # noqa: E501
        k31_oproj_push_attn_res_add_rmsnorm as rank_op,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm import (
        _attn_res_common as _c,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mods = vars(sys.modules.setdefault("_k31_push_mix", types.ModuleType("_k31_push_mix")))
    if "ar" not in mods:
        mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank)
        mods["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    ar = mods["ar"]
    inputs, weights, prefix, block, mix = _inputs(spec, seed)
    args = (inputs[rank], weights[rank], ar, prefix, block, *mix, EPS, EPS)
    with torch.inference_mode():
        rank_op.is_valid(*args)
        oproj_allreduce_push(inputs[rank], weights[rank], ar)  # grow the workspace eagerly
        updated, out = rank_op(*args)
        again_u, again_o = rank_op(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_u, g_o = rank_op(*args)
        graph.replay()
        two_kernel = oproj_allreduce_push(inputs[rank], weights[rank], ar)
        want_u, want_o = k31_attn_res_add_rmsnorm(prefix, two_kernel, block, *mix, EPS, EPS)
        # The path without the push chain: cuBLAS + the engine all-reduce, then the mix.
        today = ar(torch.nn.functional.linear(inputs[rank], weights[rank]))
        today_u, today_o = k31_attn_res_add_rmsnorm(prefix, today, block, *mix, EPS, EPS)
        ref_u, _ = rank_op.reference(inputs, weights, prefix, block, *mix, EPS, EPS)
        # The mix + norm band is the attn_res kernel's, on the residual the kernel produced.
        ref_o = _c.reference(updated, block, *mix, EPS, EPS)
        torch.cuda.synchronize()
        bits = out.view(torch.int16).to(torch.int64).flatten()
        result = dict(
            rank=rank,
            updated_bitwise=bool(
                torch.equal(updated.view(torch.int16), (prefix + two_kernel).view(torch.int16))
            ),
            out_bitwise=bool(torch.equal(out.view(torch.int16), want_o.view(torch.int16))),
            today_bitwise=bool(
                torch.equal(updated.view(torch.int16), today_u.view(torch.int16))
                and torch.equal(out.view(torch.int16), today_o.view(torch.int16))
            ),
            repeat=bool(torch.equal(out, again_o) and torch.equal(updated, again_u)),
            graph=bool(torch.equal(out, g_o) and torch.equal(updated, g_u)),
            out_vs_ref=_one_step(out, ref_o, mix[2]),
            upd_vs_ref_frac=float(
                (updated.view(torch.int16) != ref_u.view(torch.int16)).float().mean()
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
        assert r["updated_bitwise"] and r["out_bitwise"], r
        assert r["repeat"] and r["graph"], r
        assert r["out_vs_ref"], r
        # The push GEMM rounds each rank's partial (cuBLAS-class error), so the summed addend is a
        # few BF16 steps from the FP32 reference in a fraction of the elements.
        assert r["upd_vs_ref_frac"] < 0.5, r
        if cell.spec["k"] == 384 and cell.spec["rows"] > 48:
            # TP32 above 48 rows: the bits of cuBLAS + the engine all-reduce + the mix.
            assert r["today_bitwise"], r
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"


def _rank_one_shot_bits(spec: dict, seed: int) -> dict:
    """Outputs of the entry with the one-shot transport forced for a row count where the engine takes
    two-shot (``FUSED_ONE_SHOT_MAX_ROWS``) against the two-shot transport, eager and in a CUDA graph."""
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm import (
        lamport_ar as lam,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.oproj_allreduce_push import (  # noqa: E501
        oproj_allreduce_push,
        push_serves,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.norm import (
        oproj_push_attn_res_add_rmsnorm as mod,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mods = vars(sys.modules.setdefault("_k31_push_mix", types.ModuleType("_k31_push_mix")))
    if "ar" not in mods:
        mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank)
        mods["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    ar = mods["ar"]
    inputs, weights, prefix, block, mix = _inputs(spec, seed)
    args = (inputs[rank], weights[rank], ar, prefix, block, *mix, EPS, EPS)
    saved = dict(mod.FUSED_ONE_SHOT_MAX_ROWS)
    got = {}
    try:
        with torch.inference_mode():
            oproj_allreduce_push(inputs[rank], weights[rank], ar)  # grow the workspace eagerly
            for name, cap in (("two_shot", 0), ("one_shot", spec["rows"])):
                mod.FUSED_ONE_SHOT_MAX_ROWS[WORLD] = cap
                mod.lamport_workspace(ar, spec["rows"], H, True)
                eager = op(*args)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    in_graph = op(*args)
                graph.replay()
                torch.cuda.synchronize()
                got[name] = (eager, (in_graph[0].clone(), in_graph[1].clone()))
    finally:
        mod.FUSED_ONE_SHOT_MAX_ROWS.clear()
        mod.FUSED_ONE_SHOT_MAX_ROWS.update(saved)

    def same(a, b):
        return all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(a, b))

    return dict(
        rank=rank,
        served=push_serves(inputs[rank], weights[rank], ar),
        engine_two_shot=not lam.one_shot_for(
            spec["rows"], H, WORLD, mod.DEFAULT_ONE_SHOT_MAX_BYTES
        ),
        eager=same(got["one_shot"][0], got["two_shot"][0]),
        graph=same(got["one_shot"][1], got["two_shot"][1]),
        graph_eager=same(got["one_shot"][1], got["two_shot"][0]),
    )


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("rows,candidates", ((24, 5), (40, 9)))
def test_one_shot_finish_keeps_two_shot_bits(rows, candidates, mpi_pool_executor) -> None:
    spec = dict(rows=rows, k=768, candidates=candidates)
    seed = 1000 * rows + candidates
    results = list(mpi_pool_executor.map(_rank_one_shot_bits, *zip(*[(spec, seed)] * WORLD)))
    print(results[0])
    for r in results:
        assert r["served"] and r["engine_two_shot"], r
        assert r["eager"] and r["graph"] and r["graph_eager"], r
