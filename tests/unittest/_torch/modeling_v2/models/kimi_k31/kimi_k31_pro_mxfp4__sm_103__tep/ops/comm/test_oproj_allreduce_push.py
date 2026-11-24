# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 oproj_allreduce_push entry on one GB300 tray (four MPI ranks).

Every rank builds the engine `AllReduce` module the target builds for the attention TP group (bf16,
MNNVL forced: one tray), regenerates every rank's o_proj input and weight shard from the cell's
seed, calls the entry and gates it against the reference over all four ranks. It also records how
the result compares with the current target path (`F.linear` + `allreduce_module`), runs the call
inside a CUDA graph, and checks every rank got the same bits.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import pickle
import sys
import types
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the push chain is a CuTe DSL kernel pair", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.oproj_allreduce_push import (  # noqa: E402, E501
    oproj_allreduce_push,
)

__extra_import_path__ = ["../../../../.."]
from _validating import ULP, validating  # noqa: E402 — needs the path declared above

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 3), (10, 7))
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs (sm_107 driven separately)",
    ),
]

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


def _deviation(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    o, r = out.float(), ref.float()
    row = r.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / row).max().item() / ULP[torch.bfloat16]
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    return element, rms / ULP[torch.bfloat16]


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm import (
        oproj_allreduce_push as opm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (
        allreduce_module,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    reg = vars(sys.modules.setdefault("_k31_oproj_ar", types.ModuleType("_k31_oproj_ar")))
    if "ar" not in reg:
        mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
        reg["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    ar = reg["ar"]
    assert ar.mnnvl_allreduce is not None, "MNNVL was not built on this tray"

    xs, ws = _inputs(spec, seed)
    x, w = xs[rank], ws[rank]
    before = x.clone()
    with torch.inference_mode():
        with validating(oproj_allreduce_push):
            out = oproj_allreduce_push(x, w, ar)
        again = oproj_allreduce_push(x, w, ar)
        current = allreduce_module(torch.nn.functional.linear(x, w), ar, 0, None, None)
        ref = oproj_allreduce_push.reference(xs, ws, ar)
        # Under a CUDA graph (the served decode path), after the eager warm-up above.
        graph_out = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = oproj_allreduce_push(x, w, ar)
        graph.replay()
        torch.cuda.synchronize()
        element, rms = _deviation(out, ref)
        bits = out.contiguous().view(torch.int16).to(torch.int64)
        weights = torch.arange(1, bits.numel() + 1, device="cuda", dtype=torch.int64) % 65521
        digest = int((bits.flatten() * weights).sum().item())
    return dict(
        rank=rank,
        shape=tuple(out.shape),
        dtype=str(out.dtype),
        served=opm.push_serves(x, w, ar),
        element_ulp=element,
        rms_ulp=rms,
        hash=digest,
        repeat_equal=bool(torch.equal(out, again)),
        graph_equal=bool(torch.equal(out, graph_out)),
        current_equal=bool(torch.equal(out.view(torch.int16), current.view(torch.int16))),
        input_untouched=bool(torch.equal(x, before)),
    )


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell",
    oproj_allreduce_push.CELLS,
    ids=[c.why[4:72] for c in oproj_allreduce_push.CELLS],
)
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    assert spec["world"] == WORLD
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = sorted(
        mpi_pool_executor.map(_rank_cell, *zip(*[(spec, seed)] * WORLD)),
        key=lambda r: r["rank"],
    )
    worst = max(r["element_ulp"] for r in results), max(r["rms_ulp"] for r in results)
    print(
        f"{cell.why}: served {results[0]['served']}, element {worst[0]:.3f} ulp, "
        f"rms {worst[1]:.3f} ulp, bitwise vs current path {results[0]['current_equal']}"
    )
    for r in results:
        assert r["shape"] == (spec["rows"], spec["hidden"]) and r["dtype"] == "torch.bfloat16", r
        assert r["input_untouched"], r
        assert r["repeat_equal"] and r["graph_equal"], f"rank {r['rank']}: not deterministic"
        assert r["element_ulp"] <= ELEMENT_ULP and r["rms_ulp"] <= RMS_ULP, r
        if not r["served"]:
            # The fallback is the current target path itself.
            assert r["current_equal"], r
        if spec["k"] == 384 and spec["rows"] > 48:
            # TP32 above 48 rows: bitwise to the current path (cuBLAS + the engine all-reduce).
            assert r["current_equal"], r
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"


def test_gate_refuses_a_dropped_rank() -> None:
    spec = dict(world=WORLD, rows=8, k=768, hidden=7168)
    xs, ws = _inputs(spec, seed=1)
    right = oproj_allreduce_push.reference(xs, ws, None)
    wrong = oproj_allreduce_push.reference(xs[:-1], ws[:-1], None)
    element, rms = _deviation(wrong, right)
    assert element > ELEMENT_ULP and rms > RMS_ULP, (element, rms)


def test_guard_refuses_fp32() -> None:
    x = torch.randn(4, 768, device="cuda")
    w = torch.randn(7168, 768, device="cuda")
    with pytest.raises(AssertionError, match="bf16"):
        with validating(oproj_allreduce_push):
            oproj_allreduce_push(x, w, None)
