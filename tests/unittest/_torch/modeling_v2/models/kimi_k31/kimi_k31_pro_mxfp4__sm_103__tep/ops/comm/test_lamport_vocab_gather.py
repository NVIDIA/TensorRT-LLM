# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 lamport_vocab_gather entry on one GB300 tray (four MPI ranks).

Every rank builds the engine `AllReduce` module of the TP group (bf16, MNNVL forced: one tray),
regenerates every rank's BF16 shard from the cell's seed and gathers its own through the entry: the
result must equal the reference bit for bit on every rank (the FP32 concatenation, with the sentinel
word's -0.0 as +0.0) and the engine gather + `.float()` value for value, repeated, interleaved with
engine all-reduces and FP32 Lamport all-gathers on the same workspace, and under CUDA-graph replay. A gather
of more rows than one call carries runs in row chunks; it must equal one kernel call over all its rows bit for
bit (that call on a workspace grown for it, which the entry itself never does).

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
    pytest.skip("the gather is a CuTe DSL kernel", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.lamport_vocab_gather import (  # noqa: E402, E501
    lamport_vocab_gather,
)

__extra_import_path__ = ["../../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

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


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (spec["rows"], spec["shard"])
    xs = [
        (torch.randn(shape, generator=gen, device="cuda") * 8).to(torch.bfloat16)
        for _ in range(WORLD)
    ]
    if spec["special"]:
        flat = xs[1].view(-1)
        flat[0], flat[1], flat[2], flat[3] = 0.0, -0.0, float("inf"), float("-inf")
        flat[4], flat[5] = float("nan"), -0.0  # -0.0 after a non-zero even column
        flat[6], flat[7] = -0.0, 0.0
        xs[2].view(-1)[8], xs[2].view(-1)[9] = 0.0, -0.0
        xs[3].view(-1)[-2], xs[3].view(-1)[-1] = 0.0, -0.0
    return xs


#: Rows of the one-call comparison: past one call's 25 rows, up to six chunks.
ONE_CALL_ROWS = (26, 35, 48, 50, 60, 64, 75, 76, 100, 128)


def _rank_one_call(rows_list, seed: int) -> dict:
    """Every rank's chunked gathers against one kernel call over all rows (run first in the pool: it grows the
    workspace for those calls before any CUDA graph captures it)."""
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm import (
        lamport_allgather as ag,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm import (
        lamport_vocab_gather as vgm,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    reg = vars(sys.modules.setdefault("_k31_lamport_vg", types.ModuleType("_k31_lamport_vg")))
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    if "ar" not in reg:
        reg["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    ar = reg["ar"]
    shard = vgm._VOCAB // WORLD
    # Twice the largest call: a call's trailing loads read up to the same size again past its rows.
    vgm.lamport_workspace(ar, 2 * max(rows_list), shard, one_shot=True)
    results = {}
    with torch.inference_mode():
        for rows in rows_list:
            for special in (False, True):
                xs = _inputs(dict(rows=rows, shard=shard, special=special), seed + rows)
                x = xs[rank]
                chunked = lamport_vocab_gather(x, mapping, ar)
                ws = vgm.lamport_workspace(ar, 2 * rows, shard, one_shot=True)
                one = torch.empty_like(chunked)
                ag.run_lamport_vocab_gather(ws, x, one, True)
                torch.cuda.synchronize()
                results[(rows, special)] = (
                    len(vgm.row_chunks(rows, vgm.rows_per_call(shard, WORLD))),
                    bool(torch.equal(chunked.view(torch.int32), one.view(torch.int32))),
                )
    return dict(rank=rank, results=results)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_chunked_equals_one_call(mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(_rank_one_call, *zip(*[(ONE_CALL_ROWS, 1234)] * WORLD)))
    for r in results:
        for (rows, special), (chunks, equal) in r["results"].items():
            assert chunks == -(-rows // 25), (r["rank"], rows, chunks)
            assert equal, (r["rank"], rows, special)


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm import (
        lamport_vocab_gather as vgm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.lamport_allgather import (  # noqa: E501
        lamport_allgather,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.allgather_module import (
        allgather_module,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    reg = vars(sys.modules.setdefault("_k31_lamport_vg", types.ModuleType("_k31_lamport_vg")))
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    if "ar" not in reg:
        reg["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    ar = reg["ar"]
    xs = _inputs(spec, seed)
    x = xs[rank]
    ref = lamport_vocab_gather.reference(xs, mapping, ar)
    probe = torch.randn(8, 7168, device="cuda").to(torch.bfloat16)
    pairs = torch.randn(8, 64, device="cuda")
    with torch.inference_mode():
        engine = allgather_module(x, mapping, -1, None).float()
        with validating(lamport_vocab_gather):
            out = lamport_vocab_gather(x, mapping, ar)
        ar(probe)  # an engine all-reduce on the same workspace in between
        lamport_allgather(pairs, mapping, ar)  # and the FP32 all-gather
        again = lamport_vocab_gather(x, mapping, ar)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = lamport_vocab_gather(x, mapping, ar)
            ar(probe)
            graph_again = lamport_vocab_gather(x, mapping, ar)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()

    def same(a, b):
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))

    served = vgm.lamport_vocab_gather_serves(x, mapping, ar)
    return dict(
        rank=rank,
        served=served,
        shape=tuple(out.shape),
        equal=same(out, ref) if served else None,
        engine=bool(torch.equal(out.nan_to_num(), engine.nan_to_num())),
        engine_bits=same(out, engine),
        repeat=same(out, again),
        graph=same(out, graph_out) and same(out, graph_again),
    )


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", lamport_vocab_gather.CELLS, ids=[c.why[4:70] for c in lamport_vocab_gather.CELLS]
)
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = list(mpi_pool_executor.map(_rank_cell, *zip(*[(spec, seed)] * WORLD)))
    for r in results:
        assert r["shape"] == (spec["rows"], WORLD * spec["shard"]), r
        assert r["engine"] and r["repeat"] and r["graph"], r
        if r["served"]:
            assert r["equal"], r
        if not spec["special"]:
            # Random shards carry no sentinel word: bitwise the engine gather + .float().
            assert r["engine_bits"], r
    assert [r["served"] for r in results] == [True] * WORLD
