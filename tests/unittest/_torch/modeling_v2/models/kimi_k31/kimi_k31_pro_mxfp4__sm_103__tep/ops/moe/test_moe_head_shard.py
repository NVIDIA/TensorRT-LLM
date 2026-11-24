# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 sharded MoE head on one GB300 tray (four MPI ranks).

Every rank builds the engine `AllReduce` module of the TP group (bf16, MNNVL forced: one tray), regenerates the cell's
inputs and runs the sharded head on its slice of the head weight: the gathered router logits and routed latent must
equal the replicated head's (``reference``) bit for bit on every rank, for random inputs at two scales, rows of one
repeated token, large spikes and exact cancellation (K's second half repeats the first with the weight's sign flipped,
so every output is the summation order's own rounding) and all-zero rows, repeated, interleaved with engine
all-reduces on the same workspace, and under CUDA-graph replay. The routed variant (the MoE prologue folded into the
all-gather) must give the bits of the sharded head followed by the MoE prologue with the runner's weights, for three
correction biases (with the zero bias every score of an all-zero row ties, so routing takes its exact top-k fallback).

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import functools
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
    pytest.skip("the sharded head is CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe.moe_head_shard import (  # noqa: E402, E501
    MAX_ROWS,
    compile_shard,
    gather_bytes,
    k31_moe_head_shard,
    k31_moe_head_shard_routed,
    shard_config,
    shard_views,
)

__extra_import_path__ = ["../../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
HIDDEN, ROUTER, LATENT = 7168, 896, 3584
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 3)
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs",
    ),
]


def _battery(rows: int, seed: int):
    """``(x, head weight)`` pairs, identical on every rank."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = (torch.randn(ROUTER + LATENT, HIDDEN, generator=gen, device="cuda") * 0.02).to(
        torch.bfloat16
    )
    cancel = weight.clone()
    cancel[:, HIDDEN // 2 :] = -cancel[:, : HIDDEN // 2]
    pairs = [
        ((torch.randn(rows, HIDDEN, generator=gen, device="cuda") * s).to(torch.bfloat16), weight)
        for s in (0.5, 4.0)
    ]
    one = (torch.randn(1, HIDDEN, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    pairs.append((one.repeat(rows, 1), weight))
    spikes = torch.randn(rows, HIDDEN, generator=gen, device="cuda") * 0.01
    spikes[:, ::97] = 30.0
    pairs.append((spikes.to(torch.bfloat16), weight))
    for _ in range(2):
        x = torch.randn(rows, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
        x[:, HIDDEN // 2 :] = x[:, : HIDDEN // 2]
        pairs.append((x, cancel))
    pairs.append((torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda"), weight))
    return pairs


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E501
        moe_head_shard as kernels,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.oproj_allreduce_push import (  # noqa: E501
        lamport_workspace,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    reg = vars(sys.modules.setdefault("_k31_head_shard", types.ModuleType("_k31_head_shard")))
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    if "ar" not in reg:
        reg["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    ar = reg["ar"]
    rows = spec["rows"]
    width = 2 * kernels.gather_bytes(1, WORLD) // (4 * WORLD)
    lamport_workspace(ar, MAX_ROWS, width, one_shot=True)  # grown before capture
    probe = torch.randn(8, HIDDEN, device="cuda").to(torch.bfloat16)

    def ws():
        return lamport_workspace(ar, rows, width, one_shot=True)

    def same(a, b):
        return bool(torch.equal(a[0].view(torch.int32), b[0].view(torch.int32))) and bool(
            torch.equal(a[1].view(torch.int16), b[1].view(torch.int16))
        )

    equal = repeat = graph_ok = True
    with torch.inference_mode():
        for x, weight in _battery(rows, seed):
            ref = k31_moe_head_shard.reference(x, weight, rank, WORLD, None)
            with validating(k31_moe_head_shard):
                out = k31_moe_head_shard(x, weight, rank, WORLD, ws())
            ar(probe)  # an engine all-reduce on the same workspace in between
            again = k31_moe_head_shard(x, weight, rank, WORLD, ws())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = k31_moe_head_shard(x, weight, rank, WORLD, ws())
                ar(probe)
            graph.replay()
            graph.replay()
            torch.cuda.synchronize()
            equal &= same(out, ref)
            repeat &= same(out, again)
            graph_ok &= same(out, graph_out)
    return dict(rank=rank, equal=equal, repeat=repeat, graph=graph_ok)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", k31_moe_head_shard.CELLS, ids=[c.why[30:70] for c in k31_moe_head_shard.CELLS]
)
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = list(mpi_pool_executor.map(_rank_cell, *zip(*[(spec, seed)] * WORLD)))
    for r in results:
        assert r["equal"] and r["repeat"] and r["graph"], r


def _rank_cell_routed(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe.moe_prologue import (
        k31_moe_prologue,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E501
        moe_head_shard as kernels,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.oproj_allreduce_push import (  # noqa: E501
        lamport_workspace,
    )
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    reg = vars(sys.modules.setdefault("_k31_head_shard", types.ModuleType("_k31_head_shard")))
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    if "ar" not in reg:
        reg["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
    ar = reg["ar"]
    rows = spec["rows"]
    width = 2 * kernels.gather_bytes(1, WORLD) // (4 * WORLD)
    lamport_workspace(ar, MAX_ROWS, width, one_shot=True)  # grown before capture
    probe = torch.randn(8, HIDDEN, device="cuda").to(torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    biases = [torch.zeros(ROUTER, device="cuda")] + [
        torch.randn(ROUTER, generator=gen, device="cuda") * s for s in (0.05, 2.0)
    ]

    def ws():
        return lamport_workspace(ar, rows, width, one_shot=True)

    def same(a, b):
        return all(
            bool(torch.equal(u.contiguous().view(torch.uint8), v.contiguous().view(torch.uint8)))
            for u, v in zip(a, b)
        )

    equal = repeat = graph_ok = True
    with torch.inference_mode():
        for x, weight in _battery(rows, seed):
            for bias in biases:
                logits, latent = k31_moe_head_shard(x, weight, rank, WORLD, ws())
                unfused = k31_moe_prologue(logits, bias, 1.0, latent, True)
                with validating(k31_moe_head_shard_routed):
                    out = k31_moe_head_shard_routed(x, weight, bias, rank, WORLD, ws())
                ar(probe)  # an engine all-reduce on the same workspace in between
                again = k31_moe_head_shard_routed(x, weight, bias, rank, WORLD, ws())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_out = k31_moe_head_shard_routed(x, weight, bias, rank, WORLD, ws())
                    ar(probe)
                graph.replay()
                graph.replay()
                torch.cuda.synchronize()
                equal &= same(out, unfused)
                repeat &= same(out, again)
                graph_ok &= same(out, graph_out)
    return dict(rank=rank, equal=equal, repeat=repeat, graph=graph_ok)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell",
    k31_moe_head_shard_routed.CELLS,
    ids=[c.why[30:70] for c in k31_moe_head_shard_routed.CELLS],
)
def test_routed_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = list(mpi_pool_executor.map(_rank_cell_routed, *zip(*[(spec, seed)] * WORLD)))
    for r in results:
        assert r["equal"] and r["repeat"] and r["graph"], r


@pytest.mark.parametrize(
    "cell", k31_moe_head_shard.CELLS, ids=[c.why[30:70] for c in k31_moe_head_shard.CELLS]
)
def test_compile_shard_cells(cell, monkeypatch) -> None:
    """The target's load-time compile (``compile_shard``) of every cell's config, on one GPU: the slice GEMM runs
    once, the gather, the routed gather + MoE prologue and, where the config takes it, the push GEMM and its poll
    kernel compile, and compiling again adds nothing (a cell's first call under CUDA-graph capture compiles
    nothing). ``gather_bytes`` is the kernel's workspace size. The compile caches are process-wide, so each cell
    compiles into empty ones: a first call that compiled nothing fails."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E501
        moe_head_push as push,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe import (  # noqa: E501
        moe_head_shard as kernels,
    )

    rows = cell.spec["rows"]
    config = shard_config(rows)
    assert gather_bytes(rows, WORLD) == kernels.gather_bytes(rows, WORLD)
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    weight = (torch.randn(ROUTER + LATENT, HIDDEN, generator=gen, device="cuda") * 0.02).to(
        torch.bfloat16
    )
    w_latent, w_router = shard_views(weight, 1, WORLD)
    x = torch.randn(rows, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)

    for module, name in (
        (kernels, "_compiled_gather"),
        (kernels, "_compiled_route_quant"),
        (push, "_compiled_poll"),
    ):
        cached = getattr(module, name)
        monkeypatch.setattr(
            module, name, functools.lru_cache(**cached.cache_parameters())(cached.__wrapped__)
        )
    monkeypatch.setattr(push, "_PUSH_COMPILED", {})

    def compiled():
        return (
            kernels._compiled_gather.cache_info().currsize,
            kernels._compiled_route_quant.cache_info().currsize,
            len(push._PUSH_COMPILED),
            push._compiled_poll.cache_info().currsize,
        )

    assert compiled() == (0, 0, 0, 0)
    compile_shard(WORLD, x, w_latent, w_router, config, True, routed=True)
    torch.cuda.synchronize()
    first = compiled()
    assert first[0] > 0 and first[1] > 0
    if push.push_serves(config):
        assert first[2] > 0 and first[3] > 0
    compile_shard(WORLD, x, w_latent, w_router, config, True, routed=True)
    compile_shard(WORLD, x, w_latent, w_router, config, True, routed=False)
    torch.cuda.synchronize()
    assert compiled() == first


def test_split_map() -> None:
    """The K split per row count is cuBLAS's for the replicated head at 33-176 rows (CPU)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe.moe_head_shard import (  # noqa: E501
        MAX_ROWS,
        MIN_ROWS,
        shard_config,
        shard_serves,
    )

    assert shard_config(MIN_ROWS - 1) is None and shard_config(MAX_ROWS + 1) is None
    assert (MIN_ROWS, MAX_ROWS) == (33, 176)
    for rows in range(MIN_ROWS, MAX_ROWS + 1):
        _, _, latent, router = shard_config(rows)
        assert router == ((19, 38) if rows <= 64 else (28,))
        assert latent == ((28,) if rows <= 48 or 65 <= rows <= 96 else ())
    assert shard_serves(40, 16) and shard_serves(40, 8) and shard_serves(40, 4)
    assert not shard_serves(40, 1)
