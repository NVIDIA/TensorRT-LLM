# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensor-parallel collectives must trace under ``torch.compile(fullgraph=True)``.

With ``TLLM_DISABLE_MPI=1`` (the VisualGen / Ray path) every TP collective used
to break the graph twice per call: once in the ``@torch.compiler.disable``d
``DeviceMeshTopologyImpl._get_mesh_dim_by_name`` helper behind the
``*_group_pg`` properties, and once on ``ProcessGroup.boxed()`` feeding the
``trtllm::*_pg`` C++ ops (dynamo has no fake class for a c10d ProcessGroup).
That made ``fullgraph=True`` impossible for TP DiT blocks.

These tests pin the fix: mesh-dim lookups are cached dict reads, and the
``trtllm::*_pg_by_name`` ops take the group *name* (a plain ``str``) and
resolve the group inside the opaque op.  Every test compiles with
``fullgraph=True``, so any graph break fails the test outright.

The second half pins the functional-collective route ``AllReduce`` takes for
the plain NCCL all-reduce when MPI is disabled: eager results are bitwise equal
to the C++ by-name route, Inductor re-inplaces the collective onto the dead
GEMM output under ``torch.compile``, and the route captures into a CUDA graph.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_tp_fullgraph_compile.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode

import tensorrt_llm._torch.distributed.ops as dist_ops
from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
    allgather,
    reducescatter,
)
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.visual_gen.args import AttentionConfig


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (same pattern as test_tp_attention.py)
# =============================================================================


def _init_worker(rank: int, world_size: int, port: int, backend: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _distributed_worker(rank, world_size, test_fn, port, backend):
    try:
        _init_worker(rank, world_size, port, backend)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(world_size: int, test_fn: Callable, backend: str = "nccl"):
    if backend == "nccl" and torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    from ._visual_gen_dist_utils import spawn_with_retry

    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker,
            args=(world_size, test_fn, port, backend),
            nprocs=world_size,
            join=True,
        )
    )


def _fresh_tp_mapping(rank: int, world_size: int) -> VisualGenMapping:
    """Build a pure-TP VisualGenMapping on a freshly reset class-level mesh."""
    DeviceMeshTopologyImpl.device_mesh = None
    DeviceMeshTopologyImpl.tp_mesh = None
    VisualGenMapping.seq_mesh = None
    return VisualGenMapping(world_size=world_size, rank=rank, tp_size=world_size)


def _compile_fullgraph(fn):
    torch._dynamo.reset()
    return torch.compile(fn, fullgraph=True, dynamic=False)


def _allreduce_pair(mapping):
    """Same TP group, two routes: c10d functional collective vs. C++ by-name op."""
    functional = AllReduce(
        mapping=mapping, strategy=AllReduceStrategy.NCCL, use_functional_nccl=True
    )
    cpp = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL, use_functional_nccl=False)
    assert functional._use_functional_nccl and not cpp._use_functional_nccl
    return functional, cpp


class _DispatchRecorder(TorchDispatchMode):
    """Records every op that reaches the dispatcher.  Both all-reduce routes
    produce identical bytes, so equal outputs cannot prove which one ran."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func)
        return func(*args, **(kwargs or {}))


def _dispatched(fn, *args, **kwargs):
    with _DispatchRecorder() as recorder:
        out = fn(*args, **kwargs)
    return out, recorder.ops


def _functional_op():
    return torch.ops._c10d_functional.all_reduce.default


def _cpp_op():
    return torch.ops.trtllm.allreduce_pg_by_name.default


# =============================================================================
# Test bodies (run inside spawned ranks)
# =============================================================================


def _mapping_lookups_body(rank: int, world_size: int):
    """Both mapping flavours feed collectives; their group lookups must trace."""
    vgm = _fresh_tp_mapping(rank, world_size)
    # to_llm_mapping() yields a DeviceMeshTopology whose *_group_pg properties go
    # through DeviceMeshTopologyImpl -- the path that used to carry the
    # @torch.compiler.disable'd helper.  This is what AllReduce receives.
    llm_mapping = vgm.to_llm_mapping()
    device = torch.device(f"cuda:{rank}")
    x = torch.ones(8, device=device)

    def fn(t):
        return (
            t
            + vgm.tp_group_pg.size()
            + vgm.tp_rank
            + len(vgm.tp_group)
            + llm_mapping.tp_group_pg.size()
            + llm_mapping.tp_rank
            + len(llm_mapping.tp_group)
            + llm_mapping._get_mesh_dim_by_name("tp").size()
        )

    eager = fn(x)
    compiled = _compile_fullgraph(fn)(x)
    torch.testing.assert_close(compiled, eager)
    # Same cached group object served to both mappings.
    assert llm_mapping.tp_group_pg is vgm.tp_group_pg

    # pp/cp and the flattened-tp MoE layout are served by the same cache: every
    # *_group_pg must be a cached object and fold to a constant under compile.
    for kwargs in (
        {"tp_size": 1, "pp_size": world_size},
        {"tp_size": world_size, "moe_tp_size": 1, "moe_ep_size": world_size},
    ):
        DeviceMeshTopologyImpl.device_mesh = None
        DeviceMeshTopologyImpl.tp_mesh = None
        m = Mapping(world_size=world_size, rank=rank, gpus_per_node=world_size, **kwargs)
        m.build_mesh()
        names = ["tp", "pp", "cp"] + (["moe_tp", "moe_ep"] if m.moe_ep_size > 1 else [])
        for name in names:
            pg = getattr(m, f"{name}_group_pg")
            assert pg is getattr(m, f"{name}_group_pg")
            assert pg is DeviceMeshTopologyImpl._group_cache[name]
        assert m.tp_group_pg.size() == m.tp_size and m.pp_group_pg.size() == m.pp_size
        assert m.cp_group_pg.size() == 1
        if m.moe_ep_size > 1:
            assert m.moe_tp_group_pg.size() == 1 and m.moe_ep_group_pg.size() == world_size

        def h(t):
            return t + m.tp_group_pg.size() + m.pp_group_pg.size() + m.cp_group_pg.size()

        torch.testing.assert_close(_compile_fullgraph(h)(x), h(x))


def _allreduce_body(rank: int, world_size: int):
    vgm = _fresh_tp_mapping(rank, world_size)
    mapping = vgm.to_llm_mapping()
    device = torch.device(f"cuda:{rank}")
    # The constructor default follows the module switch; only strategy NCCL
    # qualifies (NCCL_SYMMETRIC is the same C++ call today but stays on it).
    default = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)
    assert default._use_functional_nccl == dist_ops._USE_FUNCTIONAL_NCCL_ALLREDUCE
    symmetric = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL_SYMMETRIC)
    assert not symmetric._use_functional_nccl

    x = torch.full((4, 16), float(rank + 1), device=device, dtype=torch.bfloat16)
    expected = torch.full_like(x, float(world_size * (world_size + 1) // 2))

    # Both routes must trace fullgraph, whichever one the module default selects.
    for allreduce in _allreduce_pair(mapping):

        def fn(t):
            return allreduce(t)

        torch.testing.assert_close(fn(x), expected)
        torch.testing.assert_close(_compile_fullgraph(fn)(x), expected)


def _allgather_reducescatter_body(rank: int, world_size: int):
    vgm = _fresh_tp_mapping(rank, world_size)
    mapping = vgm.to_llm_mapping()
    device = torch.device(f"cuda:{rank}")

    x = torch.full((2, 8), float(rank + 1), device=device, dtype=torch.bfloat16)
    expected_ag = torch.cat(
        [
            torch.full((2, 8), float(r + 1), device=device, dtype=torch.bfloat16)
            for r in range(world_size)
        ],
        dim=0,
    )

    def ag(t):
        return allgather(t, mapping, dim=0)

    torch.testing.assert_close(ag(x), expected_ag)
    torch.testing.assert_close(_compile_fullgraph(ag)(x), expected_ag)

    # Identical input on every rank -> reduce-scatter == my slice * world_size.
    y = torch.arange(world_size * 2 * 8, device=device, dtype=torch.bfloat16).view(
        world_size * 2, 8
    )
    expected_rs = y[rank * 2 : (rank + 1) * 2] * world_size

    def rs(t):
        return reducescatter(t, mapping, dim=0)

    torch.testing.assert_close(rs(y), expected_rs)
    torch.testing.assert_close(_compile_fullgraph(rs)(y), expected_rs)


def _build_tp_attention(vgm: VisualGenMapping, rank: int, device: torch.device) -> Attention:
    """Column-parallel QKV, row-parallel out-proj with all-reduce; each rank
    owns its own TP shard, seeded so two builds get identical weights."""
    config = DiffusionModelConfig(
        mapping=vgm.to_llm_mapping(),
        visual_gen_mapping=vgm,
        attention=AttentionConfig(backend="VANILLA"),
        skip_create_weights_in_init=False,
    )
    attn = Attention(256, 8, qkv_mode=QKVMode.FUSE_QKV, qk_norm=False, config=config).to(device)
    torch.manual_seed(1234 + rank)
    with torch.no_grad():
        for p in attn.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
    return attn


def _tp_attention_body(rank: int, world_size: int):
    """A real TP DiT attention block compiles with fullgraph=True, matches eager,
    and its row-parallel Linear -> AllReduce takes the default route; the other
    route gives the same bytes eager and compiled."""
    from torch._inductor.utils import run_and_get_code

    vgm = _fresh_tp_mapping(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    attn = _build_tp_attention(vgm, rank, device)
    # Linear resolves the route from the module default when it constructs its
    # AllReduce, so the twin is built with the switch flipped.
    default_route = dist_ops._USE_FUNCTIONAL_NCCL_ALLREDUCE
    dist_ops._USE_FUNCTIONAL_NCCL_ALLREDUCE = not default_route
    try:
        twin = _build_tp_attention(vgm, rank, device)
    finally:
        dist_ops._USE_FUNCTIONAL_NCCL_ALLREDUCE = default_route

    torch.manual_seed(42)  # same activations on every rank
    x = torch.randn(2, 32, 256, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        eager = attn(x)
        eager_twin = twin(x)
        compiled, codes = run_and_get_code(_compile_fullgraph(attn), x)
        compiled_twin = _compile_fullgraph(twin)(x)
        explanation = torch._dynamo.explain(attn)(x)
    torch.testing.assert_close(compiled, eager, atol=2e-2, rtol=2e-2)
    assert explanation.graph_break_count == 0, explanation.break_reasons

    code = "\n".join(codes)
    functional_marker = "_c10d_functional.all_reduce_.default"
    cpp_marker = "allreduce_pg_by_name"
    if default_route:
        assert functional_marker in code and cpp_marker not in code, code
    else:
        assert cpp_marker in code and functional_marker not in code, code
    assert torch.equal(eager, eager_twin)
    assert torch.equal(compiled, compiled_twin)


# =============================================================================
# Functional-collective route for the plain NCCL all-reduce
# =============================================================================


class _GemmAllReduce(torch.nn.Module):
    """Row-parallel GEMM feeding an all-reduce.  The GEMM output is dead after
    the reduction, so Inductor may run the functional all-reduce in place on it."""

    def __init__(self, weight: torch.Tensor, allreduce: AllReduce):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.allreduce = allreduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.allreduce(torch.nn.functional.linear(x, self.weight))


def _functional_eager_body(rank: int, world_size: int):
    """Eager: the functional route is bitwise equal to today's C++ route, each
    instance really dispatches its own op, and fused params bypass the route."""
    vgm = _fresh_tp_mapping(rank, world_size)
    functional, cpp = _allreduce_pair(vgm.to_llm_mapping())
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device).manual_seed(100 + rank)
    # Shapes a TP DiT block reduces: row-parallel GEMM outputs and the TP-aware
    # RMSNorm variance (fp32, one value per token).
    cases = [
        ((2, 1024, 5120), torch.bfloat16),
        ((1, 37, 1536), torch.bfloat16),
        ((2, 1024, 1), torch.float32),
        ((3, 7, 64), torch.float16),
    ]
    for shape, dtype in cases:
        x = torch.randn(shape, device=device, dtype=dtype, generator=gen)
        x_saved = x.clone()
        ref, ops_c = _dispatched(cpp, x)
        out, ops_f = _dispatched(functional, x)
        assert _functional_op() in ops_f and _cpp_op() not in ops_f, ops_f
        assert _cpp_op() in ops_c and _functional_op() not in ops_c, ops_c
        assert type(out) is torch.Tensor and out.shape == ref.shape and out.dtype == ref.dtype
        assert out.data_ptr() != x.data_ptr() and torch.equal(x, x_saved)
        assert torch.equal(out, ref), f"functional != cpp route for {shape} {dtype}"

    # The functional route has no epilogue: a fused request must fall through to
    # the C++ op on both instances and keep its [normed, residual] outputs.
    x = torch.randn((2, 64, 512), device=device, dtype=torch.bfloat16, generator=gen)
    residual = torch.randn_like(x)
    norm_weight = torch.randn(512, device=device, dtype=torch.bfloat16, generator=gen)

    def fused_params():
        return AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm_weight,
            eps=1e-6,
        )

    ref, ops_c = _dispatched(cpp, x, all_reduce_params=fused_params())
    out, ops_f = _dispatched(functional, x, all_reduce_params=fused_params())
    assert _cpp_op() in ops_f and _functional_op() not in ops_f, ops_f
    assert _cpp_op() in ops_c
    assert isinstance(out, (list, tuple)) and len(out) == 2, type(out)
    assert len(ref) == 2 and all(torch.equal(a, b) for a, b in zip(out, ref))


def _functional_compiled_body(rank: int, world_size: int):
    """Compiled: Inductor re-inplaces the functional all-reduce onto the dead GEMM
    output, and the result equals both eager and today's route compiled."""
    from torch._inductor.utils import run_and_get_code

    vgm = _fresh_tp_mapping(rank, world_size)
    functional, cpp = _allreduce_pair(vgm.to_llm_mapping())
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(7 + rank)  # each rank owns its own TP shard
    weight = torch.randn(256, 128, device=device, dtype=torch.bfloat16) * 0.02
    mod_f = _GemmAllReduce(weight, functional)
    mod_c = _GemmAllReduce(weight, cpp)
    torch.manual_seed(42)  # same activations on every rank
    x = torch.randn(2, 64, 128, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        eager_f, eager_c = mod_f(x), mod_c(x)
        compiled_f, codes = run_and_get_code(_compile_fullgraph(mod_f), x)
        compiled_c = _compile_fullgraph(mod_c)(x)
    code = "\n".join(codes)
    assert "_c10d_functional.all_reduce_.default" in code, code
    assert "_c10d_functional.all_reduce.default" not in code, code
    assert torch.equal(eager_f, eager_c)
    assert torch.equal(compiled_f, eager_f)
    assert torch.equal(compiled_f, compiled_c)


def _functional_cuda_graph_body(rank: int, world_size: int):
    """CUDA graph: the eager functional route captures, and replay == eager."""
    vgm = _fresh_tp_mapping(rank, world_size)
    functional, cpp = _allreduce_pair(vgm.to_llm_mapping())
    device = torch.device(f"cuda:{rank}")
    static_x = torch.randn(2, 256, 1024, device=device, dtype=torch.bfloat16)

    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(3):
            functional(static_x)
    torch.cuda.current_stream(device).wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = functional(static_x)

    for seed in (1, 2):
        torch.manual_seed(seed * 1000 + rank)
        x = torch.randn_like(static_x)
        static_x.copy_(x)
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(static_out, functional(x))
        assert torch.equal(static_out, cpp(x))


def _functional_route_gloo_body(rank: int, world_size: int):
    """CPU/gloo: the route is a plain c10d all-reduce by group name and traces
    without a graph break.  The C++ route needs CUDA, so it is not compared here."""
    from torch.distributed.device_mesh import init_device_mesh

    # build_mesh() hard-codes a CUDA mesh; install a CPU one in the class cache.
    DeviceMeshTopologyImpl.device_mesh = init_device_mesh(
        "cpu", mesh_shape=(1, world_size, 1), mesh_dim_names=("pp", "tp", "cp")
    )
    DeviceMeshTopologyImpl.tp_mesh = None
    DeviceMeshTopologyImpl._populate_group_cache()
    mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)
    allreduce = AllReduce(
        mapping=mapping, strategy=AllReduceStrategy.NCCL, use_functional_nccl=True
    )
    assert allreduce._use_functional_nccl

    x = torch.randn(3, 5, 8, generator=torch.Generator().manual_seed(11 + rank))
    expected = x.clone()
    dist.all_reduce(expected)
    out, ops = _dispatched(allreduce, x)
    assert _functional_op() in ops and _cpp_op() not in ops, ops
    assert type(out) is torch.Tensor and out.data_ptr() != x.data_ptr()
    assert torch.equal(out, expected)

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(allreduce)(x)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    compiled = torch.compile(allreduce, backend="aot_eager", fullgraph=True, dynamic=False)
    assert torch.equal(compiled(x), expected)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("world_size", [2])
def test_mapping_group_lookups_trace_fullgraph(world_size):
    _run(world_size, _mapping_lookups_body)


@pytest.mark.parametrize("world_size", [2])
def test_allreduce_nccl_fullgraph(world_size):
    _run(world_size, _allreduce_body)


@pytest.mark.parametrize("world_size", [2])
def test_allgather_reducescatter_fullgraph(world_size):
    _run(world_size, _allgather_reducescatter_body)


@pytest.mark.parametrize("world_size", [2])
def test_tp_attention_module_fullgraph(world_size):
    _run(world_size, _tp_attention_body)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_allreduce_functional_route_matches_cpp_eager(world_size):
    _run(world_size, _functional_eager_body)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_allreduce_functional_route_compiled_inplace(world_size):
    _run(world_size, _functional_compiled_body)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_allreduce_functional_route_cuda_graph(world_size):
    _run(world_size, _functional_cuda_graph_body)


@pytest.mark.cpu_only
@pytest.mark.parametrize("world_size", [2])
def test_allreduce_functional_route_gloo_cpu(world_size):
    _run(world_size, _functional_route_gloo_body, backend="gloo")
