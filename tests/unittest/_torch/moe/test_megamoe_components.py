# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for MegaMoE resource ownership, using device/collective doubles."""

from __future__ import annotations

import ast
import gc
import importlib.util
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_only
_ROOT = Path(__file__).resolve().parents[4]
_COMPONENTS = _ROOT / "tensorrt_llm/_torch/moe/fused_moe/mega_moe"
_OP = "tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op"
_PACKAGE = "_megamoe_test._torch.moe.fused_moe.mega_moe"


class _Tensor:
    def __init__(self, shape, dtype, device, trace, name, rows=None):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype, self.device = dtype, device
        self.trace, self.name = trace, name
        self.rows = rows

    def __getitem__(self, key):
        rows = key[0] if isinstance(key, tuple) else key
        return _Tensor(self.shape, self.dtype, self.device, self.trace, self.name, rows)

    def view(self, dtype):
        return self

    def copy_(self, source, *, non_blocking):
        assert non_blocking
        self.trace.append(("copy", self.name, self.rows, source.name))

    def zero_(self):
        self.trace.append(("zero", self.name, self.rows))

    def fill_(self, value):
        self.trace.append(("fill", self.name, self.rows, value))


class _MpiError(Exception):
    pass


@pytest.fixture
def components(monkeypatch):
    """Import actual component modules without loading CUDA/native package initialization."""
    trace = []
    torch = ModuleType("torch")
    for name in ("uint8", "int32", "int64", "float32", "bfloat16"):
        setattr(torch, name, name)

    def allocate(kind, shape, value=None, *, dtype, device):
        name = f"tensor{len(trace)}"
        trace.append(("allocate", kind, shape, dtype, device, value))
        return _Tensor(shape, dtype, device, trace, name)

    torch.empty = lambda shape, **kw: allocate("empty", shape, **kw)
    torch.zeros = lambda shape, **kw: allocate("zeros", shape, **kw)
    torch.full = lambda shape, value, **kw: allocate("full", shape, value, **kw)
    torch.cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 2)
    torch.device = lambda kind, index: (kind, index)
    dist = ModuleType("torch.distributed")
    dist.is_available = lambda: True
    dist.is_initialized = lambda: False
    dist.init_process_group = MagicMock()
    dist.get_world_size = lambda: 8
    dist.group = SimpleNamespace(WORLD=object())
    torch.distributed = dist
    utils = ModuleType("tensorrt_llm._utils")
    utils.get_sm_version = lambda: 107
    utils.MPI = SimpleNamespace(Exception=_MpiError)
    utils.mpi_world_size = lambda: 8
    utils.mpi_rank = lambda: 3
    comm = SimpleNamespace(bcast=MagicMock(return_value=("127.0.0.1", "29500")))
    utils.mpi_comm = lambda: comm
    op = ModuleType(_OP)
    op.megamoe_activation_sf_bytes_per_row = lambda hidden: ((hidden + 15) // 16 + 3) // 4 * 4
    op.set_active_megamoe_profiling_scratch = MagicMock()
    op.set_active_megamoe_profiling_scratch_factory = MagicMock()
    op.get_megamoe_profiling_scratch = MagicMock()
    for name, value in {
        "torch": torch,
        "torch.distributed": dist,
        "tensorrt_llm._utils": utils,
        "tensorrt_llm.logger": SimpleNamespace(logger=MagicMock()),
        "_megamoe_test._torch.cute_dsl_utils": SimpleNamespace(IS_CUTLASS_DSL_AVAILABLE=False),
        _OP: op,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)

    loaded = {}
    for name in ("runtime", "workspace", "profiling"):
        module_name = f"{_PACKAGE}.cute_dsl_{name}"
        spec = importlib.util.spec_from_file_location(
            module_name, _COMPONENTS / f"cute_dsl_{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(
        trace=trace, torch=torch, dist=dist, utils=utils, comm=comm, op=op, **loaded
    )


def _acquire(c, workspace, *, bucket=8, query=None):
    return workspace.acquire(
        top_k=2,
        hidden=80,
        device="cpu",
        output_dtype=c.torch.bfloat16,
        launch_max_T=bucket,
        query_local_kernel=query or MagicMock(return_value=(64, True)),
    )


def test_local_workspace_reuses_buffers_and_initializes_counter_prefix(components):
    c = components
    workspace = c.workspace.MegaMoeWorkspace(world_size=1, topk_index_dtype="int32", full_bucket=8)
    query = MagicMock(return_value=(64, True))
    first = _acquire(c, workspace, query=query)
    assert first.activation.shape == (8, 40)
    assert first.activation_sf.shape == (8, 8)
    assert first.combine_output.shape == (8, 1, 80)
    assert first.topk_idx_local.dtype == "int32"
    assert ("allocate", "zeros", 64, "uint8", "cpu", None) in c.trace
    assert ("allocate", "full", (8, 2), "int32", "cpu", -1) in c.trace
    c.trace.clear()
    second = _acquire(c, workspace, query=query)
    assert first.activation is second.activation
    assert first.shared_workspace is second.shared_workspace
    assert c.trace == []
    query.assert_called_once_with(8, 2, 80)
    assert workspace.supports_direct_inputs(8)
    assert not workspace.supports_direct_inputs(4)
    third = _acquire(c, workspace, bucket=4, query=query)
    assert third.activation is not first.activation
    assert query.call_count == 2


def test_multirank_workspace_only_allocates_local_routes(components):
    c = components
    workspace = c.workspace.MegaMoeWorkspace(world_size=8, topk_index_dtype="int64", full_bucket=8)
    regions = SimpleNamespace(
        **{
            name: object()
            for name in (
                "activation",
                "activation_sf",
                "topk_weights",
                "combine_output",
                "shared_workspace",
                "peer_offsets",
            )
        }
    )
    provider = SimpleNamespace(num_topk=2, get_regions=lambda: regions)
    workspace.register_provider(8, provider, supports_direct_inputs=True)
    query = MagicMock(
        side_effect=AssertionError("forward must not query/allocate symmetric memory")
    )
    bufs = _acquire(c, workspace, query=query)
    assert len(c.trace) == 1
    assert bufs.topk_idx_local.dtype == "int64"
    assert bufs.activation is regions.activation
    assert bufs.peer_offsets is regions.peer_offsets
    assert workspace.full_provider is provider
    assert workspace.supports_direct_inputs(8)
    query.assert_not_called()
    workspace.register_provider(8, provider, supports_direct_inputs=False)
    assert not workspace.supports_direct_inputs(8)
    provider.num_topk = 3
    with pytest.raises(c.runtime.MegaMoeCuteDslUnavailable, match="top_k"):
        _acquire(c, workspace, query=query)


def test_multirank_workspace_rejects_missing_build_time_provider(components):
    c = components
    workspace = c.workspace.MegaMoeWorkspace(world_size=8, topk_index_dtype="int32", full_bucket=8)
    with pytest.raises(c.runtime.MegaMoeCuteDslUnavailable, match="create_weights"):
        _acquire(c, workspace)


@pytest.mark.parametrize("stage_activation", [True, False])
def test_staging_preserves_input_order_and_resets_each_bucket_tail(components, stage_activation):
    c = components
    workspace = c.workspace.MegaMoeWorkspace(world_size=1, topk_index_dtype="int32", full_bucket=8)
    bufs = _acquire(c, workspace)
    x = _Tensor((8, 40), "uint8", "cpu", c.trace, "x")
    sf = _Tensor((8, 8), "uint8", "cpu", c.trace, "sf")
    routes = _Tensor((8, 2), "int32", "cpu", c.trace, "routes")
    weights = _Tensor((8, 2), "float32", "cpu", c.trace, "weights")

    def stage(buffer, rows):
        workspace.stage_inputs(
            bufs=buffer,
            x=x,
            x_sf=sf,
            topk_idx=routes,
            topk_weights=weights,
            num_tokens=rows,
            top_k=2,
            stage_activation=stage_activation,
        )

    stage(bufs, 8)
    c.trace.clear()
    stage(bufs, 3)
    assert ("zero", bufs.topk_weights.name, slice(3, 8)) in c.trace
    assert ("fill", bufs.topk_idx_local.name, slice(3, 8), -1) in c.trace
    route_copy = c.trace.index(("copy", bufs.topk_idx_local.name, slice(None, 3), "routes"))
    weight_copy = c.trace.index(("copy", bufs.topk_weights.name, slice(None, 3), "weights"))
    assert route_copy < weight_copy
    if stage_activation:
        activation_copy = c.trace.index(("copy", bufs.activation.name, slice(None, 3), "x"))
        assert route_copy < activation_copy < weight_copy
    else:
        assert not any(entry[0] == "copy" and entry[-1] in ("x", "sf") for entry in c.trace)
    small = _acquire(c, workspace, bucket=4)
    stage(small, 4)
    c.trace.clear()
    stage(bufs, 0)
    assert not any(entry[0] == "copy" for entry in c.trace)
    assert ("fill", bufs.topk_idx_local.name, slice(0, 3), -1) in c.trace
    assert ("zero", bufs.topk_weights.name, slice(0, 8)) in c.trace


def _profiling(c):
    scratch = c.profiling.MegaMoeProfilingScratch()
    scratch.configure(
        c.profiling.MegaMoeScratchRequest(
            process_group=object(),
            world_size=8,
            rank=3,
            hidden_size=80,
            max_tokens_per_rank=8,
            num_topk=2,
            output_dtype="bfloat16",
            combine_format="bf16",
            shared_workspace_bytes=64,
        )
    )
    return scratch


@pytest.mark.parametrize(
    "world,tuning,has_factory", [(1, True, False), (8, False, False), (8, True, True)]
)
def test_profiling_allocation_is_deferred_to_miss(components, world, tuning, has_factory):
    c = components
    scratch = _profiling(c)
    with scratch.activate(world_size=world, tuning=tuning):
        factory = c.op.set_active_megamoe_profiling_scratch_factory.call_args.args[0]
        assert callable(factory) == has_factory
        c.op.get_megamoe_profiling_scratch.assert_not_called()
    c.op.set_active_megamoe_profiling_scratch.assert_called_with(None)
    c.op.set_active_megamoe_profiling_scratch_factory.assert_called_with(None)


def test_profiling_does_not_keep_released_provider_alive(components):
    c = components
    scratch = _profiling(c)

    class Provider:
        def get_regions(self):
            return "regions"

    provider = Provider()
    c.op.get_megamoe_profiling_scratch.return_value = provider
    with scratch.activate(world_size=8, tuning=True):
        factory = c.op.set_active_megamoe_profiling_scratch_factory.call_args.args[0]
        assert factory() == "regions"
    with scratch.activate(world_size=8, tuning=True):
        c.op.set_active_megamoe_profiling_scratch.assert_called_with("regions")
        c.op.set_active_megamoe_profiling_scratch_factory.assert_called_with(None)
    c.op.get_megamoe_profiling_scratch.assert_called_once()
    reference = weakref.ref(provider)
    c.op.get_megamoe_profiling_scratch.return_value = None
    del provider
    gc.collect()
    assert reference() is None
    with scratch.activate(world_size=8, tuning=True):
        assert callable(c.op.set_active_megamoe_profiling_scratch_factory.call_args.args[0])


def test_profiling_clears_op_context_after_exception(components):
    c = components
    with pytest.raises(RuntimeError, match="launch failed"):
        with _profiling(c).activate(world_size=8, tuning=True):
            raise RuntimeError("launch failed")
    c.op.set_active_megamoe_profiling_scratch.assert_called_with(None)
    c.op.set_active_megamoe_profiling_scratch_factory.assert_called_with(None)


def test_mpi_bootstrap_preserves_executor_device_and_collective_order(components, monkeypatch):
    c = components
    monkeypatch.setenv("MASTER_ADDR", "different-per-rank-value")
    monkeypatch.setenv("MASTER_PORT", "unused")
    c.runtime.maybe_init_torch_dist_under_mpi(ep_size=8)
    c.comm.bcast.assert_called_once_with(None, root=0)
    c.dist.init_process_group.assert_called_once_with(
        backend="cuda:nccl,cpu:gloo", rank=3, world_size=8, device_id=("cuda", 2)
    )
    assert c.runtime.os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert c.runtime.os.environ["MASTER_PORT"] == "29500"


def test_mpi_bootstrap_rejects_hybrid_before_any_collective(components):
    c = components
    with pytest.raises(ValueError, match="pure DEP"):
        c.runtime.maybe_init_torch_dist_under_mpi(ep_size=4)
    c.comm.bcast.assert_not_called()
    c.dist.init_process_group.assert_not_called()


def test_ep_resolution_prefers_mapping_then_world_and_rejects_missing_subgroup(components):
    c = components
    c.dist.is_initialized = lambda: True
    mapped = object()
    assert (
        c.runtime.resolve_ep_process_group(
            SimpleNamespace(moe_ep_group_pg=mapped), ep_size=8, layer_idx=0
        )
        is mapped
    )
    assert (
        c.runtime.resolve_ep_process_group(SimpleNamespace(), ep_size=8, layer_idx=None)
        is c.dist.group.WORLD
    )
    with pytest.raises(RuntimeError, match="strict subset"):
        c.runtime.resolve_ep_process_group(SimpleNamespace(), ep_size=4, layer_idx=1)
    c.comm.bcast.assert_not_called()


def test_runtime_probe_keeps_capability_cache_per_architecture(components):
    c = components
    first = c.runtime.is_megamoe_cute_dsl_runtime_available()
    assert first[0] is False
    assert "Cutlass DSL is not importable" in first[1]
    c.runtime.IS_CUTLASS_DSL_AVAILABLE = True
    assert c.runtime.is_megamoe_cute_dsl_runtime_available() == first
    assert set(c.runtime._RUNTIME_PROBE_CACHE) == {107}
    c.runtime.IS_CUTLASS_DSL_AVAILABLE = False
    c.runtime.get_sm_version = lambda: 100
    c.runtime.is_megamoe_cute_dsl_runtime_available()
    assert set(c.runtime._RUNTIME_PROBE_CACHE) == {100, 107}


def _kernel_launcher(c, tuner):
    """Execute the actual launch method without importing CUDA backend dependencies."""
    source = _COMPONENTS / "mega_moe_cute_dsl.py"
    tree = ast.parse(source.read_text())
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrtllmCutedslMegaMoeNvfp4Impl"
    )
    method = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == "_launch_megamoe_kernel"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            method,
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": c.torch,
        "AutoTuner": tuner,
        "ActivationType": SimpleNamespace(SwigluBias="bias", SiTu="situ"),
        "_as_nvfp4": lambda tensor: tensor,
        "_as_fp8_sf": lambda tensor: tensor,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize(
    "world,tactic_autotune,queries_tuner",
    [(1, False, False), (8, False, False), (1, True, False), (8, True, True)],
)
def test_kernel_launch_only_queries_tuner_for_multirank_autotuning(
    components, world, tactic_autotune, queries_tuner
):
    c = components
    tuner = SimpleNamespace(get=MagicMock(return_value=SimpleNamespace(is_tuning_mode=True)))
    op = MagicMock()
    c.torch.ops = SimpleNamespace(trtllm=SimpleNamespace(cute_dsl_megamoe_nvfp4_blackwell=op))
    owner = SimpleNamespace(
        tactic_autotune=tactic_autotune,
        _profiling_scratch=_profiling(c),
        _use_in_kernel_fc2_reduce=lambda _: False,
        expert_size_per_partition=4,
        intermediate_size_per_partition=64,
        expand_intermediate_size_per_partition=128,
        apply_topk_in_fc1=False,
        act_clamp=None,
        act_alpha=None,
        act_beta=None,
        activation_type="plain",
        combine_format="bf16",
    )
    tensor = _Tensor((8, 1, 80), "bfloat16", "cpu", c.trace, "output")
    weights = SimpleNamespace(
        **{
            name: tensor
            for name in (
                "fc1_weight",
                "fc1_weight_sf",
                "fc2_weight",
                "fc2_weight_sf",
                "fc31_alpha",
                "fc2_alpha",
                "fc1_norm_const",
            )
        }
    )
    result = _kernel_launcher(c, tuner)(
        owner,
        activation=tensor,
        activation_sf=tensor,
        topk_idx=tensor,
        topk_weights=tensor,
        weight_view=weights,
        combine_output=tensor,
        shared_workspace=tensor,
        world_size=world,
        local_rank=0,
        top_k=2,
        hidden=80,
        peer_offsets=list(range(world)),
        num_tokens=0,
        output_dtype="bfloat16",
        launch_max_T=8,
    )
    assert result.shape == (0, 80)
    op.assert_called_once()
    if queries_tuner:
        tuner.get.assert_called_once_with()
    else:
        tuner.get.assert_not_called()
    c.op.set_active_megamoe_profiling_scratch.assert_called_with(None)
    c.op.set_active_megamoe_profiling_scratch_factory.assert_called_with(None)


@pytest.mark.parametrize("error", [_MpiError, RuntimeError, ValueError])
def test_mpi_rank_query_failure_leaves_collectives_uninitialized(components, error):
    c = components
    c.utils.mpi_world_size = MagicMock(side_effect=error("rank query unavailable"))
    c.runtime.maybe_init_torch_dist_under_mpi(ep_size=8)
    c.comm.bcast.assert_not_called()
    c.dist.init_process_group.assert_not_called()
