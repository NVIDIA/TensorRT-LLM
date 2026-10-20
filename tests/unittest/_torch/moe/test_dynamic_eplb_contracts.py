# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast CPU contracts for the per-layer dynamic EPLB serving path."""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib.util
import math
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_TORCH_ROOT = _ROOT / "tensorrt_llm/_torch"
_QUEUE = "CUDA_SCALE_LAUNCH_QUEUES"
_SCHEDULER_RUNTIME = _TORCH_ROOT / "moe/kernels/megamoe_scheduler/cuda_scheduler/runtime.py"
# The scheduler's native bound and the bound of the default 1800 s MoE
# communication warmup timeout at the nominal 2 GHz clock.
_SERVING_SPIN_CYCLES = 8_000_000_000
_WARMUP_SPIN_CYCLES = 1800 * 2_000_000_000


def _extract_class(
    path: Path,
    name: str,
    methods: set[str] | None,
    namespace: dict,
    bases: Tuple[str, ...] = (),
):
    tree = ast.parse(path.read_text())
    original = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )
    if methods is None:
        selected = original
    else:
        body = [
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
        assert {node.name for node in body} == methods
        selected = ast.ClassDef(
            name=name,
            bases=[ast.Name(id=base, ctx=ast.Load()) for base in bases],
            keywords=[],
            body=body,
            decorator_list=[],
        )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _extract_function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text())
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def test_shared_slot_provider_uses_model_owned_pool_state():
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/rebalance_live_arena.py"
    provider_cls = _extract_class(
        source,
        "SharedSlotArenaProvider",
        {"__init__"},
        {"Any": Any, "Optional": Optional, "Tuple": Tuple},
    )
    first_owner = SimpleNamespace(
        _rebalance_shared_slot_pools={},
        _rebalance_shared_slot_pool_order=[],
    )
    second_owner = SimpleNamespace(
        _rebalance_shared_slot_pools={},
        _rebalance_shared_slot_pool_order=[],
    )

    first = provider_cls(first_owner)
    same_model = provider_cls(first_owner)
    second = provider_cls(second_owner)

    assert first._pools is same_model._pools
    assert first._pool_order is same_model._pool_order
    assert first._pools is not second._pools
    with pytest.raises(TypeError, match="model-local pool state"):
        provider_cls(SimpleNamespace())


def test_shared_slot_provider_pool_key_includes_complete_bundle():
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/rebalance_live_arena.py"

    class FakePool:
        instances = []

        def __init__(self, **kwargs):
            self.bundle = kwargs["bundle"]
            self._closed = False
            self.group_sizes = ()
            self.instances.append(self)

        def build_layer(self, comm):
            return object(), {}, object(), 0

    provider_cls = _extract_class(
        source,
        "SharedSlotArenaProvider",
        {"__init__", "build_hierarchical_live_arena"},
        {
            "Any": Any,
            "Dict": dict,
            "Optional": Optional,
            "Tuple": Tuple,
            "_SharedSlotPool": FakePool,
        },
    )

    @dataclasses.dataclass(frozen=True)
    class Plane:
        name: str
        nbytes: int
        descriptor: str

    @dataclasses.dataclass(frozen=True)
    class Bundle:
        hidden: int
        intermediate: int
        planes: tuple[Plane, ...]

    # Equal names and byte sizes but different view descriptors must not share.
    first_bundle = Bundle(160, 80, (Plane("weight", 12800, "storage-a"),))
    second_bundle = Bundle(160, 80, (Plane("weight", 12800, "storage-b"),))
    owner = SimpleNamespace(
        _rebalance_shared_slot_pools={},
        _rebalance_shared_slot_pool_order=[],
    )
    kwargs = dict(
        world=8,
        rank=0,
        home_count=4,
        helper_count=1,
        device=0,
        comm=object(),
    )
    first_provider = provider_cls(owner)
    first_provider.build_hierarchical_live_arena(bundle=first_bundle, **kwargs)
    with pytest.raises(RuntimeError, match="single-geometry"):
        first_provider.build_hierarchical_live_arena(bundle=second_bundle, **kwargs)
    provider_cls(owner).build_hierarchical_live_arena(bundle=second_bundle, **kwargs)

    assert len(owner._rebalance_shared_slot_pools) == 2
    assert [pool.bundle for pool in FakePool.instances] == [first_bundle, second_bundle]


@pytest.mark.parametrize("per_layer_helper_sets", [False, True])
def test_shared_slot_provider_pool_key_includes_helper_set_mode(per_layer_helper_sets):
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/rebalance_live_arena.py"

    class FakePool:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._closed = False
            self.group_sizes = ()
            self.instances.append(self)

        def build_layer(self, comm):
            return object(), {}, object(), 0

    provider_cls = _extract_class(
        source,
        "SharedSlotArenaProvider",
        {"__init__", "build_hierarchical_live_arena"},
        {
            "Any": Any,
            "Dict": dict,
            "Optional": Optional,
            "Tuple": Tuple,
            "_SharedSlotPool": FakePool,
        },
    )
    owner = SimpleNamespace(
        _rebalance_shared_slot_pools={},
        _rebalance_shared_slot_pool_order=[],
    )
    kwargs = dict(world=8, rank=0, home_count=4, helper_count=1, bundle="bundle", device=0)
    provider_cls(owner, per_layer_helper_sets=per_layer_helper_sets).build_hierarchical_live_arena(
        comm=object(), **kwargs
    )
    provider_cls(
        owner, per_layer_helper_sets=not per_layer_helper_sets
    ).build_hierarchical_live_arena(comm=object(), **kwargs)

    # One pool per mode: a per-layer pool never hands out a shared bank.
    assert [pool.kwargs["per_layer_sets"] for pool in FakePool.instances] == [
        per_layer_helper_sets,
        not per_layer_helper_sets,
    ]
    assert len(owner._rebalance_shared_slot_pools) == 2


def test_hierarchical_fused_submit_launches_bound_scheduler_and_copy():
    source = _TORCH_ROOT / "moe/kernels/megamoe_scheduler/sami/hierarchical.py"
    launches = []

    def submit_fused(ctx, flags_mc, route_epoch, route_target, *scheduler_args):
        launches.append((ctx, flags_mc, route_epoch, route_target, scheduler_args))

    torch = SimpleNamespace(
        ops=SimpleNamespace(trtllm=SimpleNamespace(moe_rebalance_tma_submit_fused=submit_fused)),
    )
    stream_switches = []
    ticket_cls = _extract_class(
        source,
        "HierarchicalCopyTicket",
        None,
        {"dataclass": dataclasses.dataclass, "field": dataclasses.field},
    )
    broadcast_cls = _extract_class(
        source,
        "HierarchicalSamiWeightBroadcast",
        {"submit_fused", "_require_live_bank_reusable"},
        {
            "torch": torch,
            "HierarchicalCopyTicket": ticket_cls,
            "enter_cuda_stream": lambda stream: stream_switches.append(("enter", stream)) or stream,
            "exit_cuda_stream": lambda state: stream_switches.append(("exit", state)),
        },
    )
    outputs = object()
    requests = []

    def fused_submit_args(routes, stream_handle, *, spin_cycles):
        requests.append((routes, stream_handle, spin_cycles))
        return ("scheduler", routes)

    broadcast = broadcast_cls.__new__(broadcast_cls)
    broadcast._lock = nullcontext()
    broadcast._handoff_aborted = False
    broadcast._outputs = outputs
    broadcast._scheduler = SimpleNamespace(fused_submit_args=fused_submit_args)
    broadcast._reuse_authority = object()
    broadcast._validate_reuse_authority = lambda authority: None
    broadcast._active_generation = 0
    broadcast._collective_safe_generation = 0
    broadcast._generation = 1
    broadcast._stream = 19
    broadcast._stream_owner = object()
    broadcast.device = 0
    broadcast.ctx = "ctx"
    broadcast.flags_mc = 4096
    broadcast.terminal_flags = "terminals"

    ticket = broadcast.submit_fused(outputs, "routes", "epoch", 8, spin_cycles=12345)

    assert requests == [("routes", 19, 12345)]
    assert launches == [("ctx", 4096, "epoch", 8, ("scheduler", "routes"))]
    # The launch runs on the copy stream and restores the caller's stream afterwards.
    owner = broadcast._stream_owner
    assert stream_switches == [("enter", owner), ("exit", owner)]
    assert (ticket.generation, ticket.command_count) == (1, 1)
    assert (broadcast._active_generation, broadcast._generation) == (1, 2)
    # The bank stays leased until the consumer of generation 1 is proven done.
    with pytest.raises(RuntimeError, match="still leased"):
        broadcast.submit_fused(outputs, "routes", "epoch", 16, spin_cycles=12345)
    with pytest.raises(RuntimeError, match="bound CUDA scheduler"):
        broadcast.submit_fused(object(), "routes", "epoch", 16, spin_cycles=12345)


@pytest.mark.parametrize(
    (
        "hidden",
        "intermediate",
        "expected_total_bytes",
        "expected_stride_bytes",
        "expected_planes",
    ),
    [
        (
            7168,
            3072,
            37158924,
            37159424,
            (
                (
                    "mega_fc1_weight",
                    0,
                    22020096,
                    512,
                    ((3, 6144, 3584), (22020096, 3584, 1), "uint8", 1, 16),
                    (
                        (3, 3584, 6144),
                        (22020096, 1, 3584),
                        "float4_e2m1fn_x2",
                        1,
                        16,
                    ),
                ),
                (
                    "mega_fc1_weight_sf",
                    22020096,
                    2752512,
                    512,
                    ((3, 2752512), (2752512, 1), "uint8", 1, 16),
                    ((3, 2752512), (2752512, 1), "float8_e4m3fn", 1, 16),
                ),
                (
                    "mega_fc2_weight",
                    24772608,
                    11010048,
                    512,
                    ((3, 7168, 1536), (11010048, 1536, 1), "uint8", 1, 16),
                    (
                        (3, 1536, 7168),
                        (11010048, 1, 1536),
                        "float4_e2m1fn_x2",
                        1,
                        16,
                    ),
                ),
                (
                    "mega_fc2_weight_sf",
                    35782656,
                    1376256,
                    512,
                    ((3, 1376256), (1376256, 1), "uint8", 1, 16),
                    ((3, 1376256), (1376256, 1), "float8_e4m3fn", 1, 16),
                ),
                (
                    "fc31_alpha",
                    37158912,
                    4,
                    16,
                    ((3,), (1,), "float32", 4, 4),
                    ((3,), (1,), "float32", 4, 4),
                ),
                (
                    "fc2_alpha",
                    37158928,
                    4,
                    16,
                    ((3,), (1,), "float32", 4, 4),
                    ((3,), (1,), "float32", 4, 4),
                ),
                (
                    "fc1_norm_const",
                    37158944,
                    4,
                    16,
                    ((3,), (1,), "float32", 4, 4),
                    ((3,), (1,), "float32", 4, 4),
                ),
            ),
        ),
        (
            160,
            80,
            24332,
            25088,
            (
                (
                    "mega_fc1_weight",
                    0,
                    12800,
                    512,
                    ((3, 160, 80), (12800, 80, 1), "uint8", 1, 16),
                    (
                        (3, 80, 160),
                        (12800, 1, 80),
                        "float4_e2m1fn_x2",
                        1,
                        16,
                    ),
                ),
                (
                    "mega_fc1_weight_sf",
                    12800,
                    3072,
                    512,
                    ((3, 3072), (3072, 1), "uint8", 1, 16),
                    ((3, 3072), (3072, 1), "float8_e4m3fn", 1, 16),
                ),
                (
                    "mega_fc2_weight",
                    15872,
                    6400,
                    512,
                    ((3, 160, 40), (6400, 40, 1), "uint8", 1, 16),
                    (
                        (3, 40, 160),
                        (6400, 1, 40),
                        "float4_e2m1fn_x2",
                        1,
                        16,
                    ),
                ),
                (
                    "mega_fc2_weight_sf",
                    22528,
                    2048,
                    512,
                    ((3, 2048), (2048, 1), "uint8", 1, 16),
                    ((3, 2048), (2048, 1), "float8_e4m3fn", 1, 16),
                ),
                (
                    "fc31_alpha",
                    24576,
                    4,
                    16,
                    ((3,), (1,), "float32", 4, 4),
                    ((3,), (1,), "float32", 4, 4),
                ),
                (
                    "fc2_alpha",
                    24592,
                    4,
                    16,
                    ((3,), (1,), "float32", 4, 4),
                    ((3,), (1,), "float32", 4, 4),
                ),
                (
                    "fc1_norm_const",
                    24608,
                    4,
                    16,
                    ((3,), (1,), "float32", 4, 4),
                    ((3,), (1,), "float32", 4, 4),
                ),
            ),
        ),
    ],
    ids=["production", "padding-sensitive"],
)
def test_vendored_live_plane_abi_matches_independent_goldens(
    hidden,
    intermediate,
    expected_total_bytes,
    expected_stride_bytes,
    expected_planes,
):
    path = _TORCH_ROOT / "moe/kernels/megamoe_scheduler/sami/geometry.py"
    module_name = f"_live_plane_geometry_{hidden}_{intermediate}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    bundle = module.BundleLayout.create(hidden=hidden, intermediate=intermediate)
    observed = tuple(
        (
            plane.name,
            plane.offset,
            plane.nbytes,
            plane.alignment,
            plane.storage_metadata(3),
            plane.kernel_metadata(3),
        )
        for plane in bundle.planes
    )
    assert observed == expected_planes
    assert bundle.total_bytes == expected_total_bytes
    assert bundle.stride_bytes == expected_stride_bytes


def test_ep_collective_adapter_uses_the_resolved_process_group():
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/rebalance_live_arena.py"
    tree = ast.parse(source.read_text())
    selected = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_EpComm"
    )

    calls = []

    class FakeProcessGroup:
        def __init__(self, rank, world, global_ranks):
            self.rank = rank
            self.world = world
            self.global_ranks = global_ranks

    class FakeDistributed:
        @staticmethod
        def get_rank(*, group):
            calls.append(("rank", group))
            return group.rank

        @staticmethod
        def get_world_size(*, group):
            calls.append(("world", group))
            return group.world

        @staticmethod
        def get_global_rank(group, group_rank):
            calls.append(("global_rank", group, group_rank))
            return group.global_ranks[group_rank]

        @staticmethod
        def barrier(*, group):
            calls.append(("barrier", group))

        @staticmethod
        def broadcast_object_list(values, *, src, group):
            calls.append(("bcast", group, src))
            values[0] = ("from", src)

        @staticmethod
        def all_gather_object(values, value, *, group):
            calls.append(("allgather", group, value))
            for rank in range(len(values)):
                values[rank] = (rank, value)

    namespace = {
        "Any": Any,
        "_disable_current_modes": nullcontext,
        "torch": SimpleNamespace(distributed=FakeDistributed),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    process_group = FakeProcessGroup(rank=1, world=3, global_ranks=(11, 13, 17))
    comm = namespace["_EpComm"](process_group)
    comm.validate_geometry(expected_rank=1, expected_world=3)
    assert comm.allgather("handle") == [(0, "handle"), (1, "handle"), (2, "handle")]
    assert comm.bcast("payload", 2) == ("from", 17)
    comm.barrier()
    assert all(call[1] is process_group for call in calls)
    with pytest.raises(ValueError, match="rank/size differs"):
        comm.validate_geometry(expected_rank=0, expected_world=3)
    with pytest.raises(ValueError, match="broadcast root"):
        comm.bcast("payload", 3)


def test_model_engine_warmup_uses_load_balancer_public_api():
    events = []
    engine_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/model_engine.py",
        "PyTorchModelEngine",
        {"is_warmup", "moe_load_balancer_iter_info"},
        {"set_moe_comm_warmup": lambda value: events.append(("comm", value))},
    )
    manager = SimpleNamespace(
        enable_statistic=True,
        enable_update_weights=True,
        set_iter_info=lambda **kwargs: events.append(("iter", kwargs)),
    )
    engine = engine_cls()
    engine.moe_load_balancer = manager

    engine.is_warmup = True

    assert events == [
        ("comm", True),
        ("iter", {"enable_statistic": False, "enable_update_weights": False}),
    ]


def test_model_engine_cleanup_shuts_manager_after_graphs_while_model_is_alive():
    events = []
    engine_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/model_engine.py",
        "PyTorchModelEngine",
        {
            "prepare_cleanup",
            "shutdown_moe_load_balancer",
            "finish_cleanup",
            "cleanup",
        },
        {"release_gc": lambda: events.append("release_gc")},
    )
    engine = engine_cls()
    model = object()

    class Manager:
        def shutdown(self, *, local_safe):
            assert engine.model is model
            events.append(("manager.shutdown", local_safe))
            return local_safe

    engine._cleanup_done = False
    engine.model_loader = None
    engine._release_cuda_graphs = lambda: events.append("release_graphs")
    engine.moe_load_balancer = Manager()
    engine._runner = object()
    engine._model_caller = object()
    engine._mm_item_scheduler = object()
    engine.model = model
    engine.input_processor = object()

    engine.cleanup()
    engine.cleanup()

    assert events == [
        "release_graphs",
        ("manager.shutdown", True),
        "release_gc",
    ]
    assert engine.moe_load_balancer is None
    assert engine.model is None


def test_model_engine_prepare_failure_participates_in_unsafe_preflight():
    events = []
    engine_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/model_engine.py",
        "PyTorchModelEngine",
        {
            "prepare_cleanup",
            "shutdown_moe_load_balancer",
            "finish_cleanup",
            "cleanup",
        },
        {"release_gc": lambda: events.append("release_gc")},
    )

    class Manager:
        def shutdown(self, *, local_safe):
            events.append(("manager.shutdown", local_safe))
            return local_safe

    engine = engine_cls()
    engine._cleanup_done = False
    engine.model_loader = None
    engine._release_cuda_graphs = lambda: (_ for _ in ()).throw(
        RuntimeError("graph cleanup failed")
    )
    engine.moe_load_balancer = Manager()
    engine._runner = object()
    engine._model_caller = object()
    engine._mm_item_scheduler = object()
    engine.model = object()
    engine.input_processor = object()

    with pytest.raises(RuntimeError, match="graph cleanup failed"):
        engine.cleanup()

    assert events == [("manager.shutdown", False)]
    assert engine.moe_load_balancer is not None
    assert engine.model is not None


def test_model_engine_finalizer_never_enters_collective_cleanup():
    calls = []
    engine_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/model_engine.py",
        "PyTorchModelEngine",
        {"__del__"},
        {"logger": SimpleNamespace(warning=lambda *args: calls.append(("warning", args)))},
    )
    engine = engine_cls()
    engine.cleanup = lambda *, collective: calls.append(("cleanup", collective))

    engine.__del__()

    assert calls == [("cleanup", False)]


def test_executor_terminal_cleanup_is_three_phase_and_stream_scoped(monkeypatch):
    events = []
    module_name = "tensorrt_llm._torch.moe.kernels.megamoe_shared_fc12"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(
            release_shared_fc12_cache=lambda device, stream_handle=None: events.append(
                ("cache", device, stream_handle)
            )
        ),
    )
    executor_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/py_executor.py",
        "PyExecutor",
        {"terminal_cleanup"},
        {
            "__name__": "tensorrt_llm._torch.pyexecutor.py_executor",
            "__package__": "tensorrt_llm._torch.pyexecutor",
        },
    )

    class Engine:
        def __init__(self, name):
            self.name = name

        def prepare_cleanup(self):
            events.append(("prepare", self.name))

        def shutdown_moe_load_balancer(self, *, local_safe):
            events.append(("collective", self.name, local_safe))
            return local_safe

        def finish_cleanup(self):
            events.append(("finish", self.name))

    executor = executor_cls()
    executor.execution_stream = SimpleNamespace(device=SimpleNamespace(index=3), cuda_stream=99)
    executor._terminal_model_engines = (Engine("first"), Engine("second"))

    executor.terminal_cleanup()
    executor.terminal_cleanup()

    assert events == [
        ("prepare", "first"),
        ("prepare", "second"),
        ("collective", "first", True),
        ("collective", "second", True),
        ("finish", "first"),
        ("finish", "second"),
        ("cache", 3, 99),
    ]
    assert executor._terminal_model_engines == ()


def test_executor_terminal_cleanup_propagates_prepare_failure_to_all_managers(monkeypatch):
    events = []
    executor_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/py_executor.py",
        "PyExecutor",
        {"terminal_cleanup"},
        {
            "__name__": "tensorrt_llm._torch.pyexecutor.py_executor",
            "__package__": "tensorrt_llm._torch.pyexecutor",
        },
    )

    class Engine:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        def prepare_cleanup(self):
            events.append(("prepare", self.name))
            if self.fail:
                raise RuntimeError(f"{self.name} graph failure")

        def shutdown_moe_load_balancer(self, *, local_safe):
            events.append(("collective", self.name, local_safe))
            return local_safe

        def finish_cleanup(self):
            events.append(("finish", self.name))

    executor = executor_cls()
    executor.execution_stream = SimpleNamespace(device=SimpleNamespace(index=3), cuda_stream=99)
    executor._terminal_model_engines = (Engine("first", fail=True), Engine("second"))

    with pytest.raises(RuntimeError, match="first graph failure"):
        executor.terminal_cleanup()

    assert events == [
        ("prepare", "first"),
        ("prepare", "second"),
        ("collective", "first", False),
        ("collective", "second", False),
    ]
    assert len(executor._terminal_model_engines) == 2


def test_pyexecutor_shutdown_tail_failure_is_retryable():
    events = []

    class AsyncWorkerMixin:
        pass

    class RetryingPool(dict):
        def __init__(self):
            super().__init__(pool=object())
            self.delete_attempts = 0

        def __delitem__(self, key):
            self.delete_attempts += 1
            events.append(("pool_delete", self.delete_attempts))
            if self.delete_attempts == 1:
                raise RuntimeError("pool delete failed")
            super().__delitem__(key)

    class Sampler(AsyncWorkerMixin):
        def __init__(self):
            self.stop_attempts = 0

        def async_worker_enabled(self):
            return True

        def async_worker_stop(self):
            self.stop_attempts += 1
            events.append(("sampler_stop", self.stop_attempts))
            if self.stop_attempts == 1:
                raise RuntimeError("sampler stop failed")

    class Engine:
        def _release_cuda_graphs(self):
            events.append("release_graphs")

    class DwdpManager:
        def __init__(self):
            self.exit_attempts = 0

        def __exit__(self, *args):
            self.exit_attempts += 1
            events.append(("dwdp_exit", self.exit_attempts))
            if self.exit_attempts == 1:
                raise RuntimeError("dwdp exit failed")

    executor_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/py_executor.py",
        "PyExecutor",
        {"shutdown", "_finish_shutdown"},
        {
            "AsyncWorkerMixin": AsyncWorkerMixin,
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        },
    )
    worker_cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/base_worker.py",
        "BaseWorker",
        {"shutdown"},
        {},
    )

    engine = Engine()
    executor = executor_cls()
    executor.model_engine = engine
    executor.draft_model_engine = None
    executor.shutdown_all_ranks = True
    executor.can_shutdown = lambda: True
    executor.executor_request_queue = SimpleNamespace(
        enqueue_shutdown_request=lambda: events.append("enqueue")
    )
    executor.shutdown_event = SimpleNamespace(wait=lambda: events.append("wait"))
    executor._profile_manager = SimpleNamespace(cleanup=lambda: events.append("profile_cleanup"))
    executor.hang_detector = SimpleNamespace(detected=lambda: False)
    executor.worker_thread = SimpleNamespace(join=lambda: events.append("worker_join"))
    executor.kv_connector_manager = None
    executor.dist = SimpleNamespace(pp_size=1)
    executor._shutdown_sleep_wakeup_listeners = lambda: events.append("listener_shutdown")
    executor.encoder_launch_executor = None
    executor.worker_started = True
    executor.resource_manager = SimpleNamespace(
        resource_managers={
            "manager": SimpleNamespace(shutdown=lambda: events.append("resource_shutdown"))
        }
    )
    executor.virtual_memory_pools = RetryingPool()
    executor.sampler = Sampler()
    executor.dwdp_manager = DwdpManager()

    def terminal_cleanup(*, local_safe):
        events.append(("terminal_cleanup", local_safe))
        if local_safe:
            assert executor.model_engine is None
            assert executor._terminal_model_engines == (engine,)
            executor._terminal_model_engines = ()

    executor.terminal_cleanup = terminal_cleanup
    worker = worker_cls()
    worker.doing_shutdown = False
    worker.engine = executor

    for message in ("pool delete failed", "sampler stop failed", "dwdp exit failed"):
        with pytest.raises(RuntimeError, match=message):
            worker.shutdown()
        assert executor.model_engine is engine
        assert executor._terminal_model_engines == (engine,)
        assert executor._shutdown_runtime_complete
        assert worker.engine is executor
        assert not worker.doing_shutdown

    worker.shutdown()

    assert events == [
        "enqueue",
        "wait",
        "profile_cleanup",
        "worker_join",
        "listener_shutdown",
        "release_graphs",
        "resource_shutdown",
        ("pool_delete", 1),
        ("terminal_cleanup", False),
        ("pool_delete", 2),
        ("sampler_stop", 1),
        ("terminal_cleanup", False),
        ("sampler_stop", 2),
        ("dwdp_exit", 1),
        ("terminal_cleanup", False),
        ("dwdp_exit", 2),
        ("terminal_cleanup", True),
    ]
    assert executor.virtual_memory_pools == {}
    assert executor.model_engine is None
    assert executor.draft_model_engine is None
    assert executor.dwdp_manager is None
    assert executor._shutdown_sampler_complete
    assert executor._terminal_model_engines == ()
    assert worker.engine is None


class _Routes:
    def __init__(
        self,
        rows,
        trace,
        *,
        topk=6,
        dtype="int32",
        device="cuda:0",
        contiguous=True,
    ):
        self.shape = (rows, topk)
        self.ndim = 2
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        self.trace = trace

    def is_contiguous(self):
        return self._contiguous

    def record_stream(self, stream):
        self.trace.append(("lifetime", stream.cuda_stream))

    def __getitem__(self, rows):
        return _Routes(rows.stop, self.trace)


class _StageTensor:
    def __init__(self, name, trace, shape=(8, 6)):
        self.name = name
        self.trace = trace
        self.shape = shape

    def __getitem__(self, index):
        return _StageTensor(self.name, self.trace, self.shape)

    def view(self, dtype):
        return self

    def copy_(self, other, **kwargs):
        self.trace.append("copy:" + self.name)

    def zero_(self):
        self.trace.append("zero:" + self.name)

    def fill_(self, value):
        self.trace.append("fill:" + self.name)


def test_launch_queue_is_enabled_only_for_per_layer_dynamic_mode(monkeypatch):
    path = _ROOT / "tensorrt_llm/llmapi/_load_balance_env.py"
    spec = importlib.util.spec_from_file_location("load_balance_env_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    initialized = False
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: initialized),
    )
    process_queue = os.environ.get(_QUEUE)
    test_environ = os.environ.copy()
    test_environ.pop(_QUEUE, None)
    monkeypatch.setattr(module, "os", SimpleNamespace(environ=test_environ))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def config(mode="per_layer_dynamic"):
        return SimpleNamespace(
            load_balancer=SimpleNamespace(mode=mode),
            resolve_load_balancer_compatibility=lambda: None,
        )

    original = {"KEEP": "value"}
    configured = module.configure_moe_launch_queues(config(), original)
    assert configured == {"KEEP": "value", _QUEUE: "4x"}
    assert configured is not original
    assert original == {"KEEP": "value"}
    assert test_environ[_QUEUE] == "4x"
    assert module.configure_moe_launch_queues(config(), configured) == configured

    for off in (None, config(mode="standard")):
        test_environ.pop(_QUEUE, None)
        before = {"KEEP": "value"}
        assert module.configure_moe_launch_queues(off, before) is before
        assert _QUEUE not in test_environ

    initialized = True
    before = {"KEEP": "value"}
    assert module.configure_moe_launch_queues(config(mode="standard"), before) is before
    assert _QUEUE not in test_environ

    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        module.configure_moe_launch_queues(config(), original)
    assert original == {"KEEP": "value"}
    assert _QUEUE not in test_environ
    assert os.environ.get(_QUEUE) == process_queue


class _RuntimeInitReached(Exception):
    """Stops an entrypoint at its first runtime-initialization call."""


class _RecordingEnviron(dict):
    def __init__(self, events, **values):
        super().__init__(values)
        self._events = events

    def __setitem__(self, key, value):
        self._events.append(("env", key, value))
        super().__setitem__(key, value)

    def update(self, values):
        for key, value in values.items():
            self[key] = value


def test_launch_queue_setup_sees_env_overrides_before_runtime_initialization(monkeypatch):
    events = []
    environ = _RecordingEnviron(events, MASTER_ADDR="127.0.0.1")
    fake_os = SimpleNamespace(
        environ=environ,
        getenv=lambda key, default=None: environ.get(key, default),
    )
    logger = SimpleNamespace(
        level="warning",
        set_level=lambda level: None,
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    user_overrides = {"USER_SETTING": "1"}
    configured = {**user_overrides, _QUEUE: "4x"}

    def configure_moe_launch_queues(moe_config, env_overrides):
        # The override must already be visible when the launch queue is configured.
        events.append(("configure", environ.get("USER_SETTING")))
        return {**(env_overrides or {}), _QUEUE: "4x"}

    def runtime_init(name):
        def call(*args, **kwargs):
            events.append(name)
            raise _RuntimeInitReached(name)

        return call

    def run(entrypoint, stop):
        events.clear()
        environ.pop("USER_SETTING", None)
        with pytest.raises(_RuntimeInitReached, match=stop):
            entrypoint()
        return list(events)

    def worker_args():
        return SimpleNamespace(
            backend="pytorch", env_overrides=dict(user_overrides), moe_config=object()
        )

    expected = [("env", "USER_SETTING", "1"), ("configure", "1")]
    common = {
        "os": fake_os,
        "logger": logger,
        "configure_moe_launch_queues": configure_moe_launch_queues,
    }

    # LLM API: argument validation is the first step that may probe CUDA.
    class TorchLlmArgs:
        model_fields = {"env_overrides": None, "moe_config": None}

        def __init__(self, **kwargs):
            events.append(("llm_args", kwargs["env_overrides"]))
            raise _RuntimeInitReached("llm_args")

    llm_cls = _extract_class(
        _ROOT / "tensorrt_llm/llmapi/llm.py",
        "BaseLLM",
        {"__init__", "_process_env_overrides"},
        {
            **common,
            "GenerationExecutor": object,
            "MoeConfig": lambda **kwargs: SimpleNamespace(**kwargs),
            "TorchLlmArgs": TorchLlmArgs,
            "TORCH_LLMARGS_REMOVED_ARGS": set(),
            "mpi_disabled": lambda: False,
        },
    )
    assert run(
        lambda: llm_cls(
            "model", backend="pytorch", env_overrides=dict(user_overrides), moe_config={}
        ),
        "llm_args",
    ) == expected + [("llm_args", configured)]

    # MPI worker process: MPI is the first runtime initialization.
    worker_main = _extract_function(
        _ROOT / "tensorrt_llm/executor/worker.py",
        "worker_main",
        {**common, "GenerationExecutorWorker": object, "mpi_comm": runtime_init("mpi_comm")},
    )
    args = worker_args()
    assert run(
        lambda: worker_main(engine="engine", worker_queues=None, log_level="info", llm_args=args),
        "mpi_comm",
    ) == expected + ["mpi_comm"]
    assert args.env_overrides == configured

    # Direct worker construction: the executor base initializes the runtime.
    class GenerationExecutor:
        __init__ = runtime_init("executor_init")

    worker_cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/base_worker.py",
        "BaseWorker",
        {"__init__"},
        {
            **common,
            "GenerationExecutor": GenerationExecutor,
            "PostprocWorkerConfig": lambda: SimpleNamespace(
                num_postprocess_workers=0, postprocess_tokenizer_dir=None
            ),
        },
        bases=("GenerationExecutor",),
    )
    args = worker_args()
    assert run(lambda: worker_cls(engine="engine", llm_args=args), "executor_init") == expected + [
        "executor_init"
    ]
    assert args.env_overrides == configured

    # Ray worker: the first CUDA query follows launch-queue setup.
    monkeypatch.setitem(sys.modules, "tensorrt_llm.logger", SimpleNamespace(logger=logger))
    ray_cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/ray/gpu_worker.py",
        "RayWorkerWrapper",
        {"__init__"},
        {
            **common,
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(device_count=runtime_init("torch.cuda.device_count"))
            ),
        },
    )
    args = worker_args()
    assert run(lambda: ray_cls(object, {"llm_args": args}, 1, 0), "device_count") == expected + [
        "torch.cuda.device_count"
    ]
    assert args.env_overrides == configured


@pytest.mark.parametrize("auxiliary_sms", [4, 8])
def test_fused_submit_orders_generations_and_defers_the_route_wait(auxiliary_sms):
    trace = []
    state = SimpleNamespace(device=0, thread=7)
    main = SimpleNamespace(cuda_stream=44, priority=0)
    copy = SimpleNamespace(cuda_stream=19, priority=-1)
    cuda = SimpleNamespace(
        current_device=lambda: state.device,
        current_stream=lambda _: main,
        nvtx=SimpleNamespace(range=lambda _: nullcontext()),
    )
    raw = SimpleNamespace(
        _cuda_getDevice=lambda: state.device,
        _cuda_getCurrentRawStream=lambda _: main.cuda_stream,
    )
    torch = SimpleNamespace(cuda=cuda, _C=raw, Tensor=_Routes, int32="int32")
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/rebalance_slot_scheduler.py"
    namespace = {"torch": torch, "threading": SimpleNamespace(get_ident=lambda: state.thread)}
    group_cls = _extract_class(source, "RebalanceSlotSchedulerGroup", None, namespace)
    lease_cls = _extract_class(source, "_LiveBankLeaseProvider", None, namespace)
    group = group_cls.__new__(group_cls)
    for name in (
        "_generation",
        "_finished_generation",
        "_scheduled_generation",
        "_route_wait_generation",
        "_route_target",
        "plan_calls",
    ):
        setattr(group, name, 0)
    group._plan_part = None
    group._pending_release = None
    group._closed = False
    group._owner_thread_id = None
    group._execution_stream_handle = None
    group.device = 0
    group.topk = 6
    group.max_tokens_per_rank = 8192
    group.auxiliary_sms = auxiliary_sms
    group.copy_stream = group.scheduler_stream = copy
    group._stream_handle = 19
    group._input_ready_handle = 1
    group._consumer_done_handle = 3
    group._consumer_done = SimpleNamespace(cuda_event=3)
    route_epoch = object()
    group._route_epoch = route_epoch
    group._route_epoch_ptr = 77
    # Generation 1 launches during warmup, generation 2 while serving.
    bounds = iter((_WARMUP_SPIN_CYCLES, _SERVING_SPIN_CYCLES))
    group._spin_bound = SimpleNamespace(cycles=lambda: next(bounds))

    def driver_call(*entry):
        trace.append(entry)
        return (0,)

    group._driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        CUstreamWaitValue_flags=SimpleNamespace(CU_STREAM_WAIT_VALUE_GEQ="geq"),
        cuEventRecord=lambda event, stream: driver_call("record", event, stream),
        cuStreamWaitEvent=lambda stream, event, flags: driver_call("wait", stream, event),
        cuStreamWaitValue64=lambda stream, address, value, flags: driver_call(
            "wait_value", stream, address, value, flags
        ),
    )
    outputs = SimpleNamespace(physical_slot_ids=_Routes(8192, trace))

    launch_bounds = []

    def submit_fused(result, routes, epoch, target, *, spin_cycles):
        assert result is outputs and epoch is route_epoch
        generation = group._scheduled_generation
        trace.append(("FUSED", routes.shape[0], target, generation))
        launch_bounds.append(spin_cycles)
        return SimpleNamespace(generation=generation)

    group.scheduler = SimpleNamespace(device="cuda:0", outputs=outputs)
    group.broadcaster = SimpleNamespace(
        submit_fused=submit_fused,
        release_generation_after=lambda generation, event: trace.append(
            ("consumer_release", generation, event.cuda_event)
        ),
        mark_collective_reuse_safe=lambda provider, generation: trace.append(("lease", generation)),
    )
    group.lease = lease_cls(group)

    for generation, rows in enumerate((16, 0), 1):
        trace.clear()
        routes = _Routes(rows, trace)
        target = generation * auxiliary_sms
        route_wait = ("wait_value", 44, 77, target, "geq")
        part = group.plan_schedule(routes)
        physical, actual_generation = group.plan_finish(part, defer_wait=True)
        assert physical.shape == (rows, 6)
        assert actual_generation == generation
        assert route_wait not in trace
        group.wait_for_routes()
        group.wait_for_routes()
        group.finish()
        assert trace.count(route_wait) == 1
        assert trace.index(("wait", 19, 1)) < trace.index(("FUSED", rows, target, generation))
        assert trace.index(("FUSED", rows, target, generation)) < trace.index(route_wait)
        assert trace[-2:] == [("record", 3, 44), ("consumer_release", generation, 3)]
        if generation > 1:
            # The previous consumer is proven reusable before the next launch.
            assert trace.index(("lease", generation - 1)) < trace.index(
                ("FUSED", rows, target, generation)
            )
    assert launch_bounds == [_WARMUP_SPIN_CYCLES, _SERVING_SPIN_CYCLES]

    # The bound-owner fast path still rejects another thread, stream or device.
    state.thread = 8
    with pytest.raises(RuntimeError, match="one MAIN thread and stream"):
        group.plan_schedule(_Routes(8, trace))
    state.thread = 7
    main.cuda_stream = 45
    with pytest.raises(RuntimeError, match="one MAIN thread and stream"):
        group.plan_schedule(_Routes(8, trace))
    main.cuda_stream = 44
    state.device = 1
    with pytest.raises(RuntimeError, match="bound CUDA device"):
        group.plan_schedule(_Routes(8, trace))
    state.device = 0

    trace.clear()
    with pytest.raises(ValueError, match="contiguous CUDA int32"):
        group.plan_schedule(_Routes(8, trace, dtype="int64"))
    assert trace == []

    group._driver.cuStreamWaitEvent = lambda *args: (17,)
    with pytest.raises(RuntimeError, match="cuStreamWaitEvent"):
        group.plan_schedule(_Routes(8, trace))
    assert not any(entry[0] == "FUSED" for entry in trace)
    assert group._plan_part is None


def _module_constants(path: Path, names: set[str]) -> dict:
    tree = ast.parse(path.read_text())
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    }
    assert set(constants) == names
    return constants


def _scheduler_spin_namespace() -> dict:
    namespace = _module_constants(_SCHEDULER_RUNTIME, {"SPIN_CYCLES", "ASSUMED_CLOCK_HZ"})
    _extract_function(_SCHEDULER_RUNTIME, "spin_cycles_for_timeout", namespace)
    _extract_function(_SCHEDULER_RUNTIME, "_check_spin_cycles", namespace)
    return namespace


def _moe_comm_timeout_guard_module(monkeypatch) -> ModuleType:
    """Execute the timeout guard module without its torch and logger imports."""
    path = _TORCH_ROOT / "moe/fused_moe/moe_comm_timeout_guard.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            (isinstance(node, ast.Import) and node.names[0].name == "torch")
            or (isinstance(node, ast.ImportFrom) and node.module == "tensorrt_llm.logger")
        )
    ]
    module = ModuleType("moe_comm_timeout_guard_under_test")
    module.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    module.torch = None
    # Dataclasses resolve their module through sys.modules.
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def test_scheduler_spin_cycles_convert_timeouts_at_the_nominal_clock():
    namespace = _scheduler_spin_namespace()
    to_cycles = namespace["spin_cycles_for_timeout"]

    assert namespace["SPIN_CYCLES"] == _SERVING_SPIN_CYCLES
    assert to_cycles(None) == _SERVING_SPIN_CYCLES
    assert to_cycles(1800) == _WARMUP_SPIN_CYCLES
    for seconds in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="positive exact int"):
            to_cycles(seconds)
    for cycles in (0, -1, 1.0, True):
        with pytest.raises(ValueError, match="positive exact int"):
            namespace["_check_spin_cycles"](cycles)


@pytest.mark.parametrize(
    ("environ", "warmup_cycles", "serving_cycles"),
    [
        pytest.param({}, _WARMUP_SPIN_CYCLES, _SERVING_SPIN_CYCLES, id="defaults"),
        pytest.param(
            {
                "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC": "600",
                "TRTLLM_MOE_COMM_TIMEOUT_SEC": "30",
            },
            600 * 2_000_000_000,
            30 * 2_000_000_000,
            id="configured",
        ),
    ],
)
def test_scheduler_spin_bound_follows_warmup_transitions_and_autotuning(
    monkeypatch, environ, warmup_cycles, serving_cycles
):
    guard_module = _moe_comm_timeout_guard_module(monkeypatch)
    guard = guard_module.MoECommTimeoutGuard(environ=environ, is_capturing=lambda: False)
    runtime = SimpleNamespace(**_scheduler_spin_namespace())
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.moe.kernels.megamoe_scheduler.cuda_scheduler.runtime",
        runtime,
    )
    bound_cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/mega_moe/rebalance_slot_scheduler.py",
        "SchedulerSpinBound",
        None,
        {"Callable": Any, "Optional": Optional},
    )
    tuning = [False]
    bound = bound_cls(warmup_seconds=guard.budgets().warmup_seconds, is_tuning=lambda: tuning[0])
    guard.register(bound)

    # The engine starts in serving; its warmup transitions switch the bound.
    assert bound.cycles() == serving_cycles
    guard.set_warmup(True)
    assert bound.cycles() == warmup_cycles
    guard.set_warmup(False)
    assert bound.cycles() == serving_cycles
    # AutoTuner profiling outside warmup gets the warmup bound too.
    tuning[0] = True
    assert bound.cycles() == warmup_cycles
    # The startup warmup launch always uses the warmup bound.
    assert bound.warmup_cycles == warmup_cycles


def test_scheduler_launches_pass_their_spin_bound_to_the_kernel():
    # Locate spin_cycles in the registered operator schema rather than hard-coding it.
    op_source = (
        _ROOT / "cpp/tensorrt_llm/thop/moe/loadBalance/moeRebalanceHaloQOp.cpp"
    ).read_text()
    schema = "".join(re.findall(r'"([^"]*)"', op_source.split("m.def(", 1)[1].split(");", 1)[0]))
    parameters = re.fullmatch(r"moe_rebalance_halo_q\((.*)\) -> \(\)", schema).group(1)
    names = [parameter.split()[-1] for parameter in parameters.split(",")]
    spin_index = names.index("spin_cycles")

    launched = []
    torch = SimpleNamespace(
        Tensor=_Routes,
        int32="int32",
        ops=SimpleNamespace(
            trtllm=SimpleNamespace(moe_rebalance_halo_q=lambda *args: launched.append(args))
        ),
    )
    namespace = _scheduler_spin_namespace()
    namespace.update(
        {
            "torch": torch,
            "enter_cuda_stream": lambda stream: stream,
            "exit_cuda_stream": lambda state: None,
            "_stream_handle": lambda stream: int(getattr(stream, "cuda_stream", stream)),
        }
    )
    scheduler_cls = _extract_class(
        _SCHEDULER_RUNTIME,
        "CudaPhysicalSlotScheduler",
        {"launch", "_launch_op", "_halo_q_args", "fused_submit_args", "_check_routes"},
        namespace,
    )
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.cfg = SimpleNamespace(
        ep_size=8,
        logical_expert_count=384,
        extra_slots_per_rank=4,
        local_rank=0,
        route_count=8192 * 6,
        ctas=8,
        threads=512,
        enable_pdl=True,
        topk=6,
        max_tokens_per_rank=8192,
    )
    scheduler.outputs = SimpleNamespace(
        physical_slot_ids="slots",
        hot_expert_ids="ids",
        hot_expert_group_level="levels",
        hot_expert_source_ranks="owners",
    )
    for name in (
        "peer_base",
        "status",
        "partial",
        "route_aux",
        "grid_sync",
        "plan_workspace",
        "route_prefix",
    ):
        setattr(scheduler, name, name)
    scheduler.logical_expert_ids = "staged routes"
    scheduler.device = "cuda:0"
    scheduler._connected = True
    scheduler._plan_channel_lock = nullcontext()
    scheduler._plan_channel_bound = False
    scheduler._plan_channel_pending = False
    scheduler._plan_channel_ptr = 0
    scheduler._plan_channel_abi_version = 0
    scheduler._plan_channel_words = 0
    scheduler._plan_channel_route_features = 0
    scheduler._plan_channel_torch_stream = SimpleNamespace(cuda_stream=19)
    scheduler._launch_stream_handle = None

    # The standalone kernel keeps the native bound unless the caller passes one.
    scheduler.launch(19)
    scheduler.launch(19, spin_cycles=_WARMUP_SPIN_CYCLES)
    assert [args[spin_index] for args in launched] == [_SERVING_SPIN_CYCLES, _WARMUP_SPIN_CYCLES]

    # The fused launch takes the same scheduler arguments with the caller's bound.
    scheduler._gpu_direct_bound = True
    scheduler._plan_channel_stream_handle = 19
    routes = _Routes(16, [])
    routes.is_cuda = True
    args = scheduler.fused_submit_args(routes, 19, spin_cycles=_WARMUP_SPIN_CYCLES)
    assert args[spin_index] == _WARMUP_SPIN_CYCLES
    assert args[spin_index - 1] is False  # No programmatic dependency in the fused kernel.
    with pytest.raises(ValueError, match="positive exact int"):
        scheduler.fused_submit_args(routes, 19, spin_cycles=0)


def test_raw_stream_switch_restores_the_previous_stream_and_device():
    calls = []
    current = {"device": 0, "streams": {0: (10, 0, 1), 3: (30, 3, 1)}}

    def exchange_device(index):
        calls.append(("exchange", index))
        previous, current["device"] = current["device"], index
        return previous

    def set_stream(stream_id, device_index, device_type):
        calls.append(("set", stream_id, device_index))
        current["streams"][device_index] = (stream_id, device_index, device_type)

    def maybe_exchange_device(index):
        calls.append(("restore_device", index))
        current["device"] = index
        return index

    raw = SimpleNamespace(
        _cuda_exchangeDevice=exchange_device,
        _cuda_getCurrentStream=lambda index: current["streams"][index],
        _cuda_setStream=set_stream,
        _cuda_maybeExchangeDevice=maybe_exchange_device,
    )
    source = _TORCH_ROOT / "moe/kernels/megamoe_scheduler/cuda_scheduler/runtime.py"
    namespace = {"torch": SimpleNamespace(_C=raw, cuda=SimpleNamespace(Stream=object))}
    enter = _extract_function(source, "enter_cuda_stream", namespace)
    leave = _extract_function(source, "exit_cuda_stream", namespace)

    state = enter(SimpleNamespace(device_index=3, stream_id=77, device_type=1))
    assert current["device"] == 3 and current["streams"][3] == (77, 3, 1)
    leave(state)
    assert current == {"device": 0, "streams": {0: (10, 0, 1), 3: (30, 3, 1)}}
    assert calls == [
        ("exchange", 3),
        ("set", 77, 3),
        ("set", 30, 3),
        ("restore_device", 0),
    ]


def test_megamoe_cute_weight_cache_keys_every_view_field():
    conversions = []

    def to_cute(t, assumed_align):
        conversions.append((t, assumed_align))
        return ("cute", len(conversions))

    to_cute_cached = _extract_function(
        _TORCH_ROOT / "moe/custom_ops/cute_dsl_megamoe_custom_op.py",
        "_to_cute_cached",
        {"torch": SimpleNamespace(Tensor=object), "_cute_launch_helpers": lambda: (to_cute,)},
    )

    def weight(ptr=4096, shape=(8, 16), stride=(16, 1), dtype="fp4", device=0):
        return SimpleNamespace(
            data_ptr=lambda: ptr,
            shape=shape,
            stride=lambda: stride,
            dtype=dtype,
            device=SimpleNamespace(index=device),
        )

    cache = {}
    first = to_cute_cached(cache, weight())
    # A new view object of the same memory and layout reuses the cute view.
    assert to_cute_cached(cache, weight()) is first
    assert len(conversions) == 1
    for changed in (
        weight(ptr=8192),
        weight(shape=(16, 8)),
        weight(stride=(1, 8)),
        weight(dtype="fp8"),
        weight(device=1),
    ):
        assert to_cute_cached(cache, changed) is not first
    assert to_cute_cached(cache, weight(), assumed_align=4) is not first
    assert len(conversions) == 7
    # The cache belongs to the caller: another module's dict starts empty.
    assert to_cute_cached({}, weight()) is not first


def test_route_wait_occurs_after_independent_staging():
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/mega_moe_cute_dsl.py"
    impl = _extract_class(
        source,
        "TrtllmCutedslMegaMoeNvfp4Impl",
        {"_stage_inputs"},
        {"torch": SimpleNamespace(uint8="uint8")},
    )

    def run(stage_routes):
        trace = []
        owner = impl()
        owner._last_staged_T = {8: 8}
        owner._wait_rebalance_routes = lambda replica_plan=None: trace.append("wait")
        bufs = SimpleNamespace(
            **{
                key: _StageTensor(key, trace)
                for key in ("topk_idx_local", "activation", "activation_sf", "topk_weights")
            }
        )
        owner._stage_inputs(
            bufs=bufs,
            x=_StageTensor("x", trace),
            x_sf=_StageTensor("sf", trace),
            topk_idx=_StageTensor("routes", trace),
            topk_weights=_StageTensor("weights", trace),
            num_tokens=4,
            top_k=6,
            stage_activation=True,
            stage_routes=stage_routes,
        )
        return trace, owner

    trace, owner = run(True)
    assert trace.index("copy:activation") < trace.index("wait")
    assert trace.index("copy:topk_weights") < trace.index("wait")
    assert trace.index("wait") < trace.index("copy:topk_idx_local")
    assert owner._last_staged_T[8] == 4

    trace, owner = run(False)
    assert "wait" not in trace
    assert not any("topk_idx_local" in item for item in trace)
    assert owner._last_staged_T[8] == 8


def test_warmup_owner_handoff_uses_deduplicated_manager_public_api():
    events = []
    main = SimpleNamespace(cuda_stream=11, priority=0, device=0)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda device: events.append(("synchronize", device)),
        )
    )
    executor_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/py_executor.py",
        "PyExecutor",
        {"_handoff_rebalance_warmup_owners"},
        {"torch": torch},
    )

    class Manager:
        def __init__(self, name):
            self.name = name

        def handoff_submission_owner(self, stream_handle):
            events.append(("handoff", self.name, stream_handle))

    target = Manager("target")
    draft = Manager("draft")
    executor = executor_cls()
    executor.execution_stream = main
    executor.model_engine = SimpleNamespace(moe_load_balancer=target)
    executor.draft_model_engine = SimpleNamespace(moe_load_balancer=draft)

    executor._handoff_rebalance_warmup_owners()
    assert events == [
        ("synchronize", 0),
        ("handoff", "target", 11),
        ("handoff", "draft", 11),
    ]

    events.clear()
    executor.draft_model_engine = SimpleNamespace(moe_load_balancer=target)
    executor._handoff_rebalance_warmup_owners()
    assert events == [("synchronize", 0), ("handoff", "target", 11)]

    events.clear()
    executor.model_engine = SimpleNamespace(moe_load_balancer=None)
    executor.draft_model_engine = None
    executor._handoff_rebalance_warmup_owners()
    assert events == []


def test_quiesced_owner_release_uses_deduplicated_manager_public_api():
    events = []
    main = SimpleNamespace(cuda_stream=11, priority=0, device=0)
    executor_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/py_executor.py",
        "PyExecutor",
        {"_release_quiesced_rebalance_submission_owners"},
        {},
    )

    class Manager:
        def __init__(self, name):
            self.name = name

        def release_quiesced_submission_owner(self, stream_handle):
            events.append(("release", self.name, stream_handle))

    target = Manager("target")
    draft = Manager("draft")
    executor = executor_cls()
    executor.execution_stream = main
    executor.model_engine = SimpleNamespace(moe_load_balancer=target)
    executor.draft_model_engine = SimpleNamespace(moe_load_balancer=draft)

    executor._release_quiesced_rebalance_submission_owners()
    assert events == [
        ("release", "target", 11),
        ("release", "draft", 11),
    ]

    events.clear()
    executor.draft_model_engine = SimpleNamespace(moe_load_balancer=target)
    executor._release_quiesced_rebalance_submission_owners()
    assert events == [("release", "target", 11)]


def _autotune_helpers(torch):
    source = _TORCH_ROOT / "moe/custom_ops/cute_dsl_megamoe_custom_op.py"
    functions = {
        "_autotune_pl_alpha",
        "synthesize_profiling_topk",
        "_token_back_ready_granularity_choices",
        "_megamoe_tuning_op_name",
        "_unpack_tactic",
        "_is_pow2_in_range",
        "validate_megamoe_tactic",
        "_epi_flag_batch_for_tokens",
        "_expand_megamoe_ready_tactics",
        "enumerate_megamoe_candidate_tactics",
    }
    constants = {
        "_NVFP4_BLOCK_SIZE",
        "_SUPPORTED_MMA_TILE_M",
        "_SUPPORTED_MMA_TILE_N",
        "_AUTOTUNE_PL_SEED",
        "_DEFAULT_TOKEN_BACK_READY_GRANULARITY",
        "_TOKEN_BACK_READY_GRANULARITIES",
        "_TACTIC_LEN",
        "_TACTIC_LEN_V4",
        "_LEGACY_TACTIC_LEN",
        "_FLAG_BATCH_MAX",
        "_GEOMETRIES",
        "_TOKEN_BACK_MODES",
        "_TOKEN_BACK_STORE_BINDING",
        "_WORK_ID_MODE_CANDIDATES",
        "_GROUP_HINTS",
        "_FLAG_BATCHES",
        "_EPI_FLAG_BATCH_SMALL",
        "_EPI_FLAG_BATCH_LARGE",
        "_EPI_FLAG_BATCH_TOKEN_THRESHOLD",
    }
    tree = ast.parse(source.read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in constants for target in targets):
                nodes.append(node)
    namespace = {
        "torch": torch,
        "math": math,
        "functools": functools,
        "Tuple": Tuple,
        "List": List,
        "Optional": Optional,
        "Any": Any,
        "logger": SimpleNamespace(debug=lambda *args: None),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"
        ),
        namespace,
    )
    return namespace


def test_megamoe_runner_cache_reuses_only_production_runners():
    source = _TORCH_ROOT / "moe/custom_ops/cute_dsl_megamoe_custom_op.py"
    tree = ast.parse(source.read_text())
    factory_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_megamoe_get_runner"
    )

    created = []

    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._profiling_scratch = object()
            self._profiling_scratch_factory = object()
            created.append(self)

    namespace = {
        "Sm100MegaMoENvfp4Runner": FakeRunner,
        "_MEGAMOE_RUNNER_CACHE": {},
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            factory_node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    get_runner = namespace["_megamoe_get_runner"]
    kwargs = dict(
        world_size=8,
        local_rank=2,
        num_topk=6,
        num_experts_per_rank=32,
        hidden_size=7168,
        intermediate_size_per_partition=2048,
        expand_intermediate_size_per_partition=4096,
        max_tokens_per_rank=8192,
        output_dtype="bf16",
        apply_topk_in_fc1=True,
        swiglu_alpha=None,
        swiglu_beta=None,
        gate_up_clamp=None,
        situ_beta=None,
        situ_linear_beta=None,
        in_kernel_fc2_reduce=False,
        combine_format="bf16",
        helper_expert_count=4,
    )

    production = get_runner(**kwargs, tactic_autotune=False)
    production._profiling_scratch = "stale"
    production._profiling_scratch_factory = "stale"
    assert get_runner(**kwargs, tactic_autotune=False) is production
    assert production._profiling_scratch is None
    assert production._profiling_scratch_factory is None

    first_tuning = get_runner(**kwargs, tactic_autotune=True)
    second_tuning = get_runner(**kwargs, tactic_autotune=True)
    assert first_tuning is not second_tuning
    assert len(created) == 3


def test_on_autotune_uses_distinct_tactics():
    helpers = _autotune_helpers(None)

    off_key = helpers["_megamoe_tuning_op_name"](0)
    on_key = helpers["_megamoe_tuning_op_name"](4)
    assert off_key == "trtllm::cute_dsl_megamoe_nvfp4_blackwell"
    assert on_key != off_key and "alpha=0.8" in on_key

    off = helpers["enumerate_megamoe_candidate_tactics"](32, 107)
    on = helpers["enumerate_megamoe_candidate_tactics"](32, 107, load_balance=True)
    assert all(helpers["_unpack_tactic"](tactic)[10] == "expert" for tactic in off)
    assert {helpers["_unpack_tactic"](tactic)[10] for tactic in on} == {
        "expert",
        "token_tile",
    }
    assert any(tactic[2] is not None for tactic in on)


def test_on_autotune_builds_balanced_powerlaw_input(request):
    torch = pytest.importorskip("torch")
    original_num_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(original_num_threads))
    torch.set_num_threads(4)
    helpers = _autotune_helpers(torch)
    kwargs = dict(
        num_tokens=8192,
        num_topk=6,
        num_experts_per_rank=52,
        world_size=8,
        alpha=0.8,
        device=torch.device("cpu"),
        source_rank=0,
    )
    routes = helpers["synthesize_profiling_topk"](**kwargs)
    assert torch.equal(routes, helpers["synthesize_profiling_topk"](**kwargs))
    counts = torch.bincount(routes.reshape(-1), minlength=8 * 52).view(8, 52)
    assert bool((counts[:, 1:] >= counts[:, :-1]).all())
    per_rank = counts.sum(1)
    assert int(per_rank.max() - per_rank.min()) <= 1
    assert bool((counts[:, -1] > 0).all())

    uniform = helpers["synthesize_profiling_topk"](**{**kwargs, "alpha": 0.0})
    expected = (torch.arange(8192 * 6) % (8 * 52)).view(8192, 6)
    assert torch.equal(uniform, expected)


def test_per_layer_dynamic_shutdown_reports_in_iter_through_collective_preflight():
    manager_cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb.py",
        "PerLayerDynamicMoeLoadBalancer",
        {"shutdown"},
        {},
    )
    calls = []

    class Comm:
        def allgather(self, state):
            calls.append(("allgather", state))
            return [state, {"local_safe": True, "pool_count": 1}]

    class Pool:
        def close_collectively(self, *, local_safe):
            calls.append(("pool", local_safe))
            return True

    manager = manager_cls()
    manager.is_shutdown = False
    manager.in_iter = True
    manager._ep_comm = Comm()
    manager._rebalance_shared_slot_pool_order = [Pool()]

    assert manager.shutdown(local_safe=True) is False
    assert calls == [
        ("allgather", {"local_safe": False, "pool_count": 1}),
    ]


def test_base_worker_finalizes_rank_follower_pyexecutor():
    worker_cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/base_worker.py",
        "BaseWorker",
        {"shutdown"},
        {},
    )
    events = []
    engine = SimpleNamespace(
        shutdown_all_ranks=True,
        can_shutdown=lambda: False,
        shutdown=lambda: events.append("shutdown"),
        terminal_cleanup=lambda *, local_safe: events.append(("terminal_cleanup", local_safe)),
    )
    worker = worker_cls()
    worker.doing_shutdown = False
    worker.engine = engine

    worker.shutdown()

    assert events == ["shutdown", ("terminal_cleanup", True)]
    assert worker.engine is None

    events.clear()
    cleanup_attempts = 0
    gate_attempts = 0

    def can_shutdown():
        nonlocal gate_attempts
        gate_attempts += 1
        return gate_attempts == 1

    def terminal_cleanup():
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        events.append(("terminal_cleanup", True))
        if cleanup_attempts == 1:
            raise RuntimeError("terminal cleanup failed")

    engine = SimpleNamespace(
        shutdown_all_ranks=False,
        can_shutdown=can_shutdown,
        shutdown=lambda: events.append("shutdown"),
        terminal_cleanup=terminal_cleanup,
    )
    worker = worker_cls()
    worker.doing_shutdown = False
    worker.engine = engine

    with pytest.raises(RuntimeError, match="terminal cleanup failed"):
        worker.shutdown()

    assert events == ["shutdown", ("terminal_cleanup", True)]
    assert worker.engine is engine
    assert not worker.doing_shutdown

    worker.shutdown()

    assert events == [
        "shutdown",
        ("terminal_cleanup", True),
        ("terminal_cleanup", True),
    ]
    assert worker.engine is None
    assert gate_attempts == 1


def test_mpi_worker_terminal_cleanup_failure_is_retryable(monkeypatch):
    import types

    events = []
    fake_distributed = types.ModuleType("torch.distributed")
    fake_distributed.is_initialized = lambda: True
    fake_distributed.destroy_process_group = lambda: events.append("destroy_process_group")
    fake_torch = types.ModuleType("torch")
    fake_torch.distributed = fake_distributed
    fake_torch.cuda = SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_distributed)

    worker_cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/worker.py",
        "GenerationExecutorWorker",
        {"shutdown"},
        {
            "logger_debug": lambda *args, **kwargs: None,
            "mpi_rank": lambda: 0,
            "mpi_comm": lambda: SimpleNamespace(allgather=lambda value: [value]),
            "sys": sys,
            "_MEGA_MOE_DEEPGEMM_MODULE": "missing_test_megamoe_module",
            "gc": SimpleNamespace(collect=lambda: events.append("gc")),
        },
    )
    cleanup_attempts = 0

    def terminal_cleanup(*, local_safe):
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        events.append(("terminal_cleanup", local_safe))
        if cleanup_attempts == 1:
            raise RuntimeError("terminal cleanup failed")

    engine = SimpleNamespace(
        shutdown_all_ranks=True,
        can_enqueue_requests=lambda: False,
        shutdown=lambda: events.append("shutdown"),
        terminal_cleanup=terminal_cleanup,
    )
    worker = worker_cls()
    worker.doing_shutdown = False
    worker.rpc_server = None
    worker.engine = engine
    worker.llm_args = SimpleNamespace(backend="pytorch")
    worker._handle_background_error = lambda: events.append("background")

    with pytest.raises(RuntimeError, match="terminal cleanup failed"):
        worker.shutdown()

    assert events == ["shutdown", ("terminal_cleanup", True)]
    assert worker.engine is engine
    assert not worker.doing_shutdown

    worker.shutdown()

    assert events == [
        "shutdown",
        ("terminal_cleanup", True),
        ("terminal_cleanup", True),
        "destroy_process_group",
        "gc",
        "empty_cache",
        "background",
    ]
    assert worker.engine is None


def test_ray_worker_postproc_failure_still_runs_unsafe_terminal_cleanup():
    worker_cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/ray/gpu_worker.py",
        "RayGPUWorker",
        {"shutdown"},
        {
            "logger": SimpleNamespace(
                debug=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None
            ),
        },
    )
    events = []

    def fail_postproc():
        events.append("postproc")
        raise RuntimeError("postproc cleanup failed")

    worker = worker_cls()
    worker.doing_shutdown = False
    worker.rank = 0
    worker._postproc_pool = object()
    worker.shutdown_postproc_workers = fail_postproc
    engine = SimpleNamespace(
        shutdown_all_ranks=True,
        shutdown=lambda: events.append("shutdown"),
        terminal_cleanup=lambda *, local_safe: events.append(("terminal_cleanup", local_safe)),
    )
    worker.engine = engine
    worker.llm_args = SimpleNamespace(backend="pytorch")
    worker._handle_background_error = lambda: events.append("background")

    with pytest.raises(RuntimeError, match="postproc cleanup failed"):
        worker.shutdown()

    assert events == [
        "postproc",
        "shutdown",
        ("terminal_cleanup", False),
        "background",
    ]
    assert worker.engine is engine
    assert not worker.doing_shutdown

    worker._postproc_pool = None
    worker.shutdown()

    assert events == [
        "postproc",
        "shutdown",
        ("terminal_cleanup", False),
        "background",
        ("terminal_cleanup", True),
        "background",
    ]
    assert worker.engine is None


def test_ray_worker_shutdown_preserves_first_error_and_attempts_all_cleanup():
    worker_cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/ray/gpu_worker.py",
        "RayGPUWorker",
        {"shutdown"},
        {
            "logger": SimpleNamespace(
                debug=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None
            ),
        },
    )
    events = []

    def fail(name):
        def raise_error():
            events.append(name)
            raise RuntimeError(f"{name} failed")

        return raise_error

    worker = worker_cls()
    worker.doing_shutdown = False
    worker.rank = 0
    worker._postproc_pool = object()
    worker.shutdown_postproc_workers = fail("postproc")
    worker.engine = SimpleNamespace(
        shutdown_all_ranks=True,
        shutdown=lambda: events.append("shutdown"),
        terminal_cleanup=lambda *, local_safe: events.append(("terminal_cleanup", local_safe)),
    )
    worker.llm_args = SimpleNamespace(backend="pytorch")
    worker.checkpoint_loader = SimpleNamespace(cleanup=fail("checkpoint"))
    worker._handle_background_error = fail("background")

    with pytest.raises(RuntimeError, match="postproc failed"):
        worker.shutdown()

    assert events == [
        "postproc",
        "shutdown",
        ("terminal_cleanup", False),
        "checkpoint",
        "background",
    ]
    assert worker.engine is not None
    assert not worker.doing_shutdown
    assert worker.checkpoint_loader is not None


def test_ray_worker_partial_init_without_llm_args_still_checks_background():
    worker_cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/ray/gpu_worker.py",
        "RayGPUWorker",
        {"shutdown"},
        {
            "logger": SimpleNamespace(
                debug=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None
            ),
        },
    )
    events = []
    worker = worker_cls()
    worker.doing_shutdown = False
    worker.rank = 0
    worker.engine = None
    worker._handle_background_error = lambda: events.append("background")

    worker.shutdown()

    assert events == ["background"]


def test_standard_load_balancer_unsafe_preflight_skips_destructive_cleanup():
    manager_cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/moe_load_balancer.py",
        "MoeLoadBalancer",
        {"shutdown"},
        {},
    )
    calls = []

    class Comm:
        def allgather(self, local_safe):
            calls.append(("allgather", local_safe))
            return [local_safe, True]

        def barrier(self):
            calls.append("barrier")

    manager = manager_cls()
    manager.is_shutdown = False
    manager.in_iter = True
    manager.shared_mpi_comm = Comm()
    manager.single_layer_load_balancers = []
    manager.load_balancer_impl = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))

    assert manager.shutdown(local_safe=True) is False
    assert calls == [("allgather", False)]
