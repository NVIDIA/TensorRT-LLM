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
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
# Worker shutdown contracts below cover the collective path, which only
# per-layer dynamic EPLB takes.
_PER_LAYER_DYNAMIC_SHUTDOWN = {"uses_collective_shutdown": lambda llm_args: True}
_TORCH_ROOT = _ROOT / "tensorrt_llm/_torch"
_QUEUE = "CUDA_SCALE_LAUNCH_QUEUES"
_SCHEDULER_RUNTIME = _TORCH_ROOT / "moe/kernels/megamoe_scheduler/cuda_scheduler/runtime.py"
# The scheduler's native bound and the bound of the default 1800 s MoE
# communication warmup timeout at the nominal 2 GHz clock.
_SERVING_SPIN_CYCLES = 300 * 2_000_000_000
_WARMUP_SPIN_CYCLES = 1800 * 2_000_000_000


def _extract_class(
    path: Path,
    name: str,
    methods: set[str] | None,
    namespace: dict,
    bases: Tuple[str, ...] = (),
):
    tree = ast.parse(path.read_text())
    # Breadth-first, so a module-level class wins over a nested one of the same name.
    original = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name
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
    source = _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb/arena.py"
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
    source = _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb/arena.py"

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
    source = _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb/arena.py"

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
    native = []

    def submit_fused(handle, routes, route_target, spin_cycles):
        native.append(("launch", handle, routes, route_target, spin_cycles))

    torch = SimpleNamespace(
        ops=SimpleNamespace(trtllm=SimpleNamespace(moe_rebalance_tma_submit_fused=submit_fused)),
    )
    event_handle = _extract_function(source, "_event_handle", {})
    broadcast_cls = _extract_class(
        source,
        "HierarchicalSamiWeightBroadcast",
        {
            "submit_fused",
            "release_generation_after",
            "_require_live_bank_reusable",
            "mark_collective_reuse_safe",
        },
        {
            "torch": torch,
            "_event_handle": event_handle,
            "check_cuda": lambda result, operation: result,
        },
    )
    outputs = object()

    broadcast = broadcast_cls.__new__(broadcast_cls)
    broadcast._lock = nullcontext()
    broadcast._handoff_aborted = False
    broadcast._outputs = outputs
    # The native context binds every scheduler argument; a launch passes only its routes.
    broadcast._scheduler = SimpleNamespace()
    authority = object()
    broadcast._reuse_authority = authority

    def revalidate(_authority):
        raise AssertionError("the bound authority was validated when it was bound")

    # The per-layer path checks the bound authority by identity only.
    broadcast._validate_reuse_authority = revalidate
    broadcast._active_generation = 0
    broadcast._collective_safe_generation = 0
    broadcast._local_release_generation = 0
    broadcast._generation = 1
    broadcast._stream = 19
    broadcast.device = 0
    broadcast.ctx = "ctx"
    # The per-layer ops take the context's handle, not the custom-class object.
    broadcast._handle = 4242
    broadcast.terminal_flags = "terminals"
    broadcast.cuda = SimpleNamespace(
        cuStreamWaitEvent=lambda stream, event, flags: native.append(("wait", stream, event))
    )

    generation = broadcast.submit_fused(outputs, "routes", 8, spin_cycles=12345)

    # One native call orders the launch after the current stream and enqueues
    # it on the bound copy stream; Python switches no stream.
    assert native == [("launch", 4242, "routes", 8, 12345)]
    assert generation == 1
    assert (broadcast._active_generation, broadcast._generation) == (1, 2)
    # The bank stays leased until the consumer of generation 1 is proven done.
    with pytest.raises(RuntimeError, match="still leased"):
        broadcast.submit_fused(outputs, "routes", 16, spin_cycles=12345)
    with pytest.raises(RuntimeError, match="bound CUDA scheduler"):
        broadcast.submit_fused(object(), "routes", 16, spin_cycles=12345)

    # Only the bound authority, after the local consumer release, frees the bank.
    with pytest.raises(PermissionError, match="identity mismatch"):
        broadcast.mark_collective_reuse_safe(object(), 1)
    with pytest.raises(RuntimeError, match="local consumer release"):
        broadcast.mark_collective_reuse_safe(authority, 1)
    native.clear()
    # Without an event, the consumer is the work queued on the current stream.
    # The next launch orders the copy stream after that stream, so the release
    # queues no device work.
    broadcast.release_generation_after(1)
    assert native == []
    with pytest.raises(RuntimeError, match="already has a local release"):
        broadcast.release_generation_after(1)
    broadcast.mark_collective_reuse_safe(authority, 1)
    assert broadcast._collective_safe_generation == 1
    assert broadcast.submit_fused(outputs, "routes", 16, spin_cycles=12345) == 2

    # A caller-recorded consumer event is waited for on the copy stream instead.
    native.clear()
    broadcast.release_generation_after(2, SimpleNamespace(cuda_event=23))
    assert native == [("wait", 19, 23)]
    with pytest.raises(ValueError, match="positive CUDA event"):
        broadcast.release_generation_after(3, SimpleNamespace(cuda_event=0))


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
    source = _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb/arena.py"
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
        "ModelEngine",
        {"_set_warmup_effects", "warmup_scope"},
        {
            "contextmanager": contextmanager,
            "set_moe_comm_warmup": lambda value: events.append(("comm", value)),
        },
    )
    manager = SimpleNamespace(
        enable_statistic=True,
        enable_update_weights=True,
        set_iter_info=lambda **kwargs: events.append(("iter", kwargs)),
    )
    engine = engine_cls()
    engine.moe_load_balancer = manager
    engine._warmup_depth = 0

    with engine.warmup_scope():
        assert events == [
            ("comm", True),
            ("iter", {"enable_statistic": False, "enable_update_weights": False}),
        ]

    assert events[2:] == [
        ("comm", False),
        ("iter", {"enable_statistic": True, "enable_update_weights": True}),
    ]


def test_model_engine_cleanup_shuts_manager_after_graphs_while_model_is_alive():
    events = []
    engine_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/model_engine.py",
        "ModelEngine",
        {
            "prepare_cleanup",
            "shutdown_moe_load_balancer",
            "finish_cleanup",
            "_release_model_loader",
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
    engine._release_cuda_graphs = lambda **_: events.append("release_graphs")
    engine.moe_load_balancer = Manager()
    engine._runner = object()
    engine._model_caller = object()
    engine._mm_item_scheduler = object()
    engine.model = model
    engine.input_processor = object()

    engine.cleanup(collective=True)
    engine.cleanup(collective=True)

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
        "ModelEngine",
        {
            "prepare_cleanup",
            "shutdown_moe_load_balancer",
            "finish_cleanup",
            "_release_model_loader",
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
    engine._release_cuda_graphs = lambda **_: (_ for _ in ()).throw(
        RuntimeError("graph cleanup failed")
    )
    engine.moe_load_balancer = Manager()
    engine._runner = object()
    engine._model_caller = object()
    engine._mm_item_scheduler = object()
    engine.model = object()
    engine.input_processor = object()

    with pytest.raises(RuntimeError, match="graph cleanup failed"):
        engine.cleanup(collective=True)

    assert events == [("manager.shutdown", False)]
    assert engine.moe_load_balancer is not None
    assert engine.model is not None


def test_model_engine_finalizer_never_enters_collective_cleanup():
    calls = []
    engine_cls = _extract_class(
        _TORCH_ROOT / "pyexecutor/model_engine.py",
        "ModelEngine",
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
        {"terminal_cleanup", "release_rank_local_caches"},
        {
            "__name__": "tensorrt_llm._torch.pyexecutor.py_executor",
            "__package__": "tensorrt_llm._torch.pyexecutor",
            "sys": sys,
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

    class Engine:
        def release_executor_resources(self, *, release_model_owned_graphs):
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
        {"shutdown", "_release_execution_resources", "_release_execution_pools"},
        {
            "AsyncWorkerMixin": AsyncWorkerMixin,
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        },
    )
    worker_cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/base_worker.py",
        "BaseWorker",
        {"shutdown"},
        _PER_LAYER_DYNAMIC_SHUTDOWN,
    )

    engine = Engine()
    executor = executor_cls()
    executor.model_engine = engine
    executor.draft_model_engine = None
    executor._closed = False
    executor._resources_released = False
    executor.kv_cache_transceiver = None
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
    executor.enable_attention_dp = False
    executor._kv_connector_failed = False
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
            assert not hasattr(executor, "model_engine")
            assert executor._terminal_model_engines == (engine,)
            executor._terminal_model_engines = ()

    executor.terminal_cleanup = terminal_cleanup
    worker = worker_cls()
    worker.doing_shutdown = False
    worker.engine = executor

    for message in ("pool delete failed", "dwdp exit failed"):
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
        ("sampler_stop", 1),
        "release_graphs",
        "resource_shutdown",
        ("pool_delete", 1),
        ("terminal_cleanup", False),
        ("pool_delete", 2),
        ("dwdp_exit", 1),
        ("terminal_cleanup", False),
        ("dwdp_exit", 2),
        ("terminal_cleanup", True),
    ]
    assert executor.virtual_memory_pools == {}
    assert not hasattr(executor, "model_engine")
    assert executor.draft_model_engine is None
    assert executor.dwdp_manager is None
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

    def get_device(self):
        return int(self.device.split(":")[1]) if self.device.startswith("cuda:") else -1

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

    # A replayed override is too late for a context that already exists:
    # the check reads the setting the process had before the replay.
    test_environ[_QUEUE] = "4x"
    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        module.configure_moe_launch_queues(config(), original, initial_setting=None)
    assert module.configure_moe_launch_queues(config(), original, initial_setting="4x") == {
        "KEEP": "value",
        _QUEUE: "4x",
    }
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

    def configure_moe_launch_queues(moe_config, env_overrides, *, initial_setting):
        # The override must already be visible when the launch queue is configured,
        # while the late-init check sees the setting from before the replay.
        events.append(("configure", environ.get("USER_SETTING"), initial_setting))
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

    expected = [("env", "USER_SETTING", "1"), ("configure", "1", None)]
    common = {
        "os": fake_os,
        "logger": logger,
        "configure_moe_launch_queues": configure_moe_launch_queues,
        "LAUNCH_QUEUES_ENV": _QUEUE,
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
        lambda: worker_main(worker_queues=None, log_level="info", llm_args=args),
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
    assert run(lambda: worker_cls(llm_args=args), "executor_init") == expected + ["executor_init"]
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
    state = SimpleNamespace(device=0)
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
    source = _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb/planner.py"
    namespace = {"torch": torch}
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
    group._routes_in_flight = None
    group._lent_rows = -1
    group._lent_routes = None
    group._closed = False
    group._execution_stream_handle = None
    group.device = 0
    group.topk = 6
    group.max_tokens_per_rank = 8192
    group.auxiliary_sms = auxiliary_sms
    group.copy_stream = group.scheduler_stream = copy
    group._stream_handle = 19
    group._route_epoch_ptr = 77
    # Generation 1 launches during warmup, the later ones while serving.
    bounds = iter((_WARMUP_SPIN_CYCLES, _SERVING_SPIN_CYCLES, _SERVING_SPIN_CYCLES))
    group._spin_bound = SimpleNamespace(cycles=lambda: next(bounds))

    def driver_call(*entry):
        trace.append(entry)
        return (0,)

    # The planner records no event itself: the native launch and release
    # order the copy stream after MAIN.
    group._driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        CUstreamWaitValue_flags=SimpleNamespace(CU_STREAM_WAIT_VALUE_GEQ="geq"),
        cuStreamWaitValue64=lambda stream, address, value, flags: driver_call(
            "wait_value", stream, address, value, flags
        ),
    )
    group._wait_value_geq = group._driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ
    group._cuda_success = group._driver.CUresult.CUDA_SUCCESS
    outputs = SimpleNamespace(physical_slot_ids=_Routes(8192, trace))

    launch_bounds = []

    def submit_fused(result, routes, target, *, spin_cycles):
        assert result is outputs
        generation = group._scheduled_generation
        trace.append(("FUSED", routes.shape[0], target, generation))
        launch_bounds.append(spin_cycles)
        return generation

    group.scheduler = SimpleNamespace(device="cuda:0", outputs=outputs)
    group.broadcaster = SimpleNamespace(
        submit_fused=submit_fused,
        release_generation_after=lambda generation, event: trace.append(
            ("consumer_release", generation, event)
        ),
        mark_collective_reuse_safe=lambda provider, generation: trace.append(("lease", generation)),
    )
    group.lease = lease_cls(group)

    lent = {}
    for generation, rows in enumerate((16, 16, 0), 1):
        trace.clear()
        routes = _Routes(rows, trace)
        target = generation * auxiliary_sms
        route_wait = ("wait_value", 44, 77, target, "geq")
        part = group.plan_schedule(routes)
        physical, actual_generation = group.plan_finish(part, defer_wait=True)
        assert physical.shape == (rows, 6)
        assert actual_generation == generation
        # The view of the fixed output buffer is reused while the token count repeats.
        assert lent.setdefault(rows, physical) is physical
        assert route_wait not in trace
        # The group, not an allocator stream event, keeps the routes alive
        # until MAIN waits for every HALO-Q read of them.
        assert group._routes_in_flight is routes
        group.wait_for_routes()
        assert group._routes_in_flight is None
        group.wait_for_routes()
        group.finish()
        assert trace.count(route_wait) == 1
        assert not any(entry[0] == "lifetime" for entry in trace)
        assert trace.index(("FUSED", rows, target, generation)) < trace.index(route_wait)
        # Without an event, the release follows the MAIN work queued so far.
        assert trace[-1] == ("consumer_release", generation, None)
        if generation > 1:
            # The previous consumer is proven reusable before the next launch.
            assert trace.index(("lease", generation - 1)) < trace.index(
                ("FUSED", rows, target, generation)
            )
    assert launch_bounds == [_WARMUP_SPIN_CYCLES, _SERVING_SPIN_CYCLES, _SERVING_SPIN_CYCLES]

    # The bound-stream fast path still rejects another stream or device.
    main.cuda_stream = 45
    with pytest.raises(RuntimeError, match="MAIN stream cannot change"):
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

    # A submission that fails may have launched, so it keeps its routes alive,
    # and the group refuses further plans rather than let the allocator reuse them.
    group._pending_release = None
    group._spin_bound = SimpleNamespace(cycles=lambda: _SERVING_SPIN_CYCLES)
    group.broadcaster.submit_fused = lambda *args, **kwargs: 99
    stuck = _Routes(8, trace)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        group.plan_schedule(stuck)
    assert group._routes_in_flight is stuck
    with pytest.raises(RuntimeError, match="never ordered on MAIN"):
        group.plan_schedule(_Routes(8, trace))

    # A closed group rejects every route wait, including one of a waited generation.
    group._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        group.wait_for_routes()


def test_rebalance_submission_binds_one_main_stream_for_every_thread():
    priorities = {19: -1, 44: 0, 45: 0, 46: -1}
    current = SimpleNamespace(stream=46)
    torch = SimpleNamespace(
        _C=SimpleNamespace(
            _cuda_getDevice=lambda: 0,
            _cuda_getCurrentRawStream=lambda _: current.stream,
        ),
        cuda=SimpleNamespace(
            current_stream=lambda _: SimpleNamespace(
                cuda_stream=current.stream, priority=priorities[current.stream]
            )
        ),
    )
    group_cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb/planner.py",
        "RebalanceSlotSchedulerGroup",
        {"_check_owner"},
        {"torch": torch},
    )
    group = group_cls()
    group._closed = False
    group.device = 0
    group.copy_stream = SimpleNamespace(cuda_stream=19, priority=-1)
    group._stream_handle = 19
    group._execution_stream_handle = None

    # Neither COPY nor a stream of its priority can become MAIN.
    for stream in (46, 19):
        current.stream = stream
        with pytest.raises(RuntimeError, match="distinct from the higher-priority COPY"):
            group._check_owner()
    assert group._execution_stream_handle is None

    # The first other stream binds as MAIN, and any thread may submit on it:
    # warmup and serving run on different threads but share one stream.
    current.stream = 44
    group._check_owner()
    assert group._execution_stream_handle == 44
    errors = []

    def submit():
        try:
            group._check_owner()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    worker = threading.Thread(target=submit)
    worker.start()
    worker.join()
    assert errors == []

    current.stream = 45
    with pytest.raises(RuntimeError, match="MAIN stream cannot change"):
        group._check_owner()
    assert group._execution_stream_handle == 44


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
        _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb/planner.py",
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


def _tma_op_parameters(name: str) -> List[str]:
    """Parameter names of a registered moe_rebalance_tma operator schema."""
    op_source = (_ROOT / "cpp/tensorrt_llm/thop/moe/loadBalance/moeRebalanceTmaOp.cpp").read_text()
    schemas = ["".join(re.findall(r'"([^"]*)"', block)) for block in op_source.split("m.def(")[1:]]
    schema = next(text for text in schemas if text.startswith(f"{name}("))
    parameters = re.fullmatch(rf"{name}\((.*)\) -> \(\)", schema).group(1)
    return [parameter.split()[-1] for parameter in parameters.split(",")]


def test_fused_launch_binds_shared_arguments_and_passes_only_routes():
    # Map the scheduler arguments onto the registered operator schemas rather than hard-coding them.
    bind_names = _tma_op_parameters("moe_rebalance_tma_bind_fused")
    scheduler_names = bind_names[bind_names.index("out_slots") : bind_names.index("copy_stream")]
    # A launch passes its routes, route target and spin bound; the bind holds the rest.
    # They take the context's handle, since a custom-class argument costs
    # microseconds of dispatch per call.
    assert _tma_op_parameters("moe_rebalance_tma_submit_fused") == [
        "handle",
        "routes",
        "route_target",
        "spin_cycles",
    ]
    assert _tma_op_parameters("moe_rebalance_tma_release") == ["handle"]

    scheduler_cls = _extract_class(
        _SCHEDULER_RUNTIME,
        "CudaPhysicalSlotScheduler",
        {"fused_launch_args"},
        {},
    )
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.cfg = SimpleNamespace(
        ep_size=8,
        logical_expert_count=384,
        extra_slots_per_rank=4,
        local_rank=0,
        route_count=8192 * 6,
        ctas=4,
        threads=512,
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
    scheduler._bound_stream = None
    with pytest.raises(RuntimeError, match="bound GPU-direct copy"):
        scheduler.fused_launch_args()
    scheduler._bound_stream = 19

    args = scheduler.fused_launch_args()
    assert len(args) == len(scheduler_names)
    expected = {
        "out_slots": "slots",
        "out_ids": "ids",
        "out_levels": "levels",
        "out_owners": "owners",
        "peer_bases": "peer_base",
        "route_capacity": 8192 * 6,
        "topk": 6,
        "ctas": 4,
        "threads": 512,
        "ep": 8,
        "experts": 384,
        "helpers": 4,
        "local_rank": 0,
    }
    for name, value in expected.items():
        assert args[scheduler_names.index(name)] == value


def _cpp_method_body(source: str, signature: str) -> str:
    """Body of the C++ function whose definition starts with ``signature``."""
    open_brace = source.index("{", source.index(signature))
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : index]
    raise AssertionError(f"unbalanced body of {signature}")


def _assert_in_order(body: str, calls: Tuple[str, ...]) -> None:
    positions = [body.index(call) for call in calls]
    assert positions == sorted(positions), calls


def test_native_launch_and_release_order_the_copy_stream_after_main():
    # The cross-stream ordering lives in the native context, so pin its call order:
    # HALO-Q must not read routes before MAIN wrote them, and the next generation's
    # copy must not overwrite the live bank before MAIN's consumer of it.
    source = (_ROOT / "cpp/tensorrt_llm/thop/moe/loadBalance/moeRebalanceTmaOp.cpp").read_text()
    submit = _cpp_method_body(source, "void submitFused(torch::Tensor const& routes")
    _assert_in_order(
        submit,
        (
            "TORCH_CHECK_VALUE(routes.is_cuda()",
            "TORCH_CHECK_VALUE(spinCycles > 0",
            "cudaStream_t const execution = currentExecutionStream();",
            "cudaEventRecord(mInputReady, execution)",
            "cudaStreamWaitEvent(mCopyStream, mInputReady, 0)",
            "megamoe_tma_copy_submit_fused(",
        ),
    )
    launch = submit[submit.index("megamoe_tma_copy_submit_fused(") :]
    assert "reinterpret_cast<void*>(mCopyStream)" in launch
    release = _cpp_method_body(source, "void releaseAfterCurrentStream()")
    _assert_in_order(
        release,
        (
            "cudaStream_t const execution = currentExecutionStream();",
            "cudaEventRecord(mConsumerDone, execution)",
            "cudaStreamWaitEvent(mCopyStream, mConsumerDone, 0)",
        ),
    )
    # The stream a launch or release follows can never be the copy stream itself.
    current = _cpp_method_body(source, "cudaStream_t currentExecutionStream() const")
    assert "TORCH_CHECK(stream != mCopyStream" in current


def test_megamoe_cute_weight_cache_keys_every_view_field():
    conversions = []

    def to_cute(t, assumed_align):
        conversions.append((t, assumed_align))
        return ("cute", len(conversions))

    source = _TORCH_ROOT / "moe/custom_ops/cute_dsl_megamoe_custom_op.py"
    namespace = {
        "torch": SimpleNamespace(Tensor=object),
        "Tuple": Tuple,
        "_cute_launch_helpers": lambda: (to_cute,),
    }
    _extract_function(source, "_cute_view_key", namespace)
    to_cute_cached = _extract_function(source, "_to_cute_cached", namespace)

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

    # A slot keeps only the latest view of a buffer whose live rows change.
    to_cute_slot = _extract_function(source, "_to_cute_slot", namespace)
    slots = {}
    conversions.clear()
    rows_8 = to_cute_slot(slots, "output", weight(shape=(8, 16)))
    assert to_cute_slot(slots, "output", weight(shape=(8, 16))) is rows_8
    rows_4 = to_cute_slot(slots, "output", weight(shape=(4, 16)))
    assert rows_4 is not rows_8 and len(slots) == 1
    assert to_cute_slot(slots, "output", weight(shape=(8, 16))) is not rows_8
    assert len(conversions) == 3


def test_megamoe_cute_pointer_cache_keys_address_and_alignment():
    made = []

    def to_cute_ptr(t, assumed_align):
        made.append((t.data_ptr(), assumed_align))
        return ("ptr", len(made))

    source = _TORCH_ROOT / "moe/custom_ops/cute_dsl_megamoe_custom_op.py"
    cache = {}
    namespace = {
        "torch": SimpleNamespace(Tensor=object),
        "_cute_launch_helpers": lambda: (None, to_cute_ptr),
        "_MEGAMOE_CUTE_PTR_CACHE": cache,
        "_MEGAMOE_CUTE_PTR_CACHE_MAX_ENTRIES": 2,
    }
    to_cute_ptr_cached = _extract_function(source, "_to_cute_ptr_cached", namespace)

    def buffer(ptr):
        return SimpleNamespace(data_ptr=lambda: ptr)

    first = to_cute_ptr_cached(buffer(4096))
    # Another tensor at the same address gets the same pointer.
    assert to_cute_ptr_cached(buffer(4096)) is first
    assert to_cute_ptr_cached(buffer(4096), assumed_align=4) is not first
    assert len(made) == 2
    # The cache is bounded: a third entry starts it over.
    to_cute_ptr_cached(buffer(8192))
    assert len(cache) == 1
    assert to_cute_ptr_cached(buffer(4096)) is not first


def test_executors_of_one_engine_share_the_execution_stream():
    # The planner binds the first MAIN stream for good, so every executor
    # built for the engine, across profiling and serving, uses its one stream.
    tree = ast.parse((_TORCH_ROOT / "pyexecutor/py_executor_creator.py").read_text())
    create = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_create_py_executor"
    )
    calls = [
        node
        for node in ast.walk(create)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "create_py_executor_instance"
    ]
    assert calls
    for call in calls:
        stream = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "execution_stream"),
            None,
        )
        assert isinstance(stream, ast.Name) and stream.id == "execution_stream"


def test_route_wait_occurs_after_independent_staging():
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/mega_moe_cute_dsl.py"
    impl = _extract_class(
        source,
        "TrtllmCutedslMegaMoeNvfp4Impl",
        {"_stage_inputs"},
        {"torch": SimpleNamespace(uint8="uint8")},
    )

    def run():
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
        )
        return trace, owner

    trace, owner = run()
    assert trace.index("copy:activation") < trace.index("wait")
    assert trace.index("copy:topk_weights") < trace.index("wait")
    assert trace.index("wait") < trace.index("copy:topk_idx_local")
    assert owner._last_staged_T[8] == 4


@pytest.mark.parametrize("has_sm_budget", [True, False])
def test_megamoe_quantize_reserves_sms_only_for_a_replica_plan(has_sm_budget):
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/mega_moe_cute_dsl.py"
    calls = []

    class Activation:
        def __init__(self, rows, cols):
            self.shape = (rows, cols)

        def to(self, dtype):
            return self

        def contiguous(self):
            return self

        def view(self, *shape):
            return shape

    def fp4_quantize(*args):
        calls.append(("default", *args[1:]))
        return "x_fp4", Activation(3, 4)

    def sm_budget(*args):
        calls.append(("sm_budget", *args[1:]))
        return "x_fp4", Activation(3, 4)

    if has_sm_budget:
        fp4_quantize.sm_budget = sm_budget
    namespace = {
        "torch": SimpleNamespace(
            bfloat16="bf16", ops=SimpleNamespace(trtllm=SimpleNamespace(fp4_quantize=fp4_quantize))
        ),
        "logger": SimpleNamespace(warning=lambda message: None),
        "Fp4QuantizedTensor": type("Fp4QuantizedTensor", (), {}),
        "megamoe_activation_sf_bytes_per_row": lambda hidden: hidden // 16,
        "ceil_div": lambda a, b: -(-a // b),
    }
    _extract_function(source, "_fp4_quantize_has_sm_budget", namespace)
    impl = _extract_class(source, "TrtllmCutedslMegaMoeNvfp4Impl", {"quantize_input"}, namespace)
    owner = impl()
    owner.fc31_input_scale = "scale"

    owner.quantize_input(Activation(3, 64))
    owner.quantize_input(Activation(3, 64), reserved_sms=4)

    default = ("default", "scale", 16, False, False)
    budget = ("sm_budget", "scale", 16, False, False, 4) if has_sm_budget else default
    assert calls == [default, budget]


def test_megamoe_launch_buffers_are_built_once_per_bucket_and_mode():
    inference = SimpleNamespace(enabled=True)
    fake_torch = SimpleNamespace(is_inference_mode_enabled=lambda: inference.enabled)
    impl = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/mega_moe/mega_moe_cute_dsl.py",
        "TrtllmCutedslMegaMoeNvfp4Impl",
        {"_acquire_buffers"},
        {"torch": fake_torch},
    )
    built = []
    owner = impl()
    owner._launch_buffers = {}

    def build_buffers(**kwargs):
        built.append((kwargs["launch_max_T"], inference.enabled))
        return object()

    owner._build_buffers = build_buffers
    args = dict(top_k=6, hidden=7168, device="cuda:0", output_dtype="bf16")

    first = owner._acquire_buffers(**args, launch_max_T=8192)
    assert owner._acquire_buffers(**args, launch_max_T=8192) is first
    assert owner._acquire_buffers(**args, launch_max_T=4096) is not first
    # Views made under inference mode are kept apart from normal-mode views.
    inference.enabled = False
    assert owner._acquire_buffers(**args, launch_max_T=8192) is not first
    assert built == [(8192, True), (4096, True), (8192, False)]


def test_megamoe_symmetric_regions_are_built_once_per_mode():
    inference = SimpleNamespace(enabled=True)
    provider_cls = _extract_class(
        _TORCH_ROOT / "moe/custom_ops/cute_dsl_megamoe_custom_op.py",
        "MegaMoeSymmMemProvider",
        {"get_regions"},
        {"torch": SimpleNamespace(is_inference_mode_enabled=lambda: inference.enabled)},
    )
    provider = provider_cls()
    provider._regions = {}
    built = []
    provider._build_regions = lambda: built.append(inference.enabled) or object()

    regions = provider.get_regions()
    assert provider.get_regions() is regions
    inference.enabled = False
    assert provider.get_regions() is not regions
    assert built == [True, False]


def test_megamoe_tuning_config_is_built_once_per_runner():
    configs = []

    def tuning_config(**kwargs):
        configs.append(kwargs)
        return SimpleNamespace(**kwargs)

    runner_cls = _extract_class(
        _TORCH_ROOT / "moe/custom_ops/cute_dsl_megamoe_custom_op.py",
        "Sm100MegaMoENvfp4Runner",
        {"get_tuning_config", "_autotuner_inputs_pre_hook"},
        {
            "TuningConfig": tuning_config,
            "DynamicTensorSpec": lambda *args: args,
            "ConstraintSpec": lambda *args: args,
            "DistributedTuningStrategy": SimpleNamespace(MERGE="merge"),
            "get_last_power_of_2_num_tokens_buckets": None,
            "last_positive_power_of_2": None,
            "_megamoe_autotune_num_tokens": None,
        },
    )
    first, second = runner_cls(), runner_cls()
    for runner in (first, second):
        runner.max_tokens_per_rank = 8192

    config = first.get_tuning_config()
    assert first.get_tuning_config() is config
    # The profiling pre-hook stays bound to the runner that built the config.
    assert config.inputs_pre_hook.__self__ is first
    assert second.get_tuning_config().inputs_pre_hook.__self__ is second
    assert len(configs) == 2


def test_megamoe_runner_identity_is_built_once():
    runner_cls = _extract_class(
        _TORCH_ROOT / "moe/custom_ops/cute_dsl_megamoe_custom_op.py",
        "Sm100MegaMoENvfp4Runner",
        {"unique_id"},
        {},
    )
    runner = runner_cls()
    builds = []
    runner._build_unique_id = lambda: builds.append(1) or (100, 8, 6)

    assert runner.unique_id() == (100, 8, 6)
    assert runner.unique_id() == (100, 8, 6)
    assert builds == [1]


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

    tuner = SimpleNamespace(is_tuning_mode=False)
    namespace = {
        "AutoTuner": SimpleNamespace(get=lambda: tuner),
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

    # While the AutoTuner profiles, an autotune opt-in gets a fresh runner
    # that owns its profiling scratch.
    tuner.is_tuning_mode = True
    first_tuning = get_runner(**kwargs, tactic_autotune=True)
    second_tuning = get_runner(**kwargs, tactic_autotune=True)
    assert first_tuning is not second_tuning
    assert first_tuning is not production
    assert len(created) == 3

    # Outside tuning mode it only looks up its tactic: one runner is reused,
    # separate from the opted-out production runner.
    tuner.is_tuning_mode = False
    serving = get_runner(**kwargs, tactic_autotune=True)
    serving._profiling_scratch = "stale"
    assert get_runner(**kwargs, tactic_autotune=True) is serving
    assert serving is not production
    assert serving._profiling_scratch is None
    assert get_runner(**kwargs, tactic_autotune=False) is production
    assert len(created) == 4


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
        _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb/manager.py",
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


def test_post_moe_hook_releases_a_replica_plan_every_chunk():
    moe_cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/interface.py",
        "MoE",
        {"_load_balancer_start_set_cpu_stage", "_load_balancer_uses_replica_plan"},
        {},
    )
    moe_cls._using_dynamic_load_balancer = _extract_function(
        _TORCH_ROOT / "moe/fused_moe/impl_blocks.py", "_using_dynamic_load_balancer", {}
    )
    started = []

    class Layer:
        def __init__(self, *, produces_replica_plan, dynamic_routing):
            self.produces_replica_plan = produces_replica_plan
            self.dynamic_routing = dynamic_routing

        def is_dynamic_routing(self):
            return self.dynamic_routing

        def start_set_cpu_stage(self):
            started.append(self)

    cases = [
        # A per-layer dynamic layer disables standard host migration.
        (Layer(produces_replica_plan=True, dynamic_routing=False), [True, True]),
        (Layer(produces_replica_plan=False, dynamic_routing=True), [False, True]),
        (Layer(produces_replica_plan=False, dynamic_routing=False), [False, False]),
        (None, [False, False]),
    ]
    for layer, expected in cases:
        moe = moe_cls()
        moe.layer_load_balancer = layer
        observed = []
        for is_last_call in (False, True):
            started.clear()
            moe._load_balancer_start_set_cpu_stage(is_last_call)
            observed.append(started == [layer])
        assert observed == expected


def test_iteration_context_aborts_the_iteration_of_a_failed_forward():
    source = _TORCH_ROOT / "moe/fused_moe/moe_load_balancer.py"
    context_cls = _extract_class(source, "MoeLoadBalancerIterContext", None, {})
    base_cls = _extract_class(source, "BaseMoeLoadBalancer", {"abort_iter"}, {})
    events = []

    class Manager:
        static = False

        def is_static_routing(self):
            return self.static

        def set_iter_info(self, enable_statistic, enable_update_weights):
            events.append("set_iter_info")

        def start_iter(self):
            events.append("start_iter")

        def end_iter(self):
            events.append("end_iter")

        def abort_iter(self, error):
            events.append(("abort_iter", error))

    manager = Manager()
    with context_cls(manager):
        pass
    assert events == ["set_iter_info", "start_iter", "end_iter"]

    events.clear()
    error = RuntimeError("forward failed")
    with pytest.raises(RuntimeError, match="forward failed") as caught:
        with context_cls(manager):
            raise error
    assert caught.value is error
    assert events == ["set_iter_info", "start_iter", ("abort_iter", error)]

    # Static placement runs no iteration hook.
    events.clear()
    manager.static = True
    with pytest.raises(RuntimeError, match="forward failed"):
        with context_cls(manager):
            raise error
    assert events == []

    # By default an aborted iteration simply ends, as standard EPLB always did.
    class Standard(base_cls):
        def end_iter(self):
            events.append("end_iter")

    Standard().abort_iter(error)
    assert events == ["end_iter"]


def test_per_layer_dynamic_abort_releases_pending_plans_and_keeps_the_error():
    namespace = {}
    _extract_function(
        _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb/layer.py", "_append_note", namespace
    )
    manager_cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/per_layer_dynamic_eplb/manager.py",
        "PerLayerDynamicMoeLoadBalancer",
        {"abort_iter", "shutdown"},
        namespace,
    )
    released = []

    class Layer:
        def __init__(self, layer_idx, plan=None, cleanup_error=None):
            self.layer_idx = layer_idx
            self.replica_plan = plan
            self.cleanup_error = cleanup_error

        def finish_replica_plan(self):
            released.append(self.layer_idx)
            if self.cleanup_error is not None:
                raise self.cleanup_error
            self.replica_plan = None

    manager = manager_cls()
    manager.in_iter = True
    manager.iter_id = 3
    manager.single_layer_load_balancers = [
        Layer(0),
        Layer(1, plan="first", cleanup_error=RuntimeError("route wait failed")),
        Layer(2, plan="second"),
    ]
    error = ValueError("forward failed")

    manager.abort_iter(error)

    # Every pending plan is released; a failed release only annotates the
    # forward's error, which the iteration context then re-raises.
    assert released == [1, 2]
    assert [layer.replica_plan for layer in manager.single_layer_load_balancers] == [
        None,
        "first",
        None,
    ]
    assert len(error.__notes__) == 1
    assert "layer 1" in error.__notes__[0]
    assert "route wait failed" in error.__notes__[0]
    assert (manager.in_iter, manager.iter_id) == (False, 4)

    # The plan left pending makes this rank unsafe for collective teardown.
    states = []

    class Comm:
        def allgather(self, state):
            states.append(state)
            return [state, {"local_safe": True, "pool_count": 0}]

    manager.is_shutdown = False
    manager._ep_comm = Comm()
    manager._rebalance_shared_slot_pool_order = []
    assert manager.shutdown(local_safe=True) is False
    assert states == [{"local_safe": False, "pool_count": 0}]

    # The iteration ends even if the release loop itself fails.
    class Broken:
        @property
        def replica_plan(self):
            raise RuntimeError("layer state unavailable")

    manager.in_iter = True
    manager.single_layer_load_balancers = [Broken()]
    with pytest.raises(RuntimeError, match="layer state unavailable"):
        manager.abort_iter(ValueError("forward failed"))
    assert (manager.in_iter, manager.iter_id) == (False, 5)


class _ChunkRoutes:
    dtype = "int32"

    def __init__(self, name):
        self.name = name


@pytest.mark.parametrize("mode", ["per_layer_dynamic", "standard", "off"])
def test_fused_comm_chunk_routes_once_after_recording_the_planner_input(mode):
    source = _TORCH_ROOT / "moe/fused_moe/moe_scheduler.py"
    trace = []
    labels = []

    class ExpertStatistic:
        @staticmethod
        def set_layer(layer_idx):
            assert layer_idx == 7

        @staticmethod
        def maybe_add_info(num_slots, routes):
            labels.append(num_slots)
            trace.append(("statistic", routes.name))

    def maybe_collect_or_replay_slots(num_slots, routes):
        labels.append(num_slots)
        trace.append(("calibrator", routes.name))
        return _ChunkRoutes("replayed " + routes.name)

    namespace = {
        "ExpertStatistic": ExpertStatistic,
        "get_calibrator": lambda: SimpleNamespace(
            maybe_collect_or_replay_slots=maybe_collect_or_replay_slots
        ),
        "get_active_route_capture": lambda: None,
        "torch": SimpleNamespace(int32="int32", float32="float32"),
        "Fp4QuantizedTensor": type("Fp4QuantizedTensor", (), {}),
        "MoERunContext": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    _extract_function(source, "_record_routes", namespace)
    _extract_function(source, "_leading_rows", namespace)
    scheduler_cls = _extract_class(source, "FusedCommMoEScheduler", {"_forward_chunk"}, namespace)

    plan = SimpleNamespace(reserved_sms=4)
    layer = (
        None
        if mode == "off"
        else SimpleNamespace(
            produces_replica_plan=mode != "standard",
            require_replica_plan=lambda: plan,
        )
    )

    def route(routes, use_dp):
        trace.append(("route", routes.name))
        return _ChunkRoutes("physical" if mode == "per_layer_dynamic" else "slots")

    def run_moe(ctx):
        trace.append(("run_moe", ctx.token_selected_experts.name, ctx.replica_plan))
        return "out"

    moe = SimpleNamespace(
        apply_router_weight_on_input=False,
        mapping=SimpleNamespace(moe_ep_rank=0),
        layer_idx=7,
        num_experts=16,
        num_slots=24,
        use_dp=True,
        layer_load_balancer=layer,
        routing_method=SimpleNamespace(
            apply=lambda logits, input_ids: (
                _ChunkRoutes("logical"),
                SimpleNamespace(dtype="float32"),
            )
        ),
        _maybe_get_perfect_router_expert_ids=lambda routes: routes,
        _load_balancer_start_wait_gpu_stage=lambda is_first_call: trace.append("start_wait_gpu"),
        _load_balancer_done_wait_gpu_stage=lambda is_first_call: trace.append("done_wait_gpu"),
        _load_balancer_update_statistic=lambda routes, *args, **kwargs: trace.append(
            ("update_statistic", routes.name)
        ),
        _load_balancer_uses_replica_plan=lambda: mode == "per_layer_dynamic",
        _load_balancer_route=route,
        _load_balancer_start_set_cpu_stage=lambda is_last_call: trace.append("start_set_cpu"),
        _load_balancer_done_set_cpu_stage=lambda is_last_call: trace.append("done_set_cpu"),
        backend=SimpleNamespace(
            quantize_input=lambda x, **kwargs: trace.append(("quantize", kwargs)) or ("x", "sf"),
            set_adaptive_launch_tokens=lambda tokens: trace.append(("bucket", tokens)),
            run_moe=run_moe,
        ),
    )
    scheduler = scheduler_cls()
    scheduler.moe = moe
    x = SimpleNamespace(shape=(4, 16))

    out = scheduler._forward_chunk(
        x,
        SimpleNamespace(shape=(4, 24)),
        output_dtype="bf16",
        all_rank_num_tokens=[4, 6],
        do_finalize=True,
    )

    assert out == "out"
    tail = [("quantize", {}), ("bucket", 6)]
    if mode == "per_layer_dynamic":
        # The planner consumes the logical routes the calibrator records or
        # replays, and quantization leaves the plan's SMs to its copy kernel.
        # The layer has no CPU stage or host statistics, so only the post-MoE
        # hook that releases its plan runs.
        expected = [
            ("statistic", "logical"),
            ("calibrator", "logical"),
            ("route", "replayed logical"),
            ("quantize", {"reserved_sms": 4}),
            ("bucket", 6),
            ("run_moe", "physical", plan),
            "start_set_cpu",
        ]
    elif mode == "standard":
        expected = [
            "start_wait_gpu",
            "done_wait_gpu",
            ("update_statistic", "logical"),
            ("route", "logical"),
            ("statistic", "slots"),
            ("calibrator", "slots"),
            *tail,
            ("run_moe", "replayed slots", None),
            "start_set_cpu",
            "done_set_cpu",
        ]
    else:
        expected = [
            "start_wait_gpu",
            ("statistic", "logical"),
            ("calibrator", "logical"),
            *tail,
            ("run_moe", "replayed logical", None),
            "start_set_cpu",
            "done_set_cpu",
        ]
    assert trace == expected
    # Routes recorded before the route are expert IDs; after it, slots.
    assert labels == ([16, 16] if mode == "per_layer_dynamic" else [24, 24])


def test_base_worker_finalizes_rank_follower_pyexecutor():
    worker_cls = _extract_class(
        _ROOT / "tensorrt_llm/executor/base_worker.py",
        "BaseWorker",
        {"shutdown"},
        _PER_LAYER_DYNAMIC_SHUTDOWN,
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
            **_PER_LAYER_DYNAMIC_SHUTDOWN,
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
            **_PER_LAYER_DYNAMIC_SHUTDOWN,
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
            **_PER_LAYER_DYNAMIC_SHUTDOWN,
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
            **_PER_LAYER_DYNAMIC_SHUTDOWN,
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


def test_standard_load_balancer_finalizer_releases_it_as_on_main():
    manager_cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/moe_load_balancer.py",
        "MoeLoadBalancer",
        {"__del__"},
        {},
    )
    calls = []
    manager = manager_cls()
    manager.shutdown = lambda: calls.append("shutdown")

    manager.is_shutdown = True
    manager.__del__()
    assert calls == []

    manager.is_shutdown = False
    manager.__del__()
    assert calls == ["shutdown"]
    manager.is_shutdown = True


class _FakePlane:
    """Stand-in for an arena alias: records fills, passes the layout checks."""

    is_cuda = True
    dtype = "float32"

    def __init__(self, name, address, fills):
        self.name = name
        self.shape = (6,)
        self._address = address
        self._fills = fills

    def stride(self):
        return (1,)

    def data_ptr(self):
        return self._address

    def fill_(self, value):
        self._fills.append((self.name, value))


def test_live_plane_binding_restores_per_expert_scalar_defaults():
    names = (
        "mega_fc1_weight",
        "mega_fc1_weight_sf",
        "mega_fc2_weight",
        "mega_fc2_weight_sf",
        "fc31_alpha",
        "fc2_alpha",
        "fc1_norm_const",
    )
    fills = []
    aliases = [_FakePlane(name, 0x1000 + index, fills) for index, name in enumerate(names)]
    arena = SimpleNamespace(
        bundle="bundle",
        plane_names=names,
        local_plane_views=aliases,
        tekit_alias_views=aliases,
        home_experts=4,
    )
    method_cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/quantization.py",
        "NVFP4MegaMoECuteDslMethod",
        {"_bind_live_weight_planes"},
        {"torch": SimpleNamespace(nn=SimpleNamespace(Parameter=None))},
    )
    method = method_cls()
    method.live_weight_plane_spec = lambda module: SimpleNamespace(
        planes=[SimpleNamespace(name=name) for name in names]
    )
    arena.bundle = method.live_weight_plane_spec(None)
    method.assert_live_weight_aliases = lambda module, bound: None
    module = SimpleNamespace(
        **{
            name: SimpleNamespace(
                is_meta=False, shape=(6,), stride=lambda: (1,), dtype="float32", data=None
            )
            for name in names
        }
    )

    method._bind_live_weight_planes(module, arena)

    # The arena starts zeroed; an expert the checkpoint leaves unset must keep
    # the create-time scalars (1.0) instead of silently computing zeros.
    assert sorted(fills) == sorted(
        [("fc31_alpha", 1.0), ("fc2_alpha", 1.0), ("fc1_norm_const", 1.0)]
    )
    assert all(getattr(module, name).data is alias for name, alias in zip(names, aliases))
