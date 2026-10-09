# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast CPU contracts for the Dynamic EPLB serving path."""

from __future__ import annotations

import ast
import functools
import importlib.util
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_TORCH_ROOT = _ROOT / "tensorrt_llm/_torch"
_QUEUE = "CUDA_SCALE_LAUNCH_QUEUES"


def _extract_class(path: Path, name: str, methods: set[str] | None, namespace: dict):
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
        selected = ast.ClassDef(name=name, bases=[], keywords=[], body=body, decorator_list=[])
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            selected,
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
        {"set_moe_a2a_warmup": lambda value: events.append(("a2a", value))},
    )
    manager = SimpleNamespace(
        enable_statistic=True,
        enable_update_weights=True,
        set_iter_info=lambda **kwargs: events.append(("iter", kwargs)),
        set_warmup=lambda value: events.append(("warmup", value)),
    )
    engine = engine_cls()
    engine.moe_load_balancer = manager

    engine.is_warmup = True

    assert events == [
        ("a2a", True),
        ("iter", {"enable_statistic": False, "enable_update_weights": False}),
        ("warmup", True),
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
    module_name = "tensorrt_llm._torch.cute_dsl_kernels.megamoe_shared_fc12"
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


def test_launch_queue_is_enabled_only_for_per_iteration_mode(monkeypatch):
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

    def config(mode="per_iteration"):
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


def test_launch_queue_setup_precedes_runtime_initialization():
    def call_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{call_name(node.value)}.{node.attr}"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "super"
        ):
            return "super()"
        return ""

    def calls(relative, function_name, class_name=None):
        tree = ast.parse((_ROOT / relative).read_text())
        owner = tree
        if class_name is not None:
            owner = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
        function = next(
            node
            for node in owner.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        result = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                name = call_name(node.func)
                result[name] = min(result.get(name, node.lineno), node.lineno)
        return result

    base = calls("tensorrt_llm/llmapi/llm.py", "__init__", "BaseLLM")
    assert (
        base["self._process_env_overrides"]
        < base["configure_moe_launch_queues"]
        < base["llm_args_cls"]
    )
    assert base["configure_moe_launch_queues"] < base["get_device_count"]

    worker_main = calls("tensorrt_llm/executor/worker.py", "worker_main")
    assert (
        worker_main["os.environ.update"]
        < worker_main["configure_moe_launch_queues"]
        < worker_main["worker_cls"]
    )
    assert worker_main["configure_moe_launch_queues"] < worker_main["mpi_comm"]

    worker = calls("tensorrt_llm/executor/base_worker.py", "__init__", "BaseWorker")
    assert (
        worker["os.environ.update"]
        < worker["configure_moe_launch_queues"]
        < worker["super().__init__"]
    )

    ray = calls("tensorrt_llm/executor/ray/gpu_worker.py", "__init__", "RayWorkerWrapper")
    assert (
        ray["os.environ.update"]
        < ray["configure_moe_launch_queues"]
        < ray["torch.cuda.device_count"]
        < ray["torch.cuda.set_device"]
    )


def test_direct_inputs_are_not_retained_by_dtype_view_memo():
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/mega_moe_cute_dsl.py"
    tree = ast.parse(source.read_text())
    launch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cute_dsl_megamoe_nvfp4_blackwell"
    )
    arguments = {keyword.arg: keyword.value for keyword in launch.keywords}

    for name, caster in (("activation", "_as_nvfp4"), ("activation_sf", "_as_fp8_sf")):
        value = arguments[name]
        assert isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Name) and value.func.id == caster

    for name in ("fc1_weight", "fc1_weight_sf", "fc2_weight", "fc2_weight_sf"):
        value = arguments[name]
        assert isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Attribute)
        assert value.func.attr == "_memo_dtype_view"

    reserved_sms = arguments["reserved_sms"]
    assert isinstance(reserved_sms, ast.Name)
    assert reserved_sms.id == "reserved_sms"


def test_gpu_direct_tma_fails_closed_on_stale_scheduler_plan():
    source = (
        _ROOT / "cpp/tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTma.cu"
    ).read_text()
    begin = source.index("__global__ void tma_copy_gpu_direct_kernel")
    end = source.index("struct MegamoeTmaCopyState", begin)
    kernel = source[begin:end]

    status_check = kernel.index("statusOk &= direct.status[field] == 0")
    epoch_check = kernel.index("schedulerPlanReady = statusOk && planEpoch == schedulerEpoch")
    rejection = kernel.index("if (!schedulerPlanReady)")
    trap = kernel.index('asm volatile("trap;"', rejection)
    publish = kernel.index("publish_plan_channel_warp<false>")
    build = kernel.index("megamoe_tma_build_gpu_plan")
    ready = kernel.index("finish_cta_and_notify")

    assert status_check < epoch_check < rejection <= trap < publish < build < ready

    submit = source[source.index("megamoe_tma_copy_submit_gpu_direct") :]
    assert "tma_copy_gpu_direct_kernel<<<" in submit
    assert "cudaLaunchAttributeProgrammaticStreamSerialization" not in submit
    assert "cudaLaunchKernelEx" not in submit


def test_halo_q_rendezvous_resets_arrivals_and_wraps_epoch_safely():
    source = (
        _ROOT / "cpp/tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQ.cu"
    ).read_text()
    begin = source.index("__device__ bool grid_rendezvous")
    end = source.index("__device__ void local_histogram_pass", begin)
    rendezvous = source[begin:end]

    assert "using Epoch = std::uint32_t;" in source
    assert "return observed != expected && expected - observed < (Epoch{1} << 31);" in source
    arrival = rendezvous.index("fetch_add_acq_rel_device(sync, 1)")
    last_cta = rendezvous.index("old == static_cast<Epoch>(blocks - 1)")
    reset = rendezvous.index("store_relaxed_device(sync, 0)")
    publish = rendezvous.index("store_release_device(sync + 1, target)")
    wait = rendezvous.index("while (epoch_before(observed, target))")
    assert arrival < last_cta < reset < publish < wait

    mask = (1 << 32) - 1

    def epoch_before(observed: int, expected: int) -> bool:
        delta = (expected - observed) & mask
        return observed != expected and delta < (1 << 31)

    epochs = (0xFFFFFFFE, 0xFFFFFFFF, 0x00000000, 0x00000001)
    arrivals = 0
    for observed, expected in zip(epochs, epochs[1:]):
        for _ in range(128):
            old = arrivals
            arrivals += 1
            if old == 127:
                arrivals = 0
        assert arrivals == 0
        assert epoch_before(observed, expected)
        assert not epoch_before(expected, observed)
        assert not epoch_before(expected, expected)


def test_direct_submit_orders_generations_and_defers_the_route_wait():
    trace = []
    state = SimpleNamespace(device=0, thread=7)
    main = SimpleNamespace(cuda_stream=44, priority=0)
    copy = SimpleNamespace(cuda_stream=19, priority=-1)
    cuda = SimpleNamespace(
        current_device=lambda: state.device,
        current_stream=lambda _: main,
        nvtx=SimpleNamespace(range=lambda _: nullcontext()),
    )
    torch = SimpleNamespace(cuda=cuda, Tensor=_Routes, int32="int32")
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
    group.copy_stream = group.scheduler_stream = copy
    group._stream_handle = 19
    group._input_ready_handle = 1
    group._plan_ready_handle = 2
    group._consumer_done_handle = 3
    group._consumer_done = SimpleNamespace(cuda_event=3)

    def driver_call(*entry):
        trace.append(entry)
        return (0,)

    group._driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        cuEventRecord=lambda event, stream: driver_call("record", event, stream),
        cuStreamWaitEvent=lambda stream, event, flags: driver_call("wait", stream, event),
    )
    outputs = SimpleNamespace(physical_slot_ids=_Routes(8192, trace))
    pending = False

    def submit(routes, handle):
        nonlocal pending
        assert not pending and handle == 19
        pending = True
        trace.append(("HALO", routes.shape[0], id(routes)))
        return outputs

    def copy_submit(result):
        nonlocal pending
        assert pending and result is outputs
        pending = False
        generation = group._scheduled_generation
        trace.append(("TMA", generation))
        return SimpleNamespace(generation=generation)

    group.scheduler = SimpleNamespace(device="cuda:0", submit=submit)
    group.broadcaster = SimpleNamespace(
        submit=copy_submit,
        release_generation_after=lambda generation, event: trace.append(
            ("consumer_release", generation, event.cuda_event)
        ),
        mark_collective_reuse_safe=lambda provider, generation: trace.append(("lease", generation)),
    )
    group.lease = lease_cls(group)

    for generation, rows in enumerate((16, 0), 1):
        trace.clear()
        routes = _Routes(rows, trace)
        part = group.plan_schedule(routes)
        physical, actual_generation = group.plan_finish(part, defer_wait=True)
        assert physical.shape == (rows, 6)
        assert actual_generation == generation
        assert not any(entry[0] == "wait" and entry[1:] == (44, 2) for entry in trace)
        group.wait_for_routes()
        group.wait_for_routes()
        group.finish()
        assert sum(entry == ("wait", 44, 2) for entry in trace) == 1
        assert trace.index(("HALO", rows, id(routes))) < trace.index(("TMA", generation))
        assert trace.index(("TMA", generation)) < trace.index(("wait", 44, 2))
        assert trace[-2:] == [("record", 3, 44), ("consumer_release", generation, 3)]

    trace.clear()
    with pytest.raises(ValueError, match="contiguous CUDA int32"):
        group.plan_schedule(_Routes(8, trace, dtype="int64"))
    assert trace == []

    group._driver.cuStreamWaitEvent = lambda *args: (17,)
    with pytest.raises(RuntimeError, match="cuStreamWaitEvent"):
        group.plan_schedule(_Routes(8, trace))
    assert not any(entry[0] == "HALO" for entry in trace)
    assert group._plan_part is None


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
        "_launch_cluster_configuration",
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
        reserved_sms=0,
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
    different_reservation = get_runner(**{**kwargs, "reserved_sms": 8}, tactic_autotune=False)
    assert different_reservation is not production
    assert different_reservation.kwargs["reserved_sms"] == 8
    assert len(created) == 4


def test_megamoe_runner_launch_geometry_uses_reserved_sm_budget():
    source = _TORCH_ROOT / "moe/custom_ops/cute_dsl_megamoe_custom_op.py"
    tree = ast.parse(source.read_text())
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_build_kernel_once", "_tactic_cache_key"}
    }
    for method_name in ("_build_kernel_once", "_tactic_cache_key"):
        launch_call = next(
            node
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_launch_cluster_configuration"
        )
        keywords = {keyword.arg: keyword.value for keyword in launch_call.keywords}
        for keyword_name in ("reserved_sms", "max_sm_count"):
            value = keywords[keyword_name]
            assert isinstance(value, ast.Attribute)
            assert isinstance(value.value, ast.Name)
            assert value.value.id == "self"
            assert value.attr == keyword_name


def test_megamoe_persistent_grid_reserves_sms_for_copy_work():
    helpers = _autotune_helpers(None)
    max_clusters = {2: 72, 4: 32}
    helpers["_max_active_clusters"] = lambda cluster_size, sm_version=None: max_clusters[
        cluster_size
    ]
    launch_config = helpers["_launch_cluster_configuration"]

    uniform = launch_config([2, 1, 1], None, 107, reserved_sms=8, max_sm_count=144)
    mixed = launch_config([4, 1, 1], [2, 1, 1], 107, reserved_sms=8, max_sm_count=144)
    rounded = launch_config([4, 1, 1], [2, 1, 1], 107, reserved_sms=9, max_sm_count=144)

    assert uniform == (68, None, None, 136)
    assert mixed == (34, 32, 4, 136)
    assert rounded == (33, 32, 2, 132)
    with pytest.raises(ValueError, match="reserved_sms"):
        launch_config([2, 1, 1], None, 107, reserved_sms=144, max_sm_count=144)
    with pytest.raises(TypeError, match="reserved_sms"):
        launch_config([2, 1, 1], None, 107, reserved_sms=True, max_sm_count=144)


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


def test_per_iteration_shutdown_reports_in_iter_through_collective_preflight():
    manager_cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/per_iteration_eplb.py",
        "PerIterationMoeLoadBalancer",
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
    manager.ep_process_group = object()
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
