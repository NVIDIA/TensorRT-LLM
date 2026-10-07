# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast CPU contracts for the Dynamic EPLB serving path."""

from __future__ import annotations

import ast
import copy
import functools
import gc
import importlib.util
import math
import os
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple
from weakref import WeakKeyDictionary

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


def test_shared_slot_arena_is_default_and_model_local():
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/rebalance_live_arena_v2.py"
    text = source.read_text()
    assert "TRTLLM_MOE_REBALANCE_SHARED_SLOTS" not in text
    assert "TRTLLM_MOE_REBALANCE_SHARED_SLOT_SETS" not in text
    assert "HierarchicalFabricArenaProvider" not in text

    tree = ast.parse(text)
    count = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_HELPER_BANK_COUNT"
            for target in node.targets
        )
    )
    assert ast.literal_eval(count.value) == 2

    scope_cls = _extract_class(source, "_PoolScope", None, {})
    registry = WeakKeyDictionary()
    provider_cls = _extract_class(
        source,
        "SharedSlotArenaProvider",
        {"__init__"},
        {
            "_POOL_SCOPE_ATTRIBUTE": "_trtllm_rebalance_shared_slot_scope",
            "_PoolScope": scope_cls,
            "_SHARED_SLOT_POOLS": registry,
        },
    )

    class StructurallyEqualOwner:
        def __eq__(self, other):
            return isinstance(other, StructurallyEqualOwner)

        def __hash__(self):
            return 1

    first = StructurallyEqualOwner()
    second = StructurallyEqualOwner()
    first_provider = provider_cls(first)
    same_model_provider = provider_cls(first)
    second_provider = provider_cls(second)
    cloned_provider = provider_cls(copy.deepcopy(first))

    assert first_provider._pools is same_model_provider._pools
    assert first_provider._pools is not second_provider._pools
    assert first_provider._pools is not cloned_provider._pools
    assert len(registry) == 3

    del second_provider, second
    gc.collect()
    assert len(registry) == 2


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


def test_launch_queue_is_on_only_and_reaches_process_entrypoints(monkeypatch):
    path = _ROOT / "tensorrt_llm/llmapi/_load_balance_env.py"
    spec = importlib.util.spec_from_file_location("load_balance_env_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    initialized = False
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: initialized),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv(_QUEUE, raising=False)

    def config(enabled=True, slots=4, active=None):
        if active is None:
            active = enabled and slots > 0
        return SimpleNamespace(
            rebalance=SimpleNamespace(
                enabled=enabled,
                helper_slots_per_rank=slots,
                is_active=active,
            )
        )

    original = {"KEEP": "value"}
    configured = module.configure_moe_launch_queues(config(), original)
    assert configured == {"KEEP": "value", _QUEUE: "4x"}
    assert configured is not original
    assert original == {"KEEP": "value"}
    assert os.environ[_QUEUE] == "4x"
    assert module.configure_moe_launch_queues(config(), configured) == configured

    for off in (None, config(enabled=False), config(slots=0)):
        monkeypatch.delenv(_QUEUE, raising=False)
        before = {"KEEP": "value"}
        assert module.configure_moe_launch_queues(off, before) is before
        assert _QUEUE not in os.environ

    initialized = True
    before = {"KEEP": "value"}
    assert module.configure_moe_launch_queues(config(active=False), before) is before
    assert _QUEUE not in os.environ

    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        module.configure_moe_launch_queues(config(), original)
    assert original == {"KEEP": "value"}
    assert _QUEUE not in os.environ

    entrypoints = (
        ("tensorrt_llm/llmapi/llm.py", "BaseLLM", "__init__", ("get_device_count",)),
        ("tensorrt_llm/executor/worker.py", None, "worker_main", ("barrier", "update")),
        ("tensorrt_llm/executor/base_worker.py", "BaseWorker", "__init__", ("__init__",)),
        (
            "tensorrt_llm/executor/ray/gpu_worker.py",
            "RayWorkerWrapper",
            "__init__",
            ("device_count", "set_device"),
        ),
    )
    for relative, class_name, method_name, boundaries in entrypoints:
        tree = ast.parse((_ROOT / relative).read_text())
        scope = tree
        if class_name is not None:
            scope = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            )
        method = next(
            node
            for node in scope.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
        calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
            and (
                getattr(node.func, "id", None) == "configure_moe_launch_queues"
                or getattr(node.func, "attr", None) == "configure_moe_launch_queues"
            )
        ]
        assert len(calls) == 1, relative
        setup_line = calls[0].lineno
        for boundary in boundaries:
            boundary_calls = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, (ast.Name, ast.Attribute))
                and (
                    getattr(node.func, "id", None) == boundary
                    or getattr(node.func, "attr", None) == boundary
                )
                and node is not calls[0]
            ]
            assert boundary_calls, (relative, boundary)
            assert setup_line < min(node.lineno for node in boundary_calls), (relative, boundary)


def test_halo_q_is_the_only_scheduler_planner():
    sources = {
        "runtime": _TORCH_ROOT / "cute_dsl_kernels/megamoe_scheduler_v2/cuda_scheduler/runtime.py",
        "integration": _TORCH_ROOT / "moe/fused_moe/mega_moe/rebalance_slot_scheduler_v2.py",
        "kernel": _ROOT
        / "cpp/tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQ.cu",
        "kernel_header": _ROOT
        / "cpp/tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQ.h",
        "torch_op": _ROOT / "cpp/tensorrt_llm/thop/moe/loadBalance/moeRebalanceHaloQOp.cpp",
    }
    retired = (
        "plan_legacy",
        "TRTLLM_MOE_REBALANCE_ALGO",
        "algorithmValue",
        "params.algorithm",
    )
    for name, path in sources.items():
        text = path.read_text()
        assert all(token not in text for token in retired), name

    runtime_tree = ast.parse(sources["runtime"].read_text())
    config = next(
        node
        for node in runtime_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CudaSchedulerConfig"
    )
    fields = {
        node.target.id
        for node in config.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "algorithm" not in fields

    launch = next(
        node
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "moe_rebalance_halo_q"
    )
    assert len(launch.args) == 26


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
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/rebalance_slot_scheduler_v2.py"
    namespace = {"torch": torch, "threading": SimpleNamespace(get_ident=lambda: state.thread)}
    group_cls = _extract_class(source, "RebalanceSlotSchedulerGroupV2", None, namespace)
    lease_cls = _extract_class(source, "_V2LiveBankLeaseProvider", None, namespace)
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
        {"_nvtxwrap__stage_inputs"},
        {"torch": SimpleNamespace(uint8="uint8")},
    )

    def run(stage_routes):
        trace = []
        owner = impl()
        owner._last_staged_T = {8: 8}
        owner._wait_rebalance_routes = lambda: trace.append("wait")
        bufs = SimpleNamespace(
            **{
                key: _StageTensor(key, trace)
                for key in ("topk_idx_local", "activation", "activation_sf", "topk_weights")
            }
        )
        owner._nvtxwrap__stage_inputs(
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


def test_warmup_owner_handoff_drains_once_and_deduplicates_groups():
    source = _TORCH_ROOT / "moe/fused_moe/mega_moe/rebalance_slot_scheduler_v2.py"
    executor_source = _TORCH_ROOT / "pyexecutor/py_executor.py"
    events = []
    main = SimpleNamespace(cuda_stream=11, priority=0, device=0)
    current = SimpleNamespace(stream=main, device=0)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            current_device=lambda: current.device,
            current_stream=lambda device: current.stream,
            synchronize=lambda device: events.append(("synchronize", device)),
        )
    )
    group_cls = _extract_class(
        source,
        "RebalanceSlotSchedulerGroupV2",
        {"has_submission_owner", "_check_owner", "release_warmup_owner_after_sync"},
        {"torch": torch, "threading": threading},
    )
    executor_cls = _extract_class(
        executor_source,
        "PyExecutor",
        {"_handoff_rebalance_warmup_owners"},
        {"torch": torch},
    )

    def make_group(bound=True):
        group = group_cls()
        group.device = 0
        group._owner_thread_id = None
        group._execution_stream_handle = None
        group._stream_handle = 99
        group.copy_stream = SimpleNamespace(priority=-1)
        group._plan_part = None
        group._generation = group._finished_generation = group.plan_calls = 3
        group._pending_release = (3, object())
        if bound:
            group._check_owner()
        return group

    target, draft, unbound = make_group(), make_group(), make_group(False)
    for label, group in (("target", target), ("draft", draft)):
        original = group.release_warmup_owner_after_sync

        def logged_release(handle, original=original, label=label):
            events.append(("release", label))
            original(handle)

        group.release_warmup_owner_after_sync = logged_release

    def engine(*groups):
        modules = [SimpleNamespace(_rebalance_scheduler_group=group) for group in groups]
        return SimpleNamespace(model=SimpleNamespace(modules=lambda: [*modules, SimpleNamespace()]))

    executor = executor_cls()
    executor.execution_stream = main
    executor.model_engine = engine(target, target, unbound)
    executor.draft_model_engine = engine(draft, target)
    executor._handoff_rebalance_warmup_owners()
    assert events == [("synchronize", 0), ("release", "target"), ("release", "draft")]
    assert not target.has_submission_owner and not draft.has_submission_owner
    assert not unbound.has_submission_owner

    events.clear()
    executor.model_engine = engine(unbound)
    executor.draft_model_engine = None
    executor._handoff_rebalance_warmup_owners()
    assert events == []


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
        "_os": os,
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


def test_on_autotune_uses_distinct_tactics_and_state_contract(monkeypatch):
    helpers = _autotune_helpers(None)
    monkeypatch.delenv("MEGAMOE_AUTOTUNE_PL_ALPHA", raising=False)
    monkeypatch.delenv("MEGAMOE_TOKEN_BACK_READY_GRANULARITY", raising=False)

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

    tuning_state = SimpleNamespace(is_tuning_mode=True)
    backend_cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/mega_moe/mega_moe_cute_dsl.py",
        "TrtllmCutedslMegaMoeNvfp4Impl",
        {"is_rebalance_active", "set_rebalance_warmup"},
        {"AutoTuner": SimpleNamespace(get=lambda: tuning_state)},
    )
    backend = backend_cls()
    backend._rebalance_slots_active = 4
    backend._rebalance_arm_open = True
    backend.set_rebalance_warmup(True)
    backend.tactic_autotune = True
    assert backend.is_rebalance_active()
    backend.tactic_autotune = False
    assert not backend.is_rebalance_active()
    tuning_state.is_tuning_mode = False
    backend.set_rebalance_warmup(False)
    assert backend.is_rebalance_active()
    backend.set_rebalance_warmup(True)
    assert not backend.is_rebalance_active()
    backend.set_rebalance_warmup(False)
    backend._rebalance_slots_active = 0
    assert not backend.is_rebalance_active()


def test_on_autotune_builds_balanced_powerlaw_input():
    torch = pytest.importorskip("torch")
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
