# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts of the HALO-R producer behind HALO-Q's scheduler-group contract.

HALO-R's group subclasses ``planner.RebalanceSlotSchedulerGroup`` and builds
its own copy endpoint. These tests build the group with fake native ops, CUDA
driver and scheduler buffers, mirror HALO-Q's group contract tests
(test_dynamic_eplb_contracts.py) on it, and pin the HALO-R native sources.
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tensorrt_llm._torch.moe.fused_moe import moe_comm_timeout_guard
from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import halo_r_planner, planner
from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.halo_r_planner import (
    HaloRRelayEndpoint,
    HaloRSlotSchedulerGroup,
    build_halo_r_slot_scheduler_group,
)
from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.planner import (
    RebalanceSlotSchedulerGroup,
    SchedulerSpinBound,
    _LiveBankLeaseProvider,
    build_rebalance_slot_scheduler_group,
)

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_KERNELS = _ROOT / "cpp/tensorrt_llm/kernels/moe/loadBalance/dynamicEplb"
_HALO_R_OP = _ROOT / "cpp/tensorrt_llm/thop/moe/loadBalance/moeRebalanceHaloROp.cpp"
_SCHEDULER_RUNTIME = (
    _ROOT / "tensorrt_llm/_torch/moe/kernels/megamoe_scheduler/cuda_scheduler/runtime.py"
)
_SCHEDULER_PACKAGE = "tensorrt_llm._torch.moe.kernels.megamoe_scheduler.cuda_scheduler"

_MAIN = 44
_OTHER_MAIN = 45
_COPY = 19
_HANDLE = 0x1234
_TOPK = 6
_MAX_TOKENS = 64
# The scheduler's native bound and that of the default 1800 s MoE communication
# warmup timeout at the nominal 2 GHz clock.
_SERVING_SPIN_CYCLES = 300 * 2_000_000_000
_WARMUP_SPIN_CYCLES = 1800 * 2_000_000_000


class _Routes:
    """Just enough of a CUDA int32 [tokens, topk] tensor for the group's checks."""

    def __init__(self, rows, *, dtype="int32", device="cuda:0"):
        self.shape = (rows, _TOPK)
        self.ndim = 2
        self.dtype = dtype
        self.device = device

    def is_contiguous(self):
        return True

    def get_device(self):
        return int(self.device.split(":")[1])

    def __getitem__(self, rows):
        return _Routes(rows.stop)


class _Tensor:
    _next_address = 0x10000

    def __init__(self, name, trace=None):
        self.name = name
        self.trace = trace
        _Tensor._next_address += 0x100
        self.address = _Tensor._next_address

    def data_ptr(self):
        return self.address

    def to(self, device):
        return self

    def zero_(self):
        self.trace.append(("zero", self.name))
        return self

    def fill_(self, value):
        self.trace.append(("fill", self.name, value))
        return self

    def view(self, dtype):
        return self


def _extract_function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _spin_cycles_for_timeout():
    """The scheduler runtime's conversion, run from its source without importing CUDA modules."""
    tree = ast.parse(_SCHEDULER_RUNTIME.read_text())
    namespace = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("SPIN_CYCLES", "ASSUMED_CLOCK_HZ")
    }
    return _extract_function(_SCHEDULER_RUNTIME, "spin_cycles_for_timeout", namespace)


class _Stream(SimpleNamespace):
    def __init__(self, handle, priority, trace):
        super().__init__(cuda_stream=handle, priority=priority, trace=trace)

    def wait_event(self, event):
        self.trace.append(("main_wait", self.cuda_stream, event.recorded_on))


class _Event:
    def __init__(self, trace):
        self.trace = trace
        self.recorded_on = None

    def record(self, stream):
        self.recorded_on = stream.cuda_stream
        self.trace.append(("main_record", stream.cuda_stream))


class _Env(SimpleNamespace):
    """A built HALO-R group with the fakes it runs on."""


def _build(
    monkeypatch, *, auxiliary_sms=4, ep_size=8, spin_bound=None, helper_residency=False
) -> _Env:
    trace = []
    state = SimpleNamespace(
        stream=_MAIN,
        device=0,
        fail_submit=False,
        counters=[0, 0, 0],
        capturing=False,
        peer_reset=None,
    )
    priorities = {_MAIN: 0, _OTHER_MAIN: 0, _COPY: -1}

    def call(name):
        def record(*args):
            trace.append((name, *args))
            return (0,)

        return record

    driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        CUstreamWaitValue_flags=SimpleNamespace(CU_STREAM_WAIT_VALUE_GEQ="geq"),
        CUdevice_attribute=SimpleNamespace(
            CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS="mem64"
        ),
        cuDeviceGet=lambda device: (0, device),
        cuDeviceGetAttribute=lambda attribute, device: (0, 1),
        cuStreamWaitValue64=call("wait_value"),
        cuStreamWaitEvent=call("wait_event"),
    )
    cuda_package = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    bindings.driver = driver
    cuda_package.bindings = bindings
    monkeypatch.setitem(sys.modules, "cuda", cuda_package)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)

    class Config:
        def __init__(self, **values):
            self.__dict__.update(values)
            self.route_count = self.max_tokens_per_rank * self.topk

        def validate(self):
            trace.append(("validate_config", self.ctas))

    class Scheduler:
        def __init__(self, cfg, device):
            self.cfg = cfg
            self.device = SimpleNamespace(index=int(device.split(":")[1]))
            self.outputs = SimpleNamespace(
                physical_slot_ids=_Routes(cfg.max_tokens_per_rank),
                hot_expert_ids="ids",
                hot_expert_group_level="levels",
                hot_expert_source_ranks="owners",
            )

        def connect_peer_bases(self, bases):
            trace.append(("connect", len(bases)))

        def bind_gpu_direct(self, stream):
            trace.append(("bind_gpu_direct", stream))

        def fused_launch_args(self):
            c = self.cfg
            names = ("slots", "ids", "levels", "owners", "peer", "status", "partial", "aux", "grid")
            return (
                *names,
                "plan",
                "prefix",
                c.ep_size,
                c.logical_expert_count,
                c.extra_slots_per_rank,
                c.local_rank,
                c.topk,
                c.route_count,
                c.ctas,
                c.threads,
            )

    class Symmetric:
        def __init__(self, nbytes, device):
            self.shareable = b"rank"
            self.ptr = 0x5000

        def import_peer(self, handle):
            return 0x6000

        def close(self):
            trace.append(("symmetric_close",))

    scheduler_package = ModuleType(_SCHEDULER_PACKAGE)
    scheduler_package.CudaPhysicalSlotScheduler = Scheduler
    scheduler_package.CudaSchedulerConfig = lambda **values: Config(threads=512, **values)
    multirank = ModuleType(f"{_SCHEDULER_PACKAGE}.multirank_dist")
    multirank.FabricSymmetricBuffer = Symmetric
    runtime = ModuleType(f"{_SCHEDULER_PACKAGE}.runtime")
    runtime.symmetric_buffer_ints = lambda ep, experts: 64 + 2 * ep * experts
    runtime.spin_cycles_for_timeout = _spin_cycles_for_timeout()
    for name in (
        "tensorrt_llm._torch.moe.kernels",
        "tensorrt_llm._torch.moe.kernels.megamoe_scheduler",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, _SCHEDULER_PACKAGE, scheduler_package)
    monkeypatch.setitem(sys.modules, multirank.__name__, multirank)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    def submit_fused(handle, routes, spin_cycles):
        trace.append(("submit", handle, routes.shape[0], spin_cycles))
        if state.fail_submit:
            raise RuntimeError("HALO-R kernel launch failed: unspecified launch failure")

    def create(*args):
        trace.append(("create", len(args)))
        state.create_args = args
        return SimpleNamespace(name="ctx")

    ops = SimpleNamespace(
        moe_rebalance_halo_r_create=create,
        moe_rebalance_halo_r_handle=lambda ctx: _HANDLE,
        moe_rebalance_halo_r_submit_fused=submit_fused,
        moe_rebalance_halo_r_wait_routes=lambda handle, stream: trace.append(
            ("route_wait", handle, stream)
        ),
        moe_rebalance_halo_r_join=lambda handle: trace.append(("join", handle, state.stream)),
        moe_rebalance_halo_r_relay_counts=lambda ctx: list(state.counters),
        moe_rebalance_halo_r_destroy=lambda ctx: trace.append(("destroy",)),
    )
    fake_torch = SimpleNamespace(
        Tensor=_Routes,
        int32="int32",
        int64="int64",
        zeros=lambda *shape, dtype, device: _Tensor(f"zeros{shape}", trace),
        full=lambda shape, value, dtype, device: _Tensor(f"full{shape}={value}", trace),
        tensor=lambda values, dtype: _Tensor("table", trace),
        ops=SimpleNamespace(trtllm=ops),
        _C=SimpleNamespace(
            _cuda_getDevice=lambda: state.device,
            _cuda_getCurrentRawStream=lambda device: state.stream,
        ),
        cuda=SimpleNamespace(
            Stream=object,
            Event=lambda: _Event(trace),
            current_stream=lambda device: _Stream(state.stream, priorities[state.stream], trace),
            is_current_stream_capturing=lambda: state.capturing,
            device=lambda device: nullcontext(),
            synchronize=lambda device: trace.append(("synchronize", device)),
        ),
    )
    monkeypatch.setattr(planner, "torch", fake_torch)
    monkeypatch.setattr(halo_r_planner, "torch", fake_torch)
    monkeypatch.setattr(
        halo_r_planner,
        "_relay_tables",
        lambda arena, pool, *, ep_rank, home_experts: (
            [1] * 8,
            [2] * ep_size,
            [512],
            0x8000,
            0x9000,
        ),
    )

    def allgather(value):
        trace.append(("allgather",))
        if isinstance(value, tuple):
            # A residency reset: every peer reports the same, unless a test makes rank 0 differ.
            return [state.peer_reset or value] + [value] * (ep_size - 1)
        return [bytes([r]) for r in range(ep_size)]

    ep_comm = SimpleNamespace(
        validate_geometry=lambda **kwargs: trace.append(("validate_geometry",)),
        allgather=allgather,
        barrier=lambda: trace.append(("barrier",)),
    )
    pool = SimpleNamespace(comm=ep_comm)
    ready_row = _Tensor("ready_row", trace)
    arena = SimpleNamespace(
        provider=SimpleNamespace(
            pool=pool, register_producer=lambda producer: trace.append(("register_producer",))
        ),
        terminal_flags_tensor=lambda: ready_row,
    )
    if spin_bound is None:
        spin_bound = SimpleNamespace(cycles=lambda: _SERVING_SPIN_CYCLES)
    copy = SimpleNamespace(cuda_stream=_COPY, priority=-1, device=SimpleNamespace(index=0))
    group = HaloRSlotSchedulerGroup(
        arena=arena,
        mapping=SimpleNamespace(moe_ep_size=ep_size, moe_ep_rank=1),
        ep_comm=ep_comm,
        device=0,
        home_experts=8,
        helper_slots=3,
        topk=_TOPK,
        max_tokens_per_rank=_MAX_TOKENS,
        copy_stream=copy,
        auxiliary_sms=auxiliary_sms,
        helper_residency=helper_residency,
        spin_bound=spin_bound,
        layer_idx=5,
    )
    return _Env(group=group, endpoint=group.broadcaster, trace=trace, state=state)


def _layer_call(group, rows, *, event=None):
    routes = _Routes(rows)
    physical, generation = group.plan_finish(group.plan_schedule(routes), defer_wait=True)
    group.wait_for_routes()
    group.finish(event)
    return physical, generation


def _self_attributes(cls, method):
    """Names that ``cls.method`` assigns on ``self``."""
    source = inspect.getsource(cls)
    tree = ast.parse(source)
    function = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method
    )
    names = set()
    for node in ast.walk(function):
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        for target in targets:
            for element in getattr(target, "elts", [target]):
                if (
                    isinstance(element, ast.Attribute)
                    and isinstance(element.value, ast.Name)
                    and element.value.id == "self"
                ):
                    names.add(element.attr)
    return names


def test_construction_follows_halo_q_collective_order_and_sets_its_attributes(monkeypatch):
    env = _build(monkeypatch)
    order = [entry[0] for entry in env.trace]
    expected = [
        "validate_geometry",
        "register_producer",
        "validate_config",
        "allgather",
        "connect",
        "barrier",
        "bind_gpu_direct",
        "create",
        "fill",
        "synchronize",
        "barrier",
    ]
    assert [name for name in order if name in expected] == expected
    # The READY row starts at 1, so the first launch drains it like every other.
    assert ("fill", "ready_row", 1) in env.trace
    # The scheduler runs on exactly the auxiliary SMs, as HALO-Q's.
    assert ("validate_config", 4) in env.trace
    # HALO-Q's methods read HALO-Q's private attributes: every one is set.
    missing = _self_attributes(RebalanceSlotSchedulerGroup, "__init__") - set(vars(env.group))
    assert not missing
    assert isinstance(env.group.lease, _LiveBankLeaseProvider)
    assert env.group.lease.bound_copy_module is env.endpoint
    assert env.endpoint._reuse_authority is env.group.lease
    assert env.group.scheduler_stream is env.group.copy_stream


@pytest.mark.parametrize("auxiliary_sms", [2, 4, 8, 16])
def test_constant_ready_and_route_target_with_the_route_wait_deferred(monkeypatch, auxiliary_sms):
    env = _build(monkeypatch, auxiliary_sms=auxiliary_sms)
    group, trace = env.group, env.trace
    lent = {}
    for generation, rows in enumerate((16, 16, 0), 1):
        trace.clear()
        routes = _Routes(rows)
        # The native route wait waits for the CTA count and resets the counter;
        # it takes no target from the host, so a replay reproduces it.
        route_wait = ("route_wait", _HANDLE, _MAIN)
        part = group.plan_schedule(routes)
        physical, ready = group.plan_finish(part, defer_wait=True)
        assert ready == 1
        assert group._generation == generation
        assert physical.shape == (rows, _TOPK)
        assert lent.setdefault(rows, physical) is physical
        assert route_wait not in trace
        # The group, not an allocator stream event, keeps the routes alive
        # until MAIN waits for the kernel's route release.
        assert group._routes_in_flight is routes
        group.wait_for_routes()
        group.wait_for_routes()
        assert group._routes_in_flight is None
        group.finish()
        assert trace.count(route_wait) == 1
        submit = trace.index(("submit", _HANDLE, rows, _SERVING_SPIN_CYCLES))
        assert submit < trace.index(route_wait) < trace.index(("join", _HANDLE, _MAIN))
        # The previous consumer is proven reusable before the next launch.
        assert env.endpoint._collective_safe_generation == generation - 1


def test_main_may_change_between_layer_calls_behind_a_fence(monkeypatch):
    env = _build(monkeypatch)
    group, trace, state = env.group, env.trace, env.state
    state.stream = _COPY
    with pytest.raises(RuntimeError, match="distinct from the higher-priority COPY"):
        group.plan_schedule(_Routes(4))
    state.stream = _MAIN
    _layer_call(group, 4)
    # A change between layer calls: one record on the old MAIN and one wait on
    # the new MAIN, both before the launch.
    trace.clear()
    state.stream = _OTHER_MAIN
    _layer_call(group, 4)
    fence = [entry for entry in trace if entry[0] in ("main_record", "main_wait")]
    assert fence == [("main_record", _MAIN), ("main_wait", _OTHER_MAIN, _MAIN)]
    submit = next(i for i, entry in enumerate(trace) if entry[0] == "submit")
    assert trace.index(("main_wait", _OTHER_MAIN, _MAIN)) < submit
    assert ("route_wait", _HANDLE, _OTHER_MAIN) in trace and ("join", _HANDLE, _OTHER_MAIN) in trace
    # Staying on a MAIN needs no fence; going back fences again.
    trace.clear()
    _layer_call(group, 4)
    assert not any(entry[0] in ("main_record", "main_wait") for entry in trace)
    state.stream = _MAIN
    _layer_call(group, 4)
    assert [entry for entry in trace if entry[0] == "main_wait"] == [
        ("main_wait", _MAIN, _OTHER_MAIN)
    ]
    # A change inside a layer call raises.
    part = group.plan_schedule(_Routes(4))
    state.stream = _OTHER_MAIN
    with pytest.raises(RuntimeError, match="must stay on one MAIN stream"):
        group.plan_finish(part)
    state.stream = _MAIN
    group.plan_finish(part)
    group.finish()
    # A change into a capturing stream raises; capturing on the bound MAIN does not.
    state.stream, state.capturing = _OTHER_MAIN, True
    with pytest.raises(RuntimeError, match="into a stream that is capturing"):
        group.plan_schedule(_Routes(4))
    state.stream = _MAIN
    _layer_call(group, 4)
    state.device = 1
    with pytest.raises(RuntimeError, match="bound CUDA device"):
        group.plan_schedule(_Routes(4))


@pytest.mark.parametrize(
    "environ, warmup_cycles, serving_cycles",
    [
        pytest.param({}, _WARMUP_SPIN_CYCLES, _SERVING_SPIN_CYCLES, id="defaults"),
        pytest.param(
            {"TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC": "600", "TRTLLM_MOE_COMM_TIMEOUT_SEC": "30"},
            600 * 2_000_000_000,
            30 * 2_000_000_000,
            id="configured",
        ),
    ],
)
def test_each_launch_takes_the_spin_bound_of_its_moment(
    monkeypatch, environ, warmup_cycles, serving_cycles
):
    guard = moe_comm_timeout_guard.MoECommTimeoutGuard(environ=environ, is_capturing=lambda: False)
    tuning = [False]
    env = _build(monkeypatch, spin_bound=SimpleNamespace(cycles=lambda: bound.cycles()))
    bound = SchedulerSpinBound(
        warmup_seconds=guard.budgets().warmup_seconds, is_tuning=lambda: tuning[0]
    )
    guard.register(bound)

    def launched_cycles():
        _layer_call(env.group, 4)
        return next(entry[3] for entry in reversed(env.trace) if entry[0] == "submit")

    assert launched_cycles() == serving_cycles
    guard.set_warmup(True)
    assert launched_cycles() == warmup_cycles
    # A captured launch keeps the serving bound, which every replay reuses.
    with guard.serving_timeouts():
        assert launched_cycles() == serving_cycles
    assert launched_cycles() == warmup_cycles
    guard.set_warmup(False)
    tuning[0] = True
    assert launched_cycles() == warmup_cycles


def test_finish_joins_copy_every_call_and_waits_for_a_caller_event(monkeypatch):
    env = _build(monkeypatch)
    _layer_call(env.group, 8)
    assert not any(entry[0] == "wait_event" for entry in env.trace)
    assert [entry for entry in env.trace if entry[0] == "join"] == [("join", _HANDLE, _MAIN)]
    env.trace.clear()
    _layer_call(env.group, 8, event=SimpleNamespace(cuda_event=0x99))
    assert [entry for entry in env.trace if entry[0] == "wait_event"] == [
        ("wait_event", _COPY, 0x99, 0)
    ]
    assert env.trace.index(("wait_event", _COPY, 0x99, 0)) < env.trace.index(
        ("join", _HANDLE, _MAIN)
    )
    # A discarded plan still waits for its routes and joins.
    env.trace.clear()
    env.group.discard_plan(env.group.plan_schedule(_Routes(8)))
    assert [entry[0] for entry in env.trace if entry[0] in ("route_wait", "join")] == [
        "route_wait",
        "join",
    ]
    with pytest.raises(ValueError, match="CUDA event"):
        _layer_call(env.group, 8, event=SimpleNamespace(cuda_event=0))


def test_failed_submit_holds_its_routes_and_aborts_the_endpoint(monkeypatch):
    env = _build(monkeypatch)
    _layer_call(env.group, 8)
    env.state.fail_submit = True
    stuck = _Routes(8)
    with pytest.raises(RuntimeError, match="launch failed"):
        env.group.plan_schedule(stuck)
    # The failed submission may have launched, so its routes stay alive.
    assert env.group._routes_in_flight is stuck
    env.state.fail_submit = False
    with pytest.raises(RuntimeError, match="never ordered on MAIN"):
        env.group.plan_schedule(_Routes(8))
    endpoint = env.endpoint
    for call in (
        lambda: endpoint.submit_fused(endpoint._outputs, _Routes(8), 0, spin_cycles=1),
        lambda: endpoint.wait_routes(_MAIN),
        endpoint.join,
        lambda: endpoint.release_generation_after(1),
        lambda: endpoint.mark_collective_reuse_safe(env.group.lease, 1),
        lambda: endpoint.bind_generation_reuse_authority(env.group.lease),
    ):
        with pytest.raises(RuntimeError, match="quiesced group rebuild required"):
            call()
    # Teardown still releases the native context.
    env.group.close()
    assert ("destroy",) in env.trace and ("symmetric_close",) in env.trace


def test_endpoint_refuses_a_launch_while_the_previous_generation_is_leased(monkeypatch):
    env = _build(monkeypatch)
    endpoint = env.endpoint
    assert endpoint.submit_fused(endpoint._outputs, _Routes(4), 0, spin_cycles=1) == 1
    with pytest.raises(RuntimeError, match="still leased by the previous generation"):
        endpoint.submit_fused(endpoint._outputs, _Routes(4), 0, spin_cycles=1)
    endpoint.release_generation_after(1)
    with pytest.raises(RuntimeError, match="still leased by the previous generation"):
        endpoint.submit_fused(endpoint._outputs, _Routes(4), 0, spin_cycles=1)
    with pytest.raises(PermissionError, match="identity mismatch"):
        endpoint.mark_collective_reuse_safe(object(), 1)
    endpoint.mark_collective_reuse_safe(env.group.lease, 1)
    assert endpoint.submit_fused(endpoint._outputs, _Routes(4), 0, spin_cycles=1) == 2
    # The kernel never reads a route target; the launch passes only routes and the bound.
    assert [entry[:3] for entry in env.trace if entry[0] == "submit"] == [
        ("submit", _HANDLE, 4)
    ] * 2


def test_endpoint_accepts_only_its_own_lease_once(monkeypatch):
    env = _build(monkeypatch)
    endpoint = env.endpoint
    with pytest.raises(RuntimeError, match="single-assignment"):
        endpoint.bind_generation_reuse_authority(env.group.lease)
    for flags in (
        {"collective_safe": False},
        {"producer_reuse_guarded": False},
        {"live_bank_count": 2},
        {"bound_copy_module": object()},
    ):
        authority = SimpleNamespace(
            collective_safe=True,
            producer_reuse_guarded=True,
            live_bank_count=1,
            bound_copy_module=endpoint,
        )
        authority.__dict__.update(flags)
        with pytest.raises(ValueError):
            endpoint._validate_reuse_authority(authority)


def test_relay_counts_come_from_the_native_context(monkeypatch):
    env = _build(monkeypatch)
    env.state.counters = [3, 12, 0]
    assert env.endpoint.relay_counts() == (3, 12, 0)


def test_group_overrides_construction_and_the_four_graph_hooks():
    methods = {
        name for name, value in vars(HaloRSlotSchedulerGroup).items() if inspect.isfunction(value)
    }
    overridden = {name for name in methods if hasattr(RebalanceSlotSchedulerGroup, name)}
    assert overridden == {"__init__", "_check_owner", "_wait_route_epoch", "plan_finish", "finish"}
    # The one addition is the weight-reload hook that the layer handle looks up.
    assert methods - overridden == {"forget_helper_residency"}
    assert inspect.signature(HaloRSlotSchedulerGroup.__init__) == inspect.signature(
        RebalanceSlotSchedulerGroup.__init__
    )
    for name, value in vars(RebalanceSlotSchedulerGroup).items():
        if inspect.isfunction(value) and (not name.startswith("_") or name in overridden):
            assert inspect.signature(getattr(HaloRSlotSchedulerGroup, name)) == inspect.signature(
                value
            )
    assert list(inspect.signature(build_halo_r_slot_scheduler_group).parameters) == list(
        inspect.signature(build_rebalance_slot_scheduler_group).parameters
    )


def test_endpoint_has_the_copy_surface_the_group_and_lease_call():
    for name in (
        "submit_fused",
        "release_generation_after",
        "mark_collective_reuse_safe",
        "bind_generation_reuse_authority",
        "close",
    ):
        assert callable(getattr(HaloRRelayEndpoint, name))
    assert HaloRRelayEndpoint.live_bank_count == 1


def _function_body(source: str, signature: str) -> str:
    """Text between the braces of the first function whose header contains ``signature``."""
    start = source.index("{", source.index(signature))
    depth = 0
    for index in range(start, len(source)):
        depth += {"{": 1, "}": -1}.get(source[index], 0)
        if depth == 0:
            return source[start : index + 1]
    raise ValueError(f"unbalanced body of {signature}")


def _assert_in_order(text: str, parts) -> None:
    positions = [text.index(part) for part in parts]
    assert positions == sorted(positions), parts


def _op_parameters(name: str) -> list:
    source = _HALO_R_OP.read_text()
    schemas = ["".join(re.findall(r'"([^"]*)"', block)) for block in source.split("m.def(")[1:]]
    schema = next(text for text in schemas if text.startswith(f"{name}("))
    parameters = re.fullmatch(rf"{name}\((.*)\) -> .*", schema).group(1)
    return [parameter.split()[-1] for parameter in parameters.split(",")]


def test_native_submit_orders_copy_after_main_and_poisons_on_any_cuda_error():
    assert _op_parameters("moe_rebalance_halo_r_submit_fused") == [
        "handle",
        "routes",
        "spin_cycles",
    ]
    source = _HALO_R_OP.read_text()
    submit = _function_body(source, "void submitFused(torch::Tensor const& routes")
    _assert_in_order(
        submit,
        (
            "requireOperational();",
            "TORCH_CHECK_VALUE(routes.is_cuda()",
            "cudaStream_t const execution = currentExecutionStream();",
            "poisonOnError(cudaEventRecord(mInputReady, execution)",
            "poisonOnError(cudaStreamWaitEvent(mCopyStream, mInputReady, 0)",
            "kernels::launchMoeRebalanceHaloR(mLaunch, params, mCopyStream, mLaunched)",
        ),
    )
    # Capture is allowed: every value the launch hands the device is constant.
    assert "cudaStreamIsCapturing" not in submit and "kReadyGeneration," in source
    # Every CUDA call of a launch goes through the sticky poison.
    calls = re.findall(r"(\w*)\(\s*(cuda\w+|kernels::launch\w+)\(", submit)
    assert calls and all(wrapper == "poisonOnError" for wrapper, _ in calls)
    poison = _function_body(source, "void poisonOnError(cudaError_t error")
    assert "mPoison = error;" in poison
    # The routes stay alive through the group's route hold, not an allocator stream.
    assert "recordStream" not in source
    # The launch is prepared when the context is bound.
    create = _function_body(source, "MoeRebalanceHaloRState(torch::Tensor outSlots")
    assert "mLaunch = kernels::prepareMoeRebalanceHaloRLaunch(ep, experts, ctas);" in create


def test_kernel_traps_before_it_releases_routes_and_before_ready():
    kernel_source = (_KERNELS / "moeRebalanceHaloR.cu").read_text()
    kernel = _function_body(kernel_source, "void haloRFusedKernel(")
    _assert_in_order(
        kernel,
        (
            "schedulerRole(p);",
            "trapLaunch(p, role.reason, role.detail);",
            "__syncthreads();",
            "firstFailedStatus(p.status)",
            'trapLaunch(p, "status word set before route release", word);',
            '"red.release.sys.global.add.u64 [%0], 1;" ::"l"(p.routeCounter)',
            "relayCopy(p)",
            "completeLaunch(p);",
        ),
    )
    # The release sits in the branch where every status word is clear.
    assert re.search(
        r"if \(word >= 0\)\s*\{[^}]*trapLaunch[^}]*\}\s*else\s*\{\s*asm volatile\(\"red\.release",
        kernel,
    )
    complete = _function_body(kernel_source, "void completeLaunch(")
    _assert_in_order(
        complete,
        (
            '"atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"',
            "firstFailedStatus(p.status)",
            'trapLaunch(p, "status word set before READY", failedWord);',
            "publishReady(p.readyFlagMc, p.readyValue);",
            "p.counters[kHaloRCounterLaunches] += 1;",
        ),
    )
    # Every scheduler wait that times out fails the launch instead of routing home.
    role = _function_body(kernel_source, "RoleResult mapRoutes(")
    assert role.count('return RoleResult{"') == 3
    assert "Home" not in kernel_source
    prepare = _function_body(
        kernel_source, "MoeRebalanceHaloRLaunch prepareMoeRebalanceHaloRLaunch("
    )
    assert (
        'TLLM_CHECK_WITH_INFO(active == 1, "HALO-R kernel must fit exactly one CTA per SM'
        in prepare
    )
    assert "cudaDeviceSetLimit(cudaLimitStackSize, attributes.localSizeBytes)" in prepare


def test_every_scheduler_role_failure_traps_the_launch():
    """Each wait of the scheduler role that times out, and the planner, fails the launch the same way.

    A failure that returned instead of trapping would leave MAIN waiting for
    a route release that never comes.
    """
    kernel_source = (_KERNELS / "moeRebalanceHaloR.cu").read_text()
    reasons = re.findall(r'return RoleResult\{"([^"]+)", [^}]+\};', kernel_source)
    assert sorted(reasons) == sorted(
        [
            "READY drain timeout",
            "count exchange timeout",
            "planner status",
            "worker rendezvous timeout",
            "plan wait timeout",
            "route prefix wait timeout",
            "grid rendezvous timeout",
        ]
    )
    # Every RoleResult reaches the kernel through schedulerRole, which returns its roles' results.
    assert re.findall(r"\b(planAndPublish|mapRoutes)\(p, epoch\)", kernel_source) == [
        "planAndPublish",
        "mapRoutes",
    ]
    role = _function_body(kernel_source, "RoleResult schedulerRole(")
    _assert_in_order(
        role,
        (
            'return RoleResult{"grid rendezvous timeout", 0};',
            "return blockIdx.x == 0 ? planAndPublish(p, epoch) : mapRoutes(p, epoch);",
        ),
    )
    assert kernel_source.count("schedulerRole(p)") == 1
    # The kernel traps on any failure and returns before the route release.
    kernel = _function_body(kernel_source, "void haloRFusedKernel(")
    assert re.search(
        r"if \(role\.reason != nullptr\)\s*\{\s*trapLaunch\(p, role\.reason, role\.detail\);\s*return;\s*\}",
        kernel,
    )
    trap = _function_body(kernel_source, "void trapLaunch(")
    assert 'asm volatile("trap;" ::: "memory");' in trap


def test_no_timing_probe_or_programmatic_launch_remains():
    sources = [
        _KERNELS / "moeRebalanceHaloR.cu",
        _KERNELS / "moeRebalanceHaloR.h",
        _KERNELS / "moeRebalanceRelay.cuh",
        _HALO_R_OP,
        Path(halo_r_planner.__file__),
    ]
    for path in sources:
        text = path.read_text()
        for marker in (
            "kHaloRProbe",
            "set_probe",
            "griddepcontrol",
            "TRTLLM_HALO_R_RELAY_PDL",
            "%globaltimer",
        ):
            assert marker not in text, (path.name, marker)


def test_kernel_runs_in_the_remote_memory_sync_domain():
    kernel_source = (_KERNELS / "moeRebalanceHaloR.cu").read_text()
    prepare = _function_body(
        kernel_source, "MoeRebalanceHaloRLaunch prepareMoeRebalanceHaloRLaunch("
    )
    _assert_in_order(
        prepare,
        (
            "cudaDevAttrMemSyncDomainCount",
            "launch.attributes[0].val.memSyncDomain = cudaLaunchMemSyncDomainRemote;",
            "launch.attributes[1].id = cudaLaunchAttributeMemSyncDomainMap;",
            "launch.attributeCount = 2;",
        ),
    )
    launch = _function_body(kernel_source, "cudaError_t launchMoeRebalanceHaloR(")
    # The prepared attributes, then the launch completion event that MAIN's route wait waits for.
    _assert_in_order(
        launch,
        (
            "std::copy_n(launch.attributes, launch.attributeCount, attributes);",
            "completion.id = cudaLaunchAttributeLaunchCompletionEvent;",
            "completion.val.launchCompletionEvent.event = launched;",
            "config.numAttrs = launch.attributeCount + 1;",
        ),
    )
    # Every edge out of the remote domain is system scope.
    assert '"red.release.sys.global.add.u64 [%0], 1;"' in kernel_source
    publish = _function_body(kernel_source, "void publishReady(")
    _assert_in_order(
        publish, ('"fence.release.sys;"', '"multimem.st.release.sys.global.u64 [%0], %1;"')
    )


def _create_argument(env, name):
    """The value a fake native create received for the schema parameter ``name``."""
    names = _op_parameters("moe_rebalance_halo_r_create")
    assert len(names) == len(env.state.create_args)
    return env.state.create_args[names.index(name)]


@pytest.mark.parametrize("helper_residency", [False, True])
def test_residency_table_exists_only_for_per_layer_helper_slots(monkeypatch, helper_residency):
    env = _build(monkeypatch, helper_residency=helper_residency)
    table = _create_argument(env, "residency")
    if not helper_residency:
        assert table is None and env.endpoint.residency is None
        return
    # One int32 [ep][helpers] table per layer, unknown (-1) everywhere, before any launch.
    assert table is env.endpoint.residency
    assert table.name == "full(8, 3)=-1"
    assert env.trace.index(("synchronize", 0)) < env.trace.index(
        ("create", len(env.state.create_args))
    )


def test_kernel_skips_resident_receivers_and_records_residency_before_ready():
    kernel_source = (_KERNELS / "moeRebalanceHaloR.cu").read_text()
    relay_source = (_KERNELS / "moeRebalanceRelay.cuh").read_text()
    prepare = _function_body(relay_source, "void prepareRelayWork(")
    _assert_in_order(
        prepare,
        (
            "halo_r::relayNeedMask(",
            "p.residency, p.helpers);",
            "halo_r::relayDestinationIndex(need, p.localRank, &destinations);",
            "work->need[entries] = need;",
        ),
    )
    issue = _function_body(relay_source, "void relayIssue(")
    # Loads and stores go to the need set, never to the plan's whole receiver mask.
    assert "work.need[" in issue and "kDestinationMask" not in issue
    account = _function_body(kernel_source, "void accountBroadcasts(")
    _assert_in_order(
        account,
        (
            "halo_r::relayNeedMask(mask, low, high, experts[b], p.residency, p.helpers);",
            "atomicAdd(receiverCopies",
            "atomicAdd(receiverSkips",
            "p.residency[d * p.helpers + halo_r::helperIndex(low, high, d)] = experts[b];",
        ),
    )
    complete = _function_body(kernel_source, "void completeLaunch(")
    _assert_in_order(
        complete,
        (
            '"atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"',
            "firstFailedStatus(p.status)",
            "accountBroadcasts(p, &receiverCopies, &receiverSkips);",
            "publishReady(p.readyFlagMc, p.readyValue);",
            "p.counters[kHaloRCounterReceiverSkips] += ",
        ),
    )
    # Every thread's table change precedes thread 0's READY release.
    barrier = complete.index("__syncthreads();", complete.index("accountBroadcasts("))
    assert barrier < complete.index("publishReady(")


class _FakeVmm:
    def __init__(self, trace):
        self.trace = trace
        self.next_address = 0x100000

    def reserve(self, nbytes):
        self.next_address += 0x10000
        self.trace.append(("reserve", nbytes))
        return self.next_address

    def import_(self, shareable):
        self.trace.append(("import", shareable))
        return f"handle:{shareable}"

    def map(self, address, nbytes, handle):
        self.trace.append(("map", address, nbytes, handle))


def _pool(trace, *, rank, sets, gathered):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.arena import (
        _HelperSet,
        _SharedSlotPool,
    )

    pool = _SharedSlotPool.__new__(_SharedSlotPool)
    pool.world = len(gathered[0]) if gathered else 4
    pool.rank = rank
    pool._closed = False
    pool._peer_helper_bases = {}
    pool.large = (SimpleNamespace(name="w13"), SimpleNamespace(name="w2"))
    pool.helper_bytes = {"w13": 4096, "w2": 2048}
    pool.sets = tuple(
        _HelperSet({}, (), (), shareables={"w13": f"s{index}-w13", "w2": f"s{index}-w2"})
        for index in range(sets)
    )
    pool.vmm = _FakeVmm(trace)

    def allgather(local):
        trace.append(("allgather", dict(local)))
        return gathered.pop(0)

    pool.comm = SimpleNamespace(allgather=allgather)
    return pool


@pytest.mark.parametrize("sets", [2, 5], ids=["shared-banks", "per-layer"])
def test_peer_helper_chunks_are_mapped_once_per_set(sets):
    trace = []
    world = 4

    def peers(index):
        return [{"w13": f"r{r}-s{index}-w13", "w2": f"r{r}-s{index}-w2"} for r in range(world)]

    pool = _pool(trace, rank=1, sets=sets, gathered=[peers(index) for index in range(sets)])
    for helper_set in range(sets):
        bases = pool.peer_helper_bases(helper_set)
        # Every peer but this rank, every large plane, the peer's own chunk of this set.
        assert set(bases) == {(peer, name) for peer in (0, 2, 3) for name in ("w13", "w2")}
        assert trace[-1][0] == "map" and ("allgather", pool.sets[helper_set].shareables) in trace
        imported = [entry[1] for entry in trace if entry[0] == "import"]
        assert f"r3-s{helper_set}-w2" in imported and f"r1-s{helper_set}-w2" not in imported
    # A repeat is served from the cache, with no collective.
    count = len(trace)
    assert pool.peer_helper_bases(0) is pool.peer_helper_bases(0)
    assert len(trace) == count
    assert sum(1 for entry in trace if entry[0] == "allgather") == sets
    with pytest.raises(ValueError, match="not one of the pool"):
        pool.peer_helper_bases(sets)
    pool._closed = True
    with pytest.raises(RuntimeError, match="closed shared-slot pool"):
        pool.peer_helper_bases(0)


def test_peer_helper_mapping_refuses_mismatched_chunks():
    trace = []
    gathered = [
        [{"w13": "a", "w2": "b"}, {"w13": "c"}, {"w13": "e", "w2": "f"}, {"w13": "g", "w2": "h"}]
    ]
    pool = _pool(trace, rank=0, sets=1, gathered=gathered)
    with pytest.raises(RuntimeError, match="chunks differ"):
        pool.peer_helper_bases(0)
    pool = _pool(trace, rank=0, sets=1, gathered=[[{"w13": "a", "w2": "b"}] * 3])
    pool.world = 4
    with pytest.raises(RuntimeError, match="incomplete EP"):
        pool.peer_helper_bases(0)


class _FakeChunkVmm(_FakeVmm):
    def create(self, nbytes):
        self.trace.append(("create", nbytes))
        index = sum(1 for entry in self.trace if entry[0] == "create")
        return f"chunk{index}", f"shareable{index}"


def _set_pool(trace, *, helper_multicast):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.arena import _SharedSlotPool

    pool = _SharedSlotPool.__new__(_SharedSlotPool)
    pool.rank = 5
    pool.home_count = 4
    # EP 8: the HALO levels are the whole world and this rank's aligned half.
    pool.groups = (tuple(range(8)), (4, 5, 6, 7))
    pool.large = (SimpleNamespace(name="w13", nbytes=1024), SimpleNamespace(name="w2", nbytes=512))
    pool.home_bytes = {"w13": 8192, "w2": 4096}
    pool.helper_bytes = {"w13": 4096, "w2": 2048}
    pool.mc_offsets = {"w13": 0, "w2": 4096}
    pool.set_bytes = 6144
    pool.helper_multicast = helper_multicast
    pool.vmm = _FakeChunkVmm(trace)
    return pool


@pytest.mark.parametrize("count", [1, 2], ids=["per-layer", "shared-banks"])
def test_halo_q_helper_sets_keep_a_multicast_object_per_level(monkeypatch, count):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import arena as arena_module

    calls = []
    monkeypatch.setattr(
        arena_module,
        "_create_multicast",
        lambda vmm, comm, rank, specs: calls.append((rank, specs)),
    )
    pool = _set_pool([], helper_multicast=True)
    sets = pool._create_sets(count, comm="comm")
    # One collective creation per call: every set at every level, set-major.
    assert len(calls) == 1 and calls[0][0] == 5
    specs = calls[0][1]
    assert [spec.members for spec in specs] == list(pool.groups) * count
    for index, helper_set in enumerate(sets):
        assert helper_set.level_specs == tuple(specs[2 * index : 2 * index + 2])
        for bases, spec in zip(helper_set.level_bases, helper_set.level_specs, strict=True):
            assert spec.nbytes == pool.set_bytes
            assert spec.binds == [
                (0, helper_set.chunks["w13"], 4096),
                (4096, helper_set.chunks["w2"], 2048),
            ]
            # Row 0 of each plane's view sits H rows before the plane's chunk.
            assert bases == {
                "w13": spec.mappings[0] - 4 * 1024,
                "w2": spec.mappings[1] + 4096 - 4 * 512,
            }


@pytest.mark.parametrize("count", [1, 2], ids=["per-layer", "shared-banks"])
def test_halo_r_helper_sets_get_no_multicast_object(monkeypatch, count):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import arena as arena_module

    def create_multicast(*args):
        raise AssertionError("HALO-R helper sets must not create multicast objects")

    monkeypatch.setattr(arena_module, "_create_multicast", create_multicast)
    trace = []
    pool = _set_pool(trace, helper_multicast=False)
    sets = pool._create_sets(count, comm="comm")
    # Only the chunks: no multicast object and no view reservation for one.
    assert trace == [("create", 4096), ("create", 2048)] * count
    for helper_set in sets:
        assert helper_set.level_bases == () and helper_set.level_specs == ()
        assert set(helper_set.chunks) == set(helper_set.shareables) == {"w13", "w2"}


def _cuda_view(shape, stride, dtype, element_size, pointer):
    return SimpleNamespace(
        data_ptr=lambda: pointer,
        shape=shape,
        stride=lambda: stride,
        dtype=dtype,
        element_size=lambda: element_size,
        is_cuda=True,
        device="cuda:0",
    )


def _live_arena(world, slots, unicast_planes):
    from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami.arena import (
        LivePlaneView,
        LiveTerminalView,
        LiveWeightArena,
    )
    from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami.geometry import BundleLayout

    bundle = BundleLayout.create(hidden=256, intermediate=128)
    addresses = iter(range(1 << 32, 1 << 40, 1 << 28))

    def view(layout):
        shape, stride, dtype, element_size, _ = layout.kernel_metadata(slots)
        return _cuda_view(shape, stride, dtype, element_size, next(addresses))

    planes = tuple(
        LivePlaneView(
            name=layout.name,
            uc_views=tuple(view(layout) for _ in range(world)),
            mc_view=None if index < unicast_planes else view(layout),
            aliases_same_backing=True,
            backing_owner=object(),
        )
        for index, layout in enumerate(bundle.planes)
    )
    terminals = LiveTerminalView(
        local_view=_cuda_view((world,), (1,), "torch.uint64", 8, next(addresses)),
        mc_view=_cuda_view((world,), (1,), "torch.uint64", 8, next(addresses)),
        aliases_same_backing=True,
        backing_owner=object(),
    )
    return LiveWeightArena(planes=planes, terminals=terminals), bundle


@pytest.mark.parametrize("unicast_planes", [0, 2], ids=["multicast", "unicast-helpers"])
def test_live_arena_binds_planes_without_a_multicast_alias(unicast_planes):
    world, rank, home, helpers = 2, 1, 4, 2
    arena, bundle = _live_arena(world, home + helpers, unicast_planes)
    bound = arena.bind(
        world=world, rank=rank, home_count=home, helper_count=helpers, bundle=bundle, device=0
    )
    # The unicast copy reads every expert from its owner's view either way.
    assert len(bound.scatter_source_ptrs) == world * home * len(bundle.planes)
    assert bound.local_plane_views == tuple(plane.uc_views[rank] for plane in arena.planes)
    # Multicast destinations, which only the HALO-Q copy uses, need every alias.
    expected = 0 if unicast_planes else helpers * len(bundle.planes)
    assert len(bound.destination_ptrs) == expected


def test_hierarchical_arena_refuses_planes_without_a_multicast_alias():
    from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami import (
        HierarchicalLiveWeightArena,
        hierarchy_group_sizes,
    )

    world, home, helpers = 2, 4, 2
    base, bundle = _live_arena(world, home + helpers, unicast_planes=2)
    arena = HierarchicalLiveWeightArena(
        base=base,
        group_sizes=hierarchy_group_sizes(world),
        group_mc_views=tuple(() if p.mc_view is None else (p.mc_view,) for p in base.planes),
        group_mc_owners=tuple(() if p.mc_view is None else (object(),) for p in base.planes),
        aliases_same_backing=True,
    )
    # The HALO-Q copy cannot bind an arena whose helper rows have no multicast alias.
    with pytest.raises(ValueError, match="hierarchy depth"):
        arena.bind(
            world=world, rank=0, home_count=home, helper_count=helpers, bundle=bundle, device=0
        )


def test_weight_reload_forgets_residency_between_layer_calls(monkeypatch):
    env = _build(monkeypatch, helper_residency=True)
    table = env.endpoint.residency
    group = env.group
    _layer_call(group, 8)
    part = group.plan_schedule(_Routes(8))
    with pytest.raises(RuntimeError, match="only between layer calls"):
        group.forget_helper_residency()
    group.plan_finish(part)
    with pytest.raises(RuntimeError, match="only between layer calls"):
        group.forget_helper_residency()
    group.finish()
    env.trace.clear()
    group.forget_helper_residency()
    # Every rank resets the same layer for the same time; no launch before the
    # reset can still update the table, and every later one reads the reset.
    assert env.trace == [
        ("allgather",),
        ("synchronize", 0),
        ("fill", table.name, -1),
        ("synchronize", 0),
    ]
    # The reset does not disturb the generations: the next call proceeds as usual.
    _layer_call(group, 8)
    assert group._generation == 3


@pytest.mark.parametrize("peer_reset", [(5, 2), (6, 1), (None, 1)])
def test_weight_reload_raises_when_ranks_reset_differently(monkeypatch, peer_reset):
    """A rank that reset another layer, or this one another number of times, fails every rank."""
    env = _build(monkeypatch, helper_residency=True)
    _layer_call(env.group, 8)
    env.state.peer_reset = peer_reset
    env.trace.clear()
    with pytest.raises(RuntimeError, match="resets differ across EP ranks"):
        env.group.forget_helper_residency()
    # The table is left as it was, for no rank may reset alone.
    assert env.trace == [("allgather",)]


def test_weight_reload_without_residency_is_free(monkeypatch):
    env = _build(monkeypatch)
    _layer_call(env.group, 8)
    env.trace.clear()
    env.group.forget_helper_residency()
    assert env.trace == []
    env.state.fail_submit = True
    with pytest.raises(RuntimeError, match="launch failed"):
        env.group.plan_schedule(_Routes(8))
    # An aborted group is rebuilt, not reset.
    with pytest.raises(RuntimeError, match="quiesced group rebuild required"):
        env.group.forget_helper_residency()


def test_planner_reads_the_residency_table_from_shared_memory():
    kernel_source = (_KERNELS / "moeRebalanceHaloR.cu").read_text()
    publish = _function_body(kernel_source, "RoleResult planAndPublish(")
    _assert_in_order(
        publish,
        (
            "__shared__ int resident[halo_r::kMaxRanks * halo_r::kMaxHelperSlots];",
            "resident[entry] = p.residency[entry];",
            "p.residency != nullptr ? resident : nullptr",
        ),
    )
    load = publish.index("resident[entry] = p.residency[entry];")
    assert publish.index("__syncthreads();", load) < publish.index("? resident : nullptr")
    planner = (_KERNELS / "moeRebalanceHaloRPlanner.h").read_text()
    build = _function_body(planner, "inline PlanStatus buildHelperTable(")
    # Resident experts are placed before the others fill the free slots.
    _assert_in_order(build, ("row[q] == received[r]", "placed[r] = true;", "while (hot[q] >= 0)"))


def test_native_route_wait_and_join_are_constant_and_poisoned_together():
    assert _op_parameters("moe_rebalance_halo_r_wait_routes") == ["handle", "stream"]
    assert _op_parameters("moe_rebalance_halo_r_join") == ["handle"]
    source = _HALO_R_OP.read_text()
    wait = _function_body(source, "void waitRoutes(std::int64_t streamHandle)")
    _assert_in_order(
        wait,
        (
            "requireOperational();",
            # The launch completion event is the CUDA-visible edge from the kernel to the wait.
            "poisonOnError(cudaStreamWaitEvent(stream, mLaunched, 0)",
            "operations[0].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;",
            "operations[0].waitValue.value64 = static_cast<cuuint64_t>(mLaunch.ctas);",
            "operations[0].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;",
            "operations[1].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;",
            "operations[1].writeValue.value64 = 0;",
            "cuStreamBatchMemOp(reinterpret_cast<CUstream>(stream), 2, operations, 0)",
            "mPoison = ",
        ),
    )
    join = _function_body(source, "void join()")
    _assert_in_order(
        join,
        (
            "requireOperational();",
            "cudaStream_t const execution = currentExecutionStream();",
            "poisonOnError(cudaEventRecord(mCopyIdle, mCopyStream)",
            "poisonOnError(cudaStreamWaitEvent(execution, mCopyIdle, 0)",
        ),
    )
    operational = _function_body(source, "void requireOperational() const")
    assert "mPoison == cudaSuccess" in operational


def test_kernel_drains_ready_before_the_exchange_within_one_bound():
    kernel_source = (_KERNELS / "moeRebalanceHaloR.cu").read_text()
    publish = _function_body(kernel_source, "RoleResult planAndPublish(")
    _assert_in_order(
        publish,
        (
            "long long const start = clock64();",
            "missingReady = drainReadyRow(p.readyFlags, p.ep, start, p.spinCycles);",
            "exchangeCycles = elapsed < p.spinCycles ? p.spinCycles - elapsed : 1;",
            'return RoleResult{"READY drain timeout", missingReady};',
            "exchange_counts(",
            "epoch, exchangeCycles, p.status)",
        ),
    )
    # Every launch drains, whatever its epoch: the row starts at 1, and the epoch wraps.
    assert "epoch > 1" not in publish
    drain = _function_body(kernel_source, "int drainReadyRow(")
    _assert_in_order(
        drain,
        (
            '"ld.relaxed.sys.global.u64 %0, [%1];"',
            "return source;",
            '"fence.acq_rel.sys;"',
            '"st.relaxed.sys.global.u64 [%0], %1;"',
            "return -1;",
        ),
    )
    # The reset is fenced on both sides.
    reset = drain.index('"st.relaxed.sys.global.u64 [%0], %1;"')
    assert (
        drain.count('"fence.acq_rel.sys;"') == 2
        and drain.index('"fence.acq_rel.sys;"', reset) > reset
    )
