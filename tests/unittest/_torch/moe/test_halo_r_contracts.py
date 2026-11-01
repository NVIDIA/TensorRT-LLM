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


class _Env(SimpleNamespace):
    """A built HALO-R group with the fakes it runs on."""


def _build(monkeypatch, *, auxiliary_sms=4, ep_size=8, spin_bound=None) -> _Env:
    trace = []
    state = SimpleNamespace(stream=_MAIN, device=0, fail_submit=False, counters=[0, 0, 0])
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
        return SimpleNamespace(name="ctx")

    ops = SimpleNamespace(
        moe_rebalance_halo_r_create=create,
        moe_rebalance_halo_r_handle=lambda ctx: _HANDLE,
        moe_rebalance_halo_r_submit_fused=submit_fused,
        moe_rebalance_halo_r_relay_counts=lambda ctx: list(state.counters),
        moe_rebalance_halo_r_destroy=lambda ctx: trace.append(("destroy",)),
    )
    fake_torch = SimpleNamespace(
        Tensor=_Routes,
        int32="int32",
        int64="int64",
        zeros=lambda *shape, dtype, device: _Tensor(f"zeros{shape}", trace),
        tensor=lambda values, dtype: _Tensor("table", trace),
        ops=SimpleNamespace(trtllm=ops),
        _C=SimpleNamespace(
            _cuda_getDevice=lambda: state.device,
            _cuda_getCurrentRawStream=lambda device: state.stream,
        ),
        cuda=SimpleNamespace(
            Stream=object,
            current_stream=lambda device: SimpleNamespace(
                cuda_stream=state.stream, priority=priorities[state.stream]
            ),
            device=lambda device: nullcontext(),
            synchronize=lambda device: trace.append(("synchronize", device)),
        ),
    )
    monkeypatch.setattr(planner, "torch", fake_torch)
    monkeypatch.setattr(halo_r_planner, "torch", fake_torch)
    monkeypatch.setattr(
        halo_r_planner,
        "_relay_tables",
        lambda arena, pool, *, ep_rank, home_experts: ([1] * 8, [2] * ep_size, [512], 0x9000),
    )

    ep_comm = SimpleNamespace(
        validate_geometry=lambda **kwargs: trace.append(("validate_geometry",)),
        allgather=lambda value: trace.append(("allgather",))
        or [bytes([r]) for r in range(ep_size)],
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
        helper_residency=False,
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
        "zero",
        "synchronize",
        "barrier",
    ]
    assert [name for name in order if name in expected] == expected
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
def test_generations_accumulate_the_route_target_and_defer_the_route_wait(
    monkeypatch, auxiliary_sms
):
    env = _build(monkeypatch, auxiliary_sms=auxiliary_sms)
    group, trace = env.group, env.trace
    lent = {}
    for generation, rows in enumerate((16, 16, 0), 1):
        trace.clear()
        routes = _Routes(rows)
        route_wait = (
            "wait_value",
            _MAIN,
            group._route_epoch_ptr,
            generation * auxiliary_sms,
            "geq",
        )
        part = group.plan_schedule(routes)
        physical, actual = group.plan_finish(part, defer_wait=True)
        assert actual == generation
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
        assert trace.index(("submit", _HANDLE, rows, _SERVING_SPIN_CYCLES)) < trace.index(
            route_wait
        )
        # Without an event, finish queues no device work.
        assert not any(entry[0] == "wait_event" for entry in trace)
        # The previous consumer is proven reusable before the next launch.
        assert env.endpoint._collective_safe_generation == generation - 1


def test_main_stream_is_bound_once_and_copy_cannot_be_main(monkeypatch):
    env = _build(monkeypatch)
    env.state.stream = _COPY
    with pytest.raises(RuntimeError, match="distinct from the higher-priority COPY"):
        env.group.plan_schedule(_Routes(4))
    env.state.stream = _MAIN
    _layer_call(env.group, 4)
    env.state.stream = _OTHER_MAIN
    with pytest.raises(RuntimeError, match="MAIN stream cannot change"):
        env.group.plan_schedule(_Routes(4))
    env.state.device = 1
    with pytest.raises(RuntimeError, match="bound CUDA device"):
        env.group.plan_schedule(_Routes(4))


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


def test_finish_waits_on_copy_only_for_a_caller_event(monkeypatch):
    env = _build(monkeypatch)
    _layer_call(env.group, 8)
    assert not any(entry[0] == "wait_event" for entry in env.trace)
    env.trace.clear()
    _layer_call(env.group, 8, event=SimpleNamespace(cuda_event=0x99))
    assert [entry for entry in env.trace if entry[0] == "wait_event"] == [
        ("wait_event", _COPY, 0x99, 0)
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


def test_group_overrides_only_construction():
    overridden = {
        name for name, value in vars(HaloRSlotSchedulerGroup).items() if inspect.isfunction(value)
    }
    assert overridden == {"__init__"}
    assert inspect.signature(HaloRSlotSchedulerGroup.__init__) == inspect.signature(
        RebalanceSlotSchedulerGroup.__init__
    )
    for name, value in vars(RebalanceSlotSchedulerGroup).items():
        if inspect.isfunction(value) and not name.startswith("_"):
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
            "TORCH_CHECK(mPoison == cudaSuccess",
            "TORCH_CHECK_VALUE(routes.is_cuda()",
            "cudaStream_t const execution = currentExecutionStream();",
            "poisonOnError(cudaStreamIsCapturing(execution, &capture)",
            "TORCH_CHECK(capture == cudaStreamCaptureStatusNone",
            "poisonOnError(cudaEventRecord(mInputReady, execution)",
            "poisonOnError(cudaStreamWaitEvent(mCopyStream, mInputReady, 0)",
            "poisonOnError(kernels::launchMoeRebalanceHaloR(mLaunch, params, mCopyStream)",
            "++mGeneration;",
        ),
    )
    # Every CUDA call of a launch goes through the sticky poison.
    calls = re.findall(r"(\w*)\((cuda\w+|kernels::launch\w+)\(", submit)
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
            'trapLaunch(p, "status word set before READY", word);',
            "publishReady(p.readyFlagMc, p.readyValue);",
            "countLaunch(p);",
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
    assert "config.numAttrs = launch.attributeCount;" in launch
    # Every edge out of the remote domain is system scope.
    assert '"red.release.sys.global.add.u64 [%0], 1;"' in kernel_source
    publish = _function_body(kernel_source, "void publishReady(")
    _assert_in_order(
        publish, ('"fence.release.sys;"', '"multimem.st.release.sys.global.u64 [%0], %1;"')
    )
