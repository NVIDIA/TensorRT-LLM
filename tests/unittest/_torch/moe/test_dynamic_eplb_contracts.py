# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast CPU contracts for the per-layer dynamic EPLB serving path.

Most of these come from the upstream DynamicEPLB contract suite, restricted to
the modules this tree carries: the shared-slot arena, the HALO-Q + TMA planner
group, the SAMI copy endpoint, the scheduler runtime, the native operator
sources and the EPLB lifecycle. They execute the real source of one class or
function against stand-ins, so they need neither a GPU nor the native ops.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import re
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_TORCH_ROOT = _ROOT / "tensorrt_llm/_torch"
_SCHEDULER_RUNTIME = _TORCH_ROOT / "moe/kernels/megamoe_scheduler/cuda_scheduler/runtime.py"
_DYNAMIC_EPLB_KERNELS = _ROOT / "cpp/tensorrt_llm/kernels/moe/loadBalance/dynamicEplb"
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


def test_native_expert_bound_is_shared_by_the_kernels_and_the_python_scheduler():
    # The plan field stride, the broadcast capacity and the HALO-Q shared
    # memory all follow one bound, which must cover Kimi K3.1's 896 experts.
    plan_header = (_DYNAMIC_EPLB_KERNELS / "moeRebalanceTmaGpuPlan.h").read_text()
    native = int(re.search(r"#define MEGAMOE_TMA_GPU_PLAN_MAX_EXPERTS (\d+)", plan_header).group(1))
    python = _module_constants(_SCHEDULER_RUNTIME, {"MAX_EXPERTS"})["MAX_EXPERTS"]
    assert native == python == 1024
    assert (
        "kMoeRebalanceMaxExperts = MEGAMOE_TMA_GPU_PLAN_MAX_EXPERTS"
        in (_DYNAMIC_EPLB_KERNELS / "moeRebalanceHaloQ.h").read_text()
    )
    assert (
        "kMaxExperts = kMoeRebalanceMaxExperts"
        in (_DYNAMIC_EPLB_KERNELS / "moeRebalanceHaloQKernel.cuh").read_text()
    )
    # No source keeps the former fixed 384-expert bound.
    sources = [
        *_DYNAMIC_EPLB_KERNELS.iterdir(),
        *(_ROOT / "cpp/tensorrt_llm/thop/moe/loadBalance").iterdir(),
        *(_TORCH_ROOT / "moe/kernels/megamoe_scheduler").rglob("*.py"),
    ]
    for path in sources:
        if path.is_file():
            assert not re.search(r"\b384\b", path.read_text()), path.name


@pytest.mark.parametrize(
    ("experts", "helpers", "accepted"),
    [
        pytest.param(896, 8, True, id="kimi-k31-ep16-s8"),
        pytest.param(1024, 4, True, id="bound"),
        pytest.param(1040, 4, False, id="above-bound"),
    ],
)
def test_scheduler_config_accepts_kimi_k31_geometry(experts, helpers, accepted):
    namespace = {
        "torch": None,
        "MIN_EP": 2,
        "MAX_EP": 32,
        "hierarchy_group_sizes": lambda ep_size: (ep_size,),
        "dataclass": dataclasses.dataclass,
    }
    namespace.update(_module_constants(_SCHEDULER_RUNTIME, {"MAX_EXPERTS", "MAX_CTAS"}))
    _extract_function(_SCHEDULER_RUNTIME, "recommend_cuda_scheduler_ctas", namespace)
    module_name = "cuda_scheduler_config_under_test"
    module = ModuleType(module_name)
    module.__dict__.update(namespace)
    # The class keeps its frozen dataclass decorator, and dataclasses resolve
    # their module through sys.modules.
    sys.modules[module_name] = module
    try:
        config_cls = _extract_class(
            _SCHEDULER_RUNTIME, "CudaSchedulerConfig", None, module.__dict__
        )

        def build():
            return config_cls(
                ep_size=16,
                logical_expert_count=experts,
                extra_slots_per_rank=helpers,
                max_tokens_per_rank=8192,
                topk=16,
                local_rank=15,
                ctas=8,
            )

        if accepted:
            config = build()
            assert config.home_experts_per_rank == experts // 16
            assert config.local_physical_slot_count == experts // 16 + helpers
        else:
            with pytest.raises(ValueError, match="logical_expert_count"):
                build()
    finally:
        sys.modules.pop(module_name, None)
