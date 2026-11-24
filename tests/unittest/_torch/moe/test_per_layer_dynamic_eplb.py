# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer dynamic EPLB lifecycle: manager, layer handle, planner group and factory.

Everything here runs without a GPU: the arena, the scheduler group and the CUDA
driver are stand-ins. The kernels themselves are covered by
``multi_gpu/test_per_layer_dynamic_eplb_gpu.py``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoEReplicaPlan
from tensorrt_llm._torch.moe.fused_moe.interface import MoE
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    BaseMoeLoadBalancer,
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
    get_moe_load_balancer,
    maybe_create_moe_load_balancer,
)
from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import PerLayerDynamicMoeLoadBalancer
from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.manager import (
    LAUNCH_QUEUES_ENV,
    LAUNCH_QUEUES_VALUE,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _launch_queues(monkeypatch):
    monkeypatch.setenv(LAUNCH_QUEUES_ENV, LAUNCH_QUEUES_VALUE)


class _Config:
    mode = "per_layer_dynamic"
    num_slots = 24
    auxiliary_sms = 8

    def setup(self, ep_rank, ep_size):
        self.ep_rank = ep_rank
        self.ep_size = ep_size

    def validate_expert_capacity(self, num_experts):
        if self.num_slots <= num_experts:
            raise ValueError("num_slots must exceed num_experts")


class _StandardConfig:
    mode = "standard"
    layer_updates_per_iter = 0

    def setup(self, ep_rank, ep_size):
        raise AssertionError("unsupported standard EPLB must remain disabled")


class _Arena:
    home_experts = 4
    helper_slots = 2

    def __init__(self, bundle="bundle"):
        self.bundle = bundle
        self.flags = torch.empty(4, dtype=torch.uint64)
        self.generation = 0
        self.provider = SimpleNamespace(pool=None)

    def publish_ready_generation(self, generation):
        self.generation = generation

    def terminal_flags_tensor(self):
        return self.flags

    def assert_identity(self):
        pass


class _Backend:
    def __init__(self):
        self.tactic_autotune = False
        self.mapping = SimpleNamespace(local_rank=0)
        self.ep_process_group = object()
        self.max_tokens_per_rank_capacity = 256
        self.layer_idx = 9


class _Group:
    home_experts = 4
    helper_slots = 2

    def __init__(self):
        self.wait_calls = 0
        self.ready_waits = []
        self.finish_calls = []
        self.events = []

    def plan(self, routes, *, defer_wait):
        assert defer_wait
        self.events.append("plan")
        return routes + 4, 7

    def plan_schedule(self, routes):
        self.events.append("schedule")
        return routes

    def plan_finish(self, part, *, defer_wait):
        assert defer_wait
        self.events.append("finish")
        return part + 4, 7

    def discard_plan(self, part):
        self.events.append("discard")

    def wait_for_routes(self):
        self.wait_calls += 1

    def wait_for_helper_weights(self, generation):
        self.ready_waits.append(generation)

    def finish(self, completion_event=None):
        self.finish_calls.append(completion_event)

    def close(self):
        pass


def _manager():
    return PerLayerDynamicMoeLoadBalancer(
        ep_rank=0,
        ep_size=4,
        config=_Config(),
    )


def _allocate_arena(layer, monkeypatch, arena):
    """Allocate ``arena`` through the layer handle, as the backend's binding does."""
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import arena as arena_module

    monkeypatch.setattr(layer.manager, "get_ep_comm", lambda process_group, mapping=None: object())
    monkeypatch.setattr(layer.manager, "create_arena_provider", lambda: object())
    monkeypatch.setattr(arena_module, "allocate_rebalance_arena", MagicMock(return_value=arena))
    return layer.allocate_arena(
        bundle=arena.bundle,
        ep_process_group=object(),
        mapping=object(),
        device=0,
        layer_idx=layer.layer_idx,
    )


def test_manager_requires_the_launch_queue_setting(monkeypatch):
    monkeypatch.delenv(LAUNCH_QUEUES_ENV)
    with pytest.raises(RuntimeError, match=f"{LAUNCH_QUEUES_ENV}={LAUNCH_QUEUES_VALUE}"):
        _manager()
    monkeypatch.setenv(LAUNCH_QUEUES_ENV, "2x")
    with pytest.raises(RuntimeError, match="'2x'"):
        _manager()


def test_layer_capacity_uses_configured_compute_slots():
    manager = _manager()
    with pytest.raises(ValueError, match="configured local slot count"):
        manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=4)


def test_layer_produces_and_finishes_replica_plan():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    arena = _Arena()
    group = _Group()
    layer.bind_runtime(arena=arena, scheduler_group=group)

    logical = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    physical = layer.route(logical)
    plan = layer.require_replica_plan()

    torch.testing.assert_close(physical, logical + 4)
    assert plan.resident_experts_per_rank == 4
    assert plan.compute_slots_per_rank == 6
    assert plan.helper_slots_per_rank == 2
    assert plan.ready_generation == arena.generation == 7
    assert plan.ready_flags is arena.flags
    assert plan.reserved_sms == 8
    plan.wait_for_routes()
    assert group.wait_calls == 1
    # The stream-ordered READY wait is the plan generation's.
    plan.wait_for_helper_weights()
    assert group.ready_waits == [7]

    layer.finish_replica_plan(plan)
    assert group.finish_calls == [None]
    assert layer.replica_plan is None


def test_replica_plan_without_ready_wait_refuses_stream_ordered_consumers():
    plan = MoEReplicaPlan(
        resident_experts_per_rank=4,
        compute_slots_per_rank=6,
        ready_flags=torch.empty(4, dtype=torch.uint64),
        ready_generation=1,
        reserved_sms=4,
        _wait_for_routes=lambda: None,
    )
    with pytest.raises(RuntimeError, match="gate helper reads on ready_flags itself"):
        plan.wait_for_helper_weights()
    with pytest.raises(ValueError, match="compute slots M greater"):
        MoEReplicaPlan(
            resident_experts_per_rank=4,
            compute_slots_per_rank=4,
            ready_flags=torch.empty(4, dtype=torch.uint64),
            ready_generation=1,
            reserved_sms=4,
            _wait_for_routes=lambda: None,
        )


def test_post_moe_hook_releases_the_chunk_plan():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    group = _Group()
    layer.bind_runtime(arena=_Arena(), scheduler_group=group)

    layer.route(torch.tensor([[0, 1]], dtype=torch.int32))
    layer.start_set_cpu_stage()

    assert group.wait_calls == 1
    assert group.finish_calls == [None]
    assert layer.replica_plan is None


def test_overlap_fn_runs_between_schedule_and_plan_finish():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    arena = _Arena()
    group = _Group()
    layer.bind_runtime(arena=arena, scheduler_group=group)

    logical = torch.tensor([[0, 1]], dtype=torch.int32)
    physical = layer.route(logical, overlap_fn=lambda: group.events.append("overlap"))
    torch.testing.assert_close(physical, logical + 4)
    assert group.events == ["schedule", "overlap", "finish"]
    layer.finish_replica_plan()


def test_overlap_fn_failure_discards_the_generation():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    group = _Group()
    layer.bind_runtime(arena=_Arena(), scheduler_group=group)

    def fail():
        raise RuntimeError("shared expert failed")

    with pytest.raises(RuntimeError, match="shared expert failed"):
        layer.route(torch.tensor([[0, 1]], dtype=torch.int32), overlap_fn=fail)
    assert group.events == ["schedule", "discard"]
    assert layer.replica_plan is None


def test_load_balancer_overlap_is_consumed_once_and_always_cleared():
    calls = []

    class _Layer:
        produces_replica_plan = True

        def route(self, routes, offset_by_ep_rank, *, overlap_fn=None):
            calls.append(overlap_fn)
            return routes

    moe = SimpleNamespace(layer_load_balancer=_Layer(), _load_balancer_overlap_fn=None)
    routes = torch.tensor([[0, 1]], dtype=torch.int32)

    def overlap():
        pass

    with MoE.load_balancer_overlap(moe, overlap):
        MoE._load_balancer_route(moe, routes, False)
        MoE._load_balancer_route(moe, routes, False)
    assert calls == [overlap, None]
    assert moe._load_balancer_overlap_fn is None

    with pytest.raises(RuntimeError, match="routed path failed"):
        with MoE.load_balancer_overlap(moe, overlap):
            raise RuntimeError("routed path failed")
    assert moe._load_balancer_overlap_fn is None

    with pytest.raises(TypeError, match="overlap_fn must be callable"):
        with MoE.load_balancer_overlap(moe, None):
            pass
    assert MoE._load_balancer_uses_replica_plan(moe)


def test_layer_disables_standard_host_migration_and_checks_geometry():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    assert not layer.is_dynamic_routing()
    assert not layer.need_load_shared_weights()
    assert layer.get_load_expert_ids() == [0, 1, 2, 3]
    assert layer.get_local_statistic_tensor() is None
    layer.set_initial_weight_assignments(range(16))
    with pytest.raises(ValueError, match="custom initial_global_assignments"):
        layer.set_initial_weight_assignments([0])
    with pytest.raises(ValueError, match="arena H/S geometry"):
        layer.bind_runtime(
            arena=SimpleNamespace(home_experts=5, helper_slots=1),
            scheduler_group=_Group(),
        )


def test_bind_backend_is_idempotent_before_scheduler_construction(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import planner

    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    arena = _Arena()
    assert _allocate_arena(layer, monkeypatch, arena) is arena
    backend = _Backend()
    group = _Group()
    builder = MagicMock(return_value=group)
    monkeypatch.setattr(planner, "build_rebalance_slot_scheduler_group", builder)

    layer.bind_backend(backend)
    layer.bind_backend(backend)

    # The group is built from the layer's geometry, the manager's resources and
    # the backend's public facts; layer_idx stays the backend's model index.
    builder.assert_called_once_with(
        arena=arena,
        manager=manager,
        mapping=backend.mapping,
        ep_process_group=backend.ep_process_group,
        home_experts=4,
        helper_slots=2,
        topk=2,
        max_tokens_per_rank=256,
        layer_idx=9,
    )
    assert layer._arena is arena
    assert layer._scheduler_group is group


def test_bind_backend_requires_the_layer_arena_and_rejects_another_backend(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import planner

    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    backend = _Backend()
    builder = MagicMock(return_value=_Group())
    monkeypatch.setattr(planner, "build_rebalance_slot_scheduler_group", builder)
    with pytest.raises(RuntimeError, match="no live-weight arena"):
        layer.bind_backend(backend)

    arena = _Arena()
    _allocate_arena(layer, monkeypatch, arena)
    # The runtime binds only the arena this handle allocated.
    with pytest.raises(RuntimeError, match="did not allocate"):
        layer.bind_runtime(arena=_Arena(), scheduler_group=_Group())
    layer.bind_backend(backend)

    with pytest.raises(RuntimeError, match="different backend"):
        layer.bind_backend(_Backend())
    builder.assert_called_once()
    assert layer.live_weight_arena is arena


def test_iteration_context_reuses_common_lifecycle():
    manager = _manager()
    with manager:
        assert get_moe_load_balancer() is manager
        with MoeLoadBalancerIterContext(manager, True, True):
            assert manager.in_iter
        assert not manager.in_iter
        assert manager.iter_id == 1
    assert get_moe_load_balancer() is None


def test_iteration_context_releases_the_plan_of_a_failed_forward():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    group = _Group()
    layer.bind_runtime(arena=_Arena(), scheduler_group=group)

    with pytest.raises(RuntimeError, match="MoE kernel failed"):
        with MoeLoadBalancerIterContext(manager, True, True):
            layer.route(torch.tensor([[0, 1]], dtype=torch.int32))
            raise RuntimeError("MoE kernel failed")

    assert group.finish_calls == [None]
    assert layer.replica_plan is None
    assert not manager.in_iter
    assert manager.iter_id == 1


def test_iteration_end_refuses_an_unfinished_plan():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    layer.bind_runtime(arena=_Arena(), scheduler_group=_Group())
    manager.start_iter()
    layer.route(torch.tensor([[0, 1]], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="unfinished replica plans"):
        manager.end_iter()


def test_model_owned_shutdown():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    arena = _Arena()
    pool = MagicMock()
    pool.close_collectively.return_value = True
    arena.provider.pool = pool
    manager._rebalance_shared_slot_pools[("test",)] = pool
    manager._rebalance_shared_slot_pool_order.append(pool)
    layer.bind_runtime(arena=arena, scheduler_group=_Group())

    assert manager.shutdown()
    pool.close_collectively.assert_called_once_with(local_safe=True)
    assert manager.is_shutdown


def test_ep_communicator_is_model_owned(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import arena as arena_module

    creations = []

    class Comm:
        def __init__(self, process_group):
            self.process_group = process_group
            self.validations = []
            creations.append(self)

        def validate_geometry(self, *, expected_rank, expected_world):
            self.validations.append((expected_rank, expected_world))

    monkeypatch.setattr(arena_module, "_EpComm", Comm)
    manager = _manager()
    process_group = object()

    first = manager.get_ep_comm(process_group)
    second = manager.get_ep_comm(process_group)

    assert first is second
    assert len(creations) == 1
    assert first.validations == [(0, 4), (0, 4)]
    with pytest.raises(RuntimeError, match="cannot span EP ProcessGroups"):
        manager.get_ep_comm(object())


def test_ep_communicator_falls_back_to_an_mpi_group_of_the_ep_ranks(monkeypatch):
    """Without a ProcessGroup (MPI orchestrator), the EP ranks form an MPI communicator."""
    import tensorrt_llm._utils as utils_module
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import arena as arena_module

    created = []

    class _SubComm:
        def __init__(self, members):
            self.members = members
            self.calls = []

        def Get_rank(self):
            return self.members.index(2)

        def Get_size(self):
            return len(self.members)

        def Barrier(self):
            self.calls.append("barrier")

        def allgather(self, value):
            self.calls.append(("allgather", value))
            return [value] * len(self.members)

        def bcast(self, payload, root):
            self.calls.append(("bcast", root))
            return payload

    class _WorldComm:
        group = SimpleNamespace(Incl=lambda members: list(members))

        def Create_group(self, group):
            created.append(group)
            return _SubComm(group)

    monkeypatch.setattr(utils_module, "mpi_comm", lambda: _WorldComm())
    mapping = SimpleNamespace(moe_ep_group=[0, 1, 2, 3], rank=2)

    comm = arena_module.resolve_ep_comm(None, mapping)

    assert isinstance(comm, arena_module._EpComm)
    assert created == [[0, 1, 2, 3]]
    assert (comm.rank, comm.world) == (2, 4)
    assert comm.process_group is None
    comm.validate_geometry(expected_rank=2, expected_world=4)
    assert comm.allgather("handle") == ["handle"] * 4
    assert comm.bcast("payload", 1) == "payload"
    comm.barrier()
    assert comm.comm.calls == [("allgather", "handle"), ("bcast", 1), "barrier"]
    with pytest.raises(ValueError, match="rank/size differs"):
        comm.validate_geometry(expected_rank=0, expected_world=4)


def test_layer_allocates_arena_from_model_owned_resources(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import arena as arena_module

    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    order = []
    process_group, comm, provider, bundle, mapping = (object() for _ in range(5))
    arena = SimpleNamespace(bundle=bundle)

    def get_ep_comm(group, group_mapping=None):
        assert group is process_group
        assert group_mapping is mapping
        order.append("ep_comm")
        return comm

    def create_arena_provider():
        order.append("provider")
        return provider

    monkeypatch.setattr(manager, "get_ep_comm", get_ep_comm)
    monkeypatch.setattr(manager, "create_arena_provider", create_arena_provider)
    allocate = MagicMock(return_value=arena)
    monkeypatch.setattr(arena_module, "allocate_rebalance_arena", allocate)
    kwargs = dict(ep_process_group=process_group, mapping=mapping, device=1, layer_idx=7)

    allocated = layer.allocate_arena(bundle=bundle, **kwargs)

    assert allocated is arena
    assert layer.live_weight_arena is arena
    assert order == ["ep_comm", "provider"]
    allocate.assert_called_once_with(
        home_experts=4,
        helper_slots=2,
        bundle=bundle,
        ep_comm=comm,
        mapping=mapping,
        provider=provider,
        device=1,
        layer_idx=7,
    )

    # Allocation happens once per layer; only the same plane spec gets the arena back.
    assert layer.allocate_arena(bundle=bundle, **kwargs) is arena
    with pytest.raises(RuntimeError, match="different live-weight plane spec"):
        layer.allocate_arena(bundle=object(), **kwargs)
    allocate.assert_called_once()
    assert order == ["ep_comm", "provider"]


@pytest.mark.parametrize(
    ("helper_slot_mode", "per_layer_sets"),
    [(None, True), ("per_layer_resident", True), ("shared_banks", False)],
)
def test_helper_slot_mode_selects_helper_residency(helper_slot_mode, per_layer_sets):
    config = _Config()
    if helper_slot_mode is not None:
        config.helper_slot_mode = helper_slot_mode
    manager = PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=config)

    assert manager.helper_residency is per_layer_sets
    provider = manager.create_arena_provider()
    assert provider._per_layer_helper_sets is per_layer_sets


def test_helper_slot_mode_rejects_unknown_mode():
    config = _Config()
    config.helper_slot_mode = "private"
    with pytest.raises(ValueError, match="helper_slot_mode"):
        PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=config)


def test_scheduler_spin_bound_follows_the_warmup_guard_until_shutdown(monkeypatch):
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.moe.fused_moe import moe_comm_timeout_guard

    guard = moe_comm_timeout_guard.MoECommTimeoutGuard(environ={}, is_capturing=lambda: False)
    monkeypatch.setattr(moe_comm_timeout_guard, "_DEFAULT_GUARD", guard)
    # The real AutoTuner reads the CUDA device name; this test runs without a GPU.
    tuner = SimpleNamespace(is_tuning_mode=False)
    monkeypatch.setattr(AutoTuner, "get", lambda: tuner)
    manager = _manager()

    bound = manager.get_scheduler_spin_bound()

    assert manager.get_scheduler_spin_bound() is bound
    assert bound.cycles() == 300 * 2_000_000_000
    tuner.is_tuning_mode = True
    assert bound.cycles() == 1800 * 2_000_000_000
    tuner.is_tuning_mode = False
    guard.set_warmup(True)
    assert bound.cycles() == 1800 * 2_000_000_000
    guard.set_warmup(False)
    assert bound.cycles() == 300 * 2_000_000_000
    assert manager.shutdown()
    guard.set_warmup(True)
    # A released bound no longer follows the guard.
    assert bound.cycles() == 300 * 2_000_000_000


def test_per_layer_dynamic_manager_shares_the_base_lifecycle_only():
    manager = _manager()
    assert isinstance(manager, BaseMoeLoadBalancer)
    assert not isinstance(manager, MoeLoadBalancer)
    assert manager.is_dynamic_routing()
    assert repr(manager) == (
        "PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, num_slots=24, auxiliary_sms=8)"
    )
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    assert layer.produces_replica_plan


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV4ForCausalLM", "KimiLinearForCausalLM", "KimiK31ForConditionalGeneration"],
)
def test_factory_selects_per_layer_dynamic_manager(architecture):
    mapping = SimpleNamespace(moe_ep_rank=0, moe_ep_size=4, moe_cluster_size=1, moe_tp_size=1)
    config = _Config()
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=[architecture]),
        moe_load_balancer=config,
    )

    with maybe_create_moe_load_balancer(model_config, mapping) as manager:
        assert isinstance(manager, PerLayerDynamicMoeLoadBalancer)
        assert config.ep_rank == 0
        assert config.ep_size == 4


@pytest.mark.parametrize(
    "architecture, ep_size, cluster_size, message",
    [
        ("UnsupportedForCausalLM", 4, 1, "does not support model architecture"),
        ("DeepseekV4ForCausalLM", 1, 1, "requires expert parallelism"),
        ("DeepseekV4ForCausalLM", 4, 2, "incompatible with smart routing"),
    ],
)
def test_per_layer_dynamic_factory_fails_fast_for_unsupported_topology(
    architecture, ep_size, cluster_size, message
):
    mapping = SimpleNamespace(moe_ep_rank=0, moe_ep_size=ep_size, moe_cluster_size=cluster_size)
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=[architecture]),
        moe_load_balancer=_Config(),
    )

    with pytest.raises(ValueError, match=message):
        maybe_create_moe_load_balancer(model_config, mapping)


def test_per_layer_dynamic_factory_refuses_moe_tp():
    mapping = SimpleNamespace(moe_ep_rank=0, moe_ep_size=4, moe_cluster_size=1, moe_tp_size=2)
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        moe_load_balancer=_Config(),
    )

    with pytest.raises(ValueError, match="requires moe_tp_size 1, got 2"):
        maybe_create_moe_load_balancer(model_config, mapping)


@pytest.mark.parametrize(
    "architecture, ep_size, cluster_size",
    [
        ("UnsupportedForCausalLM", 4, 1),
        ("DeepseekV4ForCausalLM", 1, 1),
        ("DeepseekV4ForCausalLM", 4, 2),
    ],
)
def test_standard_factory_keeps_unsupported_topology_disabled(architecture, ep_size, cluster_size):
    mapping = SimpleNamespace(moe_ep_rank=0, moe_ep_size=ep_size, moe_cluster_size=cluster_size)
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=[architecture]),
        moe_load_balancer=_StandardConfig(),
    )

    with maybe_create_moe_load_balancer(model_config, mapping) as manager:
        assert manager is None


def _bare_scheduler_group(ep_size, flags_ptr, driver):
    """A ``RebalanceSlotSchedulerGroup`` bound to a fake driver and MAIN stream."""
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.planner import (
        RebalanceSlotSchedulerGroup,
    )

    group = object.__new__(RebalanceSlotSchedulerGroup)
    group.ep_size = ep_size
    group._generation = 3
    group._ready_wait_generation = 0
    group._closed = False
    group._ready_flags_ptr = flags_ptr
    group._execution_stream_handle = 0x1234
    group._driver = driver
    group._wait_value_geq = "GEQ"
    group._cuda_success = 0
    group._check_owner = lambda: None
    return group


def test_ready_wait_queues_one_stream_wait_per_ep_rank_once_per_generation():
    calls = []

    class _Driver:
        @staticmethod
        def cuStreamWaitValue64(stream, address, value, flags):
            calls.append((stream, address, value, flags))
            return (0,)

    group = _bare_scheduler_group(4, 0x10000, _Driver())

    group.wait_for_helper_weights(3)
    group.wait_for_helper_weights(3)

    assert calls == [(0x1234, 0x10000 + 8 * rank, 3, "GEQ") for rank in range(4)]
    with pytest.raises(RuntimeError, match="active generation is 3"):
        group.wait_for_helper_weights(2)


def test_ready_wait_surfaces_driver_errors():
    class _Driver:
        @staticmethod
        def cuStreamWaitValue64(stream, address, value, flags):
            return (1,)

    group = _bare_scheduler_group(2, 0x10000, _Driver())
    with pytest.raises(RuntimeError, match="cuStreamWaitValue64"):
        group.wait_for_helper_weights(3)
    # A failed wait is not recorded, so a retry queues it again.
    assert group._ready_wait_generation == 0


# ---------------------------------------------------------------------------
# ExternalCommMoEScheduler: where a replica plan's device waits are queued
# ---------------------------------------------------------------------------

_TOP_K = 2
_HIDDEN = 8
_PHYSICAL_OFFSET = 100


def _first(events, name):
    return events.index(name)


def _scheduler_harness(*, comm_kind, replica_plan=True):
    """An ``ExternalCommMoEScheduler`` over a MoE whose collaborators record calls.

    ``comm_kind`` is ``"post_quant"`` (quantize, then dispatch: NVLink
    one-sided), ``"pre_quant"`` (dispatch, then quantize) or ``None``.
    """
    from tensorrt_llm._torch.moe.fused_moe.communication import NVLinkOneSided
    from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler

    events = []
    seen = {}
    logical = torch.tensor([[0, 5], [7, 3], [1, 2]], dtype=torch.int32)
    scales = torch.rand(3, _TOP_K)

    plan = MoEReplicaPlan(
        resident_experts_per_rank=4,
        compute_slots_per_rank=6,
        ready_flags=torch.zeros(4, dtype=torch.uint64),
        ready_generation=1,
        reserved_sms=4,
        _wait_for_routes=lambda: events.append("wait_routes"),
        _wait_for_helper_weights=lambda: events.append("wait_ready"),
    )

    class _Backend:
        input_requirement = SimpleNamespace(
            routing_scales_dtype=None,
            requires_sanitized_expert_ids=False,
            requires_run_moe_workspace=False,
            onesided_workspace_dtype=None,
        )
        capabilities = SimpleNamespace(supports_moe_lora=False)

        def _supports_load_balancer(self):
            return True

        def supports_moe_output_in_alltoall_workspace(self):
            return False

        def try_fused_route_quant(self, x, router_logits, input_ready_event=None):
            # Only reached without EPLB; declining keeps routing separate.
            return None

        def quantize_input(self, x, post_quant_comm=True):
            events.append("quantize")
            return x, None

        def run_moe(self, ctx, *, workspace=None):
            events.append("run_moe")
            seen["ctx"] = ctx
            return ctx.x

    comm = None
    if comm_kind is not None:
        comm = MagicMock(spec=NVLinkOneSided)
        comm.supports_post_quant_dispatch.return_value = comm_kind == "post_quant"
        comm.uses_internal_dispatch_quantization.return_value = False

        def dispatch(*, hidden_states, hidden_states_sf, token_selected_slots, **kwargs):
            events.append("dispatch")
            seen["dispatched_slots"] = token_selected_slots
            return (
                hidden_states,
                hidden_states_sf,
                token_selected_slots,
                kwargs["token_final_scales"],
            )

        def combine(output, *, all_rank_max_num_tokens):
            events.append("combine")
            return output

        comm.dispatch.side_effect = dispatch
        comm.combine.side_effect = combine

    def route(token_selected_experts, use_dp):
        events.append("route")
        seen["routed_logical"] = token_selected_experts
        return token_selected_experts + _PHYSICAL_OFFSET

    layer = SimpleNamespace(require_replica_plan=lambda: plan) if replica_plan else None
    moe = SimpleNamespace(
        backend=_Backend(),
        comm=comm,
        routing_method=SimpleNamespace(
            requires_separated_routing=False,
            experts_per_token=_TOP_K,
            apply=lambda router_logits, input_ids: (logical, scales),
        ),
        apply_router_weight_on_input=False,
        layer_load_balancer=layer,
        layer_idx=3,
        num_experts=16,
        num_slots=24 if replica_plan else 16,
        use_dp=True,
        enable_dummy_allreduce=False,
        quant_scales=None,
        unfinalized_combine_fn=None,
        hidden_size=_HIDDEN,
        enable_alltoall=comm is not None,
        parallel_size=1,
        reduce_results=False,
        _using_load_balancer=lambda: replica_plan,
        _load_balancer_uses_replica_plan=lambda: replica_plan,
        _load_balancer_start_wait_gpu_stage=lambda is_first_call: None,
        _load_balancer_get_local_statistic_tensor=lambda: None,
        _load_balancer_route=route,
        _load_balancer_start_set_cpu_stage=lambda is_last_call: (
            events.append("release") if replica_plan else None
        ),
        _load_balancer_done_set_cpu_stage=lambda is_last_call: None,
    )
    scheduler = ExternalCommMoEScheduler(moe)

    def run():
        x = torch.randn(3, _HIDDEN)
        return scheduler._forward_chunk_impl(
            x,
            torch.randn(3, 16),
            torch.float32,
            [3, 3],
            False,
            True,
            True,
            row_offset=None,
        )

    return run, events, seen, plan, logical


@pytest.mark.parametrize("comm_kind", ["post_quant", "pre_quant", None])
def test_scheduler_orders_route_and_ready_waits_around_their_readers(comm_kind):
    run, events, seen, plan, logical = _scheduler_harness(comm_kind=comm_kind)

    run()

    # HALO-Q reads the logical routes; everything after it reads physical slots.
    torch.testing.assert_close(seen["routed_logical"], logical)
    assert seen["routed_logical"].is_contiguous()
    physical = logical + _PHYSICAL_OFFSET
    torch.testing.assert_close(seen["ctx"].token_selected_experts, physical)
    assert seen["ctx"].replica_plan is plan
    # The route goes first, so planning and the copy overlap what follows.
    assert _first(events, "route") == 0
    first_reader = "dispatch" if comm_kind is not None else "run_moe"
    assert _first(events, "wait_routes") < _first(events, first_reader)
    if comm_kind == "post_quant":
        # The input quantization runs beside HALO-Q, before the route wait.
        assert _first(events, "quantize") < _first(events, "wait_routes")
    if comm_kind is not None:
        torch.testing.assert_close(seen["dispatched_slots"], physical)
        # The helper weights are awaited only after the dispatch was queued.
        assert _first(events, "dispatch") < _first(events, "wait_ready")
    assert _first(events, "wait_ready") < _first(events, "run_moe")
    # The plan is released after its kernel, before the combine.
    assert _first(events, "run_moe") < _first(events, "release")
    if comm_kind is not None:
        assert _first(events, "release") < _first(events, "combine")


def test_scheduler_without_replica_plan_queues_no_eplb_waits():
    run, events, seen, _, logical = _scheduler_harness(comm_kind="post_quant", replica_plan=False)

    run()

    assert events == ["quantize", "dispatch", "run_moe", "combine"]
    assert seen["ctx"].replica_plan is None
    torch.testing.assert_close(seen["ctx"].token_selected_experts, logical)


@pytest.mark.parametrize("replica_plan", [True, False])
def test_chunks_of_a_replica_plan_layer_stay_on_the_main_stream(replica_plan):
    """The planner binds one MAIN stream; chunk overlap would submit from the aux stream."""
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
    from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler

    chunks = []
    moe = SimpleNamespace(
        use_dp=False,
        enable_alltoall=False,
        # Entering the overlap path touches both; neither is usable here.
        aux_stream=object(),
        event_dict=None,
        repeat_idx=0,
        repeat_count=1,
        backend=SimpleNamespace(
            input_requirement=SimpleNamespace(requires_run_moe_workspace=False)
        ),
        _load_balancer_uses_replica_plan=lambda: replica_plan,
    )
    moe.split_chunk = lambda tokens, num_chunks: ConfigurableMoE.split_chunk(
        moe, tokens, num_chunks
    )
    scheduler = ExternalCommMoEScheduler(moe)

    def chunk_impl(x, *args, **kwargs):
        chunks.append(x.shape[0])
        return x

    scheduler._forward_chunk_impl = chunk_impl
    x = torch.randn(5, _HIDDEN)

    def run():
        return scheduler._forward_multiple_chunks(
            x, torch.randn(5, 16), 2, torch.float32, [5], False
        )

    if not replica_plan:
        # Control: without a replica plan the same layer takes the overlap path.
        with pytest.raises(TypeError):
            run()
        return
    torch.testing.assert_close(run(), x)
    assert chunks == [3, 2]


# ---------------------------------------------------------------------------
# TRTLLM-Gen backend: kernel window and planner capacity
# ---------------------------------------------------------------------------


def _trtllm_gen_backend(*, ep_rank=1, resident=4, helpers=2, replica_plans=True, rows=None):
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import TrtllmTrtllmGenNvfp4Impl

    # Construction needs a GPU; the window reads only these attributes.
    backend = TrtllmTrtllmGenNvfp4Impl.__new__(TrtllmTrtllmGenNvfp4Impl)
    backend.ep_rank = ep_rank
    backend.slot_start = ep_rank * resident
    backend.expert_size_per_partition = resident
    backend.layer_load_balancer = (
        SimpleNamespace(produces_replica_plan=True, helper_slots_per_rank=helpers)
        if replica_plans
        else None
    )
    backend.w3_w1_weight = torch.empty(rows if rows is not None else resident + helpers, 1)
    return backend


def _plan(resident=4, compute=6):
    return MoEReplicaPlan(
        resident_experts_per_rank=resident,
        compute_slots_per_rank=compute,
        ready_flags=torch.zeros(4, dtype=torch.uint64),
        ready_generation=1,
        reserved_sms=4,
        _wait_for_routes=lambda: None,
    )


def test_backend_window_without_a_plan_is_the_resident_window():
    backend = _trtllm_gen_backend(replica_plans=False, rows=4)
    assert backend.compute_slot_count == 4
    assert backend._local_slot_window(None) == (4, 4)


def test_backend_window_under_a_plan_spans_the_rank_compute_slots():
    backend = _trtllm_gen_backend()
    assert backend.compute_slot_count == 6
    # Physical slot ids are rank * M + local slot.
    assert backend._local_slot_window(_plan()) == (6, 6)
    # The kernel window of a forward without a plan does not move.
    assert backend._local_slot_window(None) == (4, 4)


@pytest.mark.parametrize(
    "plan, rows",
    [
        (_plan(resident=4, compute=7), 6),  # plan M differs from the layer
        (_plan(resident=3, compute=6), 6),  # plan H differs from the layer
        (_plan(), 4),  # operands still hold H rows: planes were never bound
    ],
)
def test_backend_window_refuses_a_plan_its_weights_do_not_match(plan, rows):
    backend = _trtllm_gen_backend(rows=rows)
    with pytest.raises(RuntimeError, match="replica-plan geometry"):
        backend._local_slot_window(plan)


@pytest.mark.parametrize(
    "use_dp, dp_size, max_num_tokens, moe_max_num_tokens, expected",
    [
        # Default moe_max_num_tokens = max_num_tokens * dp_size.
        (True, 16, 8192, 16 * 8192, 8192),
        # A smaller MoE chunk never shrinks the bound below one forward.
        (True, 16, 8192, 16 * 1024, 8192),
        # A chunk larger than one forward per rank raises it.
        (True, 4, 1024, 16384, 4096),
        (False, 16, 1024, 4096, 4096),
    ],
)
def test_planner_capacity_covers_one_rank_rows_of_any_chunk(
    use_dp, dp_size, max_num_tokens, moe_max_num_tokens, expected
):
    backend = _trtllm_gen_backend()
    backend.use_dp = use_dp
    backend.mapping = SimpleNamespace(dp_size=dp_size)
    backend.max_num_tokens = max_num_tokens
    backend._moe_max_num_tokens = moe_max_num_tokens
    assert backend.max_tokens_per_rank_capacity == expected


# ---------------------------------------------------------------------------
# Live-weight planes
# ---------------------------------------------------------------------------


def test_live_weight_bundle_describes_each_per_expert_tensor():
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.live_planes import (
        live_weight_bundle,
    )

    resident = 3
    planes = [
        ("w3_w1_weight", torch.empty(resident, 64, 32, dtype=torch.uint8)),
        ("w3_w1_weight_scale", torch.empty(resident, 64, 4, dtype=torch.float8_e4m3fn)),
        ("fc31_alpha", torch.empty(resident, dtype=torch.float32)),
        ("w2_weight", torch.empty(resident, 64, 16, dtype=torch.uint8)),
    ]

    bundle = live_weight_bundle(planes, hidden=64, intermediate=32)

    assert [plane.name for plane in bundle.planes] == [name for name, _ in planes]
    weight, scale, alpha, down = bundle.planes
    assert (weight.nbytes, scale.nbytes, alpha.nbytes, down.nbytes) == (2048, 256, 4, 1024)
    assert weight.kernel.shape == (64, 32) and weight.kernel.stride == (32, 1)
    assert weight.kernel.dtype == "uint8"
    assert scale.kernel.dtype == "float8_e4m3fn"
    # A per-expert scalar is one element per slot.
    assert alpha.kernel.shape == () and alpha.kernel.pointer_alignment == 4
    # Large planes keep the alignment the kernels' TMA descriptors need.
    assert weight.kernel.pointer_alignment == 256
    assert all(plane.offset % plane.alignment == 0 for plane in bundle.planes)
    assert bundle.total_bytes == 2048 + 256 + 4 + 1024
    assert bundle.stride_bytes % bundle.alignment == 0
    # Equal specs compare equal: the layer handle reuses its arena for them.
    assert live_weight_bundle(planes, hidden=64, intermediate=32) == bundle


@pytest.mark.parametrize(
    "planes, error, message",
    [
        ([], ValueError, "at least one plane"),
        (
            [("a", torch.empty(3, 16, dtype=torch.uint8)), ("a", torch.empty(3, 16))],
            ValueError,
            "unique",
        ),
        (
            [("a", torch.empty(3, 16, dtype=torch.uint8)), ("b", torch.empty(4, 16))],
            ValueError,
            "expected 3",
        ),
        ([("a", torch.empty(16, 3, dtype=torch.uint8).t())], ValueError, "contiguous"),
        ([("a", torch.empty(3, 15, dtype=torch.uint8))], ValueError, "multiple of 16"),
    ],
)
def test_live_weight_bundle_refuses_planes_the_arena_cannot_alias(planes, error, message):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.live_planes import (
        live_weight_bundle,
    )

    with pytest.raises(error, match=message):
        live_weight_bundle(planes, hidden=64, intermediate=32)


def test_rebinding_keeps_parameter_objects_and_buffer_registration():
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.live_planes import _rebind

    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)
    module.register_buffer("constant", torch.zeros(2), persistent=False)
    module.plain = torch.zeros(2)
    parameter = module.weight
    arena_rows = torch.arange(12, dtype=torch.float32).view(3, 4)

    _rebind(module, "weight", arena_rows)
    _rebind(module, "constant", arena_rows[:, 0])
    _rebind(module, "plain", arena_rows[:, 1])

    # The Parameter object survives (state dict, optimizers, weakrefs); its
    # storage is the arena's.
    assert module.weight is parameter
    assert module.weight.data_ptr() == arena_rows.data_ptr()
    assert module.weight.shape == (3, 4)
    assert "constant" in module._buffers
    assert module.constant.data_ptr() == arena_rows.data_ptr()
    assert module.plain.data_ptr() == arena_rows[:, 1].data_ptr()
