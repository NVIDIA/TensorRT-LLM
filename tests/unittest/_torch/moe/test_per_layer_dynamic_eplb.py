# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.interface import MoE
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    BaseMoeLoadBalancer,
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
    get_moe_load_balancer,
    maybe_create_moe_load_balancer,
)
from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import PerLayerDynamicMoeLoadBalancer

pytestmark = pytest.mark.cpu_only


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
    """Allocate ``arena`` through the layer handle, as the quant method does."""
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import arena as arena_module

    monkeypatch.setattr(layer.manager, "get_ep_comm", lambda process_group: object())
    monkeypatch.setattr(layer.manager, "create_arena_provider", lambda: object())
    monkeypatch.setattr(arena_module, "allocate_rebalance_arena", MagicMock(return_value=arena))
    return layer.allocate_arena(
        bundle=arena.bundle,
        ep_process_group=object(),
        mapping=object(),
        device=0,
        layer_idx=layer.layer_idx,
    )


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

    layer.finish_replica_plan(plan)
    assert group.finish_calls == [None]
    assert layer.replica_plan is None


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


def test_overlap_fn_failure_discards_generation_and_adds_context():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    group = _Group()
    layer.bind_runtime(arena=_Arena(), scheduler_group=group)

    def fail():
        raise RuntimeError("shared expert failed")

    with pytest.raises(RuntimeError, match="shared expert failed") as caught:
        layer.route(torch.tensor([[0, 1]], dtype=torch.int32), overlap_fn=fail)
    assert group.events == ["schedule", "discard"]
    assert "safely released" in caught.value.__notes__[0]
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
    monkeypatch.setattr(
        planner,
        "build_rebalance_slot_scheduler_group",
        builder,
    )

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


def test_bind_backend_requires_the_layer_arena_and_rejects_another_backend(
    monkeypatch,
):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import planner

    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    backend = _Backend()
    builder = MagicMock(return_value=_Group())
    monkeypatch.setattr(
        planner,
        "build_rebalance_slot_scheduler_group",
        builder,
    )
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


def test_layer_allocates_arena_from_model_owned_resources(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import arena as arena_module

    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    order = []
    process_group, comm, provider, bundle, mapping = (object() for _ in range(5))
    arena = SimpleNamespace(bundle=bundle)

    def get_ep_comm(group):
        assert group is process_group
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


def test_factory_selects_per_layer_dynamic_manager():
    mapping = SimpleNamespace(
        moe_ep_rank=0,
        moe_ep_size=4,
        moe_cluster_size=1,
    )
    config = _Config()
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
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
    mapping = SimpleNamespace(
        moe_ep_rank=0,
        moe_ep_size=ep_size,
        moe_cluster_size=cluster_size,
    )
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=[architecture]),
        moe_load_balancer=_Config(),
    )

    with pytest.raises(ValueError, match=message):
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
    mapping = SimpleNamespace(
        moe_ep_rank=0,
        moe_ep_size=ep_size,
        moe_cluster_size=cluster_size,
    )
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(architectures=[architecture]),
        moe_load_balancer=_StandardConfig(),
    )

    with maybe_create_moe_load_balancer(model_config, mapping) as manager:
        assert manager is None
