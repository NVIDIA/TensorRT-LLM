# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.mega_moe.rebalance_slot_scheduler import (
    RebalanceSlotSchedulerGroup,
)
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancerIterContext,
    get_moe_load_balancer,
    maybe_create_moe_load_balancer,
)
from tensorrt_llm._torch.moe.fused_moe.per_iteration_eplb import PerIterationMoeLoadBalancer

pytestmark = pytest.mark.cpu_only


class _Config:
    mode = "per_iteration"
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

    def __init__(self):
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
    def __init__(self, arena):
        self._rebalance_arena = arena
        self.tactic_autotune = False


class _Group:
    home_experts = 4
    helper_slots = 2

    def __init__(self):
        self.has_submission_owner = False
        self.wait_calls = 0
        self.finish_calls = []
        self.handoff_calls = []
        self.release_calls = []
        self.events = []

    def plan(self, routes, *, defer_wait):
        assert defer_wait
        self.has_submission_owner = True
        self.events.append("plan")
        return routes + 4, 7

    def plan_schedule(self, routes):
        self.has_submission_owner = True
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

    def release_warmup_owner_after_sync(self, stream_handle):
        self.handoff_calls.append(stream_handle)
        self.has_submission_owner = False

    def release_quiesced_submission_owner(self, stream_handle):
        self.release_calls.append(stream_handle)
        self.has_submission_owner = False

    def close(self):
        pass


def _manager():
    return PerIterationMoeLoadBalancer(
        ep_rank=0,
        ep_size=4,
        config=_Config(),
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


def test_gap_hook_runs_between_schedule_and_plan_finish():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    arena = _Arena()
    group = _Group()
    layer.bind_runtime(arena=arena, scheduler_group=group)

    logical = torch.tensor([[0, 1]], dtype=torch.int32)
    physical = layer.route(logical, gap_hook=lambda: group.events.append("hook"))
    torch.testing.assert_close(physical, logical + 4)
    assert group.events == ["schedule", "hook", "finish"]
    layer.finish_replica_plan()


def test_gap_hook_failure_discards_generation_and_adds_context():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    group = _Group()
    layer.bind_runtime(arena=_Arena(), scheduler_group=group)

    def fail():
        raise RuntimeError("shared expert failed")

    with pytest.raises(RuntimeError, match="shared expert failed") as caught:
        layer.route(torch.tensor([[0, 1]], dtype=torch.int32), gap_hook=fail)
    assert group.events == ["schedule", "discard"]
    assert "safely released" in caught.value.__notes__[0]
    assert layer.replica_plan is None


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
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import rebalance_slot_scheduler

    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    arena = _Arena()
    backend = _Backend(arena)
    group = _Group()
    builder = MagicMock(return_value=group)
    monkeypatch.setattr(
        rebalance_slot_scheduler,
        "build_rebalance_slot_scheduler_group",
        builder,
    )

    layer.bind_backend(backend)
    layer.bind_backend(backend)

    builder.assert_called_once_with(backend)
    assert layer._arena is arena
    assert layer._scheduler_group is group


def test_bind_backend_rejects_different_backend_or_arena_before_construction(
    monkeypatch,
):
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import rebalance_slot_scheduler

    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    arena = _Arena()
    backend = _Backend(arena)
    builder = MagicMock(return_value=_Group())
    monkeypatch.setattr(
        rebalance_slot_scheduler,
        "build_rebalance_slot_scheduler_group",
        builder,
    )
    layer.bind_backend(backend)

    with pytest.raises(RuntimeError, match="different backend"):
        layer.bind_backend(_Backend(arena))
    backend._rebalance_arena = _Arena()
    with pytest.raises(RuntimeError, match="changed its rebalance arena"):
        layer.bind_backend(backend)

    builder.assert_called_once_with(backend)


def test_iteration_context_reuses_common_lifecycle():
    manager = _manager()
    with manager:
        assert get_moe_load_balancer() is manager
        with MoeLoadBalancerIterContext(manager, True, True):
            assert manager.in_iter
        assert not manager.in_iter
        assert manager.iter_id == 1
    assert get_moe_load_balancer() is None


def test_public_warmup_handoff_and_model_owned_shutdown():
    manager = _manager()
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    arena = _Arena()
    pool = MagicMock()
    pool.close_collectively.return_value = True
    arena.provider.pool = pool
    manager._rebalance_shared_slot_pools[("test",)] = pool
    manager._rebalance_shared_slot_pool_order.append(pool)
    group = _Group()
    group.has_submission_owner = True
    layer.bind_runtime(arena=arena, scheduler_group=group)

    assert manager.has_submission_owner
    manager.handoff_submission_owner(1234)
    assert group.handoff_calls == [1234]
    manager.release_quiesced_submission_owner(1234)
    assert group.release_calls == [1234]
    assert manager.shutdown()
    pool.close_collectively.assert_called_once_with(local_safe=True)
    assert manager.is_shutdown


def _scheduler_group_for_owner_release(*, finished=True):
    group = RebalanceSlotSchedulerGroup.__new__(RebalanceSlotSchedulerGroup)
    group._closed = False
    group._plan_part = None
    group._generation = 3
    group._finished_generation = 3 if finished else 2
    group.plan_calls = 3
    group._owner_thread_id = 99
    group._execution_stream_handle = 1234
    return group


def test_quiesced_owner_release_clears_worker_and_stream_binding():
    group = _scheduler_group_for_owner_release()

    group.release_quiesced_submission_owner(1234)

    assert group._owner_thread_id is None
    assert group._execution_stream_handle is None


def test_quiesced_owner_release_rejects_unfinished_generation():
    group = _scheduler_group_for_owner_release(finished=False)

    with pytest.raises(RuntimeError, match="all plans to be finished"):
        group.release_quiesced_submission_owner(1234)


def test_ep_communicator_is_model_owned(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import rebalance_live_arena

    creations = []

    class Comm:
        def __init__(self, process_group):
            self.process_group = process_group
            self.validations = []
            creations.append(self)

        def validate_geometry(self, *, expected_rank, expected_world):
            self.validations.append((expected_rank, expected_world))

    monkeypatch.setattr(rebalance_live_arena, "_EpComm", Comm)
    manager = _manager()
    process_group = object()

    first = manager.get_ep_comm(process_group)
    second = manager.get_ep_comm(process_group)

    assert first is second
    assert len(creations) == 1
    assert first.validations == [(0, 4), (0, 4)]
    with pytest.raises(RuntimeError, match="cannot span EP ProcessGroups"):
        manager.get_ep_comm(object())


def test_factory_selects_per_iteration_manager():
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
        assert isinstance(manager, PerIterationMoeLoadBalancer)
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
def test_per_iteration_factory_fails_fast_for_unsupported_topology(
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
