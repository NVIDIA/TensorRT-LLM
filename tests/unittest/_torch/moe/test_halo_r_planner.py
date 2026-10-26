# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests of the HALO-R planner reference and its group selection."""

import random
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from _torch.moe.halo_r_reference import plan_halo_r

from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import (
    PerLayerDynamicMoeLoadBalancer,
    halo_r_planner,
)
from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb.halo_r_planner import (
    HALO_R_SEND_COUNT,
    HALO_R_SPIN_BUDGET_NS,
    HaloRSlotSchedulerGroup,
    validate_halo_r_geometry,
)

pytestmark = pytest.mark.cpu_only

# Eight ranks of four experts. Rank 0 holds the hot expert 0 (300 of its 400
# routes); ranks 1, 4 and 5 are idle and the others sit at the balanced 100.
WORKED_EXAMPLE = [300, 40, 30, 30, 0, 0, 0, 0, 40, 30, 20, 10, 40, 30, 20, 10]
WORKED_EXAMPLE += [0, 0, 0, 0, 0, 0, 0, 0, 40, 30, 20, 10, 40, 30, 20, 10]


def test_worked_example_replicates_hot_expert_to_idle_ranks():
    plan = plan_halo_r(WORKED_EXAMPLE, ep=8, helper_slots=2, send_count=4)
    assert plan.target == 100
    assert len(plan.broadcasts) == 1
    broadcast = plan.broadcasts[0]
    assert (broadcast.expert, broadcast.owner) == (0, 0)
    assert broadcast.rows == {1: 100, 4: 100, 5: 100}
    assert broadcast.helper == {1: 0, 4: 0, 5: 0}
    assert plan.quota(WORKED_EXAMPLE, 8) == [[0, 100, 0, 0, 100, 100, 0, 0]]
    assert plan.load_after == [100] * 8


def test_receivers_fill_helper_slots_in_expert_order():
    histogram = [0, 0, 80, 10, 40, 200, 40, 10, 10, 80, 200, 0, 0, 80, 0, 200]
    plan = plan_halo_r(histogram, ep=4, helper_slots=4, send_count=4)
    assert [b.expert for b in plan.broadcasts] == [5, 10, 15]
    assert [b.helper[0] for b in plan.broadcasts] == [0, 1, 2]
    assert plan.helper_table[0] == [5, 10, 15, -1]
    words = plan.plan_words()
    stride = 384
    # Helper slot 2 sets the high mask bit of rank 0.
    assert words[2 + 3 * stride + 2] == 0
    assert words[2 + 5 * stride + 2] == 1


@pytest.mark.parametrize("seed", range(4))
def test_plans_respect_helper_budget_and_conserve_routes(seed):
    rng = random.Random(seed)
    for _ in range(50):
        ep = rng.choice([2, 4, 8, 16, 32])
        home = rng.randint(HALO_R_SEND_COUNT, 12)
        helper_slots = rng.randint(1, 4)
        histogram = [rng.randint(0, 300) if rng.random() < 0.7 else 0 for _ in range(ep * home)]
        histogram[rng.randrange(ep * home)] += rng.randint(0, 5000)
        plan = plan_halo_r(histogram, ep, helper_slots, HALO_R_SEND_COUNT)
        received = [0] * ep
        for b in plan.broadcasts:
            assert b.owner not in b.rows
            assert sum(b.rows.values()) <= histogram[b.expert]
            for rank, slot in b.helper.items():
                received[rank] += 1
                assert plan.helper_table[rank][slot] == b.expert
        assert max(received) <= helper_slots
        assert sum(plan.load_after) == sum(histogram)
        quotas = plan.quota(histogram, ep)
        for b, quota in zip(plan.broadcasts, quotas):
            assert sum(quota) == histogram[b.expert]


@pytest.mark.parametrize(
    "ep_size, home_experts, helper_slots, message",
    [
        pytest.param(1, 12, 2, "EP ranks", id="single-rank"),
        pytest.param(64, 6, 2, "EP ranks", id="above-one-warp"),
        pytest.param(8, 48, 5, "helper slots", id="five-helpers"),
        pytest.param(8, 3, 2, "experts per rank", id="fewer-home-experts-than-sends"),
        pytest.param(32, 16, 2, "at most 384 experts", id="too-many-experts"),
    ],
)
def test_geometry_validation_rejects_unsupported_shapes(
    ep_size, home_experts, helper_slots, message
):
    with pytest.raises(ValueError, match=message):
        validate_halo_r_geometry(
            ep_size=ep_size, home_experts=home_experts, helper_slots=helper_slots
        )


def test_geometry_validation_accepts_ep32_two_helper_slots():
    validate_halo_r_geometry(ep_size=32, home_experts=12, helper_slots=2)


class _Config:
    mode = "per_layer_dynamic"
    num_slots = 24
    auxiliary_sms = 8
    helper_slot_mode = "shared_banks"

    def __init__(self, planner):
        self.planner = planner

    def setup(self, ep_rank, ep_size):
        pass

    def validate_expert_capacity(self, num_experts):
        pass


class _Arena:
    home_experts = 4
    helper_slots = 2
    bundle = None

    def __init__(self):
        self.provider = SimpleNamespace(pool=None)


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


class _Backend:
    def __init__(self):
        self.tactic_autotune = False
        self.mapping = SimpleNamespace(local_rank=0)
        self.ep_process_group = object()
        self.max_tokens_per_rank_capacity = 256
        self.layer_idx = 3


def test_halo_r_planner_builds_the_halo_r_group(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import planner

    manager = PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=_Config("halo_r"))
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    group = SimpleNamespace(home_experts=4, helper_slots=2)
    halo_r_builder = MagicMock(return_value=group)
    halo_q_builder = MagicMock()
    monkeypatch.setattr(halo_r_planner, "build_halo_r_slot_scheduler_group", halo_r_builder)
    monkeypatch.setattr(planner, "build_rebalance_slot_scheduler_group", halo_q_builder)
    arena = _Arena()
    _allocate_arena(layer, monkeypatch, arena)
    backend = _Backend()

    layer.bind_backend(backend)

    halo_q_builder.assert_not_called()
    halo_r_builder.assert_called_once_with(
        arena=arena,
        manager=manager,
        mapping=backend.mapping,
        ep_process_group=backend.ep_process_group,
        home_experts=4,
        helper_slots=2,
        topk=2,
        max_tokens_per_rank=256,
        layer_idx=3,
    )
    assert layer._scheduler_group is group
    assert repr(manager).endswith("planner=halo_r)")


def test_default_planner_builds_the_halo_q_group(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.per_layer_dynamic_eplb import planner

    config = _Config("halo_q")
    del config.planner
    manager = PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=config)
    layer = manager.add_layer(expert_count=16, top_k=2, slot_count_per_rank=6)
    halo_r_builder = MagicMock()
    halo_q_builder = MagicMock(return_value=SimpleNamespace(home_experts=4, helper_slots=2))
    monkeypatch.setattr(halo_r_planner, "build_halo_r_slot_scheduler_group", halo_r_builder)
    monkeypatch.setattr(planner, "build_rebalance_slot_scheduler_group", halo_q_builder)

    _allocate_arena(layer, monkeypatch, _Arena())
    layer.bind_backend(_Backend())

    halo_r_builder.assert_not_called()
    halo_q_builder.assert_called_once()
    assert manager.planner == "halo_q"


def test_manager_rejects_unknown_planner():
    with pytest.raises(ValueError, match="unknown per-layer dynamic EPLB planner"):
        PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=_Config("halo_m"))


def test_manager_rejects_halo_r_with_resident_helper_slots():
    config = _Config("halo_r")
    config.helper_slot_mode = "per_layer_resident"
    with pytest.raises(ValueError, match="requires helper_slot_mode 'shared_banks'"):
        PerLayerDynamicMoeLoadBalancer(ep_rank=0, ep_size=4, config=config)


def test_reference_plan_words_have_the_kernel_layout():
    plan = plan_halo_r(WORKED_EXAMPLE, ep=8, helper_slots=2, send_count=4)
    words = plan.plan_words(epoch=7)
    assert len(words) == 2 + 6 * 384
    assert words[:2] == [7, 1]
    assert words[2 + 2 * 384] == (1 << 1) | (1 << 4) | (1 << 5)


_HALO_R_OP_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "cpp/tensorrt_llm/thop/moe/loadBalance/moeRebalanceHaloROp.cpp"
)


def _halo_r_op_parameters(name):
    """Parameter names of a registered moe_rebalance_halo_r operator schema."""
    op_source = _HALO_R_OP_SOURCE.read_text()
    schemas = ["".join(re.findall(r'"([^"]*)"', block)) for block in op_source.split("m.def(")[1:]]
    schema = next(text for text in schemas if text.startswith(f"{name}("))
    parameters = re.fullmatch(rf"{name}\((.*)\) -> \(\)", schema).group(1)
    return [parameter.split()[-1] for parameter in parameters.split(",")]


def test_per_layer_ops_take_the_context_handle():
    # The context binds every shared argument; a custom-class argument would
    # cost microseconds of dispatch per call.
    assert _halo_r_op_parameters("moe_rebalance_halo_r_submit") == ["handle", "routes", "spin_ns"]
    assert _halo_r_op_parameters("moe_rebalance_halo_r_wait_routes") == ["handle"]
    assert _halo_r_op_parameters("moe_rebalance_halo_r_release") == [
        "handle",
        "completion_event",
    ]


class _FakeRoutes:
    """Just enough of a CUDA int32 [tokens, topk] tensor for the host checks."""

    ndim = 2

    def __init__(self, fake_torch, tokens, topk):
        self.shape = (tokens, topk)
        self.dtype = fake_torch.int32
        self.device = "cuda:0"

    def is_contiguous(self):
        return True


class _FakeSlots(_FakeRoutes):
    def __getitem__(self, index):
        return ("slots", index)


def _fake_torch(calls, main_handle=11):
    def op(name):
        return lambda *args: calls.append((name, *args))

    fake = SimpleNamespace(int32=object())
    fake.Tensor = _FakeRoutes
    fake.cuda = SimpleNamespace(
        current_device=lambda: 0,
        current_stream=lambda device: SimpleNamespace(cuda_stream=main_handle, priority=0),
    )
    fake.ops = SimpleNamespace(
        trtllm=SimpleNamespace(
            moe_rebalance_halo_r_submit=op("submit"),
            moe_rebalance_halo_r_wait_routes=op("wait_routes"),
            moe_rebalance_halo_r_release=op("release"),
        )
    )
    return fake


def _bound_group(fake_torch, handle=0x1234):
    """A group past construction: bound to a native context with ``handle``."""
    group = HaloRSlotSchedulerGroup.__new__(HaloRSlotSchedulerGroup)
    group.device = 0
    group.topk = 2
    group.max_tokens_per_rank = 8
    group.copy_stream = SimpleNamespace(priority=-1)
    group.physical_slot_ids = _FakeSlots(fake_torch, 8, 2)
    group.plan_calls = 0
    group._handle = handle
    group._copy_handle = 22
    group._closed = False
    group._validated_streams = set()
    group._call_stream_handle = None
    group._pending_tokens = None
    group._routes_waited = False
    group._planned = False
    group._part = None
    return group


def test_layer_call_makes_one_native_call_per_step(monkeypatch):
    calls = []
    fake_torch = _fake_torch(calls)
    monkeypatch.setattr(halo_r_planner, "torch", fake_torch)
    group = _bound_group(fake_torch)
    routes = _FakeRoutes(fake_torch, 5, 2)

    slots, generation = group.plan(routes)
    group.finish()

    assert (slots, generation) == (("slots", slice(None, 5)), 1)
    # The native calls carry the whole fork/join; no event, stream or driver
    # call is made from Python.
    assert calls == [
        ("submit", 0x1234, routes, HALO_R_SPIN_BUDGET_NS),
        ("wait_routes", 0x1234),
        ("release", 0x1234, 0),
    ]


def test_deferred_route_wait_is_enqueued_once_and_release_takes_the_event(monkeypatch):
    calls = []
    fake_torch = _fake_torch(calls)
    monkeypatch.setattr(halo_r_planner, "torch", fake_torch)
    group = _bound_group(fake_torch)
    routes = _FakeRoutes(fake_torch, 0, 2)

    part = group.plan_schedule(routes)
    group.plan_finish(part, defer_wait=True)
    assert [call[0] for call in calls] == ["submit"]
    group.wait_for_routes()
    group.wait_for_routes()
    group.finish(SimpleNamespace(cuda_event=0x99))

    assert calls[1:] == [("wait_routes", 0x1234), ("release", 0x1234, 0x99)]
    with pytest.raises(ValueError, match="CUDA event"):
        group.plan(routes)
        group.finish(SimpleNamespace(cuda_event=0))


def test_layer_call_rejects_the_copy_stream(monkeypatch):
    calls = []
    fake_torch = _fake_torch(calls, main_handle=22)
    monkeypatch.setattr(halo_r_planner, "torch", fake_torch)
    group = _bound_group(fake_torch)

    with pytest.raises(RuntimeError, match="distinct from the higher-priority COPY"):
        group.plan(_FakeRoutes(fake_torch, 1, 2))
    assert calls == []
