# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the spare EPLB slots that hold a failed rank's experts on the active ranks."""

import functools
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.cutedsl import TrtllmCutedslGroupedGemmNvfp4Impl
from tensorrt_llm._torch.moe.fused_moe.cutlass import (
    DeepgemmCudaHopperFp8BlockScalesImpl,
    TrtllmCutlassNvfp4Impl,
    TrtllmCutlassUnquantizedImpl,
    TrtllmCutlassW4a16Nvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import (
    EplbExpertRestore,
    FaultTolerantMoeLoadBalancer,
    create_spare_slot_storage,
    plan_layer_placement,
    use_spare_slot_storage,
)
from tensorrt_llm._torch.moe.fused_moe.impl_blocks import (
    MoEEplbWeightLayoutMixin,
    MoEWeightOwnerMixin,
)
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer

pytestmark = pytest.mark.cpu_only

_ER = "tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore"
_USES_GDRCOPY = "tensorrt_llm.bindings.internal.runtime.host_accessible_device_memory_uses_gdrcopy"
WORLD_SIZE = 4
EXPERTS = 4


def _deployment():
    """Return the keyword arguments of a rank's load balancer."""
    return {"mapping": None}


def _fault_tolerance(monkeypatch, largest_failure):
    """Turn fault tolerance on for nodes [0, 1] and [2, 3] and ``largest_failure`` lost ranks."""
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", lambda self: [0, 0, 1, 1])
    recoverable = EplbExpertRestore._recoverable_failures
    monkeypatch.setattr(
        EplbExpertRestore,
        "_recoverable_failures",
        staticmethod(lambda nodes: [f for f in recoverable(nodes) if len(f) <= largest_failure]),
    )


def _load_balancer(
    monkeypatch, *, largest_failure, eplb_slots, layer_updates_per_iter=1, ep_rank=0
):
    """Return a fault-tolerant MoeLoadBalancer over one layer of ``eplb_slots`` per rank."""
    _fault_tolerance(monkeypatch, largest_failure)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(_USES_GDRCOPY, return_value=True),
        patch(f"{_ER}.DeploymentHostWeights"),
        patch("torch.cuda.Event"),
    ):
        load_balancer = FaultTolerantMoeLoadBalancer(
            ep_rank, WORLD_SIZE, layer_updates_per_iter, **_deployment()
        )
        load_balancer.add_layer(EXPERTS, 2, eplb_slots, aux_stream=Mock())
    return load_balancer


@pytest.mark.parametrize(
    "largest_failure, layer_updates_per_iter, uses_gdrcopy, refused",
    [
        (2, 1, False, True),
        (2, 1, True, False),
        # Static EPLB writes no slot from the host. Without a rank to lose there is no spare slot.
        (2, 0, False, False),
        (0, 1, False, False),
    ],
)
def test_online_spare_slots_need_gdrcopy(
    monkeypatch, largest_failure, layer_updates_per_iter, uses_gdrcopy, refused
):
    """Refuse online EPLB with spare slots unless GDRCopy maps the slots for host writes."""
    _fault_tolerance(monkeypatch, largest_failure)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(_USES_GDRCOPY, return_value=uses_gdrcopy),
        pytest.raises(ValueError, match="requires GDRCopy") if refused else nullcontext(),
    ):
        FaultTolerantMoeLoadBalancer(0, WORLD_SIZE, layer_updates_per_iter, **_deployment())


@pytest.mark.parametrize(
    "largest_failure, eplb_slots, spare_slot_count",
    [
        # After losing a node of two, the two active ranks need two slots each for four experts.
        (2, 1, 1),
        # After losing one rank, the three active ranks still need two slots each.
        (1, 1, 1),
        # A node may hold both replicas of up to four experts, which the spare slots take.
        (2, 2, 2),
        # Without a rank to lose, no spare slot is added.
        (0, 1, 0),
    ],
)
def test_spare_slots_cover_largest_failure(
    monkeypatch, largest_failure, eplb_slots, spare_slot_count
):
    """Add enough spare slots for the survivors of the largest failure to hold every expert."""
    load_balancer = _load_balancer(
        monkeypatch, largest_failure=largest_failure, eplb_slots=eplb_slots
    )
    (layer,) = load_balancer.single_layer_load_balancers
    load_balancer.load_balancer_impl.add_layer.assert_called_once_with(
        EXPERTS, 2, eplb_slots + spare_slot_count
    )
    assert layer.spare_slot_count == spare_slot_count
    assert layer.active_slot_count_per_rank == eplb_slots
    assert load_balancer.adds_spare_slots is (spare_slot_count > 0)


def test_no_spare_slots_without_fault_tolerance(monkeypatch):
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", Mock(side_effect=AssertionError("nodes")))
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch.object(MoeLoadBalancer, "host_tensor_sharer_cls"),
        patch("torch.cuda.Event"),
    ):
        load_balancer = MoeLoadBalancer(0, WORLD_SIZE, 1)
        layer = load_balancer.add_layer(EXPERTS, 2, 1, aux_stream=Mock())
    load_balancer.load_balancer_impl.add_layer.assert_called_once_with(EXPERTS, 2, 1)
    assert layer.spare_slot_count == 0
    assert not load_balancer.adds_spare_slots
    assignments = [0, 1, 2, 3]
    assert layer.with_spare_slots(assignments) is assignments
    layer.set_initial_weight_assignments(assignments)
    assert layer.initial_placement is None


@pytest.mark.parametrize("misuse", ["storage_replaced", "no_host_weights"])
def test_spare_storage_misuse_refused(monkeypatch, misuse):
    """Refuse a weight whose spare slots cannot be restored, for lost storage or no host weights."""
    load_balancer = _load_balancer(
        monkeypatch, largest_failure=2, eplb_slots=1, layer_updates_per_iter=0
    )
    (layer,) = load_balancer.single_layer_load_balancers
    with patch(f"{_ER}.torch") as mock_torch:
        mock_torch.ops.trtllm.moe_unbacked_slots_save_memory.return_value = True
        mock_torch.ops.trtllm.moe_unbacked_slot_tensor.return_value = torch.zeros(2, 4)
        create_spare_slot_storage(layer, "w", torch.zeros(2, 4))
    if misuse == "no_host_weights":
        message = r"Weights \['w'\] of MoE layer .* no host weights"
    else:
        layer.add_to_migrate_weight_fn(use_spare_slot_storage, (layer, torch.zeros(2, 4), "w"))
        message = "Weight w of MoE layer .* lost the storage"
    with pytest.raises(RuntimeError, match=message):
        load_balancer.register_weight_slots_after_to_cuda()


@pytest.mark.parametrize("fault_tolerance", [True, False], ids=["fault_tolerance", "off"])
def test_fault_tolerant_shutdown_skips_node_collectives(monkeypatch, fault_tolerance):
    """Skip the node's allgather and barrier at shutdown under fault tolerance only."""
    load_balancer = _load_balancer(monkeypatch, largest_failure=2, eplb_slots=1)
    (layer,) = load_balancer.single_layer_load_balancers
    # Without fault tolerance, the load balancer's shutdown is MoeLoadBalancer's.
    shutdown = (
        load_balancer.shutdown
        if fault_tolerance
        else functools.partial(MoeLoadBalancer.shutdown, load_balancer)
    )
    node = Mock(allgather=Mock(side_effect=lambda safe: [safe, True]))
    load_balancer.shared_mpi_comm = node
    events = []
    load_balancer.load_balancer_impl.shutdown.side_effect = lambda: events.append("shutdown")
    layer.host_tensor_sharer.pre_shutdown_cleanup.side_effect = lambda: events.append("release")
    load_balancer.in_iter = True
    assert not shutdown()
    assert events == []
    load_balancer.in_iter = False
    assert shutdown()
    assert events == ["shutdown", "release"]
    assert node.allgather.call_count == (0 if fault_tolerance else 2)
    assert node.barrier.call_count == (0 if fault_tolerance else 2)


def _placement(*rows):
    return np.array(rows, dtype=np.int32)


def test_failed_rank_experts_move_and_return():
    """Plan the placement as ranks 3 and 2 fail and then rejoin."""
    initial = _placement([0, 1, -1, -1], [2, 3, -1, -1], [4, 5, -1, -1], [6, 7, -1, -1])

    def plan(current, active, layer=0):
        return plan_layer_placement(current, initial, np.array(active, bool), 2, 8, layer)

    # Rank 3 fails, and its experts go to the spare slots of the ranks with the fewest filled.
    without_3 = plan(initial, [1, 1, 1, 0])
    np.testing.assert_array_equal(
        without_3, _placement([0, 1, 6, -1], [2, 3, 7, -1], [4, 5, -1, -1], [-1] * 4)
    )
    # The next layer starts at the next rank, so that the layers spread the lost experts.
    np.testing.assert_array_equal(
        plan(initial, [1, 1, 1, 0], layer=1),
        _placement([0, 1, -1, -1], [2, 3, 6, -1], [4, 5, 7, -1], [-1] * 4),
    )
    # Rank 2 fails too, and the experts restored before stay where they are.
    without_2_3 = plan(without_3, [1, 1, 0, 0])
    np.testing.assert_array_equal(
        without_2_3, _placement([0, 1, 6, 4], [2, 3, 7, 5], [-1] * 4, [-1] * 4)
    )
    # When rank 2 rejoins, the spare slots holding its experts are emptied.
    np.testing.assert_array_equal(plan(without_2_3, [1, 1, 1, 0]), without_3)
    # With every rank back, the placement is the initial one.
    np.testing.assert_array_equal(plan(without_3, [1, 1, 1, 1]), initial)
    # Without spare slots, lost experts take the active slots of extra replicas.
    replicated = _placement([0, 1], [2, 3], [0, 1])
    np.testing.assert_array_equal(
        plan_layer_placement(replicated, replicated, np.array([1, 0, 1], bool), 2, 4, 0),
        _placement([0, 1], [-1, -1], [2, 3]),
    )
    # Without spare slots or extra replicas, a lost expert has no slot.
    with pytest.raises(ValueError, match=r"active ranks \[0\] cannot hold expert 2"):
        plan_layer_placement(initial[:2, :2], initial[:2, :2], np.array([1, 0], bool), 2, 4, 0)


def test_single_replica_layer_keeps_one_replica_per_expert():
    """Plan a single replica per expert from ranks that an online EPLB update left mixed."""
    # A swap of experts 0 and 2 reached rank 0 only, so rank 1's slot of expert 2 takes expert 0.
    mixed = _placement([2, 1, -1, -1], [2, 3, -1, -1], [4, 5, -1, -1], [6, 7, -1, -1])
    placement = plan_layer_placement(mixed, mixed, np.array([1, 1, 1, 0], bool), 2, 8, 0)
    np.testing.assert_array_equal(
        placement, _placement([2, 1, 6, -1], [0, 3, 7, -1], [4, 5, -1, -1], [-1] * 4)
    )


def test_multi_replica_layer_restores_experts_into_extra_replicas():
    """Plan every expert from ranks that an online EPLB update left mixed in a multi-replica layer."""
    # An update moving expert 2 to rank 1 reached rank 0 only. Four experts lack a replica for
    # three spare slots, so expert 8 takes rank 1's slot of an extra replica of expert 4.
    mixed = _placement([0, 1, 4, -1], [3, 4, 5, -1], [0, 3, 1, -1], [6, 7, 8, -1])
    placement = plan_layer_placement(mixed, mixed, np.array([1, 1, 1, 0], bool), 3, 9, 0)
    np.testing.assert_array_equal(
        placement, _placement([0, 1, 4, 2], [3, 8, 5, 6], [0, 3, 1, 7], [-1] * 4)
    )


def _online_balancer(monkeypatch, current, ep_rank=0):
    """Return an online EPLB load balancer at placement ``current`` with 4 MiB spare slots."""
    load_balancer = _load_balancer(monkeypatch, largest_failure=2, eplb_slots=2, ep_rank=ep_rank)
    (layer,) = load_balancer.single_layer_load_balancers
    layer.set_initial_weight_assignments(layer.with_spare_slots([0, 1, 2, 3, 0, 1, 2, 3]))
    layer.single_layer_load_balancer_impl.get_old_rank_expert_ids.return_value = current.tolist()
    layer.unbacked_slot_tensors = {"w": Mock()}
    monkeypatch.setattr(
        torch.ops.trtllm,
        "moe_unbacked_slots_bytes_to_back",
        lambda tensor, slots: len(slots) * (4 << 20),
        raising=False,
    )
    return load_balancer, layer


# Online EPLB moved the experts away from the initial placement.
_ONLINE = _placement([1, 0, -1, -1], [3, 2, -1, -1], [2, 0, -1, -1], [1, 2, -1, -1])
_ONLINE_WITHOUT_1 = _placement([1, 0, 3, -1], [-1] * 4, [2, 0, -1, -1], [1, 2, -1, -1])


def test_active_ranks_plan_the_same_restore(monkeypatch):
    """Plan the same restore on active ranks with different views of the placement."""
    active = [True, False, True, True]
    # An interrupted online update left rank 2 with another view of the placement than rank 0's.
    views = {
        0: _ONLINE,
        2: _placement([1, 0, -1, -1], [3, 2, -1, -1], [2, 0, -1, -1], [3, 2, -1, -1]),
    }
    plans = [
        _online_balancer(monkeypatch, view, ep_rank)[0].expert_restore.plan_expert_restore(
            active, [_ONLINE]
        )
        for ep_rank, view in views.items()
    ]
    for plan in plans:
        # The active ranks' active slots keep their experts, and expert 3 goes to a spare slot.
        np.testing.assert_array_equal(plan.placements[0], _ONLINE_WITHOUT_1)
    assert plans[0].digest == plans[1].digest
    # Rank 0 backs one 4 MiB slot and rank 2 backs none.
    assert [plan.device_bytes for plan in plans] == [4 << 20, 0]
    other = _online_balancer(monkeypatch, _ONLINE)[0].expert_restore.plan_expert_restore(
        [True, True, False, True], [_ONLINE]
    )
    assert other.digest != plans[0].digest


def test_release_keeps_filled_spare_slots_during_update(monkeypatch):
    """Release only the empty spare slots while an online EPLB update swaps the placement."""
    # The old placement reads -1 during the swap, so only the unusable slots tell 2 from 3.
    load_balancer, layer = _online_balancer(monkeypatch, _placement(*[[-1] * 4] * WORLD_SIZE))
    # Active slot 1 is unusable too: a stopped restore emptied it, and another replica kept its
    # expert, so the next plan left it empty.
    layer.single_layer_load_balancer_impl.get_unusable_local_slots.return_value = [1, 3]
    released = []

    def set_backed(tensors, slots, slot_counts, backed):
        assert not backed
        # Like moe_unbacked_slots_set_backed, refuse a slot that is not a spare slot.
        assert all(slot in layer.spare_slots for slot in slots), slots
        released.extend(slots)
        return len(slots) * (4 << 20)

    monkeypatch.setattr(
        torch.ops.trtllm, "moe_unbacked_slots_set_backed", set_backed, raising=False
    )
    restore = load_balancer.expert_restore
    assert restore.empty_spare_slots() == [[3]]
    assert restore.release_spare_slots(restore.empty_spare_slots()) == 4 << 20
    assert released == [3]


def test_release_unmaps_a_few_layers_per_call(monkeypatch):
    """Release empty spare slots a few layers per call, leaving the backed bytes to the caller."""
    load_balancer, layer = _online_balancer(monkeypatch, _ONLINE)
    monkeypatch.setattr(load_balancer, "single_layer_load_balancers", [layer] * 3)
    monkeypatch.setattr(f"{_ER}._RELEASE_LAYERS_PER_CALL", 2)
    calls = []

    def set_backed(tensors, slots, slot_counts, backed):
        assert not backed
        calls.append(slot_counts)
        return len(slots) * (4 << 20)

    monkeypatch.setattr(
        torch.ops.trtllm, "moe_unbacked_slots_set_backed", set_backed, raising=False
    )
    restore = load_balancer.expert_restore
    assert restore.release_spare_slots([[2, 3], [], [3]]) == 3 * (4 << 20)
    # The first call takes layers 0 and 1, the second layer 2.
    assert calls == [[2], [1]]
    assert restore.backed_spare_slot_bytes == 0


def test_restore_reads_no_placement_while_an_update_may_swap_it(monkeypatch):
    """Refuse even a restore that changes nothing before reading the placement an update swaps."""
    load_balancer, layer = _online_balancer(monkeypatch, _ONLINE)
    restore = load_balancer.expert_restore
    # Every rank is active and keeps its experts, so the plan loads no slot.
    plan = restore.plan_expert_restore([True] * WORLD_SIZE, [_ONLINE])
    np.testing.assert_array_equal(plan.placements[0], _ONLINE)
    load_balancer.load_balancer_impl.check_slots_writable.side_effect = RuntimeError(
        "call abandonIterations first"
    )
    old_rank_expert_ids = layer.single_layer_load_balancer_impl.get_old_rank_expert_ids
    old_rank_expert_ids.reset_mock()
    with pytest.raises(RuntimeError, match="abandonIterations"):
        restore.restore_lost_experts(plan)
    old_rank_expert_ids.assert_not_called()


@pytest.mark.parametrize("ep_rank, largest", [(0, 8 << 20), (2, 4 << 20)])
def test_largest_failure_restore_bytes_skips_failures_that_lose_this_rank(
    monkeypatch, ep_rank, largest
):
    """Size the restore of each recoverable failure that this rank survives, at its placement."""
    load_balancer, _ = _online_balancer(monkeypatch, _ONLINE, ep_rank=ep_rank)
    restore = load_balancer.expert_restore
    # The recoverable failures of nodes [0, 1] and [2, 3] lose node 1 or rank 1.
    device_bytes = {(2, 3): 8 << 20, (1,): 4 << 20}

    def plan_expert_restore(active_ranks, current):
        np.testing.assert_array_equal(current[0], _ONLINE)
        lost = tuple(rank for rank, active in enumerate(active_ranks) if not active)
        return SimpleNamespace(device_bytes=device_bytes[lost])

    monkeypatch.setattr(restore, "plan_expert_restore", plan_expert_restore)
    assert restore.largest_failure_restore_bytes() == largest


def test_current_placement_hides_invalid_slots(monkeypatch):
    """Show no expert in a slot of this rank that a stopped restore may have written in part."""
    load_balancer, _ = _online_balancer(monkeypatch, _ONLINE, ep_rank=2)
    restore = load_balancer.expert_restore
    assert restore.current_placement() == [(EXPERTS, _ONLINE.tolist())]
    restore._invalid_slots[0] = {1}
    expected = _ONLINE.copy()
    expected[2, 1] = -1
    assert restore.current_placement() == [(EXPERTS, expected.tolist())]


class _SpareSlotOwner(MoEWeightOwnerMixin, MoEEplbWeightLayoutMixin):
    """A MoE layer's weight owner with the quantization method of the MoE backend ``leaf``."""

    def __init__(self, leaf, spare_slot_count):
        self.layer_load_balancer = SimpleNamespace(spare_slot_count=spare_slot_count)
        self.method = leaf.__new__(leaf)._get_quant_method()
        self.method.create_weights = Mock()

    def _get_quant_method(self):
        return self.method


@pytest.mark.parametrize("spare_slot_count", [2, 0])
def test_spare_slot_weights_created_after_the_expert_weights(monkeypatch, spare_slot_count):
    """Create the spare slots' weights after the quantization method's, only with spare slots."""
    create_spare_slot_weights = Mock()
    monkeypatch.setattr(f"{_ER}.create_spare_slot_weights", create_spare_slot_weights)
    owner = _SpareSlotOwner(TrtllmCutlassNvfp4Impl, spare_slot_count)
    owner.create_weights()
    owner.method.create_weights.assert_called_once_with(owner)
    assert create_spare_slot_weights.call_count == (1 if spare_slot_count else 0)


@pytest.mark.parametrize(
    "leaf, quant, sm, rejected",
    [
        (TrtllmCutlassNvfp4Impl, "NVFP4", 100, False),
        (TrtllmCutlassUnquantizedImpl, None, 100, False),
        (TrtllmCutedslGroupedGemmNvfp4Impl, "NVFP4", 100, False),
        # The SM107 (Rubin) kernels tune every local slot.
        (TrtllmCutedslGroupedGemmNvfp4Impl, "NVFP4", 107, True),
        (TrtllmCutlassW4a16Nvfp4Impl, "W4A16_NVFP4", 100, True),
        (DeepgemmCudaHopperFp8BlockScalesImpl, "FP8_BLOCK_SCALES", 90, True),
    ],
)
def test_spare_slots_checked_in_can_implement(leaf, quant, sm, rejected):
    """A backend that would autotune or allocate for the spare slots declines them."""
    import dataclasses

    from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
        MoEDeployment,
        MoEEnvironment,
        MoEProblem,
        MoERejectReason,
    )
    from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep

    problem = MoEProblem(quant=quant, dtype_act=torch.bfloat16)
    deployment = MoEDeployment(
        ep_size=4,
        tp_size=1,
        parallel_size=4,
        use_dp=True,
        num_slots=8,
        env=MoEEnvironment(sm=sm, available_deps=(MoEDep.CUTEDSL_RUBIN.value,)),
        eplb_enabled=True,
        eplb_mode="standard",
    )
    assert leaf.can_implement(problem, deployment).eligible

    verdict = leaf.can_implement(problem, dataclasses.replace(deployment, spare_slots_enabled=True))
    assert verdict.eligible is not rejected
    if rejected:
        assert verdict.reject_reason is MoERejectReason.SPARE_SLOTS_UNSUPPORTED
