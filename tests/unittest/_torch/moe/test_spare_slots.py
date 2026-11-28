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
import os
import secrets
import signal
import subprocess  # nosec B404
import sys
import weakref
from contextlib import nullcontext
from multiprocessing import resource_tracker, shared_memory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.distributed.membership import (
    REPLACEMENT_CANNOT_START,
    SHARED_SECRET_ENV,
    MembershipRecord,
    MembershipState,
    ReplacementCannotStart,
    host_weights_name,
    startup,
)
from tensorrt_llm._torch.distributed.startup_record import StartupRecord
from tensorrt_llm._torch.moe.fused_moe.cutedsl import (
    TrtllmCutedslGroupedGemmBf16Impl,
    TrtllmCutedslGroupedGemmNvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.cutlass import (
    DeepgemmCudaHopperFp8BlockScalesImpl,
    TrtllmCutlassNvfp4Impl,
    TrtllmCutlassUnquantizedImpl,
    TrtllmCutlassW4a16Nvfp4Impl,
)
from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import (
    DeploymentHostWeights,
    EplbExpertRestore,
    FaultTolerantMoeLoadBalancer,
    create_spare_slot_storage,
    create_spare_slot_weights,
    plan_layer_placement,
    use_spare_slot_storage,
)
from tensorrt_llm._torch.moe.fused_moe.impl_blocks import (
    MoEEplbWeightLayoutMixin,
    MoEWeightOwnerMixin,
)
from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import HostMoeTensorSharer, MoeLoadBalancer
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    UnquantizedFusedMoEMethod,
    W4A16NVFP4CutlassFusedMoEMethod,
)
from tensorrt_llm._torch.pyexecutor.rank_failure_model_load import create_moe_load_balancer
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_ER = "tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore"
_MLB = "tensorrt_llm._torch.moe.fused_moe.moe_load_balancer"
_LOAD = "tensorrt_llm._torch.pyexecutor.rank_failure_model_load"
_USES_GDRCOPY = "tensorrt_llm.bindings.internal.runtime.host_accessible_device_memory_uses_gdrcopy"
WORLD_SIZE = 4
EXPERTS = 4


def _deployment():
    """Return the arguments of an original rank with a fresh shared memory name."""
    return {
        "mapping": None,
        "shared_memory_base_name": f"test_host_weights_{os.getpid()}_{secrets.token_hex(8)}",
        "replacement": False,
        "wait_for_host_weights": Mock(),
    }


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
    # No node keeps host weights of an expert, so any rank may take a lost one.
    anywhere = np.zeros((4, 8), bool)

    def plan(current, active, layer=0):
        return plan_layer_placement(current, np.array(active, bool), 2, 8, anywhere, layer)

    def rejoin(current, rank):
        """Return ``current`` with ``rank``'s row of the initial placement, which it loaded."""
        current = current.copy()
        current[rank] = initial[rank]
        return current

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
    np.testing.assert_array_equal(plan(rejoin(without_2_3, 2), [1, 1, 1, 0]), without_3)
    # With every rank back, the placement is the initial one.
    np.testing.assert_array_equal(plan(rejoin(without_3, 3), [1, 1, 1, 1]), initial)
    # Without spare slots, lost experts take the active slots of extra replicas.
    replicated = _placement([0, 1], [2, 3], [0, 1])
    np.testing.assert_array_equal(
        plan_layer_placement(
            replicated, np.array([1, 0, 1], bool), 2, 4, np.zeros((3, 4), bool), 0
        ),
        _placement([0, 1], [-1, -1], [2, 3]),
    )
    # Without spare slots or extra replicas, a lost expert has no slot.
    with pytest.raises(ValueError, match=r"active ranks \[0\] cannot hold expert 2"):
        plan_layer_placement(initial[:2, :2], np.array([1, 0], bool), 2, 4, anywhere[:2, :4], 0)


def test_single_replica_layer_keeps_one_replica_per_expert():
    """Plan a single replica per expert from ranks that an online EPLB update left mixed."""
    # A swap of experts 0 and 2 reached rank 0 only, so rank 1's slot of expert 2 takes expert 0.
    mixed = _placement([2, 1, -1, -1], [2, 3, -1, -1], [4, 5, -1, -1], [6, 7, -1, -1])
    placement = plan_layer_placement(
        mixed, np.array([1, 1, 1, 0], bool), 2, 8, np.ones((4, 8), bool), 0
    )
    np.testing.assert_array_equal(
        placement, _placement([2, 1, 6, -1], [0, 3, 7, -1], [4, 5, -1, -1], [-1] * 4)
    )


def test_multi_replica_layer_restores_experts_into_extra_replicas():
    """Plan every expert from ranks that an online EPLB update left mixed in a multi-replica layer."""
    # An update moving expert 2 to rank 1 reached rank 0 only. Four experts lack a replica for
    # three spare slots, so expert 8 takes rank 1's slot of an extra replica of expert 4.
    mixed = _placement([0, 1, 4, -1], [3, 4, 5, -1], [0, 3, 1, -1], [6, 7, 8, -1])
    placement = plan_layer_placement(
        mixed, np.array([1, 1, 1, 0], bool), 3, 9, np.ones((4, 9), bool), 0
    )
    np.testing.assert_array_equal(
        placement, _placement([0, 1, 4, 2], [3, 8, 5, 6], [0, 3, 1, 7], [-1] * 4)
    )


@pytest.mark.parametrize(
    "lost, rejoining_rank, rejoined, expected, expert_count",
    [
        # Multi-replica layer: online EPLB left expert 0 only in rank 0's spare slot while rank 1
        # was lost. Rank 1 rejoins with experts 2 and 3, and expert 0 takes its slot of the extra
        # replica of expert 3.
        (
            _placement([1, 3, 0, -1], [-1] * 4, [3, 1, 2, -1], [2, 1, -1, -1]),
            1,
            [2, 3, -1, -1],
            _placement([1, 3, -1, -1], [2, 0, -1, -1], [3, 1, -1, -1], [2, 1, -1, -1]),
            4,
        ),
        # Single-replica layer: online EPLB swapped experts 4 and 6 on rank 2 while rank 3 was
        # lost. Rank 3 rejoins with experts 6 and 7, and expert 4 takes its slot of expert 6.
        (
            _placement([0, 1, -1, -1], [2, 3, 7, -1], [6, 5, 4, -1], [-1] * 4),
            3,
            [6, 7, -1, -1],
            _placement([0, 1, -1, -1], [2, 3, -1, -1], [6, 5, -1, -1], [4, 7, -1, -1]),
            8,
        ),
    ],
    ids=["multi_replica", "single_replica"],
)
def test_full_rejoin_keeps_survivors_and_empties_spare_slots(
    lost, rejoining_rank, rejoined, expected, expert_count
):
    """Keep the survivors' active slots on a full rejoin, and put lost experts in active slots."""
    current = lost.copy()
    current[rejoining_rank] = rejoined
    placement = plan_layer_placement(
        current, np.ones(4, bool), 2, expert_count, np.ones((4, expert_count), bool), 0
    )
    np.testing.assert_array_equal(placement, expected)
    # The survivors keep their active slots, the rejoining rank changes only the slot that takes
    # the lost expert, every spare slot is empty and every expert has a slot.
    survivors = [rank for rank in range(4) if rank != rejoining_rank]
    np.testing.assert_array_equal(placement[survivors, :2], current[survivors, :2])
    assert np.count_nonzero(placement[rejoining_rank] != current[rejoining_rank]) == 1
    assert (placement[:, 2:] == -1).all()
    assert set(placement.ravel().tolist()) - {-1} == set(range(expert_count))


def test_partial_rejoin_keeps_lost_experts_in_spare_slots():
    """Keep the experts of a rank still lost in spare slots when another lost rank rejoins."""
    # Online EPLB moved experts on ranks 0 and 1 after ranks 2 and 3 were lost, and rank 2
    # rejoins with its experts 4 and 5 of the initial placement.
    current = _placement([1, 0, 6, 4], [3, 2, 7, 5], [4, 5, -1, -1], [-1] * 4)
    placement = plan_layer_placement(
        current, np.array([1, 1, 1, 0], bool), 2, 8, np.zeros((4, 8), bool), 0
    )
    # Rank 2's experts leave the spare slots, and rank 3's stay there.
    np.testing.assert_array_equal(
        placement, _placement([1, 0, 6, -1], [3, 2, 7, -1], [4, 5, -1, -1], [-1] * 4)
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


def test_rejoin_loads_only_lost_experts_after_online_updates(monkeypatch):
    """Load only the lost experts on a rejoin, keeping the active slots online EPLB filled."""
    # Online EPLB moved experts while rank 1 was lost, and left expert 0 only in a spare slot.
    online = _placement([1, 3, 0, -1], [-1] * 4, [3, 1, 2, -1], [2, 1, -1, -1])
    load_balancer, _ = _online_balancer(monkeypatch, online)
    # Rank 1 rejoins with its experts 2 and 3 of the initial placement.
    current = online.copy()
    current[1] = [2, 3, -1, -1]
    plan = load_balancer.expert_restore.plan_expert_restore([True] * WORLD_SIZE, [current])
    (placement,), (before,) = plan.placements, plan.placement_before
    # Expert 0 takes rank 1's slot of the extra replica of expert 3. The survivors load nothing,
    # and every spare slot empties.
    np.testing.assert_array_equal(
        placement, _placement([1, 3, -1, -1], [2, 0, -1, -1], [3, 1, -1, -1], [2, 1, -1, -1])
    )
    np.testing.assert_array_equal(np.argwhere((placement >= 0) & (placement != before)), [[1, 1]])
    assert plan.staged.size == 0
    assert plan.device_bytes == 0


@pytest.mark.parametrize("initial", [[0, 1, 2, 3], [2, 0, 3, 1]], ids=["none", "offline"])
def test_static_rejoin_restores_initial_placement(monkeypatch, initial):
    """Plan the initial placement when a lost node rejoins under static EPLB."""
    load_balancer = _load_balancer(
        monkeypatch, largest_failure=2, eplb_slots=1, layer_updates_per_iter=0
    )
    (layer,) = load_balancer.single_layer_load_balancers
    layer.set_initial_weight_assignments(layer.with_spare_slots(initial))
    initial_placement = layer.initial_placement.reshape(WORLD_SIZE, -1)
    restore = load_balancer.expert_restore
    # Node 1's experts go to the spare slots of node 0, which keeps their host weights.
    without_node_1 = restore.plan_expert_restore([True, True, False, False], [initial_placement])
    current = without_node_1.placements[0].copy()
    assert sorted(current[:2, 1].tolist()) == sorted(initial[2:])
    # Ranks 2 and 3 rejoin with their experts of the initial placement.
    current[2:] = initial_placement[2:]
    rejoin = restore.plan_expert_restore([True] * WORLD_SIZE, [current])
    np.testing.assert_array_equal(rejoin.placements[0], initial_placement)
    assert rejoin.staged.size == 0


def test_static_rejoin_restores_extra_replicas(monkeypatch):
    """Restore the initial extra replicas when lost experts took their slots under static EPLB."""
    load_balancer = _load_balancer(
        monkeypatch, largest_failure=2, eplb_slots=2, layer_updates_per_iter=0
    )
    (layer,) = load_balancer.single_layer_load_balancers
    layer.set_initial_weight_assignments(layer.with_spare_slots([0, 1, 0, 1, 2, 3, 2, 3]))
    initial_placement = layer.initial_placement.reshape(WORLD_SIZE, -1)
    # While node 1 was lost, its experts 2 and 3 took the slots of rank 1's extra replicas.
    current = initial_placement.copy()
    current[1, :2] = [2, 3]
    rejoin = load_balancer.expert_restore.plan_expert_restore([True] * WORLD_SIZE, [current])
    (placement,), (before,) = rejoin.placements, rejoin.placement_before
    np.testing.assert_array_equal(placement, initial_placement)
    np.testing.assert_array_equal(np.argwhere(placement != before), [[1, 0], [1, 1]])


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
        restore.restore_lost_experts(
            plan,
            torch.empty((WORLD_SIZE, 0), dtype=torch.uint8),
            lambda value: [value] * WORLD_SIZE,
        )
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


def _original(monkeypatch):
    """Make this process an original rank, whose startup record ``Distributed`` holds."""
    monkeypatch.setattr(Distributed, "startup_record", StartupRecord())


def _replacement(monkeypatch):
    _original(monkeypatch)
    monkeypatch.setattr(StartupRecord, "replacement", True)


@pytest.fixture
def tracked(monkeypatch):
    """Yield the shared memory names the resource tracker would unlink at exit, then unlink all."""
    names = set()
    monkeypatch.setattr(
        resource_tracker, "register", lambda name, rtype: names.add(name.removeprefix("/"))
    )
    monkeypatch.setattr(
        resource_tracker, "unregister", lambda name, rtype: names.discard(name.removeprefix("/"))
    )
    yield names
    for name in _shm_names():
        os.unlink(f"/dev/shm/{name}")


def _sharer(rank, value, node_size=1, dtype=torch.float32, shape=(4,), base_name=None):
    """Return the host weights of local ``rank``'s two experts, each ``value`` plus its id."""
    replay = Distributed.startup_record
    sharer = DeploymentHostWeights(
        0,
        2 * node_size,
        Mock(Get_rank=Mock(return_value=rank), Get_size=Mock(return_value=node_size)),
        replacement=replay is not None and replay.replacement,
    )
    sharer.set_shared_memory_base_name(base_name or f"test_host_weights_{os.getpid()}")
    sharer.reattach_own_shm({"w": (dtype, torch.Size(shape))})
    for expert_id in range(2 * node_size):
        if expert_id in sharer.own_expert_ids:
            sharer.share_host_tensor_with_shape(
                expert_id, "w", torch.full(shape, value + expert_id, dtype=dtype)
            )
        else:
            sharer.pre_register_host_tensor_with_shape(expert_id, "w", dtype, torch.Size(shape))
    return sharer


def _shm_names():
    return sorted(
        name
        for name in os.listdir("/dev/shm")
        if name.startswith(f"test_host_weights_{os.getpid()}")
    )


def test_host_weights_round_trip_by_name(monkeypatch, tracked):
    """Create and reattach the host weights under the name the node supervisor removes."""
    monkeypatch.setenv("TRTLLM_EPLB_SHM_NAME", f"test_host_weights_{os.getpid()}")
    monkeypatch.setenv(SHARED_SECRET_ENV, secrets.token_hex(32))
    monkeypatch.setattr(EplbExpertRestore, "_rank_nodes", lambda self: [0, 0, 1, 1])
    _original(monkeypatch)
    with (
        patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"),
        patch(
            f"{_LOAD}.maybe_create_moe_load_balancer",
            lambda config, mapping, cls: cls(0, WORLD_SIZE, 0),
        ),
    ):
        load_balancer = create_moe_load_balancer(
            SimpleNamespace(mapping=SimpleNamespace(node_rank=0)), None
        )
    base_name = load_balancer.shared_memory_base_name
    original = _sharer(0, 1.0, base_name=base_name)
    original.finalize_layer_weights()
    assert original.get_shared_memory_name().startswith(f"{host_weights_name(0)}_")
    _replacement(monkeypatch)
    replacement = _sharer(0, -1.0, base_name=base_name)
    assert replacement.reattached_own_shm
    assert replacement.own_shm.name == original.own_shm.name
    # They belong to the deployment, even if the original or the replacement fails.
    assert not tracked
    original.pre_shutdown_cleanup()
    replacement.pre_shutdown_cleanup()


def _import_own_host_weights(sharer):
    """Finalize ``sharer`` and return the host weights it imports by expert."""
    sharer.finalize_layer_weights()
    host_weights = {}
    sharer.finalize_host_tensor_sharing(
        lambda expert_id, name, t: host_weights.update({expert_id: t.tolist()})
    )
    sharer.pre_shutdown_cleanup()
    return host_weights


@pytest.mark.parametrize(
    "found", ["missing", "incomplete", "torn", "other host weights", "open to other users"]
)
def test_replacement_rebuilds_damaged_host_weights(monkeypatch, tracked, found):
    """Refill a replacement's host weights unless they are intact and its deployment's own."""
    if found != "missing":
        # Other host weights of the same size.
        other = found == "other host weights"
        original = _sharer(
            0, 1.0, dtype=torch.float64 if other else torch.float32, shape=(2 if other else 4,)
        )
        original.finalize_layer_weights()
        if found == "incomplete":
            # The process died before writing the fingerprint.
            original.own_shm.buf[:32] = bytes(32)
        elif found == "torn":
            # The process died while writing the fingerprint.
            original.own_shm.buf[16:32] = bytes(16)
        elif found == "open to other users":
            os.fchmod(original.own_shm._fd, 0o666)
            assert not original.holds_host_weights(0)
        original.pre_shutdown_cleanup()
    _replacement(monkeypatch)
    replacement = _sharer(0, -1.0)
    assert not replacement.reattached_own_shm
    replacement.finalize_layer_weights()
    assert replacement.host_weights[(1, "w")].tolist() == [0.0] * 4
    assert replacement.holds_host_weights(0)
    assert not tracked
    replacement.pre_shutdown_cleanup()
    replacement.post_shutdown_cleanup()
    assert _shm_names() == [replacement.get_shared_memory_name()]
    successor = _sharer(0, -2.0)
    assert successor.reattached_own_shm
    assert _import_own_host_weights(successor) == {0: [-1.0] * 4, 1: [0.0] * 4}


def _attach_then(change):
    """Patch the attach of shared memory to call ``change`` right after the open."""
    attach = shared_memory.SharedMemory

    def attach_then_change(*args, **kwargs):
        shm = attach(*args, **kwargs)
        change()
        return shm

    return patch(f"{_ER}.shared_memory.SharedMemory", attach_then_change)


@pytest.mark.parametrize("change", ["removed", "swapped"])
def test_host_weights_changed_while_attaching_are_not_held(tracked, change):
    """Check the opened host weights, not the file at their path, when another rank changes it."""
    original = _sharer(0, 1.0)
    original.finalize_layer_weights()
    assert original.holds_host_weights(0)
    path = f"/dev/shm/{original.get_shared_memory_name()}"

    def swap():
        # The link keeps the opened segment linked, so only its owner and mode reject it.
        os.fchmod(original.own_shm._fd, 0o666)
        os.link(path, f"{path}_kept")
        os.unlink(path)
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600))

    with _attach_then(swap if change == "swapped" else lambda: os.unlink(path)):
        assert not original.holds_host_weights(0)
    original.pre_shutdown_cleanup()


def test_reattached_host_weights_outlive_their_name(monkeypatch, tracked):
    """Import reattached host weights after their name is reused, and release checkpoint weights."""
    original = _sharer(0, 1.0)
    original.finalize_layer_weights()
    original.pre_shutdown_cleanup()
    _replacement(monkeypatch)
    replacement = _sharer(0, -1.0)
    assert replacement.reattached_own_shm
    paged_out = []
    monkeypatch.setattr(f"{_MLB}.advise_tensor_pageout", lambda t: paged_out.append(t.data_ptr()))
    checkpoint_weight = torch.ones(4)
    checkpoint_weight_ref = weakref.ref(checkpoint_weight)
    replacement.add_raw_host_weight_for_unmap([checkpoint_weight])
    paged_out_expected = [checkpoint_weight.data_ptr()]
    del checkpoint_weight
    # Another replacement of the rank rebuilds them meanwhile.
    path = f"/dev/shm/{replacement.get_shared_memory_name()}"
    size = os.stat(path).st_size
    os.unlink(path)
    rebuilt = shared_memory.SharedMemory(
        name=replacement.get_shared_memory_name(), create=True, size=size
    )
    assert _import_own_host_weights(replacement) == {0: [1.0] * 4, 1: [2.0] * 4}
    assert paged_out == paged_out_expected
    assert checkpoint_weight_ref() is None
    rebuilt.close()


_REBUILD_AND_DIE = """
import os
import signal
import sys
from unittest.mock import Mock

import torch

from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import DeploymentHostWeights

sharer = DeploymentHostWeights(
    0, 2, Mock(Get_rank=Mock(return_value=0), Get_size=Mock(return_value=1)), replacement=True
)
sharer.set_shared_memory_base_name(sys.argv[1])
assert not sharer.reattach_own_shm({"w": (torch.float32, torch.Size([4]))})
for expert_id in range(2):
    sharer.share_host_tensor_with_shape(expert_id, "w", torch.full((4,), 1.0 + expert_id))
sharer.finalize_layer_weights()
os.kill(os.getpid(), signal.SIGKILL)
"""


def test_host_weights_of_a_failed_replacement_outlive_it(monkeypatch, tracked):
    """Keep the host weights a replacement wrote when it dies before it joins, for the next one."""
    # The replacement's own resource tracker unlinks what it leaves registered, before the run
    # returns. The replacement uses neither MPI nor a GPU.
    writer = subprocess.run(  # nosec B603
        [sys.executable, "-c", _REBUILD_AND_DIE, f"test_host_weights_{os.getpid()}"],
        env=dict(
            os.environ,
            TRT_LLM_NO_LIB_INIT="1",
            MPI4PY_RC_INITIALIZE="false",
            CUDA_VISIBLE_DEVICES="",
        ),
        stderr=subprocess.PIPE,
        timeout=300,
    )
    assert writer.returncode == -signal.SIGKILL, writer.stderr.decode()
    _replacement(monkeypatch)
    successor = _sharer(0, -1.0)
    assert successor.reattached_own_shm
    assert _import_own_host_weights(successor) == {0: [1.0] * 4, 1: [2.0] * 4}


class _Moe(MoEEplbWeightLayoutMixin, torch.nn.Module):
    """An unquantized MoE layer of EP rank 0 with experts 0 and 1 in its active and spare slot."""

    num_experts = EXPERTS
    expert_size_per_partition = 2
    bias = False
    tp_size = 1
    tp_rank = 0

    def __init__(self, layer_load_balancer):
        super().__init__()
        self.layer_load_balancer = layer_load_balancer
        self.initial_local_expert_ids = [0, 1]
        self.w3_w1_weight = torch.nn.Parameter(torch.empty(2, 4, 2), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(torch.empty(2, 2, 2), requires_grad=False)


def _load_moe(monkeypatch, replacement):
    """Load ``_Moe`` on a node's one rank and return the loaded experts and the host weights."""
    _fault_tolerance(monkeypatch, 1)
    deployment = dict(
        _deployment(),
        shared_memory_base_name=f"test_host_weights_{os.getpid()}_moe",
        replacement=replacement,
    )
    with patch("tensorrt_llm.bindings.internal.runtime.MoeLoadBalancer"):
        load_balancer = FaultTolerantMoeLoadBalancer(0, WORLD_SIZE, 0, **deployment)
        load_balancer.load_balancer_impl.add_layer.return_value.get_layer_id.return_value = 0
        layer = load_balancer.add_layer(EXPERTS, 2, 1, aux_stream=Mock())
        layer.set_initial_weight_assignments(layer.with_spare_slots(list(range(EXPERTS))))
        module = _Moe(layer)
    # The slots keep their storage, as on a system where unbacked spare slots save no memory.
    monkeypatch.setattr(f"{_ER}.create_spare_slot_storage", lambda layer, name, slots: None)
    create_spare_slot_weights(module)
    checkpoint = {
        f"{expert_id}.{name}.weight": torch.full((2, 2), 10.0 * expert_id + index)
        for expert_id in range(EXPERTS)
        for index, name in enumerate(["w1", "w2", "w3"])
    }
    method = UnquantizedFusedMoEMethod()
    load = Mock(wraps=method.load_expert_weights_to_dst)
    monkeypatch.setattr(method, "load_expert_weights_to_dst", load)
    method.load_weights(module, checkpoint, MoEWeightLoadingMode.VANILLA)
    host_weights = {}
    module.layer_load_balancer.host_tensor_sharer.finalize_host_tensor_sharing(
        lambda expert_id, name, t: host_weights.update({(expert_id, name): t.clone()})
    )
    assert load_balancer.shutdown()
    # The fourth argument of a load is the experts it loads.
    return [expert for call in load.call_args_list for expert in call.args[3]], host_weights


def test_replacement_reattaching_host_weights_loads_none(monkeypatch, tracked):
    """Load no host weights on a replacement that reattaches them, and restore the same."""
    loaded, host_weights = _load_moe(monkeypatch, replacement=False)
    # Rank 0's node keeps the host weights of experts 2 and 3, which the next node's ranks hold.
    assert loaded == [0, 1, 2, 3]
    assert sorted(host_weights) == [
        (expert_id, name) for expert_id in (2, 3) for name in ["w2_weight", "w3_w1_weight"]
    ]
    assert host_weights[(3, "w3_w1_weight")].flatten().tolist() == [32.0] * 4 + [30.0] * 4
    _replacement(monkeypatch)
    loaded, reattached_host_weights = _load_moe(monkeypatch, replacement=True)
    assert loaded == [0, 1]
    assert reattached_host_weights.keys() == host_weights.keys()
    for key, t in host_weights.items():
        assert torch.equal(reattached_host_weights[key], t), key


class _ReplacementLayer(MoEEplbWeightLayoutMixin, torch.nn.Module):
    """A MoE layer of a replacement that reattached its node's host weights, so it loads none."""

    num_experts = EXPERTS
    expert_size_per_partition = 2

    def __init__(self):
        super().__init__()
        comm = Mock(Get_rank=Mock(return_value=0), Get_size=Mock(return_value=1))
        self.sharer = HostMoeTensorSharer(0, EXPERTS, comm)
        self.sharer.keep_experts([])
        self.layer_load_balancer = SimpleNamespace(
            spare_slot_count=1,
            get_load_expert_ids=lambda: self.sharer.own_expert_ids,
            host_tensor_sharer=self.sharer,
            add_to_migrate_weight_fn=Mock(),
        )

    def _using_dynamic_load_balancer(self) -> bool:
        return False


def test_zero_row_host_weights_register_shapes():
    """A replacement's host weights have no rows, and their shapes are registered all the same."""
    layer = _ReplacementLayer()
    layer.register_all_parameter_slot_and_to_fix_weight_fns({"w": torch.empty(0, 2, 3)})
    assert layer.sharer.name_info == {"w": (torch.float32, torch.Size([2, 3]))}


def test_replacement_nvfp4_registers_no_host_rows():
    """NVFP4 weight processing of a replacement registers its empty host scales and alphas."""
    method = W4A16NVFP4CutlassFusedMoEMethod()
    method.block_scales_vec_size = 4
    module = _ReplacementLayer()
    module.tp_size, module.tp_rank = 1, 0
    module.initial_local_expert_ids = [2, 0]
    module.w3_w1_weight_scale = torch.nn.Parameter(
        torch.zeros(2, 128, 4, dtype=torch.int32), requires_grad=False
    )
    module.w2_weight_scale = torch.nn.Parameter(
        torch.zeros(2, 256, 2, dtype=torch.int32), requires_grad=False
    )
    module.local_shared_w3_w1_scale_tensors = torch.zeros(0, 128, 4, dtype=torch.int32)
    module.local_shared_w2_scale_tensors = torch.zeros(0, 256, 2, dtype=torch.int32)
    module.fc31_input_scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)
    module.fc2_input_scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)
    module.fc31_alpha = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    module.fc2_alpha = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    module.tmp_raw_input_scales = {
        i: {name: torch.tensor(1.0) for name in ("w1", "w3", "w2")} for i in range(EXPERTS)
    }
    module.tmp_weight_scale_2 = {
        slot: {"w1": torch.tensor(1.0), "w3": torch.tensor(1.0), "w2": torch.tensor(2.0)}
        for slot in range(2)
    }
    module.tmp_shared_weight_scale_2 = {}
    for slot in range(2):
        w1 = torch.ones(128, 16).to(torch.float8_e4m3fn)
        w2 = torch.ones(256, 8).to(torch.float8_e4m3fn)
        method.load_expert_w3_w1_weight_scale_nvfp4(
            module, w1, None, module.w3_w1_weight_scale[slot], expert_idx=slot
        )
        method.load_expert_w2_weight_scale_nvfp4(module, w2, module.w2_weight_scale[slot])

    method.process_weights_after_loading(module)

    assert module.sharer.name_info == {
        name: (getattr(module, name).dtype, getattr(module, name).shape[1:])
        for name in ("fc31_alpha", "fc2_alpha", "w3_w1_weight_scale", "w2_weight_scale")
    }


def test_plan_after_stopped_restore_names_loaded_experts(monkeypatch):
    """Replan from current_placement after a stopped restore, so every slot holds what it names."""
    # Online EPLB left two replicas of expert 1 on rank 0 while rank 1 was lost, and expert 0 only
    # in rank 0's spare slot.
    online = _placement([1, 1, 0, -1], [-1] * 4, [2, 3, -1, -1], [1, 2, -1, -1])
    load_balancer, layer = _online_balancer(monkeypatch, online)
    restore = load_balancer.expert_restore
    impl = load_balancer.load_balancer_impl
    host = {(expert, "w"): torch.full((4,), expert, dtype=torch.int32) for expert in range(EXPERTS)}
    layer.host_tensor_sharer.node_host_weights = host
    # Rank 0's slots hold the experts of the installed placement; slot 3 is empty.
    slots = torch.tensor([[expert] * 4 for expert in online[0].tolist()], dtype=torch.int32)
    layer.slot_tensors = {"w": slots}
    monkeypatch.setattr(
        torch.ops.trtllm,
        "moe_unbacked_slots_set_backed",
        lambda tensors, slot_ids, slot_counts, backed: 0,
        raising=False,
    )

    # Rank 1 rejoins with its experts 2 and 3 of the initial placement, so rank 0's slot of the
    # extra replica of expert 1 takes expert 0. A newer membership change stops the restore once
    # it loaded that slot, before it installed the placement.
    def rank_1_rejoins(current):
        current = np.array(current, dtype=np.int32)
        current[1] = [2, 3, -1, -1]
        return current

    rejoin = restore.plan_expert_restore([True] * WORLD_SIZE, [rank_1_rejoins(online)])
    assert rejoin.placements[0][0].tolist() == [1, 0, -1, -1]
    # Every rank keeps the host weights of online EPLB, so the restore stages nothing.
    workspace = torch.empty((WORLD_SIZE, 0), dtype=torch.uint8)

    def allgather(value):
        return [value] * WORLD_SIZE

    with pytest.raises(RuntimeError, match="newer membership change"):
        restore.restore_lost_experts(
            rejoin,
            workspace,
            allgather,
            progress=Mock(side_effect=RuntimeError("a newer membership change")),
        )
    impl.restore_placement.assert_not_called()
    assert slots[:, 0].tolist() == [1, 0, 0, -1]
    # Rank 0's view names no expert in the slot the stopped restore wrote, so the next plan does
    # not keep the installed placement's expert 1 there.
    ((_, view),) = restore.current_placement()
    assert view[0] == [1, -1, 0, -1]
    # The stop may leave the slot written in part.
    slots[1] = -7
    # The rejoin runs again and loads the slot again.
    retry = restore.plan_expert_restore([True] * WORLD_SIZE, [rank_1_rejoins(view)])
    restore.restore_lost_experts(retry, workspace, allgather)
    (dead_ranks, (installed,)), _ = impl.restore_placement.call_args
    assert dead_ranks == []
    for slot, expert in enumerate(installed[0]):
        if expert >= 0:
            assert torch.equal(slots[slot], host[(expert, "w")]), (slot, expert)
    assert installed[0] == [1, 0, -1, -1]


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


def _record(
    active_mask, incarnations=(0, 0, 0, 0), state=MembershipState.COMMITTED, given_up_mask=0
):
    return MembershipRecord(
        generation=1,
        state=state,
        active_mask=active_mask,
        incarnations=incarnations,
        given_up_mask=given_up_mask,
    )


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "record, error, message",
    [
        # Rank 2 is active, so nobody builds its host weights again.
        (_record(0b0111, (0, 0, 0, 1)), ReplacementCannotStart, r"active ranks \[2\]"),
        # Rank 2's node supervisor gave its replacements up.
        (
            _record(0b0011, (0, 0, 1, 1), given_up_mask=0b0100),
            ReplacementCannotStart,
            r"Ranks \[2\]",
        ),
        (_record(0b0011, state=MembershipState.ABORTED), RuntimeError, "deployment ended"),
    ],
    ids=["active", "given_up", "aborted"],
)
def test_replacement_without_node_host_weights_fails(monkeypatch, tracked, record, error, message):
    """Refuse to start rank 3's replacement when rank 2 on its node lacks its host weights."""
    _replacement(monkeypatch)
    monkeypatch.setattr(startup._state, "client", SimpleNamespace(record=record))
    monkeypatch.setattr(startup, "_HOST_WEIGHTS_CHECK_S", 0.01)
    replacement = _sharer(1, 0.0, node_size=2)
    replacement.finalize_layer_weights()
    load_balancer = FaultTolerantMoeLoadBalancer.__new__(FaultTolerantMoeLoadBalancer)
    load_balancer.replacement = True
    load_balancer._wait_for_host_weights = startup.wait_for_node_host_weights
    load_balancer.ep_rank, load_balancer.ep_size = 3, 4
    load_balancer.mapping = Mapping(world_size=4, tp_size=4, gpus_per_node=2)
    load_balancer.layer_updates_per_iter = 0
    load_balancer.expert_restore = EplbExpertRestore(load_balancer)
    load_balancer.single_layer_load_balancers = [
        SimpleNamespace(need_load_shared_weights=lambda: True, host_tensor_sharer=replacement)
    ]
    with pytest.raises(error, match=message) as raised:
        load_balancer.finalize_model()
    if error is ReplacementCannotStart:
        assert raised.value.code == REPLACEMENT_CANNOT_START
    replacement.pre_shutdown_cleanup()
    replacement.post_shutdown_cleanup()


@pytest.mark.timeout(60)
def test_replacement_waits_until_host_weights_are_built_again(monkeypatch):
    """Wait for the node's host weights that a later replacement of rank 2 builds again."""
    monkeypatch.setattr(
        startup._state, "client", SimpleNamespace(record=_record(0b0011, (0, 0, 1, 1)))
    )
    monkeypatch.setattr(startup, "_HOST_WEIGHTS_CHECK_S", 0.01)
    startup.wait_for_node_host_weights(2, lambda: [])
    checks = iter([[2], [2], [2], []])
    startup.wait_for_node_host_weights(3, lambda: next(checks))
    assert next(checks, None) is None


def test_host_weights_not_sized_yet_are_missing(monkeypatch, tracked):
    """Check that shared memory its rank created but has not sized yet holds no host weights."""
    _original(monkeypatch)
    sharer = _sharer(1, 0.0, node_size=2)
    path = f"/dev/shm/{sharer.get_shared_memory_name(0)}"
    os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600))
    assert not sharer.holds_host_weights(0)
    sharer.pre_shutdown_cleanup()


@pytest.mark.parametrize(
    "leaf, quant, sm, problem_fields, reason",
    [
        # Gated FC1 rows (2 * 96) are not a multiple of the 128-row NVFP4 scale tile.
        (
            TrtllmCutlassW4a16Nvfp4Impl,
            "W4A16_NVFP4",
            100,
            {"intermediate_size": 96},
            "SHAPE_UNALIGNED",
        ),
        # The BF16 FC1 op fuses only SwiGLU.
        (
            TrtllmCutedslGroupedGemmBf16Impl,
            None,
            107,
            {"activation": "Relu2"},
            "ACTIVATION_UNSUPPORTED",
        ),
    ],
)
def test_leaf_reason_wins_over_spare_slots(leaf, quant, sm, problem_fields, reason):
    """A backend without spare slots that its own gate also turns down reports its own reason."""
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
        MoEDeployment,
        MoEEnvironment,
        MoEProblem,
        MoERejectReason,
    )
    from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep

    problem = MoEProblem(quant=quant, dtype_act=torch.bfloat16, **problem_fields)
    deployment = MoEDeployment(
        ep_size=4,
        tp_size=1,
        parallel_size=4,
        use_dp=True,
        num_slots=8,
        env=MoEEnvironment(sm=sm, available_deps=(MoEDep.CUTEDSL_RUBIN.value,)),
        eplb_enabled=True,
        eplb_mode="standard",
        spare_slots_enabled=True,
    )
    assert not leaf.capabilities.supports_spare_slots
    verdict = leaf.can_implement(problem, deployment)
    assert verdict.reject_reason is MoERejectReason[reason]


def test_host_weights_outlive_a_failed_rank(monkeypatch, tracked):
    """Leave no host weights, created or attached, for a rank's resource tracker to unlink.

    The resource tracker of a rank that fails unlinks the shared memory the rank registered, the
    other ranks' host weights included. The node supervisor unlinks them once the deployment ends.
    """
    _original(monkeypatch)
    sharers = [_sharer(rank, float(rank), node_size=2) for rank in range(2)]
    for sharer in sharers:
        sharer.finalize_layer_weights()
    for sharer in sharers:
        sharer.finalize_host_tensor_sharing()
    assert sorted(sharers[0].node_host_weights) == [(e, "w") for e in range(4)]
    assert not tracked
    for sharer in sharers:
        sharer.pre_shutdown_cleanup()
        sharer.post_shutdown_cleanup()
    assert not tracked
    assert _shm_names() == sorted(sharer.get_shared_memory_name() for sharer in sharers)
