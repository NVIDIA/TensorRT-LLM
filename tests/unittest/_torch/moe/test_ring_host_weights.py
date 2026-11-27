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
"""Tests for the EPLB host weights that Wide-EP fault tolerance keeps in a ring of nodes."""

import collections
import functools
import itertools
import os
import secrets
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import (
    DeploymentHostWeights,
    EplbExpertRestore,
    ExpertRestorePlanMismatchError,
    staging_pieces,
)
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    HostMoeTensorSharer,
    SingleLayerMoeLoadBalancer,
)

pytestmark = pytest.mark.cpu_only

_USES_GDRCOPY = "tensorrt_llm.bindings.internal.runtime.host_accessible_device_memory_uses_gdrcopy"
# Three nodes of two ranks, twelve experts on two EPLB slots per rank and two spare slots.
_NODES = [0, 0, 1, 1, 2, 2]
_EXPERTS = 12
_PLACEMENT = [[2 * rank, 2 * rank + 1, -1, -1] for rank in range(len(_NODES))]
# Slot weights by name: w1 300 bytes, w2 200 bytes.
_WEIGHTS = {"w1": (300, torch.uint8), "w2": (100, torch.float16)}


def _comm(local_rank, local_size):
    return Mock(Get_rank=Mock(return_value=local_rank), Get_size=Mock(return_value=local_size))


def _sharer(local_rank, local_size, expert_count=8, expert_ids=None):
    sharer = HostMoeTensorSharer(0, expert_count, _comm(local_rank, local_size))
    if expert_ids is not None:
        sharer.keep_experts(expert_ids)
    return sharer


def _share(sharer, expert_count):
    """Share the weight "w" of the sharer's own experts, each filled with its expert id."""
    for expert in range(expert_count):
        if expert in sharer.own_expert_ids:
            sharer.share_host_tensor_with_shape(expert, "w", torch.full((4,), float(expert)))
        else:
            sharer.pre_register_host_tensor_with_shape(expert, "w", torch.float32, (4,))


@pytest.mark.parametrize("expert_count, local_size", [(8, 4), (10, 4), (3, 4)])
def test_sharer_splits_every_expert_by_default(expert_count, local_size):
    for local_rank in range(local_size):
        start = local_rank * expert_count // local_size
        end = (local_rank + 1) * expert_count // local_size
        assert _sharer(local_rank, local_size, expert_count).own_expert_ids == list(
            range(start, end)
        )


def test_sharer_splits_kept_experts():
    kept = [13, 5, 6, 7, 12]
    assert [_sharer(rank, 2, 16, kept).own_expert_ids for rank in range(2)] == [[5, 6], [7, 12, 13]]
    sharer = _sharer(1, 2, 16, kept)
    assert sharer.rank_expert_ids(0) == [5, 6]
    # Another rank's experts and experts the node does not keep are only registered by shape.
    for expert in (5, 0):
        sharer.pre_register_host_tensor_with_shape(expert, "w", torch.float32, (4,))
    with pytest.raises(AssertionError):
        sharer.pre_register_host_tensor_with_shape(7, "w", torch.float32, (4,))
    with pytest.raises(AssertionError):
        sharer.share_host_tensor_with_shape(0, "w", torch.zeros(4))


def test_kept_experts_shared_between_ranks():
    kept = [1, 4, 6]
    sharers = [_sharer(rank, 2, 8, kept) for rank in range(2)]
    for sharer in sharers:
        sharer.set_shared_memory_base_name(f"test_ring_host_weights_{os.getpid()}_{id(sharers)}")
        _share(sharer, 8)
    try:
        for sharer in sharers:
            sharer.finalize_layer_weights()
        for sharer in sharers:
            weights = {}
            sharer.finalize_host_tensor_sharing(
                lambda expert, name, t: weights.__setitem__(expert, t.tolist())
            )
            assert weights == {expert: [float(expert)] * 4 for expert in kept}
    finally:
        for sharer in sharers:
            sharer.pre_shutdown_cleanup()
        for sharer in sharers:
            sharer.post_shutdown_cleanup()


def _deployment_sharer(expert_ids=None, base_name=None):
    sharer = DeploymentHostWeights(0, 4, _comm(0, 1), replacement=False)
    if expert_ids is not None:
        sharer.keep_experts(expert_ids)
    sharer.set_shared_memory_base_name(
        base_name or f"test_ring_host_weights_{os.getpid()}_{secrets.token_hex(8)}"
    )
    _share(sharer, 4)
    return sharer


def _unlink(sharer):
    try:
        os.unlink(f"/dev/shm/{sharer.get_shared_memory_name()}")
    except FileNotFoundError:
        pass


@pytest.mark.parametrize(
    "next_node_ranks, kept",
    [
        ([2, 3], [4, 5, 6, 7]),
        # Replicas: every expert with a slot on the next node, wherever its other replicas are.
        ([1], [0, 2, 3]),
        # Without them, the node keeps every expert.
        (None, list(range(8))),
    ],
)
def test_node_keeps_next_node_experts(next_node_ranks, kept):
    sharer = DeploymentHostWeights(
        0, 8, _comm(0, 1), replacement=False, next_node_ranks=next_node_ranks
    )
    placement = np.array([[0, 1, -1], [2, 3, 0], [4, 5, -1], [6, 7, -1]])
    sharer.keep_next_node_experts(placement)
    assert sharer.expert_ids == kept


def test_reattach_needs_same_kept_experts():
    writer = _deployment_sharer([0, 1])
    try:
        writer.finalize_layer_weights()
        name = writer.shared_memory_base_name
        assert _deployment_sharer([0, 1], name).holds_host_weights(0)
        assert not _deployment_sharer([0, 2], name).holds_host_weights(0)
    finally:
        writer.pre_shutdown_cleanup()
        writer.post_shutdown_cleanup()
        _unlink(writer)


def test_node_bytes_count_kept_experts():
    sharer = DeploymentHostWeights(0, 8, _comm(0, 2), replacement=False)
    sharer.keep_experts([1, 4, 6])
    # Three experts of a 300 and a 200 byte weight, aligned to 256 bytes, and a fingerprint per rank.
    assert sharer.node_bytes([300, 200]) == 3 * (512 + 256) + 2 * 256


def _resident_bytes(segment_name: str) -> int:
    """Return this process's resident bytes (smaps Rss) in mappings of /dev/shm/``segment_name``."""
    total, current = 0, False
    with open("/proc/self/smaps") as smaps:
        for line in smaps:
            fields = line.split()
            if "-" in fields[0] and ":" not in fields[0]:
                current = fields[-1] == f"/dev/shm/{segment_name}"
            elif current and fields[0] == "Rss:":
                total += int(fields[1]) * 1024
    return total


def test_writer_drops_its_pages():
    """Keep no page of the written host weights mapped, and fault the data back in on read."""
    sharer = _deployment_sharer()
    try:
        sharer.finalize_layer_weights()
        sharer.finalize_host_tensor_sharing()
        assert _resident_bytes(sharer.get_shared_memory_name()) == 0
        assert sharer.node_host_weights[(3, "w")].tolist() == [3.0] * 4
    finally:
        sharer.pre_shutdown_cleanup()
        sharer.post_shutdown_cleanup()
        _unlink(sharer)


def _balancer(nodes, ep_rank, layer_updates_per_iter=0):
    return SimpleNamespace(
        ep_rank=ep_rank,
        ep_size=len(nodes),
        layer_updates_per_iter=layer_updates_per_iter,
        mapping=SimpleNamespace(get_node_rank=lambda rank: nodes[rank]),
        in_iter=False,
        single_layer_load_balancers=[],
        load_balancer_impl=Mock(),
    )


@pytest.mark.parametrize(
    "nodes, layer_updates_per_iter, next_node_ranks",
    [
        (_NODES, 0, [2, 3]),
        # The last node's next node is the first.
        ([2, 2, 0, 0, 1, 1], 0, [2, 3]),
        # A single node keeps its own experts.
        ([0, 0, 0, 0], 0, [0, 1, 2, 3]),
        # Online EPLB may move any expert anywhere.
        (_NODES, 1, None),
    ],
    ids=["ring", "wrap", "single_node", "online"],
)
def test_next_node_ranks(nodes, layer_updates_per_iter, next_node_ranks):
    with patch(_USES_GDRCOPY, return_value=True):
        restore = EplbExpertRestore(_balancer(nodes, 0, layer_updates_per_iter))
    assert restore.next_node_ranks() == next_node_ranks


def _restore(
    nodes, ep_rank, layers=1, placement=_PLACEMENT, expert_count=_EXPERTS, spares=2, view=None
):
    """Return ``ep_rank``'s expert restore over static layers at ``placement``, seen as ``view``."""
    balancer = _balancer(nodes, ep_rank)
    restore = EplbExpertRestore(balancer)
    view = placement if view is None else view
    for layer_idx in range(layers):
        impl = Mock(
            get_layer_id=Mock(return_value=layer_idx),
            get_old_rank_expert_ids=Mock(return_value=[list(rank) for rank in view]),
        )
        layer = SingleLayerMoeLoadBalancer(
            impl,
            _comm(0, 1),
            expert_count,
            updates_enabled=False,
            total_slot_count_per_rank=len(placement[0]),
            spare_slot_count=spares,
            host_tensor_sharer_cls=functools.partial(
                DeploymentHostWeights,
                replacement=False,
                next_node_ranks=restore.next_node_ranks(),
            ),
        )
        layer.set_initial_weight_assignments([expert for rank in placement for expert in rank])
        sharer = layer.host_tensor_sharer
        sharer.keep_next_node_experts(np.array(placement))
        for expert in sharer.expert_ids:
            for name in _WEIGHTS:
                sharer.node_host_weights[(expert, name)] = _host_weight(layer_idx, expert, name)
        layer.slot_tensors = {
            name: torch.zeros((len(placement[0]), count), dtype=dtype)
            for name, (count, dtype) in _WEIGHTS.items()
        }
        balancer.single_layer_load_balancers.append(layer)
    return restore


def _host_weight(layer_idx, expert, name):
    count, dtype = _WEIGHTS[name]
    generator = torch.Generator().manual_seed(layer_idx * 1000 + expert * 10 + len(name))
    data = torch.randint(0, 256, (count * dtype.itemsize,), dtype=torch.uint8, generator=generator)
    return data.view(dtype)


def _active(failed, size=len(_NODES)):
    return [rank not in failed for rank in range(size)]


def test_plan_stages_from_other_node():
    """Stage each lost expert into a rank off the node with its host weights, from that node."""
    plans = [
        _restore(_NODES, rank).plan_expert_restore(_active([4, 5]), [np.array(_PLACEMENT)])
        for rank in range(4)
    ]
    # Node 1 keeps the host weights of node 2's experts 8 to 11, so they go to the spare slots
    # off node 1, of ranks 0 and 1, which stage them from ranks 2 and 3 in turn.
    np.testing.assert_array_equal(
        plans[0].staged, [[0, 0, 2, 8, 2], [0, 0, 3, 10, 3], [0, 1, 2, 9, 2], [0, 1, 3, 11, 3]]
    )
    for plan in plans[1:]:
        np.testing.assert_array_equal(plan.staged, plans[0].staged)
        assert plan.digest == plans[0].digest
    # In another ring node 0 keeps them, so they go to ranks 2 and 3.
    other = _restore([0, 0, 2, 2, 1, 1], 0).plan_expert_restore(
        _active([4, 5]), [np.array(_PLACEMENT)]
    )
    np.testing.assert_array_equal(
        other.staged, [[0, 2, 2, 8, 0], [0, 2, 3, 10, 1], [0, 3, 2, 9, 0], [0, 3, 3, 11, 1]]
    )
    assert other.digest != plans[0].digest


def test_plan_rejects_expert_without_host_weights():
    """Leave the next node's experts with only their GPU copies until a lost node rejoins."""
    restore = _restore(_NODES, 0)
    without_node_1 = restore.plan_expert_restore(_active([2, 3]), [np.array(_PLACEMENT)])
    current = [without_node_1.placements[0]]
    # Node 0 keeps node 1's experts' host weights, and node 2 keeps node 0's.
    restore.plan_expert_restore(_active([1, 2, 3]), current)
    # Node 1 kept those of node 2, so rank 4's experts 8 and 9 have none left. The error names the
    # ranks that kept them.
    with pytest.raises(ValueError, match=r"expert 8 of MoE layer 0: only ranks \[2, 3\] keep them"):
        restore.plan_expert_restore(_active([2, 3, 4]), current)


def test_staging_pieces_alternate_halves():
    # Target 0 takes expert 6 in two layers; workspace halves of 512 bytes.
    staged = np.array([[0, 0, 2, 6, 2], [1, 0, 2, 6, 2]])
    pieces = staging_pieces(staged, [[300, 200], [300, 200]], 1024)
    # A weight that the rest of a half cannot take continues in the next chunk, in the other half.
    assert [
        (piece.chunk, piece.layer, piece.weight, piece.row_offset, piece.weight_offset, piece.size)
        for piece in pieces
    ] == [
        (0, 0, 0, 0, 0, 300),
        (1, 0, 1, 512, 0, 200),
        (1, 1, 0, 768, 0, 256),
        (2, 1, 0, 0, 256, 44),
        (2, 1, 1, 256, 0, 200),
    ]
    # 256 bytes: the second half has no aligned room.
    with pytest.raises(ValueError, match="no aligned room"):
        staging_pieces(staged, [[300, 200], [300, 200]], 256)


def test_staging_pieces_never_overlap():
    staged = np.array(
        [
            [layer, target, slot, 0, 4]
            for layer in range(3)
            for target in range(3)
            for slot in (2, 3)
        ]
    )
    row_bytes = 1024
    pieces = staging_pieces(staged, [[300, 200]] * 3, row_bytes)
    assert [piece.chunk for piece in pieces] == sorted(piece.chunk for piece in pieces)
    # Each weight is staged whole, in consecutive pieces.
    parts = {}
    for piece in pieces:
        parts.setdefault((piece.layer, piece.target, piece.slot, piece.weight), []).append(
            (piece.weight_offset, piece.size)
        )
    assert len(parts) == 2 * len(staged)
    for (_, _, _, weight), weight_parts in parts.items():
        starts = np.cumsum([0] + [size for _, size in sorted(weight_parts)])
        assert [start for start, _ in sorted(weight_parts)] == starts[:-1].tolist()
        assert starts[-1] == [300, 200][weight]
    # A target's pieces of a chunk and of the chunk before, which it may still read, never overlap.
    for piece in pieces:
        assert 0 <= piece.row_offset and piece.row_offset + piece.size <= row_bytes
    for piece, other in itertools.permutations(pieces, 2):
        if other.target == piece.target and piece.chunk - other.chunk in (0, 1):
            assert (
                other.row_offset + other.size <= piece.row_offset
                or piece.row_offset + piece.size <= other.row_offset
            )


def _restore_in_threads(
    restores, plans, before_gather=lambda rank: None, progress=lambda rank: None, times=None
):
    """Restore ``plans`` on one thread per rank and return each rank's number of allgathers.

    If given, ``times`` receives each rank's phase durations by rank.
    """
    barrier = threading.Barrier(len(restores))
    values = {}
    gathers = collections.Counter()
    errors = []
    workspace = torch.zeros(len(_NODES), 1024, dtype=torch.uint8)

    def allgather(rank, value):
        before_gather(rank)
        gathers[rank] += 1
        values[rank] = value
        barrier.wait(timeout=10)
        gathered = [values[r] for r in sorted(values)]
        barrier.wait(timeout=10)
        return gathered

    def run(rank):
        try:
            restores[rank].restore_lost_experts(
                plans[rank],
                workspace,
                functools.partial(allgather, rank),
                None if times is None else times.setdefault(rank, {}),
                progress=functools.partial(progress, rank),
            )
        except BaseException as error:  # noqa: BLE001 - the test thread raises it.
            errors.append(error)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in restores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    for restore in restores.values():
        restore.balancer.load_balancer_impl.restore_placement.assert_called_once()
    return gathers


def _local_slots(plan, rank):
    """Yield the layer and slot of each slot that ``rank`` loads from its node's host weights."""
    for i, (placement, before) in enumerate(zip(plan.placements, plan.placement_before)):
        staged = {
            slot
            for layer, target, slot, _, _ in plan.staged.tolist()
            if (layer, target) == (i, rank)
        }
        for slot in np.flatnonzero((placement[rank] >= 0) & (placement[rank] != before[rank])):
            if slot not in staged:
                yield i, slot


def test_restore_loads_slots_bit_exactly(monkeypatch):
    """Restore node 2's experts on four ranks, loading local slots only after the staging."""
    # With one spare slot per rank, ranks 2 and 3 take two of the experts and load them locally.
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    placement = [rank[:3] for rank in _PLACEMENT]
    restores = {
        rank: _restore(_NODES, rank, layers=2, placement=placement, spares=1) for rank in range(4)
    }
    plans = {
        rank: restore.plan_expert_restore(_active([4, 5]), [np.array(placement)] * 2)
        for rank, restore in restores.items()
    }
    assert plans[0].staged.size and any(_local_slots(plans[2], 2))
    # Each rank installs the placement only once its slots are loaded.
    for rank, restore in restores.items():
        restore.balancer.load_balancer_impl.restore_placement.side_effect = functools.partial(
            _check_loaded, restore, plans[rank]
        )
    _restore_in_threads(restores, plans)


def _loaded(restore, plan):
    """Return, by layer, whether each slot that ``plan`` loads on ``restore``'s rank holds data."""
    rank = restore.balancer.ep_rank
    loaded = []
    for i, layer in enumerate(restore.balancer.single_layer_load_balancers):
        placement, before = plan.placements[i][rank], plan.placement_before[i][rank]
        slots = np.flatnonzero((placement >= 0) & (placement != before))
        loaded.append([bool(layer.slot_tensors["w1"][slot].any()) for slot in slots])
    return loaded


def _record_backing(restore, plan):
    """Record the layers of each call that backs ``restore``'s spare slots, by call.

    Each call checks that no slot that ``plan`` loads holds data yet.
    """
    calls = []

    def set_spare_slots_backed(layer_slots, is_backed):
        assert is_backed
        assert not any(map(any, _loaded(restore, plan))), "A slot was loaded before its backing"
        calls.append([i for i, slots in enumerate(layer_slots) if slots])
        return 0

    restore.set_spare_slots_backed = set_spare_slots_backed
    return calls


def test_restore_backs_every_layer_in_one_call(monkeypatch):
    """Back the spare slots of every layer in one call, before loading any slot."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    placement = [rank[:3] for rank in _PLACEMENT]
    restores = {
        rank: _restore(_NODES, rank, layers=2, placement=placement, spares=1) for rank in range(4)
    }
    plans = {
        rank: restore.plan_expert_restore(_active([4, 5]), [np.array(placement)] * 2)
        for rank, restore in restores.items()
    }
    # Ranks 0 and 1 read both layers' experts from the staging workspace, and ranks 2 and 3 load
    # them from host weights and stage them out.
    assert all(any(_local_slots(plans[rank], rank)) == (rank >= 2) for rank in range(4))
    backing = {rank: _record_backing(restore, plans[rank]) for rank, restore in restores.items()}
    for rank, restore in restores.items():
        restore.balancer.load_balancer_impl.restore_placement.side_effect = functools.partial(
            _check_loaded, restore, plans[rank]
        )
    times = {}
    gathers = _restore_in_threads(restores, plans, times=times)
    assert backing == {rank: [[0, 1]] for rank in range(4)}
    # The ranks agree on the plan, then meet once per staged chunk.
    assert set(gathers.values()) == {4}
    assert {rank: list(phases) for rank, phases in times.items()} == {
        rank: ["back", "back op", "load", "install"] for rank in range(4)
    }


class _BackingError(RuntimeError):
    pass


def test_failed_backing_stops_the_restore(monkeypatch):
    """Raise the backing error in the restore before it loads any slot."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    placement = [rank[:3] for rank in _PLACEMENT]
    restore = _restore(_NODES, 2, layers=2, placement=placement, spares=1)
    plan = restore.plan_expert_restore(_active([4, 5]), [np.array(placement)] * 2)
    error = _BackingError("out of memory")
    backing = []

    def set_spare_slots_backed(layer_slots, is_backed):
        backing.append([i for i, slots in enumerate(layer_slots) if slots])
        raise error

    restore.set_spare_slots_backed = set_spare_slots_backed
    workspace = torch.zeros(len(_NODES), 1024, dtype=torch.uint8)
    with pytest.raises(_BackingError) as raised:
        restore.restore_lost_experts(plan, workspace, lambda value: [value] * len(_NODES))
    assert raised.value is error
    assert backing == [[0, 1]]
    assert not any(map(any, _loaded(restore, plan)))
    restore.balancer.load_balancer_impl.restore_placement.assert_not_called()


def test_restore_reports_progress_per_step(monkeypatch):
    """Report progress after each staged chunk and each layer loaded locally.

    A rank loads its local slots of a chunk's layers before the chunk's allgather.
    """
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    placement = [rank[:3] for rank in _PLACEMENT]
    restores = {
        rank: _restore(_NODES, rank, layers=2, placement=placement, spares=1) for rank in range(4)
    }
    plans = {
        rank: restore.plan_expert_restore(_active([4, 5]), [np.array(placement)] * 2)
        for rank, restore in restores.items()
    }
    local = {(i, slot) for i, slot in _local_slots(plans[2], 2)}
    assert {i for i, _ in local} == {0, 1}
    gathers = collections.Counter()
    reports = []

    def progress(rank):
        if rank == 2:
            layers = restores[2].balancer.single_layer_load_balancers
            loaded = {i for i, slot in local if layers[i].slot_tensors["w1"][slot].any()}
            reports.append((gathers[2], loaded))

    _restore_in_threads(restores, plans, lambda rank: gathers.update([rank]), progress)
    # One allgather agrees on the plan, and each of the three staged chunks takes one more. Chunk 0
    # stages layer 0 and chunk 1 starts layer 1, so rank 2 loads its local slot of each layer
    # before that chunk's allgather.
    assert gathers[2] == 4
    assert reports == [(1, {0}), (2, {0}), (2, {0, 1}), (3, {0, 1}), (4, {0, 1})]


def test_ranks_with_other_views_restore_together(monkeypatch):
    """Restore after only rank 0 installed the last restore, so the ranks' views differ."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    failed = [4, 5]
    (installed,) = (
        _restore(_NODES, 0).plan_expert_restore(_active(failed), [np.array(_PLACEMENT)]).placements
    )
    restores = {rank: _restore(_NODES, rank) for rank in range(1, 4)}
    restores[0] = _restore(_NODES, 0, view=installed)
    restores[0].active_ranks = tuple(_active(failed))
    # Every active rank reports its own slots.
    current = installed.copy()
    current[1:4] = np.array(_PLACEMENT)[1:4]
    plans = {
        rank: restore.plan_expert_restore(_active(failed), [current])
        for rank, restore in restores.items()
    }
    # Rank 0's view is the plan already, and rank 1 still stages experts 9 and 11.
    np.testing.assert_array_equal(plans[0].placements[0], installed)
    assert plans[0].staged.tolist() == [[0, 1, 2, 9, 2], [0, 1, 3, 11, 3]]
    gathers = _restore_in_threads(restores, plans)
    assert len(gathers) == 4 and len(set(gathers.values())) == 1


@pytest.mark.parametrize("differs", ["plan", "rows"])
def test_restore_refuses_other_plan(monkeypatch, differs):
    """Refuse a plan or workspace rows that differ from rank 0's before writing a slot or a row."""
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    active = _active([4, 5])
    plan = _restore(_NODES, 0).plan_expert_restore(active, [np.array(_PLACEMENT)])
    restore = _restore([0, 0, 2, 2, 1, 1] if differs == "plan" else _NODES, 1)
    own = restore.plan_expert_restore(active, [np.array(_PLACEMENT)])
    assert (own.digest == plan.digest) == (differs == "rows")
    workspace = torch.zeros(len(_NODES), 1024 if differs == "plan" else 2048, dtype=torch.uint8)

    def allgather(value):
        # The other active ranks restore rank 0's plan through rows of 1024 bytes.
        agreed = (plan.digest, 1024)
        return [agreed, value, agreed, agreed, None, None]

    with pytest.raises(
        ExpertRestorePlanMismatchError,
        match=f"EP rank 1 planned expert placement {own.digest} with workspace rows of "
        f"{workspace.shape[1]} bytes, but the lowest active rank planned {plan.digest} with rows "
        "of 1024 bytes",
    ):
        restore.restore_lost_experts(own, workspace, allgather)
    assert not workspace.any()
    restore.balancer.load_balancer_impl.restore_placement.assert_not_called()


def test_interrupted_restore_reloads_its_slots(monkeypatch):
    """Reload a slot that a restore stopped writing before it installed its placement."""
    # An interrupted online EPLB update left rank 2's first slot with an extra replica of expert 2.
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    active = _active([4, 5])
    current = np.array(_PLACEMENT)
    current[2, 0] = 2
    restore = _restore(_NODES, 2, view=current)
    plan = restore.plan_expert_restore(active, [current])
    assert [0, 2, 0, 4] in plan.staged[:, :4].tolist()
    digests = iter([[(plan.digest, 1024)] * 4 + [None] * 2])

    def allgather(value):
        gathered = next(digests, None)
        if gathered is None:
            raise RuntimeError("a newer membership change")
        return gathered

    workspace = torch.zeros(len(_NODES), 1024, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="a newer membership change"):
        restore.restore_lost_experts(plan, workspace, allgather)
    # Rank 2's view names no expert in the slot until a restore installs a placement.
    ((_, view),) = restore.current_placement()
    assert view[2] == [-1, 5, -1, -1]
    current[2] = view[2]
    retry = restore.plan_expert_restore(active, [current])
    assert [0, 2, 0, 4] in retry.staged[:, :4].tolist()
    # With redundant slots too, rank 1's emptied active slots take the lost experts before any
    # spare slot.
    redundant = _restore(
        [0, 1, 2], 1, placement=[[0, 1, -1], [2, 3, -1], [0, 2, -1]], expert_count=4, spares=1
    )
    emptied = np.array([[0, 1, -1], [-1, -1, -1], [-1, -1, -1]])
    (placement,) = redundant.plan_expert_restore([True, True, False], [emptied]).placements
    np.testing.assert_array_equal(placement, [[0, 1, -1], [2, 3, -1], [-1, -1, -1]])


def _check_loaded(restore, plan, dead_ranks, placements):
    ep_rank = restore.balancer.ep_rank
    assert dead_ranks == [4, 5]
    for layer_idx, layer in enumerate(restore.balancer.single_layer_load_balancers):
        for slot, expert in enumerate(plan.placements[layer_idx][ep_rank].tolist()):
            if expert < 0 or _PLACEMENT[ep_rank][slot] == expert:
                continue
            for name, tensor in layer.slot_tensors.items():
                loaded = tensor[slot].view(torch.uint8)
                expected = _host_weight(layer_idx, expert, name).view(torch.uint8)
                assert torch.equal(loaded, expected), (ep_rank, layer_idx, slot, name)


# DeepSeek-R1 at EP32 on eight nodes of four GPUs: 256 experts on 8 slots per rank, and the 2 spare
# slots per rank that fault tolerance adds.
_R1_EP, _R1_GPUS_PER_NODE, _R1_EXPERTS = 32, 4, 256
_R1_NODES = [rank // _R1_GPUS_PER_NODE for rank in range(_R1_EP)]
_R1_PLACEMENT = [[8 * rank + slot for slot in range(8)] + [-1, -1] for rank in range(_R1_EP)]


def _r1_node(node):
    return list(range(node * _R1_GPUS_PER_NODE, (node + 1) * _R1_GPUS_PER_NODE))


@pytest.mark.parametrize("node", range(1, 8))
def test_deepseek_r1_node_then_rank(node):
    """Restore a node, then any rank but one of the next node, whose experts it kept."""
    restore = _restore(_R1_NODES, 0, placement=_R1_PLACEMENT, expert_count=_R1_EXPERTS)
    failed = _r1_node(node)
    (placement,) = restore.plan_expert_restore(
        _active(failed, _R1_EP), [np.array(_R1_PLACEMENT)]
    ).placements
    assert (np.bincount(placement[placement >= 0], minlength=_R1_EXPERTS) == 1).all()
    unprotected = _r1_node((node + 1) % 8)
    for rank in range(1, _R1_EP):
        if rank in failed:
            continue
        active = _active(failed + [rank], _R1_EP)
        if rank in unprotected:
            with pytest.raises(ValueError, match="keeps host weights for expert"):
                restore.plan_expert_restore(active, [placement])
        else:
            restore.plan_expert_restore(active, [placement])


@pytest.mark.parametrize("node", range(2, 8))
def test_deepseek_r1_rank_then_node_before(node):
    """Restore any rank of a node, then the node before, which keeps its experts' host weights."""
    previous = _r1_node(node - 1)
    for rank in _r1_node(node):
        restore = _restore(_R1_NODES, 0, placement=_R1_PLACEMENT, expert_count=_R1_EXPERTS)
        (placement,) = restore.plan_expert_restore(
            _active([rank], _R1_EP), [np.array(_R1_PLACEMENT)]
        ).placements
        assert not np.isin(placement[previous], _R1_PLACEMENT[rank][:8]).any()
        restore.plan_expert_restore(_active([rank] + previous, _R1_EP), [placement])


@pytest.mark.parametrize("node", range(1, 8))
def test_deepseek_r1_rank_then_node(node):
    """Restore a rank of the next node, then the node: plan every expert once, or refuse."""
    restore = _restore(_R1_NODES, 0, placement=_R1_PLACEMENT, expert_count=_R1_EXPERTS)
    for rank in _r1_node((node + 1) % 8):
        if rank == 0:
            continue
        (placement,) = restore.plan_expert_restore(
            _active([rank], _R1_EP), [np.array(_R1_PLACEMENT)]
        ).placements
        active = _active([rank] + _r1_node(node), _R1_EP)
        try:
            (after,) = restore.plan_expert_restore(active, [placement]).placements
        except ValueError as error:
            assert "keeps host weights for expert" in str(error)
        else:
            assert (np.bincount(after[after >= 0], minlength=_R1_EXPERTS) == 1).all()
