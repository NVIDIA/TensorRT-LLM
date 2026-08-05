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

import gc
from collections import deque
from unittest.mock import Mock, sentinel
from uuid import uuid4

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheAllocationLease
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AllocationIdentity,
    AllocationRange,
    AllocationReuseProof,
    LeaseSettlement,
    rawref,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._core._kv_cache_manager import (
    _QUARANTINED_ALLOCATION_MANAGERS,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._core._kv_cache_manager import (
    KVCacheManager as RuntimeKVCacheManager,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._storage_manager import StorageManager


def _make_wrapper() -> KVCacheManagerV2:
    return object.__new__(KVCacheManagerV2)


def _make_runtime_manager() -> RuntimeKVCacheManager:
    manager = object.__new__(RuntimeKVCacheManager)
    manager._allocator_domain_id = uuid4().hex
    manager._next_allocation_lease_id = 1
    manager._allocation_leases = {}
    manager._in_doubt_allocation_leases = {}
    manager._settled_allocation_leases = {}
    manager._settled_allocation_lease_order = deque()
    manager._life_cycles = Mock(size=1, ssm_life_cycle_id=None)
    manager._radix_tree = Mock()
    manager._storage = Mock()
    return manager


def _make_runtime_allocation(manager: RuntimeKVCacheManager) -> Mock:
    allocation = Mock()
    allocation.manager = manager
    allocation.close_requested = False
    allocation.close_pending = False
    allocation.is_active = True
    allocation.num_blocks = 7
    allocation.beam_width = 1
    allocation.allocation_identity = AllocationIdentity(
        allocator_domain_id=manager.allocator_domain_id,
        request_id=41,
        allocation_generation=3,
    )
    allocation._snapshot_transferable_page_ranges.return_value = (
        (0, 2, (11, 12)),
        (5, 7, (21, 22)),
    )
    return allocation


def test_allocation_range_never_accepts_bad_or_ambiguous_page_descriptors() -> None:
    with pytest.raises(ValueError, match="valid GPU pages"):
        AllocationRange(0, 0, 0, 1, (-1,))
    with pytest.raises(ValueError, match="valid GPU page"):
        AllocationRange(0, 0, 0, 1, (), ssm_page_index=-1)
    with pytest.raises(ValueError, match="must not contain attention"):
        AllocationRange(0, 0, 0, 1, (3,), ssm_page_index=5)


def test_runtime_snapshot_preserves_sparse_segments_without_sentinel_pages() -> None:
    manager = _make_runtime_manager()
    allocation = _make_runtime_allocation(manager)

    lease = manager.snapshot_and_lease(allocation, layer_group_ids=[0])

    assert lease.snapshot.ranges == (
        AllocationRange(0, 0, 0, 2, (11, 12)),
        AllocationRange(0, 0, 5, 7, (21, 22)),
    )
    allocation._snapshot_transferable_page_ranges.assert_called_once_with(0, 0, 0, 7)
    assert lease.settle(AllocationReuseProof.NOT_EXPOSED) is LeaseSettlement.RELEASED
    manager.shutdown()


def test_abandoned_handle_keeps_runtime_manager_and_backing_storage_quarantined() -> None:
    manager = _make_runtime_manager()
    allocation = _make_runtime_allocation(manager)
    lease = manager.snapshot_and_lease(allocation, layer_group_ids=[0])
    domain_id = manager.allocator_domain_id
    lease_id = lease.snapshot.lease_id
    identity = lease.snapshot.identity

    del lease
    del allocation
    del manager
    gc.collect()

    quarantined = _QUARANTINED_ALLOCATION_MANAGERS[domain_id]
    assert isinstance(quarantined, RuntimeKVCacheManager)
    assert quarantined.outstanding_allocation_lease_count == 1
    assert (
        quarantined.settle_allocation_lease(
            lease_id,
            identity,
            AllocationReuseProof.QUIESCED_SUCCESS,
        )
        is LeaseSettlement.RELEASED
    )
    assert domain_id not in _QUARANTINED_ALLOCATION_MANAGERS
    quarantined.shutdown()


def test_pool_topology_mutation_is_rejected_while_lease_is_active() -> None:
    manager = _make_runtime_manager()
    allocation = _make_runtime_allocation(manager)
    lease = manager.snapshot_and_lease(allocation, layer_group_ids=[0])

    assert manager.resize(0, 1024) is False
    with pytest.raises(RuntimeError, match="active or in doubt"):
        manager._adjust_level(0, 1024)
    with pytest.raises(RuntimeError, match="active or in doubt"):
        manager.adjust()
    manager._storage.adjust_cache_level.assert_not_called()

    assert lease.settle(AllocationReuseProof.NOT_EXPOSED) is LeaseSettlement.RELEASED
    manager.shutdown()


@pytest.mark.parametrize(
    ("method_name", "args"),
    (
        ("adjust_cache_level", (0, None, [])),
        ("shrink_pool_group", (0, 0, 1, [])),
        ("expand_pool_group", (0, 0, 1)),
        ("destroy", ()),
    ),
)
def test_storage_topology_entry_points_invoke_lease_guard(
    method_name: str,
    args: tuple,
) -> None:
    storage = object.__new__(StorageManager)
    storage.__rawref__ = rawref.NULL
    storage._levels = []
    storage._topology_change_guard = Mock(side_effect=RuntimeError("lease topology fence"))

    with pytest.raises(RuntimeError, match="lease topology fence"):
        getattr(storage, method_name)(*args)

    storage._topology_change_guard.assert_called_once_with()
    storage._topology_change_guard = None


def test_physical_retirement_failure_is_stably_in_doubt_and_retains_allocation() -> None:
    manager = _make_runtime_manager()
    allocation = _make_runtime_allocation(manager)
    allocation.close_pending = True
    allocation._finalize_close.side_effect = RuntimeError("retirement failed")
    lease = manager.snapshot_and_lease(allocation, layer_group_ids=[0])

    assert lease.settle(AllocationReuseProof.QUIESCED_SUCCESS) is LeaseSettlement.IN_DOUBT
    assert lease.settle(AllocationReuseProof.QUIESCED_SUCCESS) is LeaseSettlement.IN_DOUBT
    assert manager.outstanding_allocation_lease_count == 1
    assert manager._in_doubt_allocation_leases[lease.snapshot.lease_id].allocation is allocation
    assert manager.resize(0, 1024) is False
    manager._storage.adjust_cache_level.assert_not_called()
    with pytest.raises(RuntimeError, match="1 outstanding allocation lease"):
        manager.shutdown()

    # Test-only cleanup: production deliberately requires an endpoint reset to
    # reclaim an in-doubt allocation.
    manager._in_doubt_allocation_leases.clear()
    manager._settled_allocation_leases.clear()
    manager._release_allocation_lease_quarantine()
    manager.shutdown()


def test_wrapper_resolves_request_mapping_once_before_leasing_exact_allocation() -> None:
    manager = _make_wrapper()
    manager.kv_cache_map = {41: sentinel.exact_allocation}
    manager.impl = Mock(spec=["snapshot_and_lease"])
    manager.impl.snapshot_and_lease.return_value = sentinel.lease

    result = manager.snapshot_and_lease(41, [0, 2], (3, 7))

    assert result is sentinel.lease
    manager.impl.snapshot_and_lease.assert_called_once_with(
        sentinel.exact_allocation,
        [0, 2],
        (3, 7),
    )


def test_wrapper_rejects_missing_request_before_calling_allocator() -> None:
    manager = _make_wrapper()
    manager.kv_cache_map = {}
    manager.impl = Mock(spec=["snapshot_and_lease"])

    with pytest.raises(KeyError, match="request 41"):
        manager.snapshot_and_lease(41)

    manager.impl.snapshot_and_lease.assert_not_called()


def test_wrapper_settlement_never_rechecks_request_mapping() -> None:
    lease = Mock()
    lease.settle.return_value = LeaseSettlement.RELEASED

    result = KVCacheManagerV2.settle_allocation_lease(
        lease,
        AllocationReuseProof.QUIESCED_SUCCESS,
    )

    assert result is LeaseSettlement.RELEASED
    lease.settle.assert_called_once_with(AllocationReuseProof.QUIESCED_SUCCESS)


def test_wrapper_shutdown_refuses_before_mutating_live_request_state() -> None:
    manager = _make_wrapper()
    manager.impl = Mock(spec=["outstanding_allocation_lease_count", "shutdown"])
    manager.impl.outstanding_allocation_lease_count = 2
    cache = Mock()
    manager.kv_cache_map = {41: cache}

    with pytest.raises(RuntimeError, match="2 outstanding allocation lease"):
        manager.shutdown()

    cache.close.assert_not_called()
    assert manager.kv_cache_map == {41: cache}
    manager.impl.shutdown.assert_not_called()


def _cpp_lease_backend(*, outstanding_leases: int = 0, outstanding_pins: int = 0) -> Mock:
    backend = Mock(
        spec=[
            "get_allocation_identity",
            "snapshot_and_lease",
            "settle_allocation_lease",
            "get_allocation_lease_accounting",
            "release_pools",
        ]
    )
    backend.get_allocation_lease_accounting.return_value = Mock(
        lease_state_known=True,
        outstanding_lease_count=outstanding_leases,
        outstanding_block_pin_count=outstanding_pins,
    )
    return backend


def test_wrapper_adapts_identity_based_cpp_allocation_lease() -> None:
    manager = _make_wrapper()
    manager.impl = _cpp_lease_backend()
    manager.kv_cache_map = {}
    identity = sentinel.identity
    snapshot = Mock(lease_id=7, identity=identity, blocks=())
    manager.impl.get_allocation_identity.return_value = identity
    manager.impl.snapshot_and_lease.return_value = snapshot

    lease = manager.snapshot_and_lease(41)

    assert isinstance(lease, KVCacheAllocationLease)
    assert lease.snapshot is snapshot
    manager.impl.get_allocation_identity.assert_called_once_with(41)
    manager.impl.snapshot_and_lease.assert_called_once_with(identity)
    assert manager.supports_allocation_generation_leases

    proof = sentinel.reusable_proof
    settlement = lease.settle(proof)
    assert settlement is manager.impl.settle_allocation_lease.return_value
    manager.impl.settle_allocation_lease.assert_called_once_with(7, identity, proof)


def test_wrapper_cpp_shutdown_rejects_outstanding_lease_before_mutation() -> None:
    manager = _make_wrapper()
    manager.impl = _cpp_lease_backend(outstanding_leases=2, outstanding_pins=5)
    cache = Mock()
    manager.kv_cache_map = {41: cache}
    manager.conversation_manager = None

    with pytest.raises(RuntimeError, match="2 outstanding allocation lease"):
        manager.shutdown()

    cache.close.assert_not_called()
    assert manager.kv_cache_map == {41: cache}
    manager.impl.release_pools.assert_not_called()


def test_wrapper_cpp_shutdown_releases_pools_after_closing_allocations() -> None:
    manager = _make_wrapper()
    manager.impl = _cpp_lease_backend()
    cache = Mock()
    manager.kv_cache_map = {41: cache}
    manager.conversation_manager = None

    manager.shutdown()

    cache.close.assert_called_once_with()
    assert manager.kv_cache_map == {}
    manager.impl.release_pools.assert_called_once_with()


def test_wrapper_lease_incapable_cpp_v2_keeps_legacy_shutdown_working() -> None:
    manager = _make_wrapper()
    manager.impl = Mock(spec=["shutdown"])
    cache = Mock()
    manager.kv_cache_map = {41: cache}
    manager.conversation_manager = None

    assert not manager.supports_allocation_generation_leases
    with pytest.raises(TypeError, match="does not support allocation leases"):
        manager.snapshot_and_lease(41)

    manager.shutdown()

    cache.close.assert_called_once_with()
    assert manager.kv_cache_map == {}
    manager.impl.shutdown.assert_called_once_with()
