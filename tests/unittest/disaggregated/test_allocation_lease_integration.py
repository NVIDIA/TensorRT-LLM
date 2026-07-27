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

"""CPU-only allocator lease integration tests for disaggregated transfer."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, WaitResult
from tensorrt_llm._torch.disaggregation.lifecycle import PhysicalDisposition
from tensorrt_llm._torch.disaggregation.resource.allocation_lease import (
    AllocationLease,
    AllocationLeaseValidationError,
)
from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup
from tensorrt_llm._torch.disaggregation.transceiver import (
    _NON_DRAINED_TRANSCEIVERS,
    KvCacheTransceiverV2,
)
from tensorrt_llm.bindings import LlmRequestState


@dataclass(frozen=True)
class _V1Block:
    window_size: int
    block_index: int
    beam_index: int
    primary_pool_index: int


@dataclass(frozen=True)
class _V2Range:
    layer_group_id: int
    beam_index: int
    block_begin: int
    block_end: int
    page_indices: tuple[int, ...]
    ssm_page_index: int | None = None


class _LeaseHandle:
    def __init__(
        self,
        snapshot: object,
        *,
        settlement_name: str = "RELEASED",
        events: list[str] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self._settlement_name = settlement_name
        self._events = events
        self.proofs: list[object] = []

    def settle(self, proof: object) -> object:
        if self._events is not None:
            self._events.append("settle")
        self.proofs.append(proof)
        return SimpleNamespace(name=self._settlement_name)


class _LeaseManager:
    def __init__(self, handle: _LeaseHandle, events: list[str] | None = None) -> None:
        self._handle = handle
        self._events = events
        self.request_ids: list[int] = []

    def snapshot_and_lease(self, request_id: int) -> _LeaseHandle:
        if self._events is not None:
            self._events.append("lease")
        self.request_ids.append(request_id)
        return self._handle


class _ReuseAdapter:
    tokens_per_block = 8

    def __init__(self, block_ids: tuple[int, ...], events: list[str] | None = None) -> None:
        self._block_ids = block_ids
        self._events = events

    def get_cached_token_count_per_layer_group(self, req, layer_groups):
        del req
        return [0] * len(layer_groups)

    def get_block_ids(self, req, group_idx, layer_group):
        del req, group_idx, layer_group
        if self._events is not None:
            self._events.append("copy")
        return np.asarray(self._block_ids, dtype=np.int64)


def _v1_handle(
    *,
    copied: tuple[int, ...] = (10, 11, 12),
    settlement_name: str = "RELEASED",
    events: list[str] | None = None,
) -> tuple[_LeaseHandle, tuple[int, ...]]:
    blocks = (
        _V1Block(64, 0, 0, 10),
        _V1Block(64, 0, 1, 10),
        _V1Block(64, 1, 0, 11),
        _V1Block(64, 1, 1, 12),
    )
    snapshot = SimpleNamespace(
        lease_id=1,
        identity=SimpleNamespace(allocation_generation=7),
        blocks=blocks,
    )
    return (
        _LeaseHandle(snapshot, settlement_name=settlement_name, events=events),
        copied,
    )


def _v2_handle(
    *,
    copied: tuple[int, ...] = (21, 22),
    settlement_name: str = "RELEASED",
) -> tuple[_LeaseHandle, tuple[int, ...]]:
    ranges = (
        _V2Range(0, 0, 0, 2, (21, 22)),
        _V2Range(0, 1, 0, 2, (21, 23)),
    )
    snapshot = SimpleNamespace(
        lease_id=2,
        identity=SimpleNamespace(allocation_generation=9),
        ranges=ranges,
    )
    return _LeaseHandle(snapshot, settlement_name=settlement_name), copied


def _transceiver_for_create(
    handle: _LeaseHandle,
    copied: tuple[int, ...],
    *,
    events: list[str] | None = None,
    beam_width: int = 1,
) -> tuple[KvCacheTransceiverV2, object]:
    layer_group = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=1,
        sliding_window_size=64,
    )
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._kv_cache_manager = _LeaseManager(handle, events)
    transceiver._reuse_adapter = _ReuseAdapter(copied, events)
    transceiver._page_table = SimpleNamespace(layer_groups=[layer_group])
    request = SimpleNamespace(
        prompt_len=16,
        py_request_id=41,
        py_beam_width=beam_width,
        is_generation_only_request=lambda: False,
    )
    return transceiver, request


@pytest.fixture
def backend_neutral_proof(monkeypatch):
    monkeypatch.setattr(
        AllocationLease,
        "_backend_proof",
        lambda self, disposition: disposition,
    )


@pytest.mark.usefixtures("backend_neutral_proof")
def test_v1_lease_precedes_descriptor_copy_and_is_carried_by_slice() -> None:
    events: list[str] = []
    handle, copied = _v1_handle(events=events)
    transceiver, request = _transceiver_for_create(
        handle,
        copied,
        events=events,
        beam_width=2,
    )

    kv_slice = transceiver._create_kv_slice(request)

    assert events == ["lease", "copy"]
    assert isinstance(kv_slice.allocation_lease, AllocationLease)
    assert kv_slice.allocation_lease.snapshot is handle.snapshot
    np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], copied)


@pytest.mark.usefixtures("backend_neutral_proof")
def test_v2_snapshot_validates_the_copied_beam_zero_pages() -> None:
    handle, copied = _v2_handle()
    transceiver, request = _transceiver_for_create(handle, copied)

    kv_slice = transceiver._create_kv_slice(request)

    assert isinstance(kv_slice.allocation_lease, AllocationLease)
    assert kv_slice.allocation_lease.snapshot is handle.snapshot
    np.testing.assert_array_equal(kv_slice.block_ids_per_layer_groups[0], copied)


@pytest.mark.usefixtures("backend_neutral_proof")
def test_descriptor_mismatch_rolls_back_as_not_exposed() -> None:
    handle, _ = _v1_handle()
    transceiver, request = _transceiver_for_create(handle, (10, 99))

    with pytest.raises(AllocationLeaseValidationError, match="lease 1 owns"):
        transceiver._create_kv_slice(request)

    assert handle.proofs == [PhysicalDisposition.NOT_EXPOSED]
    assert transceiver._get_session_allocation_leases() == {}
    assert transceiver._get_retained_allocation_leases() == []


@pytest.mark.usefixtures("backend_neutral_proof")
@pytest.mark.parametrize(
    ("disposition", "expected_proof"),
    [
        (PhysicalDisposition.QUIESCED_SUCCESS, PhysicalDisposition.QUIESCED_SUCCESS),
        (PhysicalDisposition.QUIESCED_FAILURE, PhysicalDisposition.QUIESCED_FAILURE),
    ],
)
def test_sync_or_async_close_settles_only_after_physical_close(
    disposition: PhysicalDisposition,
    expected_proof: PhysicalDisposition,
) -> None:
    events: list[str] = []
    handle, _ = _v1_handle(events=events)
    lease = AllocationLease(handle)
    session = Mock()
    session.close.side_effect = lambda: events.append("close")
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._bind_slice_allocation_lease(
        session,
        KVSlice(allocation_lease=lease),
    )

    transceiver._close_session_with_allocation_leases(session, disposition)

    assert events == ["close", "settle"]
    assert handle.proofs == [expected_proof]
    assert lease.settled
    assert transceiver._get_session_allocation_leases() == {}


@pytest.mark.usefixtures("backend_neutral_proof")
def test_close_exception_retains_the_bound_lease_without_settlement() -> None:
    handle, _ = _v1_handle()
    lease = AllocationLease(handle)
    session = Mock()
    session.close.side_effect = RuntimeError("still active")
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._bind_slice_allocation_lease(
        session,
        KVSlice(allocation_lease=lease),
    )

    with pytest.raises(RuntimeError, match="still active"):
        transceiver._close_session_with_allocation_leases(
            session,
            PhysicalDisposition.QUIESCED_FAILURE,
        )

    assert handle.proofs == []
    assert transceiver._get_session_allocation_leases() == {session: [lease]}


@pytest.mark.usefixtures("backend_neutral_proof")
def test_allocator_in_doubt_settlement_is_quarantined() -> None:
    handle, _ = _v2_handle(settlement_name="IN_DOUBT")
    lease = AllocationLease(handle)
    session = Mock()
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._bind_slice_allocation_lease(
        session,
        KVSlice(allocation_lease=lease),
    )

    try:
        with pytest.raises(RuntimeError, match="allocator rejected"):
            transceiver._close_session_with_allocation_leases(
                session,
                PhysicalDisposition.QUIESCED_SUCCESS,
            )

        assert not lease.settled
        assert transceiver._get_session_allocation_leases() == {session: [lease]}
        assert transceiver._get_retained_allocation_leases() == [lease]
        assert transceiver in _NON_DRAINED_TRANSCEIVERS
        assert transceiver._shutdown_started
        assert transceiver._retirement_fault is not None
        with pytest.raises(RuntimeError, match="retirement invariant failed"):
            with transceiver._status_call():
                pass
    finally:
        _NON_DRAINED_TRANSCEIVERS.discard(transceiver)


@pytest.mark.usefixtures("backend_neutral_proof")
def test_failed_settlement_retains_every_unprocessed_session_lease() -> None:
    failed_handle, _ = _v1_handle(settlement_name="IN_DOUBT")
    unprocessed_handle, _ = _v2_handle()
    failed_lease = AllocationLease(failed_handle)
    unprocessed_lease = AllocationLease(unprocessed_handle)
    session = Mock()
    transceiver = object.__new__(KvCacheTransceiverV2)
    for lease in (failed_lease, unprocessed_lease):
        transceiver._bind_slice_allocation_lease(
            session,
            KVSlice(allocation_lease=lease),
        )

    try:
        with pytest.raises(RuntimeError, match="allocator rejected"):
            transceiver._close_session_with_allocation_leases(
                session,
                PhysicalDisposition.QUIESCED_SUCCESS,
            )

        assert transceiver._get_session_allocation_leases() == {
            session: [failed_lease, unprocessed_lease]
        }
        assert failed_handle.proofs == [PhysicalDisposition.QUIESCED_SUCCESS]
        assert unprocessed_handle.proofs == []
    finally:
        _NON_DRAINED_TRANSCEIVERS.discard(transceiver)


@pytest.mark.usefixtures("backend_neutral_proof")
def test_nonreusable_disposition_never_releases_allocator_lease() -> None:
    handle, _ = _v1_handle()
    lease = AllocationLease(handle)
    transceiver = object.__new__(KvCacheTransceiverV2)

    try:
        with pytest.raises(RuntimeError, match="allocator rejected"):
            transceiver._settle_allocation_lease(
                lease,
                PhysicalDisposition.IN_DOUBT,
            )
        assert handle.proofs == []
        assert transceiver._get_retained_allocation_leases() == [lease]
        assert transceiver._shutdown_started
        assert transceiver._retirement_fault is not None
    finally:
        _NON_DRAINED_TRANSCEIVERS.discard(transceiver)


class _SyncRequest:
    def __init__(self, request_id: int) -> None:
        self.request_id = request_id
        self.py_request_id = request_id
        self.py_disaggregated_params = None
        self.py_disagg_transfer_protocol_identity = object()
        self.state = None
        self.py_beam_width = 1
        self.kv_cache_size = 0

    def set_kv_cache_size(self, size: int) -> None:
        self.kv_cache_size = size


@pytest.mark.usefixtures("backend_neutral_proof")
def test_sync_receive_success_settles_success_after_wait() -> None:
    handle, _ = _v1_handle()
    lease = AllocationLease(handle)
    request = _SyncRequest(71)
    session = Mock()
    session.wait_complete.return_value = WaitResult.COMPLETED
    session.status = SimpleNamespace()
    session.transfer_start_time = None
    session.transfer_end_time = None
    session.kv_cache_size_bytes = 8
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = False
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = Mock()
    transceiver._transfer_worker.create_rx_session.return_value = session
    transceiver._create_kv_slice = Mock(return_value=KVSlice(allocation_lease=lease))
    transceiver._need_aux_transfer = Mock(return_value=False)
    transceiver._assert_disagg_history_declared = Mock()
    transceiver._slice_num_bytes = Mock(return_value=8)
    transceiver._kv_size_rank_factor = 1

    transceiver.request_and_receive_sync(request)

    assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    assert handle.proofs == [PhysicalDisposition.QUIESCED_SUCCESS]
    session.close.assert_called_once_with()


@pytest.mark.usefixtures("backend_neutral_proof")
def test_async_publication_exception_settles_failure_after_cancel_drain() -> None:
    handle, _ = _v1_handle()
    lease = AllocationLease(handle)
    request = _SyncRequest(72)
    session = Mock()
    session.receive.side_effect = RuntimeError("publish failed")
    session.has_transferring_tasks.return_value = False
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._session_admission_lock = threading.RLock()
    transceiver._shutdown_started = False
    transceiver._ever_had_recv_session = False
    transceiver._recv_sessions = {}
    transceiver._recv_reqs = {}
    transceiver._transfer_worker = Mock()
    transceiver._transfer_worker.create_rx_session.return_value = session
    transceiver._create_kv_slice = Mock(return_value=KVSlice(allocation_lease=lease))
    transceiver._slice_num_bytes = Mock(return_value=8)
    transceiver._kv_size_rank_factor = 1

    with pytest.raises(RuntimeError, match="publish failed"):
        transceiver.request_and_receive_async(request)

    assert handle.proofs == [PhysicalDisposition.QUIESCED_FAILURE]
    assert transceiver._recv_sessions == {}
    assert transceiver._recv_reqs == {}
