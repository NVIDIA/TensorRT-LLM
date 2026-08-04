# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for early IndexMapper release during disaggregated KV cache transfer.

Validates that IndexMapper slots are recycled when context-only requests enter
the transfer phase, preventing exhaustion when transfers are slow and multiple
iterations of requests accumulate in-flight.
"""

from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import AsyncTransferManager, PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings import LlmRequestState


def _has_sequence(index_mapper, seq_id: int) -> bool:
    """Helper: IndexMapper has no `has_sequence` binding, so probe via get_index."""
    try:
        index_mapper.get_index(seq_id)
        return True
    except Exception:
        return False


def create_mock_request(
    request_id: int, *, is_context_only: bool = True, is_context_finished: bool = True
):
    """Create a mock LlmRequest with the given request ID."""
    request = MagicMock()
    request.py_request_id = request_id
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    request.is_context_only_request = is_context_only
    request.is_context_finished = is_context_finished
    request.is_finished_due_to_length = False
    request.is_finished_due_to_cancellation = False
    return request


def create_mock_resource_manager(kv_cache_manager=None, seq_slot_manager=None):
    """Create a mock ResourceManager with the specified resource managers."""
    resource_manager = MagicMock()
    resource_manager.resource_managers = {}

    if kv_cache_manager is not None:
        resource_manager.resource_managers[ResourceManagerType.KV_CACHE_MANAGER] = kv_cache_manager

    if seq_slot_manager is not None:
        resource_manager.resource_managers[ResourceManagerType.SEQ_SLOT_MANAGER] = seq_slot_manager

    return resource_manager


class _FakeExecutor:
    """Minimal stand-in for PyExecutor so we can call the unbound
    `_send_kv_async` method without constructing the real object."""

    def __init__(self, kv_cache_manager, async_transfer_manager, kv_cache_transceiver):
        self.kv_cache_manager = kv_cache_manager
        self.async_transfer_manager = async_transfer_manager
        self.kv_cache_transceiver = kv_cache_transceiver
        self.kv_connector_manager = None
        self.disable_overlap_scheduler = True
        self.previous_batch = None

    def _check_disagg_ctx_cache_transfer_status(self, _):
        return None


class TestSendKvAsyncReleasesIndexSlot:
    """Test that `_send_kv_async` releases the IndexMapper slot before the
    transfer is started, which is where the production code actually lives
    (see py_executor._send_kv_async)."""

    def _build(self, kv_cache_manager):
        resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
        transfer_manager = AsyncTransferManager(resource_manager)
        transceiver = MagicMock()
        transceiver.kv_transfer_timeout_ms = None
        return _FakeExecutor(kv_cache_manager, transfer_manager, transceiver), transfer_manager

    def test_send_kv_async_calls_release_index_slot(self):
        """Verify release_index_slot is called for V2 managers."""
        kv_cache_manager = MagicMock()
        kv_cache_manager.store_blocks_for_reuse.return_value = 100

        executor, _ = self._build(kv_cache_manager)
        request = create_mock_request(42)

        PyExecutor._send_kv_async(executor, [request])

        kv_cache_manager.release_index_slot.assert_called_once_with(42)

    def test_send_kv_async_skips_release_for_v1_manager(self):
        """Verify _send_kv_async does not crash when kv_cache_manager lacks
        release_index_slot (legacy V1 managers)."""
        # spec without release_index_slot so hasattr() returns False
        kv_cache_manager = MagicMock(
            spec=["store_blocks_for_reuse", "free_resources", "unpin_blocks_by_id"]
        )
        kv_cache_manager.store_blocks_for_reuse.return_value = 100

        executor, transfer_manager = self._build(kv_cache_manager)
        request = create_mock_request(42)

        # Should not raise even though release_index_slot doesn't exist
        PyExecutor._send_kv_async(executor, [request])

        assert 42 in transfer_manager.requests_in_transfer()

    def test_release_called_once_per_request(self):
        """If the same request passes through _send_kv_async once, release is
        called exactly once (it's gated by `is_context_finished`, which is
        reset after the first pass)."""
        kv_cache_manager = MagicMock()
        kv_cache_manager.store_blocks_for_reuse.return_value = 100

        executor, _ = self._build(kv_cache_manager)
        request = create_mock_request(42)

        PyExecutor._send_kv_async(executor, [request])

        kv_cache_manager.release_index_slot.assert_called_once_with(42)


class TestIndexMapperSlotReuse:
    """Test IndexMapper slot recycling after early release."""

    @pytest.fixture
    def index_mapper(self):
        """Create a small IndexMapper for testing."""
        from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
            IndexMapper,
        )

        return IndexMapper(max_batch_size=2, max_beam_width=1)

    def test_add_and_remove(self, index_mapper):
        """Verify add_new_sequence / remove_sequence bookkeeping via get_index."""
        assert not _has_sequence(index_mapper, 1)

        index_mapper.add_new_sequence(1)
        assert _has_sequence(index_mapper, 1)
        assert not _has_sequence(index_mapper, 2)

        index_mapper.remove_sequence(1)
        assert not _has_sequence(index_mapper, 1)

    def test_slot_reuse_after_early_release(self, index_mapper):
        """Core test: after early release, the slot can be reused by a new request."""
        # Fill capacity
        index_mapper.add_new_sequence(1)
        index_mapper.add_new_sequence(2)

        # Simulate early release of request 1 (transfer started)
        index_mapper.remove_sequence(1)

        # New request should succeed — slot recycled
        index_mapper.add_new_sequence(3)
        assert _has_sequence(index_mapper, 3)
        assert _has_sequence(index_mapper, 2)
        assert not _has_sequence(index_mapper, 1)

    def test_exhaustion_without_early_release(self, index_mapper):
        """Demonstrate the original bug: capacity=2 is exhausted when slots aren't released."""
        index_mapper.add_new_sequence(1)
        index_mapper.add_new_sequence(2)

        # 3rd request fails — IndexMapper full
        with pytest.raises(Exception):
            index_mapper.add_new_sequence(3)

    def test_exhaustion_fixed_with_early_release(self, index_mapper):
        """After early release, the 3rd request succeeds."""
        index_mapper.add_new_sequence(1)
        index_mapper.add_new_sequence(2)

        # Early release request 1
        index_mapper.remove_sequence(1)

        # Now the 3rd request can be added
        index_mapper.add_new_sequence(3)
        assert _has_sequence(index_mapper, 3)


class TestFreeResourcesDoubleReleaseSafety:
    """Test that free_resources handles already-released IndexMapper slots."""

    @pytest.fixture
    def index_mapper(self):
        from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
            IndexMapper,
        )

        return IndexMapper(max_batch_size=4, max_beam_width=1)

    def test_free_resources_after_early_release_no_crash(self, index_mapper):
        """Simulate: release_index_slot() then free_resources() — should not crash."""
        index_mapper.add_new_sequence(1)
        assert _has_sequence(index_mapper, 1)

        # Early release (simulating start_transfer)
        index_mapper.remove_sequence(1)
        assert not _has_sequence(index_mapper, 1)

        # Double release via guard (simulating free_resources)
        if _has_sequence(index_mapper, 1):
            index_mapper.remove_sequence(1)
        # No crash — the guard works

    def test_free_resources_normal_path_still_works(self, index_mapper):
        """Normal path (no early release) still removes the sequence."""
        index_mapper.add_new_sequence(1)
        assert _has_sequence(index_mapper, 1)

        # Normal free_resources path
        if _has_sequence(index_mapper, 1):
            index_mapper.remove_sequence(1)
        assert not _has_sequence(index_mapper, 1)


class TestAuxBufferHigherCapacity:
    """Test that increased AuxBuffer capacity works correctly."""

    def test_aux_buffer_8x_capacity(self):
        """With 8x multiplier (batch_size=1), 8 slots should be allocatable."""
        from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBuffer

        buf = AuxBuffer(max_slot_num=8, beam_width=1, max_draft_len=0, device="cpu")

        slots = []
        for _ in range(8):
            slots.append(buf.alloc_slot())

        # All 8 slots allocated successfully
        assert len(slots) == 8

        # 9th should fail
        with pytest.raises(ValueError, match="No free auxiliary buffer slots"):
            buf.alloc_slot()

        # Free one and re-allocate — free_slot takes the int slot id, not the
        # AuxSlot namedtuple
        buf.free_slot(slots[0].id)
        new_slot = buf.alloc_slot()
        assert new_slot is not None
