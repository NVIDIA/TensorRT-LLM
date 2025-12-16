# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings import LlmRequestState


def create_mock_request(request_id: int):
    """Create a mock LlmRequest with the given request ID."""
    request = MagicMock()
    request.py_request_id = request_id
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    return request


def create_mock_resource_manager(
    kv_cache_manager=None,
    seq_slot_manager=None,
    spec_resource_manager=None,
):
    """Create a mock ResourceManager with the specified resource managers."""
    resource_manager = MagicMock()
    resource_manager.resource_managers = {}

    if kv_cache_manager is not None:
        resource_manager.resource_managers[ResourceManagerType.KV_CACHE_MANAGER] = kv_cache_manager

    if seq_slot_manager is not None:
        resource_manager.resource_managers[ResourceManagerType.SEQ_SLOT_MANAGER] = seq_slot_manager

    if spec_resource_manager is not None:
        resource_manager.resource_managers[ResourceManagerType.SPEC_RESOURCE_MANAGER] = (
            spec_resource_manager
        )

    return resource_manager


def test_start_transfer_single_request():
    """Test starting a single transfer."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    seq_slot_manager = MagicMock()
    resource_manager = create_mock_resource_manager(
        kv_cache_manager=kv_cache_manager, seq_slot_manager=seq_slot_manager
    )
    manager = AsyncTransferManager(resource_manager)

    request = create_mock_request(42)
    manager.start_transfer(request)

    # Check request is tracked
    assert 42 in manager.requests
    req, block_id, counter = manager.requests[42]
    assert req is request
    assert block_id == 100
    assert counter == 1

    # Check state was updated
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    # Check KV cache manager was called
    kv_cache_manager.store_blocks_for_reuse.assert_called_once_with(request, True)

    # Check seq slot manager was called to free resources
    seq_slot_manager.free_resources.assert_called_once_with(request)

    # Check should_store_blocks is tracked
    assert manager.request_should_store_blocks[42] is True

    manager.end_transfer(request)
    kv_cache_manager.unpin_blocks_by_id.assert_called_once()


def test_start_transfer_multiple_transfers_same_request():
    """Test starting multiple transfers for the same request."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    request = create_mock_request(42)
    manager.start_transfer(request, True)
    manager.start_transfer(request, True)
    manager.start_transfer(request, True)

    # Counter should be incremented
    _, block_id, counter = manager.requests[42]
    assert counter == 3

    # store_blocks_for_reuse should only be called once
    kv_cache_manager.store_blocks_for_reuse.assert_called_once()

    for _ in range(2):
        manager.end_transfer(request)
        kv_cache_manager.unpin_blocks_by_id.assert_not_called()

    manager.end_transfer(request)
    kv_cache_manager.unpin_blocks_by_id.assert_called_once()


def test_start_transfer_without_storing_blocks():
    """Test starting a transfer with should_store_blocks=False."""
    kv_cache_manager = MagicMock()
    spec_resource_manager = MagicMock()
    resource_manager = create_mock_resource_manager(
        kv_cache_manager=kv_cache_manager, spec_resource_manager=spec_resource_manager
    )
    manager = AsyncTransferManager(resource_manager)

    request = create_mock_request(42)
    manager.start_transfer(request, should_store_blocks=False)

    # Check request is tracked
    assert 42 in manager.requests
    req, block_id, counter = manager.requests[42]
    assert block_id is None  # No block stored
    assert counter == 1

    # Check KV cache manager was NOT called
    kv_cache_manager.store_blocks_for_reuse.assert_not_called()
    spec_resource_manager.free_resources.assert_called_once_with(request)
    # Check should_store_blocks is tracked
    assert manager.request_should_store_blocks[42] is False

    assert manager.end_transfer(request)

    kv_cache_manager.unpin_blocks_by_id.assert_not_called()


def test_start_transfer_inconsistent_should_store_blocks():
    """Test that starting a transfer with inconsistent should_store_blocks raises an error."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    request = create_mock_request(42)
    manager.start_transfer(request, should_store_blocks=True)

    with pytest.raises(ValueError, match="already in transfer.*should_store_blocks"):
        manager.start_transfer(request, should_store_blocks=False)


def test_end_transfer_without_storing_blocks():
    """Test ending a transfer when should_store_blocks was False."""
    kv_cache_manager = MagicMock()
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    request = create_mock_request(42)
    manager.start_transfer(request, should_store_blocks=False)

    should_terminate = manager.end_transfer(request)

    # Request should be removed
    assert 42 not in manager.requests

    # Blocks should NOT be unpinned (since we didn't store any)
    kv_cache_manager.unpin_blocks_by_id.assert_not_called()

    # Should terminate (because we didn't store blocks)
    assert should_terminate is True


def test_end_transfer_preserves_error_state():
    """Test that end_transfer does not overwrite error state."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    request = create_mock_request(42)
    manager.start_transfer(request)

    # Set error state before end_transfer
    request.state = LlmRequestState.DISAGG_TRANS_ERROR

    manager.end_transfer(request)

    # Error state should be preserved
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR


def test_requests_in_transfer():
    """Test that requests_in_transfer returns correct mapping."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    request1 = create_mock_request(1)
    request2 = create_mock_request(2)
    request3 = create_mock_request(3)

    manager.start_transfer(request1)
    manager.start_transfer(request2)
    manager.start_transfer(request3)

    in_transfer = manager.requests_in_transfer()

    assert len(in_transfer) == 3
    assert in_transfer[1] is request1
    assert in_transfer[2] is request2
    assert in_transfer[3] is request3
