# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tensorrt_llm._torch.pyexecutor.py_executor import AsyncTransferManager, PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings import LlmRequestState


def create_mock_request(request_id: int):
    """Create a mock LlmRequest with the given request ID."""
    request = MagicMock()
    request.py_request_id = request_id
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    request.py_kv_transfer_timed_out = False
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
    assert 42 in manager._requests_in_transfer

    transfer_metadata = manager._request_transfer_metadata[42]

    assert transfer_metadata.block_id == 100
    assert transfer_metadata.counter == 1

    # Check state was updated
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    # Check KV cache manager was called
    kv_cache_manager.store_blocks_for_reuse.assert_called_once_with(request, True)

    # Check seq slot manager was called to free resources
    seq_slot_manager.free_resources.assert_called_once_with(request)

    manager.end_transfer(request)
    kv_cache_manager.unpin_blocks_by_id.assert_called_once()


def test_start_transfer_multiple_transfers_same_request():
    """Test starting multiple transfers for the same request."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    request = create_mock_request(42)
    manager.start_transfer(request)
    manager.start_transfer(request)
    manager.start_transfer(request)

    # Counter should be incremented
    transfer_metadata = manager._request_transfer_metadata[42]
    assert transfer_metadata.counter == 3

    # store_blocks_for_reuse should only be called once
    kv_cache_manager.store_blocks_for_reuse.assert_called_once()

    for _ in range(2):
        manager.end_transfer(request)
        kv_cache_manager.unpin_blocks_by_id.assert_not_called()

    manager.end_transfer(request)
    kv_cache_manager.unpin_blocks_by_id.assert_called_once()


def test_transfer_without_storing_blocks():
    """Test starting a transfer with should_store_blocks=False."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 0
    spec_resource_manager = MagicMock()
    resource_manager = create_mock_resource_manager(
        kv_cache_manager=kv_cache_manager, spec_resource_manager=spec_resource_manager
    )
    manager = AsyncTransferManager(resource_manager, should_store_blocks=False)

    request = create_mock_request(42)
    manager.start_transfer(request)

    # Check request is tracked
    assert 42 in manager._requests_in_transfer
    transfer_metadata = manager._request_transfer_metadata[42]
    assert transfer_metadata.block_id is None  # No block stored
    assert transfer_metadata.counter == 1

    # Check KV cache manager was NOT called
    kv_cache_manager.store_blocks_for_reuse.assert_not_called()
    spec_resource_manager.free_resources.assert_called_once_with(request)

    assert manager.end_transfer(request)

    kv_cache_manager.unpin_blocks_by_id.assert_not_called()


def test_release_locally_quiesced_kv_retains_global_transfer_state():
    kv_cache_manager = MagicMock()
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager, should_store_blocks=False)
    request = create_mock_request(42)
    manager.start_transfer(request)

    assert manager.release_locally_quiesced_kv(request)
    kv_cache_manager.free_resources.assert_called_once_with(request)
    assert manager.requests_in_transfer()[42] is request
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    assert not manager.release_locally_quiesced_kv(request)
    kv_cache_manager.free_resources.assert_called_once_with(request)

    assert manager.end_transfer(request)
    assert 42 not in manager._locally_released_kv_request_ids


def test_release_locally_quiesced_kv_rejects_pinned_reuse_blocks():
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 100
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager, should_store_blocks=True)
    request = create_mock_request(42)
    manager.start_transfer(request)

    with pytest.raises(RuntimeError, match="pinned blocks"):
        manager.release_locally_quiesced_kv(request)

    kv_cache_manager.free_resources.assert_not_called()


def create_local_reclaim_executor(manager, transfer_statuses):
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = MagicMock()
    executor.kv_cache_transceiver.check_context_transfer_status_with_local_completion.side_effect = transfer_statuses
    executor.async_transfer_manager = manager
    executor._nvbug_6448152_local_kv_reclaim = True
    executor._nvbug_6448152_local_kv_reclaim_count = 0
    executor.global_rank = 0
    executor._end_transfer_and_maybe_terminate = MagicMock(side_effect=manager.end_transfer)
    executor._check_cache_transfer_errors = MagicMock()
    return executor


def test_local_kv_reclaim_waits_for_later_global_commit():
    kv_cache_manager = MagicMock()
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager, should_store_blocks=False)
    request = create_mock_request(42)
    manager.start_transfer(request)
    executor = create_local_reclaim_executor(manager, [([], [], [42]), ([42], [], [])])

    executor._check_disagg_ctx_cache_transfer_status(0)
    kv_cache_manager.free_resources.assert_called_once_with(request)
    assert manager.requests_in_transfer()[42] is request
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    executor._check_disagg_ctx_cache_transfer_status(0)
    executor._end_transfer_and_maybe_terminate.assert_called_once_with(request)
    assert 42 not in manager.requests_in_transfer()
    assert request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE


@pytest.mark.parametrize("global_error", [False, True])
def test_local_kv_reclaim_ignores_combined_commit_and_excludes_failure(global_error):
    kv_cache_manager = MagicMock()
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager, should_store_blocks=False)
    request = create_mock_request(42)
    manager.start_transfer(request)
    if global_error:
        request.state = LlmRequestState.DISAGG_TRANS_ERROR
        transfer_status = ([], [42], [])
    else:
        transfer_status = ([42], [], [42])
    executor = create_local_reclaim_executor(manager, [transfer_status])

    executor._check_disagg_ctx_cache_transfer_status(0)

    kv_cache_manager.free_resources.assert_not_called()
    if global_error:
        assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    else:
        assert request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert 42 not in manager.requests_in_transfer()


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
