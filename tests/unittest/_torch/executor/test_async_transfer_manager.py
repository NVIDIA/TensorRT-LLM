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

from tensorrt_llm._torch.pyexecutor.py_executor import AsyncTransferManager
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
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
    resource_managers = {}

    if kv_cache_manager is not None:
        resource_managers[ResourceManagerType.KV_CACHE_MANAGER] = kv_cache_manager

    if seq_slot_manager is not None:
        resource_managers[ResourceManagerType.SEQ_SLOT_MANAGER] = seq_slot_manager

    if spec_resource_manager is not None:
        resource_managers[ResourceManagerType.SPEC_RESOURCE_MANAGER] = spec_resource_manager

    return ResourceManager(resource_managers)


def test_start_transfer_single_request():
    """Test starting a single transfer."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = [100]
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

    assert transfer_metadata.pinned_block_ids == [100]
    assert transfer_metadata.counter == 1

    # Check state was updated
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    # Check KV cache manager was called
    kv_cache_manager.store_blocks_for_reuse.assert_called_once_with(request, True)

    # Check seq slot manager was called to free resources
    seq_slot_manager.free_resources.assert_called_once_with(request)

    manager.end_transfer(request)
    kv_cache_manager.unpin_blocks_by_id.assert_called_once_with([100])


def test_start_transfer_multiple_transfers_same_request():
    """Test starting multiple transfers for the same request."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = [100]
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
    kv_cache_manager.unpin_blocks_by_id.assert_called_once_with([100])


def test_named_transfer_owner_can_retire_independently():
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = [100]
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)
    request = create_mock_request(42)

    manager.start_transfer(request, owner="python_native")
    manager.start_transfer(request)

    assert manager.requests_with_owner("python_native") == {42: request}
    assert manager.has_transfer_owner(request, "python_native")
    assert not manager.end_transfer(request, owner="python_native")
    assert manager.requests_with_owner("python_native") == {}
    assert 42 in manager.requests_in_transfer()
    kv_cache_manager.unpin_blocks_by_id.assert_not_called()

    assert manager.end_transfer(request)
    kv_cache_manager.unpin_blocks_by_id.assert_called_once_with([100])


def test_start_transfer_rejects_different_request_with_same_id():
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = [100]
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)
    request = create_mock_request(42)
    replacement = create_mock_request(42)

    manager.start_transfer(request, owner="python_native")

    with pytest.raises(RuntimeError, match="different transfer owner"):
        manager.start_transfer(replacement, owner="connector")

    metadata = manager._request_transfer_metadata[42]
    assert manager.requests_in_transfer() == {42: request}
    assert metadata.counter == 1
    assert metadata.owners == {"python_native"}
    kv_cache_manager.store_blocks_for_reuse.assert_called_once_with(request, True)


def test_end_transfer_rejects_different_request_with_same_id():
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = [100]
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)
    request = create_mock_request(42)
    replacement = create_mock_request(42)

    manager.start_transfer(request, owner="python_native")

    with pytest.raises(RuntimeError, match="different active transfer owner"):
        manager.end_transfer(replacement, owner="python_native")

    metadata = manager._request_transfer_metadata[42]
    assert manager.requests_in_transfer() == {42: request}
    assert metadata.counter == 1
    assert metadata.owners == {"python_native"}
    kv_cache_manager.unpin_blocks_by_id.assert_not_called()


def test_final_transfer_unpin_failure_preserves_owner_for_retry():
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = [100]
    kv_cache_manager.unpin_blocks_by_id.side_effect = [RuntimeError("unpin failed"), None]
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)
    request = create_mock_request(42)

    manager.start_transfer(request, owner="python_native")

    with pytest.raises(RuntimeError, match="unpin failed"):
        manager.end_transfer(request, owner="python_native")

    metadata = manager._request_transfer_metadata[42]
    assert manager.requests_in_transfer() == {42: request}
    assert metadata.counter == 1
    assert metadata.owners == {"python_native"}
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    assert manager.end_transfer(request, owner="python_native")
    assert manager.requests_in_transfer() == {}
    assert manager._request_transfer_metadata == {}
    assert request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert kv_cache_manager.unpin_blocks_by_id.call_count == 2


def test_resource_release_failure_is_retained_as_in_doubt():
    first_manager = MagicMock()
    failing_manager = MagicMock()
    failing_manager.free_resources.side_effect = RuntimeError(
        "release may have partially completed"
    )
    manager = ResourceManager(
        {
            ResourceManagerType.SEQ_SLOT_MANAGER: failing_manager,
            ResourceManagerType.SPEC_RESOURCE_MANAGER: first_manager,
        }
    )
    request = create_mock_request(42)

    with pytest.raises(RuntimeError, match="partially completed"):
        manager.free_resources(request)

    progress = manager._resource_release_progress[id(request)]
    assert progress.request is request
    assert progress.next_manager == 1
    assert progress.in_doubt_manager == 1
    assert manager.has_in_doubt_resource_releases()

    with pytest.raises(RuntimeError, match="outcome is in doubt"):
        manager.free_resources(request)

    first_manager.free_resources.assert_called_once_with(request)
    failing_manager.free_resources.assert_called_once_with(request)


def test_successful_resource_release_clears_progress():
    resource = MagicMock()
    manager = ResourceManager({ResourceManagerType.SEQ_SLOT_MANAGER: resource})
    request = create_mock_request(42)

    manager.free_resources(request)

    resource.free_resources.assert_called_once_with(request)
    assert manager._resource_release_progress == {}
    assert not manager.has_in_doubt_resource_releases()


def test_transfer_without_storing_blocks():
    """Test starting a transfer with should_store_blocks=False."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = []
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
    assert transfer_metadata.pinned_block_ids is None  # No blocks stored
    assert transfer_metadata.counter == 1

    # Check KV cache manager was NOT called
    kv_cache_manager.store_blocks_for_reuse.assert_not_called()
    spec_resource_manager.free_resources.assert_called_once_with(request)

    assert manager.end_transfer(request)

    kv_cache_manager.unpin_blocks_by_id.assert_not_called()


def test_end_transfer_preserves_error_state():
    """Test that end_transfer does not overwrite error state."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = [100]
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
    kv_cache_manager.store_blocks_for_reuse.return_value = [100]
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


def test_first_owner_admission_is_published_before_external_mutations():
    request = create_mock_request(42)
    kv_cache_manager = MagicMock()
    seq_slot_manager = MagicMock()
    spec_resource_manager = MagicMock()
    resource_manager = create_mock_resource_manager(
        kv_cache_manager=kv_cache_manager,
        seq_slot_manager=seq_slot_manager,
        spec_resource_manager=spec_resource_manager,
    )
    manager = AsyncTransferManager(resource_manager)

    def assert_seq_release_is_rooted(released_request):
        assert released_request is request
        assert manager.requests_in_admission() == {42: request}
        progress = manager._transfer_admission_progress[42]
        assert progress.manager_in_doubt == ResourceManagerType.SEQ_SLOT_MANAGER
        assert progress.completed_managers == set()

    def assert_spec_release_is_rooted(released_request):
        assert released_request is request
        assert manager.requests_in_admission() == {42: request}
        progress = manager._transfer_admission_progress[42]
        assert progress.manager_in_doubt == ResourceManagerType.SPEC_RESOURCE_MANAGER
        assert progress.completed_managers == {ResourceManagerType.SEQ_SLOT_MANAGER}

    def assert_pin_is_rooted(pinned_request, pin_blocks):
        assert pinned_request is request
        assert pin_blocks is True
        assert manager.requests_in_admission() == {42: request}
        progress = manager._transfer_admission_progress[42]
        assert progress.manager_in_doubt is None
        assert progress.completed_managers == {
            ResourceManagerType.SEQ_SLOT_MANAGER,
            ResourceManagerType.SPEC_RESOURCE_MANAGER,
        }
        assert progress.state_updated
        assert progress.pin_started
        assert not progress.pin_complete
        return [100, 101]

    seq_slot_manager.free_resources.side_effect = assert_seq_release_is_rooted
    spec_resource_manager.free_resources.side_effect = assert_spec_release_is_rooted
    kv_cache_manager.store_blocks_for_reuse.side_effect = assert_pin_is_rooted

    manager.start_transfer(request, owner="python_native")

    assert manager.requests_in_admission() == {}
    assert manager.requests_in_transfer() == {42: request}
    metadata = manager._request_transfer_metadata[42]
    assert metadata.pinned_block_ids == [100, 101]
    assert metadata.owners == {"python_native"}
    assert manager.end_transfer(request, owner="python_native")
    kv_cache_manager.unpin_blocks_by_id.assert_called_once_with([100, 101])


def test_partial_transfer_release_failure_is_fail_closed_and_not_replayed():
    request = create_mock_request(42)
    kv_cache_manager = MagicMock()
    seq_slot_manager = MagicMock()
    spec_resource_manager = MagicMock()
    spec_resource_manager.free_resources.side_effect = RuntimeError(
        "spec release may have completed"
    )
    resource_manager = create_mock_resource_manager(
        kv_cache_manager=kv_cache_manager,
        seq_slot_manager=seq_slot_manager,
        spec_resource_manager=spec_resource_manager,
    )
    manager = AsyncTransferManager(resource_manager)

    with pytest.raises(RuntimeError, match="may have completed"):
        manager.start_transfer(request, owner="python_native")

    progress = manager._transfer_admission_progress[42]
    assert progress.request is request
    assert progress.completed_managers == {ResourceManagerType.SEQ_SLOT_MANAGER}
    assert progress.manager_in_doubt == ResourceManagerType.SPEC_RESOURCE_MANAGER
    assert manager.requests_in_admission() == {42: request}
    assert manager.has_in_doubt_admissions()
    assert manager.has_any_inflight_requests()
    assert resource_manager.has_in_doubt_resource_releases()

    with pytest.raises(RuntimeError, match="refusing to replay"):
        manager.start_transfer(request, owner="python_native")
    with pytest.raises(RuntimeError, match="admission is incomplete"):
        manager.end_transfer(request, owner="python_native")
    with pytest.raises(RuntimeError, match="different transfer admission"):
        manager.start_transfer(create_mock_request(42), owner="python_native")
    with pytest.raises(RuntimeError, match="outcome is in doubt"):
        resource_manager.free_resources(request)

    seq_slot_manager.free_resources.assert_called_once_with(request)
    spec_resource_manager.free_resources.assert_called_once_with(request)
    kv_cache_manager.store_blocks_for_reuse.assert_not_called()
    kv_cache_manager.free_resources.assert_not_called()


def test_pin_failure_retains_exact_admission_and_refuses_replay():
    request = create_mock_request(42)
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.side_effect = RuntimeError("pin outcome unknown")
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    with pytest.raises(RuntimeError, match="pin outcome unknown"):
        manager.start_transfer(request, owner="python_native")

    progress = manager._transfer_admission_progress[42]
    assert progress.request is request
    assert progress.pin_started
    assert not progress.pin_complete
    assert progress.pin_in_doubt
    assert progress.pinned_block_ids is None
    assert manager.requests_in_admission() == {42: request}
    assert manager.has_in_doubt_admissions()
    assert manager.has_any_inflight_requests()

    with pytest.raises(RuntimeError, match="refusing to replay"):
        manager.start_transfer(request, owner="python_native")
    with pytest.raises(RuntimeError, match="admission is incomplete"):
        manager.end_transfer(request, owner="python_native")

    kv_cache_manager.store_blocks_for_reuse.assert_called_once_with(request, True)
    kv_cache_manager.unpin_blocks_by_id.assert_not_called()


def test_empty_pinned_block_list_is_a_valid_completed_pin():
    request = create_mock_request(42)
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = []
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    manager.start_transfer(request)

    progress = manager._request_transfer_metadata[42]
    assert progress.pinned_block_ids == []
    assert manager.end_transfer(request)
    kv_cache_manager.unpin_blocks_by_id.assert_called_once_with([])


def test_early_transfer_release_is_not_repeated_by_final_release():
    request = create_mock_request(42)
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = [100]
    seq_slot_manager = MagicMock()
    spec_resource_manager = MagicMock()
    resource_manager = create_mock_resource_manager(
        kv_cache_manager=kv_cache_manager,
        seq_slot_manager=seq_slot_manager,
        spec_resource_manager=spec_resource_manager,
    )
    manager = AsyncTransferManager(resource_manager)

    manager.start_transfer(request)
    assert resource_manager.has_pending_resource_releases()
    assert manager.end_transfer(request)
    resource_manager.free_resources(request)

    seq_slot_manager.free_resources.assert_called_once_with(request)
    spec_resource_manager.free_resources.assert_called_once_with(request)
    kv_cache_manager.free_resources.assert_called_once_with(request)
    assert resource_manager._transfer_resource_release_progress == {}
    assert resource_manager._resource_release_progress == {}
    assert not resource_manager.has_pending_resource_releases()


def test_transfer_admission_closes_before_shutdown():
    request = create_mock_request(42)
    kv_cache_manager = MagicMock()
    resource_manager = create_mock_resource_manager(kv_cache_manager=kv_cache_manager)
    manager = AsyncTransferManager(resource_manager)

    manager.begin_shutdown()

    with pytest.raises(RuntimeError, match="admission is closed"):
        manager.start_transfer(request)
    kv_cache_manager.store_blocks_for_reuse.assert_not_called()
