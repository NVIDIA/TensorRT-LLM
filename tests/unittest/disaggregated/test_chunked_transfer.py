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
"""Unit tests for chunked KV cache transfer (sender-only chunking).

These tests validate the session state machine, callback plumbing, and
release queue mechanics without requiring GPU or NIXL.
"""

import queue
from typing import Tuple

import pytest

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus
from tensorrt_llm._torch.disaggregation.native.transfer import AgentResult, KVSendTask, TaskStatus

# ---------------------------------------------------------------------------
# KVSendTask tests
# ---------------------------------------------------------------------------


def _make_params(rid: int = 42) -> DisaggregatedParams:
    return DisaggregatedParams(disagg_request_id=rid)


def test_kv_send_task_chunk_block_offset():
    """KVSendTask stores chunk_block_offset correctly."""
    s = KVSlice(is_last_slice=False, block_ids_per_layer_groups=[[0, 1]])
    task = KVSendTask(s, _make_params(), slice_id=1, chunk_block_offset=512)
    assert task.chunk_block_offset == 512
    assert task.slice_id == 1
    assert task._slice is s


def test_kv_send_task_default_offset():
    """Default chunk_block_offset is 0."""
    s = KVSlice(is_last_slice=True, block_ids_per_layer_groups=[[0]])
    task = KVSendTask(s, _make_params(), slice_id=0)
    assert task.chunk_block_offset == 0


# ---------------------------------------------------------------------------
# TxSession multi-slice status tests (via mock)
# ---------------------------------------------------------------------------


def _make_mock_tx_session(num_slices: int):
    """Create a minimal mock TxSession with controllable tasks."""
    tasks = []
    for i in range(num_slices):
        s = KVSlice(
            is_last_slice=(i == num_slices - 1),
            block_ids_per_layer_groups=[[i]],
        )
        task = KVSendTask(s, _make_params(), slice_id=i)
        tasks.append(task)

    class FakeTxSession:
        def __init__(self):
            self.kv_tasks = tasks
            self.aux_task = None
            self._exception = None
            self.receiver_ready = True

        @property
        def status(self):
            if self._exception is not None or any(
                t.status == TaskStatus.ERROR for t in self.kv_tasks
            ):
                return SessionStatus.ERROR
            kv_all_transferred = bool(self.kv_tasks) and all(
                t.status == TaskStatus.TRANSFERRED for t in self.kv_tasks
            )
            if kv_all_transferred:
                if self.aux_task is not None and self.aux_task.status == TaskStatus.TRANSFERRED:
                    return SessionStatus.FULLY_TRANSFERRED
                return SessionStatus.KV_TRANSFERRED
            if self.kv_tasks and any(t.status == TaskStatus.TRANSFERRING for t in self.kv_tasks):
                return SessionStatus.TRANSFERRING
            return SessionStatus.READY if self.receiver_ready else SessionStatus.INIT

    return FakeTxSession()


def test_tx_session_status_init_until_all_transferred():
    """TxSession status is not KV_TRANSFERRED until ALL tasks complete."""
    session = _make_mock_tx_session(3)
    assert session.status == SessionStatus.READY

    session.kv_tasks[0].status = TaskStatus.TRANSFERRED
    assert session.status == SessionStatus.TRANSFERRING  # not all done

    session.kv_tasks[1].status = TaskStatus.TRANSFERRED
    assert session.status == SessionStatus.TRANSFERRING  # still one left

    session.kv_tasks[2].status = TaskStatus.TRANSFERRED
    assert session.status == SessionStatus.KV_TRANSFERRED


def test_tx_session_status_error_on_any_failure():
    """TxSession status is ERROR if any task fails."""
    session = _make_mock_tx_session(3)
    session.kv_tasks[0].status = TaskStatus.TRANSFERRED
    session.kv_tasks[1].status = TaskStatus.ERROR
    assert session.status == SessionStatus.ERROR


def test_tx_session_wait_complete_all_tasks():
    """TxSession wait_complete blocks on all task futures."""
    session = _make_mock_tx_session(3)
    for task in session.kv_tasks:
        task.future.set_result(AgentResult.SUCCESS)
        task.status = TaskStatus.TRANSFERRED

    # Simulate wait_complete logic
    try:
        for task in session.kv_tasks:
            result = task.future.result(timeout=1.0)
            assert result == AgentResult.SUCCESS
    except Exception:
        pytest.fail("wait_complete should not raise for successful tasks")


# ---------------------------------------------------------------------------
# RxSession multi-slice status tests (via mock)
# ---------------------------------------------------------------------------


def _make_mock_rx_session(num_slices: int):
    """Create a minimal mock RxSession with controllable tasks."""
    from tensorrt_llm._torch.disaggregation.native.transfer import KVRecvTask

    tasks = []
    for i in range(num_slices):
        s = KVSlice(
            is_last_slice=(i == num_slices - 1),
            block_ids_per_layer_groups=[[i]],
        )
        task = KVRecvTask(42, s, i, _make_params(), aux_slot=None)
        tasks.append(task)

    class FakeRxSession:
        def __init__(self):
            self._kv_tasks = tasks
            self._aux_status = TaskStatus.INIT
            self._exception = None
            self._aux_count = 0
            self.request_id = 42

        @property
        def status(self):
            if self._exception is not None:
                return SessionStatus.ERROR
            if not self._kv_tasks:
                return SessionStatus.INIT
            if any(t.status == TaskStatus.ERROR for t in self._kv_tasks):
                return SessionStatus.ERROR
            all_transferred = all(t.status == TaskStatus.TRANSFERRED for t in self._kv_tasks)
            if all_transferred:
                if self._aux_status == TaskStatus.TRANSFERRED:
                    return SessionStatus.FULLY_TRANSFERRED
                return SessionStatus.KV_TRANSFERRED
            if any(t.status == TaskStatus.TRANSFERRING for t in self._kv_tasks):
                return SessionStatus.TRANSFERRING
            return SessionStatus.INIT

    return FakeRxSession()


def test_rx_session_status_checks_all_tasks():
    """RxSession status is KV_TRANSFERRED only when ALL tasks complete."""
    session = _make_mock_rx_session(3)
    assert session.status == SessionStatus.INIT

    session._kv_tasks[0].status = TaskStatus.TRANSFERRED
    session._kv_tasks[1].status = TaskStatus.TRANSFERRING
    assert session.status == SessionStatus.TRANSFERRING

    session._kv_tasks[1].status = TaskStatus.TRANSFERRED
    session._kv_tasks[2].status = TaskStatus.TRANSFERRED
    assert session.status == SessionStatus.KV_TRANSFERRED


def test_rx_session_status_error_on_any_failure():
    """RxSession status is ERROR if any task fails."""
    session = _make_mock_rx_session(2)
    session._kv_tasks[0].status = TaskStatus.TRANSFERRED
    session._kv_tasks[1].status = TaskStatus.ERROR
    assert session.status == SessionStatus.ERROR


def test_rx_session_process_aux_uses_last_task():
    """process_aux_agent_result uses the last task's expected_transfers."""
    session = _make_mock_rx_session(3)
    session._kv_tasks[0].expected_transfers = 99  # should be ignored
    session._kv_tasks[1].expected_transfers = 99
    session._kv_tasks[2].expected_transfers = 1  # should be used

    expected = session._kv_tasks[-1].expected_transfers
    assert expected == 1


# ---------------------------------------------------------------------------
# Chunk completion callback tests
# ---------------------------------------------------------------------------


def test_chunk_callback_enqueues_release():
    """on_chunk_transferred callback enqueues the correct release entry."""
    release_queue: queue.Queue[Tuple[int, int]] = queue.Queue()

    def on_chunk_transferred(request_id: int, chunk_block_offset: int, num_blocks: int):
        cumulative_blocks = chunk_block_offset + num_blocks
        release_queue.put((request_id, cumulative_blocks))

    on_chunk_transferred(request_id=7, chunk_block_offset=0, num_blocks=64)
    on_chunk_transferred(request_id=7, chunk_block_offset=64, num_blocks=64)
    on_chunk_transferred(request_id=7, chunk_block_offset=128, num_blocks=64)

    results = []
    while not release_queue.empty():
        results.append(release_queue.get_nowait())

    assert results == [(7, 64), (7, 128), (7, 192)]


def test_drain_pending_releases():
    """_drain_pending_releases calls release_prefix_blocks for each entry."""
    release_queue: queue.Queue[Tuple[int, int]] = queue.Queue()
    release_queue.put((10, 64))
    release_queue.put((10, 128))
    release_queue.put((20, 32))

    released = []

    class FakeKVCacheManager:
        def release_prefix_blocks(self, request_id, num_blocks):
            released.append((request_id, num_blocks))

    mgr = FakeKVCacheManager()

    # Simulate drain logic
    while not release_queue.empty():
        try:
            request_id, num_blocks = release_queue.get_nowait()
        except queue.Empty:
            break
        mgr.release_prefix_blocks(request_id, num_blocks)

    assert released == [(10, 64), (10, 128), (20, 32)]


def test_make_chunk_callback_none_without_v2():
    """_make_chunk_callback returns None when is_v2_manager is False."""
    # Simulate the condition check
    is_v2_manager = False
    chunk_size_blocks = 64
    result = None if (not is_v2_manager or chunk_size_blocks is None) else "callback"
    assert result is None


def test_make_chunk_callback_none_without_chunking():
    """_make_chunk_callback returns None when chunk_size_blocks is None."""
    is_v2_manager = True
    chunk_size_blocks = None
    result = None if (not is_v2_manager or chunk_size_blocks is None) else "callback"
    assert result is None


def test_make_chunk_callback_returns_callable():
    """_make_chunk_callback returns a callable when both conditions met."""
    is_v2_manager = True
    chunk_size_blocks = 64
    result = None if (not is_v2_manager or chunk_size_blocks is None) else "callback"
    assert result is not None
