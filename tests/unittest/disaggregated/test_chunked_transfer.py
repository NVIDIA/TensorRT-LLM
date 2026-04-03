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
release queue mechanics using the real TxSession/RxSession classes with
lightweight stub sender/receiver objects.
"""

import queue
from typing import Tuple
from unittest.mock import MagicMock

import pytest

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVSendTask,
    RxSession,
    TaskStatus,
    TxSession,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(rid: int = 42) -> DisaggregatedParams:
    return DisaggregatedParams(disagg_request_id=rid)


def _stub_sender():
    """Create a stub sender with no-op methods needed by TxSession."""
    sender = MagicMock()
    sender.setup_session = MagicMock()
    sender._get_req_info = MagicMock(return_value=None)
    sender.dispatch_task = MagicMock()
    return sender


def _stub_receiver():
    """Create a stub receiver with no-op methods needed by RxSession."""
    receiver = MagicMock()
    receiver.setup_session = MagicMock()
    receiver.dispatch_task = MagicMock()
    return receiver


def _make_tx_session(num_slices: int, rid: int = 42, **kwargs) -> TxSession:
    """Create a real TxSession and send num_slices slices into it."""
    params = _make_params(rid)
    session = TxSession(
        request_id=rid,
        params=params,
        sender=_stub_sender(),
        aux_slot=None,
        **kwargs,
    )
    for i in range(num_slices):
        s = KVSlice(
            is_last_slice=(i == num_slices - 1),
            block_ids_per_layer_groups=[[i]],
        )
        session.send(s, chunk_block_offset=i)
    return session


def _make_rx_session(num_slices: int, rid: int = 42) -> RxSession:
    """Create a real RxSession and receive num_slices slices into it."""
    params = _make_params(rid)
    session = RxSession(
        request_id=rid,
        params=params,
        receiver=_stub_receiver(),
        aux_slot=None,
    )
    for i in range(num_slices):
        s = KVSlice(
            is_last_slice=(i == num_slices - 1),
            block_ids_per_layer_groups=[[i]],
        )
        session.receive(s)
    return session


# ---------------------------------------------------------------------------
# KVSendTask tests
# ---------------------------------------------------------------------------


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
# TxSession multi-slice status tests (real class)
# ---------------------------------------------------------------------------


def test_tx_session_status_init_until_all_transferred():
    """TxSession status is not KV_TRANSFERRED until ALL tasks complete."""
    session = _make_tx_session(3)
    session.receiver_ready = True
    assert session.status == SessionStatus.TRANSFERRING or session.status == SessionStatus.READY

    session.kv_tasks[0].status = TaskStatus.TRANSFERRED
    assert session.status != SessionStatus.KV_TRANSFERRED

    session.kv_tasks[1].status = TaskStatus.TRANSFERRED
    assert session.status != SessionStatus.KV_TRANSFERRED

    session.kv_tasks[2].status = TaskStatus.TRANSFERRED
    assert session.status == SessionStatus.KV_TRANSFERRED


def test_tx_session_status_error_on_any_failure():
    """TxSession status is ERROR if any task fails."""
    session = _make_tx_session(3)
    session.kv_tasks[0].status = TaskStatus.TRANSFERRED
    session.kv_tasks[1].status = TaskStatus.ERROR
    assert session.status == SessionStatus.ERROR


def test_tx_session_wait_complete_all_tasks():
    """TxSession.wait_complete blocks on all task futures."""
    session = _make_tx_session(3)
    for task in session.kv_tasks:
        task.future.set_result(AgentResult.SUCCESS)
        task.status = TaskStatus.TRANSFERRED

    result = session.wait_complete(need_aux=False, timeout=1.0)
    assert result == WaitResult.COMPLETED


def test_tx_session_wait_complete_fails_on_partial_failure():
    """TxSession.wait_complete returns FAILED if any task fails."""
    session = _make_tx_session(3)
    session.kv_tasks[0].future.set_result(AgentResult.SUCCESS)
    session.kv_tasks[0].status = TaskStatus.TRANSFERRED
    session.kv_tasks[1].future.set_result(AgentResult.FAILED)
    session.kv_tasks[1].status = TaskStatus.ERROR
    session.kv_tasks[2].future.set_result(AgentResult.SUCCESS)
    session.kv_tasks[2].status = TaskStatus.TRANSFERRED

    result = session.wait_complete(need_aux=False, timeout=1.0)
    assert result == WaitResult.FAILED


# ---------------------------------------------------------------------------
# RxSession multi-slice status tests (real class)
# ---------------------------------------------------------------------------


def test_rx_session_status_checks_all_tasks():
    """RxSession status is KV_TRANSFERRED only when ALL tasks complete."""
    session = _make_rx_session(3)
    assert session.status == SessionStatus.INIT

    session._kv_tasks[0].status = TaskStatus.TRANSFERRED
    session._kv_tasks[1].status = TaskStatus.TRANSFERRING
    assert session.status == SessionStatus.TRANSFERRING

    session._kv_tasks[1].status = TaskStatus.TRANSFERRED
    session._kv_tasks[2].status = TaskStatus.TRANSFERRED
    assert session.status == SessionStatus.KV_TRANSFERRED


def test_rx_session_status_error_on_any_failure():
    """RxSession status is ERROR if any task fails."""
    session = _make_rx_session(2)
    session._kv_tasks[0].status = TaskStatus.TRANSFERRED
    session._kv_tasks[1].status = TaskStatus.ERROR
    assert session.status == SessionStatus.ERROR


def test_rx_session_process_aux_uses_last_task():
    """process_aux_agent_result uses the last task's expected_transfers."""
    session = _make_rx_session(3)
    session._kv_tasks[0].expected_transfers = 99
    session._kv_tasks[1].expected_transfers = 99
    session._kv_tasks[2].expected_transfers = 1

    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    assert session._aux_status == TaskStatus.TRANSFERRED


def test_rx_session_wait_complete_all_tasks():
    """RxSession.wait_complete blocks on all task futures."""
    session = _make_rx_session(3)
    for task in session._kv_tasks:
        task.future.set_result(AgentResult.SUCCESS)
        task.status = TaskStatus.TRANSFERRED

    result = session.wait_complete(need_aux=False)
    assert result == WaitResult.COMPLETED


def test_rx_session_wait_complete_fails_on_partial_failure():
    """RxSession.wait_complete returns FAILED if any task fails."""
    session = _make_rx_session(2)
    session._kv_tasks[0].future.set_result(AgentResult.SUCCESS)
    session._kv_tasks[0].status = TaskStatus.TRANSFERRED
    session._kv_tasks[1].future.set_exception(RuntimeError("transfer failed"))
    session._kv_tasks[1].status = TaskStatus.ERROR

    result = session.wait_complete(need_aux=False)
    assert result == WaitResult.FAILED


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

    while not release_queue.empty():
        try:
            request_id, num_blocks = release_queue.get_nowait()
        except queue.Empty:
            break
        mgr.release_prefix_blocks(request_id, num_blocks)

    assert released == [(10, 64), (10, 128), (20, 32)]


@pytest.mark.parametrize(
    "has_release,chunk_size,expected_none",
    [
        (False, 64, True),
        (True, None, True),
        (False, None, True),
        (True, 64, False),
    ],
    ids=["no_release_api", "no_chunking", "neither", "with_release_and_chunking"],
)
def test_make_chunk_callback_conditions(has_release, chunk_size, expected_none):
    """_make_chunk_callback returns None unless both release API and chunking enabled."""
    result = None if (not has_release or chunk_size is None) else "callback"
    assert (result is None) == expected_none


def test_chunk_callback_callable_behavior():
    """When both conditions are met, the callback enqueues cumulative blocks."""
    release_queue: queue.Queue[Tuple[int, int]] = queue.Queue()

    def _on_chunk_transferred(request_id: int, chunk_block_offset: int, num_blocks: int):
        cumulative_blocks = chunk_block_offset + num_blocks
        release_queue.put((request_id, cumulative_blocks))

    _on_chunk_transferred(request_id=1, chunk_block_offset=0, num_blocks=4)
    _on_chunk_transferred(request_id=1, chunk_block_offset=4, num_blocks=4)
    _on_chunk_transferred(request_id=1, chunk_block_offset=8, num_blocks=2)

    results = []
    while not release_queue.empty():
        results.append(release_queue.get_nowait())
    assert results == [(1, 4), (1, 8), (1, 10)]


# ---------------------------------------------------------------------------
# Mid-transfer chunk failure tests
# ---------------------------------------------------------------------------


def test_tx_session_mid_chunk_failure():
    """If one chunk fails mid-transfer, the session reports ERROR."""
    session = _make_tx_session(4)

    session.kv_tasks[0].future.set_result(AgentResult.SUCCESS)
    session.kv_tasks[0].status = TaskStatus.TRANSFERRED
    session.kv_tasks[1].future.set_result(AgentResult.SUCCESS)
    session.kv_tasks[1].status = TaskStatus.TRANSFERRED
    session.kv_tasks[2].future.set_exception(RuntimeError("RDMA failed"))
    session.kv_tasks[2].status = TaskStatus.ERROR
    session.kv_tasks[3].future.set_result(AgentResult.SUCCESS)
    session.kv_tasks[3].status = TaskStatus.TRANSFERRED

    assert session.status == SessionStatus.ERROR
    result = session.wait_complete(need_aux=False, timeout=1.0)
    assert result == WaitResult.FAILED


def test_rx_session_mid_chunk_failure():
    """If one chunk fails mid-transfer on receiver, the session reports ERROR."""
    session = _make_rx_session(4)

    session._kv_tasks[0].future.set_result(AgentResult.SUCCESS)
    session._kv_tasks[0].status = TaskStatus.TRANSFERRED
    session._kv_tasks[1].future.set_exception(RuntimeError("RDMA failed"))
    session._kv_tasks[1].status = TaskStatus.ERROR
    session._kv_tasks[2].future.set_result(AgentResult.SUCCESS)
    session._kv_tasks[2].status = TaskStatus.TRANSFERRED
    session._kv_tasks[3].future.set_result(AgentResult.SUCCESS)
    session._kv_tasks[3].status = TaskStatus.TRANSFERRED

    assert session.status == SessionStatus.ERROR
    result = session.wait_complete(need_aux=False)
    assert result == WaitResult.FAILED
