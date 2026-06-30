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

These tests validate the session state machine using the real
TxSession/RxSession classes with lightweight stub sender/receiver objects.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import (
    KVSlice,
    SessionStatus,
    WaitResult,
    project_blocks_to_global_chunk,
)
from tensorrt_llm._torch.disaggregation.native.transfer import (
    AgentResult,
    KVSendTask,
    RecvReqInfo,
    RxSession,
    Sender,
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
        **kwargs,
    )
    for i in range(num_slices):
        s = KVSlice(
            is_last_slice=(i == num_slices - 1),
            block_ids_per_layer_groups=[[i]],
            chunk_block_offset=i,
        )
        session.send(s)
    return session


def _make_rx_session(num_slices: int, rid: int = 42) -> RxSession:
    """Create a real RxSession and receive num_slices slices into it."""
    params = _make_params(rid)
    session = RxSession(
        request_id=rid,
        params=params,
        receiver=_stub_receiver(),
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
    """KVSendTask reads chunk_block_offset from the slice."""
    s = KVSlice(is_last_slice=False, block_ids_per_layer_groups=[[0, 1]], chunk_block_offset=512)
    task = KVSendTask(s, _make_params(), slice_id=1)
    assert task._slice.chunk_block_offset == 512
    assert task.slice_id == 1
    assert task._slice is s


def test_kv_send_task_default_offset():
    """Default chunk_block_offset on KVSlice is 0."""
    s = KVSlice(is_last_slice=True, block_ids_per_layer_groups=[[0]])
    task = KVSendTask(s, _make_params(), slice_id=0)
    assert task._slice.chunk_block_offset == 0


# ---------------------------------------------------------------------------
# Global chunk projection tests
# ---------------------------------------------------------------------------


def test_chunk_projection_noops_when_chunk_is_outside_short_layer_group():
    """A shared chunk cursor past a short layer group's resident range is a no-op."""
    block_ids = np.array([10, 11, 12], dtype=np.int64)

    projected_ids = project_blocks_to_global_chunk(
        block_ids,
        chunk_block_offset=4,
        chunk_block_count=4,
        total_blocks=3,
    )

    assert projected_ids.size == 0


def test_chunk_projection_maps_prefix_reuse_suffix_by_overlap():
    """Destination suffixes are matched by overlap, not by raw chunk-offset indexing."""
    block_ids = np.array([104, 105, 106, 107], dtype=np.int64)

    first_chunk = project_blocks_to_global_chunk(
        block_ids,
        chunk_block_offset=0,
        chunk_block_count=4,
        total_blocks=8,
    )
    second_chunk = project_blocks_to_global_chunk(
        block_ids,
        chunk_block_offset=4,
        chunk_block_count=4,
        total_blocks=8,
    )

    assert first_chunk.size == 0
    assert np.array_equal(second_chunk, block_ids)


def test_build_kv_write_meta_projects_asymmetric_layer_group_chunk():
    """A short layer group's suffix blocks transfer with the overlapping global chunk."""
    peer_ri = SimpleNamespace(
        dp_rank=0,
        device_id=0,
        instance_name="decode",
        instance_rank=0,
        self_endpoint="tcp://decode:0",
    )

    extractor = MagicMock()
    extractor.page_table = SimpleNamespace(
        tokens_per_block=8,
        layer_groups=[
            SimpleNamespace(sliding_window_size=None),
            SimpleNamespace(sliding_window_size=None),
        ],
    )
    extractor.extract.side_effect = lambda block_ids, **_: SimpleNamespace(
        memory=SimpleNamespace(
            ptrs=np.asarray(block_ids, dtype=np.int64),
            bytes_per_region=1,
        )
    )

    mapper = MagicMock()
    mapper.map.side_effect = lambda src_region, dst_region: SimpleNamespace(
        src=src_region,
        dst=dst_region,
    )

    registrar = MagicMock()
    registrar.self_rank_info = SimpleNamespace()
    registrar.self_extractor = extractor
    registrar.get_peer_rank_info.return_value = peer_ri
    registrar.get_peer_overlap.return_value = SimpleNamespace(ranks=[0])
    registrar.should_send_kv.return_value = True
    registrar.get_pool_mapping.return_value = {
        (0, 0): (0, 0),
        (1, 0): (1, 0),
    }
    registrar.peer_extractor.return_value = extractor
    registrar.get_kv_map.return_value = mapper

    sender = Sender.__new__(Sender)
    sender._registrar = registrar

    task = KVSendTask(
        KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=[
                np.array([4, 5, 6, 7], dtype=np.int64),
                np.array([10, 11, 12], dtype=np.int64),
            ],
            token_range=SimpleNamespace(end=64),
            chunk_block_offset=4,
            chunk_size_blocks=4,
            total_blocks=8,
        ),
        _make_params(),
        slice_id=1,
    )
    req_info = RecvReqInfo(
        sender_req_id=42,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[
            np.array([104, 105, 106, 107], dtype=np.int64),
            np.array([200, 201, 202], dtype=np.int64),
        ],
        unique_rid=42,
    )

    write_meta = sender._build_kv_write_meta(task, req_info)

    assert np.array_equal(
        write_meta.src_ptrs,
        np.array([4, 5, 6, 7, 10, 11, 12], dtype=np.int64),
    )
    assert np.array_equal(
        write_meta.dst_ptrs,
        np.array([104, 105, 106, 107, 200, 201, 202], dtype=np.int64),
    )
    assert np.array_equal(write_meta.sizes, np.ones(7, dtype=np.int64))


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
        task.complete()

    result = session.wait_complete()
    assert result == WaitResult.COMPLETED


def test_tx_session_wait_complete_fails_on_partial_failure():
    """TxSession.wait_complete returns FAILED if any task fails."""
    session = _make_tx_session(3)
    session.kv_tasks[0].complete()
    session.kv_tasks[1].fail(RuntimeError("transfer failed"))
    session.kv_tasks[2].complete()

    result = session.wait_complete()
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
        task.complete()

    result = session.wait_complete()
    assert result == WaitResult.COMPLETED


def test_rx_session_wait_complete_fails_on_partial_failure():
    """RxSession.wait_complete returns FAILED if any task fails."""
    session = _make_rx_session(2)
    session._kv_tasks[0].complete()
    session._kv_tasks[1].fail(RuntimeError("transfer failed"))

    result = session.wait_complete()
    assert result == WaitResult.FAILED


# ---------------------------------------------------------------------------
# Mid-transfer chunk failure tests
# ---------------------------------------------------------------------------


def test_tx_session_mid_chunk_failure():
    """If one chunk fails mid-transfer, the session reports ERROR."""
    session = _make_tx_session(4)

    session.kv_tasks[0].complete()
    session.kv_tasks[1].complete()
    session.kv_tasks[2].fail(RuntimeError("RDMA failed"))
    session.kv_tasks[3].complete()

    assert session.status == SessionStatus.ERROR
    result = session.wait_complete()
    assert result == WaitResult.FAILED


def test_rx_session_mid_chunk_failure():
    """If one chunk fails mid-transfer on receiver, the session reports ERROR."""
    session = _make_rx_session(4)

    session._kv_tasks[0].complete()
    session._kv_tasks[1].fail(RuntimeError("RDMA failed"))
    session._kv_tasks[2].complete()
    session._kv_tasks[3].complete()

    assert session.status == SessionStatus.ERROR
    result = session.wait_complete()
    assert result == WaitResult.FAILED
