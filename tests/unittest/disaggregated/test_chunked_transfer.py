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
"""Unit tests for chunked and pipelined KV cache transfer (sender-only chunking).

These tests validate the session state machine using the real
TxSession/RxSession classes with lightweight stub sender/receiver objects.
"""

import threading
import time
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import (
    ChunkCoords,
    KVSlice,
    SessionStatus,
    WaitResult,
    project_blocks_to_global_chunk,
)
from tensorrt_llm._torch.disaggregation.native.transfer import (
    _KV_RESULT_PREFIX,
    NO_SLICE_ID,
    AgentResult,
    KVSendTask,
    RecvReqInfo,
    RxSession,
    Sender,
    TaskStatus,
    TxSession,
    WriteMeta,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

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


def _make_tx_session(num_slices: int, rid: int = 42, prompt_len: int = 8, **kwargs) -> TxSession:
    """Create a real TxSession and send num_slices slices into it."""
    params = _make_params(rid)
    session = TxSession(
        request_id=rid,
        params=params,
        sender=_stub_sender(),
        prompt_len=prompt_len,
        **kwargs,
    )
    for i in range(num_slices):
        s = KVSlice(
            is_last_slice=(i == num_slices - 1),
            block_ids_per_layer_groups=[[i]],
        )
        session.send(s)
    return session


def _make_rx_session(num_slices: int, rid: int = 42, prompt_len: int = 8) -> RxSession:
    """Create a real RxSession and receive num_slices slices into it."""
    params = _make_params(rid)
    session = RxSession(
        request_id=rid,
        params=params,
        receiver=_stub_receiver(),
        prompt_len=prompt_len,
    )
    for i in range(num_slices):
        s = KVSlice(
            is_last_slice=(i == num_slices - 1),
            block_ids_per_layer_groups=[[i]],
        )
        session.receive(s)
    return session


def _make_replay_sender(rid: int = 42) -> tuple[Sender, TxSession]:
    """Create a Sender/TxSession pair with controllable replay dispatch."""
    sender = Sender.__new__(Sender)
    sender._sessions_lock = threading.Lock()
    sender._req_infos = {}
    sender._session = None
    sender.setup_session = lambda session: setattr(sender, "_session", session)
    sender._get_session = lambda unique_rid: sender._session if unique_rid == rid else None
    sender._get_req_info = lambda unique_rid: sender._req_infos.get(unique_rid)

    def save_peer_req_info(info):
        sender._req_infos.setdefault(info.unique_rid, {})[info.instance_rank] = info

    sender._save_peer_req_info = save_peer_req_info
    sender._send_failed_result_to_receiver = MagicMock()
    sender.dispatch_task = MethodType(Sender.dispatch_task, sender)

    session = TxSession(
        request_id=rid,
        params=_make_params(rid),
        sender=sender,
        prompt_len=8,
    )
    return sender, session


def _replay_info(rid: int = 42) -> RecvReqInfo:
    return RecvReqInfo(
        sender_req_id=rid,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[],
        unique_rid=rid,
    )


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
        resident_block_end=3,
    )

    assert projected_ids.size == 0


@pytest.mark.parametrize(
    "resident_block_end,chunk_block_offset,expected",
    [
        (16, 0, np.arange(16, dtype=np.int64)),
        (32, 16, np.arange(16, 32, dtype=np.int64)),
    ],
    ids=["first_chunk", "later_chunk"],
)
def test_chunk_projection_maps_incrementally_allocated_source(
    resident_block_end, chunk_block_offset, expected
):
    """Source blocks end at the current chunk, not at the full prompt."""
    block_ids = np.arange(resident_block_end, dtype=np.int64)

    projected_ids = project_blocks_to_global_chunk(
        block_ids,
        chunk_block_offset=chunk_block_offset,
        chunk_block_count=16,
        resident_block_end=resident_block_end,
    )

    assert np.array_equal(projected_ids, expected)


def test_chunk_projection_maps_prefix_reuse_suffix_by_overlap():
    """Destination suffixes are matched by overlap, not by raw chunk-offset indexing."""
    block_ids = np.array([104, 105, 106, 107], dtype=np.int64)

    first_chunk = project_blocks_to_global_chunk(
        block_ids,
        chunk_block_offset=0,
        chunk_block_count=4,
        resident_block_end=8,
    )
    second_chunk = project_blocks_to_global_chunk(
        block_ids,
        chunk_block_offset=4,
        chunk_block_count=4,
        resident_block_end=8,
    )

    assert first_chunk.size == 0
    assert np.array_equal(second_chunk, block_ids)


def _make_projection_sender() -> Sender:
    """Create a Sender wired to a stub registrar with two non-windowed layer groups."""
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
    return sender


def _make_projection_task(slice_id: int = 1) -> KVSendTask:
    return KVSendTask(
        KVSlice(
            is_last_slice=True,
            block_ids_per_layer_groups=[
                np.array([4, 5, 6, 7], dtype=np.int64),
                np.array([10, 11, 12], dtype=np.int64),
            ],
            total_blocks=8,
            chunk=ChunkCoords(block_offset=4, block_count=4),
        ),
        _make_params(),
        slice_id=slice_id,
        prompt_len=64,
    )


def _make_projection_req_info(slice_id=None) -> RecvReqInfo:
    return RecvReqInfo(
        sender_req_id=42,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[
            np.array([104, 105, 106, 107], dtype=np.int64),
            np.array([200, 201, 202], dtype=np.int64),
        ],
        unique_rid=42,
        slice_id=slice_id,
    )


def test_build_kv_write_meta_projects_asymmetric_layer_group_chunk():
    """A short layer group's suffix blocks transfer with the overlapping global chunk."""
    sender = _make_projection_sender()

    write_meta = sender._build_kv_write_meta(_make_projection_task(), _make_projection_req_info())

    assert np.array_equal(
        write_meta.src_ptrs,
        np.array([4, 5, 6, 7, 10, 11, 12], dtype=np.int64),
    )
    assert np.array_equal(
        write_meta.dst_ptrs,
        np.array([104, 105, 106, 107, 200, 201, 202], dtype=np.int64),
    )
    assert np.array_equal(write_meta.sizes, np.ones(7, dtype=np.int64))
    # The sender's chunk index and the peer's task index are tracked separately;
    # a receiver that sends no slice_id is addressed as its single task 0.
    assert write_meta.sender_slice_id == 1
    assert write_meta.receiver_slice_id == 0


def test_whole_prompt_chunk_addresses_like_a_monolithic_slice():
    """A chunk spanning [0, total_blocks) writes exactly what an unpipelined send does.

    This is the degenerate slice _build_prefill_chunk produces when the whole prompt
    fits in one chunk, so the chunked branch must not perturb its addressing.
    """
    sender = _make_projection_sender()
    src_per_group = [
        np.arange(8, dtype=np.int64),
        np.array([10, 11, 12], dtype=np.int64),
    ]

    def task_for(chunk):
        return KVSendTask(
            KVSlice(
                is_last_slice=True,
                block_ids_per_layer_groups=src_per_group,
                total_blocks=8,
                chunk=chunk,
            ),
            _make_params(),
            slice_id=0,
            prompt_len=64,
        )

    chunked = sender._build_kv_write_meta(
        task_for(ChunkCoords(block_offset=0, block_count=8)), _make_projection_req_info()
    )
    monolithic = sender._build_kv_write_meta(task_for(None), _make_projection_req_info())

    assert np.array_equal(chunked.src_ptrs, monolithic.src_ptrs)
    assert np.array_equal(chunked.dst_ptrs, monolithic.dst_ptrs)
    assert np.array_equal(chunked.sizes, monolithic.sizes)


def test_build_kv_write_meta_echoes_receiver_slice_id():
    """receiver_slice_id comes from the peer's RecvReqInfo, not the sender's chunk index."""
    sender = _make_projection_sender()

    write_meta = sender._build_kv_write_meta(
        _make_projection_task(slice_id=1), _make_projection_req_info(slice_id=3)
    )

    assert write_meta.sender_slice_id == 1
    assert write_meta.receiver_slice_id == 3


# ---------------------------------------------------------------------------
# KV_AGENT_RESULT slice-id addressing tests
# ---------------------------------------------------------------------------


def _make_write_meta(sender_slice_id, receiver_slice_id) -> WriteMeta:
    empty = np.array([], dtype=np.int64)
    return WriteMeta(
        task=MagicMock(),
        expected_transfers=1,
        peer_name="decode0",
        peer_rank=0,
        peer_endpoint="tcp://decode:0",
        unique_rid=42,
        src_ptrs=empty,
        dst_ptrs=empty,
        sizes=empty,
        sender_slice_id=sender_slice_id,
        receiver_slice_id=receiver_slice_id,
    )


def test_send_kv_result_carries_both_slice_ids():
    """The result frame reports the sender's chunk and addresses the peer's own task."""
    sender = Sender.__new__(Sender)
    sender._instance_rank = 5
    dealer = MagicMock()
    sender._get_or_connect_thread_dealer = MagicMock(return_value=dealer)

    sender._send_kv_result_to_receiver(
        _make_write_meta(sender_slice_id=4, receiver_slice_id=2),
        is_last=True,
        result=AgentResult.SUCCESS,
    )

    (msg,), _ = dealer.send.call_args
    (rank, rid, sender_slice_id, receiver_slice_id, is_last, _code, _size) = (
        _KV_RESULT_PREFIX.unpack(msg[1])
    )
    assert (rank, rid, sender_slice_id, receiver_slice_id, is_last) == (5, 42, 4, 2, True)


def test_send_kv_result_without_task_reports_no_slice_id():
    """A result with no owning KVSendTask reports NO_SLICE_ID rather than chunk 0."""
    sender = Sender.__new__(Sender)
    sender._instance_rank = 5
    dealer = MagicMock()
    sender._get_or_connect_thread_dealer = MagicMock(return_value=dealer)

    sender._send_kv_result_to_receiver(
        _make_write_meta(sender_slice_id=None, receiver_slice_id=0),
        is_last=True,
        result=AgentResult.FAILED,
    )

    (msg,), _ = dealer.send.call_args
    (_rank, _rid, sender_slice_id, _receiver_slice_id, _is_last, _code, _size) = (
        _KV_RESULT_PREFIX.unpack(msg[1])
    )
    assert sender_slice_id == NO_SLICE_ID


def test_process_kv_agent_result_resolves_task_by_receiver_slice_id():
    """A sender chunk id far past the receiver's task count still resolves task 0."""
    session = _make_rx_session(1)
    session._kv_tasks[0].expected_transfers = 1
    session._receiver._bounce.is_bounced.return_value = False

    session.process_kv_agent_result(
        peer_rank=0,
        receiver_slice_id=0,
        sender_slice_id=4,
        is_last_slice=True,
        status=AgentResult.SUCCESS,
    )

    assert session._kv_tasks[0].status == TaskStatus.TRANSFERRED


def test_process_kv_agent_result_rejects_unknown_receiver_slice_id():
    """Indexing is bounded by the receiver's own task count, and names both ids."""
    session = _make_rx_session(1)

    with pytest.raises(AssertionError, match=r"receiver_slice_id=2.*sender_slice_id=0"):
        session.process_kv_agent_result(
            peer_rank=0,
            receiver_slice_id=2,
            sender_slice_id=0,
            is_last_slice=True,
            status=AgentResult.SUCCESS,
        )


def test_process_kv_agent_result_failure_attributes_sender_chunk():
    """A failed chunk names the sender chunk so the failure is attributable."""
    session = _make_rx_session(1)

    session.process_kv_agent_result(
        peer_rank=0,
        receiver_slice_id=0,
        sender_slice_id=3,
        is_last_slice=False,
        status=AgentResult.FAILED,
    )

    task = session._kv_tasks[0]
    assert task.status == TaskStatus.ERROR
    assert "sender_slice_id=3" in str(task._exception)
    assert session.status == SessionStatus.ERROR


# ---------------------------------------------------------------------------
# TxSession multi-slice status tests (real class)
# ---------------------------------------------------------------------------


def test_late_peer_replay_enqueues_before_concurrent_final_slice():
    """Replay holds dispatch ordering until every older slice is queued."""
    sender, session = _make_replay_sender()
    enqueued_slice_ids = []
    replay_build_started = threading.Event()
    release_replay = threading.Event()
    final_send_started = threading.Event()
    final_send_done = threading.Event()

    def build_write_meta(task, info):
        if task.slice_id == 0 and not replay_build_started.is_set():
            replay_build_started.set()
            assert release_replay.wait(timeout=5)
        return SimpleNamespace(task=task, peer_rank=info.instance_rank)

    sender._build_kv_write_meta = build_write_meta
    sender._enqueue = lambda meta: enqueued_slice_ids.append(meta.task.slice_id)

    session.send(KVSlice(is_last_slice=False, block_ids_per_layer_groups=[[0]]))
    session.send(KVSlice(is_last_slice=False, block_ids_per_layer_groups=[[1]]))

    info = _replay_info()
    replay_thread = threading.Thread(
        target=sender._respond_with_kv,
        args=(b"", [b"REQUEST_DATA", info.to_bytes()]),
    )
    replay_thread.start()
    assert replay_build_started.wait(timeout=5)

    def send_final_slice():
        final_send_started.set()
        session.send(KVSlice(is_last_slice=True, block_ids_per_layer_groups=[[2]]))
        final_send_done.set()

    final_thread = threading.Thread(target=send_final_slice)
    final_thread.start()
    assert final_send_started.wait(timeout=5)
    assert not final_send_done.wait(timeout=0.1)

    release_replay.set()
    replay_thread.join(timeout=5)
    final_thread.join(timeout=5)

    assert not replay_thread.is_alive()
    assert not final_thread.is_alive()
    assert enqueued_slice_ids == [0, 1, 2]


def test_late_peer_replay_includes_final_slice_sent_before_registration():
    """A final slice buffered before replay is queued once after older slices."""
    sender, session = _make_replay_sender()
    enqueued_slice_ids = []
    sender._build_kv_write_meta = lambda task, info: SimpleNamespace(
        task=task, peer_rank=info.instance_rank
    )
    sender._enqueue = lambda meta: enqueued_slice_ids.append(meta.task.slice_id)

    session.send(KVSlice(is_last_slice=False, block_ids_per_layer_groups=[[0]]))
    session.send(KVSlice(is_last_slice=False, block_ids_per_layer_groups=[[1]]))
    session.send(KVSlice(is_last_slice=True, block_ids_per_layer_groups=[[2]]))

    info = _replay_info()
    sender._respond_with_kv(b"", [b"REQUEST_DATA", info.to_bytes()])

    assert enqueued_slice_ids == [0, 1, 2]


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


def test_rx_session_process_aux_completes_at_expected_transfers():
    """Aux completes only once the expected transfer count is reached.

    The receiver always has exactly one task.
    """
    session = _make_rx_session(1)
    session._kv_tasks[0].expected_transfers = 2

    session.process_aux_agent_result(0, AgentResult.SUCCESS)
    assert session._aux_status != TaskStatus.TRANSFERRED

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


# ---------------------------------------------------------------------------
# Pipelined transfer tests
# ---------------------------------------------------------------------------


def test_pipelined_transfer_disabled_by_default():
    """pipeline_transfer_enabled reflects the configured flag."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = MagicMock()
    transceiver._enable_pipelined_transfer = False

    result = KvCacheTransceiverV2.pipeline_transfer_enabled.fget(transceiver)
    assert result is False


def test_pipelined_transfer_requires_chunked_prefill():
    """ValueError when pipelined transfer is enabled without chunked prefill."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import create_kv_cache_transceiver

    cache_transceiver_config = CacheTransceiverConfig(
        backend="NIXL",
        enable_pipelined_transfer=True,
    )

    with pytest.raises(
        ValueError,
        match="enable_chunked_prefill is required when enable_pipelined_transfer is set.",
    ):
        create_kv_cache_transceiver(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            cache_transceiver_config,
            enable_chunked_prefill=False,
        )


def test_pipelined_transfer_rejects_pipeline_parallelism(monkeypatch):
    """ValueError for pipeline parallelism when the disaggregated role is unknown."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import create_kv_cache_transceiver

    monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    mapping = MagicMock()
    mapping.pp_size = 2
    cache_transceiver_config = CacheTransceiverConfig(
        backend="NIXL",
        enable_pipelined_transfer=True,
    )

    with pytest.raises(
        ValueError,
        match="pipeline_parallel_size=1 is required when enable_pipelined_transfer is set.",
    ):
        create_kv_cache_transceiver(
            mapping,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            cache_transceiver_config,
            enable_chunked_prefill=True,
        )


def test_pipelined_transfer_allows_pipeline_parallelism_on_generation_server(monkeypatch):
    """Pipeline parallelism is allowed when the worker only receives KV cache."""
    from tensorrt_llm._torch.disaggregation import transceiver as transceiver_module
    from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import create_kv_cache_transceiver

    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
    transceiver = MagicMock()
    transceiver_cls = MagicMock(return_value=transceiver)
    monkeypatch.setattr(transceiver_module, "KvCacheTransceiverV2", transceiver_cls)

    mapping = MagicMock()
    mapping.pp_size = 2
    cache_transceiver_config = CacheTransceiverConfig(
        backend="NIXL",
        enable_pipelined_transfer=True,
    )

    result = create_kv_cache_transceiver(
        mapping,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        cache_transceiver_config,
        enable_chunked_prefill=True,
    )

    assert result is transceiver
    assert cache_transceiver_config.transceiver_runtime == "PYTHON"
    transceiver_cls.assert_called_once()


def test_python_transceiver_rejects_cpp_mamba_cache_manager():
    """Python transceiver requires separate Python-managed Mamba state."""
    from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import create_kv_cache_transceiver
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import CppMambaHybridCacheManager

    kv_cache_manager = object.__new__(CppMambaHybridCacheManager)
    cache_transceiver_config = CacheTransceiverConfig(
        backend="NIXL",
        transceiver_runtime="PYTHON",
    )

    # A hybrid manager arrives as both kv_cache_manager and mamba_cache_manager,
    # the way _util.py passes it.
    with pytest.raises(
        ValueError,
        match="cannot drive CppMambaHybridCacheManager",
    ):
        create_kv_cache_transceiver(
            MagicMock(),
            MagicMock(),
            kv_cache_manager,
            MagicMock(),
            cache_transceiver_config,
            kv_cache_manager,
        )


def test_pipelined_transfer_requires_gen_first_flow():
    """ValueError when a real request is not using gen-first flow."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.is_warmup = False
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor._validate_token_id_range = MagicMock()
    executor.sampler.validate_request = MagicMock()

    request = MagicMock()
    request.sampling_config = None
    request.py_beam_width = 1
    request.py_disaggregated_params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST
    )

    with pytest.raises(
        ValueError,
        match="schedule_style must be generation_first when enable_pipelined_transfer is set.",
    ):
        PyExecutor._validate_request(executor, request)


def test_pipelined_transfer_allows_non_disaggregated_request():
    """Requests without disaggregated parameters do not transfer KV cache."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.is_warmup = False
    executor.max_beam_width = 1
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor._validate_token_id_range = MagicMock()
    executor.sampler.validate_request = MagicMock()

    request = MagicMock()
    request.sampling_config = None
    request.py_beam_width = 1
    request.py_disaggregated_params = None

    PyExecutor._validate_request(executor, request)

    executor.sampler.validate_request.assert_called_once_with(request)


def test_send_kv_cache_early_only_sends_reused_prefixes():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor.kv_cache_manager.tokens_per_block = 32

    def make_request(
        rid, *, is_first_chunk, prepopulated, cancelled=False, last_chunk=(None, None)
    ):
        return SimpleNamespace(
            py_request_id=rid,
            is_context_only_request=True,
            is_finished_due_to_cancellation=cancelled,
            is_first_context_chunk=is_first_chunk,
            prepopulated_prompt_len=prepopulated,
            py_last_context_chunk=last_chunk,
            py_kv_prefix_sent=False,
        )

    completed = make_request(1, is_first_chunk=False, prepopulated=0, last_chunk=(0, 64))
    first_chunk = make_request(2, is_first_chunk=True, prepopulated=0)
    reused_prefix = make_request(3, is_first_chunk=True, prepopulated=128)
    cancelled = make_request(4, is_first_chunk=True, prepopulated=128, cancelled=True)

    result = PyExecutor._send_kv_cache_early(
        executor, [completed, first_chunk, reused_prefix, cancelled]
    )

    executor._send_kv_async.assert_called_once_with([reused_prefix])
    assert reused_prefix.py_last_context_chunk == (0, 128)
    assert reused_prefix.py_kv_prefix_sent
    assert not first_chunk.py_kv_prefix_sent
    assert not cancelled.py_kv_prefix_sent
    assert result is None

    # A partial-block or full-prefix hit reports an unaligned prepopulated
    # length; only whole blocks may be shipped ahead of the forward.
    executor._send_kv_async.reset_mock()
    unaligned = make_request(5, is_first_chunk=True, prepopulated=3894)
    below_one_block = make_request(6, is_first_chunk=True, prepopulated=31)

    PyExecutor._send_kv_cache_early(executor, [unaligned, below_one_block])

    executor._send_kv_async.assert_called_once_with([unaligned])
    assert unaligned.py_last_context_chunk == (0, 3872)
    assert below_one_block.py_last_context_chunk == (None, None)
    assert not below_one_block.py_kv_prefix_sent


def test_send_kv_cache_early_requires_pipelined_transfer():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.pipeline_transfer_enabled = False

    assert PyExecutor._send_kv_cache_early(executor, []) is None
    executor._send_kv_async.assert_not_called()


def test_pipelined_last_chunk_sends_and_finalizes():
    """respond_and_send_async sends the built chunk and finalizes on the last chunk."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    session = MagicMock()
    session.kv_tasks = []

    last_slice = KVSlice(
        is_last_slice=True,
        block_ids_per_layer_groups=[np.array([0, 1], dtype=np.int64)],
    )

    transceiver = MagicMock()
    transceiver._enable_pipelined_transfer = True
    transceiver.kv_transfer_timeout_ms = None
    transceiver._get_or_create_send_session.return_value = session
    transceiver._build_prefill_chunk.return_value = last_slice

    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=42),
        request_id=42,
        prompt_len=8,
        py_beam_width=1,
        py_kv_transfer_start_time=None,
        set_kv_cache_transfer_start=lambda _ts: None,
    )

    KvCacheTransceiverV2.respond_and_send_async(transceiver, request)

    transceiver._build_prefill_chunk.assert_called_once_with(request)
    session.send.assert_called_once_with(last_slice)
    transceiver._finalize_send.assert_called_once_with(request, session)
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS


def test_pipelined_non_last_chunk_does_not_finalize():
    """respond_and_send_async sends non-final chunks without finalizing."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    session = MagicMock()
    session.kv_tasks = []

    mid_slice = KVSlice(
        is_last_slice=False,
        block_ids_per_layer_groups=[np.array([0, 1], dtype=np.int64)],
    )

    transceiver = MagicMock()
    transceiver._enable_pipelined_transfer = True
    transceiver.kv_transfer_timeout_ms = None
    transceiver._get_or_create_send_session.return_value = session
    transceiver._build_prefill_chunk.return_value = mid_slice

    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=42),
        request_id=42,
        prompt_len=8,
        py_beam_width=1,
        py_kv_transfer_start_time=None,
        set_kv_cache_transfer_start=lambda _ts: None,
    )

    KvCacheTransceiverV2.respond_and_send_async(transceiver, request)

    session.send.assert_called_once_with(mid_slice)
    transceiver._finalize_send.assert_not_called()


def test_pipelined_multiple_chunks_use_real_builder_and_tx_session():
    """Drive two chunks through respond_and_send_async and a real TxSession."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    rid = 42
    tokens_per_block = 4
    source_block_ids = np.arange(4, dtype=np.int64)
    session = TxSession(
        request_id=rid,
        params=_make_params(rid),
        sender=_stub_sender(),
        prompt_len=16,
    )

    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._enable_pipelined_transfer = True
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._ever_had_send_session = False
    transceiver._transfer_worker = SimpleNamespace(create_tx_session=lambda _req: session)
    transceiver._reuse_adapter = SimpleNamespace(
        tokens_per_block=tokens_per_block,
        get_block_ids=lambda _req, _idx, _lg: source_block_ids,
    )
    transceiver._page_table = SimpleNamespace(
        layer_groups=[SimpleNamespace(sliding_window_size=None)]
    )
    transceiver._kv_cache_manager = SimpleNamespace(tokens_per_block=tokens_per_block)
    transceiver._dp_rank = 0
    transceiver._context_info_endpoint = "ctx"

    request = SimpleNamespace(
        py_disaggregated_params=_make_params(rid),
        request_id=rid,
        py_request_id=rid,
        prompt_len=16,
        py_beam_width=1,
        py_kv_send_session_retired=False,
        prepopulated_prompt_len=0,
        py_kv_prefix_sent=False,
        is_generation_only_request=lambda: False,
        set_kv_cache_transfer_start=lambda _ts: None,
        state=LlmRequestState.CONTEXT_INIT,
    )

    request.py_last_context_chunk = (0, 8)
    request.context_remaining_length = 8
    transceiver.respond_and_send_async(request)

    request.py_last_context_chunk = (8, 16)
    request.context_remaining_length = 0
    transceiver.respond_and_send_async(request)

    assert [task._slice.chunk for task in session.kv_tasks] == [
        ChunkCoords(block_offset=0, block_count=2),
        ChunkCoords(block_offset=2, block_count=2),
    ]
    assert [task._slice.block_ids_per_layer_groups[0].tolist() for task in session.kv_tasks] == [
        [0, 1],
        [2, 3],
    ]
    assert [task._slice.is_last_slice for task in session.kv_tasks] == [False, True]
    assert transceiver._send_sessions == {rid: session}
    assert transceiver._send_reqs == {rid: request}
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    session.close()


# ---------------------------------------------------------------------------
# Retired send sessions
# ---------------------------------------------------------------------------


def _make_send_session_transceiver(sessions=None):
    transceiver = MagicMock()
    transceiver._send_sessions = dict(sessions or {})
    return transceiver


def _make_retirable_request(retired: bool, rid: int = 42):
    return SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=rid),
        request_id=rid,
        py_kv_send_session_retired=retired,
        state=LlmRequestState.CONTEXT_INIT,
    )


def test_close_failed_send_session_without_request():
    """A failed session must be retired even before its request is registered."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    session = MagicMock()
    sessions = {42: session}
    reqs = {}

    KvCacheTransceiverV2._close_failed_sessions(
        MagicMock(), sessions, reqs, failed=[42], mark_retired=True
    )

    session.close.assert_called_once_with()
    assert sessions == {}
    assert reqs == {}


def test_get_or_create_send_session_refuses_retired_request():
    """Closing a send session drops the peer registration, so a new one is inert.

    Its tasks would sit in INIT forever, which is neither completed nor failed,
    so the request would never resolve and its blocks would stay pinned.
    """
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = _make_send_session_transceiver()
    request = _make_retirable_request(retired=True)

    session = KvCacheTransceiverV2._get_or_create_send_session(transceiver, request)

    assert session is None
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    transceiver._transfer_worker.create_tx_session.assert_not_called()
    assert transceiver._send_sessions == {}


def test_get_or_create_send_session_creates_for_fresh_request():
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = _make_send_session_transceiver()
    request = _make_retirable_request(retired=False)
    created = transceiver._transfer_worker.create_tx_session.return_value

    session = KvCacheTransceiverV2._get_or_create_send_session(transceiver, request)

    assert session is created
    assert transceiver._send_sessions == {42: created}
    assert request.state == LlmRequestState.CONTEXT_INIT


def test_get_or_create_send_session_prefers_live_session_over_retired_flag():
    """A live session is the source of truth; the flag only bars re-creation."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    live = MagicMock()
    transceiver = _make_send_session_transceiver({42: live})
    request = _make_retirable_request(retired=True)

    session = KvCacheTransceiverV2._get_or_create_send_session(transceiver, request)

    assert session is live
    assert request.state == LlmRequestState.CONTEXT_INIT


def test_respond_and_send_async_returns_early_when_session_refused():
    """A refused session must not build, send, or finalize anything."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = MagicMock()
    transceiver._enable_pipelined_transfer = True
    transceiver.kv_transfer_timeout_ms = None
    transceiver._get_or_create_send_session.return_value = None

    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=42),
        request_id=42,
        prompt_len=8,
        py_beam_width=1,
        py_kv_transfer_start_time=None,
        set_kv_cache_transfer_start=lambda _ts: None,
        state=LlmRequestState.DISAGG_TRANS_ERROR,
    )

    KvCacheTransceiverV2.respond_and_send_async(transceiver, request)

    transceiver._build_prefill_chunk.assert_not_called()
    transceiver._create_kv_slice.assert_not_called()
    transceiver._finalize_send.assert_not_called()
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR


# ---------------------------------------------------------------------------
# Context-side prefix reuse
# ---------------------------------------------------------------------------

_REUSE_TPB = 4
_REUSE_TOTAL_BLOCKS = 8


def _build_prefill_chunk_tokens_for(
    prepopulated_tokens,
    chunk_start_pos,
    chunk_end_pos,
    resident_blocks=None,
    prefix_sent=False,
):
    """Drive the real _build_prefill_chunk for one chunk, in token coordinates.

    ``resident_blocks`` defaults to the block holding ``chunk_end_pos``, matching a
    source block list that has only grown through the current chunk boundary.
    ``prefix_sent`` models the executor having already shipped the reused prefix
    early.
    """
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    if resident_blocks is None:
        resident_blocks = (chunk_end_pos + _REUSE_TPB - 1) // _REUSE_TPB
    base_slice = KVSlice(
        block_ids_per_layer_groups=[np.arange(resident_blocks, dtype=np.int64)],
    )

    transceiver = MagicMock()
    transceiver._kv_cache_manager.tokens_per_block = _REUSE_TPB
    transceiver._create_kv_slice.return_value = base_slice
    transceiver._send_reqs = {}

    req = MagicMock()
    req.py_disaggregated_params = DisaggregatedParams(disagg_request_id=42)
    req.py_beam_width = 1
    req.prompt_len = _REUSE_TOTAL_BLOCKS * _REUSE_TPB
    req.prepopulated_prompt_len = prepopulated_tokens
    req.py_kv_prefix_sent = prefix_sent
    req.py_last_context_chunk = (chunk_start_pos, chunk_end_pos)
    req.context_remaining_length = req.prompt_len - chunk_end_pos

    return KvCacheTransceiverV2._build_prefill_chunk(transceiver, req)


def _build_prefill_chunk_for(
    prepopulated_blocks,
    chunk_start_block,
    chunk_end_block,
    resident_blocks=None,
    prefix_sent=False,
):
    """Drive the real _build_prefill_chunk for one block-aligned chunk."""
    return _build_prefill_chunk_tokens_for(
        prepopulated_tokens=prepopulated_blocks * _REUSE_TPB,
        chunk_start_pos=chunk_start_block * _REUSE_TPB,
        chunk_end_pos=chunk_end_block * _REUSE_TPB,
        resident_blocks=chunk_end_block if resident_blocks is None else resident_blocks,
        prefix_sent=prefix_sent,
    )


def test_build_prefill_chunk_rounds_unaligned_non_final_end_up():
    """An unaligned non-final end sends its enclosing block, stale tail and all."""
    kv_slice = _build_prefill_chunk_tokens_for(
        prepopulated_tokens=0,
        chunk_start_pos=0,
        chunk_end_pos=6,
    )

    assert kv_slice.is_last_slice is False
    assert kv_slice.chunk == ChunkCoords(block_offset=0, block_count=2)
    assert np.array_equal(kv_slice.block_ids_per_layer_groups[0], np.arange(2, dtype=np.int64))


def test_unaligned_chunk_boundary_overlaps_by_exactly_one_block():
    """Rounding the end up and the next start down covers every block, sharing one."""
    chunk_bounds = [(0, 6), (6, 12), (12, _REUSE_TOTAL_BLOCKS * _REUSE_TPB)]
    slices = [
        _build_prefill_chunk_tokens_for(
            prepopulated_tokens=0,
            chunk_start_pos=start,
            chunk_end_pos=end,
        )
        for start, end in chunk_bounds
    ]

    block_spans = [
        (s.chunk.block_offset, s.chunk.block_offset + s.chunk.block_count) for s in slices
    ]
    assert block_spans == [(0, 2), (1, 3), (3, _REUSE_TOTAL_BLOCKS)]

    # Every block is covered, and each unaligned boundary repeats a single block.
    covered = [set(range(start, end)) for start, end in block_spans]
    assert set().union(*covered) == set(range(_REUSE_TOTAL_BLOCKS))
    assert len(covered[0] & covered[1]) == 1
    assert len(covered[1] & covered[2]) == 0
    assert slices[-1].is_last_slice is True


def test_unaligned_reuse_prefix_still_extends_first_chunk_to_block_zero():
    """A partial-block reuse hit leaves the first chunk unaligned at both ends."""
    kv_slice = _build_prefill_chunk_tokens_for(
        prepopulated_tokens=6,
        chunk_start_pos=6,
        chunk_end_pos=14,
    )

    assert kv_slice.chunk == ChunkCoords(block_offset=0, block_count=4)
    assert np.array_equal(kv_slice.block_ids_per_layer_groups[0], np.arange(4, dtype=np.int64))


def test_first_chunk_covers_ctx_prefix_reuse():
    """The reused prefix is resident but no chunk spans it, so slice 0 extends to block 0."""
    kv_slice = _build_prefill_chunk_for(
        prepopulated_blocks=3,
        chunk_start_block=3,
        chunk_end_block=6,
    )

    assert kv_slice.chunk == ChunkCoords(block_offset=0, block_count=6)
    assert np.array_equal(kv_slice.block_ids_per_layer_groups[0], np.arange(6, dtype=np.int64))
    assert kv_slice.is_last_slice is False


def test_first_chunk_skips_prefix_already_sent_early():
    """An early prefix send owns blocks [0, 3), so slice 0 starts at its own block."""
    kv_slice = _build_prefill_chunk_for(
        prepopulated_blocks=3,
        chunk_start_block=3,
        chunk_end_block=6,
        resident_blocks=6,
        prefix_sent=True,
    )

    assert kv_slice.chunk == ChunkCoords(block_offset=3, block_count=3)
    assert np.array_equal(kv_slice.block_ids_per_layer_groups[0], np.arange(3, 6, dtype=np.int64))


@pytest.mark.parametrize(
    "prepopulated_blocks,chunk_start_block,chunk_end_block,expected_start_block",
    [
        (3, 6, 8, 6),
        (0, 4, 8, 4),
        (0, 0, 4, 0),
    ],
    ids=["after_reuse_hit", "no_reuse_later_chunk", "no_reuse_first_chunk"],
)
def test_only_the_first_chunk_extends_to_block_zero(
    prepopulated_blocks, chunk_start_block, chunk_end_block, expected_start_block
):
    """Chunks past the first keep their own start; without reuse nothing changes."""
    kv_slice = _build_prefill_chunk_for(
        prepopulated_blocks=prepopulated_blocks,
        chunk_start_block=chunk_start_block,
        chunk_end_block=chunk_end_block,
        resident_blocks=_REUSE_TOTAL_BLOCKS,
    )

    assert kv_slice.chunk == ChunkCoords(
        block_offset=expected_start_block,
        block_count=chunk_end_block - expected_start_block,
    )
    assert np.array_equal(
        kv_slice.block_ids_per_layer_groups[0],
        np.arange(expected_start_block, chunk_end_block, dtype=np.int64),
    )


def test_single_chunk_with_reuse_degenerates_to_monolithic_slice():
    """One chunk plus a reuse hit yields the same slice shape a monolithic send would.

    The chunk still spans [0, total_blocks), which _build_kv_write_meta addresses
    exactly as an unpipelined write — see
    test_whole_prompt_chunk_addresses_like_a_monolithic_slice.
    """
    kv_slice = _build_prefill_chunk_for(
        prepopulated_blocks=3,
        chunk_start_block=3,
        chunk_end_block=_REUSE_TOTAL_BLOCKS,
        resident_blocks=_REUSE_TOTAL_BLOCKS,
    )

    assert kv_slice.is_last_slice is True
    assert kv_slice.chunk == ChunkCoords(block_offset=0, block_count=_REUSE_TOTAL_BLOCKS)
    assert np.array_equal(
        kv_slice.block_ids_per_layer_groups[0],
        np.arange(_REUSE_TOTAL_BLOCKS, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Transfer activity as a dimension owned by the transceiver
# ---------------------------------------------------------------------------


def _make_transfer_state_transceiver(session=None, rid: int = 42):
    """Transceiver stub whose session maps are the ownership record."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    sessions = {rid: session} if session is not None else {}
    transceiver = SimpleNamespace(
        _wait_reqs={},
        _send_sessions=dict(sessions),
        _send_reqs={rid: MagicMock()} if session is not None else {},
        _recv_sessions={},
        _recv_reqs={},
    )
    # Real teardown, so the predicate is checked against the actual bookkeeping.
    transceiver._retire_send_session = MethodType(
        KvCacheTransceiverV2._retire_send_session, transceiver
    )
    return transceiver


def _make_transfer_state_request(rid=42, request_id: int = 42):
    return SimpleNamespace(
        py_disaggregated_params=(
            DisaggregatedParams(disagg_request_id=rid) if rid is not None else None
        ),
        request_id=request_id,
    )


def test_has_inflight_transfer_tracks_send_session_lifetime():
    """Session membership answers the predicate, before and after teardown."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    session = MagicMock()
    session.has_transferring_tasks.return_value = False
    transceiver = _make_transfer_state_transceiver(session)
    request = _make_transfer_state_request()

    assert KvCacheTransceiverV2.has_inflight_transfer(transceiver, request)
    assert KvCacheTransceiverV2.has_any_inflight_transfer(transceiver)

    assert KvCacheTransceiverV2.cancel_request(transceiver, request)

    assert not KvCacheTransceiverV2.has_inflight_transfer(transceiver, request)
    assert not KvCacheTransceiverV2.has_any_inflight_transfer(transceiver)


def test_has_inflight_transfer_survives_mid_write_cancel():
    """A cancel that cannot complete keeps the ownership record alive."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    session = MagicMock()
    session.has_transferring_tasks.return_value = True
    transceiver = _make_transfer_state_transceiver(session)
    request = _make_transfer_state_request()

    assert not KvCacheTransceiverV2.cancel_request(transceiver, request)
    assert KvCacheTransceiverV2.has_inflight_transfer(transceiver, request)


def test_has_inflight_transfer_false_without_disagg_params():
    """A request that never registered a session owns no transfer resources."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = _make_transfer_state_transceiver()
    request = _make_transfer_state_request(rid=None, request_id=7)

    assert not KvCacheTransceiverV2.has_inflight_transfer(transceiver, request)


def test_is_request_in_transmission_uses_transceiver_predicate():
    """A mid-prefill request still counts as transmitting despite CONTEXT_INIT."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.has_inflight_transfer.return_value = True

    request = SimpleNamespace(state=LlmRequestState.CONTEXT_INIT)

    assert PyExecutor._is_request_in_transmission(executor, request)
    executor.kv_cache_transceiver.has_inflight_transfer.assert_called_once_with(request)


def test_is_request_in_transmission_false_when_nothing_in_flight():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.has_inflight_transfer.return_value = False

    request = SimpleNamespace(state=LlmRequestState.CONTEXT_INIT)

    assert not PyExecutor._is_request_in_transmission(executor, request)


def test_try_cancel_request_propagates_mid_write_failure():
    """Cancelling mid-prefill delegates and reports the retry-needed result."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor._is_request_in_transmission.return_value = True
    executor._is_disagg_inflight_cancel_active.return_value = False
    executor.kv_cache_transceiver.cancel_request.return_value = False
    # The delegation goes through _request_kv_transfer_cancellation, so run the
    # real helper to keep the assertion on the transceiver call itself.
    executor._request_kv_transfer_cancellation = (
        lambda req: PyExecutor._request_kv_transfer_cancellation(executor, req)
    )

    request = SimpleNamespace(state=LlmRequestState.CONTEXT_INIT)

    assert not PyExecutor._try_cancel_request(executor, request)
    executor.kv_cache_transceiver.cancel_request.assert_called_once_with(request)


def _make_send_kv_executor(canceled_req_ids, retired: bool = False):
    executor = MagicMock()
    executor.kv_connector_manager = None
    executor.canceled_req_ids = list(canceled_req_ids)
    executor.kv_cache_transceiver.pipeline_transfer_enabled = True
    executor.kv_cache_transceiver.kv_transfer_timeout_ms = None
    executor.kv_cache_transceiver.has_retired_send_session.return_value = retired
    return executor


def _make_send_kv_request(
    is_last_chunk: bool,
    request_id: int = 7,
    state=LlmRequestState.CONTEXT_INIT,
):
    return SimpleNamespace(
        is_context_only_request=True,
        is_finished_due_to_cancellation=False,
        is_context_finished=is_last_chunk,
        is_finished_due_to_length=False,
        is_child=False,
        parent_request_id=None,
        py_request_id=request_id,
        py_kv_transfer_start_time=None,
        state=state,
    )


def test_send_kv_async_skips_intermediate_chunk_for_cancelled_request():
    """A cancelled session must not be fed another chunk."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = _make_send_kv_executor([7])
    request = _make_send_kv_request(is_last_chunk=False)

    PyExecutor._send_kv_async(executor, [request])

    executor.kv_cache_transceiver.respond_and_send_async.assert_not_called()


def test_send_kv_async_sends_intermediate_chunk_when_not_cancelled():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = _make_send_kv_executor([])
    request = _make_send_kv_request(is_last_chunk=False)

    PyExecutor._send_kv_async(executor, [request])

    executor.kv_cache_transceiver.respond_and_send_async.assert_called_once_with(request)


def test_send_kv_async_skips_intermediate_chunk_for_failed_request():
    """An error path already failed and freed this request, leaving its chunk bounds unset.

    _update_request_states skips GENERATION_COMPLETE requests, so
    py_last_context_chunk is still (None, None) and building a chunk from it
    would fault. The request stays in scheduled_requests either way.
    """
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = _make_send_kv_executor([])
    request = _make_send_kv_request(is_last_chunk=False, state=LlmRequestState.GENERATION_COMPLETE)

    PyExecutor._send_kv_async(executor, [request])

    executor.kv_cache_transceiver.respond_and_send_async.assert_not_called()


def test_send_kv_async_skips_retired_request_mid_prefill():
    """A retired session cannot reach its peer, so the request is failed instead."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = _make_send_kv_executor([], retired=True)
    request = _make_send_kv_request(is_last_chunk=False)

    PyExecutor._send_kv_async(executor, [request])

    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    executor.kv_cache_transceiver.respond_and_send_async.assert_not_called()


def test_send_kv_async_skips_retired_request_before_start_transfer():
    """The gate precedes start_transfer, which pins blocks only end_transfer releases.

    Reaching the final-chunk branch would register the request with the transfer
    manager for a transfer that can never complete.
    """
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = _make_send_kv_executor([], retired=True)
    request = _make_send_kv_request(is_last_chunk=True)

    PyExecutor._send_kv_async(executor, [request])

    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    executor.async_transfer_manager.start_transfer.assert_not_called()
    executor.kv_cache_transceiver.respond_and_send_async.assert_not_called()


def test_send_kv_async_still_sends_final_chunk_for_cancelled_request():
    """The final chunk stays unconditional so nothing strands in the manager."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = _make_send_kv_executor([7])
    request = _make_send_kv_request(is_last_chunk=True)

    PyExecutor._send_kv_async(executor, [request])

    executor.async_transfer_manager.start_transfer.assert_called_once_with(request)
    executor.kv_cache_transceiver.respond_and_send_async.assert_called_once_with(request)


def _make_timeout_request(request_id: int = 7, elapsed_s: float = 10.0):
    return SimpleNamespace(
        is_context_only_request=True,
        is_disagg_generation_transmission_in_progress=False,
        py_request_id=request_id,
        py_kv_transfer_start_time=time.monotonic() - elapsed_s,
        py_kv_transfer_timed_out=False,
        state=LlmRequestState.CONTEXT_INIT,
    )


def test_check_kv_transfer_timeout_flags_context_request_in_transfer():
    """Context requests are monitored via the transfer manager.

    They enter it on the last chunk, at the same time as their clock starts.
    """
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.kv_transfer_timeout_ms = 100
    request = _make_timeout_request()
    executor.async_transfer_manager.requests_in_transfer.return_value = {7: request}
    executor.active_requests = [request]

    PyExecutor._check_kv_transfer_timeout(executor)

    assert request.py_kv_transfer_timed_out


def test_check_kv_transfer_timeout_ignores_context_request_not_in_transfer():
    """A request still being prefilled is not monitored.

    It holds its KV pages because it is computing, not because a chunk is in
    flight, so the timer only needs to cover the final transfer.
    """
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.kv_transfer_timeout_ms = 100
    executor.async_transfer_manager.requests_in_transfer.return_value = {}

    request = _make_timeout_request()
    executor.active_requests = [request]

    PyExecutor._check_kv_transfer_timeout(executor)

    assert not request.py_kv_transfer_timed_out


def test_has_any_inflight_kv_transfer_ors_in_transceiver():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.async_transfer_manager.has_any_inflight_requests.return_value = False
    executor.kv_cache_transceiver.has_any_inflight_transfer.return_value = True

    assert PyExecutor._has_any_inflight_kv_transfer(executor)

    executor.kv_cache_transceiver.has_any_inflight_transfer.return_value = False
    assert not PyExecutor._has_any_inflight_kv_transfer(executor)


def test_ctx_transfer_status_leaves_mid_prefill_request_alone():
    """The timeout path never cancels a request still being prefilled.

    py_kv_transfer_timed_out can only be set for requests the transfer manager
    already knows about, so the mid-prefill case has nothing to act on.
    """
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.kv_cache_transceiver.check_context_transfer_status.return_value = ([], [])
    executor.async_transfer_manager.requests_in_transfer.return_value = {}

    request = _make_timeout_request()
    request.py_kv_transfer_timed_out = True
    executor.active_requests = [request]

    PyExecutor._check_disagg_ctx_cache_transfer_status(executor)

    executor.kv_cache_transceiver.cancel_request.assert_not_called()
    assert request.state == LlmRequestState.CONTEXT_INIT
