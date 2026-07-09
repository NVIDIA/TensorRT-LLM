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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.disaggregation.base.transfer import (
    KVSlice,
    SessionStatus,
    TokenRange,
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
            token_range=TokenRange(start=32, end=64),
            total_blocks=8,
        ),
        _make_params(),
        slice_id=1,
        prompt_len=64,
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


def test_rx_session_process_aux_completes_at_expected_transfers():
    """Aux completes only once _aux_count reaches the RxSession's single
    task's expected_transfers (the receiver always has exactly one task)."""
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
    """enable_pipelined_transfer defaults to False."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    transceiver = MagicMock()
    transceiver._enable_pipelined_transfer = False
    transceiver._chunk_size_blocks = 64

    result = KvCacheTransceiverV2.enable_pipelined_transfer.fget(transceiver)
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
            match=
            "enable_chunked_prefill is required when enable_pipelined_transfer is set."
    ):
        create_kv_cache_transceiver(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            cache_transceiver_config,
            enable_chunked_prefill=False,
        )

def test_pipelined_transfer_requires_gen_first_flow():
    """ValueError when a real request is not using gen-first flow."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = MagicMock()
    executor.is_warmup = False
    executor.kv_cache_transceiver.enable_pipelined_transfer = True
    executor._validate_token_id_range = MagicMock()
    executor.sampler.validate_request = MagicMock()

    request = MagicMock()
    request.sampling_config = None
    request.py_beam_width = 1
    request.py_disaggregated_params = SimpleNamespace(
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST)

    with pytest.raises(
            ValueError,
            match=
            "schedule_style must be generation_first when enable_pipelined_transfer is set."
    ):
        PyExecutor._validate_request(executor, request)


def test_pipelined_last_chunk_defers_aux_finalization(monkeypatch):
    """Last chunk sends KV, while aux waits until sampling has updated tokens."""
    from tensorrt_llm._torch.disaggregation import transceiver as transceiver_module
    from tensorrt_llm._torch.disaggregation.transceiver import \
        KvCacheTransceiverV2

    event = MagicMock()
    monkeypatch.setattr(transceiver_module.torch.cuda, "Event",
                        lambda: event)
    monkeypatch.setattr(transceiver_module.torch.cuda, "current_stream",
                        MagicMock(return_value=MagicMock()))

    session = MagicMock()
    session.kv_tasks = []

    transceiver = MagicMock()
    transceiver._get_or_create_send_session.return_value = session
    transceiver._send_sessions = {42: session}
    transceiver._send_reqs = {}
    transceiver._reuse_adapter.tokens_per_block = 4
    transceiver._page_table = MagicMock()
    transceiver._create_kv_slice.return_value = KVSlice(
        is_last_slice=False,
        block_ids_per_layer_groups=[
            np.array([0, 1], dtype=np.int64),
        ],
    )

    request = SimpleNamespace(
        py_disaggregated_params=DisaggregatedParams(disagg_request_id=42),
        request_id=42,
        prompt_len=8,
        py_beam_width=1,
    )

    KvCacheTransceiverV2.send_prefill_chunk(
        transceiver,
        request,
        chunk_start_block=0,
        chunk_end_block=2,
        is_last_chunk=True,
    )

    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    sent_slice = session.send.call_args.args[0]
    assert sent_slice.is_last_slice is True
    transceiver._finalize_send.assert_not_called()

    KvCacheTransceiverV2.finalize_pipelined_send(transceiver, request)

    transceiver._finalize_send.assert_called_once_with(request, session)


def test_cuda_event_stored_on_task():
    """KVSlice stores cuda_event correctly."""
    event = MagicMock()
    s = KVSlice(is_last_slice=False, cuda_event=event, block_ids_per_layer_groups=[[0, 1]])
    task = KVSendTask(s, MagicMock(disagg_request_id=1), slice_id=0, prompt_len=8)
    assert task._slice.cuda_event is event


def test_cuda_event_none_by_default():
    """KVSlice.cuda_event defaults to None."""
    s = KVSlice(is_last_slice=True, block_ids_per_layer_groups=[[0]])
    task = KVSendTask(s, MagicMock(disagg_request_id=1), slice_id=0, prompt_len=8)
    assert task._slice.cuda_event is None
