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

from datetime import timedelta

import numpy as np

import tensorrt_llm._torch.disaggregation.transceiver as transceiver_module
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, SessionStatus, WaitResult
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState


class _Request:
    def __init__(self, request_id: int = 0) -> None:
        self.request_id = request_id
        self.py_disaggregated_params = None
        self.state = None
        self.context_phase_params = None
        self.kv_cache_transfer_start = None
        self.kv_cache_transfer_end = None
        self.kv_cache_size = None

    def set_kv_cache_transfer_start(self, time: timedelta) -> None:
        self.kv_cache_transfer_start = time

    def set_kv_cache_transfer_end(self, time: timedelta) -> None:
        self.kv_cache_transfer_end = time

    def set_kv_cache_size(self, size: int) -> None:
        self.kv_cache_size = size


class _TxSession:
    def __init__(self, rid: int, transferred_kv_bytes: int = 0) -> None:
        self.disagg_request_id = rid
        self.transferred_kv_bytes = transferred_kv_bytes
        self.sent_slices = []
        self.closed = False

    def send(self, kv_slice: KVSlice) -> None:
        self.sent_slices.append(kv_slice)

    def wait_complete(self) -> WaitResult:
        return WaitResult.COMPLETED

    @property
    def status(self) -> SessionStatus:
        return SessionStatus.FULLY_TRANSFERRED

    def is_completed(self) -> bool:
        return True

    def has_failed(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


class _TransferWorker:
    def __init__(self, session: _TxSession) -> None:
        self.session = session

    def create_tx_session(self, _request: _Request) -> _TxSession:
        return self.session

    def sweep_stale_req_infos(self) -> None:
        pass


class _Dist:
    tp_size = 1

    def tp_allgather(self, value):
        return [value]

    def pp_allgather(self, value):
        return [value]


def _make_transceiver() -> KvCacheTransceiverV2:
    transceiver = KvCacheTransceiverV2.__new__(KvCacheTransceiverV2)
    transceiver._gen_need_sync = False
    transceiver._gen_allgather = lambda metrics: [metrics]
    transceiver._dist = _Dist()
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._recv_reqs = {}
    transceiver._clock_offset_seconds = lambda: 0.0
    transceiver._ctx_consensus = lambda ids: ids
    return transceiver


def test_record_transfer_start_populates_request_perf_metrics(monkeypatch) -> None:
    transceiver = _make_transceiver()
    transceiver._clock_offset_seconds = lambda: 0.25
    request = _Request()
    monkeypatch.setattr(
        transceiver_module,
        "get_steady_clock_now_in_seconds",
        lambda: 100.0,
    )

    transceiver._record_transfer_start(request)

    assert request.kv_cache_transfer_start == timedelta(seconds=100.0)
    assert request.kv_cache_size == 0


def test_record_transfer_end_uses_actual_session_bytes(monkeypatch) -> None:
    transceiver = _make_transceiver()
    request = _Request()
    session = _TxSession(rid=0, transferred_kv_bytes=123)
    monkeypatch.setattr(
        transceiver_module,
        "get_steady_clock_now_in_seconds",
        lambda: 12.5,
    )

    transceiver._record_transfer_end(request, session)

    assert request.kv_cache_transfer_end == timedelta(seconds=12.5)
    assert request.kv_cache_size == 123


def test_get_transfer_metrics_reads_request_perf_metrics() -> None:
    request = _Request()
    request.kv_cache_transfer_start = timedelta(seconds=10.0)
    request.kv_cache_transfer_end = timedelta(seconds=14.0)
    request.kv_cache_size = 64

    assert KvCacheTransceiverV2._get_transfer_metrics(request) == (10.0, 14.0, 64)


def test_publish_gen_transfer_metrics_aggregates_completed_consensus_rids(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "/tmp/kvcache")
    transceiver = _make_transceiver()
    transceiver._gen_need_sync = True
    transceiver._clock_offset_seconds = lambda: 2.0
    request = _Request()
    request.kv_cache_transfer_start = timedelta(seconds=10.0)
    request.kv_cache_transfer_end = timedelta(seconds=14.0)
    request.kv_cache_size = 64
    transceiver._recv_reqs = {5: request}
    transceiver._gen_allgather = lambda metrics: [
        metrics,
        {5: (9.0, 16.0, 128)},
    ]

    transceiver._publish_gen_transfer_metrics([5], {5})

    assert request.kv_cache_transfer_start == timedelta(seconds=7.0)
    assert request.kv_cache_transfer_end == timedelta(seconds=14.0)
    assert request.kv_cache_size == 192


def test_publish_gen_transfer_metrics_respects_v1_env_gate(monkeypatch) -> None:
    monkeypatch.delenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", raising=False)
    transceiver = _make_transceiver()
    transceiver._gen_need_sync = True
    request = _Request()
    transceiver._recv_reqs = {5: request}

    def fail_allgather(_metrics):
        raise AssertionError("aggregation should be gated by TRTLLM_KVCACHE_TIME_OUTPUT_PATH")

    transceiver._gen_allgather = fail_allgather

    transceiver._publish_gen_transfer_metrics([5], {5})

    assert request.kv_cache_transfer_start is None
    assert request.kv_cache_transfer_end is None
    assert request.kv_cache_size is None


def test_publish_gen_transfer_metrics_skips_non_consensus_rids(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "/tmp/kvcache")
    transceiver = _make_transceiver()
    transceiver._gen_need_sync = True
    request = _Request()
    transceiver._recv_reqs = {5: request}

    def fail_allgather(_metrics):
        raise AssertionError("non-consensus request should not allgather")

    transceiver._gen_allgather = fail_allgather

    transceiver._publish_gen_transfer_metrics([5], set())

    assert request.kv_cache_transfer_start is None
    assert request.kv_cache_transfer_end is None
    assert request.kv_cache_size is None


def test_context_transfer_metrics_cover_send_lifecycle(monkeypatch) -> None:
    transceiver = _make_transceiver()
    kv_slice = KVSlice(
        block_ids_per_layer_groups=[
            np.array([0, 1], dtype=np.int64),
            np.array([], dtype=np.int64),
        ]
    )
    session = _TxSession(rid=7, transferred_kv_bytes=321)
    transceiver._transfer_worker = _TransferWorker(session)
    transceiver._send_sessions = {}
    transceiver._send_reqs = {}
    transceiver._dp_rank = 1
    transceiver._context_info_endpoint = "tcp://ctx"
    transceiver._create_kv_slice = lambda _request: kv_slice
    now = iter([10.0, 12.5])
    monkeypatch.setattr(
        transceiver_module,
        "get_steady_clock_now_in_seconds",
        lambda: next(now),
    )
    request = _Request(request_id=7)

    transceiver.respond_and_send_async(request)
    completed, failed = transceiver.check_context_transfer_status(1, mark_complete=True)

    assert session.sent_slices[0] is kv_slice
    assert completed == [7]
    assert failed == []
    assert request.kv_cache_transfer_start == timedelta(seconds=10.0)
    assert request.kv_cache_transfer_end == timedelta(seconds=12.5)
    assert request.kv_cache_size == 321
    assert request.state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
    assert request.context_phase_params.req_id == 7
    assert session.closed
