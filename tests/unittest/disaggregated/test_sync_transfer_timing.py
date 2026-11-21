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
"""Unit tests for KvCacheTransceiverV2._sync_transfer_timing."""

from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2


def _ms(value: int) -> timedelta:
    return timedelta(milliseconds=value)


class _FakeRequest:
    """Minimal stand-in for LlmRequest with timing getters/setters."""

    def __init__(self, rid: int, start: int, end: int, kv_size: int) -> None:
        self.request_id = rid
        self.py_disaggregated_params = None  # forces get_unique_rid -> request_id
        self._start = _ms(start)
        self._end = _ms(end)
        self._kv_cache_size = kv_size
        self.kv_cache_transfer_local_size = kv_size
        self.local_duration = None

    def set_kv_cache_transfer_local_duration(self, duration: float) -> None:
        self.local_duration = duration

    def get_kv_cache_transfer_start(self) -> timedelta:
        return self._start

    def get_kv_cache_transfer_end(self) -> timedelta:
        return self._end

    @property
    def kv_cache_size(self) -> int:
        return self._kv_cache_size

    def set_kv_cache_transfer_start(self, v: int) -> None:
        self._start = v

    def set_kv_cache_transfer_end(self, v: int) -> None:
        self._end = v

    def set_kv_cache_size(self, v: int) -> None:
        self._kv_cache_size = v


def _make_transceiver(**overrides: Any) -> "KvCacheTransceiverV2":
    """Create a minimal mock of KvCacheTransceiverV2 with _sync_transfer_timing."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    tc = object.__new__(KvCacheTransceiverV2)
    tc._gen_need_sync = overrides.get("gen_need_sync", True)
    tc._init_gen_idle_poll(False)
    tc._gen_allgather = overrides.get("gen_allgather", lambda x: [x])
    return tc


class TestSyncTransferTiming:
    @pytest.mark.parametrize("csv_enabled", [False, True])
    def test_two_rank_summary_preserves_reported_shard_bytes(
        self, monkeypatch, csv_enabled
    ) -> None:
        if csv_enabled:
            monkeypatch.setenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "/tmp/test")
        else:
            monkeypatch.delenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", raising=False)
        req = _FakeRequest(rid=1, start=10, end=20, kv_size=100)

        def fake_allgather(local_data):
            # Simulate rank 0 = local_data, rank 1 = different timing
            rank1_data = {1: (_ms(5), _ms(25), 200)}
            return [local_data, rank1_data]

        tc = _make_transceiver(gen_need_sync=True, gen_allgather=fake_allgather)
        summary = tc._sync_transfer_timing([req])

        from tensorrt_llm.metrics.enums import MetricNames, RequestEventTiming
        from tensorrt_llm.metrics.perf_utils import process_req_perf_metrics

        duration = (req._end - req._start).total_seconds()
        assert duration == (0.02 if csv_enabled else 0.01)
        perf = {
            RequestEventTiming.KV_CACHE_TRANSFER_START: req._start.total_seconds(),
            RequestEventTiming.KV_CACHE_TRANSFER_END: req._end.total_seconds(),
        }
        if req.local_duration is not None:
            perf[RequestEventTiming.KV_CACHE_TRANSFER_LOCAL_DURATION] = req.local_duration
        assert process_req_perf_metrics(perf, 0)[MetricNames.KV_CACHE_TRANSFER_TIME] == 0.01
        assert req.kv_cache_size == 100
        assert summary == {1: (_ms(5), _ms(25), 300) if csv_enabled else (_ms(10), _ms(20), 100)}

    @patch.dict("os.environ", {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "/tmp/test"}, clear=False)
    def test_multiple_requests_batched(self) -> None:
        """Multiple requests in one allgather call."""
        req_a = _FakeRequest(rid=1, start=10, end=20, kv_size=100)
        req_b = _FakeRequest(rid=2, start=30, end=40, kv_size=200)

        def fake_allgather(local_data):
            rank1_data = {
                1: (_ms(8), _ms(22), 150),
                2: (_ms(28), _ms(45), 250),
            }
            return [local_data, rank1_data]

        tc = _make_transceiver(gen_need_sync=True, gen_allgather=fake_allgather)
        summary = tc._sync_transfer_timing([req_a, req_b])

        assert req_a._start == _ms(8)
        assert req_a._end == _ms(22)
        assert req_a.kv_cache_size == 100

        assert req_b._start == _ms(28)
        assert req_b._end == _ms(45)
        assert req_b.kv_cache_size == 200
        assert summary == {1: (_ms(8), _ms(22), 250), 2: (_ms(28), _ms(45), 450)}

    @patch.dict("os.environ", {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "/tmp/test"}, clear=False)
    def test_summary_is_available_on_each_rank(self) -> None:
        req = _FakeRequest(rid=1, start=10, end=20, kv_size=100)

        def fake_allgather(local_data):
            return [local_data, {1: (_ms(5), _ms(25), 200)}]

        tc = _make_transceiver(gen_need_sync=True, gen_allgather=fake_allgather)
        summary = tc._sync_transfer_timing([req])

        assert req._start == _ms(5)
        assert req._end == _ms(25)
        assert req.kv_cache_size == 100
        assert summary == {1: (_ms(5), _ms(25), 300)}

    @patch.dict("os.environ", {}, clear=False)
    def test_skips_when_no_env(self) -> None:
        """Without TRTLLM_KVCACHE_TIME_OUTPUT_PATH, allgather is not called."""
        import os

        os.environ.pop("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", None)

        allgather_mock = MagicMock()
        tc = _make_transceiver(gen_need_sync=True, gen_allgather=allgather_mock)

        req = _FakeRequest(rid=1, start=10, end=20, kv_size=100)
        tc._sync_transfer_timing([req])

        allgather_mock.assert_not_called()
        assert req._start == _ms(10)

    @patch.dict("os.environ", {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "/tmp/test"}, clear=False)
    def test_skips_when_single_rank(self) -> None:
        """When _gen_need_sync is False, allgather is not called."""
        allgather_mock = MagicMock()
        tc = _make_transceiver(gen_need_sync=False, gen_allgather=allgather_mock)

        req = _FakeRequest(rid=1, start=10, end=20, kv_size=100)
        tc._sync_transfer_timing([req])

        allgather_mock.assert_not_called()
        assert req._start == _ms(10)

    @patch.dict("os.environ", {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "/tmp/test"}, clear=False)
    def test_empty_list(self) -> None:
        """Empty request list should return immediately."""
        allgather_mock = MagicMock()
        tc = _make_transceiver(gen_need_sync=True, gen_allgather=allgather_mock)

        tc._sync_transfer_timing([])

        allgather_mock.assert_not_called()
