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

from unittest.mock import MagicMock, patch


class _FakeRequest:
    """Minimal stand-in for LlmRequest with timing getters/setters."""

    def __init__(self, rid, start, end, kv_size):
        self.request_id = rid
        self.py_disaggregated_params = None  # forces get_unique_rid -> request_id
        self._start = start
        self._end = end
        self._kv_cache_size = kv_size

    def get_kv_cache_transfer_start(self):
        return self._start

    def get_kv_cache_transfer_end(self):
        return self._end

    @property
    def kv_cache_size(self):
        return self._kv_cache_size

    def set_kv_cache_transfer_start(self, v):
        self._start = v

    def set_kv_cache_transfer_end(self, v):
        self._end = v

    def set_kv_cache_size(self, v):
        self._kv_cache_size = v


def _make_transceiver(**overrides):
    """Create a minimal mock of KvCacheTransceiverV2 with _sync_transfer_timing."""
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    tc = object.__new__(KvCacheTransceiverV2)
    tc._gen_need_sync = overrides.get("gen_need_sync", True)
    tc._gen_allgather = overrides.get("gen_allgather", lambda x: [x])
    return tc


class TestSyncTransferTiming:
    @patch.dict("os.environ", {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "/tmp/test"}, clear=False)
    def test_merges_correctly_two_ranks(self):
        """Two simulated ranks — verify min(start), max(end), sum(size)."""
        req = _FakeRequest(rid=1, start=10, end=20, kv_size=100)

        def fake_allgather(local_data):
            # Simulate rank 0 = local_data, rank 1 = different timing
            rank1_data = {1: (5, 25, 200)}
            return [local_data, rank1_data]

        tc = _make_transceiver(gen_need_sync=True, gen_allgather=fake_allgather)
        tc._sync_transfer_timing([req])

        assert req._start == 5  # min(10, 5)
        assert req._end == 25  # max(20, 25)
        assert req._kv_cache_size == 300  # 100 + 200

    @patch.dict("os.environ", {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "/tmp/test"}, clear=False)
    def test_multiple_requests_batched(self):
        """Multiple requests in one allgather call."""
        req_a = _FakeRequest(rid=1, start=10, end=20, kv_size=100)
        req_b = _FakeRequest(rid=2, start=30, end=40, kv_size=200)

        def fake_allgather(local_data):
            rank1_data = {
                1: (8, 22, 150),
                2: (28, 45, 250),
            }
            return [local_data, rank1_data]

        tc = _make_transceiver(gen_need_sync=True, gen_allgather=fake_allgather)
        tc._sync_transfer_timing([req_a, req_b])

        assert req_a._start == 8
        assert req_a._end == 22
        assert req_a._kv_cache_size == 250

        assert req_b._start == 28
        assert req_b._end == 45
        assert req_b._kv_cache_size == 450

    @patch.dict("os.environ", {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "/tmp/test"}, clear=False)
    def test_all_ranks_updated(self):
        """Every request object should be updated, not just rank-0."""
        req = _FakeRequest(rid=1, start=10, end=20, kv_size=100)

        def fake_allgather(local_data):
            return [local_data, {1: (5, 25, 200)}]

        tc = _make_transceiver(gen_need_sync=True, gen_allgather=fake_allgather)
        tc._sync_transfer_timing([req])

        # All fields should reflect the merged values
        assert req._start == 5
        assert req._end == 25
        assert req._kv_cache_size == 300

    def test_skips_when_no_env(self):
        """Without TRTLLM_KVCACHE_TIME_OUTPUT_PATH, allgather is not called."""
        import os

        os.environ.pop("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", None)

        allgather_mock = MagicMock()
        tc = _make_transceiver(gen_need_sync=True, gen_allgather=allgather_mock)

        req = _FakeRequest(rid=1, start=10, end=20, kv_size=100)
        tc._sync_transfer_timing([req])

        allgather_mock.assert_not_called()
        assert req._start == 10  # unchanged

    @patch.dict("os.environ", {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "/tmp/test"}, clear=False)
    def test_skips_when_single_rank(self):
        """When _gen_need_sync is False, allgather is not called."""
        allgather_mock = MagicMock()
        tc = _make_transceiver(gen_need_sync=False, gen_allgather=allgather_mock)

        req = _FakeRequest(rid=1, start=10, end=20, kv_size=100)
        tc._sync_transfer_timing([req])

        allgather_mock.assert_not_called()
        assert req._start == 10  # unchanged

    @patch.dict("os.environ", {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "/tmp/test"}, clear=False)
    def test_empty_list(self):
        """Empty request list should return immediately."""
        allgather_mock = MagicMock()
        tc = _make_transceiver(gen_need_sync=True, gen_allgather=allgather_mock)

        tc._sync_transfer_timing([])

        allgather_mock.assert_not_called()
