# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for layer-wise KV cache transfer gating with ContextProgress.

Verifies that respond_and_send_layer_wise correctly gates per-layer KV
transfers using ContextProgress events, enabling overlapped compute and
KV transfer for Chunked Pipeline Parallelism.
"""

import threading
import time

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.context_progress import ContextProgress


@pytest.fixture(autouse=True)
def requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class MockKvCacheTransceiver:
    """Mock transceiver that records per-layer send timing relative to
    ContextProgress events, verifying the gating behavior."""

    def __init__(self):
        self.send_log = []
        self.send_times = []

    def respond_and_send_layer_wise(self, requests, context_progress):
        """Simulate layer-wise send: wait for each layer, then record the send."""
        num_layers = context_progress.num_layers
        for layer_idx in range(num_layers):
            context_progress.wait(layer_idx)
            self.send_log.append(layer_idx)
            self.send_times.append(time.monotonic())

    def respond_and_send_async(self, req):
        self.send_log.append("full_send")


class TestLayerWiseTransferGating:
    """Verify that ContextProgress correctly gates per-layer KV sends."""

    def test_send_blocked_until_layer_recorded(self):
        """Each layer's KV send should block until that layer's event is recorded."""
        num_layers = 4
        cp = ContextProgress(num_layers)
        transceiver = MockKvCacheTransceiver()

        send_done = threading.Event()

        def send_thread():
            transceiver.respond_and_send_layer_wise(["req1"], cp)
            send_done.set()

        t = threading.Thread(target=send_thread)
        t.start()

        time.sleep(0.05)
        assert len(transceiver.send_log) == 0, "Send should be blocked"

        compute_stream = torch.cuda.current_stream()
        for i in range(num_layers):
            x = torch.randn(64, 64, device="cuda")
            _ = x @ x
            cp.record_event(i, compute_stream)
            time.sleep(0.02)

        t.join(timeout=5.0)
        assert send_done.is_set()
        assert transceiver.send_log == list(range(num_layers))

    def test_sends_ordered_sequentially(self):
        """Layer-wise sends should proceed in order: layer 0, 1, 2, ..."""
        num_layers = 6
        cp = ContextProgress(num_layers)
        transceiver = MockKvCacheTransceiver()

        def send_thread():
            transceiver.respond_and_send_layer_wise(["req1"], cp)

        t = threading.Thread(target=send_thread)
        t.start()

        compute_stream = torch.cuda.current_stream()
        for i in range(num_layers):
            torch.randn(32, 32, device="cuda")
            cp.record_event(i, compute_stream)

        t.join(timeout=5.0)
        assert transceiver.send_log == list(range(num_layers))

    def test_compute_transfer_overlap(self):
        """Verify that KV sends for earlier layers can overlap with compute
        on later layers, simulating the CPP pattern."""
        num_layers = 4
        cp = ContextProgress(num_layers)
        transceiver = MockKvCacheTransceiver()
        compute_log = []

        def send_thread():
            transceiver.respond_and_send_layer_wise(["req1"], cp)

        t = threading.Thread(target=send_thread)
        t.start()

        compute_stream = torch.cuda.current_stream()
        for i in range(num_layers):
            x = torch.randn(128, 128, device="cuda")
            _ = x @ x
            cp.record_event(i, compute_stream)
            compute_log.append(("compute_done", i))

        t.join(timeout=5.0)
        assert len(transceiver.send_log) == num_layers
        assert len(compute_log) == num_layers

        for i in range(num_layers):
            assert transceiver.send_log[i] == i

    def test_default_fallback_waits_all_layers(self):
        """The default respond_and_send_layer_wise on KvCacheTransceiver
        base class should wait for all layers before sending."""
        from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
            KvCacheTransceiver,
        )

        class MinimalTransceiver(KvCacheTransceiver):

            def __init__(self):
                self.sent = []

            def respond_and_send_async(self, req):
                self.sent.append(req)

            def request_and_receive_sync(self, req):
                pass

            def request_and_receive_async(self, req):
                pass

            def check_context_transfer_status(self, n):
                return [], []

            def check_gen_transfer_status(self, n):
                return [], []

            def check_gen_transfer_complete(self):
                return True

            def cancel_request(self, req):
                pass

            def prepare_context_requests(self, requests):
                pass

            def get_disaggregated_params(self):
                return {}

        num_layers = 3
        cp = ContextProgress(num_layers)
        transceiver = MinimalTransceiver()

        send_done = threading.Event()

        def send_thread():
            transceiver.respond_and_send_layer_wise(["r1", "r2"], cp)
            send_done.set()

        t = threading.Thread(target=send_thread)
        t.start()

        time.sleep(0.05)
        assert len(transceiver.sent) == 0, "Should block until all layers done"

        compute_stream = torch.cuda.current_stream()
        for i in range(num_layers):
            cp.record_event(i, compute_stream)

        t.join(timeout=5.0)
        assert send_done.is_set()
        assert transceiver.sent == ["r1", "r2"]

    def test_reset_allows_reuse_across_chunks(self):
        """ContextProgress.reset() should allow reuse for multiple chunks,
        simulating sequential chunk processing in CPP."""
        num_layers = 3
        cp = ContextProgress(num_layers)
        transceiver = MockKvCacheTransceiver()

        compute_stream = torch.cuda.current_stream()

        for chunk_id in range(3):
            if chunk_id > 0:
                cp.reset()

            def send_thread():
                transceiver.respond_and_send_layer_wise(
                    [f"chunk{chunk_id}"], cp)

            t = threading.Thread(target=send_thread)
            t.start()

            for i in range(num_layers):
                torch.randn(32, 32, device="cuda")
                cp.record_event(i, compute_stream)

            t.join(timeout=5.0)

        assert len(transceiver.send_log) == num_layers * 3
