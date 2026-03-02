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

import threading

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.context_progress import ContextProgress


@pytest.fixture(autouse=True)
def requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class TestContextProgressInit:

    def test_valid_init(self):
        cp = ContextProgress(num_layers=4)
        assert cp.num_layers == 4
        for i in range(4):
            assert not cp.is_recorded(i)

    def test_zero_layers_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            ContextProgress(num_layers=0)

    def test_negative_layers_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            ContextProgress(num_layers=-1)


class TestContextProgressRecordEvent:

    def test_record_sequential(self):
        cp = ContextProgress(num_layers=3)
        stream = torch.cuda.current_stream()
        for i in range(3):
            assert not cp.is_recorded(i)
            cp.record_event(i, stream)
            assert cp.is_recorded(i)

    def test_record_out_of_order_raises(self):
        cp = ContextProgress(num_layers=3)
        with pytest.raises(RuntimeError, match="has not been recorded"):
            cp.record_event(1)

    def test_record_duplicate_raises(self):
        cp = ContextProgress(num_layers=3)
        cp.record_event(0)
        with pytest.raises(RuntimeError, match="already been recorded"):
            cp.record_event(0)

    def test_record_out_of_range_raises(self):
        cp = ContextProgress(num_layers=2)
        with pytest.raises(IndexError):
            cp.record_event(2)
        with pytest.raises(IndexError):
            cp.record_event(-1)

    def test_record_without_explicit_stream(self):
        cp = ContextProgress(num_layers=2)
        cp.record_event(0)
        assert cp.is_recorded(0)
        cp.record_event(1)
        assert cp.is_recorded(1)


class TestContextProgressWait:

    def test_wait_after_record(self):
        cp = ContextProgress(num_layers=2)
        a = torch.randn(100, device="cuda")
        b = torch.randn(100, device="cuda")
        _ = a + b
        cp.record_event(0)
        _ = a * b
        cp.record_event(1)

        cp.wait(0)
        cp.wait(1)

    def test_wait_with_stream(self):
        cp = ContextProgress(num_layers=1)
        compute_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()

        with torch.cuda.stream(compute_stream):
            a = torch.randn(1000, device="cuda")
            _ = a @ a.unsqueeze(1)
            cp.record_event(0, compute_stream)

        cp.wait(0, transfer_stream)

    def test_wait_out_of_range_raises(self):
        cp = ContextProgress(num_layers=2)
        with pytest.raises(IndexError):
            cp.wait(2)

    def test_wait_blocks_until_recorded(self):
        cp = ContextProgress(num_layers=1)
        result = {"waited": False}

        def waiter():
            cp.wait(0)
            result["waited"] = True

        t = threading.Thread(target=waiter)
        t.start()

        torch.cuda.synchronize()
        assert not result["waited"]

        cp.record_event(0)
        t.join(timeout=5.0)
        assert result["waited"]


class TestContextProgressReset:

    def test_reset_allows_reuse(self):
        cp = ContextProgress(num_layers=2)
        cp.record_event(0)
        cp.record_event(1)
        assert cp.is_recorded(0)
        assert cp.is_recorded(1)

        cp.reset()
        assert not cp.is_recorded(0)
        assert not cp.is_recorded(1)

        cp.record_event(0)
        cp.record_event(1)
        assert cp.is_recorded(0)
        assert cp.is_recorded(1)


class TestContextProgressWithCompute:
    """Tests that verify ContextProgress correctly gates access to
    layer-by-layer computation results, simulating the CPP use case."""

    def test_layerwise_compute_and_transfer(self):
        num_layers = 4
        cp = ContextProgress(num_layers=num_layers)

        compute_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()

        x = torch.randn(256, 256, device="cuda")
        weights = [
            torch.randn(256, 256, device="cuda") for _ in range(num_layers)
        ]
        results = [None] * num_layers

        with torch.cuda.stream(compute_stream):
            hidden = x
            for i in range(num_layers):
                hidden = hidden @ weights[i]
                cp.record_event(i, compute_stream)

        for i in range(num_layers):
            cp.wait(i, transfer_stream)
            with torch.cuda.stream(transfer_stream):
                results[i] = torch.ones(1, device="cuda")

        transfer_stream.synchronize()
        for i in range(num_layers):
            assert results[i] is not None

    def test_concurrent_compute_transfer_overlap(self):
        """Simulate the CPP scenario: compute records events per layer,
        a separate thread waits and 'transfers' KV cache per layer."""
        num_layers = 8
        cp = ContextProgress(num_layers=num_layers)
        transfer_order = []

        def transfer_thread():
            for i in range(num_layers):
                cp.wait(i)
                transfer_order.append(i)

        t = threading.Thread(target=transfer_thread)
        t.start()

        compute_stream = torch.cuda.current_stream()
        x = torch.randn(128, 128, device="cuda")
        for i in range(num_layers):
            x = x @ torch.randn(128, 128, device="cuda")
            cp.record_event(i, compute_stream)

        t.join(timeout=10.0)
        assert transfer_order == list(range(num_layers))
