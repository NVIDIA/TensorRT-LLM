# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the multi_stream enable/disable switch.

Focus: ``disable_multi_stream`` nests correctly, restores on exception, and
turns every passthrough / aux-stream impl into a pure pass-through that
never touches the CUDA stream manager.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.utils import multi_stream_utils as msu


class _CountingManager:
    """Drop-in replacement for cuda_stream_manager that counts accesses."""

    MAIN_STREAM_NAME = "main"
    AUX_STREAM_NAME = "aux"

    def __init__(self):
        self.get_stream_calls = 0
        self.get_event_calls = 0
        self._caller_streams = {}

    def get_stream(self, device, name):
        self.get_stream_calls += 1
        raise AssertionError("cuda_stream_manager.get_stream must not be called when disabled")

    def get_event(self, device, name):
        self.get_event_calls += 1
        raise AssertionError("cuda_stream_manager.get_event must not be called when disabled")


class TestDisableMultiStream:
    def test_default_state_is_enabled(self):
        assert msu.is_multi_stream_enabled() is True

    def test_context_manager_toggles(self):
        assert msu.is_multi_stream_enabled() is True
        with msu.disable_multi_stream():
            assert msu.is_multi_stream_enabled() is False
        assert msu.is_multi_stream_enabled() is True

    def test_context_manager_nests(self):
        with msu.disable_multi_stream():
            assert msu.is_multi_stream_enabled() is False
            with msu.disable_multi_stream():
                assert msu.is_multi_stream_enabled() is False
            assert msu.is_multi_stream_enabled() is False
        assert msu.is_multi_stream_enabled() is True

    def test_context_manager_restores_on_exception(self):
        assert msu.is_multi_stream_enabled() is True
        try:
            with msu.disable_multi_stream():
                assert msu.is_multi_stream_enabled() is False
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert msu.is_multi_stream_enabled() is True

    def test_passthroughs_are_identity_when_disabled(self, monkeypatch):
        """Every passthrough must return x and must not touch cuda_stream_manager."""
        monkeypatch.setattr(msu, "cuda_stream_manager", _CountingManager())

        x = torch.zeros(2)
        with msu.disable_multi_stream():
            assert msu.record_event_passthrough(x) is x
            assert msu.begin_aux_stream_passthrough(x) is x
            assert msu.end_aux_stream_passthrough(x) is x
            assert msu.wait_aux_stream_passthrough(x) is x

    def test_aux_stream_impl_falls_back_to_base_when_disabled(self, monkeypatch):
        """_make_aux_stream_impl's generated impl must run base_overload directly."""
        monkeypatch.setattr(msu, "cuda_stream_manager", _CountingManager())

        called = []

        def base(*args, **kwargs):
            called.append((args, kwargs))
            return "ok"

        impl = msu._make_aux_stream_impl(base)

        with msu.disable_multi_stream():
            out = impl(1, 2, key="v")

        assert out == "ok"
        assert called == [((1, 2), {"key": "v"})]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestRecordStreamDuringCudaGraphCapture:
    """Regression guard for the multi-stream record_stream fix.

    In the monolithic decode CUDA-graph path the aux-stream passthroughs are
    captured (multi-stream stays enabled). ``begin_aux_stream_passthrough`` MUST
    call ``record_stream`` on its tensor outputs *even during CUDA graph capture*,
    otherwise the caching allocator frees ``x``'s block after the main stream's
    last use and recycles it for a later allocation inside the same captured
    graph, before the aux-stream shared-expert work has read it. That silently
    corrupts the shared-expert input on replay -> garbage logits -> no-EOS
    runaway generation (reproduced as 24/24 batch-1 requests running to
    max_tokens on SuperV3-MTP with multi_stream_moe enabled).

    A previous version skipped record_stream while capturing
    (``if not torch.cuda.is_current_stream_capturing(): ...``). This test fails
    if that guard is ever reintroduced.
    """

    def test_records_stream_even_while_capturing(self, monkeypatch):
        device = torch.cuda.current_device()
        msu.cuda_stream_manager.add_device(device)

        recorded = []
        monkeypatch.setattr(
            msu, "_record_stream_for_tensor_outputs", lambda t, stream: recorded.append((t, stream))
        )
        # Simulate being inside CUDA graph capture without actually capturing,
        # so the surrounding stream bookkeeping still executes normally.
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

        x = torch.randn(8, device="cuda")
        try:
            out = msu.begin_aux_stream_passthrough(x, device=device)
        finally:
            # begin_aux switches the current stream to aux; restore the default.
            torch.cuda.set_stream(torch.cuda.default_stream(device))

        assert out is x
        assert len(recorded) == 1, (
            "begin_aux_stream_passthrough must call record_stream during CUDA "
            "graph capture (regression of the multi-stream record_stream fix)"
        )
        assert recorded[0][0] is x
