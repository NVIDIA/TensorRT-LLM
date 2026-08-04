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
