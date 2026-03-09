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
"""Unit tests for piecewise_runner: ADPiecewiseRunner, SegmentEntry, OutputInfo."""

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.compile.piecewise_runner import (
    ADPiecewiseRunner,
    OutputInfo,
    SegmentEntry,
)

# ============================================================================
# Context management tests
# ============================================================================


class TestADPiecewiseRunnerContextManagement:
    def setup_method(self):
        ADPiecewiseRunner._current_num_tokens = None
        ADPiecewiseRunner._current_phase = "replay"

    def test_set_current_num_tokens(self):
        ADPiecewiseRunner.set_current_num_tokens(128)
        assert ADPiecewiseRunner._current_num_tokens == 128

        ADPiecewiseRunner.set_current_num_tokens(None)
        assert ADPiecewiseRunner._current_num_tokens is None

    def test_set_current_phase_valid(self):
        for phase in ("warmup", "capture", "replay"):
            ADPiecewiseRunner.set_current_phase(phase)
            assert ADPiecewiseRunner._current_phase == phase

    def test_set_current_phase_invalid_raises(self):
        with pytest.raises(AssertionError, match="Invalid phase"):
            ADPiecewiseRunner.set_current_phase("invalid_phase")


# ============================================================================
# Initialization tests
# ============================================================================


class TestADPiecewiseRunnerInit:
    def test_entries_initially_empty(self):
        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod)
        assert len(runner.entries) == 0

    def test_weight_ptrs_collected(self):
        submod = nn.Linear(4, 4, bias=True)
        runner = ADPiecewiseRunner(submod)
        assert submod.weight.data_ptr() in runner._weight_ptrs
        assert submod.bias.data_ptr() in runner._weight_ptrs

    def test_no_dynamic_out_info_by_default(self):
        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod)
        assert runner._next_dynamic_out_infos == {}


# ============================================================================
# OutputInfo and SegmentEntry tests
# ============================================================================


class TestOutputInfo:
    def test_creation(self):
        info = OutputInfo(
            shape=torch.Size([128, 32, 64]),
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
        assert info.shape == torch.Size([128, 32, 64])
        assert info.dtype == torch.float16


class TestSegmentEntry:
    def test_default_values(self):
        entry = SegmentEntry()
        assert entry.cuda_graph is None
        assert entry.static_output is None
        assert entry.dynamic_out_bufs == {}
        assert entry.input_addresses == []


# ============================================================================
# set_dynamic_out_info / get_dynamic_out_buf tests
# ============================================================================


class TestDynamicOutBuf:
    def test_set_and_get_returns_none_before_capture(self):
        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod)
        info = OutputInfo(
            shape=torch.Size([8, 16]),
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        dynamic_id = 5
        runner.set_dynamic_out_info(dynamic_id, info)
        assert runner._next_dynamic_out_infos[dynamic_id] is info
        assert runner.get_dynamic_out_buf(8, dynamic_id) is None


# ============================================================================
# Full cycle tests (CUDA required)
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestADPiecewiseRunnerFullCycle:
    def setup_method(self):
        ADPiecewiseRunner._current_num_tokens = None
        ADPiecewiseRunner._current_phase = "replay"

    def test_warmup_capture_replay_same_address(self):
        """Replay with the same input tensor (addresses match -> graph replay)."""
        device = "cuda"
        submod = nn.Linear(16, 16, bias=True).to(device)
        submod.eval()

        num_tokens = 8
        runner = ADPiecewiseRunner(submod, graph_pool=None).to(device)

        x = torch.randn(num_tokens, 16, device=device)

        with torch.inference_mode():
            ADPiecewiseRunner.set_current_num_tokens(num_tokens)
            ADPiecewiseRunner.set_current_phase("warmup")
            for _ in range(3):
                _ = runner(x)

            ADPiecewiseRunner.set_current_phase("capture")
            _ = runner(x)

            ADPiecewiseRunner.set_current_phase("replay")
            x.fill_(1.0)
            replay_out = runner(x)

            eager_out = submod(x)
            torch.cuda.synchronize()
            assert torch.allclose(replay_out, eager_out, atol=1e-5)

    def test_input_addresses_recorded_during_capture(self):
        device = "cuda"
        submod = nn.Linear(16, 16).to(device)
        submod.eval()

        num_tokens = 8
        runner = ADPiecewiseRunner(submod).to(device)
        x = torch.randn(num_tokens, 16, device=device)

        with torch.inference_mode():
            ADPiecewiseRunner.set_current_num_tokens(num_tokens)

            ADPiecewiseRunner.set_current_phase("warmup")
            for _ in range(3):
                runner(x)

            ADPiecewiseRunner.set_current_phase("capture")
            runner(x)

            entry = runner.entries[num_tokens]
            assert len(entry.input_addresses) > 0
            tensor_addrs = [a for a in entry.input_addresses if a is not None]
            assert len(tensor_addrs) > 0

    def test_eager_fallback_for_none_num_tokens(self):
        device = "cuda"
        submod = nn.Linear(16, 16).to(device)
        submod.eval()

        runner = ADPiecewiseRunner(submod).to(device)
        x = torch.randn(4, 16, device=device)

        with torch.inference_mode():
            ADPiecewiseRunner.set_current_num_tokens(None)
            ADPiecewiseRunner.set_current_phase("replay")
            out = runner(x)
            eager_out = submod(x)
            assert torch.allclose(out, eager_out, atol=1e-6)

    def test_static_output_is_weak_ref(self):
        device = "cuda"
        submod = nn.Linear(16, 16).to(device)
        submod.eval()

        num_tokens = 8
        runner = ADPiecewiseRunner(submod).to(device)
        x = torch.randn(num_tokens, 16, device=device)

        with torch.inference_mode():
            ADPiecewiseRunner.set_current_num_tokens(num_tokens)

            ADPiecewiseRunner.set_current_phase("warmup")
            for _ in range(3):
                runner(x)

            ADPiecewiseRunner.set_current_phase("capture")
            capture_out = runner(x)

            entry = runner.entries[num_tokens]
            assert entry.static_output is not None
            if isinstance(capture_out, torch.Tensor):
                assert entry.static_output.data_ptr() == capture_out.data_ptr()

    def test_dynamic_out_buf_allocated_during_capture(self):
        """When set_dynamic_out_info is called, capture allocates a buffer."""
        device = "cuda"
        submod = nn.Linear(16, 16).to(device)
        submod.eval()

        num_tokens = 8
        dynamic_id = 7
        runner = ADPiecewiseRunner(submod).to(device)

        info = OutputInfo(
            shape=torch.Size([num_tokens, 32]),
            dtype=torch.float16,
            device=torch.device(device),
        )
        runner.set_dynamic_out_info(dynamic_id, info)

        x = torch.randn(num_tokens, 16, device=device)

        with torch.inference_mode():
            ADPiecewiseRunner.set_current_num_tokens(num_tokens)

            ADPiecewiseRunner.set_current_phase("warmup")
            for _ in range(3):
                runner(x)

            ADPiecewiseRunner.set_current_phase("capture")
            runner(x)

            entry = runner.entries[num_tokens]
            buf = entry.dynamic_out_bufs.get(dynamic_id)
            assert buf is not None
            assert buf.shape == torch.Size([num_tokens, 32])
            assert buf.dtype == torch.float16

    def test_get_dynamic_out_buf_after_capture(self):
        """get_dynamic_out_buf should return the buffer after capture."""
        device = "cuda"
        submod = nn.Linear(16, 16).to(device)
        submod.eval()

        num_tokens = 8
        dynamic_id = 7
        runner = ADPiecewiseRunner(submod).to(device)

        info = OutputInfo(
            shape=torch.Size([num_tokens, 32]),
            dtype=torch.float16,
            device=torch.device(device),
        )
        runner.set_dynamic_out_info(dynamic_id, info)

        x = torch.randn(num_tokens, 16, device=device)

        with torch.inference_mode():
            ADPiecewiseRunner.set_current_num_tokens(num_tokens)

            ADPiecewiseRunner.set_current_phase("warmup")
            for _ in range(3):
                runner(x)

            ADPiecewiseRunner.set_current_phase("capture")
            runner(x)

            buf = runner.get_dynamic_out_buf(num_tokens, dynamic_id)
            assert buf is not None
            assert buf.shape == torch.Size([num_tokens, 32])

    def test_entries_created_on_demand(self):
        """Entries are created during capture, not at init."""
        device = "cuda"
        submod = nn.Linear(16, 16).to(device)
        submod.eval()

        runner = ADPiecewiseRunner(submod).to(device)
        assert len(runner.entries) == 0

        x = torch.randn(8, 16, device=device)
        with torch.inference_mode():
            ADPiecewiseRunner.set_current_num_tokens(8)
            ADPiecewiseRunner.set_current_phase("warmup")
            runner(x)
            assert 8 not in runner.entries

            ADPiecewiseRunner.set_current_phase("capture")
            runner(x)
            assert 8 in runner.entries
