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
    MultiStreamWrapper,
    OutputInfo,
    SegmentEntry,
    _copy_into_stable,
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


# ============================================================================
# _copy_into_stable helper tests (container-agnostic copy used by MultiStreamWrapper)
# ============================================================================


class TestCopyIntoStable:
    def test_single_tensor_copy_returns_stable(self):
        result = torch.randn(4, 8)
        stable = torch.empty_like(result)
        out = _copy_into_stable(result, stable)
        assert out is stable
        assert torch.equal(out, result)

    def test_single_tensor_same_identity_is_noop(self):
        """When result already IS the stable buffer, no self-copy is performed."""
        buf = torch.randn(4, 8)
        out = _copy_into_stable(buf, buf)
        assert out is buf

    def test_tuple_of_tensors(self):
        r = (torch.randn(3), torch.randn(2, 2))
        s = [torch.empty_like(t) for t in r]
        out = _copy_into_stable(r, s)
        assert isinstance(out, tuple)
        assert out[0] is s[0] and out[1] is s[1]
        assert torch.equal(out[0], r[0])
        assert torch.equal(out[1], r[1])

    def test_list_of_tensors(self):
        r = [torch.randn(3), torch.randn(2, 2)]
        s = [torch.empty_like(t) for t in r]
        out = _copy_into_stable(r, s)
        assert isinstance(out, list)
        assert out[0] is s[0] and out[1] is s[1]

    def test_tuple_with_none_passthrough(self):
        """Non-tensor entries (e.g. None) round-trip unchanged."""
        r = (torch.randn(3), None, torch.randn(4))
        s = [torch.empty_like(r[0]), None, torch.empty_like(r[2])]
        out = _copy_into_stable(r, s)
        assert isinstance(out, tuple)
        assert out[0] is s[0]
        assert out[1] is None
        assert out[2] is s[2]

    def test_length_mismatch_returns_result(self):
        """Defensive path: if the stable container doesn't match, fall back to result."""
        r = (torch.randn(3), torch.randn(4))
        s = [torch.empty_like(r[0])]  # wrong length
        out = _copy_into_stable(r, s)
        assert out is r


# ============================================================================
# MultiStreamWrapper tests
# ============================================================================


class _DummyMS(nn.Module):
    """Toy submodule that mimics a multi-stream MoE region.

    Returns a fresh tensor each call so callers can observe address drift.
    """

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.device = torch.device("cuda")

    def forward(self, x):
        # Produce a fresh tensor whose content depends on x so repeated calls
        # return different data. Allocation is via regular caching allocator,
        # so its pointer is not address-stable across calls.
        return torch.full(self.shape, float(x.sum().item()), device=self.device)


class _DummyMSTuple(nn.Module):
    """Tuple-output variant: returns (tensor, None, tensor)."""

    def __init__(self, shape_a, shape_b):
        super().__init__()
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.device = torch.device("cuda")

    def forward(self, x):
        s = float(x.sum().item())
        return (
            torch.full(self.shape_a, s, device=self.device),
            None,
            torch.full(self.shape_b, s + 1, device=self.device),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MultiStreamWrapper tests need CUDA")
class TestMultiStreamWrapper:
    def setup_method(self):
        ADPiecewiseRunner._current_num_tokens = None
        ADPiecewiseRunner._current_phase = "replay"

    def test_warmup_returns_raw_result_without_allocating(self):
        """During warmup, the wrapper forwards through the submod unchanged."""
        wrapper = MultiStreamWrapper(_DummyMS((4, 8))).cuda()
        ADPiecewiseRunner.set_current_phase("warmup")
        ADPiecewiseRunner.set_current_num_tokens(4)

        out = wrapper(torch.ones(4, device="cuda"))
        assert out.shape == (4, 8)
        assert wrapper._stable_bufs == {}

    def test_none_num_tokens_returns_raw_result(self):
        """When no bucket is active, the wrapper is a pass-through."""
        wrapper = MultiStreamWrapper(_DummyMS((4, 8))).cuda()
        ADPiecewiseRunner.set_current_phase("replay")
        ADPiecewiseRunner.set_current_num_tokens(None)

        out = wrapper(torch.ones(4, device="cuda"))
        assert out.shape == (4, 8)
        assert wrapper._stable_bufs == {}

    def test_capture_allocates_stable_and_returns_it(self):
        wrapper = MultiStreamWrapper(_DummyMS((4, 8))).cuda()
        ADPiecewiseRunner.set_current_phase("capture")
        ADPiecewiseRunner.set_current_num_tokens(4)

        out = wrapper(torch.ones(4, device="cuda"))
        assert 4 in wrapper._stable_bufs
        stable = wrapper._stable_bufs[4]
        assert out is stable
        assert out.shape == (4, 8)

    def test_replay_reuses_stable_and_address_is_fixed(self):
        """Stable buffer address must not drift across replays for one bucket.

        This is the core invariant the piecewise captured graph relies on.
        """
        wrapper = MultiStreamWrapper(_DummyMS((4, 8))).cuda()

        ADPiecewiseRunner.set_current_num_tokens(4)
        ADPiecewiseRunner.set_current_phase("capture")
        out1 = wrapper(torch.ones(4, device="cuda"))
        ptr_capture = out1.data_ptr()

        ADPiecewiseRunner.set_current_phase("replay")
        out2 = wrapper(torch.ones(4, device="cuda") * 2)
        out3 = wrapper(torch.ones(4, device="cuda") * 3)

        assert out2.data_ptr() == ptr_capture
        assert out3.data_ptr() == ptr_capture
        # Data reflects the most recent submodule call (copied in each time).
        assert torch.all(out3 == 3.0 * 4)

    def test_different_buckets_get_independent_buffers(self):
        wrapper = MultiStreamWrapper(_DummyMS((4, 8))).cuda()

        ADPiecewiseRunner.set_current_phase("capture")
        ADPiecewiseRunner.set_current_num_tokens(4)
        wrapper(torch.ones(4, device="cuda"))
        ADPiecewiseRunner.set_current_num_tokens(16)
        wrapper(torch.ones(4, device="cuda"))

        assert 4 in wrapper._stable_bufs and 16 in wrapper._stable_bufs
        assert wrapper._stable_bufs[4].data_ptr() != wrapper._stable_bufs[16].data_ptr()

    def test_tuple_output_is_preserved_and_addresses_stable(self):
        wrapper = MultiStreamWrapper(_DummyMSTuple((3,), (5,))).cuda()

        ADPiecewiseRunner.set_current_num_tokens(3)
        ADPiecewiseRunner.set_current_phase("capture")
        out1 = wrapper(torch.ones(3, device="cuda"))
        assert isinstance(out1, tuple) and len(out1) == 3
        assert out1[1] is None
        ptrs = (out1[0].data_ptr(), out1[2].data_ptr())

        ADPiecewiseRunner.set_current_phase("replay")
        out2 = wrapper(torch.ones(3, device="cuda") * 2)
        assert out2[0].data_ptr() == ptrs[0]
        assert out2[2].data_ptr() == ptrs[1]
        assert out2[1] is None
        assert torch.all(out2[0] == 2.0 * 3)
        assert torch.all(out2[2] == 2.0 * 3 + 1)

    def test_stable_buffer_outside_graph_pool_does_not_recycle(self):
        """Regression test for the Qwen3.5 IMA.

        Even when other large allocations happen between calls, the stable
        buffer address must remain valid because it is outside any graph
        pool and strong-ref'd by the wrapper.
        """
        wrapper = MultiStreamWrapper(_DummyMS((4, 8))).cuda()

        ADPiecewiseRunner.set_current_phase("capture")
        ADPiecewiseRunner.set_current_num_tokens(4)
        wrapper(torch.ones(4, device="cuda"))
        ptr = wrapper._stable_bufs[4].data_ptr()

        # Churn the caching allocator.
        for _ in range(64):
            _ = torch.empty(1024 * 1024, device="cuda")

        # Wrapper still holds its buffer at the same address.
        assert wrapper._stable_bufs[4].data_ptr() == ptr
        ADPiecewiseRunner.set_current_phase("replay")
        out = wrapper(torch.ones(4, device="cuda"))
        assert out.data_ptr() == ptr
