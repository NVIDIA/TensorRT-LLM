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
"""Unit tests for piecewise_runner: ADPiecewiseRunner and SegmentEntry."""

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.compile.piecewise_runner import ADPiecewiseRunner, SegmentEntry

# ============================================================================
# Context management tests
# ============================================================================


class TestADPiecewiseRunnerContextManagement:
    """Tests for class-level context management on ADPiecewiseRunner."""

    def setup_method(self):
        """Reset class-level state before each test."""
        ADPiecewiseRunner._current_num_tokens = None
        ADPiecewiseRunner._current_phase = "replay"
        ADPiecewiseRunner._static_output_registry.clear()

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

    def test_clear_static_output_registry(self):
        # Populate with some dummy data
        t = torch.tensor([1.0])
        ADPiecewiseRunner._static_output_registry[(8, 12345)] = t
        assert len(ADPiecewiseRunner._static_output_registry) == 1

        ADPiecewiseRunner.clear_static_output_registry()
        assert len(ADPiecewiseRunner._static_output_registry) == 0


# ============================================================================
# Initialization tests
# ============================================================================


class TestADPiecewiseRunnerInit:
    """Tests for ADPiecewiseRunner initialization."""

    def test_entries_pre_populated(self):
        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8, 16, 32])
        assert set(runner.entries.keys()) == {8, 16, 32}
        for entry in runner.entries.values():
            assert isinstance(entry, SegmentEntry)

    def test_weight_ptrs_collected(self):
        submod = nn.Linear(4, 4, bias=True)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])
        # Should have weight and bias data_ptrs
        assert submod.weight.data_ptr() in runner._weight_ptrs
        assert submod.bias.data_ptr() in runner._weight_ptrs

    def test_no_piecewise_num_tokens(self):
        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=None)
        assert len(runner.entries) == 0


# ============================================================================
# _find_entry tests
# ============================================================================


class TestADPiecewiseRunnerFindEntry:
    """Tests for _find_entry."""

    def test_exact_match(self):
        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8, 16])
        assert runner._find_entry(8) is not None
        assert runner._find_entry(16) is not None

    def test_no_match_returns_none(self):
        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8, 16])
        assert runner._find_entry(32) is None
        assert runner._find_entry(4) is None


# ============================================================================
# _identify_dynamic_indices tests
# ============================================================================


class TestIdentifyDynamicIndices:
    """Tests for _identify_dynamic_indices."""

    def test_weight_tensors_excluded(self):
        submod = nn.Linear(4, 4, bias=False)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])
        entry = runner.entries[8]

        # flat_args: [weight_tensor, activation_tensor]
        weight = submod.weight
        activation = torch.randn(2, 4)
        flat_args = [weight, activation]

        dynamic = runner._identify_dynamic_indices(entry, flat_args)
        assert 0 not in dynamic  # weight is not dynamic
        assert 1 in dynamic  # activation is dynamic

    def test_non_tensor_args_ignored(self):
        submod = nn.Linear(4, 4, bias=False)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])
        entry = runner.entries[8]

        flat_args = [42, "hello", torch.randn(2, 4)]
        dynamic = runner._identify_dynamic_indices(entry, flat_args)
        # Only the tensor at index 2 should be dynamic
        assert dynamic == {2}

    def test_all_activations_marked_dynamic(self):
        submod = nn.Linear(4, 4, bias=False)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])
        entry = runner.entries[8]

        act1 = torch.randn(2, 4)
        act2 = torch.randn(3, 4)
        flat_args = [act1, act2]

        dynamic = runner._identify_dynamic_indices(entry, flat_args)
        assert dynamic == {0, 1}


# ============================================================================
# _track_warmup_ptrs tests
# ============================================================================


class TestTrackWarmupPtrs:
    """Tests for _track_warmup_ptrs."""

    def test_first_call_records_ptrs(self):
        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])
        entry = runner.entries[8]

        t1 = torch.randn(2, 4)
        t2 = torch.randn(3, 4)
        flat_args = [t1, t2]

        runner._track_warmup_ptrs(entry, flat_args)
        assert entry._warmup_data_ptrs is not None
        assert len(entry._warmup_data_ptrs) == 2
        assert entry._warmup_data_ptrs[0] == t1.data_ptr()
        assert entry._warmup_data_ptrs[1] == t2.data_ptr()

    def test_second_call_detects_ptr_change(self):
        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])
        entry = runner.entries[8]

        t1 = torch.randn(2, 4)
        t2 = torch.randn(3, 4)

        # First warmup
        runner._track_warmup_ptrs(entry, [t1, t2])

        # Second warmup with a new tensor at index 0 (simulating changed activation)
        t1_new = torch.randn(2, 4)  # new tensor, different data_ptr
        runner._track_warmup_ptrs(entry, [t1_new, t2])

        # Index 0 should be marked None (dynamic), index 1 unchanged
        assert entry._warmup_data_ptrs[0] is None
        assert entry._warmup_data_ptrs[1] == t2.data_ptr()

    def test_non_tensor_args_tracked_as_none(self):
        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])
        entry = runner.entries[8]

        flat_args = [42, torch.randn(2, 4)]
        runner._track_warmup_ptrs(entry, flat_args)
        assert entry._warmup_data_ptrs[0] is None  # int -> None
        assert entry._warmup_data_ptrs[1] is not None  # tensor -> data_ptr


# ============================================================================
# _prepare_replay_inputs tests
# ============================================================================


class TestPrepareReplayInputs:
    """Tests for _prepare_replay_inputs."""

    def _make_entry_with_dynamic(self, static_inputs, dynamic_indices):
        entry = SegmentEntry()
        entry.static_inputs = static_inputs
        entry.dynamic_indices = dynamic_indices
        return entry

    def test_same_shape_copy(self):
        """When shapes match, static buffer should be updated."""
        static_buf = torch.zeros(4, 8)
        new_inp = torch.ones(4, 8)
        entry = self._make_entry_with_dynamic([static_buf], {0})

        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])
        runner._prepare_replay_inputs(entry, [new_inp])

        assert torch.equal(static_buf, new_inp)

    def test_same_data_ptr_skips_copy(self):
        """When data_ptr matches (zero-copy path), no copy should happen."""
        shared_tensor = torch.zeros(4, 8)
        entry = self._make_entry_with_dynamic([shared_tensor], {0})

        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])

        # Pass the same tensor -- same data_ptr
        runner._prepare_replay_inputs(entry, [shared_tensor])
        # Should still be zeros (no copy from a different source)
        assert torch.equal(shared_tensor, torch.zeros(4, 8))

    def test_padded_dim0(self):
        """When new_inp is smaller along dim 0, only prefix should be copied."""
        static_buf = torch.zeros(8, 4)
        new_inp = torch.ones(5, 4)
        entry = self._make_entry_with_dynamic([static_buf], {0})

        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])
        runner._prepare_replay_inputs(entry, [new_inp])

        # First 5 rows should be ones, remaining 3 should be zeros
        assert torch.equal(static_buf[:5], torch.ones(5, 4))
        assert torch.equal(static_buf[5:], torch.zeros(3, 4))

    def test_padded_dim1(self):
        """When new_inp is smaller along dim 1, only prefix columns should be copied."""
        static_buf = torch.zeros(1, 16, 4)
        new_inp = torch.ones(1, 10, 4)
        entry = self._make_entry_with_dynamic([static_buf], {0})

        submod = nn.Linear(4, 4)
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8])
        runner._prepare_replay_inputs(entry, [new_inp])

        # First 10 along dim 1 should be ones, rest zeros
        assert torch.equal(static_buf[:, :10, :], torch.ones(1, 10, 4))
        assert torch.equal(static_buf[:, 10:, :], torch.zeros(1, 6, 4))


# ============================================================================
# Full cycle tests (CUDA required)
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestADPiecewiseRunnerFullCycle:
    """End-to-end warmup -> capture -> replay test on CUDA."""

    def setup_method(self):
        """Reset class-level state before each test."""
        ADPiecewiseRunner._current_num_tokens = None
        ADPiecewiseRunner._current_phase = "replay"
        ADPiecewiseRunner._static_output_registry.clear()

    def test_warmup_capture_replay_linear(self):
        """Full cycle: warmup -> capture -> replay with a simple Linear."""
        device = "cuda"
        submod = nn.Linear(16, 16, bias=True).to(device)
        submod.eval()

        num_tokens = 8
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[num_tokens], graph_pool=None).to(
            device
        )

        # Fixed input for warmup/capture
        x = torch.randn(num_tokens, 16, device=device)

        with torch.inference_mode():
            # --- WARMUP ---
            ADPiecewiseRunner.set_current_num_tokens(num_tokens)
            ADPiecewiseRunner.set_current_phase("warmup")
            for _ in range(3):
                _ = runner(x)

            # --- CAPTURE ---
            ADPiecewiseRunner.set_current_phase("capture")
            _ = runner(x)

            # --- REPLAY ---
            ADPiecewiseRunner.set_current_phase("replay")

            # New input for replay
            x_new = torch.randn(num_tokens, 16, device=device)
            replay_out = runner(x_new)

            # Compare with eager output
            eager_out = submod(x_new)

            torch.cuda.synchronize()
            assert torch.allclose(replay_out, eager_out, atol=1e-5), (
                "Replay output should match eager output"
            )

    def test_eager_fallback_for_unknown_num_tokens(self):
        """Runner should fall back to eager for num_tokens not in entries."""
        device = "cuda"
        submod = nn.Linear(16, 16).to(device)
        submod.eval()

        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8]).to(device)

        x = torch.randn(4, 16, device=device)

        with torch.inference_mode():
            # num_tokens=4 is not configured -- should fall back to eager
            ADPiecewiseRunner.set_current_num_tokens(4)
            ADPiecewiseRunner.set_current_phase("replay")
            out = runner(x)

            eager_out = submod(x)
            assert torch.allclose(out, eager_out, atol=1e-6)

    def test_eager_fallback_for_none_num_tokens(self):
        """Runner should fall back to eager when num_tokens is None."""
        device = "cuda"
        submod = nn.Linear(16, 16).to(device)
        submod.eval()

        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=[8]).to(device)

        x = torch.randn(4, 16, device=device)

        with torch.inference_mode():
            ADPiecewiseRunner.set_current_num_tokens(None)
            ADPiecewiseRunner.set_current_phase("replay")
            out = runner(x)

            eager_out = submod(x)
            assert torch.allclose(out, eager_out, atol=1e-6)

    def test_multiple_bucket_sizes(self):
        """Capture and replay with multiple bucket sizes."""
        device = "cuda"
        submod = nn.Linear(16, 16).to(device)
        submod.eval()

        buckets = [4, 8, 16]
        runner = ADPiecewiseRunner(submod, piecewise_num_tokens=buckets).to(device)

        with torch.inference_mode():
            for nt in buckets:
                x = torch.randn(nt, 16, device=device)

                ADPiecewiseRunner.set_current_num_tokens(nt)

                # Warmup
                ADPiecewiseRunner.set_current_phase("warmup")
                for _ in range(3):
                    runner(x)

                # Capture
                ADPiecewiseRunner.set_current_phase("capture")
                runner(x)

            # Replay each bucket
            ADPiecewiseRunner.set_current_phase("replay")
            for nt in buckets:
                x_new = torch.randn(nt, 16, device=device)
                ADPiecewiseRunner.set_current_num_tokens(nt)
                replay_out = runner(x_new)
                eager_out = submod(x_new)

                torch.cuda.synchronize()
                assert torch.allclose(replay_out, eager_out, atol=1e-5), (
                    f"Replay mismatch for bucket {nt}"
                )
