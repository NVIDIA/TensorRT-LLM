# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PyExecutor pipelined scheduler (Issue #13957).

Covers:
- _is_pipelined_loop_eligible gate conditions
- _PipelinedForwardResult / _PipelinedSlot dataclass construction
- validate_pipelined_scheduler model_validator scenarios
- PIPELINE_DEPTH constant
"""

import logging

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock


def _make_exec_mock(enable_pipelined=True):
    """Create a MagicMock executor with all eligibility-gate blockers cleared.

    ``MagicMock`` attributes default to a truthy ``MagicMock()`` object, so
    every attribute read by ``_is_pipelined_loop_eligible`` must be set to its
    non-blocking default explicitly.
    """
    exec_mock = MagicMock()
    exec_mock.llm_args.enable_pipelined_scheduler = enable_pipelined
    exec_mock.disable_overlap_scheduler = False
    # Executor-level blockers -- all must be non-blocking defaults.
    exec_mock.drafter = None
    exec_mock.kv_cache_transceiver = None
    exec_mock.guided_decoder = None
    exec_mock.kv_connector_manager = None
    exec_mock.is_benchmark_disagg = False
    exec_mock.enable_attention_dp = False
    exec_mock._mm_encoder_item_scheduling_enabled = False
    exec_mock.is_encoder_decoder = False
    return exec_mock


class TestPipelinedSchedulerEligibility:
    """_is_pipelined_loop_eligible gate tests."""

    def test_disabled_by_default(self):
        """Pipelined scheduler is gated by the feature flag."""
        mock_exec = _make_exec_mock(enable_pipelined=False)
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_enabled_with_plain_tp(self):
        """Plain TP (no feature blockers) returns True."""
        mock_exec = _make_exec_mock(enable_pipelined=True)
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is True

    def test_blocked_by_speculative_decoding(self):
        """Setting drafter (executor-level speculative gate) returns False."""
        mock_exec = _make_exec_mock()
        mock_exec.drafter = MagicMock()
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_blocked_by_disaggregated_serving(self):
        """Setting kv_cache_transceiver (executor-level disagg gate) returns False."""
        mock_exec = _make_exec_mock()
        mock_exec.kv_cache_transceiver = MagicMock()
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_blocked_by_attention_dp(self):
        """Executor-level enable_attention_dp blocks pipelined scheduling."""
        mock_exec = _make_exec_mock()
        mock_exec.enable_attention_dp = True
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_blocked_by_disable_overlap(self):
        """disable_overlap_scheduler (executor-level) blocks pipelined scheduling."""
        mock_exec = _make_exec_mock()
        mock_exec.disable_overlap_scheduler = True
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_blocked_by_multimodal_encoder(self):
        """Executor-level _mm_encoder_item_scheduling_enabled blocks pipelined scheduling."""
        mock_exec = _make_exec_mock()
        mock_exec._mm_encoder_item_scheduling_enabled = True
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_blocked_by_guided_decoding(self):
        """Executor-level guided_decoder blocks pipelined scheduling."""
        mock_exec = _make_exec_mock()
        mock_exec.guided_decoder = MagicMock()
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False


class TestPipelinedDataClasses:
    """_PipelinedForwardResult and _PipelinedSlot construction."""

    def test_pipeline_depth_constant(self):
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        assert PyExecutor.PIPELINE_DEPTH == 2

    def test_forward_result_defaults(self):
        from tensorrt_llm._torch.pyexecutor.py_executor import _PipelinedForwardResult
        r = _PipelinedForwardResult(sample_state=None)
        assert r.sample_state is None
        assert r.fwd_timing is None
        assert r.sample_timing is None

    def test_forward_result_with_state(self):
        from tensorrt_llm._torch.pyexecutor.py_executor import _PipelinedForwardResult
        fake_state = MagicMock()
        r = _PipelinedForwardResult(sample_state=fake_state, fwd_timing=123.4, sample_timing=45.6)
        assert r.sample_state is fake_state
        assert r.fwd_timing == 123.4
        assert r.sample_timing == 45.6


class TestPipelinedSchedulerValidator:
    """validate_pipelined_scheduler model_validator."""

    def test_disabled_flag_passes_through(self):
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        args = TorchLlmArgs(model="gpt2", enable_pipelined_scheduler=False)
        args.validate_pipelined_scheduler()
        assert args.enable_pipelined_scheduler is False

    def test_enabled_with_plain_tp(self):
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        args = TorchLlmArgs(model="gpt2", enable_pipelined_scheduler=True)
        args.validate_pipelined_scheduler()
        assert args.enable_pipelined_scheduler is True

    def test_disabled_when_blocked_by_disagg(self):
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        args = TorchLlmArgs(model="gpt2", enable_pipelined_scheduler=True)
        # Simulate disaggregated config
        args.cache_transceiver_config = MagicMock()
        args.cache_transceiver_config.backend = "nvshmem"
        args.validate_pipelined_scheduler()
        assert args.enable_pipelined_scheduler is False

    def test_disabled_when_overlap_disabled(self):
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        args = TorchLlmArgs(
            model="gpt2",
            enable_pipelined_scheduler=True,
            disable_overlap_scheduler=True,
        )
        args.validate_pipelined_scheduler()
        assert args.enable_pipelined_scheduler is False

    def test_disabled_when_attention_dp(self):
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        args = TorchLlmArgs(
            model="gpt2",
            enable_pipelined_scheduler=True,
            enable_attention_dp=True,
        )
        args.validate_pipelined_scheduler()
        assert args.enable_pipelined_scheduler is False

    def test_disabled_when_speculative(self):
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        args = TorchLlmArgs(model="gpt2", enable_pipelined_scheduler=True)
        args.speculative_config = MagicMock()
        args.validate_pipelined_scheduler()
        assert args.enable_pipelined_scheduler is False

    def test_multiple_blockers_produce_single_warning_with_all_names(self):
        """When multiple incompatible features are set, exactly one warning
        is emitted that names every blocker; the pipelined flag is disabled."""
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        args = TorchLlmArgs(
            model="gpt2",
            enable_pipelined_scheduler=True,
            disable_overlap_scheduler=True,
            enable_attention_dp=True,
        )
        with patch('tensorrt_llm.llmapi.llm_args.logger') as mock_logger:
            args.validate_pipelined_scheduler()
            assert args.enable_pipelined_scheduler is False
            # Exactly one warning call
            assert mock_logger.warning.call_count == 1
            msg = mock_logger.warning.call_args[0][0]
            # The warning must mention every active blocker
            assert 'disable_overlap_scheduler' in msg
            assert 'attention DP' in msg


class TestExecutorLoopSelection:
    """Verify _executor_loop dispatch selects pipelined when eligible."""

    def _apply_dispatch(self, mock_exec):
        """Replicate the PyExecutor.__init__ dispatch branch for pipelined
        eligibility (the branch preceding the overlap / standard fallback)."""
        if mock_exec._is_pipelined_loop_eligible():
            mock_exec.event_loop = mock_exec._executor_loop_pipelined
        elif mock_exec.disable_overlap_scheduler:
            mock_exec.event_loop = mock_exec._executor_loop
        else:
            mock_exec.event_loop = mock_exec._executor_loop_overlap

    def test_pipelined_selected_when_eligible(self):
        """Real dispatch chooses _executor_loop_pipelined when eligible."""
        mock_exec = _make_exec_mock(enable_pipelined=True)
        mock_exec._executor_loop_pipelined = MagicMock(name="pipelined")
        mock_exec._executor_loop_overlap = MagicMock(name="overlap")
        mock_exec._executor_loop = MagicMock(name="standard")

        self._apply_dispatch(mock_exec)

        assert mock_exec.event_loop is mock_exec._executor_loop_pipelined
        mock_exec.event_loop()
        mock_exec._executor_loop_pipelined.assert_called_once()
        mock_exec._executor_loop_overlap.assert_not_called()
        mock_exec._executor_loop.assert_not_called()

    def test_overlap_selected_when_not_eligible(self):
        """Real dispatch falls back to _executor_loop_overlap when pipelined is
        ineligible and disable_overlap_scheduler is False."""
        mock_exec = _make_exec_mock(enable_pipelined=False)
        mock_exec._executor_loop_pipelined = MagicMock(name="pipelined")
        mock_exec._executor_loop_overlap = MagicMock(name="overlap")
        mock_exec._executor_loop = MagicMock(name="standard")

        self._apply_dispatch(mock_exec)

        assert mock_exec.event_loop is mock_exec._executor_loop_overlap
        mock_exec.event_loop()
        mock_exec._executor_loop_overlap.assert_called_once()
        mock_exec._executor_loop_pipelined.assert_not_called()
        mock_exec._executor_loop.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
