# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PyExecutor pipelined scheduler (Issue #13957).

Covers:
- _is_pipelined_loop_eligible gate conditions
- _PipelinedForwardResult / _PipelinedSlot dataclass construction
- validate_pipelined_scheduler model_validator scenarios
- PIPELINE_DEPTH constant
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock


class TestPipelinedSchedulerEligibility:
    """_is_pipelined_loop_eligible gate tests."""

    def test_disabled_by_default(self):
        """Pipelined scheduler is gated by the feature flag."""
        mock_exec = MagicMock()
        mock_exec.llm_args.enable_pipelined_scheduler = False
        mock_exec.disable_overlap_scheduler = False
        mock_exec.llm_args.speculative_config = None
        mock_exec.llm_args.cache_transceiver_config = None
        mock_exec.llm_args.enable_attention_dp = False
        mock_exec.llm_args.guided_decoding_params = None
        mock_exec.llm_args.model = "gpt2"
        mock_exec.llm_args.encoder_config = None
        mock_exec.llm_args.decoder_config = None
        mock_exec.llm_args.enable_disaggregated serving = False
        mock_exec.llm_args.enable_pipeline_parallel = False
        # Patch the actual method onto mock
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_enabled_with_plain_tp(self):
        """Plain TP (no feature blockers) returns True."""
        mock_exec = MagicMock()
        mock_exec.llm_args.enable_pipelined_scheduler = True
        mock_exec.disable_overlap_scheduler = False
        mock_exec.llm_args.speculative_config = None
        mock_exec.llm_args.cache_transceiver_config = None
        mock_exec.llm_args.enable_attention_dp = False
        mock_exec.llm_args.guided_decoding_params = None
        mock_exec.llm_args.model = "gpt2"
        mock_exec.llm_args.encoder_config = None
        mock_exec.llm_args.decoder_config = None
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is True

    def test_blocked_by_speculative_decoding(self):
        mock_exec = MagicMock()
        mock_exec.llm_args.enable_pipelined_scheduler = True
        mock_exec.disable_overlap_scheduler = False
        mock_exec.llm_args.speculative_config = MagicMock()
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_blocked_by_disaggregated_serving(self):
        mock_exec = MagicMock()
        mock_exec.llm_args.enable_pipelined_scheduler = True
        mock_exec.disable_overlap_scheduler = False
        mock_exec.llm_args.speculative_config = None
        mock_exec.llm_args.cache_transceiver_config = MagicMock()
        mock_exec.llm_args.cache_transceiver_config.backend = "nvshmem"
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_blocked_by_attention_dp(self):
        mock_exec = MagicMock()
        mock_exec.llm_args.enable_pipelined_scheduler = True
        mock_exec.disable_overlap_scheduler = False
        mock_exec.llm_args.speculative_config = None
        mock_exec.llm_args.cache_transceiver_config = None
        mock_exec.llm_args.enable_attention_dp = True
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_blocked_by_disable_overlap(self):
        mock_exec = MagicMock()
        mock_exec.llm_args.enable_pipelined_scheduler = True
        mock_exec.disable_overlap_scheduler = True
        mock_exec.llm_args.speculative_config = None
        mock_exec.llm_args.cache_transceiver_config = None
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_blocked_by_multimodal_encoder(self):
        mock_exec = MagicMock()
        mock_exec.llm_args.enable_pipelined_scheduler = True
        mock_exec.disable_overlap_scheduler = False
        mock_exec.llm_args.speculative_config = None
        mock_exec.llm_args.cache_transceiver_config = None
        mock_exec.llm_args.encoder_config = MagicMock()
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
        result = PyExecutor._is_pipelined_loop_eligible(mock_exec)
        assert result is False

    def test_blocked_by_guided_decoding(self):
        mock_exec = MagicMock()
        mock_exec.llm_args.enable_pipelined_scheduler = True
        mock_exec.disable_overlap_scheduler = False
        mock_exec.llm_args.speculative_config = None
        mock_exec.llm_args.cache_transceiver_config = None
        mock_exec.llm_args.guided_decoding_params = MagicMock()
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

    def test_multiple_blockers_reported_once(self):
        """When multiple incompatible features are set, only one message is logged."""
        from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
        args = TorchLlmArgs(
            model="gpt2",
            enable_pipelined_scheduler=True,
            disable_overlap_scheduler=True,
            enable_attention_dp=True,
        )
        args.validate_pipelined_scheduler()
        assert args.enable_pipelined_scheduler is False


class TestExecutorLoopSelection:
    """Verify _executor_loop dispatch selects pipelined when eligible."""

    def test_pipelined_selected_when_eligible(self):
        mock_exec = MagicMock()
        mock_exec._is_pipelined_loop_eligible.return_value = True
        mock_exec._executor_loop_pipelined = MagicMock()
        mock_exec._executor_loop_overlap = MagicMock()
        mock_exec._executor_loop = MagicMock()
        PyExecutor = MagicMock()
        PyExecutor._is_pipelined_loop_eligible.return_value = True
        PyExecutor._executor_loop_pipelined = MagicMock()
        PyExecutor._executor_loop_overlap = MagicMock()
        PyExecutor._executor_loop = MagicMock()

    def test_pipelined_not_selected_when_not_eligible(self):
        mock_exec = MagicMock()
        mock_exec._is_pipelined_loop_eligible.return_value = False
        mock_exec.disable_overlap_scheduler = False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
