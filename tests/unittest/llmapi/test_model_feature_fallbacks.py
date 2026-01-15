#!/usr/bin/env python3
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
"""Tests for model feature fallback mechanism.

These tests verify that _apply_model_feature_fallbacks() correctly
auto-disables unsupported features based on model architecture.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.llmapi.llm import BaseLLM


def _make_mock_llm(
    architecture: str | None = None,
    kv_cache_reuse: bool | None = None,
    chunked_prefill: bool = False,
    attention_dp: bool = False,
    disable_overlap_scheduler: bool | None = None,
    auto_disable: bool = True,
) -> SimpleNamespace:
    """Create a mock LLM-like object for testing.

    Args:
        architecture: Model architecture string, or None for missing config
        kv_cache_reuse: enable_block_reuse value, None means no kv_cache_config
        chunked_prefill: enable_chunked_prefill value
        attention_dp: enable_attention_dp value
        disable_overlap_scheduler: None means attribute doesn't exist
        auto_disable: auto_disable_unsupported_features value (defaults to True
            for testing purposes, though production default is False)
    """
    mock = SimpleNamespace()

    # Setup HF model config
    if architecture is not None:
        mock._hf_model_config = SimpleNamespace(architectures=[architecture])
    else:
        mock._hf_model_config = None

    # Setup args
    mock.args = SimpleNamespace()
    mock.args.auto_disable_unsupported_features = auto_disable
    mock.args.enable_chunked_prefill = chunked_prefill
    mock.args.enable_attention_dp = attention_dp

    if disable_overlap_scheduler is not None:
        mock.args.disable_overlap_scheduler = disable_overlap_scheduler

    if kv_cache_reuse is not None:
        mock.args.kv_cache_config = SimpleNamespace(enable_block_reuse=kv_cache_reuse)
    else:
        mock.args.kv_cache_config = None

    return mock


class TestFeatureFallbacksDisable:
    """Tests where features SHOULD be auto-disabled (NO status in matrix)."""

    def test_kv_cache_reuse_disabled_for_qwen3next(self):
        """Qwen3NextForCausalLM has KV_CACHE_REUSE=NO, should disable."""
        mock = _make_mock_llm(
            architecture="Qwen3NextForCausalLM",
            kv_cache_reuse=True,
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        # Feature should be disabled
        assert mock.args.kv_cache_config.enable_block_reuse is False

    def test_chunked_prefill_disabled_for_gptoss(self):
        """GptOssForCausalLM has CHUNKED_PREFILL=NO, should disable."""
        mock = _make_mock_llm(
            architecture="GptOssForCausalLM",
            chunked_prefill=True,
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        assert mock.args.enable_chunked_prefill is False

    def test_attention_dp_disabled_for_qwen3next(self):
        """Qwen3NextForCausalLM has ATTENTION_DP=NO, should disable."""
        mock = _make_mock_llm(
            architecture="Qwen3NextForCausalLM",
            attention_dp=True,
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        assert mock.args.enable_attention_dp is False

    def test_kv_cache_reuse_disabled_for_gptoss(self):
        """GptOssForCausalLM has KV_CACHE_REUSE=NO, should disable."""
        mock = _make_mock_llm(
            architecture="GptOssForCausalLM",
            kv_cache_reuse=True,
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        assert mock.args.kv_cache_config.enable_block_reuse is False


class TestFeatureFallbacksNoChange:
    """Tests where features should NOT be changed (edge cases)."""

    def test_no_change_when_feature_not_enabled(self):
        """No change when user didn't enable the unsupported feature."""
        mock = _make_mock_llm(
            architecture="Qwen3NextForCausalLM",
            kv_cache_reuse=False,  # User didn't enable it
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        # No change
        assert mock.args.kv_cache_config.enable_block_reuse is False

    def test_no_change_for_supported_feature(self):
        """DeepseekV3 has CHUNKED_PREFILL=YES, should not disable."""
        mock = _make_mock_llm(
            architecture="DeepseekV3ForCausalLM",
            chunked_prefill=True,
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        # Feature stays enabled
        assert mock.args.enable_chunked_prefill is True

    def test_no_change_for_unknown_architecture(self):
        """Unknown architectures should not trigger any fallback."""
        mock = _make_mock_llm(
            architecture="UnknownModelForCausalLM",
            kv_cache_reuse=True,
            chunked_prefill=True,
            attention_dp=True,
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        # All features stay enabled
        assert mock.args.kv_cache_config.enable_block_reuse is True
        assert mock.args.enable_chunked_prefill is True
        assert mock.args.enable_attention_dp is True

    def test_no_change_when_config_missing(self):
        """Gracefully handle missing HF config."""
        mock = _make_mock_llm(
            architecture=None,  # No config
            kv_cache_reuse=True,
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        # Feature stays enabled
        assert mock.args.kv_cache_config.enable_block_reuse is True

    def test_no_change_when_auto_disable_off(self):
        """Features should not be touched when auto_disable is False (the default)."""
        mock = _make_mock_llm(
            architecture="Qwen3NextForCausalLM",
            kv_cache_reuse=True,
            auto_disable=False,  # Default behavior - feature is opt-in
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        # Feature stays enabled despite being unsupported
        assert mock.args.kv_cache_config.enable_block_reuse is True

    def test_overlap_scheduler_stays_enabled_for_supported(self):
        """Qwen3Next has OVERLAP_SCHEDULER=YES, should not disable."""
        mock = _make_mock_llm(
            architecture="Qwen3NextForCausalLM",
            disable_overlap_scheduler=False,  # Scheduler is enabled
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        # Scheduler stays enabled (disable_overlap_scheduler stays False)
        assert mock.args.disable_overlap_scheduler is False


class TestMultipleFeatureFallbacks:
    """Test multiple features being disabled in one call."""

    def test_multiple_features_disabled_together(self):
        """Multiple unsupported features should all be disabled."""
        mock = _make_mock_llm(
            architecture="Qwen3NextForCausalLM",
            kv_cache_reuse=True,
            attention_dp=True,
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        # Both features disabled
        assert mock.args.kv_cache_config.enable_block_reuse is False
        assert mock.args.enable_attention_dp is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
