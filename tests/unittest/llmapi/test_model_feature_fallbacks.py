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
auto-disables unsupported features based on the support matrix.

Tests are designed to be matrix-agnostic - they test the fallback
mechanism itself, not specific model/feature combinations.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.models.feature_types import Feature, SupportStatus
from tensorrt_llm.llmapi.llm import BaseLLM

# Import doc-specific matrices from docs/source/helper.py
# Add docs/source to path temporarily for import
_docs_source = Path(__file__).parent.parent.parent.parent / "docs" / "source"
sys.path.insert(0, str(_docs_source))
from helper import KEY_MODEL_MATRIX, MULTIMODAL_MATRIX  # noqa: E402

sys.path.remove(str(_docs_source))


def _make_mock_llm(
    architecture: str | None = None,
    kv_cache_reuse: bool | None = None,
    chunked_prefill: bool = False,
    attention_dp: bool = False,
    disable_overlap_scheduler: bool | None = None,
    cuda_graph_config: object | None = None,
    guided_decoding_backend: str | None = None,
    auto_disable: bool = True,
) -> SimpleNamespace:
    """Create a mock LLM-like object for testing."""
    mock = SimpleNamespace()

    if architecture is not None:
        mock._hf_model_config = SimpleNamespace(architectures=[architecture])
    else:
        mock._hf_model_config = None

    mock.args = SimpleNamespace()
    mock.args.auto_disable_unsupported_features = auto_disable
    mock.args.enable_chunked_prefill = chunked_prefill
    mock.args.enable_attention_dp = attention_dp
    mock.args.cuda_graph_config = cuda_graph_config
    mock.args.guided_decoding_backend = guided_decoding_backend

    if disable_overlap_scheduler is not None:
        mock.args.disable_overlap_scheduler = disable_overlap_scheduler

    if kv_cache_reuse is not None:
        mock.args.kv_cache_config = SimpleNamespace(enable_block_reuse=kv_cache_reuse)
    else:
        mock.args.kv_cache_config = None

    return mock


class TestFallbackMechanism:
    """Test the fallback mechanism behavior (matrix-agnostic)."""

    def test_no_change_when_auto_disable_off(self):
        """Features should not be touched when auto_disable is False (default)."""
        mock = _make_mock_llm(
            architecture="AnyModel",
            kv_cache_reuse=True,
            chunked_prefill=True,
            attention_dp=True,
            auto_disable=False,
        )

        # Mock all features as unsupported
        with patch("tensorrt_llm.llmapi.llm.is_feature_unsupported", return_value=True):
            BaseLLM._apply_model_feature_fallbacks(mock)

        # Nothing should change when auto_disable is off
        assert mock.args.kv_cache_config.enable_block_reuse is True
        assert mock.args.enable_chunked_prefill is True
        assert mock.args.enable_attention_dp is True

    def test_no_change_for_unknown_architecture(self):
        """Unknown architectures should not trigger any fallback."""
        mock = _make_mock_llm(
            architecture="UnknownModelForCausalLM",
            kv_cache_reuse=True,
            chunked_prefill=True,
            attention_dp=True,
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        # All features stay enabled (unknown arch returns None from get_status)
        assert mock.args.kv_cache_config.enable_block_reuse is True
        assert mock.args.enable_chunked_prefill is True
        assert mock.args.enable_attention_dp is True

    def test_no_change_when_config_missing(self):
        """Gracefully handle missing HF config."""
        mock = _make_mock_llm(
            architecture=None,
            kv_cache_reuse=True,
        )

        BaseLLM._apply_model_feature_fallbacks(mock)

        assert mock.args.kv_cache_config.enable_block_reuse is True

    def test_feature_disabled_when_unsupported(self):
        """Feature should be disabled when matrix returns unsupported."""
        mock = _make_mock_llm(
            architecture="MockModel",
            chunked_prefill=True,
        )

        with patch("tensorrt_llm.llmapi.llm.is_feature_unsupported", return_value=True):
            BaseLLM._apply_model_feature_fallbacks(mock)

        assert mock.args.enable_chunked_prefill is False

    def test_feature_stays_enabled_when_supported(self):
        """Feature should stay enabled when matrix returns supported."""
        mock = _make_mock_llm(
            architecture="MockModel",
            chunked_prefill=True,
        )

        with patch("tensorrt_llm.llmapi.llm.is_feature_unsupported", return_value=False):
            BaseLLM._apply_model_feature_fallbacks(mock)

        assert mock.args.enable_chunked_prefill is True

    def test_no_change_when_feature_already_disabled(self):
        """No change needed when feature is already disabled."""
        mock = _make_mock_llm(
            architecture="MockModel",
            chunked_prefill=False,
        )

        with patch("tensorrt_llm.llmapi.llm.is_feature_unsupported", return_value=True):
            BaseLLM._apply_model_feature_fallbacks(mock)

        assert mock.args.enable_chunked_prefill is False


class TestFallbackWithRealMatrix:
    """Test fallbacks using real matrix data (dynamic discovery)."""

    @staticmethod
    def _find_model_with_status(feature: Feature, status: SupportStatus) -> str | None:
        """Find any model in the matrix with the given feature status."""
        for arch, features in KEY_MODEL_MATRIX.items():
            cell = features.get(feature)
            if cell and cell.status == status:
                return arch
        for arch, features in MULTIMODAL_MATRIX.items():
            cell = features.get(feature)
            if cell and cell.status == status:
                return arch
        return None

    def test_kv_cache_reuse_fallback_if_no_exists(self):
        """Test KV cache reuse fallback if any model has NO status."""
        arch = self._find_model_with_status(Feature.KV_CACHE_REUSE, SupportStatus.NO)
        if arch is None:
            pytest.skip("No model with KV_CACHE_REUSE=NO in matrix")

        mock = _make_mock_llm(architecture=arch, kv_cache_reuse=True)
        BaseLLM._apply_model_feature_fallbacks(mock)
        assert mock.args.kv_cache_config.enable_block_reuse is False

    def test_chunked_prefill_fallback_if_no_exists(self):
        """Test chunked prefill fallback if any model has NO status."""
        arch = self._find_model_with_status(Feature.CHUNKED_PREFILL, SupportStatus.NO)
        if arch is None:
            pytest.skip("No model with CHUNKED_PREFILL=NO in matrix")

        mock = _make_mock_llm(architecture=arch, chunked_prefill=True)
        BaseLLM._apply_model_feature_fallbacks(mock)
        assert mock.args.enable_chunked_prefill is False

    def test_attention_dp_fallback_if_no_exists(self):
        """Test attention DP fallback if any model has NO status."""
        arch = self._find_model_with_status(Feature.ATTENTION_DP, SupportStatus.NO)
        if arch is None:
            pytest.skip("No model with ATTENTION_DP=NO in matrix")

        mock = _make_mock_llm(architecture=arch, attention_dp=True)
        BaseLLM._apply_model_feature_fallbacks(mock)
        assert mock.args.enable_attention_dp is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
