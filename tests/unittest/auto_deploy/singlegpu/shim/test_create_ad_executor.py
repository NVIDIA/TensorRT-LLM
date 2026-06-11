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

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionType
from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import create_autodeploy_executor
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import AttentionTypeCpp
from tensorrt_llm.llmapi import CacheTransceiverConfig


class MockTokenizer:
    """Simple mock tokenizer."""


# Note: These mock classes are a bit fragile if the signatures of these classes change.
@dataclass
class MockGuidedDecodingConfig:
    """Mock guided decoding config for testing."""

    guided_decoding_backend: Any
    tokenizer: MockTokenizer


@dataclass
class MockGuidedDecoder:
    """Mock GuidedDecoder that stores initialization arguments."""

    guided_decoding_config: Any
    max_num_sequences: int
    vocab_size_padded: int


@dataclass
class MockPyExecutor:
    """Mock PyExecutor that stores initialization arguments."""

    resource_manager: Any
    scheduler: Any
    model_engine: Any
    sampler: Any
    dist: Any
    max_num_sequences: int
    disable_overlap_scheduler: bool
    max_input_len: int
    max_batch_size: int
    max_beam_width: int
    guided_decoder: Any
    kv_cache_transceiver: Any = None
    resource_governor_queue: Any = None
    garbage_collection_gen0_threshold: Optional[int] = None


@dataclass
class MockFactory:
    """Mock Factory that stores initialization arguments."""

    vocab_size_padded: Optional[int] = None


"""Unit tests for create_autodeploy_executor function."""


def make_mock_engine(
    *,
    max_batch_size: int = 4,
    max_seq_len: int = 128,
    max_num_tokens: int = 512,
    vocab_size_padded: int = 1000,
    attention_type: Optional[AttentionType] = AttentionType.mha,
):
    kv_cache_manager = Mock()
    kv_cache_manager.impl = Mock()

    mock_engine = Mock()
    mock_engine.llm_args = SimpleNamespace(max_seq_len=max_seq_len)
    mock_engine.cache_seq_interface.info.num_pages = 100
    mock_engine.cache_seq_interface.info.max_seq_len = max_seq_len
    mock_engine.cache_seq_interface.info.max_num_tokens = max_num_tokens
    mock_engine.cache_seq_interface.info.vocab_size_padded = vocab_size_padded
    mock_engine.cache_seq_interface.max_num_state_slots = max_batch_size
    mock_engine.cache_seq_interface.attention_type = attention_type
    mock_engine.cache_seq_interface.kv_cache_manager = kv_cache_manager
    mock_engine.cache_seq_interface.kv_cache_config_tuned = Mock()
    return mock_engine, kv_cache_manager


@pytest.mark.parametrize("guided_decoding_backend", ["xgrammar", "llguidance"])
@pytest.mark.parametrize("max_batch_size", [4, 8])
@pytest.mark.parametrize("vocab_size_padded", [42, 1000])
def test_create_autodeploy_executor_with_guided_decoding(
    guided_decoding_backend, max_batch_size, vocab_size_padded
):
    """Test create_autodeploy_executor with guided_decoding_backend."""
    mock_tokenizer = MockTokenizer()

    ad_config = LlmArgs(
        model="test-model",
        max_batch_size=max_batch_size,
        max_seq_len=128,
        max_input_len=64,
        guided_decoding_backend=guided_decoding_backend,
        backend="_autodeploy",
        cuda_graph_config={"max_batch_size": max_batch_size},
    )

    # Mock guided decoding config
    mock_guided_decoding_config = MockGuidedDecodingConfig(
        guided_decoding_backend=guided_decoding_backend, tokenizer=mock_tokenizer
    )

    mock_engine, _ = make_mock_engine(
        max_batch_size=max_batch_size, vocab_size_padded=vocab_size_padded
    )

    # Mock the specific dependencies requested, plus minimal additional mocks to prevent errors
    with (
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.get_guided_decoding_config",
            return_value=mock_guided_decoding_config,
        ) as _,
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.GuidedDecoder"
        ) as guided_decoder_cls,
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.PyExecutor") as py_executor_cls,
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.ADEngine.build_from_config"
        ) as mock_ad_engine,
        patch(
            "tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs.create_factory",
            return_value=MockFactory(vocab_size_padded=vocab_size_padded),
        ),
    ):
        mock_ad_engine.return_value = mock_engine

        # substitute the GuidedDecoder and PyExecutor classes
        guided_decoder_cls.side_effect = MockGuidedDecoder
        py_executor_cls.side_effect = MockPyExecutor

        # Call the function under test
        result = create_autodeploy_executor(ad_config, mock_tokenizer)

        # Verify that GuidedDecoder was called
        guided_decoder_cls.assert_called_once()

        # Verify that PyExecutor was called
        py_executor_cls.assert_called_once()

        # Verify the return value is a MockPyExecutor with the expected guided_decoder
        assert isinstance(result, MockPyExecutor)
        assert hasattr(result, "guided_decoder")

        guided_decoder = result.guided_decoder
        assert isinstance(guided_decoder, MockGuidedDecoder)
        assert guided_decoder.guided_decoding_config == mock_guided_decoding_config
        assert guided_decoder.max_num_sequences == ad_config.max_batch_size
        assert guided_decoder.vocab_size_padded == vocab_size_padded
        assert result.resource_governor_queue is None


@pytest.mark.parametrize(
    "cache_attention_type, expected_attention_type",
    [
        (AttentionType.mha, AttentionTypeCpp.DEFAULT),
        (AttentionType.mla, AttentionTypeCpp.MLA),
    ],
)
def test_create_executor_uses_cache_transceiver(cache_attention_type, expected_attention_type):
    """Test create_autodeploy_executor passes the configured KV cache transceiver to PyExecutor."""
    mock_tokenizer = MockTokenizer()
    mock_transceiver = Mock()

    ad_config = LlmArgs(
        model="test-model",
        max_batch_size=4,
        max_seq_len=128,
        max_input_len=64,
        backend="_autodeploy",
        cuda_graph_config={"max_batch_size": 4},
        cache_transceiver_config=CacheTransceiverConfig(backend="DEFAULT"),
    )

    mock_engine, kv_cache_manager = make_mock_engine(attention_type=cache_attention_type)

    with (
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.PyExecutor") as py_executor_cls,
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.ADEngine.build_from_config"
        ) as mock_ad_engine,
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.create_kv_cache_transceiver",
            return_value=mock_transceiver,
        ) as create_transceiver,
        patch(
            "tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs.create_factory",
            return_value=MockFactory(vocab_size_padded=1000),
        ),
    ):
        mock_ad_engine.return_value = mock_engine
        py_executor_cls.side_effect = MockPyExecutor

        result = create_autodeploy_executor(ad_config, mock_tokenizer)

    create_transceiver.assert_called_once()
    _, _, passed_kv_cache_manager, attention_type, passed_config = create_transceiver.call_args.args
    assert passed_kv_cache_manager is kv_cache_manager
    assert attention_type == expected_attention_type
    assert passed_config is ad_config.cache_transceiver_config
    assert passed_config.max_tokens_in_buffer == mock_engine.cache_seq_interface.info.max_seq_len
    assert create_transceiver.call_args.kwargs["mamba_cache_manager"] is None
    assert result.kv_cache_transceiver is mock_transceiver


@pytest.mark.parametrize(
    "cache_attention_type, expected_attention_type",
    [
        (AttentionType.mha, AttentionTypeCpp.DEFAULT),
        (AttentionType.mla, AttentionTypeCpp.MLA),
    ],
)
def test_create_executor_preserves_explicit_transceiver_buffer_size(
    cache_attention_type, expected_attention_type
):
    """Test create_autodeploy_executor preserves explicit KV cache transceiver buffer sizing."""
    mock_tokenizer = MockTokenizer()
    mock_transceiver = Mock()

    ad_config = LlmArgs(
        model="test-model",
        max_batch_size=4,
        max_seq_len=128,
        max_input_len=64,
        backend="_autodeploy",
        cuda_graph_config={"max_batch_size": 4},
        cache_transceiver_config=CacheTransceiverConfig(
            backend="DEFAULT", max_tokens_in_buffer=1024
        ),
    )

    mock_engine, _ = make_mock_engine(attention_type=cache_attention_type)

    with (
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.PyExecutor") as py_executor_cls,
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.ADEngine.build_from_config"
        ) as mock_ad_engine,
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.create_kv_cache_transceiver",
            return_value=mock_transceiver,
        ) as create_transceiver,
        patch(
            "tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs.create_factory",
            return_value=MockFactory(vocab_size_padded=1000),
        ),
    ):
        mock_ad_engine.return_value = mock_engine
        py_executor_cls.side_effect = MockPyExecutor

        create_autodeploy_executor(ad_config, mock_tokenizer)

    _, _, _, attention_type, passed_config = create_transceiver.call_args.args
    assert attention_type == expected_attention_type
    assert passed_config.max_tokens_in_buffer == 1024
    assert create_transceiver.call_args.kwargs["mamba_cache_manager"] is None


@pytest.mark.parametrize("cache_attention_type", ["mha", "unsupported"])
def test_create_executor_rejects_non_enum_attention_type(cache_attention_type):
    """Test create_autodeploy_executor requires enum KV cache attention semantics."""
    mock_tokenizer = MockTokenizer()

    ad_config = LlmArgs(
        model="test-model",
        max_batch_size=4,
        max_seq_len=128,
        max_input_len=64,
        backend="_autodeploy",
        cuda_graph_config={"max_batch_size": 4},
        cache_transceiver_config=CacheTransceiverConfig(backend="DEFAULT"),
    )

    mock_engine, _ = make_mock_engine()
    mock_engine.cache_seq_interface.attention_type = cache_attention_type

    with (
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.ADEngine.build_from_config"
        ) as mock_ad_engine,
        patch(
            "tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs.create_factory",
            return_value=MockFactory(vocab_size_padded=1000),
        ),
    ):
        mock_ad_engine.return_value = mock_engine

        with pytest.raises(TypeError):
            create_autodeploy_executor(ad_config, mock_tokenizer)


def test_create_executor_requires_attention_type():
    """Test create_autodeploy_executor requires KV cache attention semantics for disagg."""
    mock_tokenizer = MockTokenizer()

    ad_config = LlmArgs(
        model="test-model",
        max_batch_size=4,
        max_seq_len=128,
        max_input_len=64,
        backend="_autodeploy",
        cuda_graph_config={"max_batch_size": 4},
        cache_transceiver_config=CacheTransceiverConfig(backend="DEFAULT"),
    )

    mock_engine, _ = make_mock_engine(attention_type=None)

    with (
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.ADEngine.build_from_config"
        ) as mock_ad_engine,
        patch(
            "tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs.create_factory",
            return_value=MockFactory(vocab_size_padded=1000),
        ),
    ):
        mock_ad_engine.return_value = mock_engine

        with pytest.raises(RuntimeError):
            create_autodeploy_executor(ad_config, mock_tokenizer)


def test_create_executor_rejects_mamba_cache_manager_for_transceiver():
    """Test create_autodeploy_executor rejects Mamba/hybrid cache transfer."""
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import BaseMambaCacheManager

    mock_tokenizer = MockTokenizer()

    ad_config = LlmArgs(
        model="test-model",
        max_batch_size=4,
        max_seq_len=128,
        max_input_len=64,
        backend="_autodeploy",
        cuda_graph_config={"max_batch_size": 4},
        cache_transceiver_config=CacheTransceiverConfig(backend="DEFAULT"),
    )

    mock_engine, _ = make_mock_engine(attention_type=AttentionType.mha)
    mamba_cache_manager = Mock(spec=BaseMambaCacheManager)
    mamba_cache_manager.impl = Mock()
    mock_engine.cache_seq_interface.kv_cache_manager = mamba_cache_manager

    with (
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.ADEngine.build_from_config"
        ) as mock_ad_engine,
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.create_kv_cache_transceiver"
        ) as create_transceiver,
        patch(
            "tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs.create_factory",
            return_value=MockFactory(vocab_size_padded=1000),
        ),
    ):
        mock_ad_engine.return_value = mock_engine

        with pytest.raises(RuntimeError):
            create_autodeploy_executor(ad_config, mock_tokenizer)

        create_transceiver.assert_not_called()
