"""Unit tests for create_autodeploy_executor function."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import create_autodeploy_executor


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
    max_draft_len: int
    max_total_draft_tokens: int
    max_beam_width: int
    guided_decoder: Any


@pytest.mark.parametrize("guided_decoding_backend", ["xgrammar", "llguidance"])
@pytest.mark.parametrize("max_batch_size", [4, 8])
@pytest.mark.parametrize("vocab_size_padded", [42, 1000])
def test_create_autodeploy_executor_with_guided_decoding(
    guided_decoding_backend, max_batch_size, vocab_size_padded
):
    """Test create_autodeploy_executor with guided_decoding_backend."""
    mock_tokenizer = MockTokenizer

    ad_config = LlmArgs(
        model="test-model",
        max_batch_size=max_batch_size,
        max_seq_len=128,
        max_input_len=64,
        guided_decoding_backend=guided_decoding_backend,
        backend="_autodeploy",
    )

    # Mock guided decoding config
    mock_guided_decoding_config = MockGuidedDecodingConfig(
        guided_decoding_backend=guided_decoding_backend, tokenizer=mock_tokenizer
    )

    # Mock the engine attributes that are actually used by create_autodeploy_executor
    mock_engine = Mock()
    mock_engine.vocab_size_padded = vocab_size_padded
    mock_engine.cache_seq_interface.info.num_pages = (
        100  # placeholder to satisfy ADEngine.build_from_config
    )
    mock_engine.cache_seq_interface.info.max_num_tokens = (
        512  # placeholder to satisfy ADEngine.build_from_config
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
        # vocab_size_padded should be set from the engine instance created during execution
        assert guided_decoder.vocab_size_padded == mock_engine.vocab_size_padded
