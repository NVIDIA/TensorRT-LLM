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
import torch

from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import ADEngine, create_autodeploy_executor


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
    resource_governor_queue: Any = None
    garbage_collection_gen0_threshold: Optional[int] = None


@dataclass
class MockFactory:
    """Mock Factory that stores initialization arguments."""

    vocab_size_padded: Optional[int] = None


"""Unit tests for create_autodeploy_executor function."""


def test_build_from_config_creates_sa_manager_after_factory_resolution():
    from tensorrt_llm.llmapi import Eagle3DecodingConfig, SAEnhancerConfig

    events = []
    resolved_max_seq_len = 96
    sa_config = SAEnhancerConfig(threshold=5)
    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model="draft-model",
        sa_config=sa_config,
    )
    ad_config = LlmArgs(
        model="target-model",
        max_batch_size=2,
        max_seq_len=None,
        max_num_tokens=128,
        max_input_len=32,
        speculative_config=spec_config,
        backend="_autodeploy",
        device="cpu",
        cuda_graph_config={"max_batch_size": 2},
    )
    get_inference_model = Mock(return_value=Mock())
    sa_manager = Mock()

    def _create_factory():
        events.append("create_factory")
        ad_config.max_seq_len = resolved_max_seq_len
        return MockFactory(vocab_size_padded=1000)

    def _create_sa_manager(*args):
        events.append("create_sa_manager")
        return sa_manager

    def _ensure_workspace(max_draft_len):
        events.append("ensure_workspace")

    def _create_optimizer(*args, **kwargs):
        events.append("create_optimizer")
        return get_inference_model

    sa_manager._ensure_workspace.side_effect = _ensure_workspace

    # InferenceOptimizer needs the SA manager during CUDA graph capture.
    with (
        patch(
            "tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs.create_factory",
            side_effect=_create_factory,
        ),
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.SuffixAutomatonManager",
            side_effect=_create_sa_manager,
        ) as sa_manager_cls,
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.InferenceOptimizer",
            side_effect=_create_optimizer,
        ) as optimizer_cls,
    ):
        engine = ADEngine.build_from_config(ad_config)

    assert events == [
        "create_factory",
        "create_sa_manager",
        "ensure_workspace",
        "create_optimizer",
    ]
    sa_manager_cls.assert_called_once_with(
        sa_config, ad_config.max_batch_size, resolved_max_seq_len
    )
    sa_manager._ensure_workspace.assert_called_once_with(spec_config.max_draft_len)
    assert optimizer_cls.call_args.kwargs["sa_manager"] is sa_manager
    assert engine.sa_manager is sa_manager
    get_inference_model.assert_called_once_with(engine.cache_seq_interface)


def test_build_from_config_cleans_up_sa_manager_when_workspace_allocation_fails():
    from tensorrt_llm.llmapi import Eagle3DecodingConfig, SAEnhancerConfig

    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model="draft-model",
        sa_config=SAEnhancerConfig(threshold=5),
    )
    ad_config = LlmArgs(
        model="target-model",
        max_batch_size=2,
        max_seq_len=96,
        max_num_tokens=128,
        max_input_len=32,
        speculative_config=spec_config,
        backend="_autodeploy",
        device="cpu",
        cuda_graph_config={"max_batch_size": 2},
    )
    sa_manager = Mock()
    sa_manager._ensure_workspace.side_effect = torch.OutOfMemoryError("oom")

    with (
        patch(
            "tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs.create_factory",
            return_value=MockFactory(vocab_size_padded=1000),
        ),
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.SuffixAutomatonManager",
            return_value=sa_manager,
        ),
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.InferenceOptimizer",
        ) as optimizer_cls,
        pytest.raises(RuntimeError, match="Could not allocate necessary memory for the SAManager"),
    ):
        ADEngine.build_from_config(ad_config)

    sa_manager.shutdown.assert_called_once()
    optimizer_cls.assert_not_called()


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

    # Mock the engine attributes that are actually used by create_autodeploy_executor
    mock_engine = Mock()
    mock_engine.cache_seq_interface.info.num_pages = (
        100  # placeholder to satisfy ADEngine.build_from_config
    )
    mock_engine.cache_seq_interface.info.max_num_tokens = (
        512  # placeholder to satisfy ADEngine.build_from_config
    )
    mock_engine.cache_seq_interface.info.vocab_size_padded = vocab_size_padded
    mock_engine.cache_seq_interface.max_num_state_slots = max_batch_size
    mock_engine.sa_manager = None

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
        assert guided_decoder.vocab_size_padded == vocab_size_padded
        assert result.resource_governor_queue is None


def test_create_autodeploy_executor_registers_sa_resource_manager():
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
    from tensorrt_llm.llmapi import Eagle3DecodingConfig, SAEnhancerConfig

    sa_config = SAEnhancerConfig(threshold=6)
    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model="draft-model",
        sa_config=sa_config,
    )
    ad_config = LlmArgs(
        model="target-model",
        max_batch_size=4,
        max_seq_len=128,
        max_input_len=64,
        speculative_config=spec_config,
        backend="_autodeploy",
        cuda_graph_config={"max_batch_size": 4},
    )

    mock_engine = Mock()
    mock_engine.cache_seq_interface.info.max_num_tokens = 512
    mock_engine.cache_seq_interface.info.vocab_size_padded = 1000
    mock_engine.cache_seq_interface.max_num_state_slots = ad_config.max_batch_size
    mock_engine.cache_seq_interface.kv_cache_manager = Mock()
    mock_engine.cache_seq_interface.kv_cache_manager.impl = Mock()
    mock_engine.cache_seq_interface.kv_cache_config_tuned = SimpleNamespace(tokens_per_block=64)
    mock_sa_manager = Mock()
    mock_engine.sa_manager = mock_sa_manager

    with (
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.mpi_world_size", return_value=1),
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.mpi_rank", return_value=0),
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.get_free_port", return_value=12345),
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.initialize_or_skip"),
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.torch.cuda.set_device"),
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.ADEngine.build_from_config",
            return_value=mock_engine,
        ) as build_from_config_mock,
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.instantiate_sampler",
            return_value=Mock(),
        ),
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.PyExecutor",
            side_effect=MockPyExecutor,
        ),
    ):
        result = create_autodeploy_executor(ad_config)

    assert (
        result.resource_manager.get_resource_manager(ResourceManagerType.SPEC_RESOURCE_MANAGER)
        is mock_sa_manager
    )
    assert "sa_manager" not in build_from_config_mock.call_args.kwargs


def test_create_autodeploy_executor_skips_sa_resource_manager_without_sa_config():
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
    from tensorrt_llm.llmapi import Eagle3DecodingConfig

    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model="draft-model",
    )
    ad_config = LlmArgs(
        model="target-model",
        max_batch_size=4,
        max_seq_len=128,
        max_input_len=64,
        speculative_config=spec_config,
        backend="_autodeploy",
        cuda_graph_config={"max_batch_size": 4},
    )

    mock_engine = Mock()
    mock_engine.cache_seq_interface.info.max_num_tokens = 512
    mock_engine.cache_seq_interface.info.vocab_size_padded = 1000
    mock_engine.cache_seq_interface.max_num_state_slots = ad_config.max_batch_size
    mock_engine.cache_seq_interface.kv_cache_manager = Mock()
    mock_engine.cache_seq_interface.kv_cache_manager.impl = Mock()
    mock_engine.cache_seq_interface.kv_cache_config_tuned = SimpleNamespace(tokens_per_block=64)
    mock_engine.sa_manager = None

    with (
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.mpi_world_size", return_value=1),
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.mpi_rank", return_value=0),
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.get_free_port", return_value=12345),
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.initialize_or_skip"),
        patch("tensorrt_llm._torch.auto_deploy.shim.ad_executor.torch.cuda.set_device"),
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.ADEngine.build_from_config",
            return_value=mock_engine,
        ) as build_from_config_mock,
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.instantiate_sampler",
            return_value=Mock(),
        ),
        patch(
            "tensorrt_llm._torch.auto_deploy.shim.ad_executor.PyExecutor",
            side_effect=MockPyExecutor,
        ),
    ):
        result = create_autodeploy_executor(ad_config)

    assert "sa_manager" not in build_from_config_mock.call_args.kwargs
    assert (
        result.resource_manager.get_resource_manager(ResourceManagerType.SPEC_RESOURCE_MANAGER)
        is None
    )
