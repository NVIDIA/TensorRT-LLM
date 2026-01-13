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

"""Unit tests for Eagle3 model with AutoDeploy."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from _model_test_utils import get_model_path, get_small_model_config
from build_and_run_ad import ExperimentConfig, main
from transformers import AutoConfig

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    Eagle3DrafterForCausalLM,
    Eagle3LlamaConfig,
    Eagle3Model,
)
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

EAGLE_MODEL_HUB_ID = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
EAGLE_MODEL_SUBPATH = "EAGLE3-LLaMA3.1-Instruct-8B"


###############################################################################
# Mock classes for standalone Eagle testing
#
# These classes enable unit testing the Eagle checkpoint without a target model.
# In production speculative decoding, real hidden states come from the target model.
# For testing, MockEagle3ModelForCausalLM generates random hidden states.
###############################################################################


class MockEagle3Config(Eagle3LlamaConfig):
    """Test config for standalone Eagle testing with mock hidden states.

    Uses a distinct model_type to avoid conflicting with production Eagle3Config.
    """

    model_type = "mock_eagle3"


class MockEagle3ModelForCausalLM(Eagle3DrafterForCausalLM):
    """Test wrapper that provides random hidden states for standalone Eagle testing.

    In production speculative decoding, real hidden states come from the target model.
    This mock class generates random hidden states for testing the Eagle model in isolation.
    """

    config_class = MockEagle3Config

    def __init__(self, config):
        super().__init__(config)
        self._hidden_size = config.hidden_size
        self._dtype = config.dtype

    def forward(self, input_ids, **kwargs):
        # Inject mock hidden states if not provided
        if "hidden_states" not in kwargs:
            batch_size, seq_len = input_ids.shape
            kwargs["hidden_states"] = torch.randn(
                (batch_size, seq_len, self._hidden_size),
                dtype=self._dtype,
                device=input_ids.device,
            )
        return super().forward(input_ids, **kwargs)


@pytest.fixture
def use_mock_eagle3():
    """Mock the factory to use MockEagle3 classes without global registration.

    This fixture patches AutoModelForCausalLMFactory to recognize mock_eagle3 model_type
    by mocking two key methods:
    1. _override_model_type: Returns MockEagle3Config
    2. _custom_model_mapping: Includes MockEagle3ModelForCausalLM for "MockEagle3Config"

    Usage:
        def test_eagle_model(use_mock_eagle3):
            # ... test code where the model is hardcoded to use MockEagle3ModelForCausalLM
    """

    def patched_override_model_type(self, model_config, model_kwargs):
        return MockEagle3Config.from_dict(model_config.to_dict())

    # Create a copy of the existing mapping and add our mock model
    patched_mapping = AutoModelForCausalLMFactory._custom_model_mapping.copy()
    patched_mapping["MockEagle3Config"] = MockEagle3ModelForCausalLM

    with patch.object(
        AutoModelForCausalLMFactory, "_override_model_type", patched_override_model_type
    ):
        with patch.object(AutoModelForCausalLMFactory, "_custom_model_mapping", patched_mapping):
            yield  # Keep patches active during test


def test_build_ad_eagle(use_mock_eagle3):
    """Test building Eagle model with AutoDeploy using the mock fixture.

    This test uses the use_mock_eagle3 fixture which mocks the factory
    to recognize MockEagle3ModelForCausalLM.
    """
    llm_extra_args = {
        "transforms": {
            "insert_cached_attention": {"backend": "flashinfer"},
            "compile_model": {"backend": "torch-compile"},
        },
    }
    experiment_config = get_small_model_config(EAGLE_MODEL_HUB_ID, **llm_extra_args)
    experiment_config["args"]["runtime"] = "demollm"
    experiment_config["args"]["world_size"] = 0
    experiment_config["args"]["tokenizer"] = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    print(f"Experiment Config: {experiment_config}")
    experiment_config = ExperimentConfig(**experiment_config)

    main(experiment_config)


def test_eagle_model_torch_export():
    """Test that Eagle3Model can be exported with torch.export.

    This validates that the model architecture is compatible with
    torch.export for potential TensorRT compilation.

    Note: We skip loading weights since torch.export only traces the computation
    graph (model architecture), not the actual weight values. Random init is fine.
    """
    print("\n" + "=" * 80)
    print("Test: EagleModel torch.export")
    print("=" * 80)

    eagle_model_path = get_model_path(EAGLE_MODEL_SUBPATH)
    if eagle_model_path is None:
        pytest.skip("Eagle model not found (LLM_MODELS_ROOT not set or model missing)")

    eagle_path = Path(eagle_model_path)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    config_path = eagle_path / "config.json"
    config = AutoConfig.from_pretrained(config_path)

    # Create model with random weights (no need to load for export test)
    model = Eagle3Model(config)
    model.to(device)
    model.eval()

    # Create inputs for export
    batch_size = 1
    seq_len = 8
    hidden_dim = config.hidden_size

    input_ids = torch.randint(
        0, model._original_vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
    )
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    mock_hidden_states = torch.randn((batch_size, seq_len, hidden_dim), device=device, dtype=dtype)

    print("Export input shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  hidden_states: {mock_hidden_states.shape}")

    example_args = (
        input_ids,
        position_ids,
        mock_hidden_states,
    )

    # Attempt torch.export
    try:
        exported_program = torch.export.export(model, args=example_args)
        print("✅ torch.export successful!")
        print("Graph module code preview (first 20 lines):")
        code_lines = exported_program.graph_module.code.split("\n")[:20]
        print("\n".join(code_lines))
    except Exception as e:
        pytest.fail(f"torch.export failed: {e}")
