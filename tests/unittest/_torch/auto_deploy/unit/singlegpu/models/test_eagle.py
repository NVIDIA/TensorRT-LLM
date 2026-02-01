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

import pytest
import torch
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    Eagle3DrafterForCausalLM,
    Eagle3DraftOutput,
    EagleConfig,
)
from tensorrt_llm._torch.auto_deploy.models.eagle import EagleDrafterFactory
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tests.test_common.llm_data import hf_id_to_local_model_dir

EAGLE_MODEL_HUB_ID = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

###############################################################################
# Mock classes for standalone Eagle testing
#
# These classes enable unit testing the Eagle checkpoint without a target model.
# In production speculative decoding, real hidden states come from the target model.
# For testing, MockEagle3ModelForCausalLM generates random hidden states.
###############################################################################


class MockEagleConfig(EagleConfig):
    """Config for standalone Eagle testing with embedding/lm_head loaded from checkpoint.

    In production, Eagle shares embedding/lm_head with the target model.
    For standalone testing, we need to load these from the checkpoint.
    """

    _drafter_defaults = {
        "llama": {
            "load_embedding_from_target": False,
            "load_lm_head_from_target": False,
            "num_capture_layers": 1,
        },
    }


class MockEagle3ModelForCausalLM(Eagle3DrafterForCausalLM):
    """Test wrapper that provides random hidden states for standalone Eagle testing.

    In production speculative decoding, real hidden states come from the target model.
    This mock class generates random hidden states for testing the Eagle model in isolation.
    """

    def __init__(self, config):
        super().__init__(config)
        self._hidden_size = config.hidden_size
        self._dtype = config.dtype

    def forward(self, input_ids, position_ids, input_embeds=None, **kwargs):
        assert self.model.embed_tokens is not None, (
            "embed_tokens must be set before running standalone Eagle model."
        )
        assert self.lm_head is not None, (
            "lm_head must be set before running standalone Eagle model."
        )

        if input_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # Inject mock hidden states if not provided
        if "hidden_states" not in kwargs:
            batch_size, seq_len = input_ids.shape
            kwargs["hidden_states"] = torch.randn(
                (batch_size, seq_len, self._hidden_size),
                dtype=self._dtype,
                device=input_ids.device,
            )
        draft_output = super().forward(inputs_embeds, position_ids, **kwargs)
        logits = self.lm_head(draft_output.norm_hidden_state)
        return Eagle3DraftOutput(logits=logits, last_hidden_state=draft_output.last_hidden_state)


class MockEagleDrafterFactory(EagleDrafterFactory):
    """Test factory that uses MockEagle3ModelForCausalLM for standalone Eagle testing.

    This factory directly builds MockEagle3ModelForCausalLM with MockEagleConfig,
    which loads embedding/lm_head from checkpoint for standalone testing.
    """

    def _build_model(self, device):
        from contextlib import nullcontext

        from accelerate import init_empty_weights

        model_config, unused_kwargs = self._get_model_config()
        model_config = MockEagleConfig(model_config, model_config.model_type)

        with (init_empty_weights if device == "meta" else nullcontext)():
            model = MockEagle3ModelForCausalLM._from_config(model_config, **unused_kwargs)

        if device == "meta":
            if hasattr(model, "post_init"):
                model.post_init()
        else:
            model.to(device)

        self._checkpoint_conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
        model.eval()

        return model


@pytest.fixture
def register_mock_eagle_factory():
    """Register MockEagleDrafterFactory for the test and clean up afterwards.

    This fixture temporarily registers the mock factory with ModelFactoryRegistry,
    allowing tests to use model_factory="MockEagleDrafter", and removes the
    registration after the test completes.
    """
    ModelFactoryRegistry._registry["MockEagleDrafter"] = MockEagleDrafterFactory
    yield
    ModelFactoryRegistry._registry.pop("MockEagleDrafter", None)


def test_build_ad_eagle(register_mock_eagle_factory):
    """Test building Eagle model with AutoDeploy using MockEagleDrafterFactory.

    This test uses the MockEagleDrafterFactory which builds MockEagle3ModelForCausalLM,
    a mock model that generates random hidden states for standalone Eagle testing.
    """
    llm_extra_args = {
        "model_factory": "MockEagleDrafter",
        "transforms": {
            "insert_cached_attention": {"backend": "flashinfer"},
            "compile_model": {"backend": "torch-compile"},
        },
    }
    experiment_config = get_small_model_config(EAGLE_MODEL_HUB_ID, **llm_extra_args)
    experiment_config["args"]["runtime"] = "demollm"
    experiment_config["args"]["world_size"] = 0
    experiment_config["args"]["tokenizer"] = hf_id_to_local_model_dir(
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    print(f"Experiment Config: {experiment_config}")
    experiment_config = ExperimentConfig(**experiment_config)

    main(experiment_config)


def test_eagle_model_torch_export():
    """Test that Eagle3Model can be exported with torch.export.

    This validates that the model architecture is compatible with
    torch.export for potential TensorRT compilation.

    Note: We skip loading weights since torch.export only traces the computation
    graph (model architecture).
    """
    print("\n" + "=" * 80)
    print("Test: EagleModel torch.export")
    print("=" * 80)

    eagle_model_path = hf_id_to_local_model_dir(EAGLE_MODEL_HUB_ID)
    if eagle_model_path is None:
        pytest.skip("Eagle model not found (LLM_MODELS_ROOT not set or model missing)")

    eagle_path = Path(eagle_model_path)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # Create model via EagleDrafterFactory (creates Eagle3DrafterForCausalLM)
    factory = EagleDrafterFactory(model=str(eagle_path), skip_loading_weights=True)
    model = factory.build_model(device)
    config = model.config

    # Create inputs for export
    batch_size = 1
    seq_len = 8
    hidden_dim = config.hidden_size

    inputs_embeds = torch.randn((batch_size, seq_len, hidden_dim), device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    mock_hidden_states = torch.randn((batch_size, seq_len, hidden_dim), device=device, dtype=dtype)

    print("Export input shapes:")
    print(f"  inputs_embeds: {inputs_embeds.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  hidden_states: {mock_hidden_states.shape}")

    example_args = (
        inputs_embeds,
        position_ids,
    )

    # Attempt torch.export
    try:
        exported_program = torch.export.export(
            model, args=example_args, kwargs={"hidden_states": mock_hidden_states}
        )
        print("✅ torch.export successful!")
        print("Graph module code preview (first 20 lines):")
        code_lines = exported_program.graph_module.code.split("\n")[:20]
        print("\n".join(code_lines))
    except Exception as e:
        pytest.fail(f"torch.export failed: {e}")
