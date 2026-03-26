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

"""Unit tests for Eagle3 model with AutoDeploy."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    Eagle3DraftOutput,
    EagleConfig,
    EagleDrafterForCausalLM,
    EagleWrapper,
    EagleWrapperConfig,
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


class MockEagle3ModelForCausalLM(EagleDrafterForCausalLM):
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


class _DummyTargetModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)
        self.output = nn.Linear(hidden_size, 32, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output


class _DummyDraftModel(nn.Module):
    def __init__(self, hidden_size: int, num_capture_layers: int, dtype: torch.dtype):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_capture_layers=num_capture_layers,
        )
        self.model = SimpleNamespace(dtype=dtype, fc=None, d2t=None)


class _FakeCSI:
    def __init__(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        hidden_size: int,
        num_capture_layers: int,
        ids_dtype: torch.dtype = torch.int64,
        hidden_states_dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        self.info = SimpleNamespace(
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            device=device,
        )
        self.named_args = {
            "input_ids": torch.zeros(max_num_tokens, dtype=ids_dtype, device=device),
        }
        for layer_idx in range(num_capture_layers):
            self.named_args[f"r{layer_idx}_hidden_states_cache"] = torch.zeros(
                max_num_tokens,
                hidden_size,
                dtype=hidden_states_dtype,
                device=device,
            )

    def get_arg(self, name: str):
        return self.named_args[name]


def _build_test_wrapper(hidden_size: int = 8, num_capture_layers: int = 2) -> EagleWrapper:
    return EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=3,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
        ),
        target_model=_DummyTargetModel(hidden_size),
        draft_model=_DummyDraftModel(hidden_size, num_capture_layers, torch.float16),
    )


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
    # Eagle one-model speculative decoding is only supported with TRTLLM attention.
    attn_backend = "trtllm"
    llm_extra_args = {
        "model_factory": "MockEagleDrafter",
        "transforms": {
            "insert_cached_attention": {"backend": attn_backend},
            "compile_model": {"backend": "torch-simple"},
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


def test_eagle_wrapper_ensure_buffers_lazy_init():
    wrapper = _build_test_wrapper()
    csi = _FakeCSI(max_batch_size=4, max_num_tokens=12, hidden_size=8, num_capture_layers=2)

    wrapper._ensure_buffers(csi)

    assert wrapper._buffers_initialized is True
    assert wrapper._buf_new_tokens_2d is not None
    assert wrapper._buf_next_new_tokens is not None
    assert wrapper._buf_new_tokens_lens is not None
    assert wrapper._buf_c_offset_ones is not None
    assert wrapper._buf_hidden_states is not None
    assert wrapper._buf_new_tokens_2d.shape == (4, 4)
    assert wrapper._buf_next_new_tokens.shape == (4, 4)
    assert wrapper._buf_new_tokens_lens.shape == (4,)
    assert wrapper._buf_c_offset_ones.shape == (4,)
    assert wrapper._buf_hidden_states.shape == (12, 16)

    buffer_ptrs = {
        "new_tokens_2d": wrapper._buf_new_tokens_2d.data_ptr(),
        "next_new_tokens": wrapper._buf_next_new_tokens.data_ptr(),
        "new_tokens_lens": wrapper._buf_new_tokens_lens.data_ptr(),
        "c_offset_ones": wrapper._buf_c_offset_ones.data_ptr(),
        "hidden_states": wrapper._buf_hidden_states.data_ptr(),
    }

    wrapper._ensure_buffers(csi)

    assert buffer_ptrs["new_tokens_2d"] == wrapper._buf_new_tokens_2d.data_ptr()
    assert buffer_ptrs["next_new_tokens"] == wrapper._buf_next_new_tokens.data_ptr()
    assert buffer_ptrs["new_tokens_lens"] == wrapper._buf_new_tokens_lens.data_ptr()
    assert buffer_ptrs["c_offset_ones"] == wrapper._buf_c_offset_ones.data_ptr()
    assert buffer_ptrs["hidden_states"] == wrapper._buf_hidden_states.data_ptr()


def test_eagle_wrapper_collect_hidden_states_reuses_preallocated_buffer():
    wrapper = _build_test_wrapper()
    csi = _FakeCSI(max_batch_size=4, max_num_tokens=12, hidden_size=8, num_capture_layers=2)
    wrapper._ensure_buffers(csi)

    first_buffers = {
        "r0_hidden_states_cache": torch.arange(96, dtype=torch.float16).view(12, 8),
        "r1_hidden_states_cache": torch.arange(96, 192, dtype=torch.float16).view(12, 8),
    }
    first_hidden_states = wrapper._collect_hidden_states(first_buffers, num_tokens=3)

    assert wrapper._buf_hidden_states is not None
    assert first_hidden_states.data_ptr() == wrapper._buf_hidden_states.data_ptr()
    assert torch.equal(
        first_hidden_states,
        torch.cat(
            [
                first_buffers["r0_hidden_states_cache"][:3],
                first_buffers["r1_hidden_states_cache"][:3],
            ],
            dim=1,
        ),
    )

    second_buffers = {
        "r0_hidden_states_cache": torch.full((12, 8), 7, dtype=torch.float16),
        "r1_hidden_states_cache": torch.full((12, 8), 11, dtype=torch.float16),
    }
    second_hidden_states = wrapper._collect_hidden_states(second_buffers, num_tokens=2)

    assert second_hidden_states.data_ptr() == wrapper._buf_hidden_states.data_ptr()
    assert torch.equal(
        second_hidden_states,
        torch.cat(
            [
                second_buffers["r0_hidden_states_cache"][:2],
                second_buffers["r1_hidden_states_cache"][:2],
            ],
            dim=1,
        ),
    )


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

    # Create model via EagleDrafterFactory (creates EagleDrafterForCausalLM)
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
