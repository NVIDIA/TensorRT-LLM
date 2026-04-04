# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Standalone tests for the Mistral4 Eagle drafter head with AutoDeploy.

Two tests mirror the pattern from test_eagle.py (Llama Eagle):

1. test_mistral4_eagle_model_torch_export  — verify the Eagle drafter can be exported
   with torch.export (architecture traceability, no weights needed).

2. test_build_ad_mistral4_eagle            — verify the Eagle drafter runs through the
   full AutoDeploy compile/run pipeline using a mock that injects random hidden states,
   so the Eagle head can be validated in isolation without the target model.

Both tests require:
  - mistralai/Mistral-Small-4-119B-2603-eagle  (Eagle checkpoint, for weights/structure)
  - mistralai/Mistral-Small-4-119B-2603        (target model, for HF config)
resolved via hf_id_to_local_model_dir.  Tests are skipped automatically if either path
is unavailable.

The Mistral4 Eagle checkpoint is in native Mistral format (no config.json), so
EagleDrafterFactory must be initialised with config_model=<target_model_path> to
load the architecture config from the target model instead.
"""

from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from _model_test_utils import get_small_model_config
from accelerate import init_empty_weights
from build_and_run_ad import ExperimentConfig, main

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    Eagle3DraftOutput,
    EagleConfig,
    EagleDrafterForCausalLM,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_mistral3 import ADMistralSmall4Processor
from tensorrt_llm._torch.auto_deploy.models.eagle import EagleDrafterFactory
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tests.test_common.llm_data import hf_id_to_local_model_dir

EAGLE_MODEL_HUB_ID = "mistralai/Mistral-Small-4-119B-2603-eagle"
TARGET_MODEL_HUB_ID = "mistralai/Mistral-Small-4-119B-2603"


# ---------------------------------------------------------------------------
# Helpers to resolve and skip
# ---------------------------------------------------------------------------


def _require_paths():
    """Return (eagle_path, target_path) or skip the test if either is missing."""
    eagle_path = hf_id_to_local_model_dir(EAGLE_MODEL_HUB_ID)
    target_path = hf_id_to_local_model_dir(TARGET_MODEL_HUB_ID)
    if eagle_path is None or not Path(eagle_path).is_dir():
        pytest.skip(f"Eagle checkpoint not found: {EAGLE_MODEL_HUB_ID}")
    if target_path is None or not Path(target_path).is_dir():
        pytest.skip(f"Target model not found: {TARGET_MODEL_HUB_ID}")
    return eagle_path, target_path


# ---------------------------------------------------------------------------
# Mock classes for standalone Mistral4 Eagle testing
# ---------------------------------------------------------------------------


class MockMistral4EagleConfig(EagleConfig):
    """EagleConfig variant for standalone Eagle testing.

    Disables loading embedding/lm_head from target (since there is no target model
    in the standalone test) and forces random initialisation for these modules so
    the model can generate logits on its own.

    Sets torch_dtype explicitly to bfloat16 so all Eagle layers (Linear, Embedding)
    initialize in BF16, which is required for flashinfer kernels (rmsnorm, mla).
    The nested text_config extracted from Mistral3Config may not carry torch_dtype.
    """

    _drafter_defaults = {
        "mistral4": {
            "load_embedding_from_target": False,
            "load_lm_head_from_target": False,
            "num_capture_layers": 1,
            "normalize_target_hidden_state": False,
            "layers_handle_final_norm": False,
            # Ensure BF16 so flashinfer kernels can dispatch correctly.
            # Use string form to avoid omegaconf serialization issues with torch.dtype objects.
            "torch_dtype": "bfloat16",
            "_checkpoint_conversion_mapping": {
                r"^eagle_linear": "model.layers.0.eagle_proj",
                r"^layers": "model.layers",
            },
        }
    }


class MockMistral4EagleDrafterForCausalLM(EagleDrafterForCausalLM):
    """Eagle drafter that injects random hidden states for standalone testing.

    In production, hidden states come from the target model.  Here we generate
    them randomly so the Eagle head can be exercised without a target model.
    The forward signature is changed to accept input_ids (like a normal LM) and
    produce logits, making it compatible with build_and_run_ad / demollm.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._hidden_size = config.hidden_size
        self._dtype = getattr(config, "torch_dtype", torch.bfloat16)

    def forward(self, input_ids, position_ids, **kwargs):
        assert self.model.embed_tokens is not None, (
            "embed_tokens must be initialised for standalone Mistral4 Eagle testing."
        )
        assert self.lm_head is not None, (
            "lm_head must be initialised for standalone Mistral4 Eagle testing."
        )
        inputs_embeds = self.model.embed_tokens(input_ids)
        if "hidden_states" not in kwargs:
            batch_size, seq_len = input_ids.shape
            kwargs["hidden_states"] = torch.randn(
                (batch_size, seq_len, self._hidden_size),
                dtype=inputs_embeds.dtype,
                device=input_ids.device,
            )
        draft_output = super().forward(inputs_embeds, position_ids, **kwargs)
        logits = self.lm_head(draft_output.norm_hidden_state)
        return Eagle3DraftOutput(logits=logits, last_hidden_state=draft_output.last_hidden_state)


class MockMistral4EagleDrafterFactory(EagleDrafterFactory):
    """Factory that builds MockMistral4EagleDrafterForCausalLM for standalone testing.

    Passes config_model=<target_path> so that the architecture config is loaded from
    the HF-format target model rather than from the native-format Eagle checkpoint.
    """

    def _build_model(self, device):
        model_config, unused_kwargs = self._get_model_config()
        model_config = MockMistral4EagleConfig(model_config, model_config.model_type)

        with (init_empty_weights if device == "meta" else nullcontext)():
            model = MockMistral4EagleDrafterForCausalLM._from_config(model_config, **unused_kwargs)

        if device == "meta":
            if hasattr(model, "post_init"):
                model.post_init()
        else:
            # Cast to bfloat16 so that flashinfer kernels (rmsnorm, mla) can dispatch
            # correctly. Random-init weights default to float32 without this.
            model.to(device=device, dtype=torch.bfloat16)

        self._checkpoint_conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
        model.eval()
        return model

    def _load_quantization_config(self, fetched_dir):
        """Skip quant-config detection for standalone mock testing.

        The Eagle checkpoint's params.json declares fp8 quantization.  When running
        with skip_loading_weights=True (random BF16 init), applying FP8 graph
        transforms on top of BF16 weights causes runtime dtype mismatches in
        flashinfer kernels.  Disable quant config auto-detection for mock tests.
        """

    def init_tokenizer(self):
        """Load the Mistral4 tekken.json tokenizer via ADMistralSmall4Processor."""
        if self.tokenizer is None:
            return None
        processor = ADMistralSmall4Processor.from_pretrained(self.tokenizer)
        return processor.tokenizer


@pytest.fixture
def register_mock_mistral4_eagle_factory():
    """Temporarily register MockMistral4EagleDrafterFactory in the model registry."""
    key = "MockMistral4EagleDrafter"
    ModelFactoryRegistry._registry[key] = MockMistral4EagleDrafterFactory
    yield
    ModelFactoryRegistry._registry.pop(key, None)


# ---------------------------------------------------------------------------
# Test 1 — torch.export traceability
# ---------------------------------------------------------------------------


def test_mistral4_eagle_model_torch_export():
    """Mistral4 Eagle drafter can be traced with torch.export.

    Validates that the model architecture (Mistral4EagleMLA + torch_mla,
    Mistral4EagleMLP, eagle_proj) is fully torch.export-compatible.  Weights are
    not loaded (skip_loading_weights=True); random initialisations are used so
    only the graph structure matters.
    """
    eagle_path, target_path = _require_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    factory = EagleDrafterFactory(
        model=str(eagle_path),
        config_model=str(target_path),
        skip_loading_weights=True,
        # Reduce to a tiny model so the export runs quickly
        model_kwargs={"num_hidden_layers": 2},
    )
    model = factory.build_model(device)
    model = model.to(dtype=dtype)
    config = model.config

    batch_size, seq_len = 1, 8
    hidden_dim = config.hidden_size
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    print(
        f"\nExporting Mistral4 Eagle drafter: hidden_size={hidden_dim}, "
        f"num_layers={config.num_hidden_layers}"
    )

    try:
        exported = torch.export.export(
            model,
            args=(inputs_embeds, position_ids),
            kwargs={"hidden_states": hidden_states},
        )
        print("torch.export succeeded.")
        print("Graph (first 20 lines):")
        print("\n".join(exported.graph_module.code.split("\n")[:20]))
    except Exception as e:
        pytest.fail(f"torch.export failed: {e}")


# ---------------------------------------------------------------------------
# Test 2 — full AutoDeploy compile + run pipeline
# ---------------------------------------------------------------------------


def test_build_ad_mistral4_eagle(register_mock_mistral4_eagle_factory):
    """Mistral4 Eagle drafter runs through the full AutoDeploy compile/run pipeline.

    Uses MockMistral4EagleDrafterFactory which:
    - Loads architecture config from the target model (Mistral-Small-4-119B-2603).
    - Injects random hidden states so the Eagle head can run without a target model.
    - Uses skip_loading_weights=True and tiny model_kwargs for fast execution.

    The MLA attention transform (insert_cached_mla_attention) is applied since
    Mistral4EagleMLA uses the torch_mla canonical op.
    """
    eagle_path, target_path = _require_paths()

    llm_extra_args = {
        "model_factory": "MockMistral4EagleDrafter",
        # config_model is passed to the factory via model_kwargs plumbing below
        "transforms": {
            "insert_cached_mla_attention": {"backend": "torch_mla"},
            "compile_model": {"backend": "torch-simple"},
        },
    }
    experiment_config = get_small_model_config(TARGET_MODEL_HUB_ID, **llm_extra_args)
    # Point the experiment at the Eagle checkpoint; config comes from target via config_model
    experiment_config["args"]["model"] = str(eagle_path)
    experiment_config["args"]["runtime"] = "demollm"
    experiment_config["args"]["world_size"] = 0
    experiment_config["args"]["tokenizer"] = str(target_path)
    experiment_config["args"]["skip_loading_weights"] = True

    # config_model is consumed by MockMistral4EagleDrafterFactory.__init__ via **kwargs
    # passed through ModelFactoryRegistry.  We inject it via a monkey-patch on the factory
    # class so that target_path is captured without changing the LlmArgs schema.
    original_init = MockMistral4EagleDrafterFactory.__init__

    def _patched_init(self, model, **kwargs):
        kwargs.setdefault("config_model", str(target_path))
        original_init(self, model=model, **kwargs)

    MockMistral4EagleDrafterFactory.__init__ = _patched_init
    try:
        cfg = ExperimentConfig(**experiment_config)
        main(cfg)
    finally:
        MockMistral4EagleDrafterFactory.__init__ = original_init


# ---------------------------------------------------------------------------
# Test 3 & 4 — E2E smoke: Eagle one-model spec-dec (skip_loading_weights)
# ---------------------------------------------------------------------------


def _run_mistral4_eagle_one_model_smoke(num_hidden_layers: int):
    """Shared implementation for Eagle one-model smoke tests.

    Builds the full Eagle one-model pipeline (target + draft) with
    skip_loading_weights=True and reduced model_kwargs.  Constructs LLM
    directly (not via ExperimentConfig + main) because
    Eagle3DecodingConfig.eagle3_layers_to_capture (Set[int]) does not survive
    the model_dump -> LlmArgs OmegaConf round-trip that main() performs.
    TODO: fix the Set[int] OmegaConf issue and switch to ExperimentConfig + main().
    """
    from tensorrt_llm._torch.auto_deploy.llm import LLM as ADLLM
    from tensorrt_llm.llmapi import Eagle3DecodingConfig, SamplingParams

    eagle_path, target_path = _require_paths()

    small_config = get_small_model_config(TARGET_MODEL_HUB_ID)
    small_dims = dict(small_config["args"]["model_kwargs"]["text_config"])
    small_dims["num_hidden_layers"] = num_hidden_layers

    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model=str(eagle_path),
        eagle3_one_model=True,
        eagle3_model_arch="mistral_large3",
    )

    with ADLLM(
        model=str(target_path),
        model_factory="Mistral3ForConditionalGeneration",
        model_kwargs={"text_config": small_dims},
        skip_loading_weights=True,
        transforms={"insert_cached_mla_attention": {"backend": "torch_mla"}},
        speculative_config=spec_config,
        speculative_model_kwargs=dict(small_dims),
        disable_overlap_scheduler=True,
        compile_backend="torch-simple",
        max_num_tokens=256,
        world_size=1,
    ) as llm:
        outputs = llm.generate(
            ["What is the capital of France?"],
            SamplingParams(max_tokens=16, top_k=None, temperature=0.0, seed=42),
        )

    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0


def test_mistral4_eagle_one_model_smoke():
    """Eagle one-model spec-dec with a tiny (1-layer) Mistral-Small-4-119B target."""
    _run_mistral4_eagle_one_model_smoke(num_hidden_layers=1)


def test_mistral4_eagle_one_model_smoke_3layers():
    """Eagle one-model spec-dec with 3 layers.

    Exercises multi-layer graph transforms (layer boundary detection, residual add
    identification) to surface boundary-detection issues such as MLA's
    multi-projection pattern or unused placeholders in the exported graph.
    """
    _run_mistral4_eagle_one_model_smoke(num_hidden_layers=3)
