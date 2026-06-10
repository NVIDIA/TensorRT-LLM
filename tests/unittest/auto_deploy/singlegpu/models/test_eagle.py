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
from typing import Any, ClassVar, Dict
from unittest.mock import patch

import pytest
import torch
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main
from test_common.llm_data import hf_id_to_local_model_dir

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    Eagle3DraftOutput,
    EagleConfig,
    EagleDrafterForCausalLM,
    EagleRMSNorm,
    EagleWrapper,
    EagleWrapperConfig,
)
from tensorrt_llm._torch.auto_deploy.models.eagle import EagleDrafterFactory, EagleOneModelFactory
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.utils.node_utils import (
    get_weight_shape,
    infer_draft_embedding_size,
    is_any_lin_op,
)

EAGLE_MODEL_HUB_ID = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
NEMOTRON_SUPER_MODEL_HUB_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

NEMOTRON_SUPER_DRAFT_MODEL_KWARGS = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "head_dim": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "attention_bias": False,
    "layer_norm_epsilon": 1e-5,
    "residual_in_fp32": False,
    "mlp_bias": False,
    "mlp_hidden_act": "relu2",
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "moe_shared_expert_intermediate_size": 64,
    "moe_latent_size": 16,
    "n_group": 1,
    "topk_group": 1,
}

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

    _drafter_defaults: ClassVar[Dict[str, Dict[str, Any]]] = {
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
        # transformers>=5.5 applies @dataclass(kw_only=True) to PretrainedConfig
        # subclasses, overriding EagleConfig.__init__. Use the factory classmethod.
        model_config = MockEagleConfig.from_base_config(model_config, model_config.model_type)

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


def _build_small_draft_factory(
    model_hub_id: str, model_kwargs: dict | None = None
) -> EagleDrafterFactory:
    draft_model_path = hf_id_to_local_model_dir(model_hub_id)
    if draft_model_path is None or not Path(draft_model_path).is_dir():
        pytest.skip(
            f"Draft model {model_hub_id} not found (LLM_MODELS_ROOT not set or model missing)"
        )

    return EagleDrafterFactory(
        model=str(draft_model_path),
        model_kwargs=model_kwargs,
        skip_loading_weights=True,
        max_seq_len=64,
    )


def test_eagle_rmsnorm_keeps_fp32_weights():
    norm = EagleRMSNorm(hidden_size=16)

    assert norm.weight.dtype == torch.float32


def test_eagle_wrapper_instantiates_sa_enhancer():
    from tensorrt_llm.llmapi import SAEnhancerConfig

    wrapper = EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=3,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
            sa_config=SAEnhancerConfig(threshold=2),
        ),
        target_model=torch.nn.Module(),
        draft_model=torch.nn.Module(),
    )

    assert wrapper.sa_enhancer is not None
    assert wrapper.sa_enhancer.threshold == 2


def test_eagle_one_model_factory_populates_sa_enhancer_config(monkeypatch):
    from tensorrt_llm.llmapi import Eagle3DecodingConfig, SAEnhancerConfig

    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model="draft-model",
        sa_config=SAEnhancerConfig(threshold=5),
    )
    factory = EagleOneModelFactory(
        model="target-model",
        speculative_config=spec_config,
        skip_loading_weights=True,
        max_seq_len=64,
    )
    target_model = torch.nn.Module()
    draft_model = torch.nn.Module()
    draft_model.config = SimpleNamespace(
        load_embedding_from_target=True,
        load_lm_head_from_target=True,
        normalize_target_hidden_state=False,
    )
    monkeypatch.setattr(factory.target_factory, "build_model", lambda device: target_model)
    monkeypatch.setattr(factory.draft_factory, "build_model", lambda device: draft_model)

    wrapper = factory._build_model("cpu")

    assert isinstance(wrapper, EagleWrapper)
    assert wrapper.sa_enhancer is not None
    assert wrapper.sa_enhancer.threshold == 5


def test_eagle_wrapper_sa_override_updates_next_new_tokens():
    class FakeSAEnhancer:
        def __init__(self):
            self.seen_draft_tokens = None

        def maybe_override_all_draft_tokens(self, draft_tokens):
            self.seen_draft_tokens = draft_tokens.clone()
            return draft_tokens + 100

    wrapper = EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=2,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
        ),
        target_model=torch.nn.Module(),
        draft_model=torch.nn.Module(),
    )
    fake_sa_enhancer = FakeSAEnhancer()
    wrapper.sa_enhancer = fake_sa_enhancer
    next_new_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

    wrapper._maybe_apply_sa_draft_override(
        next_new_tokens,
        num_prefill=1,
        sa_manager=object(),
    )

    torch.testing.assert_close(
        next_new_tokens,
        torch.tensor([[1, 2, 3], [4, 105, 106]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        fake_sa_enhancer.seen_draft_tokens,
        torch.tensor([[5, 6]], dtype=torch.int32),
    )


def test_eagle_wrapper_sa_override_noop_without_manager():
    """SA override must no-op when sa_manager is None even though sa_enhancer is set.

    This is the KV-cache resize-transform flow: the resize runs forward before the executor's
    resource managers exist, so it passes sa_manager=None. It guards the class-docstring invariant
    that the two guards (sa_enhancer is not None AND sa_manager is not None) are not redundant --
    collapsing them to a single sa_enhancer check would wrongly fire SA here.
    """

    class FakeSAEnhancer:
        def maybe_override_all_draft_tokens(self, draft_tokens):
            raise AssertionError("SA override must not run when sa_manager is None")

    wrapper = EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=2,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
        ),
        target_model=torch.nn.Module(),
        draft_model=torch.nn.Module(),
    )
    wrapper.sa_enhancer = FakeSAEnhancer()
    next_new_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    original = next_new_tokens.clone()

    wrapper._maybe_apply_sa_draft_override(next_new_tokens, num_prefill=1, sa_manager=None)

    torch.testing.assert_close(next_new_tokens, original)


def test_eagle_wrapper_forward_unpacks_spec_dec_args():
    """forward() routes spec-dec inputs through SpeculativeDecodingModelArgs.

    The struct is the executor->model contract: forward must unpack it and call the cached
    path with the struct's cache_seq_interface and sa_manager. With no struct (export time),
    it must fall back to the prefill-only path.
    """
    from tensorrt_llm._torch.auto_deploy.shim.interface import SpeculativeDecodingModelArgs

    wrapper = EagleWrapper(
        EagleWrapperConfig(
            max_draft_len=2,
            load_embedding_from_target=True,
            load_lm_head_from_target=True,
        ),
        target_model=torch.nn.Module(),
        draft_model=torch.nn.Module(),
    )

    csi = object()
    sa_manager = object()
    with (
        patch.object(wrapper, "_forward_with_kv_cache") as mock_cached,
        patch.object(wrapper, "_forward_prefill_only") as mock_prefill,
    ):
        wrapper.forward(
            spec_dec_args=SpeculativeDecodingModelArgs(
                cache_seq_interface=csi, sa_manager=sa_manager
            )
        )
        mock_cached.assert_called_once_with(csi, sa_manager=sa_manager)
        mock_prefill.assert_not_called()

    with (
        patch.object(wrapper, "_forward_with_kv_cache") as mock_cached,
        patch.object(wrapper, "_forward_prefill_only") as mock_prefill,
    ):
        wrapper.forward(input_ids="tokens", position_ids="pos")
        mock_prefill.assert_called_once_with(input_ids="tokens", position_ids="pos")
        mock_cached.assert_not_called()


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
            "insert_cached_attention": {"backend": "trtllm"},
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
    device = torch.device("cuda")
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


@pytest.mark.parametrize(
    ("model_hub_id", "model_kwargs", "expected_is_eagle"),
    [
        (
            EAGLE_MODEL_HUB_ID,
            get_small_model_config(EAGLE_MODEL_HUB_ID)["args"]["model_kwargs"],
            True,
        ),
        (NEMOTRON_SUPER_MODEL_HUB_ID, NEMOTRON_SUPER_DRAFT_MODEL_KWARGS, False),
    ],
)
def test_infer_draft_hidden_size_from_exported_draft_graph(
    model_hub_id, model_kwargs, expected_is_eagle
):
    factory = _build_small_draft_factory(model_hub_id, model_kwargs=model_kwargs)
    model = factory.build_model("cuda")
    inner_model = model.model.eval()
    hidden_size = model.config.hidden_size
    dtype = model.config.torch_dtype

    batch_size = 2
    seq_len = 4
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    position_ids = (
        torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    )
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)

    gm = torch_export_to_gm(
        inner_model,
        args=(inputs_embeds, position_ids, hidden_states),
        clone=True,
    )
    gm.is_draft = True

    linear_nodes = [node for node in gm.graph.nodes if is_any_lin_op(node)]
    assert linear_nodes, "Expected exported draft graph to contain linear nodes"
    assert get_weight_shape(linear_nodes[-1], dim=0) == hidden_size

    embd, in_eagle_drafter = infer_draft_embedding_size(gm, linear_nodes)
    assert embd == hidden_size
    assert in_eagle_drafter is expected_is_eagle
