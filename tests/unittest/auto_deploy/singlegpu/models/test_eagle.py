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

import pytest
import torch
import torch.nn as nn
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main
from test_common.llm_data import hf_id_to_local_model_dir

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    Eagle3DraftOutput,
    EagleConfig,
    EagleDrafterForCausalLM,
    EagleRMSNorm,
    EagleWrapper,
)
from tensorrt_llm._torch.auto_deploy.models.eagle import EagleDrafterFactory, EagleOneModelFactory
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.auto_deploy.utils.node_utils import (
    get_weight_shape,
    infer_draft_embedding_size,
    is_any_lin_op,
)
from tensorrt_llm.llmapi import MTPDecodingConfig

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
        assert self.eagle_drafter.embed_tokens is not None, (
            "embed_tokens must be set before running standalone Eagle model."
        )
        assert self.lm_head is not None, (
            "lm_head must be set before running standalone Eagle model."
        )

        if input_embeds is None:
            inputs_embeds = self.eagle_drafter.embed_tokens(input_ids)

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


def _mtp_speculative_config() -> MTPDecodingConfig:
    return MTPDecodingConfig(
        max_draft_len=1,
        mtp_eagle_one_model=True,
        speculative_model="test-model",
    )


class _StaticTargetFactory:
    def __init__(self, model: nn.Module, export_infos: list) -> None:
        self.model = model
        self.export_infos = export_infos

    def build_model(self, device: str) -> nn.Module:
        return self.model

    def get_export_infos(self, model: nn.Module) -> list:
        assert model is self.model
        return self.export_infos


class _StaticDraftFactory:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def build_model(self, device: str) -> nn.Module:
        return self.model


def test_mtp_llm_args_preserves_target_factory_for_one_model():
    args = LlmArgs(
        model="test-model",
        model_factory="AutoModelForCausalLM",
        max_seq_len=64,
        speculative_config=_mtp_speculative_config(),
    )

    factory = args.create_factory()

    assert args.model_factory == "eagle_one_model"
    assert args.target_model_factory == "AutoModelForCausalLM"
    assert isinstance(factory, EagleOneModelFactory)
    assert isinstance(factory.target_factory, AutoModelForCausalLMFactory)


def test_mtp_one_model_factory_honors_draft_config():
    target_model = nn.Module()
    draft_model = nn.Module()
    draft_model.config = SimpleNamespace(
        load_embedding_from_target=False,
        load_lm_head_from_target=False,
        normalize_target_hidden_state=True,
    )
    export_info = SimpleNamespace(submodule_name="model.language_model")
    factory = EagleOneModelFactory(
        model="test-model",
        skip_loading_weights=True,
        max_seq_len=64,
        speculative_config=_mtp_speculative_config(),
        target_model_factory="AutoModelForCausalLM",
    )
    factory.target_factory = _StaticTargetFactory(target_model, [export_info])
    factory.draft_factory = _StaticDraftFactory(draft_model)

    wrapper = factory._build_model("meta")

    assert isinstance(wrapper, EagleWrapper)
    assert wrapper.target_model is target_model
    assert wrapper.draft_model is draft_model
    assert wrapper.max_draft_len == 1
    assert wrapper.load_embedding_from_target is False
    assert wrapper.load_lm_head_from_target is False
    assert wrapper.normalize_target_hidden_state is True
    assert factory._target_export_submodule_name == "model.language_model"


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
    inner_model = model.eagle_drafter.eval()
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


# Draft exclude remap fixture used by the quant-config unit tests below. This keeps the tests
# focused on EagleOneModelFactory.get_quant_config without depending on a production config factory.
_MTP_DRAFT_EXCLUDE_MAP = {
    r"^mtp\.layers\.0(?=\.|\*)": "eagle_drafter.layers",
}


def _eagle_quant_stub(target_excludes, draft_map, export_submodule_name):
    """A stand-in EagleOneModelFactory exposing only what get_quant_config reads (no model build)."""
    return SimpleNamespace(
        target_factory=SimpleNamespace(
            get_quant_config=lambda: {"exclude_modules": list(target_excludes)}
        ),
        draft_factory=SimpleNamespace(_quant_exclude_conversion_mapping=draft_map),
        _target_export_submodule_name=export_submodule_name,
    )


def test_eagle_get_quant_config_aliases_target_excludes():
    """VLM target (non-trivial submodule name): target excludes are re-rooted, draft remap maps MTP.

    The target text model is exported rooted at its own submodule, so its graph node names are
    relative (e.g. ``layers.0.linear_attn*``) while the checkpoint exclude patterns are full paths
    (e.g. ``model.language_model.layers.0.linear_attn*``). The factory must rewrite the target
    patterns into the relative namespace so the excluded bf16 modules are skipped, while applying
    the draft-namespace remap (``mtp.layers.0* -> eagle_drafter.layers*``) only to the draft head
    and not over-matching the target graph.
    """
    from tensorrt_llm._torch.auto_deploy.models.eagle import EagleOneModelFactory
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import should_skip_quantization

    # Glob forms as they appear in the NVFP4 hf_quant_config.json exclude_modules.
    target_excludes = [
        "lm_head",
        "model.language_model.layers.0.linear_attn*",
        "model.language_model.layers.11.self_attn*",
        "model.language_model.layers.0.mlp.shared_expert_gate",
        "model.visual*",  # irrelevant to the text export; must survive untouched
        "mtp.layers.0*",  # the unquantized MTP head; mapped to draft namespace
    ]
    stub = _eagle_quant_stub(target_excludes, _MTP_DRAFT_EXCLUDE_MAP, "model.language_model")

    result = EagleOneModelFactory.get_quant_config(stub)["exclude_modules"]

    # Text-model excludes are rewritten in place into the relative graph namespace...
    assert "layers.0.linear_attn*" in result
    assert "layers.11.self_attn*" in result
    assert "layers.0.mlp.shared_expert_gate" in result
    # ...with the checkpoint-namespace originals dropped (not kept), and the draft-namespace
    # mapping applied into the dedicated eagle_drafter namespace.
    assert "model.language_model.layers.0.linear_attn*" not in result
    assert "eagle_drafter.layers*" in result  # mtp.layers.0* -> eagle_drafter.layers*
    # Unrelated entries are preserved.
    assert "lm_head" in result
    assert "model.visual*" in result

    # Behavioral checks against the FULL resulting list: the alias skips the relative target
    # self_attn module that the checkpoint excludes...
    assert should_skip_quantization("layers.11.self_attn.q_proj", result)
    # ...while a relative target module that is NOT excluded stays quantized. This guards against
    # the draft pattern (or any over-broad alias) accidentally skipping the whole target graph --
    # the reason we add namespaced aliases instead of globally stripping prefixes.
    assert not should_skip_quantization("layers.5.self_attn.q_proj", result)


def test_eagle_get_quant_config_full_model_export_no_target_collision():
    """Non-VLM target (trivial submodule ""): the draft's eagle_drafter excludes don't collide.

    This is the non-VLM NVFP4 MTP case (e.g. Ultra): the target is exported at the root, so its
    node names are full ``model.layers.N.*``. Because the draft is rooted at the dedicated
    ``eagle_drafter.*`` namespace, its remapped excludes can never over-match the target -- even a
    broad ``mtp.layers.0*`` exclude maps to ``eagle_drafter.layers*`` and stays disjoint from
    ``model.layers.*``. (Before the eagle_drafter rename, ``mtp.layers.0* -> model.layers*`` and
    this collided, wrongly skipping the whole target.)
    """
    from tensorrt_llm._torch.auto_deploy.models.eagle import EagleOneModelFactory
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import should_skip_quantization

    target_excludes = [
        "lm_head",
        "model.embed_tokens*",  # full-path target module; kept verbatim (no prefix to strip)
        "model.layers.0.linear_attn*",  # a specific target-layer exclude
        "mtp.layers.0*",  # broad MTP head exclude -> eagle_drafter.layers*
    ]
    stub = _eagle_quant_stub(target_excludes, _MTP_DRAFT_EXCLUDE_MAP, "")

    result = EagleOneModelFactory.get_quant_config(stub)["exclude_modules"]

    # No prefix to strip -> target full-path patterns survive verbatim (they match the full-model
    # graph), and the draft remap maps the MTP head into the *distinct* eagle_drafter namespace.
    assert "model.embed_tokens*" in result
    assert "model.layers.0.linear_attn*" in result
    assert "lm_head" in result
    # Non-triviality anchor for the collision check below: this confirms the MTP head WAS remapped
    # (into "eagle_drafter.layers*"). Without it, the "stays quantizable" assertion could be
    # satisfied by simply dropping the draft exclude rather than by it landing in a disjoint
    # namespace.
    assert "eagle_drafter.layers*" in result
    assert "model.layers*" not in result  # crucially NOT the generic target namespace
    assert "mtp.layers.0*" not in result

    # The collision regression itself: the target's own layers stay quantizable -- the draft's broad
    # "eagle_drafter.layers*" does NOT over-match a full-model target node "model.layers.5...".
    # (This is a non-triviality check: meaningful only together with the "eagle_drafter.layers*"
    # assertion above, which proves the remap actually ran.)
    assert not should_skip_quantization("model.layers.5.self_attn.q_proj.weight", result)
    # ...while the specific target exclude and the full-path target module still apply.
    assert should_skip_quantization("model.layers.0.linear_attn.q_proj.weight", result)
    assert should_skip_quantization("model.embed_tokens.weight", result)


def test_eagle_get_quant_config_rejects_target_in_draft_namespace():
    """A target exclude that strips into the draft's reserved "eagle_drafter.*" namespace must fail.

    That is the core no-collision assumption: targets live under "model.*"/etc., never
    "eagle_drafter.*". If a target's own (stripped) names land in the draft namespace the two
    sub-graphs' excludes overlap and the partition is unsafe -- fail loudly rather than mis-quantize.
    """
    from tensorrt_llm._torch.auto_deploy.models.eagle import EagleOneModelFactory

    # VLM target whose checkpoint (hypothetically) names a module in the draft's reserved namespace:
    # after stripping the "model.language_model." prefix it becomes "eagle_drafter.foo*", colliding
    # with the draft sub-graph's namespace.
    stub = _eagle_quant_stub(
        ["model.language_model.eagle_drafter.foo*"],
        _MTP_DRAFT_EXCLUDE_MAP,
        "model.language_model",
    )
    with pytest.raises(AssertionError):
        EagleOneModelFactory.get_quant_config(stub)


def test_eagle_single_target_export_info_rejects_multiple_submodules():
    """The one-model path assumes a single target export root; >1 must fail loudly.

    Otherwise the code would silently use the first export info, mis-namespacing the prefix strip
    in get_quant_config.
    """
    from tensorrt_llm._torch.auto_deploy.models.eagle import EagleOneModelFactory

    def stub_with(infos):
        return SimpleNamespace(
            target_factory=SimpleNamespace(get_export_infos=lambda _model: infos)
        )

    # Real target factories always return exactly one export info: full-model export is
    # [FullModelExportInfo()] (submodule_name ""), a VLM is one TextModelExportInfo
    # (submodule_name "model.language_model"). Both are returned as-is.
    full_model = SimpleNamespace(submodule_name="")
    got = EagleOneModelFactory._single_target_export_info(stub_with([full_model]), object())
    assert got is full_model
    vlm = SimpleNamespace(submodule_name="model.language_model")
    got = EagleOneModelFactory._single_target_export_info(stub_with([vlm]), object())
    assert got is vlm

    # >1 export infos -> assert (the prefix-strip logic only threads a single export root).
    two = [
        SimpleNamespace(submodule_name="model.language_model"),
        SimpleNamespace(submodule_name="model.vision_model"),
    ]
    with pytest.raises(AssertionError):
        EagleOneModelFactory._single_target_export_info(stub_with(two), object())
