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
"""CPU-only checks for Nemotron 3.5 VL LoRA wiring.

These run without a GPU or a real checkpoint. They cover the two failure modes
that would otherwise only surface as a wasted GPU job: an adapter whose keys the
HF loader silently drops, and a missing ``lora_config`` hook on the VL wrapper.
"""

import json
import os
from pathlib import Path

import pytest
from nemotron_h_lora_utils import (
    assert_adapter_keys_parse,
    create_nemotron_h_lora_adapter,
    create_vision_only_lora_adapter,
    layer_plan,
    projection_dims,
    read_llm_config,
)

from tensorrt_llm._torch.peft.lora.layer import LoraModuleType
from tensorrt_llm._torch.peft.lora.manager import LoraManager

pytestmark = pytest.mark.cpu_only

# A structurally faithful miniature of
# nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B-SourceOfTruth: same key layout and
# same block vocabulary, small dimensions. Everything these tests assert is
# about shape and naming, not about weights, so the miniature is sufficient and
# keeps the checks runnable on any box.
_MINI_LLM_CONFIG = {
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "mamba_head_dim": 8,
    "mamba_num_heads": 8,
    "n_groups": 2,
    "ssm_state_size": 16,
    "conv_kernel": 4,
    "moe_shared_expert_intermediate_size": 96,
    "moe_latent_size": 32,
    "moe_intermediate_size": 64,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "vocab_size": 512,
    "layers_block_type": ["mamba", "moe", "attention", "moe", "mamba", "moe"],
}


@pytest.fixture
def mini_checkpoint(tmp_path):
    """A checkpoint directory carrying only config.json, VL-nested."""
    path = tmp_path / "nemotron35-vl-mini"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NemotronH_Omni_Reasoning_V3"],
                "model_type": "nemotron_h_omni",
                "img_context_token_id": 18,
                "sound_config": None,
                "vision_config": {"hidden_size": 64},
                "llm_config": _MINI_LLM_CONFIG,
            }
        )
    )
    return str(path)


def test_read_llm_config_unwraps_the_vl_nesting(mini_checkpoint):
    """The VL checkpoint nests decoder dims one level down."""
    llm_config = read_llm_config(mini_checkpoint)
    assert llm_config["hidden_size"] == 128
    assert "layers_block_type" in llm_config


def test_read_llm_config_accepts_a_flat_text_only_config(tmp_path):
    """The text-only Super checkpoint has the same fields at the top level."""
    path = tmp_path / "flat"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_MINI_LLM_CONFIG))
    assert read_llm_config(str(path))["hidden_size"] == 128


def test_layer_plan_accepts_both_spellings():
    """3.5 uses layers_block_type; the older checkpoints use a letter pattern."""
    words = layer_plan({"layers_block_type": ["mamba", "moe", "attention"]})
    letters = layer_plan({"hybrid_override_pattern": "ME*"})
    assert words == letters == ["mamba", "moe", "attention"]


def test_layer_plan_accepts_the_current_nemotron_h_names():
    """The NVFP4 exports and the bf16 SourceOfTruth disagree on layer names.

    SourceOfTruth says mamba / attention; the quantized exports say
    linear_attention / full_attention, which are the current Nemotron-H names.
    The checkpoint's own `_nemotron_h_compatible_config` maps current to legacy.
    Both must yield the same plan, or an adapter built against one checkpoint
    silently targets the wrong layers of the other.
    """
    current = layer_plan(
        {"layers_block_type": ["linear_attention", "moe", "full_attention", "moe"]}
    )
    legacy = layer_plan({"layers_block_type": ["mamba", "moe", "attention", "moe"]})
    assert current == legacy == ["mamba", "moe", "attention", "moe"]


def test_layer_plan_rejects_an_unknown_block():
    with pytest.raises(KeyError):
        layer_plan({"layers_block_type": ["mamba", "quantum"]})


@pytest.mark.parametrize("invalid_index", [-1, len(_MINI_LLM_CONFIG["layers_block_type"])])
def test_adapter_rejects_invalid_layer_indices(
    mini_checkpoint: str, tmp_path: Path, invalid_index: int
) -> None:
    """Reject invalid indices even when the selection also contains a valid layer."""
    # Silently dropping an invalid index leaves layer-specific tests with incomplete coverage.
    with pytest.raises(ValueError, match="layer index .* is out of range"):
        create_nemotron_h_lora_adapter(
            str(tmp_path / "adapter"), mini_checkpoint, layer_indices=[0, invalid_index]
        )


def test_mamba_in_proj_dim_matches_the_runtime_derivation():
    """d_in_proj here must match _util.py's, or the adapter loads at the wrong shape.

    Mirror of the derivation at
    ``tensorrt_llm/_torch/pyexecutor/_util.py`` (mamba_in_proj_size):
    ``2 * d_inner + 2 * n_groups * d_state + mamba_num_heads``.
    """
    dims = projection_dims(_MINI_LLM_CONFIG)
    d_inner = _MINI_LLM_CONFIG["mamba_head_dim"] * _MINI_LLM_CONFIG["mamba_num_heads"]
    expected = (
        2 * d_inner
        + 2 * _MINI_LLM_CONFIG["n_groups"] * _MINI_LLM_CONFIG["ssm_state_size"]
        + _MINI_LLM_CONFIG["mamba_num_heads"]
    )
    assert dims["in_proj"] == (_MINI_LLM_CONFIG["hidden_size"], expected)
    assert dims["out_proj"] == (d_inner, _MINI_LLM_CONFIG["hidden_size"])


@pytest.mark.parametrize("key_prefix", ["", "language_model."])
def test_every_adapter_key_reaches_a_module(mini_checkpoint, tmp_path, key_prefix):
    """Both key layouts must fully parse.

    ``iterate_hf_lora`` warns and skips an unrecognized module instead of
    raising, so a prefix the pattern cannot handle yields an adapter that loads
    clean and never fires. This is the check that makes the GPU run meaningful.
    """
    adapter = create_nemotron_h_lora_adapter(
        str(tmp_path / f"adapter{len(key_prefix)}"),
        mini_checkpoint,
        lora_rank=8,
        seed=0,
        key_prefix=key_prefix,
    )
    parsed = assert_adapter_keys_parse(adapter)

    # 6 layers: 2 mamba x 2 proj + 3 moe x 4 proj + 1 attention x 4 proj.
    assert sum(len(modules) for modules in parsed.values()) == 2 * 2 + 3 * 4 + 1 * 4


def test_adapter_targets_every_block_type(mini_checkpoint, tmp_path):
    """Attention, Mamba, shared expert and latent MoE must all be covered.

    Any of the four silently missing would still produce a passing multi-LoRA
    run, since divergence only needs one live module.
    """
    adapter = create_nemotron_h_lora_adapter(
        str(tmp_path / "adapter"), mini_checkpoint, lora_rank=8
    )
    with open(os.path.join(adapter, "adapter_config.json")) as f:
        targets = set(json.load(f)["target_modules"])
    assert targets == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj",
        "out_proj",
        "shared_experts.up_proj",
        "shared_experts.down_proj",
        "fc1_latent_proj",
        "fc2_latent_proj",
    }


def test_adapters_differing_only_by_seed_have_different_weights(mini_checkpoint, tmp_path):
    """The multi-LoRA assertion depends on adapters actually differing."""
    from safetensors.torch import load_file

    a = create_nemotron_h_lora_adapter(str(tmp_path / "a"), mini_checkpoint, seed=0)
    b = create_nemotron_h_lora_adapter(str(tmp_path / "b"), mini_checkpoint, seed=1)
    wa = load_file(os.path.join(a, "adapter_model.safetensors"))
    wb = load_file(os.path.join(b, "adapter_model.safetensors"))
    assert wa.keys() == wb.keys()
    assert all(not wa[k].equal(wb[k]) for k in wa)


def test_alpha_tracks_rank_so_scaling_is_rank_invariant(mini_checkpoint, tmp_path):
    """manager.py applies alpha / r; a fixed alpha would weaken high ranks.

    Without this, a rank-32 adapter perturbs the logits a quarter as much as a
    rank-8 one and the divergence assertion becomes rank-dependent.
    """
    for rank in (8, 16, 32):
        adapter = create_nemotron_h_lora_adapter(
            str(tmp_path / f"r{rank}"), mini_checkpoint, lora_rank=rank
        )
        with open(os.path.join(adapter, "adapter_config.json")) as f:
            config = json.load(f)
        assert config["lora_alpha"] / config["r"] == 2.0


def test_vision_only_adapter_is_dropped_entirely(mini_checkpoint, tmp_path):
    """The negative control must be inert, and provably so.

    No VLM in the repo routes lora_params to a vision tower. Asserting that the
    loader drops every vision key is what makes the GPU test's divergence mean
    "the language-model modules changed the output" rather than "something
    changed the output".
    """
    adapter = create_vision_only_lora_adapter(str(tmp_path / "vision"), mini_checkpoint)
    with pytest.raises(AssertionError, match="dropped by iterate_hf_lora"):
        assert_adapter_keys_parse(adapter)


def test_vl_wrapper_exposes_lora_config():
    """The wrapper must answer lora_config; callers resolve it on the outer class.

    ``examples/llm-api/quickstart_multimodal.py`` calls
    ``model_class.lora_config(model_dir)`` on the registered architecture, which
    for this model is the VL wrapper and not ``NemotronHForCausalLM``.
    """
    from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import NemotronHMultimodalModel

    config = NemotronHMultimodalModel.lora_config("")
    assert set(config.lora_target_modules) == {
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_dense",
        "mamba_in_proj",
        "mamba_out_proj",
        "shared_expert_h_to_4h",
        "shared_expert_4h_to_h",
        "moe_latent_fc1",
        "moe_latent_fc2",
    }
    # Every declared target must resolve to a real module id, or engine setup
    # fails later with a much less obvious error. The lookup is
    # LoraManager.LORA_MODULE_IDS, not LoraModuleType.from_string: from_string
    # upper-cases its argument, so it resolves "mamba_in_proj" but not "attn_q"
    # (the member is ATTENTION_Q). These target names only round-trip via the
    # manager's table.
    for name in config.lora_target_modules:
        assert name in LoraManager.LORA_MODULE_IDS, f"{name} is not a known LoRA module"
        assert LoraManager.LORA_MODULE_IDS[name] == LoraModuleType(
            LoraManager.LORA_MODULE_IDS[name]
        ), f"{name} maps to an id with no matching LoraModuleType"


def test_vl_wrapper_hf_mapping_matches_the_adapter_builder():
    """The builder's projection names must be exactly the mapping's HF names.

    A drift here is the silent-inert-adapter failure: keys parse, modules do not
    match, ``iterate_hf_lora`` warns and skips.
    """
    from nemotron_h_lora_utils import NEMOTRON_H_HF_MODULES

    from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import NemotronHMultimodalModel

    mapping = NemotronHMultimodalModel.lora_config("").trtllm_modules_to_hf_modules
    assert set(mapping.values()) == set(NEMOTRON_H_HF_MODULES)


def test_vl_wrapper_lora_request_fails_loudly_without_a_bundled_adapter(mini_checkpoint):
    """Nemotron VL ships no adapter; --load_lora must not become a silent no-op.

    The message points the caller at the class that hands out a LoraConfig, and
    takes that name from the class rather than spelling it out, so a rename
    cannot leave the message naming something that no longer exists.
    """
    from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import NemotronHMultimodalModel

    with pytest.raises(FileNotFoundError, match="No bundled LoRA adapter") as raised:
        NemotronHMultimodalModel.lora_request(1, "image", mini_checkpoint)
    assert f"{NemotronHMultimodalModel.__name__}.lora_config()" in str(raised.value)


def test_vl_wrapper_lora_request_returns_one_bundled_adapter_per_request(mini_checkpoint):
    """With a bundled adapter present, every prompt gets its own request object.

    `quickstart_multimodal.py` passes the returned list straight to
    `llm.generate` alongside a prompt list, so a wrong length silently drops
    LoRA from the tail of the batch rather than raising.
    """
    from tensorrt_llm._torch.models.modeling_nemotron_h_multimodal import NemotronHMultimodalModel

    bundled = os.path.join(mini_checkpoint, "lora")
    os.makedirs(bundled)

    requests = NemotronHMultimodalModel.lora_request(3, "image", mini_checkpoint)

    assert len(requests) == 3
    for request in requests:
        assert request.path == bundled
        assert request.name == "nemotron-vl-lora"
        assert request.ckpt_source == "hf"
    # One adapter, so one cache slot: distinct ids would make the peft cache
    # load the same weights three times.
    assert {request.adapter_id for request in requests} == {0}
