# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Phi-4 multimodal AutoDeploy custom model."""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.phi4_multimodal.configuration_phi4_multimodal import (
    Phi4MultimodalAudioConfig,
    Phi4MultimodalConfig,
    Phi4MultimodalVisionConfig,
)
from transformers.models.phi4_multimodal.modeling_phi4_multimodal import (
    Phi4MultimodalAttention,
    Phi4MultimodalDecoderLayer,
    Phi4MultimodalForCausalLM,
    Phi4MultimodalMLP,
    Phi4MultimodalRMSNorm,
    Phi4MultimodalRotaryEmbedding,
)

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_phi4mm import (
    InputMode,
    Phi4MMAttention,
    Phi4MMConfig,
    Phi4MMDecoderLayer,
    Phi4MMForCausalLM,
    Phi4MMMLP,
    Phi4MMRMSNorm,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_hf_config() -> Phi4MultimodalConfig:
    config = Phi4MultimodalConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        original_max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=1.0,
        attention_dropout=0.0,
        resid_pdrop=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        vision_config=Phi4MultimodalVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=28,
            patch_size=14,
        ),
        audio_config=Phi4MultimodalAudioConfig(
            hidden_size=32,
            intermediate_size=48,
            num_blocks=2,
            num_attention_heads=4,
            input_size=16,
            ext_pw_out_channel=32,
            depthwise_separable_out_channel=32,
            nemo_conv_channels=32,
            time_reduction=2,
        ),
    )
    config._attn_implementation = "eager"
    return config


def _create_custom_config() -> Phi4MMConfig:
    return Phi4MMConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        original_max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=1.0,
        attention_dropout=0.0,
        resid_pdrop=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        embd_layer=None,
        vision_lora={"r": 4, "lora_alpha": 4, "dp": 0.0, "layer": ""},
        speech_lora={"r": 4, "lora_alpha": 4, "dp": 0.0, "layer": ""},
    )


def _make_causal_mask(batch_size: int, seq_len: int, device, dtype):
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)


def _map_hf_state_dict_to_custom(hf_state_dict):
    mapped = {}
    for key, value in hf_state_dict.items():
        new_key = key
        for pattern in [
            "qkv_proj.weight",
            "o_proj.weight",
            "gate_up_proj.weight",
            "down_proj.weight",
        ]:
            if key.endswith(pattern):
                new_key = key.replace(".weight", ".base_layer.weight")
                break
        mapped[new_key] = value
    return mapped


def _load_text_weights(custom_module, hf_module):
    custom_module.load_state_dict(
        _map_hf_state_dict_to_custom(hf_module.state_dict()), strict=False
    )


def test_phi4mm_lora_structure():
    config = _create_custom_config()
    model = Phi4MMForCausalLM(config)
    qkv_proj = model.model.layers[0].self_attn.qkv_proj
    assert "vision" in qkv_proj.lora_A
    assert "speech" in qkv_proj.lora_A
    assert qkv_proj.base_layer.weight.shape[1] == config.hidden_size


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_phi4mm_rmsnorm_equivalence(B, S):
    device = "cpu"
    config = _create_custom_config()
    hidden_size = config.hidden_size

    hf_norm = Phi4MultimodalRMSNorm(hidden_size, eps=config.rms_norm_eps).to(device)
    custom_norm = Phi4MMRMSNorm(hidden_size, eps=config.rms_norm_eps).to(device)
    custom_norm.load_state_dict(hf_norm.state_dict())

    x = torch.randn(B, S, hidden_size, device=device)
    torch.testing.assert_close(custom_norm(x), hf_norm(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_phi4mm_mlp_equivalence(B, S):
    device = "cpu"
    hf_config = _create_hf_config()
    custom_config = _create_custom_config()

    hf_mlp = Phi4MultimodalMLP(hf_config).to(device).eval()
    custom_mlp = Phi4MMMLP(custom_config).to(device).eval()
    _load_text_weights(custom_mlp, hf_mlp)
    custom_mlp.disable_adapter()

    x = torch.randn(B, S, custom_config.hidden_size, device=device)
    torch.testing.assert_close(custom_mlp(x), hf_mlp(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_phi4mm_attention_equivalence(B, S):
    device = "cpu"
    hf_config = _create_hf_config()
    custom_config = _create_custom_config()

    hf_attn = Phi4MultimodalAttention(hf_config, layer_idx=0).to(device).eval()
    hf_rotary = Phi4MultimodalRotaryEmbedding(hf_config).to(device).eval()
    custom_attn = Phi4MMAttention(custom_config, layer_idx=0).to(device).eval()
    _load_text_weights(custom_attn, hf_attn)
    custom_attn.disable_adapter()

    x = torch.randn(B, S, custom_config.hidden_size, device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = hf_rotary(x, position_ids)
    attention_mask = _make_causal_mask(B, S, device, x.dtype)

    hf_out = hf_attn(
        hidden_states=x,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )[0]
    custom_out = custom_attn(x, position_ids)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_phi4mm_decoder_layer_equivalence(B, S):
    device = "cpu"
    hf_config = _create_hf_config()
    custom_config = _create_custom_config()

    hf_layer = Phi4MultimodalDecoderLayer(hf_config, layer_idx=0).to(device).eval()
    custom_layer = Phi4MMDecoderLayer(custom_config, layer_idx=0).to(device).eval()
    _load_text_weights(custom_layer, hf_layer)
    custom_layer.disable_adapter()

    x = torch.randn(B, S, custom_config.hidden_size, device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_embeddings = Phi4MultimodalRotaryEmbedding(hf_config).to(device)(x, position_ids)
    attention_mask = _make_causal_mask(B, S, device, x.dtype)

    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    custom_out = custom_layer(x, position_ids)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_phi4mm_full_model_equivalence(B, S):
    device = "cpu"
    hf_config = _create_hf_config()
    custom_config = _create_custom_config()

    hf_model = Phi4MultimodalForCausalLM(hf_config).to(device).eval()
    custom_model = Phi4MMForCausalLM(custom_config).to(device).eval()
    custom_model.load_state_dict(_map_hf_state_dict_to_custom(hf_model.state_dict()), strict=False)
    custom_model.model.unset_lora_adapter()

    input_ids = torch.randint(0, custom_config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    attention_mask = torch.ones(B, S, device=device, dtype=torch.long)

    hf_logits = hf_model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    ).logits
    custom_logits = custom_model(
        input_ids=input_ids,
        position_ids=position_ids,
        input_mode=InputMode.LANGUAGE,
    ).logits

    assert_rmse_close(custom_logits, hf_logits, rmse_ratio_tol=0.05)


def test_phi4mm_model_can_be_exported():
    device = "cpu"
    config = _create_custom_config()
    model = Phi4MMForCausalLM(config).to(device).eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        None,
    )
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={
            "input_ids": input_ids,
            "position_ids": position_ids,
            "input_mode": InputMode.LANGUAGE,
        },
        dynamic_shapes=dynamic_shapes,
    )
    move_to_device(gm, device)

    out_gm = gm(input_ids=input_ids, position_ids=position_ids, input_mode=InputMode.LANGUAGE)
    assert "logits" in out_gm
    logits = out_gm["logits"]
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert torch.isfinite(logits).all()

    batch_size_2, seq_len_2 = 1, 4
    input_ids_2 = torch.randint(0, config.vocab_size, (batch_size_2, seq_len_2), device=device)
    position_ids_2 = torch.arange(seq_len_2, device=device).unsqueeze(0).expand(batch_size_2, -1)
    out_gm_2 = gm(
        input_ids=input_ids_2,
        position_ids=position_ids_2,
        input_mode=InputMode.LANGUAGE,
    )
    logits_2 = out_gm_2["logits"]
    assert logits_2.shape == (batch_size_2, seq_len_2, config.vocab_size)
    assert torch.isfinite(logits_2).all()
