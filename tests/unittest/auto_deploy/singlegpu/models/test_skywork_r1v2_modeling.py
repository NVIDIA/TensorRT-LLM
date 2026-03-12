# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for Skywork-R1V2 custom model implementation.

Skywork-R1V2-38B is an InternVL-based VLM with a Qwen2 LLM backbone.
The AD custom model only exports the LLM text path.  The LLM is standard
Qwen2 (GQA with bias on Q/K/V, SwiGLU MLP, RMSNorm, RoPE).

Reference implementations for equivalence tests are imported directly from
transformers.models.qwen2.modeling_qwen2.
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers import AutoConfig, Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_skywork_r1v2 import (
    SkyworkR1V2Attention,
    SkyworkR1V2DecoderLayer,
    SkyworkR1V2ForConditionalGeneration,
    SkyworkR1V2MLP,
    SkyworkR1V2RMSNorm,
    SkyworkR1V2RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

# Load SkyworkChatConfig the same way AutoDeploy's factory does: via AutoConfig with
# trust_remote_code.  Skip all tests if the checkpoint is not in the local HF cache.
try:
    SkyworkChatConfig = type(
        AutoConfig.from_pretrained(
            "Skywork/Skywork-R1V2-38B", trust_remote_code=True, local_files_only=True
        )
    )
except Exception:
    SkyworkChatConfig = None

if SkyworkChatConfig is None:
    pytest.skip(
        "Skywork/Skywork-R1V2-38B not found in local HF cache; skipping tests.",
        allow_module_level=True,
    )

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Small test config
# ---------------------------------------------------------------------------


def _create_small_llm_config() -> Qwen2Config:
    """Create a small Qwen2 config for the LLM backbone (used by block/layer tests)."""
    return Qwen2Config(
        architectures=["Qwen2ForCausalLM"],  # required by SkyworkChatConfig.__init__
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        tie_word_embeddings=False,
    )


def _create_small_chat_config() -> SkyworkChatConfig:
    """Create a small SkyworkChatConfig wrapping the Qwen2 LLM config.

    Mirrors how AutoDeploy's factory builds the model: AutoConfig returns a
    SkyworkChatConfig, which is then passed to SkyworkR1V2ForConditionalGeneration._from_config.
    tie_word_embeddings is forwarded explicitly to prevent PretrainedConfig from
    defaulting to True and spuriously tying lm_head to embed_tokens.
    """
    llm_dict = _create_small_llm_config().to_dict()
    return SkyworkChatConfig(
        llm_config=llm_dict,
        tie_word_embeddings=llm_dict.get("tie_word_embeddings", False),
    )


def _convert_hf_to_custom_state_dict(hf_state_dict):
    """Convert Qwen2ForCausalLM state dict keys to our custom model hierarchy.

    Qwen2ForCausalLM: model.embed_tokens.weight, model.layers.0.*, lm_head.weight
    Custom model:     language_model.model.embed_tokens.weight, ..., language_model.lm_head.weight
    """
    return {f"language_model.{key}": value for key, value in hf_state_dict.items()}


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_skywork_r1v2_rmsnorm_equivalence(B, S, dtype):
    """Test RMSNorm produces numerically equivalent output to HF Qwen2 RMSNorm."""
    device = "cpu"
    config = _create_small_llm_config()

    hf_norm = (
        Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        .to(device=device, dtype=dtype)
        .eval()
    )
    custom_norm = SkyworkR1V2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
        device=device, dtype=dtype
    )
    custom_norm.load_state_dict(hf_norm.state_dict())
    custom_norm.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_norm(x), hf_norm(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_skywork_r1v2_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF Qwen2 MLP."""
    device = "cpu"
    config = _create_small_llm_config()

    hf_mlp = Qwen2MLP(config).to(device=device, dtype=dtype).eval()
    custom_mlp = SkyworkR1V2MLP(config).to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_mlp(x), hf_mlp(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_skywork_r1v2_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF Qwen2 Attention."""
    device = "cuda"
    config = _create_small_llm_config()
    config._attn_implementation = "eager"

    hf_attn = Qwen2Attention(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    custom_attn = SkyworkR1V2Attention(config, layer_idx=0).to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_rotary = Qwen2RotaryEmbedding(config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    head_dim = config.hidden_size // config.num_attention_heads
    custom_rotary = SkyworkR1V2RotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    ).to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Build causal mask in additive form to match AD's is_causal=True behaviour.
    # HF eager attention adds this to QK^T before softmax; shape [1, 1, S, S].
    causal_mask = (
        torch.triu(torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    hf_out, _ = hf_attn(x, (hf_cos, hf_sin), attention_mask=causal_mask)
    custom_out = custom_attn(x, (custom_cos, custom_sin), position_ids)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_skywork_r1v2_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF Qwen2."""
    device = "cuda"
    config = _create_small_llm_config()
    config._attn_implementation = "eager"

    hf_layer = Qwen2DecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    custom_layer = SkyworkR1V2DecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_rotary = Qwen2RotaryEmbedding(config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    head_dim = config.hidden_size // config.num_attention_heads
    custom_rotary = SkyworkR1V2RotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    ).to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Build causal mask in additive form to match AD's is_causal=True behaviour.
    causal_mask = (
        torch.triu(torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=(hf_cos, hf_sin),
    )
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]
    custom_out = custom_layer(x, (custom_cos, custom_sin), position_ids)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu"])
@torch.no_grad()
def test_skywork_r1v2_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF Qwen2ForCausalLM."""
    config = _create_small_llm_config()
    chat_config = _create_small_chat_config()

    hf_model = Qwen2ForCausalLM(config).to(device=device, dtype=dtype).eval()

    custom_model = SkyworkR1V2ForConditionalGeneration(chat_config).to(device=device, dtype=dtype)
    custom_model.load_state_dict(
        _convert_hf_to_custom_state_dict(hf_model.state_dict()), strict=False
    )
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_logits = hf_model(input_ids=input_ids, position_ids=position_ids).logits
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits.float(),
        hf_logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model logits: ",
    )


# =========================================================================
# Export test (Level 4)
# =========================================================================


@torch.no_grad()
def test_skywork_r1v2_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    chat_config = _create_small_chat_config()

    model = SkyworkR1V2ForConditionalGeneration(chat_config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, 1000, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    eager_out = model(input_ids=input_ids, position_ids=position_ids)

    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    move_to_device(gm, device)

    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm, "Output should contain 'logits' key"
    logits = out_gm["logits"]
    assert logits.shape == (B, S, 1000)
    assert_rmse_close(
        logits.float(), eager_out.logits.float(), rmse_ratio_tol=0.05, msg="Export vs eager: "
    )

    # Test dynamic shapes with different input
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, 1000, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)
    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    assert out_gm2["logits"].shape == (B2, S2, 1000)
    assert_rmse_close(
        out_gm2["logits"].float(),
        eager_out2.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager (dynamic shape): ",
    )


# =========================================================================
# Structural tests
# =========================================================================


def test_skywork_r1v2_config_parsing():
    """Test that SkyworkChatConfig correctly wraps the llm_config as a Qwen2Config."""
    config = _create_small_chat_config()
    assert config.model_type == "skywork_chat"
    assert isinstance(config.llm_config, Qwen2Config)
    assert config.llm_config.hidden_size == 64
    assert config.llm_config.num_attention_heads == 4
    assert config.llm_config.num_key_value_heads == 2


def test_skywork_r1v2_gqa_structure():
    """Test that attention uses GQA with bias on QKV."""
    model = SkyworkR1V2ForConditionalGeneration(_create_small_chat_config())

    attn = model.language_model.model.layers[0].self_attn
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 2
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.o_proj.bias is None


def test_skywork_r1v2_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format.

    With a full SkyworkChatConfig (which includes vision_config), the model also
    instantiates the vision tower and mlp1 projector.  Their keys follow the HF
    checkpoint layout: vision_model.* and mlp1.*.
    """
    model = SkyworkR1V2ForConditionalGeneration(_create_small_chat_config())
    state_dict = model.state_dict()

    # LLM backbone keys
    expected_llm_keys = [
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.bias",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.bias",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.bias",
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    ]
    for key in expected_llm_keys:
        assert key in state_dict, (
            f"Expected LLM key '{key}' in state_dict, got keys: {list(state_dict.keys())[:10]}..."
        )

    # Vision tower keys (present because _create_small_chat_config includes vision_config)
    expected_vision_keys = [
        "vision_model.embeddings.class_embedding",
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.position_embedding",
        "vision_model.encoder.layers.0.attn.qkv.weight",
        "vision_model.encoder.layers.0.attn.proj.weight",
        "vision_model.encoder.layers.0.norm1.weight",
        "vision_model.encoder.layers.0.norm2.weight",
        "vision_model.encoder.layers.0.ls1",
        "vision_model.encoder.layers.0.ls2",
        "mlp1.0.weight",  # LayerNorm
        "mlp1.0.bias",
        "mlp1.1.weight",  # Linear
        "mlp1.1.bias",
        "mlp1.3.weight",  # Linear
        "mlp1.3.bias",
    ]
    for key in expected_vision_keys:
        assert key in state_dict, (
            f"Expected vision key '{key}' in state_dict, got keys: {list(state_dict.keys())[:10]}..."
        )

    # All keys must be under language_model.*, vision_model.*, or mlp1.*
    valid_prefixes = ("language_model.", "vision_model.", "mlp1.")
    for key in state_dict:
        assert any(key.startswith(p) for p in valid_prefixes), (
            f"Unexpected key '{key}' — expected prefix in {valid_prefixes}"
        )
