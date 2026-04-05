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

"""Tests for HunYuan Dense custom model implementation.

Hierarchical test levels:
1. Block equivalence — MLP, RMSNorm, Attention individually
2. Layer equivalence — Full decoder layer
3. Full model equivalence — End-to-end logits comparison
4. Export test — torch_export_to_gm with dynamic shapes
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers import PretrainedConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_hunyuan_dense import (
    HunYuanDenseAttention,
    HunYuanDenseDecoderLayer,
    HunYuanDenseForCausalLM,
    HunYuanDenseMLP,
    HunYuanDenseRMSNorm,
    HunYuanDenseRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# HF reference helpers
# ---------------------------------------------------------------------------


def _get_hf_config_class():
    try:
        from transformers.models.hunyuan_v1_dense.configuration_hunyuan_v1_dense import (
            HunYuanDenseV1Config,
        )

        return HunYuanDenseV1Config
    except ImportError:
        return None


def _get_hf_model_class():
    try:
        from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
            HunYuanDenseV1ForCausalLM,
        )

        return HunYuanDenseV1ForCausalLM
    except ImportError:
        return None


def _get_hf_attention_class():
    try:
        from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
            HunYuanDenseV1Attention,
        )

        return HunYuanDenseV1Attention
    except ImportError:
        return None


def _get_hf_mlp_class():
    try:
        from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1MLP

        return HunYuanDenseV1MLP
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    try:
        from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
            HunYuanDenseV1DecoderLayer,
        )

        return HunYuanDenseV1DecoderLayer
    except ImportError:
        return None


def _get_hf_rope_class():
    try:
        from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
            HunYuanDenseV1RotaryEmbedding,
        )

        return HunYuanDenseV1RotaryEmbedding
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Small config for testing
# ---------------------------------------------------------------------------


def _create_small_hf_config():
    """Create a small HF config for testing."""
    HFConfig = _get_hf_config_class()
    if HFConfig is None:
        return None
    cfg = HFConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_qk_norm=True,
        pad_token_id=0,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )
    # Required for HF attention modules to use eager (non-flash) attention
    cfg._attn_implementation = "eager"
    return cfg


def _create_small_custom_config() -> PretrainedConfig:
    """Create a small custom config for testing."""
    return PretrainedConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        pad_token_id=0,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


# ---------------------------------------------------------------------------
# Weight transfer helpers
# ---------------------------------------------------------------------------


def _transfer_mlp_weights(hf_mlp, custom_mlp):
    """Copy weights from HF MLP to custom MLP."""
    custom_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
    custom_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
    custom_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)


def _transfer_attention_weights(hf_attn, custom_attn):
    """Copy weights from HF attention to custom attention."""
    custom_attn.q_proj.weight.data.copy_(hf_attn.q_proj.weight.data)
    custom_attn.k_proj.weight.data.copy_(hf_attn.k_proj.weight.data)
    custom_attn.v_proj.weight.data.copy_(hf_attn.v_proj.weight.data)
    custom_attn.o_proj.weight.data.copy_(hf_attn.o_proj.weight.data)
    custom_attn.query_layernorm.weight.data.copy_(hf_attn.query_layernorm.weight.data)
    custom_attn.key_layernorm.weight.data.copy_(hf_attn.key_layernorm.weight.data)


def _transfer_decoder_layer_weights(hf_layer, custom_layer):
    """Copy weights from HF decoder layer to custom decoder layer."""
    _transfer_attention_weights(hf_layer.self_attn, custom_layer.self_attn)
    _transfer_mlp_weights(hf_layer.mlp, custom_layer.mlp)
    custom_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
    custom_layer.post_attention_layernorm.weight.data.copy_(
        hf_layer.post_attention_layernorm.weight.data
    )


def _transfer_full_model_weights(hf_model, custom_model):
    """Copy weights from HF model to custom model."""
    custom_model.model.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
    custom_model.model.norm.weight.data.copy_(hf_model.model.norm.weight.data)
    for i in range(len(hf_model.model.layers)):
        _transfer_decoder_layer_weights(hf_model.model.layers[i], custom_model.model.layers[i])
    # lm_head may be tied to embed_tokens
    if not hf_model.config.tie_word_embeddings:
        custom_model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)


# ===========================================================================
# Level 1: Block equivalence tests
# ===========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mlp_equivalence(B, S, dtype):
    """Test MLP block produces identical output to HF reference."""
    HFMLPCls = _get_hf_mlp_class()
    if HFMLPCls is None:
        pytest.skip("HF HunYuanDenseV1MLP not available")

    hf_config = _create_small_hf_config()

    hf_mlp = HFMLPCls(hf_config).to(dtype=dtype)
    hf_mlp.eval()

    custom_mlp = HunYuanDenseMLP(
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
    ).to(dtype=dtype)
    custom_mlp.eval()

    _transfer_mlp_weights(hf_mlp, custom_mlp)

    x = torch.randn(B, S, hf_config.hidden_size, dtype=dtype)
    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_rmsnorm_equivalence(B, S, dtype):
    """Test RMSNorm produces identical output to HF reference."""
    try:
        from transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense import (
            HunYuanDenseV1RMSNorm as HFRMSNorm,
        )
    except ImportError:
        pytest.skip("HF HunYuanDenseV1RMSNorm not available")

    hidden_size = 64
    eps = 1e-5

    hf_norm = HFRMSNorm(hidden_size, eps=eps).to(dtype=dtype)
    hf_norm.eval()

    custom_norm = HunYuanDenseRMSNorm(hidden_size, eps=eps).to(dtype=dtype)
    custom_norm.eval()
    custom_norm.weight.data.copy_(hf_norm.weight.data)

    x = torch.randn(B, S, hidden_size, dtype=dtype)
    hf_out = hf_norm(x)
    custom_out = custom_norm(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_attention_equivalence(B, S, dtype):
    """Test attention block equivalence with HF reference."""
    HFAttnCls = _get_hf_attention_class()
    HFRopeCls = _get_hf_rope_class()
    if HFAttnCls is None or HFRopeCls is None:
        pytest.skip("HF HunYuanDenseV1 attention classes not available")

    hf_config = _create_small_hf_config()

    # HF attention
    hf_attn = HFAttnCls(hf_config, layer_idx=0).to(dtype=dtype)
    hf_attn.eval()

    # Custom attention
    custom_attn = HunYuanDenseAttention(
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=hf_config.head_dim,
        rms_norm_eps=hf_config.rms_norm_eps,
        attention_bias=hf_config.attention_bias,
        layer_idx=0,
    ).to(dtype=dtype)
    custom_attn.eval()

    _transfer_attention_weights(hf_attn, custom_attn)

    # Create inputs
    x = torch.randn(B, S, hf_config.hidden_size, dtype=dtype)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    # HF reference — pass causal mask to match custom model's is_causal=True
    hf_rope = HFRopeCls(hf_config).to(dtype=dtype)
    hf_cos, hf_sin = hf_rope(x, position_ids)
    # Causal additive mask: 0 for attended, -inf for masked (future) positions
    causal_mask = torch.full((S, S), float("-inf"), dtype=torch.float32)
    causal_mask = torch.triu(causal_mask, diagonal=1)  # upper-tri = -inf
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)  # [B, 1, S, S]
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=causal_mask,
    )

    # Custom
    custom_rope = HunYuanDenseRotaryEmbedding(
        dim=hf_config.head_dim,
        max_position_embeddings=hf_config.max_position_embeddings,
        base=hf_config.rope_theta,
        rope_scaling=hf_config.rope_scaling,
    ).to(dtype=dtype)
    custom_position_embeddings = custom_rope(x)
    custom_out = custom_attn(x, position_ids, custom_position_embeddings)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# ===========================================================================
# Level 2: Layer equivalence tests
# ===========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer equivalence with HF reference."""
    HFLayerCls = _get_hf_decoder_layer_class()
    HFRopeCls = _get_hf_rope_class()
    if HFLayerCls is None or HFRopeCls is None:
        pytest.skip("HF HunYuanDenseV1 decoder layer classes not available")

    hf_config = _create_small_hf_config()
    custom_config = _create_small_custom_config()

    hf_layer = HFLayerCls(hf_config, layer_idx=0).to(dtype=dtype)
    hf_layer.eval()

    custom_layer = HunYuanDenseDecoderLayer(custom_config, layer_idx=0).to(dtype=dtype)
    custom_layer.eval()

    _transfer_decoder_layer_weights(hf_layer, custom_layer)

    # Create inputs
    x = torch.randn(B, S, hf_config.hidden_size, dtype=dtype)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    # HF reference — pass causal mask to match custom model's is_causal=True
    hf_rope = HFRopeCls(hf_config).to(dtype=dtype)
    hf_cos, hf_sin = hf_rope(x, position_ids)
    # Causal additive mask: 0 for attended, -inf for masked (future) positions
    causal_mask = torch.full((S, S), float("-inf"), dtype=torch.float32)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)  # [B, 1, S, S]
    # HF decoder layer returns a tuple; unpack the hidden states
    hf_out = hf_layer(
        hidden_states=x,
        position_ids=position_ids,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=causal_mask,
    )
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    # Custom
    custom_rope = HunYuanDenseRotaryEmbedding(
        dim=custom_config.head_dim,
        max_position_embeddings=custom_config.max_position_embeddings,
        base=custom_config.rope_theta,
        rope_scaling=custom_config.rope_scaling,
    ).to(dtype=dtype)
    custom_position_embeddings = custom_rope(x)
    custom_out = custom_layer(x, position_ids, custom_position_embeddings)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="DecoderLayer: ")


# ===========================================================================
# Level 3: Full model equivalence tests
# ===========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_full_model_equivalence(B, S, dtype):
    """Test full model logits equivalence with HF reference."""
    HFModelCls = _get_hf_model_class()
    if HFModelCls is None:
        pytest.skip("HF HunYuanDenseV1ForCausalLM not available")

    hf_config = _create_small_hf_config()
    custom_config = _create_small_custom_config()

    hf_model = HFModelCls(hf_config).to(dtype=dtype)
    hf_model.eval()

    custom_model = HunYuanDenseForCausalLM(custom_config).to(dtype=dtype)
    custom_model.eval()

    _transfer_full_model_weights(hf_model, custom_model)

    input_ids = torch.randint(0, custom_config.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(custom_out.logits, hf_out.logits, rmse_ratio_tol=0.05, msg="FullModel: ")


# ===========================================================================
# Level 4: Export test
# ===========================================================================


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@torch.no_grad()
def test_model_can_be_exported(device):
    """Test that the custom model can be exported with torch_export_to_gm.

    Verifies:
    1. The model exports successfully without graph breaks
    2. Exported graph module produces numerically equivalent output to eager model
    3. Dynamic shapes work with different batch/sequence sizes
    """
    dtype = torch.bfloat16
    custom_config = _create_small_custom_config()

    model = HunYuanDenseForCausalLM(custom_config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, custom_config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Get eager model output for comparison
    eager_out = model(input_ids=input_ids, position_ids=position_ids)

    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    dynamic_shapes = {
        "input_ids": {0: batch_size_dynamic, 1: seq_len_dynamic},
        "position_ids": {0: batch_size_dynamic, 1: seq_len_dynamic},
    }

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
    assert logits.shape == (B, S, custom_config.vocab_size)
    assert torch.isfinite(logits).all(), "Exported logits should be finite"

    # Compare exported vs eager output
    assert_rmse_close(logits, eager_out.logits, rmse_ratio_tol=0.05, msg="Export: ")

    # Test with different shape to verify dynamic dims produce correct results
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, custom_config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, custom_config.vocab_size)
    assert_rmse_close(logits2, eager_out2.logits, rmse_ratio_tol=0.05, msg="Export dynamic shape: ")


# ===========================================================================
# Registration and structure tests
# ===========================================================================


def test_config_registration():
    """Factory knows the model class under the real HF config name."""
    from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

    assert "HunYuanDenseV1Config" in AutoModelForCausalLMFactory._custom_model_mapping


def test_tied_weights():
    """Test that tie_word_embeddings works correctly."""
    config = _create_small_custom_config()
    model = HunYuanDenseForCausalLM(config)

    # With tied weights, lm_head.weight should be the same tensor as embed_tokens.weight
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_qk_norm_structure():
    """Test that QK normalization layers exist in attention."""
    config = _create_small_custom_config()
    model = HunYuanDenseForCausalLM(config)

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        assert hasattr(attn, "query_layernorm"), f"Layer {i} missing query_layernorm"
        assert hasattr(attn, "key_layernorm"), f"Layer {i} missing key_layernorm"
        assert isinstance(attn.query_layernorm, HunYuanDenseRMSNorm)
        assert isinstance(attn.key_layernorm, HunYuanDenseRMSNorm)


def test_state_dict_keys_match_checkpoint():
    """Test that model state_dict keys match the expected checkpoint format."""
    config = _create_small_custom_config()
    model = HunYuanDenseForCausalLM(config)
    state_dict = model.state_dict()

    expected_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.query_layernorm.weight",
        "model.layers.0.self_attn.key_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ]
    for key in expected_keys:
        assert key in state_dict, f"Expected key '{key}' not in state_dict"
