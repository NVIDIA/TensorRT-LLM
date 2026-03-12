# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical equivalence tests for the custom Gemma 3 AutoDeploy model."""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3ForConditionalGeneration,
    Gemma3MLP,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.models.hf import (
    AutoModelForCausalLMFactory,
    AutoModelForImageTextToTextFactory,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device


def _hf_gemma3_modules():
    try:
        from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention as HFGemma3Attention
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3DecoderLayer as HFGemma3DecoderLayer,
        )
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3ForConditionalGeneration as HFGemma3ForConditionalGeneration,
        )
        from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP as HFGemma3MLP
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm as HFGemma3RMSNorm
    except ImportError:
        return None

    return {
        "attention": HFGemma3Attention,
        "decoder_layer": HFGemma3DecoderLayer,
        "for_conditional_generation": HFGemma3ForConditionalGeneration,
        "mlp": HFGemma3MLP,
        "rms_norm": HFGemma3RMSNorm,
    }


def _small_text_config() -> Gemma3TextConfig:
    config = Gemma3TextConfig(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=16,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        rope_scaling={"rope_type": "linear", "factor": 2.0},
        rope_local_base_freq=10000.0,
        use_bidirectional_attention=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return config


def _small_full_config() -> Gemma3Config:
    vision_config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
    )
    return Gemma3Config(text_config=_small_text_config(), vision_config=vision_config)


def _make_position_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.LongTensor:
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)


def _make_attention_mask(
    seq_len: int, sliding_window: int | None, device: torch.device
) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    allowed = pos_diff >= 0
    if sliding_window is not None:
        allowed = allowed & (pos_diff < sliding_window)

    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.where(allowed, torch.zeros_like(mask), mask)
    return mask.unsqueeze(0).unsqueeze(0)


def _make_shared_inputs(config: Gemma3TextConfig, dtype: torch.dtype, device: torch.device):
    batch_size, seq_len = 2, 6
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=dtype, device=device)
    position_ids = _make_position_ids(batch_size, seq_len, device)
    return hidden_states, position_ids


def _slice_position_embeddings(
    position_embeddings: tuple[torch.Tensor, torch.Tensor], position_ids: torch.LongTensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    flat_position_ids = position_ids.reshape(-1)
    cos = cos.index_select(0, flat_position_ids).view(*position_ids.shape, -1)
    sin = sin.index_select(0, flat_position_ids).view(*position_ids.shape, -1)
    return cos, sin


def _remap_custom_state_dict_to_hf(
    custom_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in custom_state_dict.items():
        new_key = key
        if key.startswith("vision_tower."):
            new_key = f"model.{key}"
        elif key.startswith("multi_modal_projector."):
            new_key = f"model.{key}"
        elif key.startswith("language_model.model."):
            new_key = key.replace("language_model.model.", "model.language_model.", 1)
        elif key.startswith("language_model.lm_head."):
            new_key = key.replace("language_model.lm_head.", "lm_head.", 1)
        remapped[new_key] = value
    return remapped


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


def test_gemma3_rms_norm_equivalence():
    modules = _hf_gemma3_modules()
    if modules is None:
        pytest.skip("HF Gemma3 reference modules are unavailable")

    dim = 64
    device = torch.device("cpu")
    dtype = torch.float32
    custom = Gemma3RMSNorm(dim).to(device=device, dtype=dtype)
    reference = modules["rms_norm"](dim).to(device=device, dtype=dtype)
    reference.load_state_dict(custom.state_dict())

    hidden_states = torch.randn(2, 5, dim, dtype=dtype, device=device)
    custom_out = custom(hidden_states)
    reference_out = reference(hidden_states)
    torch.testing.assert_close(custom_out, reference_out)


def test_gemma3_mlp_equivalence():
    modules = _hf_gemma3_modules()
    if modules is None:
        pytest.skip("HF Gemma3 reference modules are unavailable")

    config = _small_text_config()
    device = torch.device("cpu")
    dtype = torch.float32
    custom = Gemma3MLP(config).to(device=device, dtype=dtype)
    reference = modules["mlp"](config).to(device=device, dtype=dtype)
    reference.load_state_dict(custom.state_dict())

    hidden_states = torch.randn(2, 5, config.hidden_size, dtype=dtype, device=device)
    custom_out = custom(hidden_states)
    reference_out = reference(hidden_states)
    torch.testing.assert_close(custom_out, reference_out)


@torch.no_grad()
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_gemma3_attention_equivalence(layer_idx: int):
    modules = _hf_gemma3_modules()
    if modules is None:
        pytest.skip("HF Gemma3 reference modules are unavailable")

    config = _small_text_config()
    device = torch.device("cpu")
    dtype = torch.float32
    custom = Gemma3Attention(config, layer_idx=layer_idx).to(device=device, dtype=dtype)
    reference = modules["attention"](config, layer_idx=layer_idx).to(device=device, dtype=dtype)
    reference.load_state_dict(custom.state_dict())

    hidden_states, position_ids = _make_shared_inputs(config, dtype=dtype, device=device)
    rotary_emb = Gemma3RotaryEmbedding(config)
    local_rotary_emb = Gemma3RotaryEmbedding(
        config=config,
        rope_theta=config.rope_local_base_freq,
        rope_scaling={"rope_type": "default"},
    )
    if config.layer_types[layer_idx] == "sliding_attention":
        position_embeddings = local_rotary_emb(dtype=dtype)
        attention_mask = _make_attention_mask(
            seq_len=hidden_states.shape[1], sliding_window=config.sliding_window, device=device
        )
    else:
        position_embeddings = rotary_emb(dtype=dtype)
        attention_mask = _make_attention_mask(
            seq_len=hidden_states.shape[1], sliding_window=None, device=device
        )

    custom_out = custom(
        hidden_states=hidden_states,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    reference_position_embeddings = _slice_position_embeddings(position_embeddings, position_ids)
    reference_out, _ = reference(
        hidden_states=hidden_states,
        position_embeddings=reference_position_embeddings,
        attention_mask=attention_mask,
    )
    assert_rmse_close(custom_out, reference_out, rmse_ratio_tol=0.10, msg="Attention: ")


@torch.no_grad()
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_gemma3_decoder_layer_equivalence(layer_idx: int):
    modules = _hf_gemma3_modules()
    if modules is None:
        pytest.skip("HF Gemma3 reference modules are unavailable")

    config = _small_text_config()
    device = torch.device("cpu")
    dtype = torch.float32
    custom = Gemma3DecoderLayer(config, layer_idx=layer_idx).to(device=device, dtype=dtype)
    reference = modules["decoder_layer"](config, layer_idx=layer_idx).to(device=device, dtype=dtype)
    reference.load_state_dict(custom.state_dict())

    hidden_states, position_ids = _make_shared_inputs(config, dtype=dtype, device=device)
    rotary_emb = Gemma3RotaryEmbedding(config)
    local_rotary_emb = Gemma3RotaryEmbedding(
        config=config,
        rope_theta=config.rope_local_base_freq,
        rope_scaling={"rope_type": "default"},
    )
    position_embeddings_global = rotary_emb(dtype=dtype)
    position_embeddings_local = local_rotary_emb(dtype=dtype)
    attention_mask = _make_attention_mask(
        seq_len=hidden_states.shape[1],
        sliding_window=(
            config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        ),
        device=device,
    )

    custom_out = custom(
        hidden_states=hidden_states,
        position_ids=position_ids,
        position_embeddings_global=position_embeddings_global,
        position_embeddings_local=position_embeddings_local,
    )
    reference_position_embeddings_global = _slice_position_embeddings(
        position_embeddings_global, position_ids
    )
    reference_position_embeddings_local = _slice_position_embeddings(
        position_embeddings_local, position_ids
    )
    reference_out = reference(
        hidden_states=hidden_states,
        position_embeddings_global=reference_position_embeddings_global,
        position_embeddings_local=reference_position_embeddings_local,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=False,
        use_cache=False,
    )[0]
    assert_rmse_close(custom_out, reference_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


@torch.no_grad()
def test_gemma3_full_model_equivalence():
    modules = _hf_gemma3_modules()
    if modules is None:
        pytest.skip("HF Gemma3 reference modules are unavailable")

    config = _small_full_config()
    device = torch.device("cpu")
    dtype = torch.float32
    custom = Gemma3ForConditionalGeneration(config).to(device=device, dtype=dtype)
    reference = modules["for_conditional_generation"](config).to(device=device, dtype=dtype)

    load_result = reference.load_state_dict(
        _remap_custom_state_dict_to_hf(custom.state_dict()), strict=True
    )
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(
        0, config.text_config.vocab_size, (batch_size, seq_len), device=device
    )
    position_ids = _make_position_ids(batch_size, seq_len, device)

    custom_out = custom(input_ids=input_ids, position_ids=position_ids)
    reference_out = reference(
        input_ids=input_ids,
        position_ids=position_ids,
        pixel_values=None,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=False,
    )
    assert_rmse_close(
        custom_out.logits, reference_out.logits, rmse_ratio_tol=0.05, msg="Full model: "
    )


def test_gemma3_weight_layout_and_registration():
    model = Gemma3ForConditionalGeneration(_small_full_config())
    state_dict = model.state_dict()

    assert any(key.startswith("language_model.model.layers.0") for key in state_dict)
    assert any(key.startswith("language_model.lm_head.weight") for key in state_dict)
    assert any(key.startswith("vision_tower.vision_model.embeddings") for key in state_dict)
    assert any(
        key.startswith("multi_modal_projector.mm_input_projection_weight") for key in state_dict
    )
    assert (
        AutoModelForCausalLMFactory._custom_model_mapping["Gemma3Config"]
        == Gemma3ForConditionalGeneration
    )
    assert (
        AutoModelForImageTextToTextFactory._custom_model_mapping["Gemma3Config"]
        == Gemma3ForConditionalGeneration
    )


def test_gemma3_rope_buffers():
    rope = Gemma3RotaryEmbedding(_small_text_config())
    assert hasattr(rope, "_ad_cos_cached")
    assert hasattr(rope, "_ad_sin_cached")
    assert rope._ad_cos_cached.shape == (128, 16)
    assert rope._ad_sin_cached.shape == (128, 16)


def test_gemma3_vlm_wrapper_does_not_forward_non_text_kwargs():
    model = Gemma3ForConditionalGeneration(_small_full_config())
    captured_kwargs = {}

    def _capture_kwargs(_module, args, kwargs):
        assert not args
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)

    handle = model.language_model.register_forward_pre_hook(
        _capture_kwargs, prepend=True, with_kwargs=True
    )
    try:
        batch_size, seq_len = 2, 6
        input_ids = torch.randint(0, model.config.text_config.vocab_size, (batch_size, seq_len))
        position_ids = _make_position_ids(batch_size, seq_len, input_ids.device)
        cache_position = torch.arange(seq_len, device=input_ids.device)
        attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)

        model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
    finally:
        handle.remove()

    assert set(captured_kwargs) == {"input_ids", "position_ids"}
    torch.testing.assert_close(captured_kwargs["input_ids"], input_ids)
    torch.testing.assert_close(captured_kwargs["position_ids"], position_ids)


@torch.no_grad()
def test_gemma3_model_can_be_exported():
    device = torch.device("cpu")
    dtype = torch.float32
    config = _small_full_config()
    model = Gemma3ForConditionalGeneration(config).to(device=device, dtype=dtype)
    model.eval()

    batch_size, seq_len = 2, 6
    input_ids = torch.randint(
        0, config.text_config.vocab_size, (batch_size, seq_len), device=device
    )
    position_ids = _make_position_ids(batch_size, seq_len, device)

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

    out = gm(input_ids=input_ids, position_ids=position_ids)
    assert "logits" in out
    logits = out["logits"]
    assert logits.shape == (batch_size, seq_len, config.text_config.vocab_size)
    assert torch.isfinite(logits).all()

    input_ids_2 = torch.randint(0, config.text_config.vocab_size, (1, 4), device=device)
    position_ids_2 = _make_position_ids(1, 4, device)
    out_2 = gm(input_ids=input_ids_2, position_ids=position_ids_2)
    logits_2 = out_2["logits"]
    assert logits_2.shape == (1, 4, config.text_config.vocab_size)
    assert torch.isfinite(logits_2).all()
