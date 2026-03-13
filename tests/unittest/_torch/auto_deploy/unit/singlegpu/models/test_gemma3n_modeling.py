# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Tuple

import pytest
import torch
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_gemma3n import (
    Gemma3nAudioConfig,
    Gemma3nConditionalOutput,
    Gemma3nConfig,
    Gemma3nForCausalLM,
    Gemma3nForConditionalGeneration,
    Gemma3nTextAttention,
    Gemma3nTextConfig,
    Gemma3nTextDecoderLayer,
    Gemma3nTextMLP,
    Gemma3nVisionConfig,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device


def assert_rmse_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rmse_ratio_tol: float,
    msg: str = "",
) -> None:
    diff = actual.float() - expected.float()
    rmse_diff = torch.sqrt(torch.mean(diff**2))
    rmse_ref = torch.sqrt(torch.mean(expected.float() ** 2))
    ratio = (rmse_diff / rmse_ref).item()
    assert ratio < rmse_ratio_tol, (
        f"{msg}RMSE ratio {ratio:.6f} exceeds tolerance {rmse_ratio_tol}. "
        f"(rmse_diff={rmse_diff.item():.6f}, rmse_ref={rmse_ref.item():.6f})"
    )


def _get_hf_classes():
    try:
        from transformers.models.gemma3n.modeling_gemma3n import (
            Gemma3nForCausalLM as HFGemma3nForCausalLM,
        )
        from transformers.models.gemma3n.modeling_gemma3n import (
            Gemma3nTextAttention as HFGemma3nTextAttention,
        )
        from transformers.models.gemma3n.modeling_gemma3n import (
            Gemma3nTextDecoderLayer as HFGemma3nTextDecoderLayer,
        )
        from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextMLP as HFGemma3nTextMLP
    except ImportError:
        return None
    return HFGemma3nForCausalLM, HFGemma3nTextAttention, HFGemma3nTextDecoderLayer, HFGemma3nTextMLP


HF_CLASSES = _get_hf_classes()


def _device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _small_text_config() -> Gemma3nTextConfig:
    config = Gemma3nTextConfig(
        vocab_size=256,
        vocab_size_per_layer_input=256,
        hidden_size=64,
        hidden_size_per_layer_input=8,
        intermediate_size=[128, 128, 128],
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_local_base_freq=1000.0,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=16,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        final_logit_softcapping=30.0,
        altup_active_idx=0,
        altup_correct_scale=True,
        altup_num_inputs=3,
        num_kv_shared_layers=0,
        laurel_rank=8,
        activation_sparsity_pattern=[0.5, 0.0, 0.0],
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    config._attn_implementation = "eager"
    return config


def _small_full_config() -> Gemma3nConfig:
    return Gemma3nConfig(
        text_config=_small_text_config(),
        vision_config=Gemma3nVisionConfig(
            hidden_size=32,
            vocab_size=8,
            vocab_offset=256,
            rms_norm_eps=1e-6,
        ),
        audio_config=Gemma3nAudioConfig(
            vocab_size=8,
            vocab_offset=264,
            hidden_size=32,
            rms_norm_eps=1e-6,
            conf_num_attention_heads=4,
            conf_num_hidden_layers=2,
            sscp_conv_channel_size=(16, 8),
        ),
    )


def _extended_text_config(num_hidden_layers: int) -> Gemma3nTextConfig:
    config = copy.deepcopy(_small_text_config())
    config.num_hidden_layers = num_hidden_layers
    config.intermediate_size = [128] * num_hidden_layers
    config.layer_types = ["sliding_attention"] * (num_hidden_layers - 1) + ["full_attention"]
    config.activation_sparsity_pattern = [0.0] * num_hidden_layers
    return config


def _shared_kv_text_config() -> Gemma3nTextConfig:
    config = Gemma3nTextConfig(
        vocab_size=256,
        vocab_size_per_layer_input=256,
        hidden_size=64,
        hidden_size_per_layer_input=8,
        intermediate_size=[128] * 6,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_local_base_freq=1000.0,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=16,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        final_logit_softcapping=30.0,
        altup_active_idx=0,
        altup_correct_scale=True,
        altup_num_inputs=3,
        num_kv_shared_layers=2,
        laurel_rank=8,
        activation_sparsity_pattern=[0.0] * 6,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    config._attn_implementation = "eager"
    return config


def _position_ids(batch_size: int, seq_len: int, device: str) -> torch.Tensor:
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)


def _load_equivalent_modules(custom_module: torch.nn.Module, hf_module: torch.nn.Module) -> None:
    missing, unexpected = custom_module.load_state_dict(hf_module.state_dict(), strict=False)
    assert not missing, f"Missing keys when loading HF weights into custom module: {missing}"
    assert not unexpected, (
        f"Unexpected keys when loading HF weights into custom module: {unexpected}"
    )


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(42)


def test_hf_reference_available():
    if HF_CLASSES is None:
        pytest.skip("transformers gemma3n reference classes are unavailable")
    hf_model_cls, hf_attention_cls, hf_layer_cls, hf_mlp_cls = HF_CLASSES
    assert hf_model_cls.__name__ == "Gemma3nForCausalLM"
    assert hf_attention_cls.__name__ == "Gemma3nTextAttention"
    assert hf_layer_cls.__name__ == "Gemma3nTextDecoderLayer"
    assert hf_mlp_cls.__name__ == "Gemma3nTextMLP"


@torch.no_grad()
def test_gemma3n_mlp_equivalence():
    if HF_CLASSES is None:
        pytest.skip("transformers gemma3n reference classes are unavailable")

    _, _, _, hf_mlp_cls = HF_CLASSES
    device, dtype = _device_and_dtype()
    config = _small_text_config()
    custom_mlp = Gemma3nTextMLP(config, layer_idx=0).to(device=device, dtype=dtype)
    hf_mlp = hf_mlp_cls(config, layer_idx=0).to(device=device, dtype=dtype)
    _load_equivalent_modules(custom_mlp, hf_mlp)

    hidden_states = torch.randn(2, 6, config.hidden_size, device=device, dtype=dtype)
    custom_out = custom_mlp(hidden_states)
    hf_out = hf_mlp(hidden_states)
    torch.testing.assert_close(custom_out.float(), hf_out.float(), rtol=1e-3, atol=1e-3)


@torch.no_grad()
def test_gemma3n_attention_equivalence():
    if HF_CLASSES is None:
        pytest.skip("transformers gemma3n reference classes are unavailable")

    _, hf_attention_cls, _, _ = HF_CLASSES
    device, dtype = _device_and_dtype()
    config = _small_text_config()
    custom_attn = Gemma3nTextAttention(config, layer_idx=2).to(device=device, dtype=dtype)
    hf_attn = hf_attention_cls(config, layer_idx=2).to(device=device, dtype=dtype)
    _load_equivalent_modules(custom_attn, hf_attn)

    hidden_states = torch.randn(2, 6, config.hidden_size, device=device, dtype=dtype)
    position_ids = _position_ids(2, 6, device)
    custom_rope = Gemma3nForCausalLM(config).model.rotary_emb.to(device=device)
    full_cos, full_sin = custom_rope(hidden_states, position_ids)
    position_embeddings = (full_cos[position_ids], full_sin[position_ids])

    custom_out = custom_attn(hidden_states, position_embeddings)
    hf_out = hf_attn(hidden_states, position_embeddings, attention_mask=None)[0]
    assert_rmse_close(custom_out[:, -1:], hf_out[:, -1:], rmse_ratio_tol=0.10, msg="Attention: ")


@torch.no_grad()
def test_gemma3n_decoder_layer_equivalence():
    if HF_CLASSES is None:
        pytest.skip("transformers gemma3n reference classes are unavailable")

    _, _, hf_layer_cls, _ = HF_CLASSES
    device, dtype = _device_and_dtype()
    config = _small_text_config()
    custom_layer = Gemma3nTextDecoderLayer(config, layer_idx=2).to(device=device, dtype=dtype)
    hf_layer = hf_layer_cls(config, layer_idx=2).to(device=device, dtype=dtype)
    _load_equivalent_modules(custom_layer, hf_layer)

    batch_size, seq_len = 2, 1
    hidden_states = torch.randn(
        config.altup_num_inputs, batch_size, seq_len, config.hidden_size, device=device, dtype=dtype
    )
    per_layer_input = torch.randn(
        batch_size, seq_len, config.hidden_size_per_layer_input, device=device, dtype=dtype
    )
    position_ids = _position_ids(batch_size, seq_len, device)
    rope_model = Gemma3nForCausalLM(config).model.to(device=device)
    global_cos, global_sin = rope_model.rotary_emb(hidden_states[0], position_ids)
    local_cos, local_sin = rope_model.rotary_emb_local(hidden_states[0], position_ids)
    position_embeddings_global = (global_cos[position_ids], global_sin[position_ids])
    position_embeddings_local = (local_cos[position_ids], local_sin[position_ids])

    custom_out = custom_layer(
        hidden_states,
        position_embeddings_global,
        position_embeddings_local,
        per_layer_input,
    )
    hf_out = hf_layer(
        hidden_states,
        position_embeddings_global,
        position_embeddings_local,
        per_layer_input,
        attention_mask=None,
        position_ids=position_ids,
    )[0]
    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


@torch.no_grad()
def test_gemma3n_full_model_equivalence():
    if HF_CLASSES is None:
        pytest.skip("transformers gemma3n reference classes are unavailable")

    hf_model_cls, _, _, _ = HF_CLASSES
    device, dtype = "cpu", torch.float32
    config = _small_text_config()
    custom_model = Gemma3nForCausalLM(config).to(device=device, dtype=dtype)
    hf_model = hf_model_cls(config).to(device=device, dtype=dtype)
    _load_equivalent_modules(custom_model, hf_model)
    custom_model.eval()
    hf_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 6), device=device)
    position_ids = _position_ids(2, 6, device)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)
    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    assert_rmse_close(custom_out.logits, hf_out.logits, rmse_ratio_tol=0.05, msg="Full model: ")


@torch.no_grad()
def test_gemma3n_conditional_wrapper_equivalence():
    if HF_CLASSES is None:
        pytest.skip("transformers gemma3n reference classes are unavailable")

    hf_model_cls, _, _, _ = HF_CLASSES
    device, dtype = "cpu", torch.float32
    config = _small_full_config()
    wrapper = Gemma3nForConditionalGeneration(config).to(device=device, dtype=dtype)
    hf_model = hf_model_cls(config.text_config).to(device=device, dtype=dtype)
    _load_equivalent_modules(wrapper.model.language_model, hf_model.model)
    _load_equivalent_modules(wrapper.lm_head, hf_model.lm_head)
    wrapper.eval()
    hf_model.eval()

    input_ids = torch.randint(
        0, config.text_config.vocab_size_per_layer_input, (2, 6), device=device
    )
    position_ids = _position_ids(2, 6, device)
    wrapper_out = wrapper(input_ids=input_ids, position_ids=position_ids)
    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    assert isinstance(wrapper_out, Gemma3nConditionalOutput)
    assert_rmse_close(wrapper_out.logits, hf_out.logits, rmse_ratio_tol=0.05, msg="Wrapper: ")


def test_gemma3n_conditional_wrapper_load_hook_drops_unsupported_tower_weights():
    config = _small_full_config()
    wrapper = Gemma3nForConditionalGeneration(config)
    state_dict = wrapper.state_dict()
    state_dict["model.vision_tower.fake.weight"] = torch.randn(2, 2)
    state_dict["model.audio_tower.fake.weight"] = torch.randn(2, 2)

    missing, unexpected = wrapper.load_state_dict(state_dict, strict=True)

    assert missing == []
    assert unexpected == []


def test_gemma3n_conditional_wrapper_ignores_hf_init_kwargs():
    config = _small_full_config()
    wrapper = Gemma3nForConditionalGeneration(config, use_cache=False)
    assert isinstance(wrapper, Gemma3nForConditionalGeneration)


def test_gemma3n_reduced_layer_load_hook_slices_per_layer_weights():
    source_model = Gemma3nForCausalLM(_extended_text_config(5))
    target_model = Gemma3nForCausalLM(_small_text_config())

    missing, unexpected = target_model.load_state_dict(source_model.state_dict(), strict=False)

    assert missing == []
    assert "model.layers.3.self_attn.q_proj.weight" in unexpected


def test_gemma3n_causal_lm_ties_lm_head_to_input_embeddings():
    model = Gemma3nForCausalLM(_small_text_config())
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_gemma3n_conditional_lm_ties_lm_head_to_input_embeddings():
    model = Gemma3nForConditionalGeneration(_small_full_config())
    assert model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()


def test_gemma3n_shared_kv_layer_metadata_matches_config():
    model = Gemma3nForCausalLM(_shared_kv_text_config())
    layer_expectations = [
        (False, None),
        (False, None),
        (False, None),
        (False, None),
        (True, 2),
        (True, 3),
    ]

    for layer, (is_shared, source_idx) in zip(model.model.layers, layer_expectations):
        assert layer.self_attn.is_kv_shared_layer is is_shared
        assert layer.self_attn.kv_shared_layer_index == source_idx


def test_gemma3n_export_uses_shared_kv_attention_for_shared_layers():
    config = _shared_kv_text_config()
    model = Gemma3nForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    position_ids = _position_ids(1, 4, "cpu")

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
    )

    attn_nodes = [node for node in gm.graph.nodes if node.op == "call_function"]
    regular_nodes = [
        node for node in attn_nodes if node.target == torch.ops.auto_deploy.torch_attention.default
    ]
    shared_nodes = [
        node
        for node in attn_nodes
        if node.target == torch.ops.auto_deploy.torch_attention_shared_kv.default
    ]

    assert len(regular_nodes) == config.num_hidden_layers - config.num_kv_shared_layers
    assert len(shared_nodes) == config.num_kv_shared_layers
    assert [regular.args[-1] for regular in regular_nodes] == [0, 1, 2, 3]
    assert [shared.args[-2] for shared in shared_nodes] == [4, 5]
    assert [shared.args[-1] for shared in shared_nodes] == [2, 3]


def test_gemma3n_model_can_be_exported():
    if not torch.cuda.is_available():
        pytest.skip("Export test requires CUDA")

    device = "cuda"
    dtype = torch.bfloat16
    config = _small_full_config()
    model = Gemma3nForConditionalGeneration(config).to(device=device, dtype=dtype)
    model.eval()

    input_ids = torch.randint(0, config.text_config.vocab_size, (2, 8), device=device)
    position_ids = _position_ids(2, 8, device)

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=(
            {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        ),
    )
    move_to_device(gm, device)

    with torch.inference_mode():
        eager_out = model(input_ids=input_ids, position_ids=position_ids)
        export_out = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in export_out
    assert_rmse_close(export_out["logits"], eager_out.logits, rmse_ratio_tol=0.05, msg="Export: ")

    input_ids_2 = torch.randint(0, config.text_config.vocab_size, (1, 5), device=device)
    position_ids_2 = _position_ids(1, 5, device)
    with torch.inference_mode():
        export_out_2 = gm(input_ids=input_ids_2, position_ids=position_ids_2)
        eager_out_2 = model(input_ids=input_ids_2, position_ids=position_ids_2)
    assert_rmse_close(
        export_out_2["logits"], eager_out_2.logits, rmse_ratio_tol=0.05, msg="Export dynamic: "
    )
