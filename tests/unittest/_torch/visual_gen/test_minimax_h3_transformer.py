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

"""Focused architecture tests for the MiniMax-H3 transformer."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("diffusers")

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import AttentionConfig

try:
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel as HFMiniMaxH3Transformer3DModel,
    )
except ImportError:
    HFMiniMaxH3Transformer3DModel = None


_TINY_CONFIG: dict[str, object] = {
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "hidden_size": 12,
    "num_layers": 0,
    "num_refiner_layers": 0,
    "ffn_dim": 16,
    "in_channels": 2,
    "audio_in_channels": 3,
    "patch_size": (1, 1, 1),
    "text_dim": 5,
    "freq_dim": 8,
    "time_embed_hidden_dim": 12,
    "time_embed_dim": 6,
    "rope_freq_dim": 1,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}


def _make_model_config(
    *,
    omitted_fields: tuple[str, ...] = (),
    quant_config: QuantConfig | None = None,
    dynamic_weight_quant: bool = False,
    force_dynamic_quantization: bool = False,
    **overrides: object,
) -> DiffusionModelConfig:
    pretrained_config_dict = _TINY_CONFIG | overrides
    for field in omitted_fields:
        del pretrained_config_dict[field]
    pretrained_config = SimpleNamespace(**pretrained_config_dict)
    return DiffusionModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config or QuantConfig(),
        dynamic_weight_quant=dynamic_weight_quant,
        force_dynamic_quantization=force_dynamic_quantization,
        mapping=Mapping(),
        attention=AttentionConfig(backend="VANILLA"),
    )


def _make_dynamic_fp8_model_config(**overrides: object) -> DiffusionModelConfig:
    return _make_model_config(
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
        dynamic_weight_quant=True,
        **overrides,
    )


@pytest.mark.parametrize("required_field", tuple(_TINY_CONFIG))
def test_transformer_rejects_missing_required_architecture_field(
    required_field: str,
) -> None:
    with pytest.raises(AttributeError, match=required_field):
        h3.MiniMaxH3Transformer3DModel(_make_model_config(omitted_fields=(required_field,)))


def _initialize_weights(module: nn.Module, scale: float = 0.02) -> None:
    generator = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1.0)
            else:
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=parameter.device,
                        generator=generator,
                    )
                    * scale
                )


def test_transformer_accepts_dynamic_fp8_configuration() -> None:
    model = h3.MiniMaxH3Transformer3DModel(_make_dynamic_fp8_model_config())

    assert model.context_embedder.weight.dtype == torch.float8_e4m3fn
    assert not model.context_embedder.force_dynamic_quantization


def test_adaln_modulation_keeps_dynamic_fp8_input_high_precision() -> None:
    class _RecordingFP8Linear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(1, 1, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.dtype = torch.bfloat16
            self.input_dtypes: list[torch.dtype] = []

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            self.input_dtypes.append(hidden_states.dtype)
            return torch.zeros(1, 216, dtype=torch.bfloat16)

    modulation = h3.MiniMaxH3AdaLayerNormModulation(
        time_embed_dim=6,
        hidden_size=12,
        model_config=_make_dynamic_fp8_model_config(),
    )
    spy = _RecordingFP8Linear()
    modulation.linear = spy

    modulation(torch.randn(1, 6))

    assert spy.input_dtypes == [torch.bfloat16]


@pytest.mark.parametrize(
    ("quant_algo", "dynamic_weight_quant"),
    [
        (QuantAlgo.FP8, False),
        (QuantAlgo.FP8_BLOCK_SCALES, True),
        (QuantAlgo.NVFP4, True),
    ],
)
def test_transformer_rejects_unvalidated_quantization_modes(
    quant_algo: QuantAlgo,
    dynamic_weight_quant: bool,
) -> None:
    with pytest.raises(NotImplementedError, match="dynamic per-tensor FP8"):
        h3.MiniMaxH3Transformer3DModel(
            _make_model_config(
                quant_config=QuantConfig(quant_algo=quant_algo),
                dynamic_weight_quant=dynamic_weight_quant,
            )
        )


def test_post_load_preserves_dynamic_fp8_projection_weights() -> None:
    model = h3.MiniMaxH3Transformer3DModel(_make_dynamic_fp8_model_config())

    model.post_load_weights()

    assert model.proj_in.weight.dtype == torch.float8_e4m3fn
    assert model.audio_proj_in.weight.dtype == torch.float8_e4m3fn
    assert model.proj_out.weight.dtype == torch.float8_e4m3fn
    assert model.audio_proj_out.weight.dtype == torch.float8_e4m3fn


def test_transformer_keeps_dynamic_fp8_linear_inputs_high_precision() -> None:
    class _RecordingFP8Linear(nn.Module):
        def __init__(self, out_features: int, output_dtype: torch.dtype) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(out_features, 1, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.dtype = output_dtype
            self.out_features = out_features
            self.output_dtype = output_dtype
            self.input_dtypes: list[torch.dtype] = []

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            self.input_dtypes.append(hidden_states.dtype)
            return torch.zeros(
                (*hidden_states.shape[:-1], self.out_features),
                dtype=self.output_dtype,
                device=hidden_states.device,
            )

    model = h3.MiniMaxH3Transformer3DModel(_make_dynamic_fp8_model_config())
    spies = {
        "context_embedder": _RecordingFP8Linear(12, torch.bfloat16),
        "proj_in": _RecordingFP8Linear(12, torch.float32),
        "audio_proj_in": _RecordingFP8Linear(12, torch.float32),
        "norm_out.linear": _RecordingFP8Linear(24, torch.bfloat16),
        "proj_out": _RecordingFP8Linear(2, torch.float32),
        "audio_proj_out": _RecordingFP8Linear(3, torch.float32),
    }
    model.context_embedder = spies["context_embedder"]
    model.proj_in = spies["proj_in"]
    model.audio_proj_in = spies["audio_proj_in"]
    model.norm_out.linear = spies["norm_out.linear"]
    model.proj_out = spies["proj_out"]
    model.audio_proj_out = spies["audio_proj_out"]

    model(**_model_inputs())

    assert all(
        input_dtype != torch.float8_e4m3fn
        for spy in spies.values()
        for input_dtype in spy.input_dtypes
    )


def _initialize_diffusers_golden_weights(module: nn.Module) -> None:
    """Initialize reproducible weights shared with the pinned Diffusers oracle."""
    offset = 0
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1.0)
            else:
                values = torch.arange(
                    offset,
                    offset + parameter.numel(),
                    dtype=torch.float32,
                    device=parameter.device,
                ).reshape(parameter.shape)
                parameter.copy_((torch.sin(values * 0.137) * 0.02).to(parameter.dtype))
            offset += parameter.numel()


def _diffusers_golden_inputs() -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.tensor([[[-0.75, -0.25], [0.25, 0.75]]]),
        "audio_hidden_states": torch.tensor([[[-0.5, 0.125, 0.875]]]),
        "encoder_hidden_states": torch.tensor([[[0.1, -0.2, 0.3, -0.4, 0.5]]]),
        "timestep": torch.tensor([0.0, 0.75]),
        "timestep_indices": torch.tensor([0, 1, 1, 0]),
        "token_tags": torch.tensor([1, 0, 2, 0]),
        "position_ids": torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 1]]),
        "video_indices": torch.tensor([1, 3]),
        "audio_indices": torch.tensor([2]),
        "text_indices": torch.tensor([0]),
    }


def _model_inputs(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.randn(1, 2, 2, device=device),
        "audio_hidden_states": torch.randn(1, 1, 3, device=device),
        "encoder_hidden_states": torch.randn(1, 1, 5, device=device),
        "timestep": torch.tensor([0.0, 0.75], device=device),
        "timestep_indices": torch.tensor([0, 1, 1, 0], device=device),
        "token_tags": torch.tensor([1, 0, 2, 0], device=device),
        "position_ids": torch.tensor(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 1]],
            device=device,
        ),
        "video_indices": torch.tensor([1, 3], device=device),
        "audio_indices": torch.tensor([2], device=device),
        "text_indices": torch.tensor([0], device=device),
    }


def _reference_linear(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden_states, module.weight, module.bias)


def _reference_rms_norm(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_states.to(torch.float32) * torch.rsqrt(variance + module.variance_epsilon)
    return module.weight * normalized.to(hidden_states.dtype)


def _reference_rotary_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotary_states = hidden_states[..., :rotary_dim]
    first_half, second_half = rotary_states.chunk(2, dim=-1)
    rotated_states = torch.cat((-second_half, first_half), dim=-1)
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    return torch.cat(
        (
            rotary_states * cos + rotated_states * sin,
            hidden_states[..., rotary_dim:],
        ),
        dim=-1,
    )


def _reference_rope(
    module: nn.Module,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = position_ids.to(torch.float32).unsqueeze(-1) * module.inv_freq.view(1, 1, -1)
    frequencies = torch.cat(frequencies.unbind(dim=1), dim=-1)
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


def _reference_attention(
    module: nn.Module,
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    batch_size, sequence_length = hidden_states.shape[:2]
    qkv = _reference_linear(module.qkv_proj, hidden_states)
    inner_dim = module.local_num_attention_heads * module.head_dim
    query, key, value = qkv.split((inner_dim, inner_dim, inner_dim), dim=-1)
    query = query.view(
        batch_size,
        sequence_length,
        module.local_num_attention_heads,
        module.head_dim,
    )
    key = key.view(
        batch_size,
        sequence_length,
        module.local_num_key_value_heads,
        module.head_dim,
    )
    query = _reference_rms_norm(module.norm_q, query)
    key = _reference_rms_norm(module.norm_k, key)
    if rotary_emb is not None:
        query = _reference_rotary_emb(query, *rotary_emb)
        key = _reference_rotary_emb(key, *rotary_emb)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.view(
        batch_size,
        sequence_length,
        module.local_num_key_value_heads,
        module.head_dim,
    ).transpose(1, 2)
    attention_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=module.head_dim**-0.5,
    )
    attention_output = attention_output.transpose(1, 2).flatten(2)
    return _reference_linear(module.to_out[0], attention_output)


def _reference_feed_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    gate, up = _reference_linear(module.gate_up_proj, hidden_states).chunk(2, dim=-1)
    return _reference_linear(module.down_proj, F.silu(gate) * up)


def _reference_timestep_embedding(
    model: h3.MiniMaxH3Transformer3DModel,
    timestep: torch.Tensor,
) -> torch.Tensor:
    half_dim = model.config.freq_dim // 2
    exponent = -torch.log(torch.tensor(10000.0)) * torch.arange(half_dim, dtype=torch.float32)
    frequencies = torch.exp(exponent / half_dim)
    sinusoidal = timestep[:, None].to(torch.float32) * frequencies[None, :]
    sinusoidal = torch.cat((sinusoidal.cos(), sinusoidal.sin()), dim=-1)
    hidden_states = _reference_linear(model.time_embedder.linear_1, sinusoidal)
    hidden_states = F.silu(hidden_states)
    return _reference_linear(model.time_embedder.linear_2, hidden_states)


def _reference_one_layer_forward(
    model: h3.MiniMaxH3Transformer3DModel,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(model.token_refiner.refiner_blocks) == 1
    assert len(model.transformer_blocks) == 1

    text_embeds = _reference_linear(
        model.context_embedder,
        inputs["encoder_hidden_states"].to(model.context_embedder.weight.dtype),
    )
    refiner_block = model.token_refiner.refiner_blocks[0]
    normalized = _reference_rms_norm(refiner_block.norm1, text_embeds)
    text_embeds = text_embeds + _reference_attention(refiner_block.attn, normalized)
    normalized = _reference_rms_norm(refiner_block.norm2, text_embeds)
    text_embeds = text_embeds + _reference_feed_forward(refiner_block.ff, normalized)
    text_embeds = _reference_rms_norm(model.token_refiner.final_norm, text_embeds)

    video_embeds = _reference_linear(
        model.proj_in,
        inputs["hidden_states"].to(model.proj_in.weight.dtype),
    ).to(text_embeds.dtype)
    audio_embeds = _reference_linear(
        model.audio_proj_in,
        inputs["audio_hidden_states"].to(model.audio_proj_in.weight.dtype),
    ).to(text_embeds.dtype)
    packed_hidden_states = text_embeds.new_zeros(
        (text_embeds.shape[0], inputs["position_ids"].shape[0], text_embeds.shape[-1])
    )
    packed_hidden_states[:, inputs["text_indices"]] = text_embeds
    packed_hidden_states[:, inputs["video_indices"]] = video_embeds
    packed_hidden_states[:, inputs["audio_indices"]] = audio_embeds

    temb = _reference_timestep_embedding(model, inputs["timestep"])
    adaln_indices = inputs["timestep_indices"] * h3.MINIMAX_H3_MODALITY_NUM + inputs[
        "token_tags"
    ].clamp(min=0)
    rotary_emb = _reference_rope(model.rope, inputs["position_ids"])
    block = model.transformer_blocks[0]
    modulation = _reference_linear(
        block.adaln_proj.linear,
        F.silu(temb).to(block.adaln_proj.linear.weight.dtype),
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.view(
        -1, 6 * model.config.hidden_size
    ).chunk(6, dim=-1)

    normalized = _reference_rms_norm(block.norm1, packed_hidden_states)
    normalized = normalized * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
    packed_hidden_states = packed_hidden_states + gate_msa[adaln_indices] * _reference_attention(
        block.attn, normalized, rotary_emb
    )
    normalized = _reference_rms_norm(block.norm2, packed_hidden_states)
    normalized = normalized * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
    packed_hidden_states = packed_hidden_states + gate_mlp[adaln_indices] * _reference_feed_forward(
        block.ff, normalized
    )

    shift, scale = _reference_linear(
        model.norm_out.linear,
        F.silu(temb).to(model.norm_out.linear.weight.dtype),
    ).chunk(2, dim=-1)
    packed_hidden_states = _reference_rms_norm(model.norm_out.norm, packed_hidden_states)
    packed_hidden_states = (
        packed_hidden_states * (1.0 + scale[inputs["timestep_indices"]])
        + shift[inputs["timestep_indices"]]
    )
    packed_hidden_states = packed_hidden_states.to(model.proj_out.weight.dtype)
    video_output = _reference_linear(model.proj_out, packed_hidden_states).index_select(
        1, inputs["video_indices"]
    )
    audio_output = _reference_linear(model.audio_proj_out, packed_hidden_states).index_select(
        1, inputs["audio_indices"]
    )
    return video_output, audio_output


def test_partial_rope_matches_split_half_reference_and_preserves_tail() -> None:
    hidden_states = torch.arange(1 * 2 * 1 * 8, dtype=torch.float32).view(1, 2, 1, 8)
    position_ids = torch.tensor([[0, 0, 0], [1, 2, 3]])
    cos, sin = h3.MiniMaxH3RotaryPosEmbed(rope_freq_dim=1)(position_ids)

    actual = h3.apply_minimax_h3_rotary_emb(hidden_states, cos, sin)

    rotary = hidden_states[..., :6]
    first_half, second_half = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second_half, first_half), dim=-1)
    expected_prefix = rotary * cos[None, :, None, :] + rotated * sin[None, :, None, :]
    expected = torch.cat((expected_prefix, hidden_states[..., 6:]), dim=-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[..., 6:], hidden_states[..., 6:])


def test_attention_supports_inner_width_larger_than_residual_width() -> None:
    attention = h3.MiniMaxH3Attention(
        hidden_size=12,
        num_attention_heads=2,
        attention_head_dim=8,
        qk_norm_eps=1e-5,
        model_config=_make_model_config(),
        layer_idx=0,
        module_name="test.attn",
    )
    _initialize_weights(attention)
    captured_shapes = {}

    def _identity_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        captured_shapes.update(query=query.shape, key=key.shape, value=value.shape)
        return query

    attention._attn_impl = _identity_attention
    output = attention(torch.randn(1, 3, 12, dtype=torch.bfloat16))

    assert captured_shapes == {
        "query": torch.Size([1, 3, 16]),
        "key": torch.Size([1, 3, 16]),
        "value": torch.Size([1, 3, 16]),
    }
    assert output.shape == (1, 3, 12)


def test_transformer_blocks_flatten_tokens_for_trt_gated_mlp() -> None:
    class _ZeroAttention(nn.Module):
        def forward(
            self,
            hidden_states: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> torch.Tensor:
            del args, kwargs
            return torch.zeros_like(hidden_states)

    class _RequireTwoDimensionalFFN(nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            assert hidden_states.ndim == 2
            return torch.zeros_like(hidden_states)

    class _ZeroAdaLN(nn.Module):
        def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
            table = torch.zeros(temb.shape[0] * 3, 12, dtype=temb.dtype)
            return (table,) * 6

    model_config = _make_model_config()
    refiner_block = h3.MiniMaxH3TokenRefinerBlock(
        hidden_size=12,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_dim=16,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        model_config=model_config,
        layer_idx=0,
    )
    refiner_block.norm1 = nn.Identity()
    refiner_block.norm2 = nn.Identity()
    refiner_block.attn = _ZeroAttention()
    refiner_block.ff = _RequireTwoDimensionalFFN()
    refiner_block(torch.randn(1, 4, 12))

    transformer_block = h3.MiniMaxH3TransformerBlock(
        hidden_size=12,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_dim=16,
        time_embed_dim=6,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        model_config=model_config,
        layer_idx=0,
    )
    transformer_block.norm1 = nn.Identity()
    transformer_block.norm2 = nn.Identity()
    transformer_block.attn = _ZeroAttention()
    transformer_block.ff = _RequireTwoDimensionalFFN()
    transformer_block.adaln_proj = _ZeroAdaLN()
    transformer_block(
        torch.randn(1, 4, 12),
        torch.randn(2, 6),
        torch.tensor([0, 1, 2, 3]),
        (torch.empty(0), torch.empty(0)),
    )


def test_forward_builds_timestep_modality_adaln_indices() -> None:
    model = h3.MiniMaxH3Transformer3DModel(_make_model_config())
    _initialize_weights(model)

    class _CaptureBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adaln_indices: torch.Tensor | None = None

        def forward(
            self,
            hidden_states: torch.Tensor,
            temb: torch.Tensor,
            adaln_indices: torch.Tensor,
            rotary_emb: tuple[torch.Tensor, torch.Tensor],
            key_padding_mask: torch.Tensor | None,
            timestep: torch.Tensor,
        ) -> torch.Tensor:
            del temb, rotary_emb, key_padding_mask, timestep
            self.adaln_indices = adaln_indices.clone()
            return hidden_states

    capture_block = _CaptureBlock()
    model.transformer_blocks = nn.ModuleList([capture_block])
    model(**_model_inputs())

    # row = timestep_index * 3 + modality_tag
    torch.testing.assert_close(capture_block.adaln_indices, torch.tensor([1, 3, 5, 0]))


def test_static_context_is_numerically_identical_to_uncached_path() -> None:
    model = h3.MiniMaxH3Transformer3DModel(_make_model_config())
    _initialize_weights(model)
    inputs = _model_inputs()

    uncached = model(**inputs)
    static_context = model.prepare_static_context(
        inputs["encoder_hidden_states"],
        inputs["position_ids"],
    )
    cached_inputs = inputs | {
        "encoder_hidden_states": None,
        "position_ids": None,
        "static_context": static_context,
    }
    cached = model(**cached_inputs)

    torch.testing.assert_close(cached.sample, uncached.sample, rtol=0, atol=0)
    torch.testing.assert_close(cached.audio_sample, uncached.audio_sample, rtol=0, atol=0)


def test_tiny_nonzero_layer_transformer_matches_cpu_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _cpu_swiglu(hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        gate, up = hidden_states.chunk(2, dim=-1)
        return F.silu(gate) * up

    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.gated_mlp.swiglu",
        _cpu_swiglu,
    )
    torch.manual_seed(11)
    model = h3.MiniMaxH3Transformer3DModel(_make_model_config(num_layers=1, num_refiner_layers=1))
    _initialize_weights(model, scale=0.1)
    inputs = _model_inputs()

    expected_video, expected_audio = _reference_one_layer_forward(model, inputs)
    actual = model(**inputs)

    torch.testing.assert_close(actual.sample, expected_video, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(
        actual.audio_sample,
        expected_audio,
        rtol=2e-2,
        atol=2e-3,
    )


def test_transformer_block_is_fullgraph_compile_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _cpu_swiglu(hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        gate, up = hidden_states.chunk(2, dim=-1)
        return F.silu(gate) * up

    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.gated_mlp.swiglu",
        _cpu_swiglu,
    )
    model = h3.MiniMaxH3Transformer3DModel(_make_model_config(num_layers=1))
    _initialize_weights(model, scale=0.1)
    block = model.transformer_blocks[0].eval()
    hidden_states = torch.randn(1, 4, 12, dtype=torch.bfloat16)
    temb = torch.randn(2, 6, dtype=torch.bfloat16)
    adaln_indices = torch.tensor([0, 1, 2, 3])
    rotary_emb = model.rope(torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 1]]))

    try:
        expected = block(hidden_states, temb, adaln_indices, rotary_emb)
        compiled_block = torch.compile(block, backend="eager", fullgraph=True)
        actual = compiled_block(hidden_states, temb, adaln_indices, rotary_emb)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()


def test_released_checkpoint_mixed_dtype_contract() -> None:
    model = h3.MiniMaxH3Transformer3DModel(_make_model_config())

    assert model.proj_in.weight.dtype == torch.float32
    assert model.audio_proj_in.weight.dtype == torch.float32
    assert model.time_embedder.linear_1.weight.dtype == torch.float32
    assert model.time_embedder.linear_2.weight.dtype == torch.float32
    assert model.proj_out.weight.dtype == torch.float32
    assert model.audio_proj_out.weight.dtype == torch.float32
    assert model.rope.inv_freq.dtype == torch.float32
    assert model.context_embedder.weight.dtype == torch.bfloat16
    assert model.norm_out.linear.weight.dtype == torch.bfloat16


def test_transformer_reuses_optimized_rms_norm() -> None:
    model = h3.MiniMaxH3Transformer3DModel(_make_model_config(num_layers=1, num_refiner_layers=1))

    assert isinstance(model.token_refiner.refiner_blocks[0].norm1, h3.RMSNorm)
    assert isinstance(model.token_refiner.refiner_blocks[0].norm2, h3.RMSNorm)
    assert isinstance(model.token_refiner.final_norm, h3.RMSNorm)
    assert isinstance(model.transformer_blocks[0].norm1, h3.RMSNorm)
    assert isinstance(model.transformer_blocks[0].norm2, h3.RMSNorm)
    assert isinstance(model.norm_out.norm, h3.RMSNorm)


def test_diffusers_swiglu_weight_is_reordered_from_up_gate_to_gate_up() -> None:
    up = torch.full((2, 3), 11.0)
    gate = torch.full((2, 3), 29.0)
    down = torch.full((3, 2), 41.0)
    remapped = h3.MiniMaxH3Transformer3DModel._remap_feed_forward_weights(
        {
            "transformer_blocks.0.ff.net.0.proj.weight": torch.cat((up, gate)),
            "transformer_blocks.0.ff.net.2.weight": down,
        }
    )

    torch.testing.assert_close(remapped["transformer_blocks.0.ff.gate.weight"], gate)
    torch.testing.assert_close(remapped["transformer_blocks.0.ff.up.weight"], up)
    torch.testing.assert_close(remapped["transformer_blocks.0.ff.down_proj.weight"], down)


def test_weight_loader_reports_missing_linear_biases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = h3.MiniMaxH3Transformer3DModel(_make_model_config())

    class _WeightOnlyLoader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def get_linear_weights(
            self,
            module: nn.Module,
            name: str,
            weights: dict[str, torch.Tensor],
        ) -> list[dict[str, torch.Tensor]]:
            del module, name, weights
            return [{"weight": torch.empty(0)}]

        def load_linear_weights(
            self,
            module: nn.Module,
            name: str,
            weight_dicts: list[dict[str, torch.Tensor]],
        ) -> None:
            del module, name, weight_dicts

        def filter_weights(
            self,
            name: str,
            weights: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            del name, weights
            return {}

    monkeypatch.setattr(h3, "DynamicLinearWeightLoader", _WeightOnlyLoader)
    with pytest.raises(ValueError, match=r"proj_in\.bias"):
        model.load_weights({})


def test_deferred_linear_weights_follow_transformer_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = _make_model_config()
    model_config.skip_create_weights_in_init = True
    model = h3.MiniMaxH3Transformer3DModel(model_config).to("meta")

    class _NoOpWeightLoader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def get_linear_weights(
            self,
            module: nn.Module,
            name: str,
            weights: dict[str, torch.Tensor],
        ) -> list[dict[str, torch.Tensor]]:
            del name, weights
            return [
                {
                    parameter_name: torch.empty_like(parameter, device="cpu")
                    for parameter_name, parameter in module._parameters.items()
                    if parameter is not None
                }
            ]

        def load_linear_weights(
            self,
            module: nn.Module,
            name: str,
            weight_dicts: list[dict[str, torch.Tensor]],
        ) -> None:
            del module, name, weight_dicts

        def filter_weights(
            self,
            name: str,
            weights: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            del weights
            module = dict(model.named_modules())[name]
            return {
                parameter_name: torch.empty_like(parameter, device="cpu")
                for parameter_name, parameter in module._parameters.items()
                if parameter is not None
            }

    monkeypatch.setattr(h3, "DynamicLinearWeightLoader", _NoOpWeightLoader)
    model.load_weights({})

    linear_parameters = [
        parameter
        for module in model.modules()
        if isinstance(module, h3.Linear)
        for parameter in module.parameters(recurse=False)
    ]
    assert linear_parameters
    assert all(parameter.device.type == "meta" for parameter in linear_parameters)


def _copy_golden_parameters_to_hf(
    reference: nn.Module,
    target: h3.MiniMaxH3Transformer3DModel,
) -> None:
    """Reverse-map deterministic TRT weights into the Diffusers parameter layout."""
    reference_parameters = dict(reference.named_parameters())
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            if name.endswith("attn.qkv_proj.weight"):
                prefix = name.removesuffix("qkv_proj.weight")
                query, key, value = parameter.chunk(3)
                for projection, source in zip(
                    ("to_q", "to_k", "to_v"),
                    (query, key, value),
                    strict=True,
                ):
                    reference_parameters[f"{prefix}{projection}.weight"].copy_(source)
            elif name.endswith("ff.gate_up_proj.weight"):
                gate, up = parameter.chunk(2)
                reference_parameters[
                    name.replace("gate_up_proj.weight", "net.0.proj.weight")
                ].copy_(torch.cat((up, gate)))
            elif name.endswith("ff.down_proj.weight"):
                reference_parameters[name.replace("down_proj.weight", "net.2.weight")].copy_(
                    parameter
                )
            else:
                reference_parameters[name].copy_(parameter)


def test_tiny_transformer_matches_pinned_diffusers_golden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match Diffusers PR 14355 at abc5e9bf with no draft-package dependency."""

    def _cpu_swiglu(hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        gate, up = hidden_states.chunk(2, dim=-1)
        return F.silu(gate) * up

    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.gated_mlp.swiglu",
        _cpu_swiglu,
    )
    model = h3.MiniMaxH3Transformer3DModel(_make_model_config(num_layers=1, num_refiner_layers=1))
    _initialize_diffusers_golden_weights(model)

    actual = model(**_diffusers_golden_inputs())
    expected_video = torch.tensor([[[-0.1232801974, -0.1542745978], [-0.0780100822, 0.0215485524]]])
    expected_audio = torch.tensor([[[0.1819977611, 0.1006751955, -0.1606836915]]])

    torch.testing.assert_close(actual.sample, expected_video, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(
        actual.audio_sample,
        expected_audio,
        rtol=2e-2,
        atol=2e-3,
    )


@pytest.mark.skipif(
    HFMiniMaxH3Transformer3DModel is None or not torch.cuda.is_available(),
    reason="MiniMax-H3 Diffusers reference and CUDA are required",
)
def test_pinned_diffusers_golden_matches_live_hf_reference() -> None:
    parity_config = _TINY_CONFIG | {"num_layers": 1, "num_refiner_layers": 1}
    reference = HFMiniMaxH3Transformer3DModel(**parity_config).to(
        device="cuda", dtype=torch.bfloat16
    )
    for module_name in (
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "rope",
        "proj_out",
        "audio_proj_out",
    ):
        getattr(reference, module_name).to(torch.float32)

    target = h3.MiniMaxH3Transformer3DModel(
        _make_model_config(num_layers=1, num_refiner_layers=1)
    ).to("cuda")
    _initialize_diffusers_golden_weights(target)
    _copy_golden_parameters_to_hf(reference, target)
    inputs = {name: tensor.to("cuda") for name, tensor in _diffusers_golden_inputs().items()}

    expected = reference(**inputs)
    actual = target(**inputs)
    golden_video = torch.tensor(
        [[[-0.1232801974, -0.1542745978], [-0.0780100822, 0.0215485524]]],
        device="cuda",
    )
    golden_audio = torch.tensor(
        [[[0.1819977611, 0.1006751955, -0.1606836915]]],
        device="cuda",
    )
    torch.testing.assert_close(expected.sample, golden_video, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(
        expected.audio_sample,
        golden_audio,
        rtol=2e-2,
        atol=2e-3,
    )
    torch.testing.assert_close(actual.sample, expected.sample, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        actual.audio_sample,
        expected.audio_sample,
        rtol=2e-2,
        atol=2e-2,
    )
