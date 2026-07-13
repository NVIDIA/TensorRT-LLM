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

import torch
from torch import nn
from transformers import BartConfig, MBartConfig
from transformers.models.mbart.modeling_mbart import (
    MBartForConditionalGeneration as HFMBartForConditionalGeneration,
)

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_bart import (
    BartDecoder,
    BartDecoderLayer,
    BartEncoder,
    BartEncoderLayer,
    _convert_hf_bart_weights,
)
from tensorrt_llm._torch.modules.layer_norm import LayerNorm


class _Affine(nn.Module):
    def __init__(self, scale: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states * self.scale + self.bias


def _model_config(config_class, activation_function: str) -> ModelConfig:
    config = config_class(
        vocab_size=32,
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        max_position_embeddings=8,
        activation_function=activation_function,
        dtype=torch.float32,
    )
    if config.model_type == "mbart":
        config.normalize_before = True
        config.add_final_layer_norm = True
        config.scale_embedding = True
    return ModelConfig(pretrained_config=config)


def test_mbart_encoder_layer_uses_pre_norm_and_relu() -> None:
    layer = BartEncoderLayer(_model_config(MBartConfig, "relu"), layer_idx=0)

    assert layer.normalize_before
    assert isinstance(layer.mlp.activation, nn.ReLU)

    layer.self_attn = _Affine(scale=2.0)
    layer.self_attn_layer_norm = _Affine(bias=1.0)
    layer.mlp = _Affine(scale=3.0)
    layer.final_layer_norm = _Affine(bias=10.0)

    output = layer(torch.ones(1, 8), attn_metadata=None)

    torch.testing.assert_close(output, torch.full((1, 8), 50.0))


def test_bart_encoder_layer_keeps_post_norm() -> None:
    layer = BartEncoderLayer(_model_config(BartConfig, "gelu"), layer_idx=0)

    assert not layer.normalize_before

    layer.self_attn = _Affine(scale=2.0)
    layer.self_attn_layer_norm = _Affine(bias=1.0)
    layer.mlp = _Affine(scale=3.0)
    layer.final_layer_norm = _Affine(bias=10.0)

    output = layer(torch.ones(1, 8), attn_metadata=None)

    torch.testing.assert_close(output, torch.full((1, 8), 26.0))


def test_mbart_decoder_layer_uses_pre_norm() -> None:
    layer = BartDecoderLayer(_model_config(MBartConfig, "relu"), layer_idx=0)
    layer.self_attn = _Affine(scale=2.0)
    layer.self_attn_layer_norm = _Affine(bias=1.0)
    layer.cross_attn = _Affine(scale=3.0)
    layer.cross_attn_layer_norm = _Affine(bias=10.0)
    layer.mlp = _Affine(scale=4.0)
    layer.final_layer_norm = _Affine(bias=100.0)

    output = layer(
        position_ids=torch.zeros(1, dtype=torch.int32),
        hidden_states=torch.ones(1, 8),
        attn_metadata=None,
    )

    torch.testing.assert_close(output, torch.full((1, 8), 650.0))


def test_mbart_stacks_have_final_layer_norms() -> None:
    mbart_config = _model_config(MBartConfig, "relu")
    bart_config = _model_config(BartConfig, "gelu")

    assert isinstance(BartEncoder(mbart_config).layer_norm, LayerNorm)
    assert isinstance(BartDecoder(mbart_config).layer_norm, LayerNorm)
    assert BartEncoder(bart_config).layer_norm is None
    assert BartDecoder(bart_config).layer_norm is None


def test_mbart_weight_conversion_loads_final_layer_norms() -> None:
    config = _model_config(MBartConfig, "relu").pretrained_config
    hf_model = HFMBartForConditionalGeneration(config)
    hf_weights = hf_model.state_dict()

    converted = _convert_hf_bart_weights(hf_weights, config)

    for stack in ("encoder", "decoder"):
        converted_layer_norm = converted[f"model.{stack}.layer_norm"][0]
        torch.testing.assert_close(
            converted_layer_norm["weight"],
            hf_weights[f"model.{stack}.layer_norm.weight"],
        )
        torch.testing.assert_close(
            converted_layer_norm["bias"],
            hf_weights[f"model.{stack}.layer_norm.bias"],
        )
