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
"""Unit tests for Qwen3.5/Qwen3.6 checkpoint weight mapping."""

import types

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _make_mapper() -> Qwen3_5MoeHfWeightMapper:
    pretrained_config = types.SimpleNamespace(
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        num_hidden_layers=1,
        num_experts=1,
        torch_dtype=torch.bfloat16,
    )
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
    )
    mapper = object.__new__(Qwen3_5MoeHfWeightMapper)
    mapper._config = model_config
    return mapper


def test_fp8_pertensor_linear_attn_weights_are_dequantized_before_pack():
    mapper = _make_mapper()
    scale = torch.tensor(0.25, dtype=torch.float32)
    qkv_fp8 = torch.tensor(
        [
            [1.0, -2.0],
            [3.0, -4.0],
            [0.5, -0.5],
            [1.5, -1.5],
            [2.0, -1.0],
            [4.0, -3.0],
        ],
        dtype=torch.float8_e4m3fn,
    )
    z_fp8 = torch.tensor([[2.0, -1.0], [1.0, -0.5]], dtype=torch.float8_e4m3fn)
    weights = {
        "model.layers.0.linear_attn.in_proj_qkv.weight": qkv_fp8,
        "model.layers.0.linear_attn.in_proj_qkv.weight_scale": scale,
        "model.layers.0.linear_attn.in_proj_qkv.input_scale": torch.tensor(
            1.0, dtype=torch.float32
        ),
        "model.layers.0.linear_attn.in_proj_z.weight": z_fp8,
        "model.layers.0.linear_attn.in_proj_z.weight_scale": scale,
        "model.layers.0.linear_attn.in_proj_z.input_scale": torch.tensor(1.0, dtype=torch.float32),
    }

    packed = mapper.preprocess_weights(weights)

    packed_weight = packed["model.layers.0.linear_attn.in_proj_qkvz.weight"]
    expected = torch.cat(
        [
            qkv_fp8[0:2].to(torch.float32) * scale,
            qkv_fp8[2:4].to(torch.float32) * scale,
            qkv_fp8[4:6].to(torch.float32) * scale,
            z_fp8.to(torch.float32) * scale,
        ],
        dim=0,
    ).to(torch.bfloat16)
    assert packed_weight.dtype == torch.bfloat16
    torch.testing.assert_close(packed_weight, expected)
    assert "model.layers.0.linear_attn.in_proj_qkvz.weight_scale" not in packed
    assert "model.layers.0.linear_attn.in_proj_qkvz.input_scale" not in packed
