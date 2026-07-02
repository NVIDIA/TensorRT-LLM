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

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper


def _make_mapper(dtype=torch.bfloat16):
    mapper = Qwen3_5MoeHfWeightMapper()
    mapper._config = SimpleNamespace(pretrained_config=SimpleNamespace(torch_dtype=dtype))
    return mapper


def test_qwen35_modelopt_preprocess_preserves_scalar_fp8_scale_name():
    mapper = _make_mapper()
    scale = torch.tensor(0.5, dtype=torch.float32)

    is_modelopt_ckpt, weights = mapper._preprocess_modelopt_ckpt(
        {
            "model.layers.0.linear_attn.out_proj.weight_scale": scale,
        }
    )

    assert is_modelopt_ckpt
    assert "model.layers.0.linear_attn.out_proj.weight_scale" in weights
    assert weights["model.layers.0.linear_attn.out_proj.weight_scale"].shape == torch.Size([])


def test_qwen35_rescales_per_tensor_fp8_linear_attention_qkvz_projection():
    mapper = _make_mapper(dtype=torch.bfloat16)
    mapper._config.pretrained_config.linear_num_key_heads = 1
    mapper._config.pretrained_config.linear_num_value_heads = 1
    mapper._config.pretrained_config.linear_key_head_dim = 2
    mapper._config.pretrained_config.linear_value_head_dim = 2
    weight_name = "model.layers.0.linear_attn.in_proj_qkv.weight"
    scale_name = "model.layers.0.linear_attn.in_proj_qkv.weight_scale"
    input_scale_name = "model.layers.0.linear_attn.in_proj_qkv.input_scale"
    z_weight_name = "model.layers.0.linear_attn.in_proj_z.weight"
    z_scale_name = "model.layers.0.linear_attn.in_proj_z.weight_scale"
    z_input_scale_name = "model.layers.0.linear_attn.in_proj_z.input_scale"

    weights = {
        weight_name: torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=torch.float8_e4m3fn,
        ),
        scale_name: torch.tensor(0.5, dtype=torch.float32),
        input_scale_name: torch.tensor(2.0, dtype=torch.float32),
        z_weight_name: torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float8_e4m3fn),
        z_scale_name: torch.tensor(1.0, dtype=torch.float32),
        z_input_scale_name: torch.tensor(1.0, dtype=torch.float32),
    }

    updated = mapper._rescale_linear_attn_qkvz_per_tensor_fp8(weights)
    packed = mapper._pack_split_projections(updated)

    assert scale_name not in updated
    assert input_scale_name not in updated
    assert z_scale_name not in updated
    assert z_input_scale_name not in updated
    assert packed["model.layers.0.linear_attn.in_proj_qkvz.weight"].dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        packed["model.layers.0.linear_attn.in_proj_qkvz.weight_scale"],
        torch.tensor(1.0, dtype=torch.float32),
    )
    torch.testing.assert_close(
        packed["model.layers.0.linear_attn.in_proj_qkvz.input_scale"],
        torch.tensor(2.0, dtype=torch.float32),
    )
    torch.testing.assert_close(
        packed["model.layers.0.linear_attn.in_proj_qkvz.weight"][:6].to(torch.float32),
        weights[weight_name].to(torch.float32) * 0.5,
    )
