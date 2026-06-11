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


def test_qwen35_dequantizes_per_tensor_fp8_linear_attention_projection():
    mapper = _make_mapper(dtype=torch.bfloat16)
    weight_name = "model.layers.0.linear_attn.in_proj_qkv.weight"
    scale_name = "model.layers.0.linear_attn.in_proj_qkv.weight_scale"
    input_scale_name = "model.layers.0.linear_attn.in_proj_qkv.input_scale"

    weights = {
        weight_name: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        scale_name: torch.tensor(0.5, dtype=torch.float32),
        input_scale_name: torch.tensor(1.0, dtype=torch.float32),
    }

    updated = mapper._dequantize_linear_attn_per_tensor_fp8(weights)

    assert scale_name not in updated
    assert input_scale_name not in updated
    assert updated[weight_name].dtype == torch.bfloat16
    torch.testing.assert_close(
        updated[weight_name],
        torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.bfloat16),
    )
