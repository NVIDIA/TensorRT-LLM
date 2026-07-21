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
from unittest.mock import patch

import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_qwen3_5 import (
    _lm_head_nvfp4_enabled,
    _normalize_qwen35_exclude_modules,
    _normalize_qwen35_quant_config_dict,
)
from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextSparseMoeBlock
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def test_qwen36_normalizes_modelopt_quantized_layer_paths_on_sm121():
    w4a16_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16)
    fp8_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(
            num_hidden_layers=40,
            tie_word_embeddings=False,
            vocab_size=248320,
        ),
        quant_config=QuantConfig(
            exclude_modules=[
                "model.language_model.layers.0.linear_attn.in_proj_qkv",
                "mtp.layers.0*",
            ]
        ),
        quant_config_dict={
            "model.language_model.layers.0.linear_attn.in_proj_qkv": fp8_config,
            "model.language_model.layers.0.linear_attn.in_proj_z": fp8_config,
            "model.language_model.layers.0.mlp.experts": w4a16_config,
            "model.language_model.layers.0.mlp.shared_expert.gate_proj": w4a16_config,
            "model.language_model.layers.0.mlp.shared_expert.up_proj": w4a16_config,
            "model.language_model.layers.0.mlp.shared_expert.down_proj": w4a16_config,
            "model.visual.patch_embed": fp8_config,
            "mtp.layers.0.mlp.experts": w4a16_config,
            "lm_head": w4a16_config,
        },
    )

    with patch("tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=121):
        keep_lm_head_quant = _lm_head_nvfp4_enabled(model_config)
        assert keep_lm_head_quant
        _normalize_qwen35_exclude_modules(model_config, keep_lm_head_quant=keep_lm_head_quant)
        _normalize_qwen35_quant_config_dict(model_config, keep_lm_head_quant=keep_lm_head_quant)

    assert model_config.quant_config.exclude_modules == [
        "*linear_attn.conv1d",
        "model.layers.0.linear_attn.in_proj_qkvz*",
        "model.layers.40*",
    ]
    assert set(model_config.quant_config_dict) == {
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_expert.down_proj",
        "model.layers.0.mlp.shared_expert.gate_proj",
        "model.layers.0.mlp.shared_expert.up_proj",
        "model.layers.40.mlp.experts",
        "lm_head",
    }
    for name, quant_config in model_config.quant_config_dict.items():
        if name.endswith("in_proj_qkvz"):
            assert quant_config.quant_algo == QuantAlgo.FP8
        else:
            assert quant_config.quant_algo == QuantAlgo.W4A16_NVFP4


def test_qwen36_sparse_moe_uses_layer_w4a16_quant_config():
    experts_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16)
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=1,
            shared_expert_intermediate_size=8,
            torch_dtype=torch.bfloat16,
            model_type="qwen3_5_moe_text",
            mlp_bias=False,
        ),
        moe_backend="CUTEDSL",
        quant_config=QuantConfig(),
        quant_config_dict={
            "model.layers.0.mlp.experts": experts_config,
        },
    )
    captured = {}

    def fake_create_moe(**kwargs):
        captured.update(kwargs)
        return nn.Identity()

    with (
        patch(
            "tensorrt_llm._torch.models.modeling_qwen3_next.create_moe", side_effect=fake_create_moe
        ),
        patch(
            "tensorrt_llm._torch.models.modeling_qwen3_next.AllReduce",
            side_effect=lambda **kwargs: nn.Identity(),
        ),
        patch(
            "tensorrt_llm._torch.models.modeling_qwen3_next.GatedMLP",
            side_effect=lambda **kwargs: nn.Identity(),
        ),
        patch("torch.cuda.Event", side_effect=lambda: object()),
    ):
        Qwen3NextSparseMoeBlock(model_config=model_config, aux_stream=None, layer_idx=0)

    assert captured["model_config"] is model_config
    assert captured["override_quant_config"] is experts_config
    assert captured["layer_idx"] == 0
