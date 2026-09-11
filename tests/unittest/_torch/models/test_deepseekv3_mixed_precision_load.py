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

"""Load per-layer MLA weights without applying FP8 scales to other layers."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3WeightLoader
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


@pytest.mark.parametrize(("tp_size", "tp_rank"), [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("exclusion", ["model.layers.0.self_attn.kv_b_proj", "*kv_b_proj*"])
def test_mla_loader_dequantizes_only_excluded_fp8_checkpoint_layers(
    tp_size: int, tp_rank: int, exclusion: str
) -> None:
    """Real loader/kernel covers unaligned FP8 heads alongside scaleless BF16."""
    heads, qk_dim, v_dim, kv_rank = 4, 192, 128, 128
    local_heads = heads // tp_size
    model = nn.Module()
    model.config = SimpleNamespace(
        q_lora_rank=128,
        num_attention_heads=heads,
        qk_nope_head_dim=qk_dim,
        v_head_dim=v_dim,
        kv_lora_rank=kv_rank,
        num_hidden_layers=2,
        num_nextn_predict_layers=0,
    )
    model.model_config = SimpleNamespace(
        mapping=Mapping(world_size=tp_size, tp_size=tp_size, rank=tp_rank),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION, exclude_modules=[exclusion]),
    )
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    weights = {}
    references = []
    for layer_idx in range(2):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.kv_b_proj = nn.Linear(
            kv_rank,
            local_heads * (qk_dim + v_dim),
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
        )
        layer.self_attn.k_b_proj_trans = nn.Parameter(
            torch.empty(local_heads, kv_rank, qk_dim, dtype=torch.bfloat16, device="cuda"),
            requires_grad=False,
        )
        model.model.layers.append(layer)
        name = f"model.layers.{layer_idx}.self_attn.kv_b_proj"
        # Distinct block scales expose incorrect splitting at 192-wide heads.
        codes = (
            (torch.arange(heads * (qk_dim + v_dim) * kv_rank) % 15 - 7)
            .reshape(heads * (qk_dim + v_dim), kv_rank)
            .float()
        )
        if layer_idx == 0:
            codes = codes.to(torch.float8_e4m3fn)
            scales = torch.arange(1, codes.shape[0] // 128 + 1).float().reshape(-1, 1) / 16
            weights[f"{name}.weight"] = codes
            weights[f"{name}.weight_scale_inv"] = scales
            reference = codes.float() * scales.repeat_interleave(128, dim=0)
        else:
            reference = codes.to(torch.bfloat16)
            weights[f"{name}.weight"] = reference
            # No scale exists for the BF16 layer, even under a user wildcard.
        references.append(reference)

    DeepseekV3WeightLoader(model).load_weights(weights)

    for layer, reference in zip(model.model.layers, references):
        local = reference.reshape(heads, qk_dim + v_dim, kv_rank).chunk(tp_size, dim=0)[tp_rank]
        key, value = local.split([qk_dim, v_dim], dim=1)
        expected_weight = torch.cat(
            [key.reshape(-1, kv_rank), value.reshape(-1, kv_rank)], dim=0
        ).to(torch.bfloat16)
        torch.testing.assert_close(layer.self_attn.kv_b_proj.weight.cpu(), expected_weight)
        torch.testing.assert_close(
            layer.self_attn.k_b_proj_trans.cpu(), key.transpose(1, 2).to(torch.bfloat16)
        )
        torch.testing.assert_close(layer.self_attn.v_b_proj.cpu(), value.to(torch.bfloat16))
