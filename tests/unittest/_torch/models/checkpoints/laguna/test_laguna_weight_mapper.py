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

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_laguna import LagunaHfWeightMapper
from tensorrt_llm._torch.modules.fused_moe.interface import MoE


class _FakeQuantMode:
    def __init__(self, *, nvfp4: bool = False, fp8_block_scales: bool = False):
        self._nvfp4 = nvfp4
        self._fp8_block_scales = fp8_block_scales

    def has_nvfp4(self):
        return self._nvfp4

    def has_fp8_block_scales(self):
        return self._fp8_block_scales


class _FakeMoE(MoE):
    @classmethod
    def can_implement(cls, *args, **kwargs):
        return True, None

    def __init__(self):
        nn.Module.__init__(self)
        self.loaded_weights = None
        self.allow_partial_loading = None

    def load_weights(self, weights, allow_partial_loading: bool = False):
        self.loaded_weights = weights
        self.allow_partial_loading = allow_partial_loading


def test_laguna_hf_weight_mapper_preprocesses_nvfp4_weights():
    mapper = LagunaHfWeightMapper()
    mapper._config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_mode=_FakeQuantMode(nvfp4=True))
    )

    packed_weight = torch.tensor([1, 2], dtype=torch.uint8)
    correction_bias = torch.tensor([3.0])
    weights = {
        "model.layers.0.mlp.experts.e_score_correction_bias": correction_bias,
        "model.layers.0.mlp.experts.0.gate_proj.weight_packed": packed_weight,
        "model.layers.0.mlp.experts.0.gate_proj.weight_global_scale": torch.tensor([2.0, 0.0]),
        "model.layers.0.mlp.experts.0.gate_proj.input_global_scale": torch.tensor([4.0]),
    }

    result = mapper.preprocess_weights(weights)

    assert result["model.layers.0.mlp.gate.e_score_correction_bias"] is correction_bias
    assert result["model.layers.0.mlp.experts.0.gate_proj.weight"] is packed_weight
    torch.testing.assert_close(
        result["model.layers.0.mlp.experts.0.gate_proj.weight_scale_2"],
        torch.tensor([0.5, 0.0]),
    )
    torch.testing.assert_close(
        result["model.layers.0.mlp.experts.0.gate_proj.input_scale"],
        torch.tensor([0.25]),
    )


@pytest.mark.parametrize(
    ("fp8_block_scales", "scale_key"),
    [
        (False, "0.w1.weight_scale"),
        (True, "0.w1.weight_scale_inv"),
    ],
)
def test_laguna_hf_weight_mapper_handles_special_moe_module(fp8_block_scales: bool, scale_key: str):
    mapper = LagunaHfWeightMapper()
    mapper._config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_mode=_FakeQuantMode(fp8_block_scales=fp8_block_scales))
    )
    module = _FakeMoE()

    gate_weight = torch.tensor([1.0])
    up_weight = torch.tensor([2.0])
    down_weight = torch.tensor([3.0])
    weight_scale = torch.tensor([4.0])
    module_weights = {
        "experts.0.gate_proj.weight": gate_weight,
        "experts.0.up_proj.weight": up_weight,
        "experts.0.down_proj.weight": down_weight,
        "experts.0.gate_proj.weight_scale": weight_scale,
    }

    mapper.handle_special_instance_module(
        module,
        "model.layers.0.mlp.experts",
        module_weights,
        allow_partial_loading=True,
    )

    assert module.loaded_weights == [
        {
            "0.w1.weight": gate_weight,
            "0.w3.weight": up_weight,
            "0.w2.weight": down_weight,
            scale_key: weight_scale,
        }
    ]
    assert module.allow_partial_loading is True
