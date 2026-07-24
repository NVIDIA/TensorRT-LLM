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
from typing import Optional

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.nemotron_h_weight_mapper import (
    NemotronHHfWeightMapper,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _make_mapper(
    quant_algo: Optional[QuantAlgo] = None,
) -> NemotronHHfWeightMapper:
    mapper = NemotronHHfWeightMapper()
    mapper._config = ModelConfig(
        pretrained_config=SimpleNamespace(
            mamba_head_dim=1,
            mamba_num_heads=1,
            n_groups=1,
            num_hidden_layers=52,
            quantization_config={
                "producer": {"name": "modelopt", "version": "0.37.0"},
                "quant_method": "modelopt",
            },
            ssm_state_size=1,
        ),
        mapping=Mapping(),
        moe_backend="CUTLASS",
        quant_config=QuantConfig(quant_algo=quant_algo),
    )
    return mapper


def test_nemotron_h_mapper_preserves_w4a16_lm_head_weights_without_input_scale():
    mapper = _make_mapper()
    weight_scale_2 = torch.tensor(0.291 / (448 * 6), dtype=torch.float32)
    weights = {
        "lm_head.weight": torch.empty((8, 4), dtype=torch.uint8),
        "lm_head.weight_scale": torch.empty((8, 1), dtype=torch.float8_e4m3fn),
        "lm_head.weight_scale_2": weight_scale_2,
    }

    mapped = mapper.preprocess_weights(weights)

    assert mapped["lm_head.weight"] is weights["lm_head.weight"]
    assert mapped["lm_head.weight_scale"] is weights["lm_head.weight_scale"]
    assert mapped["lm_head.weight_scale_2"] is weight_scale_2
    assert "lm_head.input_scale" not in mapped


def test_nemotron_h_mapper_remaps_w4a16_moe_weights_without_input_scale():
    mapper = _make_mapper()
    up_prefix = "backbone.layers.1.mixer.experts.0.up_proj"
    down_prefix = "backbone.layers.1.mixer.experts.0.down_proj"
    up_weight_scale_2 = torch.tensor(0.134 / (448 * 6), dtype=torch.float32)
    down_weight_scale_2 = torch.tensor(0.214 / (448 * 6), dtype=torch.float32)
    weights = {
        f"{up_prefix}.weight": torch.empty((8, 4), dtype=torch.uint8),
        f"{up_prefix}.weight_scale": torch.empty((8, 1), dtype=torch.float8_e4m3fn),
        f"{up_prefix}.weight_scale_2": up_weight_scale_2,
        f"{down_prefix}.weight": torch.empty((8, 4), dtype=torch.uint8),
        f"{down_prefix}.weight_scale": torch.empty((8, 1), dtype=torch.float8_e4m3fn),
        f"{down_prefix}.weight_scale_2": down_weight_scale_2,
    }

    mapped = mapper.preprocess_weights(weights)

    assert "model.layers.1.mixer.experts.0.w1.weight" in mapped
    assert "model.layers.1.mixer.experts.0.w3.weight" in mapped
    assert "model.layers.1.mixer.experts.0.w2.weight" in mapped
    assert mapped["model.layers.1.mixer.experts.0.w3.weight"].shape == (0, 4)
    assert mapped["model.layers.1.mixer.experts.0.w3.weight_scale_2"].shape == ()
    assert mapped["model.layers.1.mixer.experts.0.w1.weight_scale_2"] is up_weight_scale_2
    assert mapped["model.layers.1.mixer.experts.0.w3.weight_scale_2"] is up_weight_scale_2
    assert mapped["model.layers.1.mixer.experts.0.w2.weight_scale_2"] is down_weight_scale_2
    assert not any(key.endswith(".input_scale") for key in mapped)


def test_nemotron_h_mapper_converts_compressed_tensors_global_scale():
    mapper = _make_mapper(QuantAlgo.W4A16_NVFP4)
    global_scale = torch.tensor(9362.2861328125, dtype=torch.float32)
    weights = {
        "lm_head.weight_packed": torch.empty((8, 4), dtype=torch.uint8),
        "lm_head.weight_scale": torch.empty((8, 1), dtype=torch.float8_e4m3fn),
        "lm_head.weight_global_scale": global_scale,
    }

    mapped = mapper.preprocess_weights(weights)

    assert mapped["lm_head.weight"] is weights["lm_head.weight_packed"]
    assert "lm_head.weight_packed" not in mapped
    assert "lm_head.weight_global_scale" not in mapped
    assert "lm_head.input_scale" not in mapped
    torch.testing.assert_close(mapped["lm_head.weight_scale_2"], global_scale.reciprocal())
