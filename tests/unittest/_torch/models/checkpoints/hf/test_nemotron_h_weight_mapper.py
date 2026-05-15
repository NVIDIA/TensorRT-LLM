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

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.nemotron_h_weight_mapper import (
    NemotronHHfWeightMapper,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _make_mapper() -> NemotronHHfWeightMapper:
    mapper = NemotronHHfWeightMapper()
    mapper._config = ModelConfig(
        pretrained_config=SimpleNamespace(
            mamba_head_dim=1,
            mamba_num_heads=1,
            n_groups=1,
            num_hidden_layers=52,
            ssm_state_size=1,
        ),
        mapping=Mapping(),
        moe_backend="CUTLASS",
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4),
    )
    return mapper


def test_nemotron_h_mapper_canonicalizes_w4a16_nvfp4_checkpoint_keys():
    mapper = _make_mapper()
    weights = {
        "lm_head.weight_packed": torch.empty((8, 4), dtype=torch.uint8),
        "lm_head.weight_scale": torch.empty((8, 1), dtype=torch.float8_e4m3fn),
        "lm_head.weight_global_scale": torch.tensor(0.25, dtype=torch.float32),
    }

    mapped = mapper.preprocess_weights(weights)

    assert "lm_head.weight" in mapped
    assert "lm_head.weight_scale" in mapped
    assert "lm_head.weight_scale_2" in mapped
    assert "lm_head.input_scale" in mapped
    assert "lm_head.weight_packed" not in mapped
    assert "lm_head.weight_global_scale" not in mapped
    assert mapped["lm_head.weight"] is weights["lm_head.weight_packed"]
    torch.testing.assert_close(
        mapped["lm_head.weight_scale_2"], torch.tensor(4.0, dtype=torch.float32)
    )
    torch.testing.assert_close(
        mapped["lm_head.input_scale"], torch.tensor([1.0], dtype=torch.float32)
    )


def test_nemotron_h_mapper_handles_scalar_w4a16_nvfp4_moe_global_scales():
    mapper = _make_mapper()
    prefix = "backbone.layers.1.mixer.experts.0.up_proj"
    weights = {
        f"{prefix}.weight_packed": torch.empty((8, 4), dtype=torch.uint8),
        f"{prefix}.weight_scale": torch.empty((8, 1), dtype=torch.float8_e4m3fn),
        f"{prefix}.weight_global_scale": torch.tensor(0.25, dtype=torch.float32),
    }

    mapped = mapper.preprocess_weights(weights)

    assert "model.layers.1.mixer.experts.0.w1.weight" in mapped
    assert "model.layers.1.mixer.experts.0.w3.weight" in mapped
    assert "model.layers.1.mixer.experts.0.w1.weight_scale_2" in mapped
    assert "model.layers.1.mixer.experts.0.w3.weight_scale_2" in mapped
    assert mapped["model.layers.1.mixer.experts.0.w3.weight"].shape == (0, 4)
    assert mapped["model.layers.1.mixer.experts.0.w3.weight_scale_2"].shape == ()
