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

"""ModelOpt ``NVFP4_SVD`` (SVDQuant) checkpoint configs resolve to static NVFP4."""

import pytest

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

# Shape of the quantization_config a ModelOpt NVFP4_SVD export writes
# (modelopt.torch.export.convert_hf_config).
MODELOPT_NVFP4_SVD_QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            },
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "has_zero_point": False,
            "pre_quant_scale": True,
            "targets": ["Linear"],
            "lora_rank": 32,
        }
    },
    "ignore": ["proj_in", "proj_out", "norm_out*"],
    "quant_algo": "NVFP4_SVD",
    "group_size": 16,
    "quant_method": "modelopt",
}


def test_nvfp4_svd_resolves_to_static_nvfp4_with_pre_quant_scale() -> None:
    quant_config, layer_quant_config, dynamic_weight, dynamic_activation = (
        DiffusionPipelineConfig.load_diffusion_quant_config(MODELOPT_NVFP4_SVD_QUANT_CONFIG)
    )

    assert quant_config.quant_algo == QuantAlgo.NVFP4
    assert quant_config.group_size == 16
    assert quant_config.pre_quant_scale is True
    assert quant_config.exclude_modules == MODELOPT_NVFP4_SVD_QUANT_CONFIG["ignore"]
    assert dynamic_weight is False
    assert dynamic_activation is False
    assert layer_quant_config is None


def test_plain_nvfp4_config_has_no_pre_quant_scale() -> None:
    config = {
        "quant_algo": "NVFP4",
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            }
        },
    }
    quant_config, _, dynamic_weight, _ = DiffusionPipelineConfig.load_diffusion_quant_config(config)

    assert quant_config.quant_algo == QuantAlgo.NVFP4
    assert quant_config.pre_quant_scale is False
    assert dynamic_weight is False


def test_bare_nvfp4_svd_algo_string_sets_pre_quant_scale() -> None:
    quant_config, _, _, _ = DiffusionPipelineConfig.load_diffusion_quant_config(
        {"quant_algo": "NVFP4_SVD"}
    )

    assert quant_config.quant_algo == QuantAlgo.NVFP4
    assert quant_config.pre_quant_scale is True
