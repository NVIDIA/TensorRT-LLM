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

from tensorrt_llm.quantization.mode import QuantAlgo, QuantMode


def test_mxfp8_quant_algo_exists():
    assert QuantAlgo.MXFP8 == "MXFP8"


def test_mxfp8_quant_mode_helpers():
    mode = QuantMode.from_quant_algo(QuantAlgo.MXFP8)
    assert mode.has_mxfp8()
    assert mode.has_any_quant()
    # MXFP8 must not be confused with the MXFP4 variants.
    assert not mode.has_mxfp4()
    assert not mode.has_w4a8_mxfp4_mxfp8()


def test_mxfp8_from_description():
    mode = QuantMode.from_description(use_mxfp8=True)
    assert mode.has_mxfp8()


def test_load_mxfp8_hf_quant_config():
    from tensorrt_llm._torch.model_config import ModelConfig

    hf_quant_config = {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, 32],
        "ignored_layers": [
            "lm_head",
            "model.embed_tokens",
            "vision_tower",
            "model.layers.10.block_sparse_moe.gate",
        ],
    }
    quant_config, _ = ModelConfig.load_hf_quant_config(hf_quant_config, moe_backend="CUTLASS")
    assert quant_config.quant_algo == QuantAlgo.MXFP8
    assert quant_config.group_size == 32
    assert "lm_head" in quant_config.exclude_modules
    assert "vision_tower" in quant_config.exclude_modules
    assert "model.layers.10.block_sparse_moe.gate" in quant_config.exclude_modules
