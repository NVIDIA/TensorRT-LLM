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

from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def test_quant_config_exclude_modules_regex_match():
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        exclude_modules=[
            r"re:model\.layers\.[02468]\.mixer",
            r"re:model\.layers\.\d+\.self_attn\.(q|k)_proj",
        ],
    )

    for name in (
        "model.layers.2.mixer",
        "model.layers.2.mixer.down_proj",
        "model.layers.10.self_attn.q_proj",
        "model.layers.10.self_attn.q_proj.weight",
        "model.layers.10.self_attn.k_proj",
    ):
        assert quant_config.is_module_excluded_from_quantization(name), name
    for name in (
        "model.layers.3.mixer",
        "model.layers.10.self_attn.v_proj",
        "model.layers.10.self_attn.o_proj",
    ):
        assert not quant_config.is_module_excluded_from_quantization(name), name
