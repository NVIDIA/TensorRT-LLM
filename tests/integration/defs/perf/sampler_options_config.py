# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# -*- coding: utf-8 -*-
"""
Sampler options config for trtllm-bench perf tests
"""


def get_sampler_options_config(model_label: str) -> dict:
    """
    Return the sampler options config corresponding to the model label.
    Args:
        model_label: model label from self._config.to_string()
    Returns:
        dict: sampler options config
    """
    base_config = {}
    if model_label in [
            'llama_v3.1_70b_instruct-bench-pytorch-bfloat16-maxbs:512-maxnt:2048-input_output_len:200,2000-reqs:64-con:200-gpus:8',
            'llama_v3.1_70b_instruct_fp8-bench-pytorch-float8-maxbs:512-maxnt:2048-input_output_len:128,128-gpus:8',
            'llama_v3.2_1b-bench-pytorch-bfloat16-maxbs:512-maxnt:2048-input_output_len:500,2000-gpus:2',
            'llama_v3.3_70b_instruct_fp8-bench-pytorch-float8-maxbs:512-maxnt:2048-input_output_len:128,128-gpus:4',
            'llama_v4_maverick_17b_128e_instruct_fp8-bench-pytorch-float8-maxbs:1024-maxnt:20000-kv_frac:0.6-input_output_len:20000,2000-reqs:1000-ep:8-gpus:8',
            'llama_v4_maverick_17b_128e_instruct_fp8-bench-pytorch-float8-maxbs:1024-maxnt:4096-kv_frac:0.85-input_output_len:1000,1000-reqs:3000-ep:8-gpus:8',
            'llama_v4_scout_17b_16e_instruct_fp8-bench-pytorch-float8-maxbs:1024-maxnt:4096-kv_frac:0.85-input_output_len:500,2000-reqs:3000-ep:8-gpus:8',
            'mistral_small_v3.1_24b-bench-pytorch-bfloat16-maxbs:512-maxnt:2048-input_output_len:1000,2000-reqs:500-con:200-gpus:2',
            'phi_4_mini_instruct-bench-pytorch-bfloat16-maxbs:512-maxnt:2048-input_output_len:128,128'
    ]:
        base_config['top_k'] = 4
        base_config['top_p'] = 0.5
        base_config['temperature'] = 0.5
    return base_config
