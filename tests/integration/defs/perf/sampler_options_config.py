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
    # Labels are compared for equality, so they must be spelled exactly as
    # PerfTestConfig.to_string() emits them: maxbs:/maxnt: are always injected
    # and tp: is dropped when tp_size == num_gpus.
    base_config = {}
    if model_label in [
            'llama_v3.3_70b_instruct_fp8-bench-pytorch-float8-maxbs:512-maxnt:2048-input_output_len:128,128-gpus:8',
    ]:
        base_config['top_k'] = 4
        base_config['top_p'] = 0.5
        base_config['temperature'] = 0.5
    return base_config
