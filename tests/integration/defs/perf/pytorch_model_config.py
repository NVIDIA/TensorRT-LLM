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
Model pytorch yaml config for trtllm-bench perf tests
"""


def get_model_yaml_config(model_label: str) -> dict:
    """
        Return the yaml config corresponding to the model label.
        Args:
            model_label: model label from self._config.to_string()
        Returns:
            dict: yaml config
        """
    base_config = {
        'print_iter_log': True,
        'use_cuda_graph': True,
        'cuda_graph_padding_enabled': True,
    }

    # Pattern-based configurations for models matching specific substrings
    # This allows for flexible configuration of models based on naming patterns
    pattern_configs = [
        # DeepSeek R1 models with MTP speculative decoding
        {
            'patterns': [
                'deepseek_r1-bench-pytorch-float16-maxbs:1-maxnt:8192-input_output_len:1000,2000-quant:fp8-reqs:10-ep:4-gpus:8',
                'deepseek_r1_nvfp4-bench-pytorch-float16-maxbs:1-maxnt:8192-input_output_len:1000,2000-quant:nvfp4-reqs:10-ep:4-tp:8-gpus:8'
            ],
            'config': {
                'enable_attention_dp': True,
                'use_cuda_graph': True,
                'speculative_config': {
                    'decoding_type': 'MTP',
                    'num_nextn_predict_layers': 3
                }
            }
        },
        # DeepSeek R1 models with large batch sizes and cuda graph padding
        {
            'patterns': [
                'deepseek_r1-bench-pytorch-float16-maxbs:384-maxnt:1536-input_output_len:1000,2000-quant:nvfp4-reqs:49152-con:3072-ep:8-gpus:8',
                'deepseek_r1_nvfp4-bench-pytorch-float16-maxbs:384-maxnt:1536-input_output_len:1000,2000-quant:nvfp4-reqs:49152-con:3072-ep:8-gpus:8'
            ],
            'config': {
                'enable_attention_dp': True,
                'cuda_graph_padding_enabled': True,
                'cuda_graph_batch_sizes':
                [1, 2, 4, 8, 16, 32, 64, 128, 256, 384]
            }
        },
        # DeepSeek R1 model with specific batch size 128
        {
            'patterns':
            'deepseek_r1-bench-pytorch-float16-maxbs:128-maxnt:1127-input_output_len:1000,2000-quant:fp8-reqs:5120-con:1024-ep:8-gpus:8',
            'config': {
                'enable_attention_dp': True,
                'cuda_graph_batch_sizes': [128]
            }
        },
        # Llama Nemotron models with attention_dp disabled to prevent hangs
        {
            'patterns': [
                'llama_v3.1_nemotron_ultra_253b_fp8-bench-pytorch-float8',
                'llama_v3.3_nemotron_super_49b_fp8-bench-pytorch-float8',
                'llama_v3.3_nemotron_super_49b-bench-pytorch-bfloat16'
            ],
            'config': {
                # True causes hang, needs model-specific fix.
                'enable_attention_dp': False,
            }
        },
    ]

    # Apply pattern-based configurations on top of base config
    for pattern_config in pattern_configs:
        patterns = pattern_config['patterns']
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            if pattern in model_label.lower():
                base_config.update(pattern_config['config'])
                break  # Stop checking other patterns for this config once we find a match

    return base_config
