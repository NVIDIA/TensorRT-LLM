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


def get_model_yaml_config(model_label: str,
                          lora_dirs: list[str] = None) -> dict:
    """
        Return the yaml config corresponding to the model label.
        Args:
            model_label: model label from self._config.to_string()
        Returns:
            dict: yaml config
        """
    base_config = {
        'enable_attention_dp': True,
        'print_iter_log': True,
        'use_cuda_graph': True,
        'cuda_graph_padding_enabled': True,
    }
    model_configs = {
        'deepseek_r1-bench-pytorch-float16-maxbs:1-maxnt:8192-input_output_len:1000,2000-quant:fp8-reqs:10-ep:4-gpus:8':
        {
            'use_cuda_graph': True,
            'speculative_config': {
                'decoding_type': 'MTP',
                'num_nextn_predict_layers': 3
            }
        },
        'deepseek_r1_nvfp4-bench-pytorch-float16-maxbs:1-maxnt:8192-input_output_len:1000,2000-quant:nvfp4-reqs:10-ep:4-tp:8-gpus:8':
        {
            'use_cuda_graph': True,
            'speculative_config': {
                'decoding_type': 'MTP',
                'num_nextn_predict_layers': 3
            }
        },
        'deepseek_r1-bench-pytorch-float16-maxbs:128-maxnt:1127-input_output_len:1000,2000-quant:fp8-reqs:5120-con:1024-ep:8-gpus:8':
        {
            'cuda_graph_batch_sizes': [128]
        },
        'deepseek_r1-bench-pytorch-float16-maxbs:384-maxnt:1536-input_output_len:1000,2000-quant:nvfp4-reqs:49152-con:3072-ep:8-gpus:8':
        {
            'cuda_graph_padding_enabled': True,
            'cuda_graph_batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256, 384]
        },
        'deepseek_r1_nvfp4-bench-pytorch-float16-maxbs:384-maxnt:1536-input_output_len:1000,2000-quant:nvfp4-reqs:49152-con:3072-ep:8-gpus:8':
        {
            'cuda_graph_padding_enabled': True,
            'cuda_graph_batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256, 384]
        },
    }

    # lora-specific change for pytorch
    if 'pytorch' in model_label and 'loras' in model_label:
        lora_config = {
            'lora_config': {
                'lora_dir': lora_dirs if lora_dirs is not None else [],
                'max_lora_rank': 64,
                'lora_target_modules': ['attn_q', 'attn_k', 'attn_v'],
                'trtllm_modules_to_hf_modules': {
                    "attn_q": "q_proj",
                    "attn_k": "k_proj",
                    "attn_v": "v_proj"
                }
            }
        }
        print(f"lora_config: {lora_config}")
        base_config.update(lora_config)
    else:
        # get model name from model_label
        model_name = next(
            (key for key in model_configs if key in model_label.lower()), None)
        if model_name:
            base_config.update(model_configs[model_name])

    return base_config
