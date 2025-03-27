# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from utils.runtime_defaults import assert_runtime_defaults_are_parsed_correctly

from tensorrt_llm.models.modeling_utils import PretrainedConfig


def test_pretrained_config_parses_runtime_defaults_correctly():
    assert_runtime_defaults_are_parsed_correctly(
        lambda defaults: PretrainedConfig.from_dict(
            {
                'architecture': 'DeciLMForCausalLM',
                'dtype': 'float16',
                'num_hidden_layers': 2,
                'num_attention_heads': 12,
                'hidden_size': 768,
                'vocab_size': 51200,
                'max_position_embeddings': 1024,
                'runtime_defaults': defaults,
            }).runtime_defaults)
