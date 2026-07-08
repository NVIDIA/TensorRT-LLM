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

from transformers import PretrainedConfig

from tensorrt_llm._torch.pyexecutor.config_utils import _normalize_chatglm_config


def _chatglm_config(**overrides: object) -> PretrainedConfig:
    config = {
        "num_layers": 28,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "kv_channels": 128,
        "multi_query_group_num": 2,
        "ffn_hidden_size": 13696,
        "seq_length": 8192,
        "layernorm_epsilon": 1e-5,
        "padded_vocab_size": 65024,
    }
    config.update(overrides)
    return PretrainedConfig(**config)


def test_chatglm_normalization_uses_rope_ratio() -> None:
    config = _chatglm_config(rope_ratio=50.0)

    _normalize_chatglm_config(config)

    assert config.rope_theta == 500000.0


def test_chatglm_normalization_preserves_explicit_rope_theta() -> None:
    config = _chatglm_config(rope_ratio=50.0, rope_theta=12345.0)

    _normalize_chatglm_config(config)

    assert config.rope_theta == 12345.0


def test_chatglm_normalization_infers_gqa_from_group_count() -> None:
    config = _chatglm_config()

    _normalize_chatglm_config(config)

    assert config.num_key_value_heads == 2
