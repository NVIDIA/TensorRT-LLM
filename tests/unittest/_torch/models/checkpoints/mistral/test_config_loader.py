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

from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import adapt_config_dict


def test_mistral_large3_vision_config_preserves_max_position_embeddings():
    config = {
        "dim": 7168,
        "hidden_dim": 16384,
        "llama_4_scaling": {
            "beta": 0.1,
            "original_max_position_embeddings": 8192,
        },
        "max_position_embeddings": 294912,
        "max_seq_len": 262144,
        "moe": {
            "expert_hidden_dim": 4096,
            "first_k_dense_replace": 3,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "num_shared_experts": 1,
            "route_every_n": 1,
            "routed_scale": 1.0,
        },
        "n_heads": 128,
        "n_kv_heads": 128,
        "n_layers": 61,
        "norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tied_embeddings": False,
        "vision_encoder": {
            "hidden_size": 1664,
            "image_size": 1540,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 48,
            "patch_size": 14,
            "rope_theta": 10000.0,
        },
        "vocab_size": 131072,
        "yarn": {
            "alpha": 1,
            "apply_scale": False,
            "beta": 32,
            "factor": 36,
            "original_max_position_embeddings": 8192,
        },
    }

    adapted = adapt_config_dict(config)

    assert adapted.text_config.max_position_embeddings == 294912
    assert adapted.text_config.rope_scaling["rope_type"] == "yarn"
