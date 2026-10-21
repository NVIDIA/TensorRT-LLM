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

from typing import List, Optional

from transformers.configuration_utils import PretrainedConfig


class Kolibri1Config(PretrainedConfig):
    """Configuration class for Kolibri 1 model.

    Kolibri 1 is an MoE architecture featuring:
    - Alternating Sliding Window Attention (SWA) and Full Attention (RNoPE)
    - Per-head QK RMSNorm
    - 4-norm sandwich structure per decoder layer
    - 384 fine-grained experts + 1 shared expert with biased sigmoid top-k routing
    """

    model_type = "kolibri1"

    def __init__(
        self,
        vocab_size: int = 128000,
        hidden_size: int = 2560,
        intermediate_size: int = 512,
        num_hidden_layers: int = 50,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        sliding_window: int = 513,
        layer_types: Optional[List[str]] = None,
        # MoE parameters
        num_experts: int = 384,
        num_experts_per_tok: int = 6,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        norm_topk_prob: bool = False,
        eos_token_id: int = 127906,
        pad_token_id: int = 127901,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window

        # Default layer_types schedule: 4 SWA layers followed by 1 full attention layer
        if layer_types is None:
            layer_types = [
                "full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        self.layer_types = layer_types

        # MoE parameters
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
