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
"""Config for the XHToken Spark-X2.5/X2.6 dense model family (model_type
``spark2_5``). Adapted from the Apache-2.0 ``configuration_spark.py`` shipped
in the HF checkpoint, so ``spark2_5`` checkpoints load without
``trust_remote_code``.

The family mixes sliding-window and full-attention decoder layers via an
explicit per-layer ``layer_types`` list, with a per-layer-type RoPE dict
(``rope_parameters["sliding_attention"|"full_attention"]``) and an optional
headwise sigmoid attention output gate (``headwise_attn_output_gate`` /
``gate_attn_act_mode``).
"""

from transformers import PretrainedConfig


class Spark2_5Config(PretrainedConfig):
    model_type = "spark2_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="gelu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        headwise_attn_output_gate=False,
        gate_attn_act_mode="sigmoid",
        sliding_window=None,
        layer_types=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible "
                f"by num_key_value_heads ({num_key_value_heads})"
            )
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = (
            head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        )
        self.headwise_attn_output_gate = headwise_attn_output_gate
        if gate_attn_act_mode not in ("sigmoid", "silu"):
            raise ValueError(
                f"Unsupported gate_attn_act_mode: {gate_attn_act_mode!r}; "
                "expected 'sigmoid' or 'silu'."
            )
        self.gate_attn_act_mode = gate_attn_act_mode
        self.sliding_window = sliding_window
        self.rope_parameters = rope_parameters if rope_parameters is not None else {}

        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) must match "
                f"num_hidden_layers ({num_hidden_layers})"
            )
        # Without this, sliding layers would silently run as full attention:
        # both the model (attention_window_size=None) and the KV-cache window
        # derivation treat a missing sliding_window as "no window".
        if any("sliding" in str(lt) for lt in layer_types) and (
            not isinstance(sliding_window, int)
            or isinstance(sliding_window, bool)
            or sliding_window <= 0
        ):
            raise ValueError(
                f"layer_types contains sliding attention layers, which require "
                f"a positive integer sliding_window, got {sliding_window!r}"
            )
        self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_rope_theta(self, layer_type: str) -> float:
        params = self.rope_parameters.get(layer_type, {})
        return params.get("rope_theta", 10000)

    def get_partial_rotary_factor(self, layer_type: str) -> float:
        params = self.rope_parameters.get(layer_type, {})
        return params.get("partial_rotary_factor", 1.0)
