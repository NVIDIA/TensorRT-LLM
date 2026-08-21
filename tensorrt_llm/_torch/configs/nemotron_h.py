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

from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm.logger import logger

# Layer-type alphabet used by Nemotron-H `hybrid_override_pattern` strings:
# "M" mamba, "*" attention, "-" MLP, "E" MoE.
#
# transformers 5.5.x `NemotronHConfig` only understands "M", "E" and "*": it
# converts the pattern to `layers_block_type` in `__post_init__` and raises
# `KeyError: '-'` on any checkpoint with MLP layers, which is every dense
# Nemotron-H release (e.g. nvidia/NVIDIA-Nemotron-Nano-9B-v2 and its NVFP4
# variant). It also exposes `hybrid_override_pattern` as a read-only property
# derived from `layers_block_type`, so the "-" entries could not be recovered
# even if the conversion were widened. Upstream fixed this in transformers
# v5.9.0; requirements.txt pins 5.5.4.
_NATIVE_PATTERN_CHARS = frozenset("ME*")

_PATTERN_TO_BLOCK_TYPE = {
    "M": "mamba",
    "*": "attention",
    "-": "mlp",
    "E": "moe",
}

_BLOCK_TYPE_TO_PATTERN = {block_type: char for char, block_type in _PATTERN_TO_BLOCK_TYPE.items()}


def _pattern_to_block_types(pattern: str) -> list[str]:
    try:
        return [_PATTERN_TO_BLOCK_TYPE[char] for char in pattern]
    except KeyError as exc:
        raise ValueError(
            f"Invalid hybrid_override_pattern {pattern!r}: expected characters "
            f"in {sorted(_PATTERN_TO_BLOCK_TYPE)}"
        ) from exc


def _is_natively_representable(pattern) -> bool:
    """Whether the installed transformers `NemotronHConfig` can hold `pattern`."""
    return not pattern or set(pattern) <= _NATIVE_PATTERN_CHARS


class NemotronHConfig(PretrainedConfig):
    """Nemotron-H config that preserves `hybrid_override_pattern` verbatim.

    TRT-LLM builds Nemotron-H by iterating over `hybrid_override_pattern`
    (`modeling_nemotron_h.py`, `model_config.get_num_attention_layers`,
    `config_utils`' mamba/attention masks), so the literal checkpoint string --
    including "-" MLP layers -- has to survive config loading.

    This class is only used for checkpoints the installed transformers cannot
    represent. `from_dict` delegates to the native `NemotronHConfig` whenever
    the pattern is expressible there, so checkpoints that load today keep their
    exact current behaviour.
    """

    model_type = "nemotron_h"
    keys_to_ignore_at_inference = ["past_key_values"]

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        pattern = config_dict.get("hybrid_override_pattern")
        # Only take over when transformers would raise; otherwise the native
        # class stays authoritative (defaults, validation, future fixes).
        if _is_natively_representable(pattern):
            from transformers.models.nemotron_h.configuration_nemotron_h import (
                NemotronHConfig as HFNemotronHConfig,
            )

            return HFNemotronHConfig.from_dict(config_dict, **kwargs)
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 4096,
        intermediate_size: int = 21504,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        max_position_embeddings: int = 4096,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window=None,
        hybrid_override_pattern: str = "M*E",
        layers_block_type=None,
        num_hidden_layers=None,
        mlp_hidden_act: str = "relu2",
        mlp_bias: bool = False,
        use_bias: bool = False,
        use_mamba_kernels: bool = True,
        ssm_state_size: int = 128,
        mamba_num_heads: int = 128,
        mamba_head_dim: int = 64,
        mamba_hidden_act: str = "silu",
        mamba_proj_bias: bool = False,
        mamba_ssm_cache_dtype: str = "float32",
        n_groups: int = 8,
        conv_kernel: int = 4,
        expand: int = 2,
        chunk_size: int = 128,
        use_conv_bias: bool = True,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_limit=(0.0, float("inf")),
        time_step_floor: float = 1e-4,
        n_routed_experts: int = 8,
        n_shared_experts: int = 1,
        moe_intermediate_size: int = 7688,
        moe_shared_expert_intermediate_size: int = 7688,
        moe_latent_size=None,
        moe_shared_expert_overlap: bool = True,
        num_experts_per_tok: int = 2,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        num_nextn_predict_layers: int = 0,
        mtp_hybrid_override_pattern=None,
        mtp_layers_block_type=None,
        num_logits_to_keep: int = 1,
        initializer_range: float = 0.02,
        layer_norm_epsilon: float = 1e-5,
        rms_norm_eps=None,
        residual_in_fp32: bool = False,
        hidden_dropout: float = 0.0,
        rescale_prenorm_residual: bool = True,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        # Older Nemotron-H checkpoints ship the mamba-prefixed spellings; the
        # native config renames them in __post_init__, so mirror that here.
        n_groups = kwargs.pop("mamba_n_groups", n_groups)
        conv_kernel = kwargs.pop("mamba_d_conv", conv_kernel)
        expand = kwargs.pop("mamba_expand", expand)
        time_step_min = kwargs.pop("mamba_dt_min", time_step_min)
        time_step_max = kwargs.pop("mamba_dt_max", time_step_max)
        time_step_limit = kwargs.pop("mamba_dt_limit", time_step_limit)
        time_step_floor = kwargs.pop("mamba_dt_init_floor", time_step_floor)
        use_conv_bias = kwargs.pop("mamba_conv_bias", use_conv_bias)
        chunk_size = kwargs.pop("mamba_chunk_size", chunk_size)

        self.hybrid_override_pattern = hybrid_override_pattern
        # The pattern is authoritative for depth, matching the native config's
        # `num_hidden_layers` property.
        pattern_layers = len(hybrid_override_pattern)
        if num_hidden_layers is not None and num_hidden_layers != pattern_layers:
            logger.warning(
                f"num_hidden_layers ({num_hidden_layers}) does not match "
                f"hybrid_override_pattern length ({pattern_layers}); using the "
                f"pattern length."
            )
        self.num_hidden_layers = pattern_layers
        self.layers_block_type = (
            layers_block_type
            if layers_block_type is not None
            else _pattern_to_block_types(hybrid_override_pattern)
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        )
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_bias = mlp_bias
        self.use_bias = use_bias

        self.use_mamba_kernels = use_mamba_kernels
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.mamba_head_dim = mamba_head_dim
        self.mamba_hidden_act = mamba_hidden_act
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_ssm_cache_dtype = mamba_ssm_cache_dtype
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.chunk_size = chunk_size
        self.use_conv_bias = use_conv_bias
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_limit = time_step_limit
        self.time_step_floor = time_step_floor

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.moe_shared_expert_overlap = moe_shared_expert_overlap
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob

        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_layers_block_type = (
            mtp_layers_block_type if mtp_layers_block_type is not None else ["attention", "moe"]
        )
        # Plain attribute (the native config derives it from
        # mtp_layers_block_type); modeling_nemotron_h reads it directly.
        self.mtp_hybrid_override_pattern = (
            mtp_hybrid_override_pattern
            if mtp_hybrid_override_pattern is not None
            else "".join(
                _BLOCK_TYPE_TO_PATTERN[block_type] for block_type in self.mtp_layers_block_type
            )
        )

        self.num_logits_to_keep = num_logits_to_keep
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        # modeling_nemotron_h reads `rms_norm_eps`; released checkpoints ship
        # both spellings, older ones only `layer_norm_epsilon`.
        self.rms_norm_eps = rms_norm_eps if rms_norm_eps is not None else layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = hidden_dropout
        self.rescale_prenorm_residual = rescale_prenorm_residual

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
