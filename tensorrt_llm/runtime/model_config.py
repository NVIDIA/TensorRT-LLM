# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .._utils import binding_layer_type_to_str, binding_to_str_dtype
from ..llmapi.kv_cache_type import KVCacheType
from ..quantization import QuantMode


@dataclass
class ModelConfig:
    max_batch_size: int
    max_beam_width: int
    vocab_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_size: int
    gpt_attention_plugin: bool
    gemm_allreduce_plugin: str = None
    remove_input_padding: bool = False
    model_name: str = ""
    kv_cache_type: KVCacheType = KVCacheType.CONTINUOUS
    cross_attention: bool = False
    head_size: int = None
    has_position_embedding: bool = True
    has_token_type_embedding: bool = False
    tokens_per_block: int = 32
    max_prompt_embedding_table_size: int = 0
    quant_mode: QuantMode = QuantMode(0)
    gather_context_logits: bool = False
    gather_generation_logits: bool = False
    dtype: str = ""
    lora_plugin: bool = False
    lora_target_modules: List[str] = field(default_factory=list)
    trtllm_modules_to_hf_modules: dict = None
    skip_cross_kv: bool = False
    num_medusa_heads: int = 0
    max_medusa_tokens: int = 0
    paged_state: bool = True
    mamba_conv1d_plugin: bool = True
    conv_kernel: int = 0
    layer_types: List[str] = field(default_factory=list)
    rnn_hidden_size: int = 0
    rnn_head_size: int = 0
    rnn_conv_dim_size: int = 0
    state_size: int = 0
    state_dtype: str = ""
    gpu_weights_percent: float = 1.0
    # ReDrafter
    redrafter_num_beams: int = 0
    redrafter_draft_len_per_beam: int = 0
    num_kv_heads_per_layer: Optional[List[int]] = None
    num_kv_heads_per_cross_attn_layer: Optional[List[int]] = None
    skip_cross_attn_blocks: bool = False
    # language adapter (typed as Optional[Any]; the concrete config type is
    # not imported here)
    language_adapter_config: Optional[Any] = None

    @classmethod
    def from_model_config_cpp(cls, model_config_cpp) -> "ModelConfig":
        """Create a partially initialized ModelConfig instance from a given ModelConfig CPP binding instance.

        Note that each of these classes have fields that don't exist in the other, so the created ModelConfigPython
        won't have all of its fields initialized.
        """
        return cls(
            max_batch_size=model_config_cpp.max_batch_size,
            max_beam_width=model_config_cpp.max_beam_width,
            vocab_size=model_config_cpp.vocab_size,
            num_layers=model_config_cpp.num_layers(),
            num_heads=model_config_cpp.num_heads,
            num_kv_heads=model_config_cpp.num_kv_heads(0),
            hidden_size=model_config_cpp.hidden_size,
            remove_input_padding=model_config_cpp.use_packed_input,
            kv_cache_type=model_config_cpp.kv_cache_type,
            cross_attention=model_config_cpp.use_cross_attention,
            head_size=model_config_cpp.head_size,
            max_prompt_embedding_table_size=model_config_cpp.max_prompt_embedding_table_size,
            quant_mode=QuantMode(model_config_cpp.quant_mode.value),
            gather_context_logits=model_config_cpp.compute_context_logits,
            gather_generation_logits=model_config_cpp.compute_generation_logits,
            gpt_attention_plugin=model_config_cpp.use_gpt_attention_plugin,
            dtype=binding_to_str_dtype(model_config_cpp.data_type),
            num_kv_heads_per_layer=model_config_cpp.num_kv_heads_per_layer,
            tokens_per_block=model_config_cpp.tokens_per_block,
            lora_plugin=model_config_cpp.use_lora_plugin,
            layer_types=[binding_layer_type_to_str(lt) for lt in model_config_cpp.layer_types],
        )
