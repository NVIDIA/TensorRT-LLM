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
"""Swiss AI Apertus (dense decoder) for the PyTorch backend."""

from typing import Dict, Optional

import torch
from torch import nn
from transformers import ApertusConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import PositionalEmbeddingParams, RopeParams
from ..attention.qk_norm_attention import QKNormRoPEAttention
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.linear import TensorParallelMode
from ..modules.mlp import MLP
from ..modules.rms_norm import RMSNorm
from ..modules.xielu import XIELU
from ..speculative import SpecMetadata
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model

# The xIELU module is MLP.activation; HF names it mlp.act_fn.
_APERTUS_PARAMS_MAP = {r"(.*)\.mlp\.act_fn\.(.*)": r"\1.mlp.activation.\2"}


class ApertusAttention(QKNormRoPEAttention):
    def __init__(
        self,
        model_config: ModelConfig[ApertusConfig],
        layer_idx: Optional[int] = None,
        reduce_output: bool = True,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            # The fused QK-norm + RoPE kernel only implements YaRN scaling, not
            # the llama3 scaling Apertus uses, so apply RoPE separately.
            fuse_qk_norm_rope=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=config.attention_bias,
            config=model_config,
            reduce_output=reduce_output,
        )


class ApertusDecoderLayer(DecoderLayer):
    def __init__(self, model_config: ModelConfig[ApertusConfig], layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.self_attn = ApertusAttention(
            model_config,
            layer_idx=layer_idx,
            reduce_output=not self.enable_attention_dp and self.mapping.tp_size > 1,
        )

        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=getattr(config, "mlp_bias", False),
            activation=XIELU(),
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            overridden_tp_size=1 if self.enable_attention_dp else None,
        )

        self.attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

        self.disable_attn_allreduce = self.mapping.tp_size == 1 or self.enable_attention_dp

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_layernorm(hidden_states)
        else:
            hidden_states, residual = self.attention_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not self.disable_attn_allreduce),
            **kwargs,
        )

        hidden_states, residual = self.feedforward_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, lora_params=kwargs.get("lora_params"))

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)

        return hidden_states, residual


class ApertusModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[ApertusConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        if config.hidden_act != "xielu":
            raise ValueError(f"Apertus supports hidden_act='xielu' only, got {config.hidden_act!r}")

        self.embed_tokens = Embedding(
            getattr(config, "input_vocab_size", config.vocab_size),
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList(
            [
                ApertusDecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, "
                "and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("ApertusForCausalLM")
class ApertusForCausalLM(SpecDecOneEngineForCausalLM[ApertusModel, ApertusConfig]):
    _params_map = _APERTUS_PARAMS_MAP

    def __init__(self, model_config: ModelConfig[ApertusConfig]):
        super().__init__(ApertusModel(model_config), model_config)

    def load_weights(
        self,
        weights: Dict,
        weight_mapper: Optional[BaseWeightMapper] = None,
        params_map: Optional[Dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ):
        super().load_weights(
            weights=weights,
            weight_mapper=weight_mapper,
            params_map={**self._params_map, **(params_map or {})},
            allow_partial_loading=allow_partial_loading,
        )


@register_auto_model("Apertus1p5ForConditionalGeneration")
class Apertus1p5ForConditionalGeneration(ApertusForCausalLM):
    """Apertus 1.5, text input only.

    The decoder is Apertus. The input embedding (``input_vocab_size`` rows) also
    covers the discrete image and audio codes produced by the checkpoint's
    vision and audio tokenizers, which are not run here; ``vocab_size`` and
    ``lm_head`` cover only the text tokens.
    """

    # First match wins, so the combined rename comes first.
    _params_map = {
        r"^model\.language_model\.(.*)\.mlp\.act_fn\.(.*)$": r"model.\1.mlp.activation.\2",
        r"^model\.language_model\.(.*)$": r"model.\1",
    }

    def __init__(self, model_config: ModelConfig[ApertusConfig]):
        _alias_language_model_quant_names(model_config)
        super().__init__(model_config)


def _strip_language_model(name: str) -> str:
    return name.replace("language_model\\.", "", 1).replace("language_model.", "", 1)


def _alias_language_model_quant_names(model_config: ModelConfig) -> None:
    """Quantization metadata of Apertus 1.5 checkpoints names decoder modules
    ``model.language_model.*``, as in the checkpoint; here they are ``model.*``.
    Add the runtime names alongside the checkpoint names."""
    # Copy before changing: the configs may be shared with the caller.
    updates = {}
    quant_config = model_config.quant_config
    if quant_config is not None and quant_config.exclude_modules:
        aliases = [_strip_language_model(m) for m in quant_config.exclude_modules]
        updates["quant_config"] = quant_config.model_copy(
            update={
                "exclude_modules": list(dict.fromkeys([*quant_config.exclude_modules, *aliases]))
            }
        )
    if model_config.quant_config_dict:
        quant_config_dict = dict(model_config.quant_config_dict)
        for name, layer_config in model_config.quant_config_dict.items():
            quant_config_dict.setdefault(_strip_language_model(name), layer_config)
        updates["quant_config_dict"] = quant_config_dict
    if updates:
        frozen = model_config._frozen
        model_config._frozen = False
        for key, value in updates.items():
            setattr(model_config, key, value)
        model_config._frozen = frozen
