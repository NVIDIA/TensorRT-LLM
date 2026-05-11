import re
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..models.modeling_utils import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
from ..modules.linear import Linear
from ..modules.rms_norm import RMSNorm
from ..utils import AuxStreamType
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class MixtralMoE(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp

        # moe gate (linear layer) only runs in half/full precision for now
        self.gate = Linear(self.hidden_dim,
                           self.num_experts,
                           bias=False,
                           dtype=config.torch_dtype,
                           quant_config=None)

        reduce_results = True

        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=RenormalizeMoeRoutingMethod(top_k=self.top_k),
            hidden_size=self.hidden_dim,
            intermediate_size=self.ffn_dim,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            dtype=config.torch_dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=False)
        return final_hidden_states


class MixtralAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        super().__init__(hidden_size=config.hidden_size,
                         num_attention_heads=config.num_attention_heads,
                         num_key_value_heads=config.num_key_value_heads,
                         max_position_embeddings=config.max_position_embeddings,
                         bias=False,
                         pos_embd_params=PositionalEmbeddingParams(
                             type=PositionEmbeddingType.rope_gpt_neox,
                             rope=RopeParams.from_config(config),
                         ),
                         layer_idx=layer_idx,
                         dtype=config.torch_dtype,
                         config=model_config)

        self.attention_window_size = getattr(config, "sliding_window", None)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_window_size=self.attention_window_size,
            **kwargs,
        )


class MixtralDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int, aux_stream: torch.cuda.Stream):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size

        self.self_attn = MixtralAttention(model_config, layer_idx=layer_idx)

        self.block_sparse_moe = MixtralMoE(model_config,
                                           aux_stream,
                                           layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.mapping = model_config.mapping
        self.layer_idx = layer_idx

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states, attn_metadata)
        return hidden_states, residual


class MixtralModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            enable_torch_compile_for_embedding=model_config.
            enable_torch_compile_for_embedding,
        )

        self.layers = nn.ModuleList([
            MixtralDecoderLayer(model_config, layer_idx, self.aux_stream)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(position_ids=position_ids,
                                                    hidden_states=hidden_states,
                                                    attn_metadata=attn_metadata,
                                                    residual=residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def _unfuse_mixtral_moe_weights(weights: Dict) -> Dict:
    """Unfuse HF transformers 5.x fused Mixtral MoE weights to per-expert format.

    Transformers 5.x changed Mixtral to use fused expert weights:
      - ``model.layers.N.mlp.experts.gate_up_proj`` [num_experts, 2*intermediate, hidden]
      - ``model.layers.N.mlp.experts.down_proj`` [num_experts, hidden, intermediate]
    and renamed ``block_sparse_moe`` to ``mlp``.

    This function detects the new format and converts it to the per-expert
    format that TRT-LLM expects:
      - ``model.layers.N.block_sparse_moe.experts.{i}.w1.weight``
      - ``model.layers.N.block_sparse_moe.experts.{i}.w3.weight``
      - ``model.layers.N.block_sparse_moe.experts.{i}.w2.weight``

    Modifies and returns the weights dict in-place.
    """
    keys_to_remove = []
    keys_to_add = {}

    for key in list(weights.keys()):
        # Detect fused gate_up_proj: model.layers.N.mlp.experts.gate_up_proj
        m = re.match(r'^(model\.layers\.\d+\.)mlp\.experts\.gate_up_proj$', key)
        if m:
            prefix = m.group(1)
            value = weights[key]
            if hasattr(value, 'dim') and value.dim() == 3:
                num_experts = value.shape[0]
                half = value.shape[1] // 2
                for i in range(num_experts):
                    # gate_up_proj first half = gate_proj (w1),
                    # second half = up_proj (w3)
                    keys_to_add[
                        f"{prefix}block_sparse_moe.experts.{i}.w1.weight"] = value[
                            i, :half, :]
                    keys_to_add[
                        f"{prefix}block_sparse_moe.experts.{i}.w3.weight"] = value[
                            i, half:, :]
                keys_to_remove.append(key)
            continue

        # Detect fused down_proj: model.layers.N.mlp.experts.down_proj
        m = re.match(r'^(model\.layers\.\d+\.)mlp\.experts\.down_proj$', key)
        if m:
            prefix = m.group(1)
            value = weights[key]
            if hasattr(value, 'dim') and value.dim() == 3:
                num_experts = value.shape[0]
                for i in range(num_experts):
                    keys_to_add[
                        f"{prefix}block_sparse_moe.experts.{i}.w2.weight"] = value[
                            i]
                keys_to_remove.append(key)
            continue

        # Rename mlp.gate -> block_sparse_moe.gate (router weights)
        m = re.match(r'^(model\.layers\.\d+\.)mlp\.gate\.(.+)$', key)
        if m:
            prefix = m.group(1)
            suffix = m.group(2)
            keys_to_add[f"{prefix}block_sparse_moe.gate.{suffix}"] = weights[
                key]
            keys_to_remove.append(key)
            continue

    if not keys_to_remove:
        return weights

    for key in keys_to_remove:
        del weights[key]
    weights.update(keys_to_add)

    return weights


@register_auto_model("MixtralForCausalLM")
class MixtralForCausalLM(DecoderModelForCausalLM[MixtralModel,
                                                 PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(MixtralModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def load_weights(self,
                     weights: Dict,
                     weight_mapper=None,
                     skip_modules: List[str] = [],
                     params_map: Optional[Dict[str, str]] = None,
                     allow_partial_loading: bool = False):
        # Preprocess weights to handle transformers 5.x fused MoE format.
        # This handles both the v1 (no mapper) and v2 (with mapper) paths.
        _unfuse_mixtral_moe_weights(weights)
        super().load_weights(weights,
                             weight_mapper=weight_mapper,
                             skip_modules=skip_modules,
                             params_map=params_map,
                             allow_partial_loading=allow_partial_loading)
