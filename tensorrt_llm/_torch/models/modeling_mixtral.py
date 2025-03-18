from typing import Dict, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.distributed import ParallelConfig, allgather
from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..models.modeling_utils import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.fused_moe import FusedMoE, RenormalizeMoeRoutingMethod
from ..modules.linear import Linear
from ..modules.rms_norm import RMSNorm
from ..modules.rotary_embedding import RotaryEmbedding
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class MixtralMoE(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # moe gate (linear layer) only runs in half/full precision for now
        self.gate = Linear(self.hidden_dim,
                           self.num_experts,
                           bias=False,
                           dtype=config.torch_dtype,
                           quant_config=None)

        reduce_results = True

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            routing_method=RenormalizeMoeRoutingMethod(top_k=self.top_k),
            hidden_size=self.hidden_dim,
            intermediate_size=self.ffn_dim,
            dtype=config.torch_dtype,
            reduce_results=reduce_results,
            tune_max_num_tokens=config.max_position_embeddings // 4,
            model_config=model_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_dim
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape), router_logits


class MixtralRotaryEmbedding(RotaryEmbedding):

    def __init__(self,
                 config: PretrainedConfig,
                 device: Optional[torch.device] = None):
        head_dim = config.hidden_size // config.num_attention_heads
        super().__init__(config,
                         head_dim=head_dim,
                         num_attention_heads=config.num_attention_heads,
                         max_position_embeddings=config.max_position_embeddings,
                         device=device,
                         rope_type="default")


class MixtralAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        if model_config.fuse_pos_embd:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )
        else:
            pos_embd_params = None

        super().__init__(hidden_size=config.hidden_size,
                         num_attention_heads=config.num_attention_heads,
                         num_key_value_heads=config.num_key_value_heads,
                         max_position_embeddings=config.max_position_embeddings,
                         bias=False,
                         rotary_emb=MixtralRotaryEmbedding(config),
                         pos_embd_params=pos_embd_params,
                         layer_idx=layer_idx,
                         dtype=config.torch_dtype,
                         config=model_config)


class MixtralDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size

        self.self_attn = MixtralAttention(model_config, layer_idx=layer_idx)

        self.block_sparse_moe = MixtralMoE(model_config)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        # TODO: add pipeline parallel config
        self.parallel_config = ParallelConfig(
            tensor_parallel_rank=model_config.mapping.tp_rank,
            tensor_parallel_size=model_config.mapping.tp_size,
            gpus_per_node=model_config.mapping.gpus_per_node)
        self.layer_idx = layer_idx

    def all_gather(self, input_tensor, attn_metadata):
        rank = self.parallel_config.tensor_parallel_rank
        world_size = self.parallel_config.tensor_parallel_size
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        max_num_token = max(all_rank_num_tokens)
        if world_size == 1:
            return input_tensor, 0, max_num_token

        pad_tensor = torch.nn.functional.pad(
            input_tensor, (0, 0, 0, max_num_token - input_tensor.shape[0]))
        outputs = allgather(pad_tensor, self.parallel_config, gather_dim=0)
        depad_tensors = torch.concat([
            outputs[i * max_num_token:i * max_num_token +
                    all_rank_num_tokens[i]] for i in range(world_size)
        ])

        cur_rank_start = 0 if rank == 0 else sum(all_rank_num_tokens[:rank])
        cur_rank_end = cur_rank_start + all_rank_num_tokens[rank]
        return depad_tensors, cur_rank_start, cur_rank_end

    def forward(
        self,
        position_ids: torch.LongTensor,
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
        if self.enable_attention_dp:
            hidden_states, cur_rank_start, cur_rank_end = self.all_gather(
                hidden_states, attn_metadata)
        hidden_states, _router_logits = self.block_sparse_moe(hidden_states)
        if self.enable_attention_dp:
            hidden_states = hidden_states[cur_rank_start:cur_rank_end]
        return hidden_states, residual


class MixtralModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         config.pad_token_id,
                                         dtype=config.torch_dtype)

        self.layers = nn.ModuleList([
            MixtralDecoderLayer(model_config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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


@register_auto_model("MixtralForCausalLM")
class MixtralForCausalLM(DecoderModelForCausalLM[MixtralModel,
                                                 PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(MixtralModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: Dict):

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        params_map = {
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        }

        for name, module in self.named_modules():
            if len(module._parameters) > 0:
                names = name.split('.')
                if names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        module_weights.append(
                            filter_weights('.'.join(names[:-1] + [new_name]),
                                           weights))
                    module.load_weights(weights=module_weights)
                else:
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module.named_parameters():
                            p.data.copy_(module_weights[n][:])
