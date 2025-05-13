from typing import Dict, Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers import Qwen3MoeConfig

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import FusedMoE, RenormalizeMoeRoutingMethod
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from .modeling_qwen3 import Qwen3Attention
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             duplicate_kv_weight, register_auto_model)


class Qwen3MoE(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3MoeConfig],
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.moe_intermediate_size = config.moe_intermediate_size
        # self.shared_expert_intermediate_size = config.shared_expert_intermediate_size # not used in qwen3
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp

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
            intermediate_size=self.moe_intermediate_size,
            aux_stream=aux_stream,
            dtype=config.torch_dtype,
            reduce_results=reduce_results,
            model_config=model_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_dim
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)

        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        if self.enable_attention_dp and len(all_rank_num_tokens) > 1:
            max_num_token = max(all_rank_num_tokens)
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, 0, 0, max_num_token - hidden_states.shape[0]))
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens)

        return final_hidden_states.view(orig_shape)


class Qwen3MoEDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[Qwen3MoeConfig],
                 layer_idx: int, aux_stream: torch.cuda.Stream):
        super().__init__()
        config = model_config.pretrained_config
        self.self_attn = Qwen3Attention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = Qwen3MoE(model_config, aux_stream)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.layer_idx = layer_idx

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
        hidden_states = self.mlp(hidden_states, attn_metadata)
        return hidden_states, residual


class Qwen3MoEModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Qwen3MoeConfig]):
        super().__init__(model_config)
        config = self.model_config
        self.padding_idx = config.pretrained_config.pad_token_id
        self.aux_stream = torch.cuda.Stream()

        if model_config.mapping.enable_attention_dp:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            self.embed_tokens = nn.Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype)
        else:
            self.embed_tokens = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )
        self.layers = nn.ModuleList([
            Qwen3MoEDecoderLayer(
                model_config,
                layer_idx,
                self.aux_stream,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            hidden_size=config.pretrained_config.hidden_size,
            eps=config.pretrained_config.rms_norm_eps,
            dtype=config.pretrained_config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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


@register_auto_model("Qwen3MoeForCausalLM")
class Qwen3MoeForCausalLM(DecoderModelForCausalLM[Qwen3MoEModel,
                                                  Qwen3MoeConfig]):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3MoeConfig],
    ):
        super().__init__(
            Qwen3MoEModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: Dict):
        tp_size = self.model_config.mapping.tp_size
        enable_attention_dp = self.model_config.mapping.enable_attention_dp

        head_dim = getattr(
            self.config, "head_dim",
            self.config.hidden_size // self.config.num_attention_heads)

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        params_map = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"]
        }
        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                # skip load weights if tie word embeddings is enabled and layer is lm_head
                if self.config.tie_word_embeddings and name.startswith(
                        "lm_head"):
                    continue

                names = name.split(".")
                if names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        fw = filter_weights(".".join(names[:-1] + [new_name]),
                                            weights)
                        tensors_need_duplication = ["weight", "bias"]
                        if module.quant_config.quant_mode.has_nvfp4():
                            tensors_need_duplication.append("weight_scale")
                        if new_name in ["k_proj", "v_proj"]:
                            fw = {
                                k: (duplicate_kv_weight(
                                    weight=v[:],
                                    head_dim=head_dim,
                                    tensor_parallel_size=tp_size
                                    if not enable_attention_dp else 1)
                                    if k in tensors_need_duplication else v)
                                for k, v in fw.items()
                            }
                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                else:
                    module_weights = filter_weights(name, weights)
                    if isinstance(module, FusedMoE):
                        updated_module_weights = {}
                        for weight_name, weight_value in module_weights.items():
                            new_weight_name = (weight_name.replace(
                                "gate_proj",
                                "w1").replace("up_proj",
                                              "w3").replace("down_proj", "w2"))
                            updated_module_weights[
                                new_weight_name] = weight_value
                        del module_weights
                        module.load_weights(weights=[updated_module_weights])
                    elif hasattr(module, "load_weights"):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                p.data.copy_(module_weights[n][:])
