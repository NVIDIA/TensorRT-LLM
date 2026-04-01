from typing import Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers import Cohere2Config
from transformers.activations import ACT2FN

from tensorrt_llm._torch.attention_backend import AttentionMetadata, FlashInferAttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionMask,
    CustomAttentionMask,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm.functional import PositionEmbeddingType

from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear, TensorParallelMode
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model


class Cohere2MLP(nn.Module):
    def __init__(self, model_config: ModelConfig[Cohere2Config]):
        """
        A SwiGLU implementation
        """
        config = model_config.pretrained_config

        super().__init__()

        self.gate_proj = Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            dtype=config.dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.up_proj = Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            dtype=config.dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=model_config.get_quant_config(),
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.down_proj = Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=config.dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=model_config.get_quant_config(),
            allreduce_strategy=model_config.allreduce_strategy,
        )

    @torch.inference_mode()
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Cohere2Attention(Attention):
    def __init__(
        self,
        model_config: ModelConfig[Cohere2Config],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        rope_params = RopeParams.from_config(config)
        if config.layer_types[layer_idx] == "sliding_attention":
            # Sliding window attention with RoPE
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj,
                rope=rope_params,
            )
            self.attention_window_size = config.sliding_window
        else:
            # Full attention without positional embedding (NoPE)
            pos_embd_params = None
            self.attention_window_size = None
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.dtype,
            config=model_config,
        )

    @torch.inference_mode()
    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        attention_mask_data: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if attention_mask_data is not None:
            assert isinstance(attn_metadata, FlashInferAttentionMetadata), (
                "Only FlashInfer backend supports custom attention mask currently."
            )
            assert attention_mask == CustomAttentionMask.CUSTOM
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            attention_window_size=self.attention_window_size,
            attention_mask_data=attention_mask_data,
            **kwargs,
        )


class Cohere2DecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[Cohere2Config],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config

        self.self_attn = Cohere2Attention(model_config, layer_idx=layer_idx)
        self.mlp = Cohere2MLP(model_config)

        self.input_layernorm = LayerNorm(
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=config.dtype,
            has_weights=True,
            has_bias=False,
        )

    @torch.inference_mode()
    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states_attention = self.self_attn(
            position_ids=None,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        hidden_states_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states_attention + hidden_states_mlp
        return hidden_states


class Cohere2Model(DecoderModel):
    def __init__(self, model_config: ModelConfig[Cohere2Config]):
        super().__init__(model_config)
        config = model_config.pretrained_config

        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=False,
            reduce_output=True,
        )

        self.norm = LayerNorm(
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=config.dtype,
            has_weights=True,
            has_bias=False,
        )

        self.layers = nn.ModuleList(
            [
                Cohere2DecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    @torch.inference_mode()
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

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("Cohere2ForCausalLM")
class Cohere2ForCausalLM(DecoderModelForCausalLM[Cohere2Model, Cohere2Config]):
    def __init__(
        self,
        model_config: ModelConfig[Cohere2Config],
    ):
        super().__init__(
            Cohere2Model(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )
        self.logit_scale = model_config.pretrained_config.logit_scale

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        logits = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return logits * self.logit_scale

    def load_weights(self, weights: dict):
        def filter_weights(prefix: str, weights: dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1 :]
                    result[new_k] = v
            return result

        params_map = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }

        for name, module in tqdm(list(self.named_modules()), desc="Loading weights"):
            if len(module._parameters) <= 0:
                continue

            # skip load weights if tie word embeddings is enabled and layer is lm_head
            if self.config.tie_word_embeddings and name.startswith("lm_head"):
                continue

            names = name.split(".")
            if names[-1] in params_map:
                module_weights = []
                for new_name in params_map[names[-1]]:
                    fw = filter_weights(".".join(names[:-1] + [new_name]), weights)
                    module_weights.append(fw)
                module.load_weights(weights=module_weights)
            else:
                module_weights = filter_weights(name, weights)
                if hasattr(module, "load_weights"):
                    module.load_weights(weights=[module_weights])
                else:
                    for n, p in module._parameters.items():
                        if p is not None:
                            p.data.copy_(module_weights[n][:])
