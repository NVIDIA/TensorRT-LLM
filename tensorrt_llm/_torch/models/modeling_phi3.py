from typing import Dict, Optional

import torch
from torch import nn
from transformers import PhiConfig
from transformers.activations import ACT2FN

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           RopeParams, RotaryScalingType)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class PhiAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[PhiConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config

        self.small_variant = config.model_type == "phi3-small"
        self.moe_variant = config.model_type == "phi3-moe"

        position_embedding_type = PositionEmbeddingType.rope_gpt_neox

        q_scaling = 1.0
        if self.small_variant:
            hidden_size = config.hidden_size
            num_attention_heads = config.num_attention_heads
            attention_head_size = hidden_size / num_attention_heads
            q_scaling = attention_head_size**.5

        head_dim = getattr(config, "head_dim",
                           config.hidden_size // config.num_attention_heads)
        rotary_percentage = getattr(config, "rotary_pct", 1.0)
        rotary_dim = int(head_dim * rotary_percentage)

        rope_params = RopeParams(
            dim=rotary_dim,
            theta=getattr(config, "rope_theta", 10000.0),
            scale_type=RotaryScalingType.none,
            scale=1.0,
            max_positions=config.max_position_embeddings,
        )

        pos_embd_params = PositionalEmbeddingParams(
            type=position_embedding_type,
            rope=rope_params,
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=self.small_variant or self.moe_variant,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            q_scaling=q_scaling,
        )
        self.head_dim = head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout


class PhiMLP(nn.Module):

    def __init__(self, config: PhiConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dtype = config.torch_dtype

        self.gate_up_proj = Linear(self.hidden_size,
                                   2 * self.intermediate_size,
                                   bias=False,
                                   dtype=self.dtype)
        self.down_proj = Linear(self.intermediate_size,
                                self.hidden_size,
                                bias=False,
                                dtype=self.dtype)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        up_states = self.gate_up_proj(x)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


class PhiDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[PhiConfig],
        layer_idx: Optional[int] = None,
    ):
        super().__init__()

        config = model_config.pretrained_config

        self.small_variant = config.model_type == "phi3-small"
        self.moe_variant = config.model_type == "phi3-moe"

        self.self_attn = PhiAttention(
            model_config=model_config,
            layer_idx=layer_idx,
        )

        self.mlp = PhiMLP(config=config, )

        eps = config.layer_norm_epsilon if self.small_variant else config.rms_norm_eps

        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=eps,
            dtype=config.torch_dtype,
        )
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=eps,
            dtype=config.torch_dtype,
        )

        self.dropout = nn.Dropout(
            config.attention_dropout) if config.attention_dropout > 0 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)

        return hidden_states


class PhiModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[PhiConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size
        self.padding_idx = config.pad_token_id

        self.small_variant = config.model_type == "phi3-small"
        self.moe_variant = config.model_type == "phi3-moe"

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )

        self.layers = nn.ModuleList([
            PhiDecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.num_hidden_layers)
        ])

        if self.small_variant or self.moe_variant:
            self.norm = nn.LayerNorm(normalized_shape=config.hidden_size,
                                     eps=config.norm_epsilon)
        else:
            self.norm = RMSNorm(hidden_size=config.hidden_size,
                                eps=config.rms_norm_eps,
                                dtype=config.torch_dtype)


@register_auto_model("Phi3ForCausalLM")
class Phi3ForCausalLM(DecoderModelForCausalLM[PhiModel, PhiConfig]):

    def __init__(
        self,
        model_config: ModelConfig[PhiConfig],
    ):
        super().__init__(PhiModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

        self.trtllm_modules_to_hf_modules = {
            "attn_qkv": ["qkv_proj", "query_key_value"],
            "attn_dense": ["o_proj", "dense"],
            "mlp_h_to_4h": ["gate_up_proj", "up_proj"],
            "mlp_4h_to_h": "down_proj",
        }

    def load_weights(self, weights: Dict):
        """Load weights from the provided dictionary."""
        for name, module in self.named_modules():
            if len(module._parameters) > 0:
                try:
                    module_weights = weights.get(name, {})
                    if not module_weights:
                        continue

                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=module_weights)
                    else:
                        for n, p in module._parameters.items():
                            if p is not None and n in module_weights:
                                p.data.copy_(module_weights[n][:])
                except Exception as e:
                    raise Exception(
                        f"Error loading weights for module {name}: {str(e)}")

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )
