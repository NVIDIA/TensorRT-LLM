from typing import Optional, Tuple

import torch
from torch import nn
from transformers import Qwen3Config

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention, QkNormType
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class Qwen3Attention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        if getattr(config, "rope_scaling", None) is not None:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.from_string(
                    config.rope_scaling["type"]),
                rope=RopeParams.from_config(config),
            )
        else:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=config.attention_bias,
            config=model_config,
            qk_norm_type=QkNormType.pre_rope,
        )

        self.q_norm = RMSNorm(hidden_size=self.head_dim,
                              eps=1e-6,
                              dtype=config.torch_dtype,
                              has_weights=True)
        self.k_norm = RMSNorm(hidden_size=self.head_dim,
                              eps=1e-6,
                              dtype=config.torch_dtype,
                              has_weights=True)
        self.aux_stream = torch.cuda.Stream()
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

    def apply_qk_norm(self, q, k):

        def q_l2norm():
            return self.q_norm(q.reshape(-1, self.head_dim)).reshape(
                -1, self.q_size)

        def k_l2norm():
            return self.k_norm(k.reshape(-1, self.head_dim)).reshape(
                -1, self.kv_size)

        q, k = maybe_execute_in_parallel(
            q_l2norm,
            k_l2norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        return q, k


class Qwen3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        self.self_attn = Qwen3Attention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias if hasattr(config, "mlp_bias") else False,
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        mrope_config: Optional[Tuple[torch.Tensor, int]] = None,
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
            mrope_config=mrope_config,
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[Qwen3Config]):
        super().__init__(model_config)
        config = self.model_config
        self.padding_idx = config.pretrained_config.pad_token_id

        self.embed_tokens = Embedding(
            config.pretrained_config.vocab_size,
            config.pretrained_config.hidden_size,
            dtype=config.pretrained_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                model_config,
                layer_idx,
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
        mrope_config: Optional[Tuple[torch.Tensor, int]] = None,
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
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                mrope_config=mrope_config,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("Qwen3ForCausalLM")
class Qwen3ForCausalLM(DecoderModelForCausalLM[Qwen3Model, Qwen3Config]):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
    ):
        super().__init__(
            Qwen3Model(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    # NOTE: Qwen2-VL needs special mrope_config so adding separate forward() function to accept 'mrope_config'.
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        mrope_config: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            mrope_config=mrope_config,
        )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )
