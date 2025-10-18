from typing import Optional

import torch
from torch import nn
from transformers import Qwen3Config

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.qk_norm_attention import QKNormRoPEAttention
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model


class Qwen3Attention(QKNormRoPEAttention):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
        layer_idx: Optional[int] = None,
        fuse_qk_norm_rope: bool = True,
        attn_output_gate: bool = False,
        use_gemma_rms_norm: bool = False,
        disable_deep_gemm: bool = False,
    ):
        config = model_config.pretrained_config
        self.pretrained_config = config
        self.attn_output_gate = attn_output_gate

        if getattr(config, "rope_scaling", None) is not None:
            if "type" in config.rope_scaling:
                pos_type = config.rope_scaling["type"]
            elif "rope_type" in config.rope_scaling:
                pos_type = config.rope_scaling["rope_type"]
            else:
                raise ValueError(
                    "rope_scaling must have type or rope_type field")
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.from_string(pos_type),
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
            bias=getattr(config, "attention_bias", None),
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=fuse_qk_norm_rope,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=getattr(config, "attention_bias", None),
            config=model_config,
            attn_output_gate=self.attn_output_gate,
            use_gemma_rms_norm=use_gemma_rms_norm,
            disable_deep_gemm=disable_deep_gemm,
        )


class Qwen3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        self.self_attn = Qwen3Attention(
            model_config,
            layer_idx=layer_idx,
        )
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias if hasattr(config, "mlp_bias") else False,
            dtype=config.torch_dtype,
            overridden_tp_size=1 if self.enable_attention_dp else None,
            config=model_config,
        )

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.disable_allreduce = (self.mapping.tp_size == 1
                                  or self.enable_attention_dp)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
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
            all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_allreduce),
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(
            hidden_states,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_allreduce),
            cutlass_min_latency_mode=False,
        )

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)

        return hidden_states, residual


class Qwen3Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[Qwen3Config]):
        super().__init__(model_config)
        config = self.model_config

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
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
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
                spec_metadata=spec_metadata,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("Qwen3ForCausalLM")
class Qwen3ForCausalLM(SpecDecOneEngineForCausalLM[Qwen3Model, Qwen3Config]):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
    ):
        super().__init__(
            Qwen3Model(model_config),
            model_config,
        )
