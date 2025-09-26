from typing import Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class QwenAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[Qwen2Config],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        if getattr(config, "rope_scaling", None) is not None:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.from_string(
                    config.rope_scaling["type"]),
                rope=RopeParams.from_config(config),
                mrope_section=config.rope_scaling.get('mrope_section', None))
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
            bias=True,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            rope_fusion=not getattr(config, 'disable_fuse_rope', False),
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
        )


class QwenDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Qwen2Config],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        self.self_attn = QwenAttention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias if hasattr(config, 'mlp_bias') else False,
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
        position_ids: torch.IntTensor,
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


class QwenModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Qwen2Config]):
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
            QwenDecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.pretrained_config.hidden_size,
                            eps=config.pretrained_config.rms_norm_eps,
                            dtype=config.pretrained_config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
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
            hidden_states, residual = decoder_layer(position_ids=position_ids,
                                                    hidden_states=hidden_states,
                                                    attn_metadata=attn_metadata,
                                                    residual=residual,
                                                    mrope_config=mrope_config)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("Qwen2ForCausalLM")
class Qwen2ForCausalLM(DecoderModelForCausalLM[QwenModel, Qwen2Config]):

    def __init__(
        self,
        model_config: ModelConfig[Qwen2Config],
    ):
        super().__init__(QwenModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    # NOTE: Qwen2-VL needs special mrope_config so adding separate forward() function to accept 'mrope_config'.
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
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


@register_auto_model("Qwen2ForProcessRewardModel")
class Qwen2ForProcessRewardModel(DecoderModelForCausalLM[QwenModel,
                                                         Qwen2Config]):
    """
    Qwen/Qwen2.5-Math-PRM.
    The Qwen2 Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """

    def __init__(self, model_config: ModelConfig[Qwen2Config]):
        nn.Module.__init__(self)
        self.model_config = model_config
        self.model = QwenModel(model_config)
        self.num_labels = 2

        config = model_config.pretrained_config

        # TODO: add parallel config
        self.score = nn.Sequential(
            Linear(config.hidden_size,
                   config.hidden_size,
                   dtype=config.torch_dtype), nn.ReLU(),
            Linear(config.hidden_size,
                   self.num_labels,
                   dtype=config.torch_dtype))

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: torch.IntTensor,
                position_ids: Optional[torch.IntTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                **kwargs) -> torch.Tensor:
        assert attn_metadata.seq_lens is not None

        hidden_states = self.model(attn_metadata,
                                   input_ids,
                                   position_ids=position_ids,
                                   inputs_embeds=inputs_embeds)
        logits = self.score(hidden_states)

        # reshape as PRM scores each token.
        # [[input_seq_len_1, num_labels], [input_seq_len_2, num_labels], ...]
        return torch.nested.nested_tensor(
            list(torch.split(logits, attn_metadata.seq_lens.tolist(), dim=0)))


@register_auto_model("Qwen2ForRewardModel")
class Qwen2ForRewardModel(DecoderModelForCausalLM[QwenModel, Qwen2Config]):
    """
    Qwen/Qwen2.5-Math-RM
    """

    def __init__(self, model_config: ModelConfig[Qwen2Config]):
        nn.Module.__init__(self)
        self.model_config = model_config
        self.model = QwenModel(model_config)
        self.num_labels = 1

        config = model_config.pretrained_config
        self.pad_token_id = config.pad_token_id

        # TODO: add parallel config
        self.score = nn.Sequential(
            Linear(config.hidden_size,
                   config.hidden_size,
                   dtype=config.torch_dtype), nn.ReLU(),
            Linear(config.hidden_size,
                   self.num_labels,
                   dtype=config.torch_dtype))

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: torch.IntTensor,
                position_ids: Optional[torch.IntTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                **kwargs) -> torch.Tensor:
        assert attn_metadata.seq_lens is not None

        hidden_states = self.model(attn_metadata,
                                   input_ids,
                                   position_ids=position_ids,
                                   inputs_embeds=inputs_embeds)
        logits = self.score(hidden_states)

        # get score of last token of each batch item
        end_indices = torch.cumsum(attn_metadata.seq_lens, dim=0) - 1
        return logits[end_indices]
