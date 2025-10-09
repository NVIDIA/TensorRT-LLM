from typing import Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.modules.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)

try:
    from transformers import Exaone4Config
except ImportError:
    # TODO: Remove this once we have a proper transformers package
    from transformers import AutoConfig, PretrainedConfig

    class Exaone4Config(PretrainedConfig):
        model_type = "exaone4"

    AutoConfig.register(Exaone4Config.model_type, Exaone4Config)


def check_is_sliding(config: Exaone4Config, layer_idx: int) -> bool:
    """
    Check if the current layer is a sliding window (local attention) layer.
    """
    if config.sliding_window is None:
        return False
    if isinstance(config.sliding_window_pattern, int):
        return ((layer_idx + 1) % config.sliding_window_pattern) != 0
    elif isinstance(config.sliding_window_pattern, str):
        assert isinstance(config.sliding_window, int), (
            f"Sliding window must be positive integer, but got {config.sliding_window}"
        )
        return (layer_idx != config.num_hidden_layers - 1
                and config.sliding_window_pattern[layer_idx % len(
                    config.sliding_window_pattern)] == "L")
    return False


class Exaone4Attention(QKNormRoPEAttention):

    def __init__(self,
                 model_config: ModelConfig[Exaone4Config],
                 layer_idx: Optional[int] = None,
                 fuse_qk_norm_rope: bool = False):
        config = model_config.pretrained_config

        self.attention_window_size = None

        # NOTE: In EXAONE4, only sliding layers apply rope.
        self.sliding_window = config.sliding_window
        self.is_sliding = check_is_sliding(config, layer_idx)
        pos_embd_params = None
        if self.sliding_window is None or self.is_sliding:
            self.attention_window_size = config.sliding_window

            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )

        fuse_qk_norm_rope = (self.is_sliding and fuse_qk_norm_rope)

        # TODO: Fusing qk norm with rope has an issue that slightly hurts accuracy.
        assert fuse_qk_norm_rope is False, "Fusing qk norm and rope is having issue now"

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=fuse_qk_norm_rope,
            skip_rope=self.sliding_window and not self.is_sliding,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:

        # TODO LoRA has not been tested yet but there is no need to prevent it.
        assert lora_params is None, "LORA is not supported for Exaone4Attention"

        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            lora_params=lora_params,
            attention_window_size=self.attention_window_size,
            **kwargs,
        )


class Exaone4DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Exaone4Config],
        layer_idx: int,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.is_quanted = model_config.quant_config and model_config.quant_config.quant_mode.has_any_quant(
        )

        self.self_attn = Exaone4Attention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=getattr(config, "mlp_bias", False),
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.post_feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype)

        self.mapping = model_config.mapping

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:

        residual = hidden_states

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states


class Exaone4Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[Exaone4Config]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )

        self.layers = nn.ModuleList([
            Exaone4DecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(self.num_hidden_layers)
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
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params=None,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at "
                "the same time, and must specify either one.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds.to(self.dtype)

        for decoder_layer in self.layers[:self.num_hidden_layers]:
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("Exaone4ForCausalLM")
class Exaone4ForCausalLM(DecoderModelForCausalLM[Exaone4Model, Exaone4Config]):

    def __init__(
        self,
        model_config: ModelConfig[Exaone4Config],
    ):
        model_config.pretrained_config.torch_dtype = torch.bfloat16
        super().__init__(Exaone4Model(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
