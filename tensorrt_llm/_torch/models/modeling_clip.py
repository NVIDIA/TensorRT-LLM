from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import (get_parameter_device,
                                         get_parameter_dtype)
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings

from ..attention_backend.interface import (AttentionMetadata,
                                           PredefinedAttentionMask)
from ..attention_backend.utils import get_attention_backend
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.mlp import MLP
from .modeling_utils import _load_weights_impl, register_auto_model


class CLIPAttention(Attention):

    def __init__(self, model_config: ModelConfig[CLIPVisionConfig],
                 layer_idx: int):
        config = model_config.pretrained_config
        pos_embd_params = None
        max_position_embeddings = None

        # CLIP uses bias in attention QKV projections
        bias = True
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # CLIP uses MHA
            max_position_embeddings=
            max_position_embeddings,  # does not matter for CLIP
            bias=bias,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype
            if hasattr(config, 'torch_dtype') else torch.float32,
            config=model_config,
        )


class CLIPEncoderLayer(nn.Module):

    def __init__(self, model_config: ModelConfig[CLIPVisionConfig],
                 layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(model_config=model_config,
                                       layer_idx=layer_idx)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=ACT2FN[config.hidden_act],
            bias=True,  # CLIP MLP bias=True
            config=model_config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.self_attn(
            position_ids=None,  # CLIP doesn't use explicit position_ids here
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.
            FULL  # Always FULL for Vision
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        return outputs


class CLIPEncoder(nn.Module):

    def __init__(self, model_config: ModelConfig[CLIPVisionConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.config = config  # Keep HF config accessible
        self.model_config = model_config  # Keep TRT-LLM config accessible
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(model_config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds,
        attn_metadata: AttentionMetadata,
    ) -> Union[Tuple, BaseModelOutput]:

        encoder_states = ()
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # hidden_states is (batch_size * seq_len, embed_dim) because TRT-LLM Attention is applied to flattened tokens
            # we want the output shape align with HF output shape (batch_size, seq_len, embed_dim)
            encoder_states = encoder_states + (hidden_states.view(
                attn_metadata.seq_lens.shape[0], attn_metadata.seq_lens[0],
                -1), )

            layer_outputs = encoder_layer(
                hidden_states,
                attn_metadata=attn_metadata,
            )
            hidden_states = layer_outputs[0]

        # hidden_states is (batch_size * seq_len, embed_dim) because TRT-LLM Attention is applied to flattened tokens
        # we want the output shape align with HF output shape (batch_size, seq_len, embed_dim)
        encoder_states = encoder_states + (hidden_states.view(
            attn_metadata.seq_lens.shape[0], attn_metadata.seq_lens[0], -1), )

        return encoder_states


class CLIPVisionTransformer(nn.Module):
    """
    This CLIPVisionTransformer is tailored for multimodal models that use CLIP as the vision encoder.
    For example, it is different from the regular CLIPVisionTransformer in the sense that it does not return a pooled output.
    """

    def __init__(self, model_config: ModelConfig[CLIPVisionConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.config = config
        embed_dim = config.hidden_size

        # Use HF Embeddings
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(model_config)

    def forward(
        self,
        pixel_values,
        attn_metadata: AttentionMetadata,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        hidden_states = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        hidden_states = self.pre_layrnorm(hidden_states)

        # Reshape for TRT-LLM Attention: (batch * seq_len, hidden)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0] * hidden_states.shape[1],
            hidden_states.shape[2])
        encoder_outputs: Tuple[torch.Tensor] = self.encoder(
            inputs_embeds=hidden_states,
            attn_metadata=attn_metadata,
        )

        return encoder_outputs


@register_auto_model("CLIPVisionModel")
class CLIPVisionModel(nn.Module):

    def __init__(self, model_config: ModelConfig[CLIPVisionConfig]):
        super().__init__()
        self.model_config = model_config
        self.config = self.model_config.pretrained_config  # HF Vision Config
        self.vision_model = CLIPVisionTransformer(self.model_config)

        # Needed for prepare_attn_metadata
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size

        self.metadata_cls = get_attention_backend(
            model_config.attn_backend).Metadata
        self.attn_metadata = self.metadata_cls(
            max_num_requests=
            8192,  #TODO(yechank-nvidia): Make this along with the LLM's max_num_requests
            max_num_tokens=model_config.max_num_tokens,
            kv_cache_manager=None,
        )

    def prepare_attn_metadata(self, batch_size):
        """
        To simplify the usage of the model, this function aims to fill the metadata for Attention
        Call this function before forward pass
        """
        seq_len = (self.image_size // self.patch_size)**2 + 1
        request_ids = list(range(1, batch_size + 1))
        prompt_lens = [seq_len] * batch_size
        seq_lens = torch.tensor([seq_len] * batch_size,
                                dtype=torch.int,
                                pin_memory=True)

        self.attn_metadata.num_contexts = batch_size
        self.attn_metadata.request_ids = request_ids
        self.attn_metadata.prompt_lens = prompt_lens
        self.attn_metadata.seq_lens = seq_lens
        self.attn_metadata.max_seq_len = seq_len
        self.attn_metadata.prepare()
        return self.attn_metadata

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    @property
    def device(self):
        return get_parameter_device(self)

    @torch.inference_mode()
    def forward(self,
                pixel_values,
                attn_metadata: AttentionMetadata,
                interpolate_pos_encoding: Optional[bool] = False):

        return self.vision_model(
            pixel_values=pixel_values,
            attn_metadata=attn_metadata,
            interpolate_pos_encoding=interpolate_pos_encoding)

    def load_weights(self, weights: Dict):
        # Pattern mapping for CLIP based on Siglip's example and CLIP HF names
        pattern_mapping = {
            r'(.*?)self_attn\.out_proj(.*)': r'\1self_attn.o_proj\2',
            r'(.*?)mlp\.fc1(.*)': r'\1mlp.up_proj\2',
            r'(.*?)mlp\.fc2(.*)': r'\1mlp.down_proj\2',
        }
        _load_weights_impl(self, weights, params_map=pattern_mapping)
