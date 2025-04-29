from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    SiglipMultiheadAttentionPoolingHead, SiglipVisionConfig,
    SiglipVisionEmbeddings)

from ..attention_backend.interface import (AttentionMetadata,
                                           PredefinedAttentionMask)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.mlp import MLP
from .modeling_utils import register_auto_model
"""
NOTE:
Configuration Class from HF to TRT-LLM ModelConfig naming convention:
In this file, all `config`s or `self.config`s are HF Configs
all `model_config`s or `self.model_config`s are TRT-LLM ModelConfig
At the same time, the `model_config.pretrained_config` is also HF Config as this is defined by the ModelConfig class
"""


class SiglipAttention(Attention):

    def __init__(self, model_config: ModelConfig[SiglipVisionConfig],
                 layer_idx: int):
        config = model_config.pretrained_config
        pos_embd_params = None
        rotary_emb = None
        num_patches = (config.image_size // config.patch_size)**2
        # the max_position_embeddings is the number of patches for Vision Transformer
        max_position_embeddings = num_patches
        # Siglip uses bias in attention, ref:https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L399
        bias = True
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            bias=bias,
            rotary_emb=rotary_emb,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


class SiglipEncoderLayer(nn.Module):

    def __init__(self, model_config: ModelConfig[SiglipVisionConfig],
                 layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(model_config=model_config,
                                         layer_idx=layer_idx)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        # Siglip uses bias in MLP, ref:https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L458
        self.mlp = MLP(hidden_size=config.hidden_size,
                       intermediate_size=config.intermediate_size,
                       activation=ACT2FN[config.hidden_act],
                       bias=True)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        # FIXME: temporary config for AttentionMetadata and Mask
        attention_mask = PredefinedAttentionMask.FULL
        hidden_states = self.self_attn(position_ids=None,
                                       hidden_states=hidden_states,
                                       attn_metadata=attn_metadata,
                                       attention_mask=attention_mask)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )

        return outputs


class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(self, model_config: ModelConfig[SiglipVisionConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.config = config
        self.layers = nn.ModuleList([
            SiglipEncoderLayer(model_config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attn_metadata: AttentionMetadata,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        """
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)

        encoder_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                # hidden_states is (batch_size * seq_len, embed_dim) because TRT-LLM Attention is applied to flattened tokens
                seq_len = (
                    self.config.image_size // self.config.patch_size
                )**2  # the "seq_len" is fixed, it is the number of patches
                batch_size = hidden_states.shape[0] // seq_len
                # we want the output shape align with HF output shape (batch_size, seq_len, embed_dim)
                encoder_states = encoder_states + (hidden_states.view(
                    batch_size, seq_len, -1), )
                print(
                    f"hidden_states is added with shape: {hidden_states.view(batch_size, seq_len, -1).shape}, batch_size: {batch_size}, seq_len: {seq_len}"
                )
            layer_outputs = encoder_layer(
                hidden_states,
                attn_metadata=attn_metadata,
            )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            # same as above
            seq_len = (self.config.image_size // self.config.patch_size)**2
            batch_size = hidden_states.shape[0] // seq_len
            encoder_states = encoder_states + (hidden_states.view(
                batch_size, seq_len, -1), )

        return BaseModelOutput(last_hidden_state=hidden_states,
                               hidden_states=encoder_states)


class SiglipVisionTransformer(nn.Module):

    def __init__(self, model_config: ModelConfig[SiglipVisionConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(model_config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(
            config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        attn_metadata: AttentionMetadata,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> BaseModelOutputWithPooling:
        r"""
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            attn_metadata (`AttentionMetadata`):
                Attention metadata.
        """
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)

        hidden_states = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        print(f"hidden_states: {hidden_states.shape}")
        # reshape hidden_states to (batch_size * seq_len, embed_dim)
        # TODO: check the overhead here because of the reshape
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0] * hidden_states.shape[1],
            hidden_states.shape[2])
        print(f"after reshape hidden_states: {hidden_states.shape}")
        print(f"attn_metadata: {attn_metadata}")

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attn_metadata=attn_metadata,
            output_hidden_states=output_hidden_states,
        )
        print(f"encoder_outputs: {encoder_outputs}")
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state) if self.use_head else None

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@register_auto_model("SiglipVisionModel")
class SiglipVisionModel(nn.Module):

    def __init__(self, model_config: ModelConfig[SiglipVisionConfig]):
        super().__init__()
        self.config = model_config.pretrained_config
        self.vision_model = SiglipVisionTransformer(model_config)
        self.model_config = model_config

    @property
    def dtype(self):
        return self.config.torch_dtype

    def forward(self,
                pixel_values,
                attn_metadata: AttentionMetadata,
                output_hidden_states: Optional[bool] = None):
        return self.vision_model(pixel_values, attn_metadata,
                                 output_hidden_states)

    def load_weights(self, weights: Dict):
        # FIXME: Temporary import
        from .modeling_utils import (duplicate_kv_weight,
                                     missing_layer_parameter,
                                     rename_weights_with_regex)

        # FIXME: Temporary import

        pattern_mapping = {
            r'(.*?)out_proj(.*)': r'\1o_proj\2',
            r'(.*?)fc1(.*)': r'\1up_proj\2',
            r'(.*?)fc2(.*)': r'\1down_proj\2',
        }
        weights = rename_weights_with_regex(pattern_mapping, weights)

        tp_size = self.model_config.mapping.tp_size
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        params_map = {
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
            'gate_up_proj': ['gate_proj', 'up_proj']
        }
        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) > 0:
                # skip load weights if tie word embeddings is enabled and layer is lm_head
                if self.config.tie_word_embeddings and name.startswith(
                        "lm_head"):
                    continue

                # Skip if parameter belongs to a missing layer
                if missing_layer_parameter(name, self):
                    continue
                print(f"loading {name}")

                names = name.split('.')
                if names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        fw = filter_weights('.'.join(names[:-1] + [new_name]),
                                            weights)
                        if new_name in ['k_proj', 'v_proj']:
                            fw = {
                                k:
                                duplicate_kv_weight(
                                    weight=v[:],
                                    head_dim=head_dim,
                                    tensor_parallel_size=tp_size)
                                if k in ["weight", "bias"] else v
                                for k, v in fw.items()
                            }
                        module_weights.append(fw)
                    module.load_weights(weights=module_weights)
                    print(
                        f"loaded {name}: {[w_dic.keys() for w_dic in module_weights]}"
                    )
                else:
                    module_weights = filter_weights(name, weights)
                    print(f"module_weights: {module_weights.keys()}")
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                p.data.copy_(module_weights[n][:])
                    print(f"loaded {name}: {module_weights.keys()}")
