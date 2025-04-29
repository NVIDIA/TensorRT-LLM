from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings

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


# --- Attention Implementation ---
class CLIPAttention(Attention):

    # Updated to only take CLIPVisionConfig
    def __init__(self, model_config: ModelConfig[CLIPVisionConfig],
                 layer_idx: int):
        config = model_config.pretrained_config
        pos_embd_params = None
        rotary_emb = None

        # Vision model has class token
        num_patches = (config.image_size // config.patch_size)**2
        max_position_embeddings = num_patches + 1

        # CLIP uses bias in attention QKV projections
        bias = True
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,  # CLIP uses MHA
            max_position_embeddings=max_position_embeddings,
            bias=bias,
            rotary_emb=rotary_emb,
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
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states if hasattr(
                self.config, 'output_hidden_states') else False)

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            if output_hidden_states:
                # hidden_states is (batch_size * seq_len, embed_dim) because TRT-LLM Attention is applied to flattened tokens
                seq_len = (
                    self.config.image_size // self.config.patch_size
                )**2 + 1  # the "seq_len" is fixed, it is the number of patches +1 (cls token).
                batch_size = hidden_states.shape[0] // seq_len
                # we want the output shape align with HF output shape (batch_size, seq_len, embed_dim)
                encoder_states = encoder_states + (hidden_states.view(
                    batch_size, seq_len, -1), )
                # print(
                #     f"hidden_states is added with shape: {hidden_states.view(batch_size, seq_len, -1).shape}, batch_size: {batch_size}, seq_len: {seq_len}"
                # )

            layer_outputs = encoder_layer(
                hidden_states,
                attn_metadata=attn_metadata,
            )
            hidden_states = layer_outputs[0]

        if output_hidden_states:
            # hidden_states is (batch_size * seq_len, embed_dim) because TRT-LLM Attention is applied to flattened tokens
            seq_len = (
                self.config.image_size // self.config.patch_size
            )**2 + 1  # the "seq_len" is fixed, it is the number of patches. Note that we have a cls token for clip, so we need to add 1
            batch_size = hidden_states.shape[0] // seq_len
            # we want the output shape align with HF output shape (batch_size, seq_len, embed_dim)
            encoder_states = encoder_states + (hidden_states.view(
                batch_size, seq_len, -1), )
            # print(
            #     f"hidden_states is added with shape: {hidden_states.view(batch_size, seq_len, -1).shape}, batch_size: {batch_size}, seq_len: {seq_len}"
            # )

        return BaseModelOutput(
            last_hidden_state=
            hidden_states,  # Keep it flattened for subsequent layers/pooling
            hidden_states=encoder_states)


# --- Vision Transformer ---
# Uses HF CLIPVisionEmbeddings


class CLIPVisionTransformer(nn.Module):

    def __init__(self, model_config: ModelConfig[CLIPVisionConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.config = config
        embed_dim = config.hidden_size

        # Use HF Embeddings
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(model_config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values,
        attn_metadata: AttentionMetadata,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> BaseModelOutputWithPooling:

        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)

        hidden_states = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        hidden_states = self.pre_layrnorm(hidden_states)

        # Reshape for TRT-LLM Attention: (batch * seq_len, hidden)
        original_shape = hidden_states.shape  # (batch, seq_len, hidden)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0] * hidden_states.shape[1],
            hidden_states.shape[2])
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attn_metadata=attn_metadata,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state_flat = encoder_outputs.last_hidden_state

        # Reshape last_hidden_state back to (batch, seq, hidden) for pooling
        last_hidden_state = last_hidden_state_flat.view(original_shape)

        # Pooling: Use the CLS token (first token) output
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)  # Apply final LN

        # Pass the reshaped hidden_states if requested
        returned_hidden_states = encoder_outputs.hidden_states

        return BaseModelOutputWithPooling(
            # Return the unflattened last_hidden_state
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=returned_hidden_states,
            # Attentions are not returned by TRT-LLM Attention module by default
            attentions=None,
        )


# --- Vision Model ---
@register_auto_model("CLIPVisionModel")
class CLIPVisionModel(nn.Module):

    def __init__(self, model_config: ModelConfig[CLIPVisionConfig]):
        super().__init__()
        # No need to check for CLIPConfig anymore
        if not isinstance(model_config.pretrained_config, CLIPVisionConfig):
            raise ValueError(
                "Invalid config type for CLIPVisionModel, expected CLIPVisionConfig"
            )

        self.model_config = model_config
        self.config = self.model_config.pretrained_config  # HF Vision Config
        self.vision_model = CLIPVisionTransformer(self.model_config)

    @property
    def dtype(self):
        # Infer dtype from a parameter
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def forward(self,
                pixel_values,
                attn_metadata: AttentionMetadata,
                output_hidden_states: Optional[bool] = None,
                interpolate_pos_encoding: Optional[bool] = False):

        return self.vision_model(
            pixel_values=pixel_values,
            attn_metadata=attn_metadata,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding)

    def load_weights(self, weights: Dict):
        # FIXME: Temporary import
        from .modeling_utils import (duplicate_kv_weight,
                                     missing_layer_parameter,
                                     rename_weights_with_regex)

        # FIXME: Temporary import
        # Pattern mapping for CLIP based on Siglip's example and CLIP HF names
        pattern_mapping = {
            r'(.*?)self_attn\.out_proj(.*)': r'\1self_attn.o_proj\2',
            r'(.*?)mlp\.fc1(.*)': r'\1mlp.up_proj\2',
            r'(.*?)mlp\.fc2(.*)': r'\1mlp.down_proj\2',
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
                # print(f"loading {name}")

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
                    # print(
                    #     f"loaded {name}: {[w_dic.keys() for w_dic in module_weights]}"
                    # )
                else:
                    module_weights = filter_weights(name, weights)
                    # print(f"module_weights: {module_weights.keys()}")
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                p.data.copy_(module_weights[n][:])
                    # print(f"loaded {name}: {module_weights.keys()}")
