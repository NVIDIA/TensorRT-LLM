from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
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
from .modeling_utils import (duplicate_kv_weight, missing_layer_parameter,
                             register_auto_model, rename_weights_with_regex)


class CLIPAttention(Attention):

    def __init__(self, model_config: ModelConfig[CLIPVisionConfig],
                 layer_idx: int):
        config = model_config.pretrained_config
        pos_embd_params = None

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
                # we want the output shape align with HF output shape (batch_size, seq_len, embed_dim)
                encoder_states = encoder_states + (hidden_states.view(
                    attn_metadata.seq_lens.shape[0], attn_metadata.seq_lens[0],
                    -1), )

            layer_outputs = encoder_layer(
                hidden_states,
                attn_metadata=attn_metadata,
            )
            hidden_states = layer_outputs[0]

        if output_hidden_states:
            # hidden_states is (batch_size * seq_len, embed_dim) because TRT-LLM Attention is applied to flattened tokens
            # we want the output shape align with HF output shape (batch_size, seq_len, embed_dim)
            encoder_states = encoder_states + (hidden_states.view(
                attn_metadata.seq_lens.shape[0], attn_metadata.seq_lens[0],
                -1), )

        return BaseModelOutput(
            last_hidden_state=
            hidden_states,  # Keep it flattened for subsequent layers/pooling
            hidden_states=encoder_states)


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
        pooled_output = self.post_layernorm(pooled_output)

        returned_hidden_states = encoder_outputs.hidden_states

        return BaseModelOutputWithPooling(
            # Return the unflattened last_hidden_state
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=returned_hidden_states,
            attentions=None,
        )


@register_auto_model("CLIPVisionModel")
class CLIPVisionModel(nn.Module):

    def __init__(self, model_config: ModelConfig[CLIPVisionConfig]):
        super().__init__()
        self.model_config = model_config
        self.config = self.model_config.pretrained_config  # HF Vision Config
        self.vision_model = CLIPVisionTransformer(self.model_config)
        self.metadata_cls = get_attention_backend(
            model_config.attn_backend).Metadata

    def prepare_attn_metadata(self, batch_size):
        """
        To simplify the usage of the model, this function aims to fill the metadata for Attention
        Call this function before forward pass
        """
        seq_len = (self.config.image_size // self.config.patch_size)**2 + 1
        request_ids = list(range(1, batch_size + 1))
        prompt_lens = [seq_len] * batch_size
        attn_metadata = self.metadata_cls(
            seq_lens=torch.tensor([seq_len] * batch_size, dtype=torch.int),
            num_contexts=batch_size,
            max_num_requests=batch_size,
            max_num_tokens=seq_len * batch_size,
            kv_cache_manager=None,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )
        attn_metadata.max_seq_len = seq_len * batch_size
        attn_metadata.prepare()
        return attn_metadata

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
                output_hidden_states: Optional[bool] = None,
                interpolate_pos_encoding: Optional[bool] = False):

        return self.vision_model(
            pixel_values=pixel_values,
            attn_metadata=attn_metadata,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding)

    def load_weights(self, weights: Dict):

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
                else:
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module._parameters.items():
                            if p is not None:
                                p.data.copy_(module_weights[n][:])
