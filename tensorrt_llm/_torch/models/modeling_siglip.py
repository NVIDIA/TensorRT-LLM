from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import (get_parameter_device,
                                         get_parameter_dtype)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (
    SiglipMultiheadAttentionPoolingHead, SiglipVisionConfig,
    SiglipVisionEmbeddings)

from ..attention_backend.interface import AttentionMetadata
from ..attention_backend.utils import get_attention_backend
from ..model_config import ModelConfig
from .modeling_clip import CLIPEncoder
from .modeling_utils import (duplicate_kv_weight, missing_layer_parameter,
                             register_auto_model, rename_weights_with_regex)

SiglipEncoder = CLIPEncoder


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
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)

        hidden_states = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        # reshape hidden_states to (batch_size * seq_len, embed_dim)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0] * hidden_states.shape[1],
            hidden_states.shape[2])

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attn_metadata=attn_metadata,
            output_hidden_states=output_hidden_states,
        )
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
        self.metadata_cls = get_attention_backend(
            model_config.attn_backend).Metadata

    def prepare_attn_metadata(self, batch_size):
        """
        To simplify the usage of the model, this function aims to fill the metadata for Attention
        Call this function before forward pass
        """
        seq_len = (self.config.image_size // self.config.patch_size)**2
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
                output_hidden_states: Optional[bool] = None):
        return self.vision_model(pixel_values, attn_metadata,
                                 output_hidden_states)

    def load_weights(self, weights: Dict):
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
