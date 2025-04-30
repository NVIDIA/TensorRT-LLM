from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_utils import (get_parameter_device,
                                         get_parameter_dtype)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import (SiglipVisionConfig,
                                                        SiglipVisionEmbeddings)

from ..attention_backend.interface import AttentionMetadata
from ..attention_backend.utils import get_attention_backend
from ..model_config import ModelConfig
from .modeling_clip import CLIPEncoder
from .modeling_utils import _load_weights_impl, register_auto_model

SiglipEncoder = CLIPEncoder


class SiglipVisionTransformer(nn.Module):
    """
    This SiglipVisionTransformer is tailored for multimodal models that use Siglip as the vision encoder.
    For example, it is different from the regular SiglipVisionTransformer in the sense that it does not return a pooled output.
    """

    def __init__(self, model_config: ModelConfig[SiglipVisionConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.config = config

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(model_config)
        if hasattr(config, "vision_use_head"):
            assert not config.vision_use_head, "Currently, we only support vision_use_head = False"

    def forward(
        self,
        pixel_values,
        attn_metadata: AttentionMetadata,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        hidden_states = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        # reshape hidden_states to (batch_size * seq_len, embed_dim)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0] * hidden_states.shape[1],
            hidden_states.shape[2])

        encoder_outputs: Tuple[torch.Tensor] = self.encoder(
            inputs_embeds=hidden_states,
            attn_metadata=attn_metadata,
        )

        return encoder_outputs


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
    def forward(self, pixel_values, attn_metadata: AttentionMetadata):
        return self.vision_model(
            pixel_values=pixel_values,
            attn_metadata=attn_metadata,
        )

    def load_weights(self, weights: Dict):
        pattern_mapping = {
            r'(.*?)out_proj(.*)': r'\1o_proj\2',
            r'(.*?)fc1(.*)': r'\1up_proj\2',
            r'(.*?)fc2(.*)': r'\1down_proj\2',
        }
        _load_weights_impl(self, weights, pattern_mapping)
