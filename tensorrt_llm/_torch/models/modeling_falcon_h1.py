import re
import time
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import FalconH1Config

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.models.modeling_utils import (DecoderModel,
                                                      DecoderModelForCausalLM,
                                                      register_auto_model)


class FalconH1ParallelHybridLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[FalconH1Config], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config

        self.layer_idx = layer_idx

        # Branch modules
        self.self_attn = Attention(hidden_size=config.hidden_size,
                                   num_attention_heads=config.num_attention_heads,
                                   num_key_value_heads=config.num_key_value_heads,
                                   max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
                                   bias=False,
                                   pos_embd_params=None,
                                   layer_idx=layer_idx,
                                   dtype=config.torch_dtype,
                                   config=model_config)

        self.mamba = Mamba2Mixer(d_model=config.hidden_size,
                                 d_state=config.mamba_d_state,
                                 d_conv=config.mamba_d_conv,
                                 nheads=config.mamba_n_heads,
                                 n_groups=config.mamba_n_groups,
                                 head_dim=config.mamba_d_head,
                                 chunk_size=getattr(config, "mamba_chunk_size", getattr(config, "chunk_size", 128)),
                                 layer_idx=layer_idx,
                                 rms_norm_eps=config.rms_norm_eps,
                                 dtype=config.torch_dtype,
                                 config=model_config)

        # FFN uses fused gate_up and down with SiLU gating
        intermediate_size = (config.intermediate_size[0]
                             if isinstance(config.intermediate_size, list)
                             else config.intermediate_size)
        self.feed_forward = MLP(hidden_size=config.hidden_size,
                                intermediate_size=intermediate_size,
                                bias=False,
                                activation=F.silu,
                                dtype=config.torch_dtype,
                                config=model_config)

        # Norms
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.pre_ff_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                        eps=config.rms_norm_eps,
                                        dtype=config.torch_dtype)

        # Multipliers (default to 1.0 if missing)
        self.register_buffer("_ssm_in_mul", torch.tensor(getattr(config, "ssm_in_multiplier", 1.0), dtype=torch.float32), persistent=False)
        self.register_buffer("_ssm_out_mul", torch.tensor(getattr(config, "ssm_out_multiplier", 1.0), dtype=torch.float32), persistent=False)
        self.register_buffer("_attn_in_mul", torch.tensor(getattr(config, "attention_in_multiplier", 1.0), dtype=torch.float32), persistent=False)
        self.register_buffer("_attn_out_mul", torch.tensor(getattr(config, "attention_out_multiplier", 1.0), dtype=torch.float32), persistent=False)

    def forward(self,
                position_ids: torch.IntTensor,
                hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata,
                **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Attention branch
        attn_input = hidden_states * self._attn_in_mul
        attn_hidden = self.self_attn(position_ids=None,
                                     hidden_states=attn_input,
                                     attn_metadata=attn_metadata)

        # Mamba branch
        ssm_input = hidden_states * self._ssm_in_mul
        mamba_hidden = self.mamba(ssm_input, attn_metadata, **kwargs)

        hidden_states = attn_hidden * self._attn_out_mul + mamba_hidden * self._ssm_out_mul
        hidden_states = hidden_states + residual

        # FFN
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class FalconH1Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[FalconH1Config]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config

        # embeddings
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        # layers
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            layers.append(FalconH1ParallelHybridLayer(model_config, layer_idx))
        self.layers = nn.ModuleList(layers)

        # final norm
        self.final_layernorm = RMSNorm(hidden_size=config.hidden_size,
                              eps=config.rms_norm_eps,
                              dtype=config.torch_dtype)

        self.mamba_metadata: Optional[Mamba2Metadata] = None

        embed_mul = getattr(config, "embedding_multiplier", 1.0)
        self.register_buffer("_embed_mul", torch.tensor(embed_mul, dtype=torch.float32), persistent=False)

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: Optional[torch.IntTensor] = None,
                position_ids: Optional[torch.IntTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                **kwargs) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.mamba_metadata is None or self.mamba_metadata.max_batch_size != attn_metadata.max_num_requests:
            chunk_size = getattr(self.model_config.pretrained_config, "mamba_chunk_size", getattr(self.model_config.pretrained_config, "chunk_size", 128))
            self.mamba_metadata = Mamba2Metadata(attn_metadata.max_num_requests, chunk_size=chunk_size)
        self.mamba_metadata.prepare(attn_metadata)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds * self._embed_mul

        for layer in self.layers:
            hidden_states = layer(position_ids,
                                  hidden_states,
                                  attn_metadata,
                                  mamba_metadata=self.mamba_metadata)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


@register_auto_model("FalconH1ForCausalLM")
class FalconH1ForCausalLM(DecoderModelForCausalLM[FalconH1Model, FalconH1Config]):

    def __init__(self, model_config: ModelConfig[FalconH1Config]):
        # import debugpy
        # debugpy.listen(5678)
        # debugpy.wait_for_client()

        if not model_config.mapping.tp_size in [1, 2, 4, 8]:
            raise ValueError("TP has to be either 1, 2, 4 or 8")

        if model_config.quant_config.exclude_modules is not None:
            model_config.quant_config.exclude_modules = [
                re.sub(r'(model\.layers\.)?backbone', 'model', k)
                for k in model_config.quant_config.exclude_modules
            ]

        super().__init__(FalconH1Model(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: dict, weight_mapper: BaseWeightMapper):
        # Normalize checkpoint keys for Mamba2Mixer compatibility
        def _normalize_keys(w: dict) -> dict:
            norm = {}
            for k, v in w.items():
                nk = k
                if '.mamba.A_log' in nk:
                    nk = nk.replace('.mamba.A_log', '.mamba.A')
                if '.dt_proj.bias' in nk:
                    nk = nk.replace('.dt_proj.bias', '.dt_bias')
                # Flatten conv1d weights [out, 1, k] -> [out, k]
                if nk.endswith('.mamba.conv1d.weight') and v.ndim == 3 and v.shape[1] == 1:
                    v = v.squeeze(1).contiguous()
                norm[nk] = v
            return norm

        weights = _normalize_keys(weights)

        new_weights = weight_mapper.preprocess_weights(weights)
        if new_weights is None:
            new_weights = weights
        else:
            new_weights = _normalize_keys(new_weights)
        super().load_weights(new_weights, weight_mapper)


