from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

try:
    from transformer_engine.pytorch import RMSNorm
except ImportError:
    RMSNorm = None
from transformers import AutoConfig, PretrainedConfig

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.embedding import Embedding
from ..modules.mamba.mixer import MambaMixer
from ..modules.mlp import MLP
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


def split(x: torch.Tensor,
          tp_size: int,
          idx: int,
          dim: int = 0) -> torch.Tensor:
    assert x.shape[dim] % tp_size == 0
    split_size = x.shape[dim] // tp_size
    if tp_size == 1:
        return x
    return torch.split(x, split_size, dim=dim)[idx]


def relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))


class NemotronHConfig(PretrainedConfig):
    model_type = "nemotron_h"


class MLPLayer(MLP):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        super().__init__(hidden_size=config.hidden_size,
                         intermediate_size=config.intermediate_size,
                         bias=False,
                         activation=relu2,
                         dtype=config.torch_dtype,
                         config=model_config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return super().forward(hidden_states)


class TransformerLayer(Attention):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=None,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return super().forward(position_ids=None,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata)


class NemotronHLayer(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
        # M -> MambaLayer
        # - -> MLPLayer
        # * -> TransformerLayer
        layer_type: str,
    ):
        super().__init__()

        config = model_config.pretrained_config

        self.layer_idx = layer_idx
        self.layer_type = layer_type

        assert RMSNorm is not None, "RMSNorm from transformer_engine is not installed, install it with `pip3 install transformer_engine[pytorch]`"

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        if layer_type == "M":
            self.mixer = MambaMixer(d_model=config.hidden_size,
                                    d_state=config.ssm_state_size,
                                    d_conv=config.conv_kernel,
                                    expand=config.expand,
                                    n_groups=config.n_groups,
                                    head_dim=config.mamba_head_dim,
                                    chunk_size=config.chunk_size,
                                    layer_idx=layer_idx,
                                    rms_norm_eps=config.rms_norm_eps,
                                    dtype=config.torch_dtype,
                                    config=model_config)
        elif layer_type == "-":
            self.mixer = MLPLayer(model_config, layer_idx)
        elif layer_type == "*":
            self.mixer = TransformerLayer(model_config, layer_idx)
        else:
            ValueError(f"{layer_type} is not supported")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states, attn_metadata)
        hidden_states = torch.add(hidden_states, residual)

        return hidden_states


class NemotronHModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[NemotronHConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config

        # calculate embeddings
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        # create layers
        mlp_idx = 0
        mamba_idx = 0
        transformer_idx = 0
        layers = []
        for layer_type in config.hybrid_override_pattern:
            # calculate layer index based on type
            if layer_type == "M":
                layer_idx = mamba_idx
                mamba_idx += 1
            elif layer_type == "-":
                layer_idx = mlp_idx
                mlp_idx += 1
            elif layer_type == "*":
                layer_idx = transformer_idx
                transformer_idx += 1
            else:
                ValueError(f"{layer_type} is not supported")
            layers.append(NemotronHLayer(model_config, layer_idx, layer_type))
        self.layers = nn.ModuleList(layers)

        assert RMSNorm is not None, "RMSNorm from transformer_engine is not installed, install it with `pip3 install transformer_engine[pytorch]`"

        # final norm
        self.norm_f = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_metadata)

        hidden_states = self.norm_f(hidden_states)

        return hidden_states


@register_auto_model("NemotronHForCausalLM")
class NemotronHForCausalLM(DecoderModelForCausalLM[NemotronHModel,
                                                   NemotronHConfig]):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
    ):
        if not model_config.mapping.tp_size in [1, 2, 4, 8]:
            raise ValueError("TP has to be either 1, 2, 4 or 8")
        super().__init__(
            NemotronHModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(self, weights: Dict):
        config = self.model_config.pretrained_config
        tp_size = self.model_config.mapping.tp_size
        tp_rank = self.model_config.mapping.tp_rank
        d_inner = config.hidden_size * config.expand
        n_groups = config.n_groups
        d_state = config.ssm_state_size
        nheads = d_inner // config.mamba_head_dim

        new_weights = {}
        for name, params in weights.items():
            key = name

            # change backbone root name to model
            if "backbone" in key:
                key = key.replace("backbone", "model")

            # change embedding layer to embed_token
            if "embeddings" in key:
                key = key.replace("embeddings", "embed_tokens")

            if "A_log" in key:
                key = key.replace("A_log", "A")

            if "A" in key:
                w = split(weights[name], tp_size, tp_rank)
                w = w.to(torch.float32)
                w = -torch.exp(w)
                new_weights[key] = w
            elif "D" in key:
                w = split(weights[name], tp_size, tp_rank)
                w = w.to(torch.float32)
                new_weights[key] = w
            elif "dt_bias" in key:
                w = split(weights[name], tp_size, tp_rank)
                w = w.to(torch.float32)
                new_weights[key] = w
            elif "mixer.in_proj" in key:
                w = weights[name]
                in_proj_z, in_proj_x, in_proj_b, in_proj_c, in_proj_dt = torch.split(
                    w, [
                        d_inner, d_inner, n_groups * d_state,
                        n_groups * d_state, nheads
                    ],
                    dim=0)

                w = []
                for rank in range(tp_size):
                    in_proj_z_rank = split(in_proj_z, tp_size, rank)
                    in_proj_x_rank = split(in_proj_x, tp_size, rank)
                    in_proj_b_rank = split(in_proj_b, tp_size, rank)
                    in_proj_c_rank = split(in_proj_c, tp_size, rank)
                    in_proj_dt_rank = split(in_proj_dt, tp_size, rank)
                    y = torch.concat([
                        in_proj_z_rank, in_proj_x_rank, in_proj_b_rank,
                        in_proj_c_rank, in_proj_dt_rank
                    ])
                    w.append(y)

                w = torch.concat(w).contiguous()
                new_weights[key] = w
            elif "conv1d" in key:
                w = weights[name]
                # removing dim(1) because we are using Linear to store conv1d weights
                if "weight" in key:
                    w = w.squeeze(1)

                conv_x, conv_b, conv_c = torch.split(
                    w, [d_inner, n_groups * d_state, n_groups * d_state], dim=0)

                w = []
                for rank in range(tp_size):
                    conv_x_rank = split(conv_x, tp_size, rank)
                    conv_b_rank = split(conv_b, tp_size, rank)
                    conv_c_rank = split(conv_c, tp_size, rank)
                    y = torch.concat([conv_x_rank, conv_b_rank, conv_c_rank])
                    w.append(y)
                w = torch.concat(w).contiguous()
                new_weights[key] = w
            elif "mixer.norm.weight" in key:
                w = split(weights[name], tp_size, tp_rank)
                new_weights[key] = w
            else:
                new_weights[key] = weights[name]

        super().load_weights(new_weights)


AutoConfig.register(NemotronHConfig.model_type, NemotronHConfig)
