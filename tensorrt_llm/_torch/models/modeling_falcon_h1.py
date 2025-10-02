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
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.models.modeling_utils import (DecoderModel,
                                                      DecoderModelForCausalLM,
                                                      register_auto_model)
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm._torch.attention_backend.interface import (
    PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.modules.linear import (Linear, TensorParallelMode,
                                                WeightMode, WeightsLoadingConfig)

class FalconH1MLP(nn.Module):

    def __init__(self, model_config: ModelConfig[FalconH1Config],
                 intermediate_size: int, bias: bool = False):
        super().__init__()
        config = model_config.pretrained_config

        # Handle list intermediate_size
        self.intermediate_size = (intermediate_size[0]
                                  if isinstance(intermediate_size, list)
                                  else intermediate_size)

        self.hidden_size = config.hidden_size
        self.dtype = config.torch_dtype
        self.tp_size = model_config.mapping.tp_size

        # Multipliers
        self.gate_multiplier, self.down_multiplier = getattr(
            config, "mlp_multipliers", (1.0, 1.0))

        # gate_up fused projection: produces [gate, up]
        self.gate_up_proj = Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=bias,
            dtype=self.dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
            quant_config=model_config.get_quant_config(),
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
        )

        # down projection
        self.down_proj = Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            dtype=self.dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=model_config.get_quant_config(),
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            allreduce_strategy=model_config.allreduce_strategy,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate_up outputs [gate, up] concatenated along last dim (local dims under TP)
        x = self.gate_up_proj(x)
        local_intermediate = self.intermediate_size // self.tp_size
        gate = x[..., :local_intermediate]
        up = x[..., local_intermediate:]
        # Apply gate multiplier to gate branch, SiLU then multiply
        gate = F.silu(gate * self.gate_multiplier)
        x = gate * up
        x = self.down_proj(x)
        x = x * self.down_multiplier
        return x


class FalconH1SSMDecoderLayer(nn.Module):

    def __init__(self, model_config: ModelConfig[FalconH1Config], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config

        self.config = config
        self.tp_size = model_config.mapping.tp_size

        self.mamba = Mamba2Mixer(d_model=config.hidden_size,
                                 d_state=config.mamba_d_state,
                                 d_conv=config.mamba_d_conv,
                                 nheads=config.mamba_n_heads,
                                 n_groups=config.mamba_n_groups,
                                 head_dim=config.mamba_d_head,
                                 chunk_size=getattr(config, "mamba_chunk_size",
                                                    getattr(config, "chunk_size", 128)),
                                 layer_idx=layer_idx,
                                 rms_norm_eps=config.rms_norm_eps,
                                 dtype=config.torch_dtype,
                                 config=model_config)

        # Prepare non-learnable per-block scaling vector (mup_vector)
        # following vLLM's FalconH1SSMDecoderLayer implementation.
        ssm_multipliers = getattr(config, "ssm_multipliers", [1.0, 1.0, 1.0, 1.0, 1.0])
        d_ssm = (int(getattr(config, "mamba_expand", 0) * config.hidden_size)
                 if getattr(config, "mamba_d_ssm", None) is None else getattr(config, "mamba_d_ssm"))
        groups_time_state_size = config.mamba_n_groups * config.mamba_d_state
        vector_shape = (2 * d_ssm + 2 * groups_time_state_size + config.mamba_n_heads) // self.tp_size
        mup_vector = torch.ones(1, vector_shape)

        # Z: [0 : d_ssm]
        mup_vector[:, : d_ssm // self.tp_size] *= ssm_multipliers[0]
        # X: [d_ssm : 2*d_ssm]
        mup_vector[:, (d_ssm // self.tp_size):(2 * d_ssm // self.tp_size)] *= ssm_multipliers[1]
        # B: [2*d_ssm : 2*d_ssm + G*S]
        start = (2 * d_ssm) // self.tp_size
        end = (2 * d_ssm + groups_time_state_size) // self.tp_size
        mup_vector[:, start:end] *= ssm_multipliers[2]
        # C: [2*d_ssm + G*S : 2*d_ssm + 2*G*S]
        start = (2 * d_ssm + groups_time_state_size) // self.tp_size
        end = (2 * d_ssm + 2 * groups_time_state_size) // self.tp_size
        mup_vector[:, start:end] *= ssm_multipliers[3]
        # dt: [2*d_ssm + 2*G*S : end]
        start = (2 * d_ssm + 2 * groups_time_state_size) // self.tp_size
        mup_vector[:, start:] *= ssm_multipliers[4]

        self.register_buffer("mup_vector", mup_vector, persistent=False)

    def forward(self,
                hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata,
                mamba_metadata: Mamba2Metadata,
                residual: Optional[torch.Tensor] = None,
                **kwargs):
        output = self.mamba(hidden_states,
                            attn_metadata,
                            mamba_metadata,
                            mup_vector=self.mup_vector)
        return output, residual


class FalconH1AttentionDecoderLayer(nn.Module):

    def __init__(self, model_config: ModelConfig[FalconH1Config], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config

        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        key_multiplier = getattr(config, "key_multiplier", 1.0)
        # Map key scaling (multiplying K) to q_scaling in TRT attention
        # qk_scale = 1 / (sqrt(head_dim) * q_scaling)
        # Multiplying K by key_multiplier is equivalent to setting q_scaling = 1 / key_multiplier
        q_scaling = 1.0 / key_multiplier if key_multiplier not in (0.0, None) else 1.0

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if hasattr(config, "partial_rotary_factor"):
            rotary_dim = head_dim * config.partial_rotary_factor
        elif hasattr(config, "attn_rotary_emb"):
            rotary_dim = config.attn_rotary_emb
        else:
            rotary_dim = head_dim
        rope = RopeParams.from_config(config)
        rope.theta = getattr(config, "rope_theta", 1e11)
        rope.dim = rotary_dim
        rope.max_positions = max_position_embeddings

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope,
            is_neox=True,
        )

        self.attn = Attention(hidden_size=config.hidden_size,
                              num_attention_heads=config.num_attention_heads,
                              num_key_value_heads=config.num_key_value_heads,
                              max_position_embeddings=max_position_embeddings,
                              bias=False,
                              pos_embd_params=pos_embd_params,
                              layer_idx=layer_idx,
                              dtype=config.torch_dtype,
                              config=model_config,
                              q_scaling=q_scaling)

    def forward(self,
                position_ids: Optional[torch.IntTensor],
                hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata,
                residual: Optional[torch.Tensor] = None,
                **kwargs):
        output = self.attn(position_ids, hidden_states, attn_metadata)
        return output, residual


class FalconH1ParallelHybridLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[FalconH1Config], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config

        self.layer_idx = layer_idx

        # Branch modules
        self.self_attn = FalconH1AttentionDecoderLayer(model_config,
                                                       layer_idx)

        self.mamba = FalconH1SSMDecoderLayer(model_config, layer_idx)

        # FFN uses fused gate_up and down with SiLU gating (FalconH1-specific)
        intermediate_size = config.intermediate_size
        self.feed_forward = FalconH1MLP(model_config,
                                        intermediate_size=intermediate_size,
                                        bias=False)

        # Norms
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.pre_ff_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                        eps=config.rms_norm_eps,
                                        dtype=config.torch_dtype)

        # Multipliers (default to 1.0 if missing)
        self.ssm_in_multiplier = getattr(config, "ssm_in_multiplier", 1.0)
        self.ssm_out_multiplier = getattr(config, "ssm_out_multiplier", 1.0)
        self.attn_in_multiplier = getattr(config, "attention_in_multiplier", 1.0)
        self.attn_out_multiplier = getattr(config, "attention_out_multiplier", 1.0)

    def forward(self,
                position_ids: torch.IntTensor,
                hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata,
                **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Attention branch
        attn_hidden, _ = self.self_attn(position_ids=position_ids,
                                        hidden_states=hidden_states * self.attn_in_multiplier,
                                        attn_metadata=attn_metadata)

        # Mamba branch
        mamba_hidden, _ = self.mamba(hidden_states=hidden_states * self.ssm_in_multiplier,
                                     attn_metadata=attn_metadata,
                                     mamba_metadata=kwargs.get(
                                         "mamba_metadata"))

        hidden_states = attn_hidden * self.attn_out_multiplier + mamba_hidden * self.ssm_out_multiplier
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
        # TODO: vllm differentiates between is_first_rank and not, do we need that?
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
        # TODO: vllm differentiates between is_last_rank and not, do we need that?
        self.final_layernorm = RMSNorm(hidden_size=config.hidden_size,
                              eps=config.rms_norm_eps,
                              dtype=config.torch_dtype)

        self.mamba_metadata: Optional[Mamba2Metadata] = None

        self.embedding_multiplier = getattr(config, "embedding_multiplier", 1.0)

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
            self.mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests,
                chunk_size=self.model_config.pretrained_config.mamba_chunk_size)
        self.mamba_metadata.prepare(attn_metadata)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds * self.embedding_multiplier

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
                if '.mamba' in nk:
                    nk = nk.replace('.mamba', '.mamba.mamba')
                if '.self_attn.' in nk:
                    nk = nk.replace('.self_attn.', '.self_attn.attn.')
                # Flatten conv1d weights [out, 1, k] -> [out, k]
                if nk.endswith('.mamba.conv1d.weight') and v.ndim == 3 and v.shape[1] == 1:
                    v = v.squeeze(1).contiguous()
                if '.mamba.A' in nk:
                    v = -torch.exp(v.to(torch.float32))
                norm[nk] = v
            return norm

        weights = _normalize_keys(weights)

        new_weights = weight_mapper.preprocess_weights(weights)
        if new_weights is None:
            new_weights = weights
        else:
            new_weights = _normalize_keys(new_weights)
        super().load_weights(new_weights, weight_mapper)


