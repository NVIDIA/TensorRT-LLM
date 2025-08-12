import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig, Qwen3Config

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.qk_norm_attention import QKNormRoPEAttention
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model


# Move out from this class
def compute_yarn_parameters(
    config: PretrainedConfig, ) -> tuple[float, float, float, float]:
    """
    Refer to https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L197C1-L288C1
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://huggingface.co/papers/2309.00071)
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
    Returns:
        factor: float, the scaling factor for the RoPE embeddings
        low: float, the lower bound of the dimension range
        high: float, the upper bound of the dimension range
        attention_factor: float, the post-processing scaling factor applied to the computed cos/sin
    """

    # The config does not contain rope_scaling, which means the model is not using yarn
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return 1.0, 0, 0, 1.0

    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(
        config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    factor = getattr(rope_scaling, "factor", 1.0)
    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")

    if "original_max_position_embeddings" in rope_scaling:
        original_max_position_embeddings = rope_scaling[
            "original_max_position_embeddings"]
        factor = config.max_position_embeddings / original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim *
                math.log(max_position_embeddings /
                         (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base,
                              max_position_embeddings, truncate):
        """Find dimension range bounds based on rotations"""
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    truncate = rope_scaling.get("truncate", True)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                      original_max_position_embeddings,
                                      truncate)

    # These parts are implemented in the fusedQKNormRopeKernel.cu
    # # def linear_ramp_factor(min, max, dim):
    # #     if min == max:
    # #         max += 0.001  # Prevent singularity

    # #     linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    # #     ramp_func = torch.clamp(linear_func, 0, 1)
    # #     return ramp_func

    # # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # # to expand the possible context length. In other words, interpolation = apply scaling factor.
    # # pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    # # inv_freq_extrapolation = 1.0 / pos_freqs
    # # inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    # # # Get n-dimensional rotational scaling corrected for extrapolation
    # # inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    # # inv_freq = (
    # #     inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
    # #     + inv_freq_extrapolation * inv_freq_extrapolation_factor
    # # )
    # # return inv_freq, attention_factor
    return factor, low, high, attention_factor


class Qwen3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        self.self_attn = QKNormRoPEAttention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias if hasattr(config, "mlp_bias") else False,
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        mrope_config: Optional[Tuple[torch.Tensor, int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            mrope_config=mrope_config,
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)

        return hidden_states, residual


class Qwen3Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[Qwen3Config]):
        super().__init__(model_config)
        config = self.model_config

        self.embed_tokens = Embedding(
            config.pretrained_config.vocab_size,
            config.pretrained_config.hidden_size,
            dtype=config.pretrained_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            hidden_size=config.pretrained_config.hidden_size,
            eps=config.pretrained_config.rms_norm_eps,
            dtype=config.pretrained_config.torch_dtype,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mrope_config: Optional[Tuple[torch.Tensor, int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                mrope_config=mrope_config,
                spec_metadata=spec_metadata,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("Qwen3ForCausalLM")
class Qwen3ForCausalLM(SpecDecOneEngineForCausalLM[Qwen3Model, Qwen3Config]):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3Config],
    ):
        super().__init__(
            Qwen3Model(model_config),
            model_config,
        )
