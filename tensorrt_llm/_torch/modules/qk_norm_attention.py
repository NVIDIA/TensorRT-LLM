# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
from transformers import PretrainedConfig

from ..attention_backend.interface import PositionalEmbeddingParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm


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

    # If config does not contain rope_scaling or rope_type is not yarn, it means the model is not using yarn
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None or getattr(rope_scaling, "rope_type",
                                       None) != "yarn":
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


class QKNormRoPEAttention(Attention):
    """
    QKNormRoPEAttention is a custom attention layer that applies QK norm and RoPE to the input tensor.
    It is used in the ExaOne4, Gemma3 and Qwen3 models.
    It is a subclass of Attention, and overrides the apply_rope method to apply QK norm and RoPE.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        bias: bool,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        skip_rope: bool = False,
        fuse_qk_norm_rope: bool = True,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: ModelConfig,
        q_scaling: float = 1.0,
        disable_deep_gemm: bool = False,
        use_gemma_rms_norm: bool = False,
        attn_output_gate: Optional[bool] = None,
    ):
        self.pretrained_config = config.pretrained_config

        self.fuse_qk_norm_rope = fuse_qk_norm_rope
        self.skip_rope = skip_rope
        if use_gemma_rms_norm:
            assert fuse_qk_norm_rope is False, "fused_qk_norm_rope is not supported for gemma rms norm."

        # If fuse_qk_norm_rope is true, do not apply fused RoPE in attention OP, and self.rotary_emb
        # will be skipped in the overridden apply_rope.
        rope_fusion = not self.fuse_qk_norm_rope and not skip_rope and not attn_output_gate and not use_gemma_rms_norm
        assert not (fuse_qk_norm_rope and skip_rope
                    ), "Fusing qk norm and skipping rope is not supported"

        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            bias=bias,
            pos_embd_params=pos_embd_params,
            rope_fusion=rope_fusion,
            layer_idx=layer_idx,
            dtype=dtype,
            dense_bias=dense_bias,
            config=config,
            q_scaling=q_scaling,
            disable_deep_gemm=disable_deep_gemm,
            attn_output_gate=attn_output_gate,
        )

        self.q_norm = RMSNorm(hidden_size=self.head_dim,
                              eps=self.pretrained_config.rms_norm_eps,
                              dtype=self.pretrained_config.torch_dtype,
                              has_weights=True,
                              use_gemma=use_gemma_rms_norm)
        self.k_norm = RMSNorm(hidden_size=self.head_dim,
                              eps=self.pretrained_config.rms_norm_eps,
                              dtype=self.pretrained_config.torch_dtype,
                              has_weights=True,
                              use_gemma=use_gemma_rms_norm)
        self.aux_stream = torch.cuda.Stream()
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

    def apply_qk_norm(self, q, k):

        def q_l2norm():
            return self.q_norm(q.reshape(-1, self.head_dim)).reshape(
                -1, self.q_size)

        def k_l2norm():
            return self.k_norm(k.reshape(-1, self.head_dim)).reshape(
                -1, self.kv_size)

        q, k = maybe_execute_in_parallel(
            q_l2norm,
            k_l2norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        return q, k

    def apply_qk_norm_rope(self, qkv, position_ids):
        factor, low, high, attention_factor = compute_yarn_parameters(
            self.pretrained_config)
        torch.ops.trtllm.fused_qk_norm_rope(
            qkv, self.num_heads, self.num_key_value_heads,
            self.num_key_value_heads, self.head_dim,
            self.q_norm.variance_epsilon, self.q_norm.weight,
            self.k_norm.weight,
            self.pos_embd_params.rope.theta, self.pos_embd_params.is_neox,
            position_ids.view(-1), factor, low, high, attention_factor)
        return qkv, None, None

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        """
        The apply_rope method is called in the forward method of the Attention class.
        The apply_rope method is overridden in this class to apply QK norm and RoPE to the input tensor.
        """
        # Apply QK norm before RoPE.
        if not self.fuse_qk_norm_rope:
            q, k, v = self.split_qkv(q, k, v)
            q, k = self.apply_qk_norm(q, k)
            if not self.skip_rope:
                return super().apply_rope(q, k, v, position_ids)
            else:
                return q, k, v

        qkv = q
        if k is not None and v is not None:
            qkv = torch.concat([q, k, v], dim=-1)
        return self.apply_qk_norm_rope(qkv, position_ids)
