# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from enum import IntEnum
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor


class RotaryScalingType(IntEnum):
    none = 0
    linear = 1
    dynamic = 2
    longrope = 3
    llama3 = 4
    yarn = 5
    mrope = 6

    @staticmethod
    def from_string(s):
        if isinstance(s, RotaryScalingType):
            return s
        if s is None:
            return RotaryScalingType.none
        key = str(s).lower()
        # Hugging Face Transformers v5+ uses type "default" for unscaled / standard RoPE.
        if key == "default":
            return RotaryScalingType.none
        try:
            return RotaryScalingType[key]
        except KeyError:
            raise ValueError(f"Unsupported rotary scaling type: {s}")


class PositionEmbeddingType(IntEnum):
    learned_absolute = 0
    rope_gptj = 1
    rope_gpt_neox = 2
    long_rope = 3
    alibi = 4
    alibi_with_scale = 5
    relative = 6
    chatglm = 7
    yarn = 8
    mrope = 9
    # Apply customized positional embedding by using an external embedder.
    # K will be cached before embedding.
    deferred = 10

    def is_rope(self) -> bool:
        return self in [
            self.rope_gptj, self.rope_gpt_neox, self.long_rope, self.mrope
        ]

    def is_mrope(self) -> bool:
        return self in [self.mrope]

    def is_alibi(self) -> bool:
        return self in [self.alibi, self.alibi_with_scale]

    def is_deferred(self) -> bool:
        return self in [self.deferred]

    @staticmethod
    def choices() -> List[str]:
        return [embedding.name for embedding in PositionEmbeddingType]

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        # Transformers 5.x uses "default" for standard RoPE (no scaling).
        if s == "default":
            return PositionEmbeddingType.rope_gpt_neox
        try:
            return PositionEmbeddingType[s]
        except KeyError:
            raise ValueError(f"Unsupported position embedding type: {s}")


class AttentionMaskType(IntEnum):
    padding = 0
    causal = 1
    sliding_window_causal = 2
    bidirectional = 3
    bidirectionalglm = 4  # TODO: merge this mask into bidirectional
    blocksparse = 5
    custom_mask = 6


class AllReduceStrategy(IntEnum):
    NCCL = 0
    MIN_LATENCY = 1
    UB = 2
    AUTO = 3
    ONESHOT = 4
    TWOSHOT = 5
    LOWPRECISION = 6
    MNNVL = 7
    NCCL_SYMMETRIC = 8
    SYMM_MEM = 9  # PyTorch symmetric memory with MULTIMEM


class AllReduceFusionOp(IntEnum):
    NONE = 0
    RESIDUAL_RMS_NORM = 1
    LAST_PROCESS_FOR_UB = 2
    RESIDUAL_RMS_PREPOST_NORM = 3
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5
    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6
    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7
    MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM = 8
    RMS_NORM = 9


class AllReduceParams:

    def __init__(
        self,
        strategy: AllReduceStrategy = AllReduceStrategy.AUTO,
        fusion_op: AllReduceFusionOp = AllReduceFusionOp.NONE,
        bias: Optional[Tensor] = None,
        residual: Optional[Tensor] = None,
        norm_weight: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        norm_pre_residual_weight: Optional[Tensor] = None,
        eps: float = 1e-06,
        enable_allreduce: bool = True,
        trigger_completion_at_end: bool = True,
    ):
        self.strategy = strategy
        self.fusion_op = fusion_op
        self.bias = bias
        self.residual = residual
        self.norm_weight = norm_weight
        self.scale = scale
        self.norm_pre_residual_weight = norm_pre_residual_weight
        self.eps = eps
        # For torch path only, has no effect on TRT path
        self.enable_allreduce = enable_allreduce
        self.trigger_completion_at_end = trigger_completion_at_end
        assert fusion_op in (AllReduceFusionOp.NONE.value,
                             AllReduceFusionOp.RMS_NORM.value) or (residual
                                                                   is not None)

    def has_affine(self):
        return 1 if self.norm_weight is not None else 0

    def has_bias(self):
        return 1 if self.bias is not None else 0

    def has_scale(self):
        return 1 if self.scale is not None else 0


class MoEAllReduceParams(AllReduceParams):

    def __init__(
        self,
        device_num_experts: Optional[Tensor] = None,
        expert_scale_factor: Optional[Tensor] = None,
        expanded_idx_to_permuted_idx: Optional[Tensor] = None,
        shared_expert_output: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        residual: Optional[Tensor] = None,
        norm_weight: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        norm_pre_residual_weight: Optional[Tensor] = None,
        eps: float = 1e-06,
        enable_allreduce: bool = True,
        is_cutlass_min_latency: bool = False,
    ):
        super().__init__(
            bias=bias,
            residual=residual,
            norm_weight=norm_weight,
            scale=scale,
            norm_pre_residual_weight=norm_pre_residual_weight,
            eps=eps,
            enable_allreduce=enable_allreduce,
        )
        self.device_num_experts = device_num_experts
        self.expert_scale_factor = expert_scale_factor
        self.expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx
        self.shared_expert_output = shared_expert_output
        self.is_cutlass_min_latency = is_cutlass_min_latency

    def is_valid(self):
        if self.is_cutlass_min_latency:
            return (self.device_num_experts is not None
                    and self.expert_scale_factor is not None
                    and self.shared_expert_output is not None)
        else:
            return self.expanded_idx_to_permuted_idx is not None


class RopeEmbeddingUtils:

    @staticmethod
    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L298
    def apply_llama3_scaling(inv_freqs: np.ndarray, rope_scaling_config: dict):
        scale_factor = rope_scaling_config.get("factor", 8.0)
        low_freq_factor = rope_scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = rope_scaling_config.get("high_freq_factor", 4.0)
        old_context_len = rope_scaling_config.get(
            "original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_inv_freqs = []
        for inv_freq in inv_freqs:
            wavelen = 2 * math.pi / inv_freq
            if wavelen < high_freq_wavelen:
                new_inv_freqs.append(inv_freq)
            elif wavelen > low_freq_wavelen:
                new_inv_freqs.append(inv_freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor)
                new_inv_freqs.append((1 - smooth) * inv_freq / scale_factor +
                                     smooth * inv_freq)
        return np.array(new_inv_freqs, dtype=inv_freqs.dtype)

    @staticmethod
    def create_sinusoidal_positions(num_pos: int,
                                    dim: int,
                                    theta: float = 10000.0,
                                    dtype=np.float32):
        inv_freq = 1.0 / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
        sinusoid_inp = np.einsum("i , j -> i j",
                                 np.arange(num_pos, dtype=dtype),
                                 inv_freq,
                                 dtype=dtype)
        concat = np.concatenate((np.sin(sinusoid_inp), np.cos(sinusoid_inp)),
                                axis=1)
        return np.expand_dims(concat, axis=0).astype(dtype)

    @staticmethod
    def create_sinusoidal_positions_for_attention_plugin(
        num_pos: int,
        dim: int,
        theta: float = 10000.0,
        scale: float = 1.0,
        scale_type: RotaryScalingType = RotaryScalingType.none,
        # Other scaling configs that only used by certain scaling types.
        rope_scaling_config: dict = None,
        duplicate_data: bool = False,
        dtype=np.float32,
    ):
        if scale_type == RotaryScalingType.linear:
            scale = 1.0 / scale
        if scale_type == RotaryScalingType.llama3:
            assert rope_scaling_config is not None, "rotary_scaling config must be provided."
            inv_freq = 1.0 / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
            inv_freq = RopeEmbeddingUtils.apply_llama3_scaling(
                inv_freq, rope_scaling_config)
        elif scale_type == RotaryScalingType.dynamic:
            # Make sure scaling_alpha exists in rope_scaling
            # Ref: https://huggingface.co/tencent/Hunyuan-A13B-Instruct-FP8/blob/main/modeling_hunyuan.py#L346
            assert rope_scaling_config["alpha"] is not None, (
                "rope_scaling_config.alpha must be provided.")
            scaling_alpha = rope_scaling_config["alpha"]
            adjusted_base = theta * (scaling_alpha**(dim / (dim - 2)))
            inv_freq = 1.0 / (adjusted_base**(
                np.arange(0, dim, 2, dtype=dtype) / dim)).astype(dtype)
        else:
            inv_freq = scale / (theta
                                **(np.arange(0, dim, 2) / dim)).astype(dtype)
        sinusoid_inp = np.expand_dims(
            np.einsum("i , j -> i j",
                      np.arange(num_pos, dtype=dtype),
                      inv_freq,
                      dtype=dtype),
            axis=-1,
        )
        if duplicate_data:
            sinusoid_inp = np.concatenate((sinusoid_inp, sinusoid_inp), axis=-2)
        # fuse cos/sin into float2 (cos, sin).
        concat = np.concatenate(
            (np.cos(sinusoid_inp), np.sin(sinusoid_inp)),
            axis=-1)  # np.cos(sinusoid_inp).shape = (32768, 64, 1)

        return inv_freq, concat.reshape(1, -1).astype(dtype)

    @staticmethod
    def create_sinusoidal_positions_for_cogvlm_attention_plugin(
        num_pos: int,
        dim: int,
        theta: float = 10000.0,
        scale: float = 1.0,
        scale_type: RotaryScalingType = RotaryScalingType.none,
        vision_start: int = 1,
        vision_length: int = 1225,
        dtype=np.float32,
    ):
        if scale_type == RotaryScalingType.linear:
            scale = 1.0 / scale
        inv_freq = scale / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
        position_id = np.hstack([
            np.arange(0, vision_start + 1, dtype=dtype),
            np.full(vision_length, vision_start + 1, dtype=dtype),
            np.arange(vision_start + 2,
                      num_pos - (vision_length - 1),
                      dtype=dtype),
        ])
        sinusoid_inp = np.expand_dims(np.einsum("i , j -> i j",
                                                position_id,
                                                inv_freq,
                                                dtype=dtype),
                                      axis=-1)
        # fuse cos/sin into float2 (cos, sin).
        concat = np.concatenate((np.cos(sinusoid_inp), np.sin(sinusoid_inp)),
                                axis=-1)

        return inv_freq, concat.reshape(1, -1).astype(dtype)

    def create_sinusoidal_positions_long_rope_for_attention_plugin(
        num_pos: int,
        num_orig_pos: int,
        dim: int,
        theta: float = 10000.0,
        scaling_short_factors: Tensor = 1.0,
        scaling_long_factors: Tensor = 1.0,
        short_mscale=None,
        long_mscale=None,
        dtype=np.float32,
    ):

        def _calc_mscale(scale):
            if scale <= 1.0:
                return 1.0
            return math.sqrt(1 + math.log(scale) / math.log(num_orig_pos))

        if short_mscale is None:
            short_mscale = _calc_mscale(num_pos / num_orig_pos)
            long_mscale = short_mscale

        def _compute_sinusoidal_positions(scale_factors, is_short,
                                          for_attention_plugin):
            inv_freq = 1 / (scale_factors *
                            (theta**(np.arange(0, dim, 2) / dim)).astype(dtype))
            sinusoid_inp = np.einsum("i , j -> i j",
                                     np.arange(num_pos, dtype=dtype),
                                     inv_freq,
                                     dtype=dtype)

            if for_attention_plugin:
                sinusoid_inp = np.expand_dims(sinusoid_inp, axis=-1)
                concat = np.concatenate(
                    (np.cos(sinusoid_inp), np.sin(sinusoid_inp)), axis=-1)
            else:
                concat = np.concatenate(
                    (np.sin(sinusoid_inp), np.cos(sinusoid_inp)), axis=1)
                concat = np.expand_dims(concat, axis=0)

            mscale = short_mscale if is_short else long_mscale
            concat = concat.astype(dtype) * mscale

            # gpt attention plugins also need inv_freq.
            if for_attention_plugin:
                return inv_freq.reshape(1, -1), concat.reshape(1, -1)
            else:
                return concat

        return (
            _compute_sinusoidal_positions(scaling_short_factors, True, False),
            _compute_sinusoidal_positions(scaling_long_factors, False, False),
            _compute_sinusoidal_positions(scaling_short_factors, True, True),
            _compute_sinusoidal_positions(scaling_long_factors, False, True),
            short_mscale,
        )

    @staticmethod
    def create_sinusoidal_positions_long_rope(
        num_pos: int,
        dim: int,
        theta: float,
        original_max_pos: int,
        short_factor: List[float],
        long_factor: List[float],
        dtype=np.float32,
        max_seq_len: Optional[int] = None,
        duplicate_data: bool = False,
    ):
        short_factor = np.array(short_factor, dtype=np.float32)
        long_factor = np.array(long_factor, dtype=np.float32)

        inv_freq = 1.0 / (theta**(np.arange(0, dim, 2, dtype=np.float32) / dim))
        t_pos = np.arange(np.max([num_pos, original_max_pos]), dtype=np.float32)

        # Choose proper freqs based on max_seq_len.
        factor = (long_factor if max_seq_len is None
                  or max_seq_len > original_max_pos else short_factor)
        inv_freq = inv_freq / factor
        freqs = np.einsum("i,j->ij", t_pos, inv_freq)
        sinusoid_inp = freqs.astype(np.float32)[..., np.newaxis]

        # Apply scaling
        scale = num_pos / original_max_pos
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = np.sqrt(1.0 +
                                     np.log(scale) / np.log(original_max_pos))

        if duplicate_data:
            sinusoid_inp = np.concatenate((sinusoid_inp, sinusoid_inp), axis=-2)

        # fuse cos/sin into float2 (cos, sin).
        concat = np.concatenate(
            (np.cos(sinusoid_inp) * scaling_factor,
             np.sin(sinusoid_inp) * scaling_factor),
            axis=-1,
        )

        return None, concat.reshape(1, -1).astype(dtype)

    @staticmethod
    def create_fake_weight(dim: int, dtype=np.half):
        return np.random.rand(dim).astype(dtype)

    # Note: When not using deepseek_yarn, make sure to set mscale_all_dim to 0.0.
    @staticmethod
    def create_sinusoidal_positions_yarn(
        num_pos: int,
        dim: int,
        base: int = 10000,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 1.0,
        duplicate_data: bool = True,
        dtype=None,
    ):
        if dtype is None:
            dtype = torch.float32

        # Copy from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py
        # Inverse dim formula to find dim based on number of rotations
        def yarn_find_correction_dim(num_rotations, dim, base,
                                     max_position_embeddings):
            return (dim * math.log(max_position_embeddings /
                                   (num_rotations * 2 * math.pi))) / (
                                       2 * math.log(base))

        # Find dim range bounds based on rotations
        def yarn_find_correction_range(low_rot, high_rot, dim, base,
                                       max_position_embeddings):
            low = math.floor(
                yarn_find_correction_dim(low_rot, dim, base,
                                         max_position_embeddings))
            high = math.ceil(
                yarn_find_correction_dim(high_rot, dim, base,
                                         max_position_embeddings))
            if low < 0:
                low = 0
            if high > dim - 1:
                high = dim - 1
            return low, high  # Clamp values just in case

        def yarn_get_mscale(scale, mscale):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min, max, dim):
            if min == max:
                max += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype=dtype) - min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        pos_freqs = base**(torch.arange(0, dim, 2, dtype=dtype) / dim)
        freq_extra = 1.0 / pos_freqs
        freq_inter = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        )
        inv_freq_mask = 1 - yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        t = torch.arange(num_pos, dtype=dtype)
        sinusoid_inp = torch.einsum("i,j -> ij", t, inv_freq).unsqueeze(-1)

        _mscale = float(
            yarn_get_mscale(scaling_factor, mscale) /
            yarn_get_mscale(scaling_factor, mscale_all_dim))

        if duplicate_data:
            emb = torch.cat((sinusoid_inp, sinusoid_inp), dim=-2)
        else:
            emb = sinusoid_inp

        concat = torch.cat((torch.cos(emb) * _mscale, torch.sin(emb) * _mscale),
                           dim=-1)
        return inv_freq.numpy(), concat.reshape((1, -1)).to(dtype).numpy()
