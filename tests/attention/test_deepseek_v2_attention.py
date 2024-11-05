# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import sys
import unittest
from itertools import product
from typing import Optional, Tuple

import numpy as np
import pytest
import torch
from parameterized import parameterized
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import (AttentionMaskConverter,
                                                   _prepare_4d_attention_mask)

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.functional import (PositionEmbeddingType, RopeEmbeddingUtils,
                                     RotaryScalingType)
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import GenerationSequence
from tensorrt_llm.runtime.memory_pools.pools_kv_cache_manager import \
    PoolsKVCacheManager

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import (getSMVersion, skip_bf16_fp32_accum,
                        skip_bf16_pre_ampere, skip_fp8_pre_ada,
                        skip_fp32_accum_pre_ampere, unittest_name_func)

from tensorrt_llm.runtime.memory_pools.memory_pools_allocator import \
    MemoryPoolsAllocator


class DeepseekV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV2Model`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeepSeek-V2.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DeepseekV2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts, None means dense model.
        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts, None means dense model.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor or routed experts.
        topk_method (`str`, *optional*, defaults to `gready`):
            Topk method used in routed gate.
        n_group (`int`, *optional*, defaults to None):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to None):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of selected experts, None means dense model.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the weights of the routed experts.
        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Method of computing expert weights.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss weight coefficient.
        seq_aux = (`bool`, *optional*, defaults to True):
            Whether to compute the auxiliary loss for each individual sample.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    ```python
    >>> from transformers import DeepseekV2Model, DeepseekV2Config
    >>> # Initializing a Deepseek-V2 style configuration
    >>> configuration = DeepseekV2Config()
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=5120,
        intermediate_size=12288,
        moe_intermediate_size=1407,
        num_hidden_layers=60,
        num_attention_heads=128,
        num_key_value_heads=128,
        n_shared_experts=None,
        n_routed_experts=None,
        ep_size=1,
        routed_scaling_factor=16.0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        topk_method='group_limited_greedy',
        n_group=8,
        topk_group=3,
        num_experts_per_tok=6,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        norm_topk_prob=False,
        scoring_func='softmax',
        aux_loss_alpha=0.001,
        seq_aux=True,
        hidden_act="silu",
        max_position_embeddings=163840,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
            "type": "yarn"
        },
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class DeepseekV2RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekV2RotaryEmbedding(nn.Module):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base**(
            torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin().to(dtype),
                             persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len,
                                    device=x.device,
                                    dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2LinearScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin().to(dtype),
                             persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->DeepseekV2
class DeepseekV2DynamicNTKScalingRotaryEmbedding(DeepseekV2RotaryEmbedding):
    """DeepseekV2RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) -
                (self.scaling_factor - 1))**(self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base**(
                torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin().to(dtype),
                             persistent=False)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations,
                             dim,
                             base=10000,
                             max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot,
                               high_rot,
                               dim,
                               base=10000,
                               max_position_embeddings=2048):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV2YarnRotaryEmbedding(DeepseekV2RotaryEmbedding):

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base**(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (self.scaling_factor * self.base**(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale) /
            yarn_get_mscale(self.scaling_factor, self.mscale_all_dim))

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype),
                             persistent=False)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepseekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 config: DeepseekV2Config,
                 layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class.")

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        self.q_a_proj = nn.Linear(self.hidden_size,
                                  config.q_lora_rank,
                                  bias=config.attention_bias)
        self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank,
                                  self.num_heads * self.q_head_dim,
                                  bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads *
            (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self._init_rope()

        self.kv_cache_buffer = torch.zeros([256, 576],
                                           dtype=torch.bfloat16,
                                           device="cuda")
        self.acc_len = 0
        self.softmax_scale = self.q_head_dim**(-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV2RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV2LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV2DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ] if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV2YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (tensor.view(bsz, seq_len, self.num_heads,
                            self.v_head_dim).transpose(1, 2).contiguous())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        compressed_q = self.q_a_proj(hidden_states)

        q = self.q_b_proj(self.q_a_layernorm(compressed_q))
        tmp_q = q.view(bsz * q_len, self.num_heads, self.q_head_dim)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        tmp_out = torch.concat([
            self.q_a_layernorm(compressed_q),
            self.kv_a_layernorm(compressed_kv), k_pe
        ],
                               dim=-1)

        kv_nope = self.kv_a_layernorm(compressed_kv)
        tmp = self.kv_b_proj.weight.reshape(128, 256, 512)
        k_w, v_w = torch.split(tmp, [128, 128], dim=1)
        tmp_q_nope, tmp_q_pe = torch.split(
            tmp_q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        fused_input = torch.bmm(tmp_q_nope.transpose(0, 1), k_w).transpose(0, 1)
        final_fused_q = torch.concat([fused_input, tmp_q_pe], dim=-1)
        # if (q_len == 1):
        #     print("generation fused q")
        #     print(final_fused_q[0,:,:])

        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz, q_len, self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2))

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index.")
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
        tmp_kv_cache = torch.concat([
            kv_nope,
            k_pe.transpose(1, 2).reshape(bsz, q_len, self.qk_rope_head_dim)
        ],
                                    dim=-1)
        self.kv_cache_buffer[self.acc_len:self.acc_len +
                             q_len, :] = tmp_kv_cache[0]
        self.acc_len += q_len

        # print("kv cache:")
        # for i in range(self.acc_len):
        #     print(i)
        #     print(self.kv_cache_buffer[i])

        if q_len == 1:
            final_fused_q = torch.concat([
                fused_input,
                q_pe.transpose(2, 1).reshape(bsz, self.num_heads,
                                             self.qk_rope_head_dim)
            ],
                                         dim=-1)
            print("generation fused q")
            print(final_fused_q[0, :, :])
            q_input = final_fused_q[0, :, :].reshape(
                self.num_heads, self.kv_lora_rank + self.qk_rope_head_dim)
            kv = self.kv_cache_buffer[:self.acc_len, :]
            qk_weight = torch.matmul(q_input, kv.transpose(
                0, 1)) * self.softmax_scale
            print(qk_weight.shape)
            qk_weight = nn.functional.softmax(qk_weight,
                                              dim=-1,
                                              dtype=torch.float32).to(
                                                  q_input.dtype)
            qkv_output = torch.matmul(qk_weight, kv[:, :self.kv_lora_rank])
            print(qkv_output)
            print(qkv_output.shape)
            print(v_w.shape)
            gpt_output = torch.bmm(qkv_output.reshape(self.num_heads, 1, -1),
                                   v_w.transpose(2, 1))
            print(gpt_output)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len,
                                      self.q_head_dim)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        # qq = query_states.transpose(2, 1).reshape(bsz, q_len, self.num_heads * self.q_head_dim)
        # kk = key_states.transpose(2, 1).reshape(bsz, q_len, self.num_heads * self.q_head_dim)
        # vv = value_states.transpose(2, 1).reshape(bsz, q_len, self.num_heads * self.qk_nope_head_dim)
        # tmp_out = torch.concat([qq, kk, vv], dim=-1)

        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) *
                        self.softmax_scale)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights,
                                             p=self.attention_dropout,
                                             training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len,
                                          self.num_heads * self.v_head_dim)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, tmp_out


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    def load_test_cases():
        test_cases = []
        # For MLA.
        #
        # Needs to test on both Hopper and non-Hopper, because Hopper may use different kernel.
        test_cases += list(
            product(
                ['Deepseek V2'],
                [90],
                ['deepseek_attention'],
                [
                    ContextFMHAType.enabled,
                ],
                ['bfloat16'],
                [None],
                [1],  # batch_size
                [128],  # in_len
                [128],  # num_q_heads
                [192],  # head_size
                [1],  # num_kv_heads
                [False],  # enable_multi_block_mmha
                [True],
                [1],  # beam_width
                [True],  # paged_kv_cache
                [False],
                [10000.0],  # rope base
                [  # rope scaling
                    None,
                ]))

        return test_cases

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_gpt_attention(self,
                           test_partition,
                           gpu_arch,
                           attention_type,
                           context_fmha_type,
                           dtype,
                           kv_cache_dtype,
                           batch_size,
                           in_len,
                           num_heads,
                           head_size,
                           num_kv_heads,
                           enable_multi_block_mmha,
                           enable_remove_input_padding,
                           beam_width,
                           paged_kv_cache,
                           fuse_bias,
                           rope_base=10000.0,
                           rope_scaling=None,
                           sink_token_len=0):
        # if attention_type != "gpt_bigcode_attention" and attention_type != "llama_attention":
        #     assert num_kv_heads == 0 # safe guard against bad test case configs

        os.environ['TRTLLM_FORCE_XQA'] = '1'
        use_int8_kv_cache = True if kv_cache_dtype == 'int8' else False
        use_fp8_kv_cache = True if kv_cache_dtype == 'fp8' else False
        output_atol = 2e-2 if kv_cache_dtype == 'int8' else 2e-3
        if kv_cache_dtype is None:
            kv_cache_dtype = dtype
        # skip tests based on the gpu_arch_lists
        if gpu_arch != 'all':
            assert gpu_arch in [70, 80, 86, 89, 90]
            if getSMVersion() != gpu_arch:
                pytest.skip(
                    "Skip the test as the target gpu arch doesn't match this gpu arch."
                )

        # Skip tests that are not supported or duplicate
        skip_bf16_pre_ampere(dtype)
        skip_fp32_accum_pre_ampere(context_fmha_type)
        skip_bf16_fp32_accum(dtype, context_fmha_type)
        skip_fp8_pre_ada(use_fp8_kv_cache)

        if num_kv_heads == 0:
            num_kv_heads = num_heads

        session = None
        if use_int8_kv_cache or use_fp8_kv_cache or True:
            # Fixing seed to avoid flakiness in tests with quantization
            torch.manual_seed(42)

        tokens_per_block = 64 if paged_kv_cache else -1
        streamingllm = sink_token_len > 0

        def _construct_execution(
                session, input_tensor, q_a_proj, q_a_layernorm, q_b_proj,
                kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, position_ids,
                past_key_value, host_kv_cache_block_offsets,
                host_kv_cache_pool_pointers, host_kv_cache_pool_mapping,
                sequence_length, host_past_key_value_lengths,
                host_max_attention_window_sizes, host_sink_token_length,
                context_lengths, host_context_lengths, cache_indirection,
                host_request_types, num_heads, hidden_size, num_kv_heads,
                output, dtype, max_context_length, shape_dict, configuration,
                host_runtime_perf_knobs):

            c_q_dim = configuration.q_lora_rank
            c_k_dim = configuration.kv_lora_rank
            rope_dim = configuration.qk_rope_head_dim
            configuration.v_head_dim + rope_dim
            configuration.rms_norm_eps
            kv_cache_block_offsets = None
            if paged_kv_cache:
                kv_cache_block_offsets = host_kv_cache_block_offsets.to('cuda')
            # construct trt network
            builder = tensorrt_llm.Builder()
            net = builder.create_network()
            net.plugin_config.gpt_attention_plugin = dtype
            net.plugin_config.set_context_fmha(context_fmha_type)
            if streamingllm:
                net.plugin_config.streamingllm = True
            if enable_remove_input_padding:
                net.plugin_config.remove_input_padding = True
            else:
                net.plugin_config.remove_input_padding = False
            if paged_kv_cache:
                net.plugin_config.enable_paged_kv_cache(tokens_per_block)
            else:
                net.plugin_config.paged_kv_cache = False
            # always enable xqa kernels for test.
            net.plugin_config.enable_xqa = True

            with tensorrt_llm.net_guard(net):
                x_tensor = Tensor(name='input',
                                  shape=tuple(input_tensor.shape),
                                  dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                sequence_length_tensor = Tensor(
                    name='sequence_length',
                    shape=tuple(sequence_length.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_past_key_value_lengths_tensor = Tensor(
                    name='host_past_key_value_lengths',
                    shape=tuple(host_past_key_value_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_max_attention_window_sizes_tensor = Tensor(
                    name='host_max_attention_window_sizes',
                    shape=tuple(host_max_attention_window_sizes.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_sink_token_length_tensor = Tensor(
                    name='host_sink_token_length',
                    shape=tuple(host_sink_token_length.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                context_lengths_tensor = Tensor(
                    name='context_lengths',
                    shape=tuple(context_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_context_lengths_tensor = Tensor(
                    name='host_context_lengths',
                    shape=tuple(context_lengths.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(
                        'int32')) if enable_remove_input_padding else None
                cache_indirection_tensor = Tensor(
                    name='cache_indirection',
                    shape=tuple(cache_indirection.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_request_types_tensor = Tensor(
                    name='host_request_types',
                    shape=tuple(host_request_types.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                host_runtime_perf_knobs_tensor = Tensor(
                    name='host_runtime_perf_knobs',
                    shape=[16],
                    dtype=tensorrt_llm.str_dtype_to_trt('int64'))
                q_a_proj_tensor = Tensor(
                    name='q_a_proj',
                    shape=tuple(q_a_proj.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                q_a_layernorm_tensor = Tensor(
                    name='q_a_layernorm',
                    shape=tuple(q_a_layernorm.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                q_b_proj_tensor = Tensor(
                    name='q_b_proj',
                    shape=tuple(q_b_proj.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                kv_a_proj_with_mqa_tensor = Tensor(
                    name='kv_a_proj_with_mqa',
                    shape=tuple(kv_a_proj_with_mqa.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                kv_a_layernorm_tensor = Tensor(
                    name='kv_a_layernorm',
                    shape=tuple(kv_a_layernorm.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                kv_b_proj_tensor = Tensor(
                    name='kv_b_proj',
                    shape=tuple(kv_b_proj.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                position_ids_tensor = Tensor(
                    name='position_ids',
                    shape=tuple(position_ids.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt(dtype))

                past_key_value_tensor = None
                kv_cache_block_offsets_tensor = None
                host_kv_cache_block_offsets_tensor = None
                host_kv_cache_pool_pointers_tensor = None
                if paged_kv_cache:
                    kv_cache_block_offsets_tensor = Tensor(
                        name='kv_cache_block_offsets',
                        shape=tuple(kv_cache_block_offsets.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                    host_kv_cache_block_offsets_tensor = Tensor(
                        name='host_kv_cache_block_offsets',
                        shape=tuple(kv_cache_block_offsets.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                    host_kv_cache_pool_pointers_tensor = Tensor(
                        name='host_kv_cache_pool_pointers',
                        shape=(
                            1,
                            1,
                        ),
                        dtype=tensorrt_llm.str_dtype_to_trt('int64'))
                    host_kv_cache_pool_mapping_tensor = Tensor(
                        name='host_kv_cache_pool_mapping',
                        shape=(1, ),
                        dtype=tensorrt_llm.str_dtype_to_trt('int32'))
                else:
                    past_key_value_tensor = Tensor(
                        name='past_key_value',
                        shape=tuple(past_key_value.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt(kv_cache_dtype))

                kv_quant_scale_tensor = None
                kv_dequant_scale_tensor = None
                # if use_int8_kv_cache or use_fp8_kv_cache:
                #     kv_quant_scale_tensor = Tensor(
                #         name='kv_quant_scale',
                #         shape=(1, ),
                #         dtype=tensorrt_llm.str_dtype_to_trt('float32'))
                #     kv_dequant_scale_tensor = Tensor(
                #         name='kv_dequant_scale',
                #         shape=(1, ),
                #         dtype=tensorrt_llm.str_dtype_to_trt('float32'))

                rotary_embedding_dim = 0
                position_embedding_type = PositionEmbeddingType.learned_absolute

                rope_base = 10000.0
                rope_scale_type = RotaryScalingType.none
                rope_scale = 1.0
                rotary_embedding_base = configuration.rope_theta
                scaling_factor = configuration.rope_scaling["factor"]
                max_position_embeddings = configuration.max_position_embeddings
                rotary_embedding_origin_max_position = configuration.rope_scaling[
                    "original_max_position_embeddings"]
                rotary_embedding_beta_fast = configuration.rope_scaling[
                    "beta_fast"]
                rotary_embedding_beta_slow = configuration.rope_scaling[
                    "beta_slow"]
                rotary_embedding_mscale = configuration.rope_scaling["mscale"]
                rotary_embedding_mscale_all_dim = configuration.rope_scaling[
                    "mscale_all_dim"]

                embed_positions_for_gpt_attention = RopeEmbeddingUtils.create_sinusoidal_positions_for_deepseek_attention_plugin(
                    max_position_embeddings, rope_dim, rotary_embedding_base,
                    scaling_factor, rotary_embedding_origin_max_position,
                    rotary_embedding_beta_fast, rotary_embedding_beta_slow,
                    rotary_embedding_mscale, rotary_embedding_mscale_all_dim)
                rotary_cos_sin = tensorrt_llm.functional.constant(
                    embed_positions_for_gpt_attention)

                mscale = yarn_get_mscale(scaling_factor,
                                         rotary_embedding_mscale_all_dim)
                q_scaling = 1.0 / (mscale * mscale)
                outputs = tensorrt_llm.functional.gpt_attention(
                    qkv=x_tensor,
                    past_key_value=past_key_value_tensor,
                    sequence_length=sequence_length_tensor,
                    host_past_key_value_lengths=
                    host_past_key_value_lengths_tensor,
                    host_max_attention_window_sizes=
                    host_max_attention_window_sizes_tensor,
                    host_sink_token_length=host_sink_token_length_tensor,
                    context_lengths=context_lengths_tensor,
                    cache_indirection=cache_indirection_tensor,
                    host_request_types=host_request_types_tensor,
                    layer_idx=0,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    hidden_size_per_head=c_k_dim + rope_dim,
                    q_scaling=q_scaling,
                    rotary_embedding_dim=rotary_embedding_dim,
                    rotary_embedding_base=rope_base,
                    rotary_embedding_scale_type=rope_scale_type,
                    rotary_embedding_scale=rope_scale,
                    rotary_embedding_max_positions=max_position_embeddings,
                    position_embedding_type=position_embedding_type,
                    rotary_cos_sin=rotary_cos_sin,
                    kv_orig_quant_scale=kv_quant_scale_tensor,
                    kv_quant_orig_scale=kv_dequant_scale_tensor,
                    host_context_lengths=host_context_lengths_tensor,
                    kv_cache_quant_mode=QuantMode.from_description(
                        use_int8_kv_cache=use_int8_kv_cache,
                        use_fp8_kv_cache=use_fp8_kv_cache),
                    kv_cache_block_offsets=kv_cache_block_offsets_tensor,
                    host_kv_cache_block_offsets=
                    host_kv_cache_block_offsets_tensor,
                    host_kv_cache_pool_pointers=
                    host_kv_cache_pool_pointers_tensor,
                    host_kv_cache_pool_mapping=host_kv_cache_pool_mapping_tensor,
                    max_context_length=max_context_length,
                    qkv_bias=None,
                    host_runtime_perf_knobs=host_runtime_perf_knobs_tensor,
                    is_mla_enabled_flag=True,
                    #eps=eps,
                    q_lora_rank=c_q_dim,
                    kv_lora_rank=c_k_dim,
                    qk_nope_head_dim=128,
                    qk_rope_head_dim=64,
                    v_head_dim=128,
                    # position_ids=position_ids_tensor,
                    # q_a_proj=q_a_proj_tensor,
                    # q_a_layernorm=q_a_layernorm_tensor,
                    fused_q_proj=kv_a_proj_with_mqa_tensor,
                    q_b_proj=q_b_proj_tensor,
                    # kv_a_proj_with_mqa=kv_a_proj_with_mqa_tensor,
                    # kv_a_layernorm=kv_a_layernorm_tensor,
                    kv_b_proj=kv_b_proj_tensor)
                # mla_rope_size=rope_dim)

                net._mark_output(outputs[0],
                                 'output',
                                 dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                if not paged_kv_cache:
                    net._mark_output(
                        outputs[1],
                        'present_key_value',
                        dtype=tensorrt_llm.str_dtype_to_trt(kv_cache_dtype))

            inputs = {
                'input': input_tensor,
                'sequence_length': sequence_length,
                'host_past_key_value_lengths': host_past_key_value_lengths,
                'host_max_attention_window_sizes':
                host_max_attention_window_sizes,
                'host_sink_token_length': host_sink_token_length,
                'context_lengths': context_lengths,
                'cache_indirection': cache_indirection,
                'host_request_types': host_request_types,
                'host_runtime_perf_knobs': host_runtime_perf_knobs
            }
            if paged_kv_cache:
                inputs['kv_cache_block_offsets'] = kv_cache_block_offsets
                inputs[
                    'host_kv_cache_block_offsets'] = host_kv_cache_block_offsets
                inputs[
                    'host_kv_cache_pool_pointers'] = host_kv_cache_pool_pointers
                inputs[
                    'host_kv_cache_pool_mapping'] = host_kv_cache_pool_mapping
            else:
                inputs['past_key_value'] = past_key_value

            if use_int8_kv_cache or use_fp8_kv_cache:
                inputs['kv_quant_scale'] = kv_quant_scale
                inputs['kv_dequant_scale'] = kv_dequant_scale

            if enable_remove_input_padding:
                inputs['host_context_lengths'] = host_context_lengths

            inputs["position_ids"] = position_ids
            inputs["q_a_proj"] = q_a_proj
            inputs["q_a_layernorm"] = q_a_layernorm
            inputs["q_b_proj"] = q_b_proj
            inputs["kv_a_proj_with_mqa"] = kv_a_proj_with_mqa
            inputs["kv_a_layernorm"] = kv_a_layernorm
            inputs["kv_b_proj"] = kv_b_proj

            outputs = {'output': output}
            if not paged_kv_cache:
                outputs['present_key_value'] = past_key_value

            stream = torch.cuda.current_stream()
            # NOTE: when 8-bit kv cache is used together with paged kv cache no 8-bit tensors are exposed to TRT
            int8_trt_flag = use_int8_kv_cache and not paged_kv_cache
            use_fp8_kv_cache and not paged_kv_cache
            quant_mode = QuantMode.from_description(
                use_fp8_kv_cache=use_fp8_kv_cache
            ) if use_fp8_kv_cache and not paged_kv_cache else QuantMode(0)
            builder_config = builder.create_builder_config(
                name=attention_type,
                precision=dtype,
                opt_level=0,
                int8=int8_trt_flag,
                quant_mode=quant_mode)

            if session is None:
                engine = builder.build_engine(net, builder_config)
                session = tensorrt_llm.runtime.Session.from_serialized_engine(
                    engine)
            session.run(inputs=inputs,
                        outputs=outputs,
                        stream=stream.cuda_stream)

            torch.cuda.synchronize()
            return session, outputs['output'], past_key_value

        hidden_size = 5120  # embed dimension
        c_q_dim = 1536
        c_k_dim = 512
        rope_dim = 64
        head_num = 128
        head_size = 192
        kv_cache_head_size = c_k_dim + rope_dim
        plugin_kv_num_heads = 1
        kv_hidden_size = plugin_kv_num_heads * kv_cache_head_size
        qkv_hidden_size = hidden_size + 2 * kv_hidden_size
        max_seq_len = in_len + 24
        sink_tokens_in_last_block = sink_token_len % tokens_per_block
        bubble_len = tokens_per_block - sink_tokens_in_last_block if sink_tokens_in_last_block > 0 else 0
        max_blocks_per_seq = math.ceil(
            (max_seq_len + bubble_len) / tokens_per_block)
        num_blocks = batch_size * beam_width * max_blocks_per_seq
        shape_dict = {
            'q_a_proj': (c_q_dim + c_k_dim + rope_dim, hidden_size),
            'q_a_layernorm': (c_q_dim, ),
            'q_b_proj': (head_num * head_size, c_q_dim),
            'kv_a_proj_with_mqa': (head_num * (c_k_dim + rope_dim), c_q_dim),
            'kv_a_layernorm': (c_k_dim, ),
            'kv_b_proj': (2 * head_num * (head_size - rope_dim), c_k_dim),
            'bias': (qkv_hidden_size, ),
            'host_past_key_value_lengths': (batch_size, ),
            'host_max_attention_window_sizes': (1, ),
            'host_sink_token_length': (1, ),
            'sequence_length': (batch_size, ),
            'context_lengths': (batch_size, ),
            'kv_quant_scale': (1, ),
            'kv_dequant_scale': (1, ),
            'cache_indirection': (batch_size, 1, max_seq_len),
            'host_request_types': (batch_size)
        }
        if paged_kv_cache:
            shape_dict['past_key_value'] = (num_blocks, 2, plugin_kv_num_heads,
                                            tokens_per_block,
                                            kv_cache_head_size)
        else:
            shape_dict['past_key_value'] = (batch_size, plugin_kv_num_heads,
                                            max_seq_len, kv_cache_head_size)
        shape_dict['present_key_value'] = shape_dict['past_key_value']
        if enable_remove_input_padding:
            shape_dict['host_context_lengths'] = (batch_size, )

        # HACK: pytorch does not have fp8 dtype yet
        torch_kv_cache_dtype = tensorrt_llm._utils.str_dtype_to_torch(
            'int8'
        ) if kv_cache_dtype == 'fp8' else tensorrt_llm._utils.str_dtype_to_torch(
            kv_cache_dtype)
        present_key_value = torch.zeros(shape_dict['past_key_value'],
                                        dtype=torch_kv_cache_dtype,
                                        device='cuda')
        host_kv_cache_pool_pointers = None
        host_kv_cache_pool_mapping = None
        # Init KV cache block manager
        if paged_kv_cache:
            memory_pools_allocator = MemoryPoolsAllocator(
                num_blocks=num_blocks,
                tokens_per_block=tokens_per_block,
                head_size=c_k_dim + rope_dim)

            num_kv_heads_per_layer = MemoryPoolsAllocator.prepare_num_kv_heads_per_layer(
                plugin_kv_num_heads, 1)
            memory_pools_allocator.allocate(dtype, num_kv_heads_per_layer)
            pools_kv_cache_manager = PoolsKVCacheManager(
                memory_pools_allocator.pools_metadata,
                max_blocks_per_seq,
                num_blocks,
                tokens_per_block,
                c_k_dim + rope_dim,
                max_attention_window_size=max_seq_len,
                beam_width=beam_width,
                sink_token_len=sink_token_len)

            host_kv_cache_pool_pointers = torch.tensor(
                [present_key_value.data_ptr(), 0], dtype=torch.int64)
            host_kv_cache_pool_mapping = memory_pools_allocator.pool_mapping

            # Add sequences to the kv_cache_manager
            for bi in range(batch_size):
                pools_kv_cache_manager.add_sequence(
                    GenerationSequence(seq_idx=bi, batch_idx=bi), in_len)

        q_a_proj = torch.randn(shape_dict['q_a_proj'],
                               dtype=str_dtype_to_torch(dtype),
                               device='cuda')
        q_a_layernorm = torch.randn(shape_dict['q_a_layernorm'],
                                    dtype=str_dtype_to_torch(dtype),
                                    device='cuda')
        q_b_proj = torch.randn(shape_dict['q_b_proj'],
                               dtype=str_dtype_to_torch(dtype),
                               device='cuda')
        kv_a_proj_with_mqa = torch.randn(shape_dict['kv_a_proj_with_mqa'],
                                         dtype=str_dtype_to_torch(dtype),
                                         device='cuda')
        kv_a_layernorm = torch.randn(shape_dict['kv_a_layernorm'],
                                     dtype=str_dtype_to_torch(dtype),
                                     device='cuda')
        kv_b_proj = torch.randn(shape_dict['kv_b_proj'],
                                dtype=str_dtype_to_torch(dtype),
                                device='cuda')

        tmp_proj = kv_b_proj.view(head_num, 2 * (head_size - rope_dim), c_k_dim)
        k_proj, v_proj = torch.split(tmp_proj, [(head_size - rope_dim),
                                                (head_size - rope_dim)],
                                     dim=1)
        k_proj = k_proj.reshape(head_num * (head_size - rope_dim), c_k_dim)
        k_nope_weight = k_proj.reshape(head_num, head_size - rope_dim, c_k_dim)
        v_proj = v_proj.reshape(head_num * (head_size - rope_dim), c_k_dim)
        k_proj = torch.concat([k_proj, v_proj], dim=0)

        torch_present = None

        ConfigCls = DeepseekV2Config
        AttentionCls = DeepseekV2Attention
        configuration = ConfigCls()

        attention = AttentionCls(configuration, 0).cuda().eval()

        q_a_input, kv_a_input = torch.split(q_a_proj,
                                            [c_q_dim, c_k_dim + rope_dim],
                                            dim=0)
        q_nope_weight, q_pe_weight = q_b_proj.reshape(
            head_num, head_size, c_q_dim).split(
                [head_size - rope_dim, rope_dim],
                dim=1,
            )
        fused_q_nope_weight = torch.einsum(
            'hdq,hdk->hkq',
            q_nope_weight,
            k_nope_weight,
        )
        fused_q_weight = torch.cat(
            [fused_q_nope_weight, q_pe_weight],
            dim=1,
        ).flatten(start_dim=0, end_dim=1)

        attention.q_a_proj.weight = torch.nn.parameter.Parameter(
            data=q_a_input.contiguous().clone().detach(), requires_grad=False)
        attention.q_a_layernorm.weight = torch.nn.parameter.Parameter(
            data=q_a_layernorm.clone().detach(), requires_grad=False)
        attention.q_b_proj.weight = torch.nn.parameter.Parameter(
            data=q_b_proj.clone().detach(), requires_grad=False)
        attention.kv_a_proj_with_mqa.weight = torch.nn.parameter.Parameter(
            data=kv_a_input.contiguous().clone().detach(), requires_grad=False)
        attention.kv_a_layernorm.weight = torch.nn.parameter.Parameter(
            data=kv_a_layernorm.clone().detach(), requires_grad=False)
        attention.kv_b_proj.weight = torch.nn.parameter.Parameter(
            data=kv_b_proj.clone().detach(), requires_grad=False)

        input_lengths = torch.ones(
            (batch_size, ), dtype=torch.int32, device='cuda') * (in_len)
        host_context_lengths = input_lengths.cpu(
        ) if enable_remove_input_padding else None
        ctx_attention_mask = torch.ones((batch_size, in_len),
                                        dtype=torch.int32,
                                        device='cuda')
        for i in range(batch_size):
            ctx_attention_mask[i, input_lengths[i]:in_len] = 0

        def remove_input_padding(tensor):
            batch_size = tensor.shape[0]
            tmp = []
            for b in range(batch_size):
                tmp.append(tensor[b, :in_len, :])
            return torch.cat(tmp, dim=1).cuda().reshape(batch_size * (in_len),
                                                        -1)

        cache_indirection = torch.full((
            batch_size,
            beam_width,
            max_seq_len,
        ),
                                       0,
                                       dtype=torch.int32,
                                       device='cuda')

        def get_kv_quant_scale(torch_present):

            torch_kv = torch.cat((torch_present[0], torch_present[1]))
            kv_dequant_scale = torch.tensor([torch.max(torch_kv).item() / 127],
                                            dtype=torch.float32,
                                            device='cuda').reshape(
                                                shape_dict['kv_dequant_scale'])

            # fp8 kv cache uses 1.0f scale.
            if not use_int8_kv_cache:
                kv_dequant_scale = torch.tensor(
                    [1.0], dtype=torch.float32,
                    device='cuda').reshape(shape_dict['kv_dequant_scale'])

            kv_quant_scale = 1.0 / kv_dequant_scale
            return kv_dequant_scale, kv_quant_scale

        def verify_kv_cache(torch_present):
            # If enable streamingllm, kv_cache stores keys and values that with no positional embedding applied
            if streamingllm:
                return

            if not use_int8_kv_cache and not use_fp8_kv_cache and num_kv_heads == num_heads and beam_width == 1:
                if paged_kv_cache:
                    assert pools_kv_cache_manager.has_single_pool(
                    ) is True, f"Current test assuming only one memory pool"
                    kv_cache_manager = pools_kv_cache_manager.get_single_kv_cache_manager(
                    )
                    kv_cache_cont = kv_cache_manager.blocks_manager.get_continuous_caches(
                        present_key_value)
                    kv_cache_cont = kv_cache_cont.permute(1, 0, 2)
                else:
                    kv_cache_cont = present_key_value
                    kv_cache_cont = kv_cache_cont.permute(1, 0, 2, 3, 4)

                key, value = kv_cache_cont.to(torch.float32).chunk(2)

                if paged_kv_cache:
                    # TRT-LLM after paged KV cache it comes with blocks
                    # K cache has shape: [batch_size, max_blocks_per_seq, num_kv_heads, tokens_per_block, head_size]
                    key = key.reshape(batch_size, max_blocks_per_seq,
                                      num_kv_heads, tokens_per_block, head_size)
                    key = key.permute(0, 2, 1, 3, 4).reshape(
                        batch_size, num_kv_heads,
                        max_blocks_per_seq * tokens_per_block, head_size)
                else:
                    key = key.reshape(batch_size, num_kv_heads, max_seq_len,
                                      head_size)

                # Note K and V shares the same layout now.
                if paged_kv_cache:
                    # TRT-LLM after paged KV cache it comes with blocks
                    # V cache has shape: [batch_size, max_blocks_per_seq, num_kv_heads, tokens_per_block, head_size]
                    value = value.reshape(batch_size, max_blocks_per_seq,
                                          num_kv_heads, tokens_per_block,
                                          head_size)
                    value = value.permute(0, 2, 1, 3, 4).reshape(
                        batch_size, num_kv_heads,
                        max_blocks_per_seq * tokens_per_block, head_size)
                else:
                    value = value.reshape(batch_size, num_kv_heads, max_seq_len,
                                          head_size)

                tols = {
                    "float32": 2e-04,
                    "float16": 2e-04,
                    "bfloat16": 2e-01,
                }

                np.testing.assert_allclose(
                    key.cpu().numpy()[:, :, :in_len // 2, :],
                    torch_present[0].to(
                        torch.float32).cpu().numpy()[:, :, :in_len // 2, :],
                    atol=tols[dtype],
                    rtol=tols[dtype])
                np.testing.assert_allclose(
                    value.cpu().numpy()[:, :, :in_len // 2, :],
                    torch_present[1].to(
                        torch.float32).cpu().numpy()[:, :, :in_len // 2, :],
                    atol=tols[dtype],
                    rtol=tols[dtype])

        max_context_length = in_len // 2 if enable_remove_input_padding else in_len
        for step in range(2):
            # The sequence_lengths = context_lengths + step for generation stage.
            sequence_length = torch.add(input_lengths, step)

            kv_cache_block_offsets = None
            if paged_kv_cache:
                # Get arrays of pointers to the "pages" of KV values
                assert pools_kv_cache_manager.has_single_pool(
                ) is True, f"Current test assuming only one memory pool"
                kv_cache_manager = pools_kv_cache_manager.get_single_kv_cache_manager(
                )
                kv_cache_block_offsets = kv_cache_manager.get_block_offsets(
                    beam_width)
            if step == 0:
                host_request_types = torch.tensor([0] * batch_size,
                                                  dtype=torch.int32)
                if paged_kv_cache:
                    # Reassemble pointer array to have KV cache for bs context invocations instead of batch_beam
                    kv_cache_block_offsets = kv_cache_block_offsets[:, 0, :, :]
                    kv_cache_block_offsets = kv_cache_block_offsets.reshape(
                        batch_size, 1, 2, max_blocks_per_seq)

                # Context stage
                shape_dict['input'] = (batch_size, in_len, hidden_size)
                #shape_dict['output'] = (batch_size, in_len, head_num * (head_size - rope_dim))
                output_dim = head_num * (head_size - rope_dim)
                shape_dict['output'] = (batch_size, in_len, output_dim)
                host_past_key_value_lengths = torch.tensor([0] * batch_size,
                                                           dtype=torch.int32)
                host_max_attention_window_sizes = torch.tensor(
                    [max_seq_len], dtype=torch.int32)
                host_sink_token_length = torch.tensor([sink_token_len],
                                                      dtype=torch.int32)

                perf_knob_tensor_size = 16
                context_host_runtime_perf_knobs = torch.tensor(
                    [-1] * perf_knob_tensor_size,
                    dtype=torch.int64,
                    device='cpu')

                context_host_runtime_perf_knobs[
                    0] = 1  # multi_block_mode is default on
                context_host_runtime_perf_knobs[1] = 1

                input_tensor = torch.randn(shape_dict['input'],
                                           dtype=str_dtype_to_torch(dtype),
                                           device='cuda') * 1e-3

                # torch execution
                position_ids = ctx_attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(ctx_attention_mask == 0, 1)

                attention_mask = _prepare_4d_attention_mask(
                    ctx_attention_mask,
                    dtype=str_dtype_to_torch(dtype),
                    tgt_len=in_len)

                attention_mask = attention_mask + AttentionMaskConverter._make_causal_mask(
                    input_tensor.shape[:2],
                    dtype=str_dtype_to_torch(dtype),
                    device='cuda',
                    past_key_values_length=0)

                torch_output, _, torch_present, tmp_out = attention(
                    input_tensor,
                    past_key_value=DynamicCache(),
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=True)
                torch_present = torch_present.to_legacy_cache()

                print(input_tensor.shape)
                input_tensor = tmp_out
                print(input_tensor.shape)

                torch.cuda.synchronize()

                # if attention_type == 'llama_attention':
                #     kv_dequant_scale, kv_quant_scale = get_kv_quant_scale(
                #         torch_present[0])
                # else:
                #     kv_dequant_scale, kv_quant_scale = get_kv_quant_scale(
                #         torch_present)

                if enable_remove_input_padding:
                    shape_dict['input'] = (batch_size * (in_len), hidden_size)
                    input_tensor = remove_input_padding(input_tensor)

                    shape_dict['output'] = (batch_size * (in_len), output_dim)
                output = torch.zeros(shape_dict['output'],
                                     dtype=str_dtype_to_torch(dtype),
                                     device='cuda')
                print("context input")
                print(input_tensor[:20])
                session, output, present_key_value = _construct_execution(
                    session, input_tensor, q_a_proj, q_a_layernorm, q_b_proj,
                    fused_q_weight, kv_a_layernorm, k_proj, position_ids,
                    present_key_value, kv_cache_block_offsets,
                    host_kv_cache_pool_pointers, host_kv_cache_pool_mapping,
                    sequence_length, host_past_key_value_lengths,
                    host_max_attention_window_sizes, host_sink_token_length,
                    input_lengths, host_context_lengths, cache_indirection,
                    host_request_types, num_heads, hidden_size, num_kv_heads,
                    output, dtype, max_context_length, shape_dict,
                    configuration, context_host_runtime_perf_knobs)
                del session
                session = None
                # Note: Volta has larger errors.
                # We speculate it’s because Volta’s TC is smaller and more calculations are required,
                # which may lead to more error accumulation.
                print("===================context===========================")
                if enable_remove_input_padding:
                    torch_output = remove_input_padding(torch_output)
                    print(torch_output)
                    print(output)

                    # np.testing.assert_allclose(
                    #     output.to(torch.float32).cpu().numpy(),
                    #     torch_output.to(torch.float32).cpu().numpy(),
                    #     atol=5e-3 if getSMVersion() > 70 else 5e-2)

                    # np.testing.assert_allclose(
                    #     base_tensor[:, :, 0, :, :].to(torch.float32).cpu().numpy(),
                    #     new_tensor[:, :, 0, :, :].to(torch.float32).cpu().numpy(),
                    #     atol=5e-3 if getSMVersion() > 70 else 5e-2)
                    # np.testing.assert_allclose(
                    #     base_tensor[:, :, 1, :, :].to(torch.float32).cpu().numpy(),
                    #     new_tensor[:, :, 1, :, :].to(torch.float32).cpu().numpy(),
                    #     atol=5e-3 if getSMVersion() > 70 else 5e-2)
                    # np.testing.assert_allclose(
                    #     base_tensor[:, :, 2, :, :128].to(torch.float32).cpu().numpy(),
                    #     new_tensor[:, :, 2, :, :128].to(torch.float32).cpu().numpy(),
                    #     atol=5e-3 if getSMVersion() > 70 else 5e-2)
                    # np.testing.assert_allclose(
                    #     base_tensor[:, :, 1, :, 128:].to(torch.float32).cpu().numpy(),
                    #     new_tensor[:, :, 1, :, 128:].to(torch.float32).cpu().numpy(),
                    #     atol=5e-3 if getSMVersion() > 70 else 5e-2)
                    # np.testing.assert_allclose(
                    #     base_tensor[:, :, 1, :, :128].to(torch.float32).cpu().numpy(),
                    #     new_tensor[:, :, 1, :, :128].to(torch.float32).cpu().numpy(),
                    #     atol=5e-3 if getSMVersion() > 70 else 5e-2)
                else:
                    np.testing.assert_allclose(
                        output[:, :in_len // 2, :].to(
                            torch.float32).cpu().numpy(),
                        torch_output[:, :in_len // 2, :].to(
                            torch.float32).cpu().numpy(),
                        atol=5e-3 if getSMVersion() > 70 else 5e-2)

                verify_kv_cache(torch_present[0])

            else:
                # Generation stage
                shape_dict['input'] = (batch_size, 1, hidden_size)
                host_past_key_value_lengths = sequence_length.cpu() - 1
                host_max_attention_window_sizes = torch.tensor(
                    [max_seq_len], dtype=torch.int32)
                host_sink_token_length = torch.tensor([sink_token_len],
                                                      dtype=torch.int32)
                input_tensor = torch.randn(shape_dict['input'],
                                           dtype=str_dtype_to_torch(dtype),
                                           device='cuda') * 1e-3

                host_request_types = torch.tensor([1] * batch_size,
                                                  dtype=torch.int32)

                ctx_attention_mask = torch.cat((ctx_attention_mask,
                                                ctx_attention_mask.new_ones(
                                                    (batch_size, 1))),
                                               dim=-1).contiguous()

                position_ids = ctx_attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(ctx_attention_mask == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1)

                attention_mask = _prepare_4d_attention_mask(
                    ctx_attention_mask,
                    dtype=str_dtype_to_torch(dtype),
                    tgt_len=1)

                perf_knob_tensor_size = 16
                generation_host_runtime_perf_knobs = torch.tensor(
                    [-1] * perf_knob_tensor_size,
                    dtype=torch.int64,
                    device='cpu')

                generation_host_runtime_perf_knobs[
                    0] = 1  # multi_block_mode is default on
                generation_host_runtime_perf_knobs[1] = 1

                attention_mask = attention_mask + AttentionMaskConverter._make_causal_mask(
                    input_tensor.shape[:2],
                    dtype=str_dtype_to_torch(dtype),
                    device='cuda',
                    past_key_values_length=in_len + step - 1)
                torch_output, _, torch_present, tmp_out = attention(
                    input_tensor,
                    past_key_value=DynamicCache.from_legacy_cache(
                        torch_present),
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=True)
                torch_present = torch_present.to_legacy_cache()
                input_tensor = tmp_out

                def tile_beam_width(tensor: torch.Tensor, num_beams: int):
                    if num_beams == 1:
                        return tensor
                    else:
                        new_shape = np.array(tensor.shape)
                        new_shape[0] = new_shape[0] * num_beams
                        tile_size = np.ones(new_shape.shape, dtype=np.int32)
                        tile_size = np.insert(tile_size, 1, num_beams)
                        new_tensor = torch.unsqueeze(tensor, 1)
                        new_tensor = new_tensor.tile(tile_size.tolist())
                        new_tensor = new_tensor.reshape(new_shape.tolist())
                        return new_tensor

                torch_output = tile_beam_width(torch_output, beam_width)
                torch_output = torch_output.reshape(
                    [batch_size, beam_width, -1])

                torch.cuda.synchronize()

                tiled_input_tensor = tile_beam_width(input_tensor, beam_width)
                tiled_attention_mask = tile_beam_width(attention_mask,
                                                       beam_width)
                tiled_input_lengths = tile_beam_width(input_lengths, beam_width)
                tiled_host_context_lengths = tiled_input_lengths.cpu(
                ) if enable_remove_input_padding else None
                tiled_host_past_key_value_lengths = tile_beam_width(
                    host_past_key_value_lengths, beam_width)
                tiled_host_request_types = tile_beam_width(
                    host_request_types, beam_width)
                tiled_present_key_value = tile_beam_width(
                    present_key_value,
                    beam_width) if not paged_kv_cache else present_key_value
                tiled_sequence_length = tile_beam_width(sequence_length,
                                                        beam_width)

                if enable_remove_input_padding:
                    # shape_dict['input'] = (batch_size, hidden_size)
                    # input_tensor = input_tensor.view(shape_dict['input'])
                    shape_dict['output'] = (batch_size,
                                            head_num * (head_size - rope_dim))
                print(input_tensor.shape)
                print(input_tensor[:20])
                print(fused_q_weight)
                # TRT LLM execution

                output = torch.zeros(shape_dict['output'],
                                     dtype=str_dtype_to_torch(dtype),
                                     device='cuda')

                input_tensor = input_tensor.reshape([batch_size, -1])
                tiled_input_tensor = tile_beam_width(input_tensor, beam_width)
                tiled_input_tensor = tiled_input_tensor.reshape(
                    [batch_size * beam_width, 1, -1])
                output = output.reshape(
                    [batch_size, head_num * (head_size - rope_dim)])
                tiled_output = tile_beam_width(output, beam_width)
                tiled_output = tiled_output.reshape([
                    batch_size * beam_width, 1,
                    head_num * (head_size - rope_dim)
                ])
                print("generation input")
                print(tiled_input_tensor.shape)
                print(tiled_input_tensor[:, :, :20])

                session, tiled_output, present_key_value = _construct_execution(
                    session, tiled_input_tensor, q_a_proj, q_a_layernorm,
                    q_b_proj, fused_q_weight, kv_a_layernorm, kv_b_proj,
                    position_ids, tiled_present_key_value,
                    kv_cache_block_offsets, host_kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping, tiled_sequence_length,
                    tiled_host_past_key_value_lengths,
                    host_max_attention_window_sizes, host_sink_token_length,
                    tiled_input_lengths, tiled_host_context_lengths,
                    cache_indirection, tiled_host_request_types, num_heads,
                    hidden_size, num_kv_heads, tiled_output, dtype,
                    max_context_length, shape_dict, configuration,
                    generation_host_runtime_perf_knobs)
                del session
                session = None

                print(tiled_output)
                # compare result
                print(
                    "====================generation==========================")
                np.testing.assert_allclose(
                    torch.flatten(tiled_output).to(torch.float32).cpu().numpy(),
                    torch.flatten(torch_output).to(torch.float32).cpu().numpy(),
                    atol=output_atol)

            if paged_kv_cache:
                # Iterate to the next step. Increase number of tokens for all unfinished sequences
                # And allocate new blocks if needed
                pools_kv_cache_manager.step([False] * batch_size)
        # assert False, "Force fail"
        return


if __name__ == "__main__":
    unittest.main()
