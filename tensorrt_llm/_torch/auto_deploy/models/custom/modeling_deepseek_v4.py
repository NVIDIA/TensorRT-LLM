# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prefill-only DeepSeek V4 model for AutoDeploy export.

Source:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py

The HuggingFace repo ships a direct reference implementation under
``inference/model.py`` rather than a standard ``modeling_*.py`` file. This file
keeps the checkpoint module hierarchy from that reference implementation:
``embed``, ``layers``, ``norm``, ``head`` and top-level HC head parameters.

Differences from the reference implementation:
* prefill-only forward with mandatory ``position_ids``
* returns logits for all sequence positions instead of only the final token
* uses AutoDeploy canonical ops where they exist: RMSNorm, RoPE, linear, sparse
  attention, MoE
* implements the ratio-4 learned sparse-attention indexer so checkpoint indexer
  tensors are loaded and used for compressed-row selection
* keeps V4-specific hyper-connections as PyTorch reference math because no
  canonical AutoDeploy op currently covers those operations
* omits decode caches and MTP blocks from the exported path
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register all ops
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


@dataclass
class DeepseekV4ModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class DeepseekV4CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        q_lora_rank: int = 1024,
        qk_rope_head_dim: int = 64,
        o_groups: int = 8,
        o_lora_rank: int = 1024,
        sliding_window: int = 128,
        compress_ratios: Optional[Tuple[int, ...]] = None,
        compress_rope_theta: float = 160000.0,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        moe_intermediate_size: int = 2048,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        num_hash_layers: int = 0,
        scoring_func: str = "sqrtsoftplus",
        routed_scaling_factor: float = 1.5,
        swiglu_limit: float = 10.0,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        rms_norm_eps: float = 1e-6,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        ad_rope_cache_len: Optional[int] = None,
        ad_compress_max_seq_len: Optional[int] = None,
        skip_mtp: bool = True,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.sliding_window = sliding_window
        self.compress_ratios = tuple(compress_ratios or (0,) * num_hidden_layers)
        self.compress_rope_theta = compress_rope_theta
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hash_layers = num_hash_layers
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 0,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        self.rms_norm_eps = rms_norm_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.ad_rope_cache_len = ad_rope_cache_len or min(max_position_embeddings, 4096)
        self.ad_compress_max_seq_len = ad_compress_max_seq_len or self.ad_rope_cache_len
        self.skip_mtp = skip_mtp
        self.attention_bias = attention_bias
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


try:
    AutoConfig.register(DeepseekV4Config.model_type, DeepseekV4Config, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register(DeepseekV4Config.model_type, DeepseekV4Config)
    except ValueError:
        pass


def _linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    tp_mode: str = "none",
    layer_type: str = "unknown",
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_linear_simple(
        x, weight, bias, tp_mode=tp_mode, layer_type=layer_type
    )


def _rope_scaling_value(config: PretrainedConfig, key: str, default):
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    return rope_scaling.get(key, default)


def _compute_freqs_cis(
    position_ids: torch.Tensor,
    dim: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=position_ids.device) / dim)
    )
    if original_seq_len > 0:

        def _find_correction_dim(num_rotations: float) -> float:
            return (
                dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(_find_correction_dim(beta_fast)), 0)
        high = min(math.ceil(_find_correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (torch.arange(dim // 2, dtype=torch.float32, device=position_ids.device) - low) / (
            high - low
        )
        smooth = 1 - torch.clamp(ramp, 0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    phases = position_ids.to(torch.float32).unsqueeze(-1) * freqs
    return torch.polar(torch.ones_like(phases), phases)


class DeepseekV4RotaryEmbedding(nn.Module):
    """Precomputed DeepSeek V4 complex RoPE table for an AD export sequence cap."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        original_seq_len: int,
        base: float,
        factor: float,
        beta_fast: int,
        beta_slow: int,
    ) -> None:
        super().__init__()
        position_ids = torch.arange(max_position_embeddings, dtype=torch.long).unsqueeze(0)
        freqs_cis = _compute_freqs_cis(
            position_ids, dim, original_seq_len, base, factor, beta_fast, beta_slow
        ).squeeze(0)
        self.register_buffer("_ad_freqs_cis_cached", freqs_cis, persistent=False)

    def forward(self) -> torch.Tensor:
        return self._ad_freqs_cis_cached


def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    freqs = freqs_cis.conj() if inverse else freqs_cis
    if x.ndim == 3:
        x_in = x.unsqueeze(2)
        x_out, _ = torch.ops.auto_deploy.torch_rope_with_complex_freqs(x_in, x_in, freqs, 2)
        return x_out.squeeze(2)
    x_out, _ = torch.ops.auto_deploy.torch_rope_with_complex_freqs(x, x, freqs, 2)
    return x_out


def _hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    last_dim = int(x.shape[-1])
    if last_dim <= 0 or last_dim & (last_dim - 1):
        raise ValueError(f"Hadamard transform requires a positive power-of-two dim, got {last_dim}")

    out = x.float().reshape(-1, last_dim)
    h = 1
    while h < last_dim:
        out = out.reshape(-1, 2, h)
        left, right = out.unbind(dim=-2)
        out = torch.stack((left + right, left - right), dim=-2).reshape(-1, last_dim)
        h *= 2
    return (out * (last_dim**-0.5)).to(x.dtype).reshape_as(x)


def _fake_fp4_activation_quant_dequant(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    last_dim = int(x.shape[-1])
    if last_dim % block_size != 0:
        return x

    original_dtype = x.dtype
    x_blocks = x.contiguous().float().reshape(-1, last_dim)
    x_blocks = x_blocks.reshape(-1, last_dim // block_size, block_size)
    amax = x_blocks.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * (2.0**-126))
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 6.0)))
    scaled = (x_blocks / scale).clamp(-6.0, 6.0)

    abs_scaled = scaled.abs()
    quantized = torch.zeros_like(abs_scaled)
    quantized = torch.where(abs_scaled > 0.25, torch.full_like(quantized, 0.5), quantized)
    quantized = torch.where(abs_scaled >= 0.75, torch.full_like(quantized, 1.0), quantized)
    quantized = torch.where(abs_scaled > 1.25, torch.full_like(quantized, 1.5), quantized)
    quantized = torch.where(abs_scaled >= 1.75, torch.full_like(quantized, 2.0), quantized)
    quantized = torch.where(abs_scaled > 2.5, torch.full_like(quantized, 3.0), quantized)
    quantized = torch.where(abs_scaled >= 3.5, torch.full_like(quantized, 4.0), quantized)
    quantized = torch.where(abs_scaled > 5.0, torch.full_like(quantized, 6.0), quantized)
    quantized = torch.where(scaled < 0, -quantized, quantized)
    return (quantized * scale).to(original_dtype).reshape_as(x)


def _window_topk_idxs(window_size: int, batch_size: int, seq_len: int, device) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    offsets = torch.arange(window_size, device=device)
    matrix = (positions[:, None] - window_size + 1).clamp(min=0) + offsets[None, :]
    matrix = torch.where(matrix <= positions[:, None], matrix, -1)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


def _compress_topk_idxs(
    ratio: int,
    batch_size: int,
    seq_len: int,
    offset: int,
    device,
    max_compressed_len: Optional[int] = None,
) -> torch.Tensor:
    compressed_len = max_compressed_len if max_compressed_len is not None else seq_len // ratio
    compressed = torch.arange(compressed_len, device=device)
    matrix = compressed.unsqueeze(0).expand(seq_len, -1)
    valid_lengths = torch.arange(1, seq_len + 1, device=device).unsqueeze(1) // ratio
    matrix = torch.where(matrix < valid_lengths, matrix + offset, -1)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


def _sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    softmax_scale: float,
    enable_sharding: bool = False,
    layer_type: str = "unknown",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_v2(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        softmax_scale,
        enable_sharding=enable_sharding,
        layer_type=layer_type,
        layer_idx=layer_idx,
        window_size=window_size,
        compress_ratio=compress_ratio,
        max_compressed_len=max_compressed_len,
        head_dim=head_dim,
        rope_dim=rope_dim,
        rms_norm_eps=rms_norm_eps,
    )


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].view(*mixes.shape[:-1], hc_mult, hc_mult)
    comb_base = hc_base[2 * hc_mult :].view(hc_mult, hc_mult)
    comb = comb * hc_scale[2] + comb_base
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class DeepseekV4Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        x: torch.Tensor,
        tp_mode: str = "none",
        layer_type: str = "unknown",
    ) -> torch.Tensor:
        return _linear(x, self.weight, self.bias, tp_mode=tp_mode, layer_type=layer_type)


class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


class DeepseekV4Compressor(nn.Module):
    def __init__(self, config: PretrainedConfig, compress_ratio: int, head_dim: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        channels = 2 if self.overlap else 1

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, channels * self.head_dim, dtype=torch.float32)
        )
        self.wkv = DeepseekV4Linear(self.hidden_size, channels * self.head_dim)
        self.wgate = DeepseekV4Linear(self.hidden_size, channels * self.head_dim)
        self.norm = DeepseekV4RMSNorm(self.head_dim, config.rms_norm_eps)
        max_seq_len = int(getattr(config, "ad_compress_max_seq_len", config.ad_rope_cache_len))
        self.max_compressed_len = max(1, (max_seq_len + compress_ratio - 1) // compress_ratio)
        self.max_compressed_tokens = self.max_compressed_len * compress_ratio

    def _overlap_transform(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        batch_size, compressed_len, _, _ = tensor.shape
        ratio = self.compress_ratio
        head_dim = self.head_dim
        out = tensor.new_full((batch_size, compressed_len, 2 * ratio, head_dim), value)
        out[:, :, ratio:] = tensor[:, :, :, head_dim:]
        out[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
        return out

    def project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.wkv.weight.dtype)
        return self.wkv(x), self.wgate(x)

    def pool_norm_rope(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        position_ids: torch.Tensor,
        freqs_cis_table: torch.Tensor,
        norm_dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = kv.shape
        ratio = self.compress_ratio
        pad_len = self.max_compressed_tokens - seq_len
        kv = F.pad(kv, (0, 0, 0, pad_len))
        score = F.pad(score, (0, 0, 0, pad_len), value=float("-inf"))
        kv = kv.view(batch_size, self.max_compressed_len, ratio, kv.shape[-1])
        score = score.view(batch_size, self.max_compressed_len, ratio, score.shape[-1]) + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0)
            score = self._overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = kv.to(norm_dtype)
        kv = self.norm(kv)

        row_idx = torch.arange(self.max_compressed_len, device=kv.device) * ratio
        row_idx = torch.minimum(row_idx, torch.full_like(row_idx, seq_len - 1))
        row_idx = row_idx.unsqueeze(0).expand(batch_size, -1)
        compressed_position_ids = torch.gather(position_ids, 1, row_idx)
        compressed_freqs = freqs_cis_table[compressed_position_ids]
        rope = _apply_rope(kv[..., -self.rope_head_dim :], compressed_freqs)
        return torch.cat([kv[..., : -self.rope_head_dim], rope], dim=-1)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, freqs_cis_table: torch.Tensor
    ) -> torch.Tensor:
        kv, score = self.project(x)
        return self.pool_norm_rope(kv, score, position_ids, freqs_cis_table, kv.dtype)


class DeepseekV4Indexer(nn.Module):
    def __init__(self, config: PretrainedConfig, compress_ratio: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = DeepseekV4Linear(self.q_lora_rank, self.num_heads * self.head_dim)
        self.weights_proj = DeepseekV4Linear(self.hidden_size, self.num_heads)
        self.compressor = DeepseekV4Compressor(config, compress_ratio, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        position_ids: torch.Tensor,
        freqs_cis_table: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        freqs_cis = freqs_cis_table[position_ids]

        q = self.wq_b(qr, tp_mode="colwise", layer_type="mla")
        q = torch.ops.auto_deploy.view(
            q,
            [batch_size, seq_len, self.num_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        q_rope = _apply_rope(q[..., -self.rope_head_dim :], freqs_cis)
        q = torch.cat([q[..., : -self.rope_head_dim], q_rope], dim=-1)
        q = _hadamard_transform(q)
        q = _fake_fp4_activation_quant_dequant(q)

        compressed_kv = self.compressor(x, position_ids, freqs_cis_table)
        compressed_kv = _hadamard_transform(compressed_kv)
        compressed_kv = _fake_fp4_activation_quant_dequant(compressed_kv)
        weights = self.weights_proj(x, tp_mode="colwise", layer_type="mla")
        weights = weights * (self.softmax_scale * self.num_heads**-0.5)

        index_score = torch.einsum("bshd,btd->bsht", q.float(), compressed_kv.float())
        index_score = (index_score.relu() * weights.float().unsqueeze(-1)).sum(dim=2)
        index_score = torch.ops.auto_deploy.all_reduce(index_score, layer_type="mla")

        compressed_positions = torch.arange(self.compressor.max_compressed_len, device=x.device)
        valid_lengths = torch.arange(1, seq_len + 1, device=x.device).unsqueeze(1)
        valid_lengths = valid_lengths // self.compress_ratio
        invalid = compressed_positions.unsqueeze(0) >= valid_lengths
        index_score = torch.where(
            invalid.unsqueeze(0),
            torch.full_like(index_score, float("-inf")),
            index_score,
        )

        topk_count = min(self.index_topk, self.compressor.max_compressed_len)
        topk_idxs = index_score.topk(topk_count, dim=-1)[1]
        valid_topk = topk_idxs < valid_lengths.unsqueeze(0)
        return torch.where(valid_topk, topk_idxs + offset, -1)


class DeepseekV4Attention(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.num_groups = config.o_groups
        self.window_size = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.eps = config.rms_norm_eps
        self.softmax_scale = self.head_dim**-0.5

        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.wq_a = DeepseekV4Linear(self.hidden_size, self.q_lora_rank)
        self.q_norm = DeepseekV4RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = DeepseekV4Linear(self.q_lora_rank, self.num_heads * self.head_dim)
        self.wkv = DeepseekV4Linear(self.hidden_size, self.head_dim)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, self.eps)
        self.wo_a = DeepseekV4Linear(
            self.num_heads * self.head_dim // self.num_groups,
            self.num_groups * self.o_lora_rank,
        )
        self.wo_b = DeepseekV4Linear(self.num_groups * self.o_lora_rank, self.hidden_size)
        self.indexer = None
        if self.compress_ratio:
            self.compressor = DeepseekV4Compressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = DeepseekV4Indexer(config, self.compress_ratio)

        if self.compress_ratio:
            self.rope_original_seq_len = _rope_scaling_value(
                config, "original_max_position_embeddings", 0
            )
            self.rope_base = config.compress_rope_theta
        else:
            self.rope_original_seq_len = 0
            self.rope_base = config.rope_theta
        self.rope_factor = _rope_scaling_value(config, "factor", 1.0)
        self.rope_beta_fast = _rope_scaling_value(config, "beta_fast", 32)
        self.rope_beta_slow = _rope_scaling_value(config, "beta_slow", 1)
        self.rotary_emb = DeepseekV4RotaryEmbedding(
            self.rope_head_dim,
            config.ad_rope_cache_len,
            self.rope_original_seq_len,
            self.rope_base,
            self.rope_factor,
            self.rope_beta_fast,
            self.rope_beta_slow,
        )

    def _freqs_cis(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.rotary_emb()[position_ids]

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        freqs_cis_table = self.rotary_emb()
        freqs_cis = freqs_cis_table[position_ids]

        qr = self.q_norm(self.wq_a(x, layer_type="mla"))
        q = self.wq_b(qr, tp_mode="colwise", layer_type="mla")
        q = torch.ops.auto_deploy.view(
            q,
            [batch_size, seq_len, self.num_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_rope = _apply_rope(q[..., -self.rope_head_dim :], freqs_cis)
        q = torch.cat([q[..., : -self.rope_head_dim], q_rope], dim=-1)

        kv = self.kv_norm(self.wkv(x, layer_type="mla"))
        kv_rope = _apply_rope(kv[..., -self.rope_head_dim :], freqs_cis)
        kv = torch.cat([kv[..., : -self.rope_head_dim], kv_rope], dim=-1)

        topk_idxs = _window_topk_idxs(self.window_size, batch_size, seq_len, x.device)
        compressor_kv = x.new_empty(batch_size, seq_len, 0)
        compressor_gate = x.new_empty(batch_size, seq_len, 0)
        compressor_ape = x.new_empty(0, 0)
        compressor_norm_weight = x.new_empty(0)
        if self.compress_ratio:
            compressor_kv, compressor_gate = self.compressor.project(x)
            if self.indexer is not None:
                compressed_idxs = self.indexer(x, qr, position_ids, freqs_cis_table, seq_len)
            else:
                compressed_idxs = _compress_topk_idxs(
                    self.compress_ratio,
                    batch_size,
                    seq_len,
                    seq_len,
                    x.device,
                    self.compressor.max_compressed_len,
                )
            topk_idxs = torch.cat([topk_idxs, compressed_idxs], dim=-1)
            compressor_ape = self.compressor.ape
            compressor_norm_weight = self.compressor.norm.weight

        max_compressed_len = self.compressor.max_compressed_len if self.compress_ratio else None
        o = _sparse_attention(
            q,
            kv,
            self.attn_sink,
            topk_idxs.int(),
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            freqs_cis_table,
            position_ids,
            self.softmax_scale,
            enable_sharding=True,
            layer_type="mla",
            layer_idx=self.layer_idx,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            max_compressed_len=max_compressed_len,
            head_dim=self.head_dim,
            rope_dim=self.rope_head_dim,
            rms_norm_eps=self.eps,
        )
        o_rope = _apply_rope(o[..., -self.rope_head_dim :], freqs_cis, inverse=True)
        o = torch.cat([o[..., : -self.rope_head_dim], o_rope], dim=-1)
        o = torch.ops.auto_deploy.view(
            o,
            [batch_size, seq_len, self.num_groups, -1],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        wo_a = self.wo_a.weight.view(self.num_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        o = self.wo_b(o.flatten(2), tp_mode="rowwise", layer_type="mla")
        return torch.ops.auto_deploy.all_reduce(o, layer_type="mla")


class DeepseekV4Gate(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int) -> None:
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.score_func = config.scoring_func
        self.route_scale = config.routed_scaling_factor
        self.is_hash = layer_idx < config.num_hash_layers
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        if self.is_hash:
            initial = torch.arange(self.topk, dtype=torch.long) % config.n_routed_experts
            tid2eid = initial.unsqueeze(0).expand(config.vocab_size, -1).clone()
            self.tid2eid = nn.Parameter(tid2eid, requires_grad=False)
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.zeros(config.n_routed_experts, dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.score_func == "sqrtsoftplus":
            selected_experts, routing_weights = torch.ops.auto_deploy.torch_deepseek_v4_router(
                x,
                input_ids,
                self.weight,
                self.bias,
                self.tid2eid if self.is_hash else None,
                self.topk,
                self.route_scale,
                self.is_hash,
            )
            return routing_weights, selected_experts

        scores = _linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        elif self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        else:
            raise ValueError(f"Unsupported DeepSeek V4 scoring_func: {self.score_func}")

        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.is_hash:
            assert input_ids is not None, "input_ids is required for DeepSeek V4 hash-routed layers"
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights * self.route_scale, indices


class DeepseekV4Expert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0) -> None:
        super().__init__()
        self.w1 = DeepseekV4Linear(hidden_size, intermediate_size)
        self.w2 = DeepseekV4Linear(intermediate_size, hidden_size)
        self.w3 = DeepseekV4Linear(hidden_size, intermediate_size)
        self.swiglu_limit = swiglu_limit

    @property
    def gate_proj(self) -> DeepseekV4Linear:
        return self.w1

    @property
    def down_proj(self) -> DeepseekV4Linear:
        return self.w2

    @property
    def up_proj(self) -> DeepseekV4Linear:
        return self.w3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x, tp_mode="colwise", layer_type="moe").float()
        up = self.w3(x, tp_mode="colwise", layer_type="moe").float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        hidden = F.silu(gate) * up
        return self.w2(hidden.to(x.dtype), tp_mode="rowwise", layer_type="moe")


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.topk = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.score_func = config.scoring_func
        self.is_hash = layer_idx < config.num_hash_layers
        self.gate = DeepseekV4Gate(config, layer_idx)
        self.experts = nn.ModuleList(
            [
                DeepseekV4Expert(
                    config.hidden_size, config.moe_intermediate_size, config.swiglu_limit
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        assert config.n_shared_experts == 1, "DeepSeek V4 reference expects one shared expert"
        self.shared_experts = DeepseekV4Expert(
            config.hidden_size, config.moe_intermediate_size, swiglu_limit=0.0
        )

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if self.score_func != "sqrtsoftplus":
            raise ValueError(f"Unsupported DeepSeek V4 scoring_func: {self.score_func}")

        shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        routed = torch.ops.auto_deploy.torch_deepseek_v4_moe(
            x_flat,
            input_ids.flatten(),
            self.gate.weight,
            self.gate.bias,
            self.gate.tid2eid if self.is_hash else None,
            torch.stack([expert.w1.weight for expert in self.experts]),
            torch.stack([expert.w2.weight for expert in self.experts]),
            torch.stack([expert.w3.weight for expert in self.experts]),
            self.shared_experts.w1.weight,
            self.shared_experts.w2.weight,
            self.shared_experts.w3.weight,
            self.topk,
            self.route_scale,
            self.swiglu_limit,
            self.is_hash,
            layer_type="moe",
        )
        return routed.to(x.dtype).view(shape)


class DeepseekV4Block(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.attn = DeepseekV4Attention(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = DeepseekV4RMSNorm(config.hidden_size, self.norm_eps)
        self.ffn_norm = DeepseekV4RMSNorm(config.hidden_size, self.norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def _hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = _linear(flat, hc_fn, None) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2)
        return y.to(x.dtype), post, comb

    @staticmethod
    def _hc_post(
        x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        y = post.unsqueeze(-1) * x.unsqueeze(-2)
        y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.to(x.dtype)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        x, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, position_ids)
        x = self._hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return self._hc_post(x, residual, post, comb)


class DeepseekV4Head(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear(x.float(), self.weight.float(), None).to(x.dtype)


class DeepseekV4PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = ""
    _no_split_modules = ["DeepseekV4Block"]
    _keys_to_ignore_on_load_unexpected = [r"^mtp\.0\."]
    supports_gradient_checkpointing = False

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.register_load_state_dict_pre_hook(self._skip_mtp_load_hook)

    @staticmethod
    def _skip_mtp_load_hook(
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        *args,
    ) -> None:
        del module, args
        if prefix:
            return
        for key in list(state_dict):
            if key.startswith("mtp.0."):
                state_dict.pop(key)

    def _init_weights(self, module: nn.Module) -> None:
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, DeepseekV4Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DeepseekV4Head):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, DeepseekV4RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, DeepseekV4Gate):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DeepseekV4Compressor):
            module.ape.data.normal_(mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            module.attn_sink.data.zero_()
        elif isinstance(module, DeepseekV4Block):
            module.hc_attn_fn.data.normal_(mean=0.0, std=std)
            module.hc_ffn_fn.data.normal_(mean=0.0, std=std)
            module.hc_attn_base.data.zero_()
            module.hc_ffn_base.data.zero_()
            module.hc_attn_scale.data.fill_(1.0)
            module.hc_ffn_scale.data.fill_(1.0)
        elif isinstance(module, DeepseekV4ForCausalLM):
            module.hc_head_fn.data.normal_(mean=0.0, std=std)
            module.hc_head_base.data.zero_()
            module.hc_head_scale.data.fill_(1.0)


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin):
    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        kwargs.pop("skip_mtp", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected DeepSeek V4 model kwarg(s): {unexpected}")
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV4Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = DeepseekV4Head(config.vocab_size, config.hidden_size)
        self.hc_mult = config.hc_mult
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, new_embeddings):
        self.embed = new_embeddings

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        mixes = _linear(flat, self.hc_head_fn, None) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
        return torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2).to(x.dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> DeepseekV4CausalLMOutput:
        assert position_ids is not None, "position_ids is required"
        assert input_ids is not None, "input_ids is required"
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, input_ids)
        hidden_states = self._hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.head(hidden_states)
        return DeepseekV4CausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("DeepseekV4Config", DeepseekV4ForCausalLM)


@ModelFactoryRegistry.register("DeepseekV4AutoModelForCausalLM")
class DeepseekV4AutoModelForCausalLMFactory(AutoModelForCausalLMFactory):
    def _get_model_config(self) -> Tuple[PretrainedConfig, Dict[str, Any]]:
        model_config, unused_kwargs = super()._get_model_config()
        if (
            "ad_rope_cache_len" in self.model_kwargs
            and "ad_compress_max_seq_len" not in self.model_kwargs
            and hasattr(model_config, "ad_rope_cache_len")
            and hasattr(model_config, "ad_compress_max_seq_len")
        ):
            model_config.ad_compress_max_seq_len = model_config.ad_rope_cache_len
        return model_config, unused_kwargs

    def get_example_inputs(self) -> Dict[str, torch.Tensor]:
        model_config, _ = self._get_model_config()
        ratios = tuple(getattr(model_config, "compress_ratios", ()))
        num_layers = getattr(model_config, "num_hidden_layers", len(ratios))
        active_ratios = [int(ratio) for ratio in ratios[:num_layers] if int(ratio) > 0]
        export_seq_len = max(4, 2 * max(active_ratios, default=0))
        export_seq_len = max(1, min(self.max_seq_len, export_seq_len))
        return {"input_ids": torch.ones(2, export_seq_len, dtype=torch.int)}


DeepseekV4AutoModelForCausalLMFactory.register_custom_model_cls(
    "DeepseekV4Config", DeepseekV4ForCausalLM
)
