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
* uses AutoDeploy canonical ops where they exist: RMSNorm, RoPE, linear, MoE
* keeps V4-specific sparse attention and hyper-connections as PyTorch reference
  math because no canonical AutoDeploy op currently covers those operations
* omits decode caches and MTP blocks from the exported path
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register all ops
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType


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


def _window_topk_idxs(window_size: int, batch_size: int, seq_len: int, device) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    offsets = torch.arange(window_size, device=device)
    matrix = (positions[:, None] - window_size + 1).clamp(min=0) + offsets[None, :]
    matrix = torch.where(matrix <= positions[:, None], matrix, -1)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


def _compress_topk_idxs(
    ratio: int, batch_size: int, seq_len: int, offset: int, device
) -> torch.Tensor:
    compressed_len = seq_len // ratio
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
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, seq_len, _, head_dim = q.shape
    gather_idxs = topk_idxs.clamp(min=0)
    gather = gather_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kv_for_gather = kv.unsqueeze(1).expand(-1, seq_len, -1, -1)
    selected_kv = torch.gather(kv_for_gather, 2, gather)

    scores = torch.einsum("bshd,bskd->bshk", q.float(), selected_kv.float()) * softmax_scale
    invalid = topk_idxs < 0
    scores = torch.where(invalid.unsqueeze(2), torch.full_like(scores, float("-inf")), scores)

    sink = attn_sink.view(1, 1, -1, 1).expand(batch_size, seq_len, -1, -1)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    output = torch.einsum("bshk,bskd->bshd", probs.to(selected_kv.dtype), selected_kv)
    return output.to(q.dtype)


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

    def _overlap_transform(self, tensor: torch.Tensor, value: float) -> torch.Tensor:
        batch_size, compressed_len, _, _ = tensor.shape
        ratio = self.compress_ratio
        head_dim = self.head_dim
        out = tensor.new_full((batch_size, compressed_len, 2 * ratio, head_dim), value)
        out[:, :, ratio:] = tensor[:, :, :, head_dim:]
        out[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
        return out

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        ratio = self.compress_ratio
        cutoff = (seq_len // ratio) * ratio
        kv = self.wkv(x.float())[:, :cutoff]
        score = self.wgate(x.float())[:, :cutoff]
        kv = kv.view(batch_size, -1, ratio, kv.shape[-1])
        score = score.view(batch_size, -1, ratio, score.shape[-1]) + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0)
            score = self._overlap_transform(score, float("-inf"))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
        kv = self.norm(kv.to(x.dtype))

        compressed_freqs = freqs_cis[:, :cutoff:ratio]
        rope = _apply_rope(kv[..., -self.rope_head_dim :], compressed_freqs)
        return torch.cat([kv[..., : -self.rope_head_dim], rope], dim=-1)


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
        if self.compress_ratio:
            self.compressor = DeepseekV4Compressor(config, self.compress_ratio, self.head_dim)

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
        freqs_cis = self._freqs_cis(position_ids)

        qr = self.q_norm(self.wq_a(x, layer_type="mla"))
        q = self.wq_b(qr, tp_mode="colwise", layer_type="mla")
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q_rope = _apply_rope(q[..., -self.rope_head_dim :], freqs_cis)
        q = torch.cat([q[..., : -self.rope_head_dim], q_rope], dim=-1)

        kv = self.kv_norm(self.wkv(x, layer_type="mla"))
        kv_rope = _apply_rope(kv[..., -self.rope_head_dim :], freqs_cis)
        kv = torch.cat([kv[..., : -self.rope_head_dim], kv_rope], dim=-1)

        topk_idxs = _window_topk_idxs(self.window_size, batch_size, seq_len, x.device)
        if self.compress_ratio:
            compressed_kv = self.compressor(x, position_ids, freqs_cis)
            compressed_idxs = _compress_topk_idxs(
                self.compress_ratio, batch_size, seq_len, seq_len, x.device
            )
            topk_idxs = torch.cat([topk_idxs, compressed_idxs], dim=-1)
            kv = torch.cat([kv, compressed_kv], dim=1)

        o = _sparse_attention(q, kv, self.attn_sink, topk_idxs.int(), self.softmax_scale)
        o_rope = _apply_rope(o[..., -self.rope_head_dim :], freqs_cis, inverse=True)
        o = torch.cat([o[..., : -self.rope_head_dim], o_rope], dim=-1)
        o = o.view(batch_size, seq_len, self.num_groups, -1)
        wo_a = self.wo_a.weight.view(self.num_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2), tp_mode="rowwise", layer_type="mla")


class DeepseekV4Gate(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int) -> None:
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.score_func = config.scoring_func
        self.route_scale = config.routed_scaling_factor
        self.is_hash = layer_idx < config.num_hash_layers
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        if self.is_hash:
            initial = torch.arange(self.topk, dtype=torch.int32) % config.n_routed_experts
            tid2eid = initial.unsqueeze(0).expand(config.vocab_size, -1).clone()
            self.tid2eid = nn.Parameter(tid2eid, requires_grad=False)
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.zeros(config.n_routed_experts, dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _dense_experts(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        for expert_idx, expert in enumerate(self.experts):
            expert_weight = torch.where(
                selected_experts == expert_idx,
                routing_weights,
                torch.zeros_like(routing_weights),
            ).sum(dim=-1, keepdim=True)
            output = output + expert(x) * expert_weight.to(x.dtype)
        return output

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        weights, indices = self.gate(x_flat, input_ids.flatten())
        if self.swiglu_limit == 0:
            routed = torch.ops.auto_deploy.torch_moe(
                x_flat,
                indices,
                weights,
                w1_weight=[expert.w1.weight for expert in self.experts],
                w2_weight=[expert.w2.weight for expert in self.experts],
                w3_weight=[expert.w3.weight for expert in self.experts],
                is_gated_mlp=True,
                act_fn=int(ActivationType.Silu),
                layer_type="moe",
            )
        else:
            routed = self._dense_experts(x_flat, indices, weights)
        routed = routed + self.shared_experts(x_flat)
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
    _keys_to_ignore_on_load_unexpected = [r"^mtp\."]
    supports_gradient_checkpointing = False

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
    def __init__(self, config: PretrainedConfig) -> None:
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
