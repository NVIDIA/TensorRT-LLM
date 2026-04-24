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

"""Hierarchical DeepSeek V4 tests against a minimal HF-reference copy.

DeepSeek V4 Flash does not ship a standard HuggingFace ``modeling_*.py`` file.
The reference implementation lives in ``inference/model.py`` in the HF repo, so
these tests carry a small plain-PyTorch copy of the prefill math.
"""

import math
import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register ops
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    DeepseekV4RMSNorm,
)


def assert_rmse_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rmse_ratio_tol: float,
    msg: str = "",
) -> None:
    diff = actual.float() - expected.float()
    rmse_diff = torch.sqrt(torch.mean(diff**2))
    rmse_ref = torch.sqrt(torch.mean(expected.float() ** 2))
    ratio = (rmse_diff / rmse_ref.clamp_min(1e-12)).item()
    assert ratio < rmse_ratio_tol, (
        f"{msg}RMSE ratio {ratio:.6f} exceeds tolerance {rmse_ratio_tol}. "
        f"(rmse_diff={rmse_diff.item():.6f}, rmse_ref={rmse_ref.item():.6f})"
    )


def _position_ids(batch: int, seq: int, device) -> torch.Tensor:
    return torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)


def _small_config(**overrides) -> DeepseekV4Config:
    values = dict(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        o_groups=1,
        o_lora_rank=8,
        sliding_window=4,
        compress_ratios=(0, 4),
        compress_rope_theta=10000.0,
        moe_intermediate_size=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.25,
        swiglu_limit=0.0,
        max_position_embeddings=32,
        rope_scaling={
            "type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 0,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        rms_norm_eps=1e-6,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
    )
    values.update(overrides)
    return DeepseekV4Config(**values)


_DEEPSEEK_V4_CHECKPOINT_SAMPLE_KEYS = {
    "embed.weight",
    "head.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "layers.0.attn.attn_sink",
    "layers.0.attn.wq_a.weight",
    "layers.0.attn.wq_b.weight",
    "layers.0.attn.wkv.weight",
    "layers.0.ffn.experts.0.w1.weight",
    "layers.0.ffn.experts.0.w2.weight",
    "layers.0.ffn.experts.0.w3.weight",
    "layers.0.ffn.gate.tid2eid",
    "layers.0.ffn.shared_experts.w1.weight",
    "layers.0.ffn.shared_experts.w2.weight",
    "layers.0.ffn.shared_experts.w3.weight",
    "layers.0.hc_attn_fn",
    "layers.0.hc_ffn_fn",
    "norm.weight",
}


_STACKED_EXPERT_RE = re.compile(
    r"^(?P<prefix>layers\.\d+\.ffn\.experts)\.(?P<proj>w[123])\.(?P<suffix>weight|scale)$"
)


def _convert_deepseek_v4_reference_state_dict(state_dict, num_experts):
    """Convert possible stacked reference MoE tensors to checkpoint-style per-expert keys.

    The observed DeepSeek V4 Flash safetensor index already uses per-expert keys
    such as ``layers.0.ffn.experts.0.w1.weight``. This helper also supports the
    common test/reference form ``layers.0.ffn.experts.w1.weight`` with expert
    stacked on dim 0, so MoE conversion behavior is explicit and covered.
    """
    converted = {}
    for key, value in state_dict.items():
        match = _STACKED_EXPERT_RE.match(key)
        if match is None:
            converted[key] = value
            continue

        assert value.shape[0] == num_experts
        prefix = match.group("prefix")
        proj = match.group("proj")
        suffix = match.group("suffix")
        for expert_idx in range(num_experts):
            converted[f"{prefix}.{expert_idx}.{proj}.{suffix}"] = value[expert_idx]
    return converted


def _ref_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
    return F.linear(x, weight, bias)


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.square().mean(-1, keepdim=True)
    return (weight * x * torch.rsqrt(variance + eps)).to(dtype)


def _ref_freqs_cis(
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

        def find_correction_dim(num_rotations):
            return (
                dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(find_correction_dim(beta_fast)), 0)
        high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        smooth = 1 - torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32, device=position_ids.device) - low)
            / (high - low),
            0,
            1,
        )
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    phase = position_ids.to(torch.float32).unsqueeze(-1) * freqs
    return torch.polar(torch.ones_like(phase), phase)


def _ref_apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    y = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    freqs = freqs_cis.conj() if inverse else freqs_cis
    if x.ndim == 3:
        freqs = freqs.view(x.shape[0], x.shape[1], -1)
    else:
        freqs = freqs.view(x.shape[0], x.shape[1], 1, -1)
    return torch.view_as_real(y * freqs).flatten(-2).to(x.dtype)


def _ref_window_topk(window_size: int, batch_size: int, seq_len: int, device):
    positions = torch.arange(seq_len, device=device)
    offsets = torch.arange(window_size, device=device)
    matrix = (positions[:, None] - window_size + 1).clamp(min=0) + offsets[None, :]
    matrix = torch.where(matrix <= positions[:, None], matrix, -1)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


def _ref_compress_topk(ratio: int, batch_size: int, seq_len: int, offset: int, device):
    compressed_len = seq_len // ratio
    compressed = torch.arange(compressed_len, device=device)
    matrix = compressed.unsqueeze(0).expand(seq_len, -1)
    valid_lengths = torch.arange(1, seq_len + 1, device=device).unsqueeze(1) // ratio
    matrix = torch.where(matrix < valid_lengths, matrix + offset, -1)
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


def _ref_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale):
    batch_size, seq_len, _, head_dim = q.shape
    gather = topk_idxs.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, head_dim)
    selected = torch.gather(kv.unsqueeze(1).expand(-1, seq_len, -1, -1), 2, gather)
    scores = torch.einsum("bshd,bskd->bshk", q.float(), selected.float()) * softmax_scale
    scores = torch.where(
        (topk_idxs < 0).unsqueeze(2), torch.full_like(scores, float("-inf")), scores
    )
    sink = attn_sink.view(1, 1, -1, 1).expand(batch_size, seq_len, -1, -1)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probs.to(selected.dtype), selected).to(q.dtype)


def _ref_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].view(*mixes.shape[:-1], hc_mult, hc_mult)
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class _RefLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter("bias", None)

    def forward(self, x):
        return _ref_linear(x, self.weight, self.bias)


class _RefRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        return _ref_rmsnorm(x, self.weight, self.eps)


class _RefCompressor(nn.Module):
    def __init__(self, config, compress_ratio, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        channels = 2 if self.overlap else 1
        self.ape = nn.Parameter(torch.empty(compress_ratio, channels * head_dim))
        self.wkv = _RefLinear(config.hidden_size, channels * head_dim)
        self.wgate = _RefLinear(config.hidden_size, channels * head_dim)
        self.norm = _RefRMSNorm(head_dim, config.rms_norm_eps)

    def _overlap_transform(self, tensor, value):
        batch_size, compressed_len, _, _ = tensor.shape
        ratio, head_dim = self.compress_ratio, self.head_dim
        out = tensor.new_full((batch_size, compressed_len, 2 * ratio, head_dim), value)
        out[:, :, ratio:] = tensor[:, :, :, head_dim:]
        out[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
        return out

    def forward(self, x, freqs_cis):
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
        rope = _ref_apply_rope(kv[..., -self.rope_head_dim :], freqs_cis[:, :cutoff:ratio])
        return torch.cat([kv[..., : -self.rope_head_dim], rope], dim=-1)


class _RefAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.num_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.window_size = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.eps = config.rms_norm_eps
        self.softmax_scale = self.head_dim**-0.5
        self.attn_sink = nn.Parameter(torch.empty(self.num_heads))
        self.wq_a = _RefLinear(config.hidden_size, config.q_lora_rank)
        self.q_norm = _RefRMSNorm(config.q_lora_rank, self.eps)
        self.wq_b = _RefLinear(config.q_lora_rank, self.num_heads * self.head_dim)
        self.wkv = _RefLinear(config.hidden_size, self.head_dim)
        self.kv_norm = _RefRMSNorm(self.head_dim, self.eps)
        self.wo_a = _RefLinear(
            self.num_heads * self.head_dim // self.num_groups,
            self.num_groups * self.o_lora_rank,
        )
        self.wo_b = _RefLinear(self.num_groups * self.o_lora_rank, config.hidden_size)
        if self.compress_ratio:
            self.compressor = _RefCompressor(config, self.compress_ratio, self.head_dim)
            self.rope_original_seq_len = config.rope_scaling["original_max_position_embeddings"]
            self.rope_base = config.compress_rope_theta
        else:
            self.rope_original_seq_len = 0
            self.rope_base = config.rope_theta
        self.rope_factor = config.rope_scaling["factor"]
        self.rope_beta_fast = config.rope_scaling["beta_fast"]
        self.rope_beta_slow = config.rope_scaling["beta_slow"]

    def _freqs_cis(self, position_ids):
        return _ref_freqs_cis(
            position_ids,
            self.rope_head_dim,
            self.rope_original_seq_len,
            self.rope_base,
            self.rope_factor,
            self.rope_beta_fast,
            self.rope_beta_slow,
        )

    def forward(self, x, position_ids):
        batch_size, seq_len, _ = x.shape
        freqs_cis = self._freqs_cis(position_ids)
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q = torch.cat(
            [
                q[..., : -self.rope_head_dim],
                _ref_apply_rope(q[..., -self.rope_head_dim :], freqs_cis),
            ],
            dim=-1,
        )
        kv = self.kv_norm(self.wkv(x))
        kv = torch.cat(
            [
                kv[..., : -self.rope_head_dim],
                _ref_apply_rope(kv[..., -self.rope_head_dim :], freqs_cis),
            ],
            dim=-1,
        )
        topk_idxs = _ref_window_topk(self.window_size, batch_size, seq_len, x.device)
        if self.compress_ratio:
            compressed = self.compressor(x, freqs_cis)
            compressed_idxs = _ref_compress_topk(
                self.compress_ratio, batch_size, seq_len, seq_len, x.device
            )
            topk_idxs = torch.cat([topk_idxs, compressed_idxs], dim=-1)
            kv = torch.cat([kv, compressed], dim=1)
        o = _ref_sparse_attention(q, kv, self.attn_sink, topk_idxs.int(), self.softmax_scale)
        o = torch.cat(
            [
                o[..., : -self.rope_head_dim],
                _ref_apply_rope(o[..., -self.rope_head_dim :], freqs_cis, inverse=True),
            ],
            dim=-1,
        )
        o = o.view(batch_size, seq_len, self.num_groups, -1)
        wo_a = self.wo_a.weight.view(self.num_groups, self.o_lora_rank, -1)
        return self.wo_b(torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2))


class _RefGate(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.score_func = config.scoring_func
        self.route_scale = config.routed_scaling_factor
        self.is_hash = layer_idx < config.num_hash_layers
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        if self.is_hash:
            initial = torch.arange(self.topk, dtype=torch.int32) % config.n_routed_experts
            self.tid2eid = nn.Parameter(
                initial.unsqueeze(0).expand(config.vocab_size, -1).clone(), requires_grad=False
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.zeros(config.n_routed_experts))

    def forward(self, x, input_ids):
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        indices = self.tid2eid[input_ids] if self.is_hash else scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights * self.route_scale, indices


class _RefExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, swiglu_limit=0.0):
        super().__init__()
        self.w1 = _RefLinear(hidden_size, intermediate_size)
        self.w2 = _RefLinear(intermediate_size, hidden_size)
        self.w3 = _RefLinear(hidden_size, intermediate_size)
        self.swiglu_limit = swiglu_limit

    def forward(self, x):
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        return self.w2((F.silu(gate) * up).to(x.dtype))


class _RefMoE(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = _RefGate(config, layer_idx)
        self.experts = nn.ModuleList(
            [
                _RefExpert(config.hidden_size, config.moe_intermediate_size, config.swiglu_limit)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.shared_experts = _RefExpert(config.hidden_size, config.moe_intermediate_size, 0.0)

    def forward(self, x, input_ids):
        shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        weights, indices = self.gate(x_flat, input_ids.flatten())
        output = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            expert_weight = torch.where(
                indices == expert_idx, weights, torch.zeros_like(weights)
            ).sum(dim=-1, keepdim=True)
            output = output + expert(x_flat) * expert_weight.to(x_flat.dtype)
        output = output + self.shared_experts(x_flat)
        return output.view(shape).to(x.dtype)


class _RefBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.attn = _RefAttention(config, layer_idx)
        self.ffn = _RefMoE(config, layer_idx)
        self.attn_norm = _RefRMSNorm(config.hidden_size, self.norm_eps)
        self.ffn_norm = _RefRMSNorm(config.hidden_size, self.norm_eps)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def _hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape = x.shape
        flat = x.flatten(2).float()
        mixes = F.linear(flat, hc_fn) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.norm_eps
        )
        pre, post, comb = _ref_hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        return torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2).to(x.dtype), post, comb

    @staticmethod
    def _hc_post(x, residual, post, comb):
        return (
            post.unsqueeze(-1) * x.unsqueeze(-2)
            + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        ).to(x.dtype)

    def forward(self, x, position_ids, input_ids):
        residual = x
        x, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self._hc_post(self.attn(self.attn_norm(x), position_ids), residual, post, comb)
        residual = x
        x, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        return self._hc_post(self.ffn(self.ffn_norm(x), input_ids), residual, post, comb)


class _RefHead(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    def forward(self, x):
        return F.linear(x.float(), self.weight.float()).to(x.dtype)


@dataclass
class _RefOutput:
    logits: torch.Tensor


class _RefForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [_RefBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = _RefRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = _RefHead(config.vocab_size, config.hidden_size)
        self.hc_mult = config.hc_mult
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))

    def _hc_head(self, x):
        shape = x.shape
        flat = x.flatten(2).float()
        mixes = F.linear(flat, self.hc_head_fn) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
        )
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
        return torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2).to(x.dtype)

    def forward(self, input_ids, position_ids):
        hidden_states = self.embed(input_ids).unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, input_ids)
        hidden_states = self.norm(self._hc_head(hidden_states))
        return _RefOutput(logits=self.head(hidden_states))


def _copy_to_reference(ad_module: nn.Module, ref_module: nn.Module) -> None:
    ref_module.load_state_dict(ad_module.state_dict(), strict=True)


def test_stacked_moe_reference_weights_convert_to_per_expert_keys():
    config = _small_config(n_routed_experts=3)
    stacked = {
        "layers.0.ffn.experts.w1.weight": torch.randn(
            config.n_routed_experts, config.moe_intermediate_size, config.hidden_size
        ),
        "layers.0.ffn.experts.w2.weight": torch.randn(
            config.n_routed_experts, config.hidden_size, config.moe_intermediate_size
        ),
        "layers.0.ffn.experts.w3.weight": torch.randn(
            config.n_routed_experts, config.moe_intermediate_size, config.hidden_size
        ),
        "layers.0.ffn.gate.weight": torch.randn(config.n_routed_experts, config.hidden_size),
    }

    converted = _convert_deepseek_v4_reference_state_dict(stacked, config.n_routed_experts)

    assert "layers.0.ffn.experts.w1.weight" not in converted
    assert "layers.0.ffn.experts.2.w3.weight" in converted
    torch.testing.assert_close(
        converted["layers.0.ffn.experts.1.w2.weight"],
        stacked["layers.0.ffn.experts.w2.weight"][1],
    )
    torch.testing.assert_close(
        converted["layers.0.ffn.gate.weight"], stacked["layers.0.ffn.gate.weight"]
    )


def test_state_dict_matches_deepseek_v4_checkpoint_hierarchy():
    config = _small_config(num_hidden_layers=2, compress_ratios=(0, 4), num_hash_layers=1)
    model = DeepseekV4ForCausalLM(config)
    keys = set(model.state_dict())

    # Sampled from deepseek-ai/DeepSeek-V4-Flash model.safetensors.index.json.
    # The reference checkpoint stores routed MoE as per-expert w1/w2/w3 keys, so
    # no stacked-MoE conversion helper is needed for these names.
    missing = _DEEPSEEK_V4_CHECKPOINT_SAMPLE_KEYS - keys
    assert not missing
    assert "layers.1.ffn.gate.bias" in keys
    assert "layers.1.ffn.gate.tid2eid" not in keys
    assert not any(key.startswith("model.") for key in keys)


def test_rmsnorm_matches_reference():
    config = _small_config()
    ad = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
    x = torch.randn(2, 5, config.hidden_size)
    expected = _ref_rmsnorm(x, ad.weight, config.rms_norm_eps)
    torch.testing.assert_close(ad(x), expected, rtol=1e-6, atol=1e-6)


def test_attention_matches_reference():
    torch.manual_seed(1)
    config = _small_config(num_hidden_layers=2, compress_ratios=(0, 4))
    layer_idx = 1
    ad = DeepseekV4ForCausalLM(config).layers[layer_idx].attn.eval()
    ref = _RefAttention(config, layer_idx).eval()
    _copy_to_reference(ad, ref)
    x = torch.randn(2, 8, config.hidden_size)
    pos = _position_ids(2, 8, x.device)
    with torch.no_grad():
        actual = ad(x, pos)
        expected = ref(x, pos)
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.01, msg="Attention: ")


def test_moe_with_swiglu_limit_matches_reference():
    torch.manual_seed(2)
    config = _small_config(swiglu_limit=2.0)
    layer_idx = 1
    ad = DeepseekV4ForCausalLM(config).layers[layer_idx].ffn.eval()
    ref = _RefMoE(config, layer_idx).eval()
    _copy_to_reference(ad, ref)
    x = torch.randn(2, 4, config.hidden_size)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))
    with torch.no_grad():
        actual = ad(x, input_ids)
        expected = ref(x, input_ids)
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.01, msg="MoE: ")


def test_block_matches_reference():
    torch.manual_seed(3)
    config = _small_config(num_hidden_layers=1, compress_ratios=(0,))
    ad = DeepseekV4ForCausalLM(config).layers[0].eval()
    ref = _RefBlock(config, 0).eval()
    _copy_to_reference(ad, ref)
    x = torch.randn(2, 6, config.hc_mult, config.hidden_size)
    pos = _position_ids(2, 6, x.device)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    with torch.no_grad():
        actual = ad(x, pos, input_ids)
        expected = ref(x, pos, input_ids)
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.05, msg="Decoder layer: ")


def test_full_model_matches_reference():
    torch.manual_seed(4)
    config = _small_config()
    ad = DeepseekV4ForCausalLM(config).eval()
    ref = _RefForCausalLM(config).eval()
    _copy_to_reference(ad, ref)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    pos = _position_ids(2, 8, input_ids.device)
    with torch.no_grad():
        actual = ad(input_ids=input_ids, position_ids=pos).logits
        expected = ref(input_ids, pos).logits
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.05, msg="Full model: ")


def test_export_produces_finite_logits_for_second_shape():
    torch.manual_seed(5)
    config = _small_config(num_hidden_layers=1, compress_ratios=(4,), num_hash_layers=0)
    model = DeepseekV4ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    pos = _position_ids(2, 6, input_ids.device)
    dynamic_shapes = {
        "input_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        "position_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    }
    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": pos},
        dynamic_shapes=dynamic_shapes,
        num_moe_experts_for_export=2,
    )
    with torch.no_grad():
        eager = model(input_ids=input_ids, position_ids=pos).logits
        exported = gm(input_ids, position_ids=pos)
    exported_logits = (
        exported[0] if isinstance(exported, tuple) else getattr(exported, "logits", exported)
    )
    assert_rmse_close(exported_logits, eager, rmse_ratio_tol=0.05, msg="Export: ")

    ids2 = torch.randint(0, config.vocab_size, (1, 4))
    pos2 = _position_ids(1, 4, ids2.device)
    with torch.no_grad():
        eager2 = model(input_ids=ids2, position_ids=pos2).logits
        out = gm(ids2, position_ids=pos2)
    logits = out[0] if isinstance(out, tuple) else getattr(out, "logits", out)
    assert logits.shape == (1, 4, config.vocab_size)
    assert torch.isfinite(logits).all()
    assert_rmse_close(logits, eager2, rmse_ratio_tol=0.05, msg="Export second shape: ")
