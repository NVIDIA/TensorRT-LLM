# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the DeepSeek V4 custom AutoDeploy model.

DeepSeek V4 is not in transformers (4.57.3), so the reference math here is
copied verbatim (and minimally) from the HF model.py
(https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py).
The reference is simplified to the prefill-only path that the AD custom model
implements: sliding-window attention with learnable attention sinks, plus
dedicated compressor/indexer references for compressed-KV sparse attention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch import nn
from torch.export import Dim
from torch.fx import Node

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Block,
    DeepseekV4Compressor,
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    DeepseekV4Indexer,
    DeepseekV4MLP,
    DeepseekV4MoE,
    DeepseekV4RMSNorm,
    _build_freqs_cis,
    _build_sparse_attn_mask,
    _fake_fp4_act_quant,
    _fake_fp8_act_quant,
    _hadamard_rotate,
    _hc_split_sinkhorn,
    _remap_deepseek_v4_checkpoint_keys,
    _window_topk_idxs,
)


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


_FLASH_BASE_INDEX_FIXTURE = (
    Path(__file__).with_name("fixtures") / "deepseek_v4_flash_base_safetensors_index_sample.json"
)
_EXPECTED_FLASH_BASE_INDEX_KEYS = {
    "embed.weight",
    "head.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "layers.0.attn.wq_a.weight",
    "layers.0.attn.wq_a.scale",
    "layers.0.ffn.experts.0.w1.weight",
    "layers.0.ffn.experts.0.w1.scale",
    "layers.0.ffn.experts.0.w2.weight",
    "layers.0.ffn.experts.0.w3.weight",
    "layers.0.ffn.gate.tid2eid",
    "layers.0.ffn.shared_experts.w1.weight",
    "layers.0.ffn.shared_experts.w1.scale",
    "layers.0.ffn.shared_experts.w2.weight",
    "layers.0.ffn.shared_experts.w3.weight",
    "layers.2.attn.compressor.wkv.weight",
    "layers.2.attn.compressor.wkv.scale",
    "layers.2.attn.indexer.wq_b.weight",
    "layers.2.attn.indexer.wq_b.scale",
    "norm.weight",
}


def _load_flash_base_index_keys() -> set[str]:
    with open(_FLASH_BASE_INDEX_FIXTURE) as f:
        index = json.load(f)
    assert index["metadata"]["source_model"] == "deepseek-ai/DeepSeek-V4-Flash-Base"
    assert index["metadata"]["source_revision"] == "e133377a559d13c857cf0c054bb46247076102a0"
    assert all(name.endswith(".safetensors") for name in index["weight_map"].values())
    return set(index["weight_map"])


# =============================================================================
# Reference (faithful copy of the subset of HF model.py we implement)
# =============================================================================


class _RefRMSNorm(nn.Module):
    """Matches RMSNorm in model.py (weight kept in fp32 intentionally)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def _freqs_cis_from_cos_sin(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reassemble the complex freqs_cis the reference math expects from real cos/sin."""
    return torch.complex(cos, sin)


def _apply_rotary_emb_ref(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Copy of apply_rotary_emb from model.py (out-of-place)."""
    x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_c.ndim == 3:
        fc = freqs_cis.view(1, x_c.size(1), x_c.size(-1))
    else:
        fc = freqs_cis.view(1, x_c.size(1), 1, x_c.size(-1))
    return torch.view_as_real(x_c * fc).flatten(-2).type_as(x)


class _RefExpert(nn.Module):
    """Matches Expert in model.py (fp32 gate/up, optional swiglu_limit clamp)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        out = F.silu(gate) * up
        if weights is not None:
            out = weights * out
        return self.w2(out.to(dtype))


class _RefGate(nn.Module):
    """Score-routed gate only (no hash routing) from model.py."""

    def __init__(
        self,
        n_experts: int,
        dim: int,
        topk: int,
        score_func: str = "sqrtsoftplus",
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.topk = topk
        self.score_func = score_func
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        self.bias = nn.Parameter(torch.zeros(n_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        scores = scores + self.bias
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


def _sliding_window_sink_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    window_size: int,
) -> torch.Tensor:
    """Pure PyTorch reference for sparse_attn when topk_idxs only covers the window.

    For seqlen <= window_size, this is equivalent to standard causal SDPA + sinks.
    q: [B, S, H, D], kv: [B, S, D] (K == V, single KV head).
    """
    bsz, seqlen, n_heads, d = q.shape
    # Expand kv to [B, S, H, D] by broadcasting over heads (MQA -> MHA).
    k = kv.unsqueeze(2).expand(bsz, seqlen, n_heads, d)
    v = k
    # Transpose to [B, H, S, D] to compute attention manually.
    qb = q.transpose(1, 2)
    kb = k.transpose(1, 2)
    vb = v.transpose(1, 2)

    scores = torch.matmul(qb, kb.transpose(-2, -1)) * softmax_scale  # [B, H, Sq, Sk]
    # Causal mask
    s_q = seqlen
    s_k = seqlen
    causal = torch.triu(torch.ones(s_q, s_k, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
    # Sliding-window mask
    idx_q = torch.arange(s_q, device=q.device).unsqueeze(1)
    idx_k = torch.arange(s_k, device=q.device).unsqueeze(0)
    pos_diff = idx_q - idx_k
    sw_mask = (pos_diff < 0) | (pos_diff >= window_size)
    scores = scores.masked_fill(sw_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    # Softmax with sinks (extra virtual key column per head).
    sinks_expanded = attn_sink.reshape(1, -1, 1, 1).expand(bsz, n_heads, s_q, 1)
    logits_max = torch.max(scores, dim=-1, keepdim=True).values
    sinks = torch.exp(sinks_expanded - logits_max)
    unn = torch.exp(scores - logits_max)
    denom = unn.sum(dim=-1, keepdim=True) + sinks
    attn = unn / denom
    out = torch.matmul(attn, vb)  # [B, H, S, D]
    return out.transpose(1, 2).contiguous()  # [B, S, H, D]


def _ref_compressor_forward(
    compressor: DeepseekV4Compressor,
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Faithful prefill ``start_pos == 0`` reference for HF Compressor."""
    bsz, seqlen, _ = hidden_states.shape
    ratio = compressor.compress_ratio
    n_compressed = seqlen // ratio
    if n_compressed == 0:
        return hidden_states.new_empty(bsz, 0, compressor.head_dim)

    cutoff = n_compressed * ratio
    dtype = hidden_states.dtype
    x = hidden_states.float()
    kv = F.linear(x, compressor.wkv.weight.float())[:, :cutoff]
    score = F.linear(x, compressor.wgate.weight.float())[:, :cutoff]
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + compressor.ape
    if compressor.overlap:
        b, chunks, _, _ = kv.shape
        kv_out = kv.new_zeros(b, chunks, 2 * ratio, compressor.head_dim)
        score_out = score.new_full((b, chunks, 2 * ratio, compressor.head_dim), float("-inf"))
        kv_out[:, :, ratio:] = kv[:, :, :, compressor.head_dim :]
        kv_out[:, 1:, :ratio] = kv[:, :-1, :, : compressor.head_dim]
        score_out[:, :, ratio:] = score[:, :, :, compressor.head_dim :]
        score_out[:, 1:, :ratio] = score[:, :-1, :, : compressor.head_dim]
        kv, score = kv_out, score_out
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
    kv = compressor.norm(kv.to(dtype))
    rd = compressor.rope_head_dim
    rotated = _apply_rotary_emb_ref(kv[..., -rd:], freqs_cis[:cutoff:ratio])
    kv = torch.cat([kv[..., :-rd], rotated], dim=-1)
    if compressor.rotate:
        return _fake_fp4_act_quant(_hadamard_rotate(kv), block_size=32)

    nope, pe = torch.split(kv, [compressor.head_dim - rd, rd], dim=-1)
    return torch.cat([_fake_fp8_act_quant(nope, block_size=64), pe], dim=-1)


def _ref_indexer_forward(
    indexer: DeepseekV4Indexer,
    hidden_states: torch.Tensor,
    q_lora: torch.Tensor,
    freqs_cis: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    """Faithful prefill ``start_pos == 0`` reference for HF Indexer."""
    bsz, seqlen, _ = hidden_states.shape
    ratio = indexer.compress_ratio
    n_compressed = seqlen // ratio
    if n_compressed == 0:
        return hidden_states.new_empty(bsz, seqlen, 0, dtype=torch.long)

    rd = indexer.rope_head_dim
    q = F.linear(q_lora, indexer.wq_b.weight).view(
        bsz, seqlen, indexer.index_n_heads, indexer.index_head_dim
    )
    q_rot = _apply_rotary_emb_ref(q[..., -rd:], freqs_cis)
    q = torch.cat([q[..., :-rd], q_rot], dim=-1)
    q = _fake_fp4_act_quant(_hadamard_rotate(q), block_size=32)
    index_k = _ref_compressor_forward(indexer.compressor, hidden_states, freqs_cis)
    weights = F.linear(hidden_states, indexer.weights_proj.weight)
    weights = weights * (indexer.softmax_scale * indexer.index_n_heads**-0.5)

    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
    compressed_positions = torch.arange(n_compressed, device=hidden_states.device)
    valid_count = torch.arange(1, seqlen + 1, device=hidden_states.device).unsqueeze(1) // ratio
    mask = compressed_positions.unsqueeze(0) >= valid_count
    index_score = index_score.masked_fill(mask.unsqueeze(0), float("-inf"))
    topk = min(indexer.index_topk, n_compressed)
    topk_idxs = index_score.topk(topk, dim=-1).indices
    invalid = topk_idxs >= valid_count.unsqueeze(0)
    return torch.where(invalid, -1, topk_idxs + offset)


def _sparse_sink_attn_from_indices(
    q: torch.Tensor,
    kv_all: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference sparse attention from explicit HF top-k indices."""
    bsz, seqlen, n_heads, d = q.shape
    total_kv = kv_all.shape[1]
    k = kv_all.expand(bsz, total_kv, n_heads, d)
    v = k
    qb = q.transpose(1, 2)
    kb = k.transpose(1, 2)
    vb = v.transpose(1, 2)
    scores = torch.matmul(qb, kb.transpose(-2, -1)) * softmax_scale
    attn_mask = _build_sparse_attn_mask(topk_idxs, total_kv).to(scores.dtype)
    scores = scores + attn_mask

    sinks_expanded = attn_sink.reshape(1, -1, 1, 1).expand(bsz, n_heads, seqlen, 1)
    logits_max = torch.max(scores, dim=-1, keepdim=True).values
    sinks = torch.exp(sinks_expanded - logits_max)
    unn = torch.exp(scores - logits_max)
    attn = unn / (unn.sum(dim=-1, keepdim=True) + sinks)
    out = torch.matmul(attn, vb)
    return out.transpose(1, 2).contiguous()


class _RefAttention(nn.Module):
    """Prefill-only MLA-variant attention matching the AD custom model's path.

    Uses the reference (non-kernel) math for everything: interleaved complex RoPE,
    parameterless per-head RMS on Q, single-head KV, sliding window + sinks,
    inverse RoPE on output, grouped low-rank O projection.
    """

    def __init__(
        self,
        hidden: int,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        q_lora_rank: int,
        o_lora_rank: int,
        n_groups: int,
        window_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.n_groups = n_groups
        self.window_size = window_size
        self.eps = eps
        self.softmax_scale = head_dim**-0.5
        self.o_lora_rank = o_lora_rank
        self.group_head_width = n_heads * head_dim // n_groups

        self.wq_a = nn.Linear(hidden, q_lora_rank, bias=False)
        self.q_norm = _RefRMSNorm(q_lora_rank, eps=eps)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.wkv = nn.Linear(hidden, head_dim, bias=False)
        self.kv_norm = _RefRMSNorm(head_dim, eps=eps)
        self.wo_a = nn.Linear(self.group_head_width, n_groups * o_lora_rank, bias=False)
        self.wo_b = nn.Linear(n_groups * o_lora_rank, hidden, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(n_heads, dtype=torch.float32))

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        rd = self.rope_head_dim
        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.eps).to(q.dtype)
        kv = self.kv_norm(self.wkv(x))  # [B, S, head_dim]
        # RoPE on last rd dims of q and kv (in-place semantics — here we do out-of-place)
        q_rope_part = _apply_rotary_emb_ref(q[..., -rd:], freqs_cis)
        kv_rope_part = _apply_rotary_emb_ref(kv[..., -rd:], freqs_cis)
        q = torch.cat([q[..., :-rd], q_rope_part], dim=-1)
        kv = torch.cat([_fake_fp8_act_quant(kv[..., :-rd], block_size=64), kv_rope_part], dim=-1)

        o = _sliding_window_sink_attn(q, kv, self.attn_sink, self.softmax_scale, self.window_size)
        # Inverse RoPE on output's last rd dims
        o_rope_part = _apply_rotary_emb_ref(o[..., -rd:], freqs_cis, inverse=True)
        o = torch.cat([o[..., :-rd], o_rope_part], dim=-1)

        # Grouped low-rank O projection
        o = o.reshape(bsz, seqlen, self.n_groups, self.group_head_width)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, self.group_head_width)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.reshape(bsz, seqlen, self.n_groups * self.o_lora_rank))


class _RefMoE(nn.Module):
    """Reference MoE: score-routed gate + per-expert manual dispatch + shared expert."""

    def __init__(self, cfg: DeepseekV4Config):
        super().__init__()
        self.topk = cfg.num_experts_per_tok
        self.hidden = cfg.hidden_size
        self.gate = _RefGate(
            cfg.n_routed_experts,
            cfg.hidden_size,
            cfg.num_experts_per_tok,
            score_func="sqrtsoftplus",
            route_scale=cfg.routed_scaling_factor,
        )
        self.experts = nn.ModuleList(
            [
                _RefExpert(
                    cfg.hidden_size, cfg.moe_intermediate_size, swiglu_limit=cfg.swiglu_limit
                )
                for _ in range(cfg.n_routed_experts)
            ]
        )
        self.shared = _RefExpert(
            cfg.hidden_size, cfg.moe_intermediate_size * cfg.n_shared_experts, swiglu_limit=0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat = x.view(-1, self.hidden)
        weights, indices = self.gate(flat)
        y = torch.zeros_like(flat)
        for tok in range(flat.size(0)):
            for k in range(self.topk):
                e = int(indices[tok, k].item())
                w = weights[tok, k].unsqueeze(0)
                y[tok] += self.experts[e](flat[tok : tok + 1], w).squeeze(0)
        return y.view(*orig_shape) + self.shared(x)


class _RefBlock(nn.Module):
    """Reference HC block matching DeepseekV4Block's forward pattern."""

    def __init__(self, cfg: DeepseekV4Config):
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.attn_norm = _RefRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.ffn_norm = _RefRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = _RefAttention(
            hidden=cfg.hidden_size,
            n_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
            rope_head_dim=cfg.qk_rope_head_dim,
            q_lora_rank=cfg.q_lora_rank,
            o_lora_rank=cfg.o_lora_rank,
            n_groups=cfg.o_groups,
            window_size=cfg.sliding_window,
            eps=cfg.rms_norm_eps,
        )
        self.ffn = _RefMoE(cfg)

    def _hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.shape, x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype), post, comb

    def _hc_post(self, x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        residual = x
        h, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.attn_norm(h)
        h = self.attn(h, freqs_cis)
        x = self._hc_post(h, residual, post, comb)
        residual = x
        h, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.ffn_norm(h)
        h = self.ffn(h)
        return self._hc_post(h, residual, post, comb)


class _RefModel(nn.Module):
    """Reference full model mirroring DeepseekV4ForCausalLM for compress_ratios=0, seqlen<=win."""

    def __init__(self, cfg: DeepseekV4Config):
        super().__init__()
        self.cfg = cfg
        self.hc_mult = cfg.hc_mult
        self.norm_eps = cfg.rms_norm_eps
        self.hc_eps = cfg.hc_eps
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.layers = nn.ModuleList([_RefBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        hc_dim = cfg.hc_mult * cfg.hidden_size
        self.hc_head_fn = nn.Parameter(torch.zeros(cfg.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.zeros(cfg.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.norm = _RefRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def _hc_head_collapse(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.shape, x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        return torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2).to(dtype)

    def forward(self, input_ids: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        h = h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        for layer in self.layers:
            h = layer(h, freqs_cis)
        h = self._hc_head_collapse(h)
        h = self.norm(h)
        return self.lm_head(h).float()


def _copy_attention_weights(dst: _RefAttention, src: DeepseekV4Attention) -> None:
    dst.wq_a.weight.data.copy_(src.wq_a.weight.data)
    dst.q_norm.weight.data.copy_(src.q_norm.weight.data.float())
    dst.wq_b.weight.data.copy_(src.wq_b.weight.data)
    dst.wkv.weight.data.copy_(src.wkv.weight.data)
    dst.kv_norm.weight.data.copy_(src.kv_norm.weight.data.float())
    dst.wo_a.weight.data.copy_(src.wo_a.weight.data)
    dst.wo_b.weight.data.copy_(src.wo_b.weight.data)
    dst.attn_sink.data.copy_(src.attn_sink.data)


def _copy_moe_weights(dst: _RefMoE, src: DeepseekV4MoE) -> None:
    dst.gate.weight.data.copy_(src.gate.weight.data.float())
    dst.gate.bias.data.copy_(src.gate.bias.data)
    for i, e in enumerate(src.experts):
        dst.experts[i].w1.weight.data.copy_(e.gate_proj.weight.data)
        dst.experts[i].w3.weight.data.copy_(e.up_proj.weight.data)
        dst.experts[i].w2.weight.data.copy_(e.down_proj.weight.data)
    dst.shared.w1.weight.data.copy_(src.shared_experts.gate_proj.weight.data)
    dst.shared.w3.weight.data.copy_(src.shared_experts.up_proj.weight.data)
    dst.shared.w2.weight.data.copy_(src.shared_experts.down_proj.weight.data)


def _copy_block_weights(dst: _RefBlock, src: DeepseekV4Block) -> None:
    dst.hc_attn_fn.data.copy_(src.hc_attn_fn.data)
    dst.hc_ffn_fn.data.copy_(src.hc_ffn_fn.data)
    dst.hc_attn_base.data.copy_(src.hc_attn_base.data)
    dst.hc_ffn_base.data.copy_(src.hc_ffn_base.data)
    dst.hc_attn_scale.data.copy_(src.hc_attn_scale.data)
    dst.hc_ffn_scale.data.copy_(src.hc_ffn_scale.data)
    dst.attn_norm.weight.data.copy_(src.attn_norm.weight.data.float())
    dst.ffn_norm.weight.data.copy_(src.ffn_norm.weight.data.float())
    _copy_attention_weights(dst.attn, src.attn)
    _copy_moe_weights(dst.ffn, src.ffn)


# =============================================================================
# Small config
# =============================================================================


def _small_config(num_hash_layers: int = 0) -> DeepseekV4Config:
    # compress_ratios all 0 so non-YaRN RoPE is used (makes test seqlens easier).
    return DeepseekV4Config(
        vocab_size=200,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        sliding_window=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=num_hash_layers,
        moe_intermediate_size=32,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        max_position_embeddings=64,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        rope_scaling=None,
        compress_ratios=[0, 0, 0],
        swiglu_limit=0.0,  # disable clamp for faithful RMSE comparison
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


def _small_compress_config(ratio: int = 4) -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=200,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        sliding_window=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=0,
        moe_intermediate_size=32,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        max_position_embeddings=64,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        rope_scaling=None,
        compress_ratios=[ratio],
        index_n_heads=4,
        index_head_dim=16,
        index_topk=2,
        swiglu_limit=0.0,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )


# =============================================================================
# Block equivalence tests
# =============================================================================


def test_rmsnorm_equivalence():
    dim = 32
    hidden = torch.randn(2, 8, dim, dtype=torch.float32)
    custom = DeepseekV4RMSNorm(dim, eps=1e-6)
    ref = _RefRMSNorm(dim, eps=1e-6)
    ref.weight.data.copy_(custom.weight.data.float())
    y_custom = custom(hidden)
    y_ref = ref(hidden)
    torch.testing.assert_close(y_custom, y_ref, rtol=1e-3, atol=1e-3)


def test_mlp_equivalence():
    dim, inter = 32, 64
    custom = DeepseekV4MLP(dim, inter, swiglu_limit=0.0)
    ref = _RefExpert(dim, inter, swiglu_limit=0.0)
    ref.w1.weight.data.copy_(custom.gate_proj.weight.data)
    ref.w3.weight.data.copy_(custom.up_proj.weight.data)
    ref.w2.weight.data.copy_(custom.down_proj.weight.data)
    x = torch.randn(2, 8, dim)
    torch.testing.assert_close(custom(x), ref(x), rtol=1e-3, atol=1e-3)


def test_moe_gate_equivalence_score_routing():
    """Score-routed gate (non-hash layers) — sqrtsoftplus + bias-shifted topk."""
    cfg = _small_config()
    gate = _RefGate(
        cfg.n_routed_experts,
        cfg.hidden_size,
        cfg.num_experts_per_tok,
        score_func="sqrtsoftplus",
        route_scale=cfg.routed_scaling_factor,
    )
    gate.weight.data.normal_(mean=0.0, std=0.02)
    gate.bias.data.copy_(torch.linspace(-0.01, 0.01, cfg.n_routed_experts))
    # Layer index >= num_hash_layers ensures score routing.
    from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import DeepseekV4MoEGate

    custom = DeepseekV4MoEGate(cfg, layer_idx=cfg.num_hash_layers)
    custom.weight.data.copy_(gate.weight.data.float())
    custom.bias.data.copy_(gate.bias.data)

    x = torch.randn(2, cfg.hidden_size)
    selected_custom, weights_custom = custom(x, torch.zeros(2, dtype=torch.long))
    weights_ref, indices_ref = gate(x)

    torch.testing.assert_close(selected_custom, indices_ref.long())
    torch.testing.assert_close(weights_custom, weights_ref, rtol=1e-4, atol=1e-4)


def test_moe_equivalence():
    """Full MoE block: gate + routed experts via torch_moe + shared expert."""
    cfg = _small_config()
    cfg.swiglu_limit = 0.5
    moe = DeepseekV4MoE(cfg, layer_idx=cfg.num_hash_layers)
    # Randomize gate weights (they're zero-init by default isn't the case here but
    # DeepseekV4MoEGate uses torch.empty via nn.Parameter directly; in practice _init_weights
    # covers Linear and Embedding, so the raw gate weight Parameter can be noise. Reset it.)
    moe.gate.weight.data.normal_()
    moe.gate.bias.data.zero_()
    moe.eval()

    # Build a matching reference: score-routed top-k to select experts, then apply
    # the same per-expert MLP and sum weighted contributions.
    ref_gate = _RefGate(
        cfg.n_routed_experts,
        cfg.hidden_size,
        cfg.num_experts_per_tok,
        score_func="sqrtsoftplus",
        route_scale=cfg.routed_scaling_factor,
    )
    ref_gate.weight.data.copy_(moe.gate.weight.data.float())
    ref_gate.bias.data.copy_(moe.gate.bias.data)

    ref_experts = nn.ModuleList(
        [
            _RefExpert(cfg.hidden_size, cfg.moe_intermediate_size, swiglu_limit=cfg.swiglu_limit)
            for _ in range(cfg.n_routed_experts)
        ]
    )
    for i, e in enumerate(moe.experts):
        ref_experts[i].w1.weight.data.copy_(e.gate_proj.weight.data)
        ref_experts[i].w3.weight.data.copy_(e.up_proj.weight.data)
        ref_experts[i].w2.weight.data.copy_(e.down_proj.weight.data)

    ref_shared = _RefExpert(cfg.hidden_size, cfg.moe_intermediate_size * cfg.n_shared_experts)
    ref_shared.w1.weight.data.copy_(moe.shared_experts.gate_proj.weight.data)
    ref_shared.w3.weight.data.copy_(moe.shared_experts.up_proj.weight.data)
    ref_shared.w2.weight.data.copy_(moe.shared_experts.down_proj.weight.data)

    B, S, H = 2, 8, cfg.hidden_size
    x = torch.randn(B, S, H) * 4.0
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    y_custom = moe(x, input_ids)

    # Reference: gate -> per-token per-expert gather
    flat = x.view(-1, H)
    weights_ref, indices_ref = ref_gate(flat)
    y_ref = torch.zeros_like(flat)
    for token_idx in range(flat.size(0)):
        for k in range(cfg.num_experts_per_tok):
            e_idx = int(indices_ref[token_idx, k].item())
            w = weights_ref[token_idx, k].unsqueeze(0)
            y_ref[token_idx] += ref_experts[e_idx](flat[token_idx : token_idx + 1], w).squeeze(0)
    y_ref = y_ref.view(B, S, H) + ref_shared(x)

    assert_rmse_close(y_custom, y_ref, rmse_ratio_tol=0.02, msg="MoE equivalence: ")


def test_attention_equivalence():
    """Attention with compress_ratio=0 and seqlen <= sliding_window — reference match."""
    cfg = _small_config()
    layer_idx = cfg.num_hash_layers  # doesn't matter for attention
    custom = DeepseekV4Attention(cfg, layer_idx=layer_idx)
    custom.eval()
    ref = _RefAttention(
        hidden=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        q_lora_rank=cfg.q_lora_rank,
        o_lora_rank=cfg.o_lora_rank,
        n_groups=cfg.o_groups,
        window_size=cfg.sliding_window,
        eps=cfg.rms_norm_eps,
    )
    ref.wq_a.weight.data.copy_(custom.wq_a.weight.data)
    ref.q_norm.weight.data.copy_(custom.q_norm.weight.data.float())
    ref.wq_b.weight.data.copy_(custom.wq_b.weight.data)
    ref.wkv.weight.data.copy_(custom.wkv.weight.data)
    ref.kv_norm.weight.data.copy_(custom.kv_norm.weight.data.float())
    ref.wo_a.weight.data.copy_(custom.wo_a.weight.data)
    ref.wo_b.weight.data.copy_(custom.wo_b.weight.data)
    custom.attn_sink.data.normal_(std=0.1)
    ref.attn_sink.data.copy_(custom.attn_sink.data)

    B, S = 2, 8  # S <= sliding_window
    x = torch.randn(B, S, cfg.hidden_size)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos_tbl, sin_tbl = _build_freqs_cis(
        cfg.qk_rope_head_dim, cfg.max_position_embeddings, cfg.rope_theta, 0, 1.0, 32, 1
    )
    cos = cos_tbl[position_ids]  # [B, S, rd/2]
    sin = sin_tbl[position_ids]
    freqs_cis_ref = _freqs_cis_from_cos_sin(cos[0], sin[0])  # [S, rd/2] for ref

    y_custom = custom(x, cos, sin, cos, sin, cos_tbl, sin_tbl)
    y_ref = ref(x, freqs_cis_ref)
    assert_rmse_close(y_custom, y_ref, rmse_ratio_tol=0.10, msg="Attention equivalence: ")


def test_compressor_prefill_equivalence():
    """Learned Compressor start_pos=0 path, including ratio-4 overlap."""
    cfg = _small_compress_config(ratio=4)
    compressor = DeepseekV4Compressor(cfg, compress_ratio=4, head_dim=cfg.head_dim).eval()
    compressor.ape.data.normal_(std=0.02)
    x = torch.randn(2, 8, cfg.hidden_size)
    cos_tbl, sin_tbl = _build_freqs_cis(
        cfg.qk_rope_head_dim, cfg.max_position_embeddings, cfg.compress_rope_theta, 0, 1.0, 32, 1
    )
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
    cos = cos_tbl[position_ids]
    sin = sin_tbl[position_ids]
    freqs_cis_ref = _freqs_cis_from_cos_sin(cos[0], sin[0])

    y_custom = compressor(x, cos, sin)
    y_ref = _ref_compressor_forward(compressor, x, freqs_cis_ref)
    torch.testing.assert_close(y_custom[:, : y_ref.shape[1]], y_ref, rtol=1e-3, atol=1e-3)
    assert torch.isfinite(y_custom).all()


def test_indexer_prefill_topk_equivalence():
    """Ratio-4 Indexer top-k compressed-token selection."""
    cfg = _small_compress_config(ratio=4)
    indexer = DeepseekV4Indexer(cfg, compress_ratio=4).eval()
    indexer.compressor.ape.data.normal_(std=0.02)
    x = torch.randn(2, 8, cfg.hidden_size)
    q_lora = torch.randn(2, 8, cfg.q_lora_rank)
    cos_tbl, sin_tbl = _build_freqs_cis(
        cfg.qk_rope_head_dim, cfg.max_position_embeddings, cfg.compress_rope_theta, 0, 1.0, 32, 1
    )
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
    cos = cos_tbl[position_ids]
    sin = sin_tbl[position_ids]
    freqs_cis_ref = _freqs_cis_from_cos_sin(cos[0], sin[0])

    topk_custom = indexer(x, q_lora, cos, sin, offset=8)
    topk_ref = _ref_indexer_forward(indexer, x, q_lora, freqs_cis_ref, offset=8)
    expected = torch.full_like(topk_custom, -1)
    expected[:, :, : topk_ref.shape[-1]] = topk_ref
    torch.testing.assert_close(topk_custom, expected)


def test_attention_compressed_prefill_equivalence():
    """Attention path with compressed KV concatenation and ratio-4 learned indexer."""
    cfg = _small_compress_config(ratio=4)
    custom = DeepseekV4Attention(cfg, layer_idx=0).eval()
    custom.compressor.ape.data.normal_(std=0.02)
    custom.indexer.compressor.ape.data.normal_(std=0.02)
    custom.attn_sink.data.normal_(std=0.1)

    B, S = 2, 8
    x = torch.randn(B, S, cfg.hidden_size)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos_tbl, sin_tbl = _build_freqs_cis(
        cfg.qk_rope_head_dim, cfg.max_position_embeddings, cfg.compress_rope_theta, 0, 1.0, 32, 1
    )
    cos = cos_tbl[position_ids]
    sin = sin_tbl[position_ids]
    freqs_cis_ref = _freqs_cis_from_cos_sin(cos[0], sin[0])

    y_custom = custom(x, cos, sin, cos, sin, cos_tbl, sin_tbl)

    rd = cfg.qk_rope_head_dim
    q_lora = custom.q_norm(custom.wq_a(x))
    q = custom.wq_b(q_lora).view(B, S, cfg.num_attention_heads, cfg.head_dim)
    q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + cfg.rms_norm_eps).to(q.dtype)
    q = torch.cat([q[..., :-rd], _apply_rotary_emb_ref(q[..., -rd:], freqs_cis_ref)], dim=-1)
    kv = custom.kv_norm(custom.wkv(x))
    kv = torch.cat(
        [
            _fake_fp8_act_quant(kv[..., :-rd], block_size=64),
            _apply_rotary_emb_ref(kv[..., -rd:], freqs_cis_ref),
        ],
        dim=-1,
    )
    kv = kv.view(B, S, 1, cfg.head_dim)
    kv_comp = _ref_compressor_forward(custom.compressor, x, freqs_cis_ref).view(
        B, -1, 1, cfg.head_dim
    )
    window_topk = _window_topk_idxs(cfg.sliding_window, B, S, x.device)
    compress_topk = _ref_indexer_forward(custom.indexer, x, q_lora, freqs_cis_ref, offset=S)
    topk_idxs = torch.cat([window_topk, compress_topk], dim=-1)
    kv_all = torch.cat([kv, kv_comp], dim=1)
    attn_ref = _sparse_sink_attn_from_indices(
        q, kv_all, custom.attn_sink, topk_idxs, custom.softmax_scale
    )
    attn_ref = torch.cat(
        [
            attn_ref[..., :-rd],
            _apply_rotary_emb_ref(attn_ref[..., -rd:], freqs_cis_ref, inverse=True),
        ],
        dim=-1,
    )
    attn_ref = attn_ref.reshape(B, S, cfg.o_groups, custom.group_head_width)
    wo_a = custom.wo_a.weight.view(cfg.o_groups, cfg.o_lora_rank, custom.group_head_width)
    attn_ref = torch.einsum("bsgd,grd->bsgr", attn_ref, wo_a)
    y_ref = custom.wo_b(attn_ref.reshape(B, S, cfg.o_groups * cfg.o_lora_rank))

    assert_rmse_close(y_custom, y_ref, rmse_ratio_tol=0.10, msg="Compressed attention: ")


def test_block_equivalence():
    """Full HC block equivalence vs _RefBlock (weights copied, same input)."""
    cfg = _small_config()
    layer_idx = cfg.num_hash_layers  # use score-routed MoE (_RefBlock has no hash routing)
    block = DeepseekV4Block(cfg, layer_idx=layer_idx)
    # Non-zero HC / gate / sink params so mixer isn't degenerate.
    block.hc_attn_fn.data.normal_(std=0.02)
    block.hc_ffn_fn.data.normal_(std=0.02)
    block.hc_attn_base.data.normal_(std=0.1)
    block.hc_ffn_base.data.normal_(std=0.1)
    block.hc_attn_scale.data.normal_(mean=1.0, std=0.1)
    block.hc_ffn_scale.data.normal_(mean=1.0, std=0.1)
    block.ffn.gate.weight.data.normal_()
    block.attn.attn_sink.data.normal_(std=0.1)
    block.eval()

    ref = _RefBlock(cfg).eval()
    _copy_block_weights(ref, block)

    B, S = 2, 8  # S <= sliding_window so reference path matches AD's sliding-window + sinks
    x = torch.randn(B, S, cfg.hc_mult, cfg.hidden_size)
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    cos_tbl, sin_tbl = _build_freqs_cis(
        cfg.qk_rope_head_dim, cfg.max_position_embeddings, cfg.rope_theta, 0, 1.0, 32, 1
    )
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos = cos_tbl[position_ids]
    sin = sin_tbl[position_ids]
    freqs_cis_ref = _freqs_cis_from_cos_sin(cos[0], sin[0])

    y_custom = block(x, input_ids, cos, sin, cos, sin)
    y_ref = ref(x, freqs_cis_ref)
    assert y_custom.shape == x.shape
    assert_rmse_close(y_custom, y_ref, rmse_ratio_tol=0.05, msg="Block equivalence: ")


def test_flash_base_checkpoint_index_matches_expected_layout():
    """Validate repo-local representative keys from the Flash-Base safetensors index."""
    index_keys = _load_flash_base_index_keys()
    assert _EXPECTED_FLASH_BASE_INDEX_KEYS.issubset(index_keys)

    state_dict = {key: torch.empty(()) for key in _EXPECTED_FLASH_BASE_INDEX_KEYS}
    _remap_deepseek_v4_checkpoint_keys(state_dict)
    assert "model.embed_tokens.weight" in state_dict
    assert "lm_head.weight" in state_dict
    assert "model.norm.weight" in state_dict
    assert "model.hc_head_fn" in state_dict
    assert "model.hc_head_scale" in state_dict
    assert "model.layers.0.attn.wq_a.weight" in state_dict
    assert "model.layers.0.attn.wq_a.weight_scale_inv" in state_dict
    assert "model.layers.0.ffn.experts.0.gate_proj.weight" in state_dict
    assert "model.layers.0.ffn.experts.0.gate_proj.weight_scale_inv" in state_dict
    assert "model.layers.0.ffn.experts.0.up_proj.weight" in state_dict
    assert "model.layers.0.ffn.experts.0.down_proj.weight" in state_dict
    assert "model.layers.0.ffn.shared_experts.gate_proj.weight" in state_dict
    assert "model.layers.0.ffn.shared_experts.gate_proj.weight_scale_inv" in state_dict
    assert "model.layers.0.ffn.shared_experts.up_proj.weight" in state_dict
    assert "model.layers.0.ffn.shared_experts.down_proj.weight" in state_dict
    assert "model.layers.2.attn.compressor.wkv.weight" in state_dict
    assert "model.layers.2.attn.compressor.wkv.weight_scale_inv" in state_dict
    assert "model.layers.2.attn.indexer.wq_b.weight" in state_dict
    assert "model.layers.2.attn.indexer.wq_b.weight_scale_inv" in state_dict


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="requires torch float8")
def test_remap_dequantizes_wo_a_fp8_checkpoint_weight():
    """HF inference dequantizes grouped wo_a weights during checkpoint conversion."""
    scale = torch.full((1, 1), 0.25, dtype=torch.float32)
    weight_fp32 = torch.linspace(-8.0, 8.0, 128 * 128, dtype=torch.float32).view(128, 128)
    weight_fp8 = (weight_fp32 / scale).to(torch.float8_e4m3fn)
    expected = (weight_fp8.float() * scale).bfloat16()

    state_dict = {
        "layers.0.attn.wo_a.weight": weight_fp8,
        "layers.0.attn.wo_a.scale": scale,
    }
    _remap_deepseek_v4_checkpoint_keys(state_dict)

    assert "model.layers.0.attn.wo_a.weight" in state_dict
    assert "model.layers.0.attn.wo_a.weight_scale_inv" not in state_dict
    assert state_dict["model.layers.0.attn.wo_a.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(state_dict["model.layers.0.attn.wo_a.weight"], expected)


def test_load_state_dict_accepts_hf_checkpoint_layout():
    """HF checkpoint names are remapped to the AD module names during load."""
    cfg = _small_config()
    model = DeepseekV4ForCausalLM(cfg).eval()
    layer = model.model.layers[0]

    embed = torch.randn_like(model.model.embed_tokens.weight)
    head = torch.randn_like(model.lm_head.weight)
    norm = torch.randn_like(model.model.norm.weight)
    expert_w1 = torch.randn_like(layer.ffn.experts[0].gate_proj.weight)
    expert_w3 = torch.randn_like(layer.ffn.experts[0].up_proj.weight)
    expert_w2 = torch.randn_like(layer.ffn.experts[0].down_proj.weight)
    shared_w1 = torch.randn_like(layer.ffn.shared_experts.gate_proj.weight)
    shared_w3 = torch.randn_like(layer.ffn.shared_experts.up_proj.weight)
    shared_w2 = torch.randn_like(layer.ffn.shared_experts.down_proj.weight)
    gate = torch.randn_like(layer.ffn.gate.weight)

    incompatible = model.load_state_dict(
        {
            "embed.weight": embed,
            "head.weight": head,
            "norm.weight": norm,
            "layers.0.ffn.experts.0.w1.weight": expert_w1,
            "layers.0.ffn.experts.0.w3.weight": expert_w3,
            "layers.0.ffn.experts.0.w2.weight": expert_w2,
            "layers.0.ffn.shared_experts.w1.weight": shared_w1,
            "layers.0.ffn.shared_experts.w3.weight": shared_w3,
            "layers.0.ffn.shared_experts.w2.weight": shared_w2,
            "layers.0.ffn.gate.weight": gate,
        },
        strict=False,
    )

    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(model.model.embed_tokens.weight, embed)
    torch.testing.assert_close(model.lm_head.weight, head)
    torch.testing.assert_close(model.model.norm.weight, norm)
    torch.testing.assert_close(layer.ffn.experts[0].gate_proj.weight, expert_w1)
    torch.testing.assert_close(layer.ffn.experts[0].up_proj.weight, expert_w3)
    torch.testing.assert_close(layer.ffn.experts[0].down_proj.weight, expert_w2)
    torch.testing.assert_close(layer.ffn.shared_experts.gate_proj.weight, shared_w1)
    torch.testing.assert_close(layer.ffn.shared_experts.up_proj.weight, shared_w3)
    torch.testing.assert_close(layer.ffn.shared_experts.down_proj.weight, shared_w2)
    torch.testing.assert_close(layer.ffn.gate.weight, gate)


def test_load_state_dict_accepts_stacked_moe_checkpoint_layout():
    """Stacked expert tensors are unstacked into the AD per-expert ModuleList layout."""
    cfg = _small_config()
    model = DeepseekV4ForCausalLM(cfg).eval()
    layer = model.model.layers[0]

    expert_w1 = torch.randn(cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size)
    expert_w3 = torch.randn_like(expert_w1)
    expert_w2 = torch.randn(cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size)

    incompatible = model.load_state_dict(
        {
            "layers.0.ffn.experts.w1.weight": expert_w1,
            "layers.0.ffn.experts.w3.weight": expert_w3,
            "layers.0.ffn.experts.w2.weight": expert_w2,
        },
        strict=False,
    )

    assert incompatible.unexpected_keys == []
    for expert_idx, expert in enumerate(layer.ffn.experts):
        torch.testing.assert_close(expert.gate_proj.weight, expert_w1[expert_idx])
        torch.testing.assert_close(expert.up_proj.weight, expert_w3[expert_idx])
        torch.testing.assert_close(expert.down_proj.weight, expert_w2[expert_idx])


def test_load_state_dict_accepts_fused_gate_up_moe_checkpoint_layout():
    """Fused gate/up expert tensors are split and unstacked when present."""
    cfg = _small_config()
    model = DeepseekV4ForCausalLM(cfg).eval()
    layer = model.model.layers[0]

    gate_up = torch.randn(cfg.n_routed_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
    down = torch.randn(cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size)

    incompatible = model.load_state_dict(
        {
            "layers.0.ffn.experts.gate_up_proj": gate_up,
            "layers.0.ffn.experts.down_proj": down,
        },
        strict=False,
    )

    assert incompatible.unexpected_keys == []
    gate, up = gate_up.chunk(2, dim=1)
    for expert_idx, expert in enumerate(layer.ffn.experts):
        torch.testing.assert_close(expert.gate_proj.weight, gate[expert_idx])
        torch.testing.assert_close(expert.up_proj.weight, up[expert_idx])
        torch.testing.assert_close(expert.down_proj.weight, down[expert_idx])


def test_full_model_equivalence_short_seq():
    """Full model equivalence vs _RefModel for compress_ratios=[0,...] and seqlen <= window.

    The AD model reduces to:
      embed -> HC-expand ->
        [HC-pre -> RMSNorm -> RefAttention -> HC-post]
        [HC-pre -> RMSNorm -> RefMoE -> HC-post]
        ... -> HC-collapse -> RMSNorm -> lm_head
    """
    cfg = _small_config()  # num_hash_layers=0 so all layers are score-routed
    model = DeepseekV4ForCausalLM(cfg).eval()
    assert len(model.model.layers) == cfg.num_hidden_layers
    assert isinstance(model.model.layers[0].ffn.experts, nn.ModuleList)
    assert len(model.model.layers[0].ffn.experts) == cfg.n_routed_experts
    sd = model.state_dict()
    assert "model.embed_tokens.weight" in sd
    assert "model.layers.0.ffn.experts.0.gate_proj.weight" in sd
    assert "lm_head.weight" in sd
    # Randomize HC, gate, attn_sink params (not covered by default init).
    for blk in model.model.layers:
        blk.hc_attn_fn.data.normal_(std=0.02)
        blk.hc_ffn_fn.data.normal_(std=0.02)
        blk.hc_attn_base.data.normal_(std=0.1)
        blk.hc_ffn_base.data.normal_(std=0.1)
        blk.hc_attn_scale.data.normal_(mean=1.0, std=0.1)
        blk.hc_ffn_scale.data.normal_(mean=1.0, std=0.1)
        blk.ffn.gate.weight.data.normal_()
        blk.attn.attn_sink.data.normal_(std=0.1)
    model.model.hc_head_fn.data.normal_(std=0.02)
    model.model.hc_head_base.data.normal_(std=0.1)

    # Build reference and copy weights.
    ref = _RefModel(cfg).eval()
    ref.embed_tokens.weight.data.copy_(model.model.embed_tokens.weight.data)
    ref.norm.weight.data.copy_(model.model.norm.weight.data.float())
    ref.hc_head_fn.data.copy_(model.model.hc_head_fn.data)
    ref.hc_head_base.data.copy_(model.model.hc_head_base.data)
    ref.hc_head_scale.data.copy_(model.model.hc_head_scale.data)
    ref.lm_head.weight.data.copy_(model.lm_head.weight.data)
    for i, blk in enumerate(model.model.layers):
        _copy_block_weights(ref.layers[i], blk)

    B, S = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    cos_tbl, sin_tbl = _build_freqs_cis(
        cfg.qk_rope_head_dim, cfg.max_position_embeddings, cfg.rope_theta, 0, 1.0, 32, 1
    )
    cos = cos_tbl[position_ids]
    sin = sin_tbl[position_ids]
    freqs_cis_ref = _freqs_cis_from_cos_sin(cos[0], sin[0])

    logits_custom = model(input_ids=input_ids, position_ids=position_ids).logits
    logits_ref = ref(input_ids, freqs_cis_ref)
    assert logits_custom.shape == (B, S, cfg.vocab_size)
    assert_rmse_close(logits_custom, logits_ref, rmse_ratio_tol=0.05, msg="Full-model logits: ")


# =============================================================================
# Export tests
# =============================================================================


def test_model_can_be_exported():
    cfg = _small_config()
    model = DeepseekV4ForCausalLM(cfg).eval()

    B, S = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )
    with torch.inference_mode():
        out = gm(input_ids=input_ids, position_ids=position_ids)
    assert "logits" in out
    assert out["logits"].shape == (B, S, cfg.vocab_size)
    assert torch.isfinite(out["logits"]).all()

    # Second shape to verify dynamic shapes
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, cfg.vocab_size, (B2, S2))
    position_ids2 = torch.arange(S2).unsqueeze(0)
    with torch.inference_mode():
        out2 = gm(input_ids=input_ids2, position_ids=position_ids2)
    assert out2["logits"].shape == (B2, S2, cfg.vocab_size)
    assert torch.isfinite(out2["logits"]).all()


def test_sparse_prefill_op_matches_inline_math():
    """Sparse-prefill custom op matches the previous inline math.

    ``torch_deepseek_v4_sparse_attn`` must produce identical output to the
    previous inline math for both ratio-4 (indexer) and ratio-128 (no indexer)
    sparse layers. Current AD model already routes through the op; this test
    also dispatches the op explicitly with its argument pack to catch regressions
    in the op registration / schema.
    """
    for ratio in (4, 128):
        cfg = _small_compress_config(ratio=ratio)
        custom = DeepseekV4Attention(cfg, layer_idx=0).eval()
        custom.compressor.ape.data.normal_(std=0.02)
        if custom.indexer is not None:
            custom.indexer.compressor.ape.data.normal_(std=0.02)
        custom.attn_sink.data.normal_(std=0.1)

        B, S = 2, 8
        x = torch.randn(B, S, cfg.hidden_size)
        position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
        cos_tbl, sin_tbl = _build_freqs_cis(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.compress_rope_theta,
            0,
            1.0,
            32,
            1,
        )
        cos = cos_tbl[position_ids]  # [B, S, rd/2]
        sin = sin_tbl[position_ids]

        # Reference: full DeepseekV4Attention forward — emits the op internally.
        y_ref = custom(x, cos, sin, cos, sin, cos_tbl, sin_tbl)

        # Explicit invocation of the op with manually prepared inputs
        rd = cfg.qk_rope_head_dim
        qr = custom.q_norm(custom.wq_a(x))
        q = custom.wq_b(qr).view(B, S, cfg.num_attention_heads, cfg.head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + cfg.rms_norm_eps).to(
            q.dtype
        )
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention import (
            _apply_interleaved_rope,
        )

        q_nope, q_pe = torch.split(q, [cfg.head_dim - rd, rd], dim=-1)
        q_pe = _apply_interleaved_rope(q_pe, cos.unsqueeze(2), sin.unsqueeze(2))
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = custom.kv_norm(custom.wkv(x)).view(B, S, 1, cfg.head_dim)
        kv_nope, kv_pe = torch.split(kv, [cfg.head_dim - rd, rd], dim=-1)
        kv_pe = _apply_interleaved_rope(kv_pe, cos.unsqueeze(2), sin.unsqueeze(2))
        kv_nope = _fake_fp8_act_quant(kv_nope, block_size=64)
        kv = torch.cat([kv_nope, kv_pe], dim=-1)

        if custom.indexer is not None:
            indexer_args = (
                custom.indexer.wq_b.weight,
                custom.indexer.weights_proj.weight,
                custom.indexer.compressor.wkv.weight,
                custom.indexer.compressor.wgate.weight,
                custom.indexer.compressor.ape,
                custom.indexer.compressor.norm.weight,
            )
            index_n_heads = custom.indexer.index_n_heads
            index_head_dim = custom.indexer.index_head_dim
        else:
            indexer_args = (None,) * 6
            index_n_heads = 0
            index_head_dim = 0

        op_out = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn(
            q,
            kv,
            x,
            qr,
            cos,
            sin,
            cos_tbl,
            sin_tbl,
            custom.attn_sink,
            custom.compressor.wkv.weight,
            custom.compressor.wgate.weight,
            custom.compressor.ape,
            custom.compressor.norm.weight,
            *indexer_args,
            custom.softmax_scale,
            custom.window_size,
            custom.compress_ratio,
            custom.rope_head_dim,
            custom.head_dim,
            index_n_heads,
            index_head_dim,
            custom.indexer.index_topk if custom.indexer is not None else 0,
            cfg.rms_norm_eps,
            0,  # layer_idx
            "mha_sparse",
        )

        # Post-op path: inverse RoPE + grouped low-rank O projection, exactly as in
        # DeepseekV4Attention.forward.
        attn_nope, attn_pe = torch.split(op_out, [cfg.head_dim - rd, rd], dim=-1)
        attn_pe = _apply_interleaved_rope(attn_pe, cos.unsqueeze(2), sin.unsqueeze(2), inverse=True)
        attn_out = torch.cat([attn_nope, attn_pe], dim=-1)
        attn_out = attn_out.reshape(B, S, cfg.o_groups, custom.group_head_width)
        wo_a = custom.wo_a.weight.view(cfg.o_groups, cfg.o_lora_rank, custom.group_head_width)
        o = torch.einsum("bsgd,grd->bsgr", attn_out, wo_a)
        y_from_op = custom.wo_b(o.reshape(B, S, cfg.o_groups * cfg.o_lora_rank))

        torch.testing.assert_close(y_ref, y_from_op, rtol=1e-5, atol=1e-5)


def test_exported_graph_has_sparse_and_dense_attention_nodes():
    """Exported graph places dense vs sparse attention nodes in the right ops.

    The AD custom model must export a graph where:
      * each dense layer emits ``auto_deploy::torch_attention`` with
        ``layer_idx`` and ``layer_type="mha"`` set, and
      * each sparse layer emits ``auto_deploy::torch_deepseek_v4_sparse_attn``.
    This is the precondition the cache transforms rely on.
    """
    # Build a 3-layer fixture: 1 dense, 1 ratio-4, 1 ratio-128.
    cfg = _small_config()
    cfg.num_hidden_layers = 3
    cfg.compress_ratios = [0, 4, 128]
    model = DeepseekV4ForCausalLM(cfg).eval()

    B, S = 2, 4
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    dense_op = torch.ops.auto_deploy.torch_attention.default
    sparse_op = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn.default
    dense_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is dense_op]
    sparse_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is sparse_op]

    assert len(dense_nodes) == 1, f"Expected 1 dense attention node, got {len(dense_nodes)}"
    assert len(sparse_nodes) == 2, (
        f"Expected 2 sparse attention nodes (ratio-4 + ratio-128), got {len(sparse_nodes)}"
    )

    # layer_idx and layer_type must appear on every dense attention node.
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args

    for n in dense_nodes:
        layer_idx, layer_type = extract_op_args(n, "layer_idx", "layer_type")
        assert layer_idx == 0
        assert layer_type == "mha"

    # Sparse nodes must also carry layer_idx/layer_type + the constants the
    # descriptor reads in get_constants.
    seen_sparse_layer_idxs = set()
    for n in sparse_nodes:
        layer_idx, layer_type = extract_op_args(n, "layer_idx", "layer_type")
        assert layer_type == "mha_sparse"
        seen_sparse_layer_idxs.add(layer_idx)
        scale, window_size, compress_ratio, index_topk = extract_op_args(
            n, "scale", "window_size", "compress_ratio", "index_topk"
        )
        assert compress_ratio in (4, 128)
        assert window_size == cfg.sliding_window
        assert scale == cfg.head_dim**-0.5
        if compress_ratio == 4:
            assert index_topk == cfg.index_topk
        else:
            assert index_topk == 0
    assert seen_sparse_layer_idxs == {1, 2}


def test_sparse_cache_transform_replaces_sparse_attention_nodes():
    """KV-cache transform replaces sparse source nodes with cached sparse op nodes."""
    cfg = _small_config()
    cfg.num_hidden_layers = 3
    cfg.compress_ratios = [0, 4, 128]
    model = DeepseekV4ForCausalLM(cfg).eval()

    B, S = 2, 4
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
    from tensorrt_llm._torch.auto_deploy.transform.interface import Stages
    from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import (
        InsertCachedDeepseekV4SparseAttention,
    )
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    cm = CachedSequenceInterface(
        max_seq_len=cfg.max_position_embeddings,
        max_batch_size=B,
        max_num_tokens=B * cfg.max_position_embeddings,
        device="cpu",
        kv_cache_config=KvCacheConfig(dtype="bfloat16", free_gpu_memory_fraction=0.0),
    )
    transform = InsertCachedDeepseekV4SparseAttention.from_kwargs(stage=Stages.CACHE_INIT)
    gm, info = transform._apply(gm, cm, factory=None, shared_config=None)

    source_op = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn.default
    cached_op = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn_with_cache.default
    source_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is source_op]
    cached_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is cached_op]

    assert info.num_matches == 2
    assert source_nodes == []
    assert len(cached_nodes) == 2
    assert len(cm._resource_lookup) == 14
    # 19 source tensor args + 5 standard metadata args + 7 caches + 9 constants.
    assert all(len(n.args) == 40 for n in cached_nodes)


def test_dense_cache_transform_routes_dense_layers_to_triton_paged_only():
    """Dense DS-V4 layers route to triton_paged while sparse layers stay sparse."""
    cfg = _small_config()
    cfg.num_hidden_layers = 3
    cfg.compress_ratios = [0, 4, 128]
    model = DeepseekV4ForCausalLM(cfg).eval()

    B, S = 2, 4
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
    from tensorrt_llm._torch.auto_deploy.transform.interface import Stages
    from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import (
        InsertCachedAttention,
        InsertCachedDeepseekV4SparseAttention,
    )
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    dense_source_op = torch.ops.auto_deploy.torch_attention.default
    sparse_source_op = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn.default
    dense_nodes = [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target is dense_source_op
    ]
    assert len(dense_nodes) == 1
    dense_sink_arg = extract_op_args(dense_nodes[0], "sinks")[0]

    cm = CachedSequenceInterface(
        max_seq_len=cfg.max_position_embeddings,
        max_batch_size=B,
        max_num_tokens=B * cfg.max_position_embeddings,
        device="cpu",
        kv_cache_config=KvCacheConfig(
            dtype="bfloat16", free_gpu_memory_fraction=0.0, tokens_per_block=64
        ),
    )
    transform = InsertCachedAttention.from_kwargs(stage=Stages.CACHE_INIT, backend="triton_paged")
    gm, info = transform._apply(gm, cm, factory=None, shared_config=None)

    dense_cached_op = torch.ops.auto_deploy.triton_paged_mha_with_cache.default
    sparse_cached_op = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn_with_cache.default
    dense_source_nodes = [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target is dense_source_op
    ]
    dense_cached_nodes = [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target is dense_cached_op
    ]
    sparse_source_nodes = [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target is sparse_source_op
    ]
    sparse_cached_nodes = [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target is sparse_cached_op
    ]

    assert info.num_matches == 1
    assert dense_source_nodes == []
    assert len(dense_cached_nodes) == 1
    assert len(sparse_source_nodes) == 2
    assert sparse_cached_nodes == []

    cached_node = dense_cached_nodes[0]
    assert cached_node.args[-3] == cfg.head_dim**-0.5
    assert cached_node.args[-2] == cfg.sliding_window
    assert cached_node.args[-1] is dense_sink_arg

    sparse_transform = InsertCachedDeepseekV4SparseAttention.from_kwargs(stage=Stages.CACHE_INIT)
    gm, sparse_info = sparse_transform._apply(gm, cm, factory=None, shared_config=None)

    dense_cached_nodes = [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target is dense_cached_op
    ]
    sparse_source_nodes = [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target is sparse_source_op
    ]
    sparse_cached_nodes = [
        n for n in gm.graph.nodes if n.op == "call_function" and n.target is sparse_cached_op
    ]

    assert sparse_info.num_matches == 2
    assert len(dense_cached_nodes) == 1
    assert sparse_source_nodes == []
    assert len(sparse_cached_nodes) == 2


def test_exported_hc_mixes_use_fp32_matmul_not_linear_ops():
    """HC mixers must not be exported as quantizable linear ops.

    DeepSeek V4 HC parameters are fp32 routing/mixing parameters, not model
    projection weights. Keeping them as explicit fp32 matmuls prevents the
    fine-grained FP8 linear rewrite from consuming them.
    """
    cfg = _small_config()
    cfg.num_hidden_layers = 1
    cfg.compress_ratios = [0]
    model = DeepseekV4ForCausalLM(cfg).eval()

    B, S = 2, 2
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    hc_fn_nodes = {
        n
        for n in gm.graph.nodes
        if n.op == "get_attr" and str(n.target).endswith(("hc_attn_fn", "hc_ffn_fn", "hc_head_fn"))
    }
    assert len(hc_fn_nodes) == 3

    def node_depends_on_hc_fn(node: Node) -> bool:
        seen: set[Node] = set()
        stack: list[Node] = [node]
        while stack:
            current = stack.pop()
            if current in hc_fn_nodes:
                return True
            if current in seen:
                continue
            seen.add(current)
            for arg in current.all_input_nodes:
                stack.append(arg)
        return False

    def arg_depends_on_hc_fn(arg) -> bool:
        if isinstance(arg, Node):
            return node_depends_on_hc_fn(arg)
        if isinstance(arg, (tuple, list)):
            return any(arg_depends_on_hc_fn(item) for item in arg)
        return False

    linear_targets = {
        torch.ops.aten.linear.default,
        torch.ops.auto_deploy.torch_linear_simple.default,
    }
    hc_linear_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target in linear_targets
        and len(n.args) > 1
        and arg_depends_on_hc_fn(n.args[1])
    ]
    assert not hc_linear_nodes

    hc_matmul_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.matmul.default
        and len(n.args) > 1
        and arg_depends_on_hc_fn(n.args[1])
    ]
    assert len(hc_matmul_nodes) == 3


def test_sparse_layers_do_not_break_tp_sharding_detection():
    """Sparse attention's hidden-state input must not pull previous layers into TP analysis."""
    cfg = _small_config()
    cfg.num_hidden_layers = 3
    cfg.compress_ratios = [0, 4, 128]
    model = DeepseekV4ForCausalLM(cfg).eval()

    B, S = 2, 4
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer

    optimizer = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "sharding_dims": ["tp", "ep", "bmm"],
                "sharding_source": ["heuristic"],
                "shard_all_unprocessed": False,
            },
        },
    )
    optimizer.shared_config.local_rank = 0
    optimizer.shared_config.world_size = 2
    optimizer(None, gm)

    container = gm._sharding_transform_container
    sharded_by_target = {t.target_node: t for t in container.weight_sharding_transforms}

    from tensorrt_llm._torch.auto_deploy.transform.library.sharding import SplitDimension
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import LayerType

    for layer_idx in (1, 2):
        for projection_name in ("wq_a", "wq_b", "wkv", "wo_b"):
            matches = [
                t
                for name, t in sharded_by_target.items()
                if f"model_layers_{layer_idx}_attn_{projection_name}" in name
            ]
            assert len(matches) == 1
            transform = matches[0]
            assert transform.split_dim == SplitDimension.COLUMN
            assert transform.dist_op == "all_gather"
            assert transform.layer_type == LayerType.UNKNOWN

        assert not any(f"model_layers_{layer_idx}_attn_wo_a" in name for name in sharded_by_target)

    assert len(container.ep_transforms) == cfg.num_hidden_layers


def test_sharded_sparse_layers_still_rewrite_to_cached_sparse_attention():
    """Sharding executor must compose with sparse attention cache insertion."""
    cfg = _small_config()
    cfg.num_hidden_layers = 3
    cfg.compress_ratios = [0, 4, 128]
    model = DeepseekV4ForCausalLM(cfg).eval()

    B, S = 2, 4
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
    from tensorrt_llm._torch.auto_deploy.transform.interface import Stages
    from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import (
        InsertCachedDeepseekV4SparseAttention,
    )
    from tensorrt_llm._torch.auto_deploy.transform.library.sharding import ShardingTransformExecutor
    from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    optimizer = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "sharding_dims": ["tp", "ep", "bmm"],
                "sharding_source": ["heuristic"],
                "shard_all_unprocessed": False,
            },
        },
    )
    optimizer.shared_config.local_rank = 0
    optimizer.shared_config.world_size = 2
    optimizer(None, gm)

    executor = ShardingTransformExecutor.from_kwargs(
        stage=Stages.SHARDING,
        run_graph_cleanup=False,
        requires_clean_graph=False,
    )
    gm, shard_info = executor._apply(gm, cm=None, factory=None, shared_config=None)
    assert shard_info.num_matches > 0

    cm = CachedSequenceInterface(
        max_seq_len=cfg.max_position_embeddings,
        max_batch_size=B,
        max_num_tokens=B * cfg.max_position_embeddings,
        device="cpu",
        kv_cache_config=KvCacheConfig(dtype="bfloat16", free_gpu_memory_fraction=0.0),
    )
    cache_transform = InsertCachedDeepseekV4SparseAttention.from_kwargs(
        stage=Stages.CACHE_INIT,
        run_graph_cleanup=False,
        requires_clean_graph=False,
    )
    gm, cache_info = cache_transform._apply(gm, cm, factory=None, shared_config=None)

    source_op = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn.default
    cached_op = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn_with_cache.default
    source_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is source_op]
    cached_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is cached_op]

    assert cache_info.num_matches == 2
    assert source_nodes == []
    assert len(cached_nodes) == 2


def test_sparse_descriptor_resource_handler_shapes():
    """Sparse descriptor returns resource handlers with expected dtypes and shapes.

    ``DeepseekV4SparseAttentionDescriptor.get_cache_initializers`` must return
    the right set of resource handlers with sensible dtypes and token-shapes so
    the KV cache transform wires the caches correctly.
    """
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention import (
        DeepseekV4SparseAttentionDescriptor,
    )
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
        StateResourceHandler,
        UnpagedResourceHandler,
    )
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    cfg = _small_compress_config(ratio=4)
    model = DeepseekV4ForCausalLM(cfg).eval()

    B, S = 2, 4
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    sparse_op = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn.default
    sparse_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is sparse_op]
    assert sparse_nodes, "Expected at least one sparse attention node"
    assert DeepseekV4SparseAttentionDescriptor.get_num_qkv_args() == 19

    cache_config = KvCacheConfig(dtype="bfloat16")
    handlers = DeepseekV4SparseAttentionDescriptor.get_cache_initializers(
        sparse_nodes[0], cache_config
    )

    expected_keys = {
        "window_cache",
        "compressed_kv_cache",
        "compressor_kv_state",
        "compressor_score_state",
        "indexer_compressed_kv_cache",
        "indexer_kv_state",
        "indexer_score_state",
    }
    assert set(handlers.keys()) == expected_keys

    # Window / compressed caches are per-token (Unpaged); rolling state is
    # fixed-shape (StateResourceHandler).
    assert isinstance(handlers["window_cache"], UnpagedResourceHandler)
    assert isinstance(handlers["compressed_kv_cache"], UnpagedResourceHandler)
    assert handlers["window_cache"].token_shape == (cfg.head_dim,)
    for key in (
        "compressor_kv_state",
        "compressor_score_state",
        "indexer_kv_state",
        "indexer_score_state",
    ):
        assert isinstance(handlers[key], StateResourceHandler), key
        assert handlers[key].dtype == torch.float32, key
    # Rolling state: coff * compress_ratio tokens × coff * head_dim dims (coff=2 for ratio 4).
    coff = 2  # ratio == 4
    assert handlers["compressor_kv_state"].state_shape == (
        coff * cfg.compress_ratios[0],
        coff * cfg.head_dim,
    )


def test_compressed_model_can_be_exported():
    cfg = _small_compress_config(ratio=4)
    model = DeepseekV4ForCausalLM(cfg).eval()

    B, S = 2, 4
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )
    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )
    with torch.inference_mode():
        out = gm(input_ids=input_ids, position_ids=position_ids)
    assert out["logits"].shape == (B, S, cfg.vocab_size)
    assert torch.isfinite(out["logits"]).all()

    for s2 in (1, 2):
        input_ids2 = torch.randint(0, cfg.vocab_size, (1, s2))
        position_ids2 = torch.arange(s2).unsqueeze(0)
        with torch.inference_mode():
            out2 = gm(input_ids=input_ids2, position_ids=position_ids2)
        assert out2["logits"].shape == (1, s2, cfg.vocab_size)
        assert torch.isfinite(out2["logits"]).all()


def test_config_registration():
    cfg = _small_config()
    assert cfg.model_type == "deepseek_v4"
    assert hasattr(cfg, "hc_mult")
    assert hasattr(cfg, "num_hash_layers")
    assert hasattr(cfg, "o_groups")


def test_hash_routing_layer_types():
    """First num_hash_layers blocks use hash routing; later layers use score routing."""
    cfg = _small_config(num_hash_layers=1)
    model = DeepseekV4ForCausalLM(cfg).eval()
    assert model.model.layers[0].ffn.gate.hash_routing is True
    assert hasattr(model.model.layers[0].ffn.gate, "tid2eid")
    for i in range(1, cfg.num_hidden_layers):
        assert model.model.layers[i].ffn.gate.hash_routing is False
        assert model.model.layers[i].ffn.gate.bias is not None
