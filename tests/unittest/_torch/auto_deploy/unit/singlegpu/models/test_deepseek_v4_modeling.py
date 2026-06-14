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

"""Hierarchical DeepSeek V4 Flash semantic tests for AutoDeploy."""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import tensorrt_llm._torch.auto_deploy.transform.library  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models import custom as custom_models
from tensorrt_llm._torch.auto_deploy.models.custom import modeling_deepseek_v4 as dsv4
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Compressor,
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    DeepseekV4Indexer,
    DeepseekV4MLP,
    DeepseekV4MoE,
    DeepseekV4RMSNorm,
)
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args, is_op

DeepseekV4Router = (
    getattr(dsv4, "DeepseekV4Router", None)
    or getattr(dsv4, "DeepseekV4MoEGate", None)
    or getattr(dsv4, "DeepseekV4Gate", None)
)


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    torch.manual_seed(1234)


def _small_config(**overrides) -> DeepseekV4Config:
    values = {
        "vocab_size": 32,
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 8,
        "qk_rope_head_dim": 2,
        "o_groups": 2,
        "o_lora_rank": 8,
        "sliding_window": 3,
        "compress_ratios": (0, 4, 128),
        "compress_rope_theta": 16000.0,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        "moe_intermediate_size": 32,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "num_hash_layers": 1,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.25,
        "norm_topk_prob": True,
        "swiglu_limit": 0.5,
        "hidden_act": "silu",
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 256,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        "rms_norm_eps": 1e-6,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-6,
        "ad_rope_cache_len": 256,
        "ad_compress_max_seq_len": 256,
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    return DeepseekV4Config(**values)


def _make_apply_sharding_hints_optimizer(
    world_size: int,
    rank: int,
    dist_backend: str | None = None,
) -> InferenceOptimizer:
    apply_config = {"stage": "sharding"}
    if dist_backend is not None:
        apply_config["dist_backend"] = dist_backend

    opt = InferenceOptimizer(
        factory=None,
        config={"apply_sharding_hints": apply_config},
    )
    opt.shared_config = SharedConfig(
        local_rank=rank,
        world_size=world_size,
        dist_config=DistConfig(
            world_size=world_size,
            rank=rank,
            tp_size=world_size,
            moe_ep_size=world_size,
        ),
    )
    return opt


def _call_nodes(gm: torch.fx.GraphModule, op) -> list[torch.fx.Node]:
    return [node for node in gm.graph.nodes if is_op(node, op)]


def _position_ids(batch: int, seq: int, device: torch.device | str) -> torch.Tensor:
    return torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)


def _output_logits(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output["logits"]
    return output.logits


def assert_rmse_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rmse_ratio_tol: float,
    msg: str = "",
) -> None:
    diff = actual.float() - expected.float()
    rmse_diff = torch.sqrt(torch.mean(diff**2))
    rmse_ref = torch.sqrt(torch.mean(expected.float() ** 2))
    ratio = (rmse_diff / rmse_ref).item()
    assert ratio < rmse_ratio_tol, (
        f"{msg}RMSE ratio {ratio:.6f} exceeds tolerance {rmse_ratio_tol}. "
        f"(rmse_diff={rmse_diff.item():.6f}, rmse_ref={rmse_ref.item():.6f})"
    )


def _get_linear(module: nn.Module, *names: str) -> nn.Module:
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    raise AssertionError(f"{type(module).__name__} is missing one of {names}")


def _linear_ref(x: torch.Tensor, linear: nn.Module) -> torch.Tensor:
    bias = getattr(linear, "bias", None)
    return F.linear(x.float(), linear.weight.float(), None if bias is None else bias.float()).to(
        x.dtype
    )


def _copy_deterministic_linear(linear: nn.Module, offset: float) -> None:
    values = torch.linspace(-0.7 + offset, 0.7 + offset, linear.weight.numel())
    with torch.no_grad():
        linear.weight.copy_(values.view_as(linear.weight))
        bias = getattr(linear, "bias", None)
        if bias is not None:
            bias.zero_()


def _sequential_tensor(shape: torch.Size | tuple[int, ...], offset: float = 0.0) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return (torch.arange(numel, dtype=torch.float32) + offset).view(shape)


def _set_mlp_weights(module: nn.Module, offset: float = 0.0) -> None:
    _copy_deterministic_linear(_get_linear(module, "gate_proj", "w1"), offset)
    _copy_deterministic_linear(_get_linear(module, "up_proj", "w3"), offset + 0.1)
    _copy_deterministic_linear(_get_linear(module, "down_proj", "w2"), offset + 0.2)


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def _ref_ceil_pow2_scale(amax: torch.Tensor, max_value: float, min_value: float) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(amax.clamp_min(min_value) / max_value)))


def _ref_fake_fp8_act_quant(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(*x_float.shape[:-1], dim // block_size, block_size)
    scale = _ref_ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 448.0, 1.0e-4)
    quant = torch.clamp(grouped / scale, -448.0, 448.0).to(dtype).float()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _ref_fake_fp4_act_quant(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(*x_float.shape[:-1], dim // block_size, block_size)
    scale = _ref_ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 6.0, 6.0 * 2.0**-126)
    normalized = torch.clamp(grouped / scale, -6.0, 6.0)
    levels = normalized.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    level_idx = (normalized.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    quant = levels[level_idx] * normalized.sign()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _ref_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    if dim <= 1:
        return x
    if dim & (dim - 1):
        raise ValueError(f"Hadamard rotation requires power-of-two dimension, got {dim}.")

    original_shape = x.shape
    out = x.float()
    width = 1
    while width < dim:
        out = out.reshape(*out.shape[:-1], dim // (2 * width), 2, width)
        left = out[..., 0, :]
        right = out[..., 1, :]
        out = torch.cat((left + right, left - right), dim=-1).reshape(original_shape)
        width *= 2
    return (out * (dim**-0.5)).to(x.dtype)


def _ref_overlap_transform(tensor: torch.Tensor, head_dim: int, value: float) -> torch.Tensor:
    batch_size, compressed_len, ratio, _ = tensor.shape
    previous = tensor[:, :, :, :head_dim]
    current = tensor[:, :, :, head_dim:]
    prefix = tensor.new_full((batch_size, 1, ratio, head_dim), value)
    previous = torch.cat((prefix, previous[:, :-1]), dim=1)
    return torch.cat((previous, current), dim=2)


def _ref_mlp(module: nn.Module, x: torch.Tensor, swiglu_limit: float | None = None) -> torch.Tensor:
    gate_proj = _get_linear(module, "gate_proj", "w1")
    up_proj = _get_linear(module, "up_proj", "w3")
    down_proj = _get_linear(module, "down_proj", "w2")
    gate = _linear_ref(x, gate_proj).float()
    up = _linear_ref(x, up_proj).float()
    limit = getattr(module, "swiglu_limit", 0.0) if swiglu_limit is None else swiglu_limit
    if limit > 0:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    hidden = F.silu(gate) * up
    return _linear_ref(hidden.to(x.dtype), down_proj)


def _set_router_weights(router: nn.Module) -> None:
    values = torch.linspace(-0.4, 0.5, router.weight.numel())
    with torch.no_grad():
        router.weight.copy_(values.view_as(router.weight))
        bias = getattr(router, "bias", None)
        if bias is not None:
            bias.copy_(torch.tensor([0.0, 0.4, -0.3, 0.8], dtype=bias.dtype))
        tid2eid = getattr(router, "tid2eid", None)
        if tid2eid is not None:
            pattern = torch.tensor([[0, 2], [3, 1], [1, 0], [2, 3]], dtype=tid2eid.dtype)
            repeats = (tid2eid.shape[0] + pattern.shape[0] - 1) // pattern.shape[0]
            tid2eid.copy_(pattern.repeat(repeats, 1)[: tid2eid.shape[0]])


def _ref_router(
    router: nn.Module,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    config: DeepseekV4Config,
    hash_routing: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softplus(F.linear(hidden_states, router.weight.float())).sqrt()
    if hash_routing:
        tid2eid = getattr(router, "tid2eid")
        selected = tid2eid[input_ids].to(torch.long)
        selected = selected.clamp(0, config.n_routed_experts - 1)
    else:
        bias = getattr(router, "bias", None)
        assert bias is not None
        selected = (scores + bias.float()).topk(config.num_experts_per_tok, dim=-1).indices
    weights = scores.gather(1, selected)
    if getattr(config, "norm_topk_prob", True):
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return selected, weights * config.routed_scaling_factor


def _interleaved_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    if inverse:
        sin = -sin
    x_pair = x.float().reshape(*x.shape[:-1], -1, 2)
    cos = cos.float().unsqueeze(-1)
    sin = sin.float().unsqueeze(-1)
    first = x_pair[..., 0:1] * cos - x_pair[..., 1:2] * sin
    second = x_pair[..., 1:2] * cos + x_pair[..., 0:1] * sin
    return torch.cat([first, second], dim=-1).flatten(-2).to(x.dtype)


def _rope_cos_sin(
    position_ids: torch.Tensor, rope_dim: int, base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, device=position_ids.device).float() / rope_dim)
    )
    phase = position_ids.float().unsqueeze(-1) * inv_freq
    return phase.cos(), phase.sin()


def _position_embeddings(
    config: DeepseekV4Config, position_ids: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    rotary_cls = getattr(dsv4, "DeepseekV4RotaryEmbedding", None)
    if rotary_cls is not None:
        try:
            return rotary_cls(config)()
        except TypeError:
            return rotary_cls(config)(position_ids)

    table_positions = torch.arange(config.ad_rope_cache_len, device=position_ids.device)
    cos, sin = _rope_cos_sin(table_positions, config.qk_rope_head_dim, config.rope_theta)
    cos_comp, sin_comp = _rope_cos_sin(
        table_positions, config.qk_rope_head_dim, config.compress_rope_theta
    )
    return cos, sin, cos_comp, sin_comp


def _ref_compressor(
    compressor: DeepseekV4Compressor,
    hidden_states: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    ratio = compressor.compress_ratio
    max_compressed_len = compressor.max_compressed_len
    row_offsets = torch.arange(max_compressed_len, device=hidden_states.device)
    token_offsets = torch.arange(ratio, device=hidden_states.device)
    gather_idxs = row_offsets.unsqueeze(1) * ratio + token_offsets
    valid = gather_idxs < seq_len
    gather_idxs = torch.where(valid, gather_idxs, torch.zeros_like(gather_idxs))
    flat_idxs = gather_idxs.reshape(-1)

    kv_all = _linear_ref(hidden_states, compressor.wkv).float()
    score_all = _linear_ref(hidden_states, compressor.wgate).float()
    kv = kv_all[:, flat_idxs].view(batch_size, max_compressed_len, ratio, -1)
    score = score_all[:, flat_idxs].view(batch_size, max_compressed_len, ratio, -1)
    score = score + compressor.ape.float()
    score = torch.where(
        valid.view(1, max_compressed_len, ratio, 1),
        score,
        score.new_full((), -1.0e20),
    )
    if compressor.overlap:
        kv = _ref_overlap_transform(kv, compressor.head_dim, 0.0)
        score = _ref_overlap_transform(score, compressor.head_dim, -1.0e20)

    compressed = (kv * score.softmax(dim=2)).sum(dim=2).to(hidden_states.dtype)
    compressed = _ref_rmsnorm(compressed, compressor.norm.weight, compressor.norm.eps)

    row_start = row_offsets * ratio
    row_start = torch.minimum(row_start, torch.full_like(row_start, seq_len - 1))
    compressed_position_ids = position_ids[:, row_start]
    cos = cos_table[compressed_position_ids]
    sin = sin_table[compressed_position_ids]
    nope_dim = compressor.head_dim - compressor.rope_head_dim
    nope, pe = torch.split(compressed, [nope_dim, compressor.rope_head_dim], dim=-1)
    pe = _interleaved_rope(pe, cos, sin)
    compressed = torch.cat((nope, pe), dim=-1)

    if compressor.rotate:
        return _ref_fake_fp4_act_quant(_ref_hadamard_rotate(compressed), block_size=32)

    nope, pe = torch.split(compressed, [nope_dim, compressor.rope_head_dim], dim=-1)
    nope = _ref_fake_fp8_act_quant(nope, block_size=64)
    return torch.cat((nope, pe), dim=-1)


def _ref_indexer(
    indexer: DeepseekV4Indexer,
    hidden_states: torch.Tensor,
    q_lora: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    q = _linear_ref(q_lora, indexer.wq_b)
    q = q.view(batch_size, seq_len, indexer.index_n_heads, indexer.index_head_dim)
    q_nope, q_pe = torch.split(
        q,
        [indexer.index_head_dim - indexer.rope_head_dim, indexer.rope_head_dim],
        dim=-1,
    )
    q_pe = _interleaved_rope(q_pe, cos.unsqueeze(2), sin.unsqueeze(2))
    q = _ref_fake_fp4_act_quant(_ref_hadamard_rotate(torch.cat((q_nope, q_pe), dim=-1)))

    index_k = _ref_compressor(indexer.compressor, hidden_states, cos_table, sin_table, position_ids)
    weights = _linear_ref(hidden_states, indexer.weights_proj).float()
    weights = weights * (indexer.softmax_scale * indexer.index_n_heads**-0.5)

    index_score = torch.matmul(
        q.transpose(1, 2),
        index_k.transpose(1, 2).unsqueeze(1),
    ).transpose(1, 2)
    index_score = index_score.float()
    index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)

    compressed_positions = torch.arange(
        indexer.compressor.max_compressed_len,
        device=hidden_states.device,
    )
    valid_lengths = torch.arange(1, seq_len + 1, device=hidden_states.device).unsqueeze(1)
    valid_lengths = valid_lengths // indexer.compress_ratio
    index_score = index_score.masked_fill(
        (compressed_positions.unsqueeze(0) >= valid_lengths).unsqueeze(0),
        -1.0e20,
    )

    if indexer.index_topk == 0:
        return torch.empty(batch_size, seq_len, 0, device=hidden_states.device, dtype=torch.int64)

    topk_count = min(indexer.index_topk, indexer.compressor.max_compressed_len)
    topk_idxs = index_score.topk(topk_count, dim=-1).indices
    invalid = topk_idxs >= valid_lengths.unsqueeze(0)
    topk_idxs = torch.where(invalid, -1, topk_idxs + offset)
    if topk_count < indexer.index_topk:
        pad = torch.full(
            (batch_size, seq_len, indexer.index_topk - topk_count),
            -1,
            device=hidden_states.device,
            dtype=topk_idxs.dtype,
        )
        topk_idxs = torch.cat((topk_idxs, pad), dim=-1)
    return topk_idxs.to(torch.int64)


def _ref_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, _ = q.shape
    batch_idx = torch.arange(batch_size, device=q.device).view(batch_size, 1, 1)
    batch_idx = batch_idx.expand(batch_size, seq_len, topk_idxs.shape[-1])

    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    gather_idxs = topk_idxs.to(torch.long).clamp(min=0)
    selected_kv = kv[batch_idx, gather_idxs].to(compute_dtype)
    logits = torch.matmul(q.to(compute_dtype), selected_kv.transpose(-1, -2))
    logits = logits * softmax_scale
    logits = logits.masked_fill((topk_idxs < 0).unsqueeze(2), float("-inf"))

    sink_logits = attn_sink.to(dtype=compute_dtype).view(1, 1, num_heads, 1)
    sink_logits = sink_logits.expand(batch_size, seq_len, num_heads, 1)
    weights = torch.softmax(torch.cat([logits, sink_logits], dim=-1), dim=-1)
    output = torch.matmul(weights[..., :-1], selected_kv)
    return output.to(q.dtype)


def _ref_window_topk_idxs(
    window_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(seq_len, device=device).unsqueeze(1)
    key_positions = query_positions - window_size + 1 + torch.arange(window_size, device=device)
    key_positions = torch.where(
        (key_positions < 0) | (key_positions > query_positions),
        -1,
        key_positions,
    )
    return key_positions.unsqueeze(0).expand(batch_size, -1, -1)


def _ref_compress_topk_idxs(
    ratio: int,
    batch_size: int,
    seq_len: int,
    offset: int,
    max_compressed_len: int,
    device: torch.device,
) -> torch.Tensor:
    compressed_positions = torch.arange(max_compressed_len, device=device)
    valid_lengths = torch.arange(1, seq_len + 1, device=device).unsqueeze(1) // ratio
    topk_idxs = compressed_positions.unsqueeze(0).expand(seq_len, -1)
    topk_idxs = torch.where(topk_idxs < valid_lengths, topk_idxs + offset, -1)
    return topk_idxs.unsqueeze(0).expand(batch_size, -1, -1)


# `transformers.models.deepseek_v4` is unavailable in the installed transformers package.
# The HFDeepseekV4* classes below are test-only, HF-style minimal faithful copies of the
# local DeepSeek V4 `inference/model.py` semantics for equivalence checks.
class HFDeepseekV4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ref_rmsnorm(x, self.weight, self.eps)


class HFDeepseekV4Compressor(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.rotate = rotate
        self.overlap = compress_ratio == 4
        channels = 2 if self.overlap else 1
        self.max_compressed_len = max(
            1,
            (int(config.ad_compress_max_seq_len) + compress_ratio - 1) // compress_ratio,
        )
        self.max_compressed_tokens = self.max_compressed_len * compress_ratio
        self.ape = nn.Parameter(
            torch.zeros(compress_ratio, channels * head_dim, dtype=torch.float32)
        )
        self.wkv = nn.Linear(config.hidden_size, channels * head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, channels * head_dim, bias=False)
        self.norm = HFDeepseekV4RMSNorm(head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _ref_compressor(self, hidden_states, cos_table, sin_table, position_ids)


class HFDeepseekV4Indexer(nn.Module):
    def __init__(self, config: DeepseekV4Config, compress_ratio: int) -> None:
        super().__init__()
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = nn.Linear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        self.compressor = HFDeepseekV4Compressor(
            config,
            compress_ratio,
            config.index_head_dim,
            rotate=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        position_ids: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        return _ref_indexer(
            self,
            hidden_states,
            q_lora,
            cos,
            sin,
            cos_table,
            sin_table,
            position_ids,
            offset,
        )


class HFDeepseekV4Attention(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.n_groups = config.o_groups
        self.window_size = config.sliding_window
        self.compress_ratio = int(config.compress_ratios[layer_idx])
        self.rms_eps = config.rms_norm_eps
        self.softmax_scale = self.head_dim**-0.5
        self.group_head_width = (self.n_heads * self.head_dim) // self.n_groups

        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = HFDeepseekV4RMSNorm(self.q_lora_rank, self.rms_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = HFDeepseekV4RMSNorm(self.head_dim, self.rms_eps)
        self.wo_a = nn.Linear(self.group_head_width, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        if self.compress_ratio:
            self.compressor = HFDeepseekV4Compressor(config, self.compress_ratio, self.head_dim)
            self.indexer = (
                HFDeepseekV4Indexer(config, self.compress_ratio)
                if self.compress_ratio == 4
                else None
            )
        else:
            self.compressor = None
            self.indexer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        cos_base_table, sin_base_table, cos_compress_table, sin_compress_table = position_embeddings
        cos_base = cos_base_table[position_ids]
        sin_base = sin_base_table[position_ids]
        cos_compress = cos_compress_table[position_ids]
        sin_compress = sin_compress_table[position_ids]
        cos = cos_compress if self.compress_ratio else cos_base
        sin = sin_compress if self.compress_ratio else sin_base

        q_lora = self.q_norm(_linear_ref(hidden_states, self.wq_a))
        q = _linear_ref(q_lora, self.wq_b).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_eps).to(q.dtype)

        kv = self.kv_norm(_linear_ref(hidden_states, self.wkv))
        q_nope, q_pe = torch.split(q, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        kv_nope, kv_pe = torch.split(kv, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = _interleaved_rope(q_pe, cos.unsqueeze(2), sin.unsqueeze(2))
        kv_pe = _interleaved_rope(kv_pe, cos, sin)
        kv_nope = _ref_fake_fp8_act_quant(kv_nope, block_size=64)
        q = torch.cat((q_nope, q_pe), dim=-1)
        kv = torch.cat((kv_nope, kv_pe), dim=-1)

        if self.compress_ratio:
            window_idxs = _ref_window_topk_idxs(
                self.window_size, batch_size, seq_len, hidden_states.device
            )
            compressed_kv = self.compressor(
                hidden_states, cos_compress_table, sin_compress_table, position_ids
            )
            if self.indexer is not None:
                compressed_idxs = self.indexer(
                    hidden_states,
                    q_lora,
                    cos_compress,
                    sin_compress,
                    cos_compress_table,
                    sin_compress_table,
                    position_ids,
                    seq_len,
                )
            else:
                compressed_idxs = _ref_compress_topk_idxs(
                    self.compress_ratio,
                    batch_size,
                    seq_len,
                    seq_len,
                    self.compressor.max_compressed_len,
                    hidden_states.device,
                )
            kv_for_attention = torch.cat((kv, compressed_kv), dim=1)
            topk_idxs = torch.cat((window_idxs, compressed_idxs), dim=-1).to(torch.int64)
            attn_output = _ref_sparse_attention(
                q, kv_for_attention, self.attn_sink, topk_idxs, self.softmax_scale
            )
        else:
            kv_heads = kv.unsqueeze(2).expand(batch_size, seq_len, self.n_heads, self.head_dim)
            q_bh = q.transpose(1, 2).float()
            kv_bh = kv_heads.transpose(1, 2).float()
            scores = torch.matmul(q_bh, kv_bh.transpose(-1, -2))
            scores = scores * self.softmax_scale
            query_pos = position_ids[:, :, None]
            key_pos = position_ids[:, None, :]
            causal = key_pos <= query_pos
            window = (query_pos - key_pos) < self.window_size
            scores = scores.masked_fill(~(causal.unsqueeze(1) & window.unsqueeze(1)), float("-inf"))
            sink_logits = self.attn_sink.float().view(1, self.n_heads, 1, 1)
            sink_logits = sink_logits.expand(batch_size, self.n_heads, seq_len, 1)
            weights = torch.softmax(torch.cat([scores, sink_logits], dim=-1), dim=-1)[..., :-1]
            attn_output = torch.matmul(weights, kv_bh).transpose(1, 2).to(hidden_states.dtype)

        out_nope, out_pe = torch.split(
            attn_output,
            [self.nope_head_dim, self.rope_head_dim],
            dim=-1,
        )
        out_pe = _interleaved_rope(out_pe, cos.unsqueeze(2), sin.unsqueeze(2), inverse=True)
        attn_output = torch.cat((out_nope, out_pe), dim=-1)
        attn_output = attn_output.view(batch_size, seq_len, self.n_groups, -1)
        wo_a = self.wo_a.weight.float().view(self.n_groups, self.o_lora_rank, -1)
        attn_output = torch.matmul(
            attn_output.float().unsqueeze(-2),
            wo_a.transpose(-1, -2),
        ).squeeze(-2)
        attn_output = attn_output.flatten(2)
        return _linear_ref(attn_output.to(hidden_states.dtype), self.wo_b)


def _ref_dense_attention(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    config: DeepseekV4Config,
) -> torch.Tensor:
    bsz, seq_len, _ = hidden_states.shape
    num_heads = config.num_attention_heads
    head_dim = config.head_dim
    rope_dim = config.qk_rope_head_dim
    nope_dim = head_dim - rope_dim

    q_lora = _ref_rmsnorm(
        _linear_ref(hidden_states, attn.wq_a),
        _get_linear(attn, "q_norm").weight,
        config.rms_norm_eps,
    )
    q = _linear_ref(q_lora, attn.wq_b).view(bsz, seq_len, num_heads, head_dim)
    q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + config.rms_norm_eps).to(q.dtype)

    kv = _ref_rmsnorm(
        _linear_ref(hidden_states, attn.wkv),
        _get_linear(attn, "kv_norm").weight,
        config.rms_norm_eps,
    ).view(bsz, seq_len, 1, head_dim)

    cos, sin = _rope_cos_sin(position_ids, rope_dim, config.rope_theta)
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_nope, q_pe = torch.split(q, [nope_dim, rope_dim], dim=-1)
    kv_nope, kv_pe = torch.split(kv, [nope_dim, rope_dim], dim=-1)
    q = torch.cat([q_nope, _interleaved_rope(q_pe, cos, sin)], dim=-1)
    kv = torch.cat([kv_nope, _interleaved_rope(kv_pe, cos, sin)], dim=-1)

    q_bh = q.transpose(1, 2).float()
    kv_bh = kv.expand(bsz, seq_len, num_heads, head_dim).transpose(1, 2).float()
    scores = torch.matmul(q_bh, kv_bh.transpose(-2, -1)) * (head_dim**-0.5)
    query_pos = position_ids[:, :, None]
    key_pos = position_ids[:, None, :]
    causal = key_pos <= query_pos
    window = (query_pos - key_pos) < config.sliding_window
    scores = scores.masked_fill(~(causal.unsqueeze(1) & window.unsqueeze(1)), float("-inf"))

    sink_logits = attn.attn_sink.float().view(1, num_heads, 1, 1)
    sink_logits = sink_logits.expand(bsz, num_heads, seq_len, 1)
    weights = torch.softmax(torch.cat([scores, sink_logits], dim=-1), dim=-1)[..., :-1]
    out = torch.matmul(weights, kv_bh).transpose(1, 2).to(hidden_states.dtype)

    out_nope, out_pe = torch.split(out, [nope_dim, rope_dim], dim=-1)
    out = torch.cat([out_nope, _interleaved_rope(out_pe, cos, sin, inverse=True)], dim=-1)

    group_count = config.o_groups
    group_width = (num_heads * head_dim) // group_count
    out = out.reshape(bsz, seq_len, group_count, group_width)
    wo_a_weight = attn.wo_a.weight.float().view(group_count, config.o_lora_rank, group_width)
    out = torch.matmul(out.float().unsqueeze(-2), wo_a_weight.transpose(-1, -2)).squeeze(-2)
    return _linear_ref(out.reshape(bsz, seq_len, group_count * config.o_lora_rank), attn.wo_b)


def _call_attention(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    config: DeepseekV4Config,
) -> torch.Tensor:
    signature = inspect.signature(attn.forward)
    if "position_embeddings" in signature.parameters:
        return attn(hidden_states, _position_embeddings(config, position_ids), position_ids)
    if "position_ids" in signature.parameters:
        return attn(hidden_states, position_ids)

    cos, sin = _rope_cos_sin(position_ids, config.qk_rope_head_dim, config.rope_theta)
    cos_comp, sin_comp = _rope_cos_sin(
        position_ids, config.qk_rope_head_dim, config.compress_rope_theta
    )
    return attn(hidden_states, cos, sin, cos_comp, sin_comp, cos_comp[0], sin_comp[0])


def _decoder(model: DeepseekV4ForCausalLM) -> nn.Module:
    return getattr(model, "model", model)


def test_small_config_exercises_flash_semantics() -> None:
    config = _small_config()
    model = DeepseekV4ForCausalLM(config).eval()
    layers = _decoder(model).layers

    assert config.compress_ratios[0] == 0
    assert set(config.compress_ratios[1:]) == {4, 128}
    assert config.num_hash_layers == 1
    assert config.n_shared_experts == 1
    assert config.hc_mult == 2
    assert config.swiglu_limit > 0
    assert all(hasattr(layer.attn, "attn_sink") for layer in layers)
    assert hasattr(layers[0].ffn.gate, "tid2eid")
    assert getattr(layers[1].ffn.gate, "bias", None) is not None
    assert hasattr(layers[0].ffn.experts, "gate_up_proj_blocks")
    assert layers[0].ffn.experts.gate_up_proj_blocks.dtype == torch.uint8


def test_load_hook_preserves_converted_keys_and_drops_mtp_layer() -> None:
    config = _small_config(num_hidden_layers=1, compress_ratios=(0,), num_hash_layers=0)
    model = DeepseekV4ForCausalLM(config).eval()
    embed = _sequential_tensor(model.embed.weight.shape)

    result = model.load_state_dict(
        {
            "embed.weight": embed,
            "mtp.0.e_proj.weight": torch.ones(2, 2),
        },
        strict=False,
    )

    assert result.unexpected_keys == []
    torch.testing.assert_close(model.embed.weight, embed, rtol=0, atol=0)


def test_load_hook_maps_wrapper_root_keys_and_shared_expert_aliases() -> None:
    config = _small_config(num_hidden_layers=1, compress_ratios=(0,), num_hash_layers=0)
    model = DeepseekV4ForCausalLM(config).eval()
    embed = _sequential_tensor(model.embed.weight.shape)
    head = _sequential_tensor(model.head.weight.shape, offset=10.0)
    norm = _sequential_tensor(model.norm.weight.shape, offset=20.0)
    shared_w1 = _sequential_tensor(model.layers[0].ffn.shared_experts.w1.weight.shape, offset=30.0)
    shared_w3 = _sequential_tensor(model.layers[0].ffn.shared_experts.w3.weight.shape, offset=40.0)
    shared_w2 = _sequential_tensor(model.layers[0].ffn.shared_experts.w2.weight.shape, offset=50.0)

    result = model.load_state_dict(
        {
            "model.embed_tokens.weight": embed,
            "lm_head.weight": head,
            "model.norm.weight": norm,
            "model.layers.0.ffn.shared_experts.gate_proj.weight": shared_w1,
            "model.layers.0.ffn.shared_experts.up_proj.weight": shared_w3,
            "model.layers.0.ffn.shared_experts.down_proj.weight": shared_w2,
        },
        strict=False,
    )

    assert result.unexpected_keys == []
    torch.testing.assert_close(model.embed.weight, embed, rtol=0, atol=0)
    torch.testing.assert_close(model.head.weight, head, rtol=0, atol=0)
    torch.testing.assert_close(model.norm.weight, norm, rtol=0, atol=0)
    torch.testing.assert_close(
        model.layers[0].ffn.shared_experts.w1.weight, shared_w1, rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.layers[0].ffn.shared_experts.w3.weight, shared_w3, rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.layers[0].ffn.shared_experts.w2.weight, shared_w2, rtol=0, atol=0
    )


def test_rotary_embedding_returns_full_cached_tables() -> None:
    config = _small_config(ad_rope_cache_len=16, max_position_embeddings=16)
    rotary = dsv4.DeepseekV4RotaryEmbedding(config)
    position_ids = _position_ids(2, 5, "cpu")

    cos_base, sin_base, cos_compress, sin_compress = rotary()

    assert cos_base.shape == (config.ad_rope_cache_len, config.qk_rope_head_dim // 2)
    assert sin_base.shape == cos_base.shape
    assert cos_compress.shape == cos_base.shape
    assert sin_compress.shape == cos_base.shape
    assert cos_base[position_ids].shape == (2, 5, config.qk_rope_head_dim // 2)


def test_rmsnorm_matches_reference() -> None:
    norm = DeepseekV4RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 16))
    x = torch.randn(2, 3, 16)

    actual = norm(x)
    expected = _ref_rmsnorm(x, norm.weight, 1e-6)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_mlp_swiglu_limit_matches_reference() -> None:
    mlp = DeepseekV4MLP(16, 12, swiglu_limit=0.5).eval()
    _set_mlp_weights(mlp)
    x = torch.linspace(-2.0, 2.0, 2 * 3 * 16).view(2, 3, 16)

    actual = mlp(x)
    expected = _ref_mlp(mlp, x)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_deepseek_v4_chat_prompt_format_matches_reference() -> None:
    assert (
        dsv4._format_deepseek_v4_chat_messages(
            [{"role": "user", "content": "How big is the universe? "}]
        )
        == "<｜begin▁of▁sentence｜><｜User｜>How big is the universe? <｜Assistant｜></think>"
    )
    assert dsv4._format_deepseek_v4_chat_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u2"},
        ]
    ) == (
        "<｜begin▁of▁sentence｜>sys<｜User｜>u<｜Assistant｜></think>"
        "a<｜end▁of▁sentence｜><｜User｜>u2<｜Assistant｜></think>"
    )


def test_moe_shared_experts_use_configured_swiglu_limit() -> None:
    config = _small_config(num_hidden_layers=1, compress_ratios=(0,))
    assert config.swiglu_limit > 0
    moe = DeepseekV4MoE(config, layer_idx=0).eval()
    _set_mlp_weights(moe.shared_experts, offset=0.4)
    x = torch.linspace(-2.0, 2.0, 2 * 3 * config.hidden_size).view(2, 3, config.hidden_size)

    assert moe.shared_experts.swiglu_limit == config.swiglu_limit
    actual = moe.shared_experts(x)
    expected = _ref_mlp(moe.shared_experts, x, swiglu_limit=config.swiglu_limit)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_router_hash_and_score_routing_match_reference() -> None:
    assert DeepseekV4Router is not None, "DeepSeek V4 router class must be exposed for tests"
    config = _small_config()
    hidden_states = torch.linspace(-1.0, 1.0, 5 * config.hidden_size).view(5, config.hidden_size)
    input_ids = torch.tensor([0, 1, 2, 3, 4])

    hash_router = DeepseekV4Router(config, layer_idx=0).eval()
    _set_router_weights(hash_router)
    expected_hash = _ref_router(hash_router, hidden_states, input_ids, config, hash_routing=True)
    actual_hash = hash_router(hidden_states, input_ids)
    torch.testing.assert_close(actual_hash[0], expected_hash[0])
    torch.testing.assert_close(actual_hash[1], expected_hash[1], rtol=1e-6, atol=1e-6)

    score_router = DeepseekV4Router(config, layer_idx=config.num_hash_layers).eval()
    _set_router_weights(score_router)
    expected_score = _ref_router(score_router, hidden_states, input_ids, config, hash_routing=False)
    actual_score = score_router(hidden_states, input_ids)
    torch.testing.assert_close(actual_score[0], expected_score[0])
    torch.testing.assert_close(actual_score[1], expected_score[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["hash-routing", "score-routing"])
def test_moe_uses_packed_mxfp4_routed_experts(layer_idx: int) -> None:
    config = _small_config(num_hidden_layers=2, compress_ratios=(0, 0))
    moe = DeepseekV4MoE(config, layer_idx=layer_idx).eval()
    _set_router_weights(moe.gate)
    _set_mlp_weights(moe.shared_experts, offset=0.4)

    hidden_states = torch.linspace(-1.5, 1.5, 2 * 3 * config.hidden_size).view(
        2, 3, config.hidden_size
    )
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])

    actual = moe(hidden_states, input_ids)

    assert actual.shape == hidden_states.shape
    assert torch.isfinite(actual).all()
    assert moe.experts.gate_up_proj_blocks.shape == (
        config.n_routed_experts,
        2 * config.moe_intermediate_size,
        config.hidden_size // 32,
        16,
    )
    assert moe.experts.down_proj_blocks.shape == (
        config.n_routed_experts,
        config.hidden_size,
        config.moe_intermediate_size // 32,
        16,
    )


def test_dense_attention_with_sinks_matches_reference() -> None:
    config = _small_config(num_hidden_layers=1, compress_ratios=(0,), num_hash_layers=0)
    attn = DeepseekV4Attention(config, layer_idx=0).eval()
    with torch.no_grad():
        attn.attn_sink.copy_(torch.tensor([-0.5, 0.0, 0.25, 0.75]))
    hidden_states = torch.randn(2, 5, config.hidden_size)
    position_ids = _position_ids(2, 5, hidden_states.device)

    actual = _call_attention(attn, hidden_states, position_ids, config)
    expected = _ref_dense_attention(attn, hidden_states, position_ids, config)

    assert_rmse_close(actual, expected, rmse_ratio_tol=0.10, msg="Dense attention: ")


def test_torch_attention_with_sinks_accepts_bfloat16_inputs() -> None:
    query = torch.randn(2, 5, 4, 8, dtype=torch.bfloat16)
    key = torch.randn(2, 5, 4, 8, dtype=torch.bfloat16)
    value = torch.randn(2, 5, 4, 8, dtype=torch.bfloat16)
    sinks = torch.tensor([-0.5, 0.0, 0.25, 0.75], dtype=torch.bfloat16)

    actual = torch.ops.auto_deploy.torch_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
        scale=0.5,
        sinks=sinks,
        sliding_window=4,
        layout="bsnd",
    )
    expected = torch.ops.auto_deploy.torch_attention(
        query.float(),
        key.float(),
        value.float(),
        dropout_p=0.0,
        is_causal=True,
        scale=0.5,
        sinks=sinks.float(),
        sliding_window=4,
        layout="bsnd",
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)


def test_compressor_matches_independent_reference_with_fake_fp8_nope_path() -> None:
    config = _small_config(
        hidden_size=8,
        num_attention_heads=1,
        head_dim=128,
        qk_rope_head_dim=64,
        o_groups=1,
        compress_ratios=(2,),
        ad_rope_cache_len=16,
        ad_compress_max_seq_len=6,
        max_position_embeddings=16,
    )
    compressor = DeepseekV4Compressor(config, compress_ratio=2, head_dim=config.head_dim).eval()
    compressor = compressor.to(torch.bfloat16)
    _copy_deterministic_linear(compressor.wkv, offset=0.0)
    _copy_deterministic_linear(compressor.wgate, offset=0.17)
    with torch.no_grad():
        compressor.ape.copy_(
            torch.linspace(-0.2, 0.3, compressor.ape.numel()).view_as(compressor.ape)
        )
        compressor.norm.weight.copy_(torch.linspace(0.6, 1.4, config.head_dim))

    hidden_states = torch.linspace(-1.5, 1.75, 2 * 5 * config.hidden_size)
    hidden_states = hidden_states.view(2, 5, config.hidden_size).to(torch.bfloat16)
    position_ids = _position_ids(2, 5, hidden_states.device)
    table_positions = torch.arange(config.ad_rope_cache_len, device=hidden_states.device)
    cos_table, sin_table = _rope_cos_sin(
        table_positions,
        config.qk_rope_head_dim,
        config.compress_rope_theta,
    )

    assert compressor.head_dim - compressor.rope_head_dim == 64
    actual = compressor(hidden_states, cos_table, sin_table, position_ids)
    expected = _ref_compressor(compressor, hidden_states, cos_table, sin_table, position_ids)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=0.01)


def test_ratio4_indexer_matches_independent_reference_and_masks_invalid_prefix() -> None:
    config = _small_config(
        hidden_size=8,
        q_lora_rank=8,
        num_attention_heads=1,
        head_dim=64,
        qk_rope_head_dim=32,
        index_n_heads=1,
        index_head_dim=64,
        index_topk=3,
        compress_ratios=(4,),
        ad_rope_cache_len=16,
        ad_compress_max_seq_len=12,
        max_position_embeddings=16,
    )
    indexer = DeepseekV4Indexer(config, compress_ratio=4).eval()
    _copy_deterministic_linear(indexer.wq_b, offset=0.11)
    _copy_deterministic_linear(indexer.weights_proj, offset=-0.09)
    _copy_deterministic_linear(indexer.compressor.wkv, offset=0.03)
    _copy_deterministic_linear(indexer.compressor.wgate, offset=0.21)
    with torch.no_grad():
        indexer.compressor.ape.copy_(
            torch.linspace(-0.25, 0.35, indexer.compressor.ape.numel()).view_as(
                indexer.compressor.ape
            )
        )
        indexer.compressor.norm.weight.copy_(torch.linspace(0.7, 1.3, config.index_head_dim))

    hidden_states = torch.linspace(-1.4, 1.6, 2 * 5 * config.hidden_size).view(
        2, 5, config.hidden_size
    )
    q_lora = torch.linspace(-0.9, 1.1, 2 * 5 * config.q_lora_rank).view(2, 5, config.q_lora_rank)
    position_ids = _position_ids(2, 5, hidden_states.device)
    cos, sin = _rope_cos_sin(position_ids, config.qk_rope_head_dim, config.compress_rope_theta)
    table_positions = torch.arange(config.ad_rope_cache_len, device=hidden_states.device)
    cos_table, sin_table = _rope_cos_sin(
        table_positions,
        config.qk_rope_head_dim,
        config.compress_rope_theta,
    )
    offset = 10

    assert config.index_head_dim % 32 == 0
    assert config.index_head_dim & (config.index_head_dim - 1) == 0
    actual = indexer(
        hidden_states,
        q_lora,
        cos,
        sin,
        cos_table,
        sin_table,
        position_ids,
        offset,
    )
    expected = _ref_indexer(
        indexer,
        hidden_states,
        q_lora,
        cos,
        sin,
        cos_table,
        sin_table,
        position_ids,
        offset,
    )

    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[:, :3], torch.full_like(actual[:, :3], -1))
    assert (actual[:, 3:] >= 0).sum(dim=-1).eq(1).all()


def test_export_dynamic_shapes_finite_logits_and_expected_ops() -> None:
    config = _small_config(num_hidden_layers=2, compress_ratios=(0, 4), num_hash_layers=1)
    model = DeepseekV4ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    position_ids = _position_ids(2, 6, input_ids.device)

    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": position_ids},
        dynamic_shapes={
            "input_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            "position_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        },
        num_moe_experts_for_export=2,
    )

    with torch.no_grad():
        eager = model(input_ids=input_ids, position_ids=position_ids).logits
        exported = _output_logits(gm(input_ids, position_ids=position_ids))
    assert torch.isfinite(exported).all()
    assert_rmse_close(exported, eager, rmse_ratio_tol=0.05, msg="Export first shape: ")

    input_ids_2 = torch.randint(0, config.vocab_size, (1, 4))
    position_ids_2 = _position_ids(1, 4, input_ids_2.device)
    with torch.no_grad():
        out_2 = _output_logits(gm(input_ids_2, position_ids=position_ids_2))
    assert out_2.shape == (1, 4, config.vocab_size)
    assert torch.isfinite(out_2).all()

    target_names = [str(node.target) for node in gm.graph.nodes if node.op == "call_function"]
    assert target_names.count("auto_deploy.torch_deepseek_v4_sparse_attention.default") == 2
    assert "auto_deploy.torch_linear_simple.default" in target_names
    assert "auto_deploy.torch_mxfp4_moe_from_routing.default" in target_names
    assert "auto_deploy.torch_moe.default" not in target_names
    assert "auto_deploy.torch_attention.default" not in target_names
    non_torch_ad_ops = sorted(
        name
        for name in set(target_names)
        if name.startswith("auto_deploy.") and not name.startswith("auto_deploy.torch_")
    )
    assert non_torch_ad_ops == [
        "auto_deploy.all_reduce.default",
        "auto_deploy.view.default",
    ]


def test_export_mxfp4_experts_apply_sharding_hints_rank1_ep_graph() -> None:
    config = _small_config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        o_groups=2,
        o_lora_rank=8,
        compress_ratios=(0,),
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=0,
        ad_rope_cache_len=16,
        ad_compress_max_seq_len=16,
        max_position_embeddings=16,
    )
    model = DeepseekV4ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 4))
    position_ids = _position_ids(2, 4, input_ids.device)

    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": position_ids},
        strict=False,
    )

    base_op = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing.default
    ep_op = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing_ep.default
    assert len(_call_nodes(gm, base_op)) == 1

    gm_out = _make_apply_sharding_hints_optimizer(
        world_size=2,
        rank=1,
        dist_backend="torch",
    )(None, gm)

    assert len(_call_nodes(gm_out, base_op)) == 0
    ep_nodes = _call_nodes(gm_out, ep_op)
    assert len(ep_nodes) == 1
    ep_node = ep_nodes[0]

    [expert_start, gate_up_blocks] = extract_op_args(
        ep_node,
        "expert_start",
        "gate_up_blocks",
    )
    assert expert_start == 2
    assert getattr(gate_up_blocks, "op", None) == "get_attr"
    assert gm_out.get_buffer(gate_up_blocks.target).shape[0] == 2

    dist_all_reduce_op = torch.ops.auto_deploy.torch_dist_all_reduce.default
    all_reduce_nodes = _call_nodes(gm_out, dist_all_reduce_op)
    assert len(all_reduce_nodes) == 3
    assert any(node.args[0] is ep_node for node in all_reduce_nodes)
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.all_reduce.default)) == 0

    layer = gm_out.get_submodule("layers.0")
    assert layer.attn.attn_sink.shape == (config.num_attention_heads // 2,)
    assert layer.attn.wq_b.weight.shape == (
        config.num_attention_heads * config.head_dim // 2,
        config.q_lora_rank,
    )
    assert layer.attn.wo_a.weight.shape == (
        config.o_lora_rank,
        config.num_attention_heads * config.head_dim // config.o_groups,
    )
    assert layer.attn.wo_b.weight.shape == (
        config.hidden_size,
        config.o_groups * config.o_lora_rank // 2,
    )

    sparse_nodes = _call_nodes(
        gm_out,
        torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention.default,
    )
    assert len(sparse_nodes) == 1
    attn_sink = sparse_nodes[0].args[2]
    assert getattr(attn_sink, "op", None) == "get_attr"
    assert gm_out.get_parameter(attn_sink.target).shape == (config.num_attention_heads // 2,)

    group_width = config.num_attention_heads * config.head_dim // config.o_groups
    view_shapes = [
        extract_op_args(node, "shape")[0]
        for node in _call_nodes(gm_out, torch.ops.auto_deploy.view.default)
    ]
    assert [2, 4, -1, config.head_dim] in view_shapes
    assert [2, 4, -1, group_width] in view_shapes
    assert [-1, config.o_lora_rank, group_width] in view_shapes


def test_export_mxfp4_experts_apply_sharding_hints_rank6_ep8_tp8_graph() -> None:
    config = _small_config(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        o_groups=8,
        o_lora_rank=8,
        compress_ratios=(0,),
        moe_intermediate_size=64,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=0,
        ad_rope_cache_len=16,
        ad_compress_max_seq_len=16,
        max_position_embeddings=16,
    )
    model = DeepseekV4ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 4))
    position_ids = _position_ids(2, 4, input_ids.device)

    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": position_ids},
        strict=False,
    )

    base_op = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing.default
    ep_op = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing_ep.default
    assert len(_call_nodes(gm, base_op)) == 1

    gm_out = _make_apply_sharding_hints_optimizer(
        world_size=8,
        rank=6,
        dist_backend="torch",
    )(None, gm)

    assert len(_call_nodes(gm_out, base_op)) == 0
    ep_nodes = _call_nodes(gm_out, ep_op)
    assert len(ep_nodes) == 1
    ep_node = ep_nodes[0]

    [expert_start, gate_up_blocks, down_blocks] = extract_op_args(
        ep_node,
        "expert_start",
        "gate_up_blocks",
        "down_blocks",
    )
    local_expert_count = config.n_routed_experts // 8
    assert expert_start == 192
    assert local_expert_count == 32
    assert getattr(gate_up_blocks, "op", None) == "get_attr"
    assert getattr(down_blocks, "op", None) == "get_attr"
    assert gm_out.get_buffer(gate_up_blocks.target).shape[0] == local_expert_count
    assert gm_out.get_buffer(down_blocks.target).shape[0] == local_expert_count

    dist_all_reduce_op = torch.ops.auto_deploy.torch_dist_all_reduce.default
    all_reduce_nodes = _call_nodes(gm_out, dist_all_reduce_op)
    assert any(node.args[0] is ep_node for node in all_reduce_nodes)
    assert len(_call_nodes(gm_out, torch.ops.auto_deploy.all_reduce.default)) == 0

    layer = gm_out.get_submodule("layers.0")
    local_head_count = config.num_attention_heads // 8
    assert local_head_count == 1
    assert layer.attn.attn_sink.shape == (local_head_count,)
    assert layer.attn.wq_b.weight.shape == (
        local_head_count * config.head_dim,
        config.q_lora_rank,
    )
    assert layer.attn.wo_a.weight.shape == (
        config.o_lora_rank,
        config.num_attention_heads * config.head_dim // config.o_groups,
    )
    assert layer.attn.wo_b.weight.shape == (
        config.hidden_size,
        config.o_groups * config.o_lora_rank // 8,
    )

    sparse_nodes = _call_nodes(
        gm_out,
        torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention.default,
    )
    assert len(sparse_nodes) == 1
    attn_sink = sparse_nodes[0].args[2]
    assert getattr(attn_sink, "op", None) == "get_attr"
    assert gm_out.get_parameter(attn_sink.target).shape == (local_head_count,)


def test_ratio4_sparse_attention_sharding_only_splits_sink() -> None:
    config = _small_config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        o_groups=2,
        o_lora_rank=8,
        compress_ratios=(4,),
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=0,
        ad_rope_cache_len=16,
        ad_compress_max_seq_len=16,
        max_position_embeddings=16,
    )
    model = DeepseekV4ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 4))
    position_ids = _position_ids(2, 4, input_ids.device)

    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": position_ids},
        strict=False,
    )
    gm_out = _make_apply_sharding_hints_optimizer(
        world_size=2,
        rank=1,
        dist_backend="torch",
    )(None, gm)

    layer = gm_out.get_submodule("layers.0")
    assert layer.attn.attn_sink.shape == (config.num_attention_heads // 2,)
    assert layer.attn.compressor.norm.weight.shape == (config.head_dim,)
    assert layer.attn.indexer.compressor.norm.weight.shape == (config.index_head_dim,)

    sparse_nodes = _call_nodes(
        gm_out,
        torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention.default,
    )
    assert len(sparse_nodes) == 1
    attn_sink = sparse_nodes[0].args[2]
    compressor_norm = sparse_nodes[0].args[7]
    indexer_norm = sparse_nodes[0].args[16]
    assert getattr(attn_sink, "op", None) == "get_attr"
    assert getattr(compressor_norm, "op", None) == "get_attr"
    assert getattr(indexer_norm, "op", None) == "get_attr"
    assert gm_out.get_parameter(attn_sink.target).shape == (config.num_attention_heads // 2,)
    assert gm_out.get_parameter(compressor_norm.target).shape == (config.head_dim,)
    assert gm_out.get_parameter(indexer_norm.target).shape == (config.index_head_dim,)


def test_factory_registration() -> None:
    assert custom_models.DeepseekV4ForCausalLM is DeepseekV4ForCausalLM
    assert "DeepseekV4ForCausalLM" in custom_models.__all__
    assert (
        AutoModelForCausalLMFactory._custom_model_mapping["DeepseekV4Config"]
        is DeepseekV4ForCausalLM
    )
    factory_cls = getattr(dsv4, "DeepseekV4AutoModelForCausalLMFactory", None)
    assert factory_cls is not None, "DeepseekV4AutoModelForCausalLMFactory must be exposed"
    assert factory_cls._custom_model_mapping["DeepseekV4Config"] is DeepseekV4ForCausalLM
    assert ModelFactoryRegistry.get("DeepseekV4AutoModelForCausalLM") is factory_cls


def test_factory_example_inputs_cover_forward_contract(tmp_path) -> None:
    config = _small_config(
        num_hidden_layers=2,
        compress_ratios=(0, 4),
        ad_rope_cache_len=16,
        ad_compress_max_seq_len=12,
        max_position_embeddings=32,
    )
    config.save_pretrained(tmp_path)
    factory_cls = ModelFactoryRegistry.get("DeepseekV4AutoModelForCausalLM")
    factory = factory_cls(model=str(tmp_path), max_seq_len=32)

    example_inputs = factory.get_example_inputs()

    assert set(example_inputs) == {"input_ids", "position_ids"}
    input_ids = example_inputs["input_ids"]
    position_ids = example_inputs["position_ids"]
    assert input_ids.dtype == torch.long
    assert position_ids.dtype == torch.long
    assert input_ids.shape == (2, 8)
    assert position_ids.shape == input_ids.shape
    assert torch.equal(position_ids, _position_ids(2, 8, input_ids.device))

    model = DeepseekV4ForCausalLM(config).eval()
    with torch.no_grad():
        logits = model(**example_inputs).logits
    assert logits.shape == (2, 8, config.vocab_size)
