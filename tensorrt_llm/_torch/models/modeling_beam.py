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
"""Beam decoder components for the PyTorch backend.

Beam attention is grouped-query attention with an explicit ``head_dim`` (the query
width is independent of ``hidden_size``), learned per-head RMSNorm on Q and K, NeoX
RoPE on sliding-window layers, Scalable-Softmax query scaling on full (NoPE) layers,
and a sigmoid output gate with a per-head learned bias. Decoder blocks use sandwich
norms (four RMSNorms) and depth-scaled residual adds. Sparse layers route with a
sigmoid grouped top-k router whose fp32 correction bias only affects selection.

The transforms reproduce the reference model's dtype staging exactly:

* RMSNorm computes the statistics, normalization and affine multiply in fp32 and
  rounds once to the activation dtype.
* RoPE builds cos/sin in fp32 and rotates in fp32 before one cast; SSMax scales the
  query in fp32 before one cast.
* The output gate adds its bias, applies the sigmoid and multiplies in the
  activation dtype.
* SwiGLU applies SiLU(gate) * up in fp32 and casts before the down projection.
* The router GEMM, scores, selection and weights stay fp32. Each weighted expert
  route is rounded to the activation dtype, routes are accumulated in fp32 in
  ascending expert order, and the routed scaling factor is applied in fp32.
* The depth residual multiplies and adds in the activation dtype.
* The logit softcap runs in fp32 on the LM-head logits.

Projections, the paged KV cache and the attention core run through the standard
``Attention`` module and the selected attention backend; Q/K/V and the gate are
packed into the fused projection (per head ``[q_h, gate_h]``, then K, then V). The
cache holds post-norm, post-RoPE keys and projected values.
"""

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from ..attention.attention import Attention
from ..attention.backends import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.linear import Linear, TensorParallelMode
from .modeling_utils import DecoderModel, DecoderModelForCausalLM

BEAM_LAYER_TYPES = ("sliding_attention", "full_attention")
BEAM_MLP_LAYER_TYPES = ("dense", "sparse")
BEAM_ROPE_TYPES = ("default", "linear")
# Router parameters that keep their fp32 values regardless of the activation dtype.
BEAM_FP32_PARAMETER_SUFFIXES = ("mlp.gate.weight", "mlp.gate.e_score_correction_bias")


def _default_layer_types(num_hidden_layers: int) -> List[str]:
    layer_types = [
        "sliding_attention" if (i + 1) % 6 else "full_attention" for i in range(num_hidden_layers)
    ]
    if layer_types:
        layer_types[-1] = "full_attention"
    return layer_types


def _default_mlp_layer_types(num_hidden_layers: int) -> List[str]:
    num_dense = min(3, num_hidden_layers)
    return ["dense"] * num_dense + ["sparse"] * (num_hidden_layers - num_dense)


def _normalize_rope_parameters(raw: Optional[dict]) -> dict:
    """Validate Beam's sliding-RoPE / full-NoPE rope parameters."""
    if raw is None:
        return {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": None,
        }
    if not isinstance(raw, dict) or set(raw) != {"sliding_attention", "full_attention"}:
        raise ValueError(
            "rope_parameters must contain exactly 'sliding_attention' and 'full_attention': "
            f"{raw!r}"
        )
    if raw["full_attention"] is not None:
        raise ValueError("Beam full-attention layers run without rotary embeddings")
    sliding = raw["sliding_attention"]
    if not isinstance(sliding, dict):
        raise ValueError("sliding_attention rope parameters must be a dictionary")
    rope_type = sliding.get("rope_type", "default")
    if rope_type not in BEAM_ROPE_TYPES:
        raise ValueError(f"rope_type must be one of {BEAM_ROPE_TYPES}: {rope_type!r}")
    theta = sliding.get("rope_theta", 10000.0)
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise ValueError(f"rope_theta must be a number: {theta!r}")
    normalized = {"rope_type": rope_type, "rope_theta": float(theta)}
    if rope_type == "linear":
        factor = sliding.get("factor")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 0:
            raise ValueError(f"linear RoPE needs a positive factor: {factor!r}")
        normalized["factor"] = float(factor)
    return {"sliding_attention": normalized, "full_attention": None}


class BeamConfig(PretrainedConfig):
    """Architecture configuration of a Beam checkpoint (``config.json`` fields)."""

    model_type = "beam"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 5120,
        intermediate_size: int = 20480,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-6,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: Optional[int] = 512,
        moe_intermediate_size: int = 2560,
        num_experts_per_tok: int = 8,
        num_experts: int = 256,
        norm_topk_prob: bool = True,
        n_group: int = 8,
        topk_group: int = 4,
        routed_scaling_factor: float = 2.5,
        final_logit_softcapping: Optional[float] = 30.0,
        layer_types: Optional[List[str]] = None,
        mlp_layer_types: Optional[List[str]] = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype = dtype if dtype is not None else torch_dtype
        if dtype is None:
            dtype = torch.bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        kwargs.setdefault("architectures", ["BeamForCausalLM"])
        super().__init__(tie_word_embeddings=tie_word_embeddings, dtype=dtype, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = _normalize_rope_parameters(rope_parameters)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.final_logit_softcapping = final_logit_softcapping
        self.tie_word_embeddings = tie_word_embeddings
        self.layer_types = (
            list(layer_types) if layer_types else _default_layer_types(num_hidden_layers)
        )
        self.mlp_layer_types = (
            list(mlp_layer_types)
            if mlp_layer_types
            else _default_mlp_layer_types(num_hidden_layers)
        )
        if getattr(self, "torch_dtype", None) != dtype:
            try:
                self.torch_dtype = dtype
            except AttributeError:
                # A read-only torch_dtype property derives from ``dtype``.
                self.dtype = dtype
        if self.torch_dtype != dtype:
            raise ValueError(f"BeamConfig dtype {self.torch_dtype!r} != requested {dtype!r}")
        self._validate()

    def _validate(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError("mlp_layer_types length must equal num_hidden_layers")
        if set(self.layer_types) - set(BEAM_LAYER_TYPES):
            raise ValueError(f"unsupported layer_types: {sorted(set(self.layer_types))}")
        if set(self.mlp_layer_types) - set(BEAM_MLP_LAYER_TYPES):
            raise ValueError(f"unsupported mlp_layer_types: {sorted(set(self.mlp_layer_types))}")
        if self.hidden_act != "silu":
            raise ValueError(f"hidden_act must be 'silu': {self.hidden_act!r}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if "sliding_attention" in self.layer_types and not self.sliding_window:
            raise ValueError("sliding_attention layers need a positive sliding_window")
        if "sparse" in self.mlp_layer_types:
            if self.num_experts % self.n_group:
                raise ValueError("num_experts must be divisible by n_group")
            if not 1 <= self.topk_group <= self.n_group:
                raise ValueError("topk_group must be in [1, n_group]")
            if self.num_experts // self.n_group < 2:
                raise ValueError("Beam group scores sum the top-2 experts of each group")
            if self.num_experts_per_tok > self.topk_group * (self.num_experts // self.n_group):
                raise ValueError("num_experts_per_tok exceeds the experts of the selected groups")


# --------------------------------------------------------------------------- primitives


def beam_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim: fp32 statistics and affine, one cast to ``x.dtype``."""
    in_dtype = x.dtype
    x32 = x.float()
    rms = x32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x32 * rms
    return (normed * weight).to(in_dtype)


def beam_rope_cos_sin(
    positions: torch.Tensor, head_dim: int, rope_theta: float, factor: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(cos, sin)`` of shape ``positions.shape + (head_dim,)`` (NeoX halves).

    Linear position interpolation divides the inverse frequencies by ``factor``.
    """
    exponent = torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device)
    inv_freq = 1.0 / (rope_theta ** (exponent / head_dim))
    inv_freq = inv_freq / factor
    freqs = positions.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def beam_apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q``/``k`` of shape ``(..., heads, head_dim)`` in fp32, then cast back.

    ``cos``/``sin`` have shape ``(..., head_dim)`` and broadcast over the head axis.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q = q.float()
    k = k.float()
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out.to(q_dtype), k_out.to(k_dtype)


def beam_ssmax_scale(positions: torch.Tensor) -> torch.Tensor:
    """Scalable-Softmax query scale ``exp(0.1 * log1p((p + 1) / 2048))`` in fp32."""
    return torch.exp(0.1 * torch.log1p((positions.float() + 1) / 2048))


def beam_apply_ssmax(q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Scale ``q`` of shape ``positions.shape + (heads, head_dim)`` in fp32, then cast."""
    scale = beam_ssmax_scale(positions).unsqueeze(-1).unsqueeze(-1)
    return (q * scale).to(q.dtype)


def beam_output_gate(
    attention_output: torch.Tensor, gate: torch.Tensor, gate_bias: torch.Tensor
) -> torch.Tensor:
    """Sigmoid output gate on ``(..., heads, head_dim)`` tensors, in the activation dtype."""
    output_dtype = attention_output.dtype
    gate = gate + gate_bias.unsqueeze(-1)
    return (attention_output * torch.sigmoid(gate)).to(output_dtype)


def beam_depth_coefficients(layer_idx: int, is_attention: bool) -> Tuple[float, float]:
    """``(x_scale, residual_scale)`` for the attention or FFN add of a zero-based layer."""
    one_based = layer_idx + 1
    num_adds = 2 * one_based if is_attention else 2 * one_based + 1
    denom = math.sqrt(num_adds)
    return 1.0 / denom, math.sqrt(num_adds - 1) / denom


def beam_depth_residual(
    x: torch.Tensor, residual: torch.Tensor, scales: Tuple[float, float]
) -> torch.Tensor:
    """``x * x_scale + residual * residual_scale`` in the activation dtype."""
    x_scale, residual_scale = scales
    return x * x_scale + residual * residual_scale


def beam_softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """``tanh(logits / cap) * cap`` computed in fp32."""
    return (logits.float() / cap).tanh() * cap


class BeamRMSNorm(nn.Module):
    """RMSNorm whose affine weight is stored in the activation dtype."""

    def __init__(self, dim: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return beam_rms_norm(x, self.weight, self.eps)


class BeamSwiGLU(nn.Module):
    """Bias-free SwiGLU with the activation evaluated in fp32."""

    def __init__(self, model_config: ModelConfig, hidden_size: int, intermediate_size: int):
        super().__init__()
        dtype = model_config.pretrained_config.torch_dtype
        quant_config = model_config.get_quant_config()
        skip = model_config.skip_create_weights_in_init
        self.gate_proj = Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip,
        )
        self.up_proj = Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip,
        )
        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = (F.silu(gate.float()) * up.float()).to(gate.dtype)
        return self.down_proj(activated)


class BeamRouting(NamedTuple):
    """Router outputs and the intermediates that define them (all fp32 except indices)."""

    logits: torch.Tensor
    scores: torch.Tensor
    group_scores: torch.Tensor
    top_groups: torch.Tensor
    excluded: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class BeamRouter(nn.Module):
    """Sigmoid grouped top-k router; the correction bias only affects selection.

    Selection: sigmoid scores plus the correction bias, group score = sum of the top-2
    corrected scores in each group, sorted top-``topk_group`` groups, then sorted
    top-``top_k`` experts among them (other groups masked to ``-inf``). Weights are
    the unbiased scores of the selected experts in selection order, optionally
    normalized by their sum plus 1e-20; the routed scaling factor is not applied here.
    """

    def __init__(self, config: BeamConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(config.num_experts, dtype=torch.float32)
        )

    def route(self, hidden_states: torch.Tensor) -> BeamRouting:
        """Route ``(N, hidden)`` tokens; graph-capturable (no host synchronization)."""
        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        num_tokens, num_experts = scores.shape
        experts_per_group = num_experts // self.n_group
        grouped = scores_for_choice.view(num_tokens, self.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        top_groups = group_scores.topk(self.topk_group, dim=-1, sorted=True).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, top_groups, 1)
        expert_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, experts_per_group)
            .reshape(num_tokens, num_experts)
        )
        excluded = ~expert_mask.bool()
        scores_for_choice = scores_for_choice.masked_fill(excluded, float("-inf"))
        indices = scores_for_choice.topk(self.top_k, dim=-1, sorted=True).indices
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return BeamRouting(logits, scores, group_scores, top_groups, excluded, indices, weights)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        routing = self.route(hidden_states)
        return routing.weights, routing.indices


class BeamMoE(nn.Module):
    """Routed experts with the reference combine, executed with static shapes.

    Every expert evaluates every token row in ascending expert order; a route
    contributes only where the router selected that expert. Each selected route is
    ``expert(x) * weight`` rounded to the activation dtype and accumulated into an
    fp32 sum, which is scaled by ``routed_scaling_factor`` and cast once. The loop has
    no data-dependent branch or host synchronization, so it runs inside CUDA graphs.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        config = model_config.pretrained_config
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = BeamRouter(config)
        self.experts = nn.ModuleList(
            [
                BeamSwiGLU(model_config, config.hidden_size, config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

    def combine(
        self, hidden_states: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Expert outputs for given routing: ``hidden_states`` is ``(N, hidden)``."""
        out = torch.zeros(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
        for expert_id, expert in enumerate(self.experts):
            selected = indices == expert_id
            route_weight = (weights * selected).sum(dim=-1, keepdim=True)
            route = (expert(hidden_states) * route_weight).to(hidden_states.dtype)
            hit = selected.any(dim=-1, keepdim=True)
            out = torch.where(hit, out + route.float(), out)
        out = out * self.routed_scaling_factor
        return out.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights, indices = self.gate(hidden_states)
        return self.combine(hidden_states, weights, indices)


# ---------------------------------------------------------------------------- attention


class BeamAttention(Attention):
    """Beam GQA attention on the standard module/backend/KV-cache path.

    Q/K RMSNorm, RoPE (sliding layers) or SSMax (full layers) and the gate bias run in
    the module before the backend appends K/V, so the cache holds post-norm,
    post-RoPE keys. Backend RoPE fusion is disabled.
    """

    def __init__(self, model_config: ModelConfig[BeamConfig], layer_idx: int):
        config = model_config.pretrained_config
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.attention_window_size = config.sliding_window if self.is_sliding else None
        rope = config.rope_parameters["sliding_attention"]
        self.rope_theta = float(rope["rope_theta"])
        self.rope_factor = float(rope.get("factor", 1.0))
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=None,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            # The output-projection bias is added after the bias-free GEMM.
            dense_bias=False,
            config=model_config,
            attn_output_gate=True,
            head_dim=config.head_dim,
        )
        dtype = config.torch_dtype
        self.gate_bias = nn.Parameter(torch.zeros(self.num_heads, dtype=dtype), requires_grad=False)
        self.o_proj_bias = (
            nn.Parameter(torch.zeros(config.hidden_size, dtype=dtype), requires_grad=False)
            if config.attention_bias
            else None
        )
        self.q_norm = BeamRMSNorm(self.head_dim, config.rms_norm_eps, dtype)
        self.k_norm = BeamRMSNorm(self.head_dim, config.rms_norm_eps, dtype)

    def apply_rope(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ):
        """Q/K norm, then RoPE (sliding) or SSMax (full) at the tokens' true positions."""
        q, k, v = self.split_qkv(q, k, v)
        if position_ids is None:
            raise ValueError("Beam attention needs per-token position ids")
        num_tokens = q.shape[0]
        positions = position_ids.reshape(-1)[:num_tokens]
        q = self.q_norm(q.reshape(num_tokens, self.num_heads, self.head_dim))
        k = self.k_norm(k.reshape(num_tokens, self.num_key_value_heads, self.head_dim))
        if self.is_sliding:
            cos, sin = beam_rope_cos_sin(
                positions, self.head_dim, self.rope_theta, self.rope_factor
            )
            q, k = beam_apply_rotary(q, k, cos, sin)
        else:
            q = beam_apply_ssmax(q, positions)
        return (q.reshape(num_tokens, self.q_size), k.reshape(num_tokens, self.kv_size), v)

    def apply_output_gate(self, attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        num_tokens = attention_output.shape[0]
        out = attention_output.reshape(num_tokens, self.num_heads, self.head_dim)
        gate = gate[:num_tokens].reshape(num_tokens, self.num_heads, self.head_dim)
        gated = beam_output_gate(out, gate, self.gate_bias)
        return gated.reshape(num_tokens, self.num_heads * self.head_dim)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.setdefault("attention_window_size", self.attention_window_size)
        output = super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        if self.o_proj_bias is not None:
            output = output + self.o_proj_bias
        return output


# ------------------------------------------------------------------------------- blocks


class BeamDecoderLayer(DecoderLayer):
    """Sandwich-norm decoder block with depth-scaled residual adds."""

    def __init__(self, model_config: ModelConfig[BeamConfig], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        dtype = config.torch_dtype
        eps = config.rms_norm_eps
        self.layer_idx = layer_idx
        self.input_layernorm = BeamRMSNorm(config.hidden_size, eps, dtype)
        self.self_attn = BeamAttention(model_config, layer_idx)
        self.post_attention_layernorm = BeamRMSNorm(config.hidden_size, eps, dtype)
        self.pre_feedforward_layernorm = BeamRMSNorm(config.hidden_size, eps, dtype)
        self.is_sparse = config.mlp_layer_types[layer_idx] == "sparse"
        if self.is_sparse:
            self.mlp = BeamMoE(model_config)
        else:
            self.mlp = BeamSwiGLU(model_config, config.hidden_size, config.intermediate_size)
        self.post_feedforward_layernorm = BeamRMSNorm(config.hidden_size, eps, dtype)
        self.attn_scales = beam_depth_coefficients(layer_idx, is_attention=True)
        self.mlp_scales = beam_depth_coefficients(layer_idx, is_attention=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self.self_attn(
            position_ids=position_ids, hidden_states=x, attn_metadata=attn_metadata, **kwargs
        )
        x = self.post_attention_layernorm(x)
        hidden_states = beam_depth_residual(x, residual, self.attn_scales)
        residual = hidden_states
        x = self.pre_feedforward_layernorm(hidden_states)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        return beam_depth_residual(x, residual, self.mlp_scales)


class BeamModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[BeamConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList(
            [
                BeamDecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = BeamRMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                position_ids=position_ids, hidden_states=hidden_states, attn_metadata=attn_metadata
            )
        return self.norm(hidden_states)


class BeamForCausalLM(DecoderModelForCausalLM[BeamModel, BeamConfig]):
    """Beam causal LM: last-token (or context) logits in fp32, softcapped when enabled."""

    def __init__(self, model_config: ModelConfig[BeamConfig]):
        super().__init__(
            BeamModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        logits = super().forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            **kwargs,
        )
        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = beam_softcap(logits, cap)
        return logits

    def load_weights(
        self,
        weights: Dict[str, torch.Tensor],
        weight_mapper=None,
        skip_modules: Optional[List[str]] = None,
        params_map: Optional[Dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ) -> None:
        """Load a reference-layout state dict exactly (values and dtypes, no casts)."""
        if weight_mapper is not None or params_map or skip_modules or allow_partial_loading:
            raise ValueError("BeamForCausalLM loads complete reference-layout state dicts only")
        if self.model_config.mapping.tp_size > 1:
            raise ValueError(
                "BeamForCausalLM loads unsharded state dicts; use tp_size=1 or the "
                "ModelingV2 TP4 target"
            )
        load_beam_state_dict(self, weights)


# ------------------------------------------------------------------------ weight mapping


def beam_parameter_dtype(name: str, activation_dtype: torch.dtype) -> torch.dtype:
    """Source dtype policy: router weight and correction bias fp32, all else activation."""
    return torch.float32 if name.endswith(BEAM_FP32_PARAMETER_SUFFIXES) else activation_dtype


def pack_beam_qkv_gate(
    q: torch.Tensor,
    gate: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Fused projection rows: per head ``[q_h, gate_h]``, then all K rows, then all V rows."""
    trailing = q.shape[1:]
    q_gate = torch.stack(
        (q.reshape(num_heads, head_dim, *trailing), gate.reshape(num_heads, head_dim, *trailing)),
        dim=1,
    )
    return torch.cat((q_gate.reshape(2 * num_heads * head_dim, *trailing), k, v), dim=0)


def unpack_beam_qkv_gate(
    fused: torch.Tensor, num_heads: int, num_kv_heads: int, head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_beam_qkv_gate`: returns ``(q, gate, k, v)``."""
    q_width = num_heads * head_dim
    kv_width = num_kv_heads * head_dim
    q_gate, k, v = fused.split([2 * q_width, kv_width, kv_width], dim=0)
    q_gate = q_gate.reshape(num_heads, 2, head_dim, *fused.shape[1:])
    q = q_gate[:, 0].reshape(q_width, *fused.shape[1:])
    gate = q_gate[:, 1].reshape(q_width, *fused.shape[1:])
    return q, gate, k, v


def map_beam_attention_state(
    state: Dict[str, torch.Tensor], config: BeamConfig, prefix: str = ""
) -> Dict[str, torch.Tensor]:
    """Map one attention module's reference entries (under ``prefix``) to candidate names.

    The separate Q, gate, K and V projections become the fused ``qkv_proj``. With
    ``attention_bias`` the gate rows of the fused bias are zero (the gate projection has
    no bias) and the output-projection bias becomes ``o_proj_bias``. Consumed entries are
    removed from ``state``.
    """
    heads, head_dim = config.num_attention_heads, config.head_dim
    mapped: Dict[str, torch.Tensor] = {}
    q = state.pop(prefix + "q_proj.weight")
    gate = state.pop(prefix + "gate_proj.weight")
    k = state.pop(prefix + "k_proj.weight")
    v = state.pop(prefix + "v_proj.weight")
    mapped[prefix + "qkv_proj.weight"] = pack_beam_qkv_gate(q, gate, k, v, heads, head_dim)
    if config.attention_bias:
        qb = state.pop(prefix + "q_proj.bias")
        kb = state.pop(prefix + "k_proj.bias")
        vb = state.pop(prefix + "v_proj.bias")
        mapped[prefix + "qkv_proj.bias"] = pack_beam_qkv_gate(
            qb, torch.zeros_like(qb), kb, vb, heads, head_dim
        )
        mapped[prefix + "o_proj_bias"] = state.pop(prefix + "o_proj.bias")
    mapped[prefix + "o_proj.weight"] = state.pop(prefix + "o_proj.weight")
    for name in ("gate_bias", "q_norm.weight", "k_norm.weight"):
        mapped[prefix + name] = state.pop(prefix + name)
    return mapped


def map_beam_state_dict(
    state: Dict[str, torch.Tensor],
    config: BeamConfig,
    attention_prefixes: Optional[List[str]] = None,
    tied_lm_head: bool = False,
) -> Dict[str, torch.Tensor]:
    """Map reflection-core state-dict names to candidate parameter/buffer names.

    ``attention_prefixes`` lists the attention modules contained in ``state`` (default:
    every ``model.layers.{i}.self_attn.`` of the full model). Other entries keep their
    names. With ``tied_lm_head`` a supplied ``lm_head.weight`` must equal the embedding.
    """
    remaining = dict(state)
    if attention_prefixes is None:
        attention_prefixes = [
            f"model.layers.{i}.self_attn." for i in range(config.num_hidden_layers)
        ]
    mapped: Dict[str, torch.Tensor] = {}
    for prefix in attention_prefixes:
        mapped.update(map_beam_attention_state(remaining, config, prefix))
    if tied_lm_head:
        tied = remaining.pop("lm_head.weight", None)
        if tied is not None and not torch.equal(tied, state["model.embed_tokens.weight"]):
            raise ValueError("tied lm_head.weight differs from model.embed_tokens.weight")
    mapped.update(remaining)
    return mapped


def beam_state_targets(module: nn.Module, tied_lm_head: bool = False) -> Dict[str, torch.Tensor]:
    """Every loadable tensor of ``module``: all parameters plus router correction buffers.

    With ``tied_lm_head`` the LM-head weight is the embedding parameter and appears once.
    """
    targets: Dict[str, torch.Tensor] = dict(module.named_parameters())
    for name, buffer in module.named_buffers():
        if name.endswith("e_score_correction_bias"):
            targets[name] = buffer
    if tied_lm_head:
        targets.pop("lm_head.weight", None)
    return targets


def load_beam_state_dict(
    module: nn.Module,
    state: Dict[str, torch.Tensor],
    config: Optional[BeamConfig] = None,
    attention_prefixes: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Copy a reference-layout state dict into ``module``; return each tensor's dtype.

    ``module`` is a ``BeamForCausalLM`` (default prefixes) or a standalone Beam module
    with explicit ``config`` and ``attention_prefixes`` (``["self_attn."]`` for a
    decoder layer, ``[""]`` for an attention module, ``[]`` for none). Requires an exact
    name match after mapping, identical shapes and the source dtype policy on both
    sides (no implicit casts). Returns the loaded dtypes for accounting.
    """
    is_model = config is None
    config = config if config is not None else module.config
    tied = is_model and bool(config.tie_word_embeddings)
    mapped = map_beam_state_dict(state, config, attention_prefixes, tied_lm_head=tied)
    targets = beam_state_targets(module, tied_lm_head=tied)
    missing = sorted(set(targets) - set(mapped))
    unexpected = sorted(set(mapped) - set(targets))
    if missing or unexpected:
        raise ValueError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    accounting: Dict[str, str] = {}
    activation_dtype = config.torch_dtype
    with torch.no_grad():
        for name, target in targets.items():
            source = mapped[name]
            expected = beam_parameter_dtype(name, activation_dtype)
            if source.dtype != expected or target.dtype != expected:
                raise ValueError(
                    f"{name}: source {source.dtype} / target {target.dtype}, policy {expected}"
                )
            if source.shape != target.shape:
                raise ValueError(f"{name}: shape {tuple(source.shape)} != {tuple(target.shape)}")
            target.copy_(source)
            accounting[name] = str(target.dtype)
    return accounting
