# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-Torch reference for one dense sliding-attention decoder block (block 0).

An independent restatement of the source block semantics for generated-input
component replay: sandwich RMSNorms (fp32 statistics and affine, one cast),
separate bias-free Q/K/V/gate projections in [out, in] layout, per-head Q/K
RMSNorm rounded to the activation dtype before fp32 NeoX RoPE, GQA attention
with a causal window mask, per-head sigmoid output gate in the activation
dtype, bias-free output projection, fp32 SiLU(gate) * up cast once before the
down projection, and depth-scaled residual adds in the activation dtype.

Only torch is imported. `Variant` selects documented equivalent formulations
(valid alternatives) or deliberately wrong ones (fault witnesses); the default
is the source formulation.
"""

from __future__ import annotations

import contextlib
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Dims:
    hidden: int = 5120
    intermediate: int = 20480
    heads: int = 48
    kv_heads: int = 8
    head_dim: int = 128
    eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    window: int = 512
    layer_id: int = 0
    max_len: int = 129


BLOCK0 = Dims()
SEEDS = (17, 41)
LENGTHS = (1, 17, 129)

STATE_RECIPE = {
    "construction": "torch.manual_seed(1300 + seed + layer_id); fp32 kaiming_uniform_(a=sqrt(5)) on CPU for the "
    "eight projection matrices in module construction order (q, k, v, gate, o, mlp gate, up, down)",
    "overrides": "torch.Generator().manual_seed(10000 + seed) in state_dict order: *norm.weight = 1 + 0.25 * N(0, 1); "
    "self_attn.gate_bias = 0.75 * N(0, 1)",
    "dtype": "every tensor cast once to bfloat16",
    "input": "torch.Generator().manual_seed(1200 + seed): 2 * N(0, 1) of shape [1, max_len, hidden], cast once to "
    "bfloat16; length-T cases use the first T rows; positions arange(T)",
}


def depth_coefs(layer_id: int, is_attention: bool) -> tuple[float, float]:
    one_based = layer_id + 1
    num_adds = 2 * one_based if is_attention else 2 * one_based + 1
    denom = math.sqrt(num_adds)
    return 1.0 / denom, math.sqrt(num_adds - 1) / denom


def param_shapes(d: Dims = BLOCK0) -> "OrderedDict[str, tuple[int, ...]]":
    """Block parameters in source state_dict order."""
    q, kv = d.heads * d.head_dim, d.kv_heads * d.head_dim
    return OrderedDict(
        [
            ("input_layernorm.weight", (d.hidden,)),
            ("self_attn.gate_bias", (d.heads,)),
            ("self_attn.q_proj.weight", (q, d.hidden)),
            ("self_attn.k_proj.weight", (kv, d.hidden)),
            ("self_attn.v_proj.weight", (kv, d.hidden)),
            ("self_attn.gate_proj.weight", (q, d.hidden)),
            ("self_attn.o_proj.weight", (d.hidden, q)),
            ("self_attn.q_norm.weight", (d.head_dim,)),
            ("self_attn.k_norm.weight", (d.head_dim,)),
            ("pre_feedforward_layernorm.weight", (d.hidden,)),
            ("mlp.gate_proj.weight", (d.intermediate, d.hidden)),
            ("mlp.up_proj.weight", (d.intermediate, d.hidden)),
            ("mlp.down_proj.weight", (d.hidden, d.intermediate)),
            ("post_attention_layernorm.weight", (d.hidden,)),
            ("post_feedforward_layernorm.weight", (d.hidden,)),
        ]
    )


INIT_ORDER = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.gate_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def generate_state(seed: int, d: Dims = BLOCK0) -> "OrderedDict[str, torch.Tensor]":
    """Deterministic bf16 block state on CPU (see STATE_RECIPE)."""
    shapes = param_shapes(d)
    torch.manual_seed(1300 + seed + d.layer_id)
    matrices = {}
    for name in INIT_ORDER:
        weight = torch.empty(shapes[name], dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        matrices[name] = weight
    gen = torch.Generator().manual_seed(10000 + seed)
    state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, shape in shapes.items():
        if name.endswith("norm.weight"):
            value = 1.0 + 0.25 * torch.randn(shape, generator=gen)
        elif name == "self_attn.gate_bias":
            value = 0.75 * torch.randn(shape, generator=gen)
        else:
            value = matrices[name]
        state[name] = value.to(torch.bfloat16).contiguous()
    return state


def generate_input(seed: int, d: Dims = BLOCK0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1200 + seed)
    return (2.0 * torch.randn(1, d.max_len, d.hidden, generator=gen)).to(torch.bfloat16)


def state_problems(state: "dict[str, torch.Tensor]", d: Dims = BLOCK0) -> list[str]:
    """Exact accounting plus the nontrivial-state properties the comparisons rely on."""
    problems = []
    shapes = param_shapes(d)
    if list(state) != list(shapes):
        problems.append(f"names/order {list(state)} != {list(shapes)}")
    for name, shape in shapes.items():
        t = state.get(name)
        if t is None:
            continue
        if tuple(t.shape) != shape or t.dtype != torch.bfloat16:
            problems.append(f"{name}: {tuple(t.shape)} {t.dtype}, expected {shape} bfloat16")
        if not bool(torch.isfinite(t.float()).all()):
            problems.append(f"{name}: nonfinite")
    norms = [n for n in shapes if n.endswith("norm.weight")]
    for name in norms:
        if name in state and bool((state[name].float() == 1.0).all()):
            problems.append(f"{name}: unit affine")
    for i, a in enumerate(norms):
        for b in norms[i + 1 :]:
            if (
                a in state
                and b in state
                and state[a].shape == state[b].shape
                and torch.equal(state[a], state[b])
            ):
                problems.append(f"{a} and {b} are identical")
    bias = state.get("self_attn.gate_bias")
    if bias is not None:
        if not bool((bias.float() != 0).any()):
            problems.append("gate_bias is all zero")
        if bool((bias == bias[0]).all()):
            problems.append("gate_bias does not vary across heads")
    return problems


@dataclass(frozen=True)
class Variant:
    """Source formulation by default; see module docstring."""

    # Valid alternatives (same mathematics, different evaluation).
    sdpa_backend: str | None = (
        None  # None: default dispatch; MATH | EFFICIENT_ATTENTION | FLASH_ATTENTION | CUDNN_ATTENTION
    )
    sdpa_form: str = "mask"  # mask: explicit window mask; is_causal: causal flag (equal for T <= window); enable_gqa
    gemm: str = "batched"  # batched | per_row | fused | matmul_t
    rmsnorm: str = "source"  # source | alt_order | chunked
    gate: str = "source"  # source | fp32_sigmoid_then_cast
    residual: str = "source"  # source | torch_add
    silu: str = "source"  # source: F.silu | x_sigmoid: x * sigmoid(x) | exp_form: x / (1 + exp(-x)); all fp32
    # Fault witnesses (wrong semantics).
    attn_coefs: tuple[float, float] | None = None
    mlp_coefs: tuple[float, float] | None = None
    residual_fp32_fused: bool = False
    omit_post_attention_norm: bool = False
    swap_post_attention_and_pre_ffn_norms: bool = False
    rmsnorm_round_before_affine: bool = False
    softmax_scale: float | None = None
    gqa_tiled: bool = False
    noncausal: bool = False
    omit_k_norm: bool = False
    swap_qk_norm_weights: bool = False
    gate_bias: str = "source"  # source | omitted | uniform_mean | heads_rolled
    swiglu_bf16_activation: bool = False
    rope_interleaved: bool = False
    rope_positions_offset: int = 0
    normalize_v: bool = False
    qk_norm_fp32_into_rope: bool = False


SOURCE = Variant()

_SDPA = {
    "MATH": "MATH",
    "EFFICIENT_ATTENTION": "EFFICIENT_ATTENTION",
    "FLASH_ATTENTION": "FLASH_ATTENTION",
    "CUDNN_ATTENTION": "CUDNN_ATTENTION",
}


@contextlib.contextmanager
def sdpa_backend(name: str | None) -> Iterator[None]:
    if name is None:
        yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([getattr(SDPBackend, _SDPA[name])]):
        yield


Record = Callable[[str, torch.Tensor], None]


def _noop(name: str, value: torch.Tensor) -> None:
    del name, value


def linear(x: torch.Tensor, weight: torch.Tensor, v: Variant) -> torch.Tensor:
    if v.gemm == "per_row":
        rows = [F.linear(x[..., i : i + 1, :], weight) for i in range(x.shape[-2])]
        return torch.cat(rows, dim=-2)
    if v.gemm == "matmul_t":
        return torch.matmul(x, weight.t())
    return F.linear(x, weight)


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, v: Variant = SOURCE
) -> torch.Tensor:
    in_dtype = x.dtype
    x32 = x.float()
    if v.rmsnorm == "alt_order":
        normed = x32 / torch.sqrt(x32.pow(2).sum(-1, keepdim=True) / x32.shape[-1] + eps)
    elif v.rmsnorm == "chunked":
        n = x32.shape[-1]
        sq = x32.pow(2).unflatten(-1, (n // 32, 32)).sum(-1).sum(-1, keepdim=True)
        normed = x32 * (sq / n + eps).rsqrt()
    else:
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
        normed = x32 * rms
    if v.rmsnorm_round_before_affine:
        return normed.to(in_dtype) * weight
    return (normed * weight).to(in_dtype)


def rope_cos_sin(positions: torch.Tensor, d: Dims = BLOCK0) -> tuple[torch.Tensor, torch.Tensor]:
    exponent = torch.arange(0, d.head_dim, 2, dtype=torch.float32, device=positions.device)
    inv_freq = 1.0 / (d.rope_theta ** (exponent / d.head_dim))
    inv_freq = inv_freq / d.rope_factor
    freqs = positions.float().unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, v: Variant = SOURCE
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: (B, T, heads, head_dim); cos, sin: (B, T, head_dim) fp32."""
    q_dtype, k_dtype = q.dtype, k.dtype
    cos = cos.unsqueeze(2).float()
    sin = sin.unsqueeze(2).float()
    q, k = q.float(), k.float()
    if v.rope_interleaved:

        def rot(x: torch.Tensor) -> torch.Tensor:
            half = x.shape[-1] // 2
            c, s = cos[..., :half], sin[..., :half]
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)

        return rot(q).to(q_dtype), rot(k).to(k_dtype)
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out.to(q_dtype), k_out.to(k_dtype)


def causal_window_mask(length: int, window: int | None, device: torch.device) -> torch.Tensor:
    q = torch.arange(length, device=device).unsqueeze(1)
    k = torch.arange(length, device=device).unsqueeze(0)
    mask = q >= k
    if window is not None:
        mask = mask & ((q - k) < window)
    return mask.unsqueeze(0).unsqueeze(0)


def attention_core(
    q: torch.Tensor, k: torch.Tensor, val: torch.Tensor, d: Dims = BLOCK0, v: Variant = SOURCE
) -> torch.Tensor:
    """q: (B, T, heads, D) post-RoPE; k, val: (B, T, kv_heads, D). Returns (B, T, heads, D)."""
    length = q.shape[1]
    q, k, val = q.transpose(1, 2), k.transpose(1, 2), val.transpose(1, 2)
    repeat = d.heads // d.kv_heads
    gqa_native = v.sdpa_form == "enable_gqa"
    if not gqa_native and repeat != 1:
        if v.gqa_tiled:
            k, val = k.repeat(1, repeat, 1, 1), val.repeat(1, repeat, 1, 1)
        else:
            k, val = k.repeat_interleave(repeat, dim=1), val.repeat_interleave(repeat, dim=1)
    scale = d.head_dim**-0.5 if v.softmax_scale is None else v.softmax_scale
    kwargs = {"dropout_p": 0.0, "scale": scale}
    if gqa_native:
        kwargs["enable_gqa"] = True
    with sdpa_backend(v.sdpa_backend):
        if v.noncausal:
            out = F.scaled_dot_product_attention(q, k, val, **kwargs)
        elif v.sdpa_form == "is_causal":
            out = F.scaled_dot_product_attention(q, k, val, is_causal=True, **kwargs)
        else:
            mask = causal_window_mask(length, d.window, q.device)
            out = F.scaled_dot_product_attention(q, k, val, attn_mask=mask, **kwargs)
    return out.transpose(1, 2)


def output_gate(
    attn: torch.Tensor, gate: torch.Tensor, gate_bias: torch.Tensor, v: Variant = SOURCE
) -> torch.Tensor:
    """attn, gate: (B, T, heads, D) activation dtype; gate_bias: (heads,)."""
    if v.gate_bias == "omitted":
        gate_bias = torch.zeros_like(gate_bias)
    elif v.gate_bias == "uniform_mean":
        gate_bias = torch.full_like(gate_bias, float(gate_bias.float().mean()))
    elif v.gate_bias == "heads_rolled":
        gate_bias = gate_bias.roll(1)
    out_dtype = attn.dtype
    biased = gate + gate_bias[None, None, :, None]
    if v.gate == "fp32_sigmoid_then_cast":
        return attn * torch.sigmoid(biased.float()).to(out_dtype)
    return (attn * torch.sigmoid(biased)).to(out_dtype)


def attention(
    x: torch.Tensor,
    positions: torch.Tensor,
    s: "dict[str, torch.Tensor]",
    d: Dims = BLOCK0,
    v: Variant = SOURCE,
    record: Record = _noop,
    fused: "dict[str, torch.Tensor] | None" = None,
) -> torch.Tensor:
    b, t, _ = x.shape
    qw, kvw = d.heads * d.head_dim, d.kv_heads * d.head_dim
    if v.gemm == "fused":
        assert fused is not None
        q, k, val, gate = (
            y.contiguous() for y in F.linear(x, fused["qkvg"]).split([qw, kvw, kvw, qw], dim=-1)
        )
    else:
        q = linear(x, s["self_attn.q_proj.weight"], v)
        k = linear(x, s["self_attn.k_proj.weight"], v)
        val = linear(x, s["self_attn.v_proj.weight"], v)
        gate = linear(x, s["self_attn.gate_proj.weight"], v)
    for name, value in (("q_proj", q), ("k_proj", k), ("v_proj", val), ("gate_proj", gate)):
        record(name, value)
    q = q.view(b, t, d.heads, d.head_dim)
    k = k.view(b, t, d.kv_heads, d.head_dim)
    val = val.view(b, t, d.kv_heads, d.head_dim)
    gate = gate.view(b, t, d.heads, d.head_dim)
    qn_w, kn_w = s["self_attn.q_norm.weight"], s["self_attn.k_norm.weight"]
    if v.swap_qk_norm_weights:
        qn_w, kn_w = kn_w, qn_w
    if v.qk_norm_fp32_into_rope:
        q32 = q.float()
        q = (q32 * q32.pow(2).mean(-1, keepdim=True).add(d.eps).rsqrt()) * qn_w
        k32 = k.float()
        k = (k32 * k32.pow(2).mean(-1, keepdim=True).add(d.eps).rsqrt()) * kn_w
        qn, kn = q.to(torch.bfloat16), k.to(torch.bfloat16)
    else:
        q = rms_norm(q, qn_w, d.eps, v)
        if not v.omit_k_norm:
            k = rms_norm(k, kn_w, d.eps, v)
        qn, kn = q, k
    if v.normalize_v:
        val = rms_norm(val, kn_w, d.eps, v)
    record("q_norm", qn)
    record("k_norm", kn)
    cos, sin = rope_cos_sin(positions + v.rope_positions_offset, d)
    q, k = apply_rope(q, k, cos, sin, v)
    if v.qk_norm_fp32_into_rope:
        q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)
    record("q_rope", q)
    record("k_rope", k)
    attn = attention_core(q, k, val, d, v)
    record("attention_core", attn)
    attn = output_gate(attn, gate, s["self_attn.gate_bias"], v)
    attn = attn.reshape(b, t, qw)
    record("gated", attn)
    out = linear(attn, s["self_attn.o_proj.weight"], v)
    record("attention_output", out)
    return out


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor, v: Variant = SOURCE) -> torch.Tensor:
    """fp32 SiLU(gate) * up, cast once to the activation dtype."""
    if v.swiglu_bf16_activation:
        return F.silu(gate) * up
    g, u = gate.float(), up.float()
    if v.silu == "x_sigmoid":
        act = g * torch.sigmoid(g)
    elif v.silu == "exp_form":
        act = g / (1.0 + torch.exp(-g))
    else:
        act = F.silu(g)
    return (act * u).to(gate.dtype)


def swiglu(
    x: torch.Tensor,
    s: "dict[str, torch.Tensor]",
    d: Dims = BLOCK0,
    v: Variant = SOURCE,
    record: Record = _noop,
    fused: "dict[str, torch.Tensor] | None" = None,
) -> torch.Tensor:
    if v.gemm == "fused":
        assert fused is not None
        gate, up = (
            y.contiguous()
            for y in F.linear(x, fused["gate_up"]).split([d.intermediate] * 2, dim=-1)
        )
    else:
        gate = linear(x, s["mlp.gate_proj.weight"], v)
        up = linear(x, s["mlp.up_proj.weight"], v)
    record("mlp_gate", gate)
    record("mlp_up", up)
    act = silu_and_mul(gate, up, v)
    record("mlp_act", act)
    out = linear(act, s["mlp.down_proj.weight"], v)
    record("mlp_output", out)
    return out


def residual_add(
    x: torch.Tensor, residual: torch.Tensor, scales: tuple[float, float], v: Variant = SOURCE
) -> torch.Tensor:
    x_scale, r_scale = scales
    if v.residual_fp32_fused:
        return (x.float() * x_scale + residual.float() * r_scale).to(x.dtype)
    if v.residual == "torch_add":
        return torch.add(x * x_scale, residual * r_scale)
    return x * x_scale + residual * r_scale


def fused_weights(s: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
    return {
        "qkvg": torch.cat(
            [s[f"self_attn.{n}_proj.weight"] for n in ("q", "k", "v", "gate")], dim=0
        ).contiguous(),
        "gate_up": torch.cat(
            [s["mlp.gate_proj.weight"], s["mlp.up_proj.weight"]], dim=0
        ).contiguous(),
    }


def block(
    x: torch.Tensor,
    positions: torch.Tensor,
    s: "dict[str, torch.Tensor]",
    d: Dims = BLOCK0,
    v: Variant = SOURCE,
    record: Record = _noop,
    fused: "dict[str, torch.Tensor] | None" = None,
) -> torch.Tensor:
    """x: (1, T, hidden) bf16; positions: (1, T) int64. Returns (1, T, hidden) bf16."""
    attn_scales = v.attn_coefs or depth_coefs(d.layer_id, True)
    mlp_scales = v.mlp_coefs or depth_coefs(d.layer_id, False)
    post_attn_w = s["post_attention_layernorm.weight"]
    pre_ffn_w = s["pre_feedforward_layernorm.weight"]
    if v.swap_post_attention_and_pre_ffn_norms:
        post_attn_w, pre_ffn_w = pre_ffn_w, post_attn_w
    residual = x
    h = rms_norm(x, s["input_layernorm.weight"], d.eps, v)
    record("input_layernorm", h)
    h = attention(h, positions, s, d, v, record, fused)
    if not v.omit_post_attention_norm:
        h = rms_norm(h, post_attn_w, d.eps, v)
    record("post_attention_layernorm", h)
    hidden = residual_add(h, residual, attn_scales, v)
    record("attention_residual", hidden)
    h = rms_norm(hidden, pre_ffn_w, d.eps, v)
    record("pre_feedforward_layernorm", h)
    h = swiglu(h, s, d, v, record, fused)
    h = rms_norm(h, s["post_feedforward_layernorm.weight"], d.eps, v)
    record("post_feedforward_layernorm", h)
    out = residual_add(h, hidden, mlp_scales, v)
    record("block_output", out)
    return out
