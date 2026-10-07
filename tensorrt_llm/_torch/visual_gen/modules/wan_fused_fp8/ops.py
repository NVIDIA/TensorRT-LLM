# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom ops for the Wan fused FP8 path (torch.compile friendly).

Q/K scales fold the softmax scale, so attention runs with scaleSoftmaxLog2 == 1.
"""

import math
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from tensorrt_llm._torch.distributed import all_to_all_4d

from . import _ext

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_HEAD_DIM = 128
_LOG2E = 1.0 / math.log(2.0)
# sq = sk = 1 / sqrt(c), c = 1 / (sm_scale * log2 e): unit softmax scale.
_QK_MUL = math.sqrt(_HEAD_DIM**-0.5 * _LOG2E)
_UNSUPPORTED_SP = ("Attention2DAttention", "RingAttention")

_buffers = {}


def _bufs(batch: int, seq: int, device: torch.device) -> dict:
    """Per-shape device constants and scratch, reused across calls."""
    key = (batch, seq, str(device))
    if key not in _buffers:
        f32 = dict(device=device, dtype=torch.float32)
        _buffers[key] = dict(
            bmm1=torch.tensor([1.0 / _LOG2E, 1.0], **f32),
            cu_seqlens=torch.arange(0, (batch + 1) * seq, seq, device=device, dtype=torch.int32),
            seqlens=torch.full((batch,), seq, device=device, dtype=torch.int32),
            qk_mul=torch.tensor([_QK_MUL, _QK_MUL], **f32),
            mul=torch.tensor([_QK_MUL, _QK_MUL, 1.0], **f32),
            scale_v=torch.ones(1, **f32),
            amax=torch.zeros(3, **f32),
            amax_v=torch.zeros(1, **f32),
        )
    return _buffers[key]


def _rope_tables(cos: torch.Tensor, sin: torch.Tensor, tokens: int):
    cos2d = cos.reshape(-1, _HEAD_DIM).float().contiguous()
    sin2d = sin.reshape(-1, _HEAD_DIM).float().contiguous()
    # Fewer rows than tokens: one table shared by every batch entry.
    seq_per_batch = 0 if cos2d.shape[0] == tokens else cos2d.shape[0]
    return cos2d, sin2d, seq_per_batch


def _norm_rope_quant(
    qkv2d, num_heads, norm_q_w, norm_k_w, cos, sin, eps, interleave, mul, bufs, out_shape
):
    cos2d, sin2d, seq_per_batch = _rope_tables(cos, sin, qkv2d.shape[0])
    q8, k8, v8 = (torch.empty(out_shape, device=qkv2d.device, dtype=_FP8) for _ in range(3))
    _ext.prep().norm_rope_quant(
        qkv2d,
        num_heads,
        eps,
        norm_q_w,
        norm_k_w,
        cos2d,
        sin2d,
        interleave,
        seq_per_batch,
        mul,
        bufs["amax"],
        q8,
        k8,
        v8,
    )
    return q8, k8, v8


def _fmha(q8, k8, v8, scale_v, batch, seq, bufs):
    heads = q8.shape[-2]
    tokens = batch * seq
    q, k, v = (t.reshape(tokens, heads, _HEAD_DIM).contiguous() for t in (q8, k8, v8))
    out = _ext.fmha().fmha(
        q,
        k,
        v,
        bufs["cu_seqlens"],
        bufs["seqlens"],
        bufs["bmm1"],
        scale_v,
        batch,
        seq,
        True,
        True,
        None,
    )
    return out.view(batch, seq, heads * _HEAD_DIM)


@torch.library.custom_op("wanfused::gemm_fp8", mutates_args=())
def gemm_fp8(
    a: torch.Tensor,
    w: torch.Tensor,
    scale_a: torch.Tensor,
    scale_w: torch.Tensor,
    bias: Optional[torch.Tensor],
    epilogue: int,
    d_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """FP8 GEMM; epilogue 0 none, 1 bias, 2 bias+GELU."""
    out_dtype = _FP8 if d_scale is not None else torch.bfloat16
    out = torch.empty(a.shape[0], w.shape[0], device=a.device, dtype=out_dtype)
    _ext.block().gemm_fp8(
        a,
        w,
        scale_a.float().reshape(1),
        scale_w.float().reshape(1),
        bias if epilogue else None,
        epilogue,
        d_scale,
        out,
    )
    return out


@gemm_fp8.register_fake
def _(a, w, scale_a, scale_w, bias, epilogue, d_scale):
    return a.new_empty(
        a.shape[0], w.shape[0], dtype=_FP8 if d_scale is not None else torch.bfloat16
    )


@torch.library.custom_op("wanfused::resid_ln_quant", mutates_args=())
def resid_ln_quant(
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    ybias: Optional[torch.Tensor],
    gate: Optional[torch.Tensor],
    mode: int,
    w: Optional[torch.Tensor],
    b: Optional[torch.Tensor],
    seq_per_batch: int,
    eps: float,
    inv_scale: Optional[torch.Tensor],
) -> List[torch.Tensor]:
    """Gated residual, then LayerNorm/AdaLN and FP8 quant; returns [x_new, q]."""
    x = x.contiguous()
    x_out = torch.empty_like(x) if y is not None else x.new_empty(0)
    q = (
        torch.empty(x.shape, device=x.device, dtype=_FP8)
        if mode != 0
        else x.new_empty(0, dtype=_FP8)
    )
    _ext.block().resid_ln_quant(
        x,
        y.contiguous() if y is not None else None,
        ybias,
        gate,
        mode,
        w,
        b,
        seq_per_batch,
        eps,
        inv_scale,
        x_out if y is not None else None,
        q if mode != 0 else None,
    )
    return [x_out, q]


@resid_ln_quant.register_fake
def _(x, y, ybias, gate, mode, w, b, seq_per_batch, eps, inv_scale):
    x_out = torch.empty_like(x) if y is not None else x.new_empty(0)
    q = x.new_empty(x.shape, dtype=_FP8) if mode != 0 else x.new_empty(0, dtype=_FP8)
    return [x_out, q]


@torch.library.custom_op("wanfused::fp8_self_attention", mutates_args=())
def fp8_self_attention(
    qkv: torch.Tensor,
    norm_q_w: torch.Tensor,
    norm_k_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    eps: float,
    interleave: bool,
) -> torch.Tensor:
    """Packed QKV [B, S, 3*H*D] to attention output [B, S, H*D]."""
    batch, seq, _ = qkv.shape
    bufs = _bufs(batch, seq, qkv.device)
    qkv2d = qkv.reshape(batch * seq, -1).contiguous()
    _ext.prep().v_scale(qkv2d, num_heads, bufs["amax_v"], bufs["mul"], bufs["scale_v"])
    q8, k8, v8 = _norm_rope_quant(
        qkv2d,
        num_heads,
        norm_q_w,
        norm_k_w,
        cos,
        sin,
        eps,
        interleave,
        bufs["mul"],
        bufs,
        (batch * seq, num_heads, _HEAD_DIM),
    )
    return _fmha(q8, k8, v8, bufs["scale_v"], batch, seq, bufs)


@fp8_self_attention.register_fake
def _(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave):
    batch, seq, _ = qkv.shape
    return qkv.new_empty(batch, seq, num_heads * _HEAD_DIM)


@torch.library.custom_op("wanfused::v_amax", mutates_args=())
def v_amax(qkv: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Local |V| amax of packed QKV, shape [1]."""
    batch, seq, _ = qkv.shape
    amax = torch.zeros(1, device=qkv.device, dtype=torch.float32)
    mul = torch.ones(3, device=qkv.device, dtype=torch.float32)
    scale_v = torch.empty(1, device=qkv.device, dtype=torch.float32)
    _ext.prep().v_scale(qkv.reshape(batch * seq, -1).contiguous(), num_heads, amax, mul, scale_v)
    return amax


@v_amax.register_fake
def _(qkv, num_heads):
    return qkv.new_empty(1, dtype=torch.float32)


@torch.library.custom_op("wanfused::norm_rope_quant_fp8", mutates_args=())
def norm_rope_quant_fp8(
    qkv: torch.Tensor,
    norm_q_w: torch.Tensor,
    norm_k_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    eps: float,
    interleave: bool,
    mul_v: torch.Tensor,
) -> List[torch.Tensor]:
    """FP8 Q/K/V [B, S, H, D] with the given V quant multiplier."""
    batch, seq, _ = qkv.shape
    bufs = _bufs(batch, seq, qkv.device)
    mul = torch.cat([bufs["qk_mul"], mul_v.float().reshape(1)])
    q8, k8, v8 = _norm_rope_quant(
        qkv.reshape(batch * seq, -1).contiguous(),
        num_heads,
        norm_q_w,
        norm_k_w,
        cos,
        sin,
        eps,
        interleave,
        mul,
        bufs,
        (batch, seq, num_heads, _HEAD_DIM),
    )
    return [q8, k8, v8]


@norm_rope_quant_fp8.register_fake
def _(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, mul_v):
    batch, seq, _ = qkv.shape
    return [qkv.new_empty(batch, seq, num_heads, _HEAD_DIM, dtype=_FP8) for _ in range(3)]


@torch.library.custom_op("wanfused::fmha_fp8", mutates_args=())
def fmha_fp8(
    q8: torch.Tensor, k8: torch.Tensor, v8: torch.Tensor, scale_v: torch.Tensor
) -> torch.Tensor:
    """trtllm-gen FP8 attention on [B, S, H, D] inputs."""
    batch, seq = q8.shape[:2]
    return _fmha(
        q8,
        k8,
        v8,
        scale_v.float().reshape(1).contiguous(),
        batch,
        seq,
        _bufs(batch, seq, q8.device),
    )


@fmha_fp8.register_fake
def _(q8, k8, v8, scale_v):
    batch, seq, heads, _ = q8.shape
    return q8.new_empty(batch, seq, heads * _HEAD_DIM, dtype=torch.bfloat16)


def sp_mode(attn_module):
    """Returns ("none"|"ulysses"|"unsupported", process group or None)."""
    backend = getattr(attn_module, "attn", None)
    if type(backend).__name__ in _UNSUPPORTED_SP:
        return "unsupported", None
    if type(backend).__name__ != "UlyssesAttention":
        return "none", None
    # Ulysses over Attention2D / Ring needs LSE merging: unsupported here.
    if type(backend.inner_backend).__name__ in _UNSUPPORTED_SP:
        return "unsupported", None
    pg = backend.process_group
    if pg is None or dist.get_world_size(group=pg) == 1:
        return "none", None
    return "ulysses", pg


def fp8_self_attention_ulysses(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, pg):
    """Ulysses variant: local [B, S/P, 3*H*D] to [B, S/P, H*D]."""
    batch, seq_local, _ = qkv.shape
    world = dist.get_world_size(group=pg)
    # Global V amax keeps one V scale across the group.
    amax = funcol.all_reduce(v_amax(qkv, num_heads), "max", pg)
    scale_v = amax.clamp_min(1e-12) / _FP8_MAX
    q8, k8, v8 = norm_rope_quant_fp8(
        qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, 1.0 / scale_v
    )
    # Sequence-sharded to head-sharded, exchanged as FP8 bytes.
    q8, k8, v8 = (
        all_to_all_4d(t.view(torch.uint8), scatter_dim=2, gather_dim=1, process_group=pg).view(_FP8)
        for t in (q8, k8, v8)
    )
    out = fmha_fp8(q8, k8, v8, scale_v)
    seq = out.shape[1]
    out = out.view(batch, seq, num_heads // world, _HEAD_DIM).contiguous()
    out = all_to_all_4d(out, scatter_dim=1, gather_dim=2, process_group=pg)
    return out.reshape(batch, seq_local, num_heads * _HEAD_DIM)


def self_attention(
    attn, qkv: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    """Fused FP8 self-attention for a packed-QKV Attention module."""
    mode, pg = sp_mode(attn)
    args = (
        attn.norm_q.weight,
        attn.norm_k.weight,
        freqs_cos,
        freqs_sin,
        attn.local_num_attention_heads,
        float(attn.eps),
        bool(attn.interleave),
    )
    if mode == "ulysses":
        return fp8_self_attention_ulysses(qkv, *args, pg)
    if mode == "none":
        return fp8_self_attention(qkv, *args)
    # Attention2D / Ring: use the module's own backend.
    attn.apply_packed_qk_norm_rope(qkv, freqs_cos, freqs_sin)
    q, k, v = qkv.split([attn.local_q_dim, attn.local_kv_dim, attn.local_kv_dim], dim=-1)
    return attn._attn_impl(q, k, v)
