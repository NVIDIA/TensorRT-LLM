# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom ops for the Wan fused FP8 path (torch.compile friendly).

Q/K scales fold the softmax scale, so attention runs with scaleSoftmaxLog2 == 1.
Entry: self_attention -> fused_self_attention -> fp8_self_attention | ulysses_self_attention
(overlap or a2a) | fp8_self_attention_attn2d_op.
"""

import math
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.distributed_c10d import _resolve_process_group

from tensorrt_llm._torch.distributed import all_to_all_4d

from ...mapping import VisualGenMapping
from . import _ext, ulysses_overlap
from ._common import FP8, HEAD_DIM, norm_rope_quant_into, qkv_mul, shape_buffers, v_scale_from_amax

_UNSUPPORTED_SP = ("RingAttention",)
# Head groups per rank for the overlapped Ulysses path.
_ULYSSES_GROUPS = 2


def _fmha(q8, k8, v8, scale_v, batch, seq, bufs, out=None):
    heads = q8.shape[-2]
    tokens = batch * seq
    q, k, v = (t.reshape(tokens, heads, HEAD_DIM).contiguous() for t in (q8, k8, v8))
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
        out=out,
    )
    return out.view(batch, seq, heads * HEAD_DIM)


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
    """FP8 GEMM a @ w.T; epilogue 0 none, 1 bias, 2 bias+GELU.

    scale_a/scale_w: dequant scales; d_scale (1/output scale) given -> FP8 output, else BF16.
    """
    out_dtype = FP8 if d_scale is not None else torch.bfloat16
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
    return a.new_empty(a.shape[0], w.shape[0], dtype=FP8 if d_scale is not None else torch.bfloat16)


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
    """x_new = x + gate * (y + ybias) when y is given, then norm and FP8 quant (D must be 5120).

    mode 0: residual only (needs y, q empty); 1: LayerNorm with affine w/b;
    2: AdaLN, LN(x_new) * (1 + w) + b, with w/b [B, D] per batch of seq_per_batch rows.
    ybias is required whenever y is given.
    q = FP8(norm * inv_scale), inv_scale = 1/input_scale. Returns [x_new or empty, q or empty].
    """
    x = x.contiguous()
    x_out = torch.empty_like(x) if y is not None else x.new_empty(0)
    q = torch.empty(x.shape, device=x.device, dtype=FP8) if mode != 0 else x.new_empty(0, dtype=FP8)
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
    q = x.new_empty(x.shape, dtype=FP8) if mode != 0 else x.new_empty(0, dtype=FP8)
    return [x_out, q]


def _prep_qkv(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave):
    """V scale, then normed/RoPE'd FP8 Q/K/V for one GPU."""
    batch, seq, _ = qkv.shape
    bufs = shape_buffers(batch, seq, qkv.device)
    qkv2d = qkv.reshape(batch * seq, -1).contiguous()
    _ext.prep().v_scale(qkv2d, num_heads, bufs["amax_v"], bufs["mul"], bufs["scale_v"])
    shape = (batch * seq, num_heads, HEAD_DIM)
    outs = [torch.empty(shape, device=qkv.device, dtype=FP8) for _ in range(3)]
    norm_rope_quant_into(
        qkv2d, num_heads, norm_q_w, norm_k_w, cos, sin, eps, interleave, bufs["mul"], outs
    )
    return (*outs, bufs)


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
    q8, k8, v8, bufs = _prep_qkv(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave)
    return _fmha(q8, k8, v8, bufs["scale_v"], batch, seq, bufs)


@fp8_self_attention.register_fake
def _(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave):
    batch, seq, _ = qkv.shape
    return qkv.new_empty(batch, seq, num_heads * HEAD_DIM)


@torch.library.custom_op("wanfused::fp8_self_attention_fp8_out", mutates_args=())
def fp8_self_attention_fp8_out(
    qkv: torch.Tensor,
    norm_q_w: torch.Tensor,
    norm_k_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    eps: float,
    interleave: bool,
    out_scale: torch.Tensor,
) -> torch.Tensor:
    """As fp8_self_attention, but FP8 output quantized by out_scale."""
    batch, seq, _ = qkv.shape
    q8, k8, v8, bufs = _prep_qkv(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave)
    # Unit-scale cubin applies scale_v / out_scale, rounding once to FP8.
    scale_o = (bufs["scale_v"] / out_scale.float().reshape(1)).contiguous()
    out = torch.empty(batch * seq, num_heads, HEAD_DIM, device=qkv.device, dtype=FP8)
    return _fmha(q8, k8, v8, scale_o, batch, seq, bufs, out)


@fp8_self_attention_fp8_out.register_fake
def _(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, out_scale):
    batch, seq, _ = qkv.shape
    return qkv.new_empty(batch, seq, num_heads * HEAD_DIM, dtype=FP8)


@torch.library.custom_op("wanfused::v_amax", mutates_args=())
def v_amax(qkv: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Local |V| amax of packed QKV, shape [1]."""
    batch, seq, _ = qkv.shape
    amax = torch.zeros(1, device=qkv.device, dtype=torch.float32)
    # v_scale also writes mul[2] and scale_v; throwaway here, callers derive the scale from the reduced amax.
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
    mul = qkv_mul(shape_buffers(batch, seq, qkv.device), mul_v)
    shape = (batch, seq, num_heads, HEAD_DIM)
    outs = [torch.empty(shape, device=qkv.device, dtype=FP8) for _ in range(3)]
    norm_rope_quant_into(
        qkv.reshape(batch * seq, -1).contiguous(),
        num_heads,
        norm_q_w,
        norm_k_w,
        cos,
        sin,
        eps,
        interleave,
        mul,
        outs,
    )
    return outs


@norm_rope_quant_fp8.register_fake
def _(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, mul_v):
    batch, seq, _ = qkv.shape
    return [qkv.new_empty(batch, seq, num_heads, HEAD_DIM, dtype=FP8) for _ in range(3)]


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
        shape_buffers(batch, seq, q8.device),
    )


@fmha_fp8.register_fake
def _(q8, k8, v8, scale_v):
    batch, seq, heads, _ = q8.shape
    return q8.new_empty(batch, seq, heads * HEAD_DIM, dtype=torch.bfloat16)


@torch.library.custom_op("wanfused::norm_rope_quant_fp8_packed", mutates_args=())
def norm_rope_quant_fp8_packed(
    qkv: torch.Tensor,
    norm_q_w: torch.Tensor,
    norm_k_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    eps: float,
    interleave: bool,
    mul_v: torch.Tensor,
) -> torch.Tensor:
    """FP8 Q/K/V packed as [3, B, S, H, D] with the given V quant multiplier."""
    batch, seq, _ = qkv.shape
    mul = qkv_mul(shape_buffers(batch, seq, qkv.device), mul_v)
    out = torch.empty(3, batch, seq, num_heads, HEAD_DIM, device=qkv.device, dtype=FP8)
    norm_rope_quant_into(
        qkv.reshape(batch * seq, -1).contiguous(),
        num_heads,
        norm_q_w,
        norm_k_w,
        cos,
        sin,
        eps,
        interleave,
        mul,
        out.unbind(0),
    )
    return out


@norm_rope_quant_fp8_packed.register_fake
def _(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, mul_v):
    batch, seq, _ = qkv.shape
    return qkv.new_empty(3, batch, seq, num_heads, HEAD_DIM, dtype=FP8)


@torch.library.custom_op("wanfused::fmha_fp8_stats", mutates_args=())
def fmha_fp8_stats(
    q8: torch.Tensor, k8: torch.Tensor, v8: torch.Tensor, scale_v: torch.Tensor
) -> List[torch.Tensor]:
    """FP8 attention of [B, Sq, H, D] over [B, Skv, H, D]; returns out and stats.

    Stats are [B*Sq, H, 2] float {max, sum}; LSE = max + ln(sum) in natural log.
    """
    batch, seq_q, heads, _ = q8.shape
    seq_kv = k8.shape[1]
    bq = shape_buffers(batch, seq_q, q8.device)
    bkv = shape_buffers(batch, seq_kv, q8.device)
    stats = torch.empty(batch * seq_q, heads, 2, device=q8.device, dtype=torch.float32)
    out = _ext.fmha().fmha(
        q8.reshape(batch * seq_q, heads, HEAD_DIM),
        k8.reshape(batch * seq_kv, heads, HEAD_DIM),
        v8.reshape(batch * seq_kv, heads, HEAD_DIM),
        bq["cu_seqlens"],
        bq["seqlens"],
        bq["bmm1"],
        scale_v.float().reshape(1).contiguous(),
        batch,
        seq_q,
        softmax_stats=stats,
        cu_seqlens_kv=bkv["cu_seqlens"],
        seqlens_kv=bkv["seqlens"],
        seq_len_kv=seq_kv,
    )
    return [out.view(batch, seq_q, heads, HEAD_DIM), stats]


@fmha_fp8_stats.register_fake
def _(q8, k8, v8, scale_v):
    batch, seq_q, heads, _ = q8.shape
    return [
        q8.new_empty(batch, seq_q, heads, HEAD_DIM, dtype=torch.bfloat16),
        q8.new_empty(batch * seq_q, heads, 2, dtype=torch.float32),
    ]


@torch.library.custom_op("wanfused::lse_combine", mutates_args=())
def lse_combine(o: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    """Merge partials o [N, R, H, D] by stats [N, R, H, 2] into [R, H, D]."""
    out = o.new_empty(o.shape[1:])
    _ext.prep().lse_combine(o.contiguous(), stats.contiguous(), out)
    return out


@lse_combine.register_fake
def _(o, stats):
    return o.new_empty(o.shape[1:])


_SP_SPEC_ATTR = "_wan_fused_sp_spec"


def sp_mode(attn_module):
    """Returns (mode, groups) with mode "none", "ulysses", "attn2d" or "unsupported".

    groups: the Ulysses group, or (ulysses, row, col, amax groups) names for attn2d.
    Cached on the module, so compiled code reads constants after the first call.
    """
    spec = getattr(attn_module, _SP_SPEC_ATTR, None)
    if spec is None:
        spec = _resolve_sp_mode(attn_module)
    return spec


@torch.compiler.disable
def _resolve_sp_mode(attn_module):
    backend = getattr(attn_module, "attn", None)
    ulysses_pg = None
    if type(backend).__name__ == "UlyssesAttention":
        pg = backend.process_group
        if pg is not None and dist.get_world_size(group=pg) > 1:
            ulysses_pg = pg
        backend = backend.inner_backend
    name = type(backend).__name__
    if name in _UNSUPPORTED_SP:
        spec = ("unsupported", None)
    elif name == "Attention2DAttention":
        row_pg, col_pg = backend.row_process_group, backend.col_process_group
        amax = ",".join(g.group_name for g in _amax_groups(ulysses_pg, row_pg, col_pg))
        uly = ulysses_pg.group_name if ulysses_pg is not None else ""
        spec = ("attn2d", (uly, row_pg.group_name, col_pg.group_name, amax))
    elif ulysses_pg is None:
        spec = ("none", None)
    else:
        spec = ("ulysses", ulysses_pg)
    setattr(attn_module, _SP_SPEC_ATTR, spec)
    return spec


def fp8_self_attention_ulysses(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, pg):
    """Ulysses with plain all-to-all (no overlap): local [B, S/P, 3*H*D] to [B, S/P, H*D].

    Fallback and numerical reference for ulysses_overlap.
    """
    batch, seq_local, _ = qkv.shape
    world = dist.get_world_size(group=pg)
    # Global V amax keeps one V scale across the group.
    amax = funcol.all_reduce(v_amax(qkv, num_heads), "max", pg)
    scale_v = v_scale_from_amax(amax)
    q8, k8, v8 = norm_rope_quant_fp8(
        qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, 1.0 / scale_v
    )
    # Sequence-sharded to head-sharded, FP8 moved as int64 words for wide copies.
    q8, k8, v8 = (
        all_to_all_4d(t.view(torch.int64), scatter_dim=2, gather_dim=1, process_group=pg).view(FP8)
        for t in (q8, k8, v8)
    )
    out = fmha_fp8(q8, k8, v8, scale_v)
    seq = out.shape[1]
    out = out.view(batch, seq, num_heads // world, HEAD_DIM).contiguous()
    out = all_to_all_4d(out, scatter_dim=1, gather_dim=2, process_group=pg)
    return out.reshape(batch, seq_local, num_heads * HEAD_DIM)


def _amax_groups(ulysses_pg, row_pg, col_pg):
    """Groups to max-reduce V amax over; the seq mesh group when it spans them."""
    groups = tuple(
        g
        for g in (ulysses_pg, row_pg, col_pg)
        if g is not None and dist.get_world_size(group=g) > 1
    )
    mesh = VisualGenMapping.seq_mesh
    if len(groups) > 1 and mesh is not None:
        seq = mesh.get_group()
        if dist.get_world_size(group=seq) == math.prod(
            dist.get_world_size(group=g) for g in groups
        ):
            return (seq,)
    return groups


def _all_gather_seq(tensors, pg):
    """Gathers FP8 [B, S, H, D] tensors along S over pg, one coalesced launch."""
    world = dist.get_world_size(group=pg)
    outs = [t.new_empty(world, *t.shape) for t in tensors]
    with dist._coalescing_manager(group=pg, device=tensors[0].device):
        for out, t in zip(outs, tensors):
            dist.all_gather_into_tensor(out.view(-1), t.contiguous().view(-1), group=pg)
    batch, seq = tensors[0].shape[:2]
    if batch == 1:
        return [o.view(1, world * seq, *o.shape[3:]) for o in outs]
    return [o.transpose(0, 1).reshape(batch, world * seq, *o.shape[3:]) for o in outs]


def _exchange_dest_major(t, world, pg):
    """[B, W*S, ...] split by destination rank; returns received [W, B, S, ...]."""
    batch = t.shape[0]
    send = t.view(batch, world, t.shape[1] // world, *t.shape[2:]).transpose(0, 1)
    send = send.contiguous() if batch > 1 else send.reshape(world, *send.shape[1:])
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=pg)
    return recv


def fp8_self_attention_attn2d(
    qkv,
    norm_q_w,
    norm_k_w,
    cos,
    sin,
    num_heads,
    eps,
    interleave,
    ulysses_pg,
    row_pg,
    col_pg,
    amax_groups=None,
):
    """Attention2D, optionally inside Ulysses: local [B, S/P, 3*H*D] to [B, S/P, H*D].

    Eager form taking ProcessGroups; compiled code calls fp8_self_attention_attn2d_op.
    """
    batch, seq_local, _ = qkv.shape
    if amax_groups is None:
        amax_groups = _amax_groups(ulysses_pg, row_pg, col_pg)
    # One V scale across all ranks whose partial outputs are merged.
    amax = v_amax(qkv, num_heads)
    for pg in amax_groups:
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=pg)
    scale_v = v_scale_from_amax(amax)
    qkv8 = norm_rope_quant_fp8_packed(
        qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, 1.0 / scale_v
    )
    if ulysses_pg is not None:
        # One all-to-all for Q/K/V; FP8 moved as int64 words for wide copies.
        qkv8 = all_to_all_4d(
            qkv8.view(torch.int64).view(3 * batch, seq_local, num_heads, HEAD_DIM // 8),
            scatter_dim=2,
            gather_dim=1,
            process_group=ulysses_pg,
        )
        qkv8 = qkv8.view(FP8).view(3, batch, qkv8.shape[1], qkv8.shape[2], HEAD_DIM)
    out = _attn2d_core(qkv8, scale_v, row_pg, col_pg)
    if ulysses_pg is not None:
        out = all_to_all_4d(out, scatter_dim=1, gather_dim=2, process_group=ulysses_pg)
    return out.reshape(batch, seq_local, num_heads * HEAD_DIM)


@torch.library.custom_op("wanfused::fp8_self_attention_attn2d", mutates_args=())
def fp8_self_attention_attn2d_op(
    qkv: torch.Tensor,
    norm_q_w: torch.Tensor,
    norm_k_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    eps: float,
    interleave: bool,
    ulysses_group: str,
    row_group: str,
    col_group: str,
    amax_groups: str,
) -> torch.Tensor:
    """fp8_self_attention_attn2d with groups by name; amax groups comma-joined, opaque to dynamo."""
    return fp8_self_attention_attn2d(
        qkv,
        norm_q_w,
        norm_k_w,
        cos,
        sin,
        num_heads,
        eps,
        interleave,
        _resolve_process_group(ulysses_group) if ulysses_group else None,
        _resolve_process_group(row_group),
        _resolve_process_group(col_group),
        tuple(_resolve_process_group(g) for g in amax_groups.split(",") if g),
    )


@fp8_self_attention_attn2d_op.register_fake
def _(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, uly, row, col, amax):
    batch, seq, _ = qkv.shape
    return qkv.new_empty(batch, seq, num_heads * HEAD_DIM)


def _attn2d_core(qkv8, scale_v, row_pg, col_pg):
    """Gathers, FMHA with stats and LSE reduce-scatter; [3, B, S, H, D] to [B, S, H, D]."""
    q8, k8, v8 = qkv8.unbind(0)
    batch, seq_u, heads = q8.shape[:3]
    rows = dist.get_world_size(group=row_pg)
    if rows > 1:
        (q8,) = _all_gather_seq([q8], row_pg)
    if dist.get_world_size(group=col_pg) > 1:
        k8, v8 = _all_gather_seq([k8, v8], col_pg)
    out, stats = fmha_fp8_stats(q8, k8, v8, scale_v)
    if rows > 1:
        # Reduce-scatter along S: exchange partials, then merge by LSE.
        o_recv = _exchange_dest_major(out, rows, row_pg)
        st_recv = _exchange_dest_major(stats.view(batch, rows * seq_u, heads, 2), rows, row_pg)
        out = lse_combine(
            o_recv.view(rows, batch * seq_u, heads, HEAD_DIM),
            st_recv.view(rows, batch * seq_u, heads, 2),
        )
    return out.view(batch, seq_u, heads, HEAD_DIM)


def ulysses_self_attention(qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, pg):
    """Ulysses fused attention; the overlap op itself falls back when it cannot overlap."""
    if ulysses_overlap.supported(num_heads, dist.get_world_size(group=pg), _ULYSSES_GROUPS):
        return ulysses_overlap.fp8_self_attention_ulysses_overlap(
            qkv,
            norm_q_w,
            norm_k_w,
            cos,
            sin,
            num_heads,
            eps,
            interleave,
            pg.group_name,
            _ULYSSES_GROUPS,
        )
    return fp8_self_attention_ulysses(
        qkv, norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave, pg
    )


def fused_self_attention(mode, groups, qkv, *args):
    """Fused path for a supported sp_mode() result.

    args = (norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave), in this order;
    attn2d appends the group names after them.
    """
    if mode == "attn2d":
        return fp8_self_attention_attn2d_op(qkv, *args, *groups)
    if mode == "ulysses":
        return ulysses_self_attention(qkv, *args, groups)
    return fp8_self_attention(qkv, *args)


def supports(attn) -> bool:
    """Static shape conditions of the fused kernels: head_dim 128, <=64 local heads, MHA,
    full-width QK RMSNorm."""
    return (
        attn.head_dim == HEAD_DIM
        and attn.local_num_attention_heads <= 64
        and attn.local_num_attention_heads == attn.local_num_key_value_heads
        and getattr(attn.norm_q, "weight", None) is not None
        and attn.norm_q.weight.numel() == attn.local_q_dim
    )


def attn_args(attn, freqs_cos, freqs_sin):
    """(norm_q_w, norm_k_w, cos, sin, num_heads, eps, interleave) for the fused ops."""
    return (
        attn.norm_q.weight,
        attn.norm_k.weight,
        freqs_cos,
        freqs_sin,
        attn.local_num_attention_heads,
        float(attn.eps),
        bool(attn.interleave),
    )


def module_self_attention(attn, qkv, freqs_cos, freqs_sin, timestep=None):
    """Unfused self-attention through the module's own norm/RoPE and backend."""
    attn.apply_packed_qk_norm_rope(qkv, freqs_cos, freqs_sin)
    q, k, v = qkv.split([attn.local_q_dim, attn.local_kv_dim, attn.local_kv_dim], dim=-1)
    return attn._attn_impl(q, k, v, timestep=timestep)


def self_attention(
    attn,
    qkv: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    timestep: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused FP8 self-attention for a packed-QKV Attention module; Ring falls back to the
    module's backend."""
    mode, groups = sp_mode(attn)
    if mode != "unsupported":
        return fused_self_attention(mode, groups, qkv, *attn_args(attn, freqs_cos, freqs_sin))
    return module_self_attention(attn, qkv, freqs_cos, freqs_sin, timestep)
