# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused per-head RMSNorm + partial split-half RoPE + FP8 E4M3 per-tensor cast of the
MiniMax-H3 attention inputs, for the cuDNN FP8 attention path.

Per attention layer the unfused path runs, on the packed QKV projection output: two row
reductions and two pointwise kernels (QK-norm + RoPE for Q and for K, inductor-fused), then
``torch.ops.trtllm.quantize_e4m3_per_tensor`` three times (one amax kernel and one cast kernel
each, plus the clone that makes the input contiguous). Here the same work is two Triton
launches over the packed buffer:

  pass 1  per-tensor amax of the normalised, rotated Q and K and of V      (dynamic scales)
  pass 2  recompute, scale, cast to FP8; program 0 writes the three descales

With static scales (``fp8_scale_fusion=static``, scales from ``fp8_static_scales``)
pass 1 is skipped.

The scale arithmetic reproduces TensorRT-LLM's per-tensor quantizer
(``cpp/tensorrt_llm/common/cudaFp8Utils.cu``): ``scale = bf16(max(amax / 448, 1 / (448 * 512)))``,
``fp8 = cvt.rn.satfinite.e4m3(x * (1 / float(scale)))``, ``descale = float(scale)``. The norm and
RoPE arithmetic: fp32 math, one rounding to BF16 after ``x * rsqrt(mean(x^2) + eps)``, the weight
multiply and the rotation in fp32 with the BF16-rounded ``cos`` / ``sin``, one rounding to BF16
before the cast. This is not bit-identical to ``torch.compile`` of the eager norm + RoPE, which
rounds to BF16 before the rotation.

``fp8_scale_fusion`` selects ``qkstatic`` (default), ``dynamic`` or ``static``.
The private calibration and saturation logs are disabled in this scenario.
"""

from __future__ import annotations

import atexit
import json
import math
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from ..config import SCENARIO

FP8_E4M3_MAX = 448.0
MIN_SCALING_FACTOR = 1.0 / (FP8_E4M3_MAX * 512.0)

if hasattr(tl, "div_rn"):

    @triton.jit
    def _div_rn(a, b):
        return tl.div_rn(a, b)

else:

    @triton.jit
    def _div_rn(a, b):
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )


@triton.jit
def _norm_rope_tile(
    x_ptrs,
    xp_ptrs,
    mask,
    w_d,
    w_p,
    cos_v,
    sin_v,
    sign,
    inv_d,
    eps,
):
    """Normalise one [BLOCK_H, D] tile per head and rotate the leading ROT columns.

    ``x_ptrs`` address the tile, ``xp_ptrs`` the same rows with the rotary partner column
    (d + ROT/2 for the first half, d - ROT/2 for the second, d itself past ROT). Returns the
    BF16-rounded result as fp32.
    """
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    xp = tl.load(xp_ptrs, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=1)
    inv = tl.rsqrt(ss * inv_d + eps)
    yn = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w_d[None, :]
    yp = (xp * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w_p[None, :]
    y = yn * cos_v[None, :] + (sign * sin_v)[None, :] * yp
    return y.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _rope_vectors(cos_ptr, sin_ptr, t, d, ROT: tl.constexpr):
    half: tl.constexpr = ROT // 2
    is_rot = d < ROT
    partner = tl.where(d < half, d + half, tl.where(is_rot, d - half, d))
    sign = tl.where(d < half, -1.0, 1.0).to(tl.float32)
    cos_v = tl.load(cos_ptr + t * ROT + d, mask=is_rot, other=1.0)
    sin_v = tl.load(sin_ptr + t * ROT + d, mask=is_rot, other=0.0)
    # The eager path casts cos / sin to the activation dtype before the rotation.
    cos_v = cos_v.to(tl.float32).to(tl.bfloat16).to(tl.float32)
    sin_v = sin_v.to(tl.float32).to(tl.bfloat16).to(tl.float32)
    return partner, sign, cos_v, sin_v


@triton.jit
def _row_valid(len_ptr, b, t, HAS_LEN: tl.constexpr):
    """1.0 when row ``t`` of batch item ``b`` is a valid (unpadded) token, else 0.0."""
    if HAS_LEN:
        return (t < tl.load(len_ptr + b)).to(tl.float32)
    else:
        return 1.0


@triton.jit
def _qkv_amax_kernel(
    qkv_ptr,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    len_ptr,
    S,
    eps,
    HAS_LEN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    r = tl.program_id(0)
    t = r % S
    # Padded rows (Ulysses trailing padding) carry the AdaLN shift and are never read by the
    # attention; they must not set the scale.
    valid = _row_valid(len_ptr, r // S, t, HAS_LEN)
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    partner, sign, cos_v, sin_v = _rope_vectors(cos_ptr, sin_ptr, t, d, ROT)
    wq_d = tl.load(wq_ptr + d).to(tl.float32)
    wq_p = tl.load(wq_ptr + partner).to(tl.float32)
    wk_d = tl.load(wk_ptr + d).to(tl.float32)
    wk_p = tl.load(wk_ptr + partner).to(tl.float32)
    inv_d = 1.0 / D
    row = qkv_ptr + r.to(tl.int64) * (3 * H * D)
    m_q = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    m_k = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    m_v = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        offs_p = h[:, None] * D + partner[None, :]
        yq = _norm_rope_tile(
            row + offs, row + offs_p, mask, wq_d, wq_p, cos_v, sin_v, sign, inv_d, eps
        )
        yk = _norm_rope_tile(
            row + H * D + offs,
            row + H * D + offs_p,
            mask,
            wk_d,
            wk_p,
            cos_v,
            sin_v,
            sign,
            inv_d,
            eps,
        )
        v = tl.load(row + 2 * H * D + offs, mask=mask, other=0.0).to(tl.float32)
        m_q = tl.maximum(m_q, tl.abs(yq))
        m_k = tl.maximum(m_k, tl.abs(yk))
        m_v = tl.maximum(m_v, tl.abs(v))
    tl.atomic_max(amax_ptr + 0, tl.max(tl.max(m_q, axis=1), axis=0) * valid)
    tl.atomic_max(amax_ptr + 1, tl.max(tl.max(m_k, axis=1), axis=0) * valid)
    tl.atomic_max(amax_ptr + 2, tl.max(tl.max(m_v, axis=1), axis=0) * valid)


@triton.jit
def _load_amax(amax_ptr, AMAX_N: tl.constexpr, AMAX_STRIDE: tl.constexpr):
    """The per-tensor amax: one value, or the max over ``AMAX_N`` per-rank values ``AMAX_STRIDE``
    apart (the gathered symmetric-memory rows). ``max`` is exact, so the result equals
    an all-reduce MAX of the same values."""
    a = tl.load(amax_ptr)
    for j in tl.static_range(1, AMAX_N):
        a = tl.maximum(a, tl.load(amax_ptr + j * AMAX_STRIDE))
    return a


@triton.jit
def _scale_from_amax(amax):
    # cudaFp8Utils.cu: scale = (bf16) max(amax / FP8_E4M3_MAX, 1 / (FP8_E4M3_MAX * 512)); the
    # cast kernel multiplies by 1 / float(scale) and the backend dequantises with float(scale).
    s = tl.maximum(_div_rn(amax, 448.0), 1.0 / (448.0 * 512.0))
    return s.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _packed_out_offs(
    h, d, t, S, WHICH: tl.constexpr, HL: tl.constexpr, D: tl.constexpr, PLANES: tl.constexpr = 3
):
    """Element offsets of head tile ``h`` x ``d`` of token ``t`` inside the packed Ulysses send
    buffer ``[U, S, PLANES, HL, D]``: head ``h`` of the local shard goes to rank ``h // HL`` as its
    local head ``h % HL``; ``WHICH`` selects the Q / K / V plane. Rank ``u``'s chunk is contiguous,
    so one ``all_to_all_single`` on the flat buffer is the whole exchange. With ``PLANES = 1`` the
    buffer is one plane ``[U, S, HL, D]`` of the tensor-major send ``[3, U, S, HL, D]``
    (``RECV_INPLACE=1``), exchanged as three all-to-alls into contiguous Q / K / V."""
    dst = h // HL
    hl = h - dst * HL
    base = ((dst.to(tl.int64) * S + t) * PLANES + WHICH) * (HL * D) + hl.to(tl.int64) * D
    return base[:, None] + d[None, :]


@triton.jit
def _qkv_cast_kernel(
    qkv_ptr,
    wq_ptr,
    wk_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    descale_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    S,
    eps,
    STATIC: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PACKED: tl.constexpr,
    HL: tl.constexpr,
):
    r = tl.program_id(0)
    t = r % S
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    partner, sign, cos_v, sin_v = _rope_vectors(cos_ptr, sin_ptr, t, d, ROT)
    wq_d = tl.load(wq_ptr + d).to(tl.float32)
    wq_p = tl.load(wq_ptr + partner).to(tl.float32)
    wk_d = tl.load(wk_ptr + d).to(tl.float32)
    wk_p = tl.load(wk_ptr + partner).to(tl.float32)
    inv_d = 1.0 / D
    if STATIC:
        s_q = tl.load(descale_ptr + 0)
        s_k = tl.load(descale_ptr + 1)
        s_v = tl.load(descale_ptr + 2)
    else:
        s_q = _scale_from_amax(tl.load(amax_ptr + 0))
        s_k = _scale_from_amax(tl.load(amax_ptr + 1))
        s_v = _scale_from_amax(tl.load(amax_ptr + 2))
        if r == 0:
            tl.store(descale_ptr + 0, s_q)
            tl.store(descale_ptr + 1, s_k)
            tl.store(descale_ptr + 2, s_v)
    f_q = _div_rn(1.0, s_q)
    f_k = _div_rn(1.0, s_k)
    f_v = _div_rn(1.0, s_v)
    row = qkv_ptr + r.to(tl.int64) * (3 * H * D)
    out_row = r.to(tl.int64) * (H * D)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        offs_p = h[:, None] * D + partner[None, :]
        yq = _norm_rope_tile(
            row + offs, row + offs_p, mask, wq_d, wq_p, cos_v, sin_v, sign, inv_d, eps
        )
        yk = _norm_rope_tile(
            row + H * D + offs,
            row + H * D + offs_p,
            mask,
            wk_d,
            wk_p,
            cos_v,
            sin_v,
            sign,
            inv_d,
            eps,
        )
        v = tl.load(row + 2 * H * D + offs, mask=mask, other=0.0).to(tl.float32)
        if PACKED:
            # q_out_ptr is the one packed send buffer; k_out_ptr / v_out_ptr are unused.
            tl.store(
                q_out_ptr + _packed_out_offs(h, d, t, S, 0, HL, D),
                (yq * f_q).to(tl.float8e4nv),
                mask=mask,
            )
            tl.store(
                q_out_ptr + _packed_out_offs(h, d, t, S, 1, HL, D),
                (yk * f_k).to(tl.float8e4nv),
                mask=mask,
            )
            tl.store(
                q_out_ptr + _packed_out_offs(h, d, t, S, 2, HL, D),
                (v * f_v).to(tl.float8e4nv),
                mask=mask,
            )
        else:
            tl.store(q_out_ptr + out_row + offs, (yq * f_q).to(tl.float8e4nv), mask=mask)
            tl.store(k_out_ptr + out_row + offs, (yk * f_k).to(tl.float8e4nv), mask=mask)
            tl.store(v_out_ptr + out_row + offs, (v * f_v).to(tl.float8e4nv), mask=mask)


_REDUCE_GROUP = None


def set_reduce_group(group) -> None:
    """Process group whose ranks hold shards of the same sequence (Ulysses): the dynamic amax is
    all-reduced (MAX) over it before the cast so every rank casts and dequantizes with the same
    scale. ``None`` (default) means a single rank."""
    global _REDUCE_GROUP
    _REDUCE_GROUP = group


def _reduce_amax(amax: torch.Tensor) -> None:
    if _REDUCE_GROUP is not None:
        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=_REDUCE_GROUP)


def _amax_allreduce_mode() -> str:
    """``amax_allreduce`` = ``nccl`` (``all_reduce(MAX)`` on the Ulysses group) |
    ``symm`` (default): the split path gathers the per-rank amax rows through torch symmetric memory
    (``one_shot_all_reduce`` of a zero-padded ``[U, 4]`` buffer: each rank fills only its own row,
    so the sum is an exact gather) and the cast kernels take the max over the U rows. Bit-identical
    to ``nccl``: every rank casts with the same max."""
    mode = SCENARIO.amax_allreduce
    if mode not in ("nccl", "symm"):
        raise ValueError(f"amax_allreduce={mode!r}: expected nccl or symm")
    return mode


class _SymmAmax:
    """Two alternating symmetric ``[U, 4]`` fp32 buffers on the Ulysses group (row ``rank`` holds
    this rank's Q / K / V amax, column 3 is padding to 16 B). Alternating makes the zeroing for call
    n + 2 safe against a peer still reading call n."""

    def __init__(self, group, device: torch.device):
        import torch.distributed._symmetric_memory as symm_mem

        self.group_name = group.group_name
        self.rank = torch.distributed.get_rank(group=group)
        self.world = torch.distributed.get_world_size(group=group)
        enable = getattr(symm_mem, "enable_symm_mem_for_group", None)
        if enable is not None:
            enable(self.group_name)
        self.bufs = []
        for _ in range(2):
            buf = symm_mem.empty((self.world, 4), dtype=torch.float32, device=device)
            symm_mem.rendezvous(buf, self.group_name)
            buf.zero_()
            self.bufs.append(buf)
        self.i = 0

    def begin(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero the next buffer; returns (buffer, this rank's row) for the amax kernels."""
        buf = self.bufs[self.i]
        self.i ^= 1
        buf.zero_()
        return buf, buf[self.rank]

    def gather(self, buf: torch.Tensor) -> torch.Tensor:
        return torch.ops.symm_mem.one_shot_all_reduce(buf, "sum", self.group_name)


_SYMM_AMAX: Dict[Tuple[int, int], _SymmAmax] = {}


def _symm_amax_state(device: torch.device) -> Optional[_SymmAmax]:
    if _REDUCE_GROUP is None or _amax_allreduce_mode() != "symm":
        return None
    key = (id(_REDUCE_GROUP), device.index if device.index is not None else 0)
    st = _SYMM_AMAX.get(key)
    if st is None:
        st = _SymmAmax(_REDUCE_GROUP, device)
        _SYMM_AMAX[key] = st
    return st


def _launch_config() -> Tuple[int, int]:
    return (
        SCENARIO.fp8_fusion_block_h,
        SCENARIO.fp8_fusion_warps,
    )


def _split_launch() -> bool:
    """``fp8_fusion_split=1``: one launch per tensor (three amax, three cast) through the
    per-tensor kernels with the packed row stride, instead of one launch over the packed row:
    the same arithmetic, three times the programs with a third of the work each."""
    return SCENARIO.fp8_fusion_split


# With the private RECV_INPLACE path enabled, the Ulysses send buffer is tensor-major ``[3, U, S, H / U, D]`` and the
# exchange is three all-to-alls landing Q, K and V in contiguous ``[S, H / U, D]`` buffers, so the
# attention op reads them without per-tensor copies. The scenario default uses one buffer
# ``[U, S, 3, H / U, D]``, one all-to-all, strided Q / K / V views on the receive side.
RECV_INPLACE = False  # The scenario uses one packed all-to-all and strided receive views.


_VAMAX_FOLD = SCENARIO.vamax_fold


def _launch_cast_g(
    qkv,
    weights,
    cos,
    sin,
    amax,
    descale,
    outs,
    sat,
    s,
    eps,
    h,
    d,
    rot,
    width,
    grid,
    c0,
    c1,
    tail,
    block_h,
    num_warps,
    packed_hl,
    static_or_qk,
    i,
    amax_n,
    amax_stride,
    vfold_args=None,
    rope_halves_equal=False,
):
    """One grouped Q / K cast launch (tensor ``i``); ``vfold_args`` = (V view, V amax slot, valid_len)
    folds V's amax into it."""
    v, vamax, valid_len = vfold_args if vfold_args is not None else (None, None, None)
    _tensor_cast_g_kernel[grid](
        qkv[..., i * width : (i + 1) * width],
        weights[i],
        cos,
        sin,
        amax.view(-1)[i:] if amax_n > 1 else amax[i : i + 1],
        descale[i : i + 1],
        outs[i],
        sat[i : i + 1] if sat is not None else descale,
        s,
        eps,
        STATIC=static_or_qk,
        SAT_COUNT=sat is not None,
        H=h,
        D=d,
        ROT=rot,
        BLOCK_H=block_h,
        ROW_STRIDE=3 * width,
        PACKED=packed_hl > 0,
        HL=max(packed_hl, 1),
        WHICH=i,
        C0=c0,
        C1=c1,
        TAIL=tail,
        AMAX_N=amax_n,
        AMAX_STRIDE=amax_stride,
        v_ptr=v if v is not None else qkv,
        vamax_ptr=vamax if vamax is not None else descale,
        len_ptr=valid_len if valid_len is not None else descale,
        VAMAX=v is not None,
        HAS_LEN=valid_len is not None,
        HALF_EQ=rope_halves_equal,
        num_warps=num_warps,
    )


def _qk_norm_rope_fp8_split(
    qkv,
    weight_q,
    weight_k,
    cos,
    sin,
    static_descale,
    eps,
    h,
    d,
    s,
    rows,
    outs,
    block_h,
    num_warps,
    valid_len=None,
    packed_hl=0,
    qk_static=False,
    tensor_major=False,
    rope_halves_equal=False,
):
    width = h * d
    grid = (rows,)
    # Tensor-major send: outs[i] is plane i of [3, U, S, HL, D]; each kernel writes plane 0 of a
    # one-plane buffer. Row-major send: one [U, S, 3, HL, D] buffer, plane i of 3.
    planes = 1 if tensor_major else 3
    which = (0, 0, 0) if tensor_major else (0, 1, 2)
    static = static_descale is not None and not qk_static
    symm = None
    if not static:
        symm = _symm_amax_state(qkv.device)
    if static:
        # A custom op's output may not alias an input: copy the static table.
        descale = static_descale.to(torch.float32).contiguous().clone()
        amax = descale
    elif qk_static:
        # Q / K descales fixed (slots 0 and 1 of the table, fp32); V dynamic: one amax launch
        # writes slot 2, and the cast of V stores its descale there.
        if symm is not None:
            symm_buf, amax = symm.begin()
        else:
            amax = torch.zeros(3, dtype=torch.float32, device=qkv.device)
        descale = static_descale.to(torch.float32).contiguous().clone()
    else:
        if symm is not None:
            symm_buf, amax = symm.begin()
        else:
            amax = torch.zeros(3, dtype=torch.float32, device=qkv.device)
        descale = torch.empty(3, dtype=torch.float32, device=qkv.device)
    weights = (weight_q, weight_k, weight_q)
    rot = cos.shape[-1]
    sat = _sat_buffer(qkv.device, 3)
    v2 = _v2_eligible(d, rot)
    if v2:
        c0, c1, tail = _v2_groups(d, rot)
    # with Q / K static (qkstatic) and the grouped kernels, V's amax rides in the Q
    # cast launch (which does not read the amax), so that launch runs before the gather / reduce.
    vfold = _VAMAX_FOLD and qk_static and v2 and not static
    if vfold:
        _launch_cast_g(
            qkv,
            weights,
            cos,
            sin,
            amax,
            descale,
            outs,
            sat,
            s,
            eps,
            h,
            d,
            rot,
            width,
            grid,
            c0,
            c1,
            tail,
            block_h,
            num_warps,
            packed_hl,
            True,
            0,
            1,
            1,
            vfold_args=(qkv[..., 2 * width : 3 * width], amax[2:3], valid_len),
            rope_halves_equal=rope_halves_equal,
        )
    if not static and not vfold:
        for i in (2,) if qk_static else range(3):
            if v2 and i < 2:
                _tensor_amax_g_kernel[grid](
                    qkv[..., i * width : (i + 1) * width],
                    weights[i],
                    cos,
                    sin,
                    amax[i : i + 1],
                    valid_len if valid_len is not None else amax,
                    s,
                    eps,
                    HAS_LEN=valid_len is not None,
                    H=h,
                    D=d,
                    ROT=rot,
                    BLOCK_H=block_h,
                    ROW_STRIDE=3 * width,
                    C0=c0,
                    C1=c1,
                    TAIL=tail,
                    num_warps=num_warps,
                )
                continue
            _tensor_amax_kernel[grid](
                qkv[..., i * width : (i + 1) * width],
                weights[i],
                cos,
                sin,
                amax[i : i + 1],
                valid_len if valid_len is not None else amax,
                s,
                eps,
                HAS_LEN=valid_len is not None,
                NORM_ROPE=i < 2,
                H=h,
                D=d,
                ROT=rot,
                BLOCK_H=block_h,
                ROW_STRIDE=3 * width,
                num_warps=num_warps,
            )
    amax_n, amax_stride = 1, 1
    if not static:
        if symm is not None:
            amax = symm.gather(symm_buf)  # [U, 4]: row u = rank u's amax, exact
            amax_n, amax_stride = symm.world, 4
        else:
            _reduce_amax(amax)
    for i in range(3):
        if vfold and i == 0:
            continue
        if v2 and i < 2:
            _tensor_cast_g_kernel[grid](
                qkv[..., i * width : (i + 1) * width],
                weights[i],
                cos,
                sin,
                amax.view(-1)[i:] if amax_n > 1 else amax[i : i + 1],
                descale[i : i + 1],
                outs[i],
                sat[i : i + 1] if sat is not None else descale,
                s,
                eps,
                STATIC=static or qk_static,
                SAT_COUNT=sat is not None,
                H=h,
                D=d,
                ROT=rot,
                BLOCK_H=block_h,
                ROW_STRIDE=3 * width,
                PACKED=packed_hl > 0,
                HL=max(packed_hl, 1),
                WHICH=which[i],
                C0=c0,
                C1=c1,
                TAIL=tail,
                AMAX_N=amax_n,
                AMAX_STRIDE=amax_stride,
                PLANES=planes,
                HALF_EQ=rope_halves_equal,
                num_warps=num_warps,
            )
            continue
        _tensor_cast_kernel[grid](
            qkv[..., i * width : (i + 1) * width],
            weights[i],
            cos,
            sin,
            amax.view(-1)[i:] if amax_n > 1 else amax[i : i + 1],
            descale[i : i + 1],
            outs[i],
            sat[i : i + 1] if sat is not None else descale,
            s,
            eps,
            NORM_ROPE=i < 2,
            OUT_FP8=True,
            STATIC=static or (qk_static and i < 2),
            SAT_COUNT=sat is not None,
            H=h,
            D=d,
            ROT=rot,
            BLOCK_H=block_h,
            ROW_STRIDE=3 * width,
            PACKED=packed_hl > 0,
            HL=max(packed_hl, 1),
            WHICH=which[i],
            AMAX_N=amax_n,
            AMAX_STRIDE=amax_stride,
            PLANES=planes,
            num_warps=num_warps,
        )
    return descale, static, sat


def _qk_norm_rope_fp8_impl(
    qkv: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    static_descale: Optional[torch.Tensor],
    eps: float,
    num_heads: int,
    head_dim: int,
    layer_idx: int,
    valid_len: Optional[torch.Tensor],
    ulysses: int,
    rope_halves_equal=False,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    """Shared body of the two ops. ``ulysses == 0``: three ``[B, S, H * D]`` FP8 outputs.
    ``ulysses = U > 1``: one packed Ulysses send buffer ``[U, S, 3, H / U, D]`` (B must be 1)."""
    b, s, width = qkv.shape
    h, d = num_heads, head_dim
    if width != 3 * h * d:
        raise ValueError(f"qkv width {width} != 3 * {h} * {d}")
    rot = cos.shape[-1]
    qkv = qkv.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    rows = b * s
    if ulysses > 1:
        if b != 1 or h % ulysses:
            raise ValueError(
                f"packed exchange needs B == 1 and H % U == 0 (B={b}, H={h}, U={ulysses})"
            )
        hl = h // ulysses
        if RECV_INPLACE:
            send = torch.empty((3, ulysses, s, hl, d), dtype=torch.float8_e4m3fn, device=qkv.device)
            outs = (send[0], send[1], send[2])
        else:
            send = torch.empty((ulysses, s, 3, hl, d), dtype=torch.float8_e4m3fn, device=qkv.device)
            outs = (send, send, send)
    else:
        hl = 0
        q8 = torch.empty((b, s, h * d), dtype=torch.float8_e4m3fn, device=qkv.device)
        outs = (q8, torch.empty_like(q8), torch.empty_like(q8))
    block_h, num_warps = _launch_config()
    grid = (rows,)
    qk_static = _MODE == "qkstatic"
    if qk_static and static_descale is None:
        raise ValueError("fp8_scale_fusion=qkstatic needs the qk_static_descale table")
    if _split_launch():
        descale, static, sat = _qk_norm_rope_fp8_split(
            qkv,
            weight_q,
            weight_k,
            cos,
            sin,
            static_descale,
            eps,
            h,
            d,
            s,
            rows,
            outs,
            block_h,
            num_warps,
            valid_len,
            packed_hl=hl,
            qk_static=qk_static,
            tensor_major=hl > 0 and RECV_INPLACE,
            rope_halves_equal=rope_halves_equal,
        )
        if _AMAX_LOG is not None and not static:
            _record_descale(layer_idx, descale)
        if sat is not None:
            _record_sat(layer_idx, sat)
        if hl > 0:
            # The packed op returns the whole send buffer (5-D in both layouts), not plane 0.
            outs = (send, send, send)
        return outs, descale
    if qk_static:
        raise ValueError("fp8_scale_fusion=qkstatic needs fp8_fusion_split=1")
    if hl > 0 and RECV_INPLACE:
        raise ValueError("RECV_INPLACE=1 needs fp8_fusion_split=1 (per-tensor launches)")
    if static_descale is None:
        amax = torch.zeros(3, dtype=torch.float32, device=qkv.device)
        descale = torch.empty(3, dtype=torch.float32, device=qkv.device)
        _qkv_amax_kernel[grid](
            qkv,
            weight_q,
            weight_k,
            cos,
            sin,
            amax,
            valid_len if valid_len is not None else amax,
            s,
            eps,
            HAS_LEN=valid_len is not None,
            H=h,
            D=d,
            ROT=rot,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
        _reduce_amax(amax)
        static = False
    else:
        amax = static_descale  # unused in the static branch; keeps the signature
        descale = static_descale.to(torch.float32).contiguous().clone()  # no aliasing of an input
        static = True
    _qkv_cast_kernel[grid](
        qkv,
        weight_q,
        weight_k,
        cos,
        sin,
        amax,
        descale,
        outs[0],
        outs[1],
        outs[2],
        s,
        eps,
        STATIC=static,
        H=h,
        D=d,
        ROT=rot,
        BLOCK_H=block_h,
        PACKED=hl > 0,
        HL=max(hl, 1),
        num_warps=num_warps,
    )
    if _AMAX_LOG is not None and not static:
        _record_descale(layer_idx, descale)
    return outs, descale


@torch.library.custom_op("h3_lever8::qk_norm_rope_fp8_packed", mutates_args=())
def qk_norm_rope_fp8_packed(
    qkv: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    static_descale: Optional[torch.Tensor],
    eps: float,
    num_heads: int,
    head_dim: int,
    layer_idx: int,
    valid_len: Optional[torch.Tensor],
    ulysses: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The fused cast writing straight into the Ulysses send buffer ``[U, S, 3, H / U, D]`` (FP8),
    plus ``descale`` ``[3]``. ``qkv`` is ``[1, S, 3 * H * D]`` (one sequence shard); chunk ``u`` of
    the buffer holds this shard's tokens for the heads rank ``u`` owns, so a single
    ``all_to_all_single`` over the flat buffer replaces the three per-tensor exchanges and their
    permute copies. Same codes and scales as :func:`qk_norm_rope_fp8`."""
    outs, descale = _qk_norm_rope_fp8_impl(
        qkv,
        weight_q,
        weight_k,
        cos,
        sin,
        static_descale,
        eps,
        num_heads,
        head_dim,
        layer_idx,
        valid_len,
        ulysses,
    )
    return outs[0], descale


@qk_norm_rope_fp8_packed.register_fake
def _qk_norm_rope_fp8_packed_fake(
    qkv,
    weight_q,
    weight_k,
    cos,
    sin,
    static_descale,
    eps,
    num_heads,
    head_dim,
    layer_idx,
    valid_len,
    ulysses,
):
    _, s, _ = qkv.shape
    shape = (
        (3, ulysses, s, num_heads // ulysses, head_dim)
        if RECV_INPLACE
        else (ulysses, s, 3, num_heads // ulysses, head_dim)
    )
    return (
        qkv.new_empty(shape, dtype=torch.float8_e4m3fn),
        qkv.new_empty((3,), dtype=torch.float32),
    )


@torch.library.custom_op("h3_lever8::qk_norm_rope_fp8", mutates_args=())
def qk_norm_rope_fp8(
    qkv: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    static_descale: Optional[torch.Tensor],
    eps: float,
    num_heads: int,
    head_dim: int,
    layer_idx: int,
    valid_len: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP8 ``q``, ``k``, ``v`` of shape ``[B, S, H * D]`` and ``descale`` ``[3]`` fp32.

    ``qkv`` is the packed projection output ``[B, S, 3 * H * D]`` (BF16, contiguous);
    ``cos`` / ``sin`` are ``[S, ROT]``; ``static_descale`` ``[3]`` fp32 skips the amax pass.
    """
    outs, descale = _qk_norm_rope_fp8_impl(
        qkv,
        weight_q,
        weight_k,
        cos,
        sin,
        static_descale,
        eps,
        num_heads,
        head_dim,
        layer_idx,
        valid_len,
        0,
    )
    return outs[0], outs[1], outs[2], descale


@qk_norm_rope_fp8.register_fake
def _qk_norm_rope_fp8_fake(
    qkv,
    weight_q,
    weight_k,
    cos,
    sin,
    static_descale,
    eps,
    num_heads,
    head_dim,
    layer_idx,
    valid_len,
):
    b, s, _ = qkv.shape
    out = qkv.new_empty((b, s, num_heads * head_dim), dtype=torch.float8_e4m3fn)
    return (
        out,
        qkv.new_empty(out.shape, dtype=out.dtype),
        qkv.new_empty(out.shape, dtype=out.dtype),
        qkv.new_empty((3,), dtype=torch.float32),
    )


# ----------------------------------------------------------------------------------------------
# Per-tensor variant (async Ulysses: Q, K, V arrive as three [B, S, H, D] tensors)
# ----------------------------------------------------------------------------------------------


@triton.jit
def _tensor_amax_kernel(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    len_ptr,
    S,
    eps,
    HAS_LEN: tl.constexpr,
    NORM_ROPE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
):
    r = tl.program_id(0)
    t = r % S
    valid = _row_valid(len_ptr, r // S, t, HAS_LEN)
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    row = x_ptr + r.to(tl.int64) * ROW_STRIDE
    m = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    if NORM_ROPE:
        partner, sign, cos_v, sin_v = _rope_vectors(cos_ptr, sin_ptr, t, d, ROT)
        w_d = tl.load(w_ptr + d).to(tl.float32)
        w_p = tl.load(w_ptr + partner).to(tl.float32)
        inv_d = 1.0 / D
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        if NORM_ROPE:
            offs_p = h[:, None] * D + partner[None, :]
            y = _norm_rope_tile(
                row + offs, row + offs_p, mask, w_d, w_p, cos_v, sin_v, sign, inv_d, eps
            )
        else:
            y = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
        m = tl.maximum(m, tl.abs(y))
    tl.atomic_max(amax_ptr, tl.max(tl.max(m, axis=1), axis=0) * valid)


@triton.jit
def _tensor_cast_kernel(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    descale_ptr,
    out_ptr,
    sat_ptr,
    S,
    eps,
    NORM_ROPE: tl.constexpr,
    OUT_FP8: tl.constexpr,
    STATIC: tl.constexpr,
    SAT_COUNT: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    PACKED: tl.constexpr,
    HL: tl.constexpr,
    WHICH: tl.constexpr,
    AMAX_N: tl.constexpr = 1,
    AMAX_STRIDE: tl.constexpr = 1,
    PLANES: tl.constexpr = 3,
):
    r = tl.program_id(0)
    t = r % S
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    row = x_ptr + r.to(tl.int64) * ROW_STRIDE
    out_row = out_ptr + r.to(tl.int64) * (H * D)
    if NORM_ROPE:
        partner, sign, cos_v, sin_v = _rope_vectors(cos_ptr, sin_ptr, t, d, ROT)
        w_d = tl.load(w_ptr + d).to(tl.float32)
        w_p = tl.load(w_ptr + partner).to(tl.float32)
        inv_d = 1.0 / D
    if OUT_FP8:
        if STATIC:
            s = tl.load(descale_ptr)
        else:
            s = _scale_from_amax(_load_amax(amax_ptr, AMAX_N, AMAX_STRIDE))
            if r == 0:
                tl.store(descale_ptr, s)
        f = _div_rn(1.0, s)
    if SAT_COUNT:
        nsat = tl.zeros((BLOCK_H, D), dtype=tl.int32)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        if PACKED:
            out_offs = _packed_out_offs(h, d, t, S, WHICH, HL, D, PLANES)
            out_tile = out_ptr + out_offs
        else:
            out_tile = out_row + offs
        if NORM_ROPE:
            offs_p = h[:, None] * D + partner[None, :]
            y = _norm_rope_tile(
                row + offs, row + offs_p, mask, w_d, w_p, cos_v, sin_v, sign, inv_d, eps
            )
        else:
            y = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
        if OUT_FP8:
            scaled = y * f
            if SAT_COUNT:
                # Elements beyond the E4M3 range are clamped by the saturating cast.
                nsat += (tl.abs(scaled) > 448.0).to(tl.int32)
            tl.store(out_tile, scaled.to(tl.float8e4nv), mask=mask)
        else:
            tl.store(out_tile, y.to(tl.bfloat16), mask=mask)
    if SAT_COUNT:
        tl.atomic_add(sat_ptr, tl.sum(tl.sum(nsat, axis=1), axis=0))


# ----------------------------------------------------------------------------------------------
# Grouped norm + RoPE kernels (``grouped_cast=1``): the Q / K passes of the split path with
# each element normalised once. The rotated columns ``[0, ROT)`` are handled as power-of-two
# column groups whose RoPE partner is the group ``ROT / 2`` columns away (``ROT = 96``: 32 + 16
# columns and their partners at +48, the 32 unrotated columns as a tail), so the partner value is
# the already normalised tile of the other group instead of a second load and a second
# normalisation. The row statistic ``inv`` is computed from the full ``[BLOCK_H, D]`` tile with
# the same ``tl.sum`` as the shipping kernels (same reduction order), the per-element formulas
# and rounding points are unchanged (fp32 math, BF16 rounding after ``x * inv``, BF16 cos / sin,
# BF16 rounding before the scale, saturating E4M3), and the two BF16 round trips of a group pair
# go through one packed ``cvt.rn.bf16x2.f32`` per element pair (the same RNE per lane).
# ----------------------------------------------------------------------------------------------


@triton.jit
def _bf16x2_round(a, b):
    """BF16 round trip of two fp32 tiles of one shape: ``a`` and ``b`` return as fp32 values
    rounded to BF16 (RNE), one ``cvt.rn.bf16x2.f32`` per pair (``a`` in the upper half)."""
    p = tl.inline_asm_elementwise(
        "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.int32, is_pure=True, pack=1
    )
    a2 = (p & -65536).to(tl.float32, bitcast=True)
    b2 = (p << 16).to(tl.float32, bitcast=True)
    return a2, b2


@triton.jit
def _group_vectors(
    cos_ptr,
    sin_ptr,
    w_ptr,
    t,
    c0,
    ROT: tl.constexpr,
    HALF: tl.constexpr,
    HALF_EQ: tl.constexpr = False,
):
    """cos / sin (BF16-rounded as the eager path) and the norm weight for the rotated column
    group ``c0`` (first half, ``sign = -1``) and its partner group ``c0 + HALF`` (``sign = +1``).
    ``HALF_EQ``: the table's two halves hold the same values, so the partner's are not loaded."""
    c1 = c0 + HALF
    cos_a = tl.load(cos_ptr + t * ROT + c0).to(tl.float32).to(tl.bfloat16).to(tl.float32)
    sin_a = tl.load(sin_ptr + t * ROT + c0).to(tl.float32).to(tl.bfloat16).to(tl.float32)
    if HALF_EQ:
        cos_b = cos_a
        sin_b = sin_a
    else:
        cos_b = tl.load(cos_ptr + t * ROT + c1).to(tl.float32).to(tl.bfloat16).to(tl.float32)
        sin_b = tl.load(sin_ptr + t * ROT + c1).to(tl.float32).to(tl.bfloat16).to(tl.float32)
    # (sign * sin) of the shipping kernel: -1.0 * sin for the first half, 1.0 * sin for the second
    ss_a = -1.0 * sin_a
    ss_b = 1.0 * sin_b
    w_a = tl.load(w_ptr + c0).to(tl.float32)
    w_b = tl.load(w_ptr + c1).to(tl.float32)
    return cos_a, ss_a, w_a, cos_b, ss_b, w_b


@triton.jit
def _rotate_pair(x_a, x_b, inv, w_a, w_b, cos_a, ss_a, cos_b, ss_b):
    """Norm + RoPE of a column group (``x_a``) and its partner group (``x_b``), both ``[BLOCK_H,
    C]`` fp32: returns the BF16-rounded rotated values of both groups as fp32."""
    n_a = x_a * inv[:, None]
    n_b = x_b * inv[:, None]
    n_a, n_b = _bf16x2_round(n_a, n_b)
    y_a = n_a * w_a[None, :]
    y_b = n_b * w_b[None, :]
    # yn * cos + (sign * sin) * yp, contracted as fma(cos, yn, (sign * sin) * yp) like the
    # shipping kernel's PTX
    r_a = tl.fma(cos_a[None, :], y_a, ss_a[None, :] * y_b)
    r_b = tl.fma(cos_b[None, :], y_b, ss_b[None, :] * y_a)
    return _bf16x2_round(r_a, r_b)


@triton.jit
def _tensor_amax_g_kernel(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    len_ptr,
    S,
    eps,
    HAS_LEN: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    TAIL: tl.constexpr,
):
    HALF: tl.constexpr = ROT // 2
    r = tl.program_id(0)
    t = r % S
    valid = _row_valid(len_ptr, r // S, t, HAS_LEN)
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    row = x_ptr + r.to(tl.int64) * ROW_STRIDE
    inv_d = 1.0 / D
    g0 = tl.arange(0, C0)
    cos_a0, ss_a0, w_a0, cos_b0, ss_b0, w_b0 = _group_vectors(
        cos_ptr, sin_ptr, w_ptr, t, g0, ROT, HALF
    )
    m0 = tl.zeros((BLOCK_H, C0), dtype=tl.float32)
    if C1 > 0:
        g1 = C0 + tl.arange(0, C1)
        cos_a1, ss_a1, w_a1, cos_b1, ss_b1, w_b1 = _group_vectors(
            cos_ptr, sin_ptr, w_ptr, t, g1, ROT, HALF
        )
        m1 = tl.zeros((BLOCK_H, C1), dtype=tl.float32)
    if TAIL > 0:
        gt = ROT + tl.arange(0, TAIL)
        w_t = tl.load(w_ptr + gt).to(tl.float32)
        mt = tl.zeros((BLOCK_H, TAIL), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        x = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=1)
        inv = tl.rsqrt(ss * inv_d + eps)
        hm = (h < H)[:, None]
        base = row + h[:, None] * D
        x_a = tl.load(base + g0[None, :], mask=hm & (g0 < C0)[None, :], other=0.0).to(tl.float32)
        x_b = tl.load(base + (HALF + g0)[None, :], mask=hm & (g0 < C0)[None, :], other=0.0).to(
            tl.float32
        )
        r_a, r_b = _rotate_pair(x_a, x_b, inv, w_a0, w_b0, cos_a0, ss_a0, cos_b0, ss_b0)
        m0 = tl.maximum(m0, tl.maximum(tl.abs(r_a), tl.abs(r_b)))
        if C1 > 0:
            x_a = tl.load(base + g1[None, :], mask=hm & (g1 < D)[None, :], other=0.0).to(tl.float32)
            x_b = tl.load(base + (HALF + g1)[None, :], mask=hm & (g1 < D)[None, :], other=0.0).to(
                tl.float32
            )
            r_a, r_b = _rotate_pair(x_a, x_b, inv, w_a1, w_b1, cos_a1, ss_a1, cos_b1, ss_b1)
            m1 = tl.maximum(m1, tl.maximum(tl.abs(r_a), tl.abs(r_b)))
        if TAIL > 0:
            x_t = tl.load(base + gt[None, :], mask=hm & (gt < D)[None, :], other=0.0).to(tl.float32)
            n_t = (x_t * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w_t[None, :]
            mt = tl.maximum(mt, tl.abs(n_t.to(tl.bfloat16).to(tl.float32)))
    m = tl.max(tl.max(m0, axis=1), axis=0)
    if C1 > 0:
        m = tl.maximum(m, tl.max(tl.max(m1, axis=1), axis=0))
    if TAIL > 0:
        m = tl.maximum(m, tl.max(tl.max(mt, axis=1), axis=0))
    tl.atomic_max(amax_ptr, m * valid)


@triton.jit
def _tensor_cast_g_kernel(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    descale_ptr,
    out_ptr,
    sat_ptr,
    S,
    eps,
    STATIC: tl.constexpr,
    SAT_COUNT: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    PACKED: tl.constexpr,
    HL: tl.constexpr,
    WHICH: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    TAIL: tl.constexpr,
    AMAX_N: tl.constexpr = 1,
    AMAX_STRIDE: tl.constexpr = 1,
    PLANES: tl.constexpr = 3,
    v_ptr=None,
    vamax_ptr=None,
    len_ptr=None,
    VAMAX: tl.constexpr = False,
    HAS_LEN: tl.constexpr = False,
    HALF_EQ: tl.constexpr = False,
):
    HALF: tl.constexpr = ROT // 2
    r = tl.program_id(0)
    t = r % S
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    row = x_ptr + r.to(tl.int64) * ROW_STRIDE
    if VAMAX:
        # V's per-tensor amax folded into this pass (vamax_fold=1): the same
        # loads, |x| and max as _tensor_amax_kernel's identity path over V's columns of this row, so
        # the max (exact, order-free) is bit-identical; one launch fewer per block.
        vrow = v_ptr + r.to(tl.int64) * ROW_STRIDE
        vm = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    out_row = out_ptr + r.to(tl.int64) * (H * D)
    inv_d = 1.0 / D
    if STATIC:
        s = tl.load(descale_ptr)
    else:
        s = _scale_from_amax(_load_amax(amax_ptr, AMAX_N, AMAX_STRIDE))
        if r == 0:
            tl.store(descale_ptr, s)
    f = _div_rn(1.0, s)
    g0 = tl.arange(0, C0)
    cos_a0, ss_a0, w_a0, cos_b0, ss_b0, w_b0 = _group_vectors(
        cos_ptr, sin_ptr, w_ptr, t, g0, ROT, HALF, HALF_EQ
    )
    if C1 > 0:
        g1 = C0 + tl.arange(0, C1)
        cos_a1, ss_a1, w_a1, cos_b1, ss_b1, w_b1 = _group_vectors(
            cos_ptr, sin_ptr, w_ptr, t, g1, ROT, HALF, HALF_EQ
        )
    if TAIL > 0:
        gt = ROT + tl.arange(0, TAIL)
        w_t = tl.load(w_ptr + gt).to(tl.float32)
    if SAT_COUNT:
        nsat = tl.zeros((BLOCK_H,), dtype=tl.int32)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        x = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
        if VAMAX:
            vm = tl.maximum(vm, tl.abs(tl.load(vrow + offs, mask=mask, other=0.0).to(tl.float32)))
        ss = tl.sum(x * x, axis=1)
        inv = tl.rsqrt(ss * inv_d + eps)
        hm = (h < H)[:, None]
        base = row + h[:, None] * D
        # group 0 and its partner
        ma = hm & (g0 < C0)[None, :]
        x_a = tl.load(base + g0[None, :], mask=ma, other=0.0).to(tl.float32)
        x_b = tl.load(base + (HALF + g0)[None, :], mask=ma, other=0.0).to(tl.float32)
        r_a, r_b = _rotate_pair(x_a, x_b, inv, w_a0, w_b0, cos_a0, ss_a0, cos_b0, ss_b0)
        sc_a = r_a * f
        sc_b = r_b * f
        if SAT_COUNT:
            nsat += tl.sum(
                (tl.abs(sc_a) > 448.0).to(tl.int32) + (tl.abs(sc_b) > 448.0).to(tl.int32), axis=1
            )
        if PACKED:
            tl.store(
                out_ptr + _packed_out_offs(h, g0, t, S, WHICH, HL, D, PLANES),
                sc_a.to(tl.float8e4nv),
                mask=ma,
            )
            tl.store(
                out_ptr + _packed_out_offs(h, HALF + g0, t, S, WHICH, HL, D, PLANES),
                sc_b.to(tl.float8e4nv),
                mask=ma,
            )
        else:
            tl.store(out_row + h[:, None] * D + g0[None, :], sc_a.to(tl.float8e4nv), mask=ma)
            tl.store(
                out_row + h[:, None] * D + (HALF + g0)[None, :], sc_b.to(tl.float8e4nv), mask=ma
            )
        if C1 > 0:
            mb = hm & (g1 < D)[None, :]
            x_a = tl.load(base + g1[None, :], mask=mb, other=0.0).to(tl.float32)
            x_b = tl.load(base + (HALF + g1)[None, :], mask=mb, other=0.0).to(tl.float32)
            r_a, r_b = _rotate_pair(x_a, x_b, inv, w_a1, w_b1, cos_a1, ss_a1, cos_b1, ss_b1)
            sc_a = r_a * f
            sc_b = r_b * f
            if SAT_COUNT:
                nsat += tl.sum(
                    (tl.abs(sc_a) > 448.0).to(tl.int32) + (tl.abs(sc_b) > 448.0).to(tl.int32),
                    axis=1,
                )
            if PACKED:
                tl.store(
                    out_ptr + _packed_out_offs(h, g1, t, S, WHICH, HL, D, PLANES),
                    sc_a.to(tl.float8e4nv),
                    mask=mb,
                )
                tl.store(
                    out_ptr + _packed_out_offs(h, HALF + g1, t, S, WHICH, HL, D, PLANES),
                    sc_b.to(tl.float8e4nv),
                    mask=mb,
                )
            else:
                tl.store(out_row + h[:, None] * D + g1[None, :], sc_a.to(tl.float8e4nv), mask=mb)
                tl.store(
                    out_row + h[:, None] * D + (HALF + g1)[None, :], sc_b.to(tl.float8e4nv), mask=mb
                )
        if TAIL > 0:
            mc = hm & (gt < D)[None, :]
            x_t = tl.load(base + gt[None, :], mask=mc, other=0.0).to(tl.float32)
            n_t = (x_t * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w_t[None, :]
            sc_t = n_t.to(tl.bfloat16).to(tl.float32) * f
            if SAT_COUNT:
                nsat += tl.sum((tl.abs(sc_t) > 448.0).to(tl.int32), axis=1)
            if PACKED:
                tl.store(
                    out_ptr + _packed_out_offs(h, gt, t, S, WHICH, HL, D, PLANES),
                    sc_t.to(tl.float8e4nv),
                    mask=mc,
                )
            else:
                tl.store(out_row + h[:, None] * D + gt[None, :], sc_t.to(tl.float8e4nv), mask=mc)
    if SAT_COUNT:
        tl.atomic_add(sat_ptr, tl.sum(nsat, axis=0))
    if VAMAX:
        vvalid = _row_valid(len_ptr, r // S, t, HAS_LEN)
        tl.atomic_max(vamax_ptr, tl.max(tl.max(vm, axis=1), axis=0) * vvalid)


def _v2_groups(d: int, rot: int) -> Optional[Tuple[int, int, int]]:
    """``(C0, C1, TAIL)`` column groups of the grouped kernels for head dim ``d`` and RoPE width
    ``rot``, or None when the shape does not decompose into power-of-two groups."""
    half = rot // 2
    if rot <= 0 or rot % 2 or rot > d or half <= 0:
        return None
    c0 = 1 << (half.bit_length() - 1)
    c1 = half - c0
    tail = d - rot
    if c1 and (c1 & (c1 - 1)):
        return None
    if tail and (tail & (tail - 1)):
        return None
    return c0, c1, tail


_V2 = SCENARIO.grouped_cast


def _v2_eligible(d: int, rot: int) -> bool:
    return _V2 and _v2_groups(d, rot) is not None


@torch.library.custom_op("h3_lever8::norm_rope_fp8_tensor", mutates_args=())
def norm_rope_fp8_tensor(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    static_descale: Optional[torch.Tensor],
    eps: float,
    out_fp8: bool,
    valid_len: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One ``[B, S, H, D]`` tensor: per-head RMSNorm + RoPE when ``weight`` is given (Q or K),
    identity otherwise (V); output BF16, or FP8 E4M3 with the per-tensor ``descale`` ``[1]`` fp32
    when ``out_fp8``. The BF16 output is the same value the compiled eager path produces; the FP8
    output uses the same scale arithmetic as ``qk_norm_rope_fp8``. ``S`` is the local sequence
    (``cos`` / ``sin`` are ``[S, ROT]`` for these rows). Returns ``(out, descale)``; ``descale``
    is a ``[1]`` fp32 tensor holding 1.0 for the BF16 output.
    """
    b, s, h, d = x.shape
    x = x.contiguous()
    norm_rope = weight is not None
    if norm_rope:
        if cos is None or sin is None:
            raise ValueError("norm + RoPE needs cos and sin")
        cos = cos.contiguous()
        sin = sin.contiguous()
        rot = cos.shape[-1]
    else:
        rot = 2  # unused
        cos = sin = weight = x  # placeholders, never dereferenced
    rows = b * s
    out = torch.empty(
        (b, s, h, d), dtype=torch.float8_e4m3fn if out_fp8 else torch.bfloat16, device=x.device
    )
    block_h, num_warps = _launch_config()
    grid = (rows,)
    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    if out_fp8 and static_descale is None:
        descale = torch.empty(1, dtype=torch.float32, device=x.device)
        _tensor_amax_kernel[grid](
            x,
            weight,
            cos,
            sin,
            amax,
            valid_len if valid_len is not None else amax,
            s,
            eps,
            HAS_LEN=valid_len is not None,
            NORM_ROPE=norm_rope,
            H=h,
            D=d,
            ROT=rot,
            BLOCK_H=block_h,
            ROW_STRIDE=h * d,
            num_warps=num_warps,
        )
        _reduce_amax(amax)
        static = False
    elif out_fp8:
        descale = static_descale.to(torch.float32).reshape(1).contiguous().clone()
        static = True
    else:
        descale = torch.ones(1, dtype=torch.float32, device=x.device)
        static = True
    _tensor_cast_kernel[grid](
        x,
        weight,
        cos,
        sin,
        amax,
        descale,
        out,
        descale,
        s,
        eps,
        NORM_ROPE=norm_rope,
        OUT_FP8=out_fp8,
        STATIC=static,
        SAT_COUNT=False,
        H=h,
        D=d,
        ROT=rot,
        BLOCK_H=block_h,
        ROW_STRIDE=h * d,
        PACKED=False,
        HL=1,
        WHICH=0,
        num_warps=num_warps,
    )
    return out, descale


@norm_rope_fp8_tensor.register_fake
def _norm_rope_fp8_tensor_fake(x, weight, cos, sin, static_descale, eps, out_fp8, valid_len):
    return (
        x.new_empty(x.shape, dtype=torch.float8_e4m3fn if out_fp8 else torch.bfloat16),
        x.new_empty((1,), dtype=torch.float32),
    )


def norm_rope_tensor(
    x: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    eps: float,
    fp8: bool = False,
    static_descale: Optional[torch.Tensor] = None,
    valid_len: Optional[torch.Tensor] = None,
):
    """Per-tensor entry point for the async Ulysses path.

    ``norm_rope_tensor(q, norm_q.weight, cos, sin, eps)`` -> BF16 ``[B, S, H, D]``;
    ``norm_rope_tensor(q, norm_q.weight, cos, sin, eps, fp8=True)`` -> ``(q_fp8, descale[1])``;
    ``norm_rope_tensor(v, None, None, None, eps, fp8=True)`` casts V without norm or RoPE.
    The three descales concatenated (``torch.cat``) are the ``fp8_descales`` the cuDNN backend takes.
    """
    out, descale = torch.ops.h3_lever8.norm_rope_fp8_tensor(
        x, norm_weight, cos, sin, static_descale, float(eps), bool(fp8), valid_len
    )
    return (out, descale) if fp8 else out


# ----------------------------------------------------------------------------------------------
# Mode, static scales and calibration log
# ----------------------------------------------------------------------------------------------


def _read_fusion_mode() -> str:
    mode = SCENARIO.fp8_scale_fusion
    if mode in ("", "0", "off", "false", "no"):
        return "off"
    if mode in ("1", "dynamic", "dyn", "on"):
        return "dynamic"
    if mode == "static":
        return "static"
    if mode in ("qkstatic", "qk-static"):
        return "qkstatic"
    raise ValueError(f"fp8_scale_fusion={mode!r}: expected 0, dynamic, static or qkstatic")


# Read once at import so the compiled transformer sees a constant (no environment access in
# the traced forward).
_MODE = _read_fusion_mode()


def fusion_mode() -> str:
    """``off`` | ``dynamic`` | ``static`` | ``qkstatic`` from ``fp8_scale_fusion`` (read at
    import from the frozen process config). ``qkstatic``: Q and K take the fixed descale of
    :func:`qk_static_descale` (the softmax scale folded into the codes); V keeps its dynamic per-tensor amax."""
    return _MODE


_QK_STATIC_CACHE: Dict[Tuple[int, torch.device], torch.Tensor] = {}


def qk_static_descale_value(head_dim: int) -> float:
    """Fixed Q and K descale ``s`` with ``s * s * sm_scale * log2(e) == 1``, ``sm_scale =
    head_dim ** -0.5``: the FP8 codes then satisfy ``q8 . k8 = q . k * sm_scale * log2 e``, the
    base-2 logit an FMHA kernel with a unit ``scaleSoftmaxLog2`` consumes."""
    return math.sqrt(1.0 / ((1.0 / math.sqrt(head_dim)) * math.log2(math.e)))


@torch.compiler.disable
def qk_static_descale(head_dim: int, device: torch.device) -> torch.Tensor:
    """``[3]`` fp32 ``[s, s, 0]`` for the ``qkstatic`` mode: slots 0 and 1 are the fixed Q / K
    descale (fp32, not rounded to BF16: the fold must hold to fp32 precision), slot 2 is written
    by the cast from V's amax."""
    key = (head_dim, device)
    cached = _QK_STATIC_CACHE.get(key)
    if cached is None:
        s = qk_static_descale_value(head_dim)
        cached = torch.tensor([s, s, 0.0], dtype=torch.float32, device=device)
        _QK_STATIC_CACHE[key] = cached
    return cached


_STATIC_TABLE: Optional[Dict[str, list]] = None
_STATIC_CACHE: Dict[Tuple[int, torch.device], torch.Tensor] = {}


@torch.compiler.disable
def static_descale(layer_idx: int, device: torch.device) -> torch.Tensor:
    """Per-layer ``[3]`` fp32 descale (q, k, v) from the JSON named by ``fp8_static_scales``.

    The file maps ``str(layer_idx)`` to ``[descale_q, descale_k, descale_v]`` (the running maxima
    written by the calibration log, optionally scaled by ``fp8_static_margin``, default 1.0).
    Each value is rounded to BF16 like the dynamic quantizer's scale.
    """
    global _STATIC_TABLE
    key = (layer_idx, device)
    cached = _STATIC_CACHE.get(key)
    if cached is not None:
        return cached
    if _STATIC_TABLE is None:
        path = SCENARIO.fp8_static_scales
        if not path:
            raise RuntimeError("fp8_scale_fusion=static needs fp8_static_scales=<json>")
        with open(path) as f:
            table = json.load(f)
        # Either a bare {layer: [dq, dk, dv]} map or the calibration log's {"descale_max": {...}}.
        _STATIC_TABLE = table.get("descale_max", table)
    margin = float(SCENARIO.fp8_static_margin)
    values = _STATIC_TABLE[str(layer_idx)]
    t = torch.tensor([float(v) * margin for v in values], dtype=torch.float32)
    t = t.clamp_min(MIN_SCALING_FACTOR).to(torch.bfloat16).to(torch.float32).to(device)
    _STATIC_CACHE[key] = t
    return t


_AMAX_LOG = None  # the study's amax / saturation logs are not part of the scenario
_SAT_LOG = None
_RUNNING: Dict[int, torch.Tensor] = {}
_SAT: Dict[int, torch.Tensor] = {}
_SAT_CALLS: Dict[int, int] = {}


def _sat_buffer(device: torch.device, n: int) -> Optional[torch.Tensor]:
    """Per-call saturation counters (elements clipped at +-448), only with ``_SAT_LOG``."""
    if _SAT_LOG is None:
        return None
    return torch.zeros(n, dtype=torch.int32, device=device)


def _record_sat(layer_idx: int, sat: torch.Tensor) -> None:
    total = _SAT.get(layer_idx)
    if total is None:
        _SAT[layer_idx] = sat.to(torch.int64)
    else:
        total += sat
    _SAT_CALLS[layer_idx] = _SAT_CALLS.get(layer_idx, 0) + 1


_CALLS: Dict[int, int] = {}


def _record_descale(layer_idx: int, descale: torch.Tensor) -> None:
    running = _RUNNING.get(layer_idx)
    if running is None:
        _RUNNING[layer_idx] = descale.clone()
    else:
        torch.maximum(running, descale, out=running)
    _CALLS[layer_idx] = _CALLS.get(layer_idx, 0) + 1


def _dump_amax_log() -> None:
    if _AMAX_LOG is None or not _RUNNING:
        return
    table = {str(k): [float(x) for x in v.cpu().tolist()] for k, v in sorted(_RUNNING.items())}
    payload = {
        "descale_max": table,
        "calls": {str(k): v for k, v in sorted(_CALLS.items())},
        "note": "descale = max(amax / 448, 1 / (448 * 512)) rounded to bf16; q, k, v per layer",
    }
    path = _AMAX_LOG
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)


def _dump_sat_log() -> None:
    if _SAT_LOG is None or not _SAT:
        return
    table = {str(k): [int(x) for x in v.cpu().tolist()] for k, v in sorted(_SAT.items())}
    payload = {
        "saturated_elements": table,
        "calls": {str(k): v for k, v in sorted(_SAT_CALLS.items())},
        "note": "elements with |x / scale| > 448 before the saturating FP8 cast, run total; q, k, v per layer",
    }
    with open(_SAT_LOG, "w") as f:
        json.dump(payload, f, indent=1)


atexit.register(_dump_amax_log)
atexit.register(_dump_sat_log)
