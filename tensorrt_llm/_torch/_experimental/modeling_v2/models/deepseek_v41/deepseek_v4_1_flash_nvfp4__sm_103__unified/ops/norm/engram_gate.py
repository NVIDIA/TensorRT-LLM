# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram gated residual update, per (token, hc copy), in one launch.

The DeepSeek-V4.1 Engram gate (``Engram.forward`` of the checkpoint's
``inference/model.py``, after the ``wkv`` projection): for every token ``t``
and hc copy ``c`` of the residual stream ``x [T, HC, D]``, with
``key [T, HC, D]``, ``value [T, D]`` and ``w = q_weight * k_weight [HC, D]``,
all in fp32::

    rstd = rsqrt(mean_d(x[t,c]^2) + eps) * rsqrt(mean_d(key[t,c]^2) + eps)
    dot  = sum_d(x[t,c] * w[c] * key[t,c]) * rstd * D**-0.5
    gate = sigmoid(copysign(sqrt(max(|dot|, clamp_value)), dot))    # 0 where token_mask is False
    out[t,c] = x[t,c] + gate * value[t]                              # one rounding to out's dtype

The statistics are per (token, copy) over ``D``, never joint over the copies.

The kernel is a Triton kernel defined in this file (a port of SGLang's
``fused_engram_gate``), not a ``torch.ops.trtllm`` op: no engine op computes
this gate.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Launch configuration: fixed, no autotune. One program per (token, hc copy)
# row, the whole row in one block of next_pow2(D) lanes, _NUM_WARPS warps.
_NUM_WARPS = 8


@triton.jit
def _engram_gate_kernel(
    x_ptr,
    key_ptr,
    value_ptr,
    qw_ptr,
    kw_ptr,
    mask_ptr,
    out_ptr,
    gate_ptr,
    stride_xt,
    stride_xc,
    stride_kt,
    stride_kc,
    stride_vt,
    stride_qc,
    stride_wc,
    stride_ot,
    stride_oc,
    stride_mt,
    stride_gt,
    stride_gc,
    eps,
    clamp_value,
    D: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    WRITE_GATE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    t = row // HC
    c = row - t * HC
    col = tl.arange(0, BLOCK)
    live = col < D
    x = tl.load(x_ptr + t * stride_xt + c * stride_xc + col, mask=live, other=0.0).to(tl.float32)
    key = tl.load(key_ptr + t * stride_kt + c * stride_kc + col, mask=live, other=0.0).to(
        tl.float32
    )
    qw = tl.load(qw_ptr + c * stride_qc + col, mask=live, other=0.0).to(tl.float32)
    kw = tl.load(kw_ptr + c * stride_wc + col, mask=live, other=0.0).to(tl.float32)
    w = qw * kw
    # mean as sum * fp32(1/D), rsqrt / sqrt / exp / division as the CUDA math
    # library computes them (what torch's float kernels call), so the only
    # difference from the torch reference is the summation order.
    inv_d = 1.0 / D
    ms_x = tl.sum(x * x, axis=0) * inv_d
    ms_k = tl.sum(key * key, axis=0) * inv_d
    rstd = libdevice.rsqrt(ms_x + eps) * libdevice.rsqrt(ms_k + eps)
    dot = tl.sum((x * w) * key, axis=0) * rstd * (D**-0.5)
    z = libdevice.copysign(
        tl.sqrt_rn(tl.maximum(tl.abs(dot), clamp_value, propagate_nan=tl.PropagateNan.ALL)), dot
    )
    gate = tl.div_rn(1.0, 1.0 + libdevice.exp(-z))
    if HAS_MASK:
        keep = tl.load(mask_ptr + t * stride_mt) != 0
        gate = tl.where(keep, gate, 0.0)
    if WRITE_GATE:
        tl.store(gate_ptr + t * stride_gt + c * stride_gc, gate)
    v = tl.load(value_ptr + t * stride_vt + col, mask=live, other=0.0).to(tl.float32)
    tl.store(out_ptr + t * stride_ot + c * stride_oc + col, x + gate * v, mask=live)


_HC = 4  # hc_mult
_DIM = 5120  # hidden_size
_EPS = 1e-20  # rms_norm_eps
_CLAMP = 1e-6  # Engram.clamp_value
_BF16 = torch.bfloat16
_FP32 = torch.float32

#: Token counts driven against the reference in the plain call form (out=None,
#: gate_out given), decode-like to the per-rank max_num_tokens ceiling.
_CALL_T = (1, 2, 5, 37, 160, 1024, 8192, 16384)

#: The keep masks driven at T = 16384 in the multimodal target's call form.
_TARGET_MASKS = (
    "none_masked",
    "all_masked",
    "span_at_row0",
    "span_at_end",
    "one_span_mid",
    "multi_span",
    "alternating",
) + tuple(f"rand{k}" for k in range(9))


def _call_spec(t_len, masked, x_dtype=_BF16, kv_dtype=_BF16, w_dtype=_BF16, mask_dtype=torch.bool):
    return dict(
        kind="call",
        T=t_len,
        masked=masked,
        x_dtype=x_dtype,
        kv_dtype=kv_dtype,
        w_dtype=w_dtype,
        mask_dtype=mask_dtype,
        mask_stride=1,
        HC=_HC,
        D=_DIM,
        eps=_EPS,
        clamp_value=_CLAMP,
    )


def _dt(dtype):
    return str(dtype).removeprefix("torch.")


class EngramGate(OpWrapper):
    """Engram gate + residual update: ``out[t, c] = x[t, c] + gate[t, c] * value[t]``.

    One Triton launch over ``T * HC`` (token, copy) rows. Inside the call: both
    RMS statistics, the weighted dot, the signed-sqrt sigmoid gate, the mask,
    the broadcast of the one ``value`` row to every copy, and the output cast.
    The caller owns the Engram lookup and the ``wkv`` projection that produce
    ``key`` / ``value`` (two views of one fused buffer are fine, no copy), the
    choice of layers, applying the update before the layer's ``hc_pre`` mixing,
    and building ``token_mask``.

    ``x`` / ``key`` are ``[T, HC, D]``, ``value`` is ``[T, D]``, ``q_weight`` /
    ``k_weight`` are ``[HC, D]``, each with unit stride along ``D`` and any other
    strides. All arithmetic is fp32. ``token_mask`` (``[T]``, bool or integer,
    any stride, nonzero = keep) forces ``gate = 0`` where it is False/0. The
    result goes to ``out`` (``[T, HC, D]``, unit stride along ``D``; may be
    ``x`` itself for an in-place update) or to a new contiguous ``x.dtype``
    tensor, rounded once from fp32. ``gate_out`` (``[T, HC]``, any strides)
    receives the fp32 gate when given. Returns the output tensor (``out`` when
    given); ``T = 0`` returns it without a launch.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"T={t} {'masked' if m else 'no mask'}: new out, split wkv views, bf16",
                spec=_call_spec(t, m),
            )
            for t in _CALL_T
            for m in (False, True)
        )
        + tuple(
            Cell(
                why=f"dtypes x={_dt(xd)} kv={_dt(kd)} w={_dt(wd)}: T=160 masked",
                spec=_call_spec(160, True, xd, kd, wd),
            )
            for xd, kd, wd in ((_BF16, _FP32, _FP32), (_FP32, _FP32, _FP32), (_FP32, _BF16, _BF16))
        )
        + tuple(
            Cell(
                why=f"mask {_dt(md)} stride 3: T=64, masked rows pass x through",
                spec={**_call_spec(64, True, mask_dtype=md), "mask_stride": 3},
            )
            for md in (torch.bool, torch.int32, torch.uint8)
        )
        + (
            Cell(
                why="clamp and sign edges: |dot| 0 and 1e-12..1e2, both signs, zero rows",
                spec=dict(kind="edges", HC=_HC, D=_DIM, eps=_EPS, clamp_value=_CLAMP),
            ),
        )
        + tuple(
            Cell(
                why=f"mm T=16384 {kind}: in place, bool keep, fused wkv",
                spec=dict(
                    kind="target_masked",
                    T=16384,
                    mask=kind,
                    HC=_HC,
                    D=_DIM,
                    eps=_EPS,
                    clamp_value=_CLAMP,
                ),
            )
            for kind in _TARGET_MASKS
        )
    )

    note = """
    DeepSeek-V4.1: HC = 4 (hc_mult), D = 5120 (hidden_size), eps = 1e-20 (the
    config's rms_norm_eps, passed as norm_eps), clamp_value = 1e-6 (a constant
    of the reference's Engram.__init__, not a config field). x is the bf16 hc
    stream entering layers 1 and 14; key / value are the first 4 * 5120 and the
    last 5120 columns of the bf16 wkv output [T, 25600]; q_weight / k_weight are
    the checkpoint's bf16 [4, 5120] tensors as stored; out=x updates in place.
    token_mask is None for text-only batches; the multimodal target passes a
    contiguous bool keep (False on image-span rows) in steps holding an image
    request's context chunk, and that mask may be all True.

    Metadata consumed: none.

    Semantics the members state only by example: the statistics are per (token,
    copy) over D, never joint over the copies; the sqrt is signed and |dot| is
    clamped before it, so dot = 0 gives sigmoid(+sqrt(clamp_value)), not 0.5;
    masked rows get gate exactly 0 and equal x bit for bit; kept rows are bit
    for bit what the same call computes with token_mask=None.

    Preconditions: every tensor on the same CUDA device (a CPU tensor makes
    Triton's launcher raise ValueError); unit stride along D on x, key, value,
    the weights and out; out is x itself or overlaps no input (a partial
    overlap was not run). The kernel reads its whole row before writing it,
    which is what makes out=x safe.

    Batch invariant: one program per (token, copy) row with a fixed block, and
    T only sizes the grid, so a row does not depend on T or on its position.
    Measured on sm_103: every prefix r = 1..16384 of a 16384-token call is bit
    identical to the first r rows of it, with no mask and with two masks.

    CUDA-graph capturable: no host synchronization, no allocation when out is
    given. Certified captured at T = G and 6 G over the engine's 46 graph
    batches (82 counts, 1..1536) in the core's form: no mask, no gate_out, in
    place, key / value split views of one static fused buffer. The masked form
    runs eagerly only (captured steps are generation-only); its capture was run
    only at T = 1, 5, 160.

    Accuracy: the kernel's three sums over D run in Triton's reduction order;
    the pointwise tail uses the functions torch's float kernels call. That is
    the only difference from the reference, and rows whose sums are exact in
    any order match it bit for bit. The launch is fixed (8 warps, fp fusion
    off, no autotune); the reduction order depends on the warp count, so a
    different launch gives different last-bit gates and needs its own cells.

    Not certified: other HC or D; fp16; T > 16384 in one call; capture at
    another T; masked capture beyond T = 1, 5, 160; other eps / clamp_value;
    an out dtype other than x.dtype; a gate_out dtype other than fp32; NaN or
    inf inputs; a partial overlap of out with an input.
    """

    def raw_call(
        self,
        x: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        clamp_value: float,
        token_mask: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        gate_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_len, hc, dim = x.shape
        if out is None:
            out = torch.empty((t_len, hc, dim), dtype=x.dtype, device=x.device)
        if t_len == 0:
            return out
        _engram_gate_kernel[(t_len * hc,)](
            x,
            key,
            value,
            q_weight,
            k_weight,
            token_mask if token_mask is not None else x,
            out,
            gate_out if gate_out is not None else x,
            x.stride(0),
            x.stride(1),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            q_weight.stride(0),
            k_weight.stride(0),
            out.stride(0),
            out.stride(1),
            token_mask.stride(0) if token_mask is not None else 0,
            gate_out.stride(0) if gate_out is not None else 0,
            gate_out.stride(1) if gate_out is not None else 0,
            eps,
            clamp_value,
            D=dim,
            HC=hc,
            BLOCK=triton.next_power_of_2(dim),
            HAS_MASK=token_mask is not None,
            WRITE_GATE=gate_out is not None,
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
        )
        return out

    def is_valid(
        self,
        x: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        clamp_value: float,
        token_mask: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        gate_out: Optional[torch.Tensor] = None,
    ) -> None:
        """The layouts the kernel reads or writes as if they were the expected one.

        Each was run through the launch without these checks on sm_103 and
        returned without an error: a stride-2 view along ``D`` is read as if
        contiguous (output bit-exact on 0.1-18% of elements), and a view with
        fewer rows / columns / copies than ``x`` implies is read or written past
        its end (the parent buffer's data is used, or 8 rows past it written).
        Not guarded, because the kernel reads them correctly: any non-unit
        stride other than the one along ``D``, integer mask dtypes, a strided
        mask.
        """
        # Every guard below is a layout the kernel would otherwise read or write
        # as if it were the expected one, without an error (probed on sm_103).
        assert x.dim() == 3 and x.stride(2) == 1, (
            f"x must be [T, HC, D] with unit stride along D, got {tuple(x.shape)} {x.stride()}"
        )
        t_len, hc, dim = x.shape
        assert tuple(key.shape) == (t_len, hc, dim) and key.stride(2) == 1, (
            f"key must be [{t_len}, {hc}, {dim}] with unit stride along D, got {tuple(key.shape)} {key.stride()}"
        )
        assert tuple(value.shape) == (t_len, dim) and value.stride(1) == 1, (
            f"value must be [{t_len}, {dim}] with unit stride along D, got {tuple(value.shape)} {value.stride()}"
        )
        for name, wt in (("q_weight", q_weight), ("k_weight", k_weight)):
            assert tuple(wt.shape) == (hc, dim) and wt.stride(1) == 1, (
                f"{name} must be [{hc}, {dim}] with unit stride along D, got {tuple(wt.shape)} {wt.stride()}"
            )
        if token_mask is not None:
            assert tuple(token_mask.shape) == (t_len,), (
                f"token_mask must be [{t_len}], got {tuple(token_mask.shape)}"
            )
        if out is not None:
            assert tuple(out.shape) == (t_len, hc, dim) and out.stride(2) == 1, (
                f"out must be [{t_len}, {hc}, {dim}] with unit stride along D, got {tuple(out.shape)} {out.stride()}"
            )
        if gate_out is not None:
            assert tuple(gate_out.shape) == (t_len, hc), (
                f"gate_out must be [{t_len}, {hc}], got {tuple(gate_out.shape)}"
            )

    def reference(
        self,
        x: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        clamp_value: float,
        token_mask: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        gate_out: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(out, gate)``: the expected output and the expected fp32 gate ``[T, HC]``.

        Torch ops in the order of the checkpoint's ``Engram.forward``: per-copy
        ``mean`` over ``D``, ``rsqrt``, the product sum, the signed sqrt with
        the clamp, ``sigmoid``, ``masked_fill``, one cast to the output dtype.
        The gate is what ``gate_out`` receives (0 on masked rows) and is
        returned whether or not ``gate_out`` is given. ``x`` is the stream as it
        was before the call -- for an in-place call pass a copy taken before
        it. ``out`` only sets the output dtype (``x.dtype`` when None) and
        ``gate_out`` is ignored; neither is written.
        """
        dim = x.shape[-1]
        k = key.float()
        weight = q_weight.float() * k_weight.float()
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
        dot = (h * weight * k).sum(-1) * rstd * dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.bool().unsqueeze(-1), 0)
        out_dtype = x.dtype if out is None else out.dtype
        expected = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(out_dtype)
        return expected, gate

    def compare(
        self,
        out: tuple[torch.Tensor, Optional[torch.Tensor]],
        ref: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Gate ``(out, gate_out)`` -- the two tensors a call fills -- against ``reference``.

        ``gate_out`` may be None for a call made without one; then only the
        output is gated. The kernel differs from the reference only in the order
        of its three fp32 sums over ``D``, so:

        1. the gate at torch's default fp32 tolerance;
        2. the output at torch's default tolerance of its dtype. A per-ulp bound
           is not usable: where ``x`` and ``gate * value`` nearly cancel, one
           last-bit gate difference is several ulps of the small result.

        Measured on sm_103 at T = 1..16384 and on the golden layers: gate max abs
        diff 1.5e-08 to 8.6e-06, output bit-exact on 0.99996 to 1.0 of bf16
        elements (about 0.91 for fp32 x, max abs diff 1.5e-05).
        The test also checks the output bit for bit against ``x + gate * value``
        formed in torch from the kernel's own gate (no FMA contraction, one
        rounding), which needs ``x`` and ``value`` and so lives there. The
        joint-RMS, unsigned-sqrt, unclamped and wrong-mask controls each fail
        these gates on every case where they differ.
        """
        (got, gate), (ref_out, ref_gate) = out, ref
        assert got.dtype == ref_out.dtype, (got.dtype, ref_out.dtype)
        assert got.shape == ref_out.shape, (got.shape, ref_out.shape)
        if gate is not None:
            torch.testing.assert_close(gate, ref_gate)
        torch.testing.assert_close(got, ref_out)


engram_gate = EngramGate()
