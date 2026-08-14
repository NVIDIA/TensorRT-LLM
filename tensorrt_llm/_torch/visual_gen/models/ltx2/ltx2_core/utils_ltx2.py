# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import Optional

import torch

from tensorrt_llm._torch.modules.linear import NVFP4LinearMethod
from tensorrt_llm._torch.utils import Fp4QuantizedTensor

# NVFP4 fused-quant AdaLN: every fused AdaLN CUDA kernel below has a ``_quant``
# template specialization that emits packed FP4 + per-16-elem FP8 (e4m3) scale
# factors (128x4 SWIZZLED layout) directly. Call sites pass ``fp4_input_scale``
# (the calibrated NVFP4 ``input_scale`` of the downstream Linear) to dispatch
# to the quant variant; passing ``None`` runs the bf16-output variant.


def get_nvfp4_input_scale(linear) -> Optional[torch.Tensor]:
    """Return the calibrated NVFP4 ``input_scale`` for a Linear-like module, or
    None if the module isn't NVFP4-quantized or hasn't been calibrated.

    Used by call sites to dispatch the fused-quant variant of the AdaLN wrappers
    when the downstream Linear can consume a pre-quantized Fp4QuantizedTensor.
    """
    if not isinstance(getattr(linear, "quant_method", None), NVFP4LinearMethod):
        return None
    input_scale = getattr(linear, "input_scale", None)
    if input_scale is None:
        return None
    # Need scaling_vector_size == 16 (NVFP4 group). Other configs not supported.
    if getattr(linear, "scaling_vector_size", None) != 16:
        return None
    # The downstream Linear must be able to CONSUME a pre-quantized FP4 tensor. It cannot when it
    # has to quantize activations locally: an NVFP4_AWQ pre_quant_scale must be folded into the (bf16)
    # input first, or dynamic quantization recomputes input_scale per forward. In those cases our FP4
    # output is rejected by NVFP4LinearMethod._input_prepare ("FP4 output was not disabled in the
    # previous layer"); mirror that guard here and fall back to the bf16-output variant.
    if getattr(linear, "pre_quant_scale", None) is not None:
        return None
    if getattr(linear, "force_dynamic_quantization", False):
        return None
    return input_scale


def get_nvfp4_self_attn_input_scale(attn) -> Optional[torch.Tensor]:
    """NVFP4 ``input_scale`` for fusing quant into a self-attention block's AdaLN, or None.

    FUSE_QKV exposes a fused ``qkv_proj``; SEPARATE_QKV (which async-Ulysses
    ``Attention.forward_async`` requires) exposes ``to_q``/``to_k``/``to_v`` instead. In
    SEPARATE_QKV the three projections share ONE input quant only when
    ``_maybe_share_qkv_quantize`` holds (modelopt calibrates q/k/v to the same input_scale);
    there ``to_q``'s scale is the correct one for a single pre-quantized AdaLN output, which
    ``forward_async`` then feeds to all three (0 extra quant launches). When there is no
    ``qkv_proj`` and sharing is not eligible, return None so the AdaLN stays bf16 and each
    projection quantizes itself. ``get_nvfp4_input_scale`` further returns None when the chosen
    projection can't consume external FP4 (no input_scale / pre_quant_scale / forced-dynamic).
    """
    qkv_proj = getattr(attn, "qkv_proj", None)
    if qkv_proj is not None:
        return get_nvfp4_input_scale(qkv_proj)
    if getattr(attn, "_maybe_share_qkv_quantize", False):
        return get_nvfp4_input_scale(getattr(attn, "to_q", None))
    return None


def to_velocity(
    sample: torch.Tensor,
    sigma: torch.Tensor,
    denoised: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert denoised prediction to flow velocity (flow-matching parameterization).

    velocity = (sample - denoised) / sigma
    """
    # Tensor sigma: keep on device. `.item()` would force a D2H sync that
    # deadlocks under nsys profiling combined with CUDA graph replay.
    # The scheduler guarantees sigma > 0 inside the denoise loop, so we
    # skip the zero check on the tensor path (re-checking would re-sync).
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    elif sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised.to(calc_dtype)) / sigma).to(sample.dtype)


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Functional RMS normalization without learnable weights.

    Used for adaptive layer norm where scale/shift come from external modulation.
    Must match the reference: torch.nn.functional.rms_norm.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=None, eps=eps)


# Hidden dims supported by the fused AdaLN CUDA kernels. Resolved once at
# block construction into a boolean flag; call sites do not introspect tensors.
# Non-matching shapes/dtypes raise from the kernel's TLLM_CHECK -- this module
# does no runtime input validation.
_FUSED_ADALN_SUPPORTED_DIMS = (2048, 4096)


def is_fused_adaln_supported_dim(dim: int) -> bool:
    return dim in _FUSED_ADALN_SUPPORTED_DIMS


def _maybe_reshape_fp4(out_fp4: torch.Tensor, orig_shape: torch.Size, D: int) -> torch.Tensor:
    """Reshape the kernel's flat [M, D/2] FP4 output back to the caller's
    original rank (e.g. [B, S, D/2]) when the input was higher rank.
    """
    if len(orig_shape) != 2:
        return out_fp4.reshape(*orig_shape[:-1], D // 2)
    return out_fp4


def apply_fused_rmsnorm_shift_scale(
    x: torch.Tensor,
    scale_table: torch.Tensor,
    scale_ts: torch.Tensor,
    shift_table: torch.Tensor,
    shift_ts: torch.Tensor,
    eps: float,
    fuse: bool,
    *,
    fp4_input_scale: Optional[torch.Tensor] = None,
) -> "torch.Tensor | Fp4QuantizedTensor":
    """Fused RMSNorm + AdaLN affine modulation.

    Modulator is composed inline by the C++ op:
        scale[b, t, d] = scale_table[d] + scale_ts[b, t, d]
        shift[b, t, d] = shift_table[d] + shift_ts[b, t, d]
        out            = rms_norm(x) * (1 + scale) + shift

    Returns bf16 when ``fp4_input_scale is None``, else an ``Fp4QuantizedTensor``
    holding the packed FP4 + 128x4 SWIZZLED SF.

    ``fuse`` is the top-level dispatch knob: when False, runs the eager torch
    expression (always bf16 output; Linear handles its own quant -- see the
    autotuner-bug note below). Resolve ``fuse`` once at module construction.
    """
    D = x.size(-1)

    if not fuse:
        # Eager fallback returns bf16 even when fp4_input_scale is provided: routing
        # through tunable_fp4_quantize hits a known FP4 tactic bug on the audio
        # CFG-doubled shape (252, 2048) -- "sfVecSize can only be 32, when sfUseUE8M0
        # is true" -- that crashes pipeline load. Linear's internal quantize path
        # bypasses the autotuner and works.
        del fp4_input_scale
        scale = scale_table.to(scale_ts.dtype) + scale_ts
        shift = shift_table.to(shift_ts.dtype) + shift_ts
        return rms_norm(x, eps=eps) * (1 + scale) + shift

    assert x.is_contiguous(), "x must be contiguous (fused kernel assumes flat row layout)"

    if fp4_input_scale is None:
        (out,) = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
            x.view(-1, D),
            scale_table=[scale_table],
            scale_ts=[scale_ts],
            shift_table=[shift_table],
            shift_ts=[shift_ts],
            eps=eps,
        )
        return out.view_as(x)

    out_fp4, out_sf = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x.view(-1, D),
        scale_table=[scale_table],
        scale_ts=[scale_ts],
        shift_table=[shift_table],
        shift_ts=[shift_ts],
        sf_scale=[fp4_input_scale],
        eps=eps,
    )
    out_fp4 = _maybe_reshape_fp4(out_fp4, x.shape, D)
    return Fp4QuantizedTensor(out_fp4, out_sf)


def apply_shift_scale(
    x_normed: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Affine-only AdaLN modulation: ``x_normed * (1 + scale) + shift``.

    Used for the LTX-2 output head and rare cross-attention rms-precomputed
    fallback sites (fires once per inference / never in the hot path). Always
    runs eager -- the dedicated CUDA kernel was deleted because the saving on
    these rare call sites did not justify the maintenance burden.
    """
    return x_normed * (1 + scale) + shift


def apply_fused_gate_resid_rmsnorm_shift_scale(
    x: torch.Tensor,
    attn: torch.Tensor,
    gate_table: torch.Tensor,
    gate_ts: torch.Tensor,
    scale_table: torch.Tensor,
    scale_ts: torch.Tensor,
    shift_table: torch.Tensor,
    shift_ts: torch.Tensor,
    eps: float,
    fuse: bool,
    *,
    fp4_input_scale: Optional[torch.Tensor] = None,
) -> "tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, Fp4QuantizedTensor]":
    """Fused gate-residual + RMSNorm + shift_scale:
    ``x_new = x + attn * gate``; ``normed = rms_norm(x_new)``;
    ``out = (1+scale)*normed + shift``.

    Each modulator is built inline by the C++ op:
        gate[b,t,d]  = gate_table[d]  + gate_ts[b,t,d]
        scale[b,t,d] = scale_table[d] + scale_ts[b,t,d]
        shift[b,t,d] = shift_table[d] + shift_ts[b,t,d]

    Returns ``(x_new, out_shift_scaled)``. The op is functional: ``x`` is read-only
    and ``x_new`` is a fresh buffer (callers rebind ``x`` to the result).

    When ``fp4_input_scale`` is provided, the second tuple element is an
    ``Fp4QuantizedTensor`` holding packed FP4 + 128x4 SWIZZLED SF instead of bf16.
    """
    D = x.size(-1)

    if not fuse:
        # Eager fallback: bf16 output (see apply_fused_rmsnorm_shift_scale autotuner-bug note).
        del fp4_input_scale
        gate = gate_table.to(gate_ts.dtype) + gate_ts
        scale = scale_table.to(scale_ts.dtype) + scale_ts
        shift = shift_table.to(shift_ts.dtype) + shift_ts
        x_new = x + attn * gate
        normed = rms_norm(x_new, eps=eps)
        return x_new, normed * (1 + scale) + shift

    assert x.is_contiguous(), "x must be contiguous (fused kernel assumes flat row layout)"
    assert attn.is_contiguous(), "attn must be contiguous"

    if fp4_input_scale is None:
        x_new, out = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
            x.view(-1, D),
            attn=attn.view(-1, D),
            gate_table=gate_table,
            gate_ts=gate_ts,
            scale_table=[scale_table],
            scale_ts=[scale_ts],
            shift_table=[shift_table],
            shift_ts=[shift_ts],
            eps=eps,
        )
        return x_new.view_as(x), out.view_as(x)

    x_new, out_fp4, out_sf = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x.view(-1, D),
        attn=attn.view(-1, D),
        gate_table=gate_table,
        gate_ts=gate_ts,
        scale_table=[scale_table],
        scale_ts=[scale_ts],
        shift_table=[shift_table],
        shift_ts=[shift_ts],
        sf_scale=[fp4_input_scale],
        eps=eps,
    )
    out_fp4 = _maybe_reshape_fp4(out_fp4, x.shape, D)
    return x_new.view_as(x), Fp4QuantizedTensor(out_fp4, out_sf)


def apply_fused_gate_resid_rmsnorm(
    x: torch.Tensor,
    attn_out: torch.Tensor,
    gate_table: torch.Tensor,
    gate_ts: torch.Tensor,
    eps: float,
    fuse: bool,
    *,
    fp4_input_scale: Optional[torch.Tensor] = None,
) -> "tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, Fp4QuantizedTensor]":
    """Fused gate-residual + RMSNorm (no shift_scale):
    ``x_new = x + attn_out * gate``; ``normed = rms_norm(x_new)``; optional
    NVFP4 quant of ``normed``.

    Gate modulator is composed inline by the C++ op from a (table, ts) pair:
        gate[b, t, d] = gate_table[d].to(bf16) + gate_ts[b, t, d]
    The kernel folds the broadcast-add prep AND the ``attn * gate`` mul into Phase 0b.

    Used between MSA self-attn and text cross-attn. Any perturbation mask is applied
    to ``attn_out`` BEFORE this call (mask is commutative with the gate multiply).

    Returns ``(x_new, normed)`` (or ``(x_new, Fp4QuantizedTensor)`` when
    ``fp4_input_scale`` is provided). The op is functional: ``x`` is read-only
    and ``x_new`` is a fresh buffer (callers rebind ``x`` to the result).
    """
    D = x.size(-1)

    if not fuse:
        # Eager fallback: bf16 output (see apply_fused_rmsnorm_shift_scale autotuner-bug note).
        del fp4_input_scale
        gate = gate_table.to(gate_ts.dtype) + gate_ts
        x_new = x + attn_out * gate
        return x_new, rms_norm(x_new, eps=eps)

    assert x.is_contiguous(), "x must be contiguous (fused kernel assumes flat row layout)"
    assert attn_out.is_contiguous(), "attn_out must be contiguous"

    if fp4_input_scale is not None:
        # Quant variant: residual_add + gate_mul + rms_norm + NVFP4 quant in one kernel.
        # Kernel returns flat [num_tokens, D/2] FP4; reshape back to caller's
        # rank so downstream (e.g. attn2 Q-norm) sees the same [B, S, D/2].
        x_new, out_fp4, out_sf = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
            x.view(-1, D),
            attn=attn_out.view(-1, D),
            gate_table=gate_table,
            gate_ts=gate_ts,
            sf_scale=[fp4_input_scale],
            eps=eps,
        )
        out_fp4 = _maybe_reshape_fp4(out_fp4, x.shape, D)
        return x_new.view_as(x), Fp4QuantizedTensor(out_fp4, out_sf)

    # bf16 variant: residual_add + gate_mul + rms_norm, no quant.
    x_new, out = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x.view(-1, D), attn=attn_out.view(-1, D), gate_table=gate_table, gate_ts=gate_ts, eps=eps
    )
    return x_new.view_as(x), out.view_as(x)


def apply_fused_gate_resid(
    x: torch.Tensor,
    attn_out: torch.Tensor,
    gate_table: torch.Tensor,
    gate_ts: torch.Tensor,
    fuse: bool,
) -> torch.Tensor:
    """Fused gate-residual (no RMSNorm, no shift_scale):
    ``gate[b,d] = gate_table[d].to(bf16) + gate_ts[b,d]``
    ``x_new   = x + attn_out * gate``

    Used by the FFN output gate site: ``vx = vx + ff(vx_scaled) * vgate_mlp``,
    where ``(gate_table, gate_ts)`` is the pair-form of the FFN gate modulator
    (slot 5 of the AdaLN table). Functional: ``x`` is read-only; ``x_new`` is a fresh buffer.
    """
    D = x.size(-1)

    if not fuse:
        # Eager fallback: gate = gate_table + gate_ts is [B, T_t, D]; reshape x/attn to
        # [B, T, D] (T via -1 to stay torch.compile-safe under symbolic shapes) so the
        # per-batch gate broadcasts over the T tokens. ``view(B, -1, D)`` avoids the
        # static reshape of the symbolic T_t dim that a ``gate_ts.view(B, D)`` would force.
        gate = gate_table.to(gate_ts.dtype) + gate_ts
        B = gate_ts.shape[0]
        return (x.view(B, -1, D) + attn_out.view(B, -1, D) * gate).view_as(x)

    assert x.is_contiguous(), "x must be contiguous (fused kernel assumes flat row layout)"
    assert attn_out.is_contiguous(), "attn_out must be contiguous"
    # num_out=0 -> gate-residual only (no norm): op returns [x_new] (x + attn_out * gate).
    (x_new,) = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
        x.view(-1, D), attn=attn_out.view(-1, D), gate_table=gate_table, gate_ts=gate_ts, num_out=0
    )
    return x_new.view_as(x)


def apply_fused_resid_rmsnorm_shift_scale_dual(
    x: torch.Tensor,
    attn2_out: torch.Tensor,
    scale1_table: torch.Tensor,
    scale1_ts: torch.Tensor,
    shift1_table: torch.Tensor,
    shift1_ts: torch.Tensor,
    scale2_table: torch.Tensor,
    scale2_ts: torch.Tensor,
    shift2_table: torch.Tensor,
    shift2_ts: torch.Tensor,
    eps: float,
    fuse: bool,
    *,
    fp4_input_scale1: Optional[torch.Tensor] = None,
    fp4_input_scale2: Optional[torch.Tensor] = None,
) -> (
    "tuple[torch.Tensor, torch.Tensor, torch.Tensor] "
    "| tuple[torch.Tensor, Fp4QuantizedTensor, Fp4QuantizedTensor]"
):
    """Fused residual + RMSNorm + dual shift_scale:
    ``x_new = x + attn2_out``; ``normed = rms_norm(x_new)``;
    ``out1 = (1+scale1)*normed + shift1``; ``out2 = (1+scale2)*normed + shift2``.

    Each modulator is built inline by the C++ op:
        m[b,t,d] = m_table[d].to(bf16) + m_ts[b,t,d]    (bf16 narrow first, bf16 add)

    Returns ``(x_new, out1, out2)``. The op is functional: ``x`` is read-only
    and ``x_new`` is a fresh buffer (callers rebind ``x`` to the result).

    When both ``fp4_input_scale1`` and ``fp4_input_scale2`` are provided, the
    second and third tuple elements are ``Fp4QuantizedTensor`` (FP4 + SWIZZLED SF).
    Both must be either present or absent together (the kernel takes both or neither).
    """
    D = x.size(-1)

    if not fuse:
        # Eager fallback: bf16 outputs (see apply_fused_rmsnorm_shift_scale autotuner-bug note).
        del fp4_input_scale1, fp4_input_scale2
        scale1 = scale1_table.to(scale1_ts.dtype) + scale1_ts
        shift1 = shift1_table.to(shift1_ts.dtype) + shift1_ts
        scale2 = scale2_table.to(scale2_ts.dtype) + scale2_ts
        shift2 = shift2_table.to(shift2_ts.dtype) + shift2_ts
        x_new = x + attn2_out
        normed = rms_norm(x_new, eps=eps)
        return x_new, normed * (1 + scale1) + shift1, normed * (1 + scale2) + shift2

    assert x.is_contiguous(), "x must be contiguous (fused kernel assumes flat row layout)"
    assert attn2_out.is_contiguous(), "attn2_out must be contiguous"

    if fp4_input_scale1 is None and fp4_input_scale2 is None:
        x_new, out1, out2 = torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
            x.view(-1, D),
            attn=attn2_out.view(-1, D),
            scale_table=[scale1_table, scale2_table],
            scale_ts=[scale1_ts, scale2_ts],
            shift_table=[shift1_table, shift2_table],
            shift_ts=[shift1_ts, shift2_ts],
            eps=eps,
            num_out=2,
        )
        return x_new.view_as(x), out1.view_as(x), out2.view_as(x)

    assert fp4_input_scale1 is not None and fp4_input_scale2 is not None, (
        "dual fp4: both input_scales must be provided or both absent"
    )
    x_new, out1_fp4, out1_sf, out2_fp4, out2_sf = (
        torch.ops.trtllm.fused_dit_gate_resid_norm_shift_scale(
            x.view(-1, D),
            attn=attn2_out.view(-1, D),
            scale_table=[scale1_table, scale2_table],
            scale_ts=[scale1_ts, scale2_ts],
            shift_table=[shift1_table, shift2_table],
            shift_ts=[shift1_ts, shift2_ts],
            sf_scale=[fp4_input_scale1, fp4_input_scale2],
            eps=eps,
            num_out=2,
        )
    )
    out1_fp4 = _maybe_reshape_fp4(out1_fp4, x.shape, D)
    out2_fp4 = _maybe_reshape_fp4(out2_fp4, x.shape, D)
    return (
        x_new.view_as(x),
        Fp4QuantizedTensor(out1_fp4, out1_sf),
        Fp4QuantizedTensor(out2_fp4, out2_sf),
    )
