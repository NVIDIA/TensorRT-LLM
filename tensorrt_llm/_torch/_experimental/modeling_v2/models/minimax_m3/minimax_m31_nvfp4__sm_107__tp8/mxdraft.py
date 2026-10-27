# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXDRAFT: the DSpark drafter's decode projections on this target's MXFP8 GEMV.

The drafter checkpoint ships its linears as MXFP8 (E4M3 codes, one UE8M0 scale per 32 columns) and the stock drafter
dequantizes them to BF16 at load, so every decode pass streams twice the bytes it needs. MXDRAFT recovers the MXFP8
form of each BF16 weight (exactly: the recovery is checked to dequantize back to the same BF16 bits, else the
projection is left alone) and routes the projection's GEMM to ``m3_mx_gemv_bf16`` for decode inputs it serves; any
other call runs the Linear unchanged. The activation is quantized to MXFP8 inside the GEMV, so the output is not the
stock BF16 GEMM's: drafter-only precision, gated on natural AL. Row-parallel projections keep
their all-reduce (only the Linear's GEMM is replaced).
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from .kernels.mx_gemv import op as _gemv
from .ops.comm.m3_mc_allreduce_norm import swizzle_sf

E4M3_MAX = 448.0
BLOCK = 32


def mxfp8_from_bf16(weight: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The MXFP8 form (E4M3 codes [N, K], UE8M0 scale bytes [N, K / 32]) of a BF16 weight that is exactly an MXFP8
    dequantization, or None. Each block takes the smallest power-of-two scale that fits its amax in E4M3, which is at
    most the shipped scale, so the codes are exact whenever the weight came from MXFP8."""
    if weight.dtype != torch.bfloat16 or weight.dim() != 2 or weight.shape[1] % BLOCK != 0:
        return None
    n, k = weight.shape
    blocks = weight.float().view(n, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(-1)
    exp = torch.ceil(torch.log2(amax.clamp_min(2.0**-126) / E4M3_MAX)).clamp(-127, 127)
    scale = torch.exp2(exp).unsqueeze(-1)
    codes = (blocks / scale).to(torch.float8_e4m3fn)
    back = (codes.float() * scale).view(n, k).to(torch.bfloat16)
    if not torch.equal(back, weight):
        return None
    return codes.view(n, k).contiguous(), (exp + 127).to(torch.uint8).contiguous()


def _projections(draft_model) -> Iterable[torch.nn.Module]:
    for layer in getattr(getattr(draft_model, "model", None), "layers", []) or []:
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        for lin in (
            getattr(attn, "qkv_proj", None),
            getattr(attn, "o_proj", None),
            getattr(mlp, "gate_up_proj", None),
            getattr(mlp, "down_proj", None),
        ):
            if lin is not None:
                yield lin


def route_mx_linears(draft_model, rows: Iterable[int]) -> int:
    """Route each drafter projection whose BF16 weight is an exact MXFP8 dequantization to the MXFP8 GEMV for 2-D
    contiguous bf16 inputs of the given row counts (the decode passes'); compiles those kernels now (outside capture).
    Returns how many projections were routed."""
    rows = sorted({int(m) for m in rows if 0 < int(m) <= _gemv.MAX_TOKENS})
    n_routed = 0
    for lin in _projections(draft_model):
        weight = getattr(lin, "weight", None)
        if (
            weight is None
            or getattr(lin, "bias", None) is not None
            or getattr(lin, "_m31_mxdraft", None) is not None
        ):
            continue
        rec = mxfp8_from_bf16(weight.data)
        if rec is None:
            continue
        n, k = weight.shape
        served = tuple(m for m in rows if _gemv.serves(n, k, m, True))
        if not served:
            continue
        w8, sf = rec[0], swizzle_sf(rec[1])
        for m in served:
            _gemv.precompile(w8, sf, m, True)
        original = lin.apply_linear

        def apply_linear(
            input,
            bias,
            lora_params=None,
            layer_idx=None,
            *,
            _o=original,
            _w8=w8,
            _sf=sf,
            _served=served,
        ):
            if (
                bias is None
                and not lora_params
                and isinstance(input, torch.Tensor)
                and input.dtype == torch.bfloat16
                and input.dim() == 2
                and input.shape[0] in _served
                and input.is_contiguous()
            ):
                out = _gemv.m3_mx_gemv_bf16(input, _w8, _sf)
                if out is not None:
                    return out
            return _o(input, bias, lora_params, layer_idx)

        lin._m31_mxdraft = (w8, sf, served)
        lin.apply_linear = apply_linear
        n_routed += 1
    return n_routed


# DRAFT_GU_SWIGLU_EPI (F5, exact): the drafter MLP's gate_up GEMV writes swigluoai(gate, up) from its epilogue (the
# interleaved gate_up weight, m3_mx_gemv_gu_swiglu) instead of the bf16 gate | up rows plus the separate Triton
# silu_and_mul launch; bit for bit the same h (tests: test_mxdraft_gu_swiglu.py). Kill switch
# TRTLLM_M31_DISABLE_DRAFT_GU_SWIGLU_EPI=1. Engagement: "[M3.1 mxdraft] DRAFT_GU_SWIGLU_EPI taken rows=<m>" once per
# row count and process, from the code doing the work.
DRAFT_GU_SWIGLU_EPI_PATH = "DRAFT_GU_SWIGLU_EPI"
_GU_SWIGLU = (7.0, 1.702, 1.0)  # (limit, alpha, beta) of the kernel's swigluoai
_gu_logged: set = set()


def route_gu_swiglu(draft_model, rows: Iterable[int]) -> int:
    """Fold each drafter MLP's SwiGLU into its gate_up GEMV epilogue where the MLP is the swigluoai GatedMLP the kernel
    implements; compiles those kernels now (outside capture). Returns how many MLPs were routed."""
    import os

    if os.environ.get("TRTLLM_M31_DISABLE_DRAFT_GU_SWIGLU_EPI", "0") == "1":
        return 0
    import torch.nn.functional as F

    rows = sorted({int(m) for m in rows if 0 < int(m) <= _gemv.MAX_TOKENS})
    n_routed = 0
    for layer in getattr(getattr(draft_model, "model", None), "layers", []) or []:
        mlp = getattr(layer, "mlp", None)
        gu = getattr(mlp, "gate_up_proj", None)
        if (
            gu is None
            or getattr(mlp, "split_gate_up", False)
            or getattr(mlp, "activation", None) is not F.silu
            or (mlp.swiglu_limit, mlp.swiglu_alpha, mlp.swiglu_beta) != _GU_SWIGLU
            or getattr(gu, "bias", None) is not None
            or getattr(gu, "_m31_gu_swiglu", None) is not None
            or gu.weight.shape[0] % 128 != 0
        ):
            continue
        rec = mxfp8_from_bf16(_gemv.gu_interleave(gu.weight.data))
        if rec is None:
            continue
        w8, sf = rec[0], swizzle_sf(rec[1])
        served = tuple(
            m for m in rows if _gemv.precompile(w8, sf, m, True, gu_swiglu=True) is not None
        )
        if not served:
            continue
        original = gu.apply_linear
        original_act = mlp._apply_activation

        def apply_linear(
            input,
            bias,
            lora_params=None,
            layer_idx=None,
            *,
            _o=original,
            _w8=w8,
            _sf=sf,
            _served=served,
        ):
            if (
                bias is None
                and not lora_params
                and isinstance(input, torch.Tensor)
                and input.dtype == torch.bfloat16
                and input.dim() == 2
                and input.shape[0] in _served
                and input.is_contiguous()
            ):
                out = _gemv.m3_mx_gemv_gu_swiglu(input, _w8, _sf)
                if out is not None:
                    out._m31_gu_swiglu = True
                    m = int(input.shape[0])
                    if m not in _gu_logged:
                        _gu_logged.add(m)
                        logger.info(f"[M3.1 mxdraft] {DRAFT_GU_SWIGLU_EPI_PATH} taken rows={m}")
                    return out
            return _o(input, bias, lora_params, layer_idx)

        def _apply_activation(x, *args, _oa=original_act, **kwargs):
            # The GEMV already applied the SwiGLU (its output is h, half the width): pass it through.
            if getattr(x, "_m31_gu_swiglu", False):
                return x
            return _oa(x, *args, **kwargs)

        gu._m31_gu_swiglu = (w8, sf, served)
        gu.apply_linear = apply_linear
        mlp._apply_activation = _apply_activation
        n_routed += 1
    return n_routed
