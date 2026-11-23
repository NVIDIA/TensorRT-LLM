# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drafter MXFP8 projections (DRAFTER_MXFP8): the DSpark drafter's decode projections on this target's MXFP8 GEMV.

The drafter checkpoint ships its linears as MXFP8 (E4M3 codes, one UE8M0 scale per 32 columns) and the stock drafter
dequantizes them to BF16 at load, so every decode pass streams twice the bytes it needs. This module recovers the MXFP8
form of each BF16 weight (exactly: the recovery is checked to dequantize back to the same BF16 bits, else the
projection is left alone) and routes the projection's GEMM to ``m3_mx_gemv_bf16`` for decode inputs it serves; any
other call runs the Linear unchanged. The activation is quantized to MXFP8 inside the GEMV, so the output is not the
stock BF16 GEMM's: drafter-only precision (it changes draft tokens, not the target's distribution). Row-parallel
projections keep their all-reduce (only the Linear's GEMM is replaced).
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ..m31_drafter_layout import route_his_layout as _route_his_layout
from ..ops.mxfp8_recover import (  # noqa: F401 (family helper, re-exported)
    BLOCK,
    E4M3_MAX,
    mxfp8_from_bf16,
)
from .kernels.mx_gemv import op as _gemv
from .m31_switches import m31_disabled
from .ops.comm.m3_mc_allreduce_norm import swizzle_sf


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


# DRAFT_QKV_RING2: the drafter QKV GEMV at 7 rows on ring depth 2 (exact: only the ring depth changes). Default ON;
# TRTLLM_M31_DISABLE_DRAFT_QKV_RING2=1 restores the default 7-row key. GU is unchanged.
def _draft_qkv_ring2_cfg(m: int, is_qkv: bool, key: tuple, disabled: bool) -> Optional[tuple]:
    """Select only the exact M7 drafter QKV key qualified by the MB.

    GU retains its default. Pinning the full old key fails closed if an upstream
    tactic, precision, PDL or epilogue changes; the MB then no longer applies.
    """
    qualified = (
        1280,
        6144,
        8,
        6,
        True,
        False,
        False,
        1,
        True,
        True,
        True,
        0,
        1,
        8,
        8,
        True,
        False,
        False,
        False,
    )
    if disabled or m != 7 or not is_qkv or key != qualified:
        return None
    return (key[2], 2, key[5], key[7], key[8], key[9])


_qkv_ring2_logged = False


def route_mx_linears(draft_model, rows: Iterable[int]) -> int:
    """Route each drafter projection whose BF16 weight is an exact MXFP8 dequantization to the MXFP8 GEMV for 2-D
    contiguous bf16 inputs of the given row counts (the decode passes'); compiles those kernels now (outside capture).
    Returns how many projections were routed."""
    rows = sorted({int(m) for m in rows if 0 < int(m) <= _gemv.MAX_TOKENS})
    n_routed = 0
    qkv_ids = {
        id(getattr(getattr(layer, "self_attn", None), "qkv_proj", None))
        for layer in getattr(getattr(draft_model, "model", None), "layers", []) or []
    }
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
        qkv_cfg = None
        if 7 in served and id(lin) in qkv_ids:
            resolved = _gemv._resolve(7, n, k, True, True, None, None, ar=None)
            if resolved is not None:
                qkv_cfg = _draft_qkv_ring2_cfg(
                    7, True, resolved[0], m31_disabled("DRAFT_QKV_RING2")
                )
                if qkv_cfg is not None:
                    candidate = _gemv._resolve(7, n, k, True, True, qkv_cfg, None, ar=None)
                    expected = resolved[0][:3] + (2,) + resolved[0][4:]
                    if candidate is None or candidate[0] != expected:
                        raise RuntimeError("DRAFT_QKV_RING2 is no longer a ring-only key")
        for m in served:
            # Prebuild the selected key before capture, even when ordinary JIT is frozen.
            _gemv.precompile(w8, sf, m, True, cfg=qkv_cfg if m == 7 else None)
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
            _qkv_cfg=qkv_cfg,
        ):
            global _qkv_ring2_logged
            if (
                bias is None
                and not lora_params
                and isinstance(input, torch.Tensor)
                and input.dtype == torch.bfloat16
                and input.dim() == 2
                and input.shape[0] in _served
                and input.is_contiguous()
            ):
                cfg = _qkv_cfg if input.shape[0] == 7 else None
                out = _gemv.m3_mx_gemv_bf16(input, _w8, _sf, cfg=cfg)
                if out is not None:
                    if cfg is not None and not _qkv_ring2_logged:
                        import sys

                        print(
                            "[M31 DRAFT_QKV_RING2] taken rows=7 split=8 ring=2",
                            file=sys.stderr,
                            flush=True,
                        )
                        _qkv_ring2_logged = True
                    return out
            return _o(input, bias, lora_params, layer_idx)

        lin._m31_mxdraft = (w8, sf, served)
        lin.apply_linear = apply_linear
        n_routed += 1
    return n_routed


# DRAFT_GU_SWIGLU_EPI (F5, exact): the drafter MLP's gate_up GEMV writes swigluoai(gate, up) from its epilogue (the
# interleaved gate_up weight, m3_mx_gemv_gu_swiglu) instead of the bf16 gate | up rows plus the separate Triton
# silu_and_mul launch; bit for bit the same h (tests: test_mxdraft_gu_swiglu.py). Kill switch
# TRTLLM_M31_DISABLE_DRAFT_GU_SWIGLU_EPI=1. Engagement: "[M3.1 drafter_mxfp8] DRAFT_GU_SWIGLU_EPI taken rows=<m>" once
# per row count and process, from the code doing the work.
DRAFT_GU_SWIGLU_EPI_PATH = "DRAFT_GU_SWIGLU_EPI"
_GU_SWIGLU = (7.0, 1.702, 1.0)  # (limit, alpha, beta) of the kernel's swigluoai
_gu_logged: set = set()


def route_gu_swiglu(draft_model, rows: Iterable[int]) -> int:
    """Fold each drafter MLP's SwiGLU into its gate_up GEMV epilogue where the MLP is the swigluoai GatedMLP the kernel
    implements; compiles those kernels now (outside capture). Returns how many MLPs were routed."""
    if m31_disabled("DRAFT_GU_SWIGLU_EPI"):
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


# TP8 dimensions with supported target tactics; QKV uses the matching TP8 key of the reference implementation.
# Drafter-only precision and layout: draft tokens change, the target's distribution does not.
HIS_MAX_ROWS = 8
HIS_SHAPES = {
    "qkv_proj": (1280, 6144),
    "o_proj": (6144, 1024),
    "gate_up_proj": (3072, 6144),
    "down_proj": (6144, 1536),
}
# DRAFT_GEMV_RING_RETUNE: QKV ring 6 -> 2 and o ring 12 -> 2 (bitwise-identical outputs);
# TRTLLM_M31_DISABLE_DRAFT_GEMV_RING_RETUNE=1 restores rings 6 and 12.
HIS_CFGS = (
    {(1280, 6144): (8, 6), (6144, 1024): (2, 12), (3072, 6144): (4, 12)}
    if m31_disabled("DRAFT_GEMV_RING_RETUNE")
    else {(1280, 6144): (8, 2), (6144, 1024): (2, 2), (3072, 6144): (4, 12)}
)
HIS_DOWN_CFGS = {(6144, 1536): (2, 3, 0)}


def route_his_layout(draft_model, rows: Iterable[int]) -> Tuple[int, int]:
    """Reuse family layout routing with TP8 providers and dimensions, including ROW forward dispatch."""
    logger.info(
        f"[M31_TP8_GVC8] RING_RETUNE={int(not m31_disabled('DRAFT_GEMV_RING_RETUNE'))} "
        f"qkv={HIS_CFGS[(1280, 6144)]} o={HIS_CFGS[(6144, 1024)]}"
    )
    return _route_his_layout(
        draft_model,
        rows,
        gemv=_gemv,
        recover=mxfp8_from_bf16,
        swizzle=swizzle_sf,
        configs=HIS_CFGS,
        down_configs=HIS_DOWN_CFGS,
    )
