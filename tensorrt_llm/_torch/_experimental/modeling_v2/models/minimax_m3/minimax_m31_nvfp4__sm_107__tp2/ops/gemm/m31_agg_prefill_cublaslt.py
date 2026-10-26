# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned cuBLASLt for agg prefill GEMMs of at least 8192 rows (not bitwise, pending agg quality gates.); on by default.
TRTLLM_M31_DISABLE_AGG_PREFILL_CUBLASLT=1 keeps the original WIDE dispatch.

Pins and first-use checks reuse the sm_107 tp2ctx implementation.
The scaled_mm reference certifies cuBLASLt's own numerics, not FlashInfer parity.
"""

from typing import Callable

import torch

from tensorrt_llm.logger import logger

from ...m31_switches import m31_disabled

ENABLED = not m31_disabled("AGG_PREFILL_CUBLASLT")
PINS = {
    (6144, 4096): (
        (8192, 12287, (66, 513, 36, 1, 0, 0, 1, 0, 6)),
        (12288, 16384, (66, 513, 36, 1, 0, 0, 1, 0, 9)),
    ),
    (12288, 6144): ((8192, 16384, (66, 513, 36, 1, 0, 0, 3, 0, 9)),),
    (6144, 6144): (
        (8192, 12287, (66, 513, 36, 1, 0, 0, 1, 0, 6)),
        (12288, 16384, (66, 513, 36, 1, 0, 0, 1, 0, 9)),
    ),
    (3072, 6144): ((8192, 16384, (66, 513, 36, 1, 0, 0, 1, 0, 6)),),
}
_STATE = {}


def _op() -> (
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]], torch.Tensor]
    | None
):
    """The built native op, looked up once; absence preserves the original dispatch."""
    if "op" not in _STATE:
        _STATE["op"] = getattr(torch.ops.trtllm, "minimax_m31_mxfp8_cublaslt_gemm", None)
    return _STATE["op"]


def self_check(weight: torch.Tensor, scale: torch.Tensor, band: tuple) -> str | None:
    """Reuse tp2ctx's deterministic, scaled_mm-exact check at both band edges."""
    n, k = weight.shape
    gen = torch.Generator(device=weight.device).manual_seed(n * 31 + k)
    op = _op()
    if op is None:
        return "native op unavailable"
    for rows in (band[0], band[1]):
        x = torch.randn(rows, k, device=weight.device, dtype=torch.float32, generator=gen)
        exponent = torch.randint(
            -12, 13, (rows, k // 32, 1), device=x.device, generator=gen
        ).float()
        act, act_sf = torch.ops.trtllm.mxfp8_quantize(
            (x.view(rows, k // 32, 32) * torch.exp2(exponent)).view(rows, k).to(torch.bfloat16),
            True,
        )
        del x, exponent
        try:
            pinned = op(act, act_sf, weight, scale, list(band[2]))
            again = op(act, act_sf, weight, scale, list(band[2]))
        except RuntimeError as exc:
            return f"pinned launch failed at M={rows}: {exc}"
        if not torch.equal(again.view(torch.int16), pinned.view(torch.int16)):
            return f"repeated pinned calls differ at M={rows}"
        reference = torch._scaled_mm(
            act.view(torch.float8_e4m3fn),
            weight.view(torch.float8_e4m3fn).t(),
            act_sf.reshape(-1).view(torch.float8_e8m0fnu),
            scale.reshape(-1).view(torch.float8_e8m0fnu),
            out_dtype=torch.bfloat16,
        )
        if not torch.equal(pinned.view(torch.int16), reference.view(torch.int16)):
            return f"pinned output differs from scaled_mm at M={rows}"
    return None


def try_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    alpha: torch.Tensor,
    prequant: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor | None:
    """Run an eligible eager band, or leave the original WIDE path unchanged."""
    if not ENABLED or x.dim() != 2 or x.dtype != torch.bfloat16:
        return None
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None
    rows = x.shape[0]
    n, k = weight.shape
    band = next((b for b in PINS.get((n, k), ()) if b[0] <= rows <= b[1]), None)
    if band is None:
        return None
    if torch.cuda.get_device_capability(weight.device) != (10, 7):
        return None
    alpha_key = ("alpha", alpha.device, alpha.data_ptr())
    if alpha_key not in _STATE:
        _STATE[alpha_key] = float(alpha) == 1.0
    if not _STATE[alpha_key]:
        return None
    key = (weight.device, n, k, band[0], band[1])
    if key not in _STATE:
        why = self_check(weight, scale, band)
        _STATE[key] = why is None
        if why is not None:
            logger.warning(f"AGG_PREFILL_CUBLASLT not taken N={n} K={k} band={band[:2]}: {why}")
    if not _STATE[key]:
        return None
    act, act_sf = (
        prequant if prequant is not None else torch.ops.trtllm.mxfp8_quantize(x.contiguous(), True)
    )
    output = _op()(act, act_sf, weight, scale, list(band[2]))
    logged = ("engaged", key)
    if logged not in _STATE:
        _STATE[logged] = True
        logger.info(f"AGG_PREFILL_CUBLASLT engaged N={n} K={k} band={band[:2]} M={rows}")
    return output
