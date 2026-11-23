# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_DRAFTER_MXFP8: the DSpark drafter's four decoder-layer projections as MXFP8 GEMMs (vLLM's DEP drafter layout).

The drafter checkpoint ships qkv / o / gate_up / down as MXFP8 (E4M3 codes, one UE8M0 scale per 32 columns, dynamic
activation) and the stock drafter dequantizes them to BF16 at load, so every decode pass streams twice the weight bytes.
vLLM's DEP drafter runs them as MXFP8 GEMMs; this does the same on this target's kernels:

* qkv_proj / o_proj / gate_up_proj: ``trtllm::mxfp8_quantize`` of the BF16 rows + the family cuBLASLt MXFP8 GEMM at
  ``gemm_provider.CUBLASLT_ALGO66``;
* down_proj: the MLP's SwiGLU-OAI by ``swiglu_mxfp8`` (``silu_and_mul`` bit for bit, plus the MXFP8 form of its output)
  feeding the same cuBLASLt GEMM.

The weights' MXFP8 form is recovered from the BF16 weights (``ops.mxfp8_recover``: exact, checked to dequantize back to
the same BF16 bits, else the projection stays BF16); the BF16 weights stay for the rows this does not take and for the
context K/V projection, which keeps its fused BF16 weight. fc, the Markov / confidence heads and the LM heads are BF16
in the checkpoint and stay so. Rows: MIN_ROWS..MAX_ROWS (the draft block rows of a decode step, 7 per request); other
row counts, LoRA, bias, under torch.compile, or a GEMM cuBLASLt declines run the stock BF16 path.

The output is not the BF16 GEMM's (the MXFP8 activation: rel-L2 about 2.7e-2 against it): drafter-only precision,
gated on the natural acceptance length. ``TRTLLM_M31_DISABLE_DEP_DRAFTER_MXFP8=1`` installs nothing (read at import).
Engagement: the install line and one "[M3.1 DEP] MXDRAFT_DEP taken" line per projection kind and row count, from the
code doing the work."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.utils import is_torch_compiling
from tensorrt_llm.logger import logger

from .. import m31_mxfp8_cublaslt as _cublaslt
from ..ops.mxfp8_recover import mxfp8_from_bf16, swizzle_sf
from .gemm_provider import CUBLASLT_ALGO66
from .m31_switches import m31_disabled
from .merge_mxfp8 import scale_bytes

DEP_MXDRAFT = not m31_disabled("DEP_DRAFTER_MXFP8")

TAG = "[M3.1 DEP] MXDRAFT_DEP"
#: The drafter block rows taken (7 rows a request), inclusive.
MIN_ROWS, MAX_ROWS = 33, 128
#: The attribute an installed projection / MLP carries.
ATTR = "_m31_dep_mxdraft"

_LOGGED: set = set()


def _log_once(kind: str, rows: int, how: str) -> None:
    key = (kind, rows)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(f"{TAG} taken: {kind} rows={rows} via {how}")


def served(x: torch.Tensor) -> bool:
    """Whether a projection input is one this takes: contiguous 16-byte-aligned 2-D BF16 CUDA rows in the band."""
    return (
        isinstance(x, torch.Tensor)
        and not is_torch_compiling()
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and MIN_ROWS <= x.shape[0] <= MAX_ROWS
    )


def gemm(
    act_e4m3: torch.Tensor, act_sf: torch.Tensor, w8: torch.Tensor, w_sf: torch.Tensor
) -> Optional[torch.Tensor]:
    """[m, k] x [n, k]^T in BF16 on the family cuBLASLt MXFP8 GEMM at algo 66, or None (declined / not built)."""
    return _cublaslt.run(act_e4m3, act_sf, w8, w_sf, CUBLASLT_ALGO66, TAG)


def quantize_gemm(x: torch.Tensor, w8: torch.Tensor, w_sf: torch.Tensor) -> Optional[torch.Tensor]:
    """The MXFP8 form of BF16 rows ``x`` (``trtllm::mxfp8_quantize``, swizzled scales) times the MXFP8 weight."""
    act, act_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    return gemm(act, act_sf, w8, w_sf)


def swiglu_kernel():
    """``swiglu_mxfp8``, imported on first use."""
    from ..kernels.megamoe.swiglu_mxfp8 import swiglu_mxfp8

    return swiglu_mxfp8


def swiglu_constants(mlp) -> Tuple[float, float, Optional[float]]:
    """(alpha, beta, limit) as ``torch.ops.trtllm.silu_and_mul`` applies the GatedMLP's SwiGLU settings."""
    alpha = 1.0 if mlp.swiglu_alpha is None else float(mlp.swiglu_alpha)
    beta = 0.0 if mlp.swiglu_beta is None else float(mlp.swiglu_beta)
    limit = mlp.swiglu_limit
    return alpha, beta, (float(limit) if limit is not None and limit > 0.0 else None)


def weight_form(linear) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(E4M3 codes, swizzled UE8M0 scales) of a bias-free BF16 projection whose weight is exactly an MXFP8
    dequantization with a K / 32 that is a multiple of 4, else None."""
    weight = getattr(linear, "weight", None)
    if weight is None or getattr(linear, "bias", None) is not None or weight.dim() != 2:
        return None
    if weight.shape[1] % 128 != 0:
        return None
    rec = mxfp8_from_bf16(weight.data)
    if rec is None:
        return None
    return rec[0], swizzle_sf(rec[1])


def route_linear(linear, kind: str) -> bool:
    """Wrap ``linear.apply_linear`` so the rows ``served`` admits take the MXFP8 GEMM; returns whether it is routed."""
    if getattr(linear, ATTR, None) is not None:
        return True
    form = weight_form(linear)
    if form is None:
        return False
    w8, w_sf = form
    original = linear.apply_linear

    def apply_linear(
        input, bias, lora_params=None, layer_idx=None, *, _o=original, _w8=w8, _sf=w_sf
    ):
        if bias is None and not lora_params and served(input):
            out = quantize_gemm(input, _w8, _sf)
            if out is not None:
                _log_once(kind, int(input.shape[0]), "mxfp8_quantize + cuBLASLt algo 66")
                return out
        return _o(input, bias, lora_params, layer_idx)

    linear.__dict__[ATTR] = (w8, w_sf)
    linear.apply_linear = apply_linear
    return True


def mlp_decline_reason(mlp) -> Optional[str]:
    """Why a drafter MLP keeps the stock SwiGLU + BF16 down_proj, or None if it is the fused-gate_up SwiGLU GatedMLP
    whose down_proj is routed."""
    if getattr(mlp, "split_gate_up", False) or getattr(mlp, "activation", None) is not F.silu:
        return "not one fused gate_up_proj with a SwiGLU"
    down = getattr(mlp, "down_proj", None)
    if down is None or getattr(down, ATTR, None) is None:
        return "down_proj is not routed"
    if getattr(down, "tp_size", 1) > 1:
        return "down_proj is tensor-parallel (the chain would skip its all-reduce)"
    if down.weight.shape[1] % 128 != 0:
        return "the SwiGLU width is not a multiple of 128"
    return None


def route_mlp(mlp) -> bool:
    """Wrap ``mlp.forward`` so the rows ``served`` admits run gate_up_proj (routed or not), ``swiglu_mxfp8`` with its
    MXFP8 form, and the MXFP8 down GEMM; every other call is the stock forward. Returns whether it is routed."""
    if mlp.__dict__.get(ATTR) is not None:
        return True
    if mlp_decline_reason(mlp) is not None:
        return False
    w8, w_sf = mlp.down_proj.__dict__[ATTR]
    stock = mlp.forward

    def forward(
        x,
        all_rank_num_tokens=None,
        final_all_reduce_params=None,
        lora_params=None,
        *args,
        _stock=stock,
        _mlp=mlp,
        _w8=w8,
        _sf=w_sf,
        **kwargs,
    ):
        if bool(lora_params) or not served(x):
            return _stock(
                x, all_rank_num_tokens, final_all_reduce_params, lora_params, *args, **kwargs
            )
        h1 = _mlp.gate_up_proj(x)
        if served(h1) and h1.shape[1] % 2 == 0:
            rows, width = h1.shape[0], h1.shape[1] // 2
            h2 = torch.empty((rows, width), dtype=h1.dtype, device=h1.device)
            q = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=h1.device)
            sf = torch.zeros(scale_bytes(rows, width), dtype=torch.uint8, device=h1.device)
            alpha, beta, limit = swiglu_constants(_mlp)
            swiglu_kernel()(
                h1, h2, q, sf, alpha=alpha, beta=beta, limit=limit, enable_pdl=get_env_enable_pdl()
            )
            out = gemm(q, sf, _w8, _sf)
            if out is not None:
                _log_once("down_proj (swiglu_mxfp8)", rows, "swiglu_mxfp8 + cuBLASLt algo 66")
                return out
            return _mlp.down_proj(
                h2, all_reduce_params=final_all_reduce_params, layer_idx=_mlp.layer_idx
            )
        h2 = _mlp._apply_activation(h1)
        return _mlp.down_proj(
            h2, all_reduce_params=final_all_reduce_params, layer_idx=_mlp.layer_idx
        )

    mlp.__dict__[ATTR] = (w8, w_sf)
    mlp.forward = forward
    return True


def _probe(linear, kind: str) -> bool:
    """Whether cuBLASLt runs this projection's GEMM (an eager call at MIN_ROWS, outside capture); CPU weights pass."""
    w8, w_sf = linear.__dict__[ATTR]
    if not w8.is_cuda or torch.cuda.is_current_stream_capturing():
        return True
    x = torch.zeros((MIN_ROWS, w8.shape[1]), dtype=torch.bfloat16, device=w8.device)
    ok = quantize_gemm(x, w8, w_sf) is not None
    if not ok:
        logger.warning(
            f"{TAG}: {kind} [{w8.shape[0]}, {w8.shape[1]}] declined by cuBLASLt; it stays BF16"
        )
    return ok


def _unroute(linear) -> None:
    linear.__dict__.pop(ATTR, None)
    linear.__dict__.pop("apply_linear", None)


def install(draft_model) -> Tuple[int, int]:
    """Route each drafter layer's qkv_proj / o_proj / gate_up_proj / down_proj (and the MLP's down chain) whose weight
    is an exact MXFP8 dequantization; compiles the SwiGLU kernel and probes each GEMM now (outside capture). Returns
    (projections routed, MLP down chains routed); (0, 0) with TRTLLM_M31_DISABLE_DEP_DRAFTER_MXFP8=1 or no drafter."""
    if not DEP_MXDRAFT or draft_model is None:
        return 0, 0
    projections = mlps = 0
    for layer in getattr(getattr(draft_model, "model", None), "layers", []) or []:
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        for kind, linear in (
            ("qkv_proj", getattr(attn, "qkv_proj", None)),
            ("o_proj", getattr(attn, "o_proj", None)),
            ("gate_up_proj", getattr(mlp, "gate_up_proj", None)),
            ("down_proj", getattr(mlp, "down_proj", None)),
        ):
            if linear is None or not route_linear(linear, kind):
                continue
            if not _probe(linear, kind):
                _unroute(linear)
                continue
            projections += 1
        if mlp is not None and route_mlp(mlp):
            mlps += 1
            down = mlp.down_proj
            if down.weight.is_cuda and not torch.cuda.is_current_stream_capturing():
                width = down.weight.shape[1]
                g = torch.zeros((1, 2 * width), dtype=torch.bfloat16, device=down.weight.device)
                h = torch.empty((1, width), dtype=torch.bfloat16, device=g.device)
                q = torch.empty((1, width), dtype=torch.float8_e4m3fn, device=g.device)
                sf = torch.zeros(scale_bytes(1, width), dtype=torch.uint8, device=g.device)
                alpha, beta, limit = swiglu_constants(mlp)
                swiglu_kernel()(
                    g,
                    h,
                    q,
                    sf,
                    alpha=alpha,
                    beta=beta,
                    limit=limit,
                    enable_pdl=get_env_enable_pdl(),
                )
    return projections, mlps


def engagement(counts: Tuple[int, int]) -> str:
    """The install line's state."""
    if not DEP_MXDRAFT:
        return "off (TRTLLM_M31_DISABLE_DEP_DRAFTER_MXFP8=1); drafter projections BF16"
    projections, mlps = counts
    if not projections:
        return "not installed (no drafter, or no projection is an exact MXFP8 dequantization)"
    return (
        f"MXFP8 on {projections} projections, swiglu_mxfp8 down chain on {mlps} MLPs "
        f"(rows {MIN_ROWS}..{MAX_ROWS}; cuBLASLt algo {list(CUBLASLT_ALGO66)})"
    )
