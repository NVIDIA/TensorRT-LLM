# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_MLP_MXFP8_14_ROWS: the DSpark drafter MLP's gate_up + SwiGLU at 14 rows (draft length 7, batch 2) on the MX
GEMV.

By default, at 14 rows, the drafter's BF16-input MX GEMV (DRAFTER_MXFP8) declines the gate_up ([14, 6144] x
[12288, 6144]^T: its in-kernel quantize would need more than 8 passes per CTA) and DRAFT_LOWM_GEMM has no gate_up entry,
so the gate_up runs cuBLAS and the swigluoai activation a separate Triton launch. This path quantizes the activation
once with FlashInfer's CuTe DSL MXFP8 quantizer and runs this target's MX GEMV on the prequantized activation against
an MXFP8 copy of the gate_up whose rows are interleaved per 128-row tile (``op.gu_interleave``), with the SwiGLU in the
GEMV epilogue (its owner loop strided so 32 pairs x 16 token columns fit the 128 epilogue threads at split 2): the
epilogue writes h = swigluoai(gate, up) [14, 6144] directly and the activation launch is skipped.

- Numerics: the MXFP8 activation (as DRAFTER_MXFP8 at 7 rows; the BF16-input GEMV and quantize + prequantized GEMV are
  bitwise equal) instead of cuBLAS BF16: drafter-only precision. The epilogue itself is bit for bit the GEMV's bf16
  gate / up rows followed by the Triton swigluoai (same rounding, same instructions).
- Rows: ``ROWS`` = (14,) only. Every other call, and any input it does not take, runs the Linear and the activation
  unchanged. The down projection keeps its route.
- Memory: one interleaved MXFP8 copy of each drafter gate_up (12288 x 6144 codes + scales per rank, about 78.0 MB per
  layer, 0.39 GB for 5 layers), allocated at prebuild (before the KV cache is sized).
- Set-up: per rank, every drafter MLP is checked (swigluoai with limit 7 / alpha 1.702 / beta 1, fused gate_up, no
  bias, an exact MXFP8 dequantization), the kernels are compiled and run once outside capture, the first MLP is
  self-checked against its own BF16 GEMM + activation (relative RMS below ``SELF_CHECK_REL_RMS``), and the ranks agree
  (all or none) before any MLP is routed.

Engagement line: ``[M3 dispatch] DRAFT_MLP_MXFP8_14_ROWS taken: rank R: ... under CUDA-graph capture``, once per process
and row count, at the first routed call recorded into a CUDA graph (the first eager routed call logs an ``eager``
line). It is not evidence that a graph replayed.

Opt-in: ``TRTLLM_M31_ENABLE_DRAFT_MLP_MXFP8_14_ROWS=1``, read at import.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm.logger import logger

from . import dspark_draft_head as _dhead8
from .drafter_mxfp8 import mxfp8_from_bf16
from .kernels import m3_report
from .kernels.mx_gemv import op as _gemv
from .m31_switches import m31_enabled
from .ops.comm.m3_mc_allreduce_norm import swizzle_sf

DMLP_MX14 = m31_enabled("DRAFT_MLP_MXFP8_14_ROWS")
#: Draft rows taken: B x K for the draft-length-7 batch-2 block.
ROWS = (14,)  # capacity: fixed: the draft-length-7 batch-2 block
#: The swigluoai the epilogue implements: (limit, alpha, beta) of the drafter MLP.
GU_SWIGLU = (7.0, 1.702, 1.0)
#: Set-up self-check bound: relative RMS of the routed MLP's h against its own BF16 GEMM + activation (MXFP8
#: activation rounding is about 0.03 on random data; a wrong epilogue mapping is about 1.4).
SELF_CHECK_REL_RMS = 0.1


def unfit(mlp) -> Optional[str]:
    """Why a drafter MLP cannot take the epilogue (None: it can). Config and shapes only, the same on every rank."""
    gu = getattr(mlp, "gate_up_proj", None)
    if gu is None or getattr(mlp, "split_gate_up", False):
        return "no fused gate_up projection"
    if getattr(mlp, "activation", None) is not F.silu:
        return f"activation {getattr(mlp, 'activation', None)!r} (SwiGLU needed)"
    params = (
        getattr(mlp, "swiglu_limit", None),
        getattr(mlp, "swiglu_alpha", None),
        getattr(mlp, "swiglu_beta", None),
    )
    if params != GU_SWIGLU:
        return f"swiglu (limit, alpha, beta) {params} (the epilogue implements {GU_SWIGLU})"
    if getattr(gu, "bias", None) is not None:
        return "the gate_up has a bias"
    w = getattr(gu, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16:
        return "no 2-D BF16 gate_up weight"
    if not w.is_cuda or not w.is_contiguous():
        return "the gate_up weight is not a contiguous CUDA tensor"
    if int(w.shape[0]) % 128 or int(w.shape[1]) % 128:
        return f"gate_up {tuple(w.shape)} (whole 128-row interleaved tiles and 128-column k-tiles needed)"
    if getattr(gu, "_m31_dmlp", None) is not None:
        return "already routed"
    return None


class _Route:
    """One drafter MLP's interleaved MXFP8 gate_up copy and the rows it serves."""

    def __init__(
        self, layer: int, mlp, w8: torch.Tensor, sf: torch.Tensor, served: Tuple[int, ...]
    ):
        self.layer = layer
        self.mlp = mlp
        self.w8 = w8
        self.sf = sf
        self.served = served
        self.n = int(w8.shape[0])
        self.k = int(w8.shape[1])

    @property
    def bytes(self) -> int:
        return self.w8.numel() * self.w8.element_size() + self.sf.numel()


def _layers(draft_model):
    for i, layer in enumerate(getattr(getattr(draft_model, "model", None), "layers", []) or []):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            yield i, mlp


def run(route: _Route, x: torch.Tensor, quantize: Callable) -> Optional[torch.Tensor]:
    """h = swigluoai(x @ gate_up^T) [rows, F] in two launches (quantize, GEMV + epilogue), or None (not served)."""
    m = int(x.shape[0])
    xq, xsf = quantize(x)
    return _gemv.m3_mx_gemv_gu_swiglu(xq.view(m, route.k), xsf, route.w8, route.sf)


def prepare(
    draft_model, rows=ROWS, quantizer: str = "cute-dsl"
) -> Tuple[List[_Route], Optional[str]]:
    """Build every drafter MLP's route (copy, compiles, one run per served row count, the first MLP's self-check),
    without installing it. Returns (routes, why not): ``why`` None means every MLP can be routed."""
    rows = tuple(sorted({int(m) for m in rows if 0 < int(m) <= _gemv.MAX_TOKENS}))
    quantize = _dhead8._quantizer(quantizer)
    routes: List[_Route] = []
    for i, mlp in _layers(draft_model):
        why = unfit(mlp)
        if why is not None:
            return [], f"layer {i}: {why}"
        weight = mlp.gate_up_proj.weight.data
        rec = mxfp8_from_bf16(_gemv.gu_interleave(weight))
        if rec is None:
            return [], f"layer {i}: the gate_up weight is not an exact MXFP8 dequantization"
        w8, sf = rec[0], swizzle_sf(rec[1])
        served = tuple(
            m for m in rows if _gemv.precompile(w8, sf, m, False, gu_swiglu=True) is not None
        )
        if not served:
            return [], f"layer {i}: the GEMV declines the SwiGLU epilogue at rows {rows}"
        route = _Route(i, mlp, w8, sf, served)
        # A private generator: the set-up must not move the process's default RNG streams.
        gen = torch.Generator(device=weight.device).manual_seed(20261101 + i)
        for m in served:
            x = torch.randn(m, route.k, device=weight.device, generator=gen).to(torch.bfloat16)
            h = run(route, x, quantize)
            if h is None or tuple(h.shape) != (m, route.n // 2) or h.dtype != torch.bfloat16:
                return [], f"layer {i}: no h of [{m}, {route.n // 2}] BF16 at {m} rows"
            if not routes:
                ref = mlp._apply_activation(F.linear(x, weight)).float()
                rel = ((h.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
                if not rel < SELF_CHECK_REL_RMS:
                    return [], (
                        f"layer {i}: self-check relative RMS {rel:.3g} at {m} rows (bound {SELF_CHECK_REL_RMS})"
                    )
                route.self_check = rel
        routes.append(route)
    if not routes:
        return [], "no drafter MLPs"
    torch.cuda.synchronize()
    return routes, None


def install(routes: List[_Route], quantizer: str = "cute-dsl", rank=None) -> int:
    """Route each prepared MLP: its gate_up's ``apply_linear`` takes the served rows, and its activation passes the
    epilogue's h through (its width is half the gate_up's). Returns how many MLPs were routed."""
    quantize = _dhead8._quantizer(quantizer)
    logged = set()
    for route in routes:
        mlp, gu = route.mlp, route.mlp.gate_up_proj
        original = gu.apply_linear
        original_act = mlp._apply_activation

        def apply_linear(
            input, bias, lora_params=None, layer_idx=None, *, _o=original, _r=route, _q=quantize
        ):
            if (
                bias is None
                and not lora_params
                and isinstance(input, torch.Tensor)
                and input.dtype == torch.bfloat16
                and input.dim() == 2
                and int(input.shape[0]) in _r.served
                and int(input.shape[1]) == _r.k
                and input.is_contiguous()
            ):
                h = run(_r, input, _q)
                if h is not None:
                    m = int(input.shape[0])
                    captured = torch.cuda.is_current_stream_capturing()
                    if (m, captured) not in logged:
                        logged.add((m, captured))
                        what = (
                            f"drafter gate_up + SwiGLU at {m} rows (MXFP8 quantize + MX GEMV [{m}, {_r.k}] x "
                            f"[{_r.n}, {_r.k}]^T interleaved, SwiGLU epilogue -> h [{m}, {_r.n // 2}])"
                        )
                        logger.info(
                            f"[M3 dispatch] DRAFT_MLP_MXFP8_14_ROWS taken: rank {rank}: {what} under CUDA-graph capture"
                            if captured
                            else f"[M3 sm107 tp2] DRAFT_MLP_MXFP8_14_ROWS eager on rank {rank}: {what}"
                        )
                    m3_report.work("DRAFT_MLP_MXFP8_14_ROWS", f"layer {_r.layer}, {m} rows")
                    return h
            return _o(input, bias, lora_params, layer_idx)

        def _apply_activation(x, *args, _oa=original_act, _half=route.n // 2, **kwargs):
            # The epilogue already applied the SwiGLU: its h is half the gate_up's width.
            if isinstance(x, torch.Tensor) and x.dim() >= 1 and int(x.shape[-1]) == _half:
                return x
            return _oa(x, *args, **kwargs)

        gu._m31_dmlp = route
        gu.apply_linear = apply_linear
        mlp._apply_activation = _apply_activation
    return len(routes)


def _raise_if_cuda_fault(agree: Optional[Callable[[bool], bool]] = None) -> None:
    """A set-up failure that left the CUDA context faulted is raised (the rank cannot serve), not declined. Before it
    raises, this rank votes no in the set-up agreement (``agree``, a host-side collective) so the other ranks leave it
    and decline instead of waiting for a rank that will not reach it."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except RuntimeError as fault:
        if agree is not None:
            try:
                agree(False)
            except Exception:  # noqa: BLE001 - the CUDA fault below is the error to report
                pass
        raise RuntimeError("DRAFT_MLP_MXFP8_14_ROWS set-up left a CUDA fault") from fault


def prebuild(draft_model, k: int, rank=None, agree: Optional[Callable[[bool], bool]] = None) -> str:
    """The worker's set-up of DRAFT_MLP_MXFP8_14_ROWS (every rank or none); returns its set-up line. Off by its switch:
    no collective (the environment is the same on every rank). On: every rank prepares its routes (or names why not)
    and the ranks agree (``agree``, the worker's all-ranks check) before any rank installs them."""
    if not DMLP_MX14:
        why = "default off: opt in with TRTLLM_M31_ENABLE_DRAFT_MLP_MXFP8_14_ROWS=1"
        return f"M3.1 DSpark DRAFT_MLP_MXFP8_14_ROWS: off on rank {rank} ({why}); the default gate_up routing"
    start = time.perf_counter()
    rows = tuple(r for r in ROWS if k > 0 and r % k == 0)
    # Config-only declines (the draft length, no drafter MLP layers) are the same on every rank: no collective.
    if not rows:
        return (
            f"M3.1 DSpark DRAFT_MLP_MXFP8_14_ROWS: off on rank {rank} (draft length {k}: no row count of {ROWS}); "
            "the default gate_up routing"
        )
    if not any(True for _ in _layers(draft_model)):
        return f"M3.1 DSpark DRAFT_MLP_MXFP8_14_ROWS: off on rank {rank} (no drafter MLPs); the default gate_up routing"
    try:
        routes, why = prepare(draft_model, rows)
    except Exception as e:  # noqa: BLE001 - any set-up failure (a compile, memory) declines on every rank
        _raise_if_cuda_fault(agree)
        routes, why = [], f"set-up failed: {type(e).__name__}: {e}"
    ok = why is None
    if agree is not None:
        ok = agree(ok)
    if not ok:
        return (
            f"M3.1 DSpark DRAFT_MLP_MXFP8_14_ROWS: off on rank {rank} ({why or 'off on another rank'}); the default "
            "gate_up routing"
        )
    n = install(routes, rank=rank)
    total = sum(r.bytes for r in routes)
    return (
        f"M3.1 DSpark DRAFT_MLP_MXFP8_14_ROWS: drafter gate_up + SwiGLU epilogue set up on rank {rank} ({n} MLPs, rows "
        f"{routes[0].served}, gate_up [{routes[0].n}, {routes[0].k}] interleaved MXFP8, quantize cute-dsl, copies "
        f"{total / 1e6:.1f} MB, self-check rel RMS {getattr(routes[0], 'self_check', float('nan')):.3g}, set-up "
        f"{time.perf_counter() - start:.2f} s)"
    )


__all__ = ["DMLP_MX14", "GU_SWIGLU", "ROWS", "install", "prebuild", "prepare", "run", "unfit"]
