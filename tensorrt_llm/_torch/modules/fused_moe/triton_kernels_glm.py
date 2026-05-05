# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5 fused-MoE adapter glue for the TRITON_GLM5 backend.

Per nvbug 6108841 / `MOE_FUSION_DESIGN.md` §7.11.4 and §7.8/§7.9 the GLM-5
fused-MoE backend in TRT-LLM uses TWO fused kernels:

  * Kernel B — `fused_expert_up_gate_silu_fp8`
        (Phase 2.6 CuTe DSL `kernel_b_cute_dsl_v26`)

  * Kernel C — `fused_expert_down_combine_fp8`
        (Phase 2.7 CuTe DSL `kernel_c_cute_dsl_v26`)

Kernel A (RMSNorm + router GEMM) is NOT fused into the MoE backend per the
§7.11.4 decision — it stays in `DeepseekV3Gate.forward` upstream of the MoE
call and the MoE backend receives router logits as today.

This module is a thin adapter:

  1. Import the kernel B / kernel C entry points from
     `nvbugs/6108841/kernels/` (the Phase 2 dev location).  They are vanilla
     Python, but pull in CuTe DSL via `cutlass.cute` and a vendored
     CUTLASS example (`cute_examples/contiguous_grouped_gemm.py`).
  2. Provide a small `glm5_fused_moe_apply()` that runs the routing method,
     calls Kernel B + Kernel C in sequence, and returns the MoE output.
  3. Provide `is_glm5_fused_shape()` for backend dispatch.

The path injection at import time is necessary because the kernels are
file-based and import other peer modules (`kernel_b_fused_expert_up_gate_silu_fp8`
for the route+quant Triton kernel, plus `cute_examples/...`).
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch

# GLM-5 per-device constants after TP=8.  We only enable the fused path when
# all of these match — for everything else we fall back.
GLM5_NUM_EXPERTS = 256
GLM5_TOP_K = 8
GLM5_HIDDEN = 6144
GLM5_INTERMEDIATE_PER_DEVICE = 256

# Phase-2 kernel directory (dev-time scratch).  See task brief: keeping
# kernels here is option (a) — clean integration without copying the
# vendored CUTLASS examples is option (b)'s deferred work.
_GLM_KERNEL_DIR = (
    "/scratch/fsw/portfolios/coreai/projects/coreai_mlperf_inference/"
    "users/yijingl/dev/cluster-workspace/nvbugs/6108841/kernels"
)
_GLM_KERNEL_DIR_OVERRIDE_ENV = "TRTLLM_GLM5_KERNEL_DIR"


def _get_kernel_dir() -> str:
    return os.environ.get(_GLM_KERNEL_DIR_OVERRIDE_ENV, _GLM_KERNEL_DIR)


_KERNELS_LOADED = False
_kernel_b_fn = None
_kernel_c_fn = None
_load_error: Optional[str] = None


def _try_load_kernels():
    """Best-effort import of Kernel B and Kernel C.

    Records the failure in `_load_error` rather than raising, so callers can
    decide between hard-fail (when the GLM5 backend is explicitly selected)
    and soft-fall-through (when probing).
    """
    global _KERNELS_LOADED, _kernel_b_fn, _kernel_c_fn, _load_error
    if _KERNELS_LOADED:
        return _kernel_b_fn is not None and _kernel_c_fn is not None
    _KERNELS_LOADED = True

    kdir = _get_kernel_dir()
    if not os.path.isdir(kdir):
        _load_error = f"GLM5 kernel dir not found: {kdir}"
        return False

    if kdir not in sys.path:
        sys.path.insert(0, kdir)
    cute_examples_dir = os.path.join(kdir, "cute_examples")
    if os.path.isdir(cute_examples_dir) and cute_examples_dir not in sys.path:
        sys.path.insert(0, cute_examples_dir)

    try:
        # Kernel B: fused_expert_up_gate_silu_fp8 (CuTe DSL v26)
        from kernel_b_cute_dsl_v26 import fused_expert_up_gate_silu_fp8 as _kb  # type: ignore

        # Kernel C: fused_expert_down_combine_fp8 (CuTe DSL v26)
        from kernel_c_cute_dsl_v26 import fused_expert_down_combine_fp8 as _kc  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _load_error = f"failed to import GLM5 kernels: {exc!r}"
        return False

    _kernel_b_fn = _kb
    _kernel_c_fn = _kc
    return True


def is_glm5_fused_shape(
    num_experts: int, hidden_size: int, intermediate_size_per_partition: int, top_k: int
) -> bool:
    """Return True iff the (num_experts, hidden_size, intermediate, top_k)
    tuple matches the GLM-5 per-device shape after TP=8."""
    return (
        num_experts == GLM5_NUM_EXPERTS
        and hidden_size == GLM5_HIDDEN
        and intermediate_size_per_partition == GLM5_INTERMEDIATE_PER_DEVICE
        and top_k == GLM5_TOP_K
    )


def kernels_available() -> Tuple[bool, Optional[str]]:
    """Try to import the kernels; return (ok, error_message_if_any)."""
    ok = _try_load_kernels()
    return ok, _load_error


def glm5_fused_moe_apply(
    x: torch.Tensor,  # [M, H] bf16  (RMSNorm'd activation)
    router_logits: torch.Tensor,  # [M, E] fp32  (BF16 also accepted)
    e_score_correction_bias: torch.Tensor,  # [E] fp32
    gate_weight_per_expert: torch.Tensor,  # [E, N, H] fp8e4m3
    up_weight_per_expert: torch.Tensor,  # [E, N, H] fp8e4m3
    gate_scale_per_expert: torch.Tensor,  # [E, N//128, H//128] fp32
    up_scale_per_expert: torch.Tensor,  # [E, N//128, H//128] fp32
    down_weight_per_expert: torch.Tensor,  # [E, H, N] fp8e4m3
    down_scale_per_expert: torch.Tensor,  # [E, H//128, N//128] fp32
    residual: Optional[torch.Tensor] = None,  # [M, H] bf16 or None
    top_k: int = GLM5_TOP_K,
    routed_scaling_factor: float = 2.5,
) -> torch.Tensor:
    """Run Kernel B (up+gate+silu) then Kernel C (down+combine[+residual]).

    Allreduce is NOT folded — the caller (TritonFusedMoE) does it via
    `reducescatter_or_allreduce` exactly as today.

    The standard TRT-LLM MoE contract returns only the MoE output (the
    surrounding decoder layer adds the residual).  Kernel C's `x_in`
    parameter always adds a residual; pass `residual=None` (default) to
    use a zero tensor and match the TRT-LLM contract.

    Returns: out [M, H] bfloat16 (NOT all-reduced; no residual when
    `residual=None`).
    """
    if not _try_load_kernels():
        raise RuntimeError(
            f"GLM-5 fused kernels unavailable: {_load_error}. "
            f"Set {_GLM_KERNEL_DIR_OVERRIDE_ENV} to override the kernel dir."
        )

    # --- Kernel B ---
    # `scores_in` is the *pre-sigmoid* router logits (fp32).  The kernel does
    # sigmoid + bias + topk + renorm internally.
    if router_logits.dtype != torch.float32:
        scores_in = router_logits.to(torch.float32)
    else:
        scores_in = router_logits
    bias_in = e_score_correction_bias.to(torch.float32)

    gated_hidden, probs, indices = _kernel_b_fn(  # type: ignore[misc]
        x,
        scores_in,
        bias_in,
        gate_weight_per_expert,
        up_weight_per_expert,
        gate_scale_per_expert,
        up_scale_per_expert,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
    )

    # --- Kernel C ---
    # Kernel C unconditionally adds `x_in` to the output.  Use a zero
    # tensor when the caller didn't pass an explicit residual so the MoE
    # output is returned standalone (matching the TRT-LLM contract).
    if residual is None:
        residual = torch.zeros_like(x)
    out = _kernel_c_fn(  # type: ignore[misc]
        gated_hidden,
        down_weight_per_expert,
        down_scale_per_expert,
        indices,
        probs,
        residual,
    )

    return out


__all__ = [
    "GLM5_NUM_EXPERTS",
    "GLM5_TOP_K",
    "GLM5_HIDDEN",
    "GLM5_INTERMEDIATE_PER_DEVICE",
    "is_glm5_fused_shape",
    "kernels_available",
    "glm5_fused_moe_apply",
]
