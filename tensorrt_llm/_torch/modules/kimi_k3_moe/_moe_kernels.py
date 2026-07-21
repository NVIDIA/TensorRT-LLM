# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel dispatch for the in-tree Kimi K3 sparse MoE block.

The K3 MoE block has two mutually exclusive kernel paths:

1. **Python fallback** — MXFP4 group-32 routed expert weights are
   dequantized on the fly, then fed through per-expert
   ``gate_up_proj + activation + down_proj`` linears. Byte-exact HF
   parity under random weights.

2. **Private-SiTU fused cubin** — routed compute goes through
   ``flashinfer.fused_moe.trtllm_mxint4_block_scale_moe(is_private=True,
   activation_type=Situ)`` on a locally-allocated MXINT4 packed weight
   bank. The private SiTU cubins live in
   ``exisiting_optimization_work/trtllmgen_MOE/local_cubins/...``; the
   caller is expected to set ``FLASHINFER_PRIVATE_CUBIN_DIR`` BEFORE
   ``import flashinfer`` — flashinfer's ``jit_env`` module snapshots
   that env var at import time and never re-reads it.

This dispatch module handles the fused-cubin path. The Python fallback
lives inside :mod:`kimi_k3_moe_block` because it is intertwined with
the module state (MXFP4 expert bank + activation).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

KIMI_K3_MOE_PRIVATE_CUBIN_ENV = "FLASHINFER_PRIVATE_CUBIN_DIR"
"""Env variable pointing at the private SiTU cubin pool root.

Must be set before ``import flashinfer`` for the private cubin selection
to take effect. Callers can verify the setting by inspecting
``flashinfer.jit.env.FLASHINFER_PRIVATE_CUBIN_DIR`` after import.
"""

KIMI_K3_MOE_OPTIMIZED_KERNEL_ENV = "KIMI_K3_MOE_OPTIMIZED_KERNEL_DIR"
"""Env variable pointing at the ``exisiting_optimization_work`` root.

Provides a stable default location the module can suggest to callers
who forgot to set ``FLASHINFER_PRIVATE_CUBIN_DIR``.
"""


def resolve_private_cubin_dir() -> Optional[str]:
    """Return the currently-configured private cubin dir, if any."""
    val = os.environ.get(KIMI_K3_MOE_PRIVATE_CUBIN_ENV)
    if val and os.path.isdir(val):
        return val
    return None


def resolve_optimization_root() -> Optional[str]:
    """Return the optimization-work root (contains the SiTU pool)."""
    val = os.environ.get(KIMI_K3_MOE_OPTIMIZED_KERNEL_ENV)
    if val and os.path.isdir(val):
        return val
    return None


def _default_get_sm_version() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return -1
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor


def get_moe_sm_version() -> int:
    """Return the runtime SM version used for cubin selection."""
    try:
        from tensorrt_llm._utils import get_sm_version as _tllm_get_sm_version

        return int(_tllm_get_sm_version())
    except Exception:
        return _default_get_sm_version()


def is_private_situ_supported() -> bool:
    """Private SiTU cubins are Blackwell sm_100 only."""
    return get_moe_sm_version() in (100, 103)


def invoke_private_situ_moe(
    *,
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    routed_scaling_factor: float = 1.0,
    activation_type_value: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
) -> torch.Tensor:
    """Invoke the private-SiTU MXINT4 fused MoE cubin.

    Parameters mirror ``flashinfer.fused_moe.trtllm_mxint4_block_scale_moe``
    (see ``exisiting_optimization_work/trtllmgen_MOE/tests/moe/test_trtllm_gen_routed_fused_moe.py``
    for the reference driver). Buffers are documented in
    :meth:`KimiK3SparseMoeBlock._init_fused_cubin_bank`.

    ``activation_type_value`` defaults to
    ``ActivationType.Situ.value`` (evaluated lazily against the imported
    flashinfer enum so callers can pass any enum value for negative
    controls without an extra import here).

    Returns the routed output (bf16 ``[num_tokens, hidden_size]``).
    Writes into ``output`` in place when supplied — this lets a caller
    pre-fill a sentinel to detect silent no-op fallbacks.
    """
    # Lazy imports — flashinfer is only available inside the GB200
    # runner, and importing it too early skips the private cubin
    # env-var snapshot in ``flashinfer.jit.env``.
    from flashinfer import fused_moe as _fused_moe
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType

    if activation_type_value is None:
        activation_type_value = int(ActivationType.Situ.value)

    out_list = _fused_moe.trtllm_mxint4_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=int(RoutingMethodType.Renormalize.value),
        do_finalize=True,
        enable_pdl=None,
        activation_type=activation_type_value,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
        norm_topk_prob=norm_topk_prob,
        is_private=True,
    )
    torch.cuda.synchronize(hidden_states.device)

    if output is not None:
        return output
    if isinstance(out_list, (list, tuple)) and len(out_list) > 0:
        return out_list[0]
    return out_list
