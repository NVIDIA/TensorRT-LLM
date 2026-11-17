# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The environment variables the Kimi K3 targets read: their names, the values each one accepts and how a value is
read. Both targets read them through these functions, once at import or when a module is built; ``routing.py`` checks
the same values with `declines` before it imports a target, so a value a target would reject declines the deployment
instead. This module imports nothing from the targets."""

from __future__ import annotations

import os
from collections.abc import Mapping

#: Which residual value the DSpark hidden-state tap hands the drafter: "1" (the default) the pre-norm
#: attention-residual mixture, any other value the running prefix sum. It has to match what the drafter checkpoint was
#: distilled against; a mismatch lowers acceptance without an error.
AUX_ATTN_RES_STREAM = "KIMI_K3_AUX_ATTN_RES_STREAM"
#: "1" (the default) runs the fused attention-residual kernels where they take the shape; any other value turns them
#: off.
FUSED_ATTN_RES = "KIMI_K3_FUSED_ATTN_RES"
#: Which attention-residual kernel serves a decode step: one of `ATTN_RES_TOPOLOGIES`, or `ATTN_RES_TOPOLOGY_ON`.
ATTN_RES_TOPOLOGY = "KIMI_K3_ATTN_RES_TOPOLOGY"
#: The most tokens the fused attention-residual epilogue takes, a positive integer.
FUSED_ATTN_RES_MAX_TOKENS = "KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS"
#: "1" runs the latent MoE projections as plain linear layers instead of the min-latency fused GEMM.
DISABLE_MIN_LATENCY_LATENT_PROJ = "TLLM_K3_DISABLE_MIN_LATENCY_LATENT_PROJ"
#: "1" routes up to 64 tokens with noaux_tc_op and mxfp8_quantize instead of the fused moe/kimi_k3_noaux_tc_mxfp8_quant.
DISABLE_FUSED_ROUTE_QUANT = "TLLM_K3_DISABLE_FUSED_ROUTE_QUANT"
#: Unset or "1": the router gate GEMM runs bf16 x bf16 with fp32 output; any other value upcasts to fp32 first.
ROUTER_BF16 = "KIMI_K3_ROUTER_BF16"
#: The starting position count of the MLA layers' identity-RoPE table, a positive integer; unset, the checkpoint's
#: ``max_position_embeddings``.
MLA_MAX_POSITIONS = "KIMI_K3_MLA_MAX_POSITIONS"

#: The attention-residual topologies: ``per_token`` (the default) never uses the persistent kernel, ``persistent``
#: uses it at every shape it implements, ``split`` uses it above `fused_attn_res_max_tokens` tokens.
ATTN_RES_TOPOLOGIES = ("per_token", "persistent", "split")
#: The one value that turns the persistent kernel on without naming a topology: it selects ``split``, the measured
#: policy.
ATTN_RES_TOPOLOGY_ON = "1"
#: The most tokens (rows T) the native attention-residual ops take: ``check_attn_res_contract`` in
#: ``csrc/k3AttnResOp.cpp`` raises above it.
ATTN_RES_MAX_TOKENS = 16384


def aux_attn_res_stream(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether the drafter's hidden-state tap captures the attention-residual mixture, not the running prefix sum
    (`AUX_ATTN_RES_STREAM`)."""
    return environ.get(AUX_ATTN_RES_STREAM, "1") == "1"


def fused_attn_res(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether the fused attention-residual kernels are on (`FUSED_ATTN_RES`)."""
    return environ.get(FUSED_ATTN_RES, "1") == "1"


def attn_res_topology(environ: Mapping[str, str] = os.environ) -> str:
    """The attention-residual topology `ATTN_RES_TOPOLOGY` selects, one of `ATTN_RES_TOPOLOGIES`.

    Raises:
        ValueError: a value that names no topology. A mistyped A/B arm that fell back to the default would measure
            one side twice and report no difference, so the value is not ignored.
    """
    raw = environ.get(ATTN_RES_TOPOLOGY, "per_token")
    if raw == ATTN_RES_TOPOLOGY_ON:
        return "split"
    if raw not in ATTN_RES_TOPOLOGIES:
        raise ValueError(
            f"{ATTN_RES_TOPOLOGY} must be one of {ATTN_RES_TOPOLOGIES} or {ATTN_RES_TOPOLOGY_ON!r} "
            f"(which means 'split'), got {raw!r}"
        )
    return raw


def fused_attn_res_max_tokens(topology: str, environ: Mapping[str, str] = os.environ) -> int:
    """The most tokens the fused attention-residual epilogue takes (`FUSED_ATTN_RES_MAX_TOKENS`). Unset, it follows
    ``topology``: 1 for ``per_token`` (the single-token decode shape only), else 32, the top of the measured range.

    Raises:
        ValueError: a value that is not a positive integer (not ignored, for the same reason as the topology).
    """
    default = "1" if topology == "per_token" else "32"
    raw = environ.get(FUSED_ATTN_RES_MAX_TOKENS, default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{FUSED_ATTN_RES_MAX_TOKENS} must be a positive integer, got {raw!r}"
        ) from None
    if value < 1:
        raise ValueError(f"{FUSED_ATTN_RES_MAX_TOKENS} must be >= 1, got {value}")
    return value


def disable_min_latency_latent_proj(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether the latent MoE projections run as plain linear layers (`DISABLE_MIN_LATENCY_LATENT_PROJ`)."""
    return environ.get(DISABLE_MIN_LATENCY_LATENT_PROJ, "0") == "1"


def disable_fused_route_quant(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether routing skips the fused moe/kimi_k3_noaux_tc_mxfp8_quant (`DISABLE_FUSED_ROUTE_QUANT`)."""
    return environ.get(DISABLE_FUSED_ROUTE_QUANT, "0") == "1"


def router_bf16(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether the router gate GEMM reads bf16 operands (`ROUTER_BF16`)."""
    raw = environ.get(ROUTER_BF16)
    return raw == "1" if raw is not None else True


def mla_max_positions(default: int, environ: Mapping[str, str] = os.environ) -> int:
    """The starting position count of the MLA identity-RoPE table: `MLA_MAX_POSITIONS`, else ``default``.

    Raises:
        ValueError: a value that is not a positive integer.
    """
    if MLA_MAX_POSITIONS not in environ:
        return default
    raw = environ[MLA_MAX_POSITIONS]
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{MLA_MAX_POSITIONS} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{MLA_MAX_POSITIONS} must be >= 1, got {value}")
    return value


def declines(environ: Mapping[str, str] = os.environ) -> list[str]:
    """Each variable whose value a target would reject when it reads it, with the reason. The others accept every
    value."""
    reasons: list[str] = []
    topology = "per_token"
    try:
        topology = attn_res_topology(environ)
    except ValueError as error:
        reasons.append(str(error))
    try:
        fused_attn_res_max_tokens(topology, environ)
    except ValueError as error:
        reasons.append(str(error))
    try:
        mla_max_positions(1, environ)
    except ValueError as error:
        reasons.append(str(error))
    return reasons
