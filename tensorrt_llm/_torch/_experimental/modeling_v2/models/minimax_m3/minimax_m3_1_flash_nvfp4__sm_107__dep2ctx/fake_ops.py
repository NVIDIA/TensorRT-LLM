# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops, built from its csrc/.

The loader imports this module right after the target is found in the family library, so a fake for an op the
library does not define fails the load.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)


def _returns_none(*args, **kwargs):
    return None


def _in_place(name: str) -> None:
    """An op that only writes its tensor arguments in place returns nothing."""
    torch.library.register_fake(f"{_NS}::{name}")(_returns_none)


def _host_query(name: str) -> None:
    """Ops that answer a host-side question at setup (workspace sizes, launch shapes); tracing never runs them."""

    def fake(*args, **kwargs):
        raise NotImplementedError(f"{_NS}::{name} is a setup-time query; call it outside tracing")

    torch.library.register_fake(f"{_NS}::{name}")(fake)


def _ctx_residual_rmsnorm_mxfp8_fake(
    inter, residual, weight, eps, write_norm, fp4_global_scale=None, enable_pdl=False
):
    """sm103CtxNormMxfp8Op.cpp: (norm_out, MXFP8 rows, swizzled scale bytes, NVFP4 rows, linear NVFP4 scale
    bytes); inter is updated in place. Outputs that are off are empty 1-D tensors of inter's dtype."""
    hidden = inter.shape[-1]
    rows = inter.numel() // hidden
    empty = inter.new_empty((0,))
    norm_out = torch.empty_like(inter) if write_norm else empty
    mxfp8 = torch.empty_like(inter, dtype=torch.float8_e4m3fn)
    sf = inter.new_empty(
        ((rows + 127) // 128 * 128 * ((hidden // 32 + 3) // 4 * 4),), dtype=torch.uint8
    )
    if fp4_global_scale is None:
        return (norm_out, mxfp8, sf, empty, empty)
    fp4 = inter.new_empty((*inter.shape[:-1], hidden // 2), dtype=torch.uint8)
    fp4_sf = inter.new_empty((rows * (hidden // 16),), dtype=torch.uint8)
    return (norm_out, mxfp8, sf, fp4, fp4_sf)


def _ctx_boundary_normq_fake(
    inter,
    residual,
    weight,
    eps,
    write_norm,
    write_f32=False,
    fp4_global_scale=None,
    enable_pdl=False,
    p2=True,
    two_cta=True,
    stages=3,
    p2_stages=3,
):
    """ctxBoundaryNormQuantOp.cpp (DEP2CTX_NORMQ_V2): the served op's five outputs plus the float32 norm;
    outputs that are off are empty 1-D tensors of inter's dtype (write_f32 does not need write_norm)."""
    outs = _ctx_residual_rmsnorm_mxfp8_fake(
        inter, residual, weight, eps, write_norm, fp4_global_scale, enable_pdl
    )
    f32 = torch.empty_like(inter, dtype=torch.float32) if write_f32 else inter.new_empty((0,))
    return (*outs, f32)


def _ctx_boundary_normq_out_fake(
    inter,
    residual,
    weight,
    eps,
    norm_out,
    q_out,
    sf_out,
    f32_out=None,
    fp4_global_scale=None,
    fp4_out=None,
    fp4_sf_out=None,
    stage_ids=None,
    stage_weights=None,
    stage_topk=0,
    enable_pdl=False,
    p2=True,
    two_cta=True,
    stages=3,
    p2_stages=3,
):
    """minimax_m31_ctx_boundary_normq_out (DEP2CTX_NORMQ_V2 / _STAGE): writes into the given buffers; returns
    nothing."""
    return None


def _ctx_boundary_normq_topk_fake(
    shared, plane, scores, residual, weight, eps, write_norm, enable_pdl=False, dual=True
):
    """minimax_m31_ctx_boundary_normq_topk (DEP2CTX_TOPK_FOLD): shared becomes moe + residual in place; returns
    (norm_out, MXFP8 rows, swizzled scale bytes), norm_out an empty 1-D tensor unless write_norm."""
    outs = _ctx_residual_rmsnorm_mxfp8_fake(shared, residual, weight, eps, write_norm)
    return outs[:3]


torch.library.register_fake(f"{_NS}::minimax_m31_ctx_boundary_normq")(_ctx_boundary_normq_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_ctx_boundary_normq_out")(
    _ctx_boundary_normq_out_fake
)
_host_query("minimax_m31_ctx_boundary_normq_shape")
torch.library.register_fake(f"{_NS}::minimax_m31_ctx_boundary_normq_topk")(
    _ctx_boundary_normq_topk_fake
)
_host_query("minimax_m31_ctx_boundary_normq_form")
_host_query("minimax_m31_ctx_boundary_normq_topk_form")
_in_place("minimax_m31_route_stage")
