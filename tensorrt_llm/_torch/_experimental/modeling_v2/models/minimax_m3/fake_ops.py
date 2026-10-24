# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of the torch ops that the MiniMax-M3 decode targets add (``trtllm::step_inputs_scatter``
and the ``trtllm::minimax_m3_*`` decode ops in ``cpp/tensorrt_llm/thop/minimaxM3/``), for torch.compile and fake-tensor
tracing. Only loaded native ops without an existing fake are registered. Importing the family must remain safe
when the native extension is absent, so unsupported routes can fall back to the built-in model. Call register()
again after loading an overlay; registration is idempotent."""

import torch

from ...catalog.fakes.m31_fake_ops import _has_fake

_REGISTERED = set()


def _step_inputs_scatter_fake(stage):
    # Writes the staged rows into their destinations; returns nothing.
    return None


def _minimax_m3_mxfp8_low_m_gemm_fake(x, w, wsf, warps, cluster_k, row_tiles, pdl):
    return x.new_empty((x.shape[0], w.shape[0]))


def _minimax_m3_router_gemv_fake(x, w, rows, pdl):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32)


def _minimax_m3_router_fp4_outputs(x, w):
    m, k = x.shape
    return [
        x.new_empty((m, w.shape[0]), dtype=torch.float32),
        x.new_empty((m, k // 2), dtype=torch.uint8),
        x.new_empty((m * (k // 16),), dtype=torch.uint8),
    ]


def _minimax_m3_router_gemv_fp4_fake(x, w, sf_scale, pdl):
    return _minimax_m3_router_fp4_outputs(x, w)


def _minimax_m3_router_tc_fake(x, w, sf_scale, pdl):
    outputs = _minimax_m3_router_fp4_outputs(x, w)
    return outputs if sf_scale is not None else outputs[:1]


def _minimax_m3_producer_pw_fake(
    packed,
    kv_data,
    kv_scale,
    index_k,
    out_cache_loc,
    kv_quant_scale,
    eps,
    q_weight,
    k_weight,
    index_q_weight,
    index_k_weight,
    rotary_cos_sin,
    position_ids,
    heads_q,
    pdl,
    early,
):
    num_tokens = packed.shape[0]
    return [
        packed.new_empty((num_tokens, heads_q, 128), dtype=torch.float8_e4m3fn),
        packed.new_empty((num_tokens, 1, 128), dtype=torch.float8_e4m3fn),
    ]


def _minimax_m3_mc_allreduce_norm_fake(
    x,
    x2,
    residual,
    norm_w,
    eps,
    ws_uc,
    ws_mc_ptr,
    flags,
    rank,
    world,
    cluster,
    pdl,
    perm,
    scales,
    maxt,
    quant,
    prepushed,
    opts,
):
    outputs = [torch.empty_like(residual), torch.empty_like(residual)]
    if quant:
        m, h = residual.shape
        outputs += [
            residual.new_empty((m, h), dtype=torch.float8_e4m3fn),
            residual.new_empty(((m + 127) // 128 * 128 * (h // 32),), dtype=torch.uint8),
        ]
    return outputs


FAKES = {
    "step_inputs_scatter": _step_inputs_scatter_fake,
    "minimax_m3_mxfp8_low_m_gemm": _minimax_m3_mxfp8_low_m_gemm_fake,
    "minimax_m3_router_gemv": _minimax_m3_router_gemv_fake,
    "minimax_m3_router_gemv_fp4": _minimax_m3_router_gemv_fp4_fake,
    "minimax_m3_router_tc": _minimax_m3_router_tc_fake,
    "minimax_m3_producer_pw": _minimax_m3_producer_pw_fake,
    "minimax_m3_mc_allreduce_norm": _minimax_m3_mc_allreduce_norm_fake,
}


def register():
    """Register available native ops, preserving existing Meta kernels or Python fakes."""
    for name, fake in FAKES.items():
        if name in _REGISTERED or not hasattr(torch.ops.trtllm, name):
            continue
        qualname = f"trtllm::{name}"
        if not _has_fake(qualname):
            torch.library.register_fake(qualname)(fake)
        _REGISTERED.add(name)


register()
