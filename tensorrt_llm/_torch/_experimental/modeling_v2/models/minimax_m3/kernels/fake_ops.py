# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake implementations of the MiniMax-M3 decode ops for compile and fake-tensor tracing.

Target packages register these implementations when imported after routing.
"""

import torch


@torch.library.register_fake("trtllm::minimax_m3_mxfp8_low_m_gemm")
def _(x, w, wsf, warps, cluster_k, row_tiles, pdl):
    return x.new_empty((x.shape[0], w.shape[0]))


@torch.library.register_fake("trtllm::minimax_m3_router_gemv")
def _(x, w, rows, pdl):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32)


def _minimax_m3_router_fp4_outputs(x, w):
    m, k = x.shape
    return [
        x.new_empty((m, w.shape[0]), dtype=torch.float32),
        x.new_empty((m, k // 2), dtype=torch.uint8),
        x.new_empty((m * (k // 16),), dtype=torch.uint8),
    ]


@torch.library.register_fake("trtllm::minimax_m3_router_gemv_fp4")
def _(x, w, sf_scale, pdl):
    return _minimax_m3_router_fp4_outputs(x, w)


@torch.library.register_fake("trtllm::minimax_m3_router_tc")
def _(x, w, sf_scale, pdl):
    outputs = _minimax_m3_router_fp4_outputs(x, w)
    return outputs if sf_scale is not None else outputs[:1]


@torch.library.register_fake("trtllm::minimax_m3_producer_pw")
def _(
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


@torch.library.register_fake("trtllm::minimax_m3_mc_allreduce_norm")
def _(
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
