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


# The MiniMax-M3 sm_107 TP2 / TP4 / TP8 decode ops: one producer and one all-reduce + norm op per TP width (Q and
# index-Q heads per rank from the kernels' producerPw{2,4,8}.h), and the dense QK-norm + RoPE ops.
def _register_minimax_m3_sm107_fakes(tp: int, heads_q: int, heads_index: int) -> None:
    @torch.library.register_fake(f"trtllm::minimax_m3_sm107_tp{tp}_producer_pw")
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
        pdl,
        early,
        prewait_step_inputs=False,
    ):
        # [Q, index Q] in FP8; the K / V / index-K rows go into the caches in place.
        num_tokens = packed.shape[0]
        return [
            packed.new_empty((num_tokens, heads_q, 128), dtype=torch.float8_e4m3fn),
            packed.new_empty((num_tokens, heads_index, 128), dtype=torch.float8_e4m3fn),
        ]

    @torch.library.register_fake(f"trtllm::minimax_m3_sm107_tp{tp}_mc_allreduce_norm")
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
        cap_out=None,
        cap_w=None,
        cap_eps=0.0,
    ):
        # [normed, updated residual] (+ [E4M3 rows, swizzled scale bytes] with quant); cap_out is written in place.
        outputs = [torch.empty_like(residual), torch.empty_like(residual)]
        if quant:
            m, h = residual.shape
            outputs += [
                residual.new_empty((m, h), dtype=torch.float8_e4m3fn),
                residual.new_empty(((m + 127) // 128 * 128 * (h // 32),), dtype=torch.uint8),
            ]
        return outputs


for _tp, _heads_q, _heads_index in ((2, 32, 2), (4, 16, 1), (8, 8, 1)):
    _register_minimax_m3_sm107_fakes(_tp, _heads_q, _heads_index)


def _minimax_m3_sm107_dense_qk_norm_rope_fp8_fake(qkv, *args, **kwargs):
    # A new FP8 tensor shaped like qkv; the KV variant also stores K / V (and the compact Q) in place.
    return torch.empty_like(qkv, dtype=torch.float8_e4m3fn)


for _name in (
    "minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8",
    "minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8_kv",
    "minimax_m3_sm107tp4_dense_qk_norm_rope_fp8",
    "minimax_m3_sm107tp4_dense_qk_norm_rope_fp8_kv",
):
    torch.library.register_fake(f"trtllm::{_name}")(_minimax_m3_sm107_dense_qk_norm_rope_fp8_fake)
