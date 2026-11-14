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
    """Ops that answer a host-side question at setup (launch shapes); tracing never runs them."""

    def fake(*args, **kwargs):
        raise NotImplementedError(f"{_NS}::{name} is a setup-time query; call it outside tracing")

    torch.library.register_fake(f"{_NS}::{name}")(fake)


def _qkv_front_fake(
    residual,
    norm_weight,
    eps,
    out_residual,
    out_normed,
    ws_uc,
    flags,
    rank,
    world,
    maxt,
    prepushed,
    weight,
    weight_scale,
    xq_ws,
    xsf_ws,
    ctr,
    kv_data,
    kv_scale,
    index_k,
    out_cache_loc,
    kv_quant_scale,
    peps,
    q_weight,
    k_weight,
    index_q_weight,
    index_k_weight,
    rotary_cos_sin,
    position_ids,
    heads_q,
    split,
    pdl,
    early,
):
    """minimax_m31_qkv_front: the boundary's residual / normed rows, the K / V / index-K cache bytes and the workspaces
    are written in place; returns new FP8 q [T, heads_q, 128] and index-q [T, 1, 128] (T = residual rows)."""
    num_tokens = residual.shape[0]
    return [
        residual.new_empty((num_tokens, heads_q, 128), dtype=torch.float8_e4m3fn),
        residual.new_empty((num_tokens, 1, 128), dtype=torch.float8_e4m3fn),
    ]


# The TP4 MoE-boundary post (kernels/hoist): writes out_res / out_norm / route / x4 in place.
_in_place("minimax_m3_sm107tp4_hoist_post")
# The fused drafter context store + advance + forward indices (dspark_tailfold TAILGLUE_A).
_in_place("minimax_m31_tail_glue_a")
torch.library.register_fake(f"{_NS}::minimax_m31_qkv_front")(_qkv_front_fake)
_host_query("minimax_m31_qkv_front_max_clusters")
