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


def _sparse_decode_own_t_prepare_fake(anchor, max_items):
    """minimax_m3_sm107_sparse_decode_own_t_prepare configures the device and reserves the block-owner decode's
    workspace once, before graph capture; traced, it reserves nothing."""
    return 0


def _nvfp4_qkv_indexer_norm_rope_kv_insert_fake(
    packed,
    kv_data_cache,
    kv_scale_cache,
    index_k_cache,
    out_cache_loc,
    kv_quant_scale,
    num_heads_q,
    num_heads_kv,
    num_heads_index,
    head_dim,
    rotary_dim,
    eps,
    q_weight,
    k_weight,
    index_q_weight,
    index_k_weight,
    rotary_cos_sin,
    position_ids,
    early_trigger=False,
):
    """FP8 q [T, Hq, head_dim] and index-q [T, Hidx, head_dim]; the paged caches are written in place."""
    num_tokens = packed.shape[0]
    return (
        packed.new_empty((num_tokens, num_heads_q, head_dim), dtype=torch.float8_e4m3fn),
        packed.new_empty((num_tokens, num_heads_index, head_dim), dtype=torch.float8_e4m3fn),
    )


def _fp8_qkv_indexer_norm_rope_kv_insert_fake(
    packed,
    kv_cache,
    index_k_cache,
    out_cache_loc,
    num_heads_q,
    num_heads_kv,
    num_heads_index,
    head_dim,
    rotary_dim,
    eps,
    q_weight,
    k_weight,
    index_q_weight,
    index_k_weight,
    rotary_cos_sin,
    position_ids,
):
    """FP8 q [T, Hq, head_dim] and index-q [T, Hidx, head_dim]; the paged caches are written in place."""
    num_tokens = packed.shape[0]
    return (
        packed.new_empty((num_tokens, num_heads_q, head_dim), dtype=torch.float8_e4m3fn),
        packed.new_empty((num_tokens, num_heads_index, head_dim), dtype=torch.float8_e4m3fn),
    )


def _sampling_from_probs_mc_fi_fake(probs, seed, offset, workspace, ctas_per_row):
    """One sampled token per row of probs, int32 [rows]."""
    return probs.new_empty((probs.shape[0],), dtype=torch.int32)


torch.library.register_fake(f"{_NS}::minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert")(
    _fp8_qkv_indexer_norm_rope_kv_insert_fake
)
torch.library.register_fake(f"{_NS}::minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert")(
    _nvfp4_qkv_indexer_norm_rope_kv_insert_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_sampling_from_probs_mc_fi")(
    _sampling_from_probs_mc_fi_fake
)
_host_query("minimax_m31_sampling_from_probs_mc_fi_ws_bytes")
torch.library.register_fake(f"{_NS}::minimax_m3_sm107_sparse_decode_own_t_prepare")(
    _sparse_decode_own_t_prepare_fake
)
_in_place("minimax_m3_sm107_sparse_decode_own_t")
