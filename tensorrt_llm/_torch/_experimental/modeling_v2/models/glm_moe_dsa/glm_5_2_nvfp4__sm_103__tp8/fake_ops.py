# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) implementations of this target's native ops, built from its csrc/.

The loader imports this module right after the target is found in the family
library, so a fake for an op the library does not define fails the load.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)

# The layer shape the kernels are compiled for (csrc/postAttentionHop.h, csrc/draftRoutedMoe.h).
_HIDDEN_SIZE = 6144
_NUM_EXPERTS = 256
_TOP_K = 8


def _host_query(qualname: str):
    """Ops without tensor arguments answer a host-side question at setup; tracing never runs them."""

    def fake(*args, **kwargs):
        raise NotImplementedError(f"{qualname} is a setup-time query; call it outside tracing")

    return torch.library.register_fake(qualname)(fake)


_host_query(f"{_NS}::cross_rank_topk_workspace_info")
_host_query(f"{_NS}::cross_rank_topk_cluster_merge_workspace_info")
_host_query(f"{_NS}::cross_rank_topk_global_emit_workspace_info")
_host_query(f"{_NS}::sparse_mla_decode_supported")


@torch.library.register_fake(f"{_NS}::cross_rank_topk")
def _(
    logits,
    kv_lens,
    out,
    state,
    mc_workspace,
    uc_workspace,
    rank,
    world_size,
    next_n,
    debug_times=None,
):
    return None


@torch.library.register_fake(f"{_NS}::cross_rank_topk_cluster_merge")
def _(
    logits,
    kv_lens,
    out,
    state,
    mc_workspace,
    uc_workspace,
    rank,
    world_size,
    next_n,
    debug_times=None,
):
    return None


@torch.library.register_fake(f"{_NS}::cross_rank_topk_global_emit")
def _(
    logits,
    kv_lens,
    out,
    state,
    mc_workspace,
    uc_workspace,
    rank,
    world_size,
    next_n,
    req_idx,
    req_idx_bias,
    block_table,
    block_size,
    stride_factor,
    layer_ids,
    layer_id,
    copy_dst,
    copy_row_start,
    copy_rows,
    copy_mode,
    copy_next_n,
    num_accepted,
    debug_times=None,
):
    shape = tuple(out.shape) if layer_ids is None else (layer_ids.shape[0], *out.shape)
    return out.new_empty(shape)


@torch.library.register_fake(f"{_NS}::dense_mlp_gate_up_swiglu")
def _(x, gate_up_weight, variant=1):
    return x.new_empty((x.shape[0], gate_up_weight.shape[0] // 2))


@torch.library.register_fake(f"{_NS}::dense_mlp_down")
def _(h, down_weight, variant=0):
    return h.new_empty((h.shape[0], down_weight.shape[0]))


@torch.library.register_fake(f"{_NS}::dense_mlp")
def _(x, gate_up_weight, down_weight, gate_up_variant=1, down_variant=0):
    return x.new_empty((x.shape[0], down_weight.shape[0]))


@torch.library.register_fake(f"{_NS}::draft_live_row_hop")
def _(input, residual, row_ids, gamma, eps, router_weight, uc_buffer, mc_ptr, flags, rank, nranks):
    rows = row_ids.shape[0]
    return [
        input.new_empty((rows, _HIDDEN_SIZE)),
        residual.new_empty((rows, _HIDDEN_SIZE)),
        input.new_empty((rows, _NUM_EXPERTS), dtype=torch.float32),
    ]


@torch.library.register_fake(f"{_NS}::draft_routed_moe")
def _(
    x,
    router_logits,
    routing_bias,
    w3_w1,
    w2,
    h_scratch,
    route_scratch,
    routed_scaling,
    fc1_ctas,
    fc1_warps,
    fc2_ctas,
    fc2_warps,
    flags,
    debug_fc2=None,
    timeline=None,
):
    return x.new_empty((x.shape[0], _HIDDEN_SIZE))


@torch.library.register_fake(f"{_NS}::draft_tail")
def _(
    hidden,
    weight,
    tokens,
    scratch,
    keys_out,
    vocab_offset,
    tp_rank,
    tp_size,
    mode,
    mc_ptr,
    uc_ptr,
    peer_ptrs,
    variant,
    num_ctas,
    keep_permille,
    pdl,
    timeout_cycles,
    logits_out=None,
):
    return None


@torch.library.register_fake(f"{_NS}::draft_tail_from_logits")
def _(
    logits,
    tokens,
    scratch,
    keys_out,
    vocab_offset,
    tp_rank,
    tp_size,
    mode,
    mc_ptr,
    uc_ptr,
    peer_ptrs,
    pdl,
    timeout_cycles,
):
    return None


@torch.library.register_fake(f"{_NS}::rope_kv_append")
def _(*args, **kwargs):
    return None


@torch.library.register_fake(f"{_NS}::indexer_layer_rope_kv_append")
def _(*args, **kwargs):
    return None


@torch.library.register_fake(f"{_NS}::topk_index_bridge")
def _(
    topk,
    num_rows,
    req_idx,
    req_idx_bias,
    block_table,
    block_size,
    stride_factor,
    layer_ids,
    layer_id,
    copy_dst,
    copy_row_start,
    copy_rows,
    copy_mode,
    next_n,
    num_accepted,
):
    rows_topk = (num_rows, topk.shape[1])
    return topk.new_empty(rows_topk if layer_ids is None else (layer_ids.shape[0], *rows_topk))


@torch.library.register_fake(f"{_NS}::post_attention_hop")
def _(input, residual, gamma, eps, router_weight, uc_buffer, mc_ptr, flags, rank, nranks):
    rows = input.numel() // input.shape[-1]
    logits = (
        input.new_empty((rows, _NUM_EXPERTS), dtype=torch.float32)
        if router_weight is not None
        else input.new_empty((0,), dtype=torch.float32)
    )
    return [torch.empty_like(input), torch.empty_like(residual), logits]


@torch.library.register_fake(f"{_NS}::post_attention_hop_from_partials")
def _(
    residual, gamma, eps, router_weight, uc_buffer, flags, rank, nranks, early_poll, router_load=1
):
    rows = residual.numel() // _HIDDEN_SIZE
    logits = (
        residual.new_empty((rows, _NUM_EXPERTS), dtype=torch.float32)
        if router_weight is not None
        else residual.new_empty((0,), dtype=torch.float32)
    )
    return [torch.empty_like(residual), torch.empty_like(residual), logits]


@torch.library.register_fake(f"{_NS}::post_attention_hop_routing")
def _(
    residual,
    gamma,
    eps,
    router_weight,
    routing_bias,
    uc_buffer,
    flags,
    gather,
    zstate,
    rank,
    nranks,
    early_poll,
    router_load,
    routed_scaling,
    logits_in=None,
    stamps=None,
):
    rows = residual.numel() // _HIDDEN_SIZE
    return [
        torch.empty_like(residual),
        torch.empty_like(residual),
        residual.new_empty((rows, _NUM_EXPERTS), dtype=torch.float32),
        residual.new_empty((2, rows, _TOP_K), dtype=torch.int32),
    ]


@torch.library.register_fake(f"{_NS}::sparse_mla_decode")
def _(q, kv_pool_ptr, indices, bmm1_scale, bmm2_scale, seq_lens_kv, seq_len_q, out):
    return None
