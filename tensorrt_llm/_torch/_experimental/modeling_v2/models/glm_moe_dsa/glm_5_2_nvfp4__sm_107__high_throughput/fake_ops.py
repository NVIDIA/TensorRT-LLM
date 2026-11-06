# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake implementations for this target's native CUDA operators."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import namespace_of

_NS = namespace_of(__name__)


@torch.library.register_fake(f"{_NS}::glm_rubin_mtp_head_gather")
def head_gather(
    input: torch.Tensor,
    comm_buffer: torch.Tensor,
    buffer_flags: torch.Tensor,
    group_size: int,
    float_output: bool = False,
) -> torch.Tensor:
    shape = (*input.shape[:-1], group_size * input.shape[-1])
    return input.new_empty(shape, dtype=torch.float32 if float_output else input.dtype)


@torch.library.register_fake(f"{_NS}::glm_rubin_mtp_sample_glue_adp")
def sample_glue_adp(
    logits: torch.Tensor,
    exchange_buffer: torch.Tensor,
    exchange_flags: torch.Tensor,
    group_size: int,
    segment_rows: int,
    draft_tokens: torch.Tensor,
    draft_step: int,
    embed_table: torch.Tensor | None,
    hidden: torch.Tensor | None,
    enorm_weight: torch.Tensor | None,
    hnorm_weight: torch.Tensor | None,
    eps: float,
    col_lo: int,
    col_hi: int,
    glue_out: torch.Tensor | None,
    positions_in: torch.Tensor | None,
    positions_out: torch.Tensor | None,
    gather_ids: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
    fill_seq_lens: bool,
    kv_lens: torch.Tensor | None,
    kv_update: int,
    rewind_accepted: torch.Tensor | None,
    draft_len: int,
    accepted_tokens: torch.Tensor | None,
    batch_indices: torch.Tensor | None,
    num_accepted: torch.Tensor | None,
    next_new_tokens: torch.Tensor | None,
) -> None:
    return None


@torch.library.register_fake(f"{_NS}::glm_rubin_nvfp4_mla_gather_worklist_pipelined")
def worklist_pipelined(
    host_pool_pointers: torch.Tensor,
    host_pool_mapping: torch.Tensor,
    member_indices: torch.Tensor,
    query_req_indices: torch.Tensor,
    output: torch.Tensor,
    compact_indices: torch.Tensor,
    owner_table: torch.Tensor,
    member_owners: torch.Tensor,
    owner_count: torch.Tensor,
    owner_state: torch.Tensor,
    global_dequant_scale: torch.Tensor,
    layer_idx: int,
    num_requests: int,
    num_pool_tokens: int,
) -> None:
    return None


@torch.library.register_fake(f"{_NS}::glm_rubin_nvfp4_mla_gather_worklist_member")
def worklist_member(
    host_pool_pointers: torch.Tensor,
    host_pool_mapping: torch.Tensor,
    member_owners: torch.Tensor,
    output: torch.Tensor,
    compact_indices: torch.Tensor,
    owner_count: torch.Tensor,
    global_dequant_scale: torch.Tensor,
    layer_idx: int,
    num_pool_tokens: int,
) -> None:
    return None


@torch.library.register_fake(f"{_NS}::glm_rubin_noaux_tc_fp4_quantize_out")
def _fake_route_quant(
    scores,
    bias,
    hidden_states,
    global_scale,
    n_group,
    topk_group,
    topk,
    routed_scaling_factor,
    topk_values,
    topk_indices,
    fp4_out,
    sf_out,
):
    return None
