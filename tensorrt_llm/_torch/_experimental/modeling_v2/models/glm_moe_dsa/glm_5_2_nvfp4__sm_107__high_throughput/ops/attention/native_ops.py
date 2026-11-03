# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake registrations for the target-owned NVFP4 gather bindings."""

import torch


@torch.library.register_fake("trtllm::glm_rubin_nvfp4_mla_gather_worklist_pipelined")
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


@torch.library.register_fake("trtllm::glm_rubin_nvfp4_mla_gather_worklist_member")
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
