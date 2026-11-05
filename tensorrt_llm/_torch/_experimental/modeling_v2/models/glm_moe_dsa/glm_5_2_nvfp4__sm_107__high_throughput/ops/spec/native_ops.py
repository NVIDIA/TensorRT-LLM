# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake registrations for the owned MTP head gather and draft step."""

import torch


@torch.library.register_fake("trtllm::glm_rubin_mtp_head_gather")
def head_gather(
    input: torch.Tensor,
    comm_buffer: torch.Tensor,
    buffer_flags: torch.Tensor,
    group_size: int,
    float_output: bool = False,
) -> torch.Tensor:
    shape = (*input.shape[:-1], group_size * input.shape[-1])
    return input.new_empty(shape, dtype=torch.float32 if float_output else input.dtype)


@torch.library.register_fake("trtllm::glm_rubin_mtp_sample_glue_adp")
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
