# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill causal depthwise conv1d over variable-length sequences, with per-slot conv states."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    silu_activation: bool,
    pad_slot_id: int,
    out: Optional[torch.Tensor] = None,
) -> None:
    """Convolve each sequence of `x` with `weight` causally, from its slot's conv state, in place.

    Writes the result into `out` (or back into `x`) and each sequence's last `width - 1`
    inputs into `conv_states[cache_indices[b]]`. Returns None.
    """
    torch.ops.trtllm.causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        pad_slot_id,
        out,
    )
