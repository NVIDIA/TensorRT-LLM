# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store PLE recurrent windows at prefill snapshot points."""

import torch


def store_ple_snapshots(
    conv_input: torch.Tensor,
    input_ids: torch.Tensor,
    conv_states: torch.Tensor,
    ngram_context: torch.Tensor,
    conv_indices: torch.Tensor,
    token_indices: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Write short-convolution and token-history windows to reserved slots.

    ``conv_input`` is ``[tokens, channels]`` and integer ``input_ids`` is
    ``[tokens]``. Pools are ``conv_states[pool_slots, channels, width]`` and
    integer ``ngram_context[pool_slots, history]``. For ``count`` integer
    ``slots``, integer ``conv_indices`` and ``token_indices`` have lengths
    ``count * width`` and ``count * history``, respectively.
    """
    count = slots.numel()
    width = conv_states.shape[-1]
    conv = (
        conv_input.index_select(0, conv_indices)
        .view(count, width, conv_input.shape[-1])
        .transpose(1, 2)
        .contiguous()
    )
    history = input_ids.index_select(0, token_indices).view(count, ngram_context.shape[-1])
    slots = slots.to(torch.long)
    conv_states.index_copy_(0, slots, conv.to(conv_states.dtype))
    ngram_context.index_copy_(0, slots, history.to(ngram_context.dtype))
