# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move periodic GDN checkpoint state into reserved recurrent-state slots."""

import torch

from tensorrt_llm._torch.modules.fla.fused_state_io import cast_scatter_fp32_vk_to_vk


def gather_conv_snapshot_windows(
    mixed_qkv: torch.Tensor, token_indices: torch.Tensor, count: int, width: int
) -> torch.Tensor:
    """Gather raw convolution inputs ending at each checkpoint.

    ``mixed_qkv`` is ``[tokens, channels]``; integer ``token_indices`` has
    ``count * width`` entries. Returns ``[count, channels, width]``.
    """
    return (
        mixed_qkv.index_select(0, token_indices)
        .view(count, width, mixed_qkv.shape[-1])
        .transpose(1, 2)
        .contiguous()
    )


def store_gdn_snapshots(
    ssm_states: torch.Tensor,
    conv_states: torch.Tensor,
    checkpoint_states: torch.Tensor,
    conv_windows: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Write one layer's packed SSM and convolution checkpoints to its slots.

    ``ssm_states`` is ``[pool_slots, heads, value_dim, key_dim]`` and
    ``conv_states`` is ``[pool_slots, channels, width]``. FP32
    ``checkpoint_states`` has one SSM row per integer slot index;
    ``conv_windows`` is ``[count, channels, width]`` and ``slots`` is ``[count]``.
    """
    cast_scatter_fp32_vk_to_vk(checkpoint_states, ssm_states, slots)
    conv_states.index_copy_(0, slots.to(torch.long), conv_windows.to(conv_states.dtype))
