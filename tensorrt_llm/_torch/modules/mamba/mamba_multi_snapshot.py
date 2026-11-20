# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store Mamba SSD chunk-boundary states in reserved recurrent slots."""

import torch


def store_mamba_snapshots(
    checkpoint_states: torch.Tensor,
    ssm_states: torch.Tensor,
    conv_states: torch.Tensor,
    conv_windows: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Copy each selected SSD boundary and its convolution window.

    ``checkpoint_states`` has shape ``[count, *state_shape]`` and
    ``ssm_states`` has shape ``[pool_slots, *state_shape]``. ``conv_windows``
    is ``[count, channels, width]`` and ``conv_states`` is
    ``[pool_slots, channels, width]``; integer ``slots`` is ``[count]``.
    """
    slots = slots.to(torch.long)
    ssm_states.index_copy_(0, slots, checkpoint_states.to(ssm_states.dtype))
    conv_states.index_copy_(0, slots, conv_windows.to(conv_states.dtype))
