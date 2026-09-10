# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Uninitialized tensor allocation (thin mirror of torch.empty).

The general-purpose allocation primitive for scratch and output buffers
whose sizing the caller owns; layer-aware output allocation has dedicated
entries (e.g. create_attn_outputs). Contents are garbage until written.
"""

import torch


def empty(shape: list[int], dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Return an uninitialized tensor; identical semantics to torch.empty."""
    return torch.empty(shape, dtype=dtype, device=device)
