# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor reshape (thin mirror of torch.reshape).

Covers view use cases too: returns a view when the layout allows, copies
otherwise — unlike Tensor.view, it never fails on non-contiguous input.
"""

import torch


def reshape(x: torch.Tensor, shape: list[int]) -> torch.Tensor:
    """Reshape `x` to `shape`; identical semantics to torch.reshape."""
    return torch.reshape(x, shape)
