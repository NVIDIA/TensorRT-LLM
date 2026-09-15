# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor split (thin mirror of torch.split)."""

import torch


def split(
    tensor: torch.Tensor, split_size_or_sections: int | list[int], dim: int = 0
) -> tuple[torch.Tensor, ...]:
    """Split `tensor` along `dim`; identical semantics to torch.split."""
    return torch.split(tensor, split_size_or_sections, dim=dim)
