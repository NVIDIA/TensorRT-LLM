# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor concatenation (thin mirror of torch.cat)."""

import torch


def concat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Concatenate `tensors` along `dim`; identical semantics to torch.cat."""
    return torch.cat(tensors, dim=dim)
