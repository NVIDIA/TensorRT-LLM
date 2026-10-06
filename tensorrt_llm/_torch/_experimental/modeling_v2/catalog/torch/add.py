# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Elementwise addition (thin mirror of torch.add).

torch does not implement float8 arithmetic; inputs are expected in the
high-precision residual dtypes (fp32/bf16/fp16).
"""

import torch


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return `a + b`; identical semantics to torch.add."""
    return torch.add(a, b)
