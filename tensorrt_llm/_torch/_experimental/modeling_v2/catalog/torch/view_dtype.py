# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise dtype reinterpretation of a tensor (thin mirror of torch.Tensor.view(dtype)).

Reinterprets the same bytes under another dtype of equal itemsize — no
conversion, no copy. The canonical bridge between a producer that emits a
byte buffer (e.g. packed block scales as uint8) and a consumer that demands
the typed view of those bytes (float8_e4m3fn).
"""

import torch


def view_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Reinterpret `x`'s bytes as `dtype`; identical semantics to Tensor.view(dtype)."""
    return x.view(dtype)
