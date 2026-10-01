# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero-copy swap of two dimensions (thin mirror of torch.transpose).

Returns a strided view sharing storage with the input; the canonical way to
present a `[tokens, heads, dim]` activation to a batched-gemm entry whose
batch axis is the head axis.
"""

import torch


def transpose(x: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Return `x` with `dim0` and `dim1` swapped; identical semantics to torch.transpose."""
    return torch.transpose(x, dim0, dim1)
