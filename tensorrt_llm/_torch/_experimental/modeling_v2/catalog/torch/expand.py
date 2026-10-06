# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Broadcast view over singleton dims (thin mirror of torch.Tensor.expand).

Returns a zero-copy view with stride 0 on the expanded dims; only singleton
dims can be expanded, and -1 keeps a dim unchanged. Writing through the view
is unsafe (aliased elements) — copy first if mutation is needed.
"""

import torch


def expand(x: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """Return `x` broadcast to `sizes`; identical semantics to Tensor.expand."""
    return x.expand(sizes)
