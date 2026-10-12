# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place copy into a tensor or view (thin mirror of torch.Tensor.copy_).

Writes `src` into `dst` element-wise (broadcasting and dtype conversion follow
torch semantics) and returns `dst`. The canonical way to fill a slice of a
larger buffer (e.g. `copy_(k[..., :nope], k_nope)`); the destination's
pre-call contents are destroyed.
"""

import torch


def copy_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Copy `src` into `dst` in place; identical semantics to Tensor.copy_."""
    return dst.copy_(src)
