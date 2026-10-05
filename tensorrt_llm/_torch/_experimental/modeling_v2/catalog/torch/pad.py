# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constant padding of a tensor's trailing dims (thin mirror of torch.nn.functional.pad).

Layout glue: widens a tensor to a kernel's required operand width by
appending constant-valued columns (the padded region multiplies zero-valued
padded weights in block-scaled GEMMs, so the fill value must be finite).
"""

import torch
import torch.nn.functional as F


def pad(x: torch.Tensor, padding: list[int], value: float = 0.0) -> torch.Tensor:
    """Return `x` padded with `value` per the last-dim-first `padding` pairs."""
    return F.pad(x, padding, mode="constant", value=value)
