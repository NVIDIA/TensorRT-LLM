# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 copy of normed2 that the prepushed post writes for the GEMM after it (persistent per blk)."""

import torch


def _mx_bufs(blk):
    """Persistent MXFP8 copy of normed2 written by the boundary (padding pre-zeroed once)."""
    if not hasattr(blk, "h10_mxq"):
        from .stock_stages import mx_quant

        xq, xs = mx_quant(blk.normed2)
        blk.h10_mxq = torch.empty_like(xq)
        blk.h10_mxs = torch.zeros_like(xs)
    return blk.h10_mxq, blk.h10_mxs
