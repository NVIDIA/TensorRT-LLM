# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 operands for the MXFP8 GEMM entries' tests: E4M3 values and 128x4-swizzled UE8M0 block scales."""

import torch


def swizzle(byte: torch.Tensor) -> torch.Tensor:
    """[N, K / 32] scale bytes into the 128x4-swizzled buffer of ``mxfp8_quantize(w, True)``."""
    rows, cols = byte.shape
    padded_cols, padded_rows = (cols + 3) // 4 * 4, (rows + 127) // 128 * 128
    r = torch.arange(rows, device=byte.device).view(-1, 1)
    c = torch.arange(cols, device=byte.device).view(1, -1)
    index = (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )
    out = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8, device=byte.device)
    out[index.reshape(-1)] = byte.reshape(-1)
    return out


def mxfp8(rows: int, k: int, g: torch.Generator, lo: int = 118, hi: int = 125):
    """An MXFP8 operand: E4M3 values of a unit normal, UE8M0 block scales 2^(lo-127) .. 2^(hi-128)."""
    v = torch.randn(rows, k, generator=g, device="cuda").to(torch.float8_e4m3fn)
    sf = torch.randint(lo, hi, (rows, k // 32), generator=g, device="cuda", dtype=torch.int32)
    return v, swizzle(sf.to(torch.uint8))
