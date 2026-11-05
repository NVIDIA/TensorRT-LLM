# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic fp8_ds_mla pools for the entry tests: rows as the writer stores them."""

import torch


def ds_mla_rows(n: int, g: torch.Generator, nope: bool = False) -> torch.Tensor:
    """`n` uint8 [656] rows: e4m3 codes in (224, 448] magnitude as the writer leaves them,
    power-of-two tile scales 2^-13..2^0, bf16 rope (zero for a NoPE layer)."""
    codes = (torch.rand(n, 512, generator=g) * 2 - 1) * 448
    codes = codes.to(torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.exp2(torch.randint(-13, 1, (n, 4), generator=g).float())
    rope = torch.zeros(n, 64) if nope else torch.randn(n, 64, generator=g) * 3
    return torch.cat([codes, scale.view(torch.uint8), rope.to(torch.bfloat16).view(torch.uint8)], 1)
