# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locality policy by candidate count and resource-normalized rows."""

import torch

COSCHEDULED_SM_COUNT = 4


def choose_backfill(
    dtype: torch.dtype, k: int, n: int, rows: int, sms: int, sm_version: int
) -> bool:
    """Select FP32 backfill after allocation validation."""
    if sm_version < 107 or sms < 1 or dtype != torch.float32 or k not in (512, 1024):
        return False
    reference_rows = (rows * 208 + sms - 1) // sms
    if n < 131072 or reference_rows < 737:
        return False
    if n > 262143 or reference_rows > 4096:
        return True
    return (
        737 <= reference_rows <= 832
        or 921 <= reference_rows <= 1024
        or 1473 <= reference_rows <= 1664
    )


# Inclusive (K, N range, resource-normalized row range, execution mode).
# None denotes an open upper bound.
_REGIONS: tuple[tuple[int, int, int | None, int, int | None, str], ...] = (
    (512, 131072, None, 17, 128, "local_preserve"),
    (512, 65536, None, 129, None, "local_preserve"),
    (1024, 32768, 131071, 129, None, "local_preserve"),
    (1024, 131072, None, 17, None, "local_preserve"),
    (2048, 4096, 32767, 513, None, "local_preserve"),
    (2048, 32768, None, 129, None, "local_preserve"),
)

_LOWP_REGIONS: dict[torch.dtype, tuple[tuple[int, int, int | None, int, int | None, str], ...]] = {
    torch.float16: (
        (512, 131072, None, 257, None, "local_preserve"),
        (512, 32768, 131071, 513, None, "local_preserve"),
        (512, 258048, None, 129, 256, "local_preserve"),
        (1024, 131072, None, 257, None, "local_preserve"),
        (1024, 65536, 131071, 513, None, "local_preserve"),
        (1024, 258048, None, 129, 256, "local_preserve"),
        (2048, 131072, None, 257, None, "local_preserve"),
        (2048, 32768, 131071, 513, None, "local_preserve"),
        (2048, 159744, None, 129, 256, "local_preserve"),
    ),
    torch.bfloat16: (
        (512, 131072, None, 257, None, "local_preserve"),
        (512, 258048, None, 129, 256, "local_preserve"),
        (512, 65536, 131071, 513, None, "local_preserve"),
        (1024, 131072, None, 257, None, "local_preserve"),
        (1024, 258048, None, 129, 256, "local_preserve"),
        (1024, 65536, 131071, 513, None, "local_preserve"),
        (2048, 65536, None, 257, None, "local_preserve"),
        (2048, 159744, None, 129, 256, "local_preserve"),
        (2048, 32768, 65535, 513, None, "local_preserve"),
    ),
}


def choose_locality(
    dtype: torch.dtype, k: int, n: int, rows: int, sms: int, sm_version: int
) -> tuple[str, int] | None:
    """Return (execution mode, coscheduled SM count), or retain global GVR.

    The caller must verify allocation ownership and device resource compatibility.
    """
    if sm_version < 107 or rows < 1 or sms < 1 or n <= k:
        return None
    regions = _REGIONS if dtype == torch.float32 else _LOWP_REGIONS.get(dtype, ())
    reference_rows = (rows * 208 + sms - 1) // sms
    return next(
        (
            (mode, COSCHEDULED_SM_COUNT)
            for kk, lo, hi, blo, bhi, mode in regions
            if k == kk
            and lo <= n
            and (hi is None or n <= hi)
            and blo <= reference_rows
            and (bhi is None or reference_rows <= bhi)
        ),
        None,
    )
