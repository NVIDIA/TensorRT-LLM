# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured locality choices by candidate envelope and resource-normalized rows."""

import torch

COSCHEDULED_SM_COUNT = 4
# Inclusive (K, N range, resource-normalized row range, execution mode).
_REGIONS: tuple[tuple[int, int, int, int, int, str], ...] = (
    (512, 131072, 262143, 17, 128, "local_preserve"),
    (512, 65536, 262143, 129, 1024, "local_preserve"),
    (1024, 32768, 131071, 129, 1024, "local_preserve"),
    (1024, 131072, 262143, 17, 1024, "local_preserve"),
    (2048, 4096, 16383, 513, 1024, "local_preserve"),
    (2048, 16384, 32767, 513, 1024, "global_original"),
    (2048, 32768, 163839, 129, 1024, "local_preserve"),
)


def choose_locality(
    dtype: torch.dtype, k: int, n: int, rows: int, sms: int, sm_version: int
) -> tuple[str, int] | None:
    """Return (execution mode, coscheduled SM count), or retain global GVR.

    The caller must separately verify device locality capability and input
    placement. Unmeasured dtypes and shapes outside the calibrated regions retain global execution.
    """
    if dtype != torch.float32 or sm_version < 107 or rows < 1 or sms < 1 or n <= k:
        return None
    reference_rows = (rows * 208 + sms - 1) // sms
    return next(
        (
            (mode, COSCHEDULED_SM_COUNT)
            for kk, lo, hi, blo, bhi, mode in _REGIONS
            if k == kk and lo <= n <= hi and blo <= reference_rows <= bhi
        ),
        None,
    )
