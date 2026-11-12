# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured locality choices by candidate envelope and resource-normalized rows."""

import torch

_RULES: dict[tuple[int, int, int], tuple[str, int]] = {
    (512, 16, 512): ("local_preserve", 8),
    (512, 16, 1024): ("local_partition", 2),
    (512, 32, 256): ("local_preserve", 12),
    (512, 32, 512): ("local_preserve", 4),
    (512, 32, 1024): ("local_preserve", 2),
    (512, 64, 64): ("local_preserve", 2),
    (512, 64, 256): ("local_preserve", 4),
    (512, 64, 512): ("local_preserve", 4),
    (512, 64, 1024): ("local_partition", 16),
    (512, 127, 32): ("local_preserve", 4),
    (512, 127, 64): ("local_preserve", 2),
    (512, 127, 128): ("local_preserve", 2),
    (512, 127, 256): ("local_preserve", 4),
    (512, 127, 512): ("local_preserve", 4),
    (512, 127, 1024): ("local_preserve", 8),
    (1024, 4, 1024): ("local_preserve", 2),
    (1024, 16, 512): ("local_preserve", 2),
    (1024, 32, 256): ("local_preserve", 10),
    (1024, 32, 512): ("local_preserve", 8),
    (1024, 32, 1024): ("local_preserve", 8),
    (1024, 64, 64): ("local_preserve", 2),
    (1024, 64, 128): ("local_preserve", 4),
    (1024, 64, 256): ("local_preserve", 2),
    (1024, 64, 512): ("local_preserve", 4),
    (1024, 64, 1024): ("local_preserve", 2),
    (1024, 127, 32): ("local_preserve", 4),
    (1024, 127, 64): ("local_preserve", 2),
    (1024, 127, 128): ("local_preserve", 10),
    (1024, 127, 256): ("local_preserve", 2),
    (1024, 127, 512): ("local_partition", 4),
    (1024, 127, 1024): ("local_preserve", 8),
    (2048, 4, 1024): ("local_preserve", 2),
    (2048, 8, 1024): ("global_original", 2),
    (2048, 16, 1024): ("local_preserve", 2),
    (2048, 32, 512): ("local_preserve", 8),
    (2048, 32, 1024): ("local_preserve", 4),
    (2048, 64, 64): ("local_partition", 2),
    (2048, 64, 128): ("local_preserve", 10),
    (2048, 64, 256): ("local_preserve", 12),
    (2048, 64, 512): ("local_preserve", 8),
    (2048, 64, 1024): ("local_preserve", 8),
    (2048, 79, 64): ("local_partition", 4),
    (2048, 79, 256): ("local_preserve", 4),
    (2048, 79, 512): ("local_partition", 2),
    (2048, 79, 1024): ("local_preserve", 2),
}


def choose_locality(
    dtype: torch.dtype, k: int, n: int, rows: int, sms: int, sm_version: int
) -> tuple[str, int] | None:
    """Return (execution mode, coscheduled SM count), or retain global GVR.

    The caller must separately verify device locality capability and input
    placement. Unmeasured dtypes and shape buckets retain global execution.
    """
    if dtype != torch.float32 or sm_version < 107 or rows < 1 or sms < 1 or n <= k:
        return None
    reference_rows = (rows * 208 + sms - 1) // sms
    row_bucket = 1 << (reference_rows - 1).bit_length()
    return _RULES.get((k, n // 2048, row_bucket))
