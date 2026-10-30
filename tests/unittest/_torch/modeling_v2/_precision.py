# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped precision settings for modeling_v2 test reference calculations."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def true_fp32_matmul() -> Iterator[None]:
    """Use IEEE FP32 CUDA matmuls and restore the caller's previous policy."""
    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    try:
        matmul.fp32_precision = "ieee"
        yield
    finally:
        matmul.fp32_precision = previous
