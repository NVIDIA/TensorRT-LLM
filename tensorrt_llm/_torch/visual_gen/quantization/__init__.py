# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quantization support for diffusion models.
"""

from .loader import DynamicLinearWeightLoader
from .ops import quantize_fp8_blockwise, quantize_fp8_per_tensor

__all__ = [
    "DynamicLinearWeightLoader",
    "quantize_fp8_per_tensor",
    "quantize_fp8_blockwise",
]
