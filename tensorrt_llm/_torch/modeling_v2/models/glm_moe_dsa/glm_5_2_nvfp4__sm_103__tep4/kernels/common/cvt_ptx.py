# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed f32x2 -> bf16x2 -> fp8x2 conversions for the PDL-chain GEMM epilogue."""

from cutlass.experimental.primitives.nvvm_wrapper import (  # noqa: F401
    FPRoundingMode,
    SaturationMode,
)
from cutlass.experimental.primitives.nvvm_wrapper import (
    _convert_bf16x2_to_f8x2 as cvt_bf16x2_to_f8x2,
)
from cutlass.experimental.primitives.nvvm_wrapper import (
    _convert_f32x2_to_bf16x2 as cvt_f32x2_to_bf16x2,
)

__all__ = ["FPRoundingMode", "SaturationMode", "cvt_bf16x2_to_f8x2", "cvt_f32x2_to_bf16x2"]
