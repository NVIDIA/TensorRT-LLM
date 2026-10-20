# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""PrimsTS dense FP8/NVFP4 GEMM APIs."""

from .api import (
    PreparedFp4Linear,
    fp4_linear,
    fp4_linear_swiglu,
    fp4_qkv_qknorm_rope,
    fp8_linear,
    fp8_linear_swiglu,
    fp8_qkv_qknorm_rope,
    prepare_fp4_linear,
    prepare_fp4_linear_swiglu,
    prepare_fp4_qkv_qknorm_rope,
)
from .config import PrimsTsGemmConfig

__all__ = [
    "PrimsTsGemmConfig",
    "PreparedFp4Linear",
    "fp8_linear",
    "fp8_linear_swiglu",
    "fp8_qkv_qknorm_rope",
    "fp4_linear",
    "fp4_linear_swiglu",
    "fp4_qkv_qknorm_rope",
    "prepare_fp4_linear",
    "prepare_fp4_linear_swiglu",
    "prepare_fp4_qkv_qknorm_rope",
]
