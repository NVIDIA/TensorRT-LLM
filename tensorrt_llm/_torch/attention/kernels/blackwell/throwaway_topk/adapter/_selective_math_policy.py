# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Numerical modes for selective TopK."""

import torch


def _validate_math_mode(math_mode):
    if not isinstance(math_mode, str) or math_mode not in ("relu_fast", "relu_strict"):
        raise ValueError("math_mode must be 'relu_fast' or 'relu_strict'")
    return math_mode


def validate_math_mode(math_mode, *, qkv_dtype, num_heads, logits_dtype):
    if qkv_dtype == "mxfp8" and math_mode == "relu_fast":
        raise ValueError("MXFP8 requires strict ReLU; relu_fast is unsupported")
    if math_mode in ("native_bf16", "native_fp16"):
        expected_dtype = torch.bfloat16 if math_mode == "native_bf16" else torch.float16
        if (
            qkv_dtype not in ("fp8", "mxfp4", "mxfp8")
            or num_heads not in (4, 32, 64)
            or logits_dtype != expected_dtype
        ):
            raise ValueError(
                f"{math_mode} requires FP8, MXFP8 or MXFP4 inputs, H4/H32/H64 and {expected_dtype} logits"
            )
        return math_mode
    return _validate_math_mode(math_mode)
