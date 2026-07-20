# Adapted from SGLang's breakable CUDA graph implementation.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings import runtime as rt


def _cuda_get_error_string(error: rt.cudaError_t) -> str:
    result, message = rt.cudaGetErrorString(error)
    if result != rt.cudaError_t.cudaSuccess:
        return "<unknown>"
    if isinstance(message, bytes):
        return message.decode("utf-8", "replace")
    return str(message)


def check_cuda_errors(result):
    """Raise a Python exception for a failed cuda-python runtime call."""
    if result[0] != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error {int(result[0])}({_cuda_get_error_string(result[0])})")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]
