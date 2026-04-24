# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

E8M0_DTYPE_NAME = "torch.float8_e8m0fnu"

__all__ = [
    "e8m0_to_fp32",
    "e8m0_to_uint8",
    "maybe_e8m0_to_fp32",
]


def _get_e8m0_dtype() -> torch.dtype | None:
    return getattr(torch, "float8_e8m0fnu", None)


def _require_e8m0_dtype() -> torch.dtype:
    dtype = _get_e8m0_dtype()
    if dtype is None:
        raise RuntimeError(
            f"{E8M0_DTYPE_NAME} is not available in this PyTorch build; "
            "E8M0 scale tensors cannot be reinterpreted or decoded."
        )
    return dtype


def _check_e8m0_tensor(scale: torch.Tensor) -> None:
    dtype = _require_e8m0_dtype()
    if scale.dtype != dtype:
        raise TypeError(f"expected an {E8M0_DTYPE_NAME} tensor, got {scale.dtype}")


def e8m0_to_uint8(scale: torch.Tensor) -> torch.Tensor:
    """Return raw E8M0 exponent bytes without numeric conversion.

    Args:
        scale: E8M0 scale tensor of any shape.

    Returns:
        A ``torch.uint8`` view over the same raw bytes as ``scale``.

    Raises:
        RuntimeError: If this PyTorch build does not expose E8M0 tensors.
        TypeError: If ``scale`` is not an E8M0 tensor.
    """
    _check_e8m0_tensor(scale)
    return scale.view(torch.uint8)


def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 exponent-only scale values to FP32 powers of two.

    Args:
        scale: E8M0 scale tensor of any shape.

    Returns:
        A ``torch.float32`` tensor with each E8M0 byte decoded as an IEEE-754
        FP32 exponent field.

    Raises:
        RuntimeError: If this PyTorch build does not expose E8M0 tensors.
        TypeError: If ``scale`` is not an E8M0 tensor.
    """
    exp_bits = e8m0_to_uint8(scale).to(torch.int32)
    fp32_bits = exp_bits << 23
    return fp32_bits.view(torch.float32)


def maybe_e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 scales; return non-E8M0 scales as FP32.

    Args:
        scale: E8M0, BF16, FP16, FP32, or other numeric scale tensor of any shape.

    Returns:
        ``scale`` decoded from E8M0 when applicable, otherwise ``scale`` converted
        numerically to ``torch.float32``.
    """
    dtype = _get_e8m0_dtype()
    if dtype is not None and scale.dtype == dtype:
        return e8m0_to_fp32(scale)
    return scale.to(torch.float32)
