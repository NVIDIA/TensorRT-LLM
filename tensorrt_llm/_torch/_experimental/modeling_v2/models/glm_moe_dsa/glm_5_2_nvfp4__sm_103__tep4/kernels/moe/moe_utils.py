# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Shared utilities for MoE (Mixture of Experts) contiguous grouped GEMM kernels.

Contains common constants, data type definitions, and helper functions used
across all MoE kernel variants (fc1/fc2, swiglu, gather, sfc).
"""

from __future__ import annotations

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

# Note: c_dtype and torch_c_dtype are NOT shared — they differ per kernel variant:
#   - Non-SFC kernels: c_dtype = cutlass.BFloat16, torch_c_dtype = torch.bfloat16
#   - SFC kernels:     c_dtype = cutlass.Float8E4M3FN, torch_c_dtype = torch.uint8
#   - Finalize kernel: c_dtype = cutlass.Float32, torch_c_dtype = torch.float32


def cute_tensor_from_torch(
    torch_tensor: torch.Tensor,
    element_type: type | None,
    leading_dim: int = 0,
    assumed_align: int = 16,
) -> cute.Tensor:
    """Create CUTE tensor from PyTorch tensor via DLPack.

    Wraps from_dlpack with common setup: marking layout as dynamic and
    optionally overriding the element type for narrow precision types.

    Args:
        torch_tensor: Source PyTorch tensor (must be on CUDA).
        element_type: CUTLASS element type (e.g., cutlass.Float8E4M3FN).
            Use None for types that don't need explicit setting (e.g., Float16).
        leading_dim: Dimension index with stride 1 for mark_layout_dynamic.
            Default 0 for K-major tensors, use 1 for N-major output tensor.
        assumed_align: Memory alignment hint in bytes. Default 16.

    Returns:
        CUTE tensor with dynamic layout marked and element type set.
    """
    dlpack_source = _dlpack_source_tensor(torch_tensor, element_type)
    tensor = from_dlpack(dlpack_source, assumed_align=assumed_align).mark_layout_dynamic(
        leading_dim=leading_dim
    )
    if element_type is not None:
        tensor.element_type = element_type
    return tensor


def _dlpack_source_tensor(torch_tensor: torch.Tensor, element_type: type | None) -> torch.Tensor:
    if (
        element_type is not None
        and getattr(element_type, "width", None) in (4, 8)
        and torch_tensor.dtype != torch.uint8
    ):
        # Narrow types can't go through PyTorch DLPack directly: FP4 x2
        # exposes unpacked 4-bit elements, and float8 dtypes are rejected by
        # torch.__dlpack__ outright. Import the raw bytes as uint8, then
        # restore the CUTLASS element type below.
        return torch_tensor.view(torch.uint8)
    return torch_tensor
