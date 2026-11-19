# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
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
"""Input contract of the per-token AdaLN kernels, importable without CuTe DSL."""

from typing import Optional

import torch

# GPUs the kernels are built for (SM100 and SM103).
SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
# Byte alignment of every operand's data pointer (LDG.128 vector loads).
REQUIRED_ALIGNMENT = 32
# Elements per thread: 8 = one LDG.128 per bf16 operand (D/256 warps per CTA).
# The plain op also accepts 16; the residual op requires 8 (see fused_pertoken_adaln_residual).
DEFAULT_VEC = 8
WARP_SIZE = 32


def supports_hidden_size(hidden_size: int, vec: int = DEFAULT_VEC) -> bool:
    """Whether the kernels take this hidden size: a multiple of 256 and of 32 * vec, at most 8192."""
    return (
        vec in (8, 16)
        and 0 < hidden_size <= 8192
        and hidden_size % 256 == 0
        and hidden_size % (WARP_SIZE * vec) == 0
    )


def is_supported_device(device: Optional[torch.device] = None) -> bool:
    """Whether ``device`` (default: the current CUDA device) is a GPU the kernels are built for."""
    if not torch.cuda.is_available():
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    try:
        return torch.cuda.get_device_capability(device) in SUPPORTED_CAPABILITIES
    except RuntimeError:
        return False


def is_supported(
    hidden_size: int, dtype: torch.dtype, device: Optional[torch.device] = None
) -> bool:
    """Whether the per-token AdaLN kernels can run BF16 rows of ``hidden_size`` on ``device``."""
    return (
        dtype == torch.bfloat16
        and supports_hidden_size(hidden_size)
        and is_supported_device(device)
    )
