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
"""CuTe distributed operations for AutoDeploy experiments."""

import torch


@torch.library.custom_op(
    "auto_deploy::cute_dist_fused_allreduce_residual_rmsnorm",
    mutates_args=(),
    device_types="cuda",
)
def cute_dist_fused_allreduce_residual_rmsnorm(
    tensor: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental CuTe allreduce + residual + RMSNorm backend.

    This op is intentionally explicit in the AD graph. Selecting it means the
    CuTe POC must be available and inside its narrow bf16 eager execution
    envelope; unsupported cases fail instead of falling back silently.
    """
    from tensorrt_llm._torch.cute_dsl_kernels.allreduce_rmsnorm import (
        fused_allreduce_residual_rmsnorm_bf16,
    )

    return fused_allreduce_residual_rmsnorm_bf16(tensor, residual, norm_weight, eps)


@cute_dist_fused_allreduce_residual_rmsnorm.register_fake
def cute_dist_fused_allreduce_residual_rmsnorm_fake(
    tensor: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(tensor), torch.empty_like(tensor)
