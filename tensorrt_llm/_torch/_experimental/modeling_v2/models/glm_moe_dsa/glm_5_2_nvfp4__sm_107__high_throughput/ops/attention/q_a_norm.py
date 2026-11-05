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
"""Generation q_a normalization with an early PDL-dependent launch."""

from functools import cache

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl
from tensorrt_llm._torch.modules.rms_norm import RMSNorm


@cache
def _uses_cute_rmsnorm(device_index: int) -> bool:
    from flashinfer.norm import _use_cuda_norm

    return not _use_cuda_norm(torch.device("cuda", device_index))


def is_q_a_norm_early_supported(input: torch.Tensor, norm: RMSNorm) -> bool:
    """Whether plain BF16 q_a[M,2048] matches the owned FlashInfer-derived kernel."""
    return (
        type(norm) is RMSNorm
        and IS_FLASHINFER_AVAILABLE
        and get_env_enable_pdl()
        and not norm.use_gemma
        and not norm.use_cuda_tile
        and not norm.is_nvfp4
        and not norm.return_hp_output
        and input.is_cuda
        and input.dtype == torch.bfloat16
        and input.ndim == 2
        and input.shape[0] > 0
        and input.shape[1] == 2048
        and input.stride(1) == 1
        and input.stride(0) % 8 == 0
        and input.data_ptr() % 16 == 0
        and norm.weight.device == input.device
        and norm.weight.dtype == input.dtype
        and norm.weight.shape == (2048,)
        and norm.weight.is_contiguous()
        and norm.weight.data_ptr() % 16 == 0
        and not torch.is_grad_enabled()
        and _uses_cute_rmsnorm(input.device.index)
    )


@torch.library.custom_op(
    "trtllm::glm_rubin_high_throughput_q_a_norm_early", mutates_args=(), device_types="cuda"
)
def q_a_norm_early(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize BF16 q_a[M,2048] and release PDL dependents before waiting on its producer."""
    from ...kernels.gemm.rmsnorm_early_trigger import rmsnorm_early_trigger

    output = torch.empty_like(input, memory_format=torch.contiguous_format)
    rmsnorm_early_trigger(input, weight, output, eps, enable_pdl=get_env_enable_pdl())
    return output


@q_a_norm_early.register_fake
def _fake_q_a_norm_early(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(input, memory_format=torch.contiguous_format)
