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

# Adapted from https://github.com/NVIDIA/cutile-python/blob/main/test/bench_rms_norm.py

import torch

from ..cuda_tile_utils import IS_CUDA_TILE_AVAILABLE

if IS_CUDA_TILE_AVAILABLE:
    import cuda.tile as ct

    from ..cuda_tile_kernels import (
        rms_norm_fuse_residual_kernel,
        rms_norm_fuse_residual_kernel_gather,
        rms_norm_fuse_residual_kernel_static_persistent,
        rms_norm_kernel,
        rms_norm_kernel_gather,
        rms_norm_kernel_static_persistent,
    )
    from ..cuda_tile_utils import ceil_div, next_power_of_2

    @torch.library.custom_op("trtllm::cuda_tile_rms_norm", mutates_args=())
    def cuda_tile_rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
        use_gemma: bool,
    ) -> torch.Tensor:
        x = x.contiguous()
        weight = weight.contiguous()

        # Allocate output tensor
        y = torch.empty_like(x)
        M, N = x.shape

        if static_persistent:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            TILE_SIZE_M = 4  # Default value, could be made configurable
            TILE_SIZE_N = next_power_of_2(N)

            # Other tile sizes are more optimal when other dimension is too large/too small
            if TILE_SIZE_N <= 1024:
                TILE_SIZE_M = 16
            elif TILE_SIZE_N >= 16384:
                TILE_SIZE_M = 2

            grid_size = min(
                NUM_SMS,
                ceil_div(M, TILE_SIZE_M) * ceil_div(N, TILE_SIZE_N),
            )
            grid = (grid_size,)
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                rms_norm_kernel_static_persistent,
                (
                    x,
                    y,
                    weight,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    eps,
                    use_gemma,
                ),
            )
        else:
            # Standard RMSNorm kernel
            rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
            MAX_FUSED_SIZE = 2048 // x.element_size()
            TILE_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(N))
            grid = (M,)
            kernel = rms_norm_kernel_gather if gather else rms_norm_kernel
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                kernel,
                (
                    x,
                    weight,
                    y,
                    rstd,
                    N,
                    eps,
                    TILE_SIZE,
                    use_gemma,
                ),
            )
        return y.view(*x.shape)

    @cuda_tile_rms_norm.register_fake
    def _(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
        use_gemma: bool,
    ) -> torch.Tensor:
        return torch.empty_like(x.contiguous())

    @torch.library.custom_op(
        "trtllm::cuda_tile_rms_norm_fuse_residual_",
        mutates_args=("x", "residual"),
    )
    def cuda_tile_rms_norm_fuse_residual_(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        static_persistent: bool,
        gather: bool,
        use_gemma: bool,
    ) -> None:
        assert x.is_contiguous(), "x must be contiguous for in-place operation"
        assert residual.is_contiguous(), "residual must be contiguous for in-place operation"
        weight = weight.contiguous()

        M, N = x.shape

        if static_persistent:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            TILE_SIZE_M = 4  # Default value, could be made configurable
            TILE_SIZE_N = next_power_of_2(N)

            # Other tile sizes are more optimal when other dimension is too large/too small
            if TILE_SIZE_N <= 1024:
                TILE_SIZE_M = 16
            elif TILE_SIZE_N >= 16384:
                TILE_SIZE_M = 2

            grid_size = min(
                NUM_SMS,
                ceil_div(M, TILE_SIZE_M) * ceil_div(N, TILE_SIZE_N),
            )
            grid = (grid_size,)
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                rms_norm_fuse_residual_kernel_static_persistent,
                (
                    x,
                    residual,
                    weight,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    eps,
                    use_gemma,
                ),
            )
        else:
            # Standard RMSNorm kernel with residual fusion
            rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
            MAX_FUSED_SIZE = 2048 // x.element_size()
            TILE_SIZE = min(MAX_FUSED_SIZE, next_power_of_2(N))
            grid = (M,)
            kernel = (
                rms_norm_fuse_residual_kernel_gather if gather else rms_norm_fuse_residual_kernel
            )
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                kernel,
                (
                    x,
                    residual,
                    weight,
                    rstd,
                    N,
                    eps,
                    TILE_SIZE,
                    use_gemma,
                ),
            )
