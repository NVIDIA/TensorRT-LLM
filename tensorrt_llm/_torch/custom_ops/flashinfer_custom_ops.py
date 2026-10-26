# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Tuple

import torch

from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

if IS_FLASHINFER_AVAILABLE:
    from flashinfer.activation import gelu_tanh_and_mul, silu_and_mul
    from flashinfer.norm import (fused_add_rmsnorm, fused_add_rmsnorm_quant,
                                 gemma_fused_add_rmsnorm, gemma_rmsnorm,
                                 rmsnorm)
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    # Warp this into custom op since flashinfer didn't warp it properly and we want to avoid graph break between mlp layer for user buffer optimization
    @torch.library.custom_op("trtllm::flashinfer_silu_and_mul", mutates_args=())
    def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x, enable_pdl=get_env_enable_pdl())

    @flashinfer_silu_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1].contiguous()

    @torch.library.custom_op("trtllm::flashinfer_gelu_tanh_and_mul",
                             mutates_args=())
    def flashinfer_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
        return gelu_tanh_and_mul(x, enable_pdl=get_env_enable_pdl())

    @flashinfer_gelu_tanh_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1].contiguous()

    # Warp this into custom op since flashinfer provides default value for eps with would produce two different graphs depends on the eps value.
    @torch.library.custom_op("trtllm::flashinfer_rmsnorm", mutates_args=())
    def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                           eps: float) -> torch.Tensor:
        return rmsnorm(input, weight, eps, enable_pdl=get_env_enable_pdl())

    @flashinfer_rmsnorm.register_fake
    def _(input: torch.Tensor, weight: torch.Tensor,
          eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_gemma_rmsnorm",
                             mutates_args=())
    def flashinfer_gemma_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                                 eps: float) -> torch.Tensor:
        return gemma_rmsnorm(input,
                             weight,
                             eps,
                             enable_pdl=get_env_enable_pdl())

    @flashinfer_gemma_rmsnorm.register_fake
    def _(input: torch.Tensor, weight: torch.Tensor,
          eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_fused_add_rmsnorm(input: torch.Tensor,
                                     residual: torch.Tensor,
                                     weight: torch.Tensor, eps: float) -> None:
        fused_add_rmsnorm(input,
                          residual,
                          weight,
                          eps,
                          enable_pdl=get_env_enable_pdl())

    # FlashInfer's fused add-RMSNorm (CuTe DSL path) that also returns the NVFP4 of its BF16 output, the bytes
    # fp4_quantize(input, global_scale, 16, False, False) returns (linear scale factors). Used only behind
    # TRTLLM_MOE_ENTRY_NORM_NVFP4=1 (models/modeling_deepseekv3.py).
    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm_nvfp4",
                             mutates_args=("input", "residual"))
    def flashinfer_fused_add_rmsnorm_nvfp4(
            input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
            global_scale: torch.Tensor,
            eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.utils import device_support_pdl

        from ..cute_dsl_kernels.flashinfer_fused_add_rmsnorm_nvfp4 import \
            fused_add_rmsnorm_nvfp4_cute

        # as flashinfer.norm.fused_add_rmsnorm resolves enable_pdl
        enable_pdl = bool(get_env_enable_pdl()
                          and device_support_pdl(input.device))
        return fused_add_rmsnorm_nvfp4_cute(input,
                                            residual,
                                            weight,
                                            global_scale,
                                            eps,
                                            enable_pdl=enable_pdl)

    @flashinfer_fused_add_rmsnorm_nvfp4.register_fake
    def _(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
          global_scale: torch.Tensor,
          eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        m, k = input.shape
        return (input.new_empty((m, k // 2), dtype=torch.uint8),
                input.new_empty((m * (k // 16), ), dtype=torch.uint8))

    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm_quant",
                             mutates_args=("out", "residual"))
    def flashinfer_fused_add_rmsnorm_quant(out: torch.Tensor,
                                           input: torch.Tensor,
                                           residual: torch.Tensor,
                                           weight: torch.Tensor,
                                           scale: torch.Tensor,
                                           eps: float) -> None:
        fused_add_rmsnorm_quant(out,
                                input,
                                residual,
                                weight,
                                scale,
                                eps,
                                enable_pdl=get_env_enable_pdl())

    @flashinfer_fused_add_rmsnorm_quant.register_fake
    def _(out: torch.Tensor, input: torch.Tensor, residual: torch.Tensor,
          weight: torch.Tensor, scale: torch.Tensor, eps: float) -> None:
        pass

    @torch.library.custom_op("trtllm::flashinfer_gemma_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_gemma_fused_add_rmsnorm(input: torch.Tensor,
                                           residual: torch.Tensor,
                                           weight: torch.Tensor,
                                           eps: float) -> None:
        gemma_fused_add_rmsnorm(input,
                                residual,
                                weight,
                                eps,
                                enable_pdl=get_env_enable_pdl())

    @torch.library.custom_op(
        "trtllm::flashinfer_apply_rope_with_cos_sin_cache_inplace",
        mutates_args=("query", "key"))
    def flashinfer_apply_rope_with_cos_sin_cache_inplace(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ) -> None:
        apply_rope_with_cos_sin_cache_inplace(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
        )

    @flashinfer_apply_rope_with_cos_sin_cache_inplace.register_fake
    def _(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool = True,
    ):
        return
