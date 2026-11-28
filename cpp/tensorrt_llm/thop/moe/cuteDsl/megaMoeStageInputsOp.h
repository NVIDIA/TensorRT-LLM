/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/moe/cuteDsl/megaMoeStageRoutes.h"

#include <ATen/ATen.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! The quantization a input staging launch runs, checked: x [M, K] contiguous bfloat16 with K a multiple of 16 and
//! 16-byte rows, global_scale one float32, x_fp4 contiguous uint8 [M, K / 2] and x_sf contiguous uint8 [M, K / 16] (the
//! linear layout), all on x's device.
inline kernels::MegaMoeQuantize megaMoeQuantizeOf(
    at::Tensor const& x, at::Tensor const& globalScale, at::Tensor& xFp4, at::Tensor& xSf)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.scalar_type() == at::kBFloat16,
        "x must be a contiguous bfloat16 CUDA [M, K]");
    auto const m = x.size(0);
    auto const k = x.size(1);
    TORCH_CHECK(k % 16 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
        "K must be a multiple of 16 and x 16-byte aligned");
    TORCH_CHECK(
        globalScale.device() == x.device() && globalScale.scalar_type() == at::kFloat && globalScale.numel() == 1,
        "global_scale must be one float32 on x's device");
    TORCH_CHECK(xFp4.device() == x.device() && xFp4.is_contiguous() && xFp4.scalar_type() == at::kByte
            && xFp4.dim() == 2 && xFp4.size(0) == m && xFp4.size(1) == k / 2
            && reinterpret_cast<uintptr_t>(xFp4.data_ptr()) % 4 == 0,
        "x_fp4 must be contiguous 4-byte aligned uint8 [M, K / 2] on x's device");
    TORCH_CHECK(xSf.device() == x.device() && xSf.is_contiguous() && xSf.scalar_type() == at::kByte && xSf.dim() == 2
            && xSf.size(0) == m && xSf.size(1) == k / 16,
        "x_sf must be contiguous uint8 [M, K / 16] on x's device");
    return {static_cast<__nv_bfloat16 const*>(x.data_ptr()), globalScale.data_ptr<float>(),
        static_cast<uint32_t*>(xFp4.data_ptr()), xSf.data_ptr<uint8_t>(), static_cast<int>(m), static_cast<int>(k)};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
