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

#include "gpt_oss/gpt_oss_120b__sm_10x__default/csrc/swizzleBlockScales.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

at::Tensor swizzleBlockScales(at::Tensor const& scales)
{
    TORCH_CHECK(
        scales.is_cuda() && scales.is_contiguous(), "swizzle_block_scales: scales must be a contiguous CUDA tensor");
    TORCH_CHECK(scales.scalar_type() == at::kByte, "swizzle_block_scales: scales must be uint8 (E8M0), got ",
        scales.scalar_type());
    TORCH_CHECK(
        scales.dim() == 3, "swizzle_block_scales: scales must be [experts, rows, cols], got ", scales.dim(), " dims");
    int64_t const rows = scales.size(1);
    int64_t const cols = scales.size(2);
    TORCH_CHECK(rows % 128 == 0 && cols % 4 == 0,
        "swizzle_block_scales: needs rows % 128 == 0 and cols % 4 == 0, got rows ", rows, ", cols ", cols);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(scales.data_ptr()) % 4 == 0,
        "swizzle_block_scales: scales must start on a 4-byte boundary");
    c10::cuda::CUDAGuard const deviceGuard{scales.device()};
    at::Tensor out = at::empty_like(scales);
    cudaError_t const status = kernels::invokeSwizzleBlockScales(scales.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(),
        scales.size(0), rows, cols, at::cuda::getCurrentCUDAStream(scales.get_device()));
    TLLM_CHECK_WITH_INFO(status == cudaSuccess, "swizzle_block_scales launch failed: %s", cudaGetErrorString(status));
    return out;
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("swizzle_block_scales(Tensor scales) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("swizzle_block_scales", &TRTLLM_MV2_NS::swizzleBlockScales);
}
