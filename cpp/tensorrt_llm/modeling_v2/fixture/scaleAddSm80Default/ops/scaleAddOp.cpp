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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/modeling_v2/fixture/scaleAddSm80Default/kernels/scaleAdd.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

at::Tensor scaleAdd(at::Tensor const& x, double alpha, double beta)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "scale_add: x must be a contiguous CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "scale_add: x must be float32 or bfloat16, got ", x.scalar_type());
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    at::Tensor out = at::empty_like(x);
    auto const dtype
        = x.scalar_type() == at::kFloat ? kernels::ScaleAddDtype::kFloat32 : kernels::ScaleAddDtype::kBfloat16;
    cudaError_t const status = kernels::invokeScaleAdd(x.data_ptr(), out.data_ptr(), x.numel(), dtype,
        static_cast<float>(alpha), static_cast<float>(beta), at::cuda::getCurrentCUDAStream(x.get_device()));
    // TLLM_CHECK_WITH_INFO throws a TllmException, whose code lives in libtensorrt_llm.so: the fixture keeps the
    // dynamic link to the stock library exercised.
    TLLM_CHECK_WITH_INFO(status == cudaSuccess, "scale_add launch failed: %s", cudaGetErrorString(status));
    return out;
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("scale_add(Tensor x, float alpha, float beta) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("scale_add", &TRTLLM_MV2_NS::scaleAdd);
}
