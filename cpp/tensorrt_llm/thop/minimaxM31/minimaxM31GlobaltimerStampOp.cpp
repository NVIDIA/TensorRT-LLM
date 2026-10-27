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

#include "tensorrt_llm/kernels/minimaxM31/globaltimerStamp.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! Store the GPU's %globaltimer (ns) into out[0] on the current stream (see kernels/minimaxM31/globaltimerStamp.h).
//! out: an int64 CUDA tensor of >= 1 element.
void minimaxM31GlobaltimerStamp(torch::Tensor& out)
{
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt64 && out.numel() >= 1,
        "minimax_m31_globaltimer_stamp: out must be an int64 CUDA tensor of >= 1 element");
    at::cuda::CUDAGuard const guard(out.device());
    cudaError_t const e = tensorrt_llm::kernels::minimaxM31::invokeGlobaltimerStamp(
        out.data_ptr<int64_t>(), at::cuda::getCurrentCUDAStream(out.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_globaltimer_stamp launch: ", cudaGetErrorString(e));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m31_globaltimer_stamp(Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_globaltimer_stamp", &tensorrt_llm::torch_ext::minimaxM31GlobaltimerStamp);
}
