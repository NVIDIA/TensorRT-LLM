/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/reduceAddKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace torch_ext
{

torch::Tensor reduceAdd(torch::Tensor const& input, torch::Tensor const& residual)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");

    TORCH_CHECK(input.dim() == 3, "input must be a 3D tensor [num_tokens, topk, hidden_size]");
    TORCH_CHECK(residual.dim() == 2, "residual must be a 2D tensor [num_tokens, hidden_size]");

    int64_t const num_tokens = input.size(0);
    int64_t const topk = input.size(1);
    int64_t const hidden_size = input.size(2);

    TORCH_CHECK(residual.size(0) == num_tokens, "residual.size(0) must equal input.size(0)");
    TORCH_CHECK(residual.size(1) == hidden_size, "residual.size(1) must equal input.size(2)");

    auto const input_dtype = input.scalar_type();
    auto const residual_dtype = residual.scalar_type();
    TORCH_CHECK(input_dtype == residual_dtype, "input and residual must have the same dtype");

    TORCH_CHECK(input_dtype == torch::ScalarType::Half || input_dtype == torch::ScalarType::BFloat16,
        "input must be fp16 or bf16");

    // Allocate output tensor
    auto output = torch::empty({num_tokens, hidden_size}, input.options());

    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    // Launch kernel based on dtype
    if (input_dtype == torch::ScalarType::Half)
    {
        tensorrt_llm::kernels::invokeReduceAdd<half>(reinterpret_cast<half const*>(input.data_ptr()),
            reinterpret_cast<half const*>(residual.data_ptr()), reinterpret_cast<half*>(output.data_ptr()),
            static_cast<int32_t>(num_tokens), static_cast<int32_t>(topk), static_cast<int32_t>(hidden_size), stream);
    }
    else if (input_dtype == torch::ScalarType::BFloat16)
    {
        tensorrt_llm::kernels::invokeReduceAdd<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), static_cast<int32_t>(num_tokens),
            static_cast<int32_t>(topk), static_cast<int32_t>(hidden_size), stream);
    }

    return output;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("reduce_add(Tensor input, Tensor residual) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("reduce_add", &torch_ext::reduceAdd);
}
