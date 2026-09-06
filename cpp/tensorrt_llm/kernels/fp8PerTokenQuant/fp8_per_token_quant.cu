/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/fp8PerTokenQuant/fp8_per_token_quant.cuh"
#include "vectorized_fp8_per_token_quant_kernel.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

template <typename T_IN>
void invokeVectorizedPerTokenFP8Quant(
    void* output, float* scales, T_IN const* input, int hidden_size, int64_t num_tokens, cudaStream_t stream)
{
    dim3 const grid(static_cast<unsigned>(num_tokens));
    dim3 const block(vllm::kPerTokenQuantBlockSize);
    vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<T_IN, c10::Float8_e4m3fn>
        <<<grid, block, 0, stream>>>(static_cast<c10::Float8_e4m3fn*>(output), scales, input,
            /*scale_ub=*/nullptr, hidden_size,
            /*in_row_stride=*/static_cast<int64_t>(hidden_size),
            /*out_row_stride=*/static_cast<int64_t>(hidden_size));
}

template void invokeVectorizedPerTokenFP8Quant<__nv_bfloat16>(
    void*, float*, __nv_bfloat16 const*, int, int64_t, cudaStream_t);
template void invokeVectorizedPerTokenFP8Quant<half>(void*, float*, half const*, int, int64_t, cudaStream_t);
template void invokeVectorizedPerTokenFP8Quant<float>(void*, float*, float const*, int, int64_t, cudaStream_t);

} // namespace kernels

TRTLLM_NAMESPACE_END
