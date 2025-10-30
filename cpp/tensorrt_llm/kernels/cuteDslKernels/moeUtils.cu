/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/cuteDslKernels/moeUtils.h"

namespace tensorrt_llm::kernels::cute_dsl
{
template <typename InputType, typename SFType, int32_t kThreadsPerBlock>
__global__ void moePermuteKernel(InputType const* input, InputType* permuted_input, SFType const* input_sf,
    SFType* permuted_sf, int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles,
    int32_t const hidden_size, int32_t const top_k, int32_t const tile_size)
{
    using CopyType = float4;
    int32_t constexpr kElemPerThread = sizeof(CopyType) / sizeof(InputType);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif

    int32_t const num_tokens = num_non_exiting_tiles[0] * tile_size;
    for (int32_t permuted_idx = blockIdx.x; permuted_idx < num_tokens; permuted_idx += gridDim.x)
    {
        int32_t const expanded_idx = permuted_idx_to_expanded_idx[permuted_idx];
        if (expanded_idx < 0)
        {
            continue;
        }
        int32_t const token_idx = expanded_idx / top_k;

        auto const* src_ptr = reinterpret_cast<CopyType const*>(input + hidden_size * token_idx);
        auto* dst_ptr = reinterpret_cast<CopyType*>(permuted_input + hidden_size * permuted_idx);

        for (int32_t offset = threadIdx.x; offset < hidden_size / kElemPerThread; offset += kThreadsPerBlock)
        {
            dst_ptr[offset] = src_ptr[offset];
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename InputType, typename SFType>
void moePermute(InputType const* input, InputType* permuted_input, SFType const* input_sf, SFType* permuted_sf,
    int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles, int32_t const hidden_size,
    int32_t const top_k, int32_t const tile_size, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    static int64_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const blocks = smCount;
    int32_t const threads = kThreadsPerBlock;

    auto kernel = &moePermuteKernel<InputType, SFType, kThreadsPerBlock>;

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel, input, permuted_input, input_sf, permuted_sf, permuted_idx_to_expanded_idx,
        num_non_exiting_tiles, hidden_size, top_k, tile_size);
}

#define INSTANTIATE_MOE_PERMUTE(InputType, SFType)                                                                     \
    template void moePermute<InputType, SFType>(InputType const* input, InputType* permuted_input,                     \
        SFType const* input_sf, SFType* permuted_sf, int32_t const* permuted_idx_to_expanded_idx,                      \
        int32_t const* num_non_exiting_tiles, int32_t const hidden_size, int32_t const top_k, int32_t const tile_size, \
        cudaStream_t stream);

INSTANTIATE_MOE_PERMUTE(half, uint8_t);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_PERMUTE(__nv_bfloat16, uint8_t);
#endif
} // namespace tensorrt_llm::kernels::cute_dsl
