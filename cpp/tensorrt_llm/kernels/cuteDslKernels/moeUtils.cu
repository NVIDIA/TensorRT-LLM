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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/cuteDslKernels/moeUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/kernels/quantization.h"

#include <cuda_fp4.h>
#include <cute/numeric/numeric_types.hpp>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cute_dsl
{
namespace
{
using ElemCopyType = uint4;
using SFCopyType = uint32_t;
using ActivationType = tensorrt_llm::kernels::cutlass_kernels::ActivationType;

template <typename T>
auto constexpr bitsPerElem()
{
#ifdef ENABLE_FP4
    return std::is_same_v<T, __nv_fp4_e2m1> ? 4 : cute::sizeof_bits_v<T>;
#else
    return cute::sizeof_bits_v<T>;
#endif
}

template <typename T>
auto constexpr elemPerCopy()
{
    return bitsPerElem<ElemCopyType>() / bitsPerElem<T>();
}

template <typename T>
auto constexpr sfElemPerCopy()
{
    return bitsPerElem<SFCopyType>() / bitsPerElem<T>();
}
} // namespace

template <typename InputType, typename SFType, int32_t kSFVecSize, int32_t kThreadsPerBlock>
__global__ void moePermuteKernel(InputType const* input, InputType* permuted_output, SFType const* input_sf,
    SFType* permuted_sf, int32_t const* tile_idx_to_mn_limit, int32_t const* permuted_idx_to_expanded_idx,
    int32_t const* num_non_exiting_tiles, int32_t const hidden_size, int32_t const top_k, int32_t const tile_size)
{
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    int32_t constexpr kSFElemPerCopy = sfElemPerCopy<SFType>();
    // Need int64_t to prevent overflow when computing pointer offsets.
    int64_t const kCopyPerToken = hidden_size / kElemPerCopy;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int32_t const num_tokens = num_non_exiting_tiles[0] * tile_size;
    for (int32_t permuted_idx = blockIdx.x; permuted_idx < num_tokens; permuted_idx += gridDim.x)
    {
        int32_t const tile_idx = permuted_idx / tile_size;
        if (permuted_idx >= tile_idx_to_mn_limit[tile_idx])
        {
            continue;
        }
        int32_t const expanded_idx = permuted_idx_to_expanded_idx[permuted_idx];
        int32_t const token_idx = expanded_idx / top_k;

        auto const* src_ptr = reinterpret_cast<ElemCopyType const*>(input) + token_idx * kCopyPerToken;
        auto* dst_ptr = reinterpret_cast<ElemCopyType*>(permuted_output) + permuted_idx * kCopyPerToken;
        for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock)
        {
            dst_ptr[i] = src_ptr[i];
        }

#ifdef ENABLE_FP4
        if constexpr (std::is_same_v<InputType, __nv_fp4_e2m1>)
        {
            int32_t const sf_hidden_size = hidden_size / kSFVecSize;
            int64_t const kSFCopyPerToken = sf_hidden_size / kSFElemPerCopy;
            auto const* sf_src_ptr = reinterpret_cast<SFCopyType const*>(input_sf);
            auto* sf_dst_ptr = reinterpret_cast<SFCopyType*>(permuted_sf);
            for (int32_t i = threadIdx.x; i < kSFCopyPerToken; i += kThreadsPerBlock)
            {
                // input_sf is not swizzled, while permuted_sf is swizzled.
                int64_t const src_offset = token_idx * kSFCopyPerToken + i;
                int64_t const dst_offset = get_sf_out_offset_128x4(/* batchIdx= */ std::nullopt, permuted_idx,
                                               i * kSFElemPerCopy, /* numRows= */ std::nullopt, sf_hidden_size)
                    / kSFElemPerCopy;

                sf_dst_ptr[dst_offset] = sf_src_ptr[src_offset];
            }
        }
#endif
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType, typename SFType>
void moePermute(InputType const* input, InputType* permuted_output, SFType const* input_sf, SFType* permuted_sf,
    int32_t const* tile_idx_to_mn_limit, int32_t const* permuted_idx_to_expanded_idx,
    int32_t const* num_non_exiting_tiles, int32_t const max_num_permuted_tokens, int32_t const hidden_size,
    int32_t const top_k, int32_t const tile_size, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int32_t constexpr kSFVecSize = 16;
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    TLLM_CHECK_WITH_INFO(hidden_size % kElemPerCopy == 0, "hidden_size must be divisible by %d.", kElemPerCopy);

#ifdef ENABLE_FP4
    if constexpr (std::is_same_v<InputType, __nv_fp4_e2m1>)
    {
        int32_t constexpr kSFMAlignment = 128;
        int32_t constexpr kSFKAlignment = 4;
        int32_t constexpr kSFElemPerCopy = sfElemPerCopy<SFType>();
        static_assert(kSFElemPerCopy == kSFKAlignment);
        TLLM_CHECK_WITH_INFO(max_num_permuted_tokens % kSFMAlignment == 0,
            "max_num_permuted_tokens must be divisible by %d.", kSFMAlignment);
        TLLM_CHECK_WITH_INFO(hidden_size % (kSFVecSize * kSFKAlignment) == 0, "hidden_size must be divisible by %d.",
            kSFVecSize * kSFKAlignment);
        TLLM_CHECK_WITH_INFO(input_sf != nullptr, "input_sf is required for NVFP4.");
        TLLM_CHECK_WITH_INFO(permuted_sf != nullptr, "permuted_sf is required for NVFP4.");
    }
#endif

    auto kernel = &moePermuteKernel<InputType, SFType, kSFVecSize, kThreadsPerBlock>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, max_num_permuted_tokens);
    int32_t const threads = kThreadsPerBlock;

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
    cudaLaunchKernelEx(&config, kernel, input, permuted_output, input_sf, permuted_sf, tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx, num_non_exiting_tiles, hidden_size, top_k, tile_size);
}

#define INSTANTIATE_MOE_PERMUTE(InputType, SFType)                                                                     \
    template void moePermute<InputType, SFType>(InputType const* input, InputType* permuted_output,                    \
        SFType const* input_sf, SFType* permuted_sf, int32_t const* tile_idx_to_mn_limit,                              \
        int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles,                             \
        int32_t const max_num_permuted_tokens, int32_t const hidden_size, int32_t const top_k,                         \
        int32_t const tile_size, cudaStream_t stream)

INSTANTIATE_MOE_PERMUTE(half, uint8_t);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_PERMUTE(__nv_bfloat16, uint8_t);
#endif
#ifdef ENABLE_FP8
INSTANTIATE_MOE_PERMUTE(__nv_fp8_e4m3, uint8_t);
#endif
#ifdef ENABLE_FP4
INSTANTIATE_MOE_PERMUTE(__nv_fp4_e2m1, uint8_t);
#endif
#undef INSTANTIATE_MOE_PERMUTE

template <typename InputType, typename TopKScaleType, int32_t kThreadsPerBlock>
__global__ void moeUnpermuteKernel(InputType const* permuted_input, InputType* output,
    int32_t const* expanded_idx_to_permuted_idx, TopKScaleType const* topk_scales, int32_t const hidden_size,
    int32_t const top_k)
{
    using AccumType = float;
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    // Need int64_t to prevent overflow when computing pointer offsets.
    int64_t const kCopyPerToken = hidden_size / kElemPerCopy;
    InputType rmem[kElemPerCopy];
    AccumType rmemAccum[kElemPerCopy];

    int32_t const token_idx = blockIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    auto* dst_ptr = reinterpret_cast<ElemCopyType*>(output) + token_idx * kCopyPerToken;
    for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock)
    {
#pragma unroll
        for (int32_t j = 0; j < kElemPerCopy; j++)
        {
            rmemAccum[j] = 0;
        }
        for (int32_t k = 0; k < top_k; k++)
        {
            int32_t const permuted_idx = expanded_idx_to_permuted_idx[token_idx * top_k + k];
            if (permuted_idx < 0)
            {
                continue;
            }
            auto const* src_ptr = reinterpret_cast<ElemCopyType const*>(permuted_input) + permuted_idx * kCopyPerToken;
            *reinterpret_cast<ElemCopyType*>(rmem) = src_ptr[i];
            TopKScaleType const scale = topk_scales[token_idx * top_k + k];

#pragma unroll
            for (int32_t j = 0; j < kElemPerCopy; j++)
            {
                rmemAccum[j] += static_cast<AccumType>(rmem[j]) * static_cast<AccumType>(scale);
            }
        }
#pragma unroll
        for (int32_t j = 0; j < kElemPerCopy; j++)
        {
            rmem[j] = static_cast<InputType>(rmemAccum[j]);
        }
        dst_ptr[i] = *reinterpret_cast<ElemCopyType*>(rmem);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType, typename TopKScaleType>
void moeUnpermute(InputType const* permuted_input, InputType* output, int32_t const* expanded_idx_to_permuted_idx,
    TopKScaleType const* topk_scales, int32_t const num_tokens, int32_t const hidden_size, int32_t const top_k,
    cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    TLLM_CHECK_WITH_INFO(hidden_size % kElemPerCopy == 0, "hidden_size must be divisible by %d.", kElemPerCopy);

    int32_t const blocks = num_tokens;
    int32_t const threads = kThreadsPerBlock;

    auto kernel = &moeUnpermuteKernel<InputType, TopKScaleType, kThreadsPerBlock>;

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
    cudaLaunchKernelEx(
        &config, kernel, permuted_input, output, expanded_idx_to_permuted_idx, topk_scales, hidden_size, top_k);
}

#define INSTANTIATE_MOE_UNPERMUTE(InputType, TopKScaleType)                                                            \
    template void moeUnpermute<InputType>(InputType const* permuted_input, InputType* output,                          \
        int32_t const* expanded_idx_to_permuted_idx, TopKScaleType const* topk_scales, int32_t const num_tokens,       \
        int32_t const hidden_size, int32_t const top_k, cudaStream_t stream)

INSTANTIATE_MOE_UNPERMUTE(half, float);
INSTANTIATE_MOE_UNPERMUTE(half, half);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_UNPERMUTE(__nv_bfloat16, float);
INSTANTIATE_MOE_UNPERMUTE(__nv_bfloat16, __nv_bfloat16);
#endif
#undef INSTANTIATE_MOE_UNPERMUTE

template <typename InputType, int32_t kThreadsPerBlock>
__global__ void moeOutputMemsetKernel(InputType* input, int32_t const* tile_idx_to_mn_limit,
    int32_t const* expanded_idx_to_permuted_idx, int32_t const* permuted_idx_to_expanded_idx,
    int32_t const* num_non_exiting_tiles, int32_t const hidden_size, int32_t const top_k, int32_t const tile_size)
{
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    int64_t const kCopyPerToken = hidden_size / kElemPerCopy;

    InputType rmem[kElemPerCopy];
#pragma unroll
    for (int32_t j = 0; j < kElemPerCopy; j++)
    {
        rmem[j] = 0;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int32_t const num_tokens = num_non_exiting_tiles[0] * tile_size;
    for (int32_t permuted_idx = blockIdx.x; permuted_idx < num_tokens; permuted_idx += gridDim.x)
    {
        int32_t const tile_idx = permuted_idx / tile_size;
        if (permuted_idx >= tile_idx_to_mn_limit[tile_idx])
        {
            continue;
        }
        int32_t const expanded_idx = permuted_idx_to_expanded_idx[permuted_idx];
        int32_t const token_idx = expanded_idx / top_k;
        int32_t const topk_idx = expanded_idx % top_k;

        bool is_first_in_topk = true;
        for (int32_t k = 0; k < topk_idx; k++)
        {
            if (expanded_idx_to_permuted_idx[token_idx * top_k + k] >= 0)
            {
                is_first_in_topk = false;
                break;
            }
        }
        if (!is_first_in_topk)
        {
            continue;
        }

        auto* dst_ptr = reinterpret_cast<ElemCopyType*>(input) + token_idx * kCopyPerToken;
        for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock)
        {
            dst_ptr[i] = *reinterpret_cast<ElemCopyType*>(rmem);
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType>
void moeOutputMemset(InputType* input, int32_t const* tile_idx_to_mn_limit, int32_t const* expanded_idx_to_permuted_idx,
    int32_t const* permuted_idx_to_expanded_idx, int32_t const* num_non_exiting_tiles,
    int32_t const max_num_permuted_tokens, int32_t const hidden_size, int32_t const top_k, int32_t const tile_size,
    cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    TLLM_CHECK_WITH_INFO(hidden_size % kElemPerCopy == 0, "hidden_size must be divisible by %d.", kElemPerCopy);

    auto kernel = &moeOutputMemsetKernel<InputType, kThreadsPerBlock>;
    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, max_num_permuted_tokens);
    int32_t const threads = kThreadsPerBlock;

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
    cudaLaunchKernelEx(&config, kernel, input, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx, num_non_exiting_tiles, hidden_size, top_k, tile_size);
}

#define INSTANTIATE_MOE_OUTPUT_MEMSET(InputType)                                                                       \
    template void moeOutputMemset<InputType>(InputType * input, int32_t const* tile_idx_to_mn_limit,                   \
        int32_t const* expanded_idx_to_permuted_idx, int32_t const* permuted_idx_to_expanded_idx,                      \
        int32_t const* num_non_exiting_tiles, int32_t const max_num_permuted_tokens, int32_t const hidden_size,        \
        int32_t const top_k, int32_t const tile_size, cudaStream_t stream)

INSTANTIATE_MOE_OUTPUT_MEMSET(half);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_OUTPUT_MEMSET(__nv_bfloat16);
#endif
#undef INSTANTIATE_MOE_OUTPUT_MEMSET

template <typename InputType, typename OutputType, typename SFType, int32_t kSFVecSize, typename ActFn,
    int32_t kThreadsPerBlock>
__global__ void moeActivationKernel(InputType const* input, OutputType* output, float const* global_sf,
    SFType* output_sf, int32_t const* tile_idx_to_mn_limit, int32_t const* num_non_exiting_tiles,
    int32_t const interm_size, int32_t const tile_size)
{
    using ComputeType = float;
#ifdef ENABLE_FP4
    using ElemOutputCopyType = std::conditional_t<std::is_same_v<OutputType, __nv_fp4_e2m1>, uint32_t, ElemCopyType>;
#else
    using ElemOutputCopyType = ElemCopyType;
#endif
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    // Need int64_t to prevent overflow when computing pointer offsets.
    int64_t const kCopyPerToken = interm_size / kElemPerCopy;
    InputType rmem[kElemPerCopy];
    InputType rmemGate[kElemPerCopy];
    ActFn act{};

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    float global_sf_val = global_sf == nullptr ? 1.0f : global_sf[0];

    int32_t const num_tokens = num_non_exiting_tiles[0] * tile_size;
    for (int32_t permuted_idx = blockIdx.x; permuted_idx < num_tokens; permuted_idx += gridDim.x)
    {
        int32_t const tile_idx = permuted_idx / tile_size;
        if (permuted_idx >= tile_idx_to_mn_limit[tile_idx])
        {
            continue;
        }
        auto const* src_ptr
            = reinterpret_cast<ElemCopyType const*>(input) + permuted_idx * kCopyPerToken * (ActFn::IS_GLU ? 2 : 1);
        auto* dst_ptr = reinterpret_cast<ElemOutputCopyType*>(output) + permuted_idx * kCopyPerToken;
        for (int32_t i = threadIdx.x; i < kCopyPerToken; i += kThreadsPerBlock)
        {
            *reinterpret_cast<ElemCopyType*>(rmem) = src_ptr[i];
            if constexpr (ActFn::IS_GLU)
            {
                *reinterpret_cast<ElemCopyType*>(rmemGate) = src_ptr[i + kCopyPerToken];
#pragma unroll
                for (int32_t j = 0; j < kElemPerCopy; j++)
                {
                    rmem[j] = static_cast<InputType>(
                        act(static_cast<ComputeType>(rmemGate[j]), static_cast<ComputeType>(rmem[j])));
                }
            }
            else
            {
#pragma unroll
                for (int32_t j = 0; j < kElemPerCopy; j++)
                {
                    rmem[j] = static_cast<InputType>(act(static_cast<ComputeType>(rmem[j])));
                }
            }

#ifdef ENABLE_FP4
            if constexpr (std::is_same_v<OutputType, __nv_fp4_e2m1>)
            {
                auto* sf_dst_ptr = cvt_quant_get_sf_out_offset<SFType, kSFVecSize / kElemPerCopy>(
                    /* batchIdx= */ std::nullopt, permuted_idx, i, /*numRows=*/std::nullopt, interm_size / kSFVecSize,
                    output_sf, QuantizationSFLayout::SWIZZLED);
                dst_ptr[i] = cvt_warp_fp16_to_fp4<InputType, kSFVecSize, false>(
                    *reinterpret_cast<PackedVec<InputType>*>(rmem), global_sf_val, sf_dst_ptr);
            }
            else
#endif
            {
                dst_ptr[i] = *reinterpret_cast<ElemCopyType*>(rmem);
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputType, typename OutputType, typename SFType>
void moeActivation(InputType const* input, OutputType* output, float const* global_sf, SFType* output_sf,
    int32_t const* tile_idx_to_mn_limit, int32_t const* num_non_exiting_tiles,
    cutlass_kernels::ActivationParams activation_params, int32_t const max_num_permuted_tokens,
    int32_t const interm_size, int32_t const tile_size, cudaStream_t stream)
{
    int32_t constexpr kThreadsPerBlock = 256;
    int32_t constexpr kSFVecSize = 16;
    int32_t constexpr kElemPerCopy = elemPerCopy<InputType>();
    TLLM_CHECK_WITH_INFO(interm_size % kElemPerCopy == 0, "interm_size must be divisible by %d.", kElemPerCopy);

#ifdef ENABLE_FP4
    if constexpr (std::is_same_v<InputType, __nv_fp4_e2m1>)
    {
        int32_t constexpr kSFMAlignment = 128;
        int32_t constexpr kSFKAlignment = 4;
        TLLM_CHECK_WITH_INFO(max_num_permuted_tokens % kSFMAlignment == 0,
            "max_num_permuted_tokens must be divisible by %d.", kSFMAlignment);
        TLLM_CHECK_WITH_INFO(interm_size % (kSFVecSize * kSFKAlignment) == 0, "interm_size must be divisible by %d.",
            kSFVecSize * kSFKAlignment);
        TLLM_CHECK_WITH_INFO(global_sf != nullptr, "global_sf is required for NVFP4.");
        TLLM_CHECK_WITH_INFO(output_sf != nullptr, "output_sf is required for NVFP4.");
    }
#endif

    auto get_act_kernel = [](ActivationType activation_type) -> void (*)(InputType const* input, OutputType* output,
                                                                 float const* global_sf, SFType* output_sf,
                                                                 int32_t const* tile_idx_to_mn_limit,
                                                                 int32_t const* num_non_exiting_tiles,
                                                                 int32_t const interm_size, int32_t const tile_size)
    {
        switch (activation_type)
        {
        case ActivationType::Identity:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::IdentityAdaptor<cutlass::epilogue::thread::Identity>, kThreadsPerBlock>;
        case ActivationType::Gelu:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::IdentityAdaptor<cutlass::epilogue::thread::GELU>, kThreadsPerBlock>;
        case ActivationType::Geglu:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::GLUAdaptor<cutlass::epilogue::thread::GELU>, kThreadsPerBlock>;
        case ActivationType::Relu:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::IdentityAdaptor<cutlass::epilogue::thread::ReLu>, kThreadsPerBlock>;
        case ActivationType::Silu:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::IdentityAdaptor<cutlass::epilogue::thread::SiLu>, kThreadsPerBlock>;
        case ActivationType::Swiglu:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize,
                cutlass_kernels::GLUAdaptor<cutlass::epilogue::thread::SiLu>, kThreadsPerBlock>;
        case ActivationType::SwigluBias:
            return &moeActivationKernel<InputType, OutputType, SFType, kSFVecSize, cutlass_kernels::SwigluBiasAdaptor,
                kThreadsPerBlock>;
        case ActivationType::Relu2:
            // Unsupported activation type
            break;
        }
        TLLM_CHECK_WITH_INFO(false, "Unsupported activation type: %d", int(activation_type));
        return nullptr;
    };
    auto kernel = get_act_kernel(activation_params.activation_type);

    static int32_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
    int32_t const maxBlocksPerSM = tensorrt_llm::common::getMaxActiveBlocksPerSM(kernel, kThreadsPerBlock, 0);
    int32_t const blocks = std::min(smCount * maxBlocksPerSM, max_num_permuted_tokens);
    int32_t const threads = kThreadsPerBlock;

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
    cudaLaunchKernelEx(&config, kernel, input, output, global_sf, output_sf, tile_idx_to_mn_limit,
        num_non_exiting_tiles, interm_size, tile_size);
}

#define INSTANTIATE_MOE_ACTIVATION(InputType, OutputType, SFType)                                                      \
    template void moeActivation<InputType, OutputType, SFType>(InputType const* input, OutputType* output,             \
        float const* global_sf, SFType* output_sf, int32_t const* tile_idx_to_mn_limit,                                \
        int32_t const* num_non_exiting_tiles, cutlass_kernels::ActivationParams activation_params,                     \
        int32_t const max_num_permuted_tokens, int32_t const interm_size, int32_t const tile_size,                     \
        cudaStream_t stream)

INSTANTIATE_MOE_ACTIVATION(half, half, uint8_t);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_ACTIVATION(__nv_bfloat16, __nv_bfloat16, uint8_t);
#endif
#ifdef ENABLE_FP4
INSTANTIATE_MOE_ACTIVATION(half, __nv_fp4_e2m1, uint8_t);
#ifdef ENABLE_BF16
INSTANTIATE_MOE_ACTIVATION(__nv_bfloat16, __nv_fp4_e2m1, uint8_t);
#endif
#endif
#undef INSTANTIATE_MOE_ACTIVATION

} // namespace kernels::cute_dsl

TRTLLM_NAMESPACE_END
