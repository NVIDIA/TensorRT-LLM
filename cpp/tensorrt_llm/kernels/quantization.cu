/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/quantTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/kernels/quantization.h"
#include <float.h>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeQuantization(
    int8_t* dst, T const* src, int64_t const size, float const* scalePtr, cudaStream_t stream, int maxGridSize)
{
    TLLM_CHECK_WITH_INFO(size % 4 == 0, "[ERROR][invokeQuantization] size should be a multiple of 4.\n");

    int numBlocks{static_cast<int>((size + 255) / 256)};
    dim3 grid(std::min(numBlocks, maxGridSize));
    TLLM_CHECK_WITH_INFO(grid.x <= maxGridSize, "[ERROR][invokeQuantization] grid max size is exceeded\n");
    dim3 block(64);
    if (std::is_same_v<T, float>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (float4 const*) src, size / 4, scalePtr);
    }
    else if (std::is_same_v<T, half>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (half2 const*) src, size / 4, scalePtr);
    }
#ifdef ENABLE_BF16
    else if (std::is_same_v<T, __nv_bfloat16>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (__nv_bfloat162 const*) src, size / 4, scalePtr);
    }
#endif
}

template void invokeQuantization<float>(
    int8_t* dst, float const* src, int64_t const size, float const* scalePtr, cudaStream_t stream, int maxGridSize);

template void invokeQuantization<half>(
    int8_t* dst, half const* src, int64_t const size, float const* scalePtr, cudaStream_t stream, int maxGridSize);

#ifdef ENABLE_BF16
template void invokeQuantization<__nv_bfloat16>(int8_t* dst, __nv_bfloat16 const* src, int64_t const size,
    float const* scalePtr, cudaStream_t stream, int maxGridSize);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// Do per-token (row) quantization from fp16/bf16/fp32 to int8/fp8_e4m3.
template <typename T, typename QuantT>
void invokePerTokenQuantization(QuantT* dst, T const* src, int64_t const numRows, int64_t const numCols,
    float const* clampPtr, float* scalePtr, float* sumPtr, QuantMode quantMode, cudaStream_t stream)
{
    // each block is responsible for a single row
    dim3 const block(512);
    dim3 const grid(numRows);

    // The number of elements in the packed uint4 vec.
    static constexpr int NUM_ELTS_PER_VEC = sizeof(uint4) / sizeof(T);
    TLLM_CHECK_WITH_INFO(numCols % NUM_ELTS_PER_VEC == 0, "Not supported.");

    // Cache vectors to smem to avoid reloading.
    size_t const dynamicSmemSz = numCols * sizeof(T);
    // Need to check if smem capacity is enough.
    bool useSmem = true;
    if (dynamicSmemSz >= 48 * 1024)
    {
        cudaError_t res = cudaFuncSetAttribute(
            perTokenQuantization<T, QuantT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamicSmemSz);
        // Fall back to reloading-reversion if smem is not enough.
        useSmem = (res == cudaSuccess);
    }

    // Enable min_scaling_factor if it is fp8 rowwise per-token quantization.
    bool hasFp8MinScaling = quantMode.hasFp8RowWise();
    // Do we use smem ?
    if (useSmem)
    {
        perTokenQuantization<T, QuantT, true><<<grid, block, dynamicSmemSz, stream>>>(
            dst, src, numRows, numCols, clampPtr, scalePtr, sumPtr, hasFp8MinScaling);
    }
    else
    {
        perTokenQuantization<T, QuantT, false>
            <<<grid, block, 0, stream>>>(dst, src, numRows, numCols, clampPtr, scalePtr, sumPtr, hasFp8MinScaling);
    }
}

#define INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(T, QuantT)                                                           \
    template void invokePerTokenQuantization(QuantT* dst, const T* src, const int64_t numRows, const int64_t numCols,  \
        float const* clampPtr, float* scalePtr, float* sumPtr, QuantMode quantMode, cudaStream_t stream)

INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(float, int8_t);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(half, int8_t);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(__nv_bfloat16, int8_t);
#endif

#ifdef ENABLE_FP8
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(float, __nv_fp8_e4m3);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(half, __nv_fp8_e4m3);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(__nv_bfloat16, __nv_fp8_e4m3);
#endif
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP4 Quantization

template <typename T, int SF_VEC_SIZE>
void invokeFP4Quantization(int m, int n, T const* input, float const* SFScale, int64_t* output, int32_t* SFOuput,
    bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream)
{
#ifdef ENABLE_FP8
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
    {
        // Grid, Block size.
        // Each thread converts 16 values.
        dim3 block(std::min(int(n / CVT_FP8_TO_FP4_ELTS_PER_THREAD), 512));
        // Get number of blocks per SM (assume we can fully utilize the SM).
        int const numBlocksPerSM = std::max(1u, 2048u / block.x);
        dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

        // Launch the cvt kernel.
        auto* kernel_instance = useUE8M0 ? &cvt_fp8_to_fp4<SF_VEC_SIZE, true> : &cvt_fp8_to_fp4<SF_VEC_SIZE, false>;
        kernel_instance<<<grid, block, 0, stream>>>(
            m, n, input, SFScale, reinterpret_cast<uint64_t*>(output), reinterpret_cast<uint32_t*>(SFOuput), layout);
    }
    else
#endif
    {
        // Grid, Block size.
        // Each thread converts 8 values.
        dim3 block(std::min(int(n / CVT_FP4_ELTS_PER_THREAD), 512));
        // Get number of blocks per SM (assume we can fully utilize the SM).
        int const numBlocksPerSM = std::max(1u, 2048u / block.x);
        dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

        // Launch the cvt kernel.
        auto* kernel_instance
            = useUE8M0 ? &cvt_fp16_to_fp4<T, SF_VEC_SIZE, true> : &cvt_fp16_to_fp4<T, SF_VEC_SIZE, false>;
        cudaLaunchConfig_t config;
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;
        cudaLaunchKernelEx(&config, kernel_instance, m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(SFOuput), layout);
    }
}

template <typename T, int SF_VEC_SIZE>
void invokeBatchedFP4Quantization(int b, int m, int n, T const* input, float const* SFScale, int64_t* output,
    int32_t* SFOuput, bool useUE8M0, int multiProcessorCount, cudaStream_t stream)
{
#ifdef ENABLE_FP8
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
    {
        // Grid, Block size.
        // Each thread converts 16 values.
        dim3 block(std::min(int(n / CVT_FP8_TO_FP4_ELTS_PER_THREAD), 512));
        // Get number of blocks per SM (assume we can fully utilize the SM).
        int const numBlocksPerSM = std::max(1u, 2048u / block.x);
        dim3 grid(std::min(m, multiProcessorCount * numBlocksPerSM));

        // Launch the cvt kernel.
        auto* kernel_instance
            = useUE8M0 ? &cvt_fp8_to_fp4_3d<SF_VEC_SIZE, true> : &cvt_fp8_to_fp4_3d<SF_VEC_SIZE, false>;
        kernel_instance<<<grid, block, 0, stream>>>(b, m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(SFOuput), FP4QuantizationSFLayout::SWIZZLED);
    }
    else
#endif
    {
        // Grid, Block size.
        // Each thread converts 8 values.
        dim3 block(std::min(int(n / CVT_FP4_ELTS_PER_THREAD), 512));
        // Get number of blocks per SM (assume we can fully utilize the SM).
        int const numBlocksPerSM = std::max(1u, 2048u / block.x);
        dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

        // Launch the cvt kernel.
        auto* kernel_instance
            = useUE8M0 ? &cvt_fp16_to_fp4_3d<T, SF_VEC_SIZE, true> : &cvt_fp16_to_fp4_3d<T, SF_VEC_SIZE, false>;
        cudaLaunchConfig_t config;
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;
        cudaLaunchKernelEx(&config, kernel_instance, b, m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(SFOuput), FP4QuantizationSFLayout::SWIZZLED);
    }
}

__global__ void nvfp4_block_scale_interleave_kernel(
    int numBatches, int numRows, int numCols, uint8_t const* SFIn, uint8_t* SFOutput)
{
    constexpr int SF_VEC_SIZE = 16;
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
        {
            for (int colIdx = threadIdx.x; colIdx < numCols; colIdx += blockDim.x)
            {
                int64_t inOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
                auto sf = SFIn[inOffset];

                std::optional<int> batchIdxOpt = batchIdx;
                std::optional<int> numRowsOpt = numRows;

                // Without batching, the math in get_sf_out_offset is the same as
                // int const numSfTilesK = (numCols + 4 - 1) / 4;
                // int const tileOffset = ((mi / 128) * numSfTilesK + ki / 4) * 512;
                // int const dstIdx = tileOffset + (mi % 32) * 16 + ((mi % 128) / 32) * 4 + ki % 4;
                auto dstIdx = get_sf_out_offset_128x4<SF_VEC_SIZE>(
                    batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols * SF_VEC_SIZE);
                SFOutput[dstIdx] = sf;
            }
        }
    }
}

__global__ void nvfp4_block_scale_interleave_reverse_kernel(
    int numBatches, int numRows, int numCols, uint8_t const* SFIn, uint8_t* SFOutput)
{
    constexpr int SF_VEC_SIZE = 16;
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
        {
            for (int colIdx = threadIdx.x; colIdx < numCols; colIdx += blockDim.x)
            {
                std::optional<int> batchIdxOpt = batchIdx;
                std::optional<int> numRowsOpt = numRows;

                // Get the swizzled input index using the same swizzling pattern
                auto srcIdx = get_sf_out_offset_128x4<SF_VEC_SIZE>(
                    batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols * SF_VEC_SIZE);
                auto sf = SFIn[srcIdx];

                // Output goes to linear layout
                int64_t outOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
                SFOutput[outOffset] = sf;
            }
        }
    }
}

// This is intended for weight loading, so m and n are large, b <= 256
void invokeNVFP4BlockScaleInterleave(
    int b, int m, int n, uint8_t const* SFIn, uint8_t* SFOutput, int multiProcessorCount, cudaStream_t stream)
{
    // Each thread reads 1 int8 value
    dim3 block(std::min(n, 1024));
    // Get number of blocks per SM (assume we can fully utilize the SM).
    int const numBlocksPerSM = std::max(1u, 4096u / block.x);
    dim3 grid(std::min(m, multiProcessorCount * numBlocksPerSM));

    nvfp4_block_scale_interleave_kernel<<<grid, block, 0, stream>>>(b, m, n, SFIn, SFOutput);
}

// This is intended for weight loading, so m and n are large, b <= 256
void invokeNVFP4BlockScaleInterleaveReverse(
    int b, int m, int n, uint8_t const* SFIn, uint8_t* SFOutput, int multiProcessorCount, cudaStream_t stream)
{
    // Each thread reads 1 int8 value
    dim3 block(std::min(n, 1024));
    // Get number of blocks per SM (assume we can fully utilize the SM).
    int const numBlocksPerSM = std::max(1u, 4096u / block.x);
    dim3 grid(std::min(m, multiProcessorCount * numBlocksPerSM));

    nvfp4_block_scale_interleave_reverse_kernel<<<grid, block, 0, stream>>>(b, m, n, SFIn, SFOutput);
}

// Instantiate the function.
template void invokeFP4Quantization<half, 16>(int m, int n, half const* input, float const* SFScale, int64_t* output,
    int32_t* SFOuput, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream);
template void invokeFP4Quantization<half, 32>(int m, int n, half const* input, float const* SFScale, int64_t* output,
    int32_t* SFOuput, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream);
template void invokeBatchedFP4Quantization<half, 16>(int b, int m, int n, half const* input, float const* SFScale,
    int64_t* output, int32_t* SFOuput, bool useUE8M0, int multiProcessorCount, cudaStream_t stream);
template void invokeBatchedFP4Quantization<half, 32>(int b, int m, int n, half const* input, float const* SFScale,
    int64_t* output, int32_t* SFOuput, bool useUE8M0, int multiProcessorCount, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeFP4Quantization<__nv_bfloat16, 16>(int m, int n, __nv_bfloat16 const* input, float const* SFScale,
    int64_t* output, int32_t* SFOuput, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream);
template void invokeFP4Quantization<__nv_bfloat16, 32>(int m, int n, __nv_bfloat16 const* input, float const* SFScale,
    int64_t* output, int32_t* SFOuput, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream);
template void invokeBatchedFP4Quantization<__nv_bfloat16, 16>(int b, int m, int n, __nv_bfloat16 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, int multiProcessorCount,
    cudaStream_t stream);
template void invokeBatchedFP4Quantization<__nv_bfloat16, 32>(int b, int m, int n, __nv_bfloat16 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, int multiProcessorCount,
    cudaStream_t stream);
#endif

#ifdef ENABLE_FP8
template void invokeFP4Quantization<__nv_fp8_e4m3, 16>(int m, int n, __nv_fp8_e4m3 const* input, float const* SFScale,
    int64_t* output, int32_t* SFOuput, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream);
template void invokeFP4Quantization<__nv_fp8_e4m3, 32>(int m, int n, __nv_fp8_e4m3 const* input, float const* SFScale,
    int64_t* output, int32_t* SFOuput, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream);
template void invokeBatchedFP4Quantization<__nv_fp8_e4m3, 16>(int b, int m, int n, __nv_fp8_e4m3 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, int multiProcessorCount,
    cudaStream_t stream);
template void invokeBatchedFP4Quantization<__nv_fp8_e4m3, 32>(int b, int m, int n, __nv_fp8_e4m3 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, int multiProcessorCount,
    cudaStream_t stream);
#endif

} // namespace kernels
} // namespace tensorrt_llm
