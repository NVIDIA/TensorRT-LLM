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
void invokeFP4Quantization(int b, int m, int n, T const* input, float const* SFScale, int64_t* output, int32_t* SFOuput,
    bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream)
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
        auto* kernel_instance = useUE8M0
            ? &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T, SF_VEC_SIZE, true>
            : &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T, SF_VEC_SIZE, false>;
        kernel_instance<<<grid, block, 0, stream>>>(b, m, n, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(SFOuput), layout);
    }
    else
#endif
    {
        // Grid, Block size.
        // Each thread converts 8 values.
        dim3 block(std::min(int(n / CVT_ELTS_PER_THREAD), 512));
        // Get number of blocks per SM (assume we can fully utilize the SM).
        int const numBlocksPerSM = std::max(1u, 2048u / block.x);
        dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

        // Launch the cvt kernel.
        auto* kernel_instance = useUE8M0
            ? &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, true>
            : &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, false>;
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
        cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(SFOuput), layout);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// MXFP8 Quantization

template <typename T>
void invokeMxFP8Quantization(int b, int m, int n, int padded_n, T const* input, int64_t* output, int32_t* SFOuput,
    QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream)
{
    // Fixed SF_VEC_SIZE as 32
    static constexpr int SF_VEC_SIZE = 32;

    // Grid, Block size.
    // Each thread converts 8 values.
    dim3 block(std::min(int(padded_n / CVT_ELTS_PER_THREAD), 512));
    // Get number of blocks per SM (assume we can fully utilize the SM).
    int const numBlocksPerSM = std::max(1u, 2048u / block.x);
    dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

    // Launch the cvt kernel.
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
    cudaLaunchKernelEx(&config,
        quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_MXFP8, T, SF_VEC_SIZE, true>, b, m, n, padded_n,
        input, nullptr, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput), layout);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void block_scale_interleave_kernel(int numBatches, int numRows, int numRowsPadded, int numCols,
    int numColsPadded, uint8_t const* SFIn, uint8_t* SFOutput)
{
    for (int rowIdx = blockIdx.x; rowIdx < numRowsPadded; rowIdx += gridDim.x)
    {
        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
        {
            for (int colIdx = threadIdx.x; colIdx < numColsPadded; colIdx += blockDim.x)
            {
                uint8_t sf = 0;
                if (rowIdx < numRows && colIdx < numCols)
                {
                    int64_t inOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
                    sf = SFIn[inOffset];
                }

                std::optional<int> batchIdxOpt = batchIdx;
                std::optional<int> numRowsOpt = numRows;

                // Without batching, the math in get_sf_out_offset is the same as
                // int const numSfTilesK = (numCols + 4 - 1) / 4;
                // int const tileOffset = ((mi / 128) * numSfTilesK + ki / 4) * 512;
                // int const dstIdx = tileOffset + (mi % 32) * 16 + ((mi % 128) / 32) * 4 + ki % 4;
                auto dstIdx = get_sf_out_offset_128x4(batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols);
                SFOutput[dstIdx] = sf;
            }
        }
    }
}

__global__ void block_scale_interleave_reverse_kernel(
    int numBatches, int numRows, int numCols, uint8_t const* SFIn, uint8_t* SFOutput)
{
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
        {
            for (int colIdx = threadIdx.x; colIdx < numCols; colIdx += blockDim.x)
            {
                std::optional<int> batchIdxOpt = batchIdx;
                std::optional<int> numRowsOpt = numRows;

                // Get the swizzled input index using the same swizzling pattern
                auto srcIdx = get_sf_out_offset_128x4(batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols);
                auto sf = SFIn[srcIdx];

                // Output goes to linear layout
                int64_t outOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
                SFOutput[outOffset] = sf;
            }
        }
    }
}

// This is intended for weight loading, so m and n are large, b <= 256
void invokeBlockScaleInterleave(int b, int m, int m_padded, int n, int n_padded, uint8_t const* SFIn, uint8_t* SFOutput,
    int multiProcessorCount, cudaStream_t stream)
{
    // Each thread reads 1 int8 value
    dim3 block(std::min(n_padded, 1024));
    // Get number of blocks per SM (assume we can fully utilize the SM).
    int const numBlocksPerSM = std::max(1u, 4096u / block.x);
    dim3 grid(std::min(m_padded, multiProcessorCount * numBlocksPerSM));

    block_scale_interleave_kernel<<<grid, block, 0, stream>>>(b, m, m_padded, n, n_padded, SFIn, SFOutput);
}

// This is intended for weight loading, so m and n are large, b <= 256
void invokeBlockScaleInterleaveReverse(
    int b, int m, int n, uint8_t const* SFIn, uint8_t* SFOutput, int multiProcessorCount, cudaStream_t stream)
{
    // Each thread reads 1 int8 value
    dim3 block(std::min(n, 1024));
    // Get number of blocks per SM (assume we can fully utilize the SM).
    int const numBlocksPerSM = std::max(1u, 4096u / block.x);
    dim3 grid(std::min(m, multiProcessorCount * numBlocksPerSM));

    block_scale_interleave_reverse_kernel<<<grid, block, 0, stream>>>(b, m, n, SFIn, SFOutput);
}

// FP4 Dequantization

// Convert 8 e2m1 values (represented as one uint32_t) into 8 float32 values.
inline __device__ void e2m1_to_fp32_vec(uint32_t e2m1Vec, float (&array)[8])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t out_fp16[4];
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2   %0, byte0;\n"
        "cvt.rn.f16x2.e2m1x2   %1, byte1;\n"
        "cvt.rn.f16x2.e2m1x2   %2, byte2;\n"
        "cvt.rn.f16x2.e2m1x2   %3, byte3;\n"
        "}"
        : "=r"(out_fp16[0]), "=r"(out_fp16[1]), "=r"(out_fp16[2]), "=r"(out_fp16[3])
        : "r"(e2m1Vec));

    // Convert FP16x2 values to float2 values using vectorized conversion
    float2 res0 = __half22float2(reinterpret_cast<__half2&>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2&>(out_fp16[1]));
    float2 res2 = __half22float2(reinterpret_cast<__half2&>(out_fp16[2]));
    float2 res3 = __half22float2(reinterpret_cast<__half2&>(out_fp16[3]));

    array[0] = res0.x;
    array[1] = res0.y;
    array[2] = res1.x;
    array[3] = res1.y;
    array[4] = res2.x;
    array[5] = res2.y;
    array[6] = res3.x;
    array[7] = res3.y;
#else
    // Fallback for older architectures
    static float const kE2M1ToFloatArray[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    for (int i = 0; i < 8; i++)
    {
        uint8_t e2m1Val = (e2m1Vec >> (i * 4)) & 0xF;
        bool signBit = e2m1Val & 8;
        auto absValue = e2m1Val & 7;
        float result = kE2M1ToFloatArray[absValue];
        if (signBit)
            result = -result;
        array[i] = result;
    }
#endif
}

// Main FP4 dequantization kernel
template <typename T, int SF_VEC_SIZE, bool UE8M0_SF>
__global__ void cvt_fp4_to_fp16(int m, int n, uint32_t const* input, uint32_t const* SFInput, float const* globalScale,
    T* output, FP4QuantizationSFLayout layout)
{
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const totalThreads = gridDim.x * blockDim.x;

    int const groupsPerRow = n / SF_VEC_SIZE;
    int const totalGroups = m * groupsPerRow;

    float const globalScaleVal = *globalScale;

    // In dequantization, each thread processes one complete scale factor group (SF_VEC_SIZE elements)
    constexpr int CVT_FP4_NUM_THREADS_PER_SF = 1;

    for (int groupIdx = tid; groupIdx < totalGroups; groupIdx += totalThreads)
    {
        int const rowIdx = groupIdx / groupsPerRow;
        int const colGroupIdx = groupIdx % groupsPerRow;

        // Convert group index to column index - each thread processes SF_VEC_SIZE elements
        int const colIdx = colGroupIdx * SF_VEC_SIZE;

        // Use the existing function to get scale factor offset
        // Cast to uint32_t* for the template function (safe since we're only reading)
        std::optional<int> optionalNumRows = m;
        uint8_t const* sfPtr = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF, SF_VEC_SIZE>(
            std::nullopt /* batchIdx */, rowIdx, colGroupIdx, optionalNumRows, n, const_cast<uint32_t*>(SFInput),
            layout);

        // Load scale factor (the function returns pointer, so we dereference it)
        uint8_t sfVal = *sfPtr;
        float scaleFloat;
        if constexpr (UE8M0_SF)
        {
            // UE8M0 format: direct bit manipulation
            uint32_t tmp = uint32_t(sfVal) << 23;
            scaleFloat = reinterpret_cast<float const&>(tmp);
        }
        else
        {
            // UE4M3 format
            __nv_fp8_e4m3 fp8Val;
            fp8Val.__x = sfVal;
            scaleFloat = float(fp8Val);
        }
        scaleFloat *= globalScaleVal;

        // Process SF_VEC_SIZE elements in this group (typically 16 elements)
        for (int elemIdx = 0; elemIdx < SF_VEC_SIZE; elemIdx += 8)
        {
            int const packedColIdx = (colIdx + elemIdx) / 8;
            int const packedInputIdx = rowIdx * (n / 8) + packedColIdx;

            // Load packed FP4 data (8 elements = 32 bits)
            uint32_t fp4Packed = input[packedInputIdx];

            // Convert E2M1 to float
            float fp32Values[8];
            e2m1_to_fp32_vec(fp4Packed, fp32Values);

            // Scale and convert to output type
#pragma unroll
            for (int i = 0; i < 8; i++)
            {
                int const outputIdx = rowIdx * n + colIdx + elemIdx + i;
                if (outputIdx < m * n)
                {
                    float scaledValue = fp32Values[i] * scaleFloat;
                    output[outputIdx] = cuda_cast<T>(scaledValue);
                }
            }
        }
    }
}

template <typename T, int SF_VEC_SIZE>
void invokeFP4Dequantization(int m, int n, int64_t const* input, int32_t const* SFInput, float const* globalScale,
    T* output, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream)
{
    // Grid, Block size.
    // Each thread processes multiple groups
    dim3 block(std::min(512, multiProcessorCount * 32));
    dim3 grid(std::min(multiProcessorCount * 8, int((m * n / SF_VEC_SIZE + block.x - 1) / block.x)));

    // Launch the dequantization kernel
    if (useUE8M0)
    {
        cvt_fp4_to_fp16<T, SF_VEC_SIZE, true><<<grid, block, 0, stream>>>(m, n,
            reinterpret_cast<uint32_t const*>(input), reinterpret_cast<uint32_t const*>(SFInput), globalScale, output,
            layout);
    }
    else
    {
        cvt_fp4_to_fp16<T, SF_VEC_SIZE, false><<<grid, block, 0, stream>>>(m, n,
            reinterpret_cast<uint32_t const*>(input), reinterpret_cast<uint32_t const*>(SFInput), globalScale, output,
            layout);
    }
}

// Instantiate the function.
template void invokeFP4Quantization<half, 16>(int b, int m, int n, half const* input, float const* SFScale,
    int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream);
template void invokeFP4Quantization<half, 32>(int b, int m, int n, half const* input, float const* SFScale,
    int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream);
template void invokeMxFP8Quantization<half>(int b, int m, int n, int padded_n, half const* input, int64_t* output,
    int32_t* SFOuput, QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeFP4Quantization<__nv_bfloat16, 16>(int b, int m, int n, __nv_bfloat16 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream);
template void invokeFP4Quantization<__nv_bfloat16, 32>(int b, int m, int n, __nv_bfloat16 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream);
template void invokeMxFP8Quantization<__nv_bfloat16>(int b, int m, int n, int padded_n, __nv_bfloat16 const* input,
    int64_t* output, int32_t* SFOuput, QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream);
#endif

#ifdef ENABLE_FP8
template void invokeFP4Quantization<__nv_fp8_e4m3, 16>(int b, int m, int n, __nv_fp8_e4m3 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream);
template void invokeFP4Quantization<__nv_fp8_e4m3, 32>(int b, int m, int n, __nv_fp8_e4m3 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream);
#endif

// FP4 Dequantization template instantiations
template void invokeFP4Dequantization<half, 16>(int m, int n, int64_t const* input, int32_t const* SFInput,
    float const* globalScale, half* output, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream);
template void invokeFP4Dequantization<half, 32>(int m, int n, int64_t const* input, int32_t const* SFInput,
    float const* globalScale, half* output, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream);
template void invokeFP4Dequantization<float, 16>(int m, int n, int64_t const* input, int32_t const* SFInput,
    float const* globalScale, float* output, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream);
template void invokeFP4Dequantization<float, 32>(int m, int n, int64_t const* input, int32_t const* SFInput,
    float const* globalScale, float* output, bool useUE8M0, FP4QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeFP4Dequantization<__nv_bfloat16, 16>(int m, int n, int64_t const* input, int32_t const* SFInput,
    float const* globalScale, __nv_bfloat16* output, bool useUE8M0, FP4QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream);
template void invokeFP4Dequantization<__nv_bfloat16, 32>(int m, int n, int64_t const* input, int32_t const* SFInput,
    float const* globalScale, __nv_bfloat16* output, bool useUE8M0, FP4QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream);
#endif

} // namespace kernels
} // namespace tensorrt_llm
