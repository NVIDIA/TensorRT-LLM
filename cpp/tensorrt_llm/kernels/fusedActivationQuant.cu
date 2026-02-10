/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/fusedActivationQuant.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/kernels/quantization.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

constexpr int kEltsPerThread = 8;

__device__ __forceinline__ float relu2(float x)
{
    float r = fmaxf(0.0f, x);
    return r * r;
}

template <typename T>
__global__ void fusedRelu2QuantizeKernel(T const* __restrict__ input, float const* __restrict__ sfScale,
    uint32_t* __restrict__ outputFp4, uint32_t* __restrict__ outputSf, int m, int n)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    constexpr int kSfVecSize = 16;
    float const SFScaleVal = sfScale[0];
    int numColThreads = n / kEltsPerThread;
    int numColVecs = n / kSfVecSize;
    int rowIdx = blockIdx.x;

    if (rowIdx >= m)
        return;

    for (int colIdx = threadIdx.x; colIdx < numColThreads; colIdx += blockDim.x)
    {
        int inputOffset = rowIdx * n + colIdx * kEltsPerThread;
        float vals[kEltsPerThread];

#pragma unroll
        for (int i = 0; i < kEltsPerThread; i++)
        {
            vals[i] = relu2(static_cast<float>(input[inputOffset + i]));
        }

        float localMax = vals[0];
#pragma unroll
        for (int i = 1; i < kEltsPerThread; i++)
        {
            localMax = fmaxf(localMax, vals[i]);
        }

        localMax = fmaxf(localMax, __shfl_xor_sync(0xffffffff, localMax, 1));
        float vecMax = localMax;

        float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
        __nv_fp8_e4m3 fp8SF = __nv_fp8_e4m3(SFValue);
        uint8_t fp8SFVal = fp8SF.__x;
        SFValue = static_cast<float>(fp8SF);

        float outputScale
            = vecMax != 0.0f ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

        constexpr int NUM_THREADS_PER_SF = kSfVecSize / kEltsPerThread;
        if (colIdx % NUM_THREADS_PER_SF == 0)
        {
            auto sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(std::nullopt, rowIdx, colIdx,
                std::optional<int>(m), numColVecs, outputSf, QuantizationSFLayout::SWIZZLED);
            if (sfOutPtr != nullptr)
            {
                *sfOutPtr = fp8SFVal;
            }
        }

        float2 fp2Vals[kEltsPerThread / 2];
#pragma unroll
        for (int i = 0; i < kEltsPerThread / 2; i++)
        {
            fp2Vals[i].x = vals[i * 2] * outputScale;
            fp2Vals[i].y = vals[i * 2 + 1] * outputScale;
        }

        outputFp4[rowIdx * numColThreads + colIdx] = fp32_vec_to_e2m1(fp2Vals);
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("FP4 quantization requires SM100 (Blackwell) or later!\n");
    }
#endif
}

template <typename T>
void invokeFusedRelu2Quantize(T const* input, float const* sfScale, uint8_t* outputFp4, uint8_t* outputSf, int m, int n,
    int sfVecSize, cudaStream_t stream)
{
    int numColThreads = n / kEltsPerThread;
    int threadsPerBlock = min(512, numColThreads);
    threadsPerBlock = max(32, ((threadsPerBlock + 31) / 32) * 32);

    fusedRelu2QuantizeKernel<T><<<m, threadsPerBlock, 0, stream>>>(
        input, sfScale, reinterpret_cast<uint32_t*>(outputFp4), reinterpret_cast<uint32_t*>(outputSf), m, n);
}

template void invokeFusedRelu2Quantize<half>(
    half const*, float const*, uint8_t*, uint8_t*, int, int, int, cudaStream_t);

#ifdef ENABLE_BF16
template void invokeFusedRelu2Quantize<__nv_bfloat16>(
    __nv_bfloat16 const*, float const*, uint8_t*, uint8_t*, int, int, int, cudaStream_t);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
