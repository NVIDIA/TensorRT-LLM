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

#include "cutlass/numeric_conversion.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemmW4A16NVFP4.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/nvfp4ScaleLayout.h"

#include <cub/cub.cuh>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cuda_core_gemm_w4a16_nvfp4
{
namespace
{

template <typename T>
__device__ float toFloat(T value)
{
    return static_cast<float>(value);
}

template <>
__device__ float toFloat<half>(half value)
{
    return __half2float(value);
}

template <>
__device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

} // namespace

template <typename ActType, typename OutputType, typename ScaleType, SizeType32 kTileM, SizeType32 kTileN,
    SizeType32 kBlockSize>
__device__ void cudaCoreGemmImpl(ActType const* __restrict__ act, __nv_fp4_e2m1 const* __restrict__ weight,
    ScaleType const* __restrict__ weightScale, float const weightGlobalScale, OutputType* __restrict__ output,
    SizeType32 m, SizeType32 n, SizeType32 k)
{
    using VecType = int4;
    using ScaleVecType = __nv_fp8x2_e4m3;
    using CvtWeightType =
        typename tensorrt_llm::kernels::cutlass_kernels::TllmToCutlassTypeAdapter<__nv_fp4_e2m1>::type;
    using Converter = cutlass::NumericArrayConverter<float, CvtWeightType, 8>;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;

    static constexpr SizeType32 kStepK = 32;
    static constexpr SizeType32 kStepKScale = kStepK / w4a16_nvfp4::kScaleGranularity;
    static constexpr SizeType32 kTileK = kStepK * kBlockSize;
    static constexpr SizeType32 kCvtCount = static_cast<SizeType32>(sizeof(VecType) / sizeof(CvtSrcType));

    static_assert(kStepK % w4a16_nvfp4::kScaleGranularity == 0);

    auto const tileIdM = static_cast<SizeType32>(blockIdx.x * kTileM);
    auto const tileIdN = static_cast<SizeType32>(blockIdx.y * kTileN);
    auto const tid = static_cast<SizeType32>(threadIdx.x);
    (void) m;

    float tileAct[kStepK];
    float tileWeight[kTileN * kStepK];
    float tileWeightScale[kTileN * kStepKScale];
    float acc[kTileM * kTileN];

#pragma unroll
    for (SizeType32 i = 0; i < kTileM * kTileN; ++i)
    {
        acc[i] = 0.0F;
    }

    act += tileIdM * k;
    weight += tileIdN * k / 2;
    output += tileIdM * n + tileIdN;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    for (SizeType32 idxK = tid * kStepK; idxK < k; idxK += kTileK)
    {
#pragma unroll
        for (SizeType32 j = 0; j < kTileN; ++j)
        {
            auto tileWeightQuantized = reinterpret_cast<VecType const*>(weight + (j * k + idxK) / 2)[0];
#pragma unroll
            for (SizeType32 cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx)
            {
                reinterpret_cast<CvtResType*>(tileWeight)[j * kCvtCount + cvtIdx]
                    = Converter::convert(reinterpret_cast<CvtSrcType*>(&tileWeightQuantized)[cvtIdx]);
            }
        }

#pragma unroll
        for (SizeType32 j = 0; j < kTileN; ++j)
        {
            SizeType32 const rowIdx = tileIdN + j;
            SizeType32 const colIdx = idxK / w4a16_nvfp4::kScaleGranularity;
            SizeType32 const dstIdx = w4a16_nvfp4::getScaleIndex(rowIdx, colIdx, k);
            auto const tileWeightScaleFp8x2 = reinterpret_cast<ScaleVecType const*>(weightScale + dstIdx)[0];
            char2 const tmp = reinterpret_cast<char2 const&>(tileWeightScaleFp8x2);
            tileWeightScale[j * kStepKScale + 0] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.x));
            tileWeightScale[j * kStepKScale + 1] = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.y));
        }

#pragma unroll
        for (SizeType32 i = 0; i < kTileM; ++i)
        {
#pragma unroll
            for (SizeType32 l = 0; l < kStepK; ++l)
            {
                tileAct[l] = toFloat(act[i * k + idxK + l]);
            }

#pragma unroll
            for (SizeType32 j = 0; j < kTileN; ++j)
            {
#pragma unroll
                for (SizeType32 l = 0; l < kStepK; ++l)
                {
                    float const scaledWeight = tileWeight[j * kStepK + l]
                        * tileWeightScale[j * kStepKScale + l / w4a16_nvfp4::kScaleGranularity] * weightGlobalScale;
                    acc[i * kTileN + j] = fma(tileAct[l], scaledWeight, acc[i * kTileN + j]);
                }
            }
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    static constexpr SizeType32 kWarpSize = 32;
    static constexpr SizeType32 kWarpNum = kBlockSize / kWarpSize;
    SizeType32 const warpId = tid / kWarpSize;
    SizeType32 const laneId = tid % kWarpSize;
    __shared__ float shmem[kTileM * kTileN * kWarpNum];
    __shared__ typename WarpReduce::TempStorage tempStorage[kWarpNum];

#pragma unroll
    for (SizeType32 mi = 0; mi < kTileM; ++mi)
    {
#pragma unroll
        for (SizeType32 ni = 0; ni < kTileN; ++ni)
        {
            float const val = WarpReduce(tempStorage[warpId]).Sum(acc[mi * kTileN + ni]);
            if (laneId == 0)
            {
                shmem[mi * kTileN + ni + warpId * kTileM * kTileN] = val;
            }
        }
    }
    __syncthreads();

    for (SizeType32 ii = tid; ii < kTileM * kTileN; ii += kBlockSize)
    {
        SizeType32 const mid = ii / kTileN;
        SizeType32 const nid = ii % kTileN;
        float val = 0.0F;
#pragma unroll
        for (SizeType32 jj = 0; jj < kWarpNum; ++jj)
        {
            val += shmem[jj * kTileM * kTileN + ii];
        }
        output[mid * n + nid] = static_cast<OutputType>(val);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename ActType, typename OutputType, typename ScaleType, SizeType32 kTileM, SizeType32 kTileN,
    SizeType32 kBlockSize>
__global__ void cudaCoreGemm(ActType const* __restrict__ act, __nv_fp4_e2m1 const* __restrict__ weight,
    ScaleType const* __restrict__ weightScale, float const* weightGlobalScale, OutputType* __restrict__ output,
    SizeType32 m, SizeType32 n, SizeType32 k)
{
    cudaCoreGemmImpl<ActType, OutputType, ScaleType, kTileM, kTileN, kBlockSize>(
        act, weight, weightScale, weightGlobalScale[0], output, m, n, k);
}

template <typename ActType, typename OutputType, typename ScaleType, SizeType32 kTileM, SizeType32 kTileN,
    SizeType32 kBlockSize>
void cudaCoreGemmKernel(Params const& params, cudaStream_t stream)
{
    dim3 const block(kBlockSize);
    dim3 const grid(params.m / kTileM, params.n / kTileN);
    cudaCoreGemm<ActType, OutputType, ScaleType, kTileM, kTileN, kBlockSize><<<grid, block, 0, stream>>>(
        reinterpret_cast<ActType const*>(params.act), reinterpret_cast<__nv_fp4_e2m1 const*>(params.weight),
        reinterpret_cast<ScaleType const*>(params.weightScale), params.weightGlobalScale,
        reinterpret_cast<OutputType*>(params.output), params.m, params.n, params.k);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <typename ActType, typename OutputType, typename ScaleType, int kTileM, int kTileN, int kBlockSize>
bool cudaCoreGemmTemplateCaller(Params const& params, cudaStream_t stream)
{
    constexpr int kCudaCoreGemmTemplateMaxM = 16;
    if (params.m == kTileM)
    {
        cudaCoreGemmKernel<ActType, OutputType, ScaleType, kTileM, kTileN, kBlockSize>(params, stream);
        return true;
    }
    if constexpr (kTileM < kCudaCoreGemmTemplateMaxM)
    {
        return cudaCoreGemmTemplateCaller<ActType, OutputType, ScaleType, kTileM + 1, kTileN, kBlockSize>(
            params, stream);
    }
    return false;
}

template <typename ActType, typename OutputType, typename ScaleType = __nv_fp8_e4m3>
bool cudaCoreGemmLauncher(Params const& params, cudaStream_t stream)
{
    constexpr int kDefaultTileN = 2;
    constexpr int kWideTileN = 4;
    constexpr int kMaxGridDimY = 65535;
    if (params.n / kDefaultTileN <= kMaxGridDimY)
    {
        return cudaCoreGemmTemplateCaller<ActType, OutputType, ScaleType, 1, kDefaultTileN, 128>(params, stream);
    }
    if (params.n % kWideTileN == 0 && params.n / kWideTileN <= kMaxGridDimY)
    {
        return cudaCoreGemmTemplateCaller<ActType, OutputType, ScaleType, 1, kWideTileN, 128>(params, stream);
    }
    return false;
}

template <typename ActType>
bool dispatchOutputType(Params const& params, cudaStream_t stream)
{
    if (params.outputType == CUDA_R_16F)
    {
        return cudaCoreGemmLauncher<ActType, half>(params, stream);
    }
    if (params.outputType == CUDA_R_16BF)
    {
        return cudaCoreGemmLauncher<ActType, __nv_bfloat16>(params, stream);
    }
    if (params.outputType == CUDA_R_32F)
    {
        return cudaCoreGemmLauncher<ActType, float>(params, stream);
    }
    return false;
}

bool cudaCoreGemmDispatcher(Params const& params, cudaStream_t stream)
{
    bool dispatched = true;
    int const smVersion = tensorrt_llm::common::getSMVersion();
    if (smVersion != 120 && smVersion != 121)
    {
        dispatched = false;
    }
    else if (params.n % 2 != 0 || params.k % 32 != 0)
    {
        dispatched = false;
    }
    else if (params.m < 1 || params.m > 16)
    {
        dispatched = false;
    }
    else if (params.weightScale == nullptr || params.weightGlobalScale == nullptr)
    {
        dispatched = false;
    }
    else if (params.inputType == CUDA_R_16F)
    {
        dispatched = dispatchOutputType<half>(params, stream);
    }
    else if (params.inputType == CUDA_R_16BF)
    {
        dispatched = dispatchOutputType<__nv_bfloat16>(params, stream);
    }
    else
    {
        dispatched = false;
    }

    if (!dispatched)
    {
        TLLM_LOG_WARNING(
            "tensorrt_llm::kernels::cuda_core_gemm_w4a16_nvfp4::cudaCoreGemmDispatcher [NOT DISPATCHED], "
            "inputType=%d, outputType=%d, m=%d, n=%d, k=%d, sm=%d",
            params.inputType, params.outputType, params.m, params.n, params.k, smVersion);
    }
    return dispatched;
}

} // namespace cuda_core_gemm_w4a16_nvfp4
} // namespace kernels

TRTLLM_NAMESPACE_END
