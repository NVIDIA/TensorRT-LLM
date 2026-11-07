/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/csrc/glmNoAuxTcFp4Quant.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace reduce_topk = ::tensorrt_llm::kernels::reduce_topk;
using ::tensorrt_llm::kernels::CVT_ELTS_PER_THREAD;
using ::tensorrt_llm::kernels::PackedVec;
using ::tensorrt_llm::kernels::cvt_quant_get_sf_out_offset;
using ::tensorrt_llm::kernels::cvt_warp_fp16_to_fp4;

static __device__ inline float sigmoid_accurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

// The routed scaling factor as a float when it is a float with at most 20 significant bits (the low 4 mantissa bits
// zero) between 2^-20 and 2^20. Decided from the bits of the double, so no FP64 instruction runs.
static __device__ __forceinline__ bool scalingFactorAsShortFloat(double factor, float& factorFloat)
{
    uint64_t const bits = static_cast<uint64_t>(__double_as_longlong(factor));
    int const exponent = static_cast<int>((bits >> 52) & 0x7FFU) - 1023;
    uint64_t const mantissa = bits & ((uint64_t{1} << 52) - 1);
    factorFloat
        = __uint_as_float((static_cast<uint32_t>(exponent + 127) << 23) | static_cast<uint32_t>(mantissa >> 29));
    bool const shortMantissa = (mantissa & ((uint64_t{1} << 33) - 1)) == 0;
    bool const inRange = exponent >= -20 && (exponent < 20 || (exponent == 20 && mantissa == 0));
    return (bits >> 63) == 0 && shortMantissa && inRange;
}

// The routing weight static_cast<float>(s * c / (S + 1e-20)) evaluated in double, with s the expert's sigmoid score,
// S the sum of the selected scores and c the routed scaling factor.
static __device__ __noinline__ float noAuxTcRoutingWeightFp64(float s, float S, double c)
{
    return static_cast<float>(s * c / (S + 1e-20));
}

// noAuxTcRoutingWeightFp64 without FP64 instructions on the common path: on Rubin an FP64 instruction drops the SM
// clock of its GPC to about 1 GHz, recovering over tens of microseconds, which slows the kernels that follow. When c
// is a float with at most 20 significant bits, S >= 2^-13 and s * c >= 2^-60, the double expression is the float
// rounding of the exact quotient s * c / S: s * c is exact in double, S + 1e-20 rounds to S, and a quotient of these
// operands that is not itself a float rounding boundary lies at least 2^-49 relative from one, farther than rounding to
// double can move it. The quotient comes from an exact two-float product, a correctly rounded division and its exact
// remainder (no intermediate is subnormal); it is returned when it lies more than 2^-20 ulp inside its rounding
// interval and is not a power of two. Otherwise the FP64 expression runs, out of line.
static __device__ __forceinline__ float noAuxTcRoutingWeight(float s, float S, double c)
{
    float cFloat;
    if (scalingFactorAsShortFloat(c, cFloat) && S >= 0x1p-13f)
    {
        float const pHi = __fmul_rn(s, cFloat);
        if (pHi >= 0x1p-60f)
        {
            float const pLo = __fmaf_rn(s, cFloat, -pHi);
            float const q = __fdiv_rn(pHi, S);
            float const r = __fmaf_rn(-q, S, pHi);
            float const e = __fdiv_rn(__fadd_rn(r, pLo), S);
            float const y = __fadd_rn(q, e);
            uint32_t const bits = __float_as_uint(y);
            if ((bits & 0x7FFFFFU) != 0 && (bits & 0x7F800000U) != 0x7F800000U)
            {
                float const ulp = __uint_as_float(bits & 0x7F800000U) * 0x1p-23f;
                float const g = __fsub_rn(e, __fsub_rn(y, q));
                if (fabsf(g) < 0.5F * ulp - 0x1p-20f * ulp)
                {
                    return y;
                }
            }
        }
    }
    return noAuxTcRoutingWeightFp64(s, S, c);
}

static constexpr int MaxSupportedTopExperts = 32;

__device__ __forceinline__ void routeToken(float const* scores, float const* bias, float* topkValues,
    int32_t* topkIndices, int64_t topk, double routedScalingFactor)
{
    __shared__ float __align__(128) sigmoidScores[256];
    __shared__ float __align__(128) biasedScores[256];
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    int const lane = threadIdx.x % 32;
    int const expert = threadIdx.x;
    float const score = sigmoid_accurate(scores[blockIdx.x * 256 + expert]);
    sigmoidScores[expert] = score;
    biasedScores[expert] = score + bias[expert];
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float localScores[8];
        int32_t localExperts[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            localExperts[i] = i * 32 + lane;
            localScores[i] = biasedScores[localExperts[i]];
        }
        float topScores[MaxSupportedTopExperts];
        int32_t topExperts[MaxSupportedTopExperts];
        reduce_topk::reduceTopK(warp, topScores, topExperts, localScores, localExperts, -INFINITY, topk);
        int const selected = lane < topk ? topExperts[lane] : 255;
        float const selectedScore = lane < topk ? sigmoidScores[selected] : 0.F;
        float const sum = cg::reduce(warp, selectedScore, cg::plus<float>{});
        if (lane < topk)
        {
            topkValues[blockIdx.x * topk + lane] = noAuxTcRoutingWeight(selectedScore, sum, routedScalingFactor);
            topkIndices[blockIdx.x * topk + lane] = selected;
        }
    }
}

static constexpr int NoAuxTcFp4QuantNumExperts = 256;
static constexpr int NoAuxTcFp4QuantThreads = NoAuxTcFp4QuantNumExperts;
static constexpr int NvFp4SfVecSize = 16;

__global__ __launch_bounds__(NoAuxTcFp4QuantThreads) void glmRubinNoAuxTcFp4QuantKernel(float* scores,
    float* routingBias, __nv_bfloat16 const* hiddenStates, float const* globalScale, float* topkValues,
    int32_t* topkIndices, uint32_t* fp4Out, uint32_t* sfOut, int64_t numTokens, int64_t topk, int hiddenSize,
    double routedScalingFactor)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    if (blockIdx.x < numTokens)
    {
        routeToken(scores, routingBias, topkValues, topkIndices, topk, routedScalingFactor);
    }
    else
    {
        using QuantPackedVec = PackedVec<__nv_bfloat16>;
        static constexpr int CvtNumThreadsPerSf = NvFp4SfVecSize / CVT_ELTS_PER_THREAD;
        int const rowIdx = static_cast<int>(blockIdx.x - numTokens);
        int const numColThreads = hiddenSize / CVT_ELTS_PER_THREAD;
        float const sfScaleVal = globalScale[0];
        std::optional<int> const optionalBatchIdx = 0;
        std::optional<int> const optionalNumRows = static_cast<int>(numTokens);
        for (int colIdx = threadIdx.x; colIdx < numColThreads; colIdx += blockDim.x)
        {
            auto sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, CvtNumThreadsPerSf>(optionalBatchIdx, rowIdx, colIdx,
                optionalNumRows, hiddenSize / NvFp4SfVecSize, sfOut, QuantizationSFLayout::LINEAR);
            int64_t const offset = static_cast<int64_t>(rowIdx) * numColThreads + colIdx;
            QuantPackedVec inVec = reinterpret_cast<QuantPackedVec const*>(hiddenStates)[offset];
            fp4Out[offset] = cvt_warp_fp16_to_fp4<__nv_bfloat16, NvFp4SfVecSize, false>(inVec, sfScaleVal, sfOutPtr);
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Every thread's stores are visible before any thread of the CTA reports completion.
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void invokeGlmRubinNoAuxTcFp4Quant(float* scores, float* bias, __nv_bfloat16 const* hiddenStates,
    float const* globalScale, float* topkValues, int32_t* topkIndices, uint32_t* fp4Out, uint8_t* sfOut,
    int64_t const numTokens, int64_t const numExperts, int64_t const topk, int64_t const hiddenSize,
    double const routedScalingFactor, cudaStream_t const stream)
{
    TLLM_CHECK_WITH_INFO(numExperts == NoAuxTcFp4QuantNumExperts && topk >= 1 && topk <= MaxSupportedTopExperts,
        "invokeGlmRubinNoAuxTcFp4Quant: 256 experts and top-1..32 expected (got %ld experts, top-%ld)", numExperts,
        topk);
    TLLM_CHECK_WITH_INFO(hiddenSize > 0 && hiddenSize % (2 * CVT_ELTS_PER_THREAD) == 0,
        "invokeGlmRubinNoAuxTcFp4Quant: the hidden size must be a multiple of 16 (got %ld)", hiddenSize);
    cudaLaunchConfig_t config;
    config.gridDim = static_cast<unsigned int>(2 * numTokens);
    config.blockDim = NoAuxTcFp4QuantThreads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, glmRubinNoAuxTcFp4QuantKernel, scores, bias, hiddenStates, globalScale,
        topkValues, topkIndices, fp4Out, reinterpret_cast<uint32_t*>(sfOut), numTokens, topk,
        static_cast<int>(hiddenSize), routedScalingFactor));
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
