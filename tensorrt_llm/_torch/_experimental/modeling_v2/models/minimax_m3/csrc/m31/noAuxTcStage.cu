/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 * SPDX-License-Identifier: Apache-2.0
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

// The MegaMoE input staging of the MiniMax-M3.1 decode MoE layers: single-group noaux_tc routing (the ungrouped
// routing of kernels/noAuxTcKernels.cu's deepseek_v3_topk_block, with the routing weights formed without FP64
// instructions on the common path) and NVFP4 quantization written straight into the stages.

#include "minimax_m3/csrc/m31/noAuxTcStage.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <optional>

namespace cg = cooperative_groups;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::minimaxM31
{

namespace
{

constexpr int WARP_SIZE = 32;
constexpr int NumDeepseekExperts = 256;
constexpr int MaxSupportedTopExperts = 32;

__device__ inline float sigmoid_accurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

// The routed scaling factor as a float when it is a float with at most 20 significant bits (the low 4 mantissa bits
// zero) between 2^-20 and 2^20. Decided from the bits of the double, so no FP64 instruction runs.
__device__ __forceinline__ bool scalingFactorAsShortFloat(double factor, float& factorFloat)
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
__device__ __noinline__ float noAuxTcRoutingWeightFp64(float s, float S, double c)
{
    return static_cast<float>(s * c / (S + 1e-20));
}

// noAuxTcRoutingWeightFp64 without FP64 instructions on the common path: on SM107 an FP64 instruction drops the SM
// clock of its GPC to about 1 GHz, recovering over tens of microseconds, which slows the kernels that follow. When c
// is a float with at most 20 significant bits, S >= 2^-13 and s * c >= 2^-60, the double expression is the float
// rounding of the exact quotient s * c / S: s * c is exact in double, S + 1e-20 rounds to S, and a quotient of these
// operands that is not itself a float rounding boundary lies at least 2^-49 relative from one, farther than rounding to
// double can move it. The quotient comes from an exact two-float product, a correctly rounded division and its exact
// remainder (no intermediate is subnormal); it is returned when it lies more than 2^-20 ulp inside its rounding
// interval and is not a power of two. Otherwise the FP64 expression runs, out of line.
__device__ __forceinline__ float noAuxTcRoutingWeight(float s, float S, double c)
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

// One token's single-group routing (deepseek_v3_topk_block's ungrouped path at MaxNumTopExperts = 32): the sigmoid
// scores plus bias in shared memory, warp 0's iterated top-k over them, then the normalized, scaled weights of the
// selected experts. scores, topkValues and topkIndices point at the token's row.
template <int MaxNumExperts>
__device__ __forceinline__ void routeRow(float const* scores, float* topkValues, int32_t* topkIndices,
    float const* routingBias, int64_t const topk, int64_t const numExperts, double const routedScalingFactor)
{
    __shared__ float __attribute((aligned(128))) smemScoreSigmoid[MaxNumExperts];
    __shared__ float __attribute((aligned(128))) smemScoreBias[MaxNumExperts];

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    int32_t laneIdx = threadIdx.x % WARP_SIZE;
    int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);

    static constexpr float invalidScoreFloat = float{-INFINITY};

    for (int e = threadIdx.x; e < numExperts; e += blockDim.x)
    {
        auto biasVal = routingBias[e];
        float score = scores[e];
        auto scoreSigmoid = sigmoid_accurate(score);
        smemScoreSigmoid[e] = scoreSigmoid;
        smemScoreBias[e] = scoreSigmoid + biasVal;
    }

    __syncthreads();

    float topScores[MaxSupportedTopExperts];
    int32_t topExperts[MaxSupportedTopExperts];

    if (warpIdx == 0)
    {
        constexpr int NumChunks = (MaxNumExperts + WARP_SIZE - 1) / WARP_SIZE;
        float localScores[NumChunks];
        int32_t localIdx[NumChunks];
#pragma unroll
        for (int ii = 0; ii < NumChunks; ++ii)
        {
            auto expertIdx = ii * WARP_SIZE + laneIdx;
            localIdx[ii] = expertIdx;
            localScores[ii] = expertIdx < numExperts ? smemScoreBias[expertIdx] : invalidScoreFloat;
        }
        tensorrt_llm::kernels::reduce_topk::reduceTopK(warp, topScores, topExperts, localScores, localIdx,
            /* minValue */ invalidScoreFloat, topk);

        int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
        float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
        auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
        // Only the lanes that store a weight form one: a zero score would take the FP64 path.
        if (laneIdx < topk)
        {
            topkValues[laneIdx] = noAuxTcRoutingWeight(scoreNorm, redNorm, routedScalingFactor);
            topkIndices[laneIdx] = expertIdx;
        }
    }
}

// Routing, NVFP4 quantization and staging of a decode MoE layer's tokens into two MegaMoE instances in one launch:
// CTAs [0, M) each route one token with the single-group device function and expert tier noaux_tc picks for the same
// expert count, CTAs [M, 2M) each quantize one row with the per-thread conversion of quantize_with_block_size, and CTA
// 2M resets both stages' tails. Neither result depends on the block size: routing loops over the experts and reduces
// in warp 0, and each 16-element scale block is converted by a fixed pair of adjacent threads.
constexpr int NoAuxTcFp4QuantStageThreads = 256;
constexpr int NvFp4SfVecSize = 16;

__device__ __forceinline__ void resetNoAuxTcFp4QuantStageTail(NoAuxTcFp4QuantStage const& stage, int topk)
{
    int64_t const indexCells = static_cast<int64_t>(stage.indexCapRows - stage.numRows) * topk;
    for (int64_t cell = threadIdx.x; cell < indexCells; cell += blockDim.x)
    {
        int64_t const row = stage.numRows + cell / topk;
        stage.topkIndices[row * stage.indexStride + cell % topk] = -1;
    }
    int64_t const weightCells = static_cast<int64_t>(stage.weightCapRows - stage.numRows) * topk;
    for (int64_t cell = threadIdx.x; cell < weightCells; cell += blockDim.x)
    {
        int64_t const row = stage.numRows + cell / topk;
        stage.topkWeights[row * stage.weightStride + cell % topk] = 0.F;
    }
}

template <int MaxNumExperts>
__global__ __launch_bounds__(NoAuxTcFp4QuantStageThreads) void noaux_tc_fp4_quant_stage_kernel(float const* scores,
    float const* routingBias, __nv_bfloat16 const* hiddenStates, float const* globalScale, uint8_t const* padMask,
    NoAuxTcFp4QuantStage stage0, NoAuxTcFp4QuantStage stage1, int numTokens, int topk, int numExperts, int hiddenSize,
    double routedScalingFactor)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    int const ctaIdx = static_cast<int>(blockIdx.x);
    if (ctaIdx < 2 * numTokens)
    {
        int const tokenIdx = ctaIdx < numTokens ? ctaIdx : ctaIdx - numTokens;
        bool const inStage0 = tokenIdx < stage0.numRows;
        NoAuxTcFp4QuantStage const& stage = inStage0 ? stage0 : stage1;
        int64_t const stageRow = inStage0 ? tokenIdx : tokenIdx - stage0.numRows;
        if (ctaIdx < numTokens && padMask != nullptr && padMask[tokenIdx] != 0)
        {
            // A pad row takes the tail's values; the branch is uniform across the CTA.
            for (int k = threadIdx.x; k < topk; k += blockDim.x)
            {
                stage.topkIndices[stageRow * stage.indexStride + k] = -1;
                stage.topkWeights[stageRow * stage.weightStride + k] = 0.F;
            }
        }
        else if (ctaIdx < numTokens)
        {
            routeRow<MaxNumExperts>(scores + static_cast<int64_t>(tokenIdx) * numExperts,
                stage.topkWeights + stageRow * stage.weightStride, stage.topkIndices + stageRow * stage.indexStride,
                routingBias, topk, numExperts, routedScalingFactor);
        }
        else
        {
            using QuantPackedVec = tensorrt_llm::kernels::PackedVec<__nv_bfloat16>;
            static constexpr int CvtNumThreadsPerSf = NvFp4SfVecSize / tensorrt_llm::kernels::CVT_ELTS_PER_THREAD;
            int const numColThreads = hiddenSize / tensorrt_llm::kernels::CVT_ELTS_PER_THREAD;
            float const sfScaleVal = globalScale[0];
            int64_t const inRowOffset = static_cast<int64_t>(tokenIdx) * numColThreads;
            QuantPackedVec const* inRow = reinterpret_cast<QuantPackedVec const*>(hiddenStates) + inRowOffset;
            uint32_t* fp4Row = stage.fp4 + stageRow * stage.fp4StrideWords;
            uint32_t* sfRow = reinterpret_cast<uint32_t*>(stage.sf + stageRow * stage.sfStride);
            std::optional<int> const optionalBatchIdx = 0;
            std::optional<int> const optionalNumRows = 1;
            for (int colIdx = threadIdx.x; colIdx < numColThreads; colIdx += blockDim.x)
            {
                auto sfOutPtr
                    = tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<uint32_t, CvtNumThreadsPerSf>(optionalBatchIdx,
                        0, colIdx, optionalNumRows, hiddenSize / NvFp4SfVecSize, sfRow, QuantizationSFLayout::LINEAR);
                QuantPackedVec inVec = inRow[colIdx];
                fp4Row[colIdx] = tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<__nv_bfloat16, NvFp4SfVecSize, false>(
                    inVec, sfScaleVal, sfOutPtr);
            }
        }
    }
    else
    {
        resetNoAuxTcFp4QuantStageTail(stage0, topk);
        resetNoAuxTcFp4QuantStageTail(stage1, topk);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // Every thread's stores are visible before any thread of the CTA reports completion.
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// The routing CTAs of noaux_tc_fp4_quant_stage_kernel without its quantizing CTAs: CTAs [0, numRows) each route one
// token, and the CTAs past them share the tail reset, about NoAuxTcRouteStageTailCellsPerCta cells each, since at
// prefill rows one CTA resetting the tail outlasts the routing CTAs.
constexpr int64_t NoAuxTcRouteStageTailCellsPerCta = 1024;
constexpr int64_t NoAuxTcRouteStageMaxTailCtas = 64;

template <int MaxNumExperts>
__global__ __launch_bounds__(NoAuxTcFp4QuantStageThreads) void noaux_tc_route_stage_kernel(float const* scores,
    float const* routingBias, uint8_t const* padMask, NoAuxTcFp4QuantStage stage, int topk, int numExperts,
    double routedScalingFactor)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    int const ctaIdx = static_cast<int>(blockIdx.x);
    if (ctaIdx < stage.numRows)
    {
        int64_t const row = ctaIdx;
        if (padMask != nullptr && padMask[ctaIdx] != 0)
        {
            for (int k = threadIdx.x; k < topk; k += blockDim.x)
            {
                stage.topkIndices[row * stage.indexStride + k] = -1;
                stage.topkWeights[row * stage.weightStride + k] = 0.F;
            }
        }
        else
        {
            routeRow<MaxNumExperts>(scores + row * numExperts, stage.topkWeights + row * stage.weightStride,
                stage.topkIndices + row * stage.indexStride, routingBias, topk, numExperts, routedScalingFactor);
        }
    }
    else
    {
        int64_t const first = static_cast<int64_t>(ctaIdx - stage.numRows) * blockDim.x + threadIdx.x;
        int64_t const step = static_cast<int64_t>(gridDim.x - stage.numRows) * blockDim.x;
        int64_t const indexCells = static_cast<int64_t>(stage.indexCapRows - stage.numRows) * topk;
        for (int64_t cell = first; cell < indexCells; cell += step)
        {
            stage.topkIndices[(stage.numRows + cell / topk) * stage.indexStride + cell % topk] = -1;
        }
        int64_t const weightCells = static_cast<int64_t>(stage.weightCapRows - stage.numRows) * topk;
        for (int64_t cell = first; cell < weightCells; cell += step)
        {
            stage.topkWeights[(stage.numRows + cell / topk) * stage.weightStride + cell % topk] = 0.F;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

} // namespace

void invokeNoAuxTcFp4QuantStage(float const* scores, float const* bias, __nv_bfloat16 const* hiddenStates,
    float const* globalScale, uint8_t const* padMask, NoAuxTcFp4QuantStage const& stage0,
    NoAuxTcFp4QuantStage const& stage1, int64_t numTokens, int64_t numExperts, int64_t topk, int64_t hiddenSize,
    double routedScalingFactor, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(numExperts >= 1 && numExperts <= NumDeepseekExperts && topk >= 1 && topk <= numExperts
            && topk <= MaxSupportedTopExperts,
        "invokeNoAuxTcFp4QuantStage: 1..256 experts and top-1..32 expected (got %ld experts, top-%ld)", numExperts,
        topk);
    TLLM_CHECK_WITH_INFO(hiddenSize > 0 && hiddenSize % NvFp4SfVecSize == 0,
        "invokeNoAuxTcFp4QuantStage: the hidden size must be a multiple of 16 (got %ld)", hiddenSize);
    TLLM_CHECK_WITH_INFO(
        numTokens >= 0 && stage0.numRows >= 0 && stage1.numRows >= 0 && stage0.numRows + stage1.numRows == numTokens,
        "invokeNoAuxTcFp4QuantStage: the stages' rows (%d + %d) must add up to the tokens (%ld)", stage0.numRows,
        stage1.numRows, numTokens);
    for (NoAuxTcFp4QuantStage const* stage : {&stage0, &stage1})
    {
        TLLM_CHECK_WITH_INFO(stage->numRows <= stage->indexCapRows && stage->numRows <= stage->weightCapRows,
            "invokeNoAuxTcFp4QuantStage: a stage holds fewer rows than it receives");
    }
    auto* kernelInstance = numExperts <= 128 ? &noaux_tc_fp4_quant_stage_kernel<128>
                                             : &noaux_tc_fp4_quant_stage_kernel<NumDeepseekExperts>;
    cudaLaunchConfig_t config;
    config.gridDim = static_cast<unsigned int>(2 * numTokens + 1);
    config.blockDim = NoAuxTcFp4QuantStageThreads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernelInstance, scores, bias, hiddenStates, globalScale, padMask,
        stage0, stage1, static_cast<int>(numTokens), static_cast<int>(topk), static_cast<int>(numExperts),
        static_cast<int>(hiddenSize), routedScalingFactor));
    sync_check_cuda_error(stream);
}

void invokeNoAuxTcRouteStage(float const* scores, float const* bias, uint8_t const* padMask,
    NoAuxTcFp4QuantStage const& stage, int64_t numExperts, int64_t topk, double routedScalingFactor,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(numExperts >= 1 && numExperts <= NumDeepseekExperts && topk >= 1 && topk <= numExperts
            && topk <= MaxSupportedTopExperts,
        "invokeNoAuxTcRouteStage: 1..256 experts and top-1..32 expected (got %ld experts, top-%ld)", numExperts, topk);
    TLLM_CHECK_WITH_INFO(
        stage.numRows >= 0 && stage.numRows <= stage.indexCapRows && stage.numRows <= stage.weightCapRows,
        "invokeNoAuxTcRouteStage: the stage holds fewer rows than it receives");
    int64_t const tailCells
        = static_cast<int64_t>(std::max(stage.indexCapRows, stage.weightCapRows) - stage.numRows) * topk;
    int64_t const numTailCtas = std::min(
        std::max((tailCells + NoAuxTcRouteStageTailCellsPerCta - 1) / NoAuxTcRouteStageTailCellsPerCta, int64_t{1}),
        NoAuxTcRouteStageMaxTailCtas);
    auto* kernelInstance
        = numExperts <= 128 ? &noaux_tc_route_stage_kernel<128> : &noaux_tc_route_stage_kernel<NumDeepseekExperts>;
    cudaLaunchConfig_t config;
    config.gridDim = static_cast<unsigned int>(stage.numRows + numTailCtas);
    config.blockDim = NoAuxTcFp4QuantStageThreads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernelInstance, scores, bias, padMask, stage, static_cast<int>(topk),
        static_cast<int>(numExperts), routedScalingFactor));
    sync_check_cuda_error(stream);
}

} // namespace kernels::minimaxM31

TRTLLM_MV2_NAMESPACE_END
