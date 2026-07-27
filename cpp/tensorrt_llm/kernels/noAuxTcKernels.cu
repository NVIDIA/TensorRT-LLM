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

#include "moeTopKFuncs.cuh"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/noAuxTcKernels.h"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
static constexpr int WARP_SIZE = 32;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxSupportedExpertCount = 1024;
static constexpr int NumTopGroupScores = 2;
static constexpr int DefaultMaxNumTopExperts = 8;
static constexpr int MaxSupportedTopExperts = 32;
static constexpr int DefaultMaxNumTopGroups = 4;
static constexpr int LargeMaxNumTopGroups = 8;

static __device__ inline float sigmoid_accurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

template <typename InputT, typename BiasT, typename OutputT, typename IdxT, int MaxNumExperts, bool UseGroups,
    int MaxNumTopExperts = DefaultMaxNumTopExperts, int MaxNumTopGroups = DefaultMaxNumTopGroups>
__global__ void deepseek_v3_topk_kernel(InputT* scores, OutputT* topkValues, IdxT* topkIndices, BiasT* routingBias,
    int64_t const numTokens, int64_t const numGroup, int64_t const topkGroup, int64_t const topk,
    int64_t const numExperts, int64_t const numExpertsPerGroup, double const routedScalingFactor)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    __shared__ float __attribute((aligned(128))) smemScoreSigmoid[MaxNumExperts];
    __shared__ float __attribute((aligned(128))) smemScoreBias[MaxNumExperts];

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    int32_t laneIdx = threadIdx.x % WARP_SIZE;
    int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);

    static constexpr float invalidScoreFloat = float{-INFINITY};

    topkValues += blockIdx.x * topk;
    topkIndices += blockIdx.x * topk;

    if constexpr (UseGroups)
    {
        int constexpr NumWarps = MaxNumExperts / WARP_SIZE;
        __shared__ float __attribute((aligned(128))) smemGroupScores[NumWarps];

        if (warpIdx >= numGroup)
        {
            return;
        }

        auto threadExpert = warpIdx * numExpertsPerGroup + laneIdx;
        bool expertSelected = laneIdx < numExpertsPerGroup;

        auto scoreIdx = int64_t{blockIdx.x} * int64_t{numExperts} + threadExpert;
        auto biasVal = expertSelected ? static_cast<float>(routingBias[threadExpert]) : invalidScoreFloat;
        float score = expertSelected ? static_cast<float>(scores[scoreIdx]) : invalidScoreFloat;
        auto scoreSigmoid = sigmoid_accurate(score);
        if (expertSelected)
        {
            smemScoreSigmoid[threadExpert] = scoreSigmoid;
        }
        auto scoreBias = float{scoreSigmoid + float{biasVal}};
        if (expertSelected)
        {
            smemScoreBias[threadExpert] = scoreBias;
        }

        float topExpGroupScores[NumTopGroupScores];
        [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
        reduce_topk::reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
            /* minValue */ invalidScoreFloat);

        if (warp.thread_rank() == 0)
        {
            auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
            smemGroupScores[warpIdx] = groupScore;
        }

        __syncthreads();

        float topScores[MaxNumTopExperts];
        int32_t topExperts[MaxNumTopExperts];

        if (warpIdx == 0)
        {
            float topGroups[MaxNumTopGroups];
            int32_t topGroupIdx[MaxNumTopGroups];
            float groupScore = laneIdx < numGroup ? smemGroupScores[laneIdx] : invalidScoreFloat;
            reduce_topk::reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
                /* minValue */ invalidScoreFloat);

            float expertScoreGroup[MaxNumTopGroups];
            int32_t expertIdxGroup[MaxNumTopGroups];
#pragma unroll
            for (int ii = 0; ii < MaxNumTopGroups; ++ii)
            {
                auto groupIdx = topGroupIdx[ii];
                expertIdxGroup[ii] = groupIdx * numExpertsPerGroup + laneIdx;
                expertScoreGroup[ii]
                    = (ii < topkGroup) && expertSelected ? smemScoreBias[expertIdxGroup[ii]] : invalidScoreFloat;
            }

            reduce_topk::reduceTopK(
                warp, topScores, topExperts, expertScoreGroup, expertIdxGroup, /* minValue */ invalidScoreFloat, topk);

            int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
            float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
            auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
            auto finalScore = static_cast<OutputT>(scoreNorm * routedScalingFactor / (redNorm + 1e-20));
            if (laneIdx < topk)
            {
                topkValues[laneIdx] = static_cast<OutputT>(finalScore);
                topkIndices[laneIdx] = expertIdx;
            }
        }
    }
    else
    {
        for (int e = threadIdx.x; e < numExperts; e += blockDim.x)
        {
            auto scoreIdx = int64_t{blockIdx.x} * int64_t{numExperts} + e;
            auto biasVal = static_cast<float>(routingBias[e]);
            float score = static_cast<float>(scores[scoreIdx]);
            auto scoreSigmoid = sigmoid_accurate(score);
            smemScoreSigmoid[e] = scoreSigmoid;
            smemScoreBias[e] = scoreSigmoid + biasVal;
        }

        __syncthreads();

        float topScores[MaxNumTopExperts];
        int32_t topExperts[MaxNumTopExperts];

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
            reduce_topk::reduceTopK(warp, topScores, topExperts, localScores, localIdx,
                /* minValue */ invalidScoreFloat, topk);

            int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
            float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
            auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
            auto finalScore = static_cast<OutputT>(scoreNorm * routedScalingFactor / (redNorm + 1e-20));
            if (laneIdx < topk)
            {
                topkValues[laneIdx] = static_cast<OutputT>(finalScore);
                topkIndices[laneIdx] = expertIdx;
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename InputT, typename BiasT, typename OutputT, typename IdxT>
void invokeNoAuxTc(InputT* scores, BiasT* bias, OutputT* topk_values, IdxT* topk_indices, int64_t const num_tokens,
    int64_t const num_experts, int64_t const n_group, int64_t const topk_group, int64_t const topk,
    double const routed_scaling_factor, cudaStream_t const stream)
{
    bool const is_single_group
        = (n_group <= 1) && (num_experts <= MaxSupportedExpertCount) && (topk <= MaxSupportedTopExperts);

    int64_t const experts_per_group = num_experts / n_group;
    bool const is_multi_group = (n_group > 1) && (num_experts <= NumDeepseekExperts) && (experts_per_group <= WARP_SIZE)
        && (topk <= DefaultMaxNumTopExperts) && (experts_per_group * topk_group <= LargeMaxNumTopGroups * WARP_SIZE);

    if (is_single_group || is_multi_group)
    {
        cudaLaunchConfig_t config;
        auto* kernel_instance = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, NumDeepseekExperts, true>;
        int num_threads = NumDeepseekExperts;

        if (is_multi_group)
        {
            if (experts_per_group * topk_group <= DefaultMaxNumTopGroups * WARP_SIZE)
            {
                kernel_instance = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, NumDeepseekExperts, true>;
            }
            else
            {
                kernel_instance = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, NumDeepseekExperts, true,
                    DefaultMaxNumTopExperts, LargeMaxNumTopGroups>;
            }
            num_threads = NumDeepseekExperts;
        }
        else if (is_single_group)
        {
            if (num_experts <= 128)
            {
                kernel_instance
                    = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 128, false, MaxSupportedTopExperts>;
                num_threads = 128;
            }
            else if (num_experts <= 256)
            {
                kernel_instance
                    = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 256, false, MaxSupportedTopExperts>;
                num_threads = 256;
            }
            else if (num_experts <= 512)
            {
                kernel_instance
                    = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 512, false, MaxSupportedTopExperts>;
                num_threads = 256;
            }
            else
            {
                kernel_instance
                    = &deepseek_v3_topk_kernel<InputT, BiasT, OutputT, IdxT, 1024, false, MaxSupportedTopExperts>;
                num_threads = 256;
            }
        }

        config.gridDim = num_tokens;
        config.blockDim = num_threads;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        cudaLaunchKernelEx(&config, kernel_instance, scores, topk_values, topk_indices, bias, num_tokens, n_group,
            topk_group, topk, num_experts, num_experts / n_group, routed_scaling_factor);
        sync_check_cuda_error(stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "invokeNoAuxTc: unsupported configuration (n_group=%ld, num_experts=%ld, topk_group=%ld, topk=%ld). "
            "Please use original pytorch implementation.",
            n_group, num_experts, topk_group, topk);
    }
}

#define INSTANTIATE_NOAUX_TC(InputT, BiasT, OutputT, IdxT)                                                             \
    template void invokeNoAuxTc<InputT, BiasT, OutputT, IdxT>(InputT * scores, BiasT * bias, OutputT * topk_values,    \
        IdxT * topk_indices, int64_t const num_tokens, int64_t const num_experts, int64_t const n_group,               \
        int64_t const topk_group, int64_t const topk, double const routed_scaling_factor, cudaStream_t const stream);

INSTANTIATE_NOAUX_TC(float, float, float, int32_t);
INSTANTIATE_NOAUX_TC(float, half, float, int32_t);

INSTANTIATE_NOAUX_TC(half, float, half, int32_t);
INSTANTIATE_NOAUX_TC(half, half, half, int32_t);

#ifdef ENABLE_BF16
INSTANTIATE_NOAUX_TC(float, __nv_bfloat16, float, int32_t);
INSTANTIATE_NOAUX_TC(half, __nv_bfloat16, half, int32_t);

INSTANTIATE_NOAUX_TC(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, float, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, half, __nv_bfloat16, int32_t);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
