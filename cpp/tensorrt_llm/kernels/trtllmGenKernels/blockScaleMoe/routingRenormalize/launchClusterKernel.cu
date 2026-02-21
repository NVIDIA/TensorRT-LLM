/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "RoutingRenormalizeCommon.cuh"

namespace moe::dev::routing
{
namespace routingRenormalize
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params)
{
    // number of tokens/expanded idx is bounded by total number of warps
    using OutputT = typename KernelParams::OutputT;
    using InputT = typename KernelParams::InputT;

    using BaseType = std::conditional_t<KernelParams::DoSoftmaxBeforeTopK, float, InputT>;
    using TypePacked = PackedScoreIdx<BaseType>;

    static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

    __shared__ TypePacked __attribute((aligned(128))) smemPackedScoreIdx[NumWarps * KernelParams::MaxNumTopExperts];

    uint32_t const clusterBlockRank = blockIdx.x;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();

    auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
    auto scoreOffset = warpTokenIdx * params.mNumExperts;
    bool validToken = warpTokenIdx < params.mNumTokens;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);

    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }

    if (params.mPtrScores != nullptr)
    {
        // in this case, each warp represents a token
        BaseType score[VecSize];
        int32_t idx[VecSize];

        BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
        int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];

        BaseType minScore = BaseType{-INFINITY};
        if (validToken)
        {
            routingTopKExperts<BaseType, InputT, VecSize, KernelParams::MaxNumTopExperts,
                KernelParams::DoSoftmaxBeforeTopK>(warp, score, idx, warpTopKScore, warpTopKExpertIdx, laneIdx,
                params.mNumExperts, params.mTopK, params.mPtrScores + scoreOffset, params.mNormTopkProb);

            if (laneIdx < params.mTopK)
            {
                smemPackedScoreIdx[warpIdx * params.mTopK + laneIdx]
                    = TypePacked{warpTopKScore[laneIdx], static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
            }
        } // end if (validToken)
    }

    // make packed scores available to all threads in cluster
    __cluster_barrier_arrive();
    __cluster_barrier_wait();

    if (params.mPtrScores != nullptr)
    {
        routingPermutation<KernelParams, BaseType, NumThreads, NumWarps, KernelParams::MaxNumTopExperts,
            /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
    }
    else
    {
        routingPermutation<KernelParams, BaseType, NumThreads, NumWarps, KernelParams::MaxNumTopExperts,
            /*LoadExpertIdxFromGlobal=*/true>(params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
    }
}
#else
__global__ void __launch_bounds__(NumThreads) routingIndicesClusterKernel(KernelParams /* params */)
{
    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

////////////////////////////////////////////////////////////////////////////////////////////////////

void launchClusterKernel(Data const& data, void* stream)
{
    LAUNCH_ROUTING_RENORNALIZE(data, false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
        /*smemSize=*/0, // No dynamic smem
        stream, data.mDoSoftmaxBeforeTopK);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingRenormalize
} // namespace moe::dev::routing
