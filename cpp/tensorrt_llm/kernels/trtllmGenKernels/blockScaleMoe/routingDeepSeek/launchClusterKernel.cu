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
#include "RoutingDeepSeekCommon.cuh"

namespace moe::dev::routing
{
namespace routingDeepSeek
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(KernelParams::MaxNumExperts)
    routingIndicesClusterKernel(KernelParams params)
{
    using OutputT = typename KernelParams::OutputT;

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const clusterBlockRank = blockIdx.x;

    //@todo: try to move it into routingPermutation
    // then wait on primary grid
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
    routingPermutation<KernelParams, OutputT, KernelParams::MaxNumExperts, KernelParams::MaxNumExperts / WarpSize,
        KernelParams::MaxNumTopExperts, /*LoadExpertIdxFromGlobal=*/true>(params, nullptr, warpIdx, clusterBlockRank);
}
#else
__global__ void routingIndicesClusterKernel(KernelParams params)
{
    assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

void launchClusterKernel(Data& data, int numThreadsHist, void* stream)
{
    LAUNCH_ROUTING_DEEPSEEK(data,
        /*coopLaunch=*/false, routingIndicesClusterKernel, NumBlocksPerCluster, numThreadsHist,
        /*smemSize=*/0, // No dynamic smem
        stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace routingDeepSeek
} // namespace moe::dev::routing
