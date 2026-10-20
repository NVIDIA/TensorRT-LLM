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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQ.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQKernel.cuh"

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

template <int FixedEp, int FixedExperts, int FixedHelpers>
__global__ __launch_bounds__(512, 1) void halo_q_scheduler_kernel(int const* routes, int* out_slots, int* out_ids,
    int* out_levels, int* out_owners, unsigned long long const* peer_bases, int* status, int* partial, int* route_aux,
    int* grid_sync, int* plan, int* route_prefix, int* plan_channel, int runtime_ep, int runtime_experts,
    int runtime_helpers, int local_rank, int route_capacity, int ctas, int enable_pdl, unsigned long long spin_cycles,
    int plan_abi_version, int plan_channel_words, int route_features, int capacity, int route_count)
{
    halo_q_scheduler_body<FixedEp, FixedExperts, FixedHelpers>(routes, out_slots, out_ids, out_levels, out_owners,
        peer_bases, status, partial, route_aux, grid_sync, plan, route_prefix, plan_channel, runtime_ep,
        runtime_experts, runtime_helpers, local_rank, route_capacity, ctas, enable_pdl, spin_cycles, plan_abi_version,
        plan_channel_words, route_features, capacity, route_count);
}

bool validPlanChannelLaunch(std::uintptr_t pointer, int abiVersion, int words, int features, int ep, int helpers)
{
    if (pointer == 0)
    {
        return abiVersion == 0 && words == 0 && features == 0;
    }
    if (pointer % 16 != 0 || helpers <= 0)
    {
        return false;
    }
    auto const stride = (static_cast<std::int64_t>(helpers) + 3) & ~std::int64_t{3};
    auto const expected = 4 + 7 * stride;
    return abiVersion == 6 && ep >= 2 && ep <= kMaxEp && ep % 2 == 0 && (features == 0 || features == 1)
        && expected <= INT_MAX && words == expected;
}

cudaError_t ensureDynamicSharedMemory(void const* kernel, int sharedBytes)
{
    int device = -1;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
        return error;

    using KernelCache = std::map<void const*, int, std::less<void const*>>;
    static std::mutex mutex;
    static std::map<int, KernelCache> configuredBytes;
    std::lock_guard<std::mutex> lock(mutex);
    auto& current = configuredBytes[device][kernel];
    if (current >= sharedBytes)
        return cudaSuccess;
    error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes);
    if (error == cudaSuccess)
        current = sharedBytes;
    return error;
}

} // namespace

void checkMoeRebalanceHaloQParams(MoeRebalanceHaloQParams const& params)
{
    TLLM_CHECK_WITH_INFO((params.routes != nullptr || params.validRouteCount == 0) && params.outSlots != nullptr
            && params.outIds != nullptr && params.outLevels != nullptr && params.outOwners != nullptr
            && params.peerBases != nullptr && params.status != nullptr && params.partial != nullptr
            && params.routeAux != nullptr && params.gridSync != nullptr && params.planWorkspace != nullptr
            && params.routePrefix != nullptr && params.ep >= 2 && params.ep <= kMaxEp && params.ep % 2 == 0
            && params.experts > 0 && params.experts <= kMaxExperts && params.experts % params.ep == 0
            && params.helpers > 0 && params.helpers <= INT_MAX / params.ep - params.experts / params.ep
            && params.localRank >= 0 && params.localRank < params.ep && params.validRouteCount >= 0
            && params.validRouteCount <= params.routeCapacity && params.routeCapacity > 0
            && params.routeCapacity <= INT_MAX / params.ep && params.routeCapacity <= INT_MAX - params.ctas
            && params.ctas > 0 && params.ctas <= 128 && params.threads == 512,
        "Invalid HALO-Q scheduler launch parameters");
    TLLM_CHECK_WITH_INFO(
        validPlanChannelLaunch(reinterpret_cast<std::uintptr_t>(params.planChannel), params.planAbiVersion,
            params.planChannelWords, params.routeFeatures, params.ep, params.helpers),
        "Invalid HALO-Q PlanChannel launch ABI");
}

void invokeMoeRebalanceHaloQ(MoeRebalanceHaloQParams const& params, cudaStream_t stream)
{
    checkMoeRebalanceHaloQParams(params);

    int capacity = broadcast_capacity(params.ep, params.experts, params.helpers);
    std::size_t const sharedBytes
        = static_cast<std::size_t>(halo_q_shared_ints(params.ep, params.experts, capacity, params.threads))
        * sizeof(int);

    void const* kernel = reinterpret_cast<void const*>(&halo_q_scheduler_kernel<0, 0, 0>);
    if (params.experts == 384 && params.helpers == 4)
    {
        if (params.ep == 8)
        {
            kernel = reinterpret_cast<void const*>(&halo_q_scheduler_kernel<8, 384, 4>);
        }
        else if (params.ep == 4)
        {
            kernel = reinterpret_cast<void const*>(&halo_q_scheduler_kernel<4, 384, 4>);
        }
    }
    if (sharedBytes > 48 * 1024)
    {
        TLLM_CUDA_CHECK(ensureDynamicSharedMemory(kernel, static_cast<int>(sharedBytes)));
    }

    auto routes = params.routes;
    auto outSlots = params.outSlots;
    auto outIds = params.outIds;
    auto outLevels = params.outLevels;
    auto outOwners = params.outOwners;
    auto peerBases = params.peerBases;
    auto status = params.status;
    auto partial = params.partial;
    auto routeAux = params.routeAux;
    auto gridSync = params.gridSync;
    auto planWorkspace = params.planWorkspace;
    auto routePrefix = params.routePrefix;
    auto planChannel = params.planChannel;
    int ep = params.ep;
    int experts = params.experts;
    int helpers = params.helpers;
    int localRank = params.localRank;
    int routeCapacity = params.routeCapacity;
    int ctas = params.ctas;
    int enablePdl = params.enablePdl ? 1 : 0;
    std::uint64_t spinCycles = params.spinCycles;
    int planAbiVersion = params.planAbiVersion;
    int planChannelWords = params.planChannelWords;
    int routeFeatures = params.routeFeatures;
    int validRouteCount = params.validRouteCount;
    void* arguments[]
        = {&routes, &outSlots, &outIds, &outLevels, &outOwners, &peerBases, &status, &partial, &routeAux, &gridSync,
            &planWorkspace, &routePrefix, &planChannel, &ep, &experts, &helpers, &localRank, &routeCapacity, &ctas,
            &enablePdl, &spinCycles, &planAbiVersion, &planChannelWords, &routeFeatures, &capacity, &validRouteCount};
    TLLM_CUDA_CHECK(
        cudaLaunchKernel(kernel, dim3(ctas, 1, 1), dim3(params.threads, 1, 1), arguments, sharedBytes, stream));
}

} // namespace kernels

TRTLLM_NAMESPACE_END
