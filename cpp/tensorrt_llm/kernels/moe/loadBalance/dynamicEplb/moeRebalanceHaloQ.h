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

#ifndef TRTLLM_KERNELS_MOE_REBALANCE_HALO_Q_H
#define TRTLLM_KERNELS_MOE_REBALANCE_HALO_Q_H

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

struct MoeRebalanceHaloQParams
{
    int const* routes;
    int* outSlots;
    int* outIds;
    int* outLevels;
    int* outOwners;
    std::uint64_t const* peerBases;
    int* status;
    int* partial;
    int* routeAux;
    int* gridSync;
    int* planWorkspace;
    int* routePrefix;
    int* planChannel;
    int ep;
    int experts;
    int helpers;
    int localRank;
    int routeCapacity;
    int ctas;
    int threads;
    bool enablePdl;
    std::uint64_t spinCycles;
    int planAbiVersion;
    int planChannelWords;
    int routeFeatures;
    int validRouteCount;
};

void invokeMoeRebalanceHaloQ(MoeRebalanceHaloQParams const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_KERNELS_MOE_REBALANCE_HALO_Q_H
