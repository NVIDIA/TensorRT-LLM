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

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// The HALO-Q scheduler phase of the fused scheduler + in-switch copy kernel
// (megamoe_tma_copy_submit_fused).
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
    int ep;
    int experts;
    int helpers;
    int localRank;
    int routeCapacity;
    int ctas;
    int threads;
    std::uint64_t spinCycles;
    int validRouteCount;
};

// Throws unless params describe a valid scheduler launch.
void checkMoeRebalanceHaloQParams(MoeRebalanceHaloQParams const& params);

} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // TRTLLM_KERNELS_MOE_REBALANCE_HALO_Q_H
