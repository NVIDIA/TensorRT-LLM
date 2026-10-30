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
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQ.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQKernel.cuh"

#include <climits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

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
}

} // namespace kernels

TRTLLM_NAMESPACE_END
