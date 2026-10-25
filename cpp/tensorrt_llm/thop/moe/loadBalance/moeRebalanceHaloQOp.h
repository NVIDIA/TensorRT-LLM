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

#pragma once

#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloQ.h"

#include <torch/extension.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Validates the HALO-Q scheduler arguments that moe_rebalance_tma_bind_fused binds and returns its launch parameters.
kernels::MoeRebalanceHaloQParams makeMoeRebalanceHaloQParams(torch::Tensor const& routes, torch::Tensor& outSlots,
    torch::Tensor& outIds, torch::Tensor& outLevels, torch::Tensor& outOwners, torch::Tensor const& peerBases,
    torch::Tensor& status, torch::Tensor& partial, torch::Tensor& routeAux, torch::Tensor& gridSync,
    torch::Tensor& planWorkspace, torch::Tensor& routePrefix, std::int64_t ep, std::int64_t experts,
    std::int64_t helpers, std::int64_t localRank, std::int64_t routeCapacity, std::int64_t ctas, std::int64_t threads,
    std::int64_t spinCycles, std::int64_t validRouteCount);

} // namespace torch_ext

TRTLLM_NAMESPACE_END
