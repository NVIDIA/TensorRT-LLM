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

#include "tensorrt_llm/thop/moe/loadBalance/moeRebalanceHaloQOp.h"

#include <torch/extension.h>

#include <algorithm>
#include <climits>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void checkCudaTensor(torch::Tensor const& tensor, c10::Device const& device, c10::ScalarType dtype,
    std::int64_t minElements, char const* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must be on the routes CUDA device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.numel() >= minElements, name, " is too small: expected at least ", minElements,
        " elements, got ", tensor.numel());
}

int checkedInt(std::int64_t value, char const* name)
{
    TORCH_CHECK(value >= INT_MIN && value <= INT_MAX, name, " does not fit in int32");
    return static_cast<int>(value);
}

} // namespace

kernels::MoeRebalanceHaloQParams makeMoeRebalanceHaloQParams(torch::Tensor const& routes, torch::Tensor& outSlots,
    torch::Tensor& outIds, torch::Tensor& outLevels, torch::Tensor& outOwners, torch::Tensor const& peerBases,
    torch::Tensor& status, torch::Tensor& partial, torch::Tensor& routeAux, torch::Tensor& gridSync,
    torch::Tensor& planWorkspace, torch::Tensor& routePrefix, std::int64_t epValue, std::int64_t expertsValue,
    std::int64_t helpersValue, std::int64_t localRankValue, std::int64_t routeCapacityValue, std::int64_t ctasValue,
    std::int64_t threadsValue, std::int64_t spinCycles, std::int64_t validRouteCountValue)
{
    int const ep = checkedInt(epValue, "ep");
    int const experts = checkedInt(expertsValue, "experts");
    int const helpers = checkedInt(helpersValue, "helpers");
    int const localRank = checkedInt(localRankValue, "local_rank");
    int const routeCapacity = checkedInt(routeCapacityValue, "route_capacity");
    int const ctas = checkedInt(ctasValue, "ctas");
    int const threads = checkedInt(threadsValue, "threads");
    int const validRouteCount = checkedInt(validRouteCountValue, "valid_route_count");
    TORCH_CHECK(ep >= 2 && ep <= 32 && ep % 2 == 0, "ep must be even and in [2, 32]");
    TORCH_CHECK(experts > 0 && experts <= 384 && experts % ep == 0,
        "experts must be positive, no greater than 384, and divisible by ep");
    TORCH_CHECK(helpers > 0 && helpers <= INT_MAX / ep - experts / ep, "invalid helpers");
    TORCH_CHECK(localRank >= 0 && localRank < ep, "local_rank must be in [0, ep)");
    TORCH_CHECK(routeCapacity > 0 && routeCapacity <= INT_MAX / ep, "invalid route_capacity");
    TORCH_CHECK(
        validRouteCount >= 0 && validRouteCount <= routeCapacity, "valid_route_count must be in [0, route_capacity]");
    TORCH_CHECK(ctas > 0 && ctas <= 128 && routeCapacity <= INT_MAX - ctas, "invalid ctas");
    TORCH_CHECK(threads == 512, "threads must be 512");
    TORCH_CHECK(spinCycles >= 0, "spin_cycles cannot be negative");

    c10::Device const device = routes.device();
    checkCudaTensor(routes, device, torch::kInt32, validRouteCount, "routes");
    checkCudaTensor(outSlots, device, torch::kInt32, routeCapacity, "out_slots");
    checkCudaTensor(outIds, device, torch::kInt32, helpers, "out_ids");
    checkCudaTensor(outLevels, device, torch::kInt32, helpers, "out_levels");
    checkCudaTensor(outOwners, device, torch::kInt32, helpers, "out_owners");
    checkCudaTensor(peerBases, device, torch::kInt64, ep, "peer_bases");
    checkCudaTensor(status, device, torch::kInt32, 6, "status");
    checkCudaTensor(partial, device, torch::kInt32, static_cast<std::int64_t>(ctas) * experts, "partial");
    std::int64_t const bins = static_cast<std::int64_t>(experts) + 1;
    std::int64_t routeAuxElements = static_cast<std::int64_t>(ctas) * (threads / 32) * bins;
    if (ctas > 1)
    {
        routeAuxElements = std::max(routeAuxElements, 2 + static_cast<std::int64_t>(ctas - 1) * bins + routeCapacity);
    }
    checkCudaTensor(routeAux, device, torch::kInt32, routeAuxElements, "route_aux");
    checkCudaTensor(gridSync, device, torch::kInt32, 4, "grid_sync");
    checkCudaTensor(planWorkspace, device, torch::kInt32, 2 + 6 * 384, "plan_workspace");
    checkCudaTensor(routePrefix, device, torch::kInt32, static_cast<std::int64_t>(384) * (ep + 1), "route_prefix");

    return kernels::MoeRebalanceHaloQParams{
        routes.data_ptr<int>(),
        outSlots.data_ptr<int>(),
        outIds.data_ptr<int>(),
        outLevels.data_ptr<int>(),
        outOwners.data_ptr<int>(),
        reinterpret_cast<std::uint64_t const*>(peerBases.data_ptr<std::int64_t>()),
        status.data_ptr<int>(),
        partial.data_ptr<int>(),
        routeAux.data_ptr<int>(),
        gridSync.data_ptr<int>(),
        planWorkspace.data_ptr<int>(),
        routePrefix.data_ptr<int>(),
        ep,
        experts,
        helpers,
        localRank,
        routeCapacity,
        ctas,
        threads,
        static_cast<std::uint64_t>(spinCycles),
        validRouteCount,
    };
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
