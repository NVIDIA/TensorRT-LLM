/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloR.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceRelay.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

//! Plan words of the shared per-layer dynamic EPLB layout and its per-field entry count.
constexpr std::int64_t kPlanWords = 2 + 6 * 384;
constexpr std::int64_t kPlanFieldStride = 384;

void checkCudaTensor(torch::Tensor const& tensor, c10::Device const& device, c10::ScalarType dtype,
    std::int64_t minElements, char const* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must be on the plan's CUDA device");
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

template <typename T>
T* checkedPointer(std::int64_t value, char const* name)
{
    TORCH_CHECK(value > 0 && value % static_cast<std::int64_t>(alignof(T)) == 0, name, " must be a valid aligned ",
        "device address");
    return reinterpret_cast<T*>(static_cast<std::uintptr_t>(value));
}

} // namespace

void moeRebalanceHaloR(torch::Tensor const& routes, torch::Tensor& outSlots, torch::Tensor& outIds,
    torch::Tensor& outOwners, torch::Tensor const& peerBases, torch::Tensor& status, torch::Tensor& partial,
    torch::Tensor& routeAux, torch::Tensor& gridSync, torch::Tensor& plan, torch::Tensor& routePrefix,
    std::int64_t readyFlagsPtr, std::int64_t epValue, std::int64_t expertsValue, std::int64_t helpersValue,
    std::int64_t sendCountValue, std::int64_t sendWeightValue, std::int64_t localRankValue,
    std::int64_t routeCapacityValue, std::int64_t ctasValue, std::int64_t spinNs, std::int64_t validRouteCountValue)
{
    int const ep = checkedInt(epValue, "ep");
    int const experts = checkedInt(expertsValue, "experts");
    int const helpers = checkedInt(helpersValue, "helpers");
    int const routeCapacity = checkedInt(routeCapacityValue, "route_capacity");
    int const ctas = checkedInt(ctasValue, "ctas");
    int const validRouteCount = checkedInt(validRouteCountValue, "valid_route_count");
    TORCH_CHECK(ep >= 2 && ep <= 32, "ep must be in [2, 32]");
    TORCH_CHECK(experts > 0 && experts <= kPlanFieldStride && experts % ep == 0,
        "experts must be positive, no greater than ", kPlanFieldStride, ", and divisible by ep");
    TORCH_CHECK(helpers >= 1 && helpers <= 4, "helpers must be in [1, 4]");
    TORCH_CHECK(ctas >= 2 && ctas <= 128, "ctas must be in [2, 128]");
    TORCH_CHECK(routeCapacity > 0 && validRouteCount >= 0 && validRouteCount <= routeCapacity,
        "valid_route_count must be in [0, route_capacity]");
    TORCH_CHECK(spinNs > 0, "spin_ns must be positive");

    c10::Device const device = plan.device();
    if (validRouteCount > 0)
    {
        checkCudaTensor(routes, device, torch::kInt32, validRouteCount, "routes");
    }
    checkCudaTensor(outSlots, device, torch::kInt32, routeCapacity, "out_slots");
    checkCudaTensor(outIds, device, torch::kInt32, helpers, "out_ids");
    checkCudaTensor(outOwners, device, torch::kInt32, helpers, "out_owners");
    checkCudaTensor(peerBases, device, torch::kInt64, ep, "peer_bases");
    checkCudaTensor(status, device, torch::kInt32, kernels::kHaloRStatusWords, "status");
    checkCudaTensor(partial, device, torch::kInt32, static_cast<std::int64_t>(ctas) * experts, "partial");
    checkCudaTensor(routeAux, device, torch::kInt32,
        kernels::getMoeRebalanceHaloRRouteAuxEntries(experts, ctas, routeCapacity), "route_aux");
    checkCudaTensor(gridSync, device, torch::kInt32, 4, "grid_sync");
    checkCudaTensor(plan, device, torch::kInt32, kPlanWords, "plan");
    checkCudaTensor(routePrefix, device, torch::kInt32, kPlanFieldStride * (ep + 1), "route_prefix");

    c10::cuda::CUDAGuard const deviceGuard(device);
    auto const stream = at::cuda::getCurrentCUDAStream(device.index());
    kernels::MoeRebalanceHaloRParams const params{
        validRouteCount > 0 ? routes.data_ptr<int>() : nullptr,
        outSlots.data_ptr<int>(),
        outIds.data_ptr<int>(),
        outOwners.data_ptr<int>(),
        reinterpret_cast<std::uint64_t const*>(peerBases.data_ptr<std::int64_t>()),
        status.data_ptr<int>(),
        partial.data_ptr<int>(),
        routeAux.data_ptr<int>(),
        gridSync.data_ptr<int>(),
        plan.data_ptr<int>(),
        routePrefix.data_ptr<int>(),
        checkedPointer<std::uint64_t>(readyFlagsPtr, "ready_flags_ptr"),
        ep,
        experts,
        helpers,
        checkedInt(sendCountValue, "send_count"),
        checkedInt(sendWeightValue, "send_weight"),
        checkedInt(localRankValue, "local_rank"),
        routeCapacity,
        validRouteCount,
        ctas,
        static_cast<std::uint64_t>(spinNs),
    };
    kernels::invokeMoeRebalanceHaloR(params, stream);
}

void moeRebalanceRelay(torch::Tensor const& plan, torch::Tensor const& sourceTable,
    torch::Tensor const& destinationTable, torch::Tensor const& planeBytes, torch::Tensor const& status,
    torch::Tensor& completedCtas, torch::Tensor& launchCount, std::int64_t readyFlagMcPtr, std::int64_t epValue,
    std::int64_t expertsValue, std::int64_t localRankValue, std::int64_t ctasValue)
{
    int const ep = checkedInt(epValue, "ep");
    int const experts = checkedInt(expertsValue, "experts");
    TORCH_CHECK(ep >= 2 && ep <= 32, "ep must be in [2, 32]");
    TORCH_CHECK(experts > 0 && experts % ep == 0, "experts must be positive and divisible by ep");
    int const planes = checkedInt(planeBytes.numel(), "planes");

    c10::Device const device = plan.device();
    checkCudaTensor(plan, device, torch::kInt32, kPlanWords, "plan");
    checkCudaTensor(sourceTable, device, torch::kInt64, static_cast<std::int64_t>(experts) * planes, "source_table");
    checkCudaTensor(
        destinationTable, device, torch::kInt64, static_cast<std::int64_t>(ep) * planes, "destination_table");
    checkCudaTensor(planeBytes, device, torch::kInt64, planes, "plane_bytes");
    checkCudaTensor(status, device, torch::kInt32, kernels::kHaloRStatusWords, "status");
    checkCudaTensor(completedCtas, device, torch::kInt32, 1, "completed_ctas");
    checkCudaTensor(launchCount, device, torch::kInt32, 1, "launch_count");

    c10::cuda::CUDAGuard const deviceGuard(device);
    auto const stream = at::cuda::getCurrentCUDAStream(device.index());
    kernels::MoeRebalanceRelayParams const params{
        plan.data_ptr<int>(),
        reinterpret_cast<std::uint64_t const*>(sourceTable.data_ptr<std::int64_t>()),
        reinterpret_cast<std::uint64_t const*>(destinationTable.data_ptr<std::int64_t>()),
        reinterpret_cast<std::uint64_t const*>(planeBytes.data_ptr<std::int64_t>()),
        status.data_ptr<int>(),
        reinterpret_cast<unsigned*>(completedCtas.data_ptr<int>()),
        reinterpret_cast<unsigned*>(launchCount.data_ptr<int>()),
        checkedPointer<std::uint64_t>(readyFlagMcPtr, "ready_flag_mc_ptr"),
        ep,
        experts,
        checkedInt(localRankValue, "local_rank"),
        planes,
        checkedInt(ctasValue, "ctas"),
    };
    kernels::invokeMoeRebalanceRelay(params, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_rebalance_halo_r(Tensor routes, Tensor(a!) out_slots, Tensor(b!) out_ids, Tensor(c!) out_owners, "
        "Tensor peer_bases, Tensor(d!) status, Tensor(e!) partial, Tensor(f!) route_aux, Tensor(g!) grid_sync, "
        "Tensor(h!) plan, Tensor(i!) route_prefix, int ready_flags_ptr, int ep, int experts, int helpers, "
        "int send_count, int send_weight, int local_rank, int route_capacity, int ctas, int spin_ns, "
        "int valid_route_count) -> ()");
    m.def(
        "moe_rebalance_relay(Tensor plan, Tensor source_table, Tensor destination_table, Tensor plane_bytes, "
        "Tensor status, Tensor(a!) completed_ctas, Tensor(b!) launch_count, int ready_flag_mc_ptr, int ep, "
        "int experts, int local_rank, int ctas) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_rebalance_halo_r", &tensorrt_llm::torch_ext::moeRebalanceHaloR);
    m.impl("moe_rebalance_relay", &tensorrt_llm::torch_ext::moeRebalanceRelay);
}
