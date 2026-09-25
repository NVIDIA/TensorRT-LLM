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

#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceTma.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

void checkTmaResult(int error, char const* operation)
{
    TORCH_CHECK(error == 0, operation, " failed: ", megamoe_tma_copy_error_string(error), " (", error, ")");
}

int checkedInt(int64_t value, char const* name)
{
    TORCH_CHECK(value >= std::numeric_limits<int>::min() && value <= std::numeric_limits<int>::max(), name,
        " is outside the int32 range");
    return static_cast<int>(value);
}

uint64_t checkedUnsigned(int64_t value, char const* name)
{
    TORCH_CHECK(value >= 0, name, " must be nonnegative");
    return static_cast<uint64_t>(value);
}

std::vector<uint64_t> checkedU64Vector(
    std::vector<int64_t> const& values, uint64_t expectedSize, char const* name)
{
    TORCH_CHECK(expectedSize <= std::numeric_limits<size_t>::max(), name, " size overflows size_t");
    TORCH_CHECK(values.size() == static_cast<size_t>(expectedSize), name, " has ", values.size(),
        " entries; expected ", expectedSize);
    std::vector<uint64_t> result;
    result.reserve(values.size());
    for (int64_t value : values)
    {
        result.push_back(checkedUnsigned(value, name));
    }
    return result;
}

void checkBoundTensor(torch::Tensor const& tensor, int device, int64_t minimumElements, char const* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Int, name, " must have dtype torch.int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.get_device() == device, name, " must be on CUDA device ", device);
    TORCH_CHECK(tensor.numel() >= minimumElements, name, " has ", tensor.numel(), " elements; expected at least ",
        minimumElements);
}

} // namespace

class MoeRebalanceTmaState : public torch::CustomClassHolder
{
public:
    MoeRebalanceTmaState(int64_t maxSegments, int64_t sms, int64_t warps)
    {
        TORCH_CHECK(maxSegments > 0, "max_segments must be positive");
        auto const maxSegmentsUnsigned = checkedUnsigned(maxSegments, "max_segments");
        int const error = megamoe_tma_copy_create(
            maxSegmentsUnsigned, checkedInt(sms, "sms"), checkedInt(warps, "warps"), &mState);
        checkTmaResult(error, "moe rebalance TMA create");

        MegamoeTmaCopyConfig config{};
        int const configError = megamoe_tma_copy_config_info(mState, &config);
        if (configError != 0)
        {
            megamoe_tma_copy_destroy(&mState);
            checkTmaResult(configError, "moe rebalance TMA config query");
        }
        mDevice = config.device;
    }

    MoeRebalanceTmaState(MoeRebalanceTmaState const&) = delete;
    MoeRebalanceTmaState& operator=(MoeRebalanceTmaState const&) = delete;

    ~MoeRebalanceTmaState() noexcept override
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mState != nullptr)
        {
            // Destructors cannot report CUDA cleanup failures. Explicit destroy() is
            // available when the caller needs a synchronous error result.
            static_cast<void>(megamoe_tma_copy_destroy(&mState));
        }
    }

    void configureGpuPlan(int64_t world, int64_t rank, int64_t helperCount, int64_t planes,
        int64_t globalExperts, int64_t homeCount, int64_t ownerStride, int64_t levelCount,
        int64_t targetCount, int64_t planAbiVersion, int64_t planWords, int64_t routeFeatures,
        int64_t tmaRoute, int64_t tmaSourceLoadPercent, std::vector<int64_t> const& groupSizes,
        std::vector<int64_t> const& sourceTable, std::vector<int64_t> const& destinationTable,
        std::vector<int64_t> const& foreignDestinationTable, std::vector<int64_t> const& planeBytes)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireLive();
        TORCH_CHECK(!mConfigured, "moe rebalance TMA state is already configured");

        int const worldValue = checkedInt(world, "world");
        int const helperCountValue = checkedInt(helperCount, "helper_count");
        int const planesValue = checkedInt(planes, "planes");
        int const globalExpertsValue = checkedInt(globalExperts, "global_experts");
        int const levelCountValue = checkedInt(levelCount, "level_count");
        int const targetCountValue = checkedInt(targetCount, "target_count");
        int const routeFeaturesValue = checkedInt(routeFeatures, "route_features");
        TORCH_CHECK(worldValue > 0 && helperCountValue > 0 && planesValue > 0 && globalExpertsValue > 0,
            "world, helper_count, planes, and global_experts must be positive");
        TORCH_CHECK(worldValue <= MEGAMOE_TMA_GPU_PLAN_MAX_WORLD && planesValue <= 32 && globalExpertsValue <= 384,
            "world, planes, or global_experts exceeds the GPU plan ABI");
        TORCH_CHECK(helperCountValue <= (std::numeric_limits<int>::max() - 25) / 7,
            "helper_count exceeds the GPU plan ABI");
        TORCH_CHECK(levelCountValue > 0 && targetCountValue >= 0,
            "level_count must be positive and target_count must be nonnegative");
        TORCH_CHECK(targetCountValue <= 32 && (routeFeaturesValue == 0 || routeFeaturesValue == 1),
            "target_count or route_features exceeds the GPU plan ABI");
        TORCH_CHECK(levelCountValue <= MEGAMOE_TMA_GPU_PLAN_MAX_LEVELS,
            "level_count exceeds the GPU plan ABI");
        TORCH_CHECK(groupSizes.size() == static_cast<size_t>(levelCountValue),
            "group_sizes must contain level_count entries");

        std::array<int, MEGAMOE_TMA_GPU_PLAN_MAX_LEVELS> groups{};
        for (int index = 0; index < levelCountValue; ++index)
        {
            groups[index] = checkedInt(groupSizes[index], "group_sizes entry");
        }

        uint64_t const sourceCount = static_cast<uint64_t>(globalExpertsValue) * planesValue;
        uint64_t const destinationCount
            = static_cast<uint64_t>(levelCountValue) * helperCountValue * planesValue;
        uint64_t const foreignCount = routeFeaturesValue
            ? static_cast<uint64_t>(targetCountValue) * helperCountValue * planesValue
            : 0;
        auto sources = checkedU64Vector(sourceTable, sourceCount, "source_table");
        auto destinations = checkedU64Vector(destinationTable, destinationCount, "destination_table");
        auto foreign = checkedU64Vector(foreignDestinationTable, foreignCount, "foreign_destination_table");
        auto bytes = checkedU64Vector(planeBytes, planesValue, "plane_bytes");

        MegamoeTmaGpuPlanConfig config{};
        config.world = worldValue;
        config.rank = checkedInt(rank, "rank");
        config.helper_count = helperCountValue;
        config.planes = planesValue;
        config.global_experts = globalExpertsValue;
        config.home_count = checkedInt(homeCount, "home_count");
        config.owner_stride = checkedInt(ownerStride, "owner_stride");
        config.level_count = levelCountValue;
        config.target_count = targetCountValue;
        config.plan_abi_version = checkedInt(planAbiVersion, "plan_abi_version");
        config.plan_words = checkedInt(planWords, "plan_words");
        config.route_features = routeFeaturesValue;
        config.tma_route = checkedInt(tmaRoute, "tma_route");
        config.tma_source_load_percent = checkedInt(tmaSourceLoadPercent, "tma_source_load_percent");
        for (size_t index = 0; index < groups.size(); ++index)
        {
            config.group_sizes[index] = groups[index];
        }
        config.src_table = sources.data();
        config.dst_table = destinations.data();
        config.foreign_dst_table = foreign.empty() ? nullptr : foreign.data();
        config.plane_bytes = bytes.data();

        c10::cuda::CUDAGuard const deviceGuard(c10::Device(c10::DeviceType::CUDA, mDevice));
        checkTmaResult(
            megamoe_tma_copy_configure_gpu_plan(mState, &config), "moe rebalance TMA GPU plan configure");
        mConfigured = true;
        mHelperCount = helperCountValue;
    }

    void bindGpuDirect(torch::Tensor ids, torch::Tensor levels, torch::Tensor owners, torch::Tensor workspace,
        int64_t capacity)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireLive();
        TORCH_CHECK(mConfigured, "moe rebalance TMA state must be configured before bind");
        TORCH_CHECK(!mBound, "moe rebalance TMA state is already bound");
        int const capacityValue = checkedInt(capacity, "capacity");
        TORCH_CHECK(capacityValue > 0, "capacity must be positive");
        checkBoundTensor(ids, mDevice, mHelperCount, "ids");
        checkBoundTensor(levels, mDevice, mHelperCount, "levels");
        checkBoundTensor(owners, mDevice, mHelperCount, "owners");
        checkBoundTensor(workspace, mDevice, 2 + 6LL * capacityValue, "workspace");

        c10::cuda::CUDAGuard const deviceGuard(ids.device());
        checkTmaResult(megamoe_tma_copy_bind_gpu_direct(mState, reinterpret_cast<uint64_t>(ids.data_ptr()),
                           reinterpret_cast<uint64_t>(levels.data_ptr()),
                           reinterpret_cast<uint64_t>(owners.data_ptr()),
                           reinterpret_cast<uint64_t>(workspace.data_ptr()), capacityValue),
            "moe rebalance TMA GPU-direct bind");

        // The native state stores raw device addresses. Retaining the tensors in
        // the owner prevents allocator reuse until destroy() has drained work.
        mIds = std::move(ids);
        mLevels = std::move(levels);
        mOwners = std::move(owners);
        mWorkspace = std::move(workspace);
        mBound = true;
    }

    int64_t submitGpuDirect(int64_t flagMc)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireBound();
        c10::cuda::CUDAGuard const deviceGuard(c10::Device(c10::DeviceType::CUDA, mDevice));
        auto const stream = at::cuda::getCurrentCUDAStream(mDevice).stream();
        submitLocked(flagMc, reinterpret_cast<void*>(stream));
        return 0;
    }

    int64_t currentGeneration() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireLive();
        return checkedDiagnostic(mGeneration);
    }

    // Result order: error, direct/scatter slots, four domain counts, segment/range/slice counts,
    // weighted total/quota/local slices. This diagnostic synchronizes the last submitted stream.

    std::vector<int64_t> result()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireLive();
        TORCH_CHECK(mConfigured, "moe rebalance TMA state is not configured");
        c10::cuda::CUDAGuard const deviceGuard(c10::Device(c10::DeviceType::CUDA, mDevice));
        MegamoeTmaGpuPlanResult value{};
        checkTmaResult(megamoe_tma_copy_gpu_plan_result(mState, &value), "moe rebalance TMA diagnostic result");
        return {value.error, value.direct_slots, value.scatter_slots, value.weighted_domains, value.uniform_domains,
            value.mixed_source_domains, value.foreign_source_domains, checkedDiagnostic(value.segment_count),
            checkedDiagnostic(value.range_count), checkedDiagnostic(value.total_slices),
            checkedDiagnostic(value.weighted_total_slices), checkedDiagnostic(value.weighted_source_quota_slices),
            checkedDiagnostic(value.weighted_local_slices)};
    }

    // Config order follows MegamoeTmaCopyConfig through max_segments.

    std::vector<int64_t> config() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireLive();
        MegamoeTmaCopyConfig value{};
        checkTmaResult(megamoe_tma_copy_config_info(mState, &value), "moe rebalance TMA config query");
        return {value.abi_version, value.device, value.device_sm_count, value.sms, value.warps, value.threads_per_cta,
            value.slots_per_warp, value.bank0_slots_per_warp, value.bank1_slots_per_warp, value.slice_bytes,
            value.dynamic_shared_bytes, value.device_optin_shared_bytes, value.device_shared_bytes_per_sm,
            value.max_active_ctas_per_sm, value.compute_major, value.compute_minor, value.total_slots,
            value.max_slots_per_warp, value.extra_slot_warps, value.max_warps,
            checkedDiagnostic(value.max_segments)};
    }

    void destroy()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mState == nullptr)
        {
            return;
        }
        c10::cuda::CUDAGuard const deviceGuard(c10::Device(c10::DeviceType::CUDA, mDevice));
        int const error = megamoe_tma_copy_destroy(&mState);
        if (mState == nullptr)
        {
            mIds = torch::Tensor();
            mLevels = torch::Tensor();
            mOwners = torch::Tensor();
            mWorkspace = torch::Tensor();
            mConfigured = false;
            mBound = false;
        }
        checkTmaResult(error, "moe rebalance TMA destroy");
    }

private:
    void requireLive() const
    {
        TORCH_CHECK(mState != nullptr, "moe rebalance TMA state has been destroyed");
    }

    void requireBound() const
    {
        requireLive();
        TORCH_CHECK(mBound, "moe rebalance TMA state must be bound before submit");
    }

    static int64_t checkedDiagnostic(uint64_t value)
    {
        TORCH_CHECK(value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "moe rebalance TMA diagnostic counter exceeds int64");
        return static_cast<int64_t>(value);
    }

    void submitLocked(int64_t flagMc, void* stream)
    {
        uint64_t const flag = checkedUnsigned(flagMc, "flag_mc");
        TORCH_CHECK(flag != 0, "flag_mc must be nonzero");
        checkTmaResult(megamoe_tma_copy_submit_gpu_direct(mState, flag, mGeneration, stream),
            "moe rebalance TMA GPU-direct submit");
        ++mGeneration;
        TORCH_CHECK(mGeneration != 0, "moe rebalance TMA generation overflow");
    }

    mutable std::mutex mMutex;
    MegamoeTmaCopyState* mState{nullptr};
    int mDevice{-1};
    int mHelperCount{0};
    uint64_t mGeneration{1};
    bool mConfigured{false};
    bool mBound{false};
    torch::Tensor mIds;
    torch::Tensor mLevels;
    torch::Tensor mOwners;
    torch::Tensor mWorkspace;
};

using MoeRebalanceTmaStatePtr = c10::intrusive_ptr<MoeRebalanceTmaState>;

MoeRebalanceTmaStatePtr moeRebalanceTmaCreate(int64_t maxSegments, int64_t sms, int64_t warps)
{
    return c10::make_intrusive<MoeRebalanceTmaState>(maxSegments, sms, warps);
}

void moeRebalanceTmaConfigureGpuPlan(MoeRebalanceTmaStatePtr const& state, int64_t world, int64_t rank,
    int64_t helperCount, int64_t planes, int64_t globalExperts, int64_t homeCount, int64_t ownerStride,
    int64_t levelCount, int64_t targetCount, int64_t planAbiVersion, int64_t planWords, int64_t routeFeatures,
    int64_t tmaRoute, int64_t tmaSourceLoadPercent, std::vector<int64_t> const& groupSizes,
    std::vector<int64_t> const& sourceTable, std::vector<int64_t> const& destinationTable,
    std::vector<int64_t> const& foreignDestinationTable, std::vector<int64_t> const& planeBytes)
{
    TORCH_CHECK(state, "moe rebalance TMA state is null");
    state->configureGpuPlan(world, rank, helperCount, planes, globalExperts, homeCount, ownerStride, levelCount,
        targetCount, planAbiVersion, planWords, routeFeatures, tmaRoute, tmaSourceLoadPercent, groupSizes, sourceTable,
        destinationTable, foreignDestinationTable, planeBytes);
}

void moeRebalanceTmaBindGpuDirect(MoeRebalanceTmaStatePtr const& state, torch::Tensor ids, torch::Tensor levels,
    torch::Tensor owners, torch::Tensor workspace, int64_t capacity)
{
    TORCH_CHECK(state, "moe rebalance TMA state is null");
    state->bindGpuDirect(std::move(ids), std::move(levels), std::move(owners), std::move(workspace), capacity);
}

int64_t moeRebalanceTmaSubmitGpuDirect(MoeRebalanceTmaStatePtr const& state, int64_t flagMc)
{
    TORCH_CHECK(state, "moe rebalance TMA state is null");
    return state->submitGpuDirect(flagMc);
}

int64_t moeRebalanceTmaCurrentGeneration(MoeRebalanceTmaStatePtr const& state)
{
    TORCH_CHECK(state, "moe rebalance TMA state is null");
    return state->currentGeneration();
}

std::vector<int64_t> moeRebalanceTmaResult(MoeRebalanceTmaStatePtr const& state)
{
    TORCH_CHECK(state, "moe rebalance TMA state is null");
    return state->result();
}

std::vector<int64_t> moeRebalanceTmaConfig(MoeRebalanceTmaStatePtr const& state)
{
    TORCH_CHECK(state, "moe rebalance TMA state is null");
    return state->config();
}

void moeRebalanceTmaDestroy(MoeRebalanceTmaStatePtr const& state)
{
    TORCH_CHECK(state, "moe rebalance TMA state is null");
    state->destroy();
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::MoeRebalanceTmaState>("MoeRebalanceTmaState");
    m.def("moe_rebalance_tma_create(int max_segments, int sms=0, int warps=0) -> "
          "__torch__.torch.classes.trtllm.MoeRebalanceTmaState");
    m.def("moe_rebalance_tma_configure_gpu_plan("
          "__torch__.torch.classes.trtllm.MoeRebalanceTmaState state, int world, int rank, int helper_count, "
          "int planes, int global_experts, int home_count, int owner_stride, int level_count, int target_count, "
          "int plan_abi_version, int plan_words, int route_features, int tma_route, int tma_source_load_percent, "
          "int[] group_sizes, int[] source_table, int[] destination_table, int[] foreign_destination_table, "
          "int[] plane_bytes) -> ()");
    m.def("moe_rebalance_tma_bind_gpu_direct("
          "__torch__.torch.classes.trtllm.MoeRebalanceTmaState state, Tensor ids, Tensor levels, Tensor owners, "
          "Tensor workspace, int capacity) -> ()");
    m.def("moe_rebalance_tma_submit_gpu_direct("
          "__torch__.torch.classes.trtllm.MoeRebalanceTmaState state, int flag_mc) -> int");
    m.def("moe_rebalance_tma_current_gen("
          "__torch__.torch.classes.trtllm.MoeRebalanceTmaState state) -> int");
    m.def("moe_rebalance_tma_result("
          "__torch__.torch.classes.trtllm.MoeRebalanceTmaState state) -> int[]");
    m.def("moe_rebalance_tma_config("
          "__torch__.torch.classes.trtllm.MoeRebalanceTmaState state) -> int[]");
    m.def("moe_rebalance_tma_destroy("
          "__torch__.torch.classes.trtllm.MoeRebalanceTmaState state) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("moe_rebalance_tma_create", &tensorrt_llm::torch_ext::moeRebalanceTmaCreate);
    m.impl("moe_rebalance_tma_configure_gpu_plan", &tensorrt_llm::torch_ext::moeRebalanceTmaConfigureGpuPlan);
    m.impl("moe_rebalance_tma_bind_gpu_direct", &tensorrt_llm::torch_ext::moeRebalanceTmaBindGpuDirect);
    m.impl("moe_rebalance_tma_submit_gpu_direct", &tensorrt_llm::torch_ext::moeRebalanceTmaSubmitGpuDirect);
    m.impl("moe_rebalance_tma_current_gen", &tensorrt_llm::torch_ext::moeRebalanceTmaCurrentGeneration);
    m.impl("moe_rebalance_tma_result", &tensorrt_llm::torch_ext::moeRebalanceTmaResult);
    m.impl("moe_rebalance_tma_config", &tensorrt_llm::torch_ext::moeRebalanceTmaConfig);
    m.impl("moe_rebalance_tma_destroy", &tensorrt_llm::torch_ext::moeRebalanceTmaDestroy);
}
