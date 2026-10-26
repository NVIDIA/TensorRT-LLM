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
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/custom_class.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

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

void checkCudaResult(cudaError_t error, char const* operation)
{
    TORCH_CHECK(error == cudaSuccess, operation, " failed: ", cudaGetErrorString(error));
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

//! One layer's HALO-R scheduler and relay copy on one EP rank. It binds every argument the per-layer launches share,
//! the copy stream (COPY) and the four events that fork COPY from the calling stream (MAIN) and join it back, so a
//! layer call makes one native call per step:
//!
//!     submit:      MAIN record input_ready; COPY wait input_ready -> scheduler -> record plan_ready -> relay copy
//!     wait_routes: MAIN wait plan_ready
//!     release:     COPY wait the consumer (consumer_done on MAIN, or the caller's event) -> record copy_idle;
//!                  MAIN wait copy_idle
class MoeRebalanceHaloRState : public torch::CustomClassHolder
{
public:
    MoeRebalanceHaloRState(torch::Tensor outSlots, torch::Tensor outIds, torch::Tensor outOwners,
        torch::Tensor peerBases, torch::Tensor status, torch::Tensor partial, torch::Tensor routeAux,
        torch::Tensor gridSync, torch::Tensor plan, torch::Tensor routePrefix, std::int64_t readyFlagsPtr,
        torch::Tensor relaySources, torch::Tensor relayDestinations, torch::Tensor relayPlaneBytes,
        torch::Tensor relayCompletedCtas, torch::Tensor relayLaunchCount, std::int64_t readyFlagMcPtr,
        std::int64_t epValue, std::int64_t expertsValue, std::int64_t helpersValue, std::int64_t sendCountValue,
        std::int64_t sendWeightValue, std::int64_t localRankValue, std::int64_t topkValue,
        std::int64_t routeCapacityValue, std::int64_t ctasValue, std::int64_t relayCtasValue, std::int64_t copyStream)
    {
        int const ep = checkedInt(epValue, "ep");
        int const experts = checkedInt(expertsValue, "experts");
        int const helpers = checkedInt(helpersValue, "helpers");
        int const topk = checkedInt(topkValue, "topk");
        int const routeCapacity = checkedInt(routeCapacityValue, "route_capacity");
        int const ctas = checkedInt(ctasValue, "ctas");
        TORCH_CHECK(ep >= 2 && ep <= 32, "ep must be in [2, 32]");
        TORCH_CHECK(experts > 0 && experts <= kPlanFieldStride && experts % ep == 0,
            "experts must be positive, no greater than ", kPlanFieldStride, ", and divisible by ep");
        TORCH_CHECK(helpers >= 1 && helpers <= 4, "helpers must be in [1, 4]");
        TORCH_CHECK(ctas >= 2 && ctas <= 128, "ctas must be in [2, 128]");
        TORCH_CHECK(topk > 0 && routeCapacity > 0 && routeCapacity % topk == 0,
            "route_capacity must be a positive multiple of topk");
        TORCH_CHECK(copyStream != 0, "copy_stream must be a non-default CUDA stream");
        int const planes = checkedInt(relayPlaneBytes.numel(), "planes");

        c10::Device const device = plan.device();
        TORCH_CHECK(device.is_cuda(), "plan must be a CUDA tensor");
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
        checkCudaTensor(
            relaySources, device, torch::kInt64, static_cast<std::int64_t>(experts) * planes, "source_table");
        checkCudaTensor(
            relayDestinations, device, torch::kInt64, static_cast<std::int64_t>(ep) * planes, "destination_table");
        checkCudaTensor(relayPlaneBytes, device, torch::kInt64, planes, "plane_bytes");
        checkCudaTensor(relayCompletedCtas, device, torch::kInt32, 1, "completed_ctas");
        checkCudaTensor(relayLaunchCount, device, torch::kInt32, 1, "launch_count");

        mScheduler = kernels::MoeRebalanceHaloRParams{
            nullptr,
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
            0,
            ctas,
            0,
        };
        mRelay = kernels::MoeRebalanceRelayParams{
            plan.data_ptr<int>(),
            reinterpret_cast<std::uint64_t const*>(relaySources.data_ptr<std::int64_t>()),
            reinterpret_cast<std::uint64_t const*>(relayDestinations.data_ptr<std::int64_t>()),
            reinterpret_cast<std::uint64_t const*>(relayPlaneBytes.data_ptr<std::int64_t>()),
            status.data_ptr<int>(),
            reinterpret_cast<unsigned*>(relayCompletedCtas.data_ptr<int>()),
            reinterpret_cast<unsigned*>(relayLaunchCount.data_ptr<int>()),
            checkedPointer<std::uint64_t>(readyFlagMcPtr, "ready_flag_mc_ptr"),
            ep,
            experts,
            checkedInt(localRankValue, "local_rank"),
            planes,
            checkedInt(relayCtasValue, "relay_ctas"),
        };
        mDevice = device.index();
        mTopk = topk;
        mCopyStream = reinterpret_cast<cudaStream_t>(copyStream);

        c10::cuda::CUDAGuard const deviceGuard(device);
        for (cudaEvent_t& event : mEvents)
        {
            cudaError_t const error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            if (error != cudaSuccess)
            {
                event = nullptr;
                releaseEvents();
                checkCudaResult(error, "cudaEventCreate(HALO-R)");
            }
        }
        // The launches read and write these by address; retaining them prevents allocator reuse.
        mTensors = {std::move(outSlots), std::move(outIds), std::move(outOwners), std::move(peerBases),
            std::move(status), std::move(partial), std::move(routeAux), std::move(gridSync), std::move(plan),
            std::move(routePrefix), std::move(relaySources), std::move(relayDestinations), std::move(relayPlaneBytes),
            std::move(relayCompletedCtas), std::move(relayLaunchCount)};
        mBound = true;
    }

    MoeRebalanceHaloRState(MoeRebalanceHaloRState const&) = delete;
    MoeRebalanceHaloRState& operator=(MoeRebalanceHaloRState const&) = delete;

    ~MoeRebalanceHaloRState() noexcept override
    {
        releaseEvents();
    }

    //! Forks COPY from the current stream and enqueues the scheduler, the plan_ready record and the relay copy.
    void submit(torch::Tensor const& routes, std::int64_t spinNs)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireBound();
        TORCH_CHECK_VALUE(routes.is_cuda() && routes.get_device() == mDevice
                && routes.scalar_type() == at::ScalarType::Int && routes.is_contiguous() && routes.dim() == 2
                && routes.size(1) == mTopk && routes.numel() <= mScheduler.routeCapacity,
            "routes must be a contiguous CUDA int32 [tokens, ", mTopk, "] tensor on device ", mDevice, " with at most ",
            mScheduler.routeCapacity / mTopk, " tokens");
        TORCH_CHECK_VALUE(spinNs > 0, "spin_ns must be positive");
        auto scheduler = mScheduler;
        scheduler.validRouteCount = static_cast<int>(routes.numel());
        scheduler.routes = scheduler.validRouteCount > 0 ? routes.data_ptr<int>() : nullptr;
        scheduler.spinNs = static_cast<std::uint64_t>(spinNs);

        c10::cuda::CUDAGuard const deviceGuard(c10::Device(c10::DeviceType::CUDA, mDevice));
        cudaStream_t const execution = currentExecutionStream();
        cudaStreamCaptureStatus capture{cudaStreamCaptureStatusNone};
        checkCudaResult(cudaStreamIsCapturing(execution, &capture), "cudaStreamIsCapturing(HALO-R)");
        if (capture == cudaStreamCaptureStatusNone)
        {
            // COPY reads the routes after MAIN may free them. A captured route tensor lives in the graph's pool for
            // the graph's lifetime.
            c10::cuda::CUDACachingAllocator::recordStream(routes.storage().data_ptr(),
                c10::cuda::getStreamFromExternal(mCopyStream, static_cast<c10::DeviceIndex>(mDevice)));
        }
        checkCudaResult(cudaEventRecord(mEvents[kInputReady], execution), "cudaEventRecord(HALO-R input ready)");
        checkCudaResult(
            cudaStreamWaitEvent(mCopyStream, mEvents[kInputReady], 0), "cudaStreamWaitEvent(HALO-R input ready)");
        kernels::invokeMoeRebalanceHaloR(scheduler, mCopyStream);
        checkCudaResult(cudaEventRecord(mEvents[kPlanReady], mCopyStream), "cudaEventRecord(HALO-R plan ready)");
        kernels::invokeMoeRebalanceRelay(mRelay, mCopyStream);
    }

    //! Orders the current stream after the scheduler of the last submit, without a host wait.
    void waitRoutes()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireBound();
        c10::cuda::CUDAGuard const deviceGuard(c10::Device(c10::DeviceType::CUDA, mDevice));
        cudaStream_t const execution = currentExecutionStream();
        checkCudaResult(
            cudaStreamWaitEvent(execution, mEvents[kPlanReady], 0), "cudaStreamWaitEvent(HALO-R plan ready)");
    }

    //! Orders COPY after the consumer: `completionEvent` when nonzero, otherwise the work already queued on the
    //! current stream. Then joins COPY back into the current stream.
    void release(std::int64_t completionEvent)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireBound();
        c10::cuda::CUDAGuard const deviceGuard(c10::Device(c10::DeviceType::CUDA, mDevice));
        cudaStream_t const execution = currentExecutionStream();
        cudaEvent_t consumer = mEvents[kConsumerDone];
        if (completionEvent == 0)
        {
            checkCudaResult(cudaEventRecord(consumer, execution), "cudaEventRecord(HALO-R consumer done)");
        }
        else
        {
            consumer = reinterpret_cast<cudaEvent_t>(completionEvent);
        }
        checkCudaResult(cudaStreamWaitEvent(mCopyStream, consumer, 0), "cudaStreamWaitEvent(HALO-R consumer)");
        checkCudaResult(cudaEventRecord(mEvents[kCopyIdle], mCopyStream), "cudaEventRecord(HALO-R copy idle)");
        checkCudaResult(cudaStreamWaitEvent(execution, mEvents[kCopyIdle], 0), "cudaStreamWaitEvent(HALO-R copy idle)");
    }

    //! Releases the events and the bound tensors; the caller has drained the device work that uses them.
    void destroy()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        releaseEvents();
        mTensors.clear();
        mCopyStream = nullptr;
        mBound = false;
    }

private:
    enum EventIndex : int
    {
        kInputReady = 0,
        kPlanReady = 1,
        kConsumerDone = 2,
        kCopyIdle = 3,
        kEventCount = 4,
    };

    void requireBound() const
    {
        TORCH_CHECK(mBound, "HALO-R state is destroyed");
    }

    // The stream whose queued work a launch or release follows; it must differ from the copy stream.
    cudaStream_t currentExecutionStream() const
    {
        cudaStream_t const stream = at::cuda::getCurrentCUDAStream(mDevice).stream();
        TORCH_CHECK(stream != mCopyStream, "the current stream must differ from the bound copy stream");
        return stream;
    }

    void releaseEvents() noexcept
    {
        // Pending records complete normally after an event is destroyed.
        for (cudaEvent_t& event : mEvents)
        {
            if (event != nullptr)
            {
                static_cast<void>(cudaEventDestroy(event));
                event = nullptr;
            }
        }
    }

    mutable std::mutex mMutex;
    kernels::MoeRebalanceHaloRParams mScheduler{};
    kernels::MoeRebalanceRelayParams mRelay{};
    std::vector<torch::Tensor> mTensors;
    int mDevice{-1};
    int mTopk{0};
    cudaStream_t mCopyStream{nullptr};
    cudaEvent_t mEvents[kEventCount]{};
    bool mBound{false};
};

using MoeRebalanceHaloRStatePtr = c10::intrusive_ptr<MoeRebalanceHaloRState>;

MoeRebalanceHaloRStatePtr moeRebalanceHaloRCreate(torch::Tensor outSlots, torch::Tensor outIds, torch::Tensor outOwners,
    torch::Tensor peerBases, torch::Tensor status, torch::Tensor partial, torch::Tensor routeAux,
    torch::Tensor gridSync, torch::Tensor plan, torch::Tensor routePrefix, std::int64_t readyFlagsPtr,
    torch::Tensor relaySources, torch::Tensor relayDestinations, torch::Tensor relayPlaneBytes,
    torch::Tensor relayCompletedCtas, torch::Tensor relayLaunchCount, std::int64_t readyFlagMcPtr, std::int64_t ep,
    std::int64_t experts, std::int64_t helpers, std::int64_t sendCount, std::int64_t sendWeight, std::int64_t localRank,
    std::int64_t topk, std::int64_t routeCapacity, std::int64_t ctas, std::int64_t relayCtas, std::int64_t copyStream)
{
    return c10::make_intrusive<MoeRebalanceHaloRState>(std::move(outSlots), std::move(outIds), std::move(outOwners),
        std::move(peerBases), std::move(status), std::move(partial), std::move(routeAux), std::move(gridSync),
        std::move(plan), std::move(routePrefix), readyFlagsPtr, std::move(relaySources), std::move(relayDestinations),
        std::move(relayPlaneBytes), std::move(relayCompletedCtas), std::move(relayLaunchCount), readyFlagMcPtr, ep,
        experts, helpers, sendCount, sendWeight, localRank, topk, routeCapacity, ctas, relayCtas, copyStream);
}

// The per-layer ops take this handle instead of the custom class, whose conversion costs several microseconds of
// host time per operator call. The caller keeps the state alive while it uses the handle.
std::int64_t moeRebalanceHaloRHandle(MoeRebalanceHaloRStatePtr const& state)
{
    TORCH_CHECK(state, "HALO-R state is null");
    return reinterpret_cast<std::int64_t>(state.get());
}

namespace
{

MoeRebalanceHaloRState& stateFromHandle(std::int64_t handle)
{
    TORCH_CHECK(handle != 0, "HALO-R handle is null");
    return *reinterpret_cast<MoeRebalanceHaloRState*>(handle);
}

} // namespace

void moeRebalanceHaloRSubmit(std::int64_t handle, torch::Tensor const& routes, std::int64_t spinNs)
{
    stateFromHandle(handle).submit(routes, spinNs);
}

void moeRebalanceHaloRWaitRoutes(std::int64_t handle)
{
    stateFromHandle(handle).waitRoutes();
}

void moeRebalanceHaloRRelease(std::int64_t handle, std::int64_t completionEvent)
{
    stateFromHandle(handle).release(completionEvent);
}

void moeRebalanceHaloRDestroy(MoeRebalanceHaloRStatePtr const& state)
{
    TORCH_CHECK(state, "HALO-R state is null");
    state->destroy();
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::MoeRebalanceHaloRState>("MoeRebalanceHaloRState");
    m.def(
        "moe_rebalance_halo_r_create(Tensor out_slots, Tensor out_ids, Tensor out_owners, Tensor peer_bases, "
        "Tensor status, Tensor partial, Tensor route_aux, Tensor grid_sync, Tensor plan, Tensor route_prefix, "
        "int ready_flags_ptr, Tensor relay_sources, Tensor relay_destinations, Tensor relay_plane_bytes, "
        "Tensor relay_completed_ctas, Tensor relay_launch_count, int ready_flag_mc_ptr, int ep, int experts, "
        "int helpers, int send_count, int send_weight, int local_rank, int topk, int route_capacity, int ctas, "
        "int relay_ctas, int copy_stream) -> __torch__.torch.classes.trtllm.MoeRebalanceHaloRState");
    m.def(
        "moe_rebalance_halo_r_handle("
        "__torch__.torch.classes.trtllm.MoeRebalanceHaloRState state) -> int");
    m.def("moe_rebalance_halo_r_submit(int handle, Tensor routes, int spin_ns) -> ()");
    m.def("moe_rebalance_halo_r_wait_routes(int handle) -> ()");
    m.def("moe_rebalance_halo_r_release(int handle, int completion_event) -> ()");
    m.def(
        "moe_rebalance_halo_r_destroy("
        "__torch__.torch.classes.trtllm.MoeRebalanceHaloRState state) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("moe_rebalance_halo_r_create", &tensorrt_llm::torch_ext::moeRebalanceHaloRCreate);
    m.impl("moe_rebalance_halo_r_handle", &tensorrt_llm::torch_ext::moeRebalanceHaloRHandle);
    m.impl("moe_rebalance_halo_r_submit", &tensorrt_llm::torch_ext::moeRebalanceHaloRSubmit);
    m.impl("moe_rebalance_halo_r_wait_routes", &tensorrt_llm::torch_ext::moeRebalanceHaloRWaitRoutes);
    m.impl("moe_rebalance_halo_r_release", &tensorrt_llm::torch_ext::moeRebalanceHaloRRelease);
    m.impl("moe_rebalance_halo_r_destroy", &tensorrt_llm::torch_ext::moeRebalanceHaloRDestroy);
}
