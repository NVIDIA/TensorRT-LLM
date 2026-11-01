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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/custom_class.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>
#include <mutex>
#include <optional>
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

//! One layer's fused HALO-R kernel on one EP rank (moeRebalanceHaloR.h). It binds every argument the per-layer
//! launches share, the copy stream (COPY) and the event that forks COPY from the calling stream (MAIN), and prepares
//! the launch, so that a layer call makes one native call:
//!
//!     submit_fused: MAIN record input_ready; COPY wait input_ready -> fused kernel
//!
//! The caller then waits on MAIN for the route counter, which the kernel raises once per CTA, before it reads the
//! routes. Every launch follows the MAIN work queued before it, which includes the previous consumer of the helper
//! slots this launch overwrites.
//!
//! A CUDA error in any call poisons the context: every later submit fails with it, and the owner rebuilds the group.
class MoeRebalanceHaloRState : public torch::CustomClassHolder
{
public:
    MoeRebalanceHaloRState(torch::Tensor outSlots, torch::Tensor outIds, torch::Tensor outOwners,
        torch::Tensor peerBases, torch::Tensor status, torch::Tensor partial, torch::Tensor routeAux,
        torch::Tensor gridSync, torch::Tensor plan, torch::Tensor routePrefix, torch::Tensor routeCounter,
        torch::Tensor relaySources, torch::Tensor relayDestinations, torch::Tensor relayPlaneBytes,
        torch::Tensor relayCompletedCtas, torch::Tensor relayCounters, std::optional<torch::Tensor> residency,
        std::int64_t readyFlagMcPtr, std::int64_t epValue, std::int64_t expertsValue, std::int64_t helpersValue,
        std::int64_t sendCountValue, std::int64_t sendWeightValue, std::int64_t localRankValue, std::int64_t topkValue,
        std::int64_t routeCapacityValue, std::int64_t ctasValue, std::int64_t copyStream)
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
        checkCudaTensor(routeCounter, device, torch::kInt64, 1, "route_counter");
        checkCudaTensor(
            relaySources, device, torch::kInt64, static_cast<std::int64_t>(experts) * planes, "relay_sources");
        checkCudaTensor(
            relayDestinations, device, torch::kInt64, static_cast<std::int64_t>(ep) * planes, "relay_destinations");
        checkCudaTensor(relayPlaneBytes, device, torch::kInt64, planes, "relay_plane_bytes");
        checkCudaTensor(relayCompletedCtas, device, torch::kInt32, 1, "relay_completed_ctas");
        checkCudaTensor(relayCounters, device, torch::kInt64, kernels::kHaloRCounters, "relay_counters");
        if (residency.has_value())
        {
            checkCudaTensor(*residency, device, torch::kInt32, static_cast<std::int64_t>(ep) * helpers, "residency");
        }

        mParams = kernels::MoeRebalanceHaloRParams{
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
            reinterpret_cast<unsigned long long*>(routeCounter.data_ptr<std::int64_t>()),
            reinterpret_cast<std::uint64_t const*>(relaySources.data_ptr<std::int64_t>()),
            reinterpret_cast<std::uint64_t const*>(relayDestinations.data_ptr<std::int64_t>()),
            reinterpret_cast<std::uint64_t const*>(relayPlaneBytes.data_ptr<std::int64_t>()),
            residency.has_value() ? residency->data_ptr<int>() : nullptr,
            reinterpret_cast<unsigned*>(relayCompletedCtas.data_ptr<int>()),
            reinterpret_cast<long long*>(relayCounters.data_ptr<std::int64_t>()),
            checkedPointer<std::uint64_t>(readyFlagMcPtr, "ready_flag_mc_ptr"),
            0,
            0,
            ep,
            experts,
            helpers,
            checkedInt(sendCountValue, "send_count"),
            checkedInt(sendWeightValue, "send_weight"),
            checkedInt(localRankValue, "local_rank"),
            routeCapacity,
            0,
            planes,
        };
        kernels::checkMoeRebalanceHaloRParams(mParams, ctas);
        mDevice = device.index();
        mTopk = topk;
        mCopyStream = reinterpret_cast<cudaStream_t>(copyStream);

        c10::cuda::CUDAGuard const deviceGuard(device);
        mLaunch = kernels::prepareMoeRebalanceHaloRLaunch(ep, experts, ctas);
        checkCudaResult(
            cudaEventCreateWithFlags(&mInputReady, cudaEventDisableTiming), "cudaEventCreate(HALO-R input ready)");
        // The launches read and write these by address; retaining them prevents allocator reuse.
        mTensors = {std::move(outSlots), std::move(outIds), std::move(outOwners), std::move(peerBases),
            std::move(status), std::move(partial), std::move(routeAux), std::move(gridSync), std::move(plan),
            std::move(routePrefix), std::move(routeCounter), std::move(relaySources), std::move(relayDestinations),
            std::move(relayPlaneBytes), std::move(relayCompletedCtas), std::move(relayCounters)};
        if (residency.has_value())
        {
            mTensors.push_back(std::move(*residency));
        }
        mBound = true;
    }

    MoeRebalanceHaloRState(MoeRebalanceHaloRState const&) = delete;
    MoeRebalanceHaloRState& operator=(MoeRebalanceHaloRState const&) = delete;

    ~MoeRebalanceHaloRState() noexcept override
    {
        releaseEvents();
    }

    //! Forks COPY from the current stream and enqueues the fused kernel of generation `generation()`, whose READY
    //! value it is. The routes must stay unmodified until the caller's route wait.
    void submitFused(torch::Tensor const& routes, std::int64_t spinCycles)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireBound();
        TORCH_CHECK(mPoison == cudaSuccess, "HALO-R context is poisoned by an earlier CUDA error (",
            cudaGetErrorString(mPoison), "); a quiesced group rebuild is required");
        TORCH_CHECK_VALUE(routes.is_cuda() && routes.get_device() == mDevice
                && routes.scalar_type() == at::ScalarType::Int && routes.is_contiguous() && routes.dim() == 2
                && routes.size(1) == mTopk && routes.numel() <= mParams.routeCapacity,
            "routes must be a contiguous CUDA int32 [tokens, ", mTopk, "] tensor on device ", mDevice, " with at most ",
            mParams.routeCapacity / mTopk, " tokens");
        TORCH_CHECK_VALUE(spinCycles > 0, "spin_cycles must be positive");
        auto params = mParams;
        params.validRouteCount = static_cast<int>(routes.numel());
        params.routes = params.validRouteCount > 0 ? routes.data_ptr<int>() : nullptr;
        params.spinCycles = static_cast<unsigned long long>(spinCycles);
        params.readyValue = mGeneration;

        c10::cuda::CUDAGuard const deviceGuard(c10::Device(c10::DeviceType::CUDA, mDevice));
        cudaStream_t const execution = currentExecutionStream();
        cudaStreamCaptureStatus capture{cudaStreamCaptureStatusNone};
        poisonOnError(cudaStreamIsCapturing(execution, &capture), "cudaStreamIsCapturing(HALO-R)");
        // A captured READY generation would replay a stale value, so capture is refused before COPY joins it.
        TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
            "HALO-R submit: ", cudaGetErrorString(cudaErrorStreamCaptureUnsupported));
        poisonOnError(cudaEventRecord(mInputReady, execution), "cudaEventRecord(HALO-R input ready)");
        poisonOnError(cudaStreamWaitEvent(mCopyStream, mInputReady, 0), "cudaStreamWaitEvent(HALO-R input ready)");
        poisonOnError(kernels::launchMoeRebalanceHaloR(mLaunch, params, mCopyStream), "HALO-R kernel launch");
        ++mGeneration;
    }

    //! Waits for COPY and returns the diagnostic counters (launches, receiver copies, receiver skips). For tests and
    //! teardown only.
    std::vector<std::int64_t> relayCounts()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        requireBound();
        c10::cuda::CUDAGuard const deviceGuard(c10::Device(c10::DeviceType::CUDA, mDevice));
        long long counters[kernels::kHaloRCounters]{};
        poisonOnError(
            cudaMemcpyAsync(counters, mParams.counters, sizeof(counters), cudaMemcpyDeviceToHost, mCopyStream),
            "cudaMemcpyAsync(HALO-R counters)");
        poisonOnError(cudaStreamSynchronize(mCopyStream), "cudaStreamSynchronize(HALO-R copy stream)");
        return {counters[kernels::kHaloRCounterLaunches], counters[kernels::kHaloRCounterReceiverCopies],
            counters[kernels::kHaloRCounterReceiverSkips]};
    }

    //! Releases the event and the bound tensors; the caller has drained the device work that uses them.
    void destroy()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        releaseEvents();
        mTensors.clear();
        mCopyStream = nullptr;
        mBound = false;
    }

private:
    void requireBound() const
    {
        TORCH_CHECK(mBound, "HALO-R state is destroyed");
    }

    void poisonOnError(cudaError_t error, char const* operation)
    {
        if (error != cudaSuccess)
        {
            mPoison = error;
            checkCudaResult(error, operation);
        }
    }

    // The stream whose queued work a launch follows; it must differ from the copy stream.
    cudaStream_t currentExecutionStream() const
    {
        cudaStream_t const stream = at::cuda::getCurrentCUDAStream(mDevice).stream();
        TORCH_CHECK(stream != mCopyStream, "the current stream must differ from the bound copy stream");
        return stream;
    }

    void releaseEvents() noexcept
    {
        // A pending record completes normally after its event is destroyed.
        if (mInputReady != nullptr)
        {
            static_cast<void>(cudaEventDestroy(mInputReady));
            mInputReady = nullptr;
        }
    }

    mutable std::mutex mMutex;
    kernels::MoeRebalanceHaloRParams mParams{};
    kernels::MoeRebalanceHaloRLaunch mLaunch{};
    std::vector<torch::Tensor> mTensors;
    int mDevice{-1};
    int mTopk{0};
    //! READY value of the next launch; the caller's generation count starts at 1 too.
    std::uint64_t mGeneration{1};
    cudaStream_t mCopyStream{nullptr};
    cudaEvent_t mInputReady{nullptr};
    cudaError_t mPoison{cudaSuccess};
    bool mBound{false};
};

using MoeRebalanceHaloRStatePtr = c10::intrusive_ptr<MoeRebalanceHaloRState>;

MoeRebalanceHaloRStatePtr moeRebalanceHaloRCreate(torch::Tensor outSlots, torch::Tensor outIds, torch::Tensor outOwners,
    torch::Tensor peerBases, torch::Tensor status, torch::Tensor partial, torch::Tensor routeAux,
    torch::Tensor gridSync, torch::Tensor plan, torch::Tensor routePrefix, torch::Tensor routeCounter,
    torch::Tensor relaySources, torch::Tensor relayDestinations, torch::Tensor relayPlaneBytes,
    torch::Tensor relayCompletedCtas, torch::Tensor relayCounters, std::optional<torch::Tensor> residency,
    std::int64_t readyFlagMcPtr, std::int64_t ep, std::int64_t experts, std::int64_t helpers, std::int64_t sendCount,
    std::int64_t sendWeight, std::int64_t localRank, std::int64_t topk, std::int64_t routeCapacity, std::int64_t ctas,
    std::int64_t copyStream)
{
    return c10::make_intrusive<MoeRebalanceHaloRState>(std::move(outSlots), std::move(outIds), std::move(outOwners),
        std::move(peerBases), std::move(status), std::move(partial), std::move(routeAux), std::move(gridSync),
        std::move(plan), std::move(routePrefix), std::move(routeCounter), std::move(relaySources),
        std::move(relayDestinations), std::move(relayPlaneBytes), std::move(relayCompletedCtas),
        std::move(relayCounters), std::move(residency), readyFlagMcPtr, ep, experts, helpers, sendCount, sendWeight,
        localRank, topk, routeCapacity, ctas, copyStream);
}

// The per-layer op takes this handle instead of the custom class, whose conversion costs several microseconds of
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

void moeRebalanceHaloRSubmitFused(std::int64_t handle, torch::Tensor const& routes, std::int64_t spinCycles)
{
    stateFromHandle(handle).submitFused(routes, spinCycles);
}

std::vector<std::int64_t> moeRebalanceHaloRRelayCounts(MoeRebalanceHaloRStatePtr const& state)
{
    TORCH_CHECK(state, "HALO-R state is null");
    return state->relayCounts();
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
        "Tensor route_counter, Tensor relay_sources, Tensor relay_destinations, Tensor relay_plane_bytes, "
        "Tensor relay_completed_ctas, Tensor relay_counters, Tensor? residency, int ready_flag_mc_ptr, int ep, "
        "int experts, int helpers, int send_count, int send_weight, int local_rank, int topk, int route_capacity, "
        "int ctas, int copy_stream) -> "
        "__torch__.torch.classes.trtllm.MoeRebalanceHaloRState");
    m.def(
        "moe_rebalance_halo_r_handle("
        "__torch__.torch.classes.trtllm.MoeRebalanceHaloRState state) -> int");
    m.def("moe_rebalance_halo_r_submit_fused(int handle, Tensor routes, int spin_cycles) -> ()");
    m.def(
        "moe_rebalance_halo_r_relay_counts("
        "__torch__.torch.classes.trtllm.MoeRebalanceHaloRState state) -> int[]");
    m.def(
        "moe_rebalance_halo_r_destroy("
        "__torch__.torch.classes.trtllm.MoeRebalanceHaloRState state) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("moe_rebalance_halo_r_create", &tensorrt_llm::torch_ext::moeRebalanceHaloRCreate);
    m.impl("moe_rebalance_halo_r_handle", &tensorrt_llm::torch_ext::moeRebalanceHaloRHandle);
    m.impl("moe_rebalance_halo_r_submit_fused", &tensorrt_llm::torch_ext::moeRebalanceHaloRSubmitFused);
    m.impl("moe_rebalance_halo_r_relay_counts", &tensorrt_llm::torch_ext::moeRebalanceHaloRRelayCounts);
    m.impl("moe_rebalance_halo_r_destroy", &tensorrt_llm::torch_ext::moeRebalanceHaloRDestroy);
}
