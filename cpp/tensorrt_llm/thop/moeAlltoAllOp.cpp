/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/moeAlltoAllMeta.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <torch/extension.h>
#include <torch/types.h>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace moe_comm
{

static constexpr size_t CACHELINE_ALIGNMENT = 128;
static constexpr size_t kExecutionDeviceStatusOffset = CACHELINE_ALIGNMENT;
static constexpr size_t kExecutionDeviceAdmissionOffset = kExecutionDeviceStatusOffset + sizeof(uint64_t);
static constexpr int64_t kExecutionControlLiveEpochWord = 0;
static constexpr int64_t kExecutionControlAcknowledgedEpochWord = 1;
static constexpr int64_t kExecutionControlTimeoutCyclesWord = 2;
static constexpr int64_t kExecutionControlHostStatusWord = CACHELINE_ALIGNMENT / sizeof(uint64_t);
static constexpr int64_t kExecutionControlNumWords = 2 * CACHELINE_ALIGNMENT / sizeof(uint64_t);
static constexpr uint64_t kExecutionStatusEpochMask = (uint64_t{1} << 39) - 1;

// The custom op launches asynchronously while its control tensor lives in CPU
// memory, so PyTorch's CUDA allocator cannot defer that tensor's deletion for us.
// Retain each tiny control allocation until an explicit, post-quiescence release.
class ExecutionControlRegistration
{
public:
    ExecutionControlRegistration(
        torch::Tensor tensor, torch::Tensor workspace, uint64_t* deviceWords, int deviceIndex, int64_t epRank)
        : mTensor{std::move(tensor)}
        , mWorkspace{std::move(workspace)}
        , mDeviceWords{deviceWords}
        , mDeviceIndex{deviceIndex}
        , mEpRank{epRank}
    {
    }

    uint64_t* getDeviceWords() const
    {
        return mDeviceWords;
    }

    int getDeviceIndex() const
    {
        return mDeviceIndex;
    }

    void const* getWorkspaceBasePtr() const
    {
        return mWorkspace.const_data_ptr();
    }

    int64_t getEpRank() const
    {
        return mEpRank;
    }

    bool matchesWorkspace(torch::Tensor const& workspace) const
    {
        if (workspace.device() != mWorkspace.device() || workspace.const_data_ptr() != mWorkspace.const_data_ptr()
            || workspace.dim() != mWorkspace.dim() || workspace.storage_offset() != mWorkspace.storage_offset())
        {
            return false;
        }
        for (int64_t dimension = 0; dimension < workspace.dim(); ++dimension)
        {
            if (workspace.size(dimension) != mWorkspace.size(dimension)
                || workspace.stride(dimension) != mWorkspace.stride(dimension))
            {
                return false;
            }
        }
        return true;
    }

private:
    [[maybe_unused]] torch::Tensor mTensor; // Owns the mapped allocation until explicit release.
    torch::Tensor mWorkspace;               // Keeps the device-status words alive for every asynchronous launch.
    uint64_t* mDeviceWords{};
    int mDeviceIndex{};
    int64_t mEpRank{};
};

struct ExecutionControlRegistry
{
    std::mutex mutex;
    std::unordered_map<void const*, ExecutionControlRegistration> controls;
};

struct ExecutionControlMapping
{
    uint64_t const* hostWords{};
    uint64_t* deviceWords{};
};

ExecutionControlRegistry& getExecutionControlRegistry()
{
    // Intentionally process-lifetime state. Explicit release is the normal path;
    // leaking the registry avoids cudaFreeHost calls during CUDA/DSO teardown.
    static auto* registry = new ExecutionControlRegistry{};
    return *registry;
}

inline uint64_t atomicLoadAcquire(uint64_t const* ptr)
{
#if defined(_MSC_VER)
    auto* value = reinterpret_cast<__int64 volatile*>(const_cast<uint64_t*>(ptr));
    return static_cast<uint64_t>(_InterlockedCompareExchange64(value, 0, 0));
#else
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#endif
}

inline void atomicStoreRelease(uint64_t* ptr, uint64_t value)
{
#if defined(_MSC_VER)
    auto* destination = reinterpret_cast<__int64 volatile*>(ptr);
    (void) _InterlockedExchange64(destination, static_cast<__int64>(value));
#else
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
#endif
}

inline bool atomicCompareExchangeAcqRel(uint64_t* ptr, uint64_t& expected, uint64_t desired)
{
#if defined(_MSC_VER)
    auto* destination = reinterpret_cast<__int64 volatile*>(ptr);
    auto const observed = static_cast<uint64_t>(
        _InterlockedCompareExchange64(destination, static_cast<__int64>(desired), static_cast<__int64>(expected)));
    if (observed == expected)
    {
        return true;
    }
    expected = observed;
    return false;
#else
    return __atomic_compare_exchange_n(ptr, &expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#endif
}

// TODO: Is Alignment necessary?
// Helper function to align offset to specified byte boundary
inline size_t alignOffset(size_t offset, size_t alignment)
{
    return (offset + alignment - 1) & ~(alignment - 1);
}

inline void validateWorkspaceCudaTensor(
    torch::Tensor const& tensor, torch::Tensor const& workspace, char const* tensorName)
{
    TORCH_CHECK(tensor.is_cuda(), tensorName, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == workspace.device(), tensorName, " must be on the same CUDA device as workspace (",
        workspace.device(), "), but got ", tensor.device());
}

// Resolve an optional rank-mask tensor into a fixed-width uint64 array.
// If the caller did not provide a mask, default to "all ranks active" (all bits set), which
// reproduces the pre-fault-tolerance behavior bit-for-bit.
//
// On failure (wrong dtype / device / shape), throws via TORCH_CHECK so the error surfaces
// at the Python op boundary rather than the kernel launch.
inline void resolveActiveRankMask(torch::optional<torch::Tensor> const& maskTensor, int64_t epRank,
    uint64_t (&out)[tensorrt_llm::kernels::moe_comm::kRankMaskWords])
{
    using tensorrt_llm::kernels::moe_comm::kRankMaskWords;
    using tensorrt_llm::kernels::moe_comm::kMaxRanks;
    TORCH_CHECK(
        epRank >= 0 && epRank < kMaxRanks, "epRank must be in the range [0, ", kMaxRanks, ") for active_rank_mask");
    if (!maskTensor.has_value() || !maskTensor.value().defined())
    {
        for (int w = 0; w < kRankMaskWords; ++w)
        {
            out[w] = ~uint64_t{0};
        }
        return;
    }
    torch::Tensor const& t = maskTensor.value();
    TORCH_CHECK(t.is_cpu(), "active_rank_mask must be a CPU tensor");
    TORCH_CHECK(t.scalar_type() == torch::kUInt64, "active_rank_mask must have dtype uint64");
    TORCH_CHECK(t.dim() == 1, "active_rank_mask must be a 1D tensor");
    TORCH_CHECK(t.numel() == kRankMaskWords, "active_rank_mask must have exactly ", kRankMaskWords, " uint64 elements");
    TORCH_CHECK(t.is_contiguous(), "active_rank_mask must be contiguous");
    auto const* src = static_cast<uint64_t const*>(t.const_data_ptr());
    for (int w = 0; w < kRankMaskWords; ++w)
    {
        out[w] = src[w];
    }
    // Local rank's bit must be set; otherwise the kernel would be running on a "dead" rank.
    TORCH_CHECK((out[epRank >> 6] >> (epRank & 63)) & 1ULL, "active_rank_mask must mark the local ep_rank (", epRank,
        ") as active");
}

inline uint64_t const* validateExecutionControlHostTensor(torch::Tensor const& control)
{
    TORCH_CHECK(control.is_cpu(), "execution_control must be a CPU tensor");
    TORCH_CHECK(control.scalar_type() == torch::kUInt64, "execution_control must have dtype uint64");
    TORCH_CHECK(control.dim() == 1, "execution_control must be a 1D tensor");
    TORCH_CHECK(control.numel() == kExecutionControlNumWords, "execution_control must have exactly ",
        kExecutionControlNumWords, " uint64 elements");
    TORCH_CHECK(control.is_contiguous(), "execution_control must be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(control.const_data_ptr()) % alignof(uint64_t) == 0,
        "execution_control must be naturally aligned");
    return static_cast<uint64_t const*>(control.const_data_ptr());
}

inline uint64_t* validateMutableExecutionControlHostTensor(torch::Tensor& control)
{
    (void) validateExecutionControlHostTensor(control);
    return control.data_ptr<uint64_t>();
}

inline ExecutionControlMapping getExecutionControlRegistration(
    torch::Tensor const& control, torch::Tensor const* expectedWorkspace = nullptr, int64_t expectedEpRank = -1)
{
    void const* hostPtr = validateExecutionControlHostTensor(control);
    auto& registry = getExecutionControlRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto const it = registry.controls.find(hostPtr);
    TORCH_CHECK(it != registry.controls.end(),
        "execution_control is not registered or was already released; create it with "
        "moe_a2a_create_execution_control");
    if (expectedWorkspace != nullptr)
    {
        TORCH_CHECK(it->second.getDeviceIndex() == expectedWorkspace->get_device(),
            "execution_control was mapped for CUDA device ", it->second.getDeviceIndex(),
            " but the workspace is on CUDA device ", expectedWorkspace->get_device());
        TORCH_CHECK(it->second.matchesWorkspace(*expectedWorkspace),
            "execution_control belongs to a different workspace on CUDA device ", expectedWorkspace->get_device());
        TORCH_CHECK(it->second.getEpRank() == expectedEpRank, "execution_control belongs to ep_rank ",
            it->second.getEpRank(), " but was used with ep_rank ", expectedEpRank);
    }
    return {static_cast<uint64_t const*>(hostPtr), it->second.getDeviceWords()};
}

inline uint64_t* getExecutionDeviceStatusPtr(torch::Tensor const& workspace, int64_t epRank)
{
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor");
    TORCH_CHECK(epRank >= 0 && epRank < workspace.size(0), "ep_rank out of range");
    TORCH_CHECK(workspace.size(1) >= static_cast<int64_t>(kExecutionDeviceAdmissionOffset + sizeof(uint64_t)),
        "workspace row is too small for execution abort device state");
    uint8_t* rankWorkspacePtr = workspace.data_ptr<uint8_t>() + epRank * workspace.stride(0);
    auto* statusPtr = reinterpret_cast<uint64_t*>(rankWorkspacePtr + kExecutionDeviceStatusOffset);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(statusPtr) % alignof(uint64_t) == 0,
        "execution abort device status must be naturally aligned");
    return statusPtr;
}

inline tensorrt_llm::kernels::moe_comm::MoeA2AExecutionControl resolveExecutionControl(
    torch::optional<torch::Tensor> const& controlTensor, int64_t expectedEpoch, torch::Tensor const& workspace,
    int64_t epRank)
{
    using tensorrt_llm::kernels::moe_comm::MoeA2AExecutionControl;
    TORCH_CHECK(expectedEpoch >= 0, "expected_execution_epoch must be non-negative");
    TORCH_CHECK(static_cast<uint64_t>(expectedEpoch) <= kExecutionStatusEpochMask,
        "expected_execution_epoch exceeds the 39-bit status range");
    TORCH_CHECK(controlTensor.has_value() && controlTensor.value().defined(),
        "execution_control is required so an aborted epoch cannot return unobserved partial output");

    torch::Tensor const& controlTensorValue = controlTensor.value();
    auto const registration = getExecutionControlRegistration(controlTensorValue, &workspace, epRank);
    MoeA2AExecutionControl control{};
    control.expected_epoch = static_cast<uint64_t>(expectedEpoch);
    control.device_status = getExecutionDeviceStatusPtr(workspace, epRank);
    control.device_admission = control.device_status + 1;
    control.live_epoch = registration.deviceWords + kExecutionControlLiveEpochWord;
    control.host_status = registration.deviceWords + kExecutionControlHostStatusWord;
    control.timeout_cycles = atomicLoadAcquire(registration.hostWords + kExecutionControlTimeoutCyclesWord);
    return control;
}

torch::Tensor moeA2ACreateExecutionControlOp(torch::Tensor const& workspace, int64_t epRank)
{
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D uint8 tensor");
    TORCH_CHECK(workspace.size(0) > 0, "workspace must contain at least one rank row");
    TORCH_CHECK(epRank >= 0 && epRank < workspace.size(0), "ep_rank out of range for execution_control workspace");
    TORCH_CHECK(workspace.size(1) >= static_cast<int64_t>(kExecutionDeviceAdmissionOffset + sizeof(uint64_t)),
        "workspace row is too small for execution abort device state");
    at::cuda::CUDAGuard deviceGuard(workspace.device());
    int const device = workspace.get_device();

    cudaDeviceProp properties{};
    cudaError_t result = cudaGetDeviceProperties(&properties, device);
    TORCH_CHECK(result == cudaSuccess, "failed to query CUDA device properties: ", cudaGetErrorString(result));
    TORCH_CHECK(properties.canMapHostMemory,
        "NVLinkOneSided execution abort requires mapped page-locked host memory on the current CUDA device");

    void* hostPtr = nullptr;
    result = cudaHostAlloc(
        &hostPtr, kExecutionControlNumWords * sizeof(uint64_t), cudaHostAllocMapped | cudaHostAllocPortable);
    TORCH_CHECK(result == cudaSuccess, "failed to allocate mapped execution_control: ", cudaGetErrorString(result));
    auto deleter = [](void* ptr) { (void) cudaFreeHost(ptr); };
    std::unique_ptr<void, decltype(deleter)> hostAllocation{hostPtr, deleter};

    void* devicePtr = nullptr;
    result = cudaHostGetDevicePointer(&devicePtr, hostPtr, 0);
    if (result != cudaSuccess)
    {
        TORCH_CHECK(
            false, "failed to map execution_control into the current CUDA device: ", cudaGetErrorString(result));
    }
    std::fill_n(static_cast<uint64_t*>(hostPtr), kExecutionControlNumWords, uint64_t{0});

    torch::Tensor control = torch::from_blob(hostPtr, {kExecutionControlNumWords}, deleter,
        torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU));
    (void) hostAllocation.release();
    {
        auto& registry = getExecutionControlRegistry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        for (auto const& registeredControl : registry.controls)
        {
            TORCH_CHECK(registeredControl.second.getWorkspaceBasePtr() != workspace.const_data_ptr()
                    || registeredControl.second.getDeviceIndex() != device
                    || registeredControl.second.getEpRank() != epRank,
                "workspace ep_rank ", epRank, " already has a registered execution_control");
        }
        bool const inserted = registry.controls
                                  .emplace(hostPtr,
                                      ExecutionControlRegistration{
                                          control, workspace, static_cast<uint64_t*>(devicePtr), device, epRank})
                                  .second;
        TORCH_CHECK(inserted, "execution_control allocation is already registered");
    }
    return control;
}

void moeA2AReleaseExecutionControlOp(torch::Tensor const& control)
{
    void const* hostPtr = validateExecutionControlHostTensor(control);
    auto& registry = getExecutionControlRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    TORCH_CHECK(registry.controls.erase(hostPtr) == 1, "execution_control is not registered or was already released");
}

int64_t moeA2ARequestExecutionAbortOp(torch::Tensor& control)
{
    // This path is used by a recovery-coordinator callback thread. Keep it CPU-only:
    // validation must not make a CUDA runtime call that could couple abort publication
    // to device work. Detection/watchdog threads report evidence; they do not own this write.
    (void) getExecutionControlRegistration(control);
    uint64_t* words = validateMutableExecutionControlHostTensor(control);
    uint64_t* liveEpoch = words + kExecutionControlLiveEpochWord;
    uint64_t current = atomicLoadAcquire(liveEpoch);
    while (true)
    {
        TORCH_CHECK(current < kExecutionStatusEpochMask, "execution abort epoch exhausted its 39-bit status range");
        uint64_t const next = current + 1;
        if (atomicCompareExchangeAcqRel(liveEpoch, current, next))
        {
            return static_cast<int64_t>(next);
        }
    }
}

std::tuple<int64_t, int64_t> moeA2AGetExecutionAbortStateOp(torch::Tensor const& control)
{
    (void) getExecutionControlRegistration(control);
    uint64_t const* words = validateExecutionControlHostTensor(control);
    uint64_t const liveEpoch = atomicLoadAcquire(words + kExecutionControlLiveEpochWord);
    uint64_t const hostStatus = atomicLoadAcquire(words + kExecutionControlHostStatusWord);
    TORCH_CHECK(liveEpoch <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "execution abort epoch exceeds int64 range");
    TORCH_CHECK(hostStatus <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "execution abort status exceeds int64 range");
    return {static_cast<int64_t>(liveEpoch), static_cast<int64_t>(hostStatus)};
}

void moeA2ASetExecutionTimeoutForTestingOp(torch::Tensor& control, int64_t timeoutCycles)
{
    TORCH_CHECK(timeoutCycles >= 0, "timeout_cycles must be non-negative");
    (void) getExecutionControlRegistration(control);
    uint64_t* words = validateMutableExecutionControlHostTensor(control);
    atomicStoreRelease(words + kExecutionControlTimeoutCyclesWord, static_cast<uint64_t>(timeoutCycles));
}

void moeA2ABeginExecutionEpochOp(
    torch::Tensor& workspace, int64_t epRank, torch::Tensor& control, int64_t executionEpoch)
{
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor");
    TORCH_CHECK(epRank >= 0 && epRank < workspace.size(0), "ep_rank out of range");
    TORCH_CHECK(executionEpoch >= 0, "execution_epoch must be non-negative");
    TORCH_CHECK(static_cast<uint64_t>(executionEpoch) <= kExecutionStatusEpochMask,
        "execution_epoch exceeds the 39-bit status range");

    (void) getExecutionControlRegistration(control, &workspace, epRank);
    uint64_t* words = validateMutableExecutionControlHostTensor(control);
    uint64_t const liveEpoch = atomicLoadAcquire(words + kExecutionControlLiveEpochWord);
    uint64_t const acknowledgedEpoch = atomicLoadAcquire(words + kExecutionControlAcknowledgedEpochWord);
    TORCH_CHECK(liveEpoch == static_cast<uint64_t>(executionEpoch), "cannot begin execution epoch ", executionEpoch,
        "; the latest requested epoch is ", liveEpoch);
    TORCH_CHECK(static_cast<uint64_t>(executionEpoch) > acknowledgedEpoch,
        "execution_epoch must advance beyond the acknowledged epoch ", acknowledgedEpoch);

    at::cuda::CUDAGuard deviceGuard(workspace.device());
    uint64_t* deviceStatus = getExecutionDeviceStatusPtr(workspace, epRank);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(workspace.get_device());
    cudaError_t result = cudaMemsetAsync(deviceStatus, 0, 2 * sizeof(uint64_t), stream);
    TORCH_CHECK(result == cudaSuccess, "failed to clear execution abort device state: ", cudaGetErrorString(result));
    result = cudaStreamSynchronize(stream);
    TORCH_CHECK(
        result == cudaSuccess, "failed to synchronize execution abort status reset: ", cudaGetErrorString(result));
    atomicStoreRelease(words + kExecutionControlHostStatusWord, uint64_t{0});
    atomicStoreRelease(words + kExecutionControlAcknowledgedEpochWord, static_cast<uint64_t>(executionEpoch));
}

// Calculate auxiliary data offsets
MoeA2ADataOffsets calculateOffsets(int epSize, int maxNumTokens, int eplbStatsNumExperts)
{
    // TODO: Use lambdas to encapsulate offset and alignment for each entry, which is less error prone and easier to
    // read.
    constexpr size_t SIZEOF_INT32 = 4;

    MoeA2ADataOffsets offsets;
    size_t offset = 0;

    // flag_val
    offsets[FLAG_VAL_OFFSET_INDEX] = offset;
    offset += SIZEOF_INT32;

    // Local device-side sticky execution status and combine admission word. Keep
    // both on a dedicated cacheline; their offsets are intentionally private so
    // the ten-entry metainfo ABI remains stable.
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    TLLM_CHECK(offset == kExecutionDeviceStatusOffset);
    offset += 2 * sizeof(uint64_t);
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);

    // local_token_counter
    offsets[LOCAL_TOKEN_COUNTER_OFFSET_INDEX] = offset;
    offset += SIZEOF_INT32;

    // send_counters
    offsets[SEND_COUNTERS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // recv_counters
    offsets[RECV_COUNTERS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // dispatch completion flags
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // combine completion flags
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[COMBINE_COMPLETION_FLAGS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // topk_target_ranks: [maxNumTokens, kMaxTopK]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[TOPK_TARGET_RANKS_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(maxNumTokens) * static_cast<size_t>(tensorrt_llm::kernels::moe_comm::kMaxTopK)
        * SIZEOF_INT32;

    // topk_send_indices: [maxNumTokens, kMaxTopK]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[TOPK_SEND_INDICES_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(maxNumTokens) * static_cast<size_t>(tensorrt_llm::kernels::moe_comm::kMaxTopK)
        * SIZEOF_INT32;

    // eplb gathered stats: [epSize, eplbStatsNumExperts]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[EPLB_GATHERED_STATS_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(epSize) * static_cast<size_t>(eplbStatsNumExperts) * SIZEOF_INT32;

    // payload data
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[PAYLOAD_DATA_OFFSET_INDEX] = offset;

    return offsets;
}

// Initialize auxiliary data in workspace
// This function sets up the initial values for flag_val and completion_flags
//
// Inputs:
//   - workspace: [ep_size, size_per_rank] unified virtual memory workspace
//   - epRank: Current expert parallel rank
//   - epSize: Total expert parallel size
//   - maxNumTokens: Maximum number of tokens supported
//   - eplbStatsNumExperts: (Optional) Number of experts used for EPLB stats
//
// Returns:
//   - metainfo: Tensor containing offsets for auxiliary data
torch::Tensor moeA2AInitializeOp(torch::Tensor const& workspace, int64_t epRank, int64_t epSize, int64_t maxNumTokens,
    torch::optional<int64_t> eplbStatsNumExperts)
{
    using tensorrt_llm::kernels::moe_comm::kMaxRanks;

    // Validate inputs
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");
    TORCH_CHECK(epSize > 0 && epSize <= kMaxRanks, "epSize must be in the range (0, ", kMaxRanks, "]");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

    // Initialize workspace to zero
    workspace[epRank].zero_();

    int64_t eplbStatsNumExpertsValue = eplbStatsNumExperts.value_or(0);
    TORCH_CHECK(eplbStatsNumExpertsValue >= 0, "eplbStatsNumExperts must be positive if not None.");

    // Calculate auxiliary data offsets
    MoeA2ADataOffsets offsets = calculateOffsets(epSize, maxNumTokens, static_cast<int>(eplbStatsNumExpertsValue));

    // Return metainfo as a tensor containing offsets
    torch::Tensor metainfo = torch::empty(
        {static_cast<int64_t>(NUM_METAINFO_FIELDS)}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    for (int i = 0; i < static_cast<int>(NUM_METAINFO_FIELDS); i++)
    {
        metainfo[i] = static_cast<int64_t>(offsets[i]);
    }

    // Synchronize among ranks
    cudaDeviceSynchronize();
    tensorrt_llm::mpi::MpiComm::session().barrier();

    return metainfo;
}

// MoE All-to-All Dispatch Operation
// This operation dispatches tokens and their associated payloads to different expert ranks.
//
// Inputs:
//   - tokenSelectedExperts: [local_num_tokens, top_k] tensor of expert indices
//   - inputPayloads: List of tensors with shape [local_num_tokens, ...] containing data to dispatch
//   - workspace: [ep_size, size_per_rank] unified virtual memory workspace where size_per_rank is large enough to store
//   all the auxiliary data and recv payloads.
//   - metainfo: [NUM_METAINFO_FIELDS] tensor containing offsets for auxiliary data
//   - runtimeMaxTokensPerRank: Maximum of the number of tokens of each DP rank's local batch. This is a dynamic value
//   during runtime.
//   - maxNumTokens: Maximum number of tokens that could be supported. This is a static value that is setup during
//   initialization.
//   - epRank: Current expert parallel rank
//   - epSize: Total expert parallel size
//   - topK: Number of experts selected per token
//   - numExperts: Total number of routing slots (tokenSelectedExperts values are in [0, numExperts))
//   - eplbStatsNumExperts: Number of experts used for EPLB stats (may be <= numExperts)
//   - eplbLocalStats: [eplbStatsNumExperts] tensor containing local statistics for EPLB.
//
// Return values:
//   - recvTensors: Vector of receive buffers (one tensor per payload), each [ep_size, runtimeMaxTokensPerRank,
//   elements_per_token]
//   - combinePayloadOffset: Offset into workspace for the combine payload region, to be used by the combine operation
//   - eplbGatheredStats: (Optional) [ep_size, eplbStatsNumExperts] tensor containing gathered statistics for EPLB, or
//   an empty tensor if eplbLocalStats is None.
//
// Note: token_selected_experts is used for routing but is NOT automatically included as a payload.
//       If you want to dispatch token_selected_experts, include it explicitly in inputPayloads.
std::tuple<std::vector<torch::Tensor>, int64_t, torch::Tensor> moeA2ADispatchOp(
    torch::Tensor const& tokenSelectedExperts, std::vector<torch::Tensor> const& inputPayloads,
    torch::Tensor const& workspace, torch::Tensor const& metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank,
    int64_t epSize, int64_t topK, int64_t numExperts, torch::optional<torch::Tensor> eplbLocalStats,
    torch::optional<torch::Tensor> activeRankMask, torch::optional<torch::Tensor> executionControl,
    int64_t expectedExecutionEpoch)
{
    using tensorrt_llm::kernels::moe_comm::PayloadDescriptor;
    using tensorrt_llm::kernels::moe_comm::MoeA2ADispatchParams;
    using tensorrt_llm::kernels::moe_comm::moe_a2a_dispatch_launch;
    using tensorrt_llm::kernels::moe_comm::kMaxTopK;
    using tensorrt_llm::kernels::moe_comm::kMaxPayloads;
    using tensorrt_llm::kernels::moe_comm::kMaxRanks;

    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    at::cuda::CUDAGuard deviceGuard(workspace.device());
    int const workspaceDevice = workspace.get_device();

    // Validate inputs
    CHECK_INPUT(tokenSelectedExperts, torch::kInt32);
    validateWorkspaceCudaTensor(tokenSelectedExperts, workspace, "tokenSelectedExperts");
    TORCH_CHECK(tokenSelectedExperts.dim() == 2, "tokenSelectedExperts must be a 2D tensor");
    TORCH_CHECK(tokenSelectedExperts.size(1) == topK, "tokenSelectedExperts must have topK columns");

    CHECK_CPU(metainfo);
    TORCH_CHECK(metainfo.is_cpu(), "metainfo must be a CPU tensor");
    CHECK_TYPE(metainfo, torch::kInt64);
    TORCH_CHECK(metainfo.is_contiguous(), "metainfo must be contiguous");
    TORCH_CHECK(metainfo.dim() == 1, "metainfo must be a 1D tensor");
    TORCH_CHECK(metainfo.size(0) == static_cast<int64_t>(NUM_METAINFO_FIELDS),
        "metainfo must have NUM_METAINFO_FIELDS elements");
    MoeA2ADataOffsets const& offsets = *reinterpret_cast<MoeA2ADataOffsets const*>(metainfo.data_ptr<int64_t>());

    int64_t localNumTokens = tokenSelectedExperts.size(0);
    TORCH_CHECK(runtimeMaxTokensPerRank > 0, "runtimeMaxTokensPerRank must be positive");
    TORCH_CHECK(epSize > 0 && epSize <= kMaxRanks, "epSize must be in the range (0, ", kMaxRanks, "]");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");
    TORCH_CHECK(topK > 0 && topK <= kMaxTopK, "topK must be in the range (0, kMaxTopK]");
    TORCH_CHECK(!inputPayloads.empty(), "inputPayloads must not be empty");
    TORCH_CHECK(inputPayloads.size() <= kMaxPayloads, "Too many input payloads");
    TORCH_CHECK(numExperts >= epSize, "numExperts must be greater than or equal to epSize");
    // numExperts does not need to be divisible by epSize: the kernel performs
    // ceil/floor contiguous partitioning so ranks [0, numExperts % epSize)
    // own (numExperts / epSize + 1) experts and the rest own (numExperts / epSize).
    bool enableEplb = eplbLocalStats.has_value();
    int64_t eplbStatsNumExperts = 0;
    if (enableEplb)
    {
        TORCH_CHECK(eplbLocalStats.has_value(), "enable_eplb requires eplb_local_stats");
        torch::Tensor const& eplbLocalStatsTensor = eplbLocalStats.value();
        eplbStatsNumExperts = eplbLocalStatsTensor.size(0);
        TORCH_CHECK(eplbStatsNumExperts > 0, "eplb_local_stats must not be empty");
        TORCH_CHECK(eplbStatsNumExperts <= numExperts, "eplb_local_stats size must be <= numExperts (slots)");
        CHECK_INPUT(eplbLocalStatsTensor, torch::kInt32);
        validateWorkspaceCudaTensor(eplbLocalStatsTensor, workspace, "eplb_local_stats");
        TORCH_CHECK(eplbLocalStatsTensor.is_contiguous(), "eplb_local_stats must be contiguous");
        TORCH_CHECK(eplbLocalStatsTensor.dim() == 1, "eplb_local_stats must be a 1D tensor");
    }

    // All input payloads must have the same first dimension (localNumTokens)
    for (auto const& payload : inputPayloads)
    {
        TORCH_CHECK(payload.dim() >= 1, "All payloads must have at least 1 dimension");
        TORCH_CHECK(payload.size(0) == localNumTokens,
            "All payloads must have the same first dimension as tokenSelectedExperts");
        TORCH_CHECK(payload.is_contiguous(), "All payloads must be contiguous");
    }

    // Record the cacheline aligned start offset for each payload's recv buffer.
    // 1. We assume the base workspace ptr of each rank is aligned (checked in this OP)
    // 2. offsets[PAYLOAD_DATA_OFFSET_INDEX] is aligned (ensured in calculateOffsets)
    // 3. We align the currentOffset during update.
    // In this way, it is guaranteed that the recv buffer is (over-)aligned, sufficient for 128bit vectorized ld/st.

    std::vector<int> payloadElementSizes;
    std::vector<int> payloadElementsPerToken;
    std::vector<size_t> payloadRecvBufferOffsets;

    // Start offset for the first payload
    size_t currentOffset = static_cast<size_t>(offsets[PAYLOAD_DATA_OFFSET_INDEX]);
    for (auto const& payload : inputPayloads)
    {
        CHECK_CONTIGUOUS(payload);
        CHECK_TH_CUDA(payload);
        validateWorkspaceCudaTensor(payload, workspace, "payload");
        TORCH_CHECK(payload.dim() == 2, "payload must be a 2D tensor");
        TORCH_CHECK(
            payload.size(0) == localNumTokens, "payload must have the same first dimension as tokenSelectedExperts");
        // Unlike recv buffer for payloads, payload itself is not allocated by us and we cannot control its alignment.
        // We only make sure the payload start offset is 16-byte aligned, while the actual vectorized ld/st width is
        // dynamically determined based on bytes per token of this payload.
        TORCH_CHECK(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16 == 0, "payload must be 16-byte aligned");

        int elementsPerToken = static_cast<int>(payload.size(1));
        int elementSize = static_cast<int>(payload.dtype().itemsize());
        // Each payload buffer stores data from ALL ranks
        int64_t bytesPerPayload = epSize * runtimeMaxTokensPerRank * elementsPerToken * elementSize;

        payloadElementSizes.push_back(elementSize);
        payloadElementsPerToken.push_back(elementsPerToken);

        payloadRecvBufferOffsets.push_back(currentOffset);

        // Update offset and align to cacheline boundary for the next payload recv buffer.
        currentOffset += bytesPerPayload;
        currentOffset = alignOffset(currentOffset, CACHELINE_ALIGNMENT);
    }

    // Don't check contiguous - MnnvlMemory creates strided tensors for multi-GPU
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");

    // Validate workspace size - must include space for auxiliary data + payloads
    int64_t sizePerRank = workspace.size(1);
    int64_t requiredSize = static_cast<int64_t>(currentOffset);
    TORCH_CHECK(sizePerRank >= requiredSize,
        "Workspace size per rank insufficient for dispatch. "
        "Need at least ",
        requiredSize, " bytes (", offsets[PAYLOAD_DATA_OFFSET_INDEX], " for auxiliary data + payloads), but got ",
        sizePerRank);

    // Get base workspace pointer
    uint8_t* workspacePtr = workspace.data_ptr<uint8_t>();
    uint8_t* rankWorkSpacePtr = workspacePtr + epRank * workspace.stride(0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(rankWorkSpacePtr) % CACHELINE_ALIGNMENT == 0,
        "rankWorkSpacePtr must be %d-byte aligned", CACHELINE_ALIGNMENT);

    // Setup payload descriptors for source data
    int num_payloads = static_cast<int>(inputPayloads.size());
    std::vector<PayloadDescriptor> payloadDescriptors(num_payloads);
    for (int i = 0; i < num_payloads; i++)
    {
        payloadDescriptors[i].src_data = inputPayloads[i].data_ptr();
        payloadDescriptors[i].element_size = payloadElementSizes[i];
        payloadDescriptors[i].elements_per_token = payloadElementsPerToken[i];
    }

    // Setup dispatch parameters
    MoeA2ADispatchParams params{};
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.num_experts = static_cast<int>(numExperts);
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
    params.top_k = static_cast<int>(topK);
    params.enable_eplb = enableEplb;
    params.eplb_stats_num_experts = static_cast<int>(eplbStatsNumExperts);

    params.token_selected_experts = tokenSelectedExperts.data_ptr<int32_t>();

    params.num_payloads = num_payloads;
    std::copy(payloadDescriptors.begin(), payloadDescriptors.end(), &params.payloads[0]);

    params.flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);
    params.local_token_counter = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[LOCAL_TOKEN_COUNTER_OFFSET_INDEX]);
    params.send_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[SEND_COUNTERS_OFFSET_INDEX]);
    params.topk_target_ranks = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_RANKS_OFFSET_INDEX]);
    params.topk_send_indices = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_SEND_INDICES_OFFSET_INDEX]);

    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        uint8_t* targetWorkSpacePtr = workspacePtr + (target_rank * workspace.stride(0));

        params.recv_counters[target_rank]
            = reinterpret_cast<int*>(targetWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);
        params.completion_flags[target_rank]
            = reinterpret_cast<uint32_t*>(targetWorkSpacePtr + offsets[DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX]);
        if (enableEplb)
        {
            params.eplb_gathered_stats[target_rank]
                = reinterpret_cast<int*>(targetWorkSpacePtr + offsets[EPLB_GATHERED_STATS_OFFSET_INDEX]);
        }
        else
        {
            params.eplb_gathered_stats[target_rank] = nullptr;
        }

        for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
        {
            // Store pointer for current payload using pre-calculated aligned offset
            params.recv_buffers[target_rank][payload_idx] = targetWorkSpacePtr + payloadRecvBufferOffsets[payload_idx];
        }
    }

    if (enableEplb)
    {
        params.eplb_local_stats = eplbLocalStats.value().data_ptr<int32_t>();
    }
    else
    {
        params.eplb_local_stats = nullptr;
    }

    // Resolve the optional active-rank mask. Default (no mask) = all bits set, which
    // exactly reproduces the pre-fault-tolerance kernel behavior.
    resolveActiveRankMask(activeRankMask, epRank, params.active_rank_mask);
    auto const resolvedExecutionControl
        = resolveExecutionControl(executionControl, expectedExecutionEpoch, workspace, epRank);

    params.stream = at::cuda::getCurrentCUDAStream(workspaceDevice);

    // Prepare for dispatch (zero counters/indices and increment flag_val)
    moe_a2a_prepare_dispatch_launch(params, resolvedExecutionControl);

    // Launch the dispatch kernel
    moe_a2a_dispatch_launch(params, resolvedExecutionControl);
    cudaError_t result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "moe_a2a_dispatch kernel launch failed: ", cudaGetErrorString(result));

    // Create tensor views for the current rank's receive buffers only
    std::vector<torch::Tensor> recvTensors;
    for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
    {
        auto const& payload = inputPayloads[payload_idx];
        // Create tensor view for this payload using pre-calculated aligned offset
        auto recvTensor = torch::from_blob(rankWorkSpacePtr + payloadRecvBufferOffsets[payload_idx],
            {epSize, runtimeMaxTokensPerRank, payloadElementsPerToken[payload_idx]}, payload.options());
        recvTensors.push_back(recvTensor);
    }

    // Compute aligned offset after dispatch payloads for combine payload region
    int64_t combinePayloadOffset = static_cast<int64_t>(alignOffset(currentOffset, CACHELINE_ALIGNMENT));

    torch::Tensor eplbGatheredStats;
    if (enableEplb)
    {
        int* gatheredStatsPtr = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[EPLB_GATHERED_STATS_OFFSET_INDEX]);
        auto statsOptions = workspace.options().dtype(torch::kInt32);
        eplbGatheredStats = torch::from_blob(
            gatheredStatsPtr, {static_cast<int64_t>(epSize), static_cast<int64_t>(eplbStatsNumExperts)}, statsOptions);
    }
    else
    {
        eplbGatheredStats = torch::empty({0}, workspace.options().dtype(torch::kInt32));
    }

    return std::make_tuple(std::move(recvTensors), combinePayloadOffset, std::move(eplbGatheredStats));
}

// MoE All-to-All Combine Operation
// Combine the per-rank expert outputs into the originating tokens' buffers on the local rank.
//
// Two payload modes are supported:
//   1) External payload tensor: 'payload' is a tensor with shape [ep_size, max_tokens_per_rank, elements_per_token]
//      that is NOT backed by the shared workspace. In this mode, the op stages the current rank's
//      slice into the workspace region at 'payloadRegionOffset' via the prepare kernel.
//   2) Workspace-backed payload tensor: 'payload' is a view into the shared workspace. Set
//      payloadInWorkspace=true to skip staging. The op will read directly from the workspace region
//      at 'combinePayloadOffset'.
// In both cases, the combine kernel reads from the workspace at 'combinePayloadOffset'.
torch::Tensor moeA2ACombineOp(torch::Tensor const& payload, int64_t localNumTokens, torch::Tensor const& workspace,
    torch::Tensor const& metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank, int64_t epSize, int64_t topK,
    int64_t combinePayloadOffset, bool payloadInWorkspace, bool useLowPrecision = false,
    torch::optional<torch::Tensor> activeRankMask = torch::nullopt,
    torch::optional<torch::Tensor> executionControl = torch::nullopt, int64_t expectedExecutionEpoch = 0)
{
    using tensorrt_llm::kernels::moe_comm::MoeA2ACombineParams;
    using tensorrt_llm::kernels::moe_comm::moe_a2a_combine_launch;
    using tensorrt_llm::kernels::moe_comm::kMaxTopK;
    using tensorrt_llm::kernels::moe_comm::kMaxRanks;

    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    at::cuda::CUDAGuard deviceGuard(workspace.device());
    int const workspaceDevice = workspace.get_device();

    // Validate inputs
    CHECK_TH_CUDA(payload);
    validateWorkspaceCudaTensor(payload, workspace, "payload");
    CHECK_CONTIGUOUS(payload);
    TORCH_CHECK(payload.dim() == 3, "payload must be a 3D tensor [ep_size, max_tokens_per_rank, elements_per_token]");
    TORCH_CHECK(payload.size(0) == epSize, "payload first dimension must equal epSize");
    TORCH_CHECK(
        payload.size(1) == runtimeMaxTokensPerRank, "payload second dimension must equal runtimeMaxTokensPerRank");
    // We only make sure the payload start offset is 16-byte aligned, while the actual vectorized ld/st width is
    // dynamically determined based on bytes per token of this payload.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16 == 0, "payload must be 16-byte aligned");
    int64_t elementsPerToken = payload.size(2);
    TORCH_CHECK(elementsPerToken > 0, "elementsPerToken must be positive");
    TORCH_CHECK(epSize > 0 && epSize <= kMaxRanks, "epSize must be in the range (0, ", kMaxRanks, "]");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");
    TORCH_CHECK(topK > 0 && topK <= kMaxTopK, "topK must be in the range (0, kMaxTopK]");

    // Map torch dtype to nvinfer1::DataType
    nvinfer1::DataType nvDtype = nvinfer1::DataType::kFLOAT;
    auto scalarType = payload.scalar_type();
    if (scalarType == at::kHalf)
    {
        nvDtype = nvinfer1::DataType::kHALF;
    }
    else if (scalarType == at::kBFloat16)
    {
        nvDtype = nvinfer1::DataType::kBF16;
    }
    else if (scalarType == at::kFloat)
    {
        nvDtype = nvinfer1::DataType::kFLOAT;
    }
    else
    {
        TORCH_CHECK(false, "Unsupported data type for payload");
    }
    // use_low_precision is passed through to the kernel via params.use_low_precision; dtype is not mutated.

    CHECK_CPU(metainfo);
    TORCH_CHECK(metainfo.is_cpu(), "metainfo must be a CPU tensor");
    CHECK_TYPE(metainfo, torch::kInt64);
    TORCH_CHECK(metainfo.is_contiguous(), "metainfo must be contiguous");
    TORCH_CHECK(metainfo.dim() == 1, "metainfo must be a 1D tensor");
    TORCH_CHECK(metainfo.size(0) == static_cast<int64_t>(NUM_METAINFO_FIELDS),
        "metainfo must have NUM_METAINFO_FIELDS elements");
    MoeA2ADataOffsets const& offsets = *reinterpret_cast<MoeA2ADataOffsets const*>(metainfo.data_ptr<int64_t>());

    // Validate workspace and set synchronization pointers
    TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) == epSize, "workspace must be [ep_size, size_per_rank]");
    uint8_t* workspacePtr = workspace.data_ptr<uint8_t>();
    int64_t sizePerRank = workspace.size(1);
    uint8_t* rankWorkSpacePtr = workspacePtr + epRank * workspace.stride(0);

    // If user claims payload is in workspace, ensure payload tensor matches combinePayloadOffset
    if (payloadInWorkspace)
    {
        TORCH_CHECK(payload.data_ptr() == rankWorkSpacePtr + combinePayloadOffset,
            "payload_in_workspace is true but 'payload' dataptr does not match combinePayloadOffset");
    }

    int64_t payloadSize = payload.numel() * payload.element_size();
    TORCH_CHECK(combinePayloadOffset >= 0 && combinePayloadOffset + payloadSize <= sizePerRank,
        "Workspace size per rank insufficient for combine. "
        "Need at least ",
        combinePayloadOffset + payloadSize, " bytes (", combinePayloadOffset, " for offset + ", payloadSize,
        " for payload), but got ", sizePerRank);

    // Create output tensor (local on current rank), no need for initialization
    // Typically, newly allocated GPU torch tensors are at least 16-byte aligned.
    // Output dtype always matches the payload dtype: low-precision accumulates FP8 back to payload dtype.
    auto output_options = payload.options();
    torch::Tensor output = torch::empty({localNumTokens, elementsPerToken}, output_options);

    // Setup combine parameters
    MoeA2ACombineParams params{};
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
    params.top_k = static_cast<int>(topK);
    // If payload is not in workspace, stage it into current rank's region at prepare phase
    if (!payloadInWorkspace)
    {
        params.prepare_payload = payload.data_ptr();
    }
    params.output_data = output.data_ptr();
    params.elements_per_token = static_cast<int>(elementsPerToken);
    params.dtype = nvDtype;
    params.use_low_precision = useLowPrecision;

    params.flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);
    params.topk_target_ranks = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_RANKS_OFFSET_INDEX]);
    params.topk_send_indices = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_SEND_INDICES_OFFSET_INDEX]);
    params.recv_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);

    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        uint8_t* target_workspace_ptr = workspacePtr + target_rank * workspace.stride(0);
        params.completion_flags[target_rank]
            = reinterpret_cast<uint32_t*>(target_workspace_ptr + offsets[COMBINE_COMPLETION_FLAGS_OFFSET_INDEX]);
        params.recv_buffers[target_rank] = target_workspace_ptr + combinePayloadOffset;
    }

    // Resolve the optional active-rank mask. Default (no mask) = all bits set.
    resolveActiveRankMask(activeRankMask, epRank, params.active_rank_mask);
    auto const resolvedExecutionControl
        = resolveExecutionControl(executionControl, expectedExecutionEpoch, workspace, epRank);

    params.stream = at::cuda::getCurrentCUDAStream(workspaceDevice);

    moe_a2a_prepare_combine_launch(params, resolvedExecutionControl);

    // Launch the combine kernel
    moe_a2a_combine_launch(params, resolvedExecutionControl);
    cudaError_t result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "moe_a2a_combine kernel launch failed: ", cudaGetErrorString(result));

    return output;
}

// Op: moe_a2a_sanitize_expert_ids
void moeA2ASanitizeExpertIdsOp(torch::Tensor& expert_ids, torch::Tensor& workspace, torch::Tensor const& metainfo,
    int64_t epRank, int64_t invalid_expert_id)
{
    CHECK_INPUT(expert_ids, torch::kInt32);
    TORCH_CHECK(expert_ids.dim() == 3, "expert_ids must be [ep_size, runtime_max_tokens_per_rank, top_k]");

    int ep_size = static_cast<int>(expert_ids.size(0));
    int runtime_max_tokens_per_rank = static_cast<int>(expert_ids.size(1));
    int top_k = static_cast<int>(expert_ids.size(2));

    CHECK_CPU(metainfo);
    CHECK_TYPE(metainfo, torch::kInt64);
    TORCH_CHECK(metainfo.dim() == 1, "metainfo must be a 1D tensor");
    TORCH_CHECK(metainfo.size(0) == static_cast<int64_t>(NUM_METAINFO_FIELDS),
        "metainfo must have NUM_METAINFO_FIELDS elements");
    MoeA2ADataOffsets const& offsets = *reinterpret_cast<MoeA2ADataOffsets const*>(metainfo.data_ptr<int64_t>());

    uint8_t* rankWorkSpacePtr = workspace.data_ptr<uint8_t>() + epRank * workspace.stride(0);
    int* recv_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);

    tensorrt_llm::kernels::moe_comm::moe_a2a_sanitize_expert_ids_launch(expert_ids.data_ptr<int32_t>(), recv_counters,
        static_cast<int32_t>(invalid_expert_id), ep_size, runtime_max_tokens_per_rank, top_k,
        at::cuda::getCurrentCUDAStream());
}

// Return a workspace-backed tensor for combine payload region using from_blob
torch::Tensor moeA2AGetCombinePayloadTensorOp(torch::Tensor const& workspace, int64_t epRank, int64_t epSize,
    int64_t runtimeMaxTokensPerRank, int64_t combinePayloadOffset, c10::ScalarType outDtype, int64_t hiddenSize)
{
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be [ep_size, size_per_rank_bytes]");
    TORCH_CHECK(epRank >= 0 && epRank < workspace.size(0), "epRank out of range");
    TORCH_CHECK(epSize == workspace.size(0), "epSize mismatch with workspace");
    TORCH_CHECK(runtimeMaxTokensPerRank > 0, "runtimeMaxTokensPerRank must be positive");
    TORCH_CHECK(hiddenSize > 0, "hidden must be positive");

    int64_t sizePerRank = workspace.size(1); // bytes
    int64_t elementSize = static_cast<int64_t>(c10::elementSize(outDtype));
    int64_t bytesNeeded = epSize * runtimeMaxTokensPerRank * hiddenSize * elementSize;
    TORCH_CHECK(combinePayloadOffset >= 0, "combine_payload_offset must be non-negative");
    TORCH_CHECK(combinePayloadOffset + bytesNeeded <= sizePerRank,
        "workspace does not have enough space for combine payload tensor. combine payload offset=",
        combinePayloadOffset, ", payload size needed=", bytesNeeded, ", workspace size per rank=", sizePerRank);

    uint8_t* base = workspace.data_ptr<uint8_t>();
    uint8_t* rankBase = base + epRank * workspace.stride(0);
    uint8_t* dataPtr = rankBase + combinePayloadOffset;

    auto options = workspace.options().dtype(outDtype);
    torch::Tensor t = torch::from_blob(dataPtr, {epSize * runtimeMaxTokensPerRank, hiddenSize}, options);
    return t;
}

// Return the size of auxiliary data in workspace
int64_t moeA2AGetAuxDataSizeOp(int64_t epSize, int64_t maxNumTokens, torch::optional<int64_t> eplbStatsNumExperts)
{
    int64_t eplbStatsNumExpertsValue = eplbStatsNumExperts.value_or(0);
    TORCH_CHECK(eplbStatsNumExpertsValue >= 0, "eplbStatsNumExperts must be positive if not None.");
    MoeA2ADataOffsets offsets = calculateOffsets(
        static_cast<int>(epSize), static_cast<int>(maxNumTokens), static_cast<int>(eplbStatsNumExpertsValue));
    return static_cast<int64_t>(offsets[PAYLOAD_DATA_OFFSET_INDEX]);
}

} // namespace moe_comm

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// PyTorch bindings
TORCH_LIBRARY_FRAGMENT(trtllm, module)
{
    // Note that we returns recv_tensors as a list of views into workspace, we need to upcast its alias
    // group to wildcard (a!->*). See
    // https://github.com/pytorch/pytorch/blob/b1eb6dede556136f9fdcee28415b0358d58ad877/aten/src/ATen/native/README.md#annotations
    module.def(
        "moe_a2a_dispatch(Tensor token_selected_experts, Tensor[] input_payloads, "
        "Tensor(a!->*) workspace, Tensor metainfo, int runtime_max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k, int num_experts, "
        "Tensor? eplb_local_stats=None, "
        "Tensor? active_rank_mask=None, Tensor(b!)? execution_control=None, "
        "int expected_execution_epoch=0) -> (Tensor(a!)[], int, Tensor(a!))");
    module.def(
        "moe_a2a_combine(Tensor(a) payload, int local_num_tokens,"
        "Tensor(a!) workspace, Tensor metainfo, int runtime_max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k, int combine_payload_offset, "
        "bool payload_in_workspace, bool use_low_precision=False, "
        "Tensor? active_rank_mask=None, Tensor(b!)? execution_control=None, "
        "int expected_execution_epoch=0) -> Tensor");
    module.def(
        "moe_a2a_initialize(Tensor(a!) workspace, int ep_rank, int ep_size, int max_num_tokens_per_rank, "
        "int? eplb_stats_num_experts=None) -> Tensor");
    module.def(
        "moe_a2a_sanitize_expert_ids(Tensor(a!) expert_ids, Tensor(a!) workspace, Tensor metainfo, int ep_rank, int "
        "invalid_expert_id) -> ()");
    module.def(
        "moe_a2a_get_combine_payload_tensor(Tensor(a) workspace, int ep_rank, int ep_size, int "
        "runtime_max_tokens_per_rank, "
        "int combine_payload_offset, ScalarType out_dtype, int hidden_size) -> Tensor(a)");
    module.def("moe_a2a_get_aux_data_size(int ep_size, int max_num_tokens, int? eplb_stats_num_experts=None) -> int",
        &tensorrt_llm::torch_ext::moe_comm::moeA2AGetAuxDataSizeOp);
    module.def("moe_a2a_create_execution_control(Tensor workspace, int ep_rank) -> Tensor",
        &tensorrt_llm::torch_ext::moe_comm::moeA2ACreateExecutionControlOp);
    module.def("moe_a2a_request_execution_abort(Tensor(a!) execution_control) -> int",
        &tensorrt_llm::torch_ext::moe_comm::moeA2ARequestExecutionAbortOp);
    module.def("moe_a2a_get_execution_abort_state(Tensor execution_control) -> (int, int)",
        &tensorrt_llm::torch_ext::moe_comm::moeA2AGetExecutionAbortStateOp);
    module.def("moe_a2a_set_execution_timeout_for_testing(Tensor(a!) execution_control, int timeout_cycles) -> ()",
        &tensorrt_llm::torch_ext::moe_comm::moeA2ASetExecutionTimeoutForTestingOp);
    module.def("moe_a2a_release_execution_control(Tensor(a!) execution_control) -> ()",
        &tensorrt_llm::torch_ext::moe_comm::moeA2AReleaseExecutionControlOp);
    module.def(
        "moe_a2a_begin_execution_epoch(Tensor(a!) workspace, int ep_rank, Tensor(b!) execution_control, int "
        "execution_epoch) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, module)
{
    module.impl("moe_a2a_dispatch", &tensorrt_llm::torch_ext::moe_comm::moeA2ADispatchOp);
    module.impl("moe_a2a_combine", &tensorrt_llm::torch_ext::moe_comm::moeA2ACombineOp);
    module.impl("moe_a2a_initialize", &tensorrt_llm::torch_ext::moe_comm::moeA2AInitializeOp);
    module.impl("moe_a2a_sanitize_expert_ids", &tensorrt_llm::torch_ext::moe_comm::moeA2ASanitizeExpertIdsOp);
    module.impl(
        "moe_a2a_get_combine_payload_tensor", &tensorrt_llm::torch_ext::moe_comm::moeA2AGetCombinePayloadTensorOp);
    module.impl("moe_a2a_begin_execution_epoch", &tensorrt_llm::torch_ext::moe_comm::moeA2ABeginExecutionEpochOp);
}
