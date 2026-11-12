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

#include "tensorrt_llm/common/config.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

//! \file
//! A ledger of device memory that TensorRT-LLM's C++ code obtains outside the PyTorch allocator.
//!
//! Such memory shows in the device's usage but in no PyTorch counter. Each allocation site records here under one
//! of the labels below; Python reads the totals per device through `trtllm::get_native_memory_usage` and says what
//! each label is for (`tensorrt_llm/_torch/memory/monitor/providers/native.py`).
//!
//! Contract:
//! - A recorded allocation is released at most once, by its owner, with the record or the size it recorded, when its
//!   memory is returned. Memory an owner keeps mapped stays recorded.
//! - Recording is a map update under a mutex, made only when memory is allocated or released, and never throws.
//! - A label stays listed after its last release, with a count of 0.
//! - The ledger is never destroyed, so releases recorded from static destructors are safe.

//! \brief Labels the ledger accepts. `providers/native.py` describes each one; a test keeps the two in step.
namespace native_memory_label
{
constexpr char const* kNcclCommunicators = "nccl_communicators";
constexpr char const* kGemmAllreduceNvls = "gemm_allreduce_nvls";
constexpr char const* kUlyssesA2aSlots = "ulysses_a2a_slots";
constexpr char const* kMoeGemmProfilerWorkspace = "moe_gemm_profiler_workspace";
constexpr char const* kEplbDevice = "eplb_device";
constexpr char const* kEplbGpuNuma = "eplb_gpu_numa";
constexpr char const* kEplbManaged = "eplb_managed";

//! \brief Every label above.
constexpr std::array<char const*, 7> kAll{kNcclCommunicators, kGemmAllreduceNvls, kUlyssesA2aSlots,
    kMoeGemmProfilerWorkspace, kEplbDevice, kEplbGpuNuma, kEplbManaged};
} // namespace native_memory_label

//! \brief Totals for one label on one device.
struct NativeMemoryUsage
{
    std::int64_t count{0};      //!< Live allocations, measured or not.
    std::int64_t bytes{0};      //!< Bytes of the live allocations whose size is known.
    std::int64_t unmeasured{0}; //!< Live allocations whose size could not be read; `bytes` is then a lower bound.
};

//! \brief What one recorded allocation added to the ledger, for releasing exactly that.
struct NativeMemoryRecord
{
    //! Size of a recorded allocation whose size could not be read.
    static constexpr std::int64_t kUnmeasured = -1;

    char const* label{nullptr}; //!< Null when nothing was recorded; releasing such a record does nothing.
    int device{-1};             //!< Device the allocation was recorded on.
    std::int64_t bytes{0};      //!< Bytes recorded, or kUnmeasured.
};

//! \brief Record an allocation of exactly `bytes` on `device`.
void recordNativeAllocation(char const* label, int device, std::int64_t bytes) noexcept;

//! \brief Record the release of an allocation recorded with recordNativeAllocation(label, device, bytes).
void recordNativeRelease(char const* label, int device, std::int64_t bytes) noexcept;

//! \brief Release what `record` added. Does nothing for a record with no label.
void recordNativeRelease(NativeMemoryRecord const& record) noexcept;

//! \brief Change the bytes recorded under `label` on `device` by `deltaBytes`, keeping its allocation count. For an
//! owner that frees part of one recorded allocation and keeps the rest mapped.
void adjustNativeAllocation(char const* label, int device, std::int64_t deltaBytes) noexcept;

//! \brief Free memory on the current device, or -1 if it cannot be read.
std::int64_t currentDeviceFreeBytes() noexcept;

//! \brief Record an allocation on the current device whose size only the allocator knows, as the drop in free device
//! memory since `freeBefore` (a value from currentDeviceFreeBytes()).
//!
//! The reading is device-wide, so memory another process allocates or frees in between is included. A reading that
//! cannot be taken (`freeBefore` < 0) or comes out negative is recorded as an allocation of unknown size, which
//! makes the label's total a lower bound until that allocation is released.
//!
//! \return The record to pass to recordNativeRelease() when the memory is freed.
NativeMemoryRecord recordNativeAllocationMeasured(char const* label, std::int64_t freeBefore) noexcept;

//! \brief Every label with an entry on `device`, including those whose count has returned to 0.
std::vector<std::pair<std::string, NativeMemoryUsage>> getNativeMemoryUsage(int device);

//! \brief While alive, allocations on this thread by allocators that cannot name their caller are recorded under
//! `label`. Such allocators read it with currentNativeMemoryLabel().
class NativeMemoryLabelScope
{
public:
    explicit NativeMemoryLabelScope(char const* label) noexcept;
    ~NativeMemoryLabelScope();

    NativeMemoryLabelScope(NativeMemoryLabelScope const&) = delete;
    NativeMemoryLabelScope& operator=(NativeMemoryLabelScope const&) = delete;

private:
    char const* mPrevious; //!< Label of the enclosing scope, restored on destruction.
};

//! \brief The label of the innermost NativeMemoryLabelScope on this thread, or null.
char const* currentNativeMemoryLabel() noexcept;

} // namespace common

TRTLLM_NAMESPACE_END
