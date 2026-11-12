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

#include "tensorrt_llm/common/nativeMemory.h"

#include <cuda_runtime_api.h>

#include <map>
#include <mutex>

TRTLLM_NAMESPACE_BEGIN

namespace common
{
namespace
{
struct NativeMemoryLedger
{
    std::mutex mutex;
    //! device -> label -> usage.
    std::map<int, std::map<std::string, NativeMemoryUsage>> usage;
};

//! Never destroyed: owners record releases from destructors that may run during static destruction.
NativeMemoryLedger& nativeMemoryLedger()
{
    static auto* ledger = new NativeMemoryLedger();
    return *ledger;
}

void addNativeMemory(
    char const* label, int device, std::int64_t count, std::int64_t bytes, std::int64_t unmeasured) noexcept
{
    if (label == nullptr)
    {
        return;
    }
    try
    {
        auto& ledger = nativeMemoryLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        auto& usage = ledger.usage[device][label];
        usage.count += count;
        usage.bytes += bytes;
        usage.unmeasured += unmeasured;
    }
    catch (...) // NOLINT(bugprone-empty-catch): bookkeeping for a diagnostic must never fail an allocation.
    {
    }
}

thread_local char const* tCurrentLabel = nullptr;
} // namespace

void recordNativeAllocation(char const* label, int device, std::int64_t bytes) noexcept
{
    addNativeMemory(label, device, 1, bytes, 0);
}

void recordNativeRelease(char const* label, int device, std::int64_t bytes) noexcept
{
    addNativeMemory(label, device, -1, -bytes, 0);
}

void recordNativeRelease(NativeMemoryRecord const& record) noexcept
{
    if (record.bytes == NativeMemoryRecord::kUnmeasured)
    {
        addNativeMemory(record.label, record.device, -1, 0, -1);
    }
    else
    {
        addNativeMemory(record.label, record.device, -1, -record.bytes, 0);
    }
}

void adjustNativeAllocation(char const* label, int device, std::int64_t deltaBytes) noexcept
{
    addNativeMemory(label, device, 0, deltaBytes, 0);
}

std::int64_t currentDeviceFreeBytes() noexcept
{
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
    {
        cudaGetLastError();
        return -1;
    }
    return static_cast<std::int64_t>(freeBytes);
}

NativeMemoryRecord recordNativeAllocationMeasured(char const* label, std::int64_t freeBefore) noexcept
{
    NativeMemoryRecord record;
    if (cudaGetDevice(&record.device) != cudaSuccess)
    {
        cudaGetLastError();
        return record;
    }
    record.label = label;
    auto const freeAfter = freeBefore < 0 ? -1 : currentDeviceFreeBytes();
    auto const delta = freeBefore - freeAfter;
    if (freeAfter < 0 || delta < 0)
    {
        record.bytes = NativeMemoryRecord::kUnmeasured;
        addNativeMemory(label, record.device, 1, 0, 1);
    }
    else
    {
        record.bytes = delta;
        addNativeMemory(label, record.device, 1, delta, 0);
    }
    return record;
}

std::vector<std::pair<std::string, NativeMemoryUsage>> getNativeMemoryUsage(int device)
{
    auto& ledger = nativeMemoryLedger();
    std::lock_guard<std::mutex> lock(ledger.mutex);
    std::vector<std::pair<std::string, NativeMemoryUsage>> usage;
    auto const it = ledger.usage.find(device);
    if (it != ledger.usage.end())
    {
        usage.assign(it->second.begin(), it->second.end());
    }
    return usage;
}

NativeMemoryLabelScope::NativeMemoryLabelScope(char const* label) noexcept
    : mPrevious(tCurrentLabel)
{
    tCurrentLabel = label;
}

NativeMemoryLabelScope::~NativeMemoryLabelScope()
{
    tCurrentLabel = mPrevious;
}

char const* currentNativeMemoryLabel() noexcept
{
    return tCurrentLabel;
}

} // namespace common

TRTLLM_NAMESPACE_END
