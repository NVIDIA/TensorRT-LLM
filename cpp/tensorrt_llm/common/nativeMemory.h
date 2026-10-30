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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

// Device memory that TensorRT-LLM's C++ code obtains outside the PyTorch allocator, by label and device.
//
// Such memory shows in the device's usage but in no PyTorch counter, so a memory report cannot say what it is
// for. Each allocation site records here under a fixed label; Python reads the totals through
// trtllm::get_native_memory_usage. Recording is a map update under a mutex, made only on allocation and release.
struct NativeMemoryUsage
{
    std::int64_t count{0};
    std::int64_t bytes{0};
};

// Record an allocation of exactly `bytes` on `device`.
void recordNativeAllocation(char const* label, int device, std::int64_t bytes) noexcept;

// Record the release of an allocation recorded with recordNativeAllocation.
void recordNativeRelease(char const* label, int device, std::int64_t bytes) noexcept;

// Free memory on the current device, or -1 if it cannot be read.
std::int64_t currentDeviceFreeBytes() noexcept;

// Record an allocation on the current device whose size only the allocator knows -- a library that rounds or adds
// internal buffers -- as the drop in free device memory since `freeBefore`. The reading is device-wide, so memory
// another process allocates on the same device in between is counted too.
void recordNativeAllocationMeasured(char const* label, std::int64_t freeBefore) noexcept;

std::vector<std::pair<std::string, NativeMemoryUsage>> getNativeMemoryUsage(int device);

} // namespace common

TRTLLM_NAMESPACE_END
