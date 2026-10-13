/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/common/profiler.h"

#include <cuda_runtime_api.h>
#include <limits>

namespace tensorrt_llm::batch_manager
{
namespace detail
{
// Called only inside a lazy metadata factory at a direct host/device copy boundary.
inline void addKvTransferMetadata(common::ProfilerMetadata& metadata, bool offload, std::string_view managerVersion,
    std::size_t bytesPerCopy, std::size_t copyCount, cudaStream_t stream)
{
    metadata.add("kv.operation", offload ? "offload" : "restore");
    metadata.add("kv.source_tier", offload ? "gpu" : "host");
    metadata.add("kv.destination_tier", offload ? "host" : "gpu");
    metadata.add("kv.manager_version", managerVersion);
    constexpr auto kMax = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
    if (copyCount <= kMax)
    {
        metadata.add("kv.copy_count", static_cast<std::int64_t>(copyCount));
    }
    if (bytesPerCopy <= kMax)
    {
        metadata.add("kv.bytes_per_copy", static_cast<std::int64_t>(bytesPerCopy));
    }
    if (copyCount != 0 && bytesPerCopy <= kMax / copyCount)
    {
        metadata.add("kv.payload_bytes", static_cast<std::int64_t>(bytesPerCopy * copyCount));
    }
    unsigned long long streamId{};
    if (cudaStreamGetId(stream, &streamId) == cudaSuccess
        && streamId <= static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max()))
    {
        metadata.add("kv.cuda_stream_id", static_cast<std::int64_t>(streamId));
    }
}
} // namespace detail
} // namespace tensorrt_llm::batch_manager
