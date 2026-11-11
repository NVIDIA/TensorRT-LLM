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

#include "tensorrt_llm/runtime/iBuffer.h"

#include <cuda_runtime_api.h>

#if defined(TORCH_CUDA)
#include "tensorrt_llm/common/profiler.h"
#include <limits>
#endif

namespace tensorrt_llm::batch_manager::detail
{
enum class KvTransferOperation
{
    kOffload,
    kOnboard,
};

enum class KvManagerVersion
{
    kV1,
    kV2,
};

#if defined(TORCH_CUDA)
//! Called only inside a lazy metadata factory at a direct host/device copy boundary.
inline void addKvTransferMetadata(common::ProfilerMetadata& metadata, KvTransferOperation operation,
    KvManagerVersion managerVersion, std::size_t bytesPerCopy, std::size_t copyCount, cudaStream_t stream)
{
    bool const offload = operation == KvTransferOperation::kOffload;
    metadata.add("kv.operation", offload ? "offload" : "onboard");
    metadata.add("kv.source_tier", offload ? "gpu" : "host");
    metadata.add("kv.destination_tier", offload ? "host" : "gpu");
    metadata.add("kv.manager_version", managerVersion == KvManagerVersion::kV1 ? "v1" : "v2");
    // Sizes and CUDA stream IDs are unsigned; omit values that cannot be represented by int64 metadata.
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
    auto const streamError = cudaStreamGetId(stream, &streamId);
    if (streamError != cudaSuccess)
    {
        static_cast<void>(cudaGetLastError());
    }
    else if (streamId <= static_cast<unsigned long long>(std::numeric_limits<std::int64_t>::max()))
    {
        metadata.add("kv.cuda_stream_id", static_cast<std::int64_t>(streamId));
    }
}

//! Own one optional host/device submission scope without affecting the copy's error handling.
class KvCacheTransferScope
{
public:
    //! Start an observed migration; all field extraction remains lazy.
    template <typename MetadataFactory>
    void start(KvTransferOperation operation, MetadataFactory&& metadataFactory) noexcept
    {
        if (!mScope.isActive())
        {
            return;
        }
        mScope.startWithLazyName(
            [operation]() -> std::string {
                return operation == KvTransferOperation::kOffload ? "trtllm::kv_cache_offload"
                                                                  : "trtllm::kv_cache_onboard";
            },
            std::forward<MetadataFactory>(metadataFactory));
    }

    //! Start a nonempty migration with common payload fields and optional caller fields.
    template <typename MetadataFactory>
    void start(KvTransferOperation operation, KvManagerVersion version, std::size_t bytesPerCopy, std::size_t copyCount,
        cudaStream_t stream, MetadataFactory&& extraMetadata) noexcept
    {
        if (bytesPerCopy == 0 || copyCount == 0)
        {
            return;
        }
        start(operation,
            [&](common::ProfilerMetadata& metadata)
            {
                addKvTransferMetadata(metadata, operation, version, bytesPerCopy, copyCount, stream);
                std::forward<MetadataFactory>(extraMetadata)(metadata);
            });
    }

    //! Infer direction once for direct host/device buffers; device reuse emits no migration scope.
    template <typename MetadataFactory>
    void startHostDeviceCopy(runtime::IBuffer const& source, runtime::IBuffer const& destination,
        KvManagerVersion version, std::size_t bytes, cudaStream_t stream, MetadataFactory&& extraMetadata) noexcept
    {
        if (!mScope.isActive())
        {
            return;
        }
        try
        {
            auto const srcType = source.getMemoryType();
            auto const dstType = destination.getMemoryType();
            auto const isHost = [](runtime::MemoryType type)
            {
                return type == runtime::MemoryType::kCPU || type == runtime::MemoryType::kPINNED
                    || type == runtime::MemoryType::kPINNEDPOOL;
            };
            bool const offload = srcType == runtime::MemoryType::kGPU && isHost(dstType);
            if (offload || (isHost(srcType) && dstType == runtime::MemoryType::kGPU))
            {
                start(offload ? KvTransferOperation::kOffload : KvTransferOperation::kOnboard, version, bytes, 1,
                    stream, std::forward<MetadataFactory>(extraMetadata));
            }
        }
        catch (std::exception const& error)
        {
            common::reportProfilerFailure(error.what());
        }
        catch (...)
        {
            common::reportProfilerFailure("non-standard exception observing a KV transfer");
        }
    }

private:
    common::RecordFunctionWithMetadata mScope{common::ProfilerRegion::kKv};
};
#else
template <typename Metadata>
void addKvTransferMetadata(
    Metadata&, KvTransferOperation, KvManagerVersion, std::size_t, std::size_t, cudaStream_t) noexcept
{
}

//! Native builds without Torch keep the submission boundary as a compile-time no-op.
class KvCacheTransferScope
{
public:
    template <typename... Args>
    void start(Args&&...) noexcept
    {
    }

    template <typename... Args>
    void startHostDeviceCopy(Args&&...) noexcept
    {
    }
};
#endif
} // namespace tensorrt_llm::batch_manager::detail
