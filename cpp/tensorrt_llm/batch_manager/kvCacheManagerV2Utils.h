/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "kv_cache_manager_v2/common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <ATen/ATen.h>
#include <cstdint>
#include <cuda.h>
#include <set>
#include <unordered_map>
#include <vector>

namespace tk = tensorrt_llm::kernels;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using ITensor = tensorrt_llm::runtime::ITensor;

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
// DiskAddress and MemAddress are defined in kv_cache_manager_v2/common.h (included above).

// Please make sure to align with the definition in tensorrt_llm/runtime/kv_cache_manager_v2/_common.py
constexpr tk::KVCacheIndex::UnderlyingType BAD_PAGE_INDEX = -1;

template <typename DstAddr, typename SrcAddr>
struct Task
{
    DstAddr dst;
    SrcAddr src;
};

using PackedInt = union
{
    int4 packed;
    tk::KVCacheIndex::UnderlyingType unpacked[4];
};

class IndexMapper
{
public:
    //! maxBeamWidth is the source page-table stride. maxCopyBeamWidth can be
    //! larger for request-scoped caches whose beam-0 row is broadcast to beams.
    IndexMapper(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxCopyBeamWidth = 0);

    ~IndexMapper();

    IndexMapper(IndexMapper const&) = delete;
    IndexMapper& operator=(IndexMapper const&) = delete;

    SizeType32 addNewSequence(LlmRequest::RequestIdType requestId);

    SizeType32 getIndex(LlmRequest::RequestIdType requestId);

    void removeSequence(LlmRequest::RequestIdType requestId);

    //! Map scheduled sequence rows to source page-table rows. When
    //! replicateBeamZero is true, every generation beam maps to beam 0.
    at::Tensor getCopyIndex(std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 numContext,
        SizeType32 beamWidth, bool replicateBeamZero = false);

    //! Gathers each request's beam-0 K block offsets into a host snapshot.
    void gatherKBlockOffsets(at::Tensor const& source, at::Tensor destination,
        std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 numBlocks);

    /// Number of sequences currently tracked (i.e. active IndexMapper slots).
    [[nodiscard]] SizeType32 size() const noexcept
    {
        return static_cast<SizeType32>(indexMap_.size());
    }

    /// Number of free IndexMapper slots available for new sequences.
    [[nodiscard]] SizeType32 numFreeSlots() const noexcept
    {
        return static_cast<SizeType32>(freeIndices_.size());
    }

private:
    std::unordered_map<LlmRequest::RequestIdType, SizeType32> indexMap_;
    std::set<SizeType32> freeIndices_;
    SizeType32 maxBeamWidth_;
    SizeType32 maxCopyBeamWidth_;
    at::Tensor copyIndex_;
};

CUresult copyDiskToDisk(std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyDiskToHost(std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToDisk(std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToHost(std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
//! copyHostToDevice / copyDeviceToHost return CUDA_SUCCESS without enqueuing anything for an empty call, use the
//! hardware copy engines when numBytes > 0 and useCopyEngineForHostCopy() says so, and the batched copy kernel
//! otherwise (numBytes == 0 takes the kernel). The kernel moves 16 B grains: it throws when numBytes is not a multiple
//! of 16 and, unchecked, faults on a task address that is not 16 B aligned, and it takes less than 4 GiB per task (the
//! launcher's byte count is a uint32_t); the copy engines accept any size and any alignment. Any stream is accepted;
//! on the copy-engine path the legacy NULL stream, as 0 or as CU_STREAM_LEGACY, takes one cuMemcpy{HtoD,DtoH}Async per
//! task. Both throw TllmException on a negative numBytes. On main, hot<->host migrations go through
//! IKvCacheColdPageCodec / BatchedPageCopier; these two functions serve CopyEngine::transfer (migrations between two
//! cold levels when one of them is GPU_MEM) and the bindings. copyDeviceToDevice always uses the kernel.
CUresult copyHostToDevice(std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream);
CUresult copyDeviceToHost(std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream);
CUresult copyDeviceToDevice(std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream);

//! Smallest per-task size at which copyHostToDevice / copyDeviceToHost use the copy engines instead of the batched copy
//! kernel. From TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MIN_BYTES (default 4 MiB; 0 = any size, negative = never), read
//! once per process.
size_t hostCopyDmaMinBytes() noexcept;
//! Name of the environment variable behind hostCopyDmaMinBytes().
inline constexpr char const kHostCopyDmaMinBytesEnv[] = "TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MIN_BYTES";
//! The threshold for a raw value of that variable (nullptr = unset): the default for an unset, empty, non-integer or
//! out-of-range value (the last two are reported at WARNING), SIZE_MAX for a negative value. hostCopyDmaMinBytes() is
//! this function applied to the process environment, cached.
size_t hostCopyDmaMinBytesFor(char const* rawValue) noexcept;
//! True when a host<->device call with numBytes per task takes the hardware copy-engine (DMA) path rather than the
//! batched copy kernel in this process; "copy engine" here means the GPU DMA engines, not the CopyEngine class in
//! kv_cache_manager_v2/copyEngine.h. Always false where pinned host memory is registered in chunks
//! (HostMem::shouldUseChunkedRegistration()).
bool useCopyEngineForHostCopy(size_t numBytes) noexcept;

void copyBatchBlockOffsetsToDevice(ITensor const& input, ITensor& output, ITensor const& copyIndex,
    ITensor const& indexScales, ITensor const& kvOffset, CUstream stream) noexcept;

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
