/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
struct DiskAddress
{
    int fd;
    ssize_t pos;
};

using MemAddress = std::uintptr_t;

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
    IndexMapper(SizeType32 maxBatchSize, SizeType32 maxBeamWidth);

    ~IndexMapper();

    IndexMapper(IndexMapper const&) = delete;
    IndexMapper& operator=(IndexMapper const&) = delete;

    SizeType32 addNewSequence(LlmRequest::RequestIdType requestId);

    SizeType32 getIndex(LlmRequest::RequestIdType requestId);

    void removeSequence(LlmRequest::RequestIdType requestId);

    at::Tensor getCopyIndex(
        std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 numContext, SizeType32 beamWidth);

private:
    std::unordered_map<LlmRequest::RequestIdType, SizeType32> indexMap_;
    std::set<SizeType32> freeIndices_;
    SizeType32 maxBeamWidth_;
    at::Tensor copyIndex_;
};

CUresult copyDiskToDisk(std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyDiskToHost(std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToDisk(std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToHost(std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyHostToDevice(
    std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyDeviceToHost(
    std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream) noexcept;
CUresult copyDeviceToDevice(
    std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream) noexcept;

void copyBatchBlockOffsetsToDevice(
    ITensor const& input, ITensor& output, ITensor const& copyIndex, bool copyVIdx, CUstream stream) noexcept;

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
