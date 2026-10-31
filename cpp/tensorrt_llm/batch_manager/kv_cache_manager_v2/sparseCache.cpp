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

#include "kv_cache_manager_v2/sparseCache.h"

#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/utils/math.h"

#include <algorithm>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{

constexpr size_t kAlignment = 256;

size_t alignedBytes(size_t bytes) noexcept
{
    return roundUp(bytes, kAlignment);
}

} // namespace

namespace
{

//! Bytes one lease's entry words take. Arrays shared by every lease of a life cycle live
//! on the arena instead.
size_t leaseBytes(SparseCacheInit const& shape)
{
    auto const entries = static_cast<size_t>(shape.numSelections) * shape.entriesPerSelection;
    return alignedBytes(sizeof(uint32_t) * entries);
}

} // namespace

// ---------------------------------------------------------------------------
// SparseCellArena
// ---------------------------------------------------------------------------

SparseCellArena::SparseCellArena(SparseCacheInit const& shape, int maxLeases)
    : mShape(shape)
{
    CUstream const stream = nullptr;
    TLLM_CHECK_WITH_INFO(shape.numLayers > 0 && shape.numSelections > 0 && shape.numBuffers > 0
            && shape.entriesPerSelection > 0 && shape.cellsPerBuffer > 0,
        "SparseCellArena requires a positive layer, selection, buffer, entry and cell count");
    TLLM_CHECK_WITH_INFO(shape.numSelections <= shape.numLayers,
        "SparseCellArena: a selection is shared by layers, so there cannot be more of them than layers");
    TLLM_CHECK_WITH_INFO(
        !shape.pools.empty() && maxLeases > 0, "SparseCellArena requires at least one pool and a positive lease count");
    TLLM_CHECK_WITH_INFO(shape.bufferLayer.size() == static_cast<size_t>(shape.numBuffers),
        "SparseCellArena requires one layer index per buffer");
    TLLM_CHECK_WITH_INFO(shape.bufferSlotInPool.size() == static_cast<size_t>(shape.numBuffers),
        "SparseCellArena requires one pool slot per buffer");
    TLLM_CHECK_WITH_INFO(shape.layerSelection.size() == static_cast<size_t>(shape.numLayers),
        "SparseCellArena requires one selection index per layer");
    for (size_t i = 0; i < shape.bufferLayer.size(); ++i)
    {
        TLLM_CHECK_WITH_INFO(shape.bufferLayer[i] >= 0 && shape.bufferLayer[i] < shape.numLayers,
            "SparseCellArena: a buffer's layer index is out of range");
        TLLM_CHECK_WITH_INFO(i == 0 || shape.bufferLayer[i] >= shape.bufferLayer[i - 1],
            "SparseCellArena: buffers must be grouped by layer");
    }
    for (auto const selection : shape.layerSelection)
    {
        TLLM_CHECK_WITH_INFO(selection >= 0 && selection < shape.numSelections,
            "SparseCellArena: a layer's selection index is out of range");
    }
    for (auto const& [cellBytes, buffers] : shape.pools)
    {
        // {0, 0} is a pool index no sparse buffer of this life cycle uses: it holds its place
        // so that a storage pool index indexes this arena directly, and carries no cells. Any
        // other combination describes a pool that cannot be addressed.
        TLLM_CHECK_WITH_INFO((cellBytes == 0) == (buffers == 0),
            "SparseCellArena: a pool needs both a cell size and buffers drawing on it, or neither");
        TLLM_CHECK_WITH_INFO(buffers >= 0, "SparseCellArena: a pool cannot have a negative buffer count");
    }

    mMaxLeases = maxLeases;
    mMetadataBytes = leaseBytes(mShape);

    // One region per pool, each aligned so that a buffer's cell indices stay relative to its own
    // pool's base.
    std::vector<size_t> regionOffset;
    size_t cellBytes = 0;
    for (auto const& [bytes, buffers] : mShape.pools)
    {
        regionOffset.push_back(cellBytes);
        cellBytes += alignedBytes(static_cast<size_t>(maxLeases) * buffers * mShape.cellsPerBuffer * bytes);
    }

    size_t const layerBytes = alignedBytes(sizeof(int32_t) * static_cast<size_t>(mShape.numBuffers));
    size_t const slotBytes = alignedBytes(sizeof(int32_t) * static_cast<size_t>(mShape.numBuffers));
    size_t const selectionBytes = alignedBytes(sizeof(int32_t) * static_cast<size_t>(mShape.numLayers));
    size_t const sharedBytes = layerBytes + slotBytes + selectionBytes;

    mNumBytes = cellBytes + sharedBytes + static_cast<size_t>(maxLeases) * mMetadataBytes;

    CUdeviceptr base = 0;
    cuCheck(cuMemAlloc(&base, mNumBytes));
    mMemory = CudaUniqPtr<>{reinterpret_cast<std::byte*>(base)};

    for (auto offset : regionOffset)
    {
        mPoolBase.push_back(base + offset);
    }
    auto* shared = reinterpret_cast<std::byte*>(base) + cellBytes;
    mSharedLayer = reinterpret_cast<int32_t*>(shared);
    mSharedSlotInPool = reinterpret_cast<int32_t*>(shared + layerBytes);
    mSharedLayerSelection = reinterpret_cast<int32_t*>(shared + layerBytes + slotBytes);
    mMetadataBase = shared + sharedBytes;

    cuCheck(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(mSharedLayer), mShape.bufferLayer.data(),
        sizeof(int32_t) * mShape.bufferLayer.size(), stream));
    cuCheck(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(mSharedSlotInPool), mShape.bufferSlotInPool.data(),
        sizeof(int32_t) * mShape.bufferSlotInPool.size(), stream));
    cuCheck(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(mSharedLayerSelection), mShape.layerSelection.data(),
        sizeof(int32_t) * mShape.layerSelection.size(), stream));

    // An unused cell is a clean one, so the arena is born that way and a row hands its
    // lease back clean.
    for (int slot = 0; slot < mMaxLeases; ++slot)
    {
        initialiseLease(SparseLeaseIndex{slot}, stream);
    }

    // The arena has no stream of its own, so the staging above is ordered on the null
    // stream only, and the host vectors it reads die with this scope.
    cuCheck(cuStreamSynchronize(stream));
}

void SparseCellArena::initialiseLease(SparseLeaseIndex slot, CUstream stream)
{
    if (empty())
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(slot != kNoSparseLease && slot < mMaxLeases,
        "SparseCellArena::initialiseLease: slot %d is outside [0, %d)", slot.value(), mMaxLeases);
    int const index = slot.value();

    auto const entries = static_cast<size_t>(mShape.numSelections) * static_cast<size_t>(mShape.entriesPerSelection);
    cuCheck(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(mMetadataBase + static_cast<size_t>(index) * mMetadataBytes),
        SparseEntry::kFreeWord, entries, stream));
}

SparseArenaView SparseCellArena::view() const noexcept
{
    SparseArenaView out;
    out.metadataBase = mMetadataBase;
    out.leaseStride = mMetadataBytes;
    out.bufferSlotInPool = mSharedSlotInPool;
    out.bufferLayer = mSharedLayer;
    out.layerSelection = mSharedLayerSelection;
    out.numLayers = mShape.numLayers;
    out.numSelections = mShape.numSelections;
    out.entriesPerSelection = mShape.entriesPerSelection;
    return out;
}

std::vector<int32_t> SparseCellArena::peekResidentOrdinals(SparseLeaseIndex slot) const
{
    CUstream const stream = nullptr;
    if (empty() || slot == kNoSparseLease)
    {
        return {};
    }
    // A resolve writes this metadata on the Batch's stream, which is non-blocking and so
    // unordered against the one below. Only a device-wide sync covers it; nothing here
    // knows which stream last touched the slot.
    cuCheck(cuCtxSynchronize());
    auto const arena = view();
    std::vector<uint32_t> words(arena.numEntries());
    cuCheck(cuMemcpyDtoHAsync(words.data(),
        reinterpret_cast<CUdeviceptr>(arena.metadataBase + static_cast<uint64_t>(slot.value()) * arena.leaseStride),
        sizeof(uint32_t) * words.size(), stream));
    cuCheck(cuStreamSynchronize(stream));
    std::vector<int32_t> out(words.size());
    std::transform(words.begin(), words.end(), out.begin(), [](uint32_t word) { return SparseEntry::ordinal(word); });
    return out;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
