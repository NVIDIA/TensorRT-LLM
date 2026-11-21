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

#pragma once

#include "tensorrt_llm/runtime/virtualMemory.h"

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace tensorrt_llm::runtime
{

//! \brief Device storage for an EPLB slot tensor whose unbacked (spare) slots hold no device memory of their own.
//!
//! The range stays fully mapped at a fixed address, because CUDA graphs capture it. It is mapped in chunks of the
//! allocation granularity. Chunks wholly inside unbacked slots map one shared scratch chunk, so a stray access reads
//! or writes scratch instead of faulting. Backing a slot maps memory of its own at the same address.
class UnbackedSlotMemory
{
public:
    static UnbackedSlotMemory& getInstance();

    //! Return whether allocate() would leave at least one chunk unbacked on the current device. If it would not, every
    //! byte gets memory of its own, so a plain allocation serves as well.
    bool savesMemory(size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots);
    //! Allocate slotCount slots of slotBytes each on the current device, leaving unbackedSlots unbacked.
    void* allocate(size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots, bool hostAccessible = false);
    //! Synchronize the range's device, then unmap and release the range. If the synchronization fails, as it does
    //! during a CUDA graph capture, log an error and leak the range instead.
    void free(void* ptr) noexcept;

    struct SlotRange
    {
        void* ptr;
        std::vector<int> slots;
    };

    //! Map memory of their own (contents undefined) or the scratch at the slots and return the bytes changed.
    //! A changed mapping synchronizes the device, so no kernel may use the slots and no CUDA graph capture may run.
    //! On failure every slot keeps its state.
    size_t setBacked(std::vector<SlotRange> const& ranges, bool backed);

    size_t setBacked(void* ptr, std::vector<int> const& slots, bool backed)
    {
        return setBacked({SlotRange{ptr, slots}}, backed);
    }

    //! Return the device bytes that setBacked(ptr, slots, true) would map now.
    size_t bytesToBack(void* ptr, std::vector<int> const& slots);

    //! Return whether address lies in an unbacked slot of host accessible slot memory. The slot maps the scratch, which
    //! has no host mapping, until backing it maps memory of its own for the host.
    bool isUnbacked(void const* address);

private:
    struct Scratch;

    //! A run of chunks whose bytes all belong to the unbacked slots [firstSlot, lastSlot], mapped as one unit. It maps
    //! the scratch while none of its slots is backed, and memory of its own once any of them is.
    struct Piece
    {
        CUdeviceptr address;
        size_t size;
        int firstSlot;
        // The slot count if the piece ends in the padding after the last slot, which rounds the range up to the
        // granularity.
        int lastSlot;
        bool backed;
        CUDAVirtualMemoryChunk chunk;
    };

    struct Allocation
    {
        // Declared first so that it is freed after the chunks are unmapped.
        std::shared_ptr<void> reservation;
        // Per slot, 1 if the slot was allocated unbacked.
        std::vector<uint8_t> unbacked;
        // Per slot, 1 if setBacked currently backs the unbacked slot.
        std::vector<uint8_t> backed;
        std::vector<Piece> pieces;
        // The chunks holding bytes of a slot allocated backed. They keep memory of their own for the allocation's
        // lifetime.
        std::vector<CUDAVirtualMemoryChunk> alwaysBacked;
        bool hostAccessible;
    };

    struct Layout
    {
        std::vector<uint8_t> unbacked;
        size_t chunks;
        // Per chunk: the unbacked slot holding all its bytes; -1 if any byte belongs to a slot allocated backed (the
        // chunk is always backed); or -2 - chunk if it spans several unbacked slots. A run of chunks with one value is
        // a piece. A chunk spanning several slots gets a value of its own, so adjacent spanning chunks never merge and
        // backing a slot backs only the chunks that hold its bytes.
        std::vector<int64_t> slotOf;
    };

    UnbackedSlotMemory() = default;

    //! Plan the layout of the slots on the current device, which becomes the device this instance serves.
    Layout plan(size_t slotBytes, int slotCount, std::vector<int> const& unbackedSlots);
    //! Return the first or last slot holding bytes of chunk, counting the tail as slot number slots.
    size_t firstSlotOfChunk(size_t chunk, size_t slotBytes, size_t slots) const;
    size_t lastSlotOfChunk(size_t chunk, size_t slotBytes, size_t slots) const;

    Allocation& findAllocation(void* ptr, std::vector<int> const& slots);
    //! Return whether backed marks any slot of piece, which is the state the piece must be mapped in.
    static bool anyBacked(Piece const& piece, std::vector<uint8_t> const& backed);

    //! Map [address, address + size) to memory of its own if backed, else to the scratch.
    CUDAVirtualMemoryChunk mapChunk(CUdeviceptr address, size_t size, bool backed, bool hostAccessible) const;
    //! Map a piece to its other state, or restore its state and throw.
    void remap(Piece& piece, bool hostAccessible) const;
    static void releaseChunks(Allocation& allocation) noexcept;

    std::mutex mMutex;
    std::map<CUdeviceptr, Allocation> mAllocations;
    int mDevice = -1;
    size_t mGranularity = 0;
    // The chunks that map the scratch keep it alive.
    mutable std::weak_ptr<Scratch const> mScratch;
};

} // namespace tensorrt_llm::runtime
