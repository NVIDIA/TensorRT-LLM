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

#include "tensorrt_llm/batch_manager/cudaVmmArena.h"

#include <cstring>
#include <sstream>

namespace tensorrt_llm::batch_manager::vmm
{

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

void CudaVmmArena::check(CUresult res, char const* where)
{
    if (res == CUDA_SUCCESS)
        return;

    char const* name = nullptr;
    char const* desc = nullptr;
    cuGetErrorName(res, &name);
    cuGetErrorString(res, &desc);

    std::ostringstream oss;
    oss << "CUDA VMM error in " << where << ": " << (name ? name : "?") << " (" << res << ")"
        << (desc ? std::string(" — ") + desc : std::string{});
    throw CudaVmmError(oss.str(), res);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

CudaVmmArena::CudaVmmArena(size_t max_size, int device)
    : device_(device)
    , granularity_(0)
    , max_size_(0)
    , committed_size_(0)
    , base_ptr_(0)
{
    // Build allocation properties: pinned device memory on the selected GPU.
    std::memset(&alloc_prop_, 0, sizeof(alloc_prop_));
    alloc_prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    alloc_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    alloc_prop_.location.id = device_;

    // Query the minimum granularity required by this device/allocation type.
    check(cuMemGetAllocationGranularity(&granularity_, &alloc_prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");

    if (granularity_ == 0)
        throw CudaVmmError("Device reported zero allocation granularity.");

    // Round requested max_size up to a granularity boundary.
    max_size_ = align_up(max_size, granularity_);
    if (max_size_ == 0)
        throw CudaVmmError("max_size rounds to zero after granularity alignment.");

    // Reserve the virtual address range.  No physical memory is allocated yet.
    check(cuMemAddressReserve(&base_ptr_, max_size_,
              /*alignment=*/0, /*hint=*/0, /*flags=*/0),
        "cuMemAddressReserve");

    // Pre-size the handle vector but leave all entries empty.
    handles_.reserve(max_size_ / granularity_);

    // Build the access descriptor once; reused for every chunk.
    std::memset(&access_desc_, 0, sizeof(access_desc_));
    access_desc_.location = alloc_prop_.location;
    access_desc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

CudaVmmArena::~CudaVmmArena()
{
    // Unmap and release all committed chunks in reverse order.
    for (size_t i = handles_.size(); i-- > 0;)
    {
        unmap_chunk(i);
    }
    handles_.clear();

    // Release the virtual address reservation.
    if (base_ptr_)
    {
        cuMemAddressFree(base_ptr_, max_size_);
        base_ptr_ = 0;
    }
}

// ---------------------------------------------------------------------------
// Private: map / unmap a single granularity-sized chunk
// ---------------------------------------------------------------------------

void CudaVmmArena::map_chunk(size_t offset)
{
    CUmemGenericAllocationHandle handle{};

    // Allocate one granularity-sized physical page.
    check(cuMemCreate(&handle, granularity_, &alloc_prop_, /*flags=*/0), "cuMemCreate");

    // Map the physical page into our reserved VA range at `offset`.
    CUresult res = cuMemMap(base_ptr_ + offset, granularity_,
        /*offset into handle=*/0, handle, /*flags=*/0);
    if (res != CUDA_SUCCESS)
    {
        cuMemRelease(handle); // best-effort cleanup
        check(res, "cuMemMap");
    }

    // Grant read/write access on the mapped range.
    res = cuMemSetAccess(base_ptr_ + offset, granularity_, &access_desc_, /*count=*/1);
    if (res != CUDA_SUCCESS)
    {
        cuMemUnmap(base_ptr_ + offset, granularity_);
        cuMemRelease(handle);
        check(res, "cuMemSetAccess");
    }

    handles_.push_back(handle);
}

void CudaVmmArena::unmap_chunk(size_t chunk_idx)
{
    const size_t offset = chunk_idx * granularity_;

    // Revoke access before unmapping (required by the CUDA VMM spec).
    CUmemAccessDesc no_access{};
    no_access.location = alloc_prop_.location;
    no_access.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    cuMemSetAccess(base_ptr_ + offset, granularity_, &no_access, 1);

    cuMemUnmap(base_ptr_ + offset, granularity_);
    cuMemRelease(handles_[chunk_idx]);
    handles_[chunk_idx] = CUmemGenericAllocationHandle{};
}

// ---------------------------------------------------------------------------
// Public: grow / shrink / resize
// ---------------------------------------------------------------------------

void CudaVmmArena::grow(size_t new_size)
{
    const size_t aligned = align_up(new_size, granularity_);

    if (aligned == 0)
        throw CudaVmmError("grow(): new_size rounds to zero.");
    if (aligned > max_size_)
        throw CudaVmmError("grow(): new_size exceeds the reserved VA range.");
    if (aligned <= committed_size_)
        throw CudaVmmError("grow(): new_size must be larger than current committed_size.");

    // Map chunks covering [committed_size_, aligned).
    size_t offset = committed_size_;
    while (offset < aligned)
    {
        map_chunk(offset); // may throw; already-mapped chunks stay valid
        offset += granularity_;
    }

    committed_size_ = aligned;
}

void CudaVmmArena::shrink(size_t new_size)
{
    // Round *down* so we never expose a partially-unmapped granule.
    const size_t aligned = align_down(new_size, granularity_);

    if (aligned >= committed_size_)
        throw CudaVmmError("shrink(): new_size must be smaller than current committed_size.");

    // Unmap chunks covering [aligned, committed_size_) in reverse order.
    size_t offset = committed_size_;
    while (offset > aligned)
    {
        offset -= granularity_;
        unmap_chunk(handles_.size() - 1);
        handles_.pop_back();
    }

    committed_size_ = aligned;
}

void CudaVmmArena::resize(size_t new_size)
{
    // Determine what the aligned target size would be without committing.
    const size_t aligned_up = align_up(new_size, granularity_);
    const size_t aligned_down = align_down(new_size, granularity_);

    if (aligned_up > committed_size_)
    {
        grow(new_size);
    }
    else if (aligned_down < committed_size_)
    {
        shrink(new_size);
    }
    // else: already at the right size, nothing to do.
}

} // namespace tensorrt_llm::batch_manager::vmm
