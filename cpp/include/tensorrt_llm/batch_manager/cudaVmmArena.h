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

#ifndef TRTLLM_CUDAVMMARENA_H
#define TRTLLM_CUDAVMMARENA_H

#include <cstddef>
#include <cuda.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensorrt_llm::batch_manager::vmm
{

/// Exception thrown for CUDA driver API errors.
class CudaVmmError : public std::runtime_error
{
public:
    explicit CudaVmmError(std::string const& msg, CUresult result = CUDA_SUCCESS)
        : std::runtime_error(msg)
        , result_(result)
    {
    }

    CUresult result() const noexcept
    {
        return result_;
    }

private:
    CUresult result_;
};

/// Manages a contiguous virtual address range backed by physical CUDA memory pages
/// that can be added and removed at runtime using the CUDA Virtual Memory Management API.
///
/// The arena reserves a fixed VA window of `max_size` bytes upfront, then commits
/// (maps) physical pages into it on demand in multiples of the device's allocation
/// granularity.  All committed memory is accessible on the owning device with
/// read/write permissions.
///
/// Typical usage:
///   CudaVmmArena arena(1ULL << 30, 0);  // Reserve 1 GiB VA on device 0
///   arena.grow(64 << 20);               // Commit first 64 MiB
///   void* p = reinterpret_cast<void*>(arena.ptr());
///   ...
///   arena.shrink(32 << 20);             // Release upper 32 MiB back to OS
///
/// Thread safety: not thread-safe; external synchronization is required.
class CudaVmmArena
{
public:
    /// Reserve `max_size` bytes of virtual address space on `device`.
    /// `max_size` is rounded up to the device's allocation granularity.
    /// No physical memory is allocated until grow() is called.
    explicit CudaVmmArena(size_t max_size, int device = 0);

    ~CudaVmmArena();

    // Non-copyable, non-movable: owns CUDA virtual/physical resources.
    CudaVmmArena(CudaVmmArena const&) = delete;
    CudaVmmArena& operator=(CudaVmmArena const&) = delete;
    CudaVmmArena(CudaVmmArena&&) = delete;
    CudaVmmArena& operator=(CudaVmmArena&&) = delete;

    // -----------------------------------------------------------------------
    // Resize operations
    // -----------------------------------------------------------------------

    /// Increase committed size to `new_size` by mapping additional physical pages.
    /// `new_size` is rounded up to granularity.
    /// Throws if new_size <= committed_size() or new_size > max_size().
    void grow(size_t new_size);

    /// Decrease committed size to `new_size` by unmapping and releasing tail pages.
    /// `new_size` is rounded down to the nearest granularity boundary.
    /// Throws if new_size >= committed_size().
    void shrink(size_t new_size);

    /// Convenience: call grow() or shrink() depending on `new_size`.
    /// A no-op if new_size (after alignment) equals committed_size().
    void resize(size_t new_size);

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Base device pointer of the reserved VA range.
    /// Only bytes in [ptr(), ptr() + committed_size()) are valid to access.
    CUdeviceptr ptr() const noexcept
    {
        return base_ptr_;
    }

    /// Number of bytes currently mapped to physical memory.
    size_t committed_size() const noexcept
    {
        return committed_size_;
    }

    /// Total reserved virtual address range (>= max_size passed to constructor).
    size_t max_size() const noexcept
    {
        return max_size_;
    }

    /// Physical allocation granularity in bytes for this device.
    size_t granularity() const noexcept
    {
        return granularity_;
    }

    /// CUDA device index this arena was created for.
    int device() const noexcept
    {
        return device_;
    }

private:
    // Allocate one granularity-sized physical handle, map it at `offset` into
    // the reserved VA range, and grant read/write access.
    void map_chunk(size_t offset);

    // Revoke access, unmap, and release the physical handle at slot `chunk_idx`.
    void unmap_chunk(size_t chunk_idx);

    // Throw CudaVmmError if `res` is not CUDA_SUCCESS.
    static void check(CUresult res, char const* where);

    // Round `n` up to the next multiple of `align` (which must be a power of 2).
    static size_t align_up(size_t n, size_t align) noexcept
    {
        return (n + align - 1) & ~(align - 1);
    }

    // Round `n` down to the previous multiple of `align`.
    static size_t align_down(size_t n, size_t align) noexcept
    {
        return n & ~(align - 1);
    }

    int device_;
    size_t granularity_;    ///< Minimum physical page granularity, bytes.
    size_t max_size_;       ///< Reserved VA range size (aligned up).
    size_t committed_size_; ///< Currently mapped byte count.
    CUdeviceptr base_ptr_;  ///< Start of the reserved VA range.

    /// One handle per committed granularity chunk, in order.
    std::vector<CUmemGenericAllocationHandle> handles_;

    CUmemAllocationProp alloc_prop_; ///< Shared allocation properties.
    CUmemAccessDesc access_desc_;    ///< Shared access descriptor.
};

} // namespace tensorrt_llm::batch_manager::vmm

#endif // TRTLLM_CUDAVMMARENA_H
