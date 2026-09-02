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

#include "coldPageCodec.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Direction of a host<->device page copy. The two directions need different optimization:
//! reads from host needs stronger latency hiding.
enum class CopyDirection : std::uint8_t
{
    //! device -> pinned host (KV cache offload / eviction).
    kD2H,
    //! pinned host -> device (KV cache onboard / recall).
    kH2D,
};

//! One pool's worth of work: a single source/destination base pointer pair plus a device-resident
//! list of (dst, src) page indices. Page size is uniform within a pool.
struct PoolCopyArgs
{
    //! Base of the destination pool/blob. Must be 16-byte aligned.
    CUdeviceptr dstBase = 0;
    //! Base of the source pool/blob. Must be 16-byte aligned.
    CUdeviceptr srcBase = 0;
    //! Bytes between consecutive destination pages. Must be a multiple of 16.
    uint64_t dstStride = 0;
    //! Bytes between consecutive source pages. Must be a multiple of 16.
    uint64_t srcStride = 0;
    //! Bytes copied per page. Must be a multiple of 16.
    uint32_t bytesPerPage = 0;
    //! Array of `numPairs` index pairs. It must reside in the memory space reported by
    //! BatchedPageCopier::pageIndexLocation(): device memory for the kernel path, host
    //! memory for the copy-engine path, which builds its descriptors on the CPU.
    PageIndexPair const* pairs = nullptr;
    //! Number of pages to copy.
    uint32_t numPairs = 0;
};

//! Chooses and launches the host<->device page-copy kernel.
//!
//! Construction performs all platform detection and grid sizing once, for both directions, so the
//! launch path is branch-light. The constants are measured; the short version:
//!
//!   * in-flight bytes per CTA is the only kernel parameter that matters, and the knees are
//!     platform-independent: 32 KiB for offload, 64 KiB for onboard;
//!   * CTA count scales with the CPU-GPU link bandwidth, sized via Little's Law;
//!   * on H100-class PCIe parts the SM store path is far slower than the copy engine, so the
//!     kernel is disabled and callers should keep using cuMemcpyBatchAsync.
class BatchedPageCopier
{
public:
    //! Kernel shape and grid for one direction. `threads` and `ilp` are fixed; only `stages`
    //! (pipeline depth) and `ctas` vary.
    struct KernelConfig
    {
        int threads = 0;
        int stages = 0;
        int ilp = 0;
        int ctas = 0;
        //! stages * threads * ilp * 16.
        uint32_t inFlightBytes = 0;
        //! Dynamic shared memory required per CTA, in bytes. Equal to inFlightBytes.
        uint32_t sharedBytes = 0;
        //! False when the copy engine is expected to beat the kernel; caller should fall back.
        bool useKernel = false;
    };

    //! Platform facts discovered at construction. Exposed for logging and tests.
    struct Topology
    {
        int device = -1;
        int smVersion = 0;
        int smCount = 0;
        //! True for NVLink-C2C (Grace); false for discrete PCIe attachment.
        bool coherentLink = false;
        //! Per-GPU CPU-GPU link bandwidth, one direction, in GB/s.
        double linkBandwidthGBs = 0.0;
        //! Host NUMA node closest to this GPU, or -1 if unknown. Pinned cache-tier memory MUST be
        //! allocated here: remote placement costs 1.6x-2.9x, more than any kernel parameter.
        int hostNumaId = -1;
        //! Number of GPUs sharing `hostNumaId`. Affects achievable bandwidth under load, not the
        //! CTA count.
        int gpusOnSameHostNuma = 1;
    };

    //! @param device CUDA device ordinal, or -1 for the current device.
    explicit BatchedPageCopier(int device = -1);

    [[nodiscard]] KernelConfig const& config(CopyDirection direction) const noexcept
    {
        return direction == CopyDirection::kD2H ? mOffload : mOnboard;
    }

    [[nodiscard]] Topology const& topology() const noexcept
    {
        return mTopology;
    }

    //! True when the copy kernel is used; false when the copy engine is used instead.
    [[nodiscard]] bool kernelEnabled() const noexcept
    {
        return mOffload.useKernel || mOnboard.useKernel;
    }

    //! Where PoolCopyArgs::pairs must live. The kernel reads the pairs on the device; the
    //! copy-engine path expands them into descriptors on the CPU, so it needs them in host
    //! memory. Callers must query this once and allocate their index array accordingly --
    //! passing an array in the wrong space is not detectable here.
    [[nodiscard]] PageIndexLocation pageIndexLocation() const noexcept
    {
        return kernelEnabled() ? PageIndexLocation::kDevice : PageIndexLocation::kHost;
    }

    //! Callers never need to split a copy at a HostMem registration-chunk boundary.
    //!
    //! The copy-engine path hands the driver explicit per-copy descriptors, which must lie inside
    //! a single registered region -- but it is only selected when chunked registration is off, so
    //! the whole host tier is one region. When chunking IS in force the dispatcher selects the
    //! kernel, which reaches pinned host memory through ordinary loads and stores on a contiguous
    //! virtual range and is therefore indifferent to how the range was registered. Either way a
    //! page may straddle a chunk boundary safely, and the caller can drop the splitting loop.

    //! Performs one pool's copy on `stream`, via whichever path this platform selected.
    //!
    //! Any alignment is accepted. A layout whose addresses, strides and `bytesPerPage` are all
    //! multiples of 16 takes the tuned path; anything else falls back to a simple generic kernel,
    //! which is correct but much slower. Real pool layouts are 16-byte aligned.
    //!
    //! A pool group with several pools is handled by calling this once per pool on the SAME
    //! stream: stream ordering serialises them, so each call may use the full grid and no CTA
    //! budget has to be divided. (Coalesced K+V gives poolsPerGroup == 1 in the common case, so
    //! this usually is a single call anyway.)
    //!
    //! Not thread-safe: the copy-engine path reuses per-dispatcher descriptor scratch.
    void launch(PoolCopyArgs const& args, CopyDirection direction, CUstream stream);

private:
    void detect(int device);
    void computeConfigs();
    void launchKernel(PoolCopyArgs const& args, CopyDirection direction, CUstream stream) const;
    //! Alignment-agnostic fallback, used when the pool layout is not a multiple of 16 bytes.
    void launchGenericKernel(PoolCopyArgs const& args, CopyDirection direction, CUstream stream) const;
    void launchCopyEngine(PoolCopyArgs const& args, CUstream stream);

    Topology mTopology{};
    KernelConfig mOffload{};
    KernelConfig mOnboard{};

    //! Descriptor scratch for the copy-engine path, reused across calls. Page identities change every
    //! eviction, so these cannot be cached -- only the allocation is.
    std::vector<CUdeviceptr> mCopyEngineDsts;
    std::vector<CUdeviceptr> mCopyEngineSrcs;
    std::vector<size_t> mCopyEngineSizes;
};

namespace detail
{

//! Copies ephemeral host page indices into device memory while leaving the destination update
//! asynchronous.
//!
//! Companion to BatchedPageCopier: it exists only because pageIndexLocation() may require the
//! index array to live in device memory, so callers on that path must stage it there first. This
//! moves the array that *describes* the copy, not the pages themselves.
void copyPageIndicesToDevice(CUdeviceptr dst, PageIndexPair const* src, size_t numPageIndices, CUstream stream);

#if CUDA_VERSION < 12080
//! Kernel fallback used when cuMemcpyBatchAsync is unavailable. Exposed for focused testing.
void copyPageIndicesToDeviceWithKernel(
    CUdeviceptr dst, PageIndexPair const* src, size_t numPageIndices, CUstream stream);
#endif

} // namespace detail

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
