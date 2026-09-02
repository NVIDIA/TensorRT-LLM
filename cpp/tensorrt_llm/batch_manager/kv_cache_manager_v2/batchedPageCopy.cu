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

#include "batchedPageCopy.h"

#include "utils/hostMem.h"

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvmlWrapper.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cuda_runtime_api.h>
#include <nvml.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{
//! Threads per CTA. Note that this kernel is optimized for perf to resource usage ratio, not raw
//! perf! Because this kernel will run in parallel with LLM inference kernels.
constexpr int kThreads = 128;
//! Consumer ILP.
constexpr int kIlp = 2;
//! In-flight bytes per CTA. Offload saturates at 32 KiB, onboard needs 64 KiB because reads from
//! host must cover a full round trip. These are the platform-independent knees.
constexpr uint32_t kOffloadInFlightBytes = 32u * 1024u;
constexpr uint32_t kOnboardInFlightBytes = 64u * 1024u;
constexpr uint32_t kStageBytesPerCta = static_cast<uint32_t>(kThreads) * kIlp * 16u;
constexpr int kOffloadStages = static_cast<int>(kOffloadInFlightBytes / kStageBytesPerCta);
constexpr int kOnboardStages = static_cast<int>(kOnboardInFlightBytes / kStageBytesPerCta);
static_assert(kOffloadStages >= 2 && kOnboardStages >= 2, "thread count too large for these depths");

//! Per-CTA throughput of this kernel, the `lambda` of a Little's Law concurrency estimate.
//! Profiled on GH200 at 1 MiB pages and fitted to `X = min(N*lambda, Xmax)` (0.78% RMS).
//!
//! C2C figure: per-CTA bandwidth is in-flight/latency, so a PCIe path would need its own value.
//! Keep branches and selects out of the page-base load's address computation; one there cost 29%.
constexpr double kPerCtaGBs = 43.0;
//! Fraction of nominal link bandwidth actually achieved (measured 79%-86%).
constexpr double kLinkEfficiency = 0.80;
//! Sustained per-socket host bandwidth on Grace, direction-symmetric: GH200 peaks at 355 GB/s
//! writing and 351 GB/s reading against a 447 GB/s link that does not bind. ~69% of the 512 GB/s
//! LPDDR5X spec. PCIe hosts are never limited by this, so it is applied only on coherent links.
constexpr double kHostSustainedGBs = 355.0;
//! Grid multipliers. The directions differ in curve shape, not per-CTA throughput: offload is
//! linear then hard-clamps, onboard rolls off gradually (soft knee, p = 2.2) and must overshoot.
//! Selects 8 and 12 CTAs on GH200, at 97% and 94% of plateau.
constexpr double kOffloadGridMult = 0.90;
constexpr double kOnboardGridMult = 1.45;

//! Conservative link bandwidths used when NVML is unavailable.
constexpr double kFallbackCoherentGBs = 223.6; //!< the narrower (2 GPUs per Grace) C2C config
constexpr double kFallbackPcieGBs = 64.0;      //!< PCIe Gen5 x16, one direction

// ---------------------------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------------------------

//! Fields common to both page-size regimes.
struct CopyArgs
{
    char* dstBase;
    char const* srcBase;
    uint64_t dstStride;
    uint64_t srcStride;
    PageIndexPair const* pairs;
    uint32_t numPairs;
    //! bytesPerPage / 16.
    uint32_t elemsPerPage;
};

//! Pages at least one tile: a tile covers part of one page.
struct LargePageArgs
{
    CopyArgs common;
    //! Tiles needed to cover one page.
    uint32_t tilesPerPage;
};

//! Pages smaller than one tile: whole pages are packed into a tile so small pages still fill the
//! CTA. A 64-byte page is 4 elements; without this it would occupy 4 of 256 slots.
struct SmallPageArgs
{
    CopyArgs common;
    //! Whole pages carried by one tile.
    uint32_t pagesPerTile;
};

__device__ __forceinline__ uint32_t toSharedAddress(void const* pointer)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

//! `srcSize` is cp.async's source-size operand: bytes actually read from global, with the balance
//! of the 16-byte destination zero-filled. Passing 0 for out-of-range elements makes the issue
//! unconditional, so the steady-state loop is straight-line and the compiler can batch the async
//! copies instead of scheduling around a branch.
//!
//! `global` is always a real, mapped address even when srcSize is 0 -- the mappers form it from a
//! live page base rather than passing null. The PTX ISA does not actually promise the address
//! operand is ignored when srcSize is 0 (it provides a separate `ignore-src` predicate form for
//! that), so relying on a null being safe here would be an unwarranted assumption.
__device__ __forceinline__ void asyncCopy16(void* shared, void const* global, uint32_t srcSize)
{
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(toSharedAddress(shared)), "l"(global), "r"(srcSize)
        : "memory");
}

__device__ __forceinline__ void commitAsyncGroup()
{
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int PendingGroups>
__device__ __forceinline__ void waitAsyncGroup()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(PendingGroups) : "memory");
}

//! Translates a page index into its source/destination base addresses. This is the only global
//! memory read on the address path, so both mappers cache its result in the cursor: a `cp.async`
//! issue that *depends* on a global load cannot be hidden by the pipeline, and reloading per
//! element costs roughly half the achievable bandwidth.
__device__ __forceinline__ void pageBasesFor(CopyArgs const& args, uint32_t page, char const*& srcPage, char*& dstPage)
{
    // Plain index: any select here would sit in the dependency chain of this load, which is the
    // only global read on the address path and therefore latency-critical. Callers guarantee the
    // page is in range instead, checked here in debug builds only.
    assert(page < args.numPairs);
    PageIndexPair const pair = args.pairs[page];
    // A negative index would sign-extend into a huge unsigned offset and produce a wild address
    // rather than a clean fault.
    assert(pair.src >= 0 && pair.dst >= 0);
    srcPage = args.srcBase + static_cast<uint64_t>(pair.src) * args.srcStride;
    dstPage = args.dstBase + static_cast<uint64_t>(pair.dst) * args.dstStride;
}

//! Maps tiles onto pages when a page spans one or more tiles.
//!
//! The cursor holds (page, chunk-within-page) plus that page's cached base addresses. Advancing is
//! a compare-and-increment, and the page bases are re-read only when a page boundary is crossed --
//! once every `tilesPerPage` tiles, so once per 1024 tiles for a 1 MiB page. The only integer
//! division in the whole kernel is the one that seeds the cursor.
template <int Threads, int Ilp, bool Exact>
class LargePageMapper
{
public:
    //! `srcTile`/`dstTile` already point at this tile's first element, so `resolve` is a single
    //! 64-bit add of a per-thread constant -- no chunk arithmetic, no shift, on the address path.
    //! `elemsThisChunk` lifts the ragged-tail bound out of the address chain: it is a plain
    //! register compare that only differs from a full tile on a page's last chunk.
    struct Cursor
    {
        uint32_t page;
        uint32_t chunk;
        uint32_t elemsThisChunk;
        char const* srcTile;
        char* dstTile;
    };

    __device__ explicit LargePageMapper(LargePageArgs const& args)
        : mArgs(args)
    {
        uint32_t const perTile = static_cast<uint32_t>(Threads * Ilp);
        uint32_t const remainder = mArgs.common.elemsPerPage % perTile;
        mLastChunkElems = remainder == 0 ? perTile : remainder;
#pragma unroll
        for (int item = 0; item < Ilp; ++item)
        {
            mOffsetInChunk[item] = static_cast<uint32_t>(item * Threads + threadIdx.x);
            mByteOffsetInChunk[item] = mOffsetInChunk[item] * 16u;
        }
    }

    __device__ Cursor cursorAt(uint32_t tile) const
    {
        Cursor cursor{};
        cursor.page = tile / mArgs.tilesPerPage; // the only division
        cursor.chunk = tile - cursor.page * mArgs.tilesPerPage;
        char const* srcPage = nullptr;
        char* dstPage = nullptr;
        pageBasesFor(mArgs.common, cursor.page, srcPage, dstPage);
        uint64_t const chunkBytes = static_cast<uint64_t>(cursor.chunk) * kStageBytes;
        cursor.srcTile = srcPage + chunkBytes;
        cursor.dstTile = dstPage + chunkBytes;
        if constexpr (!Exact)
        {
            setChunkBound(cursor);
        }
        return cursor;
    }

    __device__ void advance(Cursor& cursor) const
    {
        if (++cursor.chunk == mArgs.tilesPerPage)
        {
            cursor.chunk = 0;
            ++cursor.page;
            if (cursor.page < mArgs.common.numPairs)
            {
                pageBasesFor(mArgs.common, cursor.page, cursor.srcTile, cursor.dstTile);
            }
            // Past the end the cursor keeps the previous page's bases. It is never resolved, so
            // the stale value is unused; leaving it alone keeps the range test off the hot path.
        }
        else
        {
            // Unconditional: pageBasesFor never yields null, so the old null test was dead. With
            // tilesPerPage large this is the dominant path (255 of 256 advances at 1 MiB pages).
            cursor.srcTile += kStageBytes;
            cursor.dstTile += kStageBytes;
        }
        if constexpr (!Exact)
        {
            // Exact never reads elemsThisChunk (resolve returns true unconditionally), so the
            // per-tile bound computation is dead there.
            setChunkBound(cursor);
        }
    }

    __device__ bool resolve(Cursor const& cursor, int item, char const*& src, char*& dst) const
    {
        // `Exact` is set by the host when bytesPerPage is a whole number of tiles AND every tile
        // in the grid maps to a live page, so no element can fall out of range. Specialising at
        // launch time (rather than branching per tile) lets the compiler drop the predicates and
        // the `active[]` liveness entirely.
        // Addresses are formed unconditionally, so `src` is always a real address derived from a
        // live page base -- never null. cp.async is issued for out-of-range elements with srcSize
        // 0, and while that reads nothing, the ISA does not promise the address operand itself is
        // ignored. The out-of-range address stays within (or just past) the page whose base is
        // cached, which is mapped either way.
        //
        // Live tiles always have a page base: tile < totalTiles == numPairs * tilesPerPage implies
        // page < numPairs, so pageBasesFor cannot have nulled the cursor. Debug-only assert, to
        // keep the release path free of the test.
        assert(cursor.srcTile != nullptr && cursor.dstTile != nullptr);
        src = cursor.srcTile + mByteOffsetInChunk[item];
        dst = cursor.dstTile + mByteOffsetInChunk[item];
        if constexpr (!Exact)
        {
            // `Exact` is set by the host when bytesPerPage is a whole number of tiles AND every
            // tile in the grid maps to a live page, so no element can fall out of range.
            return mOffsetInChunk[item] < cursor.elemsThisChunk;
        }
        return true;
    }

private:
    static constexpr uint64_t kStageBytes = static_cast<uint64_t>(Threads) * Ilp * 16u;

    __device__ void setChunkBound(Cursor& cursor) const
    {
        cursor.elemsThisChunk
            = (cursor.chunk + 1 == mArgs.tilesPerPage) ? mLastChunkElems : static_cast<uint32_t>(Threads * Ilp);
    }

    LargePageArgs const& mArgs;
    uint32_t mOffsetInChunk[Ilp];
    uint32_t mByteOffsetInChunk[Ilp];
    uint32_t mLastChunkElems;
};

//! Maps tiles onto pages when several whole pages fit in one tile.
//!
//! Each thread sits at a fixed (page-within-tile, offset-within-page) for the entire kernel, so
//! the two divisions that establish it run once at construction and never in the loop. The cursor
//! is just the tile's first page and advances by addition.
template <int Threads, int Ilp>
class SmallPageMapper
{
public:
    //! Each thread owns a different page within the tile, so the cursor caches one base-address
    //! pair per item. The pages change every tile, so unlike the large-page case this cannot be
    //! amortised further -- but it still keeps the global read off the `cp.async` issue path and
    //! avoids repeating it between the prefetch and the write-back.
    //! Base addresses already include this thread's byte offset within its page, so `resolve` is a
    //! null test and two register reads -- no arithmetic on the address path at all.
    struct Cursor
    {
        uint32_t firstPage;
        char const* srcElem[Ilp];
        char* dstElem[Ilp];
        //! Whether this item maps to a live page. Kept explicitly rather than encoded as a null
        //! address, so the cp.async source is always a real address (see resolve()).
        bool live[Ilp];
    };

    __device__ explicit SmallPageMapper(SmallPageArgs const& args)
        : mArgs(args)
    {
#pragma unroll
        for (int item = 0; item < Ilp; ++item)
        {
            uint32_t const slot = static_cast<uint32_t>(item * Threads + threadIdx.x);
            mPageInTile[item] = slot / args.common.elemsPerPage;
            uint32_t const offsetInPage = slot - mPageInTile[item] * args.common.elemsPerPage;
            mByteOffsetInPage[item] = offsetInPage * 16u;
            // Slots past the last whole page in the tile are idle. This is what bounds mode-B
            // efficiency to floor(N/e)*e/N, worst (~50%) when a page is just over half a tile.
            mActive[item] = mPageInTile[item] < args.pagesPerTile;
        }
    }

    __device__ Cursor cursorAt(uint32_t tile) const
    {
        Cursor cursor{};
        cursor.firstPage = tile * mArgs.pagesPerTile;
        loadPages(cursor);
        return cursor;
    }

    __device__ void advance(Cursor& cursor) const
    {
        cursor.firstPage += mArgs.pagesPerTile;
        loadPages(cursor);
    }

    __device__ bool resolve(Cursor const& cursor, int item, char const*& src, char*& dst) const
    {
        // Always a real address (see loadPages); liveness is carried separately.
        assert(cursor.srcElem[item] != nullptr && cursor.dstElem[item] != nullptr);
        src = cursor.srcElem[item];
        dst = cursor.dstElem[item];
        return cursor.live[item];
    }

private:
    __device__ void loadPages(Cursor& cursor) const
    {
#pragma unroll
        for (int item = 0; item < Ilp; ++item)
        {
            // Unlike the large-page case, the final tile really can address past the last page,
            // and slots past the last whole page in a tile are idle by construction. Both are
            // resolved to page 0 rather than to null so that the address handed to cp.async is
            // always mapped; the element is marked not-live and neither read nor stored.
            // mByteOffsetInPage is always < bytesPerPage, so the clamped address stays inside
            // page 0.
            uint32_t const page = cursor.firstPage + mPageInTile[item];
            bool const live = mActive[item] && page < mArgs.common.numPairs;
            char const* srcPage = nullptr;
            char* dstPage = nullptr;
            pageBasesFor(mArgs.common, live ? page : 0u, srcPage, dstPage);
            cursor.srcElem[item] = srcPage + mByteOffsetInPage[item];
            cursor.dstElem[item] = dstPage + mByteOffsetInPage[item];
            cursor.live[item] = live;
        }
    }

    SmallPageArgs const& mArgs;
    uint32_t mPageInTile[Ilp];
    uint32_t mByteOffsetInPage[Ilp];
    bool mActive[Ilp];
};

//! LDGSTS copy pipeline, shared by both page-size regimes.
//!
//! Each CTA takes a contiguous run of tiles, which keeps its accesses sequential and lets the
//! pipeline stay full. `Mapper` supplies the tile -> (page, offset) mapping; everything about the
//! staging, prefetch depth and write-back is identical between the two.
template <int Threads, int Stages, int Ilp, class Mapper>
__device__ __forceinline__ void runCopyPipeline(Mapper const& mapper, uint32_t totalTiles)
{
    constexpr int kElemsPerTile = Threads * Ilp;
    constexpr uint32_t kStageBytes = static_cast<uint32_t>(kElemsPerTile) * 16u;
    using Cursor = typename Mapper::Cursor;

    extern __shared__ __align__(16) char sharedMemory[];

    uint32_t const perBlock = (totalTiles + gridDim.x - 1) / gridDim.x;
    uint32_t const begin = blockIdx.x * perBlock;
    uint32_t const end = min(begin + perBlock, totalTiles);
    if (begin >= end)
    {
        return;
    }

    Cursor store = mapper.cursorAt(begin);

    auto issue = [&](Cursor const& cursor, char* stageBuffer)
    {
#pragma unroll
        for (int item = 0; item < Ilp; ++item)
        {
            char const* src = nullptr;
            char* dst = nullptr;
            bool const valid = mapper.resolve(cursor, item, src, dst);
            asyncCopy16(stageBuffer + static_cast<size_t>(item * Threads + threadIdx.x) * 16u, src, valid ? 16u : 0u);
        }
        commitAsyncGroup();
    };

    // Prologue: fill the pipeline. `load` trails `store` by exactly Stages tiles from here on.
    Cursor load = store;
#pragma unroll
    for (int stage = 0; stage < Stages; ++stage)
    {
        if (begin + static_cast<uint32_t>(stage) < end)
        {
            issue(load, sharedMemory + static_cast<size_t>(stage) * kStageBytes);
        }
        else
        {
            commitAsyncGroup(); // keep group accounting aligned with the wait below
        }
        mapper.advance(load);
    }

    // The last `Stages` tiles have no successor to prefetch, so they need a shallower wait and no
    // issue. Hoisting that split out of the loop makes the steady-state body branch-free: both the
    // wait depth and the issue become unconditional, which is worth ~2 branches per element.
    uint32_t const tail = static_cast<uint32_t>(Stages);
    uint32_t const steadyEnd = (end - begin > tail) ? end - tail : begin;

    // Stages is a compile-time constant, and at the default 128 threads it is a power of two
    // (8 or 16), so this lowers to a mask. A thread-count override that makes it non-power-of-two
    // (e.g. 192 threads gives 5) still compiles, but pays a real division here.
    auto stageBufferFor = [&](uint32_t tile) -> char*
    {
        int const stage = static_cast<int>((tile - begin) % static_cast<uint32_t>(Stages));
        return sharedMemory + static_cast<size_t>(stage) * kStageBytes;
    };

    // Drain a stage into registers. Must happen before the buffer is reused by the prefetch.
    //
    // This is a WAR on the stage buffer with no barrier, which is safe on both counts: across
    // threads there is no hazard (each thread owns its own 16-byte slot), and within a thread the
    // reissue cannot be hoisted above these loads because asyncCopy16's asm carries a "memory"
    // clobber, a full compiler-ordering barrier. No fence instruction is needed.
    auto drain = [&](char* stageBuffer, uint4(&values)[Ilp], bool(&active)[Ilp], char*(&targets)[Ilp])
    {
#pragma unroll
        for (int item = 0; item < Ilp; ++item)
        {
            char const* src = nullptr;
            char* dst = nullptr;
            active[item] = mapper.resolve(store, item, src, dst);
            targets[item] = dst;
            if (active[item])
            {
                values[item] = *reinterpret_cast<uint4 const*>(
                    stageBuffer + static_cast<size_t>(item * Threads + threadIdx.x) * 16u);
            }
        }
    };

    auto writeOut = [&](uint4 const(&values)[Ilp], bool const(&active)[Ilp], char* const(&targets)[Ilp])
    {
#pragma unroll
        for (int item = 0; item < Ilp; ++item)
        {
            if (active[item])
            {
                *reinterpret_cast<uint4*>(targets[item]) = values[item];
            }
        }
    };

    // Steady state: every tile has a successor `Stages` ahead, so the pipeline stays exactly
    // `Stages` groups deep and nothing here is conditional on the tile index.
    for (uint32_t tile = begin; tile < steadyEnd; ++tile)
    {
        waitAsyncGroup<Stages - 1>();
        char* const stageBuffer = stageBufferFor(tile);
        uint4 values[Ilp];
        bool active[Ilp];
        char* targets[Ilp];
        drain(stageBuffer, values, active, targets);
        issue(load, stageBuffer);
        mapper.advance(load);
        writeOut(values, active, targets);
        mapper.advance(store);
    }

    // Epilogue: drain the remaining in-flight stages. `load` is dead from here, so it is not
    // advanced. waitAsyncGroup<0> is over-strict (it retires every outstanding group on the first
    // iteration rather than one at a time) but this runs at most `Stages` times.
    for (uint32_t tile = steadyEnd; tile < end; ++tile)
    {
        waitAsyncGroup<0>();
        char* const stageBuffer = stageBufferFor(tile);
        uint4 values[Ilp];
        bool active[Ilp];
        char* targets[Ilp];
        drain(stageBuffer, values, active, targets);
        writeOut(values, active, targets);
        mapper.advance(store);
    }
}

template <int Threads, int Stages, int Ilp, bool Exact>
__global__ __launch_bounds__(Threads) void largePageCopyKernel(LargePageArgs const args, uint32_t const totalTiles)
{
    runCopyPipeline<Threads, Stages, Ilp>(LargePageMapper<Threads, Ilp, Exact>(args), totalTiles);
}

template <int Threads, int Stages, int Ilp>
__global__ __launch_bounds__(Threads) void smallPageCopyKernel(SmallPageArgs const args, uint32_t const totalTiles)
{
    runCopyPipeline<Threads, Stages, Ilp>(SmallPageMapper<Threads, Ilp>(args), totalTiles);
}

//! Alignment-agnostic fallback: one page per block, byte at a time, no pipelining and no
//! vectorisation.
//!
//! The fast kernels above move 16 bytes per element and therefore require the page size, both
//! strides and both base addresses to be multiples of 16. Pool layouts are not obliged to satisfy
//! that -- cold offsets are prefix sums of arbitrary per-pool slot sizes -- so this exists to keep
//! such layouts working rather than rejecting them. It is deliberately simple; real layouts are
//! 16-byte aligned and take the fast path, so this is a correctness backstop, not a perf path.
//!
//! `bytesPerPage` is passed explicitly because CopyArgs::elemsPerPage assumes a 16-byte multiple.
template <int Threads>
__global__ __launch_bounds__(Threads) void genericPageCopyKernel(CopyArgs const args, uint32_t const bytesPerPage)
{
    for (uint32_t page = blockIdx.x; page < args.numPairs; page += gridDim.x)
    {
        PageIndexPair const pair = args.pairs[page];
        assert(pair.src >= 0 && pair.dst >= 0);
        char const* const src = args.srcBase + static_cast<uint64_t>(pair.src) * args.srcStride;
        char* const dst = args.dstBase + static_cast<uint64_t>(pair.dst) * args.dstStride;
        for (uint32_t offset = threadIdx.x; offset < bytesPerPage; offset += Threads)
        {
            dst[offset] = src[offset];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Detection helpers
// ---------------------------------------------------------------------------------------------

//! Per-lane signalling rate of one PCIe generation, in GT/s.
double pcieGigaTransfersPerSecond(unsigned int generation)
{
    switch (generation)
    {
    case 1: return 2.5;
    case 2: return 5.0;
    case 3: return 8.0;
    case 4: return 16.0;
    case 5: return 32.0;
    case 6: return 64.0;
    default: return 0.0;
    }
}

//! The two link properties the copier needs.
struct LinkInfo
{
    //! True on a coherent CPU-GPU link (NVLink-C2C). Selects the copy path.
    bool coherent = false;
    //! Aggregate one-direction link bandwidth, always populated -- a conservative default when it
    //! cannot be measured. Only feeds the CTA count.
    double bandwidthGBs = 0.0;
};

//! Best-effort NVML refinement of what CUDA already established. Every output is optional in the
//! "may be left as the caller set it" sense: nvmlDeviceGetFieldValues() returns NVML_SUCCESS when
//! *any* requested field was populated, each field carrying its own nvmlReturn, so partial answers
//! are ordinary. `coherent` is only ever raised, never cleared -- NVML failing to describe a link
//! says nothing about whether one exists. The two bandwidths stay separate because a PCIe reading
//! must not be applied to a coherent link.
void refineFromNvml(int device, bool& coherent, double& c2cBandwidthGBs, double& pcieBandwidthGBs)
{
    // NVML is reached through NVMLWrapper, which dlopens libnvidia-ml.so.1 and resolves symbols at
    // runtime. TensorRT-LLM deliberately does not link NVML: the only build-time library available
    // is the CUDA stub, and a hard dependency would break loading where the driver library is
    // absent. Optional symbols return NVML_ERROR_FUNCTION_NOT_FOUND, which lands in the same
    // fallback path as any other failure here.
    std::shared_ptr<tensorrt_llm::common::NVMLWrapper> nvml;
    try
    {
        nvml = tensorrt_llm::common::NVMLWrapper::getInstance();
    }
    catch (tensorrt_llm::common::TllmException const& error)
    {
        // getInstance() throws exactly this for the two expected conditions: libnvidia-ml.so.1
        // absent, or a required symbol missing. Anything else propagates rather than being
        // silently downgraded to "no NVML".
        TLLM_LOG_DEBUG("NVML unavailable, falling back to CUDA-only link detection: %s", error.what());
        return;
    }
    if (nvml->nvmlInit() != NVML_SUCCESS)
    {
        return;
    }
    // Match by PCI bus id: NVML enumerates all GPUs, CUDA only the visible ones.
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
    nvmlDevice_t handle{};
    if (cudaDeviceGetPCIBusId(busId, sizeof(busId), device) == cudaSuccess
        && nvml->nvmlDeviceGetHandleByPciBusId(busId, &handle) == NVML_SUCCESS)
    {
        nvmlFieldValue_t fields[2] = {};
        fields[0].fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
        fields[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
        if (nvml->nvmlDeviceGetFieldValues(handle, 2, fields) == NVML_SUCCESS && fields[0].nvmlReturn == NVML_SUCCESS)
        {
            int const linkCount = static_cast<int>(
                fields[0].valueType == NVML_VALUE_TYPE_UNSIGNED_INT ? fields[0].value.uiVal : fields[0].value.ullVal);
            if (linkCount > 0)
            {
                coherent = true;
                if (fields[1].nvmlReturn == NVML_SUCCESS)
                {
                    // NVML reports C2C link speed in MBps.
                    double const perLink = (fields[1].valueType == NVML_VALUE_TYPE_UNSIGNED_INT
                                                   ? static_cast<double>(fields[1].value.uiVal)
                                                   : static_cast<double>(fields[1].value.ullVal))
                        / 1000.0;
                    c2cBandwidthGBs = linkCount * perLink;
                }
                else
                {
                    // C2C_LINK_GET_MAX_BW reports the speed of *active* links, so it can
                    // legitimately be unavailable while the link count is known -- links still
                    // training, a driver that predates the field, or a virtualized or
                    // permission-restricted environment. The count already settles the topology;
                    // only the tuning input is missing.
                    TLLM_LOG_DEBUG("NVML reported %d C2C link(s) but no max bandwidth; using the default.", linkCount);
                }
            }
        }
        if (c2cBandwidthGBs <= 0.0)
        {
            // Read whenever no C2C bandwidth was obtained, inferring nothing about the attachment:
            // the caller applies it only if the link is not coherent. Gating it on a link count of
            // 0 would skip it on every host where the C2C fields are NOT_SUPPORTED -- ordinary
            // discrete GPUs, exactly where it is the only real reading available.
            unsigned int generation = 0;
            unsigned int width = 0;
            if (nvml->nvmlDeviceGetMaxPcieLinkGeneration(handle, &generation) == NVML_SUCCESS
                && nvml->nvmlDeviceGetMaxPcieLinkWidth(handle, &width) == NVML_SUCCESS)
            {
                pcieBandwidthGBs = pcieGigaTransfersPerSecond(generation) * width / 8.0;
            }
        }
    }
    nvml->nvmlShutdown();
}

//! Detects the CPU-GPU link. Both fields are always resolved: anything NVML cannot answer falls
//! back to a conservative default here rather than at the call site.
LinkInfo detectLink(int device)
{
    LinkInfo info;

    // Coherence comes from CUDA, not NVML: ATS-backed pageable access is the Grace/C2C signature,
    // and unlike a multi-field NVML query it cannot partially fail. (Compute capability cannot be
    // used because B200 and GB200 are both sm_100, and H100 and GH200 are both sm_90. HMM does not
    // false-positive: it sets cudaDevAttrPageableMemoryAccess, not ...UsesHostPageTables.)
    int usesHostPageTables = 0;
    if (cudaDeviceGetAttribute(&usesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device)
        != cudaSuccess)
    {
        usesHostPageTables = 0;
        cudaGetLastError();
    }
    info.coherent = usesHostPageTables != 0;

    double c2cBandwidthGBs = 0.0;
    double pcieBandwidthGBs = 0.0;
    refineFromNvml(device, info.coherent, c2cBandwidthGBs, pcieBandwidthGBs);

    double const measured = info.coherent ? c2cBandwidthGBs : pcieBandwidthGBs;
    info.bandwidthGBs = measured > 0.0 ? measured : (info.coherent ? kFallbackCoherentGBs : kFallbackPcieGBs);
    return info;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// BatchedPageCopier
// ---------------------------------------------------------------------------------------------

BatchedPageCopier::BatchedPageCopier(int device)
{
    if (device < 0)
    {
        TLLM_CUDA_CHECK(cudaGetDevice(&device));
    }
    detect(device);
    computeConfigs();
}

void BatchedPageCopier::detect(int device)
{
    mTopology.device = device;

    int major = 0;
    int minor = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    mTopology.smVersion = major * 10 + minor;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&mTopology.smCount, cudaDevAttrMultiProcessorCount, device));

    // Host NUMA affinity. Pinned cache-tier memory must be allocated on this node.
    if (cudaDeviceGetAttribute(&mTopology.hostNumaId, cudaDevAttrHostNumaId, device) != cudaSuccess)
    {
        mTopology.hostNumaId = -1;
        cudaGetLastError();
    }

    // `coherent` selects the copy path; `bandwidthGBs` only feeds the CTA count below. Keeping them
    // independent is the point: treating an unanswerable bandwidth query as "discrete PCIe" is what
    // silently disabled the kernel on Grace.
    LinkInfo const link = detectLink(device);
    mTopology.coherentLink = link.coherent;
    mTopology.linkBandwidthGBs = link.bandwidthGBs;

    // How many GPUs share this host NUMA node? Affects achievable bandwidth under load, but
    // deliberately NOT the CTA count: contention does not reduce the concurrency needed to
    // saturate a GPU's own share.
    mTopology.gpusOnSameHostNuma = 1;
    if (mTopology.hostNumaId >= 0)
    {
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) == cudaSuccess)
        {
            int shared = 0;
            for (int other = 0; other < deviceCount; ++other)
            {
                int numa = -1;
                if (cudaDeviceGetAttribute(&numa, cudaDevAttrHostNumaId, other) == cudaSuccess
                    && numa == mTopology.hostNumaId)
                {
                    ++shared;
                }
            }
            mTopology.gpusOnSameHostNuma = std::max(1, shared);
        }
        cudaGetLastError();
    }
}

void BatchedPageCopier::computeConfigs()
{
    // The kernel is used only on coherent (NVLink-C2C) links. On discrete PCIe it loses to the
    // copy engine in both directions and on both architectures measured:
    //
    //   H100 PCIe Gen4  offload 8.56 vs 17.45 GiB/s (0.49x)   onboard 23.89 vs 23.88 (1.00x)
    //   B200 PCIe Gen5  offload 25.13 vs 43.17 GiB/s (0.58x)  onboard 39.99 vs 42.58 (0.94x)
    //
    // The two have different causes. Hopper's SM->host store path is capped near 8.5 GiB/s in
    // hardware, which no amount of pipelining or extra CTAs recovers. Blackwell has no such cap,
    // but PCIe round-trip latency is high enough that a fixed 32 KiB in flight yields only
    // ~13.5 GB/s per CTA (vs ~30 on C2C), so the grid sizing below under-provisions it. That is
    // fixable with a per-link-class kPerCtaGBs, but the ceiling is the copy engine either way, so
    // PCIe simply uses cuMemcpyBatchAsync -- which also costs zero SMs.
    // Chunked host registration (a kernel-version workaround; see HostMem) breaks the copy-engine path,
    // whose descriptors may not cross a registered region. The kernel works on virtual addresses
    // and does not care, so chunking forces the kernel on regardless of link type -- accepting the
    // PCIe slowdown above rather than reintroducing per-page splitting on the CPU.
    bool const kernelViable = mTopology.coherentLink || HostMem::shouldUseChunkedRegistration();

    auto build = [&](int stages, double gridMult) -> KernelConfig
    {
        KernelConfig config{};
        config.threads = kThreads;
        config.ilp = kIlp;
        config.stages = stages;
        config.inFlightBytes = static_cast<uint32_t>(kThreads) * kIlp * 16u * static_cast<uint32_t>(stages);
        config.sharedBytes = config.inFlightBytes;
        config.useKernel = kernelViable;

        double cap = kLinkEfficiency * mTopology.linkBandwidthGBs;
        if (mTopology.coherentLink)
        {
            cap = std::min(cap, kHostSustainedGBs);
        }
        double const ideal = cap / kPerCtaGBs; // Little's Law concurrency estimate
        double const target = gridMult * ideal;

        // Round to the nearest EVEN CTA count so the grid occupies whole SM pairs and does not
        // strand half-pairs away from tcgen05 2-CTA MMA kernels. Applied unconditionally: this
        // differs from nearest-integer by at most one CTA, which is cheaper than maintaining an
        // architecture predicate (no device attribute reports tcgen05, and every concise version
        // test is wrong for some shipped part -- sm_103 has it, sm_120 does not).
        int const ctas = 2 * static_cast<int>(std::lround(target / 2.0));
        config.ctas = std::clamp(ctas, 2, std::max(2, mTopology.smCount));
        return config;
    };

    mOffload = build(kOffloadStages, kOffloadGridMult);
    mOnboard = build(kOnboardStages, kOnboardGridMult);

    if (!kernelViable)
    {
        return;
    }

    // Depths above 48 KiB need the opt-in shared-memory attribute, on all four instantiations
    // (two page-size regimes x two directions).
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(largePageCopyKernel<kThreads, kOffloadStages, kIlp, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mOffload.sharedBytes)));
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(largePageCopyKernel<kThreads, kOffloadStages, kIlp, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mOffload.sharedBytes)));
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(smallPageCopyKernel<kThreads, kOffloadStages, kIlp>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mOffload.sharedBytes)));
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(largePageCopyKernel<kThreads, kOnboardStages, kIlp, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mOnboard.sharedBytes)));
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(largePageCopyKernel<kThreads, kOnboardStages, kIlp, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mOnboard.sharedBytes)));
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(smallPageCopyKernel<kThreads, kOnboardStages, kIlp>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mOnboard.sharedBytes)));
}

void BatchedPageCopier::launch(PoolCopyArgs const& args, CopyDirection direction, CUstream stream)
{
    if (args.numPairs == 0 || args.bytesPerPage == 0)
    {
        return;
    }
    if (config(direction).useKernel)
    {
        // The fast kernels move 16 bytes per element, so every page address they touch must be
        // 16-aligned. Anything else goes to the generic kernel -- not to the copy engine, because
        // pageIndexLocation() has already told the caller to put the index array in device memory.
        bool const aligned = args.bytesPerPage % 16 == 0 && args.dstStride % 16 == 0 && args.srcStride % 16 == 0
            && args.dstBase % 16 == 0 && args.srcBase % 16 == 0;
        if (aligned)
        {
            launchKernel(args, direction, stream);
        }
        else
        {
            launchGenericKernel(args, direction, stream);
        }
    }
    else
    {
        launchCopyEngine(args, stream);
    }
}

//! Fallback for pool layouts the fast kernels cannot accept. One block per page, byte at a time.
//! Grid is capped at the configured CTA count so the fallback does not claim more of the GPU than
//! the tuned path would.
void BatchedPageCopier::launchGenericKernel(PoolCopyArgs const& args, CopyDirection direction, CUstream stream) const
{
    KernelConfig const& config = this->config(direction);
    TLLM_CHECK(config.useKernel);

    CopyArgs common{};
    common.dstBase = reinterpret_cast<char*>(args.dstBase);
    common.srcBase = reinterpret_cast<char const*>(args.srcBase);
    common.dstStride = args.dstStride;
    common.srcStride = args.srcStride;
    common.pairs = args.pairs;
    common.numPairs = args.numPairs;
    common.elemsPerPage = 0; // unused by the generic kernel; bytesPerPage is passed explicitly

    int const grid = std::min<int>(config.ctas, static_cast<int>(std::min<uint32_t>(args.numPairs, INT32_MAX)));
    genericPageCopyKernel<kThreads>
        <<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(common, args.bytesPerPage);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

//! Copy-engine path: expand the index pairs into per-page descriptors and submit them as one
//! batch. Descriptor construction is O(numPairs) on the CPU and on the critical path, which is
//! why the kernel path exists at all -- but on PCIe the copy engine still wins overall.
void BatchedPageCopier::launchCopyEngine(PoolCopyArgs const& args, CUstream stream)
{
    mCopyEngineDsts.resize(args.numPairs);
    mCopyEngineSrcs.resize(args.numPairs);
    mCopyEngineSizes.assign(args.numPairs, args.bytesPerPage);
    for (uint32_t slot = 0; slot < args.numPairs; ++slot)
    {
        PageIndexPair const pair = args.pairs[slot]; // host memory; see pageIndexLocation()
        TLLM_CHECK_DEBUG(pair.dst >= 0 && pair.src >= 0);
        mCopyEngineDsts[slot] = args.dstBase + static_cast<uint64_t>(pair.dst) * args.dstStride;
        mCopyEngineSrcs[slot] = args.srcBase + static_cast<uint64_t>(pair.src) * args.srcStride;
    }
#if CUDA_VERSION >= 12080
    CUmemcpyAttributes attributes{};
    attributes.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    attributes.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
    size_t firstCopy = 0;
#if CUDA_VERSION < 13000
    size_t failIdx = std::numeric_limits<size_t>::max();
    TLLM_CU_CHECK(cuMemcpyBatchAsync(mCopyEngineDsts.data(), mCopyEngineSrcs.data(), mCopyEngineSizes.data(),
        args.numPairs, &attributes, &firstCopy, 1, &failIdx, stream));
#else
    TLLM_CU_CHECK(cuMemcpyBatchAsync(mCopyEngineDsts.data(), mCopyEngineSrcs.data(), mCopyEngineSizes.data(),
        args.numPairs, &attributes, &firstCopy, 1, stream));
#endif
#else
    // cuMemcpyBatchAsync needs CUDA 12.8; fall back to one enqueue per page.
    for (uint32_t slot = 0; slot < args.numPairs; ++slot)
    {
        TLLM_CU_CHECK(cuMemcpyAsync(mCopyEngineDsts[slot], mCopyEngineSrcs[slot], mCopyEngineSizes[slot], stream));
    }
#endif
}

void BatchedPageCopier::launchKernel(PoolCopyArgs const& args, CopyDirection direction, CUstream stream) const
{
    KernelConfig const& config = this->config(direction);
    TLLM_CHECK(config.useKernel);
    TLLM_CHECK(args.bytesPerPage % 16 == 0);
    TLLM_CHECK(args.dstStride % 16 == 0 && args.srcStride % 16 == 0);
    TLLM_CHECK(args.dstBase % 16 == 0 && args.srcBase % 16 == 0);
    if (args.numPairs == 0 || args.bytesPerPage == 0)
    {
        return;
    }

    constexpr uint32_t kElemsPerTile = static_cast<uint32_t>(kThreads) * kIlp;

    CopyArgs common{};
    common.dstBase = reinterpret_cast<char*>(args.dstBase);
    common.srcBase = reinterpret_cast<char const*>(args.srcBase);
    common.dstStride = args.dstStride;
    common.srcStride = args.srcStride;
    common.pairs = args.pairs;
    common.numPairs = args.numPairs;
    common.elemsPerPage = args.bytesPerPage / 16u;

    bool const offload = direction == CopyDirection::kD2H;
    auto const stream_ = static_cast<cudaStream_t>(stream);

    // The kernel indexes tiles with 32 bits. One launch therefore covers up to ~4G tiles, i.e.
    // 16 TiB at 4 KiB per tile -- far beyond any single pool copy, but check rather than wrap.
    auto launchWith = [&](auto kernelOffload, auto kernelOnboard, auto const& kernelArgs, uint64_t totalTiles)
    {
        // Bounded by INT32_MAX, not UINT32_MAX: the grid dimension below is an int, so a larger
        // count would produce a negative grid rather than a caught error.
        TLLM_CHECK(totalTiles <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
        auto const tiles = static_cast<uint32_t>(totalTiles);
        int const grid = std::min<int>(config.ctas, static_cast<int>(tiles));
        if (offload)
        {
            kernelOffload<<<grid, kThreads, config.sharedBytes, stream_>>>(kernelArgs, tiles);
        }
        else
        {
            kernelOnboard<<<grid, kThreads, config.sharedBytes, stream_>>>(kernelArgs, tiles);
        }
    };

    if (common.elemsPerPage >= kElemsPerTile)
    {
        LargePageArgs largeArgs{common, (common.elemsPerPage + kElemsPerTile - 1) / kElemsPerTile};
        uint64_t const tiles = static_cast<uint64_t>(args.numPairs) * largeArgs.tilesPerPage;
        // Exact: pages are a whole number of tiles, so no ragged tail, and the tile count is an
        // exact multiple of the page count so no tile runs past the pair list.
        static bool const noExact = std::getenv("TLLM_HDC_NO_EXACT") != nullptr; // test hook
        if (common.elemsPerPage % kElemsPerTile == 0 && !noExact)
        {
            launchWith(largePageCopyKernel<kThreads, kOffloadStages, kIlp, true>,
                largePageCopyKernel<kThreads, kOnboardStages, kIlp, true>, largeArgs, tiles);
        }
        else
        {
            launchWith(largePageCopyKernel<kThreads, kOffloadStages, kIlp, false>,
                largePageCopyKernel<kThreads, kOnboardStages, kIlp, false>, largeArgs, tiles);
        }
    }
    else
    {
        SmallPageArgs smallArgs{common, kElemsPerTile / common.elemsPerPage};
        launchWith(smallPageCopyKernel<kThreads, kOffloadStages, kIlp>,
            smallPageCopyKernel<kThreads, kOnboardStages, kIlp>, smallArgs,
            (static_cast<uint64_t>(args.numPairs) + smallArgs.pagesPerTile - 1) / smallArgs.pagesPerTile);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

namespace detail
{
#if CUDA_VERSION < 12080
namespace
{

constexpr size_t kPageIndexKernelParamBytes = 2U << 10U;
constexpr size_t kPageIndicesPerKernel = kPageIndexKernelParamBytes / sizeof(PageIndexPair);
constexpr uint32_t kPageIndexCopyThreads = 256;

using PageIndexKernelParams = std::array<PageIndexPair, kPageIndicesPerKernel>;
static_assert(sizeof(PageIndexKernelParams) == kPageIndexKernelParamBytes);
static_assert(kPageIndicesPerKernel <= kPageIndexCopyThreads);

#if CUDA_VERSION >= 11070
#define TLLM_KVCM2_GRID_CONSTANT __grid_constant__
#else
#define TLLM_KVCM2_GRID_CONSTANT
#endif

__global__ void copyPageIndicesKernel(
    PageIndexPair* dst, PageIndexKernelParams const TLLM_KVCM2_GRID_CONSTANT src, size_t count)
{
    size_t const index = threadIdx.x;
    if (index < count)
    {
        dst[index] = src[index];
    }
}

#undef TLLM_KVCM2_GRID_CONSTANT

} // namespace

void copyPageIndicesToDeviceWithKernel(
    CUdeviceptr dst, PageIndexPair const* src, size_t numPageIndices, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(dst != 0 && src != nullptr, "Page-index copy requires valid source and destination");

    PageIndexKernelParams params{};
    size_t offset = 0;
    while (offset < numPageIndices)
    {
        size_t const count = std::min(numPageIndices - offset, kPageIndicesPerKernel);
        std::copy_n(src + offset, count, params.begin());
        copyPageIndicesKernel<<<1, kPageIndexCopyThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<PageIndexPair*>(dst) + offset, params, count);
        TLLM_CUDA_CHECK(cudaGetLastError());
        offset += count;
    }
}
#endif

void copyPageIndicesToDevice(CUdeviceptr dst, PageIndexPair const* src, size_t numPageIndices, CUstream stream)
{
    if (numPageIndices == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(
        numPageIndices <= std::numeric_limits<size_t>::max() / sizeof(PageIndexPair), "Page-index array is too large");
    TLLM_CHECK_WITH_INFO(dst != 0 && src != nullptr, "Page-index copy requires valid source and destination");

#if CUDA_VERSION >= 12080
    size_t numBytes = numPageIndices * sizeof(PageIndexPair);
    CUdeviceptr srcAddress = reinterpret_cast<CUdeviceptr>(src);
    CUmemcpyAttributes attributes{};
    attributes.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL;
    attributes.srcLocHint.type = CU_MEM_LOCATION_TYPE_HOST;
    attributes.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
    attributes.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
    size_t firstCopy = 0;
#if CUDA_VERSION < 13000
    size_t failIdx = std::numeric_limits<size_t>::max();
    TLLM_CU_CHECK(cuMemcpyBatchAsync(&dst, &srcAddress, &numBytes, 1, &attributes, &firstCopy, 1, &failIdx, stream));
#else
    TLLM_CU_CHECK(cuMemcpyBatchAsync(&dst, &srcAddress, &numBytes, 1, &attributes, &firstCopy, 1, stream));
#endif
#else
    copyPageIndicesToDeviceWithKernel(dst, src, numPageIndices, stream);
#endif
}

} // namespace detail

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
