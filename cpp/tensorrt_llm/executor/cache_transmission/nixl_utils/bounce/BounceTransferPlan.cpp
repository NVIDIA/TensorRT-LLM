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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransferPlan.h"

#include "tensorrt_llm/common/assert.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

namespace tensorrt_llm::executor::kv_cache::bounce
{

namespace
{
// 32-byte alignment is enough for memory-coalesced vectorized copies and imposes no stricter
// requirement than the underlying registered memory already has.
constexpr std::uint64_t kAlignment = 32ULL;
constexpr std::uint64_t kMaxDescBytes = std::numeric_limits<std::uint32_t>::max();
// Caps the per-desc arrays reserved up front for a new chunk (16384 descs ~ 448 KiB over the four arrays).
// The plan lives until the request's last ACK, so finish() trims arrays the estimate left under half full.
constexpr std::size_t kMaxReservedDescsPerChunk = 16384;

constexpr std::uint64_t alignUp(std::uint64_t value, std::uint64_t align) noexcept
{
    return (value + align - 1ULL) / align * align;
}

template <typename T>
void shrinkIfUnderHalfFull(std::vector<T>& v)
{
    if (v.capacity() > 2 * v.size())
    {
        v.shrink_to_fit();
    }
}

template <typename Fn>
void forEachDescArray(BounceChunk& chunk, Fn&& fn)
{
    fn(chunk.srcPtrs);
    fn(chunk.dstPtrs);
    fn(chunk.sizes);
    fn(chunk.bounceOffsets);
}

// Build the chunk's coalesced scatter view (see BounceScatterRun). Greedy single pass; per desc,
// try to extend the last run in one of three ways before opening a new one:
//   (a) contiguous growth (count==1): bounce AND dst both continue exactly where the run ends ->
//       grow pieceSize in place. Captures a fully-dense dst (whole chunk -> ONE run).
//   (b) stride latch (count==1, same size): the second desc fixes (dstStride, bounceStride) and the
//       run becomes count=2. Only forward, u32-representable bounce steps latch.
//   (c) stride extension (count>=2): the desc lands exactly one stride past the run's last piece.
//       Captures a uniformly-strided dst (e.g. a head slice into a wider pool -> ONE run).
// Irregular layouts simply break runs (worst case: one count==1 run per desc == the old per-desc
// plan). Correctness never depends on merging.
void buildScatterRuns(BounceChunk& chunk)
{
    auto const n = chunk.dstPtrs.size();
    chunk.scatterRuns.clear();
    for (std::size_t i = 0; i < n; ++i)
    {
        std::uint64_t const dst = chunk.dstPtrs[i];
        std::uint64_t const bounce = chunk.bounceOffsets[i];
        std::uint32_t const size = chunk.sizes[i];
        if (!chunk.scatterRuns.empty())
        {
            auto& r = chunk.scatterRuns.back();
            if (r.count == 1)
            {
                // (a) contiguous growth. Piece size stays within u32 (packedBytes <= 4 GiB - 1 is
                // enforced in build(), but guard the sum explicitly anyway).
                if (dst == r.dstAddr + r.pieceSize && bounce == r.bounceOffset + r.pieceSize
                    && static_cast<std::uint64_t>(r.pieceSize) + size <= std::numeric_limits<std::uint32_t>::max())
                {
                    r.pieceSize += size;
                    continue;
                }
                // (b) stride latch.
                if (size == r.pieceSize && dst > r.dstAddr && bounce > r.bounceOffset
                    && bounce - r.bounceOffset <= std::numeric_limits<std::uint32_t>::max())
                {
                    r.dstStride = dst - r.dstAddr;
                    r.bounceStride = static_cast<std::uint32_t>(bounce - r.bounceOffset);
                    r.count = 2;
                    continue;
                }
            }
            else if (size == r.pieceSize && dst == r.dstAddr + static_cast<std::uint64_t>(r.count) * r.dstStride
                && bounce == r.bounceOffset + static_cast<std::uint64_t>(r.count) * r.bounceStride)
            {
                // (c) stride extension.
                r.count += 1;
                continue;
            }
        }
        chunk.scatterRuns.push_back(BounceScatterRun{bounce, dst, 0, 0, size, 1});
    }
}

struct PlanLimits
{
    std::size_t maxChunkSizeBytes;
    std::size_t maxDescsPerChunk;
    std::uint32_t sourceDeviceId;
    std::uint32_t destinationDeviceId;
};

bool fitsLimits(MemoryDesc const& src, MemoryDesc const& dst, PlanLimits const& limits)
{
    return src.getLen() == dst.getLen() && src.getDeviceId() == limits.sourceDeviceId
        && dst.getDeviceId() == limits.destinationDeviceId && src.getLen() <= limits.maxChunkSizeBytes;
}

void throwLimitViolation(MemoryDesc const& src, MemoryDesc const& dst, std::size_t index, PlanLimits const& limits)
{
    TLLM_CHECK_WITH_INFO(src.getLen() == dst.getLen(), "BounceTransferPlan: src/dst len mismatch at idx %zu", index);
    TLLM_CHECK_WITH_INFO(src.getDeviceId() == limits.sourceDeviceId,
        "BounceTransferPlan: mixed source device ids are unsupported (idx %zu has %u, expected %u)", index,
        src.getDeviceId(), limits.sourceDeviceId);
    TLLM_CHECK_WITH_INFO(dst.getDeviceId() == limits.destinationDeviceId,
        "BounceTransferPlan: mixed destination device ids are unsupported (idx %zu has %u, expected %u)", index,
        dst.getDeviceId(), limits.destinationDeviceId);
    TLLM_CHECK_WITH_INFO(src.getLen() <= limits.maxChunkSizeBytes,
        "BounceTransferPlan: single desc (%zu B) exceeds maxChunkSizeBytes (%zu B)", src.getLen(),
        limits.maxChunkSizeBytes);
}

void appendDesc(
    BounceChunk& chunk, std::uint64_t srcAddr, std::uint64_t dstAddr, std::size_t len, std::uint64_t bounceOffset)
{
    chunk.srcPtrs.push_back(srcAddr);
    chunk.dstPtrs.push_back(dstAddr);
    chunk.sizes.push_back(static_cast<std::uint32_t>(len));
    chunk.bounceOffsets.push_back(bounceOffset);
    chunk.totalBytes += len;
    chunk.packedBytes = bounceOffset + len;
}

// One desc instead of two shrinks the gather plan, the scatter runs and the wire messages.
void extendLastDesc(BounceChunk& chunk, std::size_t len)
{
    chunk.sizes.back() += static_cast<std::uint32_t>(len);
    chunk.totalBytes += len;
    chunk.packedBytes += len;
}

// Descs a new chunk will likely hold, assuming the descs ahead look like its first one (KV transfers are
// uniform). Sizes the chunk's arrays once instead of growing them by doubling; a wrong guess costs a
// regrow or slack, never correctness.
std::size_t estimateDescsInChunk(std::size_t firstDescLen, std::size_t descsLeft, PlanLimits const& limits)
{
    std::size_t const fitByBytes = limits.maxChunkSizeBytes / alignUp(firstDescLen, kAlignment) + 1;
    return std::min({limits.maxDescsPerChunk, descsLeft, fitByBytes, kMaxReservedDescsPerChunk});
}

// Completes `chunk`, hands it over and leaves `chunk` empty for the next one.
BounceChunk finishChunk(BounceChunk& chunk, PlanLimits const& limits)
{
    chunk.dstDeviceId = limits.destinationDeviceId;
    chunk.maxDescBytes = *std::max_element(chunk.sizes.begin(), chunk.sizes.end());
    buildScatterRuns(chunk);
    forEachDescArray(chunk, [](auto& descArray) { shrinkIfUnderHalfFull(descArray); });
    return std::exchange(chunk, BounceChunk{});
}

struct RangePlan
{
    std::vector<BounceChunk> chunks;
    std::uint64_t totalBytes{0};
    std::size_t totalDescs{0};
};

// Plans descs [begin, end) from a fresh chunk, so the plans of consecutive ranges concatenate into a valid
// plan (only a merge across the boundary is lost). Throws at the range's first desc that breaks a limit.
RangePlan planRange(std::vector<MemoryDesc> const& srcVec, std::vector<MemoryDesc> const& dstVec, std::size_t begin,
    std::size_t end, PlanLimits const& limits)
{
    RangePlan out;
    BounceChunk chunk;
    // The packing state stays in locals: held in an object next to the chunk's vectors (a ChunkBuilder
    // class was tried), it made the plan build ~12% slower per desc.
    std::uint64_t cursor = 0; // aligned offset of the next desc in the chunk's bounce region
    std::uint64_t lastSrcEnd = 0;
    std::uint64_t lastDstEnd = 0;
    for (std::size_t i = begin; i < end; ++i)
    {
        MemoryDesc const& src = srcVec[i];
        MemoryDesc const& dst = dstVec[i];
        if (TLLM_UNLIKELY(!fitsLimits(src, dst, limits)))
        {
            throwLimitViolation(src, dst, i, limits);
        }
        std::size_t const len = src.getLen();
        out.totalDescs += 1;
        out.totalBytes += len;
        if (len == 0)
        {
            continue;
        }
        std::uint64_t const srcAddr = src.getAddr();
        std::uint64_t const dstAddr = dst.getAddr();
        bool const fitsInChunk = cursor + len <= limits.maxChunkSizeBytes;
        bool const noPaddingAfterLastDesc = cursor == chunk.packedBytes;
        bool const continuesLastDesc = !chunk.srcPtrs.empty() && fitsInChunk && srcAddr == lastSrcEnd
            && dstAddr == lastDstEnd && noPaddingAfterLastDesc
            && std::uint64_t{chunk.sizes.back()} + len <= kMaxDescBytes;
        if (continuesLastDesc)
        {
            extendLastDesc(chunk, len);
        }
        else
        {
            bool const chunkFull = !fitsInChunk || chunk.srcPtrs.size() >= limits.maxDescsPerChunk;
            if (!chunk.srcPtrs.empty() && chunkFull)
            {
                out.chunks.push_back(finishChunk(chunk, limits));
                cursor = 0;
            }
            if (chunk.srcPtrs.empty())
            {
                std::size_t const expectedDescs = estimateDescsInChunk(len, /*descsLeft=*/end - i, limits);
                forEachDescArray(chunk, [expectedDescs](auto& descArray) { descArray.reserve(expectedDescs); });
            }
            appendDesc(chunk, srcAddr, dstAddr, len, /*bounceOffset=*/cursor);
        }
        cursor = alignUp(chunk.packedBytes, kAlignment);
        lastSrcEnd = srcAddr + len;
        lastDstEnd = dstAddr + len;
    }
    if (!chunk.srcPtrs.empty())
    {
        out.chunks.push_back(finishChunk(chunk, limits));
    }
    return out;
}
} // namespace

BounceTransferPlan BounceTransferPlan::build(TransferDescs const& srcDescs, TransferDescs const& dstDescs,
    std::size_t maxChunkSizeBytes, std::size_t maxDescsPerChunk, std::size_t segments, HostWorkerPool* pool)
{
    BounceTransferPlan plan;

    auto const& srcVec = srcDescs.getDescs();
    auto const& dstVec = dstDescs.getDescs();
    TLLM_CHECK_WITH_INFO(srcVec.size() == dstVec.size(), "BounceTransferPlan: src/dst desc count mismatch (%zu vs %zu)",
        srcVec.size(), dstVec.size());
    TLLM_CHECK_WITH_INFO(maxChunkSizeBytes > 0 && maxDescsPerChunk > 0 && segments > 0,
        "BounceTransferPlan: maxChunkSizeBytes/maxDescsPerChunk/segments must be > 0");
    // A chunk's packed size flows through 32-bit fields on the wire (Grant.len, WANT chunk sizes,
    // scatter entry size, Posted.writeBytes), so a chunk must fit in 32 bits. Arena offsets are
    // 64-bit (arena may exceed 4 GiB) but a single staging chunk above 4 GiB is nonsensical.
    TLLM_CHECK_WITH_INFO(maxChunkSizeBytes <= std::numeric_limits<std::uint32_t>::max(),
        "BounceTransferPlan: maxChunkSizeBytes (%zu) must be <= 4 GiB (chunk size is 32-bit on the wire)",
        maxChunkSizeBytes);

    std::size_t const descCount = srcVec.size();
    if (descCount == 0)
    {
        return plan;
    }

    PlanLimits const limits{
        maxChunkSizeBytes, maxDescsPerChunk, srcVec.front().getDeviceId(), dstVec.front().getDeviceId()};
    std::size_t const segmentCount = std::min(segments, descCount);
    std::vector<RangePlan> parts(segmentCount);
    forEachSegment(pool, descCount, segmentCount,
        [&](std::size_t segment, std::size_t begin, std::size_t end)
        { parts[segment] = planRange(srcVec, dstVec, begin, end, limits); });

    std::size_t chunkCount = 0;
    for (auto const& part : parts)
    {
        chunkCount += part.chunks.size();
    }
    plan.mChunks.reserve(chunkCount);
    for (auto& part : parts)
    {
        std::move(part.chunks.begin(), part.chunks.end(), std::back_inserter(plan.mChunks));
        plan.mTotalBytes += part.totalBytes;
        plan.mTotalDescs += part.totalDescs;
    }
    return plan;
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
