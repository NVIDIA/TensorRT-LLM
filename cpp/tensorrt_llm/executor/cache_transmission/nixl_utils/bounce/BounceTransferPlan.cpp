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
// Cap on the per-desc array capacity reserved up front for a new chunk (16384 entries ~ 448 KiB across
// the four arrays); a chunk that really holds more descs grows past it by doubling. The reservation is
// a guess that can overshoot, and the plan keeps its chunks until the request's last ACK, so flush()
// trims a chunk whose arrays ended up less than half full.
constexpr std::size_t kMaxReservedDescsPerChunk = 16384;

template <typename T>
void trimExcessCapacity(std::vector<T>& v)
{
    if (v.capacity() > 2 * v.size())
    {
        v.shrink_to_fit();
    }
}

constexpr std::uint64_t alignUp(std::uint64_t value, std::uint64_t align) noexcept
{
    return (value + align - 1ULL) / align * align;
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
    chunk.maxDescBytes = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        std::uint64_t const dst = chunk.dstPtrs[i];
        std::uint64_t const bounce = chunk.bounceOffsets[i];
        std::uint32_t const size = chunk.sizes[i];
        chunk.maxDescBytes = std::max(chunk.maxDescBytes, size);
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

// The plan of one contiguous descriptor range: its chunks in order, plus the totals.
struct RangePlan
{
    std::vector<BounceChunk> chunks;
    std::uint64_t totalBytes{0};
    std::size_t totalDescs{0};
};

// Plan descriptors [begin, end) into chunks: the sequential core of build(). The range always starts
// a fresh chunk (cursor 0), so disjoint ranges planned independently and concatenated in order form a
// valid plan; only a merge across a range boundary is forgone. Throws (TLLM_CHECK) at the range's first
// offending descriptor, reporting its index in the whole request.
RangePlan planRange(std::vector<MemoryDesc> const& srcVec, std::vector<MemoryDesc> const& dstVec, std::size_t begin,
    std::size_t end, std::size_t maxChunkSizeBytes, std::size_t maxDescsPerChunk, std::uint32_t sourceDeviceId,
    std::uint32_t destinationDeviceId)
{
    RangePlan out;
    BounceChunk current;
    current.dstDeviceId = destinationDeviceId;
    std::uint64_t cursor = 0; // running write offset within the current chunk region (aligned)
    // End addresses of the current chunk's last plan desc (meaningful while the chunk is non-empty):
    // the in-place merge test compares against these instead of re-deriving them from the arrays.
    std::uint64_t lastSrcEnd = 0;
    std::uint64_t lastDstEnd = 0;

    auto flush = [&]()
    {
        if (!current.srcPtrs.empty())
        {
            buildScatterRuns(current);
            trimExcessCapacity(current.srcPtrs);
            trimExcessCapacity(current.dstPtrs);
            trimExcessCapacity(current.sizes);
            trimExcessCapacity(current.bounceOffsets);
            out.chunks.emplace_back(std::move(current));
            current = BounceChunk{};
            cursor = 0;
        }
    };

    for (std::size_t i = begin; i < end; ++i)
    {
        auto const& src = srcVec[i];
        auto const& dst = dstVec[i];
        std::size_t const len = src.getLen();
        std::uint64_t const srcAddr = src.getAddr();
        std::uint64_t const dstAddr = dst.getAddr();
        // One combined test on the hot path; only a failing desc re-runs the individual checks, in their
        // usual order, for the specific error. len < 4 GiB follows from len <= maxChunkSizeBytes (build()
        // bounds that cap by 4 GiB - 1) but stays checked explicitly below.
        bool const shapeOk = len == dst.getLen() && src.getDeviceId() == sourceDeviceId
            && dst.getDeviceId() == destinationDeviceId && len <= maxChunkSizeBytes;
        if (TLLM_UNLIKELY(!shapeOk))
        {
            TLLM_CHECK_WITH_INFO(len == dst.getLen(), "BounceTransferPlan: src/dst len mismatch at idx %zu", i);
            TLLM_CHECK_WITH_INFO(src.getDeviceId() == sourceDeviceId,
                "BounceTransferPlan: mixed source device ids are unsupported (idx %zu has %u, expected %u)", i,
                src.getDeviceId(), sourceDeviceId);
            TLLM_CHECK_WITH_INFO(dst.getDeviceId() == destinationDeviceId,
                "BounceTransferPlan: mixed destination device ids are unsupported (idx %zu has %u, expected %u)", i,
                dst.getDeviceId(), destinationDeviceId);
            TLLM_CHECK_WITH_INFO(len <= maxChunkSizeBytes,
                "BounceTransferPlan: single desc (%zu B) exceeds maxChunkSizeBytes (%zu B)", len, maxChunkSizeBytes);
        }
        TLLM_CHECK_WITH_INFO(len < (1ULL << 32U), "BounceTransferPlan: single desc (%zu B) exceeds 4 GiB", len);

        // A zero-length desc carries no data; skip it so it never forces an empty chunk.
        if (len == 0)
        {
            out.totalDescs += 1;
            continue;
        }

        bool const overflow = (cursor + len > maxChunkSizeBytes);
        bool const tooManyDescs = (current.srcPtrs.size() >= maxDescsPerChunk);

        // Extend the previous desc in place when src, dst AND the bounce cursor all advance
        // contiguously (the aligned cursor left no gap): one desc instead of two shrinks the gather
        // plan, the scatter runs and the wire messages. Only within the current chunk (`!overflow`)
        // and staying within the u32 per-desc size field.
        // (packedBytes is the last plan desc's bounceOffset + size, so `cursor == packedBytes` means the
        // aligned cursor left no gap.)
        bool const srcDstContig = !current.srcPtrs.empty() && !overflow && srcAddr == lastSrcEnd
            && dstAddr == lastDstEnd && cursor == current.packedBytes
            && static_cast<std::uint64_t>(current.sizes.back()) + len <= std::numeric_limits<std::uint32_t>::max();
        if (srcDstContig)
        {
            current.sizes.back() += static_cast<std::uint32_t>(len);
            current.totalBytes += len;
            current.packedBytes += len;
            cursor = alignUp(current.packedBytes, kAlignment);
            lastSrcEnd += len;
            lastDstEnd += len;
            out.totalBytes += len;
            out.totalDescs += 1;
            continue;
        }

        if (overflow || tooManyDescs)
        {
            flush();
            current.dstDeviceId = destinationDeviceId;
        }

        if (current.srcPtrs.empty())
        {
            // A new chunk: size its per-desc arrays once instead of growing them by doubling (the plan
            // build is a serial prefix before WANT). Assume the descs ahead look like this one — KV
            // transfers are uniform — bounded by the per-chunk desc cap, the descs left in the range and
            // kMaxReservedDescsPerChunk (the guess overshoots when this desc is smaller than the ones
            // after it, or when small descs merge in place). A wrong guess only costs a regrow or some
            // slack capacity, never correctness.
            std::size_t const fitByBytes = maxChunkSizeBytes / alignUp(len, kAlignment) + 1;
            std::size_t const expected = std::min({maxDescsPerChunk, end - i, fitByBytes, kMaxReservedDescsPerChunk});
            current.srcPtrs.reserve(expected);
            current.dstPtrs.reserve(expected);
            current.sizes.reserve(expected);
            current.bounceOffsets.reserve(expected);
        }
        current.srcPtrs.push_back(srcAddr);
        current.dstPtrs.push_back(dstAddr);
        current.sizes.push_back(static_cast<std::uint32_t>(len));
        current.bounceOffsets.push_back(cursor);
        current.totalBytes += len;
        current.packedBytes = cursor + len; // extent to transfer (this desc is the furthest so far)
        cursor = alignUp(cursor + len, kAlignment);
        lastSrcEnd = srcAddr + len;
        lastDstEnd = dstAddr + len;

        out.totalBytes += len;
        out.totalDescs += 1;
    }
    flush();
    return out;
}
} // namespace

BounceTransferPlan BounceTransferPlan::build(TransferDescs const& srcDescs, TransferDescs const& dstDescs,
    std::size_t maxChunkSizeBytes, std::size_t maxDescsPerChunk, std::size_t buildSegments, HostWorkerPool* pool)
{
    BounceTransferPlan plan;

    auto const& srcVec = srcDescs.getDescs();
    auto const& dstVec = dstDescs.getDescs();
    TLLM_CHECK_WITH_INFO(srcVec.size() == dstVec.size(), "BounceTransferPlan: src/dst desc count mismatch (%zu vs %zu)",
        srcVec.size(), dstVec.size());
    TLLM_CHECK_WITH_INFO(maxChunkSizeBytes > 0 && maxDescsPerChunk > 0,
        "BounceTransferPlan: maxChunkSizeBytes/maxDescsPerChunk must be > 0");
    // A chunk's packed size flows through 32-bit fields on the wire (Grant.len, WANT chunk sizes,
    // scatter entry size, Posted.writeBytes), so a chunk must fit in 32 bits. Arena offsets are
    // 64-bit (arena may exceed 4 GiB) but a single staging chunk above 4 GiB is nonsensical.
    TLLM_CHECK_WITH_INFO(maxChunkSizeBytes <= std::numeric_limits<std::uint32_t>::max(),
        "BounceTransferPlan: maxChunkSizeBytes (%zu) must be <= 4 GiB (chunk size is 32-bit on the wire)",
        maxChunkSizeBytes);

    std::size_t const n = srcVec.size();
    if (n == 0)
    {
        return plan; // 0 descs -> 0 chunks
    }

    auto const sourceDeviceId = srcVec.front().getDeviceId();
    auto const destinationDeviceId = dstVec.front().getDeviceId();
    auto planSegment = [&](std::size_t begin, std::size_t end)
    {
        return planRange(
            srcVec, dstVec, begin, end, maxChunkSizeBytes, maxDescsPerChunk, sourceDeviceId, destinationDeviceId);
    };

    std::size_t const segments = std::min(buildSegments != 0 ? buildSegments : bulkSegmentCount(n), n);
    if (segments == 1)
    {
        RangePlan whole = planSegment(0, n);
        plan.mChunks = std::move(whole.chunks);
        plan.mTotalBytes = whole.totalBytes;
        plan.mTotalDescs = whole.totalDescs;
        return plan;
    }

    // Large request: plan `segments` equal descriptor ranges (the plan build is a serial prefix of
    // submit(), before any byte moves) and concatenate them in order. Errors keep the sequential
    // semantics: parallelFor rethrows the LOWEST failing segment's exception, and the in-order loop stops
    // at it, so either way the reported failure is the one at the lowest descriptor index.
    auto const bound = [&](std::size_t s) { return n * s / segments; };
    std::vector<RangePlan> parts(segments);
    auto const planPart = [&](std::size_t s) { parts[s] = planSegment(bound(s), bound(s + 1)); };
    if (pool != nullptr)
    {
        pool->parallelFor(segments, planPart);
    }
    else
    {
        for (std::size_t s = 0; s < segments; ++s)
        {
            planPart(s);
        }
    }
    std::size_t numChunks = 0;
    for (auto const& part : parts)
    {
        numChunks += part.chunks.size();
    }
    plan.mChunks.reserve(numChunks);
    for (auto& part : parts)
    {
        std::move(part.chunks.begin(), part.chunks.end(), std::back_inserter(plan.mChunks));
        plan.mTotalBytes += part.totalBytes;
        plan.mTotalDescs += part.totalDescs;
    }
    return plan;
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
