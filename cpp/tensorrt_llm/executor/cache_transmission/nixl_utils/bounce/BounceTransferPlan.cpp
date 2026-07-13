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

#include <limits>

namespace tensorrt_llm::executor::kv_cache::bounce
{

namespace
{
// 32-byte alignment is enough for memory-coalesced vectorized copies and imposes no stricter
// requirement than the underlying registered memory already has.
constexpr std::uint64_t kAlignment = 32ULL;

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
void buildScatterRuns(BounceChunk& chunk, bool merge)
{
    auto const n = chunk.dstPtrs.size();
    chunk.scatterRuns.clear();
    for (std::size_t i = 0; i < n; ++i)
    {
        std::uint64_t const dst = chunk.dstPtrs[i];
        std::uint64_t const bounce = chunk.bounceOffsets[i];
        std::uint32_t const size = chunk.sizes[i];
        if (merge && !chunk.scatterRuns.empty())
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
} // namespace

BounceTransferPlan BounceTransferPlan::build(TransferDescs const& srcDescs, TransferDescs const& dstDescs,
    std::size_t maxChunkBytes, std::size_t maxDescsPerChunk, bool mergeScatterRuns)
{
    BounceTransferPlan plan;

    auto const& srcVec = srcDescs.getDescs();
    auto const& dstVec = dstDescs.getDescs();
    TLLM_CHECK_WITH_INFO(srcVec.size() == dstVec.size(), "BounceTransferPlan: src/dst desc count mismatch (%zu vs %zu)",
        srcVec.size(), dstVec.size());
    TLLM_CHECK_WITH_INFO(
        maxChunkBytes > 0 && maxDescsPerChunk > 0, "BounceTransferPlan: maxChunkBytes/maxDescsPerChunk must be > 0");
    // A chunk's packed size flows through 32-bit fields on the wire (Grant.len, WANT chunk sizes,
    // scatter entry size, Posted.writeBytes), so a chunk must fit in 32 bits. Arena offsets are
    // 64-bit (arena may exceed 4 GiB) but a single staging chunk above 4 GiB is nonsensical.
    TLLM_CHECK_WITH_INFO(maxChunkBytes <= std::numeric_limits<std::uint32_t>::max(),
        "BounceTransferPlan: maxChunkBytes (%zu) must be <= 4 GiB (chunk size is 32-bit on the wire)", maxChunkBytes);

    if (srcVec.empty())
    {
        return plan; // 0 descs -> 0 chunks
    }

    BounceChunk current;
    current.dstDeviceId = dstVec.front().getDeviceId();
    std::uint64_t cursor = 0; // running write offset within the current chunk region (aligned)

    auto flush = [&]()
    {
        if (!current.srcPtrs.empty())
        {
            buildScatterRuns(current, mergeScatterRuns);
            plan.mChunks.emplace_back(std::move(current));
            current = BounceChunk{};
            cursor = 0;
        }
    };

    for (std::size_t i = 0; i < srcVec.size(); ++i)
    {
        auto const& src = srcVec[i];
        auto const& dst = dstVec[i];
        std::size_t const len = src.getLen();
        TLLM_CHECK_WITH_INFO(len == dst.getLen(), "BounceTransferPlan: src/dst len mismatch at idx %zu", i);
        TLLM_CHECK_WITH_INFO(len <= maxChunkBytes,
            "BounceTransferPlan: single desc (%zu B) exceeds maxChunkBytes (%zu B)", len, maxChunkBytes);
        TLLM_CHECK_WITH_INFO(len < (1ULL << 32U), "BounceTransferPlan: single desc (%zu B) exceeds 4 GiB", len);

        // A zero-length desc carries no data; skip it so it never forces an empty chunk.
        if (len == 0)
        {
            plan.mTotalDescs += 1;
            continue;
        }

        bool const overflow = (cursor + len > maxChunkBytes);
        bool const tooManyDescs = (current.srcPtrs.size() >= maxDescsPerChunk);
        bool const deviceMismatch = !current.srcPtrs.empty() && dst.getDeviceId() != current.dstDeviceId;

        // Extend the previous desc in place when src, dst AND the bounce cursor all advance
        // contiguously (the aligned cursor left no gap): one desc instead of two shrinks the gather
        // plan, the scatter runs and the wire messages. Only within the current chunk (`!overflow`)
        // and staying within the u32 per-desc size field.
        bool const srcDstContig = !current.srcPtrs.empty() && !overflow && !deviceMismatch
            && src.getAddr() == current.srcPtrs.back() + current.sizes.back()
            && dst.getAddr() == current.dstPtrs.back() + current.sizes.back()
            && cursor == current.bounceOffsets.back() + current.sizes.back()
            && static_cast<std::uint64_t>(current.sizes.back()) + len <= std::numeric_limits<std::uint32_t>::max();
        if (srcDstContig)
        {
            current.sizes.back() += static_cast<std::uint32_t>(len);
            current.totalBytes += len;
            current.packedBytes = current.bounceOffsets.back() + current.sizes.back();
            cursor = alignUp(current.packedBytes, kAlignment);
            plan.mTotalBytes += len;
            plan.mTotalDescs += 1;
            continue;
        }

        if (overflow || tooManyDescs || deviceMismatch)
        {
            flush();
            current.dstDeviceId = dst.getDeviceId();
        }

        current.srcPtrs.push_back(static_cast<std::uint64_t>(src.getAddr()));
        current.dstPtrs.push_back(static_cast<std::uint64_t>(dst.getAddr()));
        current.sizes.push_back(static_cast<std::uint32_t>(len));
        current.bounceOffsets.push_back(cursor);
        current.totalBytes += len;
        current.packedBytes = cursor + len; // extent to transfer (this desc is the furthest so far)
        cursor = alignUp(cursor + len, kAlignment);

        plan.mTotalBytes += len;
        plan.mTotalDescs += 1;
    }
    flush();

    return plan;
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
