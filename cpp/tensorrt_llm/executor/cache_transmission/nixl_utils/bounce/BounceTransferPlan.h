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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceMessage.h"
#include "tensorrt_llm/executor/transferAgent.h"

#include <cstdint>
#include <vector>

namespace tensorrt_llm::executor::kv_cache::bounce
{

/// One chunk = the set of (src,dst) descriptors that get packed into a single bounce region
/// and moved with one RDMA write. Offsets are byte offsets within that region.
struct BounceChunk
{
    std::vector<std::uint64_t> srcPtrs;       // sender-local source addresses
    std::vector<std::uint64_t> dstPtrs;       // receiver-local final destination addresses
    std::vector<std::uint32_t> sizes;         // per-desc byte counts
    std::vector<std::uint64_t> bounceOffsets; // per-desc offset within the region
    // Coalesced scatter view — what goes on the wire as the DATA scatter plan. Adjacent descs merge
    // into one BounceScatterRun when (bounceOffset, dstPtr) advance contiguously (run grows its
    // pieceSize; the fully-dense dst, e.g. ctx tp1 -> gen tp4) OR by a uniform stride (run grows its
    // count; the strided dst, e.g. ctx tp-slice -> gen DP full-head pool). Either way thousands of
    // per-desc entries collapse to a handful of runs, shrinking the DATA control message (which sits
    // on the ACK critical path) by orders of magnitude. The per-desc arrays above stay as-is for the
    // gather (src layout is independent of dst).
    std::vector<BounceScatterRun> scatterRuns;
    std::uint64_t totalBytes{0};              // sum of desc sizes (payload only, excludes padding)
    std::uint64_t packedBytes{0};             // region extent to RDMA-write: last bounceOffset + its size
    std::uint32_t dstDeviceId{0};             // receiver device id for this chunk (uniform per chunk)
};

/// Pure bin-packing of a TransferRequest's (src,dst) descriptor pairs into chunks that each
/// fit in one bounce region (<= maxChunkBytes). No CUDA / NIXL / threads — trivially unit-testable.
///
/// Packing rules (a chunk is flushed when any holds):
///   - adding the next desc would exceed `maxChunkBytes`,
///   - the chunk already holds `maxDescsPerChunk` descs,
///   - the next desc targets a different device id than the chunk.
class BounceTransferPlan
{
public:
    /// @param maxChunkBytes        per-chunk byte cap (one region holds at most this)
    /// @param maxDescsPerChunk  upper bound on descs per chunk (bounds scatter-plan size)
    /// @param mergeScatterRuns  coalesce the scatter view into runs (default). false = one count==1
    ///                          run per desc — DEBUG ONLY (large-message control-plane A/B).
    /// Throws (TLLM_CHECK) on src/dst count or length mismatch, or a single desc > maxChunkBytes.
    [[nodiscard]] static BounceTransferPlan build(TransferDescs const& srcDescs, TransferDescs const& dstDescs,
        std::size_t maxChunkBytes, std::size_t maxDescsPerChunk, bool mergeScatterRuns = true);

    [[nodiscard]] std::vector<BounceChunk> const& chunks() const noexcept
    {
        return mChunks;
    }

    [[nodiscard]] std::size_t numChunks() const noexcept
    {
        return mChunks.size();
    }

    [[nodiscard]] std::uint64_t totalBytes() const noexcept
    {
        return mTotalBytes;
    }

    [[nodiscard]] std::size_t totalDescs() const noexcept
    {
        return mTotalDescs;
    }

private:
    std::vector<BounceChunk> mChunks;
    std::uint64_t mTotalBytes{0};
    std::size_t mTotalDescs{0};
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
