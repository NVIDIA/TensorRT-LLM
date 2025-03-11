/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cassert>

//// FIX
#include "Dtype.h" // #include <trtllm/gen/Dtype.h>

namespace gemm
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Structure to manage memory allocation with configurable reuse
class SmemAllocatorHelper
{
public:
    // The default constructor.
    SmemAllocatorHelper() {}

    // Constructor to initialize chunk sizes, alignments, and reuse flags
    SmemAllocatorHelper(std::vector<std::pair<int32_t, int32_t>> const& sizes, std::vector<bool> const& reuse)
        : mNumBytesAndAlignmentPerSmemChunk(sizes)
        , mFirstChunkReuse(reuse)
    {
    }

    // Function to calculate the size of the array from 0 to jj chunks
    int32_t getOffsetBeforeChunk(int jj) const
    {
        int32_t totalSize = 0;
        for (int32_t ii = 0; ii < jj; ++ii)
        {
            auto const& elem = mNumBytesAndAlignmentPerSmemChunk[ii];
            auto paddedSize = getSizePaddedToAlignment(elem.first, elem.second);
            // If SMEM chunk is reused but the size of the current chunk is
            // larger than currently counted size
            if (mFirstChunkReuse[ii] && paddedSize > totalSize)
            {
                // Set new size to the size of the current chunk.
                // E.g. possible in case of
                // mNumBytesAndAlignmentPerSmemChunk = {{1, 1}, {1, 1}, {1024, 1}}
                // mFirstChunkReuse = {false, false, true}
                // The last chunk is larger than the first plus second, so total size is 1024.
                totalSize = paddedSize;
            }
            else if (!mFirstChunkReuse[ii])
            {
                totalSize += paddedSize;
            }
        }
        return totalSize;
    }

    // Returns the offset of the ith chunk
    int32_t getChunkOffset(int32_t ii) const
    {
        if (mFirstChunkReuse[ii])
        {
            // Reuse the offset of the 0th chunk.
            return getChunkOffset(0);
        }

        // Get offset of ii chunks.
        auto offset = getOffsetBeforeChunk(ii);
        // Ensure alignment for the current chunk
        return getSizePaddedToAlignment(offset, mNumBytesAndAlignmentPerSmemChunk[ii].second);
    }

    // Function to calculate the total size of the SMEM array
    int32_t getTotalSize() const
    {
        return getOffsetBeforeChunk(static_cast<int32_t>(mNumBytesAndAlignmentPerSmemChunk.size()));
    }

private:
    // Helper function to calculate padded size
    int32_t getSizePaddedToAlignment(int32_t size, int32_t alignment) const
    {
        TLLM_CHECK_ERROR((alignment & (alignment - 1)) == 0, "Alignment must be a power-of-two");
        return (size + alignment - 1) & ~(alignment - 1);
    }

private:
    // Sizes and alignment requirements of each chunk
    // NOTE: be careful and make sure that the memory dependency is clear and
    // chunks in the beginning of the SMEM can be overwritten.
    std::vector<std::pair<int32_t, int32_t>> mNumBytesAndAlignmentPerSmemChunk;
    // Chunk reuse configuration. True at ith position means that ith chunk starts at smemOffset = 0.
    std::vector<bool> mFirstChunkReuse;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class KernelTraits
{
public:
    // The default constructor.
    KernelTraits() {}

    // The constructor.
    KernelTraits(tg::Dtype dtypeElt, tg::Dtype dtypeC, tg::Dtype dtypeAcc, int32_t tileM, int32_t tileN, int32_t tileK,
        int32_t epilogueTileM, int32_t epilogueTileN, int32_t numStages, int32_t numSlicesForSplitK, SplitK splitK,
        bool useTmaStore, bool transposeMmaOutput, AllReduceAlgo allReduceAlgo, bool useDeepSeekFp8)
    {

        // SMEM chunks
        //
        // [smemA ] (1024B aligned)
        // [smemB ] (1024B aligned)
        // [gmemC0] (1024B aligned) (if needed)
        // [gmemC1] (1024B aligned) (if needed)
        // [rowMax] (16B aligned) (if needed)
        //
        // SMEM for smemA and smemB might be repurposed and used for gmemC0 and gmemC1:
        //
        // [..smemA..][..smemB..]
        // [..gmemC0..][..gmemC1..][..rowMax..]
        //

        std::vector<std::pair<int32_t, int32_t>> numBytesAndAlignmentPerSmemChunk;
        std::vector<bool> firstChunkReuse;

        // LoadA
        {
            // Number of bytes in load A shared memory.
            auto const numSmemBytesLoadA = numStages * tileM * tileK * tg::dtypeGetNumBits(dtypeElt) / 8 /* bits */;
            // Number of bytes for load A alignment for TMA load.
            auto const numBytesAlignmentLoadA = 1024;
            // loadA is already at first chunk. No need to reuse it.
            auto const reuseChunksSmemLoadA = false;

            // Add info.
            numBytesAndAlignmentPerSmemChunk.emplace_back(std::make_pair(numSmemBytesLoadA, numBytesAlignmentLoadA));
            firstChunkReuse.emplace_back(reuseChunksSmemLoadA);
        }

        // LoadB
        {
            // Number of bytes in load B shared memory.
            auto const numSmemBytesLoadB = numStages * tileN * tileK * tg::dtypeGetNumBits(dtypeElt) / 8 /* bits */;
            // Number of bytes for load B alignment for TMA load.
            auto const numBytesAlignmentLoadB = 1024;
            // No need to reuse the first chunk.
            auto const reuseChunksSmemLoadB = false;

            // Add info.
            numBytesAndAlignmentPerSmemChunk.emplace_back(std::make_pair(numSmemBytesLoadB, numBytesAlignmentLoadB));
            firstChunkReuse.emplace_back(reuseChunksSmemLoadB);
        }

        // GmemC
        // FIXME we might need to fix this for GemmGatedAct, it needs less SMEM to store gated output.
        for (int resIdx = 0; resIdx < 2; ++resIdx)
        {
            // Type of the data in the SMEM for GmemC
            auto dtypeSmemC = dtypeC;
            if (allReduceAlgo == AllReduceAlgo::TwoShot || numSlicesForSplitK > 1)
            {
                dtypeSmemC = dtypeAcc;
            }
            // Smem is used for GmemC output tile for TMA store and SplitK in CGA.
            bool usesSmemForGmemC = useTmaStore || doesSplitKUseDsmem(splitK);
            // SMEM for at leader CTA in DSMEM split-k contains K slices.
            auto extraGmemCMultiplier = doesSplitKUseDsmem(splitK) ? numSlicesForSplitK : 1;
            if (resIdx != 0 && !useDeepSeekFp8)
            {
                // No data for Epilogue1 in case of non-DeepSeek GEMM.
                extraGmemCMultiplier = 0;
            }

            // Number of bytes to store the output in smem.
            auto const numBytesSmemStoreC = usesSmemForGmemC
                ? extraGmemCMultiplier * epilogueTileM * epilogueTileN * tg::dtypeGetNumBits(dtypeSmemC) / 8 /* bits */
                : 0;
            // Number of bytes for store C alignment for TMA store.
            auto const numBytesAlignmentStoreC = 1024;
            // gmemC reuses loadAb memory for split-K in DSMEM.
            // Epilogue1 does not reuse and continues after the memory allocated Epilogue0
            // NOTE: we can always reuse loadAb SMEM as long as we don't have persistent scheduler.
            auto const reuseFirstChunksSmemStoreC = doesSplitKUseDsmem(splitK) && resIdx == 0;

            // Add info.
            numBytesAndAlignmentPerSmemChunk.emplace_back(std::make_pair(numBytesSmemStoreC, numBytesAlignmentStoreC));
            firstChunkReuse.emplace_back(reuseFirstChunksSmemStoreC);
        }

        // RowMax
        {
            // Number of dqSfsC per CTA.
            auto const numDqSfsCPerCta = transposeMmaOutput ? tileM : tileN;
            // Number of bytes for rowMax in SMEM.
            auto const numBytesSmemRowMax
                = (useDeepSeekFp8 ? numDqSfsCPerCta : 0) * tg::dtypeGetNumBits(tg::Dtype::Fp32) / 8 /* bits */;
            // Number of bytes alignment for rowMax in SMEM.
            auto const numBytesAlignmentRowMax = 16;

            // Add info.
            numBytesAndAlignmentPerSmemChunk.emplace_back(std::make_pair(numBytesSmemRowMax, numBytesAlignmentRowMax));
            firstChunkReuse.emplace_back(false);
        }

        // Create SMEM helper object.
        mSmemAllocatorHelper = SmemAllocatorHelper(numBytesAndAlignmentPerSmemChunk, firstChunkReuse);
    }

public:
    // Helper for smem allocation.
    SmemAllocatorHelper mSmemAllocatorHelper;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemBufferSize(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getTotalSize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Starting address of each SMEM buffer.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetLoadA(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffset(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetLoadB(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffset(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetLoadAb(KernelTraits traits)
{
    return getSmemOffsetLoadA(traits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetGmemC(KernelTraits traits, int resIdx = 0)
{
    return traits.mSmemAllocatorHelper.getChunkOffset(2 + resIdx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetRowMax(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffset(4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
