/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "Enums.h"
#include "trtllm/gen/CommonUtils.h"
#include "trtllm/gen/DtypeDecl.h"
#include "trtllm/gen/MmaDecl.h"
#include <cassert>
#include <stdexcept>

namespace gemm
{

namespace gemm
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Structure to manage memory allocation with configurable reuse
class MemAllocatorHelper
{
public:
    // The default constructor.
    MemAllocatorHelper() {}

    // Constructor to initialize chunk sizes, alignments, and reuse flags
    MemAllocatorHelper(std::vector<std::pair<int32_t, int32_t>> const& sizes, std::vector<bool> const& reuse,
        std::vector<std::string> const& names)
        : mNumBytesAndAlignmentPerSmemChunk(sizes)
        , mFirstChunkReuse(reuse)
        , mSmemChunkNames(names)
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
    int32_t getChunkOffsetByName(std::string const& name) const
    {
        for (size_t ii = 0; ii < mSmemChunkNames.size(); ++ii)
        {
            if (mSmemChunkNames[ii] == name)
            {
                return getChunkOffset(ii);
            }
        }
        throw std::runtime_error("Name not found: " + name);
    }

    // Returns the first chunk reuse flag given chunk name.
    int getFirstChunkReuseFlagByName(std::string const& name) const
    {
        for (size_t ii = 0; ii < mSmemChunkNames.size(); ++ii)
        {
            if (mSmemChunkNames[ii] == name)
            {
                return getFirstChunkReuseFlag(ii);
            }
        }
        throw std::runtime_error("Name not found: " + name);
    }

    // Function to calculate the total size of the SMEM array
    int32_t getTotalSize() const
    {
        return getOffsetBeforeChunk(static_cast<int32_t>(mNumBytesAndAlignmentPerSmemChunk.size()));
    }

    // Print the contents of this object.
    void print() const
    {
        for (size_t ii = 0; ii < mNumBytesAndAlignmentPerSmemChunk.size(); ++ii)
        {
            printf("Chunk %zd %s: %d bytes, %d alignment, reuse %s, offset %d\n", ii, mSmemChunkNames[ii].c_str(),
                mNumBytesAndAlignmentPerSmemChunk[ii].first, mNumBytesAndAlignmentPerSmemChunk[ii].second,
                mFirstChunkReuse[ii] ? "true" : "false", getChunkOffset(ii));
        }
    }

private:
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

    // Returns the first chunk reuse flag for the ith chunk.
    int getFirstChunkReuseFlag(int32_t ii) const
    {
        return mFirstChunkReuse[ii];
    }

    // Helper function to calculate padded size
    int32_t getSizePaddedToAlignment(int32_t size, int32_t alignment) const
    {
        assert((alignment & (alignment - 1)) == 0);
        return (size + alignment - 1) & ~(alignment - 1);
    }

private:
    // Sizes and alignment requirements of each chunk
    // NOTE: be careful and make sure that the memory dependency is clear and
    // chunks in the beginning of the SMEM can be overwritten.
    std::vector<std::pair<int32_t, int32_t>> mNumBytesAndAlignmentPerSmemChunk;
    // Chunk reuse configuration. True at ith position means that ith chunk starts at smemOffset = 0.
    std::vector<bool> mFirstChunkReuse;
    // Buffer names for inspection purposes.
    std::vector<std::string> mSmemChunkNames;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

int getNumSmemBitsPerElt(tg::Dtype dtype, tg::MmaKind mmaKind)
{
    if (mmaKind == tg::MmaKind::Auto)
    {
        throw std::runtime_error("mmaKind != tg::MmaKind::Auto");
    }
    if (mmaKind == tg::MmaKind::MxFp8Fp6Fp4)
    {
        return 8;
    }
    else
    {
        return tg::dtypeGetNumBits(dtype);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

class KernelTraits
{
public:
    // The default constructor.
    KernelTraits() {}

    // The constructor.
    KernelTraits(tg::Dtype dtypeA, tg::Dtype dtypeB, tg::Dtype dtypeC, tg::Dtype dtypeAcc, tg::Dtype dtypeMmaA,
        tg::Dtype dtypeMmaB, tg::MmaKind mmaKind, int32_t mmaK, int32_t tileM, int32_t tileN, int32_t tileK,
        int32_t epilogueTileM, int32_t epilogueTileN, int32_t numStages, int32_t numStagesMma,
        int32_t numSlicesForSplitK, int32_t numSlicesForSliceK, SplitK splitK, bool useTmaStore,
        bool transposeMmaOutput, AllReduceAlgo allReduceAlgo, bool usePersistentScheduler, bool useDeepSeekFp8,
        bool usePerTokenSfA, bool usePerTokenSfB, BiasType biasType)
        : mMmaKind{mmaKind}
    {
        //
        // SMEM
        //
        {
            // [smemA        ] (1024B aligned)
            // [smemB        ] (1024B aligned)
            // [smemBShuffle ] (1024B aligned)
            // [gmemC0       ] (1024B aligned) (if needed)
            // [gmemC1       ] (1024B aligned) (if needed)
            // [rowMax       ] (16B aligned) (if needed)
            // [sliceK       ] (16B aligned) (if needed)
            // [per-token SF ] (16B aligned) (if needed)
            // [bias         ] (16B aligned) (if needed)
            //
            // SMEM for smemA and smemB might be repurposed and used for gmemC0 and gmemC1:
            //
            // [..smemA..][..smemB..][..smemBShuffle..]
            // [..gmemC0..][..gmemC1..][..rowMax..][..sliceK..][..per-token SF..][..bias..]
            //

            if (mMmaKind == tg::MmaKind::Auto)
            {
                mMmaKind = dtypeGetMmaKind(dtypeMmaA, dtypeMmaB);
            }

            std::vector<std::pair<int32_t, int32_t>> numBytesAndAlignmentPerSmemChunk;
            std::vector<bool> firstChunkReuseSmem;
            // Buffer names for inspection purposes.
            std::vector<std::string> smemChunkNames;

            // LoadA
            {
                // Number of bytes in load A shared memory.
                auto const numSmemBytesLoadA
                    = numStages * tileM * tileK * getNumSmemBitsPerElt(dtypeA, mMmaKind) / 8 /* bits */;
                // Number of bytes for load A alignment for TMA load.
                auto const numBytesAlignmentLoadA = 1024;
                // loadA is already at first chunk. No need to reuse it.
                auto const reuseChunksSmemLoadA = false;
                // Add info.
                smemChunkNames.emplace_back("smemLoadA");
                numBytesAndAlignmentPerSmemChunk.emplace_back(
                    std::make_pair(numSmemBytesLoadA, numBytesAlignmentLoadA));
                firstChunkReuseSmem.emplace_back(reuseChunksSmemLoadA);
            }

            // LoadB
            {
                // Number of bytes in load B shared memory.
                auto const numSmemBytesLoadB
                    = numStages * tileN * tileK * getNumSmemBitsPerElt(dtypeB, mMmaKind) / 8 /* bits */;
                // Number of bytes for load B alignment for TMA load.
                auto const numBytesAlignmentLoadB = 1024;
                // No need to reuse the first chunk.
                auto const reuseChunksSmemLoadB = false;
                // Add info.
                smemChunkNames.emplace_back("smemLoadB");
                numBytesAndAlignmentPerSmemChunk.emplace_back(
                    std::make_pair(numSmemBytesLoadB, numBytesAlignmentLoadB));
                firstChunkReuseSmem.emplace_back(reuseChunksSmemLoadB);
            }

            // SmemBShuffle
            // FIXME: we should be able either:
            // - Do modification in-place. For that we need to resolve pipeline dependency between
            // smemB -> shuffleSmemB -> mma
            // - Do 4 TMA SW32 loads or several LDGSTS loads.
            {
                // Number of bytes in save shuffled B in shared memory.
                auto const numSmemBytesLoadB = numSlicesForSliceK > 1
                    ? numStages * tileN * tileK * getNumSmemBitsPerElt(dtypeB, mMmaKind) / 8 /* bits */
                    : 0;
                // Number of bytes for load B alignment for TMA load.
                auto const numBytesAlignmentLoadB = 1024;
                // No need to reuse the first chunk.
                auto const reuseChunksSmemLoadB = false;

                // Add info.
                smemChunkNames.emplace_back("smemBShuffle");
                numBytesAndAlignmentPerSmemChunk.emplace_back(
                    std::make_pair(numSmemBytesLoadB, numBytesAlignmentLoadB));
                firstChunkReuseSmem.emplace_back(reuseChunksSmemLoadB);
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
                if (numSlicesForSliceK > 1)
                {
                    // TileN is expanded in N dimension for slice-K.
                    extraGmemCMultiplier *= numSlicesForSliceK;
                }

                if (resIdx != 0 && !useDeepSeekFp8)
                {
                    // No data for Epilogue1 in case of non-DeepSeek GEMM.
                    extraGmemCMultiplier = 0;
                }

                // Number of bytes to store the output in smem.
                auto const numBytesSmemStoreC = usesSmemForGmemC ? extraGmemCMultiplier * epilogueTileM * epilogueTileN
                        * tg::dtypeGetNumBits(dtypeSmemC) / 8 /* bits */
                                                                 : 0;
                // Number of bytes for store C alignment for TMA store.
                auto const numBytesAlignmentStoreC = 1024;
                // gmemC reuses loadAb memory for split-K in DSMEM.
                // Epilogue1 does not reuse and continues after the memory allocated Epilogue0
                // NOTE: we can always reuse loadAb SMEM as long as we don't have persistent scheduler.
                auto const reuseFirstChunksSmemStoreC
                    = doesSplitKUseDsmem(splitK) && resIdx == 0 && !usePersistentScheduler;

                // Add info.
                smemChunkNames.emplace_back("smemGmemC" + std::to_string(resIdx));
                numBytesAndAlignmentPerSmemChunk.emplace_back(
                    std::make_pair(numBytesSmemStoreC, numBytesAlignmentStoreC));
                firstChunkReuseSmem.emplace_back(reuseFirstChunksSmemStoreC);
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
                smemChunkNames.emplace_back("smemRowMax");
                numBytesAndAlignmentPerSmemChunk.emplace_back(
                    std::make_pair(numBytesSmemRowMax, numBytesAlignmentRowMax));
                firstChunkReuseSmem.emplace_back(false);
            }

            // SliceK
            {
                // Real tile size before slice-K reduction.
                auto const tileSize
                    = numSlicesForSliceK > 1 ? numSlicesForSliceK * tileM * numSlicesForSliceK * tileN : 0;
                // Number of bytes for tile in SMEM.
                auto const numBytesSmemTile = tileSize * tg::dtypeGetNumBits(dtypeAcc) / 8 /* bits */;
                // Number of bytes alignment for rowMax in SMEM.
                auto const numBytesAlignmentTile = 16;

                // Add info.
                smemChunkNames.emplace_back("smemSliceK");
                numBytesAndAlignmentPerSmemChunk.emplace_back(std::make_pair(numBytesSmemTile, numBytesAlignmentTile));
                firstChunkReuseSmem.emplace_back(false);
            }

            // Per-token Scale Factors
            {
                // Number of bytes for per-token scale factors
                auto const numBytesSmemPerTokenSf
                    = (usePerTokenSfA ? (tileM) * sizeof(float) : 0) + (usePerTokenSfB ? (tileN) * sizeof(float) : 0);
                // Number of bytes alignment for per-token scale factors
                auto const numBytesAlignmentPerTokenSf = 16;
                // Add info.
                smemChunkNames.emplace_back("smemPerTokenSf");
                numBytesAndAlignmentPerSmemChunk.emplace_back(
                    std::make_pair(numBytesSmemPerTokenSf, numBytesAlignmentPerTokenSf));
                firstChunkReuseSmem.emplace_back(false);
            }

            // Bias
            {
                int32_t numBytesSmemBias = 0;
                if (isBiasTypeN(biasType))
                {
                    numBytesSmemBias = tileN * sizeof(float);
                }
                else if (isBiasTypeM(biasType))
                {
                    numBytesSmemBias = tileM * sizeof(float);
                }
                else if (isBiasTypeMn(biasType))
                {
                    numBytesSmemBias = tileM * tileN * sizeof(float);
                }
                // Number of bytes alignment for bias
                auto const numBytesAlignmentBias = 16;
                // Add info.
                smemChunkNames.emplace_back("smemBias");
                numBytesAndAlignmentPerSmemChunk.emplace_back(std::make_pair(numBytesSmemBias, numBytesAlignmentBias));
                firstChunkReuseSmem.emplace_back(false);
            }

            // Per-block absolute maximum for multi-warp reduction.
            {
                // Number of bytes: number of epilogue warps * number of tile columns.
                // TODO: avoid allocating this memory when it's not needed (it's only for MxFp8 + fusedAct)
                auto const numBytesSmemBlockAmax = transposeMmaOutput ? 4 * tileN * sizeof(float) : 0;
                // Number of bytes alignment.
                auto const numBytesAlignmentBlockAmax = 16;
                // Add info.
                smemChunkNames.emplace_back("smemBlockAmax");
                numBytesAndAlignmentPerSmemChunk.emplace_back(
                    std::make_pair(numBytesSmemBlockAmax, numBytesAlignmentBlockAmax));
                firstChunkReuseSmem.emplace_back(false);
            }

            // SmemConstSfBuf
            // A buffer used to copy constant values to TMEM.
            {
                // Do we need the buffer?
                bool const useConstSfBuf = dtypeB == tg::Dtype::E4m3 && dtypeMmaB == tg::Dtype::MxE4m3;
                // Number of bytes for the buffer.
                auto const numSmemBytesConstSfBuf = useConstSfBuf ? 512 : 0;
                // Number of bytes for the alignment of the buffer.
                auto const numBytesAlignmentConstSfBuf = 16;
                // No need to reuse the first chunk.
                auto const reuseChunksSmemConstSfBuf = false;

                // Add info.
                smemChunkNames.emplace_back("smemConstSfBuf");
                numBytesAndAlignmentPerSmemChunk.emplace_back(
                    std::make_pair(numSmemBytesConstSfBuf, numBytesAlignmentConstSfBuf));
                firstChunkReuseSmem.emplace_back(reuseChunksSmemConstSfBuf);
            }

            // Create SMEM helper object.
            mSmemAllocatorHelper
                = MemAllocatorHelper(numBytesAndAlignmentPerSmemChunk, firstChunkReuseSmem, smemChunkNames);
#if 0
      // E.g.,
      // Chunk 0 smemLoadA: 32768 bytes, 1024 alignment, false, offset 0
      // Chunk 1 smemLoadB: 32768 bytes, 1024 alignment, false, offset 32768
      // Chunk 2 smemBShuffle: 0 bytes, 1024 alignment, false, offset 65536
      // Chunk 3 smemGmemC0: 65536 bytes, 1024 alignment, true, offset 0
      // Chunk 4 smemGmemC1: 65536 bytes, 1024 alignment, false, offset 65536
      // Chunk 5 smemRowMax: 512 bytes, 16 alignment, false, offset 131072
      // Chunk 6 smemSliceK: 0 bytes, 16 alignment, false, offset 131584
      // Chunk 7 smemPerTokenSf: 0 bytes, 16 alignment, false, offset 131584
      mSmemAllocatorHelper.print();
#endif
        }

        //
        // TMEM
        //
        // [..D..][..A..][.SfA.][.SfB.]
        {
            std::vector<std::pair<int32_t, int32_t>> numBytesAndAlignmentPerTmemChunk;
            std::vector<bool> firstChunkReuseTmem;
            std::vector<std::string> tmemChunkNames;
            // Matrix D
            {
                // Number of columns for accumulators.
                auto const numTmemColsD = numSlicesForSliceK * tileN * numStagesMma * tg::dtypeGetNumBits(dtypeAcc)
                    / tg::dtypeGetNumBits(tg::Dtype::UInt32);
                // Number of columns for D alignment.
                auto const numColsAlignmentD = 2;
                // No need to reuse TMEM.
                auto const reuseChunksTmemD = false;

                // Add info.
                tmemChunkNames.emplace_back("tmemD");
                numBytesAndAlignmentPerTmemChunk.emplace_back(std::make_pair(numTmemColsD, numColsAlignmentD));
                firstChunkReuseTmem.emplace_back(reuseChunksTmemD);
            }

            // Matrix A
            {
                // We use TMEM for A if we use slice-K or if we need to cast A.
                bool const useTmemA = (numSlicesForSliceK > 1) || (dtypeMmaA != dtypeA);
                // Number of columns for A.
                auto const numTmemColsA = useTmemA ? numStages * tileK
                        / (numSlicesForSliceK * tg::dtypeGetNumBits(tg::Dtype::UInt32) / tg::dtypeGetNumBits(dtypeMmaA))
                                                   : 0;
                // Number of columns for A alignment.
                auto const numColsAlignmentA = 4;
                // No need to reuse TMEM.
                auto const reuseChunksTmemA = false;

                // Add info.
                tmemChunkNames.emplace_back("tmemA");
                numBytesAndAlignmentPerTmemChunk.emplace_back(std::make_pair(numTmemColsA, numColsAlignmentA));
                firstChunkReuseTmem.emplace_back(reuseChunksTmemA);
            }

            // Sf A
            {
                // Does the MMA require block scales in TMEM for A?
                bool const useBlockScalingA = tg::dtypeIsBlockFmt(dtypeMmaA);
                // Are the block scales constant?
                bool const useConstSfA = useBlockScalingA && !tg::dtypeIsBlockFmt(dtypeA);
                // Number of columns for scaling factors of A.
                auto const numTmemColsSfA = useConstSfA
                    ? tg::roundUp((tileK / 64) * tg::getTmemColStridePerGroup(tileM, mmaK), 4)
                    : (useBlockScalingA ? ((tileK / 64) * tg::getTmemColStridePerGroup(tileM, mmaK)) * numStages : 0);
                // Number of columns for Sf alignment.
                auto const numColsAlignmentSfA = 4;
                // No need to reuse TMEM.
                auto const reuseChunksTmemSfA = false;

                // Add info.
                tmemChunkNames.emplace_back("tmemSfA");
                numBytesAndAlignmentPerTmemChunk.emplace_back(std::make_pair(numTmemColsSfA, numColsAlignmentSfA));
                firstChunkReuseTmem.emplace_back(reuseChunksTmemSfA);
            }

            // Sf B
            {
                // Does the MMA require block scales in TMEM for B?
                bool const useBlockScalingB = tg::dtypeIsBlockFmt(dtypeMmaB);
                // Are the block scales constant?
                bool const useConstSfB = useBlockScalingB && !tg::dtypeIsBlockFmt(dtypeB);
                // Number of columns for scaling factors of B.
                auto const numTmemColsSfB = useConstSfB
                    ? tg::roundUp((tileK / 64) * tg::getTmemColStridePerGroup(tileN, mmaK), 4)
                    : (useBlockScalingB ? ((tileK / 64) * tg::getTmemColStridePerGroup(tileN, mmaK)) * numStages : 0);
                // Number of columns for Sf alignment.
                auto const numColsAlignmentSfB = 4;
                // No need to reuse TMEM.
                auto const reuseChunksTmemSfB = false;

                // Add info.
                tmemChunkNames.emplace_back("tmemSfB");
                numBytesAndAlignmentPerTmemChunk.emplace_back(std::make_pair(numTmemColsSfB, numColsAlignmentSfB));
                firstChunkReuseTmem.emplace_back(reuseChunksTmemSfB);
            }

            // Create TMEM helper object.
            mTmemAllocatorHelper
                = MemAllocatorHelper(numBytesAndAlignmentPerTmemChunk, firstChunkReuseTmem, tmemChunkNames);
        }
    }

public:
    // The MMA kind.
    tg::MmaKind mMmaKind;
    // Helper for SMEM allocation.
    MemAllocatorHelper mSmemAllocatorHelper;
    // Helper for TMEM allocation.
    MemAllocatorHelper mTmemAllocatorHelper;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemBufferSize(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getTotalSize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getTmemBufferSize(KernelTraits traits)
{
    return traits.mTmemAllocatorHelper.getTotalSize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Starting address of each SMEM buffer.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetLoadA(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffsetByName("smemLoadA");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetLoadB(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffsetByName("smemLoadB");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetLoadAb(KernelTraits traits)
{
    return getSmemOffsetLoadA(traits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetLoadShuffleB(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffsetByName("smemBShuffle");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetGmemC(KernelTraits traits, int resIdx = 0)
{
    return traits.mSmemAllocatorHelper.getChunkOffsetByName("smemGmemC" + std::to_string(resIdx));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetRowMax(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffsetByName("smemRowMax");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetSliceK(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffsetByName("smemSliceK");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetPerTokenSf(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffsetByName("smemPerTokenSf");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetBias(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffsetByName("smemBias");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetBlockAmax(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffsetByName("smemBlockAmax");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getSmemOffsetConstSfBuf(KernelTraits traits)
{
    return traits.mSmemAllocatorHelper.getChunkOffsetByName("smemConstSfBuf");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t isSmemAbRepurposedToGmemC(KernelTraits traits, int resIdx = 0)
{
    return traits.mSmemAllocatorHelper.getFirstChunkReuseFlagByName("smemGmemC" + std::to_string(resIdx));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Starting address of each TMEM buffer.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getTmemOffsetD(KernelTraits traits)
{
    return traits.mTmemAllocatorHelper.getChunkOffsetByName("tmemD");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getTmemOffsetA(KernelTraits traits)
{
    return traits.mTmemAllocatorHelper.getChunkOffsetByName("tmemA");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getTmemOffsetSfA(KernelTraits traits)
{
    return traits.mTmemAllocatorHelper.getChunkOffsetByName("tmemSfA");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getTmemOffsetSfB(KernelTraits traits)
{
    return traits.mTmemAllocatorHelper.getChunkOffsetByName("tmemSfB");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm

} // namespace gemm
