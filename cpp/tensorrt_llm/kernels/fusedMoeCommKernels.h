/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <map>

#include <cuda_runtime_api.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/moeCommKernelsCommon.h"

namespace tensorrt_llm
{
namespace kernels
{

struct ALIGN_256 SenderSideFifoInfo
{
    volatile uint64_t head; // write position
    volatile uint64_t tail; // read position
};

struct ALIGN_256 ReceiverSideFifoInfo
{
    volatile uint64_t head; // write position do we use this?
    volatile uint64_t tail; // read position
};

// struct holding Send/Recv data pointer and its displacement information.
struct SendRecvIndices
{
    int const* rankCountCumSum; // length = epSize
    int* rankLocalIndices;      // length = rankCountCumSum[epRank] - rankCountCumSum[epRank - 1] if epRank > 0 else
                                // rankCountCumSum[epRank]

#ifdef __CUDACC__
    __inline__ __device__ int getCount(int rank) const
    {
        return rank == 0 ? rankCountCumSum[rank] : rankCountCumSum[rank] - rankCountCumSum[rank - 1];
    }

    __inline__ __device__ int getRankStart(int rank) const
    {
        return rank == 0 ? 0 : rankCountCumSum[rank - 1];
    }

    __inline__ __device__ int* getGroupStart(int rank, int& tokenCount) const
    {
        tokenCount = getCount(rank);
        int rankStart = getRankStart(rank);
        return rankLocalIndices + rankStart;
    }
#endif
};

struct MoeCommFieldInfo
{
    uint8_t* dataPtrBase;
    uint8_t alignedUnitBit;          // 0, 1, 2, 3, 4 (for 1, 2, 4, 8, 16 Bytes), smallest aligned unit.
    uint16_t alignedUnitCount;       // data count in aligned unit
    uint16_t alignedUnitStride;      // data stride in aligned unit

    uint8_t unalignedFieldIndex;     // the index of unaligned Field, no decrease with field index
    uint16_t compact16BOffset;       // aligned to 16 Bytes, offset is count of 16 Byte

    cudaDataType_t originalDataType; // original data type, used for low precision alltoall.

    static constexpr uint64_t kAlign16BytePtrMask = (1ULL << 4) - 1;
    static constexpr uint32_t kAligned16BMask = (1 << 4) - 1;

    // Constants for memory alignment and access
    static constexpr int BYTES_PER_128B_BLOCK = 128;
    static constexpr int INTS_PER_128B_BLOCK = BYTES_PER_128B_BLOCK / sizeof(int);
    static constexpr int UINT64_PER_128B_BLOCK = BYTES_PER_128B_BLOCK / sizeof(uint64_t);
    static constexpr int BYTES_PER_16B_BLOCK = 16;
    // Will pad one 16 byte for each unaligned field, then head and tail 16 byte might not be aligned

    // Fill single field info, the fields that need global info is not filled here.
    __host__ void fillFieldInfo(
        uint8_t* dataPtr, size_t elementSize, int vectorSize, int stride, cudaDataType_t originalDataType);

    __host__ void setUnused()
    {
        dataPtrBase = nullptr;
        alignedUnitBit = 4;
        alignedUnitCount = 0;
        alignedUnitStride = 0;
        unalignedFieldIndex = 0;
        compact16BOffset = 0;
    }

    __device__ __host__ __forceinline__ int getFieldUncompactSize() const
    {
        int alignedUnitBytes = 1 << alignedUnitBit;
        int currentFieldSize = alignedUnitCount * alignedUnitBytes;
        if (alignedUnitBytes != 16)
        {
            constexpr int alignedUnitBytes = BYTES_PER_16B_BLOCK;
            currentFieldSize = currentFieldSize / alignedUnitBytes * alignedUnitBytes;
            currentFieldSize += alignedUnitBytes * 2;
        }
        return currentFieldSize;
    }

    __device__ __host__ __forceinline__ int getFieldCompactSize() const
    {
        int alignedUnitBytes = 1 << alignedUnitBit;
        int currentFieldSize = alignedUnitCount * alignedUnitBytes;
        // Align to 16 bytes for compact size
        return (currentFieldSize + BYTES_PER_16B_BLOCK - 1) / BYTES_PER_16B_BLOCK * BYTES_PER_16B_BLOCK;
    }

    __device__ __forceinline__ int getCompactShmOffset() const
    {
        return compact16BOffset * BYTES_PER_16B_BLOCK;
    }

    __device__ __forceinline__ int getUncompactShmOffset() const
    {
        // each unaligned field need 16 byte head and 16 byte tail
        return compact16BOffset * BYTES_PER_16B_BLOCK + unalignedFieldIndex * BYTES_PER_16B_BLOCK;
    }

    __device__ __forceinline__ int getMemmoveOffsets(int index) const
    {
        int alignedBytes = 1 << alignedUnitBit;
        uint8_t* dataPtr = dataPtrBase + index * alignedBytes * alignedUnitStride;
        int offset = reinterpret_cast<uint64_t>(dataPtr) & kAlign16BytePtrMask;
        return offset + unalignedFieldIndex * BYTES_PER_16B_BLOCK;
    }

    __device__ __forceinline__ uint8_t* getRawPtr(int index, int* rawSize) const
    {
        int alignedBytes = 1 << alignedUnitBit;
        uint8_t* dataPtr = dataPtrBase + static_cast<size_t>(index) * alignedBytes * alignedUnitStride;
        if (rawSize != nullptr)
        {
            *rawSize = alignedUnitCount * alignedBytes;
        }
        return dataPtr;
    }

    __device__ __forceinline__ uint8_t* get16BAlignedLoadCopyRange(int index, int* copyByteCount) const
    {
        int rawSize;
        uint8_t* rawDataPtr = getRawPtr(index, &rawSize);
        uint8_t* rawEndPtr = rawDataPtr + rawSize;
        uint8_t* alignedDataPtr
            = reinterpret_cast<uint8_t*>(reinterpret_cast<uint64_t>(rawDataPtr) & (~kAlign16BytePtrMask));
        uint32_t copySize = rawEndPtr - alignedDataPtr;
        *copyByteCount
            = (copySize & kAligned16BMask) != 0 ? (copySize & (~kAligned16BMask)) + BYTES_PER_16B_BLOCK : copySize;
        return alignedDataPtr;
    }

    __device__ __forceinline__ uint8_t* get16BAlignedStoreCopyRange(
        int index, int* copyByteCount, int laneId, int* headTailShmIdx, int* headTailGlobalIdx) const
    {
        int rawSize;
        uint8_t* rawDataPtr = getRawPtr(index, &rawSize);
        uint8_t* rawEndPtr = rawDataPtr + rawSize;
        int offset = reinterpret_cast<uint64_t>(rawDataPtr) & kAlign16BytePtrMask;
        uint8_t* alignedDataPtr
            = reinterpret_cast<uint8_t*>(reinterpret_cast<uint64_t>(rawDataPtr) + BYTES_PER_16B_BLOCK - offset);
        uint8_t* alignedEndPtr
            = reinterpret_cast<uint8_t*>(reinterpret_cast<uint64_t>(rawEndPtr) & (~kAlign16BytePtrMask));
        int alignedCopyBytes = alignedEndPtr - alignedDataPtr;
        if (alignedCopyBytes < 0)
        {
            alignedCopyBytes = 0;
        }
        *copyByteCount = alignedCopyBytes;

        if (laneId < BYTES_PER_16B_BLOCK)
        {
            *headTailShmIdx = laneId;
        }
        else
        {
            *headTailShmIdx = laneId + alignedCopyBytes;
        }
        *headTailGlobalIdx = *headTailShmIdx - offset;
        if (*headTailGlobalIdx < 0 || *headTailGlobalIdx >= rawSize)
        {
            *headTailGlobalIdx = -1;
            *headTailShmIdx = -1;
        }
        return alignedDataPtr;
    }
};

// Maximum number of field supported, except tokenSelectedExpert and expertScales
static constexpr int MOE_COMM_FIELD_MAX_COUNT = 8;

struct MoeSingleCommMeta
{
    int singleTransferAlignedSize;  // transfer size aligned to 128 bytes.
    int singleCompactAlignedSize;   // compact buffer is always aligned to 128 bytes
    int singleUncompactAlignedSize; // uncompact shared memory size, aligned to 128 bytes, might be larger than compact
                                    // buffer if unaligned field exist.

    // TODO: Do we need reduce shared memory usage, make it able to be smaller, and enable multiple wave?

    __device__ __host__ __forceinline__ int getTransfer128ByteCount() const
    {
        return singleTransferAlignedSize / MoeCommFieldInfo::BYTES_PER_128B_BLOCK;
    }

    __device__ __host__ __forceinline__ int getCompactData128ByteCount() const
    {
        return singleCompactAlignedSize / MoeCommFieldInfo::BYTES_PER_128B_BLOCK;
    }

    __device__ __host__ __forceinline__ int getSingleShmSize() const
    {
        return std::max(singleUncompactAlignedSize, singleTransferAlignedSize);
    }
};

struct FusedMoeWorldInfo
{
    MoeEpWorldInfo epInfo;
};

struct FusedMoePairInfo
{
    int senderRank;
    int receiverRank;
    int channel;
    int runChannelCount;
};

class FusedMoeCommunicator
{
public:
    static constexpr int FIFO_DEPTH = 4;
    static constexpr int FIFO_ENTRY_BYTES = 256 * 1024;
    static constexpr int FIFO_ENTRY_128_BYTE_COUNT = FIFO_ENTRY_BYTES / 128;
    static constexpr int FIFO_TOTAL_BYTES = FIFO_ENTRY_BYTES * FIFO_DEPTH;
    static constexpr int FIFO_TOTAL_U64 = FIFO_TOTAL_BYTES / sizeof(uint64_t);
    static constexpr int MAX_GROUP_COUNT_PER_BLOCK = 8;

    static constexpr int WARP_SIZE = 32;

    static int maxSmCount;
    static bool maxSmCountUsed;

    static void setMaxUsableSmCount(int maxUsableSmCount)
    {
        TLLM_CHECK_WITH_INFO(
            FusedMoeCommunicator::maxSmCountUsed == false, "setMaxUsableSmCount can be called only before it is used");
        int smCount = tensorrt_llm::common::getMultiProcessorCount();
        if (maxUsableSmCount > smCount)
        {
            TLLM_LOG_WARNING("setMaxUsableSmCount, maxUsableSmCount=%d, larger than smCount=%d, using smCount instead",
                maxUsableSmCount, smCount);
            maxUsableSmCount = smCount;
        }
        FusedMoeCommunicator::maxSmCount = maxUsableSmCount;
    }

    static int getMaxUsableSmCount()
    {
        FusedMoeCommunicator::maxSmCountUsed = true;
        if (FusedMoeCommunicator::maxSmCount == -1)
        {
            int smCount = tensorrt_llm::common::getMultiProcessorCount();
            FusedMoeCommunicator::maxSmCount = smCount;
        }
        return FusedMoeCommunicator::maxSmCount;
    }

    static int computeMoeCommChannelCount(int epSize)
    {
        int smCount = getMaxUsableSmCount();
        int blockCountPerChannel = (epSize + MAX_GROUP_COUNT_PER_BLOCK - 1) / MAX_GROUP_COUNT_PER_BLOCK;
        blockCountPerChannel *= 2; // for send and recv
        TLLM_CHECK_WITH_INFO(
            blockCountPerChannel <= smCount, "GPU should support at lease one channel, usableSmCount=%d", smCount);
        int perferredChannel = smCount / 2 / blockCountPerChannel; // use half SMs for communication
        int channelCount = std::max(perferredChannel, 1);          // at lease one channel
        return channelCount;
    }

    static int getMoeCommChannelCount(int epSize)
    {
        static std::map<int, int> channelCountMap{};
        auto iter = channelCountMap.find(epSize);
        if (iter == channelCountMap.end())
        {
            auto channelCount = FusedMoeCommunicator::computeMoeCommChannelCount(epSize);
            channelCountMap[epSize] = channelCount;
            return channelCount;
        }
        return iter->second;
    }

    static dim3 getLaunchBlockDim(int groupCountPerCta)
    {
        return dim3(WARP_SIZE, groupCountPerCta);
    }

    static dim3 getLaunchGridDim(int epSize, int groupCountPerCta)
    {
        int maxChannelCount = FusedMoeCommunicator::getMoeCommChannelCount(epSize);
        int targetCtaCount = (epSize + MAX_GROUP_COUNT_PER_BLOCK - 1) / MAX_GROUP_COUNT_PER_BLOCK * maxChannelCount * 2;
        int ctaPerChannel = (epSize + groupCountPerCta - 1) / groupCountPerCta;
        int ctaLimitedChannelCount = targetCtaCount / 2 / ctaPerChannel;
        ctaLimitedChannelCount = std::max(1, ctaLimitedChannelCount);
        int channelCount = std::min(ctaLimitedChannelCount, maxChannelCount);
        return dim3(ctaPerChannel, channelCount, 2);
    }
};

size_t getFusedMoeCommWorkspaceSize(int epSize);

struct FusedMoeFieldInfo
{
    int8_t isBasicInterleaved; // using tokenSelectedSlots and expertScales interleaving?
    int32_t* tokenSelectedSlots;
    float* expertScales;       // can be nullptr if no scale is used(all 1.0), if so, interleaved should all be 0
    int fieldCount;
    MoeCommFieldInfo fieldsInfo[MOE_COMM_FIELD_MAX_COUNT];

    __host__ int computeSingleCompactSize(int topK, bool hasScales, bool hasBasicFields) const
    {
        int basicFieldSize = 0;
        if (hasBasicFields)
        {
            basicFieldSize = topK * sizeof(int) + (hasScales ? topK * sizeof(float) : 0);
            // align to 16 bytes
            basicFieldSize = (basicFieldSize + MoeCommFieldInfo::BYTES_PER_16B_BLOCK - 1)
                / MoeCommFieldInfo::BYTES_PER_16B_BLOCK * MoeCommFieldInfo::BYTES_PER_16B_BLOCK;
        }
        int otherFieldSize = 0;
        for (int i = 0; i < fieldCount; i++)
        {
            MoeCommFieldInfo const& fieldInfo = fieldsInfo[i];
            otherFieldSize += fieldInfo.getFieldCompactSize();
        }
        int totalSize = basicFieldSize + otherFieldSize;
        constexpr int totalSizeAlignment = MoeCommFieldInfo::BYTES_PER_128B_BLOCK;
        totalSize = (totalSize + totalSizeAlignment - 1) / totalSizeAlignment * totalSizeAlignment;
        return totalSize;
    }

    __host__ int computeSingleUncompactSize(int topK, bool hasScales, bool hasBasicFields) const
    {
        int basicFieldSize = 0;
        if (hasBasicFields)
        {
            basicFieldSize = topK * sizeof(int) + (hasScales ? topK * sizeof(float) : 0);
            // align to 16 bytes
            basicFieldSize = (basicFieldSize + MoeCommFieldInfo::BYTES_PER_16B_BLOCK - 1)
                / MoeCommFieldInfo::BYTES_PER_16B_BLOCK * MoeCommFieldInfo::BYTES_PER_16B_BLOCK;
        }
        int otherFieldSize = 0;
        for (int i = 0; i < fieldCount; i++)
        {
            MoeCommFieldInfo const& fieldInfo = fieldsInfo[i];
            otherFieldSize += fieldInfo.getFieldUncompactSize();
        }
        int totalSize = basicFieldSize + otherFieldSize;
        constexpr int totalSizeAlignment = MoeCommFieldInfo::BYTES_PER_128B_BLOCK;
        totalSize = (totalSize + totalSizeAlignment - 1) / totalSizeAlignment * totalSizeAlignment;
        return totalSize;
    }

    template <typename T = int, bool IS_SLOTS = true>
    __device__ __forceinline__ T* getBasicFieldPtr(int tokenIndex, int selectedIndex, int topK) const
    {
        T* fieldPtr = nullptr;
        fieldPtr = IS_SLOTS ? reinterpret_cast<T*>(tokenSelectedSlots) : reinterpret_cast<T*>(expertScales);
        if (fieldPtr == nullptr || selectedIndex >= topK)
        {
            return nullptr;
        }
        int tokenStride = isBasicInterleaved ? topK * 2 : topK;
        int elementStride = isBasicInterleaved ? 2 : 1;
        return fieldPtr + tokenIndex * tokenStride + selectedIndex * elementStride;
    }

    __device__ __forceinline__ int* getTokenSelectedSlotsPtr(int tokenIndex, int selectedIndex, int topK) const
    {
        return getBasicFieldPtr<int, true>(tokenIndex, selectedIndex, topK);
    }

    __device__ __forceinline__ float* getScalePtr(int tokenIndex, int selectedIndex, int topK) const
    {
        return getBasicFieldPtr<float, false>(tokenIndex, selectedIndex, topK);
    }

    void fillMetaInfo(
        MoeSingleCommMeta* singleCommMeta, int topK, bool hasScales, bool hasBasicFields, bool isLowPrecision) const;

    void fillFieldPlacementInfo(int topK, bool hasBasicFields);
};

struct FusedMoeCommKernelParam
{
    FusedMoeWorldInfo worldInfo;
    MoeExpertParallelInfo expertParallelInfo; // expertCount inside should be slotCount if using redundant experts.
    MoeSingleCommMeta sendCommMeta;
    MoeSingleCommMeta recvCommMeta;
    SendRecvIndices sendIndices;
    SendRecvIndices recvIndices;
    FusedMoeFieldInfo sendFieldInfo;
    FusedMoeFieldInfo recvFieldInfo;
    bool isLowPrecision;
};

/*
 * Workspace Layout:
 * Ri: Rank i
 * N: Number of GPUs, e.g. EpSize or WorldSize, n = N - 1
 * Ci: Channel i
 * M: Number of Channels, m = M - 1
 * MMr: Memory Mapped from Rank r, physically located at rank r, and mapped to all ranks.
 *
 *  Whole workspace memory space:
 *  ---------------------------------------------------------------------------------------------------
 *  |<--   MM0   -->   |<--   MM1   -->   |<--   MM2   -->   |       ......        |<--   MMn   -->   |
 *  ^                  ^                  ^                  ^                     ^                  ^
 *  0           rankStrideInU64   2*rankStrideInU64   3*rankStrideInU64     n*rankStrideInU64 N*rankStrideInU64
 *
 *  For each MMr, the layout is:
 *  -------------------------------------------------------------------------------------------------
 *  |<--- FIFO memory --->|<--- SenderSideFifoInfo memory --->|<--- ReceiverSideFifoInfo memory --->|
 *  -------------------------------------------------------------------------------------------------
 *
 *  For each FIFO memory, it is physically placed at the receiver rank.
 *  To find the FIFO whose receiver is rank r, we need to find that in the FIFO memory of MMr.
 *  The layout of FIFO memory of each MMR is(here rank is the sender rank):
 *  -------------------------------------------------------------------------------------------------
 *  | R0C0 | R0C1 | .... | R0Cm | R1C0 | R1C1 | .... | R1Cm | .... .... | RnC0 | RnC1 | .... | RnCm |
 *  |<-  Channels for Rank 0  ->|<-  Channels for Rank 1  ->|           |<-  Channels for Rank n  ->|
 *  -------------------------------------------------------------------------------------------------
 *  Each R*C* has length of FIFO_TOTAL_U64 in uint64_t, which is internally divided into FIFO_DEPTH entries of
 *  size FIFO_ENTRY_BYTES each.
 *
 *  For each SenderSideFifoInfo memory, it is physically placed at the sender rank.
 *  To find the SenderSideFifoInfo whose sender is rank r, we need to find that in the FIFO memory of MMr.
 *  The layout of SenderSideFifoInfo memory of each MMR is(here rank is the receiver rank):
 *  -------------------------------------------------------------------------------------------------
 *  | R0C0 | R0C1 | .... | R0Cm | R1C0 | R1C1 | .... | R1Cm | .... .... | RnC0 | RnC1 | .... | RnCm |
 *  |<-  Channels for Rank 0  ->|<-  Channels for Rank 1  ->|           |<-  Channels for Rank n  ->|
 *  -------------------------------------------------------------------------------------------------
 *  Each R*C* is one struct of SenderSideFifoInfo. There are total M * N SenderSideFifoInfo in each MMR.
 *
 *  For each ReceiverSideFifoInfo memory, it is physically placed at the receiver rank.
 *  To find the ReceiverSideFifoInfo whose receiver is rank r, we need to find that in the FIFO memory of MMr.
 *  The layout of ReceiverSideFifoInfo memory of each MMR is(here rank is the sender rank):
 *  -------------------------------------------------------------------------------------------------
 *  | R0C0 | R0C1 | .... | R0Cm | R1C0 | R1C1 | .... | R1Cm | .... .... | RnC0 | RnC1 | .... | RnCm |
 *  |<-  Channels for Rank 0  ->|<-  Channels for Rank 1  ->|           |<-  Channels for Rank n  ->|
 *  -------------------------------------------------------------------------------------------------
 *  Each R*C* is one struct of ReceiverSideFifoInfo. There are total M * N ReceiverSideFifoInfo in each MMR.
 */

struct FusedMoeWorkspace
{
    uint64_t* workspacePtr;
    size_t rankStrideInU64;
    int channelCount;

    template <bool isSenderSideBuffer>
    __device__ __forceinline__ uint8_t* commonGetPtrBase(
        FusedMoePairInfo const& pairInfo, size_t fieldOffset, int fieldSingleSize) const
    {
        int mappedMemoryrank = isSenderSideBuffer ? pairInfo.senderRank : pairInfo.receiverRank;
        int rankInsideMappedMemory = isSenderSideBuffer ? pairInfo.receiverRank : pairInfo.senderRank;
        auto* mappedMemory = reinterpret_cast<uint8_t*>(workspacePtr + mappedMemoryrank * rankStrideInU64);
        mappedMemory += fieldOffset;
        mappedMemory += rankInsideMappedMemory * channelCount * fieldSingleSize;
        mappedMemory += pairInfo.channel * fieldSingleSize;
        return mappedMemory;
    }

    __device__ __forceinline__ uint64_t* getFifoBasePtr(
        FusedMoeWorldInfo const& worldInfo, FusedMoePairInfo const& pairInfo) const
    {
        constexpr int fieldSingleSize = FusedMoeCommunicator::FIFO_TOTAL_BYTES;
        return reinterpret_cast<uint64_t*>(commonGetPtrBase<false>(pairInfo, 0, fieldSingleSize));
    }

    __device__ __forceinline__ SenderSideFifoInfo* getSenderSideFifoInfo(
        FusedMoeWorldInfo const& worldInfo, FusedMoePairInfo const& pairInfo) const
    {
        constexpr int fieldSingleSize = sizeof(SenderSideFifoInfo);
        size_t fieldOffset
            = static_cast<size_t>(FusedMoeCommunicator::FIFO_TOTAL_BYTES) * worldInfo.epInfo.epSize * channelCount;
        return reinterpret_cast<SenderSideFifoInfo*>(commonGetPtrBase<true>(pairInfo, fieldOffset, fieldSingleSize));
    }

    __device__ __forceinline__ ReceiverSideFifoInfo* getReceiverSideFifoInfo(
        FusedMoeWorldInfo const& worldInfo, FusedMoePairInfo const& pairInfo) const
    {
        constexpr int fieldSingleSize = sizeof(ReceiverSideFifoInfo);
        size_t fieldOffset
            = static_cast<size_t>(FusedMoeCommunicator::FIFO_TOTAL_BYTES) * worldInfo.epInfo.epSize * channelCount
            + sizeof(SenderSideFifoInfo) * worldInfo.epInfo.epSize * channelCount;
        return reinterpret_cast<ReceiverSideFifoInfo*>(commonGetPtrBase<false>(pairInfo, fieldOffset, fieldSingleSize));
    }

    static size_t computeWorkspaceSizePreRank(int epSize, int channelCount)
    {
        size_t fifoSize = static_cast<size_t>(FusedMoeCommunicator::FIFO_TOTAL_BYTES) * epSize * channelCount;
        size_t senderSideInfoSize = sizeof(SenderSideFifoInfo) * epSize * channelCount;
        size_t receiverSideInfoSize = sizeof(ReceiverSideFifoInfo) * epSize * channelCount;
        return fifoSize + senderSideInfoSize + receiverSideInfoSize;
    }

    void initializeLocalWorkspace(FusedMoeWorldInfo const& worldInfo);
};

void setMaxUsableSmCount(int smCount);

void moeAllToAll(FusedMoeCommKernelParam params, FusedMoeWorkspace workspace, cudaStream_t stream);

void constructWorkspace(FusedMoeWorkspace* workspace, uint64_t* workspacePtr, size_t rankStrideInU64, int epSize);

void initializeFusedMoeLocalWorkspace(FusedMoeWorkspace* workspace, FusedMoeWorldInfo const& worldInfo);

namespace fused_moe_comm_tests
{

// Functions for testing

void launchSingleG2S(FusedMoeFieldInfo const& sendFieldInfo, MoeExpertParallelInfo const& expertParallelInfo,
    int tokenCount, int* shmDump, int warpsPerBlock, bool hasBasicFields, cudaStream_t stream);

void launchSingleS2G(FusedMoeFieldInfo const& recvFieldInfo, MoeExpertParallelInfo const& expertParallelInfo,
    int tokenCount, int* shmPreload, int warpsPerBlock, bool hasBasicFields, cudaStream_t stream);

void launchLoopback(FusedMoeFieldInfo const& sendFieldInfo, FusedMoeFieldInfo const& recvFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int* recvIndexMapping, int tokenCount, int warpsPerBlock,
    bool hasBasicFields, cudaStream_t stream);

void launchLocalFifoSendRecv(FusedMoeFieldInfo const& sendFieldInfo, FusedMoeFieldInfo const& recvFieldInfo,
    MoeExpertParallelInfo const& expertParallelInfo, int* sendIndexMapping, int* recvIndexMapping,
    FusedMoeWorkspace fusedMoeWorkspace, int tokenCount, int warpsPerBlock, int blockChannelCount, bool hasBasicFields,
    cudaStream_t stream);

} // namespace fused_moe_comm_tests

} // namespace kernels
} // namespace tensorrt_llm
