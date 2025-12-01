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

#include <cuda_runtime.h>
#include <stdint.h>

#include "tensorrt_llm/kernels/moeCommKernelsCommon.h"

namespace tensorrt_llm
{
namespace kernels
{

class LL128Proto
{
public:
    static constexpr uint32_t INITIALIZED_VALUE = 0xFFFFFFFFU;

    template <bool USE_FINISH>
    static __device__ __forceinline__ int checkDataReceivedInShm(uint8_t* sharedMemoryBase, uint64_t step,
        int countIn128Bytes, int fifoEntry128ByteIndexBase, int loaded128ByteCount, int laneId)
    {
        // return value should be how many package already been received.
        // 0 means no data received, -1 means has received finish package(should be the very first 128 Byte).
        uint64_t* aligned128BytesShm = reinterpret_cast<uint64_t*>(sharedMemoryBase);
        int totalValidCount = 0;
        for (int idxBase = loaded128ByteCount; idxBase < countIn128Bytes; idxBase += WARP_SIZE)
        {
            int idx = idxBase + laneId;
            bool valid = false;
            bool finish = false;
            if (idx < countIn128Bytes)
            {
                int indexInFifoEntry = fifoEntry128ByteIndexBase + idx;
                uint64_t value
                    = aligned128BytesShm[idx * UINT64_PER_128B_BLOCK + indexInFifoEntry % UINT64_PER_128B_BLOCK];
                if (USE_FINISH)
                {
                    finish = (value == (step & (1ULL << 63ULL)));
                    valid = (value == step) || finish;
                }
                else
                {
                    valid = (value == step);
                }
            }
            __syncwarp();
            unsigned validMask = __ballot_sync(WARP_MASK, valid);
            // here we check valid in order, if previous valid is not true, we ignore the current valid.
            int validCount = (validMask == WARP_MASK) ? WARP_SIZE : (__ffs(~validMask) - 1);
            if (USE_FINISH)
            {
                unsigned finishedMask = __ballot_sync(WARP_MASK, finish);
                // finish should be the very first 128 Byte.
                if (finishedMask & 0x1)
                {
                    return -1;
                }
            }
            totalValidCount += validCount;

            if (validCount != WARP_SIZE)
            {
                break;
            }
        }
        return totalValidCount;
    }

    static __device__ __forceinline__ void protoPack(
        uint8_t* sharedMemoryBase, uint64_t step, int countIn128Bytes, int fifoEntry128ByteIndexBase, int laneId)
    {
        uint64_t* aligned128BytesShm = reinterpret_cast<uint64_t*>(sharedMemoryBase);
        int halfLaneId = laneId % 16;
        int halfIndex = laneId / 16;
        int tailOffsetIn128Bytes = countIn128Bytes + halfIndex;
        // for LL128 15 * 128 Bytes will be packed to 16 * 128 Bytes, each 16 threads is used for one 15 * 128 bytes.
        for (int idxIn128BytesBase = halfIndex * 15; idxIn128BytesBase < countIn128Bytes; idxIn128BytesBase += 30)
        {
            int tailFlagIndexFromFifoEntry = fifoEntry128ByteIndexBase + tailOffsetIn128Bytes;
            int tailFlagInnerIndex = tailFlagIndexFromFifoEntry % UINT64_PER_128B_BLOCK;
            int idxIn128Bytes = idxIn128BytesBase + halfLaneId;
            int idxFromFifoEntry = fifoEntry128ByteIndexBase + idxIn128Bytes;
            uint64_t tailValue = step;
            uint64_t tailInnerIndex = (halfLaneId >= tailFlagInnerIndex) ? halfLaneId + 1 : halfLaneId;
            if (halfLaneId == 15)
            {
                tailInnerIndex = tailFlagInnerIndex;
            }
            int targetTailIndex = tailOffsetIn128Bytes * UINT64_PER_128B_BLOCK + tailInnerIndex;
            if (idxIn128Bytes < countIn128Bytes && halfLaneId < 15)
            {
                int flagIndex = idxIn128Bytes * UINT64_PER_128B_BLOCK + idxFromFifoEntry % UINT64_PER_128B_BLOCK;
                tailValue = aligned128BytesShm[flagIndex];
                aligned128BytesShm[flagIndex] = step;
            }
            aligned128BytesShm[targetTailIndex] = tailValue;
            tailOffsetIn128Bytes += 2;
        }
        __syncwarp();
    }

    static __device__ __forceinline__ void protoUnpack(uint8_t* sharedMemoryBase, uint64_t step, int countIn128Bytes,
        int fifoEntry128ByteIndexBase, int loaded128ByteCount, int laneId)
    {
        uint64_t* aligned128BytesShm = reinterpret_cast<uint64_t*>(sharedMemoryBase);
        int halfLaneId = laneId % 16;
        int halfIndex = laneId / 16;
        int tailOffsetIn128Bytes = countIn128Bytes + halfIndex;
        for (int idxIn128BytesBase = halfIndex * 15; idxIn128BytesBase < countIn128Bytes; idxIn128BytesBase += 30)
        {
            int tailFlagIndexFromFifoEntry = fifoEntry128ByteIndexBase + tailOffsetIn128Bytes;
            int tailFlagInnerIndex = tailFlagIndexFromFifoEntry % UINT64_PER_128B_BLOCK;
            int idxIn128Bytes = idxIn128BytesBase + halfLaneId;
            int idxFromFifoEntry = fifoEntry128ByteIndexBase + idxIn128Bytes;
            uint64_t tailValue = 0;
            int tailInnerIndex = (halfLaneId >= tailFlagInnerIndex) ? halfLaneId + 1 : halfLaneId;
            int targetTailIndex = tailOffsetIn128Bytes * UINT64_PER_128B_BLOCK + tailInnerIndex;
            if (halfLaneId < 15)
            {
                tailValue = aligned128BytesShm[targetTailIndex];
            }
            if (idxIn128Bytes < countIn128Bytes && halfLaneId < 15)
            {
                int flagIndex = idxIn128Bytes * UINT64_PER_128B_BLOCK + idxFromFifoEntry % UINT64_PER_128B_BLOCK;
                aligned128BytesShm[flagIndex] = tailValue;
            }
            tailOffsetIn128Bytes += 2;
        }
        __syncwarp();
    }

    static __device__ __forceinline__ void rearm(
        uint32_t* u32FifoPtr, uint64_t step, int countIn128Bytes, int fifoEntry128ByteIndexBase, int laneId)
    {
        // LL128 don't need rearm
    }

    static __device__ __host__ __forceinline__ int computeProtoTransfer128ByteAlignedSize(
        int compact128ByteSizeBeforeProto)
    {
        // each 15 * 128 byte need one tail 128 byte
        int tail128ByteSize = (compact128ByteSizeBeforeProto + 15 * 128 - 1) / (15 * 128) * 128;
        return compact128ByteSizeBeforeProto + tail128ByteSize;
    }
};

} // namespace kernels
} // namespace tensorrt_llm
