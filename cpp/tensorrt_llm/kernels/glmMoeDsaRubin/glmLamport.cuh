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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/lamportUtils.cuh"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

//! \brief The Lamport flags of an MNNVL workspace, read and rotated as common::LamportFlags does.
//!
//! The flags are nine 32-bit words: [0] the current buffer, [1] the dirty buffer (the previous call's), [2] bytes per
//! buffer, [3] stages of the dirty buffer, [4..7] bytes to clear per stage of the dirty buffer, [8] the arrival
//! counter. A kernel publishes into the current buffer, empties the dirty one, and the thread that rotates the flags
//! makes the current buffer the dirty one once every CTA has counted its arrival. Each CTA counts its arrival with a
//! CTA barrier and a release reduction, as common::LamportFlags does, so kernels of either type follow each other on
//! one workspace. On top of common::LamportFlags: a volatile read of the flags, the arrival counter, the rotation
//! without its wait, and the fill of the dirty buffer with a value other than -0.0.
//!
//! Like common::LamportFlags, the clear and the fill need every thread of a 1D-block grid.
template <typename PackedType = float4>
class GlmLamportFlags
{
public:
    //! \param bufferFlags The nine flag words of the workspace.
    //! \param numStages Stages this kernel uses in the current buffer.
    //! \param volatileRead Read the flags with volatile loads, which do not go through L1: for a kernel that reads them
    //! before its grid dependency wait.
    __device__ GlmLamportFlags(uint32_t* bufferFlags, uint32_t numStages, bool volatileRead)
        : mBufferFlagsPtr(bufferFlags)
        , mFlagAccessPtr(&bufferFlags[kArrivalCounterWord])
    {
        uint4 flag;
        uint4 clear;
        if (volatileRead)
        {
            auto const flagWords = common::loadPackedVolatile<float4>(bufferFlags);
            auto const clearWords = common::loadPackedVolatile<float4>(bufferFlags + 4);
            flag = make_uint4(flagWords.words[0], flagWords.words[1], flagWords.words[2], flagWords.words[3]);
            clear = make_uint4(clearWords.words[0], clearWords.words[1], clearWords.words[2], clearWords.words[3]);
        }
        else
        {
            flag = reinterpret_cast<uint4 const*>(bufferFlags)[0];
            clear = reinterpret_cast<uint4 const*>(bufferFlags)[1];
        }
        mCurrentIndex = flag.x;
        mDirtyIndex = flag.y;
        mCurBufferLayout.numStages = numStages;
        mCurBufferLayout.bytesPerBuffer = flag.z;
        mDirtyBufferLayout.bytesPerBuffer = flag.z;
        mDirtyBufferLayout.numStages = flag.w;
        mBytesToClear[0] = clear.x;
        mBytesToClear[1] = clear.y;
        mBytesToClear[2] = clear.z;
        mBytesToClear[3] = clear.w;
    }

    //! The base of stage stageIdx of the current buffer.
    [[nodiscard]] __device__ void* getCurLamportBuf(void* bufferBasePtr, int stageIdx = 0) const
    {
        return mCurBufferLayout.getStagePtr(bufferBasePtr, mCurrentIndex, stageIdx);
    }

    //! Writes the Lamport empty value (-0.0) into every stage of the dirty buffer.
    __device__ void clearDirtyLamportBuf(void* bufferBasePtr)
    {
        fillDirtyLamportBuf(bufferBasePtr, common::getPackedLamportInit<PackedType, float>());
    }

    //! Writes fillValue into every stage of the dirty buffer, for a workspace whose empty value is not -0.0.
    __device__ void fillDirtyLamportBuf(void* bufferBasePtr, PackedType fillValue)
    {
        uint32_t const globalCtaIdx = blockIdx.x * gridDim.y + blockIdx.y;
        uint32_t const globalTid = globalCtaIdx * blockDim.x + threadIdx.x;
        uint32_t const numThreads = gridDim.x * gridDim.y * blockDim.x;
        for (uint32_t stage = 0; stage < mDirtyBufferLayout.numStages; stage++)
        {
            uint32_t const fillBoundary = (mBytesToClear[stage] + sizeof(PackedType) - 1) / sizeof(PackedType);
            auto* dirty = static_cast<PackedType*>(mDirtyBufferLayout.getStagePtr(bufferBasePtr, mDirtyIndex, stage));
            for (uint32_t packedIdx = globalTid; packedIdx < fillBoundary; packedIdx += numThreads)
            {
                dirty[packedIdx] = fillValue;
            }
        }
    }

    //! Counts this CTA's arrival: a barrier of the CTA's threads (named barrier 1), then a release reduction on the
    //! arrival counter. Every flag read of the CTA precedes it. For kernels of SM 90 and newer.
    __device__ void ctaArrive()
    {
        uint32_t const barrierThreads
            = ((static_cast<uint32_t>(blockDim.x) + common::kWARP_SIZE - 1U) / common::kWARP_SIZE) * common::kWARP_SIZE;
        if (threadIdx.x < common::kWARP_SIZE)
        {
            asm volatile("barrier.cta.sync 1, %0;" ::"r"(barrierThreads) : "memory");
            if (threadIdx.x == 0)
            {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
                asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1) : "memory");
#else
                asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(mFlagAccessPtr), "r"(1) : "memory");
#endif
            }
        }
        else
        {
            asm volatile("barrier.cta.arrive 1, %0;" ::"r"(barrierThreads) : "memory");
        }
    }

    //! The arrival counter that ctaArrive increments.
    [[nodiscard]] __device__ uint32_t const* arrivalCounter() const
    {
        return mFlagAccessPtr;
    }

    //! The rotation of waitAndUpdate without its wait: the current buffer becomes the dirty one. Only for the thread
    //! that waitAndUpdate elects, once it has observed every CTA's arrival with acquire semantics.
    __device__ void update(uint4 bytesToClearPerStage)
    {
        uint4* flagPtr = reinterpret_cast<uint4*>(mBufferFlagsPtr);
        flagPtr[0] = {(mCurrentIndex + 1) % common::LamportBufferLayout::sNumLamportBuffers, mCurrentIndex,
            mCurBufferLayout.bytesPerBuffer, mCurBufferLayout.numStages};
        flagPtr[1] = bytesToClearPerStage;
        *mFlagAccessPtr = 0;
    }

    //! Thread 0 of CTA 0 waits until every CTA has arrived, then rotates the flags.
    __device__ void waitAndUpdate(uint4 bytesToClearPerStage)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        {
            uint32_t const targetCount = gridDim.x * gridDim.y * gridDim.z;
            uint32_t arrivedCount = 0;
            do
            {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(arrivedCount)
                             : "l"(mFlagAccessPtr)
                             : "memory");
            } while (arrivedCount < targetCount);
            update(bytesToClearPerStage);
        }
    }

private:
    static constexpr int kArrivalCounterWord = 8;

    uint32_t* mBufferFlagsPtr;
    uint32_t* mFlagAccessPtr;
    uint32_t mCurrentIndex;
    uint32_t mDirtyIndex;
    uint32_t mBytesToClear[4];
    common::LamportBufferLayout mCurBufferLayout;
    common::LamportBufferLayout mDirtyBufferLayout;
};

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
