/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>

namespace tensorrt_llm::kernels
{

// Internal for K and V cache indexing
enum class KVIdxType : int32_t
{
    K_IDX = 0,
    V_IDX = 1
};

// Struct operates on paged kv cache providing
// only the fields necessary for context FMHA
struct KVBlockArrayForContextFMHA
{
    using DataType = KVCacheIndex const;

    // The maximum number of sequences supported by the kv-cache.
    int32_t mMaxSeqs;
    // Max number of blocks per sequence
    int32_t mMaxBlocksPerSeq;
    // Number of tokens. It must be power of 2.
    int32_t mTokensPerBlock;
    // Exponent of number of tokens with base 2.
    // E.g. for mTokensPerBlock 64, mTokensPerBlockLog2 equals to 6
    int32_t mTokensPerBlockLog2;
    // Table maps logical block idx to the data pointer of k/v cache block pool
    // Shape [B, W, 2, M], where 2 is table for K and V,
    // B is current number of sequences
    // W is beam width
    // M is Max number of blocks per sequence

    // Size of KV cache blocks in bytes (H*D*T*sizeof(DataType))
    int32_t mBytesPerBlock;
    // Pointer to beginning of pool.
    void* mPrimaryPoolPtr;
    // Pointer to block offsets.
    DataType* data;

    KVBlockArrayForContextFMHA()
        : mMaxSeqs{0}
        , mMaxBlocksPerSeq{0}
        , mTokensPerBlock{0}
        , mTokensPerBlockLog2{0}
        , mBytesPerBlock{0}
        , mPrimaryPoolPtr{nullptr}
        , data{nullptr}
    {
    }

    KVBlockArrayForContextFMHA(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock,
        int32_t bytesPerToken, void* primaryPoolPtr, DataType* data)
        : mMaxSeqs(batchSize)
        , mMaxBlocksPerSeq(maxBlocksPerSeq)
        , mTokensPerBlock(tokensPerBlock)
        , mBytesPerBlock{tokensPerBlock * bytesPerToken}
        , mPrimaryPoolPtr{primaryPoolPtr}
        , data{data}
    {
        float const tokensPerBlockSeqLog2 = log2(mTokensPerBlock);
        TLLM_CHECK_WITH_INFO(
            ceil(tokensPerBlockSeqLog2) == floor(tokensPerBlockSeqLog2), "tokensPerBlock must be power of 2");
        // NOTE: pointer offset arithmetic offset is performed on int32_t (see this.getRowPtr).
        // If needed, we could do it on uint32_t or even uint64_t, but that might have performance implications
        TLLM_CHECK_WITH_INFO(static_cast<int64_t>(mMaxSeqs - 1) * mMaxBlocksPerSeq * 2 + maxBlocksPerSeq
                <= std::numeric_limits<int32_t>::max(),
            "kv cache is too large for gpt_attention_plugin");
        mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
    }
};

// Struct operates on paged kv cache providing
// functions for accessing blocks of in K and V caches
// and elements inside these blocks
struct KVBlockArray : public KVBlockArrayForContextFMHA
{
    // Pointer to beginning of pool.
    void* mSecondaryPoolPtr;
    // Maximum kv cache length per sequence
    int32_t mMaxAttentionWindow;
    // Number of sink tokens.
    int32_t mSinkTokens;
    // Cyclic cache length.
    int32_t mCyclicCacheLen;
    // Bubble length.
    int32_t mBubbleLen;
    // Enable one more block to save the kv tokens
    bool mEnableOneMoreBlock;

    KVBlockArray()
        : mSecondaryPoolPtr(nullptr)
        , mMaxAttentionWindow{0}
        , mSinkTokens{0}
        , mCyclicCacheLen{0}
        , mBubbleLen{0}
        , mEnableOneMoreBlock{false}
    {
    }

    KVBlockArray(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t bytesPerToken,
        int32_t maxAttentionWindow, int32_t maxAttentionWindowAllLayer, int32_t sinkTokenLen, bool canUseOneMoreBlock,
        void* primaryPoolPtr, void* secondaryPoolPtr, DataType* data)
        : KVBlockArrayForContextFMHA(batchSize, maxBlocksPerSeq, tokensPerBlock, bytesPerToken, primaryPoolPtr, data)
        , mSecondaryPoolPtr{secondaryPoolPtr}
        , mMaxAttentionWindow(maxAttentionWindow)
        , mSinkTokens(sinkTokenLen)
    {
        auto sinkTokensInLastBlock = mSinkTokens % mTokensPerBlock;
        mBubbleLen = sinkTokensInLastBlock == 0 ? 0 : mTokensPerBlock - sinkTokensInLastBlock;
        mEnableOneMoreBlock = (maxBlocksPerSeq - 1) * tokensPerBlock >= maxAttentionWindowAllLayer + mBubbleLen;
        mEnableOneMoreBlock &= canUseOneMoreBlock;
        mCyclicCacheLen = (mEnableOneMoreBlock) ? mMaxAttentionWindow + mTokensPerBlock - mSinkTokens
                                                : mMaxAttentionWindow - mSinkTokens;
    }

    [[nodiscard]] KVBlockArrayForContextFMHA copyKVBlockArrayForContextFMHA() const
    {
        return KVBlockArrayForContextFMHA{
            mMaxSeqs, mMaxBlocksPerSeq, mTokensPerBlock, mBytesPerBlock / mTokensPerBlock, mPrimaryPoolPtr, data};
    }

    __host__ __device__ [[nodiscard]] inline bool isSinkToken(int32_t tokenIdx) const
    {
        return tokenIdx < mSinkTokens;
    }

    __host__ __device__ [[nodiscard]] inline int32_t getKVTokenIdx(int32_t tokenIdx) const
    {
        if (!isSinkToken(tokenIdx))
        {
            // Apply cyclic kv cache if tokenIdx >= max_attention_window_size.
            return mSinkTokens + mBubbleLen + (tokenIdx - mSinkTokens) % mCyclicCacheLen;
        }
        return tokenIdx;
    }

    __host__ __device__ [[nodiscard]] inline DataType const* getRowPtr(KVIdxType kvIdx, int32_t seqIdx) const
    {
        // Returns pointer to array of offsets to K or V cache for one specific sequence seqIdx.
        // seqIdx is in range [0; B]
        return data + (seqIdx * mMaxBlocksPerSeq * 2 + static_cast<int32_t>(kvIdx) * mMaxBlocksPerSeq);
    }

    __host__ __device__ inline void* getBlockPtr(DataType const* offsets, int32_t tokenIdx) const
    {
        auto const offset = offsets[tokenIdx >> mTokensPerBlockLog2];
        return reinterpret_cast<void*>(
            reinterpret_cast<char*>(getPoolPtr(offset)) + offset.get() * static_cast<uint64_t>(mBytesPerBlock));
    }

    __host__ __device__ [[nodiscard]] inline void* getBlockPtr(int32_t seqIdx, int32_t tokenIdx, KVIdxType kvIdx) const
    {
        return getBlockPtr(getRowPtr(kvIdx, seqIdx), tokenIdx);
    }

    __host__ __device__ [[nodiscard]] inline void* getKBlockPtr(int32_t seqIdx, int32_t tokenIdx) const
    {
        return getBlockPtr(seqIdx, tokenIdx, KVIdxType::K_IDX);
    }

    __host__ __device__ [[nodiscard]] inline void* getVBlockPtr(int32_t seqIdx, int32_t tokenIdx) const
    {
        return getBlockPtr(seqIdx, tokenIdx, KVIdxType::V_IDX);
    }

    __host__ __device__ [[nodiscard]] inline int32_t getLocalIdx(int32_t globalIdx) const
    {
        return globalIdx & ((1 << mTokensPerBlockLog2) - 1);
    }

    __host__ __device__ [[nodiscard]] inline int32_t getKVLocalIdx(
        int32_t globalTokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx) const
    {
        // For K or V, the hidden dimension per head is *not* decomposed. The layout of each block of K or V is:
        // [numHeads, tokensPerBlock, hiddenSizePerHead].
        // This member function computes the corresponding linear index.
        // NOTE: we have remapped K layout as the same of V.
        return headIdx * mTokensPerBlock * dimsPerHead + getLocalIdx(globalTokenIdx) * dimsPerHead + channelIdx;
    }

private:
    __host__ __device__ [[nodiscard]] void* getPoolPtr(DataType offset) const
    {
        return offset.isPrimary() ? mPrimaryPoolPtr : mSecondaryPoolPtr;
    }
};

// Struct operates on contiguous kv cache providing
// functions for accessing specific elements in K and V caches
struct KVLinearBuffer
{
    using DataType = int8_t;

    // Current number of sequences
    int32_t mMaxSeqs;
    // Max sequence length
    int32_t mMaxSeqLen;
    // Bytes per sequence (H*D*M_S*sizeof(DataType))
    int32_t mBytesPerSeq;
    // Maximum kv cache length per sequence
    int32_t mMaxAttentionWindow;
    // Number of sink tokens.
    int32_t mSinkTokens;
    // Cyclic cache length.
    int32_t mCyclicCacheLen;
    // Bubble length.
    int32_t mBubbleLen;
    // Valid rows per sequence
    int32_t mValidRowsPerSeq;
    // Enable one more block to save the kv tokens
    bool mEnableOneMoreBlock;
    // Pointer to the of K/V cache data
    // Shape [B, 2, S*H*D], where 2 is for K and V,
    // B is current number of sequences and
    // H is number of heads
    // S is maximum sequence length
    // D is dimension per head
    // K shape is [B, 1, H, S, D]
    // V shape is [B, 1, H, S, D]
    // NOTE: we have remapped K layout as the same of V.
    DataType* data;

    KVLinearBuffer()
        : mMaxSeqs{0}
        , mMaxSeqLen{0}
        , mBytesPerSeq{0}
        , mMaxAttentionWindow{0}
        , mSinkTokens{0}
        , mCyclicCacheLen{0}
        , mBubbleLen{0}
        , mValidRowsPerSeq{0}
        , mEnableOneMoreBlock{false}
        , data{nullptr}
    {
    }

    KVLinearBuffer(int32_t batchSize, int32_t tokensPerBlock, int32_t sizePerToken, int32_t maxAttentionWindow,
        int32_t sinkTokenLen, bool onlyKorV, DataType* data)
        : mMaxSeqs(batchSize)
        , mMaxSeqLen(tokensPerBlock)
        , mBytesPerSeq(tokensPerBlock * sizePerToken)
        , mMaxAttentionWindow(maxAttentionWindow)
        , mSinkTokens(sinkTokenLen)
        , data(data)
    {
        // NOTE: pointer offset arithmetic offset is performed on int32_t (see this.getRowPtr).
        // If needed, we could do it on uint32_t or even uint64_t, but that might have performance implications
        TLLM_CHECK_WITH_INFO(
            static_cast<int64_t>(mMaxSeqs - 1) * mBytesPerSeq * 2 + mBytesPerSeq <= std::numeric_limits<int32_t>::max(),
            "kv cache is too large for gpt_attention_plugin");
        mCyclicCacheLen = mMaxAttentionWindow - mSinkTokens;
        mBubbleLen = 0;
        mValidRowsPerSeq = (onlyKorV) ? 1 : 2;
        mEnableOneMoreBlock = false;
    }

    __host__ __device__ [[nodiscard]] inline bool isSinkToken(int32_t tokenIdx) const
    {
        return tokenIdx < mSinkTokens;
    }

    __host__ __device__ [[nodiscard]] inline void** getRowPtr(KVIdxType kvIdx, int32_t seqIdx) const
    {
        return reinterpret_cast<void**>(data + seqIdx * mBytesPerSeq * mValidRowsPerSeq
            + static_cast<int32_t>(kvIdx) * mBytesPerSeq * (mValidRowsPerSeq - 1));
    }

    __host__ __device__ [[nodiscard]] inline int32_t getKVTokenIdx(int32_t tokenIdx) const
    {
        if (!isSinkToken(tokenIdx))
        {
            // Apply cyclic kv cache if tokenIdx >= max_attention_window_size.
            return mSinkTokens + (tokenIdx - mSinkTokens) % mCyclicCacheLen;
        }
        return tokenIdx;
    }

    __host__ __device__ static inline void* getBlockPtr(void** pointer, int32_t tokenIdx)
    {
        return reinterpret_cast<void*>(pointer);
    }

    __host__ __device__ [[nodiscard]] inline void* getKBlockPtr(int32_t seqIdx, int32_t /*tokenIdx*/) const
    {
        return reinterpret_cast<void*>(getRowPtr(KVIdxType::K_IDX, seqIdx));
    }

    __host__ __device__ [[nodiscard]] inline void* getVBlockPtr(int32_t seqIdx, int32_t /*tokenIdx*/) const
    {
        return reinterpret_cast<void*>(getRowPtr(KVIdxType::V_IDX, seqIdx));
    }

    __host__ __device__ [[nodiscard]] inline int32_t getKVLocalIdx(
        int32_t tokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx) const
    {
        return headIdx * mMaxSeqLen * dimsPerHead + tokenIdx * dimsPerHead + channelIdx;
    }
};

} // namespace tensorrt_llm::kernels
