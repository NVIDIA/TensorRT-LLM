/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cuda_runtime.h>
#include <math.h>

namespace fmha
{

// This needs to be aligned with the definition in TRT-LLM
struct Kv_block_array
{
    using PtrType = int32_t;

    // Maximum number of sequences supported by the kv-cache.
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
    void* mPoolPtr;
    // Pointer to block offsets.
    PtrType* mBlockOffsets;

    Kv_block_array() = default;

    Kv_block_array(
        int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t bytesPerBlock, void* poolPtr)
        : mMaxSeqs(batchSize)
        , mMaxBlocksPerSeq(maxBlocksPerSeq)
        , mTokensPerBlock(tokensPerBlock)
        , mBytesPerBlock{bytesPerBlock}
        , mPoolPtr{poolPtr}
        , mBlockOffsets{nullptr}
    {
        float const tokensPerBlockSeqLog2 = log2(mTokensPerBlock);
        mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
