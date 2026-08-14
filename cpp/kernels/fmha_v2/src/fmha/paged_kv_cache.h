/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
