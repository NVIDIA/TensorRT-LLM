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

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

//! Value of every 32-bit word of an unwritten slot of an exchange workspace; a new workspace is filled with it.
constexpr uint32_t kMnnvlExchangeEmptyWord = 0xFFFFFFFFU;

//! \brief The one-shot MNNVL all-gather of BF16 rows along the last dimension, widened to FP32, on an exchange
//! workspace: three Lamport buffers whose unwritten words hold kMnnvlExchangeEmptyWord, and their flags (the layout
//! of GlmLamportFlags).
struct MnnvlAllgatherParams
{
    int nRanks;            //!< Ranks of the group
    int rank;              //!< This rank
    int numRows;           //!< Rows of the shard
    int shardDim;          //!< BF16 elements per row of one rank's shard, a multiple of 8
    void const* input;     //!< This rank's shard [numRows, shardDim], BF16, contiguous, 16-byte aligned
    float* output;         //!< The gathered rows [numRows, nRanks * shardDim], FP32, contiguous, 16-byte aligned
    void* bufferPtrLocal;  //!< This rank's unicast address of the workspace's three Lamport buffers
    void* multicastPtr;    //!< Multicast address of the same buffers
    uint32_t* bufferFlags; //!< Lamport flags of the workspace
    cudaStream_t stream;
};

//! \brief Gathers every rank's shard into [row][rank][column] order, in one kernel.
//!
//! Every rank stores its shard once through the multicast address at [row][rank][column] of the gathered layout and
//! copies the gathered rows out of its local copy of the buffer as the packets land, widening BF16 to FP32 exactly;
//! it then fills the previous call's buffer with the empty word. The output equals the rank-ordered concatenation of
//! the shards bit for bit, except that a payload word equal to kMnnvlExchangeEmptyWord (a pair of BF16 NaNs with every
//! bit set, which no float conversion produces) arrives as kMnnvlExchangeEmptyWord - 1, still a pair of NaNs. The
//! kernel waits for its producer before it touches any memory, and its PDL dependents launch right after that wait.
void mnnvlAllgather(MnnvlAllgatherParams const& params);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
