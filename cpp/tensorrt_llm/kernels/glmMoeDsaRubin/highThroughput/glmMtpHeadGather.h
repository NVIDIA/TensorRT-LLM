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
#include "tensorrt_llm/common/tllmDataType.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_rubin_mtp
{

//! Value of every 32-bit word of an unwritten slot in an all-gather workspace. A new workspace is filled with it.
constexpr uint32_t kAllGatherEmptyWord = 0xFFFFFFFFU;

//! \brief Parameters of the one-shot MNNVL all-gather along the last dimension.
struct AllGatherParams
{
    int nRanks;                   //!< Ranks in the group
    int rank;                     //!< This rank
    tensorrt_llm::DataType dType; //!< Input type: kBF16 or kFLOAT
    bool floatOutput;             //!< Write the output as FP32 instead of dType
    int numRows;                  //!< Rows of the input shard
    int shardDim;                 //!< Elements per row of one rank's shard; shardDim * sizeof(dType) % 16 == 0
    void const* input;            //!< This rank's shard [numRows, shardDim], contiguous, 16-byte aligned
    void* output;                 //!< Gathered rows [numRows, nRanks * shardDim], contiguous, 16-byte aligned
    void* bufferPtrLocal;         //!< This rank's unicast view of the workspace (three Lamport buffers)
    void* multicastPtr;           //!< Multicast view of the same workspace
    uint32_t* bufferFlags;        //!< Lamport flags of the workspace
    cudaStream_t stream;
};

//! \brief All-gather of a shard along its last dimension over a dedicated Lamport workspace (one kernel).
//!
//! Every rank stores its shard once through the multicast address at [row][rank][column] of the gathered layout and
//! copies the gathered rows out of its local copy of the buffer as the packets land; an unwritten 32-bit word holds
//! kAllGatherEmptyWord. The output equals the rank-ordered concatenation of the shards bit for bit, except that a
//! payload word equal to kAllGatherEmptyWord (two BF16 or one FP32 NaN with every payload bit set, which no float
//! conversion produces) arrives as kAllGatherEmptyWord - 1, still a NaN. BF16 input widens to FP32 exactly when
//! floatOutput is set. Under PDL the kernel waits for its producer before it touches any memory.
void oneshotAllGatherOp(AllGatherParams const& params);

} // namespace kernels::glm_rubin_mtp

TRTLLM_NAMESPACE_END
