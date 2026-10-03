/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mnnvl
{

/**
 * \brief Parameters of mnnvlAllGatherSplitOp: a one-shot all-gather over the MNNVL workspace of
 * fp32 rows whose first bf16Columns columns travel, and are gathered, as bf16.
 *
 * Every rank contributes input [numTokens, bf16Columns + fp32Columns] (fp32). On every rank,
 * bf16Output[t, r * bf16Columns + j] = bf16(input_r[t, j]) and
 * fp32Output[t, r * fp32Columns + j] = input_r[t, bf16Columns + j], for every rank r. The bf16
 * rounding is round-to-nearest, as a GEMV storing bf16 would round. -0.0 arrives as +0.0.
 *
 * The kernel follows the one-shot all-reduce's Lamport protocol on the shared workspace (it takes
 * one turn of the buffer rotation) and triggers its dependents as soon as it starts.
 */
struct AllGatherSplitParams
{
    float const* input;        //!< [numTokens, bf16Columns + fp32Columns], this rank's slice
    __nv_bfloat16* bf16Output; //!< [numTokens, nRanks * bf16Columns]
    float* fp32Output;         //!< [numTokens, nRanks * fp32Columns]; unused if fp32Columns == 0
    int numTokens;
    int bf16Columns;           //!< Multiple of 8
    int fp32Columns;           //!< Multiple of 4

    int nRanks;
    int rank;
    void** bufferPtrsDev;
    void* multicastPtr;
    uint32_t* bufferFlags;
    cudaStream_t stream;
};

//! Bytes of one Lamport buffer the all-gather occupies.
int64_t mnnvlAllGatherSplitFootprint(int numTokens, int bf16Columns, int fp32Columns, int nRanks);

void mnnvlAllGatherSplitOp(AllGatherSplitParams const& params);

} // namespace kernels::mnnvl

TRTLLM_NAMESPACE_END
