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
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::kimiK3Mnnvl
{

struct AllGatherSplitParams
{
    float const* input;
    __nv_bfloat16* bf16Output;
    float* fp32Output;
    int numTokens;
    int bf16Columns;
    int fp32Columns;

    int nRanks;
    int rank;
    void** bufferPtrsDev;
    void* multicastPtr;
    uint32_t* bufferFlags;
    cudaStream_t stream;
};

int64_t mnnvlAllGatherSplitFootprint(int numTokens, int bf16Columns, int fp32Columns, int nRanks);

void mnnvlAllGatherSplitOp(AllGatherSplitParams const& params);

}

TRTLLM_MV2_NAMESPACE_END
