/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/common.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Relaxed acceptance
struct ExtractRealDraftTokensParam
{
    int curDraftIdx;
    int batchSize;
    int maxDraftLen;
    int maxTotalDraftTokens;
    int maxTopK;
    int numTokensExpandThisLayer;
    int* tokensGatherIdxForDrafterModel;
    int* topKList;
    int* draftTokensIndicesCumsum;
    int64_t* newDraftTokens;
    int64_t* draftTokensBuffer;
};

void invokeExtractRealDraftTokens(ExtractRealDraftTokensParam& params, cudaStream_t const stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
