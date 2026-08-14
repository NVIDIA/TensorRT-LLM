/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeConvertReqIndexToGlobal(int32_t const* reqId, int32_t const* blockTable, int32_t const* tokenIndices,
    int32_t* output, int32_t numTokens, int32_t numTopkTokens, int32_t maxNumBlocksPerReq, int32_t blockSize,
    int32_t strideFactor, int32_t layerId, int64_t btStride0, int64_t btStride1, int64_t tiStride0, int64_t tiStride1,
    int64_t outStride0, int64_t outStride1, cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
