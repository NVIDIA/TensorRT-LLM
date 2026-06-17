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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

enum class MegaMoePrepareExpertType
{
    INT32,
    INT64,
};

enum class MegaMoePrepareScaleType
{
    FP32,
    FP16,
    BF16,
};

void invokeMegaMoePrepare(void const* input, void const* tokenSelectedExperts, void const* tokenFinalScales, void* xOut,
    void* xSfOut, int64_t* topkIdxOut, float* topkWeightsOut, int numTokens, int hiddenSize, int topK,
    MegaMoePrepareExpertType expertType, MegaMoePrepareScaleType scaleType, int multiProcessorCount,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
