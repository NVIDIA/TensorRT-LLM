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
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
// Tokens-only FULL sampling for gathered GLM logits. Parameters/RNG arrays are step
// inputs already complete before the preceding model-forward producer. Scratch and
// counters are owned per CUDA stream / graph state and must not be used concurrently.
struct GlmVerifyHeadSampleParams
{
    float const* logits;
    float const* temperatures;
    int32_t const* topK;
    float const* topP;
    float const* minP;
    int64_t const* seed;
    int64_t const* offset;
    bool perRowRng;
    int32_t* rowCounters;
    int32_t* tokens;
    void* workspace;
    int numRows;
    int vocabSize;
    cudaStream_t stream;
};

size_t getGlmVerifyHeadSampleWorkspaceSize(int numRows, int vocabSize);
void invokeGlmVerifyHeadSample(GlmVerifyHeadSampleParams const& params);
} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
