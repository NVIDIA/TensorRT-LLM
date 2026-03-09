/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/heuristicTopKDecode.h"

#include "tensorrt_llm/common/config.h"

// Reuse the standalone heuristic kernel verbatim — produces optimal SASS
// with aggressive loop unrolling and good ILP for memory latency hiding.
#include "tensorrt_llm/kernels/heuristic_topk.cuh"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void launchHeuristicTopKDecode(
    float const* logits, int N, int const* preIdx, int preIdxCount, int topK, int* outIndices, cudaStream_t stream)
{
    // heuristicTopKKernel unconditionally writes to outputValues — allocate scratch buffer
    static float* scratchValues = nullptr;
    static int scratchSize = 0;
    if (scratchValues == nullptr || scratchSize < topK)
    {
        if (scratchValues)
        {
            cudaFree(scratchValues);
        }
        cudaMalloc(&scratchValues, topK * sizeof(float));
        scratchSize = topK;
    }

    heuristic_topk::launchHeuristicTopK<float, int>(
        logits, N, preIdx, preIdxCount, topK, scratchValues, outIndices, stream, -1);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
