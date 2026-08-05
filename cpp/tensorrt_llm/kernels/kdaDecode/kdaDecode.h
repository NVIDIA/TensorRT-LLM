/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kdaDecode
{

constexpr int kCompactHeadsWorkThreshold = 144;

constexpr bool isSupportedHeadCount(int numHeads)
{
    return numHeads == 1 || numHeads == 2 || numHeads == 3 || numHeads == 4 || numHeads == 6 || numHeads == 8
        || numHeads == 12 || numHeads == 16 || numHeads == 24 || numHeads == 32 || numHeads == 48 || numHeads == 96;
}

//! Select the compact-head kernel within the measured KDA decode work threshold.
//! Division keeps the B*H threshold overflow-safe.
constexpr bool shouldUseCompactHeads(int batchSize, int numHeads, int numValueHeads)
{
    return batchSize > 0 && numHeads == numValueHeads && isSupportedHeadCount(numHeads)
        && batchSize <= kCompactHeadsWorkThreshold / numHeads;
}

//! Parameters for the fused, single-token KDA decode kernel.
struct KdaDecodeParams
{
    void const* xQ;
    void const* xK;
    void const* xV;
    void const* wQT;
    void const* wKT;
    void const* wVT;
    void const* biasQ;
    void const* biasK;
    void const* biasV;
    void* convStateQ;
    void* convStateK;
    void* convStateV;
    float const* logA; // Named differently from a_log to satisfy codespell.
    void const* gate;
    float const* dtBias;
    void const* beta;
    void const* outputNormGate;
    float const* outputNormWeight;
    int const* ssmStateIndices;
    //! Must be arange(batchSize + 1): the kernel only advances each state by one token.
    int const* cuSeqlens;
    float* state;
    int64_t stateSlotStride;
    void* output;
    int batchSize;
    int numHeads;
    int numValueHeads;
    bool applyOutputNorm;
    bool updateConvCache;
    bool useLowerBound;
    bool applyBetaSigmoid;
    float lowerBound;
    float scale;
    float outputNormEps;
};

//! Launches the tuned KDA decode kernel on the supplied CUDA stream.
void invokeKdaDecode(KdaDecodeParams const& params, cudaStream_t stream);

} // namespace kernels::kdaDecode

TRTLLM_NAMESPACE_END
