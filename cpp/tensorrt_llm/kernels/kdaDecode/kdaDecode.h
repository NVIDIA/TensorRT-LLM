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

//! Select the compact-head kernel only in the GB300 domain measured by the
//! KDA decode sweep. Division keeps the B*H threshold overflow-safe.
constexpr bool shouldUseCompactHeads(int smVersion, int batchSize, int numHeads, int numValueHeads)
{
    return smVersion == 103 && batchSize > 0 && numHeads == numValueHeads && isSupportedHeadCount(numHeads)
        && batchSize <= kCompactHeadsWorkThreshold / numHeads;
}

//! How the decode step's per-token inputs and conv-state pool are addressed.
//!
//! The row strides are the element distance between consecutive batch rows.
//! Each is the packed value (``numHeads * 128``, or ``numHeads`` for beta)
//! when the tensor was materialized for the kernel; passing the strides of a
//! fused in-projection's output instead lets the kernel read its column
//! slices where they already are, with no repacking pass.
struct KdaDecodeIoLayout
{
    int xQRowStride;
    int xKRowStride;
    int xVRowStride;
    int gateRowStride;
    int betaRowStride;
    int outputNormGateRowStride;
    //! Element distance between conv-pool slots. Only read when
    //! ``rollConvPool`` is set.
    int64_t convPoolSlotStride;
    //! Take each request's conv window straight out of the layer's
    //! ``[slots, sections * dim, W]`` pool (``W`` contiguous, addressed by
    //! ``ssmStateIndices``) and store it back rolled forward by this token,
    //! instead of reading a batch-row-dense staged copy. ``convStateQ/K/V``
    //! then point at the pool's three section views.
    bool rollConvPool;
};

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
    KdaDecodeIoLayout layout;
};

//! Launches the tuned KDA decode kernel on the supplied CUDA stream.
void invokeKdaDecode(KdaDecodeParams const& params, cudaStream_t stream);

} // namespace kernels::kdaDecode

TRTLLM_NAMESPACE_END
