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

//! Shapes of the GDN target-verify kernel: head dims, replay history length and draft step width.
constexpr int kGdnVerifyHeadDim = 128;
constexpr int kGdnVerifyHistory = 16;
constexpr int kGdnVerifyMinTokens = 2;
constexpr int kGdnVerifyMaxTokens = 8;

//! One MTP target-verify step of a gated DeltaNet layer on its cached-replay state: the replay of the step's tokens
//! (history append, or checkpoint commit when the history is full) and the gated RMSNorm of the replay output.
//! All bf16 tensors are passed as raw bits; N requests of T tokens each, HK key heads, HV value heads.
struct GdnVerifyParams
{
    void const* qkv;   //!< [N*T, >= (2*HK + HV)*128] bf16 convolved q|k|v rows (16-byte aligned rows)
    void const* z;     //!< [N*T, HV, 128] bf16 gate, heads packed
    void const* a;     //!< [N*T, HV] bf16 gating input
    void const* b;     //!< [N*T, HV] bf16 beta input, same row stride as a
    int64_t qkvStride; //!< Row strides in elements.
    int64_t zStride;
    int64_t abStride;

    float const* logA;      //!< [HV]
    float const* dtBias;    //!< [HV]
    void const* normWeight; //!< [128] bf16
    float const* fp8Scale;  //!< [1] output scale for an FP8 E4M3 output, nullptr for a bf16 output
    float normEps;

    void* ssmStates;                      //!< [slots, HV, 128, 128] bf16, updated by the checkpoint commit
    int64_t ssmSlotStride;
    int32_t const* stateIndices;          //!< [N] state slots; negative for a padded request
    void* oldU;                           //!< [slots, 2, 16, HV, 128] bf16 history values
    void* oldK;                           //!< [slots, 2, 16, HK, 128] bf16 normalized history keys
    float* oldG;                          //!< [slots, 2, HV, 16] cumulative log-decay
    int32_t const* cacheBufIdx;           //!< [slots] history buffer read this step
    int32_t const* prevNumAcceptedTokens; //!< [slots] history length

    void* out;                            //!< [N*T, HV*128] gated RMSNorm output (FP8 E4M3 or bf16)
    int64_t outStride;                    //!< Row stride in elements.

    int numRequests;
    int numTokens;
    int numKHeads;
    int numVHeads;
};

void invokeGdnVerify(GdnVerifyParams const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
