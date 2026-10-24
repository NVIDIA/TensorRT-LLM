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

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

//! How the accepted counts of the generation requests are forced (TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS).
enum class VerifyHeadForceMode : int32_t
{
    kNone = 0,   //!< The strict acceptance count
    kFixed = 1,  //!< forceBase
    kRandom = 2, //!< min(forceBase + (pool[(counter * counterStride + gen * slotStride) & poolMask] < forceFrac),
                 //!< forceMax), counter advanced by one per call
};

//! \brief Inputs and outputs of the fused strict + forced acceptance of one-model speculative decoding.
struct VerifyHeadAcceptParams
{
    void const* targetTokens;   //!< [numContexts + numGens * (draftLen + 1)], int32, or int64 when targetTokensInt64
    bool targetTokensInt64;
    int32_t const* draftTokens; //!< [numGens, draftLen]
    int32_t* numAcceptedTokens; //!< Output [numContexts + numGens]
    int32_t* acceptedTokens;    //!< Output [numContexts + numGens, draftLen + 1], or nullptr: not written
    int numContexts;
    int numGens;
    int draftLen;
    VerifyHeadForceMode forceMode;
    int forceBase;
    int forceMax;
    float forceFrac;
    float const* forcePool; //!< kRandom: fixed uniform pool
    int64_t* forceCounter;  //!< kRandom: call counter
    int64_t counterStride;
    int64_t slotStride;
    int64_t poolMask;
    cudaStream_t stream;
};

//! \brief One kernel for SpecWorkerBase._sample_and_accept_draft_tokens_base after sampling.
//!
//! Context rows accept their sampled token (count 1, accepted row [target, ...]). A generation row accepts its target
//! token plus the leading draft tokens equal to the target tokens (the cumprod count of the strict acceptance), and
//! then the forced count replaces it as forceMode says. Integer results equal the ATen ops bit for bit. Under PDL
//! the kernel waits for its producer before it reads memory and lets its dependents launch at its end.
void invokeVerifyHeadAccept(VerifyHeadAcceptParams const& params);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
