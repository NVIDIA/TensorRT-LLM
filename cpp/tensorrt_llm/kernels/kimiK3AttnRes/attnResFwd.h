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

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3AttnRes
{

//! Parameters for the fused Kimi K3 attention-residual forward kernel
//! (warp-specialised online softmax + residual selection + RMSNorm,
//! Blackwell sm_100/sm_103 only).
//!
//! Contract (checked at the Torch-op bridge): B == 1, N in [1, 12],
//! T in [1, 16384], H a multiple of 1024 in [4096, 8192]; all residual
//! tensors bf16 contiguous, rsigma/probs/logits fp32 [N, T, B].
//! blockResidual may be nullptr when N == 1.
struct AttnResFwdParams
{
    __nv_bfloat16 const* blockResidual; // [N-1, T, B, H], nullptr iff N == 1
    __nv_bfloat16 const* layerResidual; // [T, B, H]
    __nv_bfloat16 const* resWeight;     // [H]
    __nv_bfloat16 const* rmsWeight;     // [H]
    __nv_bfloat16* output;              // [T, B, H]
    float* rsigma;                      // [N, T, B]
    float* probs;                       // [N, T, B]
    float* logits;                      // [N, T, B]
    int numCandidates;                  // N = K + 1
    int seqLen;                         // T
    int batchSize;                      // B
    int hiddenSize;                     // H
    float rmsEps;
};

//! Launches the fused attention-residual forward on the supplied stream.
void invokeAttnResFwd(AttnResFwdParams const& params, cudaStream_t stream);

} // namespace kernels::kimiK3AttnRes

TRTLLM_NAMESPACE_END
