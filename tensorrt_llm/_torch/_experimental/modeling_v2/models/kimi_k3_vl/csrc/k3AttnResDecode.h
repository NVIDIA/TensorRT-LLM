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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::kimiK3DecodeAttnRes
{

//! Parameters of the Kimi K3 attention-residual kernels (k3AttnResDecode.cu, k3AttnResFwd.cu,
//! k3AttnResPersistent.cu). Each entry point states which fields it reads and its shape contract; the Torch-op
//! bridge (k3AttnResOp.cpp) checks it. All residual tensors are bf16 and contiguous.
struct AttnResFwdParams
{
    __nv_bfloat16 const* blockResidual = nullptr;    // [N-1, T, B, H], nullptr iff N == 1
    __nv_bfloat16 const* layerResidual = nullptr;    // [T, B, H]
    __nv_bfloat16 const* layerResidualAdd = nullptr; // [T, B, H], optional fused addend
    __nv_bfloat16 const* resWeight = nullptr;        // [H]
    __nv_bfloat16 const* rmsWeight = nullptr;        // [H]
    __nv_bfloat16 const* outputRmsWeight = nullptr;  // [H], nullptr unless trailing RMSNorm is fused
    __nv_bfloat16* updatedLayerResidual = nullptr;   // [T, B, H], optional fused-add output
    __nv_bfloat16* output = nullptr;                 // [T, B, H]
    float* rsigma = nullptr;                         // [N, T, B], optional
    float* probs = nullptr;                          // [N, T, B], optional
    float* logits = nullptr;                         // [N, T, B], optional
    int numCandidates = 0;                           // N = K + 1
    int seqLen = 0;                                  // T
    int batchSize = 0;                               // B
    int hiddenSize = 0;                              // H
    float rmsEps = 0.f;
    float outputRmsEps = 0.f;
    //! PDL only: release dependents before the epilogue instead of after the final store.
    bool earlyTrigger = false;
};

//! Selection with the trailing RMSNorm fused: B=1, H=7168, N in [1, 12], one CTA or cluster per token.
void invokeAttnResRmsNormFwd(AttnResFwdParams const& params, cudaStream_t stream);
//! As invokeAttnResRmsNormFwd, with updatedLayerResidual = bf16(layerResidual + layerResidualAdd) as the last
//! candidate.
void invokeAttnResAddRmsNormFwd(AttnResFwdParams const& params, cudaStream_t stream);

//! Unfused selection, writing output and the per-candidate rsigma/probs/logits. B=1, H=7168, N in [1, 12],
//! T in [1, 16384].
void invokeAttnResFwd(AttnResFwdParams const& params, cudaStream_t stream);
//! The single-token kernels of invokeAttnResFwd (k3AttnResDecode.cu): T=1 and N in {1, 2, 4, 8, 12}. Returns false,
//! launching nothing, for any other shape.
bool invokeAttnResSingleTokenFwd(AttnResFwdParams const& params, cudaStream_t stream);

//! Persistent-grid selection with the residual add (optional) and the trailing RMSNorm fused: B=1, H=7168,
//! N in [2, 9], outputRmsWeight required, no rsigma/probs/logits. smReserve SMs are left out of the grid.
void invokeAttnResPersistentFusedFwd(AttnResFwdParams const& params, int smReserve, cudaStream_t stream);
//! True when invokeAttnResPersistentFusedFwd accepts params (shape, required tensors, no in-place add).
bool attnResPersistentFusedSupported(AttnResFwdParams const& params);

}

TRTLLM_MV2_NAMESPACE_END
