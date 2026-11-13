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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::minimaxM31
{

// One MegaMoE instance's input staging: expert ids and FP32 routing weights per row (row strides in elements), NVFP4
// activations (fp4StrideWords 32-bit words per row, 8 e2m1 values per word) and linear UE4M3 scale factors (sfStride
// bytes per row). Rows [0, numRows) receive tokens; rows [numRows, indexCapRows) of the ids are set to -1 and rows
// [numRows, weightCapRows) of the weights to 0, the masked tail the MegaMoE dispatch and combine expect.
struct NoAuxTcFp4QuantStage
{
    int32_t* topkIndices;
    float* topkWeights;
    uint32_t* fp4;
    uint8_t* sf;
    int64_t indexStride;
    int64_t weightStride;
    int64_t fp4StrideWords;
    int64_t sfStride;
    int32_t numRows;
    int32_t indexCapRows;
    int32_t weightCapRows;
};

// Single-group noaux_tc routing (at most 256 experts, top-k <= 32, FP32 scores, bias and weights) and NVFP4
// quantization (16-element blocks, global scale globalScale[0]) of numTokens BF16 rows [numTokens, hiddenSize] in one
// launch. Token t goes to stage0 row t when t < stage0.numRows and to stage1 row t - stage0.numRows otherwise; both
// stages' tails are reset in the same launch. A nonzero padMask[t] (padMask may be null; numTokens bytes otherwise)
// gives token t's stage row the tail's ids and weights instead of its routing. Each other written value has the bits
// trtllm::noaux_tc_op and trtllm::fp4_quantize produce. fp32Weight (NOAUX_TC_FP32, default on) computes the routing
// weights in the FP32 form (the FP64 expression's values without FP64 instructions on the common path); false (its
// kill switch) evaluates the FP64 expression for every weight.
void invokeNoAuxTcFp4QuantStage(float const* scores, float const* bias, __nv_bfloat16 const* hiddenStates,
    float const* globalScale, uint8_t const* padMask, NoAuxTcFp4QuantStage const& stage0,
    NoAuxTcFp4QuantStage const& stage1, int64_t numTokens, int64_t numExperts, int64_t topk, int64_t hiddenSize,
    double routedScalingFactor, cudaStream_t stream, bool fp32Weight = true);

// The routing half of invokeNoAuxTcFp4QuantStage for one stage, in one launch: token t < stage.numRows goes to stage
// row t (a nonzero padMask[t] gives it the tail's ids and weights), and the tail rows past stage.numRows are reset.
// stage.fp4 and stage.sf are neither read nor written. Each written value has the bits invokeNoAuxTcFp4QuantStage
// writes for the same stage.
void invokeNoAuxTcRouteStage(float const* scores, float const* bias, uint8_t const* padMask,
    NoAuxTcFp4QuantStage const& stage, int64_t numExperts, int64_t topk, double routedScalingFactor,
    cudaStream_t stream, bool fp32Weight = true);

} // namespace kernels::minimaxM31

TRTLLM_MV2_NAMESPACE_END
