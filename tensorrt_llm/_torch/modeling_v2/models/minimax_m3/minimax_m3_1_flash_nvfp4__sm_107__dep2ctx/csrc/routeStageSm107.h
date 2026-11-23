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

// minimax_m31_route_stage (DEP_ROUTER_STOCK_BOUNDARY's route-only stage op) for DEP2CTX_NORMQ_STAGE:
// the router-fold kernel (addRmsNormSm107 of the served PFA1 image tree) verbatim, without the decode router fold.
#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! DEP_ROUTER_STOCK_BOUNDARY: the routing part of the boundary on the stock router's logits, bit for bit what
//! noaux_tc_fp4_quantize_stage_out (+ stage_appended) writes into the live rows' first topk (+ 1) columns: the
//! single-group noaux top-k ids and weights (the stock expert tier and weight form), -1 / 0 on pad-mask rows, and with
//! appendedSlot >= 0 the slot / 1.0 in column topk. One warp per row; PDL: waits on its predecessor, then releases its
//! dependents. The NVFP4 rows and the padded tail come from the boundary (on dep2ctx
//! minimax_m31_ctx_boundary_normq_out).
struct RouteStageParams
{
    float const* logits;    // [numRows, logitStride] fp32
    float const* bias;      // [numExperts] fp32
    uint8_t const* padMask; // [numRows] or null
    int32_t* topkIndices;   // [>= numRows, >= topk (+ 1)]
    float* topkWeights;     // [>= numRows, >= topk (+ 1)]
    int64_t logitStride;
    int64_t indexStride;
    int64_t weightStride;
    int numRows;
    int numExperts;
    int topk;
    int appendedSlot;
    double routedScalingFactor;
    bool fp32Weight;
};

cudaError_t invokeRouteStage(RouteStageParams const& params, bool pdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
