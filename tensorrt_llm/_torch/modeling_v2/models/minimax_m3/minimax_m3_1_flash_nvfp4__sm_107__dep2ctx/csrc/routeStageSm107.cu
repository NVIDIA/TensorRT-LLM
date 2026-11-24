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

// routeStageKernel / invokeRouteStage bit for bit as in addRmsNormSm107.cu of the router fold (see routeStageSm107.h).
#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/csrc/routeQuantDevice.cuh"
#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/csrc/routeStageSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cooperative_groups.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{
namespace cg = cooperative_groups;

//! DEP_ROUTER_STOCK_BOUNDARY: one warp per row routes it from the stock router's logits (deepseek_v3_topk_block's
//! non-group branch via route_quant::topkWarp0, as the boundary norm does on the fold's logits).
constexpr int kRouteRowsPerCta = 4;

template <int MaxE, bool Fp32W>
__global__ void __launch_bounds__(32 * kRouteRowsPerCta) routeStageKernel(RouteStageParams p, bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    __shared__ float sSig[kRouteRowsPerCta][MaxE];
    __shared__ float sBiased[kRouteRowsPerCta][MaxE];
    if (pdl)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    int const w = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    int64_t const row = static_cast<int64_t>(blockIdx.x) * kRouteRowsPerCta + w;
    if (row >= p.numRows)
    {
        return;
    }
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    float const* logits = p.logits + row * p.logitStride;
    for (int j = lane; j < p.numExperts; j += 32)
    {
        float const sg = route_quant::sigmoidAccurate(logits[j]);
        sSig[w][j] = sg;
        sBiased[w][j] = sg + static_cast<float>(p.bias[j]);
    }
    __syncwarp();
    int32_t* ids = p.topkIndices + row * p.indexStride;
    float* weights = p.topkWeights + row * p.weightStride;
    if (p.padMask != nullptr && p.padMask[row] != 0)
    {
        for (int k = lane; k < p.topk; k += 32)
        {
            ids[k] = -1;
            weights[k] = 0.F;
        }
    }
    else
    {
        route_quant::topkWarp0<MaxE, Fp32W>(
            warp, sSig[w], sBiased[w], p.numExperts, p.topk, p.routedScalingFactor, ids, weights);
    }
    if (p.appendedSlot >= 0 && lane == 0)
    {
        ids[p.topk] = p.appendedSlot;
        weights[p.topk] = 1.0F;
    }
#endif
}

template <int MaxE, bool Fp32W>
cudaError_t launchRouteStage(RouteStageParams const& p, bool pdl, cudaStream_t stream)
{
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>((p.numRows + kRouteRowsPerCta - 1) / kRouteRowsPerCta));
    cfg.blockDim = dim3(32 * kRouteRowsPerCta);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, routeStageKernel<MaxE, Fp32W>, p, pdl);
}

} // namespace

cudaError_t invokeRouteStage(RouteStageParams const& params, bool pdl, cudaStream_t stream)
{
    if (params.numRows <= 0)
    {
        return cudaSuccess;
    }
    if (params.numRows > 65535 || params.logits == nullptr || params.bias == nullptr || params.topkIndices == nullptr
        || params.topkWeights == nullptr || params.numExperts < 1 || params.numExperts > 256 || params.topk < 1
        || params.topk > 32 || params.topk > params.numExperts || params.logitStride < params.numExperts
        || (params.numRows > 1
            && (params.indexStride < params.topk + (params.appendedSlot >= 0 ? 1 : 0)
                || params.weightStride < params.topk + (params.appendedSlot >= 0 ? 1 : 0))))
    {
        return cudaErrorInvalidValue;
    }
    // The stock route-quant's expert tier (128 up to 128 experts, else 256) and weight form.
    if (params.numExperts <= 128)
    {
        return params.fp32Weight ? launchRouteStage<128, true>(params, pdl, stream)
                                 : launchRouteStage<128, false>(params, pdl, stream);
    }
    return params.fp32Weight ? launchRouteStage<256, true>(params, pdl, stream)
                             : launchRouteStage<256, false>(params, pdl, stream);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
