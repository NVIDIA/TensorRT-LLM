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
#include "tensorrt_llm/kernels/minimaxM31/s1bFoldSm107.h"
#include "tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingCustomPolicy.cuh"

#include <cooperative_groups.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{
namespace cg = cooperative_groups;
namespace rt = moe::dev::routing;
namespace btg = batchedGemm::trtllm::gen;

using Policy = rt::TopKExpertSelect<rt::SigmoidBiasPreprocess, rt::ScaledSumNormalizePostprocess>;

struct FoldParams
{
    typename Policy::template Params<__nv_bfloat16> mExpertSelectParams;
};

constexpr int kE = 128;
constexpr int kK = 4;
constexpr int kWarps = 8;

template <int S>
__global__ void __launch_bounds__(kWarps * 32) s1bFoldKernel(float const* __restrict__ partials, int32_t M,
    float* __restrict__ logitsOut, int32_t* __restrict__ ids, __nv_bfloat16* __restrict__ weights, FoldParams p)
{
    __shared__ float row[kWarps][kE];
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    int32_t const lane = threadIdx.x & 31;
    int32_t const wid = threadIdx.x >> 5;
    int32_t const m = blockIdx.x * kWarps + wid;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    if (m >= M)
    {
        return;
    }
    // Every slice's value first (all loads in flight), then the slice-order sum (the bits of cuBLASLt's reduce).
    float v[kE / 32][S];
#pragma unroll
    for (int i = 0; i < kE / 32; ++i)
    {
#pragma unroll
        for (int s = 0; s < S; ++s)
        {
            v[i][s] = __ldg(partials + (static_cast<int64_t>(s) * M + m) * kE + i * 32 + lane);
        }
    }
#pragma unroll
    for (int i = 0; i < kE / 32; ++i)
    {
        float acc = v[i][0];
#pragma unroll
        for (int s = 1; s < S; ++s)
        {
            acc = acc + v[i][s];
        }
        row[wid][i * 32 + lane] = acc;
        if (logitsOut != nullptr)
        {
            logitsOut[static_cast<int64_t>(m) * kE + i * 32 + lane] = acc;
        }
    }
    __syncwarp();
    float topScore[8];
    int32_t topIdx[8];
    Policy::template apply<float, float, kE / 32, 8>(warp, topScore, topIdx, lane, kE, kK, &row[wid][0], p);
    if (lane < kK)
    {
        ids[m * kK + lane] = topIdx[lane];
        weights[m * kK + lane] = __nv_bfloat16{topScore[lane]};
    }
}

template <int S>
cudaError_t launch(float const* partials, int32_t M, float* lo, int32_t* ids, __nv_bfloat16* w, FoldParams const& p,
    bool pdl, cudaStream_t stream)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((M + kWarps - 1) / kWarps);
    cfg.blockDim = dim3(kWarps * 32);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, s1bFoldKernel<S>, partials, M, lo, ids, w, p);
}
} // namespace

cudaError_t invokeS1bFold(float const* partials, int32_t S, int32_t M, float const* bias, float routeScale,
    float* logitsOut, int32_t* ids, __nv_bfloat16* weights, bool pdl, cudaStream_t stream)
{
    if (M <= 0)
    {
        return cudaSuccess;
    }
    FoldParams p;
    p.mExpertSelectParams.mPreprocessParams.ptrRoutingBias = bias;
    p.mExpertSelectParams.mPreprocessParams.dtypeBias = btg::Dtype::Fp32;
    p.mExpertSelectParams.mPostprocessParams.ptrRoutingBias = bias;
    p.mExpertSelectParams.mPostprocessParams.dtypeBias = btg::Dtype::Fp32;
    p.mExpertSelectParams.mPostprocessParams.routeScale = routeScale;
    p.mExpertSelectParams.mPostprocessParams.sumEpsilon = 1e-20f;
    if (S == 8)
    {
        return launch<8>(partials, M, logitsOut, ids, weights, p, pdl, stream);
    }
    if (S == 16)
    {
        return launch<16>(partials, M, logitsOut, ids, weights, p, pdl, stream);
    }
    return cudaErrorInvalidValue;
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
