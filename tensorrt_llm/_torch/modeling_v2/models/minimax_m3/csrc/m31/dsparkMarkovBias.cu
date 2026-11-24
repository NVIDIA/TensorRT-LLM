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

// DSPARK_MARKOV_BIAS_CPP (no Triton in served paths): CUDA C++ replacements of the two Triton kernels of
// ops/dspark/dspark_markov_bias.py (see dsparkMarkovBias.h for the contracts). Gated bitwise vs the Triton
// kernels and timed on R1 (prev_embeddings -0.213 us / call (4 rows per CTA),
// widen_add -0.477 us / call (4 warps, 1024 columns per CTA), median over the timed cells). Both launch with the PDL
// attribute: griddepcontrol.wait at entry, the dependents released right after it.

#include "minimax_m3/csrc/m31/dsparkMarkovBias.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

constexpr int kPrevRowsPerCta = 4; // one warp per output row
constexpr int kWidenThreads = 128; // 4 warps
constexpr int kWidenChunk = 1024;  // columns per CTA

__device__ __forceinline__ void pdlEntry()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

__device__ __forceinline__ float addRn(float a, float b)
{
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float bfLo(uint32_t w)
{
    return __uint_as_float(w << 16);
}

__device__ __forceinline__ float bfHi(uint32_t w)
{
    return __uint_as_float(w & 0xFFFF0000u);
}

__global__ void __launch_bounds__(32 * kPrevRowsPerCta) dsparkPrevEmbeddingsKernel(void const* __restrict__ firstPrev,
    bool firstPrevIs64, void const* __restrict__ tokens, bool tokensIs64, uint4 const* __restrict__ w1,
    uint4* __restrict__ out, int32_t B, int32_t K, int64_t V, int32_t R)
{
    pdlEntry();
    int const lane = threadIdx.x & 31;
    int64_t const row = static_cast<int64_t>(blockIdx.x) * kPrevRowsPerCta + (threadIdx.x >> 5);
    if (row >= static_cast<int64_t>(B) * K)
    {
        return;
    }
    int64_t const b = row / K;
    int64_t const k = row - b * K;
    int64_t prev;
    if (k == 0)
    {
        prev = firstPrevIs64 ? static_cast<int64_t const*>(firstPrev)[b]
                             : static_cast<int64_t>(static_cast<int32_t const*>(firstPrev)[b]);
    }
    else
    {
        int64_t const i = b * K + k - 1;
        prev = tokensIs64 ? static_cast<int64_t const*>(tokens)[i]
                          : static_cast<int64_t>(static_cast<int32_t const*>(tokens)[i]);
    }
    bool const valid = prev >= 0 && prev < V;
    int const nv = R / 8; // uint4 per row (8 bf16)
    uint4 const* src = w1 + (valid ? prev : 0) * nv;
    uint4* dst = out + row * nv;
    for (int i = lane; i < nv; i += 32)
    {
        dst[i] = valid ? src[i] : make_uint4(0u, 0u, 0u, 0u);
    }
}

template <int VW>
__device__ __forceinline__ void widenStep(__nv_bfloat16 const* lr, __nv_bfloat16 const* br, float* orow, int64_t c)
{
    if constexpr (VW == 8)
    {
        uint4 const a = *reinterpret_cast<uint4 const*>(lr + c);
        uint4 const b = *reinterpret_cast<uint4 const*>(br + c);
        float4 o0, o1;
        o0.x = addRn(bfLo(a.x), bfLo(b.x));
        o0.y = addRn(bfHi(a.x), bfHi(b.x));
        o0.z = addRn(bfLo(a.y), bfLo(b.y));
        o0.w = addRn(bfHi(a.y), bfHi(b.y));
        o1.x = addRn(bfLo(a.z), bfLo(b.z));
        o1.y = addRn(bfHi(a.z), bfHi(b.z));
        o1.z = addRn(bfLo(a.w), bfLo(b.w));
        o1.w = addRn(bfHi(a.w), bfHi(b.w));
        *reinterpret_cast<float4*>(orow + c) = o0;
        *reinterpret_cast<float4*>(orow + c + 4) = o1;
    }
    else if constexpr (VW == 4)
    {
        uint2 const a = *reinterpret_cast<uint2 const*>(lr + c);
        uint2 const b = *reinterpret_cast<uint2 const*>(br + c);
        float4 o;
        o.x = addRn(bfLo(a.x), bfLo(b.x));
        o.y = addRn(bfHi(a.x), bfHi(b.x));
        o.z = addRn(bfLo(a.y), bfLo(b.y));
        o.w = addRn(bfHi(a.y), bfHi(b.y));
        *reinterpret_cast<float4*>(orow + c) = o;
    }
    else
    {
        orow[c] = addRn(__bfloat162float(lr[c]), __bfloat162float(br[c]));
    }
}

template <int VW>
__global__ void __launch_bounds__(kWidenThreads) dsparkWidenAddKernel(__nv_bfloat16 const* __restrict__ logits,
    int64_t rowStride, __nv_bfloat16 const* __restrict__ bias, float* __restrict__ out, int64_t N)
{
    pdlEntry();
    int64_t const row = blockIdx.x;
    int64_t const c0 = static_cast<int64_t>(blockIdx.y) * kWidenChunk;
    int64_t const c1 = c0 + kWidenChunk < N ? c0 + kWidenChunk : N;
    __nv_bfloat16 const* lr = logits + row * rowStride;
    __nv_bfloat16 const* br = bias + row * N;
    float* orow = out + row * N;
    int64_t const step = static_cast<int64_t>(VW) * blockDim.x;
    int64_t c = c0 + static_cast<int64_t>(VW) * threadIdx.x;
    for (; c + step < c1; c += 2 * step)
    {
        widenStep<VW>(lr, br, orow, c);
        widenStep<VW>(lr, br, orow, c + step);
    }
    if (c < c1)
    {
        widenStep<VW>(lr, br, orow, c);
    }
}

cudaLaunchConfig_t pdlConfig(dim3 grid, dim3 block, cudaStream_t stream, cudaLaunchAttribute* attr)
{
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.stream = stream;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cfg;
}

} // namespace

void invokeDsparkPrevEmbeddings(void const* firstPrev, bool firstPrevIs64, void const* tokens, bool tokensIs64,
    void const* w1, void* out, int32_t B, int32_t K, int64_t V, int32_t R, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(B >= 1 && K >= 1 && V >= 1 && R >= 16 && (R & (R - 1)) == 0 && R <= 1024,
        "dspark prev_embeddings: unsupported sizes");
    TLLM_CHECK_WITH_INFO(!(reinterpret_cast<uintptr_t>(w1) & 15) && !(reinterpret_cast<uintptr_t>(out) & 15),
        "dspark prev_embeddings: w1 / out must be 16-byte aligned");
    int64_t const rows = static_cast<int64_t>(B) * K;
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = pdlConfig(dim3(static_cast<unsigned>((rows + kPrevRowsPerCta - 1) / kPrevRowsPerCta)),
        dim3(32 * kPrevRowsPerCta), stream, attr);
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsparkPrevEmbeddingsKernel, firstPrev, firstPrevIs64, tokens, tokensIs64,
        static_cast<uint4 const*>(w1), static_cast<uint4*>(out), B, K, V, R));
}

void invokeDsparkWidenAdd(void const* logits, int64_t logitsRowStride, void const* bias, float* out, int64_t M,
    int64_t N, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(M >= 1 && N >= 1 && logitsRowStride >= N, "dspark widen_add: unsupported sizes");
    auto const lp = reinterpret_cast<uintptr_t>(logits);
    auto const bp = reinterpret_cast<uintptr_t>(bias);
    auto const op = reinterpret_cast<uintptr_t>(out);
    bool const v8 = !(lp & 15) && !(logitsRowStride & 7) && !(bp & 15) && !(op & 15) && !(N & 7);
    bool const v4 = !(lp & 7) && !(logitsRowStride & 3) && !(bp & 7) && !(op & 15) && !(N & 3);
    dim3 const grid(static_cast<unsigned>(M), static_cast<unsigned>((N + kWidenChunk - 1) / kWidenChunk));
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = pdlConfig(grid, dim3(kWidenThreads), stream, attr);
    auto const* l = static_cast<__nv_bfloat16 const*>(logits);
    auto const* b = static_cast<__nv_bfloat16 const*>(bias);
    if (v8)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsparkWidenAddKernel<8>, l, logitsRowStride, b, out, N));
    }
    else if (v4)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsparkWidenAddKernel<4>, l, logitsRowStride, b, out, N));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsparkWidenAddKernel<1>, l, logitsRowStride, b, out, N));
    }
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
