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

// SwiGLU-OAI with 16-byte vectors: the in-tree TP2 swigluOai.cu pair
// function (swigluOaiBf16x2, copied verbatim: the Triton kernel's instruction sequence, bitwise) applied to 4 packed
// pairs per thread from one 16-byte load of the gate and one of the up half, with a 32-bit row index. Rows whose
// width is not a multiple of 8 bf16 take the in-tree one-pair-per-thread shape.
#include "tensorrt_llm/kernels/minimaxM31/swigluOaiVecSm107.h"

#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{
constexpr int kThreads = 256;

__device__ __forceinline__ uint32_t swigluOaiBf16x2(uint32_t gate, uint32_t up)
{
    uint32_t r;
    asm("{\n\t.reg .b32 g, glim, gl, gh, ul, uh, one, lim, t0, t1, e0, e1, s0, s1, r0, r1;\n\t.reg .b16 hl, hh;\n\t"
        "mov.b32 glim, 0x40E040E0;\n\tmin.bf16x2 g, %1, glim;\n\t"
        "mov.b32 {hl, hh}, g;\n\tcvt.f32.bf16 gl, hl;\n\tcvt.f32.bf16 gh, hh;\n\t"
        "mov.b32 {hl, hh}, %2;\n\tcvt.f32.bf16 ul, hl;\n\tcvt.f32.bf16 uh, hh;\n\t"
        "mov.b32 lim, 0f40E00000;\n\tmin.xorsign.abs.f32 ul, ul, lim;\n\tmin.xorsign.abs.f32 uh, uh, lim;\n\t"
        "mov.b32 one, 0f3F800000;\n\tadd.rn.f32 ul, ul, one;\n\tadd.rn.f32 uh, uh, one;\n\t"
        "mul.rn.f32 t0, gl, 0fBFD9DB23;\n\tmul.rn.f32 t1, gh, 0fBFD9DB23;\n\t"
        "mul.rn.f32 t0, t0, 0f3FB8AA3B;\n\tmul.rn.f32 t1, t1, 0f3FB8AA3B;\n\t"
        "ex2.approx.f32 e0, t0;\n\tex2.approx.f32 e1, t1;\n\t"
        "add.rn.f32 e0, e0, one;\n\tadd.rn.f32 e1, e1, one;\n\t"
        "div.full.f32 s0, one, e0;\n\tdiv.full.f32 s1, one, e1;\n\t"
        "mul.rn.f32 r0, s0, gl;\n\tmul.rn.f32 r1, s1, gh;\n\t"
        "mul.rn.f32 r0, ul, r0;\n\tmul.rn.f32 r1, uh, r1;\n\t"
        "cvt.rn.bf16x2.f32 %0, r1, r0;\n\t}"
        : "=r"(r)
        : "r"(gate), "r"(up));
    return r;
}

template <bool kPdl, int NV>
__global__ void __launch_bounds__(kThreads) swigluOaiVecKernel(
    uint4 const* __restrict__ x, uint4* __restrict__ out, uint32_t rows, uint32_t vecs, uint32_t span)
{
    if constexpr (kPdl)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;");
    }
    // NV output vectors per thread, span apart (span = ceil(rows * vecs / NV)); all loads issued first.
    uint32_t const i0 = blockIdx.x * kThreads + threadIdx.x;
    uint32_t const total = rows * vecs;
    uint4 g[NV], u[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v)
    {
        uint32_t const i = i0 + v * span;
        if (i0 < span && i < total)
        {
            uint32_t const row = i / vecs;
            uint32_t const col = i - row * vecs;
            uint4 const* xr = x + static_cast<size_t>(row) * 2 * vecs;
            g[v] = xr[col];
            u[v] = xr[vecs + col];
        }
    }
#pragma unroll
    for (int v = 0; v < NV; ++v)
    {
        uint32_t const i = i0 + v * span;
        if (i0 < span && i < total)
        {
            uint4 r;
            r.x = swigluOaiBf16x2(g[v].x, u[v].x);
            r.y = swigluOaiBf16x2(g[v].y, u[v].y);
            r.z = swigluOaiBf16x2(g[v].z, u[v].z);
            r.w = swigluOaiBf16x2(g[v].w, u[v].w);
            out[i] = r;
        }
    }
}

template <bool kPdl>
__global__ void __launch_bounds__(kThreads)
    swigluOaiPairKernel(uint32_t const* __restrict__ x, uint32_t* __restrict__ out, int64_t rows, int64_t halfD)
{
    if constexpr (kPdl)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;");
    }
    int64_t const i = blockIdx.x * static_cast<int64_t>(kThreads) + threadIdx.x;
    if (i >= rows * halfD)
    {
        return;
    }
    int64_t const row = i / halfD;
    int64_t const col = i - row * halfD;
    uint32_t const* xr = x + row * 2 * halfD;
    out[i] = swigluOaiBf16x2(xr[col], xr[halfD + col]);
}

template <typename K, typename... A>
cudaError_t launch(K kernel, int64_t n, bool pdl, cudaStream_t st, A... args)
{
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads));
    cfg.blockDim = dim3(kThreads);
    cfg.stream = st;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args...);
}
} // namespace

cudaError_t invokeSwigluOaiVec(void const* x, void* out, int64_t rows, int64_t d, int nv, bool pdl, cudaStream_t stream)
{
    if (rows == 0)
    {
        return cudaSuccess;
    }
    if (d % 2 != 0)
    {
        return cudaErrorInvalidValue;
    }
    bool const vec = d % 8 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0 && rows * (d / 8) < (int64_t{1} << 32);
    if (vec)
    {
        if (nv != 1 && nv != 2 && nv != 4)
        {
            return cudaErrorInvalidValue;
        }
        auto const* xp = reinterpret_cast<uint4 const*>(x);
        auto* op = reinterpret_cast<uint4*>(out);
        uint32_t const r = static_cast<uint32_t>(rows), v = static_cast<uint32_t>(d / 8);
        int64_t const total = rows * (d / 8);
        uint32_t const span = static_cast<uint32_t>((total + nv - 1) / nv);
        if (nv == 1)
            return pdl ? launch(swigluOaiVecKernel<true, 1>, span, true, stream, xp, op, r, v, span)
                       : launch(swigluOaiVecKernel<false, 1>, span, false, stream, xp, op, r, v, span);
        if (nv == 2)
            return pdl ? launch(swigluOaiVecKernel<true, 2>, span, true, stream, xp, op, r, v, span)
                       : launch(swigluOaiVecKernel<false, 2>, span, false, stream, xp, op, r, v, span);
        return pdl ? launch(swigluOaiVecKernel<true, 4>, span, true, stream, xp, op, r, v, span)
                   : launch(swigluOaiVecKernel<false, 4>, span, false, stream, xp, op, r, v, span);
    }
    auto const* xp = reinterpret_cast<uint32_t const*>(x);
    auto* op = reinterpret_cast<uint32_t*>(out);
    return pdl ? launch(swigluOaiPairKernel<true>, rows * (d / 2), true, stream, xp, op, rows, d / 2)
               : launch(swigluOaiPairKernel<false>, rows * (d / 2), false, stream, xp, op, rows, d / 2);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
