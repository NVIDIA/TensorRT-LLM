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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp8/swigluOai.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp8
{

namespace
{

constexpr int kThreads = 256;

// One packed bf16 pair of gate and up values -> its two bf16 results (the low half from the low halves): the
// instruction sequence of the Triton kernel at limit 7, alpha 1.702, beta 1 (bitwise on SM107): the
// gate clamp as min.bf16x2 before the widening, the up clamp as min.xorsign.abs, (u + 1), (g * -1.702) * log2(e),
// ex2.approx, + 1, div.full 1 / that, s * g, the product, one rounding to bf16.
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

template <bool kPdl>
__global__ void __launch_bounds__(kThreads)
    swigluOaiKernel(uint32_t const* __restrict__ x, uint32_t* __restrict__ out, int64_t rows, int64_t halfD)
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

} // namespace

cudaError_t invokeSwigluOai(
    uint32_t const* x, uint32_t* out, int64_t rows, int64_t halfD, bool pdl, cudaStream_t stream)
{
    int64_t const n = rows * halfD;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads));
    cfg.blockDim = dim3(kThreads);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return pdl ? cudaLaunchKernelEx(&cfg, swigluOaiKernel<true>, x, out, rows, halfD)
               : cudaLaunchKernelEx(&cfg, swigluOaiKernel<false>, x, out, rows, halfD);
}

} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_NAMESPACE_END
