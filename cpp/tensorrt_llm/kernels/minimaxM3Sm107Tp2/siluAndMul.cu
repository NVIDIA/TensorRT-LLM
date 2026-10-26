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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp2/siluAndMul.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp2
{

namespace
{

constexpr int kThreads = 128;

// One packed bf16 pair of gate and up values -> its two bf16 results (the low half from the low halves). The
// instruction sequence of the Triton kernel's PTX at its defaults (host/silu/silu_and_mul_plain.ptx): g * -log2(e) as
// one product, ex2.approx, + 1, div.full 1 / that, g * s, (u + 0.0) as an IEEE add, the product, one rounding to bf16.
__device__ __forceinline__ uint32_t siluMulBf16x2(uint32_t gate, uint32_t up)
{
    uint32_t r;
    asm("{\n\t.reg .b32 gl, gh, ul, uh, t0, t1, e0, e1, s0, s1, p0, p1, q0, q1, r0, r1, one, zero;\n\t"
        ".reg .b16 hl, hh;\n\t"
        "mov.b32 {hl, hh}, %1;\n\tcvt.f32.bf16 gl, hl;\n\tcvt.f32.bf16 gh, hh;\n\t"
        "mov.b32 {hl, hh}, %2;\n\tcvt.f32.bf16 ul, hl;\n\tcvt.f32.bf16 uh, hh;\n\t"
        "mul.rn.f32 t0, gl, 0fBFB8AA3B;\n\tmul.rn.f32 t1, gh, 0fBFB8AA3B;\n\t"
        "ex2.approx.f32 e0, t0;\n\tex2.approx.f32 e1, t1;\n\t"
        "mov.b32 one, 0f3F800000;\n\tadd.rn.f32 e0, e0, one;\n\tadd.rn.f32 e1, e1, one;\n\t"
        "div.full.f32 s0, one, e0;\n\tdiv.full.f32 s1, one, e1;\n\t"
        "mul.rn.f32 p0, gl, s0;\n\tmul.rn.f32 p1, gh, s1;\n\t"
        "mov.b32 zero, 0f00000000;\n\tadd.rn.f32 q0, ul, zero;\n\tadd.rn.f32 q1, uh, zero;\n\t"
        "mul.rn.f32 r0, q0, p0;\n\tmul.rn.f32 r1, q1, p1;\n\t"
        "cvt.rn.bf16x2.f32 %0, r1, r0;\n\t}"
        : "=r"(r)
        : "r"(gate), "r"(up));
    return r;
}

// Eight outputs per thread: 16-byte loads of the gate and the up half, one 16-byte store.
__global__ void __launch_bounds__(kThreads)
    siluAndMulKernel(uint4 const* __restrict__ x, uint4* __restrict__ out, int64_t rows, int64_t vecs)
{
    int64_t const row = blockIdx.y;
    int64_t const v = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (row >= rows || v >= vecs)
    {
        return;
    }
    uint4 const g = x[row * 2 * vecs + v];
    uint4 const u = x[row * 2 * vecs + vecs + v];
    uint4 o;
    o.x = siluMulBf16x2(g.x, u.x);
    o.y = siluMulBf16x2(g.y, u.y);
    o.z = siluMulBf16x2(g.z, u.z);
    o.w = siluMulBf16x2(g.w, u.w);
    out[row * vecs + v] = o;
}

} // namespace

cudaError_t invokeSiluAndMul(uint16_t const* x, uint16_t* out, int64_t rows, int64_t d, cudaStream_t stream)
{
    int64_t const vecs = d / 8;
    dim3 const grid(static_cast<unsigned>((vecs + kThreads - 1) / kThreads), static_cast<unsigned>(rows));
    siluAndMulKernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<uint4 const*>(x), reinterpret_cast<uint4*>(out), rows, vecs);
    return cudaGetLastError();
}

} // namespace minimaxM3Sm107Tp2
} // namespace kernels

TRTLLM_NAMESPACE_END
