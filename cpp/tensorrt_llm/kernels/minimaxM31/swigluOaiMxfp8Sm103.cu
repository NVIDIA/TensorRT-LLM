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

#include "tensorrt_llm/kernels/minimaxM31/swigluOaiMxfp8Sm103.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <algorithm>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

// swigluOaiSm103.cu's swigluOaiBf16x2, instruction for instruction (the Triton silu_and_mul sequence at limit 7,
// alpha 1.702, beta 1): one packed bf16 pair of gate and up values -> its two bf16 results.
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

//! One row per CTA iteration (rows strided over the grid), one 8-column chunk per thread: the 8 gate and 8 up values
//! (16 B each) -> 8 bf16 activations -> cvt_warp_fp16_to_mxfp8<bf16, 32> over the 4 lanes that share a K32 scale.
template <bool kWriteH>
__global__ void swigluOaiMxfp8Kernel(uint4 const* __restrict__ x, uint64_t* __restrict__ qOut, uint32_t* sfOut,
    uint4* __restrict__ hOut, int rows, int d)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static constexpr int kThreadsPerSf = 32 / CVT_ELTS_PER_THREAD;
    int const colIdx = threadIdx.x;
    int const colThreads = d / CVT_ELTS_PER_THREAD;
    std::optional<int> const batchIdx = 0;
    std::optional<int> const numRows = rows;

    cudaGridDependencySynchronize();

    for (int row = blockIdx.x; row < rows; row += gridDim.x)
    {
        // x row = [gate (d) | up (d)] = 2 * colThreads chunks of 8 bf16.
        int64_t const xRow = static_cast<int64_t>(row) * 2 * colThreads;
        uint4 const gate = x[xRow + colIdx];
        uint4 const up = x[xRow + colThreads + colIdx];
        uint4 h;
        h.x = swigluOaiBf16x2(gate.x, up.x);
        h.y = swigluOaiBf16x2(gate.y, up.y);
        h.z = swigluOaiBf16x2(gate.z, up.z);
        h.w = swigluOaiBf16x2(gate.w, up.w);
        int64_t const outIdx = static_cast<int64_t>(row) * colThreads + colIdx;
        if constexpr (kWriteH)
        {
            hOut[outIdx] = h;
        }
        PackedVec<__nv_bfloat16> quantIn;
        *reinterpret_cast<uint4*>(&quantIn) = h;
        auto sf = cvt_quant_get_sf_out_offset<uint32_t, kThreadsPerSf>(
            batchIdx, row, colIdx, numRows, d / 32, sfOut, QuantizationSFLayout::SWIZZLED);
        qOut[outIdx] = cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32>(quantIn, sf);
    }
    int const paddedRows = (rows + 127) / 128 * 128;
    for (int padRow = rows + blockIdx.x; padRow < paddedRows; padRow += gridDim.x)
    {
        auto sf = cvt_quant_get_sf_out_offset<uint32_t, kThreadsPerSf>(
            batchIdx, padRow, colIdx, numRows, d / 32, sfOut, QuantizationSFLayout::SWIZZLED);
        cvt_store_replicated_sf<1>(sf, 0x00);
    }
    // Every thread's stores are visible before any thread reports PDL completion (as in quantize_with_block_size).
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

} // namespace

cudaError_t invokeSwigluOaiMxfp8(uint4 const* x, uint64_t* qOut, uint32_t* sfOut, uint4* hOut, int64_t rows, int64_t d,
    int multiProcessorCount, bool pdl, cudaStream_t stream)
{
    if (d % 256 != 0 || d > 8192 || rows > (int64_t{1} << 30))
    {
        return cudaErrorInvalidValue;
    }
    if (rows == 0)
    {
        return cudaSuccess;
    }
    int const threads = static_cast<int>(d / CVT_ELTS_PER_THREAD);
    // quantize_with_block_size's grid: one CTA per (padded) row, capped at the resident CTAs.
    int const blocksPerSm = std::max(1, 2048 / threads);
    int64_t const paddedRows = (rows + 127) / 128 * 128;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim
        = dim3(static_cast<unsigned>(std::min<int64_t>(paddedRows, int64_t{multiProcessorCount} * blocksPerSm)));
    cfg.blockDim = dim3(threads);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int const r = static_cast<int>(rows);
    int const dd = static_cast<int>(d);
    return hOut != nullptr ? cudaLaunchKernelEx(&cfg, swigluOaiMxfp8Kernel<true>, x, qOut, sfOut, hOut, r, dd)
                           : cudaLaunchKernelEx(&cfg, swigluOaiMxfp8Kernel<false>, x, qOut, sfOut, hOut, r, dd);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
