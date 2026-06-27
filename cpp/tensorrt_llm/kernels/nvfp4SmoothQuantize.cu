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
// Fused smooth + NVFP4 quantize: apply the per-input-channel pre_quant_scale AND NVFP4-quantize in
// ONE pass over the input, eliminating the separate x_hat = x*s elementwise pass. A focused reuse of
// trtllm's swizzled-layout quantize (kernels/quantization.cuh quantize_with_block_size) and its exact
// helpers (cvt_warp_fp16_to_fp4, PackedVec, cvt_quant_get_sf_out_offset) so the xq+SF output is
// byte-identical to fp4_quantize(x*s); the only addition is the per-channel multiply before the
// block-amax + quantize.
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__

// quantization.cuh -> reduceKernelUtils.cuh references PackType/TopK which are
// FP8-guarded; define ENABLE_FP8 + pull cudaFp8Utils.h FOR THIS TU ONLY (the
// CUTLASS TUs stay cached). Not used at runtime — only to satisfy the header.
#ifndef ENABLE_FP8
#define ENABLE_FP8
#endif
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/nvfp4SmoothQuantize.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cstdlib>
#include <cuda_bf16.h>

namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::kernels
{
namespace
{
// bf16, NVFP4 (UE4M3 SF, SF_VEC_SIZE=16), swizzled layout, single batch.
constexpr int SF_VEC_SIZE = 16;
using Type = __nv_bfloat16;
constexpr int ELTS_PER_THREAD = tk::CVT_ELTS_PER_THREAD;
using PackedVec = tk::PackedVec<Type>;
constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;

// trtllm's PadUpFn is a function-like MACRO (quantization.h), so `tk::PadUpFn`
// won't qualify; use a plain host+device helper instead.
__host__ __device__ inline int padUp(int x, int y)
{
    return (x + y - 1) / y * y;
}

__global__ void __launch_bounds__(512, 4) smooth_quantize_kernel(int numRows, int numCols, int numPaddedCols,
    Type const* in, Type const* pqs, float const* SFScale, uint32_t* out, uint32_t* SFout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
    auto const layout = tensorrt_llm::QuantizationSFLayout::SWIZZLED;
    int const numPaddedRowsForSf = padUp(numRows, 128);
    int const numColsForSf = padUp(numPaddedCols, 4 * SF_VEC_SIZE);
    int const numColThreads = numCols / ELTS_PER_THREAD;
    int const numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
    int const numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;

    cudaGridDependencySynchronize();
    for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x)
    {
        bool const isRowPadding = (rowIdx >= numRows);
        for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
        {
            std::optional<int> optBatch = 0;
            std::optional<int> optNumRows = numRows;
            auto sf_out = tk::cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                optBatch, rowIdx, colIdx, optNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);

            if (isRowPadding || colIdx >= numColThreads)
            {
                if (sf_out != nullptr)
                    sf_out[0] = 0x00;
                if (!isRowPadding && colIdx >= numColThreads && colIdx < numPaddedColThreads)
                    reinterpret_cast<uint32_t*>(out)[static_cast<int64_t>(rowIdx) * numPaddedColThreads + colIdx] = 0u;
                continue;
            }

            int64_t const inOffset = static_cast<int64_t>(rowIdx) * numColThreads + colIdx;
            int64_t const outOffset = static_cast<int64_t>(rowIdx) * numPaddedColThreads + colIdx;
            PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
            // --- the fusion: smooth by the per-channel pre_quant_scale (broadcast over rows) ---
            PackedVec p_vec = reinterpret_cast<PackedVec const*>(pqs)[colIdx];
#pragma unroll
            for (int i = 0; i < ELTS_PER_THREAD / 2; i++)
                in_vec.elts[i] = __hmul2(in_vec.elts[i], p_vec.elts[i]);
            reinterpret_cast<uint32_t*>(out)[outOffset]
                = tk::cvt_warp_fp16_to_fp4<Type, SF_VEC_SIZE, false>(in_vec, SFScaleVal, sf_out);
        }
    }
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}
} // namespace

void nvfp4_smooth_quantize(void* out, void* sf_out, void const* in, void const* pqs, float const* sf_scale, int m,
    int n, int multiProcessorCount, cudaStream_t stream)
{
    dim3 block(std::min(n / ELTS_PER_THREAD, 512));
    int const numBlocksPerSM = std::max(1, 2048 / int(block.x));
    dim3 grid(std::min(padUp(m, 128), multiProcessorCount * numBlocksPerSM));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    bool const enablePDL = tensorrt_llm::common::getEnvEnablePDL();
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePDL ? 1 : 0;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    // No column padding here (n is the padded width); the residual GEMM and the SF layout use n.
    cudaLaunchKernelEx(&cfg, smooth_quantize_kernel, m, n, n, reinterpret_cast<Type const*>(in),
        reinterpret_cast<Type const*>(pqs), sf_scale, reinterpret_cast<uint32_t*>(out),
        reinterpret_cast<uint32_t*>(sf_out));
}
} // namespace tensorrt_llm::kernels
