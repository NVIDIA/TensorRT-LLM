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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{
// bf16, NVFP4 (UE4M3 SF, SF_VEC_SIZE=16), swizzled layout, single batch.
constexpr int SF_VEC_SIZE = 16;
using Type = __nv_bfloat16;
constexpr int ELTS_PER_THREAD = CVT_ELTS_PER_THREAD;
using SmoothPackedVec = PackedVec<Type>;
constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
constexpr int FAST_ELTS_PER_THREAD = SF_VEC_SIZE;

// Two of these make one complete 16-element NVFP4 scale block. Keeping the load granularity at
// 128 bits avoids imposing a stronger alignment requirement than the stock quantizer.
union alignas(16) Bf16x8
{
    uint4 bits;
    __nv_bfloat162 elts[4];
};
static_assert(sizeof(Bf16x8) == 16);

// trtllm's PadUpFn is a function-like macro (quantization.h); use a plain
// host+device helper here instead.
__host__ __device__ inline int padUp(int x, int y)
{
    return (x + y - 1) / y * y;
}

int getIntegerEnv(char const* name)
{
    char const* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return -1;

    char* end = nullptr;
    long const parsed = std::strtol(value, &end, 10);
    if (end == value || end[0] != '\0' || parsed < -1 || parsed > 4096)
        return -1;
    return static_cast<int>(parsed);
}

int getSmoothQuantThreads()
{
    // Cached intentionally: tuning runs use separate processes, while production calls avoid a getenv per layer.
    static int const value = getIntegerEnv("TRTLLM_NVFP4_SMOOTH_QUANT_THREADS");
    return value;
}

int getSmoothQuantBlocksPerSm()
{
    static int const value = getIntegerEnv("TRTLLM_NVFP4_SMOOTH_QUANT_BLOCKS_PER_SM");
    return value;
}

__device__ __forceinline__ void loadBf16x8(Type const* ptr, Bf16x8& result)
{
    result.bits = *reinterpret_cast<uint4 const*>(ptr);
}

__device__ __forceinline__ uint64_t quantizeSmoothed16(
    Bf16x8& lo, Bf16x8& hi, Bf16x8 const& pqsLo, Bf16x8 const& pqsHi, float SFScaleVal, uint8_t* SFout)
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        lo.elts[i] = __hmul2(lo.elts[i], pqsLo.elts[i]);
        hi.elts[i] = __hmul2(hi.elts[i], pqsHi.elts[i]);
    }

    // Match the legacy even lane's reduction order: reduce each 8-element half independently, then merge
    // the high half into the low half. For finite BF16 values this produces the same scale for both halves.
    auto loMax = cuda_abs(lo.elts[0]);
    auto hiMax = cuda_abs(hi.elts[0]);
#pragma unroll
    for (int i = 1; i < 4; ++i)
    {
        loMax = cuda_max(loMax, cuda_abs(lo.elts[i]));
        hiMax = cuda_max(hiMax, cuda_abs(hi.elts[i]));
    }
    auto const localMax = cuda_max(hiMax, loMax);
    float const vecMax = float(cuda_max(localMax.x, localMax.y));

    // This is deliberately kept instruction-for-instruction equivalent to cvt_warp_fp16_to_fp4's UE4M3 path.
    auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    uint8_t const fp8SFVal = tmp.__x;
    SFValue = static_cast<float>(tmp);
    float const outputScale
        = vecMax != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;

    if (SFout != nullptr)
        *SFout = fp8SFVal;

    float2 fp2Vals[FAST_ELTS_PER_THREAD / 2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        fp2Vals[i] = __bfloat1622float2(lo.elts[i]);
        fp2Vals[i + 4] = __bfloat1622float2(hi.elts[i]);
        fp2Vals[i].x *= outputScale;
        fp2Vals[i].y *= outputScale;
        fp2Vals[i + 4].x *= outputScale;
        fp2Vals[i + 4].y *= outputScale;
    }
    return fp32_vec_to_e2m1(fp2Vals);
}

template <int NumCols, int RowsPerCta>
__global__ void __launch_bounds__(512, 4) smooth_quantize_fast_kernel(int numRows, Type const* __restrict__ in,
    Type const* __restrict__ pqs, float const* __restrict__ SFScale, uint64_t* __restrict__ out,
    uint8_t* __restrict__ SFout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(NumCols % (4 * SF_VEC_SIZE) == 0);
    constexpr int NumSfCols = NumCols / SF_VEC_SIZE;
    constexpr int NumSfGroups = NumSfCols / 4;
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

    cudaGridDependencySynchronize();

    // Hot path: map a fixed number of complete rows onto each CTA. K=3072 uses two rows and 384
    // threads, so every thread owns exactly one 16-value scale block without a warp shuffle.
    for (int rowBase = blockIdx.x * RowsPerCta; rowBase < numRows; rowBase += gridDim.x * RowsPerCta)
    {
        int const rowsRemaining = numRows - rowBase;
        int const rowsThisCta = rowsRemaining < RowsPerCta ? rowsRemaining : RowsPerCta;
        int const workItems = rowsThisCta * NumSfCols;
        for (int item = threadIdx.x; item < workItems; item += blockDim.x)
        {
            int const rowOffset = item / NumSfCols;
            int const sfCol = item - rowOffset * NumSfCols;
            int const row = rowBase + rowOffset;
            int64_t const vecOffset = static_cast<int64_t>(row) * NumSfCols + sfCol;

            Type const* xPtr = in + vecOffset * FAST_ELTS_PER_THREAD;
            Type const* pqsPtr = pqs + sfCol * FAST_ELTS_PER_THREAD;
            Bf16x8 xLo;
            Bf16x8 xHi;
            Bf16x8 pqsLo;
            Bf16x8 pqsHi;
            loadBf16x8(xPtr, xLo);
            loadBf16x8(xPtr + 8, xHi);
            loadBf16x8(pqsPtr, pqsLo);
            loadBf16x8(pqsPtr + 8, pqsHi);

            int64_t const sfOffset
                = get_sf_out_offset_128x4(std::nullopt, row, sfCol, std::nullopt, NumSfCols);
            out[vecOffset] = quantizeSmoothed16(xLo, xHi, pqsLo, pqsHi, SFScaleVal, SFout + sfOffset);
        }
    }

    // Cold path: only scale factors have padded rows. Four consecutive SF columns are contiguous in
    // the 128x4 layout, so initialize them with one aligned 32-bit store instead of four byte stores.
    int const numPaddedRows = padUp(numRows, 128);
    int64_t const numPaddingStores = static_cast<int64_t>(numPaddedRows - numRows) * NumSfGroups;
    for (int64_t item = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; item < numPaddingStores;
         item += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        int const paddingRow = static_cast<int>(item / NumSfGroups);
        int const sfGroup = static_cast<int>(item - static_cast<int64_t>(paddingRow) * NumSfGroups);
        int const row = numRows + paddingRow;
        int const sfCol = sfGroup * 4;
        int64_t const sfOffset = get_sf_out_offset_128x4(std::nullopt, row, sfCol, std::nullopt, NumSfCols);
        *reinterpret_cast<uint32_t*>(SFout + sfOffset) = 0u;
    }

    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__global__ void __launch_bounds__(512, 4) smooth_quantize_legacy_kernel(int numRows, int numCols, int numPaddedCols,
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
            auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
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
            SmoothPackedVec in_vec = reinterpret_cast<SmoothPackedVec const*>(in)[inOffset];
            // --- the fusion: smooth by the per-channel pre_quant_scale (broadcast over rows) ---
            SmoothPackedVec p_vec = reinterpret_cast<SmoothPackedVec const*>(pqs)[colIdx];
#pragma unroll
            for (int i = 0; i < ELTS_PER_THREAD / 2; i++)
                in_vec.elts[i] = __hmul2(in_vec.elts[i], p_vec.elts[i]);
            reinterpret_cast<uint32_t*>(out)[outOffset]
                = cvt_warp_fp16_to_fp4<Type, SF_VEC_SIZE, false>(in_vec, SFScaleVal, sf_out);
        }
    }
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int NumCols, int RowsPerCta>
void launchSmoothQuantizeFast(void* out, void* sfOut, void const* in, void const* pqs, float const* sfScale,
    int numRows, int multiProcessorCount, int blockThreads, int blocksPerSm, bool enablePDL, cudaStream_t stream)
{
    constexpr int NumSfGroups = NumCols / (4 * SF_VEC_SIZE);
    int const numPaddedRows = padUp(numRows, 128);
    int64_t const hotCtas = (static_cast<int64_t>(numRows) + RowsPerCta - 1) / RowsPerCta;
    int64_t const numPaddingStores = static_cast<int64_t>(numPaddedRows - numRows) * NumSfGroups;
    int64_t const paddingCtas = (numPaddingStores + blockThreads - 1) / blockThreads;
    int64_t const wantedCtas = std::max(hotCtas, paddingCtas);
    int64_t const maxCtas = static_cast<int64_t>(multiProcessorCount) * blocksPerSm;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned int>(std::min(wantedCtas, maxCtas)));
    cfg.blockDim = dim3(blockThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePDL ? 1 : 0;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    auto* kernel = &smooth_quantize_fast_kernel<NumCols, RowsPerCta>;
    cudaLaunchKernelEx(&cfg, kernel, numRows, reinterpret_cast<Type const*>(in), reinterpret_cast<Type const*>(pqs),
        sfScale, reinterpret_cast<uint64_t*>(out), reinterpret_cast<uint8_t*>(sfOut));
}
} // namespace

void nvfp4_smooth_quantize(void* out, void* sf_out, void const* in, void const* pqs, float const* sf_scale, int m,
    int n, int multiProcessorCount, cudaStream_t stream)
{
    if (m == 0)
        return;

    bool const enablePDL = tensorrt_llm::common::getEnvEnablePDL();
    int const requestedThreads = getSmoothQuantThreads();
    bool const useFastPath = requestedThreads != 0 && (n == 3072 || n == 12288);
    if (useFastPath)
    {
        // Same-node SM100 sweeps over the Qwen image-token M values select 192 threads for K=3072
        // and 256 for K=12288. A grid cap of eight CTAs per SM is best for both.
        int const defaultThreads = n == 3072 ? 192 : 256;
        int const blockThreads = requestedThreads >= 32 && requestedThreads <= 512 && requestedThreads % 32 == 0
            ? requestedThreads
            : defaultThreads;
        int const requestedBlocksPerSm = getSmoothQuantBlocksPerSm();
        int const defaultBlocksPerSm = 8;
        int const blocksPerSm
            = requestedBlocksPerSm > 0 ? std::min(requestedBlocksPerSm, 32) : defaultBlocksPerSm;

        if (n == 3072)
            launchSmoothQuantizeFast<3072, 2>(out, sf_out, in, pqs, sf_scale, m, multiProcessorCount, blockThreads,
                blocksPerSm, enablePDL, stream);
        else
            launchSmoothQuantizeFast<12288, 1>(out, sf_out, in, pqs, sf_scale, m, multiProcessorCount, blockThreads,
                blocksPerSm, enablePDL, stream);
        return;
    }

    dim3 block(std::min(n / ELTS_PER_THREAD, 512));
    int const numBlocksPerSM = std::max(1, 2048 / int(block.x));
    dim3 grid(std::min(padUp(m, 128), multiProcessorCount * numBlocksPerSM));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePDL ? 1 : 0;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    // No column padding here (n is the padded width); the residual GEMM and the SF layout use n.
    cudaLaunchKernelEx(&cfg, smooth_quantize_legacy_kernel, m, n, n, reinterpret_cast<Type const*>(in),
        reinterpret_cast<Type const*>(pqs), sf_scale, reinterpret_cast<uint32_t*>(out),
        reinterpret_cast<uint32_t*>(sf_out));
}
} // namespace kernels

TRTLLM_NAMESPACE_END
