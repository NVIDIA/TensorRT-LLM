/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fusedLayerNormQuant.cuh"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <cmath>
#include <cstdint>
#include <optional>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Kernel design constants (V1: hardcoded for Wan 2.2 hidden_size = 5120).
// To support a different N, make ELEMS_PER_THREAD a template parameter and
// dispatch at the launcher based on params.N.
static constexpr int kN_HARDCODED = 5120;
static constexpr int ELTS_PER_VEC = 8;
static constexpr int NUM_THREADS_PER_SF
    = LN_FP4_BLOCK_SIZE / ELTS_PER_VEC; // 2 (pairs of threads share one 16-elem SF block)

/*
 * Inline FP32 -> FP4 (e2m1) quantization for 8 floats.
 *
 * Adapted from the helper in fusedGatedRMSNormQuant.cu. Duplicated here
 * intentionally because the original is static; a future refactor can
 * lift it to a shared header in tensorrt_llm/kernels/quantization.cuh.
 *
 * Algorithm:
 *   1. Find max-abs across the 8 local values.
 *   2. Combine with the next thread via warp shuffle to get the 16-element
 *      block max-abs (this thread + thread XOR-1 cover one SF block).
 *   3. Compute the FP8 (e4m3) scale factor and write it.
 *   4. Scale the 8 floats by the inverse and convert to packed FP4 (e2m1).
 */
__device__ __forceinline__ uint32_t cvt_float_to_fp4_inline(float* vals, // 8 float values to quantize (read+modified)
    float sfScaleVal,  // global activation scale (calibrated module.input_scale)
    uint8_t* sfOutPtr) // output position for the 1-byte FP8 scale factor
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // 1. Local max-abs across 8 values held by this thread.
    float localMax = fabsf(vals[0]);
#pragma unroll
    for (int i = 1; i < 8; ++i)
    {
        localMax = fmaxf(localMax, fabsf(vals[i]));
    }

    // 2. Pair with thread XOR-1 (next lane) so the resulting localMax covers
    //    16 consecutive elements -- exactly one NVFP4 SF block.
    localMax = fmaxf(__shfl_xor_sync(0xffffffff, localMax, 1), localMax);

    // 3. Compute SF: scale_factor = sfScaleVal * (max_abs / 6.0)
    //    where 6.0 is the largest representable e2m1 value.
    float sfValue = sfScaleVal * (localMax * reciprocal_approximate_ftz(LN_FP4_E2M1_MAX));

    __nv_fp8_e4m3 sfFp8 = __nv_fp8_e4m3(sfValue);
    uint8_t sfByte = sfFp8.__x;
    float sfValueQuant = static_cast<float>(sfFp8);

    // Output scale used to map each float into [-E2M1_MAX, E2M1_MAX].
    float outputScale
        = (localMax != 0.0f) ? reciprocal_approximate_ftz(sfValueQuant * reciprocal_approximate_ftz(sfScaleVal)) : 0.0f;

    if (sfOutPtr)
    {
        *sfOutPtr = sfByte;
    }

    // 4. Scale 8 floats and convert to packed e2m1 (returns one uint32 = 8 FP4 nibbles).
    float scaledVals[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        scaledVals[i] = vals[i] * outputScale;
    }
    return fp32_vec_to_e2m1(scaledVals);
#else
    return 0;
#endif
}

/*
 * Fused LayerNorm + NVFP4 Quantization kernel.
 *
 * Grid: (M,) - one block per input row.
 * Block: BLOCK_SIZE threads (default 128).
 *
 * Each thread holds ELEMS_PER_THREAD = N / BLOCK_SIZE floats (= 40 for the
 * Wan 2.2 N=5120 / BLOCK_SIZE=128 default) across CHUNKS_PER_THREAD vector
 * loads (= 5 chunks of 8 elements each).
 *
 * Thread t handles vector indices t, t + BLOCK_SIZE, t + 2*BLOCK_SIZE, ...
 * so that adjacent threads cover adjacent 8-element vectors. Threads (2k,
 * 2k+1) therefore cover one 16-element SF block, which is what
 * cvt_float_to_fp4_inline relies on for its XOR-1 shuffle.
 *
 * Memory traffic (per row):
 *   read  : x (N elts), plus N elts of {ln_weight,ln_bias} or {scale_msa,shift_msa}
 *   write : y_fp4 (N/2 bytes) + sf_out (N/16 bytes)
 * vs unfused which would round-trip x through HBM ~6 times. This kernel
 * stores all per-row values in registers between phases.
 */
template <typename T, int BLOCK_SIZE, bool HAS_LN_AFFINE, bool HAS_MODULATION>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(BLOCK_SIZE, 4)
#endif
    fusedLayerNormQuantKernel(T const* __restrict__ x, T const* __restrict__ ln_weight, T const* __restrict__ ln_bias,
        T const* __restrict__ scale_msa, T const* __restrict__ shift_msa, uint32_t* __restrict__ y_fp4,
        uint32_t* __restrict__ sf_out, float const* __restrict__ sf_scale, int M, int seq_len_per_batch, float eps)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    using T2 = typename packed_as<T, 2>::type;

    static_assert(kN_HARDCODED % BLOCK_SIZE == 0, "N must be divisible by BLOCK_SIZE");
    static_assert(
        (kN_HARDCODED / BLOCK_SIZE) % ELTS_PER_VEC == 0, "elements per thread must be a multiple of ELTS_PER_VEC (8)");
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size");
    static_assert(BLOCK_SIZE / 32 <= 32, "block-level reduction requires numWarps <= 32");

    constexpr int ELEMS_PER_THREAD = kN_HARDCODED / BLOCK_SIZE;        // 40 for default
    constexpr int CHUNKS_PER_THREAD = ELEMS_PER_THREAD / ELTS_PER_VEC; // 5  for default
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;                         // 4  for default

    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    int const warpId = tid / 32;
    int const laneId = tid % 32;

    float const invN = 1.0f / static_cast<float>(kN_HARDCODED);
    float const sfScaleVal = (sf_scale != nullptr) ? sf_scale[0] : 1.0f;
    int const numSfVecsTotal = kN_HARDCODED / LN_FP4_BLOCK_SIZE;

    // Shared memory for two cross-warp reductions (sum, sumSq) plus the
    // final broadcast of (mean, rstd) to every thread.
    __shared__ float warpSums[NUM_WARPS];
    __shared__ float warpSqSums[NUM_WARPS];
    __shared__ float meanRstd[2];

    T const* xRow = x + static_cast<int64_t>(row) * kN_HARDCODED;

    // Compute modulation row pointers (only used when HAS_MODULATION).
    // scale_msa / shift_msa are shaped [B, N]; all rows belonging to the
    // same batch element share the same modulation vector.
    T const* scale_msa_row = nullptr;
    T const* shift_msa_row = nullptr;
    if constexpr (HAS_MODULATION)
    {
        int batch_idx = row / seq_len_per_batch;
        scale_msa_row = scale_msa + static_cast<int64_t>(batch_idx) * kN_HARDCODED;
        shift_msa_row = shift_msa + static_cast<int64_t>(batch_idx) * kN_HARDCODED;
    }

    // -----------------------------------------------------------------
    // Phase 1: load x into registers, accumulate sum and sum-of-squares.
    // -----------------------------------------------------------------
    float xVals[ELEMS_PER_THREAD];
    float localSum = 0.0f;
    float localSqSum = 0.0f;

#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_PER_THREAD; ++chunk)
    {
        int vec_idx = chunk * BLOCK_SIZE + tid;
        int elem_offset = vec_idx * ELTS_PER_VEC;

        uint4 xVec = *reinterpret_cast<uint4 const*>(xRow + elem_offset);
        T2 const* xVec2 = reinterpret_cast<T2 const*>(&xVec);

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float2 xf2;
            if constexpr (std::is_same_v<T, half>)
            {
                xf2 = __half22float2(xVec2[i]);
            }
            else
            {
                xf2 = __bfloat1622float2(xVec2[i]);
            }
            xVals[chunk * ELTS_PER_VEC + i * 2] = xf2.x;
            xVals[chunk * ELTS_PER_VEC + i * 2 + 1] = xf2.y;
            localSum += xf2.x + xf2.y;
            localSqSum += xf2.x * xf2.x + xf2.y * xf2.y;
        }
    }

    // Warp-level reductions for sum and sumSq.
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        localSum += __shfl_xor_sync(0xffffffff, localSum, offset);
        localSqSum += __shfl_xor_sync(0xffffffff, localSqSum, offset);
    }

    if (laneId == 0)
    {
        warpSums[warpId] = localSum;
        warpSqSums[warpId] = localSqSum;
    }
    __syncthreads();

    // Cross-warp reduction inside warp 0, then publish mean & rstd.
    if (warpId == 0)
    {
        float s = (laneId < NUM_WARPS) ? warpSums[laneId] : 0.0f;
        float s2 = (laneId < NUM_WARPS) ? warpSqSums[laneId] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            s += __shfl_xor_sync(0xffffffff, s, offset);
            s2 += __shfl_xor_sync(0xffffffff, s2, offset);
        }
        if (laneId == 0)
        {
            float mean = s * invN;
            float var = s2 * invN - mean * mean;
            meanRstd[0] = mean;
            meanRstd[1] = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    float const mean = meanRstd[0];
    float const rstd = meanRstd[1];

    // -----------------------------------------------------------------
    // Phase 2: normalize, apply affine OR modulation, quantize, write.
    // -----------------------------------------------------------------
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_PER_THREAD; ++chunk)
    {
        int vec_idx = chunk * BLOCK_SIZE + tid;
        int elem_offset = vec_idx * ELTS_PER_VEC;

        // Combined per-element affine: y = ((x - mean) * rstd) * w + b
        // where (w, b) come from either learned LN params or AdaLN modulation.
        // For HAS_LN_AFFINE=false and HAS_MODULATION=false, w defaults to 1, b to 0.
        float wVals[ELTS_PER_VEC];
        float bVals[ELTS_PER_VEC];

        if constexpr (HAS_LN_AFFINE)
        {
            uint4 wVec = *reinterpret_cast<uint4 const*>(ln_weight + elem_offset);
            uint4 bVec = *reinterpret_cast<uint4 const*>(ln_bias + elem_offset);
            T2 const* wVec2 = reinterpret_cast<T2 const*>(&wVec);
            T2 const* bVec2 = reinterpret_cast<T2 const*>(&bVec);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 wf2, bf2;
                if constexpr (std::is_same_v<T, half>)
                {
                    wf2 = __half22float2(wVec2[i]);
                    bf2 = __half22float2(bVec2[i]);
                }
                else
                {
                    wf2 = __bfloat1622float2(wVec2[i]);
                    bf2 = __bfloat1622float2(bVec2[i]);
                }
                wVals[i * 2] = wf2.x;
                wVals[i * 2 + 1] = wf2.y;
                bVals[i * 2] = bf2.x;
                bVals[i * 2 + 1] = bf2.y;
            }
        }
        else if constexpr (HAS_MODULATION)
        {
            uint4 sVec = *reinterpret_cast<uint4 const*>(scale_msa_row + elem_offset);
            uint4 shVec = *reinterpret_cast<uint4 const*>(shift_msa_row + elem_offset);
            T2 const* sVec2 = reinterpret_cast<T2 const*>(&sVec);
            T2 const* shVec2 = reinterpret_cast<T2 const*>(&shVec);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 sf2, shf2;
                if constexpr (std::is_same_v<T, half>)
                {
                    sf2 = __half22float2(sVec2[i]);
                    shf2 = __half22float2(shVec2[i]);
                }
                else
                {
                    sf2 = __bfloat1622float2(sVec2[i]);
                    shf2 = __bfloat1622float2(shVec2[i]);
                }
                // AdaLN: y = normalized * (1 + scale_msa) + shift_msa.
                // Fold the +1 into the weight so phase 2 is one fused mul-add.
                wVals[i * 2] = 1.0f + sf2.x;
                wVals[i * 2 + 1] = 1.0f + sf2.y;
                bVals[i * 2] = shf2.x;
                bVals[i * 2 + 1] = shf2.y;
            }
        }

        // Normalize and (if applicable) apply the affine/modulation.
        float yVals[ELTS_PER_VEC];
#pragma unroll
        for (int i = 0; i < ELTS_PER_VEC; ++i)
        {
            float n = (xVals[chunk * ELTS_PER_VEC + i] - mean) * rstd;
            if constexpr (HAS_LN_AFFINE || HAS_MODULATION)
            {
                yVals[i] = n * wVals[i] + bVals[i];
            }
            else
            {
                yVals[i] = n;
            }
        }

        // Quantize 8 floats to FP4. Two adjacent threads cooperate (via the
        // XOR-1 warp shuffle inside cvt_float_to_fp4_inline) to find the
        // 16-element block max and emit one shared FP8 SF byte.
        std::optional<int> optionalBatchIdx = std::nullopt;
        std::optional<int> optionalNumRows = M;
        uint8_t* sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(
            optionalBatchIdx, row, vec_idx, optionalNumRows, numSfVecsTotal, sf_out, QuantizationSFLayout::SWIZZLED);

        uint32_t fp4Packed = cvt_float_to_fp4_inline(yVals, sfScaleVal, sfOutPtr);

        int64_t outOffset = static_cast<int64_t>(row) * (kN_HARDCODED / ELTS_PER_VEC) + vec_idx;
        y_fp4[outOffset] = fp4Packed;
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("FusedLayerNormQuant requires SM100 (Blackwell) or newer!\n");
    }
#endif
}

template <typename T>
void invokeFusedLayerNormQuant(FusedLayerNormQuantParams<T> const& params)
{
    constexpr int BLOCK_SIZE = 128;

    TLLM_CHECK_WITH_INFO(
        params.N == kN_HARDCODED, "fusedLayerNormQuant v1 supports N=%d only (got %d).", kN_HARDCODED, params.N);
    TLLM_CHECK_WITH_INFO(
        !(params.has_ln_affine && params.has_modulation), "has_ln_affine and has_modulation are mutually exclusive.");

    // Defensive pointer-presence checks. Mode-specific kernels dereference
    // the corresponding tensors unconditionally; passing a null pointer to a
    // kernel templated on HAS_LN_AFFINE / HAS_MODULATION = true would
    // surface as an "illegal memory access" with no useful context. Catch
    // mis-wired callers here instead.
    TLLM_CHECK_WITH_INFO(params.x != nullptr, "input tensor (params.x) must not be null.");
    TLLM_CHECK_WITH_INFO(params.y_fp4 != nullptr, "output tensor (params.y_fp4) must not be null.");
    TLLM_CHECK_WITH_INFO(params.sf_out != nullptr, "scale-factor output (params.sf_out) must not be null.");
    TLLM_CHECK_WITH_INFO(params.sf_scale != nullptr, "params.sf_scale must not be null.");
    if (params.has_ln_affine)
    {
        TLLM_CHECK_WITH_INFO(params.ln_weight != nullptr && params.ln_bias != nullptr,
            "ln_weight and ln_bias must be non-null when has_ln_affine=true.");
    }
    if (params.has_modulation)
    {
        TLLM_CHECK_WITH_INFO(params.scale_msa != nullptr && params.shift_msa != nullptr,
            "scale_msa and shift_msa must be non-null when has_modulation=true.");
        TLLM_CHECK_WITH_INFO(
            params.seq_len_per_batch > 0, "seq_len_per_batch must be positive when has_modulation is true.");
        TLLM_CHECK_WITH_INFO(params.M % params.seq_len_per_batch == 0,
            "M (%d) must be divisible by seq_len_per_batch (%d).", params.M, params.seq_len_per_batch);
    }

    dim3 grid(params.M);
    dim3 block(BLOCK_SIZE);

    if (params.has_ln_affine)
    {
        fusedLayerNormQuantKernel<T, BLOCK_SIZE, /*HAS_LN_AFFINE=*/true, /*HAS_MODULATION=*/false>
            <<<grid, block, 0, params.stream>>>(params.x, params.ln_weight, params.ln_bias,
                /*scale_msa=*/nullptr, /*shift_msa=*/nullptr, params.y_fp4, params.sf_out, params.sf_scale, params.M,
                params.seq_len_per_batch, params.eps);
    }
    else if (params.has_modulation)
    {
        fusedLayerNormQuantKernel<T, BLOCK_SIZE, /*HAS_LN_AFFINE=*/false, /*HAS_MODULATION=*/true>
            <<<grid, block, 0, params.stream>>>(params.x, /*ln_weight=*/nullptr, /*ln_bias=*/nullptr, params.scale_msa,
                params.shift_msa, params.y_fp4, params.sf_out, params.sf_scale, params.M, params.seq_len_per_batch,
                params.eps);
    }
    else
    {
        fusedLayerNormQuantKernel<T, BLOCK_SIZE, /*HAS_LN_AFFINE=*/false, /*HAS_MODULATION=*/false>
            <<<grid, block, 0, params.stream>>>(params.x, /*ln_weight=*/nullptr, /*ln_bias=*/nullptr,
                /*scale_msa=*/nullptr, /*shift_msa=*/nullptr, params.y_fp4, params.sf_out, params.sf_scale, params.M,
                params.seq_len_per_batch, params.eps);
    }

    sync_check_cuda_error(params.stream);
}

template void invokeFusedLayerNormQuant<half>(FusedLayerNormQuantParams<half> const&);

#ifdef ENABLE_BF16
template void invokeFusedLayerNormQuant<__nv_bfloat16>(FusedLayerNormQuantParams<__nv_bfloat16> const&);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
