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

/*
 * Fused Gated RMSNorm + NVFP4 Quantization CUDA Kernel
 *
 * Fuses three operations for Nemotron-H NVFP4 quantized path:
 * 1. SiLU gating: gated = x * z * sigmoid(z)
 * 2. Group RMSNorm: y = norm(gated) * weight
 * 3. NVFP4 quantization with block scaling
 *
 * Key optimizations:
 * - Register-based storage: gated values in registers (single HBM pass for x, z)
 * - Inline float-to-FP4 quantization (skips intermediate bf16 conversion)
 * - Vectorized loads with uint4 (8 bf16 per load)
 * - Efficient warp-level reduction using shuffle
 *
 * Performance (vs Triton + fp4_quantize baseline):
 * - M=500:   3.4x faster
 * - M=8192:  1.3x faster
 * - M=50000: 1.29x faster
 */

#include "fusedGatedRMSNormQuant.cuh"
#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaBufferUtils.cuh"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <cmath>
#include <cstdint>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Constants for FP4 quantization
static constexpr int ELTS_PER_THREAD = 8;
static constexpr int SF_VEC_SIZE = 16;
static constexpr int NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD; // 2

// Sigmoid using fast math (optimal for B200/SM100)
__device__ __forceinline__ float fast_sigmoid(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

// Fast approximate reciprocal with flush-to-zero (matches quantization.cuh)
__device__ __forceinline__ float rcp_approx_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

/*
 * Inline FP4 quantization for float32 values
 *
 * Quantizes float32 values directly to FP4 (e2m1), avoiding the
 * intermediate bf16 conversion in cvt_warp_fp16_to_fp4.
 *
 * Original flow: float32 -> bf16 -> float32 -> e2m1 (4 conversions)
 * Optimized:     float32 -> e2m1 (direct, 0 extra conversions)
 */
__device__ __forceinline__ uint32_t cvt_float_to_fp4_inline(float* vals, // 8 float values to quantize
    float sfScaleVal,                                                    // Scale factor scale
    uint8_t* sfOutPtr)                                                   // Output for scale factor (1 byte)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Find local max (across 8 values)
    float localMax = fabsf(vals[0]);
#pragma unroll
    for (int i = 1; i < 8; i++)
    {
        localMax = fmaxf(localMax, fabsf(vals[i]));
    }

    // Get max across 2 threads (for 16-element scale factor block)
    localMax = fmaxf(__shfl_xor_sync(0xffffffff, localMax, 1), localMax);

    // Compute scale factor: SF = SFScaleVal * (max / 6.0)
    // where 6.0 is the max representable value in e2m1 format
    float sfValue = sfScaleVal * (localMax * rcp_approx_ftz(6.0f));

    // Convert to E4M3 and back to get quantized scale
    __nv_fp8_e4m3 sfFp8 = __nv_fp8_e4m3(sfValue);
    uint8_t sfByte = sfFp8.__x;
    float sfValueQuant = static_cast<float>(sfFp8);

    // Compute output scale for quantization
    // outputScale = sfScaleVal / sfValueQuant
    float outputScale = (localMax != 0.0f) ? rcp_approx_ftz(sfValueQuant * rcp_approx_ftz(sfScaleVal)) : 0.0f;

    // Write scale factor
    if (sfOutPtr)
    {
        *sfOutPtr = sfByte;
    }

    // Scale all values and convert to e2m1 using PTX instruction
    float s0 = vals[0] * outputScale;
    float s1 = vals[1] * outputScale;
    float s2 = vals[2] * outputScale;
    float s3 = vals[3] * outputScale;
    float s4 = vals[4] * outputScale;
    float s5 = vals[5] * outputScale;
    float s6 = vals[6] * outputScale;
    float s7 = vals[7] * outputScale;

    uint32_t result;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(result)
        : "f"(s0), "f"(s1), "f"(s2), "f"(s3), "f"(s4), "f"(s5), "f"(s6), "f"(s7));

    return result;
#else
    return 0;
#endif
}

/*
 * Optimized Fused Gated RMSNorm + FP4 Quantization Kernel
 *
 * Grid: (M, ngroups) where ngroups = N / groupSize
 * Block: BLOCK_SIZE threads (128 for group_size=1024)
 *
 * Key optimizations:
 * 1. Register storage: Gated values stored in registers (not recomputed)
 * 2. Inline float quantization: Direct float32 -> FP4 (no intermediate bf16)
 * 3. Single HBM pass for x and z
 *
 * Memory pattern:
 *   Pass 1: Read x, z from HBM -> compute gated values -> store in registers
 *   Pass 2: Read weight from HBM -> normalize using registers -> FP4 output
 *
 * This reduces HBM traffic from 2*(x+z) + w to (x+z) + w (~47% reduction)
 */
template <typename T, int BLOCK_SIZE, int GROUP_SIZE>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(BLOCK_SIZE, 8)
#endif
    fusedGatedRMSNormQuantKernelOptimized(T const* __restrict__ x, T const* __restrict__ z,
        T const* __restrict__ weight, uint32_t* __restrict__ y_fp4, uint32_t* __restrict__ sf_out,
        float const* __restrict__ sf_scale, int M, int N, int zRowStride, int groupSize, float eps)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    using T2 = typename packed_as<T, 2>::type;

    static constexpr int ELEMS_PER_THREAD = GROUP_SIZE / BLOCK_SIZE;
    static_assert(ELEMS_PER_THREAD == 8, "Expected 8 elements per thread");

    int const row = blockIdx.x;
    int const group = blockIdx.y;
    int const tid = threadIdx.x;
    int const warpId = tid / 32;
    int const laneId = tid % 32;
    int const numWarps = BLOCK_SIZE / 32;

    int const groupOffset = group * groupSize;

    float const invGroupSize = 1.0f / static_cast<float>(groupSize);
    float const sfScaleVal = (sf_scale != nullptr) ? sf_scale[0] : 1.0f;
    int const numSfVecsTotal = N / SF_VEC_SIZE;

    __shared__ float warpSums[BLOCK_SIZE / 32];

    T const* xGroup = x + static_cast<int64_t>(row) * N + groupOffset;
    T const* zGroup = z + static_cast<int64_t>(row) * zRowStride + groupOffset;
    T const* wGroup = weight + groupOffset;

    // ================================================================
    // Phase 1: Load x, z, compute gated values, store in registers
    // ================================================================
    float gatedVals[ELEMS_PER_THREAD];
    float localSqSum = 0.0f;

    int const baseIdx = tid * ELEMS_PER_THREAD;

    // Vectorized load: 8 bf16 = 16 bytes = 1 uint4
    uint4 xVec = *reinterpret_cast<uint4 const*>(xGroup + baseIdx);
    uint4 zVec = *reinterpret_cast<uint4 const*>(zGroup + baseIdx);

    T2 const* xVec2 = reinterpret_cast<T2 const*>(&xVec);
    T2 const* zVec2 = reinterpret_cast<T2 const*>(&zVec);

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 xf2, zf2;
        if constexpr (std::is_same_v<T, half>)
        {
            xf2 = __half22float2(xVec2[i]);
            zf2 = __half22float2(zVec2[i]);
        }
        else
        {
            xf2 = __bfloat1622float2(xVec2[i]);
            zf2 = __bfloat1622float2(zVec2[i]);
        }

        float sig0 = fast_sigmoid(zf2.x);
        float sig1 = fast_sigmoid(zf2.y);
        gatedVals[i * 2] = xf2.x * zf2.x * sig0;
        gatedVals[i * 2 + 1] = xf2.y * zf2.y * sig1;

        localSqSum += gatedVals[i * 2] * gatedVals[i * 2];
        localSqSum += gatedVals[i * 2 + 1] * gatedVals[i * 2 + 1];
    }

// Warp-level reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        localSqSum += __shfl_xor_sync(0xffffffff, localSqSum, offset);
    }

    if (laneId == 0)
    {
        warpSums[warpId] = localSqSum;
    }
    __syncthreads();

    // Block-level reduction
    float rstd;
    if (warpId == 0)
    {
        float sum = (laneId < numWarps) ? warpSums[laneId] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        rstd = rsqrtf(sum * invGroupSize + eps);
        warpSums[0] = rstd;
    }
    __syncthreads();
    rstd = warpSums[0];

    // ================================================================
    // Phase 2: Normalize and quantize (direct float to FP4)
    // ================================================================
    uint4 wVec = *reinterpret_cast<uint4 const*>(wGroup + baseIdx);
    T2 const* wVec2 = reinterpret_cast<T2 const*>(&wVec);

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 wf2;
        if constexpr (std::is_same_v<T, half>)
        {
            wf2 = __half22float2(wVec2[i]);
        }
        else
        {
            wf2 = __bfloat1622float2(wVec2[i]);
        }

        // Store normalized values in float (skip intermediate bf16)
        gatedVals[i * 2] = gatedVals[i * 2] * rstd * wf2.x;
        gatedVals[i * 2 + 1] = gatedVals[i * 2 + 1] * rstd * wf2.y;
    }

    int const fp4GroupOffset = group * (groupSize / ELTS_PER_THREAD);
    int const globalVecIdx = fp4GroupOffset + tid;

    std::optional<int> optionalBatchIdx = std::nullopt;
    std::optional<int> optionalNumRows = M;

    uint8_t* sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(
        optionalBatchIdx, row, globalVecIdx, optionalNumRows, numSfVecsTotal, sf_out, QuantizationSFLayout::SWIZZLED);

    // Inline float-to-FP4 quantization (avoids intermediate bf16)
    uint32_t fp4Packed = cvt_float_to_fp4_inline(gatedVals, sfScaleVal, sfOutPtr);

    int64_t outOffset = static_cast<int64_t>(row) * (N / ELTS_PER_THREAD) + globalVecIdx;
    y_fp4[outOffset] = fp4Packed;
#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("FusedGatedRMSNormQuant requires SM100 (Blackwell) or newer!\n");
    }
#endif
}

/*
 * Fallback grouped kernel for group sizes other than 1024
 * Uses standard cvt_warp_fp16_to_fp4 quantization
 */
template <typename T, int BLOCK_SIZE>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(BLOCK_SIZE, 8)
#endif
    fusedGatedRMSNormQuantKernelGrouped(T const* __restrict__ x, T const* __restrict__ z, T const* __restrict__ weight,
        uint32_t* __restrict__ y_fp4, uint32_t* __restrict__ sf_out, float const* __restrict__ sf_scale, int M, int N,
        int zRowStride, int groupSize, float eps)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    using T2 = typename packed_as<T, 2>::type;

    int const row = blockIdx.x;
    int const group = blockIdx.y;
    int const tid = threadIdx.x;
    int const warpId = tid / 32;
    int const laneId = tid % 32;
    int const numWarps = BLOCK_SIZE / 32;

    __shared__ float warpSums[BLOCK_SIZE / 32];

    int const groupOffset = group * groupSize;
    T const* xGroup = x + static_cast<int64_t>(row) * N + groupOffset;
    T const* zGroup = z + static_cast<int64_t>(row) * zRowStride + groupOffset;
    T const* wGroup = weight + groupOffset;

    int const numVecs = groupSize / ELTS_PER_THREAD;
    float const invGroupSize = 1.0f / static_cast<float>(groupSize);
    float const sfScaleVal = (sf_scale != nullptr) ? sf_scale[0] : 1.0f;

    uint4 const* xGroup8 = reinterpret_cast<uint4 const*>(xGroup);
    uint4 const* zGroup8 = reinterpret_cast<uint4 const*>(zGroup);
    uint4 const* wGroup8 = reinterpret_cast<uint4 const*>(wGroup);

    // Phase 1: Compute variance
    float localSqSum = 0.0f;

    for (int vecIdx = tid; vecIdx < numVecs; vecIdx += BLOCK_SIZE)
    {
        uint4 xVec = xGroup8[vecIdx];
        uint4 zVec = zGroup8[vecIdx];

        T2 const* xVec2 = reinterpret_cast<T2 const*>(&xVec);
        T2 const* zVec2 = reinterpret_cast<T2 const*>(&zVec);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 xf2, zf2;
            if constexpr (std::is_same_v<T, half>)
            {
                xf2 = __half22float2(xVec2[i]);
                zf2 = __half22float2(zVec2[i]);
            }
            else
            {
                xf2 = __bfloat1622float2(xVec2[i]);
                zf2 = __bfloat1622float2(zVec2[i]);
            }

            float sig0 = fast_sigmoid(zf2.x);
            float sig1 = fast_sigmoid(zf2.y);
            float gated0 = xf2.x * zf2.x * sig0;
            float gated1 = xf2.y * zf2.y * sig1;
            localSqSum += gated0 * gated0 + gated1 * gated1;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        localSqSum += __shfl_xor_sync(0xffffffff, localSqSum, offset);
    }

    if (laneId == 0)
    {
        warpSums[warpId] = localSqSum;
    }
    __syncthreads();

    float rstd;
    if (warpId == 0)
    {
        float sum = (laneId < numWarps) ? warpSums[laneId] : 0.0f;

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }

        rstd = rsqrtf(sum * invGroupSize + eps);
        warpSums[0] = rstd;
    }
    __syncthreads();

    rstd = warpSums[0];

    // Phase 2: Normalize and quantize
    int const fp4VecsPerGroup = groupSize / ELTS_PER_THREAD;
    int const fp4GroupOffset = group * fp4VecsPerGroup;
    uint32_t* y_fp4_group = y_fp4 + static_cast<int64_t>(row) * (N / ELTS_PER_THREAD) + fp4GroupOffset;
    int const numSfVecsTotal = N / SF_VEC_SIZE;

    for (int vecIdx = tid; vecIdx < numVecs; vecIdx += BLOCK_SIZE)
    {
        uint4 xVec = xGroup8[vecIdx];
        uint4 zVec = zGroup8[vecIdx];
        uint4 wVec = wGroup8[vecIdx];

        T2 const* xVec2 = reinterpret_cast<T2 const*>(&xVec);
        T2 const* zVec2 = reinterpret_cast<T2 const*>(&zVec);
        T2 const* wVec2 = reinterpret_cast<T2 const*>(&wVec);

        PackedVec<T> packedVec;

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 xf2, zf2, wf2;
            if constexpr (std::is_same_v<T, half>)
            {
                xf2 = __half22float2(xVec2[i]);
                zf2 = __half22float2(zVec2[i]);
                wf2 = __half22float2(wVec2[i]);
            }
            else
            {
                xf2 = __bfloat1622float2(xVec2[i]);
                zf2 = __bfloat1622float2(zVec2[i]);
                wf2 = __bfloat1622float2(wVec2[i]);
            }

            float sig0 = fast_sigmoid(zf2.x);
            float sig1 = fast_sigmoid(zf2.y);
            float gated0 = xf2.x * zf2.x * sig0;
            float gated1 = xf2.y * zf2.y * sig1;
            float val0 = gated0 * rstd * wf2.x;
            float val1 = gated1 * rstd * wf2.y;

            packedVec.elts[i] = cuda_cast<T2>(make_float2(val0, val1));
        }

        int const globalVecIdx = fp4GroupOffset + vecIdx;

        std::optional<int> optionalBatchIdx = std::nullopt;
        std::optional<int> optionalNumRows = M;

        uint8_t* sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(optionalBatchIdx, row,
            globalVecIdx, optionalNumRows, numSfVecsTotal, sf_out, QuantizationSFLayout::SWIZZLED);

        uint32_t fp4Packed = cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, false>(packedVec, sfScaleVal, sfOutPtr);

        y_fp4_group[vecIdx] = fp4Packed;
    }

#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("FusedGatedRMSNormQuant requires SM100 (Blackwell) or newer!\n");
    }
#endif
}

/*
 * Fallback kernel for groupSize == N (no groups, full row normalization)
 */
template <typename T, int BLOCK_SIZE>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(BLOCK_SIZE, 4)
#endif
    fusedGatedRMSNormQuantKernelFullRow(T const* __restrict__ x, T const* __restrict__ z, T const* __restrict__ weight,
        uint32_t* __restrict__ y_fp4, uint32_t* __restrict__ sf_out, float const* __restrict__ sf_scale, int M, int N,
        int zRowStride, float eps)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    using T2 = typename packed_as<T, 2>::type;

    int const tid = threadIdx.x;
    int const numColVecs = N / ELTS_PER_THREAD;
    int const numSfVecs = N / SF_VEC_SIZE;

    extern __shared__ float smem[];
    float* warpSums = smem;

    int const warpId = tid / 32;
    int const laneId = tid % 32;
    int const numWarps = (BLOCK_SIZE + 31) / 32;

    float const sfScaleVal = (sf_scale != nullptr) ? sf_scale[0] : 1.0f;
    float const invN = 1.0f / static_cast<float>(N);

    cudaGridDependencySynchronize();

    for (int row = blockIdx.x; row < M; row += gridDim.x)
    {
        T const* xRow = x + static_cast<int64_t>(row) * N;
        T const* zRow = z + static_cast<int64_t>(row) * zRowStride;

        // Phase 1: Compute variance using vectorized loads
        float localSqSum0 = 0.0f;
        float localSqSum1 = 0.0f;
        float localSqSum2 = 0.0f;
        float localSqSum3 = 0.0f;

        uint4 const* xRow8 = reinterpret_cast<uint4 const*>(xRow);
        uint4 const* zRow8 = reinterpret_cast<uint4 const*>(zRow);

        for (int vec8Idx = tid; vec8Idx < numColVecs; vec8Idx += BLOCK_SIZE)
        {
            uint4 xVec = xRow8[vec8Idx];
            uint4 zVec = zRow8[vec8Idx];

            T2* xVec2 = reinterpret_cast<T2*>(&xVec);
            T2* zVec2 = reinterpret_cast<T2*>(&zVec);

            // Unroll with separate accumulators for ILP
            {
                float2 xf2, zf2;
                if constexpr (std::is_same_v<T, half>)
                {
                    xf2 = __half22float2(xVec2[0]);
                    zf2 = __half22float2(zVec2[0]);
                }
                else
                {
                    xf2 = __bfloat1622float2(xVec2[0]);
                    zf2 = __bfloat1622float2(zVec2[0]);
                }
                float sig0 = fast_sigmoid(zf2.x);
                float sig1 = fast_sigmoid(zf2.y);
                float gated0 = xf2.x * zf2.x * sig0;
                float gated1 = xf2.y * zf2.y * sig1;
                localSqSum0 += gated0 * gated0 + gated1 * gated1;
            }
            {
                float2 xf2, zf2;
                if constexpr (std::is_same_v<T, half>)
                {
                    xf2 = __half22float2(xVec2[1]);
                    zf2 = __half22float2(zVec2[1]);
                }
                else
                {
                    xf2 = __bfloat1622float2(xVec2[1]);
                    zf2 = __bfloat1622float2(zVec2[1]);
                }
                float sig0 = fast_sigmoid(zf2.x);
                float sig1 = fast_sigmoid(zf2.y);
                float gated0 = xf2.x * zf2.x * sig0;
                float gated1 = xf2.y * zf2.y * sig1;
                localSqSum1 += gated0 * gated0 + gated1 * gated1;
            }
            {
                float2 xf2, zf2;
                if constexpr (std::is_same_v<T, half>)
                {
                    xf2 = __half22float2(xVec2[2]);
                    zf2 = __half22float2(zVec2[2]);
                }
                else
                {
                    xf2 = __bfloat1622float2(xVec2[2]);
                    zf2 = __bfloat1622float2(zVec2[2]);
                }
                float sig0 = fast_sigmoid(zf2.x);
                float sig1 = fast_sigmoid(zf2.y);
                float gated0 = xf2.x * zf2.x * sig0;
                float gated1 = xf2.y * zf2.y * sig1;
                localSqSum2 += gated0 * gated0 + gated1 * gated1;
            }
            {
                float2 xf2, zf2;
                if constexpr (std::is_same_v<T, half>)
                {
                    xf2 = __half22float2(xVec2[3]);
                    zf2 = __half22float2(zVec2[3]);
                }
                else
                {
                    xf2 = __bfloat1622float2(xVec2[3]);
                    zf2 = __bfloat1622float2(zVec2[3]);
                }
                float sig0 = fast_sigmoid(zf2.x);
                float sig1 = fast_sigmoid(zf2.y);
                float gated0 = xf2.x * zf2.x * sig0;
                float gated1 = xf2.y * zf2.y * sig1;
                localSqSum3 += gated0 * gated0 + gated1 * gated1;
            }
        }

        float localSqSum = localSqSum0 + localSqSum1 + localSqSum2 + localSqSum3;

        // Warp-level reduction
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            localSqSum += __shfl_xor_sync(0xffffffff, localSqSum, offset);
        }

        if (laneId == 0)
        {
            warpSums[warpId] = localSqSum;
        }
        __syncthreads();

        // Block-level reduction
        float totalSqSum = 0.0f;
        if (warpId == 0)
        {
            if (laneId < numWarps)
            {
                totalSqSum = warpSums[laneId];
            }

#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
            {
                totalSqSum += __shfl_xor_sync(0xffffffff, totalSqSum, offset);
            }

            if (laneId == 0)
            {
                warpSums[0] = totalSqSum;
            }
        }
        __syncthreads();

        float const rstd = rsqrtf(warpSums[0] * invN + eps);

        // Phase 2: Normalize and quantize
        uint4 const* wRow8 = reinterpret_cast<uint4 const*>(weight);

        for (int vecIdx = tid; vecIdx < numColVecs; vecIdx += BLOCK_SIZE)
        {
            uint4 xVec = xRow8[vecIdx];
            uint4 zVec = zRow8[vecIdx];
            uint4 wVec = wRow8[vecIdx];

            T2* xVec2 = reinterpret_cast<T2*>(&xVec);
            T2* zVec2 = reinterpret_cast<T2*>(&zVec);
            T2* wVec2 = reinterpret_cast<T2*>(&wVec);

            PackedVec<T> packedVec;

#pragma unroll 4
            for (int i = 0; i < 4; i++)
            {
                float2 xf2, zf2, wf2;
                if constexpr (std::is_same_v<T, half>)
                {
                    xf2 = __half22float2(xVec2[i]);
                    zf2 = __half22float2(zVec2[i]);
                    wf2 = __half22float2(wVec2[i]);
                }
                else
                {
                    xf2 = __bfloat1622float2(xVec2[i]);
                    zf2 = __bfloat1622float2(zVec2[i]);
                    wf2 = __bfloat1622float2(wVec2[i]);
                }

                float sig0 = fast_sigmoid(zf2.x);
                float sig1 = fast_sigmoid(zf2.y);
                float gated0 = xf2.x * zf2.x * sig0;
                float gated1 = xf2.y * zf2.y * sig1;
                float val0 = gated0 * rstd * wf2.x;
                float val1 = gated1 * rstd * wf2.y;

                packedVec.elts[i] = cuda_cast<T2>(make_float2(val0, val1));
            }

            int64_t const outOffset = static_cast<int64_t>(row) * numColVecs + vecIdx;

            std::optional<int> optionalBatchIdx = std::nullopt;
            std::optional<int> optionalNumRows = M;

            uint8_t* sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(
                optionalBatchIdx, row, vecIdx, optionalNumRows, numSfVecs, sf_out, QuantizationSFLayout::SWIZZLED);

            uint32_t fp4Packed = cvt_warp_fp16_to_fp4<T, SF_VEC_SIZE, false>(packedVec, sfScaleVal, sfOutPtr);

            y_fp4[outOffset] = fp4Packed;
        }

        __syncthreads();
    }

    cudaTriggerProgrammaticLaunchCompletion();
#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("FusedGatedRMSNormQuant requires SM100 (Blackwell) or newer!\n");
    }
#endif
}

template <typename T>
void invokeFusedGatedRMSNormQuant(FusedGatedRMSNormQuantParams<T> const& params, int multiProcessorCount)
{
    int const ngroups = params.N / params.groupSize;

    if (params.groupSize < params.N && ngroups > 1)
    {
        // Grouped kernel path
        if (params.groupSize == 1024)
        {
            // Use optimized kernel for group_size=1024
            static constexpr int BLOCK_SIZE = 128; // 4 warps, 8 elements per thread

            dim3 grid(params.M, ngroups);
            dim3 block(BLOCK_SIZE);

            fusedGatedRMSNormQuantKernelOptimized<T, BLOCK_SIZE, 1024><<<grid, block, 0, params.stream>>>(params.x,
                params.z, params.weight, params.y_fp4, params.sf_out, params.sf_scale, params.M, params.N,
                params.zRowStride, params.groupSize, params.eps);
        }
        else
        {
            // Fallback kernel for other group sizes
            static constexpr int BLOCK_SIZE = 128;

            dim3 grid(params.M, ngroups);
            dim3 block(BLOCK_SIZE);

            fusedGatedRMSNormQuantKernelGrouped<T, BLOCK_SIZE><<<grid, block, 0, params.stream>>>(params.x, params.z,
                params.weight, params.y_fp4, params.sf_out, params.sf_scale, params.M, params.N, params.zRowStride,
                params.groupSize, params.eps);
        }
    }
    else
    {
        // Use full-row kernel when groupSize == N
        static constexpr int BLOCK_SIZE = 512;

        int const numWarps = (BLOCK_SIZE + 31) / 32;
        size_t const smemSize = numWarps * sizeof(float);

        int const numBlocks = std::min(params.M, multiProcessorCount * 4);

        fusedGatedRMSNormQuantKernelFullRow<T, BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, smemSize, params.stream>>>(params.x,
            params.z, params.weight, params.y_fp4, params.sf_out, params.sf_scale, params.M, params.N,
            params.zRowStride, params.eps);
    }

    CUDA_CALL(cudaGetLastError());
}

template void invokeFusedGatedRMSNormQuant<half>(
    FusedGatedRMSNormQuantParams<half> const& params, int multiProcessorCount);

#ifdef ENABLE_BF16
template void invokeFusedGatedRMSNormQuant<__nv_bfloat16>(
    FusedGatedRMSNormQuantParams<__nv_bfloat16> const& params, int multiProcessorCount);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
