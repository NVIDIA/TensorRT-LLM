/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/envUtils.h"

#include "tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.h"
#include <cstdint>
using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels::dsv3MinLatencyKernels
{

// Custom FMA implementation using PTX assembly instructions
__device__ __forceinline__ void fma(float2& d, float2 const& a, float2 const& b, float2 const& c)
{
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(d))
                 : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)),
                 "l"(reinterpret_cast<uint64_t const&>(c)));
}

// Convert 8 bfloat16 values from a uint4 to float array - optimized conversion
template <int VPT>
__device__ __forceinline__ void bf16_uint4_to_float8(uint4 const& vec, float* dst)
{
    __nv_bfloat16* bf16_ptr = reinterpret_cast<__nv_bfloat16*>(const_cast<uint4*>(&vec));

#pragma unroll
    for (int i = 0; i < VPT; i++)
    {
        dst[i] = __bfloat162float(bf16_ptr[i]);
    }
}

template <typename T, int kBlockSize, int VPT, int kNumTokens, int kNumExperts, int kHiddenDim>
__global__ __launch_bounds__(128, 1) void router_gemm_kernel(float* out, T const* mat_a, T const* mat_b)
{
    // Each block handles one expert column
    int const n_idx = blockIdx.x;
    int const tid = threadIdx.x;
    constexpr int kWarpSize = 32;
    constexpr int kNumWarps = kBlockSize / kWarpSize;
    // Constants for this kernel
    constexpr int k_elems_per_k_iteration = VPT * kBlockSize;
    constexpr int k_iterations = kHiddenDim / k_elems_per_k_iteration; // Total K iterations

    // Initialize accumulators for all M rows
    float acc[kNumTokens] = {};

    // Shared memory for warp-level reduction
    __shared__ float sm_reduction[kNumTokens][kNumWarps]; // kNumWarps

    // B matrix is in column-major order, so we can directly load a column for the n_idx expert
    T const* b_col = mat_b + n_idx * kHiddenDim;

    // Pre-compute k_base values for each iteration to help compiler optimize
    // int k_bases[k_iterations];
    int k_bases[k_iterations];
#pragma unroll
    for (int ki = 0; ki < k_iterations; ki++)
    {
        k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // Process the GEMM in chunks
    for (int ki = 0; ki < k_iterations; ki++)
    {
        int const k_base = k_bases[ki];

        // Load B matrix values using vector load (8 bf16 values)
        uint4 b_vec = *reinterpret_cast<uint4 const*>(b_col + k_base);

        // Convert B values to float
        float b_float[VPT];
        bf16_uint4_to_float8<VPT>(b_vec, b_float);

// Process each token
#pragma unroll
        for (int m_idx = 0; m_idx < kNumTokens; m_idx++)
        {
            // Load both rows of A matrix using vector loads
            uint4 a_vec = *reinterpret_cast<uint4 const*>(mat_a + (m_idx * kHiddenDim) + k_base);

            // Convert A values to float
            float a_float[VPT];
            bf16_uint4_to_float8<VPT>(a_vec, a_float);

// Process elements in this chunk
#pragma unroll
            for (int k = 0; k < VPT; k++)
            {
                float a = a_float[k];
                float b = b_float[k];
                acc[m_idx] += a * b;
            }
        }
    }

    // Perform warp-level reduction
    int const warpSize = 32;
    int const warpId = tid / warpSize;
    int const laneId = tid % warpSize;

    // Register for warp-level reduction results
    float warp_result[kNumTokens];

#pragma unroll
    for (int m_idx = 0; m_idx < kNumTokens; m_idx++)
    {
        warp_result[m_idx] = acc[m_idx];
    }

// Perform warp-level reduction using optimized butterfly pattern
#pragma unroll
    for (int m = 0; m < kNumTokens; m++)
    {
        float sum = warp_result[m];

        // Butterfly reduction pattern
        sum += __shfl_xor_sync(0xffffffff, sum, 16);
        sum += __shfl_xor_sync(0xffffffff, sum, 8);
        sum += __shfl_xor_sync(0xffffffff, sum, 4);
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        sum += __shfl_xor_sync(0xffffffff, sum, 1);

        // Only the first thread in each warp stores to shared memory
        if (laneId == 0)
        {
            sm_reduction[m][warpId] = sum;
        }
    }

    __syncthreads();

    // Final reduction across warps (only first thread)
    if (tid == 0)
    {
#pragma unroll
        for (int m = 0; m < kNumTokens; m++)
        {
            float final_sum = 0.0f;

// Sum across the kNumWarps
#pragma unroll
            for (int w = 0; w < kNumWarps; w++)
            {
                final_sum += sm_reduction[m][w];
            }

            // Write final result
            out[m * kNumExperts + n_idx] = final_sum;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemm(float* output, T const* mat_a, T const* mat_b, cudaStream_t stream)
{
    constexpr int VPT = 16 / sizeof(T);
    constexpr int kBlockSize = 128;
    cudaLaunchConfig_t config;
    config.gridDim = kNumExperts;
    config.blockDim = kBlockSize;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
        &config, router_gemm_kernel<T, kBlockSize, VPT, kNumTokens, kNumExperts, kHiddenDim>, output, mat_a, mat_b));
}

// D += A(16x16, row) * B(16x8, col), bf16 inputs, fp32 accumulate. Fragment layout (PTX ISA, m16n8k16):
//   a[0] = A[g][2q..2q+1]  a[1] = A[g+8][2q..2q+1]  a[2] = A[g][2q+8..2q+9]  a[3] = A[g+8][2q+8..2q+9]
//   b[0] = B[2q..2q+1][g]  b[1] = B[2q+8..2q+9][g]   c[0..1] = D[g][2q..2q+1]  c[2..3] = D[g+8][2q..2q+1]
// with g = lane / 4, q = lane % 4.
__device__ __forceinline__ void mma_bf16_m16n8k16(float (&c)[4], uint32_t const (&a)[4], uint32_t const (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Tensor-core variant for large expert counts (Kimi K3: 896 experts). One warp computes a 16-token x 8-expert
// tile over a K-slice with mma.sync; the kNumWarps warps of a CTA split K and are reduced through shared memory
// in warp order (deterministic). Each lane loads 16 bytes (8 consecutive k) per row/column per 32-wide k-chunk;
// the k order inside a chunk is permuted identically for A and B (lane group q owns k = q*8..q*8+7, fed to the
// two MMAs as k-pairs {0,1},{2,3} and {4,5},{6,7}), which is legal because the dot product is order-invariant.
// Motivation (B200, M=8): router_gemm_kernel executes ~4M warp-instructions (M x 896 x 7168 FMAs + bf16->fp32
// converts) and is issue-bound (~26% issue utilisation, time ~linear in M); one MMA replaces 64 warp-FFMAs.
template <int kNumTokens, int kNumExperts, int kHiddenDim, int kExpertsPerCta, int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void router_gemm_mma_kernel(
    float* out, __nv_bfloat16 const* mat_a, __nv_bfloat16 const* mat_b)
{
    static_assert(kNumTokens >= 1 && kNumTokens <= 16, "one m16 tile in the token dimension");
    static_assert(kExpertsPerCta % 8 == 0, "expert columns per CTA must be whole n8 tiles");
    constexpr int kNumWarps = kBlockSize / 32;
    constexpr int kTilesN = kExpertsPerCta / 8;
    constexpr int kChunk = 32; // k elements per lane load (8 per lane group) = two m16n8k16 MMAs
    static_assert(kHiddenDim % (kNumWarps * kChunk) == 0, "K must split evenly over warps and 32-wide chunks");
    constexpr int kChunksPerWarp = kHiddenDim / (kNumWarps * kChunk);
    constexpr int kBatch = kChunksPerWarp < 8 ? kChunksPerWarp : 8; // chunks whose loads are kept in flight
    static_assert(kChunksPerWarp % kBatch == 0, "chunks per warp must be a multiple of the load batch");
    constexpr bool kTwoRows = kNumTokens > 8;

    int const warpId = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const g = lane / 4; // token row (A) / expert column (B) owned by this lane in the fragments
    int const q = lane % 4; // 8-wide k sub-chunk owned by this lane
    int const n_base = blockIdx.x * kExpertsPerCta;
    int const k_warp = warpId * (kChunksPerWarp * kChunk);

    __shared__ float sm_red[kNumWarps][kTilesN][32][4];

    float acc[kTilesN][4] = {};

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    __nv_bfloat16 const* a_row0 = mat_a + g * kHiddenDim;
    __nv_bfloat16 const* a_row1 = mat_a + (g + 8) * kHiddenDim;
    bool const row0_valid = g < kNumTokens;
    bool const row1_valid = kTwoRows && (g + 8 < kNumTokens);
    uint4 const zero4 = make_uint4(0u, 0u, 0u, 0u);

#pragma unroll
    for (int batch = 0; batch < kChunksPerWarp; batch += kBatch)
    {
        uint4 a0[kBatch];
        uint4 a1[kBatch];
        uint4 b[kBatch][kTilesN];
#pragma unroll
        for (int c = 0; c < kBatch; c++)
        {
            int const k = k_warp + (batch + c) * kChunk + q * 8;
            a0[c] = row0_valid ? *reinterpret_cast<uint4 const*>(a_row0 + k) : zero4;
            a1[c] = row1_valid ? *reinterpret_cast<uint4 const*>(a_row1 + k) : zero4;
#pragma unroll
            for (int t = 0; t < kTilesN; t++)
            {
                b[c][t] = *reinterpret_cast<uint4 const*>(
                    mat_b + static_cast<int64_t>(n_base + t * 8 + g) * kHiddenDim + k);
            }
        }
#pragma unroll
        for (int c = 0; c < kBatch; c++)
        {
            uint32_t const afrag0[4] = {a0[c].x, a1[c].x, a0[c].y, a1[c].y};
            uint32_t const afrag1[4] = {a0[c].z, a1[c].z, a0[c].w, a1[c].w};
#pragma unroll
            for (int t = 0; t < kTilesN; t++)
            {
                uint32_t const bfrag0[2] = {b[c][t].x, b[c][t].y};
                uint32_t const bfrag1[2] = {b[c][t].z, b[c][t].w};
                mma_bf16_m16n8k16(acc[t], afrag0, bfrag0);
                mma_bf16_m16n8k16(acc[t], afrag1, bfrag1);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < kTilesN; t++)
    {
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            sm_red[warpId][t][lane][i] = acc[t][i];
        }
    }
    __syncthreads();

    // One thread per (tile, lane slot, fragment element): sum the warp partials in warp order and scatter.
    constexpr int kNumOutputs = kTilesN * 32 * 4;
    static_assert(kNumOutputs <= kBlockSize, "need one thread per output element");
    if (threadIdx.x < kNumOutputs)
    {
        int const t = threadIdx.x / 128;
        int const l = (threadIdx.x % 128) / 4;
        int const i = threadIdx.x % 4;
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kNumWarps; w++)
        {
            sum += sm_red[w][t][l][i];
        }
        int const row = (l / 4) + (i >= 2 ? 8 : 0);
        int const col = n_base + t * 8 + (l % 4) * 2 + (i & 1);
        if (row < kNumTokens)
        {
            out[row * kNumExperts + col] = sum;
        }
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmMma(float* output, __nv_bfloat16 const* mat_a, __nv_bfloat16 const* mat_b, cudaStream_t stream)
{
    // 8 expert columns per CTA and 14 warps splitting K: for 896 x 7168 that is 112 CTAs (one wave on B200)
    // each doing 16 k-chunks per warp. Measured B200 (graph replay, L2-warm) vs the cuBLAS fallback:
    // M=1 2.79 vs 7.36 us, M=8 3.77 vs 7.62 us, M=16 4.70 vs 6.64 us.
    constexpr int kExpertsPerCta = 8;
    constexpr int kBlockSize = 448;
    static_assert(kNumExperts % kExpertsPerCta == 0, "kNumExperts must be a multiple of kExpertsPerCta");
    cudaLaunchConfig_t config;
    config.gridDim = kNumExperts / kExpertsPerCta;
    config.blockDim = kBlockSize;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config,
        router_gemm_mma_kernel<kNumTokens, kNumExperts, kHiddenDim, kExpertsPerCta, kBlockSize>, output, mat_a, mat_b));
}

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 1, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 2, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 3, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 4, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 5, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 6, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 7, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 8, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 9, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 10, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 11, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 12, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 13, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 14, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 15, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 16, 256, 7168>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

// hidden_dim=6144 instantiations (GLM-5).
template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 1, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 2, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 3, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 4, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 5, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 6, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 7, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 8, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 9, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 10, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 11, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 12, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 13, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 14, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 15, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 16, 256, 6144>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

// hidden_dim=4096 instantiations (DeepSeek-V4).
template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 1, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 2, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 3, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 4, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 5, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 6, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 7, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 8, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 9, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 10, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 11, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 12, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 13, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 14, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 15, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemm<__nv_bfloat16, 16, 256, 4096>(
    float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

// num_experts=896, hidden_dim=7168 instantiations (Kimi K3), tensor-core variant.
#define INSTANTIATE_ROUTER_GEMM_MMA_K3(kNumTokens)                                                                     \
    template void tensorrt_llm::kernels::dsv3MinLatencyKernels::invokeRouterGemmMma<kNumTokens, 896, 7168>(            \
        float*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);
INSTANTIATE_ROUTER_GEMM_MMA_K3(1)
INSTANTIATE_ROUTER_GEMM_MMA_K3(2)
INSTANTIATE_ROUTER_GEMM_MMA_K3(3)
INSTANTIATE_ROUTER_GEMM_MMA_K3(4)
INSTANTIATE_ROUTER_GEMM_MMA_K3(5)
INSTANTIATE_ROUTER_GEMM_MMA_K3(6)
INSTANTIATE_ROUTER_GEMM_MMA_K3(7)
INSTANTIATE_ROUTER_GEMM_MMA_K3(8)
INSTANTIATE_ROUTER_GEMM_MMA_K3(9)
INSTANTIATE_ROUTER_GEMM_MMA_K3(10)
INSTANTIATE_ROUTER_GEMM_MMA_K3(11)
INSTANTIATE_ROUTER_GEMM_MMA_K3(12)
INSTANTIATE_ROUTER_GEMM_MMA_K3(13)
INSTANTIATE_ROUTER_GEMM_MMA_K3(14)
INSTANTIATE_ROUTER_GEMM_MMA_K3(15)
INSTANTIATE_ROUTER_GEMM_MMA_K3(16)
#undef INSTANTIATE_ROUTER_GEMM_MMA_K3

} // namespace kernels::dsv3MinLatencyKernels

TRTLLM_NAMESPACE_END
