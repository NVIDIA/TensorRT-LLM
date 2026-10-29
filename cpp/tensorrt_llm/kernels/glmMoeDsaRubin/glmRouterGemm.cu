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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmRouterGemm.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

namespace
{

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

// mat_a: the activations [kNumTokens, kHiddenDim]; mat_b: the weight [kNumExperts, kHiddenDim]; out: the logits
// [kNumTokens, kNumExperts]. Before the grid-dependency wait each CTA prefetches its weight row into L2 (no kernel of
// the step writes the weight), so the weight loads after the wait hit L2; the dependent grid may launch right after the
// wait instead of after the logits are written, and must wait on its own grid dependency before reading them.
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

    // The weight row of expert n_idx (a column of B in column-major order)
    T const* b_col = mat_b + n_idx * kHiddenDim;

    // Pre-compute k_base values for each iteration to help compiler optimize
    int k_bases[k_iterations];
#pragma unroll
    for (int ki = 0; ki < k_iterations; ki++)
    {
        k_bases[ki] = ki * k_elems_per_k_iteration + tid * VPT;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    {
        // The primary grid does not write the weights, so this CTA's column can stream into L2 while the primary
        // runs; the weight loads below then hit L2. What is loaded and the accumulation order are unchanged.
        constexpr int kLineBytes = 128;
        constexpr int kColumnBytes = kHiddenDim * static_cast<int>(sizeof(T));
        static_assert(kColumnBytes % kLineBytes == 0, "the weight column must span whole 128-byte lines");
        char const* b_col_bytes = reinterpret_cast<char const*>(b_col);
        for (int offset = tid * kLineBytes; offset < kColumnBytes; offset += kBlockSize * kLineBytes)
        {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(b_col_bytes + offset) : "memory");
        }
    }
    cudaGridDependencySynchronize();
    // Every dependent waits on its own grid dependency before it reads the logits, so it may launch now and overlap
    // its prologue with this GEMV.
    cudaTriggerProgrammaticLaunchCompletion();
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
}

template <int kNumTokens>
void launchRouterGemm(
    float* logits, __nv_bfloat16 const* hiddenStates, __nv_bfloat16 const* weight, cudaStream_t stream)
{
    using T = __nv_bfloat16;
    constexpr int VPT = 16 / sizeof(T);
    constexpr int kBlockSize = 128;
    cudaLaunchConfig_t config;
    config.gridDim = kRouterNumExperts;
    config.blockDim = kBlockSize;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config,
        router_gemm_kernel<T, kBlockSize, VPT, kNumTokens, kRouterNumExperts, kRouterHiddenSize>, logits, hiddenStates,
        weight));
}

// ---------------------------------------------------------------------------------------------------------------------
// Router GEMM on tensor cores for 17-32 tokens, 256 experts, hidden 6144.
//
// A CTA owns 16 experts and one of 8 K slices of 768; the 8 CTAs of a cluster own the same experts. The CTA copies its
// weight tile (16 x 768) and the slice of every token's activations to shared memory, and each of its 4 warps runs
// mma.sync m16n8k16 (bf16 inputs, fp32 accumulation) over its 12 k-steps. The 4 warp tiles are added in warp order,
// sent to the shared memory of the cluster's rank 0, and rank 0 adds the 8 slices in rank order and writes the logits.
// The summation order is fixed, so the logits are deterministic; it differs from cuBLAS and from router_gemm_kernel.
// The weight tile is copied before the grid-dependency wait (router_gemm_kernel prefetches its weights into L2 there),
// and the dependent grid may launch right after that wait.
namespace routerMma
{
constexpr int kHiddenDim = 6144;
constexpr int kNumExperts = 256;
// Experts per CTA: the M of one mma tile.
constexpr int kExpertsPerCta = 16;
// K slices of 768, one per CTA of a cluster.
constexpr int kSlices = 8;
constexpr int kSliceK = kHiddenDim / kSlices;
// Row pitch of the shared-memory tiles: the 8 rows of an ldmatrix start in distinct banks.
constexpr int kRowElems = kSliceK + 8;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kKStepsPerWarp = kSliceK / 16 / kWarps;
// 16-byte chunks per row of a slice.
constexpr int kChunks = kSliceK / 8;
constexpr int kNumCtas = kNumExperts / kExpertsPerCta * kSlices;
constexpr int kMinTokens = 17;
constexpr int kMaxTokens = 32;

__host__ __device__ constexpr int paddedTokens(int numTokens)
{
    return (numTokens + 7) / 8 * 8;
}

// Weight tile and activation rows (bf16), the 4 warp tiles and the 8 slice tiles received by rank 0 (fp32).
__host__ __device__ constexpr int smemBytes(int numTokens)
{
    return (kExpertsPerCta + paddedTokens(numTokens)) * kRowElems * 2
        + (kWarps + kSlices) * kExpertsPerCta * paddedTokens(numTokens) * 4;
}
} // namespace routerMma

namespace
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ uint32_t sharedAddress(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void copyAsync16(void* dst, void const* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sharedAddress(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void copyAsyncCommit()
{
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void copyAsyncWaitAll()
{
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void ldmatrixX4(uint32_t (&r)[4], void const* ptr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sharedAddress(ptr)));
}

__device__ __forceinline__ void ldmatrixX2(uint32_t (&r)[2], void const* ptr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(sharedAddress(ptr)));
}

__device__ __forceinline__ void mmaBf16(float (&c)[4], uint32_t const (&a)[4], uint32_t const (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void clusterArriveRelaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterWait()
{
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterArriveRelease()
{
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterWaitAcquire()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Stores v at the same shared-memory offset as ptr in the CTA of the given rank of this cluster.
__device__ __forceinline__ void storeToClusterRank(float* ptr, uint32_t rank, float v)
{
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(sharedAddress(ptr)), "r"(rank));
    asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(remote), "f"(v) : "memory");
}
#endif
} // namespace

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
template <int kNumTokens, bool kEarlyTrigger, bool kPrefetchWeights>
__global__ void __launch_bounds__(routerMma::kThreads, 1)
    router_gemm_mma_kernel(float* out, __nv_bfloat16 const* mat_a, __nv_bfloat16 const* mat_b)
{
    using namespace routerMma;
    constexpr int kPadded = paddedTokens(kNumTokens);
    constexpr int kTokenTiles = kPadded / 8;
    constexpr int kTile = kExpertsPerCta * kPadded;
    // Shared memory: the weight tile [kExpertsPerCta][kRowElems] and the activation rows [kPadded][kRowElems] (bf16),
    // the warp tiles [kWarps][kExpertsPerCta][kPadded] and, used in rank 0, the slice tiles
    // [kSlices][kExpertsPerCta][kPadded] (fp32).
    extern __shared__ __align__(128) uint8_t smemMma[];
    __nv_bfloat16* weightTile = reinterpret_cast<__nv_bfloat16*>(smemMma);
    __nv_bfloat16* actTile = weightTile + kExpertsPerCta * kRowElems;
    float* warpSums = reinterpret_cast<float*>(actTile + kPadded * kRowElems);
    float* sliceSums = warpSums + kWarps * kTile;

    // First phase of the cluster barrier: arrive now, wait before the first remote store, so that every CTA of the
    // cluster has started by then.
    clusterArriveRelaxed();
    int const slice = blockIdx.x % kSlices; // rank in the cluster
    int const expertBase = blockIdx.x / kSlices * kExpertsPerCta;
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;

    auto const copyWeightTile = [&]()
    {
        for (int c = threadIdx.x; c < kExpertsPerCta * kChunks; c += kThreads)
        {
            int const row = c / kChunks;
            int const chunk = c % kChunks;
            copyAsync16(weightTile + row * kRowElems + chunk * 8,
                mat_b + static_cast<size_t>(expertBase + row) * kHiddenDim + slice * kSliceK + chunk * 8);
        }
        copyAsyncCommit();
    };
    if constexpr (kPrefetchWeights)
    {
        // The primary grid does not write the weights.
        copyWeightTile();
    }
    // Rows past the last token are zero; they only feed outputs that are not stored.
    for (int c = threadIdx.x; c < (kPadded - kNumTokens) * kChunks; c += kThreads)
    {
        *reinterpret_cast<uint4*>(actTile + (kNumTokens + c / kChunks) * kRowElems + (c % kChunks) * 8)
            = make_uint4(0, 0, 0, 0);
    }
    cudaGridDependencySynchronize();
    if constexpr (kEarlyTrigger)
    {
        // Every dependent waits on its own grid dependency before it reads the logits.
        cudaTriggerProgrammaticLaunchCompletion();
    }
    if constexpr (!kPrefetchWeights)
    {
        copyWeightTile();
    }
    for (int c = threadIdx.x; c < kNumTokens * kChunks; c += kThreads)
    {
        int const row = c / kChunks;
        int const chunk = c % kChunks;
        copyAsync16(actTile + row * kRowElems + chunk * 8,
            mat_a + static_cast<size_t>(row) * kHiddenDim + slice * kSliceK + chunk * 8);
    }
    copyAsyncCommit();
    copyAsyncWaitAll();
    __syncthreads();

    float acc[kTokenTiles][4] = {};
#pragma unroll
    for (int s = 0; s < kKStepsPerWarp; s++)
    {
        int const k0 = (warp * kKStepsPerWarp + s) * 16;
        uint32_t a[4];
        ldmatrixX4(a, weightTile + (lane % 16) * kRowElems + k0 + (lane / 16) * 8);
#pragma unroll
        for (int j = 0; j < kTokenTiles; j++)
        {
            uint32_t b[2];
            ldmatrixX2(b, actTile + (j * 8 + lane % 8) * kRowElems + k0 + ((lane / 8) % 2) * 8);
            mmaBf16(acc[j], a, b);
        }
    }
    // m16n8 accumulators: acc[j][0..1] at expert lane / 4, tokens 8 j + 2 (lane % 4) + 0..1; acc[j][2..3] 8 experts on.
#pragma unroll
    for (int j = 0; j < kTokenTiles; j++)
    {
        float* row = warpSums + (warp * kExpertsPerCta + lane / 4) * kPadded + j * 8 + 2 * (lane % 4);
        *reinterpret_cast<float2*>(row) = make_float2(acc[j][0], acc[j][1]);
        *reinterpret_cast<float2*>(row + 8 * kPadded) = make_float2(acc[j][2], acc[j][3]);
    }
    __syncthreads();
    clusterWait();
    for (int i = threadIdx.x; i < kTile; i += kThreads)
    {
        if (i % kPadded < kNumTokens)
        {
            float const sum = ((warpSums[i] + warpSums[kTile + i]) + warpSums[2 * kTile + i]) + warpSums[3 * kTile + i];
            storeToClusterRank(&sliceSums[slice * kTile + i], 0, sum);
        }
    }
    // Second phase: rank 0 waits for the slices; the other CTAs are done (nothing reads their shared memory).
    clusterArriveRelease();
    if (slice != 0)
    {
        return;
    }
    clusterWaitAcquire();
    for (int p = threadIdx.x; p < kExpertsPerCta * kNumTokens; p += kThreads)
    {
        int const e = p / kNumTokens;
        int const m = p % kNumTokens;
        float sum = sliceSums[e * kPadded + m];
#pragma unroll
        for (int r = 1; r < kSlices; r++)
        {
            sum += sliceSums[r * kTile + e * kPadded + m];
        }
        out[m * kNumExperts + expertBase + e] = sum;
    }
    if constexpr (!kEarlyTrigger)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}
#else
template <int kNumTokens, bool kEarlyTrigger, bool kPrefetchWeights>
__global__ void __launch_bounds__(routerMma::kThreads, 1)
    router_gemm_mma_kernel(float* /* out */, __nv_bfloat16 const* /* mat_a */, __nv_bfloat16 const* /* mat_b */)
{
    assert(false && "router_gemm_mma_kernel is only supported on SM100+ architectures");
}
#endif

template <typename T, int kNumTokens, int kNumExperts, int kHiddenDim>
void invokeRouterGemmMma(float* output, T const* mat_a, T const* mat_b, cudaStream_t stream)
{
    static_assert(std::is_same_v<T, __nv_bfloat16>, "the tensor-core router GEMM takes bf16 inputs");
    static_assert(kNumExperts == routerMma::kNumExperts && kHiddenDim == routerMma::kHiddenDim,
        "the tensor-core router GEMM covers 256 experts and hidden size 6144");
    static_assert(kNumTokens >= routerMma::kMinTokens && kNumTokens <= routerMma::kMaxTokens,
        "the tensor-core router GEMM covers 17-32 tokens");
    constexpr int kSmemBytes = routerMma::smemBytes(kNumTokens);
    void (*kernel)(float*, __nv_bfloat16 const*, __nv_bfloat16 const*) = router_gemm_mma_kernel<kNumTokens, true, true>;
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    cudaLaunchConfig_t config;
    config.gridDim = routerMma::kNumCtas;
    config.blockDim = routerMma::kThreads;
    config.dynamicSmemBytes = kSmemBytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim.x = routerMma::kSlices;
    attrs[1].val.clusterDim.y = 1;
    attrs[1].val.clusterDim.z = 1;
    config.numAttrs = 2;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, output, mat_a, mat_b));
}

// Launches the instance of numTokens, one of kNumTokens..kRouterMaxTokens.
template <int kNumTokens>
void dispatchRouterGemm(
    float* logits, __nv_bfloat16 const* hiddenStates, __nv_bfloat16 const* weight, int numTokens, cudaStream_t stream)
{
    if constexpr (kNumTokens <= kRouterMaxTokens)
    {
        if (numTokens == kNumTokens)
        {
            if constexpr (kNumTokens <= 16)
            {
                launchRouterGemm<kNumTokens>(logits, hiddenStates, weight, stream);
            }
            else
            {
                invokeRouterGemmMma<__nv_bfloat16, kNumTokens, kRouterNumExperts, kRouterHiddenSize>(
                    logits, hiddenStates, weight, stream);
            }
        }
        else
        {
            dispatchRouterGemm<kNumTokens + 1>(logits, hiddenStates, weight, numTokens, stream);
        }
    }
    else
    {
        TLLM_THROW("The GLM router GEMV takes 1 to %d tokens, got %d", kRouterMaxTokens, numTokens);
    }
}

} // namespace

void invokeRouterGemm(
    float* logits, __nv_bfloat16 const* hiddenStates, __nv_bfloat16 const* weight, int numTokens, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        numTokens >= 1, "The GLM router GEMV takes 1 to %d tokens, got %d", kRouterMaxTokens, numTokens);
    dispatchRouterGemm<1>(logits, hiddenStates, weight, numTokens, stream);
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
