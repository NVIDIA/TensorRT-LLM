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

// Fused DiT LayerNorm + optional AdaLN/affine + optional NVFP4 quantization kernel.
// Three mode combos × two output dtypes = 6 compile-time instantiations; see
// launchFusedDiTLayerNormShiftScaleKernel and fusedDiTLayerNormShiftScaleKernel.h.

#include "fusedDiTLayerNormShiftScaleKernel.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// TMA / mbarrier helpers (shared with fusedDiTGateResidNormShiftScaleKernel).

__device__ __forceinline__ uint32_t cvta_to_smem(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void*>(ptr)));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count)
{
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : : "r"(cvta_to_smem(bar)), "r"(count));
#endif
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar)
{
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(cvta_to_smem(bar)));
#endif
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes)
{
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" : : "r"(cvta_to_smem(bar)), "r"(tx_bytes));
#endif
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase)
{
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile(
        "{ .reg .pred P;                                                   \n"
        "  WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;   \n"
        "  @P bra DONE;                                                    \n"
        "  bra WAIT;                                                       \n"
        "  DONE: }"
        :
        : "r"(cvta_to_smem(bar)), "r"(phase));
#endif
}

__device__ __forceinline__ void cp_async_bulk(void* smem_dst, void const* global_src, uint32_t bytes, uint64_t* bar)
{
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :
        : "r"(cvta_to_smem(smem_dst)), "l"(reinterpret_cast<uint64_t>(global_src)), "r"(bytes), "r"(cvta_to_smem(bar))
        : "memory");
#endif
}

} // anonymous namespace

template <int D, int BLOCK_SIZE, bool HAS_LN_AFFINE, bool HAS_MODULATION, bool HAS_QUANT>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(BLOCK_SIZE, 4)
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__launch_bounds__(BLOCK_SIZE, 8)
#endif
    fusedDiTLayerNormShiftScaleKernel(DiTLayerNormShiftScaleParams p)
{
    static_assert(D % 8 == 0, "D must be a multiple of 8");
    static_assert(D % 16 == 0, "D must be a multiple of 16 (NVFP4 SF group size)");
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size");
    static_assert(!HAS_LN_AFFINE || !HAS_MODULATION, "HAS_LN_AFFINE and HAS_MODULATION are mutually exclusive");

    constexpr int ELEMS_PER_THREAD = D / BLOCK_SIZE;        // 40 for D=5120, BLOCK_SIZE=128
    constexpr int CHUNKS_PER_THREAD = ELEMS_PER_THREAD / 8; // 5
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;              // 4
    constexpr int SF_VEC_SIZE = 16;
    constexpr int SF_PER_ROW = D / SF_VEC_SIZE;             // 320
    constexpr int NUM_THREADS_PER_SF = SF_VEC_SIZE / 8;     // 2

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif

    // TMA is beneficial for large rows on sm>=900
    constexpr bool USE_TMA = (D >= 4096);

    int const tid = threadIdx.x;
    int const warpId = tid / 32;
    int const laneId = tid % 32;
    int const row = blockIdx.x;

    int64_t const rowBase = static_cast<int64_t>(row) * D;

    // Shared memory for cross-warp reductions and mean/rstd broadcast.
    __shared__ float warpSums[NUM_WARPS];
    __shared__ float warpSqSums[NUM_WARPS];
    __shared__ float meanRstd[2];

    // Static mbarrier for TMA (compiler elides on non-TMA paths).
    __shared__ alignas(8) uint64_t mbar;

    // Dynamic SMEM: holds one row of bf16 x when USE_TMA; zero-sized otherwise.
    // The launcher passes dyn_smem = USE_TMA ? D * sizeof(bf16) : 0.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // Phase 0a: bulk-load x[row] into SMEM while Phase 0b loads modulators.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (USE_TMA)
    {
        constexpr uint32_t kXBytes = static_cast<uint32_t>(D) * sizeof(__nv_bfloat16);
        static_assert(kXBytes % 16 == 0, "cp.async.bulk requires nbBytes multiple of 16");
        if (tid == 0)
        {
            mbar_init(&mbar, 1);
            mbar_arrive_expect_tx(&mbar, kXBytes);
            cp_async_bulk(smem_x, p.x + rowBase, kXBytes, &mbar);
        }
        __syncthreads(); // ensure mbar is initialized before all threads see it
    }
    else
#endif
    {
        // Non-TMA path: cp.async into SMEM (fallback, not production path for D=5120).
#pragma unroll
        for (int chunk = 0; chunk < CHUNKS_PER_THREAD; ++chunk)
        {
            int const vecIdx = chunk * BLOCK_SIZE + tid;
            int const elemOff = vecIdx * 8;
            __pipeline_memcpy_async(smem_x + elemOff, p.x + rowBase + elemOff, 16);
        }
        __pipeline_commit();
    }

    // Phase 0b: load LN affine or AdaLN modulator rows into register caches.
    float wVals[ELEMS_PER_THREAD]; // weight (or 1 + scale_msa)
    float bVals[ELEMS_PER_THREAD]; // bias   (or shift_msa)

    if constexpr (HAS_LN_AFFINE)
    {
        // ln_weight[D], ln_bias[D] -- direct GMEM reads while waiting for SMEM fill.
#pragma unroll
        for (int chunk = 0; chunk < CHUNKS_PER_THREAD; ++chunk)
        {
            int const vecIdx = chunk * BLOCK_SIZE + tid;
            int const elemOff = vecIdx * 8;
            uint4 const wVec = *reinterpret_cast<uint4 const*>(p.ln_weight + elemOff);
            uint4 const bVec = *reinterpret_cast<uint4 const*>(p.ln_bias + elemOff);
            __nv_bfloat162 const* wVec2 = reinterpret_cast<__nv_bfloat162 const*>(&wVec);
            __nv_bfloat162 const* bVec2 = reinterpret_cast<__nv_bfloat162 const*>(&bVec);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 wf = __bfloat1622float2(wVec2[i]);
                float2 bf = __bfloat1622float2(bVec2[i]);
                wVals[chunk * 8 + i * 2 + 0] = wf.x;
                wVals[chunk * 8 + i * 2 + 1] = wf.y;
                bVals[chunk * 8 + i * 2 + 0] = bf.x;
                bVals[chunk * 8 + i * 2 + 1] = bf.y;
            }
        }
    }
    else if constexpr (HAS_MODULATION)
    {
        // scale_msa[B, D], shift_msa[B, D] -- batch_idx = row / seq_len_per_batch.
        int const batchIdx = row / p.seq_len_per_batch;
        int64_t const modBase = static_cast<int64_t>(batchIdx) * D;
#pragma unroll
        for (int chunk = 0; chunk < CHUNKS_PER_THREAD; ++chunk)
        {
            int const vecIdx = chunk * BLOCK_SIZE + tid;
            int const elemOff = vecIdx * 8;
            uint4 const sVec = *reinterpret_cast<uint4 const*>(p.scale_msa + modBase + elemOff);
            uint4 const shVec = *reinterpret_cast<uint4 const*>(p.shift_msa + modBase + elemOff);
            __nv_bfloat162 const* sVec2 = reinterpret_cast<__nv_bfloat162 const*>(&sVec);
            __nv_bfloat162 const* shVec2 = reinterpret_cast<__nv_bfloat162 const*>(&shVec);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 sf = __bfloat1622float2(sVec2[i]);
                float2 shf = __bfloat1622float2(shVec2[i]);
                // AdaLN: y = normalized * (1 + scale_msa) + shift_msa.
                // Fold +1 into wVals so Phase 2 is one fma.
                wVals[chunk * 8 + i * 2 + 0] = 1.0f + sf.x;
                wVals[chunk * 8 + i * 2 + 1] = 1.0f + sf.y;
                bVals[chunk * 8 + i * 2 + 0] = shf.x;
                bVals[chunk * 8 + i * 2 + 1] = shf.y;
            }
        }
    }

    // Phase 0c: wait for x load, then read x chunks into xVals registers.
    float xVals[ELEMS_PER_THREAD];

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (USE_TMA)
    {
        mbar_wait(&mbar, /*phase=*/0);
        __syncthreads();
        // Read x chunks from SMEM.
#pragma unroll
        for (int chunk = 0; chunk < CHUNKS_PER_THREAD; ++chunk)
        {
            int const vecIdx = chunk * BLOCK_SIZE + tid;
            int const elemOff = vecIdx * 8;
            uint4 const xVec = *reinterpret_cast<uint4 const*>(smem_x + elemOff);
            __nv_bfloat162 const* xVec2 = reinterpret_cast<__nv_bfloat162 const*>(&xVec);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 xf = __bfloat1622float2(xVec2[i]);
                xVals[chunk * 8 + i * 2 + 0] = xf.x;
                xVals[chunk * 8 + i * 2 + 1] = xf.y;
            }
        }
    }
    else
#endif
    {
        __pipeline_wait_prior(0);
        __syncthreads();
        // Read x chunks from SMEM (non-TMA path).
#pragma unroll
        for (int chunk = 0; chunk < CHUNKS_PER_THREAD; ++chunk)
        {
            int const vecIdx = chunk * BLOCK_SIZE + tid;
            int const elemOff = vecIdx * 8;
            uint4 const xVec = *reinterpret_cast<uint4 const*>(smem_x + elemOff);
            __nv_bfloat162 const* xVec2 = reinterpret_cast<__nv_bfloat162 const*>(&xVec);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                float2 xf = __bfloat1622float2(xVec2[i]);
                xVals[chunk * 8 + i * 2 + 0] = xf.x;
                xVals[chunk * 8 + i * 2 + 1] = xf.y;
            }
        }
    }

    // Phase 1: warp-reduce sum/sum-of-squares → mean and rstd (LayerNorm, not RMSNorm).
    float localSum = 0.0f;
    float localSqSum = 0.0f;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i)
    {
        localSum += xVals[i];
        localSqSum += xVals[i] * xVals[i];
    }

    // Warp-level reduction.
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

    // Cross-warp reduction in warp 0.
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
            float const invD = 1.0f / static_cast<float>(D);
            float const mean = s * invD;
            float const var = s2 * invD - mean * mean;
            meanRstd[0] = mean;
            meanRstd[1] = rsqrtf(var + p.eps);
        }
    }
    __syncthreads();

    float const mean = meanRstd[0];
    float const rstd = meanRstd[1];

    // Pre-read sf_scale scalar (HAS_QUANT only).
    float sfScaleVal = 1.0f;
    if constexpr (HAS_QUANT)
    {
        sfScaleVal = (p.sf_scale != nullptr) ? p.sf_scale[0] : 1.0f;
    }

    // Phase 2: normalize, apply affine/modulation, write output.
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_PER_THREAD; ++chunk)
    {
        int const vecIdx = chunk * BLOCK_SIZE + tid;
        int const elemOff = vecIdx * 8;

        // Normalize and apply affine/modulation, producing 8 float yVals.
        float yVals[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            float const xn = (xVals[chunk * 8 + i] - mean) * rstd;
            if constexpr (HAS_LN_AFFINE || HAS_MODULATION)
            {
                yVals[i] = xn * wVals[chunk * 8 + i] + bVals[chunk * 8 + i];
            }
            else
            {
                yVals[i] = xn;
            }
        }

        if constexpr (HAS_QUANT)
        {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            // FP4 quantization path (Blackwell+).
            // 1. Compute local max-abs across 8 elements held by this thread.
            float localMax = fabsf(yVals[0]);
#pragma unroll
            for (int i = 1; i < 8; ++i)
            {
                localMax = fmaxf(localMax, fabsf(yVals[i]));
            }

            // 2. Pair-lane max across the 16-element SF block (threads pair via XOR-1).
            float const blockMax = fmaxf(__shfl_xor_sync(0xffffffff, localMax, 1), localMax);

            // 3. Compute FP8-e4m3 scale factor (same formula as LTX-2 kernel).
            constexpr float kE2M1MaxRcp = 1.0f / 6.0f;
            float const sfValue = sfScaleVal * (blockMax * kE2M1MaxRcp);
            __nv_fp8_e4m3 const sfFp8 = __nv_fp8_e4m3(sfValue);
            uint8_t const sfByte = sfFp8.__x;
            float const sfValueQuant = static_cast<float>(sfFp8);
            float const outScale = (blockMax != 0.0f) ? (sfScaleVal / sfValueQuant) : 0.0f;

            // 4. Scale yVals and convert to packed FP4 (e2m1).
#pragma unroll
            for (int i = 0; i < 8; ++i)
                yVals[i] *= outScale;
            uint32_t const fp4Packed = fp32_vec_to_e2m1(yVals);

            // 5. Write SF via swizzled layout helper.
            uint8_t* sfOutPtr = cvt_quant_get_sf_out_offset<uint32_t, NUM_THREADS_PER_SF>(std::nullopt, row, vecIdx,
                std::optional<int>(p.M), SF_PER_ROW, reinterpret_cast<uint32_t*>(p.out_sf),
                QuantizationSFLayout::SWIZZLED);
            if (sfOutPtr != nullptr)
            {
                *sfOutPtr = sfByte;
            }

            // 6. Write packed FP4.
            int64_t const fp4Off = static_cast<int64_t>(row) * (D / 8) + vecIdx;
            p.out_fp4[fp4Off] = fp4Packed;
#endif // __CUDA_ARCH__ >= 1000
        }
        else
        {
            // bf16 output path.
            uint4 outVec;
            __nv_bfloat162* outVec2 = reinterpret_cast<__nv_bfloat162*>(&outVec);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                outVec2[i] = __float22bfloat162_rn(make_float2(yVals[i * 2 + 0], yVals[i * 2 + 1]));
            }
            *reinterpret_cast<uint4*>(p.out_bf16 + rowBase + elemOff) = outVec;
        }
    }

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void launchFusedDiTLayerNormShiftScaleKernel(DiTLayerNormShiftScaleParams const& params, bool has_ln_affine,
    bool has_modulation, bool has_quant, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.D == 5120, "fusedDiTLayerNormShiftScaleKernel only supports D=5120 (got %d)", params.D);
    TLLM_CHECK_WITH_INFO(!(has_ln_affine && has_modulation), "has_ln_affine and has_modulation are mutually exclusive");

    constexpr int D = 5120;
    constexpr int BLOCK_SIZE = 128;
    constexpr size_t dynSmem = D * sizeof(__nv_bfloat16);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(params.M);
    cfg.blockDim = dim3(BLOCK_SIZE);
    cfg.dynamicSmemBytes = dynSmem;
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

#define LAUNCH(AFF, MOD, QUANT)                                                                                        \
    cudaLaunchKernelEx(&cfg, fusedDiTLayerNormShiftScaleKernel<D, BLOCK_SIZE, AFF, MOD, QUANT>, params)

    if (!has_ln_affine && !has_modulation && !has_quant)
        LAUNCH(false, false, false);
    else if (!has_ln_affine && !has_modulation && has_quant)
        LAUNCH(false, false, true);
    else if (has_ln_affine && !has_modulation && !has_quant)
        LAUNCH(true, false, false);
    else if (has_ln_affine && !has_modulation && has_quant)
        LAUNCH(true, false, true);
    else if (!has_ln_affine && has_modulation && !has_quant)
        LAUNCH(false, true, false);
    else if (!has_ln_affine && has_modulation && has_quant)
        LAUNCH(false, true, true);
    else
        TLLM_CHECK_WITH_INFO(false, "Unsupported flag combination in launchFusedDiTLayerNormShiftScaleKernel");

#undef LAUNCH

    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
