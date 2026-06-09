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

#include "fusedDiTNormKernel.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

__device__ __forceinline__ uint32_t cvta_to_smem(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void*>(ptr)));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : : "r"(cvta_to_smem(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(cvta_to_smem(bar)));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" : : "r"(cvta_to_smem(bar)), "r"(tx_bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase)
{
    asm volatile(
        "{ .reg .pred P;                                                   \n"
        "  WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;     \n"
        "  @P bra DONE;                                                    \n"
        "  bra WAIT;                                                       \n"
        "  DONE: }"
        :
        : "r"(cvta_to_smem(bar)), "r"(phase));
}

__device__ __forceinline__ void cp_async_bulk(void* smem_dst, void const* global_src, uint32_t bytes, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :
        : "r"(cvta_to_smem(smem_dst)), "l"(reinterpret_cast<uint64_t>(global_src)), "r"(bytes), "r"(cvta_to_smem(bar))
        : "memory");
}

// Sync only the consumer warpgroup on named barrier 1; producer warp must not participate.
__device__ __forceinline__ void bar_sync_consumer(int count)
{
    asm volatile("bar.sync 1, %0;" : : "r"(count));
}

// combine_modulator_chunk: combine 8 bf16 ts + 8 fp32 table into 8 bf16 (one uint4).
// Matches PyTorch eager `_get_ada_values` semantics: narrow fp32 table to
// bf16 FIRST, then bf16 hw add. Used for gate / scale / shift modulators alike.
__device__ __forceinline__ uint4 combine_modulator_chunk(uint4 const& ts_v, float4 const& tbl_lo, float4 const& tbl_hi)
{
    __nv_bfloat162 const* ts_b2 = reinterpret_cast<__nv_bfloat162 const*>(&ts_v);
    __nv_bfloat162 out_b2[4];
    out_b2[0] = __hadd2(__float22bfloat162_rn(make_float2(tbl_lo.x, tbl_lo.y)), ts_b2[0]);
    out_b2[1] = __hadd2(__float22bfloat162_rn(make_float2(tbl_lo.z, tbl_lo.w)), ts_b2[1]);
    out_b2[2] = __hadd2(__float22bfloat162_rn(make_float2(tbl_hi.x, tbl_hi.y)), ts_b2[2]);
    out_b2[3] = __hadd2(__float22bfloat162_rn(make_float2(tbl_hi.z, tbl_hi.w)), ts_b2[3]);
    uint4 out_v;
    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&out_v) + 0) = out_b2[0];
    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&out_v) + 1) = out_b2[1];
    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&out_v) + 2) = out_b2[2];
    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&out_v) + 3) = out_b2[3];
    return out_v;
}

} // anonymous namespace

template <int D, int ROWS_PER_BLOCK, int BLOCK_SIZE, bool HAS_RESIDUAL, bool HAS_GATE, bool HAS_MODULATE, int NUM_OUT,
    bool HAS_QUANT>
__global__ void fusedDiTNormKernel(AdaLNNormParams p)
{
    static_assert(NUM_OUT == 1 || NUM_OUT == 2, "NUM_OUT must be 1 or 2");
    static_assert(!HAS_GATE || HAS_RESIDUAL, "HAS_GATE requires HAS_RESIDUAL");

    constexpr int THREADS_PER_ROW = BLOCK_SIZE / ROWS_PER_BLOCK;
    constexpr int WARPS_PER_ROW = THREADS_PER_ROW / 32;
    constexpr int CHUNK_ELEMS = 8; // uint4 = 8 bf16
    constexpr int CHUNKS_PER_ROW = (D + THREADS_PER_ROW * CHUNK_ELEMS - 1) / (THREADS_PER_ROW * CHUNK_ELEMS);
    constexpr int SF_VEC_SIZE = 16;
    constexpr int SF_PER_ROW = D / SF_VEC_SIZE;

    static_assert(D % CHUNK_ELEMS == 0, "D must be multiple of 8");
    static_assert(D % SF_VEC_SIZE == 0, "D must be multiple of 16 (NVFP4 group size)");

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif

    int const tid = threadIdx.x;
    int const row_in_block = tid / THREADS_PER_ROW;
    int const lane_in_row = tid % THREADS_PER_ROW;
    int const row_warp = lane_in_row >> 5;
    int const row_lane = lane_in_row & 31;

    int const tokenIdx = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    bool const valid = (tokenIdx < p.num_tokens);
    int const safeTokenIdx = valid ? tokenIdx : 0;

    int const batchIdx = safeTokenIdx / p.tokens_per_batch;
    int64_t const tokenBase = static_cast<int64_t>(safeTokenIdx) * D;

    // Hybrid TMA / cp.async gating, picked empirically from paired bench on B200:
    //   USE_TMA       : single-instruction cp.async.bulk for the X load. Wins on D=4096
    //                   except when (HAS_QUANT && HAS_MODULATE && NUM_OUT==1) where the
    //                   kernel is HBM-light and the mbarrier setup cost exceeds the LSU
    //                   saving. Disabled on D=2048 (audio path, 2-4us kernels).
    //   USE_TMA_ATTN  : also TMA-bulk the residual `attn` tensor into smem. Only enabled
    //                   on (USE_TMA && HAS_RESIDUAL && HAS_QUANT) -- bf16 paths regress
    //                   because the per-thread LDG for attn was already overlapping
    //                   the TMA-X load via different LSU pipes, while quant's heavy
    //                   Phase 2 hides the extra mbarrier wait.
    constexpr bool USE_TMA = (D >= 4096) && !(HAS_QUANT && HAS_MODULATE && NUM_OUT == 1);
    constexpr bool USE_TMA_ATTN = USE_TMA && HAS_RESIDUAL && HAS_QUANT;
    constexpr int kAttnSmemBytes = USE_TMA_ATTN ? (ROWS_PER_BLOCK * D * static_cast<int>(sizeof(__nv_bfloat16))) : 0;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_attn = reinterpret_cast<__nv_bfloat16*>(smem_raw + ROWS_PER_BLOCK * D * sizeof(__nv_bfloat16));
    float* warp_sums = reinterpret_cast<float*>(smem_raw + ROWS_PER_BLOCK * D * sizeof(__nv_bfloat16) + kAttnSmemBytes);

    // mbarrier in static SMEM (8B), only used when USE_TMA. Compiler elides when unused.
    __shared__ alignas(8) uint64_t mbar;

    // Phase 0a: load X -> SMEM. Either TMA bulk (1 instruction, mbarrier-synced) or
    // cp.async (32 per-thread issues, __pipeline_commit/wait_prior synced).
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    if constexpr (USE_TMA)
    {
        constexpr uint32_t kXBytes = ROWS_PER_BLOCK * D * static_cast<uint32_t>(sizeof(__nv_bfloat16));
        constexpr uint32_t kAttnBytes = USE_TMA_ATTN ? kXBytes : 0;
        constexpr uint32_t kTotalBytes = kXBytes + kAttnBytes;
        static_assert(kXBytes % 16 == 0, "cp.async.bulk requires nbBytes multiple of 16");
        if (tid == 0)
        {
            asm volatile(
                "mbarrier.init.shared.b64 [%0], 1;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar)))
                : "memory");
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                             static_cast<uint32_t>(__cvta_generic_to_shared(&mbar))),
                         "r"(kTotalBytes)
                         : "memory");
            // Bulk load X.
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"l"(
                             __cvta_generic_to_shared(smem_x + row_in_block * D)),
                         "l"(reinterpret_cast<uint64_t>(p.x + tokenBase)), "r"(kXBytes),
                         "l"(__cvta_generic_to_shared(&mbar))
                         : "memory");
            // Bulk load attn (only when USE_TMA_ATTN), into smem_attn slot.
            if constexpr (USE_TMA_ATTN)
            {
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"l"(
                        __cvta_generic_to_shared(smem_attn + row_in_block * D)),
                    "l"(reinterpret_cast<uint64_t>(p.attn + tokenBase)), "r"(kAttnBytes),
                    "l"(__cvta_generic_to_shared(&mbar))
                    : "memory");
            }
        }
        __syncthreads();
    }
    else
#endif
    {
#pragma unroll
        for (int chunk = 0; chunk < CHUNKS_PER_ROW; chunk++)
        {
            int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
            if (elemBase >= D)
                continue;
            __pipeline_memcpy_async(smem_x + row_in_block * D + elemBase, p.x + tokenBase + elemBase, 16);
        }
        __pipeline_commit();
    }

    // Phase 0b: load gate / scale / shift modulators into register caches.
    // Per-thread storage scaled by HAS_GATE / HAS_MODULATE flags; compiler elides
    // unused slots when the corresponding flag is false.
    uint4 gate_cache[HAS_GATE ? CHUNKS_PER_ROW : 1];
    uint4 scale_cache[HAS_MODULATE ? NUM_OUT * CHUNKS_PER_ROW : 1];
    uint4 shift_cache[HAS_MODULATE ? NUM_OUT * CHUNKS_PER_ROW : 1];

    if constexpr (HAS_GATE || HAS_MODULATE)
    {
        int64_t const gateBase = HAS_GATE ? static_cast<int64_t>(batchIdx) * p.gate_ts_stride : 0;
        int64_t scaleBase[NUM_OUT];
        int64_t shiftBase[NUM_OUT];
        if constexpr (HAS_MODULATE)
        {
#pragma unroll
            for (int k = 0; k < NUM_OUT; k++)
            {
                scaleBase[k] = static_cast<int64_t>(batchIdx) * p.scale_ts_stride[k];
                shiftBase[k] = static_cast<int64_t>(batchIdx) * p.shift_ts_stride[k];
            }
        }
#pragma unroll
        for (int chunk = 0; chunk < CHUNKS_PER_ROW; chunk++)
        {
            int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
            if (elemBase >= D)
                continue;
            if constexpr (HAS_GATE)
            {
                uint4 const ts_v = *reinterpret_cast<uint4 const*>(&p.gate_ts[gateBase + elemBase]);
                float4 const tbl_lo = *reinterpret_cast<float4 const*>(&p.gate_table[elemBase + 0]);
                float4 const tbl_hi = *reinterpret_cast<float4 const*>(&p.gate_table[elemBase + 4]);
                gate_cache[chunk] = combine_modulator_chunk(ts_v, tbl_lo, tbl_hi);
            }
            if constexpr (HAS_MODULATE)
            {
#pragma unroll
                for (int k = 0; k < NUM_OUT; k++)
                {
                    uint4 const s_ts_v = *reinterpret_cast<uint4 const*>(&p.scale_ts[k][scaleBase[k] + elemBase]);
                    float4 const s_tbl_lo = *reinterpret_cast<float4 const*>(&p.scale_table[k][elemBase + 0]);
                    float4 const s_tbl_hi = *reinterpret_cast<float4 const*>(&p.scale_table[k][elemBase + 4]);
                    scale_cache[k * CHUNKS_PER_ROW + chunk] = combine_modulator_chunk(s_ts_v, s_tbl_lo, s_tbl_hi);

                    uint4 const h_ts_v = *reinterpret_cast<uint4 const*>(&p.shift_ts[k][shiftBase[k] + elemBase]);
                    float4 const h_tbl_lo = *reinterpret_cast<float4 const*>(&p.shift_table[k][elemBase + 0]);
                    float4 const h_tbl_hi = *reinterpret_cast<float4 const*>(&p.shift_table[k][elemBase + 4]);
                    shift_cache[k * CHUNKS_PER_ROW + chunk] = combine_modulator_chunk(h_ts_v, h_tbl_lo, h_tbl_hi);
                }
            }
        }
    }

    // Phase 0c: load attn into regs (HAS_RESIDUAL only). Either from smem (USE_TMA_ATTN —
    // attn was bulk-loaded into smem_attn in Phase 0a) or directly from GMEM (cp.async path).
    // The smem read happens AFTER the mbarrier wait, so smem_attn is guaranteed populated.
    uint4 attn_reg[HAS_RESIDUAL ? CHUNKS_PER_ROW : 1];
    if constexpr (HAS_RESIDUAL && !USE_TMA_ATTN)
    {
#pragma unroll
        for (int chunk = 0; chunk < CHUNKS_PER_ROW; chunk++)
        {
            int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
            if (elemBase >= D)
                continue;
            attn_reg[chunk] = *reinterpret_cast<uint4 const*>(&p.attn[tokenBase + elemBase]);
        }
    }

    // Wait on Phase 0a load completion (TMA mbarrier OR cp.async pipeline, gated by USE_TMA).
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    if constexpr (USE_TMA)
    {
        uint32_t const mbar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
        asm volatile(
            "{\n"
            ".reg .pred P1;\n"
            "WAIT_LOOP:\n"
            "mbarrier.try_wait.parity.shared.b64 P1, [%0], 0;\n"
            "@P1 bra DONE;\n"
            "bra     WAIT_LOOP;\n"
            "DONE:\n"
            "}\n" ::"r"(mbar_smem)
            : "memory");
        __syncthreads();
    }
    else
#endif
    {
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    // Phase 1: compute x_new and sum^2.
    //   HAS_RESIDUAL=false:        x_new = x
    //   HAS_RESIDUAL, !HAS_GATE:   x_new = x + attn
    //   HAS_RESIDUAL,  HAS_GATE:   x_new = x + attn * gate
    // x_new is cached in regs (xnew_cache) for Phase 2 to avoid a re-read; if
    // HAS_RESIDUAL also write x_new back to x in-place so the downstream residual
    // chain sees the updated value.
    uint4 xnew_cache[CHUNKS_PER_ROW];
    float sum2 = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_PER_ROW; chunk++)
    {
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= D)
            continue;
        uint4 const xv = *reinterpret_cast<uint4 const*>(&smem_x[row_in_block * D + elemBase]);
        // USE_TMA_ATTN path: attn was TMA-bulk-loaded into smem_attn in Phase 0a;
        // populate attn_reg from smem here (post-mbarrier-wait, data is valid).
        if constexpr (USE_TMA_ATTN)
        {
            attn_reg[chunk] = *reinterpret_cast<uint4 const*>(&smem_attn[row_in_block * D + elemBase]);
        }
        uint const* xu = reinterpret_cast<uint const*>(&xv);
        uint4 new_vec;
        uint* nu = reinterpret_cast<uint*>(&new_vec);

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 xf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&xu[i]));
            float2 nf;
            if constexpr (HAS_RESIDUAL)
            {
                uint4 const av = attn_reg[chunk];
                uint const* au = reinterpret_cast<uint const*>(&av);
                float2 af = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&au[i]));
                if constexpr (HAS_GATE)
                {
                    uint4 const gv = gate_cache[chunk];
                    uint const* gu = reinterpret_cast<uint const*>(&gv);
                    float2 gf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&gu[i]));
                    nf.x = xf.x + af.x * gf.x;
                    nf.y = xf.y + af.y * gf.y;
                }
                else
                {
                    nf.x = xf.x + af.x;
                    nf.y = xf.y + af.y;
                }
            }
            else
            {
                nf = xf;
            }
            sum2 += nf.x * nf.x + nf.y * nf.y;
            __nv_bfloat162 bf = __float22bfloat162_rn(nf);
            reinterpret_cast<__nv_bfloat162&>(nu[i]) = bf;
        }
        xnew_cache[chunk] = new_vec;
        if constexpr (HAS_RESIDUAL)
        {
            if (valid)
                *reinterpret_cast<uint4*>(&p.x[tokenBase + elemBase]) = new_vec;
        }
    }

    // Per-row warp reduce + cross-warp reduce within the row.
    sum2 = tensorrt_llm::common::warpReduceSum(sum2);
    if (row_lane == 0)
        warp_sums[row_in_block * WARPS_PER_ROW + row_warp] = sum2;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS_PER_ROW; w++)
        total += warp_sums[row_in_block * WARPS_PER_ROW + w];
    float const rms_rcp = rsqrtf(total / static_cast<float>(D) + p.eps);

    // Pre-read sf_scale broadcast scalars (HAS_QUANT only).
    float sf_scale_val[NUM_OUT];
    if constexpr (HAS_QUANT)
    {
#pragma unroll
        for (int k = 0; k < NUM_OUT; k++)
            sf_scale_val[k] = (p.sf_scale[k] != nullptr) ? *p.sf_scale[k] : 1.0f;
    }

    // Phase 2: optional modulate + write outputs (NUM_OUT entries).
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_PER_ROW; chunk++)
    {
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= D)
            continue;

        uint4 const in_vec = xnew_cache[chunk];
        uint const* x_uints = reinterpret_cast<uint const*>(&in_vec);

        if constexpr (HAS_QUANT)
        {
            // FP4 quant path: fp32 modulate + max-abs scan + e2m1 pack. Inlining the
            // max scan with modulate shares the pair-lane reduction with the FP4
            // conversion (saves ~10us/call vs cvt_float_to_fp4_inline + separate scan).
            float vals[NUM_OUT][8];
            float localMax[NUM_OUT];
#pragma unroll
            for (int k = 0; k < NUM_OUT; k++)
                localMax[k] = 0.0f;

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 xv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&x_uints[i]));
                float const nx = xv.x * rms_rcp;
                float const ny = xv.y * rms_rcp;
#pragma unroll
                for (int k = 0; k < NUM_OUT; k++)
                {
                    float yx;
                    float yy;
                    if constexpr (HAS_MODULATE)
                    {
                        uint4 const sv = scale_cache[k * CHUNKS_PER_ROW + chunk];
                        uint4 const hv = shift_cache[k * CHUNKS_PER_ROW + chunk];
                        uint const* s_uints = reinterpret_cast<uint const*>(&sv);
                        uint const* h_uints = reinterpret_cast<uint const*>(&hv);
                        float2 sf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&s_uints[i]));
                        float2 hf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&h_uints[i]));
                        yx = nx * (1.0f + sf.x) + hf.x;
                        yy = ny * (1.0f + sf.y) + hf.y;
                    }
                    else
                    {
                        yx = nx;
                        yy = ny;
                    }
                    vals[k][2 * i + 0] = yx;
                    vals[k][2 * i + 1] = yy;
                    localMax[k] = fmaxf(localMax[k], fmaxf(fabsf(yx), fabsf(yy)));
                }
            }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            uint32_t fp4_packed[NUM_OUT];
            uint8_t sfBytes[NUM_OUT];
#pragma unroll
            for (int k = 0; k < NUM_OUT; k++)
            {
                // Pair-lane max-abs across 16 elements (one SF block).
                float const blockMax = fmaxf(__shfl_xor_sync(0xffffffff, localMax[k], 1), localMax[k]);
                constexpr float kE2M1MaxRcp = 1.0f / 6.0f;
                float const sfValue = sf_scale_val[k] * (blockMax * kE2M1MaxRcp);
                __nv_fp8_e4m3 const sfFp8 = __nv_fp8_e4m3(sfValue);
                sfBytes[k] = sfFp8.__x;
                float const sfQuant = static_cast<float>(sfFp8);
                float const outScale = (blockMax != 0.0f) ? (sf_scale_val[k] / sfQuant) : 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++)
                    vals[k][i] *= outScale;
                fp4_packed[k] = fp32_vec_to_e2m1(vals[k]);
            }

            int const colVecIdx = (chunk * THREADS_PER_ROW) + lane_in_row;
            uint8_t* first_sf_ptr = cvt_quant_get_sf_out_offset<uint32_t, /*CVT_NUM_THREADS_PER_SF=*/2>(std::nullopt,
                safeTokenIdx, colVecIdx, std::optional<int>(p.num_tokens), SF_PER_ROW, p.out_sf[0],
                QuantizationSFLayout::SWIZZLED);
            if (valid && first_sf_ptr != nullptr)
            {
                *first_sf_ptr = sfBytes[0];
                if constexpr (NUM_OUT == 2)
                {
                    uint8_t* second_sf_ptr = cvt_quant_get_sf_out_offset<uint32_t, /*CVT_NUM_THREADS_PER_SF=*/2>(
                        std::nullopt, safeTokenIdx, colVecIdx, std::optional<int>(p.num_tokens), SF_PER_ROW,
                        p.out_sf[1], QuantizationSFLayout::SWIZZLED);
                    *second_sf_ptr = sfBytes[1];
                }
            }
            if (valid)
            {
                int64_t const fp4_row_off = static_cast<int64_t>(safeTokenIdx) * (D / 8);
#pragma unroll
                for (int k = 0; k < NUM_OUT; k++)
                    p.out_fp4[k][fp4_row_off + colVecIdx] = fp4_packed[k];
            }
#endif
        }
        else
        {
            // bf16 modulate path: byte-matches eager apply_fused_*_modulate semantics.
            //   normed_bf16 = bf16(x_new_fp32 * rms_rcp_fp32)
            //   y_bf16      = bf16(normed_bf16 * bf16(1 + scale_bf16) + shift_bf16)   if HAS_MODULATE
            //   y_bf16      = normed_bf16                                              else
            __nv_bfloat162 const one_b2 = __float2bfloat162_rn(1.0f);
            uint4 out_vecs[NUM_OUT];
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                __nv_bfloat162 x_b2 = *reinterpret_cast<__nv_bfloat162 const*>(&x_uints[i]);
                float2 xv = __bfloat1622float2(x_b2);
                __nv_bfloat162 normed_b2 = __float22bfloat162_rn(make_float2(xv.x * rms_rcp, xv.y * rms_rcp));
#pragma unroll
                for (int k = 0; k < NUM_OUT; k++)
                {
                    uint* o_uints = reinterpret_cast<uint*>(&out_vecs[k]);
                    if constexpr (HAS_MODULATE)
                    {
                        uint4 const sv = scale_cache[k * CHUNKS_PER_ROW + chunk];
                        uint4 const hv = shift_cache[k * CHUNKS_PER_ROW + chunk];
                        uint const* s_uints = reinterpret_cast<uint const*>(&sv);
                        uint const* h_uints = reinterpret_cast<uint const*>(&hv);
                        __nv_bfloat162 s_b2 = *reinterpret_cast<__nv_bfloat162 const*>(&s_uints[i]);
                        __nv_bfloat162 h_b2 = *reinterpret_cast<__nv_bfloat162 const*>(&h_uints[i]);
                        __nv_bfloat162 one_p_s = __hadd2(one_b2, s_b2);
                        __nv_bfloat162 y_b2 = __hadd2(__hmul2(normed_b2, one_p_s), h_b2);
                        reinterpret_cast<__nv_bfloat162&>(o_uints[i]) = y_b2;
                    }
                    else
                    {
                        reinterpret_cast<__nv_bfloat162&>(o_uints[i]) = normed_b2;
                    }
                }
            }
            if (valid)
            {
#pragma unroll
                for (int k = 0; k < NUM_OUT; k++)
                    *reinterpret_cast<uint4*>(&p.out_bf16[k][tokenBase + elemBase]) = out_vecs[k];
            }
        }
    }

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Pipelined variant: multi-row CTA + circular SMEM stages + 1 producer warp / 8 consumer warps.
//
// Pipeline:
//   - Per-stage SMEM: [X (D bf16) | attn (D bf16) if HAS_RESIDUAL].
//   - full_bar[NUM_STAGES]:  producer signals "TMA done".
//   - empty_bar[NUM_STAGES]: consumers signal "slot reusable" (init=CONSUMER_WARPS arrives).
//   - Producer warp (lane 0 active): wait empty_bar, issue cp.async.bulk for X (and attn when
//     HAS_RESIDUAL), arrive full_bar with the expected tx-bytes.
//   - Consumer warps: wait full_bar, compute x_new + sum^2, normalize + modulate + write,
//     arrive empty_bar.
//
// Caller contract: tokens_per_batch >= R_CTA && tokens_per_batch % R_CTA == 0 so all R_CTA
// rows in a CTA share batchIdx and modulator load amortizes once per CTA.
template <int D, int R_CTA, int NUM_STAGES, bool HAS_RESIDUAL, bool HAS_GATE, bool HAS_MODULATE, int NUM_OUT,
    bool HAS_QUANT>
__global__ __launch_bounds__(288, 2) void fusedDiTNormKernelPipelined(AdaLNNormParams p)
{
    static_assert(D == 4096, "Pipelined variant requires D=4096");
    static_assert(R_CTA == 4, "Pipelined variant requires R_CTA=4");
    static_assert(NUM_STAGES == 2 || NUM_STAGES == 3, "NUM_STAGES must be 2 or 3");
    static_assert(NUM_OUT == 1 || NUM_OUT == 2, "NUM_OUT must be 1 or 2");
    static_assert(!HAS_GATE || HAS_RESIDUAL, "HAS_GATE requires HAS_RESIDUAL");

    constexpr int CONSUMER_WARPS = 8;
    constexpr int CONSUMER_THREADS = CONSUMER_WARPS * 32;              // 256
    constexpr int VEC_ELEMS = 8;                                       // bf16 per uint4
    constexpr int VEC_PER_THREAD = D / (CONSUMER_THREADS * VEC_ELEMS); // 2 for D=4096
    constexpr int SF_VEC_SIZE = 16;
    constexpr int SF_PER_ROW = D / SF_VEC_SIZE;
    static_assert(D % (CONSUMER_THREADS * VEC_ELEMS) == 0, "D must split evenly across 256 threads x 8 bf16");

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif

    int const tid = threadIdx.x;
    int const warp_id = tid / 32;
    int const lane_id = tid % 32;
    bool const is_producer = (warp_id == 0);
    int const consumer_id = tid - 32;      // valid for warp_id >= 1
    int const consumer_warp = warp_id - 1; // 0..7

    int const base_token = blockIdx.x * R_CTA;
    int const batchIdx = base_token / p.tokens_per_batch;

    constexpr int kStageElems = HAS_RESIDUAL ? (2 * D) : D;
    constexpr size_t kStageBytes = static_cast<size_t>(kStageElems) * sizeof(__nv_bfloat16);
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem_x_stage[NUM_STAGES];
    __nv_bfloat16* smem_attn_stage[NUM_STAGES];
#pragma unroll
    for (int s = 0; s < NUM_STAGES; s++)
    {
        smem_x_stage[s] = reinterpret_cast<__nv_bfloat16*>(smem_raw + s * kStageBytes);
        smem_attn_stage[s] = HAS_RESIDUAL ? (smem_x_stage[s] + D) : nullptr;
    }
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem_raw + NUM_STAGES * kStageBytes);
    uint64_t* empty_bar = full_bar + NUM_STAGES;
    float* warp_sums = reinterpret_cast<float*>(empty_bar + NUM_STAGES);

    if (tid == 0)
    {
#pragma unroll
        for (int s = 0; s < NUM_STAGES; s++)
        {
            mbar_init(&full_bar[s], 1);
            mbar_init(&empty_bar[s], CONSUMER_WARPS);
        }
    }

    // Pre-load modulator caches into REGs once per CTA. All R_CTA rows share batchIdx.
    constexpr int kScaleCacheSize = HAS_MODULATE ? (NUM_OUT * VEC_PER_THREAD) : 1;
    constexpr int kGateCacheSize = HAS_GATE ? VEC_PER_THREAD : 1;
    uint4 scale_cache[kScaleCacheSize];
    uint4 shift_cache[kScaleCacheSize];
    uint4 gate_cache[kGateCacheSize];

    if (!is_producer)
    {
        if constexpr (HAS_MODULATE)
        {
#pragma unroll
            for (int k = 0; k < NUM_OUT; k++)
            {
                int64_t const scaleBase = static_cast<int64_t>(batchIdx) * p.scale_ts_stride[k];
                int64_t const shiftBase = static_cast<int64_t>(batchIdx) * p.shift_ts_stride[k];
#pragma unroll
                for (int v = 0; v < VEC_PER_THREAD; v++)
                {
                    int const elem = (v * CONSUMER_THREADS + consumer_id) * VEC_ELEMS;
                    uint4 const s_ts = *reinterpret_cast<uint4 const*>(&p.scale_ts[k][scaleBase + elem]);
                    float4 const s_tbl_lo = *reinterpret_cast<float4 const*>(&p.scale_table[k][elem + 0]);
                    float4 const s_tbl_hi = *reinterpret_cast<float4 const*>(&p.scale_table[k][elem + 4]);
                    scale_cache[k * VEC_PER_THREAD + v] = combine_modulator_chunk(s_ts, s_tbl_lo, s_tbl_hi);

                    uint4 const h_ts = *reinterpret_cast<uint4 const*>(&p.shift_ts[k][shiftBase + elem]);
                    float4 const h_tbl_lo = *reinterpret_cast<float4 const*>(&p.shift_table[k][elem + 0]);
                    float4 const h_tbl_hi = *reinterpret_cast<float4 const*>(&p.shift_table[k][elem + 4]);
                    shift_cache[k * VEC_PER_THREAD + v] = combine_modulator_chunk(h_ts, h_tbl_lo, h_tbl_hi);
                }
            }
        }
        if constexpr (HAS_GATE)
        {
            int64_t const gateBase = static_cast<int64_t>(batchIdx) * p.gate_ts_stride;
#pragma unroll
            for (int v = 0; v < VEC_PER_THREAD; v++)
            {
                int const elem = (v * CONSUMER_THREADS + consumer_id) * VEC_ELEMS;
                uint4 const g_ts = *reinterpret_cast<uint4 const*>(&p.gate_ts[gateBase + elem]);
                float4 const g_tbl_lo = *reinterpret_cast<float4 const*>(&p.gate_table[elem + 0]);
                float4 const g_tbl_hi = *reinterpret_cast<float4 const*>(&p.gate_table[elem + 4]);
                gate_cache[v] = combine_modulator_chunk(g_ts, g_tbl_lo, g_tbl_hi);
            }
        }
    }

    __syncthreads();

    // Pre-fill empty_bar so producer's first iteration doesn't block.
    if (!is_producer && lane_id == 0)
    {
#pragma unroll
        for (int s = 0; s < NUM_STAGES; s++)
            mbar_arrive(&empty_bar[s]);
    }
    __syncthreads();

    uint32_t stage = 0;
    uint32_t phase_full = 0;
    uint32_t phase_empty = 0;
    constexpr uint32_t kXBytes = D * sizeof(__nv_bfloat16);
    constexpr uint32_t kAttnBytes = HAS_RESIDUAL ? kXBytes : 0;
    constexpr uint32_t kTotalBytes = kXBytes + kAttnBytes;

    if (is_producer)
    {
        if (lane_id == 0)
        {
#pragma unroll
            for (int sub = 0; sub < R_CTA; sub++)
            {
                int const token = base_token + sub;
                if (token >= p.num_tokens)
                    break;
                int64_t const tokenBase = static_cast<int64_t>(token) * D;
                mbar_wait(&empty_bar[stage], phase_empty);
                mbar_arrive_expect_tx(&full_bar[stage], kTotalBytes);
                cp_async_bulk(smem_x_stage[stage], p.x + tokenBase, kXBytes, &full_bar[stage]);
                if constexpr (HAS_RESIDUAL)
                {
                    cp_async_bulk(smem_attn_stage[stage], p.attn + tokenBase, kAttnBytes, &full_bar[stage]);
                }
                stage = stage + 1;
                if (stage == NUM_STAGES)
                {
                    stage = 0;
                    phase_empty ^= 1u;
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int sub = 0; sub < R_CTA; sub++)
        {
            int const token = base_token + sub;
            bool const valid = (token < p.num_tokens);
            int const safeToken = valid ? token : 0;
            int64_t const tokenBase = static_cast<int64_t>(safeToken) * D;

            mbar_wait(&full_bar[stage], phase_full);

            // Phase 1: x_new = x [+ attn [* gate]]; cache x_new in regs, accumulate sum^2.
            uint4 xnew_cache[VEC_PER_THREAD];
            float sum2 = 0.0f;
#pragma unroll
            for (int v = 0; v < VEC_PER_THREAD; v++)
            {
                int const elem = (v * CONSUMER_THREADS + consumer_id) * VEC_ELEMS;
                uint4 const xv = *reinterpret_cast<uint4 const*>(&smem_x_stage[stage][elem]);
                uint const* xu = reinterpret_cast<uint const*>(&xv);
                uint4 new_vec;
                uint* nu = reinterpret_cast<uint*>(&new_vec);

                if constexpr (HAS_RESIDUAL)
                {
                    uint4 const av = *reinterpret_cast<uint4 const*>(&smem_attn_stage[stage][elem]);
                    uint const* au = reinterpret_cast<uint const*>(&av);
                    if constexpr (HAS_GATE)
                    {
                        uint4 const gv = gate_cache[v];
                        uint const* gu = reinterpret_cast<uint const*>(&gv);
#pragma unroll
                        for (int i = 0; i < 4; i++)
                        {
                            float2 xf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&xu[i]));
                            float2 af = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&au[i]));
                            float2 gf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&gu[i]));
                            float2 nf;
                            nf.x = xf.x + af.x * gf.x;
                            nf.y = xf.y + af.y * gf.y;
                            sum2 += nf.x * nf.x + nf.y * nf.y;
                            reinterpret_cast<__nv_bfloat162&>(nu[i]) = __float22bfloat162_rn(nf);
                        }
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < 4; i++)
                        {
                            float2 xf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&xu[i]));
                            float2 af = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&au[i]));
                            float2 nf;
                            nf.x = xf.x + af.x;
                            nf.y = xf.y + af.y;
                            sum2 += nf.x * nf.x + nf.y * nf.y;
                            reinterpret_cast<__nv_bfloat162&>(nu[i]) = __float22bfloat162_rn(nf);
                        }
                    }
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        float2 xf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&xu[i]));
                        sum2 += xf.x * xf.x + xf.y * xf.y;
                        reinterpret_cast<uint&>(nu[i]) = xu[i];
                    }
                }
                xnew_cache[v] = new_vec;
                // Write x_new back to p.x in-place when HAS_RESIDUAL (downstream chain reads updated x).
                if constexpr (HAS_RESIDUAL)
                {
                    if (valid)
                        *reinterpret_cast<uint4*>(&p.x[tokenBase + elem]) = new_vec;
                }
            }

            sum2 = tensorrt_llm::common::warpReduceSum(sum2);
            if (lane_id == 0)
                warp_sums[consumer_warp] = sum2;
            bar_sync_consumer(CONSUMER_THREADS);

            float total = 0.0f;
#pragma unroll
            for (int w = 0; w < CONSUMER_WARPS; w++)
                total += warp_sums[w];
            float const rms_rcp = rsqrtf(total / static_cast<float>(D) + p.eps);

            // ---- Phase 2: normalize + (optional modulate) + write NUM_OUT outputs ----
            float sf_scale_val[NUM_OUT];
            if constexpr (HAS_QUANT)
            {
#pragma unroll
                for (int k = 0; k < NUM_OUT; k++)
                    sf_scale_val[k] = (p.sf_scale[k] != nullptr) ? *p.sf_scale[k] : 1.0f;
            }

            __nv_bfloat162 const one_b2 = __float2bfloat162_rn(1.0f);
#pragma unroll
            for (int v = 0; v < VEC_PER_THREAD; v++)
            {
                int const elem = (v * CONSUMER_THREADS + consumer_id) * VEC_ELEMS;
                uint4 const in_vec = xnew_cache[v];
                uint const* xu = reinterpret_cast<uint const*>(&in_vec);

                if constexpr (HAS_QUANT)
                {
                    // Per-v NVFP4 path: fp32 modulate + max-abs scan + e2m1 pack.
                    // Each thread covers 8 bf16 (one uint4) per v, half SF block.
                    // Pair of adjacent lanes (lane ^ 1) shares one 16-element SF block.
                    float vals[NUM_OUT][8];
                    float localMax[NUM_OUT];
#pragma unroll
                    for (int k = 0; k < NUM_OUT; k++)
                        localMax[k] = 0.0f;

#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        float2 xf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&xu[i]));
                        float const nx = xf.x * rms_rcp;
                        float const ny = xf.y * rms_rcp;
#pragma unroll
                        for (int k = 0; k < NUM_OUT; k++)
                        {
                            float yx, yy;
                            if constexpr (HAS_MODULATE)
                            {
                                uint4 const sv = scale_cache[k * VEC_PER_THREAD + v];
                                uint4 const hv = shift_cache[k * VEC_PER_THREAD + v];
                                uint const* su = reinterpret_cast<uint const*>(&sv);
                                uint const* hu = reinterpret_cast<uint const*>(&hv);
                                float2 sf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&su[i]));
                                float2 hf = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&hu[i]));
                                yx = nx * (1.0f + sf.x) + hf.x;
                                yy = ny * (1.0f + sf.y) + hf.y;
                            }
                            else
                            {
                                yx = nx;
                                yy = ny;
                            }
                            vals[k][2 * i + 0] = yx;
                            vals[k][2 * i + 1] = yy;
                            localMax[k] = fmaxf(localMax[k], fmaxf(fabsf(yx), fabsf(yy)));
                        }
                    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                    uint32_t fp4_packed[NUM_OUT];
                    uint8_t sfBytes[NUM_OUT];
#pragma unroll
                    for (int k = 0; k < NUM_OUT; k++)
                    {
                        // Pair-lane max-abs across 16 elements (one SF block).
                        float const blockMax = fmaxf(__shfl_xor_sync(0xffffffff, localMax[k], 1), localMax[k]);
                        constexpr float kE2M1MaxRcp = 1.0f / 6.0f;
                        float const sfValue = sf_scale_val[k] * (blockMax * kE2M1MaxRcp);
                        __nv_fp8_e4m3 const sfFp8 = __nv_fp8_e4m3(sfValue);
                        sfBytes[k] = sfFp8.__x;
                        float const sfQuant = static_cast<float>(sfFp8);
                        float const outScale = (blockMax != 0.0f) ? (sf_scale_val[k] / sfQuant) : 0.0f;
#pragma unroll
                        for (int i = 0; i < 8; i++)
                            vals[k][i] *= outScale;
                        fp4_packed[k] = fp32_vec_to_e2m1(vals[k]);
                    }

                    // colVecIdx: one uint32 of packed FP4 per (v, thread). Pair of adjacent
                    // lanes (lane ^ 1) shares one 16-element SF block (CVT_NUM_THREADS_PER_SF=2).
                    int const colVecIdx = v * CONSUMER_THREADS + consumer_id;
                    uint8_t* first_sf_ptr = cvt_quant_get_sf_out_offset<uint32_t, /*CVT_NUM_THREADS_PER_SF=*/2>(
                        std::nullopt, safeToken, colVecIdx, std::optional<int>(p.num_tokens), SF_PER_ROW, p.out_sf[0],
                        QuantizationSFLayout::SWIZZLED);
                    if (valid && first_sf_ptr != nullptr)
                    {
                        *first_sf_ptr = sfBytes[0];
                        if constexpr (NUM_OUT == 2)
                        {
                            uint8_t* second_sf_ptr
                                = cvt_quant_get_sf_out_offset<uint32_t, /*CVT_NUM_THREADS_PER_SF=*/2>(std::nullopt,
                                    safeToken, colVecIdx, std::optional<int>(p.num_tokens), SF_PER_ROW, p.out_sf[1],
                                    QuantizationSFLayout::SWIZZLED);
                            *second_sf_ptr = sfBytes[1];
                        }
                    }
                    if (valid)
                    {
                        int64_t const fp4_row_off = static_cast<int64_t>(safeToken) * (D / 8);
#pragma unroll
                        for (int k = 0; k < NUM_OUT; k++)
                            p.out_fp4[k][fp4_row_off + colVecIdx] = fp4_packed[k];
                    }
#endif
                }
                else
                {
                    uint4 out_vecs[NUM_OUT];
#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        __nv_bfloat162 x_b2 = *reinterpret_cast<__nv_bfloat162 const*>(&xu[i]);
                        float2 xf = __bfloat1622float2(x_b2);
                        __nv_bfloat162 normed = __float22bfloat162_rn(make_float2(xf.x * rms_rcp, xf.y * rms_rcp));
#pragma unroll
                        for (int k = 0; k < NUM_OUT; k++)
                        {
                            uint* o_u = reinterpret_cast<uint*>(&out_vecs[k]);
                            if constexpr (HAS_MODULATE)
                            {
                                uint4 const sv = scale_cache[k * VEC_PER_THREAD + v];
                                uint4 const hv = shift_cache[k * VEC_PER_THREAD + v];
                                uint const* su = reinterpret_cast<uint const*>(&sv);
                                uint const* hu = reinterpret_cast<uint const*>(&hv);
                                __nv_bfloat162 s_b2 = *reinterpret_cast<__nv_bfloat162 const*>(&su[i]);
                                __nv_bfloat162 h_b2 = *reinterpret_cast<__nv_bfloat162 const*>(&hu[i]);
                                __nv_bfloat162 one_p_s = __hadd2(one_b2, s_b2);
                                __nv_bfloat162 y_b2 = __hadd2(__hmul2(normed, one_p_s), h_b2);
                                reinterpret_cast<__nv_bfloat162&>(o_u[i]) = y_b2;
                            }
                            else
                            {
                                reinterpret_cast<__nv_bfloat162&>(o_u[i]) = normed;
                            }
                        }
                    }
                    if (valid)
                    {
#pragma unroll
                        for (int k = 0; k < NUM_OUT; k++)
                            *reinterpret_cast<uint4*>(&p.out_bf16[k][tokenBase + elem]) = out_vecs[k];
                    }
                }
            }

            bar_sync_consumer(CONSUMER_THREADS);
            if (lane_id == 0)
                mbar_arrive(&empty_bar[stage]);

            stage = stage + 1;
            if (stage == NUM_STAGES)
            {
                stage = 0;
                phase_full ^= 1u;
            }
        }
    }

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host launcher template + explicit specializations for each (residual, gate, modulate, num_out)
// combination consumed by LTX-2's transformer block.

template <bool HAS_RESIDUAL, bool HAS_GATE, bool HAS_MODULATE, int NUM_OUT, bool HAS_QUANT>
void launchFusedDiTNorm(AdaLNNormParams const& params, int hidden_dim, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.num_tokens >= 1, "num_tokens must be >= 1, got %d", params.num_tokens);
    TLLM_CHECK_WITH_INFO(
        params.tokens_per_batch >= 1, "tokens_per_batch must be >= 1, got %d", params.tokens_per_batch);
    TLLM_CHECK_WITH_INFO(params.num_tokens % params.tokens_per_batch == 0,
        "num_tokens (%d) must be divisible by tokens_per_batch (%d)", params.num_tokens, params.tokens_per_batch);

    // Production tile selected via NCU sweep at D=4096, B=1, T=15360: 1-row/CTA, 256 threads/CTA
    // gives 2 CTAs/SM (regs <= 80) and the highest per-element bench rate among {2r256, 1r256, 1r512}.
    constexpr int ROWS_PER_BLOCK = 1;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARPS_PER_ROW = (BLOCK_SIZE / ROWS_PER_BLOCK) / 32;

    cudaLaunchConfig_t cfg = {};
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    // Pipelined dispatch (multi-row CTA + circular TMA stages + warp specialization).
    // Auto-selected on shapes where it wins; otherwise falls through to the default path below.
    //   - hidden_dim == 4096 (video path; audio D=2048 has too small a grid to amortize the
    //     warp-specialization overhead and bench-regresses)
    //   - HAS_MODULATE || !HAS_GATE (the HAS_GATE && !HAS_MODULATE variant is HBM-bound and bench-regresses)
    //   - tokens_per_batch >= 4 && % 4 == 0 (R_CTA=4 rows share batchIdx for per-CTA mod cache)
    //   - num_tokens % 4 == 0 (whole grid must be R_CTA-aligned)
    constexpr bool kPipelinedVariantOK = HAS_MODULATE || !HAS_GATE;
    if constexpr (kPipelinedVariantOK)
    {
        // NUM_STAGES chosen per HAS_QUANT: bf16 path is HBM-latency-bound so deeper pipeline
        // (3 stages) helps; quant Phase 2 is compute-heavy and extra stages just add SMEM
        // pressure without commensurate latency hiding.
        constexpr int PIPE_NUM_STAGES = HAS_QUANT ? 2 : 3;
        constexpr int PIPE_BLOCK_SIZE = 288;
        constexpr int PIPE_D = 4096;
        constexpr int PIPE_R_CTA = 4;
        if (hidden_dim == PIPE_D && (params.num_tokens % PIPE_R_CTA == 0) && (params.tokens_per_batch >= PIPE_R_CTA)
            && (params.tokens_per_batch % PIPE_R_CTA == 0))
        {
            constexpr int pipe_stage_elems = HAS_RESIDUAL ? (2 * PIPE_D) : PIPE_D;
            size_t const pipe_smem_bytes
                = static_cast<size_t>(PIPE_NUM_STAGES) * pipe_stage_elems * sizeof(__nv_bfloat16)
                + 2 * PIPE_NUM_STAGES * sizeof(uint64_t) + 8 * sizeof(float) + 16;
            cudaLaunchConfig_t pipe_cfg = cfg;
            pipe_cfg.gridDim = dim3((params.num_tokens + PIPE_R_CTA - 1) / PIPE_R_CTA);
            pipe_cfg.blockDim = dim3(PIPE_BLOCK_SIZE);
            pipe_cfg.dynamicSmemBytes = static_cast<int>(pipe_smem_bytes);
            auto* pipe_kp = fusedDiTNormKernelPipelined<PIPE_D, PIPE_R_CTA, PIPE_NUM_STAGES, HAS_RESIDUAL, HAS_GATE,
                HAS_MODULATE, NUM_OUT, HAS_QUANT>;
            cudaFuncSetAttribute(
                pipe_kp, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(pipe_smem_bytes));
            cudaLaunchKernelEx(&pipe_cfg, pipe_kp, params);
            return;
        }
    }

#define LAUNCH(D_VAL)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        constexpr bool USE_TMA_HOST = (D_VAL >= 4096) && !(HAS_QUANT && HAS_MODULATE && NUM_OUT == 1);                 \
        constexpr bool USE_TMA_ATTN_HOST = USE_TMA_HOST && HAS_RESIDUAL && HAS_QUANT;                                  \
        int const attn_extra_bytes                                                                                     \
            = USE_TMA_ATTN_HOST ? (ROWS_PER_BLOCK * D_VAL * static_cast<int>(sizeof(__nv_bfloat16))) : 0;              \
        cfg.gridDim = dim3((params.num_tokens + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);                                 \
        cfg.blockDim = dim3(BLOCK_SIZE);                                                                               \
        cfg.dynamicSmemBytes = ROWS_PER_BLOCK * D_VAL * static_cast<int>(sizeof(__nv_bfloat16)) + attn_extra_bytes     \
            + ROWS_PER_BLOCK * WARPS_PER_ROW * static_cast<int>(sizeof(float));                                        \
        cudaLaunchKernelEx(&cfg,                                                                                       \
            fusedDiTNormKernel<D_VAL, ROWS_PER_BLOCK, BLOCK_SIZE, HAS_RESIDUAL, HAS_GATE, HAS_MODULATE, NUM_OUT,       \
                HAS_QUANT>,                                                                                            \
            params);                                                                                                   \
    } while (0)

    switch (hidden_dim)
    {
    case 2048: LAUNCH(2048); break;
    case 4096: LAUNCH(4096); break;
    default: TLLM_THROW("Unsupported hidden_dim for fusedDiTNorm: %d (only 2048, 4096)", hidden_dim);
    }
#undef LAUNCH
}

// Explicit instantiations -- one (bf16, fp4) pair per (residual, gate, modulate, num_out) variant.
//   !RESIDUAL, !GATE, MODULATE, NUM_OUT=1: post-pre-norm RMSNorm + shift_scale modulate
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/false, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/false>(AdaLNNormParams const&, int, cudaStream_t);
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/false, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/true>(AdaLNNormParams const&, int, cudaStream_t);
//   RESIDUAL, !GATE, MODULATE, NUM_OUT=2: post-text-attn residual + RMSNorm + dual shift_scale
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true, /*NUM_OUT=*/2,
    /*HAS_QUANT=*/false>(AdaLNNormParams const&, int, cudaStream_t);
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true, /*NUM_OUT=*/2,
    /*HAS_QUANT=*/true>(AdaLNNormParams const&, int, cudaStream_t);
//   RESIDUAL, GATE, MODULATE, NUM_OUT=1: post-cross-attn FFN pre-norm gate*attn + residual + RMSNorm + shift_scale
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true, /*HAS_MODULATE=*/true, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/false>(AdaLNNormParams const&, int, cudaStream_t);
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true, /*HAS_MODULATE=*/true, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/true>(AdaLNNormParams const&, int, cudaStream_t);
//   RESIDUAL, GATE, !MODULATE, NUM_OUT=1: post-self-attn gate*attn + residual + RMSNorm (+ optional quant)
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true, /*HAS_MODULATE=*/false, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/false>(AdaLNNormParams const&, int, cudaStream_t);
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true, /*HAS_MODULATE=*/false, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/true>(AdaLNNormParams const&, int, cudaStream_t);

} // namespace kernels

TRTLLM_NAMESPACE_END
