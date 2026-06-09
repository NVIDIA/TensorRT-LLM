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

// Phase 0b helper: combine 8 bf16 ts + 8 fp32 table into 8 bf16 (one uint4).
// Matches PyTorch eager `_get_all_ada_values` semantics: narrow fp32 table to
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
// Host launcher template + KABCD specializations.

template <bool HAS_RESIDUAL, bool HAS_GATE, bool HAS_MODULATE, int NUM_OUT, bool HAS_QUANT>
void launchFusedDiTNorm(AdaLNNormParams const& params, int hidden_dim, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.num_tokens >= 1, "num_tokens must be >= 1, got %d", params.num_tokens);
    TLLM_CHECK_WITH_INFO(
        params.tokens_per_batch >= 1, "tokens_per_batch must be >= 1, got %d", params.tokens_per_batch);
    TLLM_CHECK_WITH_INFO(params.num_tokens % params.tokens_per_batch == 0,
        "num_tokens (%d) must be divisible by tokens_per_batch (%d)", params.num_tokens, params.tokens_per_batch);

    // Hardcoded production tile. {2r256, 1r256, 1r512} all tied in the per-call
    // microbench (<0.5% spread). 1r256 chosen because at D=4096 it halves
    // CHUNKS_PER_ROW vs 2r256 (4 -> 2), which halves the per-thread modulator
    // cache register footprint. NCU showed 2r256 KB-bf16 at 142 regs/thread -
    // only 12.5% theoretical occupancy; switching to 1r256 puts that under the
    // 64K-regs/SM budget for 2 CTAs (25% occupancy).
    // Hardcoded production tile: 1-row/CTA, 256 threads/CTA.
    //
    // NCU sweep of {2r256, 1r256, 1r512} on B200 ran at the V1 production shape
    // (D=4096, B=1, T=15360):
    //   - 2r256: KB-bf16 142 regs/thread, 12.5% theoretical occupancy, sm_sol 23%
    //   - 1r256: KB-bf16  72 regs/thread, 37.5% occupancy, sm_sol 38%
    //   - 1r512: KB-bf16  44 regs/thread, 50%   occupancy, sm_sol 40%
    //
    // 1r512 wins on NCU metrics but LOSES on wall-time bench (KB +12% slower vs
    // 1r256). Per-wave analysis: 1r512 has more CTAs resident but each CTA has
    // 16 warps -- only 4 warp schedulers/SM can issue at once, so most warps
    // sit ready but unissued. The 512-thread __syncthreads() barrier is also
    // heavier than the 256-thread one. The net per-element rate is ~5% slower.
    //
    // 1r256 is the sweet spot: register pressure low enough for 2 CTAs/SM
    // (37.5%+ occupancy on all KABCD families), scheduler stays busy across
    // independent CTAs, and the per-element bench rate is the highest of the
    // three. Microbench shows cpp beating torch.compile on every V1/V2 shape
    // (KA -10%, KB -5%, KC tied, KD-bf16 tied, KD-quant -42%).
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

// Explicit instantiations -- one (bf16, fp4) pair per KABCD family.
//   KA: !RESIDUAL, !GATE, MODULATE, NUM_OUT=1
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/false, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/false>(AdaLNNormParams const&, int, cudaStream_t);
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/false, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/true>(AdaLNNormParams const&, int, cudaStream_t);
//   KB: RESIDUAL, !GATE, MODULATE, NUM_OUT=2
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true, /*NUM_OUT=*/2,
    /*HAS_QUANT=*/false>(AdaLNNormParams const&, int, cudaStream_t);
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true, /*NUM_OUT=*/2,
    /*HAS_QUANT=*/true>(AdaLNNormParams const&, int, cudaStream_t);
//   KC: RESIDUAL, GATE, MODULATE, NUM_OUT=1
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true, /*HAS_MODULATE=*/true, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/false>(AdaLNNormParams const&, int, cudaStream_t);
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true, /*HAS_MODULATE=*/true, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/true>(AdaLNNormParams const&, int, cudaStream_t);
//   KD: RESIDUAL, GATE, !MODULATE, NUM_OUT=1
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true, /*HAS_MODULATE=*/false, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/false>(AdaLNNormParams const&, int, cudaStream_t);
template void launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true, /*HAS_MODULATE=*/false, /*NUM_OUT=*/1,
    /*HAS_QUANT=*/true>(AdaLNNormParams const&, int, cudaStream_t);

} // namespace kernels

TRTLLM_NAMESPACE_END
