/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

// Fused post-mapping + TF32 HC prenorm GEMM on B200 (SM100).
//
// Mathematical formula:
//   new_r[i, hc, h] = post_mix[i, hc] * x[i, h]
//                   + sum_j comb_mix[i, j, hc] * residual[i, j, h]
//   D[i, n]       = sum_{hc, h} new_r[i, hc, h] * W[n, hc, h]
//   sqr[i]        = sum_{hc, h} new_r[i, hc, h]^2      (bf16-rounded)
//
// Shape:
//   residual [M, HC_MULT, hidden]   bf16
//   x        [M, hidden]            bf16
//   post_mix [M, HC_MULT]           fp32
//   comb_mix [M, HC_MULT, HC_MULT]  fp32
//   W        [N, HC_MULT*hidden]    fp32 (TF32)
//   D        [M, N]                 fp32
//   sqr      [M]                    fp32
//
// Kernel architecture (derived from DeepGEMM sm100_tf32_hc_prenorm_gemm):
//   - BLOCK_M x BLOCK_N x BLOCK_K = 64 x 32 x 64, TF32 MMA on tcgen05
//   - 256 threads per CTA (warps 0..3 = MMA group, warps 4..7 = pmap group)
//   - new_r is NEVER materialized in GMEM - it is computed on-chip directly
//     into TMEM (fp32) where UMMA consumes it.
//   - Iteration order: outer h_tile (slow), inner hc_idx (fast). residual+x
//     SMEM is reused for HC_MULT=4 consecutive MMA stages.
//
// Barriers:
//   full_B[N_B_STAGES]      TMA -> MMA (B arrived in SMEM)
//   empty_B[N_B_STAGES]     MMA -> TMA (B slot empty)
//   full_input[N_INPUT]     TMA -> pmap (residual+x arrived)
//   empty_input[N_INPUT]    pmap -> TMA (input slot empty)
//   full_cast[2]            pmap -> MMA (A ready in TMEM)
//   empty_cast[2]           MMA -> pmap (TMEM slot empty)
//   tmem_full[1]            MMA -> epilogue

#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cuda_bf16.h>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>

namespace deep_gemm
{
using math::swap;
}

#include <deep_gemm/common/reduction.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>
#include <deep_gemm/common/tma_utils.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace fused_mhc
{

// Reuse DeepGEMM's swizzle helper
using deep_gemm::sm100::make_umma_desc;
using deep_gemm::sm100::get_num_aligned_tmem_cols;
using deep_gemm::sm100::tcgen05_before_thread_sync;
using deep_gemm::sm100::tcgen05_after_thread_sync;
using deep_gemm::sm100::advance_umma_desc_lo;
using deep_gemm::tma_copy;
using deep_gemm::math::constexpr_ceil_div;
using deep_gemm::ptx::get_lane_idx;
using deep_gemm::utils::PatternVisitor;

template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
__device__ __forceinline__ uint32_t get_swizzled_smem_offset(uint32_t const& offset, uint32_t const& lane_idx)
{
    auto const& bank_group_idx = offset + lane_idx * (kSwizzleMode / kSwizzleBase);
    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
    constexpr bool kHasShortcut = (kSwizzleMode / kSwizzleBase) == kNumBankGroups;
    auto row = kHasShortcut ? (offset / kNumBankGroups + lane_idx) : (bank_group_idx / kNumBankGroups);
    auto col = kHasShortcut ? (offset) : (bank_group_idx % kNumBankGroups);
    col ^= row % (kSwizzleMode / kSwizzleBase);
    return row * 128 + col * kSwizzleBase;
}

__device__ __forceinline__ void stsm_x4_b16_rout(void* smem_dst, uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"l"(__cvta_generic_to_shared(smem_dst)),
        "r"(a), "r"(b), "r"(c), "r"(d));
}

template <uint32_t SHAPE_N, uint32_t HIDDEN, uint32_t HC_MULT, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kSwizzleCDMode, uint32_t N_B_STAGES, uint32_t N_INPUT_STAGES, uint32_t kNumMMAThreads,
    uint32_t kNumPmapThreads, uint32_t kNumSplits = 1, bool kEarlyRelease = false>
__global__ void __launch_bounds__(kNumMMAThreads + kNumPmapThreads, 1) fused_tf32_pmap_gemm_rout_atomic_impl(
    const uint32_t shape_m, const __grid_constant__ cute::TmaDescriptor tensor_map_residual,
    const __grid_constant__ cute::TmaDescriptor tensor_map_x, const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_residual_out,
    float* __restrict__ D, // [M, SHAPE_N]  (caller memsets to 0)
    float const* __restrict__ post_mix, float const* __restrict__ comb_mix, float* __restrict__ sqr_sum)
{                          // [M]            (caller memsets to 0)
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000) and (__CUDA_ARCH__ < 1100)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    constexpr uint32_t SHAPE_K = HC_MULT * HIDDEN;
    constexpr uint32_t H_TILES_PER_HC = HIDDEN / BLOCK_K;
    static_assert(H_TILES_PER_HC % kNumSplits == 0, "H_TILES_PER_HC must be divisible by kNumSplits");
    constexpr uint32_t H_TILES_PER_SPLIT = H_TILES_PER_HC / kNumSplits;
    constexpr uint32_t kNumCastStages = 4;
    constexpr uint32_t kSwizzleAMode = cute::min(BLOCK_K * sizeof(nv_bfloat16), 128);
    constexpr uint32_t kSwizzleBMode = cute::min(BLOCK_K * sizeof(float), 128);
    constexpr uint32_t kSwizzleXMode = kSwizzleAMode;
    constexpr uint32_t kSwizzleResMode = kSwizzleAMode;
    constexpr uint32_t kSwizzleRoutMode = kSwizzleAMode;
    constexpr auto kMajorA = cute::UMMA::Major::K;
    constexpr auto kMajorB = cute::UMMA::Major::K;
    static_assert(HIDDEN % BLOCK_K == 0, "HIDDEN must be multiple of BLOCK_K");
    static_assert(N_B_STAGES >= HC_MULT, "N_B_STAGES must be >= HC_MULT");
    static_assert(kSwizzleCDMode / sizeof(float) == BLOCK_N, "Invalid block N");
    static_assert(kNumMMAThreads == 128, "Invalid MMA threads");
    static_assert(kNumPmapThreads == 128, "Invalid pmap threads");
    static_assert(BLOCK_M == 64, "Invalid block M");
    static_assert(HC_MULT == 4, "Only HC_MULT=4 supported");
    static_assert(kSwizzleCDMode == 128, "Atomic variant expects kSwizzleCDMode=128");
    static_assert(SHAPE_N <= BLOCK_N, "SHAPE_N must fit within BLOCK_N");

    auto const warp_idx = cutlass::canonical_warp_idx_sync();
    auto const lane_idx = get_lane_idx();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_CD_SIZE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_B_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(float);
    constexpr uint32_t SMEM_RES_PER_ISTG = BLOCK_M * HC_MULT * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_X_PER_ISTG = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_POST_SIZE = BLOCK_M * HC_MULT * sizeof(float);
    constexpr uint32_t SMEM_COMB_SIZE = BLOCK_M * HC_MULT * HC_MULT * sizeof(float);
    constexpr uint32_t SMEM_RC_PER_HC = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16); // 8 KB
    constexpr uint32_t SMEM_RC_SIZE = HC_MULT * SMEM_RC_PER_HC;                  // 32 KB

    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<BLOCK_K * kNumCastStages + BLOCK_N>();

    // Prefetch TMA descriptors
    if (warp_idx == 0 and cute::elect_one_sync())
    {
        cute::prefetch_tma_descriptor(&tensor_map_residual);
        cute::prefetch_tma_descriptor(&tensor_map_x);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_residual_out);
    }

    // SMEM layout: [cd, B stages, res stages, x stages, post, comb, rc (HC_MULT slices)]
    auto smem_cd = reinterpret_cast<float*>(smem_buffer);
    uint8_t* cursor = smem_buffer + SMEM_CD_SIZE;
    auto smem_b = PatternVisitor(
        [&, base = cursor](uint32_t const& i) { return reinterpret_cast<float*>(base + i * SMEM_B_PER_STAGE); });
    cursor += N_B_STAGES * SMEM_B_PER_STAGE;
    auto smem_res = PatternVisitor(
        [&, base = cursor](uint32_t const& i) { return reinterpret_cast<nv_bfloat16*>(base + i * SMEM_RES_PER_ISTG); });
    cursor += N_INPUT_STAGES * SMEM_RES_PER_ISTG;
    auto smem_x_stg = PatternVisitor(
        [&, base = cursor](uint32_t const& i) { return reinterpret_cast<nv_bfloat16*>(base + i * SMEM_X_PER_ISTG); });
    cursor += N_INPUT_STAGES * SMEM_X_PER_ISTG;
    auto smem_post = reinterpret_cast<float*>(cursor);
    cursor += SMEM_POST_SIZE;
    auto smem_comb = reinterpret_cast<float*>(cursor);
    cursor += SMEM_COMB_SIZE;
    auto smem_rc = reinterpret_cast<nv_bfloat16*>(cursor); // [HC_MULT][BLOCK_M][BLOCK_K] bf16, single-buffered
    cursor += SMEM_RC_SIZE;

    cursor = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(cursor) + 7) & ~uintptr_t(7));
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(cursor);
    auto full_B = PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + i; });
    auto empty_B = PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + N_B_STAGES + i; });
    auto full_input = PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + 2 * N_B_STAGES + i; });
    auto empty_input
        = PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + 2 * N_B_STAGES + N_INPUT_STAGES + i; });
    auto full_cast = PatternVisitor(
        [=](uint32_t const& i) { return barrier_start_ptr + 2 * N_B_STAGES + 2 * N_INPUT_STAGES + i; });
    auto empty_cast = PatternVisitor([=](uint32_t const& i)
        { return barrier_start_ptr + 2 * N_B_STAGES + 2 * N_INPUT_STAGES + kNumCastStages + i; });
    auto tmem_full_barrier = barrier_start_ptr + 2 * N_B_STAGES + 2 * N_INPUT_STAGES + 2 * kNumCastStages;

    cursor += (2 * N_B_STAGES + 2 * N_INPUT_STAGES + 2 * kNumCastStages + 1) * sizeof(Barrier);
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(cursor);

    if (warp_idx == 1 and cute::elect_one_sync())
    {
#pragma unroll
        for (uint32_t i = 0; i < N_B_STAGES; ++i)
        {
            full_B[i]->init(1);
            empty_B[i]->init(1);
        }
#pragma unroll
        for (uint32_t i = 0; i < N_INPUT_STAGES; ++i)
        {
            full_input[i]->init(1);
            empty_input[i]->init(kNumPmapThreads);
        }
#pragma unroll
        for (uint32_t i = 0; i < kNumCastStages; ++i)
        {
            full_cast[i]->init(kNumPmapThreads);
            empty_cast[i]->init(1);
        }
        tmem_full_barrier->init(1);
        cutlass::arch::fence_barrier_init();
    }
    else if (warp_idx == 2)
    {
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t m_offset = m_block_idx * BLOCK_M;
    const uint32_t h_tile_start = k_split_idx * H_TILES_PER_SPLIT;
    constexpr uint32_t num_total_stages = H_TILES_PER_SPLIT * HC_MULT;

    // Prologue: pmap warp group loads post_mix, comb_mix into SMEM
    if (warp_idx >= kNumMMAThreads / 32)
    {
        const uint32_t pmap_tid = threadIdx.x - kNumMMAThreads;
#pragma unroll
        for (uint32_t t = 0; t < 2; ++t)
        {
            uint32_t idx = pmap_tid + t * kNumPmapThreads;
            if (idx < BLOCK_M * HC_MULT)
            {
                uint32_t m = idx / HC_MULT;
                uint32_t hc = idx % HC_MULT;
                uint32_t gmem_m = m_offset + m;
                float v = (gmem_m < shape_m) ? post_mix[gmem_m * HC_MULT + hc] : 0.f;
                smem_post[idx] = v;
            }
        }
#pragma unroll
        for (uint32_t t = 0; t < 8; ++t)
        {
            uint32_t idx = pmap_tid + t * kNumPmapThreads;
            if (idx < BLOCK_M * HC_MULT * HC_MULT)
            {
                uint32_t m = idx / (HC_MULT * HC_MULT);
                uint32_t jk = idx % (HC_MULT * HC_MULT);
                uint32_t gmem_m = m_offset + m;
                float v = (gmem_m < shape_m) ? comb_mix[gmem_m * HC_MULT * HC_MULT + jk] : 0.f;
                smem_comb[idx] = v;
            }
        }
    }
    __syncthreads();

    if (warp_idx < kNumMMAThreads / 32)
    {
        // ----- TMA warp (warp 0) -----
        if (warp_idx == 0 and cute::elect_one_sync())
        {
            uint32_t b_stage = 0;
            uint32_t i_stage = 0;
            uint32_t s = 0;
            for (uint32_t ht = 0; ht < H_TILES_PER_SPLIT; ++ht)
            {
                const uint32_t h_tile = h_tile_start + ht;
                empty_input[i_stage]->wait(((ht / N_INPUT_STAGES) & 1) ^ 1);
                uint32_t m_idx = m_block_idx * BLOCK_M;
                uint32_t h_idx = h_tile * BLOCK_K;
#pragma unroll
                for (uint32_t j = 0; j < HC_MULT; ++j)
                {
                    tma_copy<BLOCK_K, BLOCK_M, kSwizzleResMode>(&tensor_map_residual, full_input[i_stage],
                        smem_res[i_stage] + j * BLOCK_M * BLOCK_K, j * HIDDEN + h_idx, m_idx);
                }
                tma_copy<BLOCK_K, BLOCK_M, kSwizzleXMode>(
                    &tensor_map_x, full_input[i_stage], smem_x_stg[i_stage], h_idx, m_idx);
                constexpr uint32_t kInputBytes = SMEM_RES_PER_ISTG + SMEM_X_PER_ISTG;
                full_input[i_stage]->arrive_and_expect_tx(kInputBytes);

#pragma unroll
                for (uint32_t hc = 0; hc < HC_MULT; ++hc)
                {
                    empty_B[b_stage]->wait(((s / N_B_STAGES) & 1) ^ 1);
                    uint32_t k_idx = hc * HIDDEN + h_idx;
                    tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(
                        &tensor_map_b, full_B[b_stage], smem_b[b_stage], k_idx, 0);
                    full_B[b_stage]->arrive_and_expect_tx(SMEM_B_PER_STAGE);
                    b_stage = (b_stage + 1) % N_B_STAGES;
                    ++s;
                }
                i_stage = (i_stage + 1) % N_INPUT_STAGES;
            }
        }

        // ----- MMA issue warp (warp 1) -----
        if (warp_idx == 1)
        {
            constexpr uint32_t UMMA_M = BLOCK_M;
            constexpr uint32_t UMMA_N = BLOCK_N;
            constexpr uint32_t UMMA_K = 32 / sizeof(float);
            constexpr uint32_t BLOCK_SWIZZLED_BK = kSwizzleBMode / sizeof(float);
            using umma_t = cute::SM100_MMA_TF32_TS<cutlass::tfloat32_t, cutlass::tfloat32_t, float, BLOCK_M, BLOCK_N,
                kMajorA, kMajorB>;
            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::tfloat32_t, cutlass::tfloat32_t, float, UMMA_M,
                UMMA_N, kMajorA, kMajorB>();
            auto const& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
            static_assert(N_B_STAGES <= 32, "Too many B stages");
            auto b_desc = make_umma_desc<kMajorB, BLOCK_N, BLOCK_SWIZZLED_BK, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t const& b_desc_lo = lane_idx < N_B_STAGES ? b_desc.lo + lane_idx * SMEM_B_PER_STAGE / 16 : 0u;

            for (uint32_t s = 0; s < num_total_stages; ++s)
            {
                const uint32_t b_stage = s % N_B_STAGES;
                const uint32_t cast_stage_idx = s % kNumCastStages;
                full_cast[cast_stage_idx]->wait((s / kNumCastStages) & 1);
                full_B[b_stage]->wait((s / N_B_STAGES) & 1);
                tcgen05_after_thread_sync();
                auto const& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(b_stage));
#pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k)
                {
                    uint32_t const& atom_idx = (k * UMMA_K) / BLOCK_SWIZZLED_BK;
                    uint32_t const& in_atom_idx = (k * UMMA_K) % BLOCK_SWIZZLED_BK;
                    uint32_t const& offset = atom_idx * BLOCK_N * BLOCK_SWIZZLED_BK;
                    b_desc.lo = advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleBMode, float>(
                        b_desc_base_lo, offset, in_atom_idx);
                    umma_t::fma(BLOCK_K * cast_stage_idx + k * UMMA_K, b_desc, BLOCK_K * kNumCastStages, s > 0 or k > 0,
                        runtime_instr_desc);
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_cast[cast_stage_idx]));
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_B[b_stage]));
            }
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barrier));
        }

        // ----- Epilogue (warps 0..3, 128 threads) -----
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(float);
        static_assert(BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        tmem_full_barrier->wait(0);
        tcgen05_after_thread_sync();

#pragma unroll
        for (uint32_t i = 0; i < BLOCK_N / kNumElemsPerBankGroup; ++i)
        {
            uint32_t tmem_addr = BLOCK_K * kNumCastStages + i * kNumElemsPerBankGroup;
            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd) + warp_idx * BLOCK_M / 4 * kSwizzleCDMode
                + get_swizzled_smem_offset<kSwizzleCDMode>(i, lane_idx);
            uint32_t values[kNumElemsPerBankGroup];
            static_assert(kNumElemsPerBankGroup == 4, "Invalid type");
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
            cutlass::arch::fence_view_async_tmem_load();
            if (BLOCK_M == 128 or (BLOCK_M == 64 and lane_idx < 16))
                deep_gemm::ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
            if constexpr (BLOCK_M == 64)
                __syncwarp();
        }
        cutlass::arch::NamedBarrier::sync(kNumMMAThreads, 0);

        constexpr uint32_t kTotalOut = BLOCK_M * SHAPE_N;
        const uint32_t tid = threadIdx.x;
#pragma unroll
        for (uint32_t k = tid; k < kTotalOut; k += kNumMMAThreads)
        {
            uint32_t m = k / SHAPE_N;
            uint32_t n = k - m * SHAPE_N;
            uint32_t gm = m_block_idx * BLOCK_M + m;
            if (gm < shape_m)
            {
                uint32_t col_group = n >> 2;
                uint32_t in_group = n & 3;
                uint32_t phys_col_grp = col_group ^ (m & 7);
                uint32_t byte = m * kSwizzleCDMode + phys_col_grp * kNumBankGroupBytes + in_group * sizeof(float);
                float val = *reinterpret_cast<float const*>(reinterpret_cast<uint8_t const*>(smem_cd) + byte);
                atomicAdd(&D[gm * SHAPE_N + n], val);
            }
        }

        if (warp_idx == 1)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
    else
    {
        // ----- Pmap warp group (warps 4..7, 128 threads) -----
        const uint32_t sub_warp_idx = warp_idx - kNumMMAThreads / 32;
        const uint32_t upper_row = sub_warp_idx * 16 + lane_idx / 4;
        const uint32_t lower_row = upper_row + 8;
        const uint32_t col_lane = lane_idx % 4;

        float pm_u[HC_MULT], pm_l[HC_MULT];
        float cm_u[HC_MULT][HC_MULT], cm_l[HC_MULT][HC_MULT];
#pragma unroll
        for (uint32_t hc = 0; hc < HC_MULT; ++hc)
        {
            pm_u[hc] = smem_post[upper_row * HC_MULT + hc];
            pm_l[hc] = smem_post[lower_row * HC_MULT + hc];
        }
#pragma unroll
        for (uint32_t j = 0; j < HC_MULT; ++j)
        {
#pragma unroll
            for (uint32_t hc = 0; hc < HC_MULT; ++hc)
            {
                cm_u[j][hc] = smem_comb[upper_row * HC_MULT * HC_MULT + j * HC_MULT + hc];
                cm_l[j][hc] = smem_comb[lower_row * HC_MULT * HC_MULT + j * HC_MULT + hc];
            }
        }

        float sqr_u = 0.f, sqr_l = 0.f;
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(nv_bfloat16);
        constexpr uint32_t kNumLoads = BLOCK_K / kNumElemsPerBankGroup;
        constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;
        static_assert(BLOCK_K * sizeof(nv_bfloat16) == kSwizzleAMode, "BLOCK_K must match swizzle A mode");
        static_assert(kNumLoads % 2 == 0, "kNumLoads must be even for LDSM.x4");

        uint32_t s = 0;
        for (uint32_t ht = 0; ht < H_TILES_PER_SPLIT; ++ht)
        {
            const uint32_t i_stage = ht % N_INPUT_STAGES;
            full_input[i_stage]->wait((ht / N_INPUT_STAGES) & 1);

            uint32_t x_vals[2][kNumLoads];
            {
                uint8_t const* x_base
                    = reinterpret_cast<uint8_t*>(smem_x_stg[i_stage]) + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleXMode;
#pragma unroll
                for (uint32_t i = 0; i < kNumLoads; i += 2)
                {
                    auto smem_ptr = x_base + get_swizzled_smem_offset<kSwizzleXMode>(i + lane_idx / 16, lane_idx % 16);
                    deep_gemm::sm90::SM90_U32x4_LDSM_N::copy(x_vals[0][i + 0], x_vals[1][i + 0], x_vals[0][i + 1],
                        x_vals[1][i + 1], const_cast<uint8_t*>(smem_ptr));
                }
            }

            uint32_t r_vals[HC_MULT][2][kNumLoads];
#pragma unroll
            for (uint32_t j = 0; j < HC_MULT; ++j)
            {
                uint8_t const* r_base = reinterpret_cast<uint8_t*>(smem_res[i_stage])
                    + j * BLOCK_M * BLOCK_K * sizeof(nv_bfloat16) + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleResMode;
#pragma unroll
                for (uint32_t i = 0; i < kNumLoads; i += 2)
                {
                    auto smem_ptr
                        = r_base + get_swizzled_smem_offset<kSwizzleResMode>(i + lane_idx / 16, lane_idx % 16);
                    deep_gemm::sm90::SM90_U32x4_LDSM_N::copy(r_vals[j][0][i + 0], r_vals[j][1][i + 0],
                        r_vals[j][0][i + 1], r_vals[j][1][i + 1], const_cast<uint8_t*>(smem_ptr));
                }
            }

            float2 xf[2][kNumLoads];
#pragma unroll
            for (uint32_t u = 0; u < 2; ++u)
            {
#pragma unroll
                for (uint32_t i = 0; i < kNumLoads; ++i)
                {
                    xf[u][i] = __bfloat1622float2(*reinterpret_cast<nv_bfloat162*>(&x_vals[u][i]));
                }
            }

            if constexpr (kEarlyRelease)
            {
                empty_input[i_stage]->arrive();
            }

            // Wait for previous ht's residual_out TMA_STOREs to drain before we
            // overwrite single-buffered smem_rc with new hc values.
            if (ht > 0)
            {
                cute::tma_store_wait<0>();
            }

#pragma unroll
            for (uint32_t hc = 0; hc < HC_MULT; ++hc)
            {
                const uint32_t cast_stage_idx = s % kNumCastStages;
                empty_cast[cast_stage_idx]->wait(((s / kNumCastStages) & 1) ^ 1);

                uint32_t rc_u_buf[kNumLoads], rc_l_buf[kNumLoads];
#pragma unroll
                for (uint32_t i = 0; i < kNumLoads; ++i)
                {
                    float2 nu{pm_u[hc] * xf[0][i].x, pm_u[hc] * xf[0][i].y};
                    float2 nl{pm_l[hc] * xf[1][i].x, pm_l[hc] * xf[1][i].y};
#pragma unroll
                    for (uint32_t j = 0; j < HC_MULT; ++j)
                    {
                        float2 ruj = __bfloat1622float2(*reinterpret_cast<nv_bfloat162*>(&r_vals[j][0][i]));
                        float2 rlj = __bfloat1622float2(*reinterpret_cast<nv_bfloat162*>(&r_vals[j][1][i]));
                        nu.x = fmaf(cm_u[j][hc], ruj.x, nu.x);
                        nu.y = fmaf(cm_u[j][hc], ruj.y, nu.y);
                        nl.x = fmaf(cm_l[j][hc], rlj.x, nl.x);
                        nl.y = fmaf(cm_l[j][hc], rlj.y, nl.y);
                    }
                    nv_bfloat162 b_up = __float22bfloat162_rn(nu);
                    nv_bfloat162 b_lo = __float22bfloat162_rn(nl);
                    uint32_t b_up_bits = *reinterpret_cast<uint32_t*>(&b_up);
                    uint32_t b_lo_bits = *reinterpret_cast<uint32_t*>(&b_lo);
                    rc_u_buf[i] = b_up_bits;
                    rc_l_buf[i] = b_lo_bits;
                    float2 ru = __bfloat1622float2(b_up);
                    float2 rl = __bfloat1622float2(b_lo);
                    sqr_u = fmaf(ru.x, ru.x, sqr_u);
                    sqr_u = fmaf(ru.y, ru.y, sqr_u);
                    sqr_l = fmaf(rl.x, rl.x, sqr_l);
                    sqr_l = fmaf(rl.y, rl.y, sqr_l);
                    cute::SM100_TMEM_STORE_16dp256b1x::copy(*reinterpret_cast<uint32_t*>(&ru.x),
                        *reinterpret_cast<uint32_t*>(&ru.y), *reinterpret_cast<uint32_t*>(&rl.x),
                        *reinterpret_cast<uint32_t*>(&rl.y), cast_stage_idx * BLOCK_K + i * 8);
                }
                cutlass::arch::fence_view_async_tmem_store();
                tcgen05_before_thread_sync();
                full_cast[cast_stage_idx]->arrive();
                ++s;

                // STSM bf16 new_r values into smem_rc[hc] sub-region for this warp.
                uint8_t* rc_base = reinterpret_cast<uint8_t*>(smem_rc) + hc * SMEM_RC_PER_HC
                    + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleRoutMode;
#pragma unroll
                for (uint32_t i = 0; i < kNumLoads; i += 2)
                {
                    auto smem_ptr
                        = rc_base + get_swizzled_smem_offset<kSwizzleRoutMode>(i + lane_idx / 16, lane_idx % 16);
                    stsm_x4_b16_rout(smem_ptr, rc_u_buf[i + 0], rc_l_buf[i + 0], rc_u_buf[i + 1], rc_l_buf[i + 1]);
                }
            }
            if constexpr (!kEarlyRelease)
            {
                empty_input[i_stage]->arrive();
            }

            // Emit HC_MULT TMA_STOREs of residual_cur: one per hc slice, per-warp rows.
            cute::tma_store_fence();
            if (cute::elect_one_sync())
            {
                const uint32_t h_idx = (h_tile_start + ht) * BLOCK_K;
#pragma unroll
                for (uint32_t hc = 0; hc < HC_MULT; ++hc)
                {
                    uint8_t* rc_base = reinterpret_cast<uint8_t*>(smem_rc) + hc * SMEM_RC_PER_HC
                        + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleRoutMode;
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_residual_out, rc_base, hc * HIDDEN + h_idx,
                        m_offset + sub_warp_idx * BLOCK_M_PER_WARP);
                    cute::tma_store_arrive();
                }
            }
        }

        // Drain any in-flight residual_out TMA stores before exit.
        cute::tma_store_wait<0>();

        // Warp-reduce sqr across 4 col_lanes then atomicAdd to global.
        sqr_u += __shfl_xor_sync(0xffffffff, sqr_u, 1);
        sqr_u += __shfl_xor_sync(0xffffffff, sqr_u, 2);
        sqr_l += __shfl_xor_sync(0xffffffff, sqr_l, 1);
        sqr_l += __shfl_xor_sync(0xffffffff, sqr_l, 2);
        if (col_lane == 0)
        {
            uint32_t gm_u = m_block_idx * BLOCK_M + upper_row;
            uint32_t gm_l = m_block_idx * BLOCK_M + lower_row;
            if (gm_u < shape_m)
                atomicAdd(&sqr_sum[gm_u], sqr_u);
            if (gm_l < shape_m)
                atomicAdd(&sqr_sum[gm_l], sqr_l);
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100a");
#endif
}

// ============================================================================
// ALL-IN-ONE variant (Path D, tf32 tcgen05 MMA analogue of Path F).
//
// Single-kernel fusion of:
//   1) post_mapping : residual_cur[j,h] = post_mix_prev[j]*x_prev[h]
//                                       + sum_k comb_mix_prev[k,j]*residual_prev[k,h]
//   2) pre_GEMM     : D[i,n] = sum_{hc,h} residual_cur[i,hc,h] * W_T[n, hc*HIDDEN+h]
//                     sqr[i] = sum_{hc,h} residual_cur[i,hc,h]^2
//   3) bigFuse      : rmsnorm+sigmoid+sinkhorn on D/sqr -> post_mix_out,
//                     comb_mix_out, pre_mix; layer_input = pre_mix @ residual_cur.
//
// Semantics match Path F (fused_pmap_gemm_fma_allinone) exactly. The only
// algorithmic difference vs Path F is that we use tf32 tcgen05 MMA for the
// residual_cur @ W_T GEMM (instead of CUDA-core FMA).
//
// Pipelining, warp layout, and SMEM layout inherit from Path B
// (fused_tf32_pmap_gemm_rout_atomic_impl): the pmap warp group computes
// residual_cur, TMA-stores it to GMEM, and STSM-casts it into TMEM; the MMA
// warp group consumes TMEM for the GEMM; the epilogue atomicAdd's into
// y_acc / sqr_sum. Phase 3 elects the last-home CTA (per m_block) via
// atomicAdd on done_counter; Phase 4 runs bigfuse inline on that CTA only,
// reloading residual_cur from GMEM and writing layer_input + post_mix_out +
// comb_mix_out.
//
// Caller MUST zero D (y_acc), sqr_sum (r_acc), done_counter before launch.
// ============================================================================

template <uint32_t SHAPE_N, uint32_t HIDDEN, uint32_t HC_MULT, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kSwizzleCDMode, uint32_t N_B_STAGES, uint32_t N_INPUT_STAGES, uint32_t kNumMMAThreads,
    uint32_t kNumPmapThreads, uint32_t kNumSplits = 1>
__global__ void __launch_bounds__(kNumMMAThreads + kNumPmapThreads, 1)
    fused_allinone_tf32_pmap_gemm_atomic_impl(const uint32_t shape_m,
        const __grid_constant__ cute::TmaDescriptor tensor_map_residual,     // residual_prev, bf16
        const __grid_constant__ cute::TmaDescriptor tensor_map_x,            // x_prev,        bf16
        const __grid_constant__ cute::TmaDescriptor tensor_map_b,            // W_T,           tf32
        const __grid_constant__ cute::TmaDescriptor tensor_map_residual_out, // residual_cur,  bf16 (TMA store)
        __nv_bfloat16 const* __restrict__ residual_cur_ptr,                  // same buffer as TMA target
        __nv_bfloat16* __restrict__ layer_input_out,                         // [M, HIDDEN]    bf16
        float* __restrict__ D,                   // [M, SHAPE_N]   fp32 (y_acc, caller zeros)
        float* __restrict__ sqr_sum,             // [M]            fp32 (r_acc, caller zeros)
        int* __restrict__ done_counter,          // [ceil(M/BLOCK_M)] int (caller zeros)
        float const* __restrict__ post_mix_prev, // [M, HC_MULT]
        float const* __restrict__ comb_mix_prev, // [M, HC_MULT, HC_MULT]
        float const* __restrict__ hc_scale,      // [3]
        float const* __restrict__ hc_base,       // [HC_MULT*(2+HC_MULT)]
        float* __restrict__ post_mix_out,        // [M, HC_MULT]
        float* __restrict__ comb_mix_out,        // [M, HC_MULT, HC_MULT]
        float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value, uint32_t sinkhorn_repeat)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000) and (__CUDA_ARCH__ < 1100)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    constexpr uint32_t HC_MULT2 = HC_MULT * HC_MULT;
    constexpr uint32_t HC_MULT3 = HC_MULT * (2 + HC_MULT);
    constexpr uint32_t SHAPE_K = HC_MULT * HIDDEN;
    constexpr uint32_t H_TILES_PER_HC = HIDDEN / BLOCK_K;
    static_assert(H_TILES_PER_HC % kNumSplits == 0, "H_TILES_PER_HC must be divisible by kNumSplits");
    constexpr uint32_t H_TILES_PER_SPLIT = H_TILES_PER_HC / kNumSplits;
    constexpr uint32_t kNumCastStages = 4;
    constexpr uint32_t kSwizzleAMode = cute::min(BLOCK_K * sizeof(nv_bfloat16), 128);
    constexpr uint32_t kSwizzleBMode = cute::min(BLOCK_K * sizeof(float), 128);
    constexpr uint32_t kSwizzleXMode = kSwizzleAMode;
    constexpr uint32_t kSwizzleResMode = kSwizzleAMode;
    constexpr uint32_t kSwizzleRoutMode = kSwizzleAMode;
    constexpr auto kMajorA = cute::UMMA::Major::K;
    constexpr auto kMajorB = cute::UMMA::Major::K;
    static_assert(HIDDEN % BLOCK_K == 0, "HIDDEN must be multiple of BLOCK_K");
    static_assert(N_B_STAGES >= HC_MULT, "N_B_STAGES must be >= HC_MULT");
    static_assert(kSwizzleCDMode / sizeof(float) == BLOCK_N, "Invalid block N");
    static_assert(kNumMMAThreads == 128, "Invalid MMA threads");
    static_assert(kNumPmapThreads == 128, "Invalid pmap threads");
    static_assert(BLOCK_M == 64, "Invalid block M");
    static_assert(HC_MULT == 4, "Only HC_MULT=4 supported");
    static_assert(kSwizzleCDMode == 128, "Atomic variant expects kSwizzleCDMode=128");
    static_assert(SHAPE_N <= BLOCK_N, "SHAPE_N must fit within BLOCK_N");
    static_assert(SHAPE_N == HC_MULT3, "Path D expects SHAPE_N == HC_MULT*(2+HC_MULT)=24");

    auto const warp_idx = cutlass::canonical_warp_idx_sync();
    auto const lane_idx = get_lane_idx();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    constexpr uint32_t SMEM_CD_SIZE = BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_B_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(float);
    constexpr uint32_t SMEM_RES_PER_ISTG = BLOCK_M * HC_MULT * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_X_PER_ISTG = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_POST_SIZE = BLOCK_M * HC_MULT * sizeof(float);
    constexpr uint32_t SMEM_COMB_SIZE = BLOCK_M * HC_MULT * HC_MULT * sizeof(float);
    constexpr uint32_t SMEM_RC_PER_HC = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16); // 8 KB
    constexpr uint32_t SMEM_RC_SIZE = HC_MULT * SMEM_RC_PER_HC;                  // 32 KB

    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<BLOCK_K * kNumCastStages + BLOCK_N>();

    // Prefetch TMA descriptors
    if (warp_idx == 0 and cute::elect_one_sync())
    {
        cute::prefetch_tma_descriptor(&tensor_map_residual);
        cute::prefetch_tma_descriptor(&tensor_map_x);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_residual_out);
    }

    // SMEM layout: [cd, B stages, res stages, x stages, post, comb, rc (HC_MULT slices)]
    auto smem_cd = reinterpret_cast<float*>(smem_buffer);
    uint8_t* cursor = smem_buffer + SMEM_CD_SIZE;
    auto smem_b = PatternVisitor(
        [&, base = cursor](uint32_t const& i) { return reinterpret_cast<float*>(base + i * SMEM_B_PER_STAGE); });
    cursor += N_B_STAGES * SMEM_B_PER_STAGE;
    auto smem_res = PatternVisitor(
        [&, base = cursor](uint32_t const& i) { return reinterpret_cast<nv_bfloat16*>(base + i * SMEM_RES_PER_ISTG); });
    cursor += N_INPUT_STAGES * SMEM_RES_PER_ISTG;
    auto smem_x_stg = PatternVisitor(
        [&, base = cursor](uint32_t const& i) { return reinterpret_cast<nv_bfloat16*>(base + i * SMEM_X_PER_ISTG); });
    cursor += N_INPUT_STAGES * SMEM_X_PER_ISTG;
    auto smem_post = reinterpret_cast<float*>(cursor);
    cursor += SMEM_POST_SIZE;
    auto smem_comb = reinterpret_cast<float*>(cursor);
    cursor += SMEM_COMB_SIZE;
    auto smem_rc = reinterpret_cast<nv_bfloat16*>(cursor); // [HC_MULT][BLOCK_M][BLOCK_K] bf16
    cursor += SMEM_RC_SIZE;

    cursor = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(cursor) + 7) & ~uintptr_t(7));
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(cursor);
    auto full_B = PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + i; });
    auto empty_B = PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + N_B_STAGES + i; });
    auto full_input = PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + 2 * N_B_STAGES + i; });
    auto empty_input
        = PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + 2 * N_B_STAGES + N_INPUT_STAGES + i; });
    auto full_cast = PatternVisitor(
        [=](uint32_t const& i) { return barrier_start_ptr + 2 * N_B_STAGES + 2 * N_INPUT_STAGES + i; });
    auto empty_cast = PatternVisitor([=](uint32_t const& i)
        { return barrier_start_ptr + 2 * N_B_STAGES + 2 * N_INPUT_STAGES + kNumCastStages + i; });
    auto tmem_full_barrier = barrier_start_ptr + 2 * N_B_STAGES + 2 * N_INPUT_STAGES + 2 * kNumCastStages;

    cursor += (2 * N_B_STAGES + 2 * N_INPUT_STAGES + 2 * kNumCastStages + 1) * sizeof(Barrier);
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(cursor);

    if (warp_idx == 1 and cute::elect_one_sync())
    {
#pragma unroll
        for (uint32_t i = 0; i < N_B_STAGES; ++i)
        {
            full_B[i]->init(1);
            empty_B[i]->init(1);
        }
#pragma unroll
        for (uint32_t i = 0; i < N_INPUT_STAGES; ++i)
        {
            full_input[i]->init(1);
            empty_input[i]->init(kNumPmapThreads);
        }
#pragma unroll
        for (uint32_t i = 0; i < kNumCastStages; ++i)
        {
            full_cast[i]->init(kNumPmapThreads);
            empty_cast[i]->init(1);
        }
        tmem_full_barrier->init(1);
        cutlass::arch::fence_barrier_init();
    }
    else if (warp_idx == 2)
    {
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    const uint32_t block_idx = __shfl_sync(0xffffffff, blockIdx.x, 0);
    const uint32_t m_block_idx = block_idx / kNumSplits;
    const uint32_t k_split_idx = block_idx % kNumSplits;
    const uint32_t m_offset = m_block_idx * BLOCK_M;
    const uint32_t h_tile_start = k_split_idx * H_TILES_PER_SPLIT;
    constexpr uint32_t num_total_stages = H_TILES_PER_SPLIT * HC_MULT;

    // Prologue: pmap warp group loads post_mix_prev, comb_mix_prev into SMEM
    if (warp_idx >= kNumMMAThreads / 32)
    {
        const uint32_t pmap_tid = threadIdx.x - kNumMMAThreads;
#pragma unroll
        for (uint32_t t = 0; t < 2; ++t)
        {
            uint32_t idx = pmap_tid + t * kNumPmapThreads;
            if (idx < BLOCK_M * HC_MULT)
            {
                uint32_t m = idx / HC_MULT;
                uint32_t hc = idx % HC_MULT;
                uint32_t gmem_m = m_offset + m;
                float v = (gmem_m < shape_m) ? post_mix_prev[gmem_m * HC_MULT + hc] : 0.f;
                smem_post[idx] = v;
            }
        }
#pragma unroll
        for (uint32_t t = 0; t < 8; ++t)
        {
            uint32_t idx = pmap_tid + t * kNumPmapThreads;
            if (idx < BLOCK_M * HC_MULT * HC_MULT)
            {
                uint32_t m = idx / (HC_MULT * HC_MULT);
                uint32_t jk = idx % (HC_MULT * HC_MULT);
                uint32_t gmem_m = m_offset + m;
                float v = (gmem_m < shape_m) ? comb_mix_prev[gmem_m * HC_MULT * HC_MULT + jk] : 0.f;
                smem_comb[idx] = v;
            }
        }
    }
    __syncthreads();

    if (warp_idx < kNumMMAThreads / 32)
    {
        // ----- TMA warp (warp 0) -----
        if (warp_idx == 0 and cute::elect_one_sync())
        {
            uint32_t b_stage = 0;
            uint32_t i_stage = 0;
            uint32_t s = 0;
            for (uint32_t ht = 0; ht < H_TILES_PER_SPLIT; ++ht)
            {
                const uint32_t h_tile = h_tile_start + ht;
                empty_input[i_stage]->wait(((ht / N_INPUT_STAGES) & 1) ^ 1);
                uint32_t m_idx = m_block_idx * BLOCK_M;
                uint32_t h_idx = h_tile * BLOCK_K;
#pragma unroll
                for (uint32_t j = 0; j < HC_MULT; ++j)
                {
                    tma_copy<BLOCK_K, BLOCK_M, kSwizzleResMode>(&tensor_map_residual, full_input[i_stage],
                        smem_res[i_stage] + j * BLOCK_M * BLOCK_K, j * HIDDEN + h_idx, m_idx);
                }
                tma_copy<BLOCK_K, BLOCK_M, kSwizzleXMode>(
                    &tensor_map_x, full_input[i_stage], smem_x_stg[i_stage], h_idx, m_idx);
                constexpr uint32_t kInputBytes = SMEM_RES_PER_ISTG + SMEM_X_PER_ISTG;
                full_input[i_stage]->arrive_and_expect_tx(kInputBytes);

#pragma unroll
                for (uint32_t hc = 0; hc < HC_MULT; ++hc)
                {
                    empty_B[b_stage]->wait(((s / N_B_STAGES) & 1) ^ 1);
                    uint32_t k_idx = hc * HIDDEN + h_idx;
                    tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(
                        &tensor_map_b, full_B[b_stage], smem_b[b_stage], k_idx, 0);
                    full_B[b_stage]->arrive_and_expect_tx(SMEM_B_PER_STAGE);
                    b_stage = (b_stage + 1) % N_B_STAGES;
                    ++s;
                }
                i_stage = (i_stage + 1) % N_INPUT_STAGES;
            }
        }

        // ----- MMA issue warp (warp 1) -----
        if (warp_idx == 1)
        {
            constexpr uint32_t UMMA_M = BLOCK_M;
            constexpr uint32_t UMMA_N = BLOCK_N;
            constexpr uint32_t UMMA_K = 32 / sizeof(float);
            constexpr uint32_t BLOCK_SWIZZLED_BK = kSwizzleBMode / sizeof(float);
            using umma_t = cute::SM100_MMA_TF32_TS<cutlass::tfloat32_t, cutlass::tfloat32_t, float, BLOCK_M, BLOCK_N,
                kMajorA, kMajorB>;
            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::tfloat32_t, cutlass::tfloat32_t, float, UMMA_M,
                UMMA_N, kMajorA, kMajorB>();
            auto const& runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
            static_assert(N_B_STAGES <= 32, "Too many B stages");
            auto b_desc = make_umma_desc<kMajorB, BLOCK_N, BLOCK_SWIZZLED_BK, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t const& b_desc_lo = lane_idx < N_B_STAGES ? b_desc.lo + lane_idx * SMEM_B_PER_STAGE / 16 : 0u;

            for (uint32_t s = 0; s < num_total_stages; ++s)
            {
                const uint32_t b_stage = s % N_B_STAGES;
                const uint32_t cast_stage_idx = s % kNumCastStages;
                full_cast[cast_stage_idx]->wait((s / kNumCastStages) & 1);
                full_B[b_stage]->wait((s / N_B_STAGES) & 1);
                tcgen05_after_thread_sync();
                auto const& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(b_stage));
#pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k)
                {
                    uint32_t const& atom_idx = (k * UMMA_K) / BLOCK_SWIZZLED_BK;
                    uint32_t const& in_atom_idx = (k * UMMA_K) % BLOCK_SWIZZLED_BK;
                    uint32_t const& offset = atom_idx * BLOCK_N * BLOCK_SWIZZLED_BK;
                    b_desc.lo = advance_umma_desc_lo<kMajorB, BLOCK_N, kSwizzleBMode, float>(
                        b_desc_base_lo, offset, in_atom_idx);
                    umma_t::fma(BLOCK_K * cast_stage_idx + k * UMMA_K, b_desc, BLOCK_K * kNumCastStages, s > 0 or k > 0,
                        runtime_instr_desc);
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_cast[cast_stage_idx]));
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_B[b_stage]));
            }
            cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barrier));
        }

        // ----- Epilogue (warps 0..3, 128 threads) -----
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(float);
        static_assert(BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        tmem_full_barrier->wait(0);
        tcgen05_after_thread_sync();

#pragma unroll
        for (uint32_t i = 0; i < BLOCK_N / kNumElemsPerBankGroup; ++i)
        {
            uint32_t tmem_addr = BLOCK_K * kNumCastStages + i * kNumElemsPerBankGroup;
            auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd) + warp_idx * BLOCK_M / 4 * kSwizzleCDMode
                + get_swizzled_smem_offset<kSwizzleCDMode>(i, lane_idx);
            uint32_t values[kNumElemsPerBankGroup];
            static_assert(kNumElemsPerBankGroup == 4, "Invalid type");
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
            cutlass::arch::fence_view_async_tmem_load();
            if (BLOCK_M == 128 or (BLOCK_M == 64 and lane_idx < 16))
                deep_gemm::ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
            if constexpr (BLOCK_M == 64)
                __syncwarp();
        }
        cutlass::arch::NamedBarrier::sync(kNumMMAThreads, 0);

        constexpr uint32_t kTotalOut = BLOCK_M * SHAPE_N;
        const uint32_t tid = threadIdx.x;
#pragma unroll
        for (uint32_t k = tid; k < kTotalOut; k += kNumMMAThreads)
        {
            uint32_t m = k / SHAPE_N;
            uint32_t n = k - m * SHAPE_N;
            uint32_t gm = m_block_idx * BLOCK_M + m;
            if (gm < shape_m)
            {
                uint32_t col_group = n >> 2;
                uint32_t in_group = n & 3;
                uint32_t phys_col_grp = col_group ^ (m & 7);
                uint32_t byte = m * kSwizzleCDMode + phys_col_grp * kNumBankGroupBytes + in_group * sizeof(float);
                float val = *reinterpret_cast<float const*>(reinterpret_cast<uint8_t const*>(smem_cd) + byte);
                atomicAdd(&D[gm * SHAPE_N + n], val);
            }
        }

        if (warp_idx == 1)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
    else
    {
        // ----- Pmap warp group (warps 4..7, 128 threads) -----
        const uint32_t sub_warp_idx = warp_idx - kNumMMAThreads / 32;
        const uint32_t upper_row = sub_warp_idx * 16 + lane_idx / 4;
        const uint32_t lower_row = upper_row + 8;
        const uint32_t col_lane = lane_idx % 4;

        float pm_u[HC_MULT], pm_l[HC_MULT];
        float cm_u[HC_MULT][HC_MULT], cm_l[HC_MULT][HC_MULT];
#pragma unroll
        for (uint32_t hc = 0; hc < HC_MULT; ++hc)
        {
            pm_u[hc] = smem_post[upper_row * HC_MULT + hc];
            pm_l[hc] = smem_post[lower_row * HC_MULT + hc];
        }
#pragma unroll
        for (uint32_t j = 0; j < HC_MULT; ++j)
        {
#pragma unroll
            for (uint32_t hc = 0; hc < HC_MULT; ++hc)
            {
                cm_u[j][hc] = smem_comb[upper_row * HC_MULT * HC_MULT + j * HC_MULT + hc];
                cm_l[j][hc] = smem_comb[lower_row * HC_MULT * HC_MULT + j * HC_MULT + hc];
            }
        }

        float sqr_u = 0.f, sqr_l = 0.f;
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(nv_bfloat16);
        constexpr uint32_t kNumLoads = BLOCK_K / kNumElemsPerBankGroup;
        constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;
        static_assert(BLOCK_K * sizeof(nv_bfloat16) == kSwizzleAMode, "BLOCK_K must match swizzle A mode");
        static_assert(kNumLoads % 2 == 0, "kNumLoads must be even for LDSM.x4");

        uint32_t s = 0;
        for (uint32_t ht = 0; ht < H_TILES_PER_SPLIT; ++ht)
        {
            const uint32_t i_stage = ht % N_INPUT_STAGES;
            full_input[i_stage]->wait((ht / N_INPUT_STAGES) & 1);

            uint32_t x_vals[2][kNumLoads];
            {
                uint8_t const* x_base
                    = reinterpret_cast<uint8_t*>(smem_x_stg[i_stage]) + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleXMode;
#pragma unroll
                for (uint32_t i = 0; i < kNumLoads; i += 2)
                {
                    auto smem_ptr = x_base + get_swizzled_smem_offset<kSwizzleXMode>(i + lane_idx / 16, lane_idx % 16);
                    deep_gemm::sm90::SM90_U32x4_LDSM_N::copy(x_vals[0][i + 0], x_vals[1][i + 0], x_vals[0][i + 1],
                        x_vals[1][i + 1], const_cast<uint8_t*>(smem_ptr));
                }
            }

            uint32_t r_vals[HC_MULT][2][kNumLoads];
#pragma unroll
            for (uint32_t j = 0; j < HC_MULT; ++j)
            {
                uint8_t const* r_base = reinterpret_cast<uint8_t*>(smem_res[i_stage])
                    + j * BLOCK_M * BLOCK_K * sizeof(nv_bfloat16) + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleResMode;
#pragma unroll
                for (uint32_t i = 0; i < kNumLoads; i += 2)
                {
                    auto smem_ptr
                        = r_base + get_swizzled_smem_offset<kSwizzleResMode>(i + lane_idx / 16, lane_idx % 16);
                    deep_gemm::sm90::SM90_U32x4_LDSM_N::copy(r_vals[j][0][i + 0], r_vals[j][1][i + 0],
                        r_vals[j][0][i + 1], r_vals[j][1][i + 1], const_cast<uint8_t*>(smem_ptr));
                }
            }

            float2 xf[2][kNumLoads];
#pragma unroll
            for (uint32_t u = 0; u < 2; ++u)
            {
#pragma unroll
                for (uint32_t i = 0; i < kNumLoads; ++i)
                {
                    xf[u][i] = __bfloat1622float2(*reinterpret_cast<nv_bfloat162*>(&x_vals[u][i]));
                }
            }

            // Wait for previous ht's residual_out TMA_STOREs to drain before we
            // overwrite single-buffered smem_rc with new hc values.
            if (ht > 0)
            {
                cute::tma_store_wait<0>();
            }

#pragma unroll
            for (uint32_t hc = 0; hc < HC_MULT; ++hc)
            {
                const uint32_t cast_stage_idx = s % kNumCastStages;
                empty_cast[cast_stage_idx]->wait(((s / kNumCastStages) & 1) ^ 1);

                uint32_t rc_u_buf[kNumLoads], rc_l_buf[kNumLoads];
#pragma unroll
                for (uint32_t i = 0; i < kNumLoads; ++i)
                {
                    float2 nu{pm_u[hc] * xf[0][i].x, pm_u[hc] * xf[0][i].y};
                    float2 nl{pm_l[hc] * xf[1][i].x, pm_l[hc] * xf[1][i].y};
#pragma unroll
                    for (uint32_t j = 0; j < HC_MULT; ++j)
                    {
                        float2 ruj = __bfloat1622float2(*reinterpret_cast<nv_bfloat162*>(&r_vals[j][0][i]));
                        float2 rlj = __bfloat1622float2(*reinterpret_cast<nv_bfloat162*>(&r_vals[j][1][i]));
                        nu.x = fmaf(cm_u[j][hc], ruj.x, nu.x);
                        nu.y = fmaf(cm_u[j][hc], ruj.y, nu.y);
                        nl.x = fmaf(cm_l[j][hc], rlj.x, nl.x);
                        nl.y = fmaf(cm_l[j][hc], rlj.y, nl.y);
                    }
                    nv_bfloat162 b_up = __float22bfloat162_rn(nu);
                    nv_bfloat162 b_lo = __float22bfloat162_rn(nl);
                    uint32_t b_up_bits = *reinterpret_cast<uint32_t*>(&b_up);
                    uint32_t b_lo_bits = *reinterpret_cast<uint32_t*>(&b_lo);
                    rc_u_buf[i] = b_up_bits;
                    rc_l_buf[i] = b_lo_bits;
                    float2 ru = __bfloat1622float2(b_up);
                    float2 rl = __bfloat1622float2(b_lo);
                    sqr_u = fmaf(ru.x, ru.x, sqr_u);
                    sqr_u = fmaf(ru.y, ru.y, sqr_u);
                    sqr_l = fmaf(rl.x, rl.x, sqr_l);
                    sqr_l = fmaf(rl.y, rl.y, sqr_l);
                    cute::SM100_TMEM_STORE_16dp256b1x::copy(*reinterpret_cast<uint32_t*>(&ru.x),
                        *reinterpret_cast<uint32_t*>(&ru.y), *reinterpret_cast<uint32_t*>(&rl.x),
                        *reinterpret_cast<uint32_t*>(&rl.y), cast_stage_idx * BLOCK_K + i * 8);
                }
                cutlass::arch::fence_view_async_tmem_store();
                tcgen05_before_thread_sync();
                full_cast[cast_stage_idx]->arrive();
                ++s;

                // STSM bf16 new_r values into smem_rc[hc] sub-region for this warp.
                uint8_t* rc_base = reinterpret_cast<uint8_t*>(smem_rc) + hc * SMEM_RC_PER_HC
                    + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleRoutMode;
#pragma unroll
                for (uint32_t i = 0; i < kNumLoads; i += 2)
                {
                    auto smem_ptr
                        = rc_base + get_swizzled_smem_offset<kSwizzleRoutMode>(i + lane_idx / 16, lane_idx % 16);
                    stsm_x4_b16_rout(smem_ptr, rc_u_buf[i + 0], rc_l_buf[i + 0], rc_u_buf[i + 1], rc_l_buf[i + 1]);
                }
            }
            empty_input[i_stage]->arrive();

            // Emit HC_MULT TMA_STOREs of residual_cur: one per hc slice, per-warp rows.
            cute::tma_store_fence();
            if (cute::elect_one_sync())
            {
                const uint32_t h_idx = (h_tile_start + ht) * BLOCK_K;
#pragma unroll
                for (uint32_t hc = 0; hc < HC_MULT; ++hc)
                {
                    uint8_t* rc_base = reinterpret_cast<uint8_t*>(smem_rc) + hc * SMEM_RC_PER_HC
                        + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleRoutMode;
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_residual_out, rc_base, hc * HIDDEN + h_idx,
                        m_offset + sub_warp_idx * BLOCK_M_PER_WARP);
                    cute::tma_store_arrive();
                }
            }
        }

        // Drain any in-flight residual_out TMA stores before exit.
        cute::tma_store_wait<0>();

        // Warp-reduce sqr across 4 col_lanes then atomicAdd to global.
        sqr_u += __shfl_xor_sync(0xffffffff, sqr_u, 1);
        sqr_u += __shfl_xor_sync(0xffffffff, sqr_u, 2);
        sqr_l += __shfl_xor_sync(0xffffffff, sqr_l, 1);
        sqr_l += __shfl_xor_sync(0xffffffff, sqr_l, 2);
        if (col_lane == 0)
        {
            uint32_t gm_u = m_block_idx * BLOCK_M + upper_row;
            uint32_t gm_l = m_block_idx * BLOCK_M + lower_row;
            if (gm_u < shape_m)
                atomicAdd(&sqr_sum[gm_u], sqr_u);
            if (gm_l < shape_m)
                atomicAdd(&sqr_sum[gm_l], sqr_l);
        }
    }

    // ========================================================================
    // Phase 3: cross-split barrier.
    // For kNumSplits == 1 we only need a block-scope fence + __syncthreads.
    // For kNumSplits  > 1 ALL splits participate in Phase 4: each CTA
    // increments done_counter, then spin-waits until the counter reaches
    // kNumSplits.  Phase 4 work is then partitioned across CTAs by
    // k_split_idx — CTA i processes tokens
    //     [i * TOKS_PER_CTA, (i+1) * TOKS_PER_CTA)
    // where TOKS_PER_CTA = BLOCK_M / kNumSplits.  This replaces the old
    // "last-home CTA does all Phase 4 work serially" design that bottlenecked
    // Path D at BLOCK_M=64 (1 CTA's Phase 4 = ~28 µs regardless of M).
    // ========================================================================
    if constexpr (kNumSplits == 1)
    {
        __threadfence_block();
        __syncthreads();
    }
    else
    {
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(&done_counter[m_block_idx], 1);
            // Spin-wait until all kNumSplits CTAs finish Phase 2. The
            // atomicAdd(..., 0) is a zero-increment load with full device
            // coherence — cheap on B200 and avoids an extra flag allocation.
            while (atomicAdd(&done_counter[m_block_idx], 0) < static_cast<int>(kNumSplits))
            {
                /* spin */
            }
        }
        __syncthreads();
    }

    // ========================================================================
    // Phase 4: inline bigFuse for this CTA's subset of BLOCK_M tokens.
    //   y_acc[tok, 0..HC_MULT)           -> pre_mix (sigmoid)
    //   y_acc[tok, HC_MULT..2*HC_MULT)   -> post_mix_out
    //   y_acc[tok, 2*HC_MULT..HC_MULT3)  -> comb_mix_out (sinkhorn)
    //   layer_input[tok, h] = sum_j pre_mix[tok, j] * residual_cur[tok, j, h]
    //
    // Token subset: CTA k_split_idx handles tokens
    //   [k_split_idx * TOKS_PER_CTA, (k_split_idx + 1) * TOKS_PER_CTA),
    // where TOKS_PER_CTA = ceil(BLOCK_M / kNumSplits).  At kNumSplits == 1
    // this is the whole m_block (64 tokens); at kNumSplits == 8 it's 8 tokens
    // spread over 8 CTAs running Phase 4 concurrently — ~8× the Phase-4
    // throughput of the old single-last-home-CTA design.
    //
    // Within a CTA, tokens are parallelized across warps: each warp handles
    // max(1, TOKS_PER_CTA / NUM_WARPS_BF) tokens serially.  Within a warp,
    // lanes 0..HC_MULT-1 compute rmsnorm/sigmoid/sinkhorn; pre_mix is
    // broadcast via __shfl_sync so all 32 lanes run the layer_input dot
    // product across HIDDEN=4096.
    // ========================================================================
    constexpr uint32_t BLOCK_SIZE_BF = kNumMMAThreads + kNumPmapThreads; // 256
    constexpr uint32_t WARP_SIZE_BF = 32;
    constexpr uint32_t NUM_WARPS_BF = BLOCK_SIZE_BF / WARP_SIZE_BF;      // 8
    constexpr uint32_t TOKS_PER_CTA = (BLOCK_M + kNumSplits - 1) / kNumSplits;
    // When TOKS_PER_CTA < NUM_WARPS_BF (large kNumSplits / small BLOCK_M case),
    // have WARPS_PER_TOK warps cooperate on the HIDDEN-stride layer_input loop
    // so all 8 warps stay active. Example at BLOCK_M=64, kNumSplits=16:
    //   TOKS_PER_CTA=4, WARPS_PER_TOK=2, TOKS_PER_PASS=4, TOKEN_PASSES=1 —
    //   2 warps per token each sweep HIDDEN/2, zero idle warps.
    constexpr uint32_t WARPS_PER_TOK = (NUM_WARPS_BF > TOKS_PER_CTA) ? (NUM_WARPS_BF / TOKS_PER_CTA) : 1u;
    constexpr uint32_t TOKS_PER_PASS = NUM_WARPS_BF / WARPS_PER_TOK;
    constexpr uint32_t TOKEN_PASSES = (TOKS_PER_CTA + TOKS_PER_PASS - 1) / TOKS_PER_PASS;
    constexpr uint32_t BF16_VEC_LI = 8;
    static_assert(HIDDEN % (WARPS_PER_TOK * WARP_SIZE_BF * BF16_VEC_LI) == 0,
        "HIDDEN must be a multiple of WARPS_PER_TOK * WARP_SIZE * BF16_VEC_LI");
    const uint32_t tid_bf = threadIdx.x;
    const uint32_t lane_bf = tid_bf % WARP_SIZE_BF;
    const uint32_t warp_bf = tid_bf / WARP_SIZE_BF;
    const uint32_t warp_tok_pos = warp_bf / WARPS_PER_TOK; // which token in a pass
    const uint32_t warp_in_team = warp_bf % WARPS_PER_TOK; // which warp inside team
    const uint32_t cta_tok_base = k_split_idx * TOKS_PER_CTA;

#pragma unroll 1
    for (uint32_t pass = 0; pass < TOKEN_PASSES; ++pass)
    {
        const uint32_t t_in_cta = pass * TOKS_PER_PASS + warp_tok_pos;
        if (t_in_cta >= TOKS_PER_CTA)
            continue;
        const uint32_t t = cta_tok_base + t_in_cta;
        if (t >= BLOCK_M)
            continue;
        const uint32_t tok = m_offset + t;
        if (tok >= shape_m)
            continue;

        // Lanes 0..HC_MULT-1 compute rmsnorm / sigmoid / sinkhorn; pre_mix is
        // held in `pre_mix_local` on lanes 0..HC_MULT-1 and later broadcast to
        // all 32 lanes via __shfl_sync.  All warps in a team redundantly run
        // these ~tens of FLOPs (cheap) to avoid a cross-warp SMEM sync; only
        // warp_in_team==0 writes comb_mix_out / post_mix_out to GMEM.
        float pre_mix_local = 0.f;
        if (lane_bf < HC_MULT)
        {
            float const r_val = sqr_sum[tok];
            float y_local[HC_MULT3];
            float const* y_row = D + static_cast<long long>(tok) * SHAPE_N;
#pragma unroll
            for (uint32_t c = 0; c < HC_MULT3; ++c)
                y_local[c] = y_row[c];

            float const rstd = rsqrtf(r_val / static_cast<float>(HC_MULT * HIDDEN) + rms_eps);
            float const s0 = hc_scale[0];
            float const s1 = hc_scale[1];
            float const s2 = hc_scale[2];

            float v = y_local[lane_bf] * rstd * s0 + hc_base[lane_bf];
            pre_mix_local = 1.0f / (1.0f + __expf(-v)) + hc_pre_eps;

            v = y_local[HC_MULT + lane_bf] * rstd * s1 + hc_base[HC_MULT + lane_bf];
            float post_val = 1.0f / (1.0f + __expf(-v)) * hc_post_mult_value;
            if (warp_in_team == 0)
            {
                post_mix_out[tok * HC_MULT + lane_bf] = post_val;
            }

            float cm_vals[HC_MULT];
#pragma unroll
            for (uint32_t k = 0; k < HC_MULT; ++k)
                cm_vals[k] = y_local[2 * HC_MULT + lane_bf * HC_MULT + k] * rstd * s2
                    + hc_base[2 * HC_MULT + lane_bf * HC_MULT + k];

            constexpr unsigned LANE_MASK = (1u << HC_MULT) - 1;
            float const rowMax = fmaxf(fmaxf(cm_vals[0], cm_vals[1]), fmaxf(cm_vals[2], cm_vals[3]));
#pragma unroll
            for (uint32_t k = 0; k < HC_MULT; ++k)
                cm_vals[k] = __expf(cm_vals[k] - rowMax);
            float rs = cm_vals[0] + cm_vals[1] + cm_vals[2] + cm_vals[3];
#pragma unroll
            for (uint32_t k = 0; k < HC_MULT; ++k)
                cm_vals[k] = cm_vals[k] / rs + hc_sinkhorn_eps;
#pragma unroll
            for (uint32_t k = 0; k < HC_MULT; ++k)
            {
                float cs = cm_vals[k];
                cs += __shfl_xor_sync(LANE_MASK, cs, 1);
                cs += __shfl_xor_sync(LANE_MASK, cs, 2);
                cm_vals[k] /= (cs + hc_sinkhorn_eps);
            }
            for (uint32_t it = 1; it < sinkhorn_repeat; ++it)
            {
                rs = cm_vals[0] + cm_vals[1] + cm_vals[2] + cm_vals[3] + hc_sinkhorn_eps;
#pragma unroll
                for (uint32_t k = 0; k < HC_MULT; ++k)
                    cm_vals[k] /= rs;
#pragma unroll
                for (uint32_t k = 0; k < HC_MULT; ++k)
                {
                    float cs = cm_vals[k];
                    cs += __shfl_xor_sync(LANE_MASK, cs, 1);
                    cs += __shfl_xor_sync(LANE_MASK, cs, 2);
                    cm_vals[k] /= (cs + hc_sinkhorn_eps);
                }
            }
            if (warp_in_team == 0)
            {
                float* cm_out_ptr = comb_mix_out + tok * HC_MULT2;
#pragma unroll
                for (uint32_t k = 0; k < HC_MULT; ++k)
                    cm_out_ptr[lane_bf * HC_MULT + k] = cm_vals[k];
            }
        }

        // Broadcast pre_mix[j] from lane j to all 32 lanes (intra-warp shfl).
        float pm[HC_MULT];
#pragma unroll
        for (uint32_t j = 0; j < HC_MULT; ++j)
            pm[j] = __shfl_sync(0xffffffff, pre_mix_local, j);

        // Layer_input[tok, h] = sum_j pm[j] * residual_cur[tok, j, h].
        // When WARPS_PER_TOK>1, warp_in_team 0..WARPS_PER_TOK-1 together cover
        // HIDDEN in strides of WARPS_PER_TOK * 32 * 8.  When WARPS_PER_TOK==1,
        // each warp sweeps HIDDEN alone (same as the original single-warp case).
        __nv_bfloat16 const* rbase = residual_cur_ptr + static_cast<long long>(tok) * HC_MULT * HIDDEN;
        __nv_bfloat16* obase = layer_input_out + static_cast<long long>(tok) * HIDDEN;

        constexpr uint32_t H_STRIDE = WARPS_PER_TOK * WARP_SIZE_BF * BF16_VEC_LI;
        const uint32_t h_start = warp_in_team * WARP_SIZE_BF * BF16_VEC_LI + lane_bf * BF16_VEC_LI;

#pragma unroll
        for (uint32_t h = h_start; h < HIDDEN; h += H_STRIDE)
        {
            // Issue all HC_MULT=4 residual_cur reads first so their L2 latency
            // is hidden by the bf16→fp32 arithmetic that follows.  The compiler
            // schedules the 4 independent __ldg's in parallel.
            uint4 raws[HC_MULT];
#pragma unroll
            for (uint32_t j = 0; j < HC_MULT; ++j)
            {
                raws[j] = __ldg(reinterpret_cast<uint4 const*>(&rbase[j * HIDDEN + h]));
            }
            float acc_li[BF16_VEC_LI] = {};
#pragma unroll
            for (uint32_t j = 0; j < HC_MULT; ++j)
            {
                __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&raws[j]);
#pragma unroll
                for (uint32_t v = 0; v < BF16_VEC_LI / 2; ++v)
                {
                    float2 f = __bfloat1622float2(pairs[v]);
                    acc_li[2 * v + 0] += pm[j] * f.x;
                    acc_li[2 * v + 1] += pm[j] * f.y;
                }
            }
            uint4 out_raw;
            __nv_bfloat162* opairs = reinterpret_cast<__nv_bfloat162*>(&out_raw);
#pragma unroll
            for (uint32_t v = 0; v < BF16_VEC_LI / 2; ++v)
                opairs[v] = __float22bfloat162_rn(make_float2(acc_li[2 * v], acc_li[2 * v + 1]));
            *reinterpret_cast<uint4*>(&obase[h]) = out_raw;
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100a");
#endif
}

} // namespace fused_mhc

#pragma clang diagnostic pop
