/*
 * MIT License
 *
 * Copyright (c) 2025 DeepSeek
 * Copyright (c) 2026 NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Adapted from DeepGEMM's SM100 1D1D FP8/FP4 GEMM implementation.
 */

#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

// DeepGEMM is normally JIT-compiled and exposes several host/device helpers
// containing PTX. TRT-LLM compiles this translation unit with clang as NVCC's
// host compiler, so keep those helpers out of the host pass.
#pragma push_macro("CUTLASS_HOST_DEVICE")
#undef CUTLASS_HOST_DEVICE
#define CUTLASS_HOST_DEVICE CUTLASS_DEVICE
#include <deep_gemm/common/math.cuh>
#pragma pop_macro("CUTLASS_HOST_DEVICE")

#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/utils.cuh>

#include "dsv4Fp8SplitKScheduler.cuh"
#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mhc::dsv4_splitk
{

using namespace deep_gemm;

CUTLASS_DEVICE void clusterSyncWithRelaxedArrive()
{
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB, uint32_t kGranKA, uint32_t kGranKB, uint32_t SHAPE_M,
    uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
    uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode, uint32_t kNumStages,
    uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads, uint32_t kNumMulticast, bool kIsMulticastOnA,
    uint32_t kNumSMs, bool kSwapAB, GemmType kGemmType, bool kWithAccumulation, typename a_dtype_t, typename b_dtype_t,
    typename cd_dtype_t, typename epilogue_type_t, uint32_t kSplitKFactor = 1>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1) dsv4Fp8SplitKGemmKernel(
    int* grouped_layout, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
    __grid_constant__ const cute::TmaDescriptor tensor_map_a, __grid_constant__ const cute::TmaDescriptor tensor_map_b,
    __grid_constant__ const cute::TmaDescriptor tensor_map_sfa,
    __grid_constant__ const cute::TmaDescriptor tensor_map_sfb,
    __grid_constant__ const cute::TmaDescriptor tensor_map_cd, void* gmem_split_partials)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // C/D type: BF16 and FP32 are supported, with or without accumulation
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
        "Invalid C/D data dtype");
    DG_STATIC_ASSERT(kSplitKFactor == 1 or kGemmType == GemmType::Normal, "Split-K only supports normal GEMM");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal, "DSV4 split-K only supports normal GEMM");

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M)
            or (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)),
        "Invalid block size");

    // SF configs
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
    DG_STATIC_ASSERT(kGranKA == 32 or kGranKA == 128, "Invalid granularity K for A");
    DG_STATIC_ASSERT(kGranKB == 32 or kGranKB == 128, "Invalid granularity K for B");
    DG_STATIC_ASSERT(
        (kGemmType != GemmType::KGroupedContiguous) or kGranKA == kGranKB, "K-grouped SF requires kGranKA == kGranKB");

    // Epilogue configs
    // Always enable pipeline for better performance
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M = kSwapAB ? 16 : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads : STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    DG_STATIC_ASSERT(
        SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
        "Shared memory of A/B must be aligned to 1024 bytes");
    // NOTES: Make sure we have enough shared memory for UMMA padding
    constexpr uint32_t UMMA_A_SIZE_PER_STAGE
        = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype_t);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages,
        "Memory Out of bound for UMMA");

    // Tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols
        = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? clusterSyncWithRelaxedArrive() : void();

    // Utils
    bool const is_leader_cta = cute::block_rank_in_cluster() == 0;
    auto const warp_idx = cutlass::canonical_warp_idx_sync();
    auto const lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0)
    {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    auto const shape_sfa_k = math::ceil_div(shape_k, kGranKA * 4);
    auto const shape_sfb_k = math::ceil_div(shape_k, kGranKB * 4);

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor(
        [&](uint32_t const& i) { return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE); });
    auto smem_a = utils::PatternVisitor([&](uint32_t const& i)
        { return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE); });
    auto smem_b = utils::PatternVisitor(
        [&](uint32_t const& i)
        {
            return reinterpret_cast<b_dtype_t*>(
                smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        });

    // SFA/SFB shared memory
    auto sf_start_ptr = reinterpret_cast<uint8_t*>(smem_b[kNumStages]);
    auto smem_sfa = utils::PatternVisitor(
        [=](uint32_t const& i) { return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE); });
    auto smem_sfb = utils::PatternVisitor(
        [=](uint32_t const& i)
        {
            return reinterpret_cast<uint32_t*>(
                sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
        });

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_sfb[kNumStages]);
    ;
    auto full_barriers = utils::PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + (i); });
    auto empty_barriers
        = utils::PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers
        = utils::PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers
        = utils::PatternVisitor([=](uint32_t const& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers = utils::PatternVisitor(
        [=](uint32_t const& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync())
    {
#pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i)
        {
            // Arrive at all CTAs
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // Arrive only at the leader CTA
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
#pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++i)
        {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    else if (warp_idx == 2)
    {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? clusterSyncWithRelaxedArrive() : __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = SplitKScheduler<BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA, kNumSMs, kSplitKFactor>(
        shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx)
    {
        ++k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync())
    {
        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            // Use dynamic load block M, when swap-AB is enabled
            auto const load_block_m
                = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            auto const num_all_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            DG_TRAP_ONLY_DEVICE_ASSERT(num_all_k_blocks % kSplitKFactor == 0);
            auto const num_total_k_blocks = num_all_k_blocks / kSplitKFactor;
            auto const k_block_offset = scheduler.split_k_idx * num_total_k_blocks;
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx
                    = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), IndexType::MN>(
                        shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), IndexType::MN>(
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous
                        or kGemmType == GemmType::Batched or kMajorA == cute::UMMA::Major::K,
                    "Invalid major");
                uint32_t const global_k_block_idx = k_block_offset + k_block_idx;
                uint32_t k_idx = global_k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), IndexType::K>(
                    shape_k, BLOCK_K, global_k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), IndexType::K>(
                    shape_k, BLOCK_K, global_k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1)
                {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                uint32_t const batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, 1, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_b_idx, n_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, 1, batch_idx);
                auto num_arrival_bytes
                    = SMEM_A_SIZE_PER_STAGE / (std::is_same_v<a_dtype_t, cutlass::float_e4m3_t> ? 1 : 2)
                    + SMEM_B_SIZE_PER_STAGE / (std::is_same_v<b_dtype_t, cutlass::float_e4m3_t> ? 1 : 2);

                // Issue SFA and SFB TMAs at certain stages
                // No swizzling, so one TMA for one SF is enough
                if (k_block_idx % kNumSFAStagesPerLoad == 0)
                {
                    uint32_t sfa_m_idx = m_block_idx * BLOCK_M;
                    uint32_t sfa_k_idx
                        = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), IndexType::SF_K>(
                            shape_sfa_k, 1, math::ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad));
                    tma::copy<BLOCK_M, 1, 0>(
                        &tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx);
                    num_arrival_bytes += BLOCK_M * sizeof(uint32_t);
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0)
                {
                    uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                    uint32_t sfb_k_idx = scheduler.template get_global_idx<true, IndexType::SF_K>(
                        shape_sfb_k, 1, math::ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad), m_block_idx);
                    tma::copy<BLOCK_N, 1, 0>(
                        &tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx);
                    num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
                }

                // Arrive at full barriers
                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
            }
        }
    }
    else if (warp_idx == 1 and is_leader_cta)
    {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc_block_scaled<b_dtype_t, a_dtype_t, float,
                              cutlass::float_ue8m0_t, UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, float,
                                      cutlass::float_ue8m0_t, UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64 and UMMA_N % 8 == 0 and 8 <= UMMA_N and UMMA_N <= 256)
                or (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256)
                or (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
            "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            // Wait tensor memory empty barrier arrival
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](bool const& do_tmem_full_arrive)
            {
                auto umma_arrive = [](uint64_t const* barrier)
                {
                    if constexpr (kNumMulticast == 1)
                    {
                        cutlass::arch::umma_arrive(barrier);
                    }
                    else
                    {
                        constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    }
                };
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic update of UMMA N based on effective M, when swap-AB is enabled
            if constexpr (kSwapAB)
            {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }

            // Launch MMAs
            auto const num_all_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            DG_TRAP_ONLY_DEVICE_ASSERT(num_all_k_blocks % kSplitKFactor == 0);
            auto const num_total_k_blocks = num_all_k_blocks / kSplitKFactor;
#pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                // Wait TMA and SF-transpose arrival
                with_sf_full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                auto const a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                auto const b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                if (cute::elect_one_sync())
                {
                    // Do SF copy at certain stages
                    // TODO: process shared memory descriptor by addition
                    using cute_utccp_t = cute::conditional_t<kNumMulticast == 1, cute::SM100_UTCCP_4x32dp128bit_1cta,
                        cute::SM100_UTCCP_4x32dp128bit_2cta>;
                    uint32_t const sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
                    if (sfa_stage_in_group_idx == 0)
                    {
#pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++i)
                        {
                            auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                    }
                    uint32_t const sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad;
                    if (sfb_stage_in_group_idx == 0)
                    {
#pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++i)
                        {
                            auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                        }
                    }

                    // Issue UMMA
                    using mma_t = cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_MXF8F6F4_SS,
                        ptx::SM100_MMA_MXF8F6F4_2x1SM_SS>;
#pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k)
                    {
                        uint32_t const sfa_id = (kGranKA == 32 ? k : sfa_stage_in_group_idx);
                        uint32_t const sfb_id = (kGranKB == 32 ? k : sfb_stage_in_group_idx);
                        auto const runtime_instr_desc = kSwapAB
                            ? mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfb_id, sfa_id)
                            : mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                            a_desc_base_lo, 0, k * UMMA_K);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            b_desc_base_lo, 0, k * UMMA_K);
                        if constexpr (kSwapAB)
                        {
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N, k_block_idx > 0 or k > 0,
                                runtime_instr_desc, kTmemStartColOfSFB, kTmemStartColOfSFA);
                        }
                        else
                        {
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N, k_block_idx > 0 or k > 0,
                                runtime_instr_desc, kTmemStartColOfSFA, kTmemStartColOfSFB);
                        }
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by
                // `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        auto const iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0)
        {
            auto const accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }
    }
    else if (warp_idx == 2)
    {
        // UTCCP transposer
        auto utccp_required_smem_warp_transpose = [&](uint32_t const* smem_ptr)
        {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            auto const num_all_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            DG_TRAP_ONLY_DEVICE_ASSERT(num_all_k_blocks % kSplitKFactor == 0);
            auto const num_total_k_blocks = num_all_k_blocks / kSplitKFactor;
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);

                // Transpose for UTCCP at certain stages
                if (k_block_idx % kNumSFAStagesPerLoad == 0)
                {
#pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++i)
                        utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0)
                {
#pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++i)
                        utccp_required_smem_warp_transpose(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }

                // Arrive
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    }
    else if (warp_idx >= kNumNonEpilogueThreads / 32
        and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32)
    {
        // Epilogue warp groups
        auto const epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            auto const tmem_base_addr = accum_stage_idx * UMMA_N;
            auto const base_m_idx
                = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), IndexType::MN>(
                    shape_m, BLOCK_M, m_block_idx);
            auto const base_n_idx = n_block_idx * BLOCK_N;
            if constexpr (kSplitKFactor > 1)
            {
                // The regular TMA epilogue is tied to its 2D CD descriptor.
                // Like the SM120 split-K path, write accumulators directly to
                // a contiguous workspace. BF16 matches the downstream mHC
                // input and avoids a separate FP32-to-BF16 cast kernel.
                DG_STATIC_ASSERT(kSwapAB, "Split-K workspace store requires swap-AB");
                DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                    "Split-K workspace store requires FP32 or BF16 output");
                constexpr uint32_t kRowsPerTmemLoad = 8;
                constexpr uint32_t kColsPerEpilogueWarp = 32;
                auto const effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                auto const num_m_stores = effective_m / STORE_BLOCK_M;
                auto const split_offset = static_cast<uint64_t>(scheduler.split_k_idx) * shape_m * shape_n;

                for (uint32_t s = 0; s < num_m_stores; ++s)
                {
#pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / kRowsPerTmemLoad; ++i)
                    {
                        uint32_t const tmem_addr = tmem_base_addr + s * STORE_BLOCK_M + i * kRowsPerTmemLoad;
                        uint32_t values[kRowsPerTmemLoad];
                        cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr, values[0], values[1], values[2], values[3],
                            values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        uint32_t const col = base_n_idx + epilogue_warp_idx * kColsPerEpilogueWarp + lane_idx;
#pragma unroll
                        for (uint32_t row = 0; row < kRowsPerTmemLoad; ++row)
                        {
                            uint32_t const global_row = base_m_idx + s * STORE_BLOCK_M + i * kRowsPerTmemLoad + row;
                            if (global_row < shape_m and col < shape_n)
                            {
                                auto const idx = split_offset + static_cast<uint64_t>(global_row) * shape_n + col;
                                auto const value = __uint_as_float(values[row]);
                                if constexpr (cute::is_same_v<cd_dtype_t, float>)
                                    static_cast<float*>(gmem_split_partials)[idx] = value;
                                else
                                    static_cast<cutlass::bfloat16_t*>(gmem_split_partials)[idx]
                                        = cutlass::bfloat16_t(value);
                            }
                        }
                    }
                }

                ptx::tcgen05_before_thread_sync();
                tmem_empty_barriers[accum_stage_idx]->arrive(0u);
            }
            else if constexpr (kSwapAB)
            {
                auto const effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                epilogue::sm100_store_cd_swap_ab<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N, kSwizzleCDMode,
                    kNumTMAStoreStages, kNumUMMAStoreThreads, kGemmType, kWithAccumulation, cd_dtype_t,
                    epilogue_type_t>(smem_cd, tma_stage_idx, tmem_base_addr, base_m_idx, base_n_idx,
                    scheduler.current_group_idx, effective_m, epilogue_warp_idx, lane_idx,
                    tmem_empty_barriers[accum_stage_idx], tensor_map_cd);
            }
            else
            {
                epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N, kSwizzleCDMode,
                    kNumTMAStoreStages, kNumUMMAStoreThreads, kGemmType, kWithAccumulation, cd_dtype_t,
                    epilogue_type_t>(smem_cd, tma_stage_idx, tmem_base_addr, base_m_idx, base_n_idx,
                    scheduler.current_group_idx, epilogue_warp_idx, lane_idx, tmem_empty_barriers[accum_stage_idx],
                    tensor_map_cd);
            }
        }
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? clusterSyncWithRelaxedArrive() : __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace kernels::mhc::dsv4_splitk

TRTLLM_NAMESPACE_END

#pragma clang diagnostic pop
