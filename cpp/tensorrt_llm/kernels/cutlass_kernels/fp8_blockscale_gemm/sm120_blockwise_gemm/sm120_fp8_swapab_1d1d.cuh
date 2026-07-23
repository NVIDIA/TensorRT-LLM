/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "sm120_swapab_utils.cuh"
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_conversion.h>

namespace sm120_swapab
{

using namespace cute;

/**
 * SwapAB QMMA Kernel
 *
 * Receives ALREADY-SWAPPED arguments from dispatch:
 *   A_kernel = B_user [N_real × K],  B_kernel = A_user [M_real × K]
 *   problem_shape = (N_real, M_real, K, L)
 *
 * Computes: accum[TileM × TileN] via TMA + QMMA mainloop
 * Epilogue: STG scatter store accum[i][j] → D_real[j][i]
 *   D_real is [M_real × N_real] row-major, ld_D = N_real
 *   Transposed write: stride = (1, ld_D) in the (TileM, TileN) view
 */
template <typename KT>
struct SM120BlockScaledSwapABKernel
{
    static constexpr int kNumTMAThreads = KT::kNumTMAThreads;   // 64
    static constexpr int kNumMathThreads = KT::kNumMathThreads; // 128
    static constexpr int MaxThreadsPerBlock = kNumMathThreads + kNumTMAThreads;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    using ProblemShape = typename KT::ProblemShape;

    struct Params
    {
        typename KT::TMA_A tma_load_a;
        typename KT::TMA_B tma_load_b;
        typename KT::TMA_SFA tma_load_sfa;
        typename KT::TMA_SFB tma_load_sfb;
        typename KT::ElementD* ptr_D;
        int64_t ld_D;               // leading dim of D_real (= N_real = M_kernel)
        ProblemShape problem_shape; // (N_real, M_real, K, L) — already swapped
        float* ptr_workspace;       // FP32 workspace for sliced-K atomicAdd (nullptr = no slicing)
        int k_slices;               // number of K slices (1 = no slicing)
    };

    struct Arguments
    {
        typename KT::ElementA* ptr_A;
        typename KT::StrideA dA;
        typename KT::ElementB* ptr_B;
        typename KT::StrideB dB;
        typename KT::ElementSFLoad* ptr_SFA;
        typename KT::StrideSFA dSFA;
        typename KT::ElementSFLoad* ptr_SFB;
        typename KT::StrideSFB dSFB;
        typename KT::ElementD* ptr_D;
        typename KT::StrideD dD;
    };

    static Params to_underlying_arguments(ProblemShape const& ps, Arguments const& args)
    {
        auto [M, N, K, L] = ps; // M = N_real, N = M_real (swapped)

        auto tensor_A = make_tensor(make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, L), args.dA));
        auto tma_load_a = make_tma_copy(SM90_TMA_LOAD{}, tensor_A, typename KT::SmemLayoutA{}(_, _, Int<0>{}),
            make_shape(shape<0>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});

        auto tensor_B = make_tensor(make_gmem_ptr(args.ptr_B), make_layout(make_shape(N, K, L), args.dB));
        auto tma_load_b = make_tma_copy(SM90_TMA_LOAD{}, tensor_B, typename KT::SmemLayoutB{}(_, _, Int<0>{}),
            make_shape(shape<1>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});

        auto sfa_layout = KT::deduce_sfa_layout(ps);
        auto sfb_layout = KT::deduce_sfb_layout(ps);

        auto tensor_sfa = make_tensor(make_gmem_ptr(args.ptr_SFA), sfa_layout);
        auto tma_load_sfa = make_tma_copy(SM90_TMA_LOAD{}, tensor_sfa, typename KT::SmemLayoutSFA{}(_, _, Int<0>{}),
            make_shape(shape<0>(typename KT::ScaleTileShape{}), shape<2>(typename KT::ScaleTileShape{})), _1{});

        auto tensor_sfb = make_tensor(make_gmem_ptr(args.ptr_SFB), sfb_layout);
        auto tma_load_sfb = make_tma_copy(SM90_TMA_LOAD{}, tensor_sfb, typename KT::SmemLayoutSFB{}(_, _, Int<0>{}),
            make_shape(shape<1>(typename KT::ScaleTileShape{}), shape<2>(typename KT::ScaleTileShape{})), _1{});

        int64_t ld_D = get<0>(args.dD);

        return {tma_load_a, tma_load_b, tma_load_sfa, tma_load_sfb, args.ptr_D, ld_D, ps, nullptr, 1};
    }

    static Params to_underlying_arguments(ProblemShape const& ps, Arguments const& args, float* workspace, int k_slices)
    {
        auto p = to_underlying_arguments(ps, args);
        p.ptr_workspace = workspace;
        p.k_slices = k_slices;
        return p;
    }

    // Grid: tile over M_kernel (= N_real) and N_kernel (= M_real), z = k_slices
    static dim3 get_grid_shape(Params const& p)
    {
        auto [M, N, K, L] = p.problem_shape;
        return dim3((M + KT::kTileM - 1) / KT::kTileM, (N + KT::kTileN - 1) / KT::kTileN, L * p.k_slices);
    }

    static dim3 get_block_shape()
    {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    struct SharedStorage
    {
        typename KT::TensorStorage tensors;
        alignas(16) typename KT::BarrierStorage barriers;
    };

    static constexpr int kSmemSize = int(sizeof(SharedStorage));

    // ---- Barrier helpers ----
    CUTE_DEVICE static auto get_mbarriers(SharedStorage& ss)
    {
        using FullBarrier = typename KT::FullBarrier;
        using EmptyBarrier = typename KT::EmptyBarrier;
        auto* ab_full_mbar = recast_ptr<FullBarrier>(&ss.barriers.ab_full_mbar[0]);
        auto* ab_empty_mbar = recast_ptr<EmptyBarrier>(&ss.barriers.ab_empty_mbar[0]);
        auto* sf_full_mbar = recast_ptr<FullBarrier>(&ss.barriers.sf_full_mbar[0]);
        auto* sf_empty_mbar = recast_ptr<EmptyBarrier>(&ss.barriers.sf_empty_mbar[0]);
        return cute::make_tuple(ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar);
    }

    CUTE_DEVICE static void prefetch_tma_descriptors(Params const& params)
    {
        cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
    }

    // ---- MMA mainloop + STG scatter epilogue ----
    CUTE_DEVICE static void mma_and_epilogue(
        Params const& params, SharedStorage& ss, int32_t sf_tile_count, uint32_t& sf_phase, uint32_t& ab_phase)
    {
        int thread_idx = int(threadIdx.x);

        auto sA_ = make_tensor(make_smem_ptr(ss.tensors.load.smem_A.begin()), typename KT::SmemLayoutA{});
        auto sB_ = make_tensor(make_smem_ptr(ss.tensors.load.smem_B.begin()), typename KT::SmemLayoutB{});
        auto sSFA_ = make_tensor(make_smem_ptr(ss.tensors.load.smem_SFA.begin()), typename KT::SmemLayoutSFA{});
        auto sSFB_ = make_tensor(make_smem_ptr(ss.tensors.load.smem_SFB.begin()), typename KT::SmemLayoutSFB{});
        auto sA = as_position_independent_swizzle_tensor(sA_);
        auto sB = as_position_independent_swizzle_tensor(sB_);
        auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

        typename KT::TiledMma mma;
        auto tile_shape_mnk = tile_shape(mma);
        auto thr_mma = mma.get_thread_slice(thread_idx);
        auto accum = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));

        auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
        auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));

        // S2R: A, B
        auto s2r_copy_A = make_tiled_copy_A(typename KT::SmemCopyAtomA{}, mma);
        auto s2r_thr_A = s2r_copy_A.get_thread_slice(thread_idx);
        auto tXsA = s2r_thr_A.partition_S(sA);
        auto tXrA = s2r_thr_A.retile_D(tCrA);

        auto s2r_copy_B = make_tiled_copy_B(typename KT::SmemCopyAtomB{}, mma);
        auto s2r_thr_B = s2r_copy_B.get_thread_slice(thread_idx);
        auto tXsB = s2r_thr_B.partition_S(sB);
        auto tXrB = s2r_thr_B.retile_D(tCrB);

        // S2R: SFA, SFB
        auto s2r_copy_SFA = make_tiled_copy_impl(
            typename KT::SmemCopyAtomSF{}, KT::get_layoutSFA_TV(mma), make_shape(size<0>(tile_shape_mnk), _1{}));
        auto s2r_thr_SFA = s2r_copy_SFA.get_thread_slice(thread_idx);
        auto tXsSFA = s2r_thr_SFA.partition_S(sSFA);
        auto tCrSFA = KT::partition_fragment_SFA(sSFA(_, _, Int<0>{}), thr_mma);
        auto tXrSFA = s2r_thr_SFA.retile_D(tCrSFA);
        auto tCrSFA_frg = KT::transform_fragment_for_qmma(tCrSFA);

        auto s2r_copy_SFB = make_tiled_copy_impl(
            typename KT::SmemCopyAtomSF{}, KT::get_layoutSFB_TV(mma), make_shape(size<1>(tile_shape_mnk), _1{}));
        auto s2r_thr_SFB = s2r_copy_SFB.get_thread_slice(thread_idx);
        auto tXsSFB = s2r_thr_SFB.partition_S(sSFB);
        auto tCrSFB = KT::partition_fragment_SFB(sSFB(_, _, Int<0>{}), thr_mma);
        auto tXrSFB = s2r_thr_SFB.retile_D(tCrSFB);
        auto tCrSFB_frg = KT::transform_fragment_for_qmma(tCrSFB);

        cute::clear(accum);
        auto mbars = get_mbarriers(ss);
        auto ab_full_mbar = cute::get<0>(mbars);
        auto ab_empty_mbar = cute::get<1>(mbars);
        auto sf_full_mbar = cute::get<2>(mbars);
        auto sf_empty_mbar = cute::get<3>(mbars);

        // Main loop: sf_tile_count - 1 iterations
        for (int32_t sf_tile_idx = 0; sf_tile_idx < sf_tile_count - 1; ++sf_tile_idx)
        {
            sf_full_mbar[0].wait(sf_phase);
            cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, Int<0>{}), tXrSFA);
            cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, Int<0>{}), tXrSFB);
            sf_empty_mbar[0].arrive();

            cute::for_each(cute::make_int_sequence<KT::kNumStagePerSF>{},
                [&](auto iter)
                {
                    cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                        [&](auto read_stage)
                        {
                            ab_full_mbar[read_stage].wait(ab_phase);
                            cute::copy(s2r_copy_A, tXsA(_, _, _, read_stage), tXrA);
                            cute::copy(s2r_copy_B, tXsB(_, _, _, read_stage), tXrB);
                            ab_empty_mbar[read_stage].arrive();

                            constexpr int sf_idx = (iter * KT::AB_Stages + read_stage) / KT::kKTilesPerScale;
                            auto tCrSFA_stage = tCrSFA_frg(_, _, _, Int<sf_idx>{});
                            auto tCrSFB_stage = tCrSFB_frg(_, _, _, Int<sf_idx>{});
                            cute::gemm(
                                mma, make_zip_tensor(tCrA, tCrSFA_stage), make_zip_tensor(tCrB, tCrSFB_stage), accum);
                        });
                    ab_phase ^= 1;
                });
            sf_phase ^= 1;
        }

        // Tail iteration
        sf_full_mbar[0].wait(sf_phase);
        cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, Int<0>{}), tXrSFA);
        cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, Int<0>{}), tXrSFB);
        sf_empty_mbar[0].arrive();

        cute::for_each(cute::make_int_sequence<KT::kNumStagePerSF>{},
            [&](auto iter)
            {
                cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                    [&](auto read_stage)
                    {
                        ab_full_mbar[read_stage].wait(ab_phase);
                        cute::copy(s2r_copy_A, tXsA(_, _, _, read_stage), tXrA);
                        cute::copy(s2r_copy_B, tXsB(_, _, _, read_stage), tXrB);
                        ab_empty_mbar[read_stage].arrive();

                        constexpr int sf_idx = (iter * KT::AB_Stages + read_stage) / KT::kKTilesPerScale;
                        auto tCrSFA_stage = tCrSFA_frg(_, _, _, Int<sf_idx>{});
                        auto tCrSFB_stage = tCrSFB_frg(_, _, _, Int<sf_idx>{});
                        cute::gemm(
                            mma, make_zip_tensor(tCrA, tCrSFA_stage), make_zip_tensor(tCrB, tCrSFB_stage), accum);
                    });
                ab_phase ^= 1;
            });
        sf_phase ^= 1;

        // ---- Epilogue ----
        int M_k = cute::get<0>(params.problem_shape);
        int N_k = cute::get<1>(params.problem_shape);
        int K_k = cute::get<2>(params.problem_shape);
        int L_k = cute::get<3>(params.problem_shape);
        int m_block = blockIdx.x;
        int n_block = blockIdx.y;
        int m_base = m_block * KT::kTileM;
        int n_base = n_block * KT::kTileN;

        auto cD = make_identity_tensor(make_shape(Int<KT::kTileM>{}, Int<KT::kTileN>{}));
        auto tCcD = thr_mma.partition_C(cD);
        constexpr int kAccumSize = decltype(cute::size(accum))::value;

        if (params.ptr_workspace)
        {
            // Sliced-K: atomicAdd FP32 to workspace.
            // blockIdx.z encodes (l, slice) when k_slices>1 — same decode as operator() below.
            int l_coord = (params.k_slices > 1) ? (int(blockIdx.z) / params.k_slices) : int(blockIdx.z);
            // Per-batch workspace stride: N_real * M_real (kernel-view M_k * N_k).
            int64_t ws_stride_l = (int64_t) M_k * (int64_t) N_k;
            cute::for_each(cute::make_int_sequence<kAccumSize>{},
                [&](auto i)
                {
                    int m_in = get<0>(tCcD(i));
                    int n_in = get<1>(tCcD(i));
                    if (m_base + m_in < M_k && n_base + n_in < N_k)
                    {
                        // Workspace is tight (ws_stride_l = M_k * N_k); use M_k as the inner
                        // stride. Don't use params.ld_D — that's the caller's output ldd,
                        // which can differ from M_k when `out` is a strided view.
                        int64_t addr = l_coord * ws_stride_l + (int64_t) (m_base + m_in)
                            + (int64_t) (n_base + n_in) * (int64_t) M_k;
                        atomicAdd(&params.ptr_workspace[addr], accum(i));
                    }
                });
        }
        else
        {
            // No slicing: direct STG scatter BF16
            int l_coord = blockIdx.z;
            auto mD = make_tensor(make_gmem_ptr(params.ptr_D),
                make_layout(
                    make_shape(M_k, N_k, L_k), make_stride(Int<1>{}, params.ld_D, (int64_t) N_k * params.ld_D)));
            auto gD_mnl = local_tile(mD, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, _1, Underscore>{});
            auto gD = gD_mnl(_, _, m_block, n_block, l_coord);
            auto tCgD = thr_mma.partition_C(gD);
            auto convert = cutlass::NumericConverter<typename KT::ElementD, float>{};

            cute::for_each(cute::make_int_sequence<kAccumSize>{},
                [&](auto i)
                {
                    int m_in = get<0>(tCcD(i));
                    int n_in = get<1>(tCcD(i));
                    if (m_base + m_in < M_k && n_base + n_in < N_k)
                    {
                        tCgD(i) = convert(accum(i));
                    }
                });
        }
    }

    // ---- Kernel entry ----
    CUTE_DEVICE void operator()(Params const& params, char* smem_buf)
    {
        SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_buf);
        int thread_idx = int(threadIdx.x);
        int warp_idx = canonical_warp_idx_sync();
        int lane_predicate = cute::elect_one_sync();
        bool is_tma_warp = (warp_idx >= KT::kNumMathWarps);

        auto [M, N, K, L] = params.problem_shape;
        // Sliced-K: blockIdx.z encodes (l_coord, k_slice_idx) laid out as l_coord * k_slices + slice.
        // When k_slices==1, blockIdx.z == l_coord (single-slice fast path preserved).
        int k_slice_idx = (params.k_slices > 1) ? (int(blockIdx.z) % params.k_slices) : 0;
        int l_coord = (params.k_slices > 1) ? (int(blockIdx.z) / params.k_slices) : int(blockIdx.z);
        int K_per_slice = K / params.k_slices;
        int32_t sf_tile_count = K_per_slice / 512;
        int32_t k_tile_offset = k_slice_idx * (K_per_slice / KT::kTileK);
        int32_t sf_tile_offset = k_slice_idx * sf_tile_count;

        // Barrier init (thread 0)
        if (thread_idx == 0)
        {
            auto mbars = get_mbarriers(ss);
            auto ab_full_mbar = cute::get<0>(mbars);
            auto ab_empty_mbar = cute::get<1>(mbars);
            auto sf_full_mbar = cute::get<2>(mbars);
            auto sf_empty_mbar = cute::get<3>(mbars);
            for (uint32_t i = 0; i < KT::AB_Stages; ++i)
            {
                ab_full_mbar[i].init(1);
                ab_empty_mbar[i].init(kNumMathThreads);
            }
            for (uint32_t i = 0; i < KT::SF_Stages; ++i)
            {
                sf_full_mbar[i].init(1);
                sf_empty_mbar[i].init(kNumMathThreads);
            }
            cutlass::arch::fence_barrier_init();
        }
        if (is_tma_warp && lane_predicate)
        {
            prefetch_tma_descriptors(params);
        }
        __syncthreads();

        if (is_tma_warp)
        {
            // Warp specialization: AB load warp + SF load warp (separate)
            constexpr int ab_warp_idx = KT::kNumMathWarps;     // warp 4
            constexpr int sf_warp_idx = KT::kNumMathWarps + 1; // warp 5

            int m_block = blockIdx.x;
            int n_block = blockIdx.y;
            // l_coord is already decoded above from blockIdx.z.

            if (warp_idx == ab_warp_idx && lane_predicate)
            {
                // AB load warp
                uint32_t ab_phase = 1;
                using X = Underscore;
                auto mA = params.tma_load_a.get_tma_tensor(make_shape(M, K, L));
                auto mB = params.tma_load_b.get_tma_tensor(make_shape(N, K, L));
                auto gA_mkl = local_tile(mA, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
                auto gB_nkl = local_tile(mB, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
                auto block_tma_a = params.tma_load_a.get_slice(0);
                auto block_tma_b = params.tma_load_b.get_slice(0);
                auto gA = gA_mkl(_, _, m_block, _, l_coord);
                auto gB = gB_nkl(_, _, n_block, _, l_coord);
                auto tAgA = block_tma_a.partition_S(gA);
                auto tBgB = block_tma_b.partition_S(gB);
                auto sA = as_position_independent_swizzle_tensor(
                    make_tensor(make_smem_ptr(ss.tensors.load.smem_A.begin()), typename KT::SmemLayoutA{}));
                auto sB = as_position_independent_swizzle_tensor(
                    make_tensor(make_smem_ptr(ss.tensors.load.smem_B.begin()), typename KT::SmemLayoutB{}));
                auto tAsA = block_tma_a.partition_D(sA);
                auto tBsB = block_tma_b.partition_D(sB);

                auto mbars = get_mbarriers(ss);
                auto ab_full_mbar = cute::get<0>(mbars);
                auto ab_empty_mbar = cute::get<1>(mbars);
                auto sf_full_mbar = cute::get<2>(mbars);
                auto sf_empty_mbar = cute::get<3>(mbars);
                int32_t k_tile_count = sf_tile_count * KT::kNumTileKPerSF;
                for (int32_t k = 0; k < k_tile_count; k += KT::AB_Stages)
                {
                    cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                        [&](auto ws)
                        {
                            ab_empty_mbar[ws].wait(ab_phase);
                            auto& ab_full_barrier = ab_full_mbar[ws];
                            auto tma_a = params.tma_load_a.with(
                                *recast_ptr<typename KT::ProducerBarrierType>(&ab_full_barrier));
                            cute::copy(tma_a, tAgA(_, _, _, k_tile_offset + k + ws), tAsA(_, _, _, ws));
                            auto tma_b = params.tma_load_b.with(
                                *recast_ptr<typename KT::ProducerBarrierType>(&ab_full_barrier));
                            cute::copy(tma_b, tBgB(_, _, _, k_tile_offset + k + ws), tBsB(_, _, _, ws));
                            ab_full_mbar[ws].arrive_and_expect_tx(KT::TmaABTransactionBytes);
                        });
                    ab_phase ^= 1;
                }
            }
            if (warp_idx == sf_warp_idx && lane_predicate)
            {
                // SF load warp
                uint32_t sf_phase = 1;
                using X = Underscore;
                auto mSFA = params.tma_load_sfa.get_tma_tensor(shape(KT::deduce_sfa_layout(params.problem_shape)));
                auto mSFB = params.tma_load_sfb.get_tma_tensor(shape(KT::deduce_sfb_layout(params.problem_shape)));
                auto gSFA_mkl = local_tile(mSFA, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
                auto gSFB_nkl = local_tile(mSFB, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
                auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
                auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
                auto gSFA = gSFA_mkl(_, _, m_block, _, l_coord);
                auto gSFB = gSFB_nkl(_, _, n_block, _, l_coord);
                auto tAgSFA = block_tma_sfa.partition_S(gSFA);
                auto tBgSFB = block_tma_sfb.partition_S(gSFB);
                auto sSFA = as_position_independent_swizzle_tensor(
                    make_tensor(make_smem_ptr(ss.tensors.load.smem_SFA.begin()), typename KT::SmemLayoutSFA{}));
                auto sSFB = as_position_independent_swizzle_tensor(
                    make_tensor(make_smem_ptr(ss.tensors.load.smem_SFB.begin()), typename KT::SmemLayoutSFB{}));
                auto tAsSFA = block_tma_sfa.partition_D(sSFA);
                auto tBsSFB = block_tma_sfb.partition_D(sSFB);

                auto mbars = get_mbarriers(ss);
                auto ab_full_mbar = cute::get<0>(mbars);
                auto ab_empty_mbar = cute::get<1>(mbars);
                auto sf_full_mbar = cute::get<2>(mbars);
                auto sf_empty_mbar = cute::get<3>(mbars);
                for (int32_t i = 0; i < sf_tile_count; ++i)
                {
                    sf_empty_mbar[0].wait(sf_phase);
                    auto& sf_full_barrier = sf_full_mbar[0];
                    auto tma_sfa
                        = params.tma_load_sfa.with(*recast_ptr<typename KT::ProducerBarrierType>(&sf_full_barrier));
                    cute::copy(tma_sfa, tAgSFA(_, _, _, sf_tile_offset + i), tAsSFA(_, _, _, Int<0>{}));
                    auto tma_sfb
                        = params.tma_load_sfb.with(*recast_ptr<typename KT::ProducerBarrierType>(&sf_full_barrier));
                    cute::copy(tma_sfb, tBgSFB(_, _, _, sf_tile_offset + i), tBsSFB(_, _, _, Int<0>{}));
                    sf_full_mbar[0].arrive_and_expect_tx(KT::TmaSFTransactionBytes);
                    sf_phase ^= 1;
                }
            }
            // Other TMA warps idle
        }
        else
        {
            // Math warps
            uint32_t sf_phase = 0, ab_phase = 0;
            mma_and_epilogue(params, ss, sf_tile_count, sf_phase, ab_phase);
        }
    }
};

} // namespace sm120_swapab
