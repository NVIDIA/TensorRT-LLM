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

#include "sm120_utils.cuh"

using namespace cute;

namespace sm120_blockscaled_gemm
{

template <typename KT>
struct SM120BlockScaledKernel
{

    static constexpr int kNumTMAThreads = 128;
    static constexpr int kNumMathThreads = 128;
    static constexpr int MaxThreadsPerBlock = kNumTMAThreads + kNumMathThreads;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    using ProblemShape = typename KT::ProblemShape;

    struct Params
    {
        typename KT::TMA_A tma_load_a;
        typename KT::TMA_B tma_load_b;
        typename KT::TMA_SFA tma_load_sfa;
        typename KT::TMA_SFB tma_load_sfb;
        typename KT::TMA_D tma_store_d;
        typename KT::ProblemShape problem_shape;
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

    using TileShape = typename KT::TileShape;
    using ScaleTileShape = typename KT::ScaleTileShape;

    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args)
    {
        auto [M, N, K, L] = problem_shape;
        auto tensor_A = make_tensor(make_gmem_ptr(args.ptr_A), make_layout(make_shape(M, K, L), args.dA));
        typename KT::TMA_A tma_load_a
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_A, typename KT::SmemLayoutA{}(_, _, Int<0>{}),
                make_shape(shape<0>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});

        auto tensor_B = make_tensor(make_gmem_ptr(args.ptr_B), make_layout(make_shape(N, K, L), args.dB));
        typename KT::TMA_B tma_load_b
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_B, typename KT::SmemLayoutB{}(_, _, Int<0>{}),
                make_shape(shape<1>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});

        auto sfa_layout = KT::deduce_sfa_layout(problem_shape);
        auto sfb_layout = KT::deduce_sfb_layout(problem_shape);

        auto tensor_sfa = make_tensor(make_gmem_ptr(args.ptr_SFA), sfa_layout);
        auto tensor_sfb = make_tensor(make_gmem_ptr(args.ptr_SFB), sfb_layout);

        typename KT::TMA_SFA tma_load_sfa
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_sfa, typename KT::SmemLayoutSFA{}(_, _, Int<0>{}),
                make_shape(shape<0>(typename KT::ScaleTileShape{}), shape<2>(typename KT::ScaleTileShape{})), _1{});

        typename KT::TMA_SFB tma_load_sfb
            = make_tma_copy(SM90_TMA_LOAD{}, tensor_sfb, typename KT::SmemLayoutSFB{}(_, _, Int<0>{}),
                make_shape(shape<1>(typename KT::ScaleTileShape{}), shape<2>(typename KT::ScaleTileShape{})), _1{});

        auto tensor_d = make_tensor(make_gmem_ptr(args.ptr_D), make_layout(make_shape(M, N, L), args.dD));
        auto tma_store_d = make_tma_copy_C_sm90(
            typename KT::CopyOpS2G{}, tensor_d, take<0, 2>(typename KT::SmemLayoutD{}), typename KT::EpilogueTile_MN{});
        return {tma_load_a, tma_load_b, tma_load_sfa, tma_load_sfb, tma_store_d, problem_shape};
    }

    static dim3 get_grid_shape(Params const& params)
    {
        return KT::get_grid_shape(params.problem_shape);
    }

    static dim3 get_block_shape()
    {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTE_DEVICE
    static void prefetch_tma_descriptors(Params const& params)
    {
        cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
    }

    CUTE_DEVICE
    static auto load_init(Params const& params)
    {
        using X = Underscore;
        auto [M, N, K, L] = params.problem_shape;

        auto mA_mkl = params.tma_load_a.get_tma_tensor(make_shape(M, K, L));
        auto mB_nkl = params.tma_load_b.get_tma_tensor(make_shape(N, K, L));
        auto mSFA_mkl = params.tma_load_sfa.get_tma_tensor(shape(KT::deduce_sfa_layout(params.problem_shape)));
        auto mSFB_nkl = params.tma_load_sfb.get_tma_tensor(shape(KT::deduce_sfb_layout(params.problem_shape)));

        // Make tiled views, defer the slice
        auto gA_mkl = local_tile(
            mA_mkl, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});        // (BLK_M,BLK_K,m,k,l)
        auto gB_nkl = local_tile(
            mB_nkl, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});        // (BLK_N,BLK_K,n,k,l)
        auto gSFA_mkl = local_tile(
            mSFA_mkl, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<_1, X, _1>{}); // (TILE_M,TILE_K,m,k,l)
        auto gSFB_nkl = local_tile(
            mSFB_nkl, typename KT::ScaleTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (TILE_N,TILE_K,n,k,l)

        return cute::make_tuple(gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl);
    }

    template <class Accumulator, class SharedStorage, class BlockCoord>
    CUTE_DEVICE void tma_store(
        Accumulator const& accum, SharedStorage& shared_storage, Params const& params, BlockCoord const& blk_coord)
    {
        auto const math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        auto accum_frg = recast<Array<typename KT::ElementAccum, 2>>(accum);
        auto epi = make_fragment_like<typename KT::ElementD>(accum);
        auto epi_frg = recast<Array<typename KT::ElementD, 2>>(epi);
        cutlass::NumericArrayConverter<typename KT::ElementD, typename KT::ElementAccum, 2> converter;
        cute::for_each(
            cute::make_int_sequence<cute::size(epi_frg)>{}, [&](auto i) { epi_frg(i) = converter(accum_frg(i)); });

        int thread_idx = int(threadIdx.x);
        typename KT::TiledMma tiled_mma;
        auto tiled_copy_C_atom = make_tiled_copy_C_atom(typename KT::CopyAtomC{}, tiled_mma);

        auto tiled_copy_r2s
            = make_tiled_copy_S(cute::Copy_Atom<typename KT::CopyOpR2S, typename KT::ElementD>{}, tiled_copy_C_atom);
        auto thr_copy_r2s = tiled_copy_r2s.get_slice(thread_idx);

        auto sD_epi_ = make_tensor(make_smem_ptr(shared_storage.tensors.store.smem_D.begin()),
            typename KT::SmemLayoutD{});                                     // (BLK_M,BLK_K,PIPE)
        auto sD_epi = cute::as_position_independent_swizzle_tensor(sD_epi_); // (EPI_TILE_M,EPI_TILE_N,PIPE_D)
        auto tRS_rD = thr_copy_r2s.retile_S(epi);
        auto tRS_sD = thr_copy_r2s.partition_D(sD_epi);

        using EpilogueTile = typename KT::EpilogueTile_MN;
        auto [M, N, K, L] = params.problem_shape;
        auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
        auto mD_mn = params.tma_store_d.get_tma_tensor(make_shape(M, N, L)); // (M,N,L)
        auto mD = coalesce(mD_mn, take<0, 2>(TileShape{}));
        auto gD = local_tile(mD, take<0, 2>(TileShape{}), make_coord(m_coord, n_coord, l_coord));

        auto gD_epi = flat_divide(gD, EpilogueTile{}); // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

        auto block_tma_d = params.tma_store_d.get_slice(Int<0>{});
        auto bSG_sD = block_tma_d.partition_S(sD_epi); // (TMA,TMA_M,TMA_K, PIP)
        auto bSG_gD = block_tma_d.partition_D(gD_epi); // (TMA,TMA_M,TMA_K, EPI_M, EPI_N)

        cutlass::arch::NamedBarrier::sync(128, math_wg_idx);
        copy(tiled_copy_r2s, tRS_rD, tRS_sD(_, _, _, Int<0>{}));

        uint32_t elect_one_thr = cute::elect_one_sync();
        uint32_t elect_one_warp = (thread_idx / 32 == 0);
        bool is_tma_store = elect_one_warp && elect_one_thr;
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(128, math_wg_idx);
        if (is_tma_store)
        {
            for (int epi_n = 0; epi_n < size<3>(bSG_gD); ++epi_n)
            {
                for (int epi_m = 0; epi_m < size<2>(bSG_gD); ++epi_m)
                {
                    cute::copy(params.tma_store_d, bSG_sD(_, _, _, Int<0>{}), bSG_gD(_, _, _, epi_m, epi_n));
                }
            }
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
        cutlass::arch::NamedBarrier::sync(128, math_wg_idx);
    }

    using TensorStorage = typename KT::TensorStorage;
    using BarrierStorage = typename KT::BarrierStorage;

    struct SharedStorage
    {
        TensorStorage tensors;
        alignas(16) BarrierStorage barriers;
    };

    static constexpr int kSmemSize = int(sizeof(SharedStorage));

    CUTE_DEVICE
    void operator()(Params const& params, char* smem_buf)
    {

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
        int thread_idx = int(threadIdx.x);
        int lane_idx = canonical_lane_idx();
        int warp_idx = canonical_warp_idx_sync();
        int warp_group_idx = canonical_warp_group_idx();
        int lane_predicate = cute::elect_one_sync();
        bool is_tma_thread = warp_idx == 0 && lane_predicate;

        if (is_tma_thread)
        {
            prefetch_tma_descriptors(params);
        }
        __syncthreads();

        // producer part
        auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
            typename KT::SmemLayoutA{});                       // (BLK_M,BLK_K,PIPE)
        auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()), typename KT::SmemLayoutB{});
        auto sA = as_position_independent_swizzle_tensor(sA_); // (BLK_M,BLK_K,PIPE)
        auto sB = as_position_independent_swizzle_tensor(sB_); // (BLK_N,BLK_K,PIPE)
        auto sSFA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
            typename KT::SmemLayoutSFA{});                     // (BLK_M,BLK_K,PIPE)
        auto sSFB_
            = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()), typename KT::SmemLayoutSFB{});
        auto sSFA = as_position_independent_swizzle_tensor(sSFA_); // (BLK_M,BLK_K,PIPE)
        auto sSFB = as_position_independent_swizzle_tensor(sSFB_); // (BLK_N,BLK_K,PIPE)

        auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_init(params);
        auto block_tma_a = params.tma_load_a.get_slice(0);
        auto block_tma_b = params.tma_load_b.get_slice(0);
        auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
        auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

        auto m_coord = idx2crd(int(blockIdx.x), shape<2>(gA_mkl));
        auto n_coord = idx2crd(int(blockIdx.y), shape<2>(gB_nkl));
        auto l_coord = idx2crd(int(blockIdx.z), shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

        auto gA = gA_mkl(_, _, m_coord, _, l_coord);
        auto gB = gB_nkl(_, _, n_coord, _, l_coord);

        auto tAgA = block_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_K,k)
        auto tAsA = block_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_K,PIPE)
        auto tBgB = block_tma_b.partition_S(gB); // (TMA,TMA_N,TMA_K,k)
        auto tBsB = block_tma_b.partition_D(sB); // (TMA,TMA_N,TMA_K,PIPE)

        auto gSFA = gSFA_mkl(_, _, m_coord, _, l_coord);
        auto gSFB = gSFB_nkl(_, _, n_coord, _, l_coord);

        auto tAgSFA = block_tma_sfa.partition_S(gSFA); // (TMA,TMA_M,TMA_K,k)
        auto tAsSFA = block_tma_sfa.partition_D(sSFA); // (TMA,TMA_M,TMA_K,PIPE)
        auto tBgSFB = block_tma_sfb.partition_S(gSFB); // (TMA,TMA_N,TMA_K,k)
        auto tBsSFB = block_tma_sfb.partition_D(sSFB); // (TMA,TMA_N,TMA_K,PIPE)

        // consumer part

        typename KT::TiledMma mma;
        auto tile_shape_mnk = tile_shape(mma);
        auto thr_mma = mma.get_thread_slice(thread_idx);
        auto accum = partition_fragment_C(mma, cute::take<0, 2>(TileShape{})); // (MMA,MMA_M,MMA_N)
        auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));          // (MMA,MMA_M,MMA_K)
        auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));          // (MMA,MMA_N,MMA_K)

        auto s2r_copy_A = make_tiled_copy_A(typename KT::SmemCopyAtomA{}, mma);
        auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
        auto tXsA = s2r_thr_copy_A.partition_S(sA); // (CPY,CPY_M,CPY_K,PIPE)
        auto tXrA = s2r_thr_copy_A.retile_D(tCrA);  // (CPY,CPY_M,CPY_K)

        auto s2r_copy_B = make_tiled_copy_B(typename KT::SmemCopyAtomB{}, mma);
        auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
        auto tXsB = s2r_thr_copy_B.partition_S(sB); // (CPY,CPY_M,CPY_K,PIPE)
        auto tXrB = s2r_thr_copy_B.retile_D(tCrB);  // (CPY,CPY_M,CPY_K)

        auto s2r_copy_SFA = make_tiled_copy_impl(
            typename KT::SmemCopyAtomSF{}, KT::get_layoutSFA_TV(mma), make_shape(size<0>(tile_shape(mma)), _1{}));
        auto s2r_thr_copy_SFA = s2r_copy_SFA.get_thread_slice(thread_idx);
        auto tXsSFA = s2r_thr_copy_SFA.partition_S(sSFA);                        // (CPY,CPY_M,CPY_K,PIPE)
        auto tCrSFA = KT::partition_fragment_SFA(sSFA(_, _, Int<0>{}), thr_mma); // (MMA,MMA_M,MMA_K)
        auto tXrSFA = s2r_thr_copy_SFA.retile_D(tCrSFA);
        auto tCrSFA_frg = KT::transform_fragment_for_qmma(tCrSFA);

        auto s2r_copy_SFB = make_tiled_copy_impl(
            typename KT::SmemCopyAtomSF{}, KT::get_layoutSFB_TV(mma), make_shape(size<1>(tile_shape(mma)), _1{}));
        auto s2r_thr_copy_SFB = s2r_copy_SFB.get_thread_slice(thread_idx);
        auto tXsSFB = s2r_thr_copy_SFB.partition_S(sSFB);                        // (CPY,CPY_M,CPY_K,PIPE)
        auto tCrSFB = KT::partition_fragment_SFB(sSFB(_, _, Int<0>{}), thr_mma); // (MMA,MMA_N,MMA_K)
        auto tXrSFB = s2r_thr_copy_SFB.retile_D(tCrSFB);
        auto tCrSFB_frg = KT::transform_fragment_for_qmma(tCrSFB);

        using FullBarrier = typename KT::FullBarrier;
        using EmptyBarrier = typename KT::EmptyBarrier;
        using ProducerBarrierType = typename FullBarrier::ValueType;
        using ConsumerBarrierType = typename EmptyBarrier::ValueType;

        auto* ab_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.ab_full_mbar[0]);
        auto* ab_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.ab_empty_mbar[0]);
        auto* sf_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.sf_full_mbar[0]);
        auto* sf_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.sf_empty_mbar[0]);

        // init barriers
        if (is_tma_thread)
        {
#pragma unroll
            for (uint32_t i = 0; i < KT::SF_Stages; ++i)
            {
                sf_full_mbar[i].init(1);
                sf_empty_mbar[i].init(128);
            }

#pragma unroll
            for (uint32_t i = 0; i < KT::AB_Stages; ++i)
            {
                ab_full_mbar[i].init(1);
                ab_empty_mbar[i].init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        int32_t sf_tile_count = cute::size<2>(gSFA);
        clear(accum);

        if (warp_idx >= kNumMathThreads / 32)
        {
            if (warp_idx == kNumMathThreads / 32)
            {
                uint32_t phase = 1;
                if (lane_predicate)
                {
                    for (int32_t sf_tile_idx = 0; sf_tile_idx < sf_tile_count; ++sf_tile_idx)
                    {
                        sf_empty_mbar[0].wait(phase);
                        auto& sf_full_barrier = sf_full_mbar[0];
                        auto tma_copy_sfa
                            = params.tma_load_sfa.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
                        cute::copy(tma_copy_sfa, tAgSFA(_, _, _, sf_tile_idx), tAsSFA(_, _, _, Int<0>{}));
                        auto tma_copy_sfb
                            = params.tma_load_sfb.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
                        cute::copy(tma_copy_sfb, tBgSFB(_, _, _, sf_tile_idx), tBsSFB(_, _, _, Int<0>{}));
                        sf_full_mbar[0].arrive_and_expect_tx(KT::TmaSFTransactionBytes);

                        int32_t k_tile_idx = sf_tile_idx * 4;
                        CUTE_UNROLL
                        for (int32_t write_stage = 0; write_stage < KT::AB_Stages; ++write_stage)
                        {
                            ab_empty_mbar[write_stage].wait(phase);
                            auto& ab_full_barrier = ab_full_mbar[write_stage];
                            auto tma_copy_a
                                = params.tma_load_a.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
                            cute::copy(tma_copy_a, tAgA(_, _, _, k_tile_idx), tAsA(_, _, _, write_stage));
                            auto tma_copy_b
                                = params.tma_load_b.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
                            cute::copy(tma_copy_b, tBgB(_, _, _, k_tile_idx), tBsB(_, _, _, write_stage));
                            ab_full_mbar[write_stage].arrive_and_expect_tx(KT::TmaABTransactionBytes);
                            k_tile_idx += 1;
                        }
                        phase ^= 1;
                    }
                }
                __syncwarp();
            }
            cutlass::arch::NamedBarrier::sync(128, 2);
        }
        else
        {

            auto const math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
            clear(accum);
            uint32_t phase = 0;
            for (int32_t sf_tile_idx = 0; sf_tile_idx < sf_tile_count; ++sf_tile_idx)
            {
                sf_full_mbar[0].wait(phase);
                cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, Int<0>{}), tXrSFA);
                cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, Int<0>{}), tXrSFB);
                sf_empty_mbar[0].arrive();

                cute::for_each(cute::make_int_sequence<KT::AB_Stages>{},
                    [&](auto read_stage)
                    {
                        ab_full_mbar[read_stage].wait(phase);
                        cute::copy(s2r_copy_A, tXsA(_, _, _0{}, read_stage), tXrA(_, _, _0{}));
                        cute::copy(s2r_copy_B, tXsB(_, _, _0{}, read_stage), tXrB(_, _, _0{}));

                        auto K_BLOCK_MAX = size<2>(tCrA);
                        cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{},
                            [&](auto k_block)
                            {
                                if constexpr (k_block + 1 <= K_BLOCK_MAX - 1)
                                {
                                    cute::copy(
                                        s2r_copy_A, tXsA(_, _, k_block + 1, read_stage), tXrA(_, _, k_block + 1));
                                    cute::copy(
                                        s2r_copy_B, tXsB(_, _, k_block + 1, read_stage), tXrB(_, _, k_block + 1));
                                }
                                if constexpr (k_block + 1 == K_BLOCK_MAX - 1)
                                {
                                    ab_empty_mbar[read_stage].arrive();
                                }

                                auto tCrSFA_stage = tCrSFA_frg(_, _, _, read_stage);
                                auto tCrSFB_stage = tCrSFB_frg(_, _, _, read_stage);
                                cute::gemm(mma, make_zip_tensor(tCrA(_, _, k_block), tCrSFA_stage(_, _, k_block)),
                                    make_zip_tensor(tCrB(_, _, k_block), tCrSFB_stage(_, _, k_block)), accum);
                            });
                    });
                phase ^= 1;
            }
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);
            tma_store(accum, shared_storage, params, blk_coord);
        }
    }
};

} // namespace sm120_blockscaled_gemm
