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

#include "ada_blockwise_gemm_traits.cuh"

namespace ada_blockwise_gemm
{

template <typename KT>
struct AdaBlockwiseGemmKernel
{
    using Params = typename KT::Params;
    using Arguments = typename KT::Arguments;
    using SharedStorage = typename KT::SharedStorage;

    static constexpr int kThreadCount = KT::kThreadCount;
    static constexpr int kSmemSize = KT::kSmemSize;

    static dim3 get_grid_shape(GemmCoord problem_size)
    {

        int grid_m = (problem_size.m() + KT::kTileM - 1) / KT::kTileM;
        int grid_n = (problem_size.n() + KT::kTileN - 1) / KT::kTileN;
        int grid_k = 1;
        return dim3(grid_m, grid_n, grid_k);
    }

    static dim3 get_block_shape()
    {
        return dim3(kThreadCount, 1, 1);
    }

    static Params to_underlying_arguments(Arguments const& args)
    {
        return KT::to_underlying_arguments(args);
    }

    // Factory invocation
    CUTLASS_DEVICE
    static void invoke(Params const& params, SharedStorage& shared_storage)
    {
        AdaBlockwiseGemmKernel op;
        op(params, shared_storage);
    }

    CUTE_DEVICE auto gmem_tensor_init(Params const& params)
    {
        using X = cute::Underscore;

        int const M = params.problem_size.m();
        int const N = params.problem_size.n();
        int const K = params.problem_size.k();
        int const ScaleM = (((M + 3) >> 2) << 2); // align 4
        int const ScaleN = (N + KT::ScaleGranularityN - 1) / KT::ScaleGranularityN;
        int const ScaleK = (K + KT::ScaleGranularityK - 1) / KT::ScaleGranularityK;

        typename KT::ElementA const* ptr_A_ = params.ptr_a;
        typename KT::ElementB const* ptr_B_ = params.ptr_b;
        typename KT::ElementOutput* ptr_output_ = params.ptr_output;
        typename KT::ElementBlockScale const* ptr_scale_a_ = params.ptr_scale_a;
        typename KT::ElementBlockScale const* ptr_scale_b_ = params.ptr_scale_b;

        cute::Tensor mA_mk
            = cute::make_tensor(cute::make_gmem_ptr(ptr_A_), cute::make_shape(M, K), cute::make_stride(K, cute::_1{}));

        cute::Tensor mB_nk
            = cute::make_tensor(cute::make_gmem_ptr(ptr_B_), cute::make_shape(N, K), cute::make_stride(K, cute::_1{}));

        cute::Tensor mOutput_mn = cute::make_tensor(
            cute::make_gmem_ptr(ptr_output_), cute::make_shape(M, N), cute::make_stride(N, cute::_1{}));

        cute::Tensor mScaleA_mk = cute::make_tensor(
            cute::make_gmem_ptr(ptr_scale_a_), cute::make_shape(ScaleM, ScaleK), cute::make_stride(cute::_1{}, ScaleM));

        cute::Tensor mScaleB_nk = cute::make_tensor(
            cute::make_gmem_ptr(ptr_scale_b_), cute::make_shape(ScaleN, ScaleK), cute::make_stride(ScaleK, cute::_1{}));

        // partition the gmem tensor for each Cta
        cute::Tensor gA_mk = cute::local_tile(mA_mk, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<cute::_1, X, cute::_1>{}); // (BLK_M, BLK_K, m, k)

        cute::Tensor gB_nk = cute::local_tile(mB_nk, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<X, cute::_1, cute::_1>{}); // (BLK_N, BLK_K, n, k)

        cute::Tensor gOutput_mn = cute::local_tile(mOutput_mn, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<cute::_1, cute::_1, X>{}); // (BLK_M, BLK_N, m, n)

        cute::Tensor gScaleA_mk = cute::local_tile(mScaleA_mk, typename KT::ScalePerTileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<cute::_1, X, cute::_1>{}); // (BLK_M, BLK_K, m, k)

        cute::Tensor gScaleB_nk = cute::local_tile(mScaleB_nk, typename KT::ScalePerTileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<X, cute::_1, cute::_1>{}); // (BLK_N, BLK_K, n, k)

        return cute::make_tuple(gA_mk, gB_nk, gOutput_mn, gScaleA_mk, gScaleB_nk);
    }

    template <class TensorAccum, class TensorScaleA, class TensorScaleB>
    CUTE_DEVICE void promote(
        TensorAccum& accum, TensorAccum const& temp_accum, TensorScaleA const& tCrScaleA, TensorScaleB const& tCrScaleB)
    {

        using AccumType = typename TensorAccum::value_type;
        CUTE_UNROLL
        for (int mma_m = 0; mma_m < cute::get<1>(cute::shape<0>(accum)); ++mma_m)
        {
            AccumType sFA = tCrScaleA(mma_m);
            AccumType sFB = tCrScaleB(0);
            AccumType scale = sFA * sFB;
            CUTE_UNROLL
            for (int mma_n = 0; mma_n < cute::get<0>(cute::shape<0>(accum)); ++mma_n)
            {
                CUTE_UNROLL
                for (int mma_iter_m = 0; mma_iter_m < cute::size<1>(accum); ++mma_iter_m)
                {
                    CUTE_UNROLL
                    for (int mma_iter_n = 0; mma_iter_n < cute::size<2>(accum); ++mma_iter_n)
                    {
                        auto coord = cute::make_coord(cute::make_coord(mma_n, mma_m), mma_iter_m, mma_iter_n);
                        accum(coord) += temp_accum(coord) * scale;
                    }
                }
            }
        }
    }

    /// Executes one GEMM
    CUTE_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
        int const block_m_idx = blockIdx.x;
        int const block_n_idx = blockIdx.y;
        int const thread_idx = threadIdx.x;
        int const residue_m = params.problem_size.m() - block_m_idx * cute::size<0>(typename KT::TileShape{});
        int const residue_n = params.problem_size.n() - block_n_idx * cute::size<1>(typename KT::TileShape{});
        // gmem tensor partition ..
        auto [gA_mk, gB_nk, gOutput_mn, gScaleA_mk, gScaleB_nk] = gmem_tensor_init(params);

        // smem tensor ..
        cute::Tensor sA = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_a.data()), typename KT::SmemLayoutA{}); // (BLK_M, BLK_K, Stage)
        cute::Tensor sB = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_b.data()), typename KT::SmemLayoutB{}); // (BLK_N, BLK_K, Stage)
        cute::Tensor sO = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_o.data()), typename KT::SmemLayoutO{}); // (BLK_M, BLK_N)

        cute::Tensor sScaleA = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_scale_a.data()),
            typename KT::SmemLayoutScaleA{}); // (BLK_M, BLK_K, Stage)
        cute::Tensor sScaleB = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_scale_b.data()),
            typename KT::SmemLayoutScaleB{}); // (BLK_N, BLK_K, Stage)

        // (1) first step, get the B_res and B_gate

        // (1.1) get partition for gmem -> smem
        cute::Tensor gA = gA_mk(cute::_, cute::_, block_m_idx, cute::_); // (BLK_M, BLK_K, k)
        cute::Tensor gB = gB_nk(cute::_, cute::_, block_n_idx, cute::_); // (BLK_N, BLK_K, k)

        cute::Tensor gScaleA = gScaleA_mk(cute::_, cute::_, block_m_idx, cute::_);
        cute::Tensor gScaleB = gScaleB_nk(cute::_, cute::_, block_n_idx, cute::_);

        typename KT::GmemTiledCopyA gmem_tiled_copy_A;
        typename KT::GmemTiledCopyB gmem_tiled_copy_B;
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
        auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

        cute::Tensor tAgA = gmem_thr_copy_A.partition_S(gA); // (ACPY,ACPY_M,ACPY_K,k)
        cute::Tensor tAsA = gmem_thr_copy_A.partition_D(sA); // (ACPY,ACPY_M,ACPY_K,Stage)
        cute::Tensor tBgB = gmem_thr_copy_B.partition_S(gB); // (BCPY,BCPY_N,BCPY_K,k)
        cute::Tensor tBsB = gmem_thr_copy_B.partition_D(sB); // (BCPY,BCPY_N,BCPY_K,Stage)

        typename KT::GmemTiledCopyScaleA gmem_tiled_copy_ScaleA;
        typename KT::GmemTiledCopyScaleB gmem_tiled_copy_ScaleB;
        auto gmem_thr_copy_ScaleA = gmem_tiled_copy_ScaleA.get_slice(thread_idx);
        auto gmem_thr_copy_ScaleB = gmem_tiled_copy_ScaleB.get_slice(thread_idx);

        cute::Tensor tAgScaleA = gmem_thr_copy_ScaleA.partition_S(gScaleA); // (ACPY,ACPY_M,ACPY_K,k)
        cute::Tensor tAsScaleA = gmem_thr_copy_ScaleA.partition_D(sScaleA); // (ACPY,ACPY_M,ACPY_K,Stage)
        cute::Tensor tBgScaleB = gmem_thr_copy_ScaleB.partition_S(gScaleB); // (BCPY,BCPY_N,BCPY_K,k)
        cute::Tensor tBsScaleB = gmem_thr_copy_ScaleB.partition_D(sScaleB); // (BCPY,BCPY_N,BCPY_K,Stage)

        // Allocate predicate tensors for input and fc weight (actually we only need input predicate tensor)
        cute::Tensor tApA = cute::make_tensor<bool>(
            cute::make_shape(cute::size<1>(tAsA), cute::size<2>(tAsA)), cute::Stride<cute::_1, cute::_0>{});
        // Construct identity layout for sA
        cute::Tensor cA = make_identity_tensor(
            cute::make_shape(cute::size<0>(sA), cute::size<1>(sA))); // (BLK_M,BLK_K) -> (blk_m,blk_k)

        // Repeat the partitioning with identity layouts
        cute::Tensor tAcA = gmem_thr_copy_A.partition_S(cA); // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

        // Set predicates for m bounds
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < cute::size<0>(tApA); ++m)
        {
            tApA(m, 0) = cute::get<0>(tAcA(0, m, 0)) < residue_m; // blk_m coord < residue_m
        }

        cute::Tensor tBpB = cute::make_tensor<bool>(
            cute::make_shape(cute::size<1>(tBsB), cute::size<2>(tBsB)), cute::Stride<cute::_1, cute::_0>{});
        // Construct identity layout for sB
        cute::Tensor cB = make_identity_tensor(
            cute::make_shape(cute::size<0>(sB), cute::size<1>(sB))); // (BLK_N,BLK_K) -> (blk_n,blk_k)
        // Repeat the partitioning with identity layouts
        cute::Tensor tBcB = gmem_thr_copy_B.partition_S(cB); // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

        // Set predicates for n bounds
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < cute::size<0>(tBpB); ++n)
        {
            tBpB(n, 0) = cute::get<0>(tBcB(0, n, 0)) < residue_n; // blk_n coord < residue_n
        }

        cute::Tensor tApSFA = cute::make_tensor<bool>(
            cute::make_shape(cute::size<1>(tAsScaleA), cute::size<2>(tAsScaleA)), cute::Stride<cute::_1, cute::_0>{});
        cute::Tensor tAcSFA = gmem_thr_copy_ScaleA.partition_S(cA); // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < cute::size<0>(tApSFA); ++m)
        {
            tApSFA(m, 0) = cute::get<0>(tAcSFA(0, m, 0)) < residue_m; // blk_m coord < residue_m
        }
        // (1.2) prefetch gmem -> smem
        cute::clear(tAsA); // we don't need to clear tBsB..
        cute::clear(tBsB);

        cute::clear(tAsScaleA);
        cute::clear(tBsScaleB);

        auto k_tile_iter = cute::make_coord_iterator(cute::size<2>(gA)); // emm, iter start from 0
        int k_tile_count = cute::size<2>(gA);
        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < KT::Stages - 1; ++k_pipe)
        {
            if (k_tile_count <= 0)
            {
                cute::clear(tApA);
                cute::clear(tBpB);
                cute::clear(tApSFA);
            }
            cute::copy_if(gmem_tiled_copy_A, tApA, tAgA(cute::_, cute::_, cute::_, *k_tile_iter),
                tAsA(cute::_, cute::_, cute::_, k_pipe));
            cute::copy_if(gmem_tiled_copy_B, tBpB, tBgB(cute::_, cute::_, cute::_, *k_tile_iter),
                tBsB(cute::_, cute::_, cute::_, k_pipe));

            cute::copy_if(gmem_tiled_copy_ScaleA, tApSFA, tAgScaleA(cute::_, cute::_, cute::_, *k_tile_iter),
                tAsScaleA(cute::_, cute::_, cute::_, k_pipe));
            cute::copy(gmem_tiled_copy_ScaleB, tBgScaleB(cute::_, cute::_, cute::_, *k_tile_iter),
                tBsScaleB(cute::_, cute::_, cute::_, k_pipe));

            cute::cp_async_fence();
            k_tile_count--;
            if (k_tile_count > 0)
            {
                ++k_tile_iter;
            }
        }

        // (1.3) get partition for rf
        typename KT::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        cute::Tensor tCrA = thr_mma.partition_fragment_A(sA(cute::_, cute::_, 0)); // (MMA,MMA_M,MMA_K)
        cute::Tensor tCrB = thr_mma.partition_fragment_B(sB(cute::_, cute::_, 0)); // (MMA,MMA_N,MMA_K)

        cute::Tensor accum
            = cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename KT::TileShape{})); // (MMA,MMA_M,MMA_N)
        cute::Tensor temp_accum
            = cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename KT::TileShape{})); // (MMA,MMA_M,MMA_N)
        cute::clear(accum);
        // checkout the shape
        CUTE_STATIC_ASSERT_V(cute::size<1>(tCrA) == cute::size<1>(accum)); // MMA_M
        CUTE_STATIC_ASSERT_V(cute::size<1>(tCrB) == cute::size<2>(accum)); // MMA_N
        CUTE_STATIC_ASSERT_V(cute::size<2>(tCrA) == cute::size<2>(tCrB));  // MMA_K
        CUTE_STATIC_ASSERT_V(cute::size(gmem_tiled_copy_A) == cute::size(tiled_mma));
        CUTE_STATIC_ASSERT_V(cute::size(gmem_tiled_copy_B) == cute::size(tiled_mma));

        // (1.4)retiling the smem and rf for copy..
        auto smem_tiled_copy_A = cute::make_tiled_copy_A(typename KT::SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
        cute::Tensor tOsA = smem_thr_copy_A.partition_S(sA);                    // (CPY,CPY_M,CPY_K,Stage)
        cute::Tensor tCrA_write = smem_thr_copy_A.retile_D(tCrA);               // (CPY,CPY_M,CPY_K)
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOsA) == cute::size<1>(tCrA_write)); // CPY_M
        CUTE_STATIC_ASSERT_V(cute::size<2>(tOsA) == cute::size<2>(tCrA_write)); // CPY_K

        auto smem_tiled_copy_B = cute::make_tiled_copy_B(typename KT::SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
        cute::Tensor tOsB = smem_thr_copy_B.partition_S(sB);                    // (CPY,CPY_N,CPY_K,Stage)
        cute::Tensor tCrB_write = smem_thr_copy_B.retile_D(tCrB);               // (CPY,CPY_N,CPY_K)
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOsB) == cute::size<1>(tCrB_write)); // CPY_N
        CUTE_STATIC_ASSERT_V(cute::size<2>(tOsB) == cute::size<2>(tCrB_write)); // CPY_K

        typename KT::SmemCopyAtomScaleA smem_tiled_copy_ScaleA;
        typename KT::SmemCopyAtomScaleB smem_tiled_copy_ScaleB;
        auto smem_thr_copy_ScaleA = smem_tiled_copy_ScaleA.get_thread_slice(thread_idx);
        auto smem_thr_copy_ScaleB = smem_tiled_copy_ScaleB.get_thread_slice(thread_idx);

        cute::Tensor tOsScaleA = smem_thr_copy_ScaleA.partition_S(sScaleA);
        cute::Tensor tCrScaleA = cute::make_fragment_like(tOsScaleA(cute::_, cute::_, cute::_, 0));
        cute::Tensor tOsScaleB = smem_thr_copy_ScaleB.partition_S(sScaleB);
        cute::Tensor tCrScaleB = cute::make_fragment_like(tOsScaleB(cute::_, cute::_, cute::_, 0));

        // (1.5) mainloop
        // Current pipe index in smem to read from
        int smem_pipe_read = 0;
        // Current pipe index in smem to write to
        int smem_pipe_write = KT::Stages - 1;

        cute::Tensor tOsA_read = tOsA(cute::_, cute::_, cute::_, smem_pipe_read);
        cute::Tensor tOsB_read = tOsB(cute::_, cute::_, cute::_, smem_pipe_read);

        cute::Tensor tOsScaleA_read = tOsScaleA(cute::_, cute::_, cute::_, smem_pipe_read);
        cute::Tensor tOsScaleB_read = tOsScaleB(cute::_, cute::_, cute::_, smem_pipe_read);

        constexpr int K_BLOCK_MAX = cute::size<2>(tCrA);
        // prefetch register pipeline
        if constexpr (K_BLOCK_MAX > 1)
        {
            cute::cp_async_wait<KT::Stages - 2>();
            __syncthreads();

            // Prefetch the first k-tile smem -> reg
            cute::copy(smem_tiled_copy_A, tOsA_read(cute::_, cute::_, cute::Int<0>{}),
                tCrA_write(cute::_, cute::_, cute::Int<0>{}));
            cute::copy(smem_tiled_copy_B, tOsB_read(cute::_, cute::_, cute::Int<0>{}),
                tCrB_write(cute::_, cute::_, cute::Int<0>{}));
        }
        // k loop for mainloop
        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > 0; --k_tile_count)
        {
            cute::clear(temp_accum);
            cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{},
                [&](auto k_block)
                {
                    if (k_block == K_BLOCK_MAX - 1)
                    {
                        tOsA_read = tOsA(cute::_, cute::_, cute::_, smem_pipe_read);
                        tOsB_read = tOsB(cute::_, cute::_, cute::_, smem_pipe_read);

                        tOsScaleA_read = tOsScaleA(cute::_, cute::_, cute::_, smem_pipe_read);
                        tOsScaleB_read = tOsScaleB(cute::_, cute::_, cute::_, smem_pipe_read);

                        cute::cp_async_wait<KT::Stages - 2>();
                        __syncthreads();
                    }
                    // Load A, B smem -> reg for k_block+1
                    auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_MAX;
                    cute::copy(smem_tiled_copy_A, tOsA_read(cute::_, cute::_, k_block_next),
                        tCrA_write(cute::_, cute::_, k_block_next));
                    cute::copy(smem_tiled_copy_B, tOsB_read(cute::_, cute::_, k_block_next),
                        tCrB_write(cute::_, cute::_, k_block_next));
                    // Copy gmem -> smem before computing gemm on each k-pipe
                    if (k_block == 0)
                    {
                        cute::copy_if(gmem_tiled_copy_A, tApA, tAgA(cute::_, cute::_, cute::_, *k_tile_iter),
                            tAsA(cute::_, cute::_, cute::_, smem_pipe_write));
                        cute::copy_if(gmem_tiled_copy_B, tBpB, tBgB(cute::_, cute::_, cute::_, *k_tile_iter),
                            tBsB(cute::_, cute::_, cute::_, smem_pipe_write));

                        cute::copy_if(gmem_tiled_copy_ScaleA, tApSFA,
                            tAgScaleA(cute::_, cute::_, cute::_, *k_tile_iter),
                            tAsScaleA(cute::_, cute::_, cute::_, smem_pipe_write));
                        cute::copy(gmem_tiled_copy_ScaleB, tBgScaleB(cute::_, cute::_, cute::_, *k_tile_iter),
                            tBsScaleB(cute::_, cute::_, cute::_, smem_pipe_write));

                        cute::cp_async_fence();
                        if (k_tile_count - 1 > 0)
                        {
                            ++k_tile_iter;
                        }

                        cute::copy(smem_tiled_copy_ScaleA, tOsScaleA_read, tCrScaleA);
                        cute::copy(smem_tiled_copy_ScaleB, tOsScaleB_read, tCrScaleB);

                        // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
                        smem_pipe_write = smem_pipe_read;
                        ++smem_pipe_read;
                        smem_pipe_read = (smem_pipe_read == KT::Stages) ? 0 : smem_pipe_read;
                    }
                    // Thread-level register gemm for k_block
                    cute::gemm(tiled_mma, temp_accum, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block),
                        temp_accum);
                });

            promote(accum, temp_accum, tCrScaleA, tCrScaleB);
        }
        // load tail
        cute::for_each(cute::make_int_sequence<KT::Stages - 2>{},
            [&](auto WaitIndex)
            {
                k_tile_count--;
                using WaitIndex_t = decltype(WaitIndex);
                cute::clear(temp_accum);
                cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{},
                    [&](auto k_block)
                    {
                        if (k_block == K_BLOCK_MAX - 1)
                        {
                            tOsA_read = tOsA(cute::_, cute::_, cute::_, smem_pipe_read);
                            tOsB_read = tOsB(cute::_, cute::_, cute::_, smem_pipe_read);

                            tOsScaleA_read = tOsScaleA(cute::_, cute::_, cute::_, smem_pipe_read);
                            tOsScaleB_read = tOsScaleB(cute::_, cute::_, cute::_, smem_pipe_read);

                            cute::cp_async_wait<KT::Stages - 3 - WaitIndex_t::value>();
                            __syncthreads();
                        }
                        // Load A, B smem -> reg for k_block+1
                        auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_MAX;
                        cute::copy(smem_tiled_copy_A, tOsA_read(cute::_, cute::_, k_block_next),
                            tCrA_write(cute::_, cute::_, k_block_next));
                        cute::copy(smem_tiled_copy_B, tOsB_read(cute::_, cute::_, k_block_next),
                            tCrB_write(cute::_, cute::_, k_block_next));
                        if (k_block == 0)
                        {

                            cute::copy(smem_tiled_copy_ScaleA, tOsScaleA_read, tCrScaleA);
                            cute::copy(smem_tiled_copy_ScaleB, tOsScaleB_read, tCrScaleB);

                            // only update smem_pipe_read
                            ++smem_pipe_read;
                            smem_pipe_read = (smem_pipe_read == KT::Stages) ? 0 : smem_pipe_read;
                        }
                        // Thread-level register gemm for k_block
                        cute::gemm(tiled_mma, temp_accum, tCrA(cute::_, cute::_, k_block),
                            tCrB(cute::_, cute::_, k_block), temp_accum);
                    });

                promote(accum, temp_accum, tCrScaleA, tCrScaleB);
            });
        // mma tail
        cute::clear(temp_accum);
        cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{},
            [&](auto k_block)
            {
                // Load A, B smem -> reg for k_block+1
                auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_MAX;
                cute::copy(smem_tiled_copy_A, tOsA_read(cute::_, cute::_, k_block_next),
                    tCrA_write(cute::_, cute::_, k_block_next));
                cute::copy(smem_tiled_copy_B, tOsB_read(cute::_, cute::_, k_block_next),
                    tCrB_write(cute::_, cute::_, k_block_next));
                if (k_block == 0)
                {

                    cute::copy(smem_tiled_copy_ScaleA, tOsScaleA_read, tCrScaleA);
                    cute::copy(smem_tiled_copy_ScaleB, tOsScaleB_read, tCrScaleB);
                }
                // Thread-level register gemm for k_block
                cute::gemm(tiled_mma, temp_accum, tCrA(cute::_, cute::_, k_block), tCrB(cute::_, cute::_, k_block),
                    temp_accum);
            });

        promote(accum, temp_accum, tCrScaleA, tCrScaleB);

        // (4) push all the result to smem
        // (4.1) convert result from ElementAccum to ElementA
        cute::Tensor epi = util_convert_type<KT::ElementOutput>(accum);

        // (4.2) rf -> smem
        auto smem_tiled_copy_R2S = cute::make_tiled_copy_C(typename KT::SmemCopyAtomR2S{}, tiled_mma);
        auto smem_thr_copy_R2S = smem_tiled_copy_R2S.get_thread_slice(thread_idx);
        // cute::clear(sO);
        cute::Tensor tRS_rO = smem_thr_copy_R2S.retile_S(epi);
        cute::Tensor tRS_sO = smem_thr_copy_R2S.partition_D(sO);

        cute::copy(smem_tiled_copy_R2S, tRS_rO, tRS_sO);
        __syncthreads();

        // (4.3) smem -> rf

        typename KT::SmemTiledCopyS2R smem_tiled_copy_S2R;
        auto smem_thr_copy_S2R = smem_tiled_copy_S2R.get_thread_slice(thread_idx);
        cute::Tensor tSR_sO = smem_thr_copy_S2R.partition_S(sO);
        cute::Tensor tSR_rO = cute::make_tensor<KT::ElementOutput>(cute::shape(tSR_sO));

        cute::copy(smem_tiled_copy_S2R, tSR_sO, tSR_rO);
        __syncthreads();

        // (4.4) rf -> gmem
        cute::Tensor gO = gOutput_mn(cute::_, cute::_, block_m_idx, block_n_idx);
        cute::Tensor cO = cute::make_identity_tensor(
            cute::make_shape(cute::size<0>(typename KT::TileShape{}), cute::size<1>(typename KT::TileShape{})));
        auto tRG_rO = smem_thr_copy_S2R.retile_S(tSR_rO);
        auto tRG_gO = smem_thr_copy_S2R.partition_D(gO);
        auto tRG_cO = smem_thr_copy_S2R.partition_D(cO);
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < cute::size<1>(tRG_cO); ++m)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int n = 0; n < cute::size<2>(tRG_cO); ++n)
            {
                if (cute::get<0>(tRG_cO(0, m, n)) < residue_m && cute::get<1>(tRG_cO(0, m, n)) < residue_n)
                {
                    cute::copy(typename KT::GmemCopyAtomR2G{}, tRG_rO(cute::_, m, n), tRG_gO(cute::_, m, n));
                }
            }
        }
    }
};

} // namespace ada_blockwise_gemm
