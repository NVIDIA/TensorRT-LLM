/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

namespace cutlass::epilogue::fusion
{
template <typename TileShape_MNK, typename SystemBarrier_>
struct Sm100AllReduceArrive
{
    using SystemBarrier = SystemBarrier_;

    struct SharedStorage
    {
    };

    struct Arguments
    {
        typename SystemBarrier::Params barrier_params;
        int rank = 0;
        int world_size = 1;
    };

    struct Params
    {
        Layout<Shape<int, int>> tile_layout; // (TILE_M, TILE_N)
        typename SystemBarrier::Params barrier_params;
        int rank;
        int world_size;
    };

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(
        ProblemShape const& problem_shape, Arguments const& args, void* workspace)
    {
        // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
        auto problem_shape_mnkl = append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_mnkl;

        int m_tiles = ceil_div(M, size<0>(TileShape_MNK{}));
        int n_tiles = ceil_div(N, size<1>(TileShape_MNK{}));
        auto tile_layout = make_layout(make_shape(m_tiles, n_tiles));

        return {tile_layout, args.barrier_params, args.rank, args.world_size};
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const& problem_shape, Arguments const& args)
    {
        return true;
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args)
    {
        return 0;
    }

    template <class ProblemShape>
    static cutlass::Status initialize_workspace(ProblemShape const& problem_shape, Arguments const& args,
        void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr)
    {
        return cutlass::Status::kSuccess;
    }

    CUTLASS_HOST_DEVICE
    Sm100AllReduceArrive() {}

    CUTLASS_HOST_DEVICE
    Sm100AllReduceArrive(Params const& params, SharedStorage const& shared_storage)
        : params_ptr(&params)
    {
    }

    Params const* params_ptr; // pointer to Params from kernel(Params) (constant mem)

    CUTLASS_DEVICE bool is_producer_load_needed() const
    {
        return false;
    }

    CUTLASS_DEVICE bool is_C_load_needed() const
    {
        return false;
    }

    template <class... Args>
    CUTLASS_DEVICE auto get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args)
    {
        return EmptyProducerLoadCallbacks{};
    }

    template <class TileCoordMNL>
    struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks
    {
        CUTLASS_DEVICE
        ConsumerStoreCallbacks(Params const* params_ptr, TileCoordMNL tile_coord_mnl, int const thread_idx)
            : issued_tma_store(false)
            , tile_coord_mnl(tile_coord_mnl)
            , thread_idx(thread_idx)
            , params_ptr(params_ptr)
        {
        }

        bool issued_tma_store;
        TileCoordMNL tile_coord_mnl;
        int thread_idx;
        Params const* params_ptr;

        // Wait until at most Count committed TMA_STOREs are
        // complete and visible in gmem
        template <int Count>
        CUTLASS_DEVICE static void tma_store_wait()
        {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
            asm volatile("cp.async.bulk.wait_group %0;" : : "n"(Count) : "memory");
#endif
        }

        template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
        CUTLASS_DEVICE auto visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m,
            int epi_n, Array<ElementInput, FragmentSize> const& frg_input)
        {
            return frg_input;
        }

        CUTLASS_DEVICE void tma_store(int epi_m, int epi_n, int store_iteration, bool issue_tma_store)
        {
            // Record this so we know which thread issued the TMA store.
            // This is to ensure the same thead in end() waits on the TMA store.
            this->issued_tma_store = issue_tma_store;
        }

        CUTLASS_DEVICE bool tile_valid(int m, int n)
        {
            // Out of bounds check
            auto tiles_mn = params_ptr->tile_layout.shape();
            return m < size<0>(tiles_mn) && n < size<1>(tiles_mn);
        }

        // Tile end
        CUTLASS_DEVICE void end()
        {
            auto [m, n, l] = tile_coord_mnl;
            if (!tile_valid(m, n) || params_ptr->world_size == 1)
            {
                return; // nothing to do
            }

            // Steps for ensuring TMA store is visible in gmem
            // 1. Issue TMA op                (executing thread)
            // 2. cp.async.bulk.commit_group  (executing thread)
            // 3. cp.async.bulk.wait_group    (executing thread)
            if (issued_tma_store)
            {
                tma_store_wait<0>();

                int tile_idx = params_ptr->tile_layout(m, n);
                SystemBarrier::arrive_inc<cuda::thread_scope::thread_scope_device>(
                    params_ptr->barrier_params, thread_idx, tile_idx, params_ptr->rank, params_ptr->world_size);
            }
        }
    };

    template <bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
        class... Args>
    CUTLASS_DEVICE auto get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args)
    {
        auto [m, n, k, l] = args.tile_coord_mnkl;
        auto tile_coord_mnl = make_coord(m, n, l);
        return ConsumerStoreCallbacks<decltype(tile_coord_mnl)>(params_ptr, tile_coord_mnl, args.thread_idx);
    }
};

// D = AllReduce(activation(alpha * acc + beta * C))
template <class SystemBarrier_, class ElementOutput_, class ElementCompute_, class ElementSource_ = ElementOutput_,
    class ElementScalar_ = ElementCompute_, FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest>
struct Sm100LinCombAuxAllReduce
    : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>
{
};

template <class SystemBarrier, class ElementOutput, class ElementCompute, class ElementSource, class ElementScalar,
    FloatRoundStyle RoundStyle, class CtaTileShapeMNK, class EpilogueTile>
using Sm100LinearCombAuxAllReduce = Sm90EVT<Sm100AllReduceArrive<CtaTileShapeMNK, SystemBarrier>,  // Aux AR
    Sm90LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha
                                                                                                   // * acc)
    >;

template <
    // Dispatch policy arguments
    int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
    // Fusion Op arguments
    class SystemBarrier, class ElementD, class ElementCompute, class ElementC, class ElementScalar,
    FloatRoundStyle RoundStyle,
    // Epilogue arguments
    class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    Sm100LinCombAuxAllReduce<SystemBarrier, ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>,
    CtaTileShapeMNK, EpilogueTile> : Sm100LinearCombAuxAllReduce<SystemBarrier, ElementD, ElementCompute, ElementC,
                                         ElementScalar, RoundStyle, CtaTileShapeMNK, EpilogueTile>
{

    using Impl = Sm100LinearCombAuxAllReduce<SystemBarrier, ElementD, ElementCompute, ElementC, ElementScalar,
        RoundStyle, CtaTileShapeMNK, EpilogueTile>;
    using Operation
        = Sm100LinCombAuxAllReduce<SystemBarrier, ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

    struct Arguments
    {
        ElementScalar alpha = ElementScalar(1);
        ElementScalar beta = ElementScalar(0);
        ElementScalar const* alpha_ptr = nullptr;
        ElementScalar const* beta_ptr = nullptr;
        typename SystemBarrier::Params barrier_params{};
        int rank = 0;
        int num_ranks = 1;
        using StrideAlpha = Stride<_0, _0, int64_t>;
        using StrideBeta = Stride<_0, _0, int64_t>;
        StrideAlpha dAlpha = {_0{}, _0{}, 0};
        StrideBeta dBeta = {_0{}, _0{}, 0};

        operator typename Impl::Arguments() const
        {
            return {{
                        // ternary op : beta * C + (alpha * acc)
                        {{beta}, {beta_ptr}, {dBeta}}, // leaf args : beta
                        {},                            // leaf args : C
                        {
                            // binary op : alpha * acc
                            {{alpha}, {alpha_ptr}, {dAlpha}}, // leaf args : alpha
                            {},                               // leaf args : acc
                            {}                                // binary args : multiplies
                        },                                    // end binary op
                        {}                                    // ternary args : multiply_add
                    },                                        // end ternary op
                {barrier_params, rank, num_ranks}};
        }
    };

    // Ctor inheritance
    using Impl::Impl;
};

} // namespace cutlass::epilogue::fusion
