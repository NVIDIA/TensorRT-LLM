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
#include "cute/atom/mma_atom.hpp"
#include <cuda_runtime.h>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/config.hpp>
#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>
#include <cutlass/cutlass.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/device_kernel.h" // cutlass::device_kernel
#include "cutlass/numeric_conversion.h"

using namespace cute;
using namespace cutlass;

namespace sm120_blockscaled_gemm
{

template <int TileM_ = 32, int TileN_ = 128>
struct SM120BlockScaledBuilder
{

    using ElementA = cute::float_e4m3_t;
    using ElementB = cute::float_e4m3_t;
    using ElementSFLoad = int32_t;                // scale load type
    using ElementSFCompute = cute::float_ue8m0_t; // scale mma type
    using ElementAccum = float;
    using ElementD = cute::bfloat16_t;

    static constexpr int AB_Stages = 4;
    static constexpr int SF_Stages = 1;
    static constexpr int kTileM = TileM_;
    static constexpr int kTileN = TileN_;
    static constexpr int kSFVecSize = 128; // fixed for 1x128 quantization
    static constexpr int kTileSF = 1;      // 1 sf block contains 4 e8m0 per 512 k elements
    static constexpr int kTileK = 128;
    using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
    using ScaleTileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileSF>>;
    using ClusterShape = Shape<_1, _1, _1>;
    using ProblemShape = Shape<int, int, int, int>;

    // ====== mma ======
    using PermMmaTileM = Int<32>;
    using PermMmaTileN = Int<32>;
    using PermMmaTileK = Int<32>;
    using MMA_Atom = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t, float, float_ue8m0_t,
        32>>; // sm120 16x8x32 fp8 mma
    using TiledMma = TiledMMA<MMA_Atom, Layout<Shape<_2, _2, _1>>, Tile<PermMmaTileM, PermMmaTileN, PermMmaTileK>>;

    CUTE_HOST_DEVICE
    static auto ceil_div(int const& x, int const& y)
    {
        return (x + y - 1) / y;
    }

    CUTE_HOST_DEVICE
    static auto align(int const& x, int const& alignment)
    {
        return ceil_div(x, alignment) * alignment;
    }

    CUTE_HOST_DEVICE
    static auto get_tma_aligned_size(int const& x)
    {
        constexpr int kNumTMAAlignmentBytes = 16;
        CUTE_STATIC_ASSERT(kNumTMAAlignmentBytes % sizeof(ElementSFLoad) == 0, "element_size must be a multiple of 16");
        auto alignment = kNumTMAAlignmentBytes / sizeof(ElementSFLoad);
        return align(x, alignment);
    }

    CUTE_HOST_DEVICE
    static auto deduce_sfa_layout(ProblemShape const& problem_shape)
    {
        auto [M, N, K, L] = problem_shape;
        auto scale_m = get_tma_aligned_size(M);
        auto scale_k = ceil_div(K, 128 * 4);
        return make_ordered_layout(make_shape(scale_m, scale_k, L), Step<_1, _2, _3>{});
    }

    CUTE_HOST_DEVICE
    static auto deduce_sfb_layout(ProblemShape const& problem_shape)
    {
        auto [M, N, K, L] = problem_shape;
        auto scale_n = get_tma_aligned_size(N);
        auto scale_k = ceil_div(K, 128 * 4);
        return make_ordered_layout(make_shape(scale_n, scale_k, L), Step<_1, _2, _3>{});
    }

    template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE static constexpr auto thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
    {
        CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

        // Reorder the tensor for the TiledAtom
        auto permutation_mnk = TiledPerm{};
        auto t_tile = make_tile(get<0>(permutation_mnk), _1{});
        auto tiled_sfa = logical_divide(sfatensor, t_tile);

        using AtomShape_MNK = typename Atom::Shape_MNK;
        auto atom_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(_1{}));
        auto tiled_atom_sfa = zipped_divide(tiled_sfa, atom_tile); // ((AtomM,AtomK),(RestM,RestK))
        // Transform the Atom mode from (M,K) to (Thr,Val)
        using AtomLayoutSFA_TV = Layout<Shape<Shape<_2, _2, _8>, _1>,     // Effectively 16 threads due to the 2:0 mode
            Stride<Stride<_8, _0, _1>, _16>>;
        auto tv_atom_sfa = tiled_atom_sfa.compose(AtomLayoutSFA_TV{}, _); // ((ThrV,FrgV),(RestM,RestK))

        // Tile the tensor for the Thread
        auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
        auto thr_tile
            = make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
        auto thr_tensor = zipped_divide(tv_atom_sfa, thr_tile); // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))
        return thr_tensor;
    }

    template <class SFATensor, class ThrMma>
    CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma)
    {
        auto thr_tensor
            = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(), thread_mma));
        auto thr_vmnk = thread_mma.thr_vmnk_;
        auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
        auto partition_SFA = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
        auto frg_SFA = make_fragment_like<ElementSFLoad>(partition_SFA);
        return frg_SFA;
    }

    template <class TiledMma>
    CUTE_HOST_DEVICE static constexpr auto get_layoutSFA_TV(TiledMma& mma)
    {
        auto tile_shape_mnk = tile_shape(mma);
        auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), _1{}));
        auto thr_tensor = thrfrg_SFA(ref_A, mma);
        auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
        auto atile = make_tile(_,
            make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                          make_stride(Int<1>{}, Int<0>{})),
                _));
        auto tv_sfa = thr_tensor.compose(atile, _);
        auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
        auto tv_layout = tv_sfa.compose(thridx_2_thrid, _);
        return tv_layout;
    }

    template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE static constexpr auto thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
    {
        CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

        // Reorder the tensor for the TiledAtom
        auto permutation_mnk = TiledPerm{};
        auto t_tile = make_tile(get<1>(permutation_mnk), _1{});
        auto tiled_sfb = logical_divide(sfbtensor, t_tile);

        using AtomShape_MNK = typename Atom::Shape_MNK;
        auto atom_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(_1{}));
        auto tiled_atom_sfb = zipped_divide(tiled_sfb, atom_tile); // ((AtomM,AtomK),(RestM,RestK))
        // Transform the Atom mode from (M,K) to (Thr,Val)
        using AtomLayoutSFB_TV = Layout<Shape<Shape<_4, _8>, _1>,         // Effectively 8 threads due to the 4:0 mode
            Stride<Stride<_0, _1>, _8>>;
        auto tv_atom_sfb = tiled_atom_sfb.compose(AtomLayoutSFB_TV{}, _); // ((ThrV,FrgV),(RestM,RestK))

        // Tile the tensor for the Thread
        auto thr_layout_vmnk = mma.get_thr_layout_vmnk(); // (_32,_4,_1,_1):(_1,_32,_0,_0)
        auto thr_tile
            = make_tile(_, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
        auto thr_tensor = zipped_divide(tv_atom_sfb, thr_tile); // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))
        return thr_tensor;
    }

    template <class SFBTensor, class ThrMma>
    CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma)
    {
        // using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
        auto thr_tensor
            = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(), thread_mma));
        auto thr_vmnk = thread_mma.thr_vmnk_;
        auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
        auto partition_SFB = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
        auto frg_SFB = make_fragment_like<ElementSFLoad>(partition_SFB);
        return frg_SFB;
    }

    template <class TiledMma>
    CUTE_HOST_DEVICE static constexpr auto get_layoutSFB_TV(TiledMma& mma)
    {
        auto tile_shape_mnk = tile_shape(mma);
        auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), _1{}));
        auto thr_tensor = thrfrg_SFB(ref_B, mma);
        auto thr_layout_vmnk = mma.get_thr_layout_vmnk(); // (_32,_4,_1,_1):(_1,_32,_0,_0)
        auto btile = make_tile(_,
            make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                          make_stride(Int<0>{}, Int<1>{})),
                _));                                          // (_,((_4,_1):(_0,_1),_))
        auto tv_sfb = thr_tensor.compose(btile, _);           // mma_sfb_tv_layout
        auto thridx_2_thrid = right_inverse(thr_layout_vmnk); // (_128:_1)
        auto tv_layout = tv_sfb.compose(thridx_2_thrid, _);
        return tv_layout;
    }

    template <class Tensor>
    CUTE_HOST_DEVICE static constexpr auto transform_fragment_for_qmma(Tensor&& tensor)
    {
        CUTE_STATIC_ASSERT_V(rank(tensor) == Int<3>{});
        auto old_ptr = tensor.data();
        auto new_ptr = recast_ptr<ElementSFCompute>(old_ptr);
        auto old_layout = tensor.layout();
        auto num_mn = size<1>(shape(old_layout));
        CUTE_STATIC_ASSERT_V(size<2>(shape(old_layout)) == Int<1>{});
        auto new_layout = make_layout(make_shape(_32{}, num_mn, _4{}, _4{}), make_stride(_0{}, _4{}, _0{}, _1{}));
        auto new_tensor = make_tensor(new_ptr, new_layout);
        return new_tensor;
    }

    // ====== load smem -> rf ======
    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
    using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;

    // ====== smem layout ======
    using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<ElementA>; // ((_8,_16),(_128,_1)):((_128,_1024),(_1,_0))
    using SmemLayoutAtomB = GMMA::Layout_K_SW128_Atom<ElementB>; // ((_8,_16),(_128,_1)):((_128,_1024),(_1,_0))

    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<AB_Stages>{}), Step<_1, _2, _3>{}));

    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<AB_Stages>{}), Step<_1, _2, _3>{}));

    // ====== TMA config ======
    using StrideA = Stride<int32_t, Int<1>, int32_t>;
    using StrideB = Stride<int32_t, Int<1>, int32_t>;

    using TMA_A = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(recast_ptr<ElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_, _, Int<0>{}), make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{}));

    using TMA_B = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(recast_ptr<ElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_, _, Int<0>{}), make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})), _1{}));

    // ====== scale ======
    using SmemCopyAtomSF = Copy_Atom<AutoVectorizingCopy, ElementSFLoad>; // auto-vectorized LDS

    using SmemLayoutAtomSFA = decltype(make_ordered_layout(select<0, 2>(ScaleTileShape{}), Step<_1, _2>{}));

    using SmemLayoutAtomSFB = decltype(make_ordered_layout(select<1, 2>(ScaleTileShape{}), Step<_1, _2>{}));

    using SmemLayoutSFA = decltype(tile_to_shape(SmemLayoutAtomSFA{},
        make_shape(shape<0>(ScaleTileShape{}), shape<2>(ScaleTileShape{}), Int<SF_Stages>{}), Step<_1, _2, _3>{}));

    using SmemLayoutSFB = decltype(tile_to_shape(SmemLayoutAtomSFB{},
        make_shape(shape<1>(ScaleTileShape{}), shape<2>(ScaleTileShape{}), Int<SF_Stages>{}), Step<_1, _2, _3>{}));

    using StrideSFA = Stride<Int<1>, int32_t, int32_t>; // column major
    using StrideSFB = Stride<Int<1>, int32_t, int32_t>; // column major

    using TMA_SFA = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(recast_ptr<ElementSFLoad>(nullptr), repeat_like(StrideSFA{}, int32_t(0)), StrideSFA{}),
        SmemLayoutSFA{}(_, _, cute::Int<0>{}), make_shape(shape<0>(ScaleTileShape{}), shape<2>(ScaleTileShape{})),
        _1{}));

    using TMA_SFB = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(recast_ptr<ElementSFLoad>(nullptr), repeat_like(StrideSFB{}, int32_t(0)), StrideSFB{}),
        SmemLayoutSFB{}(_, _, cute::Int<0>{}), make_shape(shape<1>(ScaleTileShape{}), shape<2>(ScaleTileShape{})),
        _1{}));

    static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>));
    static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(
        cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>));
    static constexpr uint32_t TmaABTransactionBytes = TmaTransactionBytesA + TmaTransactionBytesB;
    static constexpr uint32_t TmaTransactionBytesSFA = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSFLoad>));
    static constexpr uint32_t TmaTransactionBytesSFB = static_cast<uint32_t>(
        cutlass::bits_to_bytes(cosize(take<0, 2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSFLoad>));
    static constexpr uint32_t TmaSFTransactionBytes = TmaTransactionBytesSFA + TmaTransactionBytesSFB;

    // ====== TMA store ======
    using StrideD = Stride<int32_t, Int<1>, int32_t>;
    using EpilogueTile_MN = Shape<Int<kTileM>, Int<kTileN>>;

    using CopyAtomC = Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>;

    using SmemLayoutAtomD = GMMA::Layout_K_SW128_Atom<ElementD>; // ((_8,_16),(_128,_1)):((_128,_1024),(_1,_0))

    static constexpr int StagesD = 1;
    using SmemLayoutD = decltype(tile_to_shape(SmemLayoutAtomD{},
        make_shape(size<0>(EpilogueTile_MN{}), size<1>(EpilogueTile_MN{}), Int<StagesD>{}), Step<_1, _2, _3>{}));

    using CopyOpR2S = SM90_U32x2_STSM_N;
    using CopyOpS2G = SM90_TMA_STORE;
    using TMA_D = decltype(make_tma_copy_C_sm90(CopyOpS2G{},
        make_tensor(make_gmem_ptr(static_cast<ElementD*>(nullptr)), repeat_like(StrideD{}, int32_t(0)), StrideD{}),
        take<0, 2>(SmemLayoutD{}), EpilogueTile_MN{}));

    struct SharedStorageLoad : cute::aligned_struct<128, _0>
    {
        alignas(1024) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
        alignas(1024) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
        cute::ArrayEngine<ElementSFLoad, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
        cute::ArrayEngine<ElementSFLoad, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;

    struct SharedStorageStore : cute::aligned_struct<128, _0>
    {
        alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
    };

    union TensorStorage
    {
        SharedStorageLoad load;
        SharedStorageStore store;
    };

    using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;
    using ProducerBarrierType = FullBarrier::ValueType;
    using ConsumerBarrierType = EmptyBarrier::ValueType;

    struct BarrierStorage
    {
        FullBarrier ab_full_mbar[AB_Stages];
        EmptyBarrier ab_empty_mbar[AB_Stages];
        FullBarrier sf_full_mbar[SF_Stages];
        EmptyBarrier sf_empty_mbar[SF_Stages];
    };

    static dim3 get_grid_shape(ProblemShape problem_shape)
    {
        auto [M, N, K, L] = problem_shape;
        int grid_m = (M + kTileM - 1) / kTileM;
        int grid_n = (N + kTileN - 1) / kTileN;
        int grid_l = L;
        return dim3(grid_m, grid_n, grid_l);
    }
};

} // namespace sm120_blockscaled_gemm
