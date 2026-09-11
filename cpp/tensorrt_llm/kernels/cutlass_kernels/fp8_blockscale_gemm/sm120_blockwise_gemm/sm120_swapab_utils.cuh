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

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace sm120_swapab
{

using namespace cute;

/**
 * SwapAB Builder: QMMA config for GEMV with swapped A/B.
 *
 * User problem:  D[M×N] = A[M×K] × B[N×K]^T,  M small (≤16), N large
 * Kernel view:   accum[TileM × TileN] = B[N×K] × A[M×K]^T
 *                TileM covers N (large), TileN covers M (small)
 * Epilogue:      STG scatter store  accum[i][j] → D[j][i]
 *
 * Dispatch must swap: A_kernel=B_user, B_kernel=A_user,
 *                     SFA_kernel=SFB_user, SFB_kernel=SFA_user,
 *                     problem_shape_kernel = (N, M, K, L)
 */
template <int TileM_ = 128, int TileN_ = 32, int TileK_ = 128, int Stages_ = 1>
struct SM120BlockScaledSwapABBuilder
{
    using ElementA = cute::float_e4m3_t;
    using ElementB = cute::float_e4m3_t;
    using ElementSFLoad = int32_t;
    using ElementSFCompute = cute::float_ue8m0_t;
    using ElementAccum = float;
    using ElementD = cute::bfloat16_t;

    static constexpr int AB_Stages = Stages_;
    static constexpr int SF_Stages = 1;
    static constexpr int kTileM = TileM_;
    static constexpr int kTileN = TileN_;
    static constexpr int kTileK = TileK_;
    static constexpr int kSFVecSize = 128;
    static constexpr int kTileSF = 1;
    static_assert(kTileK == 64 || kTileK == 128, "kTileK must be 64 or 128");
    static constexpr int kNumTileKPerSF = 512 / kTileK;
    static constexpr int kNumStagePerSF = kNumTileKPerSF / AB_Stages;
    static_assert(
        kNumStagePerSF > 0 && kNumTileKPerSF % AB_Stages == 0, "kNumTileKPerSF must be divisible by AB_Stages");
    static constexpr int kKTilesPerScale = kSFVecSize / kTileK;

    using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
    using ScaleTileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileSF>>;
    using ClusterShape = Shape<_1, _1, _1>;
    using ProblemShape = Shape<int, int, int, int>;

    // ---- MMA: 64×8 tile, 128 math threads (4 warps, 1 WG) ----
    using MMA_Atom_t
        = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 32>>;
    using TiledMma = TiledMMA<MMA_Atom_t, Layout<Shape<_4, _1, _1>, Stride<_1, _1, _0>>, Tile<_64, _8, Underscore>>;
    static constexpr int kNumMathThreads = size(TiledMma::ThrLayoutVMNK{});
    static constexpr int kNumMathWarps = kNumMathThreads / 32;
    static constexpr int kNumMathWG = kNumMathThreads / 128;
    static constexpr int kNumTMAThreads = 64; // 2 warps: AB load + SF load

    // ---- Helpers ----
    CUTE_HOST_DEVICE static auto ceil_div(int x, int y)
    {
        return (x + y - 1) / y;
    }

    CUTE_HOST_DEVICE static auto align(int x, int a)
    {
        return ceil_div(x, a) * a;
    }

    CUTE_HOST_DEVICE static auto get_tma_aligned_size(int x)
    {
        constexpr int kAlign = 16 / sizeof(ElementSFLoad);
        return align(x, kAlign);
    }

    // ---- SF layout deduction (operates on KERNEL problem_shape) ----
    CUTE_HOST_DEVICE static auto deduce_sfa_layout(ProblemShape const& ps)
    {
        auto [M, N, K, L] = ps;
        int64_t sm = static_cast<int64_t>(get_tma_aligned_size(M));
        int64_t sk = static_cast<int64_t>(ceil_div(K, 128 * 4));
        return make_layout(make_shape(sm, sk, L), make_stride(Int<1>{}, sm, sm * sk));
    }

    CUTE_HOST_DEVICE static auto deduce_sfb_layout(ProblemShape const& ps)
    {
        auto [M, N, K, L] = ps;
        int64_t sn = static_cast<int64_t>(get_tma_aligned_size(N));
        int64_t sk = static_cast<int64_t>(ceil_div(K, 128 * 4));
        return make_layout(make_shape(sn, sk, L), make_stride(Int<1>{}, sn, sn * sk));
    }

    // ---- SF partition helpers for QMMA scale fragment ----
    template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE static constexpr auto thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
    {
        CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});
        auto permutation_mnk = TiledPerm{};
        auto t_tile = make_tile(get<0>(permutation_mnk), _1{});
        auto tiled_sfa = logical_divide(sfatensor, t_tile);
        using AtomShape_MNK = typename Atom::Shape_MNK;
        auto atom_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(_1{}));
        auto tiled_atom_sfa = zipped_divide(tiled_sfa, atom_tile);
        using AtomLayoutSFA_TV = Layout<Shape<Shape<_2, _2, _8>, _1>, Stride<Stride<_8, _0, _1>, _16>>;
        auto tv_atom_sfa = tiled_atom_sfa.compose(AtomLayoutSFA_TV{}, _);
        auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
        auto thr_tile
            = make_tile(_, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
        return zipped_divide(tv_atom_sfa, thr_tile);
    }

    template <class SFATensor, class ThrMma>
    CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma)
    {
        auto thr_tensor
            = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(), thread_mma));
        auto thr_vmnk = thread_mma.thr_vmnk_;
        auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
        auto partition_SFA = thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
        return make_fragment_like<ElementSFLoad>(partition_SFA);
    }

    template <class TiledMma_>
    CUTE_HOST_DEVICE static constexpr auto get_layoutSFA_TV(TiledMma_& mma)
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
        return tv_sfa.compose(thridx_2_thrid, _);
    }

    template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
    CUTE_HOST_DEVICE static constexpr auto thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
    {
        CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});
        auto permutation_mnk = TiledPerm{};
        auto t_tile = make_tile(get<1>(permutation_mnk), _1{});
        auto tiled_sfb = logical_divide(sfbtensor, t_tile);
        using AtomShape_MNK = typename Atom::Shape_MNK;
        auto atom_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(_1{}));
        auto tiled_atom_sfb = zipped_divide(tiled_sfb, atom_tile);
        using AtomLayoutSFB_TV = Layout<Shape<Shape<_4, _8>, _1>, Stride<Stride<_0, _1>, _8>>;
        auto tv_atom_sfb = tiled_atom_sfb.compose(AtomLayoutSFB_TV{}, _);
        auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
        auto thr_tile
            = make_tile(_, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
        return zipped_divide(tv_atom_sfb, thr_tile);
    }

    template <class SFBTensor, class ThrMma>
    CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma)
    {
        auto thr_tensor
            = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(), thread_mma));
        auto thr_vmnk = thread_mma.thr_vmnk_;
        auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
        auto partition_SFB = thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
        return make_fragment_like<ElementSFLoad>(partition_SFB);
    }

    template <class TiledMma_>
    CUTE_HOST_DEVICE static constexpr auto get_layoutSFB_TV(TiledMma_& mma)
    {
        auto tile_shape_mnk = tile_shape(mma);
        auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), _1{}));
        auto thr_tensor = thrfrg_SFB(ref_B, mma);
        auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
        auto btile = make_tile(_,
            make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                          make_stride(Int<0>{}, Int<1>{})),
                _));
        auto tv_sfb = thr_tensor.compose(btile, _);
        auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
        return tv_sfb.compose(thridx_2_thrid, _);
    }

    template <class Tensor>
    CUTE_HOST_DEVICE static constexpr auto transform_fragment_for_qmma(Tensor&& tensor)
    {
        CUTE_STATIC_ASSERT_V(rank(tensor) == Int<3>{});
        auto new_ptr = recast_ptr<ElementSFCompute>(tensor.data());
        auto num_mn = size<1>(shape(tensor.layout()));
        CUTE_STATIC_ASSERT_V(size<2>(shape(tensor.layout())) == Int<1>{});
        auto new_layout = make_layout(make_shape(_32{}, num_mn, _4{}, _4{}), make_stride(_0{}, _4{}, _0{}, _1{}));
        return make_tensor(new_ptr, new_layout);
    }

    // ---- SMEM layouts (with stages) ----
    using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<ElementA>;
    using SmemLayoutAtomB = GMMA::Layout_K_SW128_Atom<ElementB>;
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<AB_Stages>{}), Step<_1, _2, _3>{}));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<AB_Stages>{}), Step<_1, _2, _3>{}));

    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
    // B S2R: U32x2 for small TileN (TileN=8 fragment=256B, U32x4=512B too wide)
    using SmemCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, ElementB>;
    using SmemCopyAtomSF = Copy_Atom<AutoVectorizingCopy, ElementSFLoad>;

    // ---- TMA load ----
    using StrideA = Stride<int64_t, Int<1>, int64_t>;
    using StrideB = Stride<int64_t, Int<1>, int64_t>;
    using StrideD = Stride<int64_t, Int<1>, int64_t>;
    using TMA_A = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<ElementA*>(nullptr)), repeat_like(StrideA{}, int64_t(0)), StrideA{}),
        SmemLayoutA{}(_, _, Int<0>{}), make_shape(Int<kTileM>{}, Int<kTileK>{}), _1{}));
    using TMA_B = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<ElementB*>(nullptr)), repeat_like(StrideB{}, int64_t(0)), StrideB{}),
        SmemLayoutB{}(_, _, Int<0>{}), make_shape(Int<kTileN>{}, Int<kTileK>{}), _1{}));

    // ---- Scale SMEM + TMA ----
    using SmemLayoutAtomSFA = decltype(make_ordered_layout(select<0, 2>(ScaleTileShape{}), Step<_1, _2>{}));
    using SmemLayoutAtomSFB = decltype(make_ordered_layout(select<1, 2>(ScaleTileShape{}), Step<_1, _2>{}));
    using SmemLayoutSFA = decltype(tile_to_shape(SmemLayoutAtomSFA{},
        make_shape(shape<0>(ScaleTileShape{}), shape<2>(ScaleTileShape{}), Int<SF_Stages>{}), Step<_1, _2, _3>{}));
    using SmemLayoutSFB = decltype(tile_to_shape(SmemLayoutAtomSFB{},
        make_shape(shape<1>(ScaleTileShape{}), shape<2>(ScaleTileShape{}), Int<SF_Stages>{}), Step<_1, _2, _3>{}));
    using StrideSFA = Stride<Int<1>, int64_t, int64_t>;
    using StrideSFB = Stride<Int<1>, int64_t, int64_t>;
    using TMA_SFA = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(recast_ptr<ElementSFLoad>(nullptr), repeat_like(StrideSFA{}, int64_t(0)), StrideSFA{}),
        SmemLayoutSFA{}(_, _, Int<0>{}), make_shape(shape<0>(ScaleTileShape{}), shape<2>(ScaleTileShape{})), _1{}));
    using TMA_SFB = decltype(make_tma_copy(SM90_TMA_LOAD{},
        make_tensor(recast_ptr<ElementSFLoad>(nullptr), repeat_like(StrideSFB{}, int64_t(0)), StrideSFB{}),
        SmemLayoutSFB{}(_, _, Int<0>{}), make_shape(shape<1>(ScaleTileShape{}), shape<2>(ScaleTileShape{})), _1{}));

    // ---- Transaction bytes ----
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

    // ---- Barriers ----
    using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
    using EmptyBarrier = cutlass::arch::ClusterBarrier;
    using ProducerBarrierType = typename FullBarrier::ValueType;
    using ConsumerBarrierType = typename EmptyBarrier::ValueType;

    // ---- Shared Storage (load only, STG epilogue needs no store SMEM) ----
    struct SharedStorageLoad : cute::aligned_struct<128, _0>
    {
        alignas(1024) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
        alignas(1024) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
        cute::ArrayEngine<ElementSFLoad, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
        cute::ArrayEngine<ElementSFLoad, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    };

    struct TensorStorage
    {
        SharedStorageLoad load;
    };

    struct BarrierStorage
    {
        FullBarrier ab_full_mbar[AB_Stages];
        EmptyBarrier ab_empty_mbar[AB_Stages];
        FullBarrier sf_full_mbar[SF_Stages];
        EmptyBarrier sf_empty_mbar[SF_Stages];
    };
};

} // namespace sm120_swapab
