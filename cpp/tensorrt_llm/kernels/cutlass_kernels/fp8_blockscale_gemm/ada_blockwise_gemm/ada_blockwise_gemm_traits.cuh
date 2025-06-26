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

// clang-format off
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>
#include <cutlass/arch/mma.h>
#include "ada_blockwise_mma_utils.cuh"
#include "ada_blockwise_copy_utils.cuh"

// clang-format on
using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm;

namespace ada_blockwise_gemm
{

template <typename ElementType, typename OutElementType, typename AccumElementType, typename BlockScaleElementType,
    int Stages_, int TileM_, int TileN_, int TileK_>
struct AdaBlockwiseGemmTraits
{
    using ElementA = ElementType;
    using LayoutA = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    using ElementB = ElementType;
    using LayoutB = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    using ElementAccum = AccumElementType;
    using ElementBlockScale = BlockScaleElementType;
    using ElementOutput = OutElementType;

    using index_t = uint32_t;
    static_assert(TileM_ % 16 == 0);
    static_assert(TileN_ % 32 == 0);
    static_assert(TileK_ % 32 == 0);
    static constexpr int Stages = Stages_;
    static constexpr int kTileM = TileM_;
    static constexpr int kTileN = TileN_;
    static constexpr int kTileK = (kTileM > 16) ? (TileK_) : (TileK_ >= 64 ? TileK_ : 64);

    // tile shape
    using TileShape = cute::Shape<cute::Int<kTileM>, cute::Int<kTileN>, cute::Int<kTileK>>;
    static constexpr int kWarpsCount = 4;
    static constexpr int kThreadCount = kWarpsCount * 32;

    static constexpr int ScaleGranularityM = 1;
    static constexpr int ScaleGranularityN = 128;
    static constexpr int ScaleGranularityK = 128;

    static constexpr int ScaleMsPerTile = (kTileM + ScaleGranularityM - 1) / ScaleGranularityM;
    static constexpr int ScaleNsPerTile = (kTileN + ScaleGranularityN - 1) / ScaleGranularityN;
    static constexpr int ScaleKsPerTile = (kTileK + ScaleGranularityK - 1) / ScaleGranularityK;

    static_assert(ScaleKsPerTile >= 1, "ScaleKsPerTile must be greater than or equal to 1");

    using ScaleGranularity
        = cute::Shape<cute::Int<ScaleGranularityM>, cute::Int<ScaleGranularityN>, cute::Int<ScaleGranularityK>>;
    using ScalePerTileShape
        = cute::Shape<cute::Int<ScaleMsPerTile>, cute::Int<ScaleNsPerTile>, cute::Int<ScaleKsPerTile>>;

    // MMA atom arch and layout
    using TiledMma = DefaultGemm_TensorOp_MMA<cute::float_e4m3_t, cutlass::arch::Sm89>::TiledMma;

    static constexpr int kBlockKSmem = 128;
    // A memory copy operand
    using DefaultOperandA
        = DefaultGemm_TensorOpSm80_OperandA<ElementA, cutlass::layout::RowMajor, AlignmentA, kBlockKSmem>;
    using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
    using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
    using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

    // ScaleA memory copy operand
    using SmemLayoutAtomScaleA = decltype(cute::make_layout(typename CopyTraitsScaleA::SmemTileShapeScale{}));
    using SmemCopyAtomScaleA = typename CopyTraitsScaleA::SmemTiledCopyScale;
    using GmemTiledCopyScaleA = typename CopyTraitsScaleA::GmemTiledCopyScale;

    // B memory copy operand
    using DefaultOperandB
        = DefaultGemm_TensorOpSm80_OperandB<ElementB, cutlass::layout::ColumnMajor, AlignmentB, kBlockKSmem>;
    using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
    using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
    using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

    // ScaleB memory copy operand
    using SmemLayoutAtomScaleB = decltype(cute::make_layout(typename CopyTraitsScaleB::SmemTileShapeScale{}));
    using SmemCopyAtomScaleB = typename CopyTraitsScaleB::SmemTiledCopyScale;
    using GmemTiledCopyScaleB = typename CopyTraitsScaleB::GmemTiledCopyScale;

    // Output memory copy operand
    using SmemLayoutAtomO = decltype(cute::composition(cute::Swizzle<3, 3, 3>{},
        cute::Layout<cute::Shape<cute::_8, cute::Shape<cute::_8, cute::_8>>,
            cute::Stride<cute::_8, cute::Stride<cute::_1, cute::_64>>>{}));

    using SmemCopyAtomR2S = cute::Copy_Atom<cute::AutoVectorizingCopy, ElementOutput>;
    using SmemCopyAtomS2R = cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, ElementOutput>;
    using GmemCopyAtomR2G = cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, ElementOutput>;
    using SmemThrLayoutS2R
        = cute::Layout<cute::Shape<cute::Int<8>, cute::Int<16>>, cute::Stride<cute::Int<16>, cute::_1>>;
    using SmemValLayoutS2R = cute::Layout<cute::Shape<cute::Int<1>, cute::Int<8>>>;
    using SmemTiledCopyS2R = decltype(cute::make_tiled_copy(SmemCopyAtomS2R{}, SmemThrLayoutS2R{}, SmemValLayoutS2R{}));

    static_assert(cute::rank(SmemLayoutAtomA{}) == 2);
    static_assert(cute::size<0>(TileShape{}) % cute::size<0>(SmemLayoutAtomA{}) == 0); // M
    static_assert(cute::size<2>(TileShape{}) % cute::size<1>(SmemLayoutAtomA{}) == 0); // K
    static_assert(cute::rank(SmemLayoutAtomB{}) == 2);
    static_assert(cute::size<1>(TileShape{}) % cute::size<0>(SmemLayoutAtomB{}) == 0); // N
    static_assert(cute::size<2>(TileShape{}) % cute::size<1>(SmemLayoutAtomB{}) == 0); // K

    using SmemLayoutA = decltype(cute::tile_to_shape(SmemLayoutAtomA{},
        cute::make_shape(
            cute::shape<0>(TileShape{}), cute::shape<2>(TileShape{}), cute::Int<Stages>{}))); // BLK_M, BLK_K, Stages
    using SmemLayoutB = decltype(cute::tile_to_shape(SmemLayoutAtomB{},
        cute::make_shape(
            cute::shape<1>(TileShape{}), cute::shape<2>(TileShape{}), cute::Int<Stages>{}))); // BLK_N, BLK_K, Stages
    using SmemLayoutO = decltype(cute::tile_to_shape(
        SmemLayoutAtomO{}, cute::make_shape(cute::shape<0>(TileShape{}), cute::shape<1>(TileShape{})))); // BLK_M, BLK_N

    using SmemLayoutScaleA = decltype(cute::tile_to_shape(SmemLayoutAtomScaleA{},
        cute::make_shape(cute::shape<0>(ScalePerTileShape{}), cute::shape<2>(ScalePerTileShape{}),
            cute::Int<Stages>{}))); // BLK_M, BLK_K, Stages
    using SmemLayoutScaleB = decltype(cute::tile_to_shape(SmemLayoutAtomScaleB{},
        cute::make_shape(cute::shape<1>(ScalePerTileShape{}), cute::shape<2>(ScalePerTileShape{}),
            cute::Int<Stages>{}))); // BLK_N, BLK_K, Stages

    // we need at least 2 stages..
    static_assert(Stages >= 2);

    struct SharedStorage : cute::aligned_struct<128>
    {
        cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
        cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>> smem_b;
        cute::array_aligned<ElementOutput, cute::cosize_v<SmemLayoutO>> smem_o;
        cute::array_aligned<ElementBlockScale, cute::cosize_v<SmemLayoutScaleA>> smem_scale_a;
        cute::array_aligned<ElementBlockScale, cute::cosize_v<SmemLayoutScaleB>> smem_scale_b;
    };

    static constexpr int kSmemSize = static_cast<int>(sizeof(SharedStorage));

    struct Params
    {
        GemmCoord problem_size{};
        ElementA const* ptr_a{};
        ElementB const* ptr_b{};
        ElementOutput* ptr_output{};
        BlockScaleElementType const* ptr_scale_a{};
        BlockScaleElementType const* ptr_scale_b{};

        Params() {}

        Params(GemmCoord problem_size_, ElementA const* ptr_a_, ElementB const* ptr_b_, ElementOutput* ptr_output_,
            BlockScaleElementType const* ptr_scale_a_, BlockScaleElementType const* ptr_scale_b_)
            : problem_size(problem_size_)
            , ptr_a(ptr_a_)
            , ptr_b(ptr_b_)
            , ptr_output(ptr_output_)
            , ptr_scale_a(ptr_scale_a_)
            , ptr_scale_b(ptr_scale_b_)
        {
        }
    };

    struct Arguments
    {
        GemmCoord problem_size{};
        void const* ptr_a;
        void const* ptr_b;
        void* ptr_d;
        float const* ptr_scale_a;
        float const* ptr_scale_b;

        Arguments(GemmCoord problem_size_, void const* ptr_a_, void const* ptr_b_, void* ptr_d_,
            float const* ptr_scale_a_, float const* ptr_scale_b_)
            : problem_size(problem_size_)
            , ptr_a(ptr_a_)
            , ptr_b(ptr_b_)
            , ptr_d(ptr_d_)
            , ptr_scale_a(ptr_scale_a_)
            , ptr_scale_b(ptr_scale_b_)
        {
        }
    };

    static Params to_underlying_arguments(Arguments const& args)
    {
        auto ptr_a = reinterpret_cast<ElementA const*>(args.ptr_a);
        auto ptr_b = reinterpret_cast<ElementB const*>(args.ptr_b);
        auto ptr_d = reinterpret_cast<ElementOutput*>(args.ptr_d);
        auto ptr_scale_a = reinterpret_cast<ElementBlockScale const*>(args.ptr_scale_a);
        auto ptr_scale_b = reinterpret_cast<ElementBlockScale const*>(args.ptr_scale_b);

        Params params(args.problem_size, ptr_a, ptr_b, ptr_d, ptr_scale_a, ptr_scale_b);

        return params;
    }
};

} // namespace ada_blockwise_gemm
