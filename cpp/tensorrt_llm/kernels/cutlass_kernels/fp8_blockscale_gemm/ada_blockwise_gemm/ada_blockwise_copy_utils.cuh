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
#include <cute/tensor.hpp>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_conversion.h>

// clang-format on
namespace ada_blockwise_gemm
{

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandA;

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB;

// specializations for float_e4m3
template <>
struct DefaultGemm_TensorOpSm80_OperandA<cute::float_e4m3_t, cutlass::layout::RowMajor, 16, 128>
{
    // Smem
    using smem_layout = cute::Layout<cute::Shape<cute::_8, cute::_128>, cute::Stride<cute::_128, cute::_1>>;
    using SmemLayoutAtom = decltype(cute::composition(cute::Swizzle<3, 4, 3>{}, smem_layout{}));
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::float_e4m3_t>;

    // Gmem
    using copy_atom = decltype(cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::float_e4m3_t>{});
    using thr_layout = decltype(cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_8, cute::_1>>{});
    using val_layout = decltype(cute::Layout<cute::Shape<cute::_1, cute::_16>>{});
    using GmemTiledCopy = decltype(cute::make_tiled_copy(copy_atom{}, thr_layout{}, val_layout{}));
};

template <>
struct DefaultGemm_TensorOpSm80_OperandA<cute::float_e4m3_t, cutlass::layout::ColumnMajor, 16, 128>
{
    // Smem
    using smem_layout = cute::Layout<cute::Shape<cute::_8, cute::_128>, cute::Stride<cute::_128, cute::_1>>;
    using SmemLayoutAtom = decltype(cute::composition(cute::Swizzle<3, 4, 3>{}, smem_layout{}));
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::float_e4m3_t>;

    // Gmem
    using copy_atom = decltype(cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::float_e4m3_t>{});
    using thr_layout = decltype(cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_1, cute::_16>>{});
    using val_layout = decltype(cute::Layout<cute::Shape<cute::_16, cute::_1>>{});
    using GmemTiledCopy = decltype(cute::make_tiled_copy(copy_atom{}, thr_layout{}, val_layout{}));
};

template <>
struct DefaultGemm_TensorOpSm80_OperandA<cute::half_t, cutlass::layout::RowMajor, 8, 64>
{
    // Smem
    using SmemLayoutAtom = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{}, cute::Layout<cute::Shape<cute::_8, cute::_64>, cute::Stride<cute::_64, cute::_1>>{}));
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::half_t>;

    // Gmem
    using GmemTiledCopy = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::half_t>{},
        cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_8, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));
};

template <>
struct DefaultGemm_TensorOpSm80_OperandA<cute::bfloat16_t, cutlass::layout::RowMajor, 8, 64>
{
    // Smem
    using SmemLayoutAtom = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{}, cute::Layout<cute::Shape<cute::_8, cute::_64>, cute::Stride<cute::_64, cute::_1>>{}));
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::bfloat16_t>;

    // Gmem
    using GmemTiledCopy = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::bfloat16_t>{},
        cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_8, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));
};

/// Operand A - Column-major (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpSm80_OperandA<cute::half_t, cutlass::layout::ColumnMajor, 8, SizeK>
{
    // Smem
    using SmemLayoutAtom = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{}, cute::Layout<cute::Shape<cute::_64, cute::_8>, cute::Stride<cute::_1, cute::_64>>{}));
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, cute::half_t>;

    // Gmem
    using GmemTiledCopy = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::half_t>{},
        cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_1, cute::_16>>{},
        cute::Layout<cute::Shape<cute::_8, cute::_1>>{}));
};

template <int SizeK>
struct DefaultGemm_TensorOpSm80_OperandA<cute::bfloat16_t, cutlass::layout::ColumnMajor, 8, SizeK>
{
    // Smem
    using SmemLayoutAtom = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{}, cute::Layout<cute::Shape<cute::_64, cute::_8>, cute::Stride<cute::_1, cute::_64>>{}));
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, cute::bfloat16_t>;

    // Gmem
    using GmemTiledCopy = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::bfloat16_t>{},
        cute::Layout<cute::Shape<cute::_16, cute::_8>, cute::Stride<cute::_1, cute::_16>>{},
        cute::Layout<cute::Shape<cute::_8, cute::_1>>{}));
};

// Because the F32F16 TiledMMA is A-B symmetric, we can reuse the
// DefaultOperands

// Operand B - Column-Major (K-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<cute::half_t, cutlass::layout::ColumnMajor, Alignment, SizeK>
    : DefaultGemm_TensorOpSm80_OperandA<cute::half_t, cutlass::layout::RowMajor, Alignment, SizeK>
{
};

template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<cute::bfloat16_t, cutlass::layout::ColumnMajor, Alignment, SizeK>
    : DefaultGemm_TensorOpSm80_OperandA<cute::bfloat16_t, cutlass::layout::RowMajor, Alignment, SizeK>
{
};

// Operand B - Row-Major (N-major)
template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<cute::half_t, cutlass::layout::RowMajor, Alignment, SizeK>
    : DefaultGemm_TensorOpSm80_OperandA<cute::half_t, cutlass::layout::ColumnMajor, Alignment, SizeK>
{
};

template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<cute::bfloat16_t, cutlass::layout::RowMajor, Alignment, SizeK>
    : DefaultGemm_TensorOpSm80_OperandA<cute::bfloat16_t, cutlass::layout::ColumnMajor, Alignment, SizeK>
{
};

template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<cute::float_e4m3_t, cutlass::layout::ColumnMajor, Alignment, SizeK>
    : DefaultGemm_TensorOpSm80_OperandA<cute::float_e4m3_t, cutlass::layout::RowMajor, Alignment, SizeK>
{
};

template <int Alignment, int SizeK>
struct DefaultGemm_TensorOpSm80_OperandB<cute::float_e4m3_t, cutlass::layout::RowMajor, Alignment, SizeK>
    : DefaultGemm_TensorOpSm80_OperandA<cute::float_e4m3_t, cutlass::layout::ColumnMajor, Alignment, SizeK>
{
};

//
// F16: 128-by-128-by-32 (small k-block)
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpSm80_OperandA<cute::half_t, cutlass::layout::RowMajor, 8, 32>
{
    // Smem
    using SmemLayoutAtom = decltype(cute::composition(
        cute::Swizzle<2, 3, 3>{}, cute::Layout<cute::Shape<cute::_8, cute::_32>, cute::Stride<cute::_32, cute::_1>>{}));
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::half_t>;

    // Gmem
    using GmemTiledCopy = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::half_t>{},
        cute::Layout<cute::Shape<cute::_32, cute::_4>, cute::Stride<cute::_4, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));
};

template <>
struct DefaultGemm_TensorOpSm80_OperandA<cute::bfloat16_t, cutlass::layout::RowMajor, 8, 32>
{
    // Smem
    using SmemLayoutAtom = decltype(cute::composition(
        cute::Swizzle<2, 3, 3>{}, cute::Layout<cute::Shape<cute::_8, cute::_32>, cute::Stride<cute::_32, cute::_1>>{}));
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cute::bfloat16_t>;

    // Gmem
    using GmemTiledCopy = decltype(cute::make_tiled_copy(
        cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cute::bfloat16_t>{},
        cute::Layout<cute::Shape<cute::_32, cute::_4>, cute::Stride<cute::_4, cute::_1>>{},
        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));
};

struct CopyTraitsScaleA
{
    using GmemCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint32_t>, float>;

    // Gmem
    using GmemLayoutTVScale
        = cute::Layout<cute::Shape<cute::Shape<cute::_32, cute::_4>, cute::Shape<cute::_1, cute::_1>>,
            cute::Stride<cute::Stride<cute::_1, cute::_0>, cute::Stride<cute::_1, cute::_1>>>;
    using GmemTileShapeScale = cute::Shape<cute::_32, cute::_1>;
    using GmemTiledCopyScale
        = decltype(cute::make_tiled_copy_impl(GmemCopyAtom{}, GmemLayoutTVScale{}, GmemTileShapeScale{}));

    // Smem
    using SmemCopyAtomScale = cute::Copy_Atom<cute::UniversalCopy<float>, float>;
    using SmemLayoutTVScale
        = cute::Layout<cute::Shape<cute::Shape<cute::_4, cute::_8, cute::_2, cute::_2>, cute::Shape<cute::_2>>,
            cute::Stride<cute::Stride<cute::_0, cute::_1, cute::_16, cute::_0>, cute::Stride<cute::_8>>>;
    using SmemTileShapeScale = cute::Shape<cute::_32, cute::_1>;
    using SmemTiledCopyScale
        = decltype(cute::make_tiled_copy_impl(SmemCopyAtomScale{}, SmemLayoutTVScale{}, SmemTileShapeScale{}));
};

struct CopyTraitsScaleB
{
    using GmemCopyAtom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint32_t>, float>;

    // Gmem
    using GmemLayoutTVScale
        = cute::Layout<cute::Shape<cute::Shape<cute::_32, cute::_4>, cute::Shape<cute::_1, cute::_1>>,
            cute::Stride<cute::Stride<cute::_0, cute::_0>, cute::Stride<cute::_1, cute::_1>>>;
    using GmemTileShapeScale = cute::Shape<cute::_1, cute::_1>;
    using GmemTiledCopyScale
        = decltype(cute::make_tiled_copy_impl(GmemCopyAtom{}, GmemLayoutTVScale{}, GmemTileShapeScale{}));

    // Smem
    using SmemCopyAtomScale = cute::Copy_Atom<cute::UniversalCopy<float>, float>;
    using SmemLayoutTVScale
        = cute::Layout<cute::Shape<cute::Shape<cute::_4, cute::_8, cute::_2, cute::_2>, cute::Shape<cute::_1>>,
            cute::Stride<cute::Stride<cute::_0, cute::_0, cute::_0, cute::_0>, cute::Stride<cute::_0>>>;
    using SmemTileShapeScale = cute::Shape<cute::_1, cute::_1>;
    using SmemTiledCopyScale
        = decltype(cute::make_tiled_copy_impl(SmemCopyAtomScale{}, SmemLayoutTVScale{}, SmemTileShapeScale{}));
};

template <typename To_type, typename Engine, typename Layout>
CUTE_DEVICE auto util_convert_type(cute::Tensor<Engine, Layout> const& tensor)
{
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(cute::size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<cutlass::Array<From_type, numel> const*>(tensor.data()));
    return cute::make_tensor(cute::make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
CUTE_DEVICE void util_copy(
    TiledCopy const& tiled_copy, cute::Tensor<Engine0, Layout0> const& S, cute::Tensor<Engine1, Layout1>& D)
{
    CUTE_STATIC_ASSERT_V(cute::rank(S) == cute::Int<3>{});
    CUTE_STATIC_ASSERT_V(cute::rank(D) == cute::Int<3>{});
    CUTE_STATIC_ASSERT_V(cute::size<0>(S) == cute::size<0>(D));
    CUTE_STATIC_ASSERT_V(cute::size<1>(S) == cute::size<1>(D));
    CUTE_STATIC_ASSERT_V(cute::size<2>(S) == cute::size<2>(D));

    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < cute::size<1>(S); ++m)
    {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < cute::size<2>(S); ++k)
        {
            cute::copy(tiled_copy, S(cute::_, m, k), D(cute::_, m, k));
        }
    }
}

} // namespace ada_blockwise_gemm
