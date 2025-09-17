/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Visitor tree store operations for the sm90 TMA warp-specialized (ws) epilogue
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

#include "cutlass_extensions/arch/copy_red_global.hpp"
#include "cutlass_extensions/util/gather_tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

template <
  class EpilogueTile,
  class StrideOutput,
  class SmemLayoutAtom,
  class CopyOpR2S,
  class ElementOutput,
  int AlignmentOutput = 128 / cute::sizeof_bits_v<ElementOutput>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct Sm90ScatterPtrArray {

  using SmemShape = decltype(make_shape(size(make_layout(get<0>(EpilogueTile{}))), size(make_layout(get<1>(EpilogueTile{})))));
  using SmemLayout = decltype(tile_to_shape(SmemLayoutAtom{}, SmemShape{}));

  using ElementIndex = int32_t;

  static constexpr bool MajorMode = cutlass::gemm::detail::is_major<0,StrideOutput>() ? 0 : 1;

  using StrideIndex = decltype(replace<1-MajorMode>(Stride<_0,_0,_0>{}, Int<1>{}));

  struct SharedStorage {};

  struct Arguments {
    ElementOutput* ptr_out{};               // output tensor pointer
    StrideOutput dOut = {};                 // output tensor stride
    ElementIndex const* const* ptr_index{}; // per-group pointer to the scatter index
    int index_modulo{};                     // modulo used to transform the index before store
    int shape_override = -1;                // override value for contiguous output tensor mode
    bool use_reduction = true;              // use reduction or regular store
  };

  struct Params {
    ElementOutput* ptr_out{};               // output tensor pointer
    StrideOutput dOut = {};                 // output tensor stride
    ElementIndex const* const* ptr_index{}; // per-group pointer to the scatter index
    cutlass::FastDivmod index_divmod{};     // modulo used to transform the index before store
    int shape_override = -1;                // override value for contiguous output tensor mode
    bool use_reduction = true;              // use reduction or regular store
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {
      args.ptr_out,
      args.dOut,
      args.ptr_index,
      cutlass::FastDivmod(args.index_modulo),
      args.shape_override,
      args.use_reduction
    };
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm90ScatterPtrArray() { }

  CUTLASS_HOST_DEVICE
  Sm90ScatterPtrArray(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params) { }

  Params const* params_ptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<
    class ArgsTuple
  >
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(ArgsTuple&& args_tuple)
      : args_tuple(std::move(args_tuple)) {}

    ArgsTuple args_tuple;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {

      auto& [tC_rOut, tiled_r2s, tRG_gOut, tRG_cD, tiled_r2g_red, tiled_r2g_stg, use_reduction, thread_idx, residue_cD] = args_tuple;

      using ConvertInput = NumericArrayConverter<ElementOutput, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Tensor tC_rOut_frg = recast<Array<ElementOutput, FragmentSize>>(coalesce(tC_rOut)); // (EPI_V)
      tC_rOut_frg(epi_v) = convert_input(frg_input);

      return tC_rOut_frg(epi_v);
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&& reduction_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {

      auto& [tC_rOut, tiled_r2s, tRG_gOut, tRG_cD, tiled_r2g_red, tiled_r2g_stg, use_reduction, thread_idx, residue_cD] = args_tuple;

      Tensor byte_buffer = recast<uint8_t>(reduction_buffer);
      static_assert(cosize(byte_buffer.layout()) * sizeof_bits_v<uint8_t> >= cosize(SmemLayout{}) * sizeof_bits_v<ElementOutput>,
                    "Not enough space in scratch smem buffer");

      Tensor sOut = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(recast_ptr<ElementOutput>(byte_buffer.data())), SmemLayout{}));

      auto thread_r2s = tiled_r2s.get_slice(thread_idx);
      Tensor tRS_sOut_epi = thread_r2s.partition_D(sOut);
      Tensor tRS_rOut_epi = thread_r2s.retile_S(tC_rOut);

      auto thread_r2g = tiled_r2g_red.get_slice(thread_idx);
      Tensor tRG_gOut_epi = tRG_gOut(_,_,_,epi_m,epi_n);
      Tensor tRG_sOut_epi = thread_r2g.partition_D(sOut);
      Tensor tRG_rOut_epi = thread_r2g.retile_S(make_tensor(tC_rOut.data(), shape(tRG_sOut_epi))); // reuse D registers

      // sanity check for register reuse
      CUTE_STATIC_ASSERT_V(cosize(tC_rOut.layout()) == cosize(tRG_rOut_epi.layout()), "Invalid register count for R2G");

      copy(tiled_r2s, tRS_rOut_epi, tRS_sOut_epi);
      sync_fn();
      copy(tRG_sOut_epi, tRG_rOut_epi);

      auto residue = residue_cD; // capturing structured bindings is a C++20 feature
      Tensor tRG_cD_epi = tRG_cD(0,_,_,epi_m,epi_n);
      auto pred = cute::lazy::transform(tRG_cD_epi, [&](auto c){ return elem_less(c, residue); });

      if (use_reduction) {
        copy_if(tiled_r2g_red, pred, tRG_rOut_epi, tRG_gOut_epi);
      }
      else {
        copy_if(tiled_r2g_stg, pred, tRG_rOut_epi, tRG_gOut_epi);
      }
    }
  };

  template <class Element, int MaxVecSize>
  static constexpr auto get_reduction_op()
  {
      using namespace cute;

      // For now only support red.add
      if constexpr (is_same_v<Element, cutlass::half_t>) {
        if constexpr (MaxVecSize % 8 == 0) {
          return SM90_RED_ADD_NOFTZ_F16x2_V4{};
        }
        else if constexpr (MaxVecSize % 4 == 0) {
          return SM90_RED_ADD_NOFTZ_F16x2_V2{};
        }
        else if constexpr (MaxVecSize % 2 == 0) {
          return SM70_RED_ADD_NOFTZ_F16x2{};
        }
        else {
          return SM70_RED_ADD_NOFTZ_F16{};
        }
      }
      else if constexpr (is_same_v<Element, cutlass::bfloat16_t>) {
        if constexpr (MaxVecSize % 8 == 0) {
          return SM90_RED_ADD_NOFTZ_BF16x2_V4{};
        }
        else if constexpr (MaxVecSize % 4 == 0) {
          return SM90_RED_ADD_NOFTZ_BF16x2_V2{};
        }
        else if constexpr (MaxVecSize % 2 == 0) {
          return SM90_RED_ADD_NOFTZ_BF16x2{};
        }
        else {
          return SM90_RED_ADD_NOFTZ_BF16{};
        }
      }
      else {
        // non-vectorized atomic add for all other types until supported
        return TypedAtomicAdd<Element>{};
      }
  }


  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    auto index_read = [index = params_ptr->ptr_index[l], divmod = params_ptr->index_divmod](auto i){ return divmod.rem(index[i]); };
    Tensor mOut = cutlass::util::make_gather_tensor(params_ptr->ptr_out, make_shape(M,N,Int<1>{}), params_ptr->dOut, index_read); // (M,N,_1)
    Tensor gOut = local_tile(mOut, take<0,2>(args.tile_shape_mnk), make_coord(m,n,Int<0>{}));                               // (CTA_M,CTA_N)
    Tensor gOut_epi = flat_divide(gOut, args.epi_tile);                                                                     // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    Tensor mIdx = make_tensor(params_ptr->ptr_index[l], make_shape(M,N,Int<1>{}), StrideIndex{});                           // (M,N,_1)
    Tensor gIdx = local_tile(mIdx, take<0,2>(args.tile_shape_mnk), make_coord(m,n,Int<0>{}));                               // (CTA_M,CTA_N)
    Tensor gIdx_epi = flat_divide(gIdx, args.epi_tile);                                                                     // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    Tensor cD_epi = flat_divide(args.cD, args.epi_tile);                                                                    // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    Tensor tC_gOut = sm90_partition_for_epilogue<ReferenceSrc>(gOut, args.epi_tile, args.tiled_copy, args.thread_idx);      // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    Tensor tC_rOut = make_tensor<ElementOutput>(take<0,3>(shape(tC_gOut)));                                                 // (CPY,CPY_M,CPY_N)

    auto tiled_r2s = conditional_return<ReferenceSrc>(
      make_tiled_copy_S(Copy_Atom<CopyOpR2S,ElementOutput>{}, args.tiled_copy),
      make_tiled_copy_D(Copy_Atom<CopyOpR2S,ElementOutput>{}, args.tiled_copy)
    );

    // Vectorization must not exceed alignment and also the number of values per thread in the tile
    int constexpr NumThreads = CUTE_STATIC_V(size(args.tiled_copy));
    int constexpr NumValTile = product(take<0,2>(shape(cD_epi)));
    int constexpr MaxVecSize = cute::min(AlignmentOutput, NumValTile / NumThreads);

    // Choose the largest available red.global op and an st.global op with matching vectorization
    using CopyOpR2GRed = decltype(get_reduction_op<ElementOutput, MaxVecSize>());
    using CopyOpR2GStg = UniversalCopy<uint_bit_t<Copy_Atom<CopyOpR2GRed,ElementOutput>::NumValSrc * sizeof_bits_v<ElementOutput>>>;

    auto make_tiled_r2g = [&](auto copy_op)
    {
      using CopyAtomR2G = Copy_Atom<decltype(copy_op),ElementOutput>;
      constexpr int VecSize = CopyAtomR2G::NumValSrc;
      if constexpr (cutlass::gemm::detail::is_k_major<StrideOutput>()) {
        constexpr int ThreadsMajor = size<1>(args.epi_tile) / VecSize;
        constexpr int ThreadsMinor = NumThreads / ThreadsMajor;
        return make_tiled_copy(CopyAtomR2G{},
          Layout<Shape<Int<ThreadsMinor>, Int<ThreadsMajor>>, Stride<Int<ThreadsMajor>, _1>>{},
          Layout<Shape<_1, Int<VecSize>>>{});
      }
      else if constexpr (cutlass::gemm::detail::is_mn_major<StrideOutput>()) {
        constexpr int ThreadsMajor = size<0>(args.epi_tile) / VecSize;
        constexpr int ThreadsMinor = NumThreads / ThreadsMajor;
        return make_tiled_copy(CopyAtomR2G{},
          Layout<Shape<Int<ThreadsMajor>, Int<ThreadsMinor>>, Stride<_1, Int<ThreadsMajor>>>{},
          Layout<Shape<Int<VecSize>, _1>>{});
      }
      else {
        static_assert(cute::is_void_v<StrideOutput>, "Unsupported D gmem layout.");
      }
    };

    auto tiled_r2g_red = make_tiled_r2g(CopyOpR2GRed{});
    auto tiled_r2g_stg = make_tiled_r2g(CopyOpR2GStg{});

    // Sanity checks - since we will be using one tiled copy with tensors partitioned with the other tiled copy,
    // ensure they have matching layouts/tilers
    using TiledR2GRed = decltype(tiled_r2g_red);
    using TiledR2GStg = decltype(tiled_r2g_stg);
    static_assert(typename TiledR2GRed::AtomLayoutSrc{} == typename TiledR2GStg::AtomLayoutSrc{}, "Mismatching AtomLayoutSrc");
    static_assert(typename TiledR2GRed::AtomLayoutDst{} == typename TiledR2GStg::AtomLayoutDst{}, "Mismatching AtomLayoutDst");
    static_assert(typename TiledR2GRed::TiledLayout_TV{} == typename TiledR2GStg::TiledLayout_TV{}, "Mismatching TiledLayout_TV");
    static_assert(typename TiledR2GRed::Tiler_MN{} == typename TiledR2GStg::Tiler_MN{}, "Mismatching Tiler_MN");

    auto thread_r2g = tiled_r2g_red.get_slice(args.thread_idx);
    Tensor tRG_gOut = thread_r2g.partition_D(gOut_epi);                      // (R2G,R2G_M,R2G_N,EPI_M,EPI_N)
    Tensor tRG_cD = thread_r2g.partition_D(cD_epi);                          // (R2G,R2G_M,R2G_N,EPI_M,EPI_N)

    auto residue_cD = args.residue_cD;

    // If shape_override is set, adjust residue_cD to change predication.
    // This is used to support fused slicing (where the output tensor is smaller than problem shape)
    if (params_ptr->shape_override >= 0) {
      get<MajorMode>(residue_cD) += params_ptr->shape_override - get<MajorMode>(args.problem_shape_mnkl);
    }

    auto args_tuple = make_tuple(
      cute::move(tC_rOut),
      tiled_r2s,
      tRG_gOut,
      tRG_cD,
      tiled_r2g_red,
      tiled_r2g_stg,
      params_ptr->use_reduction,
      args.thread_idx,
      residue_cD);

    return ConsumerStoreCallbacks<decltype(args_tuple)>(std::move(args_tuple));
  }
};

template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledAccPerRowBias
    : ScaledAcc<ElementOutput_, ElementCompute_, ElementScalar_, RoundStyle_>
{
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = AlignmentBias_;
  static constexpr bool IsPerRowBiasSupported = true;
};

template<
  class ElementOutput_,
  class ElementCompute_,
  class ElementBias_ = ElementOutput_,
  class ElementScalar_ = ElementCompute_,
  int AlignmentBias_ = 128 / cute::sizeof_bits_v<ElementBias_>,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct ScaledAccPerColBias
    : ScaledAcc<ElementOutput_, ElementCompute_, ElementScalar_, RoundStyle_>
{
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = AlignmentBias_;
  static constexpr bool IsPerColBiasSupported = true;
};

template<
  class GmemLayoutTagOut,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementScale = ElementCompute,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>,
  int AlignmentOutput = 128 / cute::sizeof_bits_v<ElementOutput>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct ScaledAccPerRowBiasPerColScaleScatter
    : ScaledAccPerRowBias<ElementOutput, ElementCompute, ElementBias, ElementScalar, AlignmentBias, RoundStyle>
{
  using ElementAux = ElementOutput;
  using GmemLayoutTagAux = GmemLayoutTagOut;
  static constexpr int AlignmentAux = AlignmentOutput;
  static constexpr bool IsAuxOutSupported = true;
};

template<
  class GmemLayoutTagOut,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementScale = ElementCompute,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>,
  int AlignmentOutput = 128 / cute::sizeof_bits_v<ElementOutput>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct ScaledAccPerColBiasPerRowScaleScatter
    : ScaledAccPerColBias<ElementOutput, ElementCompute, ElementBias, ElementScalar, AlignmentBias, RoundStyle>
{
  using ElementAux = ElementOutput;
  using GmemLayoutTagAux = GmemLayoutTagOut;
  static constexpr int AlignmentAux = AlignmentOutput;
  static constexpr bool IsAuxOutSupported = true;
};

// D = alpha * acc + per-row bias
template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledAccPerRowBiasPtrArray =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // alpha * acc + bias
    Sm90ScalarBroadcastPtrArray<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
    Sm90AccFetch, // acc
    Sm90ColBroadcast<0, CtaTileShapeMNK, ElementBias *, ElementCompute, Stride<_1,_0,int64_t>, AlignmentBias> // bias
  >;

template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledAccPerColBiasPtrArray =
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // alpha * acc + bias
    Sm90ScalarBroadcastPtrArray<ElementScalar, Stride<_0,_0,int64_t>>, // alpha
    Sm90AccFetch, // acc
    Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias *, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias> // bias
  >;

template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  class StrideOutput,
  class SmemLayoutAtom,
  class CopyOpR2S,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementScale = ElementCompute,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>,
  int AlignmentOutput = 128 / cute::sizeof_bits_v<ElementOutput>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledAccPerRowBiasPerColScaleScatterPtrArray =
  Sm90EVT<Sm90ScatterPtrArray<EpilogueTile, StrideOutput, SmemLayoutAtom, CopyOpR2S, ElementOutput, AlignmentOutput, RoundStyle>, // scatter store
    Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // scale * (alpha * acc + bias)
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementScalar *, ElementCompute, Stride<_0,_1,int64_t>, 1>, // scale
      Sm90ScaledAccPerRowBiasPtrArray<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementScalar, AlignmentBias, RoundStyle> // alpha * acc + bias
    >
  >;

template<
  class CtaTileShapeMNK,
  class EpilogueTile,
  class StrideOutput,
  class SmemLayoutAtom,
  class CopyOpR2S,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  class ElementScale = ElementCompute,
  class ElementScalar = ElementCompute,
  int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>,
  int AlignmentOutput = 128 / cute::sizeof_bits_v<ElementOutput>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Sm90ScaledAccPerColBiasPerRowScaleScatterPtrArray =
  Sm90EVT<Sm90ScatterPtrArray<EpilogueTile, StrideOutput, SmemLayoutAtom, CopyOpR2S, ElementOutput, AlignmentOutput, RoundStyle>, // scatter store
    Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // scale * (alpha * acc + bias)
      Sm90ColBroadcast<0, CtaTileShapeMNK, ElementScalar *, ElementCompute, Stride<_1,_0,int64_t>, 1>, // scale
      Sm90ScaledAccPerColBiasPtrArray<CtaTileShapeMNK, ElementCompute, ElementCompute, ElementBias, ElementScalar, AlignmentBias, RoundStyle> // alpha * acc + bias
    >
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  class GmemLayoutTagOut,
  class ElementOutput,
  class ElementCompute,
  class ElementBias,
  class ElementScale,
  class ElementScalar,
  int AlignmentBias,
  int AlignmentOutput,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpR2S
>
struct FusionCallbacks<
    epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC,
                                             StagesD,
                                             FragmentSize,
                                             ReuseSmemC,
                                             DelayTmaStore,
                                             NumEpilogueWarpGroups
                                            >,
    fusion::ScaledAccPerRowBiasPerColScaleScatter<GmemLayoutTagOut,
                                                  ElementOutput,
                                                  ElementCompute,
                                                  ElementBias,
                                                  ElementScale,
                                                  ElementScalar,
                                                  AlignmentBias,
                                                  AlignmentOutput,
                                                  RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpR2S
> : Sm90ScaledAccPerRowBiasPerColScaleScatterPtrArray<
      CtaTileShapeMNK,
      EpilogueTile,
      cutlass::gemm::TagToStrideC_t<GmemLayoutTagOut>,
      SmemLayoutAtom, CopyOpR2S,
      ElementOutput, ElementCompute, ElementBias, ElementScale, ElementScalar,
      AlignmentBias, AlignmentOutput, RoundStyle
    > {

  using StrideOutput = cutlass::gemm::TagToStrideC_t<GmemLayoutTagOut>;

  using Impl = Sm90ScaledAccPerRowBiasPerColScaleScatterPtrArray<
    CtaTileShapeMNK,
    EpilogueTile,
    StrideOutput,
    SmemLayoutAtom, CopyOpR2S,
    ElementOutput, ElementCompute, ElementBias, ElementScale, ElementScalar,
    AlignmentBias, AlignmentOutput, RoundStyle
  >;
  using Operation = fusion::ScaledAccPerRowBiasPerColScaleScatter<
    GmemLayoutTagOut,
    ElementOutput,
    ElementCompute,
    ElementBias,
    ElementScale,
    ElementScalar,
    AlignmentBias,
    AlignmentOutput,
    RoundStyle>;

  struct Arguments {

    using StrideAlpha = Stride<_0,_0,int64_t>;
    ElementScalar alpha = ElementScalar(1);
    ElementScalar const* alpha_ptr{};
    ElementScalar const* const* alpha_ptr_array{};
    StrideAlpha dAlpha{};

    using StrideBias = Stride<_1,_0,int64_t>;
    ElementBias const* const* bias_ptr{};
    StrideBias dBias{};

    using StrideScale = Stride<_0,_1,int64_t>;
    ElementScalar const* const* scale_ptr_array{};
    StrideScale dScale{};

    // Nested args not usable due to a compiler bug with constexpr evaluation
    // using ScatterArguments = typename Sm90ScatterPtrArray<EpilogueTile, StrideOutput, SmemLayoutAtom, CopyOpR2S, ElementOutput, AlignmentOutput, RoundStyle>::Arguments;
    // ScatterArguments scatter{};

    ElementOutput* ptr_out{};      // output tensor pointer
    StrideOutput dOut{};           // output tensor stride
    int const* const* ptr_index{}; // per-group pointer to the scatter index
    int index_modulo{};            // modulo used to transform the index before store
    int shape_override = -1;       // override value for contiguous output tensor mode
    bool use_reduction = true;     // use reduction or regular store

    operator typename Impl::Arguments() const {
      return
        {                                                           // unary op: reduce(scale * (beta * C + (alpha * acc)))
          {                                                             // binary op: scale * (beta * C + (alpha * acc))
            { scale_ptr_array, ElementScalar(1), dScale },                  // leaf args : scale broadcast
            {                                                                 // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}},                // leaf args : alpha
              {},                                                                 // leaf args : acc
              {bias_ptr, ElementBias(0), dBias},                                  // leaf args : bias
              {}                                                                  // ternary args : multiply_add
            },                                                                // end binary op
            {}                                                              // binary args: multiply
          },                                                            // end binary op
          //scatter                                                       // unary args: reduce
          { ptr_out, dOut, ptr_index, index_modulo, shape_override, use_reduction }
        };                                                          // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;

};

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  class GmemLayoutTagOut,
  class ElementOutput,
  class ElementCompute,
  class ElementBias,
  class ElementScale,
  class ElementScalar,
  int AlignmentBias,
  int AlignmentOutput,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtom,
  class CopyOpR2S
>
struct FusionCallbacks<
    epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC,
                                             StagesD,
                                             FragmentSize,
                                             ReuseSmemC,
                                             DelayTmaStore,
                                             NumEpilogueWarpGroups
                                            >,
    fusion::ScaledAccPerColBiasPerRowScaleScatter<GmemLayoutTagOut,
                                                  ElementOutput,
                                                  ElementCompute,
                                                  ElementBias,
                                                  ElementScale,
                                                  ElementScalar,
                                                  AlignmentBias,
                                                  AlignmentOutput,
                                                  RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtom,
    CopyOpR2S
> : Sm90ScaledAccPerColBiasPerRowScaleScatterPtrArray<
      CtaTileShapeMNK,
      EpilogueTile,
      cutlass::gemm::TagToStrideC_t<GmemLayoutTagOut>,
      SmemLayoutAtom, CopyOpR2S,
      ElementOutput, ElementCompute, ElementBias, ElementScale, ElementScalar,
      AlignmentBias, AlignmentOutput, RoundStyle
    > {

  using StrideOutput = cutlass::gemm::TagToStrideC_t<GmemLayoutTagOut>;

  using Impl = Sm90ScaledAccPerColBiasPerRowScaleScatterPtrArray<
    CtaTileShapeMNK,
    EpilogueTile,
    StrideOutput,
    SmemLayoutAtom, CopyOpR2S,
    ElementOutput, ElementCompute, ElementBias, ElementScale, ElementScalar,
    AlignmentBias, AlignmentOutput, RoundStyle
  >;
  using Operation = fusion::ScaledAccPerColBiasPerRowScaleScatter<
    GmemLayoutTagOut,
    ElementOutput,
    ElementCompute,
    ElementBias,
    ElementScale,
    ElementScalar,
    AlignmentBias,
    AlignmentOutput,
    RoundStyle>;

  struct Arguments {

    using StrideAlpha = Stride<_0,_0,int64_t>;
    ElementScalar alpha = ElementScalar(1);
    ElementScalar const* alpha_ptr{};
    ElementScalar const* const* alpha_ptr_array{};
    StrideAlpha dAlpha{};

    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* const* bias_ptr{};
    StrideBias dBias{};

    using StrideScale = Stride<_1,_0,int64_t>;
    ElementScalar const* const* scale_ptr_array{};
    StrideScale dScale{};

    // Nested args not usable due to a compiler bug with constexpr evaluation
    // using ScatterArguments = typename Sm90ScatterPtrArray<EpilogueTile, StrideOutput, SmemLayoutAtom, CopyOpR2S, ElementOutput, AlignmentOutput, RoundStyle>::Arguments;
    // ScatterArguments scatter{};

    ElementOutput* ptr_out = nullptr;
    StrideOutput dOut = {};
    int const* const* ptr_index{};   // per-group pointer to the scatter index
    int index_modulo{}; // modulo used to transform the index before store
    int shape_override = -1; // override value for contiguous output tensor mode
    bool use_reduction = true;

    operator typename Impl::Arguments() const {
      return
        {                                                           // unary op: reduce(scale * (beta * C + (alpha * acc)))
          {                                                             // binary op: scale * (beta * C + (alpha * acc))
            { scale_ptr_array, ElementScalar(1), dScale },                  // leaf args : scale broadcast
            {                                                                 // ternary op : alpha * acc + bias
              {{alpha}, {alpha_ptr}, {alpha_ptr_array}, {dAlpha}},                // leaf args : alpha
              {},                                                                 // leaf args : acc
              {bias_ptr, ElementBias(0), dBias},                                  // leaf args : bias
              {}                                                                  // ternary args : multiply_add
            },                                                                // end binary op
            {}                                                              // binary args: multiply
          },                                                            // end binary op
          //scatter                                                       // unary args: reduce
          { ptr_out, dOut, ptr_index, index_modulo, shape_override, use_reduction }
        };                                                          // end unary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;

};

} // namespace cutlass::epilogue::fusion

// clang-format on
