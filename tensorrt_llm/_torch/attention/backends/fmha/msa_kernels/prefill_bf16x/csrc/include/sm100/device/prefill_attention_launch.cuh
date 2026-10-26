// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once

#include "sm100/common/sm107_mma_ts.cuh"
#include "sm100/kernel/sm100_prefill_attention_kernel.cuh"

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#if FMHA_SM100_PREFILL_BF16X_SM107
#include "cute/arch/mma_sm107_umma.hpp"
#include "cute/atom/mma_traits_sm107.hpp"
#endif
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/version.h"

namespace fmha_sm100::prefill_bf16x {
namespace {

// One (page, head) block of `tile_rows` x 128 contiguous bytes per box; the head and page
// strides come from the cache view, so vLLM's packed pages are read in place.
cudaError_t encode_page_head_tma(cute::TmaDescriptor &descriptor, CacheView const &view,
                                 int physical_pages, int num_kv_heads, int tile_rows) {
  cuuint64_t const global_dimensions[4] = {
      128,
      static_cast<cuuint64_t>(tile_rows),
      static_cast<cuuint64_t>(num_kv_heads),
      static_cast<cuuint64_t>(physical_pages),
  };
  cuuint64_t const global_strides[3] = {
      128,
      static_cast<cuuint64_t>(view.head_stride),
      static_cast<cuuint64_t>(view.page_stride),
  };
  cuuint32_t const box_dimensions[4] = {128, static_cast<cuuint32_t>(tile_rows), 1, 1};
  cuuint32_t const element_strides[4] = {1, 1, 1, 1};
  CUresult const result = cuTensorMapEncodeTiled(
      reinterpret_cast<CUtensorMap *>(&descriptor), CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
      const_cast<uint8_t *>(view.ptr), global_dimensions, global_strides, box_dimensions,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

} // namespace

cudaError_t launch_prefill_attention(PrefillArguments const &arguments, cudaStream_t stream) {
  using namespace cute;
  using namespace detail;

#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1
  // BF16: tcgen05 kind::f16, BF16 x BF16 -> FP32, 128 x 128 tile, K = 16 per instruction (8
  // k-blocks over the 128 dims). CUTLASS's SM100 SS atom is compiled for SM100A, SM103A and
  // SM107A alike (CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED); kind::f16 has no SM107 K extension.
  using QkMmaAtom =
      SM100_MMA_F16BF16_SS<Element, Element, Accumulator, 128, 128, UMMA::Major::K, UMMA::Major::K>;
#elif FMHA_SM100_PREFILL_BF16X_SM107 && !defined(FMHA_SM100_PREFILL_BF16X_E4M3_K32) && \
    !(defined(FMHA_SM100_PREFILL_BF16X_E4M3X) && FMHA_SM100_PREFILL_BF16X_E4M3X)
  // E4M3X always takes the K = 32 atoms: the S accumulation order must be the CuTe-DSL fp8 forward's (VR oracle v2
  // 837815: bitwise on wide-range scales only at K = 32; same speed, 837675).
  // SM107: K = 64 per instruction, twice the SM100 FP8 rate. (E4M3_K32: the SM100 K = 32 atoms below, as the
  // CuTe-DSL fp8 forward issues them, for the E4M3X exact path.)
  using QkMmaAtom =
      SM107_MMA_F8F6F4_SS<Element, Element, Accumulator, 128, 128, UMMA::Major::K, UMMA::Major::K>;
#elif CUTLASS_MAJOR > 4 || (CUTLASS_MAJOR == 4 && CUTLASS_MINOR >= 8)
  using QkMmaAtom =
      SM100_MMA_F8F6F4_SS<Element, Element, Accumulator, 128, 128, UMMA::Major::K, UMMA::Major::K>;
#else
  // Before CUTLASS 4.8 the SS atom is untyped and the types live in its MMA_Traits.
  using QkMmaAtom = MMA_Atom<MMA_Traits<
      SM100_MMA_F8F6F4_SS, Element, Element, Accumulator, Int<128>, Int<128>,
      integral_constant<UMMA::Major, UMMA::Major::K>, integral_constant<UMMA::Major, UMMA::Major::K>,
      integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>,
      integral_constant<UMMA::ScaleIn, UMMA::ScaleIn::One>>>;
#endif
  using TiledMmaQK = decltype(make_tiled_mma(QkMmaAtom{}));
#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1 && FMHA_SM100_PREFILL_BF16X_SM107
  // BF16 PV: A = P (BF16) from TMEM, B = V (BF16, MN-major) from SMEM, kind::f16.
  using PvMmaAtom = detail::Sm107MmaF16Bf16Ts<Element, Element, Accumulator, 128, 128,
                                              UMMA::Major::K, UMMA::Major::MN>;
#elif FMHA_SM100_PREFILL_BF16X_DTYPE == 1
  using PvMmaAtom = SM100_MMA_F16BF16_TS<Element, Element, Accumulator, 128, 128, UMMA::Major::K,
                                         UMMA::Major::MN>;
#elif FMHA_SM100_PREFILL_BF16X_SM107 && !defined(FMHA_SM100_PREFILL_BF16X_E4M3_K32) && \
    !(defined(FMHA_SM100_PREFILL_BF16X_E4M3X) && FMHA_SM100_PREFILL_BF16X_E4M3X)
  using PvMmaAtom = detail::Sm107MmaF8F6F4Ts<Element, Element, Accumulator, 128, 128,
                                             UMMA::Major::K, UMMA::Major::MN>;
#else
  using PvMmaAtom = SM100_MMA_F8F6F4_TS<Element, Element, Accumulator, 128, 128, UMMA::Major::K,
                                        UMMA::Major::MN>;
#endif
  using TiledMmaPV = decltype(make_tiled_mma(PvMmaAtom{}));

  using QShape = decltype(make_shape(Int<128>{}, Int<128>{}, Int<kQStages>{}));
  using KShape = decltype(make_shape(Int<128>{}, Int<128>{}, _1{}));
  using VShape = decltype(make_shape(Int<128>{}, Int<128>{}, _1{}));
  using QAtom = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                         UMMA::Major::K, Element, decltype(shape<0>(QShape{})),
                         decltype(shape<1>(QShape{}))>());
  using KAtom = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                         UMMA::Major::K, Element, decltype(shape<0>(KShape{})),
                         decltype(shape<1>(KShape{}))>());
  using VAtom = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                         UMMA::Major::MN, Element, decltype(shape<0>(VShape{})),
                         decltype(shape<1>(VShape{}))>());
  using QMmaShape = decltype(partition_shape_A(TiledMmaQK{}, QShape{}));
  using KMmaShape = decltype(partition_shape_B(TiledMmaQK{}, KShape{}));
  using VMmaShape = decltype(partition_shape_B(TiledMmaPV{}, VShape{}));
  using QSmemLayout = decltype(UMMA::tile_to_mma_shape(QAtom{}, QMmaShape{}));
  using KSmemLayout = decltype(UMMA::tile_to_mma_shape(KAtom{}, KMmaShape{}));
  using VSmemLayout = decltype(UMMA::tile_to_mma_shape(VAtom{}, VMmaShape{}));
  using QLogicalLayout = decltype(tile_to_shape(QAtom{}, QShape{}, Step<_1, _2, _3>{}));
  using KLogicalLayout = decltype(tile_to_shape(KAtom{}, KShape{}, Step<_1, _2, _3>{}));
  using VLogicalLayout = decltype(tile_to_shape(VAtom{}, VShape{}, Step<_1, _2, _3>{}));
  static_assert(cosize_v<QSmemLayout> == cosize_v<QLogicalLayout>);
  static_assert(cosize_v<KSmemLayout> == cosize_v<KLogicalLayout>);
  static_assert(cosize_v<VSmemLayout> == cosize_v<VLogicalLayout>);
#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1
  // The BF16 dequant writes K and V with hand-computed SW128 addresses
  // (bf16_tile_linear_byte_offset, then sw128_swizzle_byte_offset); check them against the
  // layouts the MMA descriptors are built from. The atoms are ComposedLayout<Swizzle<3,4,3>,
  // smem_ptr_flag_bits<16>, L>, whose swizzle acts on byte addresses (cute applies it to the smem
  // pointer), so the unswizzled offsets of L (in elements) are compared with the linear formula
  // and the swizzle functor is checked to be the byte-level SW128 one.
  static_assert(cute::is_same_v<cute::remove_cvref_t<decltype(KLogicalLayout{}.layout_a())>,
                                cute::Swizzle<3, 4, 3>>);
  static_assert(cute::is_same_v<cute::remove_cvref_t<decltype(VLogicalLayout{}.layout_a())>,
                                cute::Swizzle<3, 4, 3>>);
  // K is (token, dim); V is (dim, token): the PV B operand is N = dim by K = token. The
  // coordinates cover every address component (row in atom, row group, dim chunk, dim half).
  static_assert(KLogicalLayout{}.layout_b()(0, 0, 0) * 2 == bf16_tile_linear_byte_offset<false>(0, 0));
  static_assert(KLogicalLayout{}.layout_b()(1, 8, 0) * 2 == bf16_tile_linear_byte_offset<false>(1, 8));
  static_assert(KLogicalLayout{}.layout_b()(7, 56, 0) * 2 ==
                bf16_tile_linear_byte_offset<false>(7, 56));
  static_assert(KLogicalLayout{}.layout_b()(13, 72, 0) * 2 ==
                bf16_tile_linear_byte_offset<false>(13, 72));
  static_assert(KLogicalLayout{}.layout_b()(66, 64, 0) * 2 ==
                bf16_tile_linear_byte_offset<false>(66, 64));
  static_assert(KLogicalLayout{}.layout_b()(127, 127, 0) * 2 ==
                bf16_tile_linear_byte_offset<false>(127, 127));
  static_assert(VLogicalLayout{}.layout_b()(0, 0, 0) * 2 == bf16_tile_linear_byte_offset<true>(0, 0));
  static_assert(VLogicalLayout{}.layout_b()(8, 1, 0) * 2 == bf16_tile_linear_byte_offset<true>(1, 8));
  static_assert(VLogicalLayout{}.layout_b()(56, 7, 0) * 2 ==
                bf16_tile_linear_byte_offset<true>(7, 56));
  static_assert(VLogicalLayout{}.layout_b()(72, 13, 0) * 2 ==
                bf16_tile_linear_byte_offset<true>(13, 72));
  static_assert(VLogicalLayout{}.layout_b()(64, 66, 0) * 2 ==
                bf16_tile_linear_byte_offset<true>(66, 64));
  static_assert(VLogicalLayout{}.layout_b()(127, 127, 0) * 2 ==
                bf16_tile_linear_byte_offset<true>(127, 127));
  // Q: the box of token t, k-atom kb starts at kb * 128 * kQTmaCols + t * cosize(token box), and
  // a stage is 128 * kHeadDim elements.
  static_assert(QLogicalLayout{}.layout_b()(16 * 3, 0, 0) == 3 * cosize_v<QTokenSmemLayout>);
  static_assert(QLogicalLayout{}.layout_b()(16 * 5, kQTmaCols, 0) ==
                128 * kQTmaCols + 5 * cosize_v<QTokenSmemLayout>);
  static_assert(QLogicalLayout{}.layout_b()(0, 0, 1) == 128 * kHeadDim);
  static_assert(cosize_v<QTokenSmemLayout> == kQHeadsPerKv * kQTmaCols);
#endif

  auto q_gmem_shape = make_shape(arguments.total_q * arguments.num_q_heads, Int<kHeadDim>{});
  auto q_gmem_stride = make_stride(Int<kHeadDim>{}, _1{});
  auto tma_q =
      make_tma_copy(SM90_TMA_LOAD{},
                    make_tensor(make_gmem_ptr(reinterpret_cast<Element const *>(arguments.q_ptr)),
                                make_layout(q_gmem_shape, q_gmem_stride)),
                    QTokenSmemLayout{});

  cute::TmaDescriptor tma_packed_k{};
  cute::TmaDescriptor tma_packed_v{};
  cute::TmaDescriptor tma_k_scale{};
  cute::TmaDescriptor tma_v_scale{};
  cudaError_t status = encode_page_head_tma(tma_packed_k, arguments.packed_k,
                                            arguments.physical_pages, arguments.num_kv_heads, 64);
  if (status != cudaSuccess) {
    return status;
  }
  status = encode_page_head_tma(tma_packed_v, arguments.packed_v, arguments.physical_pages,
                                arguments.num_kv_heads, 64);
  if (status != cudaSuccess) {
    return status;
  }
  status = encode_page_head_tma(tma_k_scale, arguments.k_scale, arguments.physical_pages,
                                arguments.num_kv_heads, 8);
  if (status != cudaSuccess) {
    return status;
  }
  status = encode_page_head_tma(tma_v_scale, arguments.v_scale, arguments.physical_pages,
                                arguments.num_kv_heads, 8);
  if (status != cudaSuccess) {
    return status;
  }

  using Storage = SharedStorage<QSmemLayout, KSmemLayout, VSmemLayout>;
  // 227 KB of dynamic SMEM per CTA (see the budget in prefill_attention_config.cuh).
  static_assert(sizeof(Storage) <= 232448, "SharedStorage exceeds the 227 KB SMEM limit");
  using Params = KernelParams<decltype(tma_q), decltype(q_gmem_shape)>;
  Params kernel_params{arguments,    tma_q,       q_gmem_shape, tma_packed_k,
                       tma_packed_v, tma_k_scale, tma_v_scale};
  auto *kernel =
      &prefill_attention_kernel<Storage, TiledMmaQK, TiledMmaPV, QSmemLayout, KSmemLayout,
                                VSmemLayout, decltype(tma_q), decltype(q_gmem_shape)>;
  status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage));
  if (status != cudaSuccess) {
    return status;
  }

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(arguments.work_capacity, 1, 1);
  config.blockDim = dim3(WarpSpecialization::kThreads, 1, 1);
  config.dynamicSmemBytes = sizeof(Storage);
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, kernel, kernel_params);
}

} // namespace fmha_sm100::prefill_bf16x
