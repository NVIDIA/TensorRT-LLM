// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once

// tcgen05 kind::f8f6f4 MMA with A in TMEM and K = 64 for SM107. CUTLASS 4.8 provides the SS form
// (SM107_MMA_F8F6F4_SS) but no TS form; the instruction is the same as SM100's TS form, only the
// instruction descriptor selects the 64-deep K (make_sm107_instr_desc), which runs at twice the
// SM100 FP8 rate on SM107.
//
// BF16 data path (FMHA_SM100_PREFILL_BF16X_DTYPE == 1): Sm107MmaF16Bf16Ts is the kind::f16 TS
// form (A = P in TMEM, B = V in SMEM, BF16 x BF16 -> FP32) for SM107. CUTLASS 4.8 has no SM107_*
// f16 atom; its SM100_MMA_F16BF16_TS is compiled for SM107A too (CUTE_ARCH_TCGEN05_F16F32_MMA_
// ENABLED), and the f16 instruction descriptor has no SM107 K extension (k_size_ is f8f6f4-only),
// so this atom is the same PTX and the standard descriptor with K = 16 (256 bits). It is kept
// local, like the f8f6f4 atom, so the SM107 PV path does not depend on that CUTLASS gating.

#include "cute/tensor.hpp"

#include "sm100/common/prefill_attention_config.cuh"

#if FMHA_SM100_PREFILL_BF16X_SM107

#include "cute/arch/config.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/atom/mma_traits.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace fmha_sm100::prefill_bf16x::detail {

template <class a_type, class b_type, class c_type, int M, int N, cute::UMMA::Major a_major,
          cute::UMMA::Major b_major, cute::UMMA::ScaleIn a_neg = cute::UMMA::ScaleIn::One,
          cute::UMMA::ScaleIn b_neg = cute::UMMA::ScaleIn::One,
          cute::UMMA::Saturate c_sat = cute::UMMA::Saturate::False>
struct Sm107MmaF8F6F4Ts {
  static_assert(M == 128, "Sm107MmaF8F6F4Ts: M must be 128");
  static_assert(N % 16 == 0 && N >= 16 && N <= 256, "Sm107MmaF8F6F4Ts: N must be a multiple of 16");
  static_assert(a_major == cute::UMMA::Major::K, "Sm107MmaF8F6F4Ts: A from TMEM is K-major");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t const &tmem_a, uint64_t const &desc_b,
                                   uint32_t const &tmem_c, uint32_t const &scaleC,
                                   uint64_t const &idescE) {
#if defined(CUTE_ARCH_TCGEN05_RUBIN_F8F6F4_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile("{\n\t"
                   ".reg .pred p;\n\t"
                   "setp.ne.b32 p, %4, 0;\n\t"
                   "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, "
                   "{%5, %6, %7, %8}, p; \n\t"
                   "}\n"
                   :
                   : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)),
                     "r"(scaleC), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
#else
    CUTE_INVALID_CONTROL_PATH(
        "Sm107MmaF8F6F4Ts requires CUTE_ARCH_TCGEN05_RUBIN_F8F6F4_MMA_ENABLED");
#endif
  }
};

#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1
template <class a_type, class b_type, class c_type, int M, int N, cute::UMMA::Major a_major,
          cute::UMMA::Major b_major, cute::UMMA::ScaleIn a_neg = cute::UMMA::ScaleIn::One,
          cute::UMMA::ScaleIn b_neg = cute::UMMA::ScaleIn::One,
          cute::UMMA::Saturate c_sat = cute::UMMA::Saturate::False>
struct Sm107MmaF16Bf16Ts {
  static_assert(M == 128, "Sm107MmaF16Bf16Ts: M must be 128");
  static_assert(N % 16 == 0 && N >= 16 && N <= 256,
                "Sm107MmaF16Bf16Ts: N must be a multiple of 16");
  static_assert(a_major == cute::UMMA::Major::K, "Sm107MmaF16Bf16Ts: A from TMEM is K-major");

  using DRegisters = void;
  using ARegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t const &tmem_a, uint64_t const &desc_b,
                                   uint32_t const &tmem_c, uint32_t const &scaleC,
                                   uint64_t const &idescE) {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    if (cute::elect_one_sync()) {
      uint32_t mask[4] = {0, 0, 0, 0};
      asm volatile("{\n\t"
                   ".reg .pred p;\n\t"
                   "setp.ne.b32 p, %4, 0;\n\t"
                   "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, "
                   "{%5, %6, %7, %8}, p; \n\t"
                   "}\n"
                   :
                   : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idescE >> 32)),
                     "r"(scaleC), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
#else
    CUTE_INVALID_CONTROL_PATH(
        "Sm107MmaF16Bf16Ts requires CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED");
#endif
  }
};
#endif // FMHA_SM100_PREFILL_BF16X_DTYPE == 1

} // namespace fmha_sm100::prefill_bf16x::detail

namespace cute {

template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, UMMA::Saturate c_sat>
struct MMA_Traits<fmha_sm100::prefill_bf16x::detail::Sm107MmaF8F6F4Ts<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> <= 8 && cute::sizeof_bits_v<b_type> <= 8);

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  // SM107: 512-bit logical K per instruction, 64 elements of 8 bits.
  static constexpr int K = 64;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_sm107_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg,
                                  c_sat>();

  template <class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC,
            class CLayout>
  CUTE_HOST_DEVICE constexpr friend void
  mma_unpack(MMA_Traits const &traits, Tensor<TD, DLayout> &D, Tensor<TA, ALayout> const &A,
             Tensor<TB, BLayout> const &B, Tensor<TC, CLayout> const &C) {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");
    uint64_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
    fmha_sm100::prefill_bf16x::detail::Sm107MmaF8F6F4Ts<a_type, b_type, c_type, M, N, a_major,
                                                        b_major, a_neg, b_neg, c_sat>::
        fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};

#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1
template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, UMMA::Saturate c_sat>
struct MMA_Traits<fmha_sm100::prefill_bf16x::detail::Sm107MmaF16Bf16Ts<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, c_sat>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == 16 && cute::sizeof_bits_v<b_type> == 16,
                "Sm107MmaF16Bf16Ts takes 16-bit A and B");

  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_1sm<c_type, int32_t, UMMA::TmemAllocMode::NonInterleaved>;

  // 256-bit logical K per instruction: 16 BF16 elements (no SM107 K extension for kind::f16).
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;

  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg,
                            c_sat>();

  template <class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC,
            class CLayout>
  CUTE_HOST_DEVICE constexpr friend void
  mma_unpack(MMA_Traits const &traits, Tensor<TD, DLayout> &D, Tensor<TA, ALayout> const &A,
             Tensor<TB, BLayout> const &B, Tensor<TC, CLayout> const &C) {
    static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected desc registers in MMA_Atom::call");
    static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");
    uint32_t tmem_a = raw_pointer_cast(A.data());
    uint64_t desc_b = B[0];
    uint32_t tmem_c = raw_pointer_cast(D.data());
    uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
    fmha_sm100::prefill_bf16x::detail::Sm107MmaF16Bf16Ts<a_type, b_type, c_type, M, N, a_major,
                                                         b_major, a_neg, b_neg, c_sat>::
        fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
  }
};
#endif // FMHA_SM100_PREFILL_BF16X_DTYPE == 1

} // namespace cute

#endif // FMHA_SM100_PREFILL_BF16X_SM107
