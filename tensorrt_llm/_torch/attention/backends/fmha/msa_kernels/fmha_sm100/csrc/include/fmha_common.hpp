/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::fmha::collective {

using namespace cute;

template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto get_local_tile_tensor(const MTensor& m_tensor, const Shape& tile_shape,
                                          int head_idx, int offset, int seq_len) {
  // (N, D, H)
  auto g_offset = local_tile(m_tensor(_, _, head_idx), cute::make_shape(1, get<1>(tile_shape)),
                             make_coord(offset, _0{}));
  auto g_sequence =
      make_tensor(g_offset.data(),
                  make_layout(cute::make_shape(seq_len, get<1>(tile_shape)), g_offset.stride()));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
  return g_tensor;
}

template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto get_local_tile_t_tensor(const MTensor& m_tensor, const Shape& tile_shape,
                                            int head_idx, int offset, int seq_len) {
  // (D, N, H)
  auto g_offset = local_tile(m_tensor(_, _, head_idx), cute::make_shape(get<0>(tile_shape), 1),
                             make_coord(_0{}, offset));
  auto g_sequence =
      make_tensor(g_offset.data(),
                  make_layout(cute::make_shape(get<0>(tile_shape), seq_len), g_offset.stride()));
  auto g_tensor = local_tile(g_offset, tile_shape, make_coord(_0{}, _));
  return g_tensor;
}

template <typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_reset_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  constexpr int rA = decltype(rank(tA))::value;
  constexpr int rB = decltype(rank(tB))::value;
  constexpr int rC = decltype(rank(tC))::value;
  static_assert(rA == 3 && rB == 3 && rC == 3);

  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tA); k_block++) {
    cute::gemm(atom, tA(_, _, k_block), tB(_, _, k_block), tC);
    atom.accumulate_ = decltype(atom.accumulate_)::One;
  }
}

template <typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  atom.accumulate_ = decltype(atom.accumulate_)::Zero;
  gemm_reset_zero_acc(atom, tA, tB, tC);
}

template <class Layout, class Stages = _1>
CUTE_DEVICE constexpr auto unstageSmemLayout(Layout const& layout, Stages stages = {}) {
  return composition(layout, prepend<decltype(rank(layout))::value>(make_layout(stages), _));
}

template <class T>
CUTE_DEVICE T warp_uniform(T a) {
  return __shfl_sync(0xffffffff, a, 0);
}

template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr auto to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<MMA_Traits<SM100_MMA_F8F6F4_SS, a_type, b_type, c_type, cute::C<M>,
                                 cute::C<N>, cute::integral_constant<UMMA::Major, a_major>,
                                 cute::integral_constant<UMMA::Major, b_major>,
                                 cute::integral_constant<UMMA::ScaleIn, a_neg>,
                                 cute::integral_constant<UMMA::ScaleIn, b_neg>>,
                      TAs...>,
             TMs...>) {
  return TiledMMA<
      MMA_Atom<MMA_Traits<SM100_MMA_F8F6F4_TS<a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
                                              b_neg, UMMA::Saturate::False>>,
               TAs...>,
      TMs...>{};
}

template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg, class... TAs, class... TMs>
CUTE_HOST_DEVICE constexpr auto to_tiled_mma_sm100_ts(
    TiledMMA<
        MMA_Atom<SM100_MMA_F16BF16_SS<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>,
                 TAs...>,
        TMs...>) {
  return TiledMMA<MMA_Atom<SM100_MMA_F16BF16_TS<a_type, b_type, c_type, M, N, a_major, b_major,
                                                a_neg, b_neg, UMMA::Saturate::False>,
                           TAs...>,
                  TMs...>{};
}

template <uint32_t RegCount>
CUTLASS_DEVICE void warpgroup_reg_set() {
  if constexpr (RegCount < 128) {
    cutlass::arch::warpgroup_reg_dealloc<RegCount>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<RegCount>();
  }
}

// Emulated exp2 using degree-3 polynomial approximation (ported from FA4's e2e_asm2).
// Uses f32x2 packed PTX instructions to halve instruction count vs scalar version.
// Reduces SFU pressure by replacing hardware exp2f with FMA-based computation.
__device__ __forceinline__ void exp2_emulated_2(float& x, float& y) {
  asm(
    "{\n\t"
    ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
    ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
    ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
    // Clamp to [-127, +inf)
    "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"       // f1 = max(x, -127.0)
    "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"       // f2 = max(y, -127.0)
    "mov.b64 l1, {f1, f2};\n\t"                  // pack into f32x2
    // Bias for floor extraction: 2^23 + 2^22 = 12582912.0
    "mov.f32 f3, 0f4B400000;\n\t"
    "mov.b64 l2, {f3, f3};\n\t"
    // Round-down add to extract floor(x) in lower bits
    "add.rm.ftz.f32x2 l7, l1, l2;\n\t"           // rounded = clamped + bias
    // Fractional part: frac = clamped - (rounded - bias)
    "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"           // rounded_back = rounded - bias
    "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"           // frac = clamped - rounded_back
    // Degree-3 polynomial coefficients (minimax on [0,1))
    "mov.f32 f7, 0f3D9DF09D;\n\t"                // c3 = 0.077119...
    "mov.b64 l6, {f7, f7};\n\t"
    "mov.f32 f6, 0f3E6906A4;\n\t"                // c2 = 0.227564...
    "mov.b64 l5, {f6, f6};\n\t"
    "mov.f32 f5, 0f3F31F519;\n\t"                // c1 = 0.695146...
    "mov.b64 l4, {f5, f5};\n\t"
    "mov.f32 f4, 0f3F800000;\n\t"                // c0 = 1.0
    "mov.b64 l3, {f4, f4};\n\t"
    // Horner evaluation: ((c3*f + c2)*f + c1)*f + c0
    "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"     // t = frac * c3 + c2
    "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"    // t = t * frac + c1
    "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"    // t = t * frac + c0
    // Combine integer exponent with polynomial mantissa
    "mov.b64 {r1, r2}, l7;\n\t"                  // unpack floor values
    "mov.b64 {r3, r4}, l10;\n\t"                 // unpack polynomial results
    "shl.b32 r5, r1, 23;\n\t"                    // shift floor to IEEE 754 exponent
    "add.s32 r7, r5, r3;\n\t"                    // result_x = 2^floor * poly
    "shl.b32 r6, r2, 23;\n\t"
    "add.s32 r8, r6, r4;\n\t"                    // result_y = 2^floor * poly
    "mov.b32 %0, r7;\n\t"
    "mov.b32 %1, r8;\n\t"
    "}\n"
    : "=f"(x), "=f"(y)
    : "f"(x), "f"(y)
  );
}

}  // namespace cutlass::fmha::collective
