// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once

#include "sm100/common/prefill_attention_config.cuh"
#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1
#include "sm100/common/nvfp4_to_bf16.cuh"
#endif

#include <cmath>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

namespace fmha_sm100::prefill_bf16x {
namespace detail {

using namespace cute;

constexpr float kLog2Fp8ProbabilityScale = 8.807354922057604f;
// The E4M3 path scales P by 2^kLog2Fp8ProbabilityScale (448) so it uses the E4M3 range, and takes
// it back out of the LSE. BF16 P needs no range scaling: the BF16 path uses 0 (and skips the LSE
// subtraction altogether).
constexpr float kLog2ProbabilityScale = kExactPath ? 0.0f : kLog2Fp8ProbabilityScale;

CUTLASS_DEVICE int decode_q_idx(int packed_qsplit) { return packed_qsplit & 0x00ffffff; }

CUTLASS_DEVICE int decode_split_idx(int packed_qsplit) {
  return static_cast<unsigned>(packed_qsplit) >> 24;
}

CUTLASS_DEVICE int real_col_to_stg128_half_fake_col(int col) {
  int const tile = col / 32;
  int const col32 = col % 32;
  int const lane = (col32 % 8) / 2;
  int const group = col32 / 8;
  int const element = col32 % 2;
  return tile * 32 + lane * 8 + group * 2 + element;
}

// Pure functions of their operands: plain (non-volatile) asm lets ptxas schedule the exp loop's
// independent chains across each other instead of in program order.
CUTLASS_DEVICE float rcp_approx_ftz(float value) {
  float result;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

CUTLASS_DEVICE float exp2_approx_ftz(float value) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

CUTLASS_DEVICE float log2_approx_ftz(float value) {
  float result;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

CUTLASS_DEVICE float2 add_packed_f32x2(float2 lhs, float2 rhs) {
  float2 result;
  asm("add.rn.ftz.f32x2 %0, %1, %2;"
               : "=l"(reinterpret_cast<uint64_t &>(result))
               : "l"(reinterpret_cast<uint64_t const &>(lhs)),
                 "l"(reinterpret_cast<uint64_t const &>(rhs)));
  return result;
}

CUTLASS_DEVICE float2 fma_packed_f32x2(float2 lhs, float2 rhs, float2 addend) {
  float2 result;
  asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
               : "=l"(reinterpret_cast<uint64_t &>(result))
               : "l"(reinterpret_cast<uint64_t const &>(lhs)),
                 "l"(reinterpret_cast<uint64_t const &>(rhs)),
                 "l"(reinterpret_cast<uint64_t const &>(addend)));
  return result;
}

// BF16X M3: the reference softmax's exact ops (BF16X_M0_DSL_MATH_TRACE_v1 + SASS confirm): packed f32x2 ops are .rn WITHOUT
// ftz; the poly exp2 is utils.ex2_emulation_2(poly_degree=3) op for op.
CUTLASS_DEVICE float2 add_packed_f32x2_rn(float2 lhs, float2 rhs) {
  float2 result;
  asm("add.rn.f32x2 %0, %1, %2;"
      : "=l"(reinterpret_cast<uint64_t &>(result))
      : "l"(reinterpret_cast<uint64_t const &>(lhs)), "l"(reinterpret_cast<uint64_t const &>(rhs)));
  return result;
}

CUTLASS_DEVICE float2 fma_packed_f32x2_rn(float2 lhs, float2 rhs, float2 addend) {
  float2 result;
  asm("fma.rn.f32x2 %0, %1, %2, %3;"
      : "=l"(reinterpret_cast<uint64_t &>(result))
      : "l"(reinterpret_cast<uint64_t const &>(lhs)), "l"(reinterpret_cast<uint64_t const &>(rhs)),
        "l"(reinterpret_cast<uint64_t const &>(addend)));
  return result;
}

CUTLASS_DEVICE float2 sub_packed_f32x2_rn(float2 lhs, float2 rhs) {
  float2 result;
  asm("sub.rn.f32x2 %0, %1, %2;"
      : "=l"(reinterpret_cast<uint64_t &>(result))
      : "l"(reinterpret_cast<uint64_t const &>(lhs)), "l"(reinterpret_cast<uint64_t const &>(rhs)));
  return result;
}

CUTLASS_DEVICE float2 add_packed_f32x2_rm(float2 lhs, float2 rhs) {
  float2 result;
  asm("add.rm.f32x2 %0, %1, %2;"
      : "=l"(reinterpret_cast<uint64_t &>(result))
      : "l"(reinterpret_cast<uint64_t const &>(lhs)), "l"(reinterpret_cast<uint64_t const &>(rhs)));
  return result;
}

CUTLASS_DEVICE float max_f32(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

CUTLASS_DEVICE float combine_int_frac_ex2(float x_rounded, float frac_ex2) {
  float out;
  asm("{\n\t.reg .s32 xi, fi, xe, oi;\n\tmov.b32 xi, %1;\n\tmov.b32 fi, %2;\n\tshl.b32 xe, xi, 23;\n\t"
      "add.s32 oi, xe, fi;\n\tmov.b32 %0, oi;\n\t}"
      : "=f"(out) : "f"(x_rounded), "f"(frac_ex2));
  return out;
}

// utils.ex2_emulation_2(x, y, poly_degree=3): clamp -127, floor via add.rm(2^23+2^22), frac via two sub.rn, Horner
// fma.rn.f32x2 with POLY_EX2[3] = (1.0, 0.695146143436431884765625, 0.227564394474029541015625,
// 0.077119089663028717041015625) from the highest degree down, then exponent bits.
CUTLASS_DEVICE float2 ex2_emulation_2(float2 xy) {
  float const kRound = 12582912.0f;  // 2^23 + 2^22 (0f4B400000)
  float2 const clamped = make_float2(max_f32(xy.x, -127.0f), max_f32(xy.y, -127.0f));
  float2 const rounded = add_packed_f32x2_rm(clamped, make_float2(kRound, kRound));
  float2 const rounded_back = sub_packed_f32x2_rn(rounded, make_float2(kRound, kRound));
  float2 const frac = sub_packed_f32x2_rn(clamped, rounded_back);
  float const c3 = 0.077119089663028717041015625f, c2 = 0.227564394474029541015625f;
  float const c1 = 0.695146143436431884765625f, c0 = 1.0f;
  float2 acc = make_float2(c3, c3);
  acc = fma_packed_f32x2_rn(acc, frac, make_float2(c2, c2));
  acc = fma_packed_f32x2_rn(acc, frac, make_float2(c1, c1));
  acc = fma_packed_f32x2_rn(acc, frac, make_float2(c0, c0));
  return make_float2(combine_int_frac_ex2(rounded.x, acc.x), combine_int_frac_ex2(rounded.y, acc.y));
}

template <class Tensor> CUTLASS_DEVICE auto make_16x256b_tensor_mn_view(Tensor tensor) {
  auto acc_layout = layout(tensor);
  auto acc_layout_col_major = make_layout(shape(acc_layout));
  auto acc_layout_sm90 = make_layout(
      make_shape(get<0, 0>(shape(acc_layout_col_major)), get<0, 1>(shape(acc_layout_col_major)),
                 get<1>(shape(acc_layout_col_major)), get<2>(shape(acc_layout_col_major))),
      make_stride(get<0, 0>(stride(acc_layout_col_major)), get<0, 1>(stride(acc_layout_col_major)),
                  get<1>(stride(acc_layout_col_major)), get<2>(stride(acc_layout_col_major))));
  auto converted_sm90 = composition(acc_layout, acc_layout_sm90);

  auto converted_col_major = make_layout(shape(converted_sm90));
  auto converted_mn = make_layout(
      make_shape(
          make_shape(get<0, 1>(shape(converted_col_major)), get<1>(shape(converted_col_major))),
          make_shape(get<0, 0>(shape(converted_col_major)), get<0, 2>(shape(converted_col_major)),
                     get<2>(shape(converted_col_major))),
          get<3>(shape(converted_col_major))),
      make_stride(
          make_stride(get<0, 1>(stride(converted_col_major)), get<1>(stride(converted_col_major))),
          make_stride(get<0, 0>(stride(converted_col_major)),
                      get<0, 2>(stride(converted_col_major)), get<2>(stride(converted_col_major))),
          get<3>(stride(converted_col_major))));
  return make_tensor(tensor.data(), composition(converted_sm90, converted_mn));
}

template <int Offset, class Tensor, size_t... Indices>
CUTLASS_DEVICE void tmem_store_16_impl(Tensor const &source, uint32_t destination,
                                       cute::index_sequence<Indices...>) {
  SM100_TMEM_STORE_32dp32b16x::copy(source(Offset + Indices)..., destination);
}

template <int Offset, class Tensor>
CUTLASS_DEVICE void tmem_store_16(Tensor const &source, uint32_t destination) {
  tmem_store_16_impl<Offset>(source, destination, cute::make_index_sequence<16>{});
}

template <class TiledMma, class ScoreTensor>
CUTLASS_DEVICE void issue_qk(TiledMma tiled_mma, ScoreTensor tS, auto const &tQ, auto const &tK,
                             uint64_t *done_barrier) {
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tQ); ++k_block) {
    gemm(tiled_mma, tQ(_, _, k_block), tK(_, _, k_block), tS);
    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
  }
  cutlass::arch::umma_arrive(done_barrier);
}

template <int Offset, class Tensor, size_t... Indices>
CUTLASS_DEVICE void tmem_store_32_impl(Tensor const &source, uint32_t destination,
                                       cute::index_sequence<Indices...>) {
  SM100_TMEM_STORE_32dp32b32x::copy(source(Offset + Indices)..., destination);
}

// 32 consecutive 32-bit TMEM columns of this thread's lane (BF16 P: one 64-key half).
template <int Offset, class Tensor>
CUTLASS_DEVICE void tmem_store_32(Tensor const &source, uint32_t destination) {
  tmem_store_32_impl<Offset>(source, destination, cute::make_index_sequence<32>{});
}

template <class TiledMmaQK>
CUTLASS_DEVICE void softmax_to_tmem(TiledMmaQK tiled_mma_qk, uint32_t score_address,
                                    uint32_t probability_address, float softmax_scale_log2,
                                    int visible_cols, uint64_t *p_early_full, uint64_t *p_full,
                                    float &row_sum, float &row_max) {
  Tensor tStS = partition_fragment_C(tiled_mma_qk, make_shape(Int<128>{}, Int<128>{}));
  tStS.data() = score_address;
  Tensor cS = make_identity_tensor(make_shape(Int<128>{}, Int<128>{}));
  Tensor tScS = tiled_mma_qk.get_slice(0).partition_C(cS);
  auto tiled_tmem_load = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, tStS);
  int const group_thread = static_cast<int>(threadIdx.x) & 127;
  auto thr_tmem_load = tiled_tmem_load.get_slice(group_thread);
  Tensor tStS_t2r = thr_tmem_load.partition_S(tStS);
  Tensor tScS_t2r = thr_tmem_load.partition_D(tScS);
  Tensor rS = make_tensor<Accumulator>(shape(tScS_t2r));
  static_assert(size(rS) == 128);
  copy(tiled_tmem_load, tStS_t2r, rS);
  cutlass::arch::fence_view_async_tmem_load();

  float maximum0 = -INFINITY;
  float maximum1 = -INFINITY;
  float maximum2 = -INFINITY;
  float maximum3 = -INFINITY;
  // Only the query's own page is partially visible; history pages (the common case) are read in
  // full, so the per-element select is skipped when every row of the warp sees all 128 keys. The
  // test is warp-uniform by construction (all lanes vote), so both paths are warp-uniform.
  if (__all_sync(0xffffffffu, visible_cols >= 128)) {
    CUTLASS_PRAGMA_UNROLL
    for (int col = 0; col < 128; col += 8) {
      maximum0 = fmaxf(maximum0, fmaxf(rS(col), rS(col + 1)));
      maximum1 = fmaxf(maximum1, fmaxf(rS(col + 2), rS(col + 3)));
      maximum2 = fmaxf(maximum2, fmaxf(rS(col + 4), rS(col + 5)));
      maximum3 = fmaxf(maximum3, fmaxf(rS(col + 6), rS(col + 7)));
    }
  } else {
    CUTLASS_PRAGMA_UNROLL
    for (int col = 0; col < 128; col += 8) {
      CUTLASS_PRAGMA_UNROLL
      for (int value = 0; value < 8; ++value) {
        rS(col + value) = col + value < visible_cols ? rS(col + value) : -INFINITY;
      }
      maximum0 = fmaxf(maximum0, fmaxf(rS(col), rS(col + 1)));
      maximum1 = fmaxf(maximum1, fmaxf(rS(col + 2), rS(col + 3)));
      maximum2 = fmaxf(maximum2, fmaxf(rS(col + 4), rS(col + 5)));
      maximum3 = fmaxf(maximum3, fmaxf(rS(col + 6), rS(col + 7)));
    }
  }
  float const maximum = fmaxf(fmaxf(maximum0, maximum1), fmaxf(maximum2, maximum3));

  float const maximum_safe = maximum == -INFINITY ? 0.0f : maximum;
  float2 const packed_scale = make_float2(softmax_scale_log2, softmax_scale_log2);
  // BF16X-M3: packed-op ftz flags. fma/add.rn.ftz.f32x2 flush subnormal inputs and results; the
  // exact reference softmax may need the non-ftz forms (or a different op order) here and below.
  float2 const packed_bias =
      fma_packed_f32x2(make_float2(-maximum_safe, -maximum_safe), packed_scale,
                       make_float2(kLog2ProbabilityScale, kLog2ProbabilityScale));
  if constexpr (kBf16DataPath) {
#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1
    // BF16X M3: the reference forms ms = max_safe * scale_log2 (fp32 mul) and x * scale_log2 - ms as one packed fma.
    float const ms = __fmul_rn(maximum_safe, softmax_scale_log2);
    float2 const bias_rn = make_float2(-ms, -ms);
    // BF16 P: 2 keys per 32-bit TMEM column, so each 64-key half is 32 columns. The first half
    // (keys 0-63, PV k-blocks 0-3) is published on p_early_full as soon as it is stored, the
    // second (keys 64-127) on p_full; the PV MMA waits on them in that order.
    namespace common = ::fmha_sm100::prefill_bf16x::sm100::common;
    // BF16X-M3: row_sum association. Four packed partial sums (keys 8c + {0,1}, {2,3}, {4,5},
    // {6,7}), reduced as (sum0 + sum1) + (sum2 + sum3), then .x + .y.
    float2 sum0 = make_float2(0.0f, 0.0f);
    float2 sum1 = make_float2(0.0f, 0.0f);
    float2 sum2 = make_float2(0.0f, 0.0f);
    float2 sum3 = make_float2(0.0f, 0.0f);
    Tensor rP = make_tensor<uint32_t>(make_shape(Int<32>{}));
    CUTLASS_PRAGMA_UNROLL
    for (int half = 0; half < 2; ++half) {
      CUTLASS_PRAGMA_UNROLL
      for (int chunk = 0; chunk < 8; ++chunk) {
        int const col = half * 64 + chunk * 8;
        // BF16X M3: elements 44-47, 60-63, 76-79, 92-95 (fragments 1-2, k % 16 >= 12 of the reference's
        // ex2_emu_freq 16 / res 4 / start_frg 1, causal) use the poly exp2; all others MUFU ex2.approx.ftz.
        bool const poly_chunk = (col == 40) || (col == 56) || (col == 72) || (col == 88);
        float2 p0 = fma_packed_f32x2_rn(make_float2(rS(col), rS(col + 1)), packed_scale, bias_rn);
        p0.x = exp2_approx_ftz(p0.x);
        p0.y = exp2_approx_ftz(p0.y);
        sum0 = add_packed_f32x2_rn(sum0, p0);

        float2 p1 = fma_packed_f32x2_rn(make_float2(rS(col + 2), rS(col + 3)), packed_scale, bias_rn);
        p1.x = exp2_approx_ftz(p1.x);
        p1.y = exp2_approx_ftz(p1.y);
        sum1 = add_packed_f32x2_rn(sum1, p1);

        float2 p2 = fma_packed_f32x2_rn(make_float2(rS(col + 4), rS(col + 5)), packed_scale, bias_rn);
        if (poly_chunk) {
          p2 = ex2_emulation_2(p2);
        } else {
          p2.x = exp2_approx_ftz(p2.x);
          p2.y = exp2_approx_ftz(p2.y);
        }
        sum2 = add_packed_f32x2_rn(sum2, p2);

        float2 p3 = fma_packed_f32x2_rn(make_float2(rS(col + 6), rS(col + 7)), packed_scale, bias_rn);
        if (poly_chunk) {
          p3 = ex2_emulation_2(p3);
        } else {
          p3.x = exp2_approx_ftz(p3.x);
          p3.y = exp2_approx_ftz(p3.y);
        }
        sum3 = add_packed_f32x2_rn(sum3, p3);

        // cvt.rn.bf16x2.f32, even key in the low half (TMEM A operand order).
        rP(chunk * 4) = common::pack_bf16x2_rn(p0.x, p0.y);
        rP(chunk * 4 + 1) = common::pack_bf16x2_rn(p1.x, p1.y);
        rP(chunk * 4 + 2) = common::pack_bf16x2_rn(p2.x, p2.y);
        rP(chunk * 4 + 3) = common::pack_bf16x2_rn(p3.x, p3.y);
      }
      tmem_store_32<0>(rP, probability_address + half * 32);
      cutlass::arch::fence_view_async_tmem_store();
      cutlass::arch::ClusterBarrier::arrive(half == 0 ? p_early_full : p_full);
    }

    sum0 = add_packed_f32x2_rn(sum0, sum1);
    sum2 = add_packed_f32x2_rn(sum2, sum3);
    sum0 = add_packed_f32x2_rn(sum0, sum2);
    row_sum = __fadd_rn(sum0.x, sum0.y);
    row_max = maximum;
#endif
    return;
  }
#if FMHA_SM100_PREFILL_BF16X_DTYPE == 0 && FMHA_SM100_PREFILL_BF16X_E4M3X == 1
  // E4M3X: the CuTe-DSL fp8 softmax. Same ops as the BF16X M3 path (ms = max_safe x scale_log2 in fp32, then the
  // packed non-ftz fma; the causal poly exp2 on elements 44-47 / 60-63 / 76-79 / 92-95; MUFU elsewhere; the row
  // sum over the fp32 exps in the 4-accumulator order with non-ftz adds), then P = cvt.rn.satfinite.e4m3 of the
  // UNSCALED exp (no 448 range factor), byte i = key i.
  {
    float const ms = __fmul_rn(maximum_safe, softmax_scale_log2);
    float2 const bias_rn = make_float2(-ms, -ms);
    float2 sum0 = make_float2(0.0f, 0.0f);
    float2 sum1 = make_float2(0.0f, 0.0f);
    float2 sum2 = make_float2(0.0f, 0.0f);
    float2 sum3 = make_float2(0.0f, 0.0f);
    Tensor rP = make_tensor<uint32_t>(make_shape(Int<32>{}));
    auto pack_e4m3x4 = [](float2 lo, float2 hi) -> uint32_t {
      uint16_t a;
      uint16_t b;
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(a) : "f"(lo.y), "f"(lo.x));
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(b) : "f"(hi.y), "f"(hi.x));
      return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 16);
    };
    CUTLASS_PRAGMA_UNROLL
    for (int chunk = 0; chunk < 16; ++chunk) {
      int const col = chunk * 8;
      bool const poly_chunk = (col == 40) || (col == 56) || (col == 72) || (col == 88);
      float2 p0 = fma_packed_f32x2_rn(make_float2(rS(col), rS(col + 1)), packed_scale, bias_rn);
      p0.x = exp2_approx_ftz(p0.x);
      p0.y = exp2_approx_ftz(p0.y);
      sum0 = add_packed_f32x2_rn(sum0, p0);
      float2 p1 = fma_packed_f32x2_rn(make_float2(rS(col + 2), rS(col + 3)), packed_scale, bias_rn);
      p1.x = exp2_approx_ftz(p1.x);
      p1.y = exp2_approx_ftz(p1.y);
      sum1 = add_packed_f32x2_rn(sum1, p1);
      float2 p2 = fma_packed_f32x2_rn(make_float2(rS(col + 4), rS(col + 5)), packed_scale, bias_rn);
      if (poly_chunk) {
        p2 = ex2_emulation_2(p2);
      } else {
        p2.x = exp2_approx_ftz(p2.x);
        p2.y = exp2_approx_ftz(p2.y);
      }
      sum2 = add_packed_f32x2_rn(sum2, p2);
      float2 p3 = fma_packed_f32x2_rn(make_float2(rS(col + 6), rS(col + 7)), packed_scale, bias_rn);
      if (poly_chunk) {
        p3 = ex2_emulation_2(p3);
      } else {
        p3.x = exp2_approx_ftz(p3.x);
        p3.y = exp2_approx_ftz(p3.y);
      }
      sum3 = add_packed_f32x2_rn(sum3, p3);
      rP(chunk * 2) = pack_e4m3x4(p0, p1);
      rP(chunk * 2 + 1) = pack_e4m3x4(p2, p3);
    }
    sum0 = add_packed_f32x2_rn(sum0, sum1);
    sum2 = add_packed_f32x2_rn(sum2, sum3);
    sum0 = add_packed_f32x2_rn(sum0, sum2);
    float const sum = __fadd_rn(sum0.x, sum0.y);

    tmem_store_16<0>(rP, probability_address);
    cutlass::arch::fence_view_async_tmem_store();
    cutlass::arch::ClusterBarrier::arrive(p_early_full);

    tmem_store_16<16>(rP, probability_address + 16);
    cutlass::arch::fence_view_async_tmem_store();
    cutlass::arch::ClusterBarrier::arrive(p_full);

    row_sum = sum;
    row_max = maximum;
  }
#elif FMHA_SM100_PREFILL_BF16X_DTYPE == 0
  float2 sum0 = make_float2(0.0f, 0.0f);
  float2 sum1 = make_float2(0.0f, 0.0f);
  float2 sum2 = make_float2(0.0f, 0.0f);
  float2 sum3 = make_float2(0.0f, 0.0f);
  Tensor rP = make_tensor<uint32_t>(make_shape(Int<32>{}));
  cutlass::NumericArrayConverter<Element, float, 4> convert;
  CUTLASS_PRAGMA_UNROLL
  for (int chunk = 0; chunk < 16; ++chunk) {
    int const col = chunk * 8;
    float2 probability =
        fma_packed_f32x2(make_float2(rS(col), rS(col + 1)), packed_scale, packed_bias);
    probability.x = exp2_approx_ftz(probability.x);
    probability.y = exp2_approx_ftz(probability.y);
    rS(col) = probability.x;
    rS(col + 1) = probability.y;
    sum0 = add_packed_f32x2(sum0, probability);

    probability =
        fma_packed_f32x2(make_float2(rS(col + 2), rS(col + 3)), packed_scale, packed_bias);
    probability.x = exp2_approx_ftz(probability.x);
    probability.y = exp2_approx_ftz(probability.y);
    rS(col + 2) = probability.x;
    rS(col + 3) = probability.y;
    sum1 = add_packed_f32x2(sum1, probability);

    probability =
        fma_packed_f32x2(make_float2(rS(col + 4), rS(col + 5)), packed_scale, packed_bias);
    probability.x = exp2_approx_ftz(probability.x);
    probability.y = exp2_approx_ftz(probability.y);
    rS(col + 4) = probability.x;
    rS(col + 5) = probability.y;
    sum2 = add_packed_f32x2(sum2, probability);

    probability =
        fma_packed_f32x2(make_float2(rS(col + 6), rS(col + 7)), packed_scale, packed_bias);
    probability.x = exp2_approx_ftz(probability.x);
    probability.y = exp2_approx_ftz(probability.y);
    rS(col + 6) = probability.x;
    rS(col + 7) = probability.y;
    sum3 = add_packed_f32x2(sum3, probability);

    cutlass::Array<float, 4> input0;
    input0[0] = rS(col);
    input0[1] = rS(col + 1);
    input0[2] = rS(col + 2);
    input0[3] = rS(col + 3);
    cutlass::Array<float, 4> input1;
    input1[0] = rS(col + 4);
    input1[1] = rS(col + 5);
    input1[2] = rS(col + 6);
    input1[3] = rS(col + 7);
    cutlass::Array<Element, 4> const packed0 = convert(input0);
    cutlass::Array<Element, 4> const packed1 = convert(input1);
    rP(chunk * 2) = reinterpret_cast<uint32_t const &>(packed0);
    rP(chunk * 2 + 1) = reinterpret_cast<uint32_t const &>(packed1);
  }

  sum0 = add_packed_f32x2(sum0, sum1);
  sum2 = add_packed_f32x2(sum2, sum3);
  sum0 = add_packed_f32x2(sum0, sum2);
  float const sum = sum0.x + sum0.y;

  tmem_store_16<0>(rP, probability_address);
  cutlass::arch::fence_view_async_tmem_store();
  cutlass::arch::ClusterBarrier::arrive(p_early_full);

  tmem_store_16<16>(rP, probability_address + 16);
  cutlass::arch::fence_view_async_tmem_store();
  cutlass::arch::ClusterBarrier::arrive(p_full);

  row_sum = sum;
  row_max = maximum;
#endif
}

// Thread-level views of one 64-column half of the O accumulator: the tiled TMEM copy, its source
// partition and the matching coordinate partition.
template <int ColumnPass, class TiledMmaPV, class OutputTensor, class CoordinateTensor>
CUTLASS_DEVICE auto partial_output_pass_views(TiledMmaPV tiled_mma_pv, OutputTensor tO,
                                              CoordinateTensor cO, int group_thread) {
  constexpr int kColumnPassSize = 64;
  auto thr_mma = tiled_mma_pv.get_slice(0);
  Tensor tOcO = thr_mma.partition_C(cO);
  Tensor tOcOPass =
      logical_divide(tOcO, make_layout(make_shape(Int<128>{}, Int<kColumnPassSize>{})));
  Tensor tOPass = tO;
  tOPass.data() = tO.data() + ColumnPass * kColumnPassSize;
  Tensor tOPassDivided =
      logical_divide(tOPass, make_layout(make_shape(Int<128>{}, Int<kColumnPassSize>{})));
  auto tiled_tmem_load =
      make_tmem_copy(SM100_TMEM_LOAD_16dp256b8x{}, tOPassDivided(make_coord(_, _), _0{}));
  auto thread_tmem_load = tiled_tmem_load.get_slice(group_thread);
  Tensor tTMEMtOAll = thread_tmem_load.partition_S(tOPassDivided(make_coord(_, _), _));
  Tensor tTMEMcOAll = thread_tmem_load.partition_D(tOcOPass(make_coord(_, _), _));
  return cute::make_tuple(tiled_tmem_load, tTMEMtOAll(_, _, _, _0{}), tTMEMcOAll(_, _, _, _0{}));
}

// One thread's slice of the partial output: the 16dp256b TMEM load gives each thread 2 rows
// (two query tokens of the same head) x 32 columns per 64-column pass. The row scale, the output
// row pointer and the validity test are per row, so they are computed once per row and the 32
// columns stream out as four 16-byte streaming stores in the STG.128 "fake" column order the
// combine undoes. Plain C++ arithmetic keeps the compiler free to interleave the rows and column
// groups; the rounding (rcp.approx, fp32 multiply, round-to-nearest bf16) is unchanged.
// Per-thread view of the O accumulator: the 16dp256b TMEM load gives each thread, per 64-column
// pass, two "row indices" x four 8-value chunks that cover four accumulator rows (A, A+8, A+16,
// A+24 for A = 32 * warp + lane / 4) x two column offsets (c, c+32). A row is one (token, head)
// of the group, so its output pointer, scale and validity are computed once per group and reused
// by the four 16-byte chunks the thread stores for it (two per pass).
struct PartialRowState {
  cutlass::bfloat16_t *ptr;
  float scale;
  bool valid;
};

template <int ColumnPass, class RegTensor, class CoordinateTensor>
CUTLASS_DEVICE void store_partial_output_pass(RegTensor const &rOPass,
                                              CoordinateTensor const &tTMEMcOPass,
                                              PartialRowState const (&rows)[2][2]) {
  constexpr int kColumnPassSize = 64;
  Tensor rOMnFull = make_16x256b_tensor_mn_view(rOPass);
  Tensor cOMnFull = make_16x256b_tensor_mn_view(tTMEMcOPass);
  Tensor rOMn = rOMnFull(_, _, _0{});
  Tensor cOMn = cOMnFull(_, _, _0{});
  static_assert(rank(rOMn) == 2 && size<0>(rOMn) == 2 && size<1>(rOMn) == 32);

  CUTLASS_PRAGMA_UNROLL
  for (int row_index = 0; row_index < 2; ++row_index) {
    CUTLASS_PRAGMA_UNROLL
    for (int column_group = 0; column_group < 4; ++column_group) {
      int const column_base = column_group * 8;
      PartialRowState const &state = rows[row_index][column_group / 2];
      int const col = get<1>(cOMn(row_index, column_base)) + ColumnPass * kColumnPassSize;
      int const fake_col = real_col_to_stg128_half_fake_col(col);
      uint32_t packed[4];
      CUTLASS_PRAGMA_UNROLL
      for (int pair = 0; pair < 4; ++pair) {
        // BF16X M3: the reference's o x rcp is mul.rn.f32x2 (no ftz): explicit __fmul_rn on the BF16 path.
        float2 const value = kExactPath
            ? make_float2(__fmul_rn(rOMn(row_index, column_base + 2 * pair), state.scale),
                          __fmul_rn(rOMn(row_index, column_base + 2 * pair + 1), state.scale))
            : make_float2(rOMn(row_index, column_base + 2 * pair) * state.scale,
                          rOMn(row_index, column_base + 2 * pair + 1) * state.scale);
        __nv_bfloat162 const bf16_pair = __float22bfloat162_rn(value);
        packed[pair] = *reinterpret_cast<uint32_t const *>(&bf16_pair);
      }
      if (state.valid) {
        __stcs(reinterpret_cast<uint4 *>(state.ptr + fake_col),
               make_uint4(packed[0], packed[1], packed[2], packed[3]));
      }
    }
  }
}

template <class TiledMmaPV, class OutputTensor, class CoordinateTensor>
CUTLASS_DEVICE void
store_partial_output(TiledMmaPV tiled_mma_pv, OutputTensor tO, CoordinateTensor cO,
                     uint64_t *output_empty, uint64_t *stats_empty,
                     cutlass::bfloat16_t *o_partial_ptr, int64_t o_partial_split_stride,
                     float *lse_partial_ptr, int total_q, int num_q_heads,
                     float softmax_scale_log2, float output_scale, int head_kv,
                     int q_count, int q_batch_offset, int group, int group_thread,
                     int const *qsplit_indices, float const *row_sum, float const *row_max) {
  int const lane = group_thread & 31;
  int const output_token_base = (group_thread / 32) * 2;
  int output_qsplit0 = 0;
  int output_qsplit1 = 0;
  if (lane == 0) {
    output_qsplit0 = qsplit_indices[output_token_base];
    output_qsplit1 = qsplit_indices[output_token_base + 1];
  }
  output_qsplit0 = __shfl_sync(0xffffffffu, output_qsplit0, 0);
  output_qsplit1 = __shfl_sync(0xffffffffu, output_qsplit1, 0);

  auto views0 = partial_output_pass_views<0>(tiled_mma_pv, tO, cO, group_thread);
  auto views1 = partial_output_pass_views<1>(tiled_mma_pv, tO, cO, group_thread);

  // The four rows this thread stores (see store_partial_output_pass): rows[row_index][half]
  // is the row of column groups 2 * half and 2 * half + 1 of that row index. This thread's LSE
  // row stats are read here too, before the stats stage is handed back to the softmax.
  Tensor cOMn = make_16x256b_tensor_mn_view(get<2>(views0))(_, _, _0{});
  float const lse_sum = row_sum[group_thread];
  float const lse_max = row_max[group_thread];
  PartialRowState rows[2][2];
  CUTLASS_PRAGMA_UNROLL
  for (int row_index = 0; row_index < 2; ++row_index) {
    CUTLASS_PRAGMA_UNROLL
    for (int half = 0; half < 2; ++half) {
      int const row = get<0>(cOMn(row_index, half * 16));
      int const token = row / kQHeadsPerKv;
      int const qi = group * kQueriesPerGroup + token;
      int const packed_qsplit = token == output_token_base ? output_qsplit0 : output_qsplit1;
      int const q_idx = decode_q_idx(packed_qsplit);
      int const split = decode_split_idx(packed_qsplit);
      int const q_abs = q_batch_offset + q_idx;
      int const head = head_kv * kQHeadsPerKv + row % kQHeadsPerKv;
      int64_t const row_offset =
          static_cast<int64_t>(split) * o_partial_split_stride +
          (static_cast<int64_t>(q_abs) * num_q_heads + head) * kHeadDim;
      float const sum = row_sum[row];
      rows[row_index][half].ptr = o_partial_ptr + row_offset;
      if constexpr (kExactPath) {
        // No output scale in the BF16 forward: O = acc x rcp.approx.ftz(row_sum), with
        // rcp(1) for an empty (0) or NaN sum.
        // BF16X-M3: epilogue rounding (rcp.approx, the fp32 multiply and the RN bf16 store).
        float const divisor = (sum == 0.0f || sum != sum) ? 1.0f : sum;
        rows[row_index][half].scale = rcp_approx_ftz(divisor);
      } else {
        rows[row_index][half].scale = sum > 0.0f ? rcp_approx_ftz(sum) * output_scale : 0.0f;
      }
      rows[row_index][half].valid = qi < q_count;
    }
  }
  // Every stat this thread needs is in registers: the softmax may publish the next group's.
  cutlass::arch::ClusterBarrier::arrive(stats_empty);

  // One 64-column half at a time: the epilogue register budget (96) holds one half's 32
  // accumulators plus the row state, not both halves. The O stage is handed back to the MMA as
  // soon as its second half is in registers, before this group's stores drain.
  {
    Tensor rO0 = make_tensor<Accumulator>(shape(get<2>(views0)));
    copy(get<0>(views0), get<1>(views0), rO0);
    cutlass::arch::fence_view_async_tmem_load();
    store_partial_output_pass<0>(rO0, get<2>(views0), rows);
  }
  {
    Tensor rO1 = make_tensor<Accumulator>(shape(get<2>(views1)));
    copy(get<0>(views1), get<1>(views1), rO1);
    cutlass::arch::fence_view_async_tmem_load();
    cutlass::arch::ClusterBarrier::arrive(output_empty);
    store_partial_output_pass<1>(rO1, get<2>(views1), rows);
  }

  int const row = group_thread;
  int const token = row / kQHeadsPerKv;
  int const qi = group * kQueriesPerGroup + token;
  if (qi < q_count) {
    int const packed_qsplit = token == output_token_base ? output_qsplit0 : output_qsplit1;
    int const q_idx = decode_q_idx(packed_qsplit);
    int const split = decode_split_idx(packed_qsplit);
    int const q_abs = q_batch_offset + q_idx;
    int const lse_head = head_kv * kQHeadsPerKv + row % kQHeadsPerKv;
    float lse = -INFINITY;
    if (lse_sum > 0.0f && isfinite(lse_max)) {
      constexpr float kLn2 = 0.6931471805599453f;
      if constexpr (kExactPath) {
        // BF16X-M3: LSE fma. Left to the compiler (contracted to fma(max, scale, lg2(sum)) or
        // not); the exact reference fixes the op order and rounding.
        // BF16X M3: the reference's contracted fma(row_max_raw, scale_log2, lg2(sum)) then x ln2 (PTX fma.rn.f32 + mul).
        lse = __fmul_rn(__fmaf_rn(lse_max, softmax_scale_log2, log2_approx_ftz(lse_sum)), kLn2);
      } else {
        lse = (lse_max * softmax_scale_log2 + log2_approx_ftz(lse_sum) -
               kLog2Fp8ProbabilityScale) *
              kLn2;
      }
    }
    int64_t const lse_offset =
        (static_cast<int64_t>(split) * total_q + q_abs) * num_q_heads + lse_head;
    lse_partial_ptr[lse_offset] = lse;
  }
}

} // namespace detail
} // namespace fmha_sm100::prefill_bf16x
