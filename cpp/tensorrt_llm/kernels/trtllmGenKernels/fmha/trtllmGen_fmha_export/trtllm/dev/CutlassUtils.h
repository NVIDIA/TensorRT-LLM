/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Conversion functions.

////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to accept Array<float, 2> inputs.
inline __device__ cutlass::Array<float, 2> fadd2(cutlass::Array<float, 2> inA,
                                                 cutlass::Array<float, 2> inB) {
  cutlass::Array<float, 2> output;
  cute::add(reinterpret_cast<float2&>(output),
            reinterpret_cast<float2&>(inA),
            reinterpret_cast<float2&>(inB));
  return output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to accept Array<float, 4> inputs.
inline __device__ cutlass::Array<float, 4> fadd4(cutlass::Array<float, 4> inA,
                                                 cutlass::Array<float, 4> inB) {
  // Extract the lower (Lo) and higher (Hi) parts of the input arrays.
  auto aLo = cutlass::make_Array(inA[0], inA[1]);
  auto bLo = cutlass::make_Array(inB[0], inB[1]);
  auto aHi = cutlass::make_Array(inA[2], inA[3]);
  auto bHi = cutlass::make_Array(inB[2], inB[3]);

  // Multiply pairs of elements (using the vectorized FFMA2 inst).
  auto cLo = fadd2(aLo, bLo);
  auto cHigh = fadd2(aHi, bHi);

  // Repack the 4 outputs into a single array of 4 floats.
  cutlass::Array<float, 4> result = cutlass::make_Array(cLo[0], cLo[1], cHigh[0], cHigh[1]);
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to accept Array<float, 2> inputs.
inline __device__ cutlass::Array<float, 2> ffma2(cutlass::Array<float, 2> inA,
                                                 cutlass::Array<float, 2> inB,
                                                 cutlass::Array<float, 2> inC) {
  cutlass::Array<float, 2> output;
  cute::fma(reinterpret_cast<float2&>(output),
            reinterpret_cast<float2&>(inA),
            reinterpret_cast<float2&>(inB),
            reinterpret_cast<float2&>(inC));
  return output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to accept Array<float, 1> inputs.
inline __device__ cutlass::Array<float, 1> fmul(cutlass::Array<float, 1> inA,
                                                cutlass::Array<float, 1> inB) {
  cutlass::Array<float, 1> output = {inA[0] * inB[0]};
  return output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to accept Array<float, 2> inputs.
inline __device__ cutlass::Array<float, 2> fmul2(cutlass::Array<float, 2> inA,
                                                 cutlass::Array<float, 2> inB) {
  cutlass::Array<float, 2> output;
  cute::mul(reinterpret_cast<float2&>(output),
            reinterpret_cast<float2&>(inA),
            reinterpret_cast<float2&>(inB));
  return output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to accept Array<float, 4> inputs.
inline __device__ cutlass::Array<float, 4> fmul4(cutlass::Array<float, 4> inA,
                                                 cutlass::Array<float, 4> inB) {
  // Extract the lower (Lo) and higher (Hi) parts of the input arrays.
  auto aLo = cutlass::make_Array(inA[0], inA[1]);
  auto bLo = cutlass::make_Array(inB[0], inB[1]);
  auto aHi = cutlass::make_Array(inA[2], inA[3]);
  auto bHi = cutlass::make_Array(inB[2], inB[3]);

  // Multiply pairs of elements (using the vectorized FFMA2 inst).
  auto cLo = fmul2(aLo, bLo);
  auto cHigh = fmul2(aHi, bHi);

  // Repack the 4 outputs into a single array of 4 floats.
  cutlass::Array<float, 4> result = cutlass::make_Array(cLo[0], cLo[1], cHigh[0], cHigh[1]);
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to accept Array<float, x> inputs.
template <int x>
inline __device__ cutlass::Array<float, x> fmulx(cutlass::Array<float, x> inA,
                                                 cutlass::Array<float, x> inB) {
  cutlass::Array<float, x> result;
#pragma unroll
  for (int i = 0; i < x; i += 2) {
    auto a = cutlass::make_Array(inA[i + 0], inA[i + 1]);
    auto b = cutlass::make_Array(inB[i + 0], inB[i + 1]);
    auto c = fmul2(a, b);
    result[i + 0] = c[0];
    result[i + 1] = c[1];
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Type converters.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DstType, typename SrcType, int N>
inline __device__ cutlass::Array<DstType, N> castArray(cutlass::Array<SrcType, N> src) {
  using Converter =
    cutlass::NumericArrayConverter<DstType,
                                   SrcType,
                                   N,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  return Converter::convert(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DstType, typename T, int N>
inline __device__ DstType castFromArray(cutlass::Array<T, N> arr) {
  return reinterpret_cast<DstType const&>(arr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int N, typename SrcType>
inline __device__ cutlass::Array<T, N> castToArray(SrcType src) {
  return reinterpret_cast<cutlass::Array<T, N> const&>(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ cutlass::uint128_t convertBfloat16ToFp16(cutlass::uint128_t src) {
  auto srcArray{castToArray<cutlass::bfloat16_t, 8>(src)};
  return castFromArray<cutlass::uint128_t>(castArray<cutlass::half_t>(srcArray));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ cutlass::uint128_t convertE4m3ToBfloat16(uint64_t src) {
  auto srcArray{castToArray<cutlass::float_e4m3_t, 8>(src)};
  return castFromArray<cutlass::uint128_t>(castArray<cutlass::bfloat16_t>(srcArray));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ cutlass::uint128_t convertE4m3ToFp16(uint64_t src) {
  auto srcArray{castToArray<cutlass::float_e4m3_t, 8>(src)};
  return castFromArray<cutlass::uint128_t>(castArray<cutlass::half_t>(srcArray));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ cutlass::uint128_t convertFp16ToBfloat16(cutlass::uint128_t src) {
  auto srcArray{castToArray<cutlass::half_t, 8>(src)};
  return castFromArray<cutlass::uint128_t>(castArray<cutlass::bfloat16_t>(srcArray));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert packed e2m1 to fp16.
template <int N>
inline __device__ cutlass::Array<cutlass::half_t, N> convert_e2m1_to_fp16(
  cutlass::Array<cutlass::float_e2m1_t, N> const& src) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  if constexpr (N % 8 == 0) {
    cutlass::Array<cutlass::half_t, N> dst;
    auto ptrE2m1x8 = reinterpret_cast<const uint32_t*>(src.data());
    auto ptrFp16x2 = reinterpret_cast<uint32_t*>(dst.data());

#pragma unroll
    for (int i = 0; i < N / 8; i++) {
      uint32_t const tmpE2m1x8 = ptrE2m1x8[i];

      // Cast e2m1x8 to fp16x8.
      uint32_t tmpF16x2[4];
      asm volatile("{\n"
                   ".reg .b8 byte0, byte1, byte2, byte3;\n"
                   "mov.b32 {byte0, byte1, byte2, byte3}, %4;\n"
                   "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
                   "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
                   "cvt.rn.f16x2.e2m1x2 %2, byte2;\n"
                   "cvt.rn.f16x2.e2m1x2 %3, byte3;\n"
                   "}\n"
                   : "=r"(tmpF16x2[0]), "=r"(tmpF16x2[1]), "=r"(tmpF16x2[2]), "=r"(tmpF16x2[3])
                   : "r"(tmpE2m1x8));
#pragma unroll
      for (int j = 0; j < 4; j++) {
        ptrFp16x2[4 * i + j] = tmpF16x2[j];
      }
    }

    return dst;
  } else
#endif
  {
    // Less efficient fallback. At this time, no specialization is available in cutlass, so this is
    // using a generic and less efficient code path.
    using ConverterE2m1ToFp16 =
      cutlass::NumericArrayConverter<cutlass::half_t,
                                     cutlass::float_e2m1_t,
                                     N,
                                     cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
    return ConverterE2m1ToFp16::convert(src);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert packed ue8m0 to bfloat16.
template <int N>
inline __device__ cutlass::Array<cutlass::bfloat16_t, N> convert_ue8m0_to_bfloat16(
  cutlass::Array<cutlass::float_ue8m0_t, N> const& src) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  if constexpr (N % 4 == 0) {
    cutlass::Array<cutlass::bfloat16_t, N> dst;
    auto ptrUe8m0x4 = reinterpret_cast<const uint32_t*>(src.data());
    auto ptrBf16x2 = reinterpret_cast<uint32_t*>(dst.data());

#pragma unroll
    for (int i = 0; i < N / 4; i++) {
      uint32_t const tmpUe8m0x4 = ptrUe8m0x4[i];
      uint32_t tmpBf16x2[2];
      asm volatile("{\n"
                   ".reg .b16 lo, hi;\n"
                   "mov.b32 {lo, hi}, %2;\n"
                   "cvt.rn.bf16x2.ue8m0x2 %0, lo;\n"
                   "cvt.rn.bf16x2.ue8m0x2 %1, hi;\n"
                   "}\n"
                   : "=r"(tmpBf16x2[0]), "=r"(tmpBf16x2[1])
                   : "r"(tmpUe8m0x4));
      ptrBf16x2[2 * i] = tmpBf16x2[0];
      ptrBf16x2[2 * i + 1] = tmpBf16x2[1];
    }

    return dst;
  } else
#endif
  {
    // Fallback.
    // Note: at this time, no specialization is available in cutlass, so this is using a generic and
    // less efficient code path.
    using ConverterUe8m0ToBf16 =
      cutlass::NumericArrayConverter<cutlass::bfloat16_t,
                                     cutlass::float_ue8m0_t,
                                     N,
                                     cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
    return ConverterUe8m0ToBf16::convert(src);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 2 bfloat16 to 2 floats.
template <typename T>
inline __device__ cutlass::Array<float, 2> convert_bfloat16_to_float2(T const& src) {
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, float>,
                "T must be uint32_t or float");
  auto srcArray{castToArray<cutlass::bfloat16_t, 2>(src)};
  using Converter = cutlass::NumericArrayConverter<float, cutlass::bfloat16_t, 2>;
  return Converter::convert(srcArray);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float to bfloat16.
inline __device__ cutlass::Array<cutlass::bfloat16_t, 1> convert_float_to_bfloat16(
  cutlass::Array<float, 1> const& src) {
  using Converter =
    cutlass::NumericArrayConverter<cutlass::bfloat16_t,
                                   float,
                                   1,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  return Converter::convert(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 2 float16 to 2 floats.
template <typename T>
inline __device__ cutlass::Array<float, 2> convert_float16_to_float2(T const& src) {
  // 2xFP16 is packed as uint32_t or float.
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, float>,
                "T must be uint32_t or float");
  auto srcArray{castToArray<cutlass::half_t, 2>(src)};
  using Converter = cutlass::NumericArrayConverter<float, cutlass::half_t, 2>;
  return Converter::convert(srcArray);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 2 floats to 2 bfloat16.
inline __device__ cutlass::Array<cutlass::bfloat16_t, 2> convert_float2_to_bfloat16(
  cutlass::Array<float, 2> const& src) {
  using Converter =
    cutlass::NumericArrayConverter<cutlass::bfloat16_t,
                                   float,
                                   2,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<cutlass::bfloat16_t, 2> dst = Converter::convert(src);
  return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 2 floats to 2 bfloat16 (represented as uint32_t).
inline __device__ uint32_t convert_float2_to_bfloat16(float in0, float in1) {
  using Converter =
    cutlass::NumericArrayConverter<cutlass::bfloat16_t,
                                   float,
                                   2,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<float, 2> src = cutlass::make_Array(in0, in1);
  cutlass::Array<cutlass::bfloat16_t, 2> dst = Converter::convert(src);
  return reinterpret_cast<uint32_t const&>(dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 4 floats to 4 bfloat16.
inline __device__ cutlass::Array<cutlass::bfloat16_t, 4> convert_float4_to_bfloat16(
  cutlass::Array<float, 4> const& src) {
  using Converter =
    cutlass::NumericArrayConverter<cutlass::bfloat16_t,
                                   float,
                                   4,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<cutlass::bfloat16_t, 4> dst = Converter::convert(src);
  return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float type Tensor to bfloat type Tensor.
template <class TEngine, class TLayout>
inline __device__ auto convert_float2_to_bfloat16(cute::Tensor<TEngine, TLayout> const& in) {
  auto result =
    cute::make_tensor<uint32_t>(cute::shape(cute::recast_layout<uint16_t, uint32_t>(in.layout())));
#pragma unroll
  for (int i = 0; i < size(in); i += 2) {
    result(i / 2) = convert_float2_to_bfloat16(in(i + 0), in(i + 1));
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 2 floats to 2 e2m1.
inline __device__ cutlass::Array<cutlass::float_e2m1_t, 2> convert_float2_to_e2m1(
  cutlass::Array<float, 2> const& array) {
  using Converter =
    cutlass::NumericArrayConverter<cutlass::float_e2m1_t,
                                   float,
                                   2,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<cutlass::float_e2m1_t, 2> dst = Converter::convert(array);
  return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 4 floats to 4 e2m1.
inline __device__ cutlass::Array<cutlass::float_e2m1_t, 4> convert_float4_to_e2m1(
  cutlass::Array<float, 4> const& array) {
  using Converter =
    cutlass::NumericArrayConverter<cutlass::float_e2m1_t,
                                   float,
                                   4,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<cutlass::float_e2m1_t, 4> dst = Converter::convert(array);
  return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert <x> floats to <x> e2m1.
template <int x>
inline __device__ cutlass::Array<cutlass::float_e2m1_t, x> convert_floatx_to_e2m1(
  cutlass::Array<float, x> const& array) {
  using Converter =
    cutlass::NumericArrayConverter<cutlass::float_e2m1_t,
                                   float,
                                   x,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<cutlass::float_e2m1_t, x> dst = Converter::convert(array);
  return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float to e4m3.
inline __device__ cutlass::Array<cutlass::float_e4m3_t, 1> convert_float_to_e4m3(
  cutlass::Array<float, 1> const& in) {
  // FIXME
  using Converter =
    cutlass::NumericArrayConverter<cutlass::float_e4m3_t,
                                   float,
                                   2,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<float, 2> src = cutlass::make_Array(in[0], in[0]);
  cutlass::Array<cutlass::float_e4m3_t, 2> dst = Converter::convert(src);
  // FIXME
  cutlass::Array<cutlass::float_e4m3_t, 1> ret =
    reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 1>&>(dst);
  return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 2 floats to 2 e4m3.
inline __device__ cutlass::Array<cutlass::float_e4m3_t, 2> convert_float2_to_e4m3(
  cutlass::Array<float, 2> const& array) {
  using Converter =
    cutlass::NumericArrayConverter<cutlass::float_e4m3_t,
                                   float,
                                   2,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<cutlass::float_e4m3_t, 2> dst = Converter::convert(array);
  return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 4 floats to 4 e4m3.
inline __device__ cutlass::Array<cutlass::float_e4m3_t, 4> convert_float4_to_e4m3(
  cutlass::Array<float, 4> const& array) {
  using Converter =
    cutlass::NumericArrayConverter<cutlass::float_e4m3_t,
                                   float,
                                   4,
                                   cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<cutlass::float_e4m3_t, 4> dst = Converter::convert(array);
  return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 4 floats to 4 e4m3 (represented as uint32_t).
inline __device__ uint32_t convert_float4_to_e4m3(float in0, float in1, float in2, float in3) {
  using Converter = cutlass::detail::NumericArrayConverterPacked4Element<
    cutlass::float_e4m3_t,
    float,
    cutlass::FloatRoundStyle::round_to_nearest_satfinite>;
  cutlass::Array<float, 4> src = cutlass::make_Array(in0, in1, in2, in3);
  cutlass::Array<cutlass::float_e4m3_t, 4> dst = Converter::convert(src);
  return reinterpret_cast<uint32_t const&>(dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float type Tensor to e4m3 type Tensor.
template <class TEngine, class TLayout>
inline __device__ auto convert_float4_to_e4m3(cute::Tensor<TEngine, TLayout> const& in) {
  auto result =
    cute::make_tensor<uint32_t>(cute::shape(cute::recast_layout<uint8_t, uint32_t>(in.layout())));
#pragma unroll
  for (int i = 0; i < size(in); i += 4) {
    result(i / 4) = convert_float4_to_e4m3(in(i + 0), in(i + 1), in(i + 2), in(i + 3));
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Passthrough
template <int N, class T>
inline __device__ cutlass::Array<T, N> convert_dtype_to_dtype(cutlass::Array<T, N> const& array) {
  return array;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// clamp half2 +inf/-inf
static inline __device__ uint16_t satfiniteFp16(uint16_t h) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  uint16_t out, clamp_value;
  clamp_value = 0x7bffu;
  asm volatile("min.xorsign.abs.f16 %0, %1, %2;" : "=h"(out) : "h"(h), "h"(clamp_value));
  return out;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
  // bit representation of maximum and minimum value of half.
  uint16_t umax = 0x7bffu;
  uint16_t umin = 0xfbffu;
  uint16_t out;
  asm volatile("min.f16 %0, %1, %2;" : "=h"(out) : "h"(h), "h"(umax));
  asm volatile("max.f16 %0, %0, %1;" : "+h"(out) : "h"(umin));
  return out;
#else
  // Take the absolute value of half. It should map to |Rx| in SASS.
  uint16_t p;
  asm volatile("abs.f16 %0, %1;" : "=h"(p) : "h"(h));

  // Compute a mask for each fp16: 0xffff if +INF and 0x0000 otherwise.
  uint16_t inf = 0x7c00u;
  uint16_t mask;
  asm volatile("set.eq.u32.f16 %0, %1, %2;" : "=h"(mask) : "h"(p), "h"(inf));

  // Recreate the new value. 0x7bff is the max value for FP16.
  p = (~mask & p) | (mask & 0x7bff);

  // Simply re-add the sign and we're done.
  return p | (h & 0x8000);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// clamp half2 +inf/-inf
static inline __device__ uint32_t satfiniteFp16x2(uint32_t h2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
  uint32_t out, clamp_value;
  clamp_value = 0x7bff7bffu;
  asm volatile("min.xorsign.abs.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h2), "r"(clamp_value));
  return out;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
  // bit representation of maximum and minimum value of half2
  uint32_t umax = 0x7bff7bffu;
  uint32_t umin = 0xfbfffbffu;
  uint32_t out;
  asm volatile("min.f16x2 %0, %1, %2;" : "=r"(out) : "r"(h2), "r"(umax));
  asm volatile("max.f16x2 %0, %0, %1;" : "+r"(out) : "r"(umin));
  return out;
#else
  // Take the absolute value of h2. It should map to |Rx| in SASS.
  uint32_t p2;
  asm volatile("abs.f16x2 %0, %1;" : "=r"(p2) : "r"(h2));

  // Compute a mask for each fp16: 0xffff if +INF and 0x0000 otherwise.
  uint32_t inf2 = 0x7c007c00u;
  uint32_t mask;
  asm volatile("set.eq.u32.f16x2 %0, %1, %2;" : "=r"(mask) : "r"(p2), "r"(inf2));

  // Recreate the new value. 0x7bff is the max value for FP16.
  p2 = (~mask & p2) | (mask & 0x7bff7bff);

  // Simply re-add the sign and we're done.
  return p2 | (h2 & 0x80008000);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert floats to half.
inline __device__ cutlass::Array<cutlass::half_t, 1> convert_float_to_half(
  cutlass::Array<float, 1> const& src) {
  using Converter = cutlass::
    NumericArrayConverter<cutlass::half_t, float, 1, cutlass::FloatRoundStyle::round_to_nearest>;
  return Converter::convert(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 2 floats to 2 half.
inline __device__ cutlass::Array<cutlass::half_t, 2> convert_float2_to_half(
  cutlass::Array<float, 2> const& src) {
  using Converter = cutlass::
    NumericArrayConverter<cutlass::half_t, float, 2, cutlass::FloatRoundStyle::round_to_nearest>;
  cutlass::Array<cutlass::half_t, 2> h2 = Converter::convert(src);
  uint32_t& dst = reinterpret_cast<uint32_t&>(h2);
  dst = satfiniteFp16x2(dst);
  return reinterpret_cast<cutlass::Array<cutlass::half_t, 2>&>(dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 2 floats to 2 half (represented as uint32_t).
inline __device__ uint32_t convert_float2_to_half(float in0, float in1) {
  using Converter = cutlass::
    NumericArrayConverter<cutlass::half_t, float, 2, cutlass::FloatRoundStyle::round_to_nearest>;
  cutlass::Array<float, 2> src = cutlass::make_Array(in0, in1);
  cutlass::Array<cutlass::half_t, 2> h2 = Converter::convert(src);
  uint32_t& dst = reinterpret_cast<uint32_t&>(h2);
  return satfiniteFp16x2(dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert float type Tensor to half type Tensor.
template <class TEngine, class TLayout>
inline __device__ auto convert_float2_to_half(cute::Tensor<TEngine, TLayout> const& in) {
  auto result =
    cute::make_tensor<uint32_t>(cute::shape(cute::recast_layout<uint16_t, uint32_t>(in.layout())));
#pragma unroll
  for (int i = 0; i < size(in); i += 2) {
    result(i / 2) = convert_float2_to_half(in(i + 0), in(i + 1));
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to convert 4 floats to 4 half.
inline __device__ cutlass::Array<cutlass::half_t, 4> convert_float4_to_half(
  cutlass::Array<float, 4> const& src) {
  using Converter = cutlass::
    NumericArrayConverter<cutlass::half_t, float, 4, cutlass::FloatRoundStyle::round_to_nearest>;
  cutlass::Array<cutlass::half_t, 4> dst = Converter::convert(src);
  return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to extract bit from uint32_t.
inline __device__ uint32_t extract_bit_from_uint32(uint32_t const& in, int32_t offset) {
  return in & (1u << offset);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to extract bit from uint64_t.
inline __device__ uint32_t extract_bit_from_uint64(uint64_t const& in, int32_t offset) {
  uint32_t u32 = reinterpret_cast<uint32_t const(&)[2]>(in)[offset / 32];
  return u32 & (1u << (offset % 32));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to extract bit from uint128_t.
inline __device__ uint32_t extract_bit_from_uint128(cutlass::uint128_t const& in, int32_t offset) {
  uint32_t u32 = reinterpret_cast<uint32_t const(&)[4]>(in)[offset / 32];
  return u32 & (1u << (offset % 32));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the cluster size.
inline __device__ int32_t getClusterSize() {
  auto shape = cute::cluster_shape();
  return static_cast<int32_t>(shape.x * shape.y * shape.z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the cluster size.
inline __device__ int32_t getClusterDimX() {
  auto shape = cute::cluster_shape();
  return shape.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the cluster size.
inline __device__ int32_t getClusterDimY() {
  auto shape = cute::cluster_shape();
  return shape.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the cluster size.
inline __device__ int32_t getClusterDimZ() {
  auto shape = cute::cluster_shape();
  return shape.z;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the CTA mask.
inline __device__ uint16_t getCtaMask() {
  return uint16_t(1) << cute::block_rank_in_cluster();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the CTA mask with multicast for A.
inline __device__ uint16_t getCtaMaskMcastA() {
  // The cta_idx.y == 0 CTAs are the leaders and do the multicast loads for all other CTAs in the
  // row
  //  xooo
  //  xooo
  //  xooo
  //  xooo
  auto block_id_in_cluster = cute::block_id_in_cluster();
  auto cluster_dim = cute::cluster_shape();

  uint16_t base = 0;
  for (auto block_id_y = 0; block_id_y < cluster_dim.y; block_id_y++) {
    base |= uint16_t{1} << (cluster_dim.x * block_id_y);
  }
  return base << block_id_in_cluster.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the CTA mask with multicast for B.
inline __device__ uint16_t getCtaMaskMcastB() {
  // For the 2-CTA GEMMs, the cta_idx.x = 0,2 CTAs load the first half of B while the cta_idx.y =
  // 1,3 CTAs load the second half of B. The cta_idx.x == 0,1 CTAs are the leaders and do the
  // multicast loads for cta_idx.x = 2,3 CTAs.
  //  |x |x |x |x |
  //  | x| x| x| x|
  //  |o |o |o |o |
  //  | o| o| o| o|
  auto block_id_in_cluster = cute::block_id_in_cluster();
  auto cluster_dim = cute::cluster_shape();
  uint16_t base = 0;
  for (auto block_id_x = 0; block_id_x < cluster_dim.x; block_id_x += 2) {
    base |= uint16_t(1) << (cute::block_rank_in_cluster() + block_id_x);
  }
  return base;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the CTA mask with multicast for SfB.
inline __device__ uint16_t getCtaMaskMcastSfB() {
  // The cta_idx.x == 0 CTAs are the leaders and do the multicast loads for all other CTAs in the
  // column
  //  xxxx
  //  oooo
  //  oooo
  //  oooo
  auto block_id_in_cluster = cute::block_id_in_cluster();
  auto cluster_dim = cute::cluster_shape();
  uint16_t base = 0;
  for (auto block_id_x = 0; block_id_x < cluster_dim.x; block_id_x++) {
    base |= uint16_t(1) << block_id_x;
  }
  return base << (block_id_in_cluster.y * cluster_dim.x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the CTA rank in pair.
inline __device__ int32_t getCtaRankInPair() {
  return static_cast<int32_t>(cute::block_rank_in_cluster() & 0x1u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get if lead CTA in CGA with multicast for A.
inline __device__ uint16_t getIsLeadCtaInCgaMcastA() {
  // The cta_idx.y == 0 CTAs are the leaders and do the multicast loads for all other CTAs in the
  // row
  //  xooo
  //  xooo
  //  xooo
  //  xooo
  return cute::block_id_in_cluster().y == 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get if lead CTA in CGA with multicast for B.
inline __device__ uint16_t getIsLeadCtaInCgaMcastB() {
  // For the 2-CTA GEMMs, the cta_idx.x = 0,2 CTAs load the first half of B while the cta_idx.y =
  // 1,3 CTAs load the second half of B. The cta_idx.x == 0,1 CTAs are the leaders and do the
  // multicast loads for cta_idx.x = 2,3 CTAs.
  //  |x |x |x |x |
  //  | x| x| x| x|
  //  |o |o |o |o |
  //  | o| o| o| o|
  return cute::block_id_in_cluster().x / 2 == 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get if lead CTA in CGA with multicast for B.
inline __device__ uint16_t getIsLeadCtaInCgaMcastSfB() {
  // The cta_idx.x == 0 CTAs are the leaders and do the multicast loads for all other CTAs in the
  // column
  //  xxxx
  //  oooo
  //  oooo
  //  oooo
  return cute::block_id_in_cluster().x == 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to get the lead CTA rank.
inline __device__ int32_t getLeadCtaRank() {
  return static_cast<int32_t>(cute::block_rank_in_cluster() & 0xfffffffe);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to construct smem matrix descriptor used in wgmma
inline __device__ uint64_t make_smem_desc(uint32_t start_address,
                                          uint32_t start_address_imm,
                                          uint32_t leading_byte_offset,
                                          uint32_t stride_byte_offset,
                                          uint32_t base_offset,
                                          uint32_t layout_type) {

  cute::GmmaDescriptor desc;
  desc.bitfield.start_address_ = (start_address + start_address_imm) >> 4;
  desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
  desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
  desc.bitfield.base_offset_ = base_offset;
  desc.bitfield.layout_type_ = layout_type;
  return static_cast<uint64_t const&>(desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ uint64_t make_smem_desc(T* start_address,
                                          uint32_t start_address_imm,
                                          uint32_t leading_byte_offset,
                                          uint32_t stride_byte_offset,
                                          uint32_t base_offset,
                                          uint32_t layout_type) {

  return make_smem_desc(cute::cast_smem_ptr_to_uint(start_address),
                        start_address_imm,
                        leading_byte_offset,
                        stride_byte_offset,
                        base_offset,
                        layout_type);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Descriptor for UTC*MMA operations without block scaling.
union UtcmmaDescriptor {
  uint32_t desc_;

  struct {
    // Bitfield implementation avoids the need for shifts in assignment
    uint16_t sparse_id2_ : 2, // bit [ 0, 2) : Sparse meta data id2
      sparse_flag_ : 1,       // bit [ 2, 3) : 0 = dense. 1 = sparse.
      saturate_ : 1, // bit [ 3, 4) : 0 = no saturate. 1 = saturate. 1 value valid only for S8
      c_format_ : 2, // bit [ 4, 6) : 0 = F16. 1 = F32, 2 = S32
      : 1,           //
      a_format_ : 3, // bit [ 7,10) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 =
                     // E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1
                     // signed 8 bit. Boolean MMA: 0 Boolean
      b_format_ : 3, // bit [10,13) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 =
                     // E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1
                     // signed 8 bit. Boolean MMA: 0 Boolean
      a_negate_ : 1, // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format
                     // and MXF8F6F4Format
      b_negate_ : 1, // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format
                     // and MXF8F6F4Format
      a_major_ : 1;  // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for
                     // E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
    uint16_t
      b_major_ : 1, // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for
                    // E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
      n_dim_ : 6,   // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32
                    // (N=256).  All values are not valid for all instruction formats
      : 1,          //
      m_dim_ : 5,   // bit [24,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16
                    // (M=256)
      bit_29_ : 1,  //
      max_shift_ : 2; // bit [30,32) : Maximum shift for WS instruction. Encoded as follows: 0 = no
                      // shift, 1 = maximum shift of 8, 2 = maximum shift of 16, 3 = maximum shift
                      // of 32.
  };

  // Decay to a uint32_t
  __host__ __device__ constexpr explicit operator uint32_t() const noexcept { return desc_; }
};

inline __device__ uint32_t make_utcmma_desc(uint32_t fmtC,
                                            uint32_t fmtA,
                                            uint32_t fmtB,
                                            bool majorMA,
                                            bool majorNB,
                                            uint32_t instM,
                                            uint32_t instN,
                                            uint32_t instK,
                                            bool isSparseA,
                                            [[maybe_unused]] bool isQmma) {
  // Setup the Instruction descriptors, see comment on UtcmmaDescriptor above.
  UtcmmaDescriptor desc_mma{0};
  desc_mma.sparse_flag_ = static_cast<uint32_t>(isSparseA);
  desc_mma.c_format_ = fmtC;
  desc_mma.a_format_ = fmtA;
  desc_mma.b_format_ = fmtB;
  desc_mma.a_major_ = majorMA;
  desc_mma.b_major_ = majorNB;
  desc_mma.n_dim_ = instN >> 3;
  desc_mma.m_dim_ = instM >> 4;
  return desc_mma.desc_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Descriptor for UTC*MMA operations with block scaling.
union UtcmmaDescriptorBlock {
  uint32_t desc_;

  struct {
    // Bitfield implementation avoids the need for shifts in assignment
    uint16_t : 2,       //
      sparse_flag_ : 1, // bit [2,3) : 0 = dense. 1 = sparse.
      bit_3_ : 1,       //
      b_sf_id_ : 2,     // bit [4,6) : Matrix B Scale Factor ID
      : 1,              //
      a_format_ : 3,    // bit [7,10) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 =
                        // E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1
                        // signed 8 bit. BMMA: 0 Boolean
      b_format_ : 3,    // bit [10,13) : MXF8F6F4Format:0 = E4M3, 1 = E5M2, 3 = E2M3, 4 = E3M2, 5 =
                        // E2M1. F32F16Format: 0 = F16, 1 = BF16, 2 = TF32. S8: 0 unsigned 8 bit, 1
                        // signed 8 bit. BMMA: 0 Boolean
      a_negate_ : 1, // bit [13,14) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format
                     // and MXF8F6F4Format
      b_negate_ : 1, // bit [14,15) : 0 = no negate. 1 = negate. 1 value valid only for F32F16Format
                     // and MXF8F6F4Format
      a_major_ : 1;  // bit [15,16) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for
                     // E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
    uint16_t
      b_major_ : 1, // bit [16,17) : 0 = K-major. 1 = MN-major. Major value of 1 is only valid for
                    // E4M3, E5M2, INT8 (signed and unsigned), F16, BF16 and TF32 source formats
      n_dim_ : 6,   // bit [17,23) : 3 LSBs not included. Valid values range from 1 (N=8) to 32
                    // (N=256).  All values are not valid for all instruction formats
      scale_format_ : 1, // bit [23,24) : 0=E4M3, 1=E8M0
      : 2,
      bit_26_ : 1,  // bit [26, 27) : 0 = legacy, 1 = unique SfA
      m_dim_ : 2,   // bit [27,29) : 4 LSBs not included. Valid values are: 4 (M=64), 8 (M=128), 16
                    // (M=256)
      a_sf_id_ : 2, // bit [29,31) : Matrix A Scale Factor ID
      k_size_ : 1;  // bit [31,32) : MMA-K Dim. MXF8F6F4Format: 0=[dense: K32, sparse: K64].
                    // S8Format: 0=[dense: K32, sparse: invalid]. MXF4Format: 0=[dense: K64, sparse:
                    // K128], 1=[dense: K96, sparse: invalid].
  };
  // Decay to a uint32_t
  __host__ __device__ constexpr explicit operator uint32_t() const noexcept { return desc_; }
};

inline __device__ uint32_t make_utcmma_desc_block(uint32_t fmtSf,
                                                  uint32_t fmtA,
                                                  uint32_t fmtB,
                                                  bool majorMA,
                                                  bool majorNB,
                                                  uint32_t instM,
                                                  uint32_t instN,
                                                  uint32_t instK,
                                                  bool isSparseA,
                                                  [[maybe_unused]] int version,
                                                  uint32_t byteOffsetSfA,
                                                  uint32_t byteOffsetSfB,
                                                  [[maybe_unused]] bool isOmma) {
  // Setup the Instruction descriptors, see comment on UtcmmaDescriptorBlock above.
  UtcmmaDescriptorBlock desc_mma{0};
  // desc_mma.a_negate_      = 0 (no negate)
  // desc_mma.b_negate_      = 0 (no negate)
  desc_mma.sparse_flag_ = static_cast<uint32_t>(isSparseA);
  desc_mma.scale_format_ = fmtSf;
  desc_mma.a_format_ = fmtA;
  desc_mma.b_format_ = fmtB;
  desc_mma.a_major_ = majorMA;
  desc_mma.b_major_ = majorNB;
  desc_mma.n_dim_ = instN >> 3;
  desc_mma.m_dim_ = instM >> 7;
  desc_mma.a_sf_id_ = byteOffsetSfA;
  desc_mma.b_sf_id_ = byteOffsetSfB;
  desc_mma.k_size_ = (instK == 96 ? 1 : 0);
  return desc_mma.desc_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Reduction helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns the maximum of the absolute value of the given inputs among the group with NaNs filtered
// out.
template <int GroupSize, int GroupStride>
inline __device__ float reduce_group_max_abs_f32(float value, int laneIdx) {
#if __CUDA_HAS_ARCH_FAMILY_SPECIFIC(100)
  constexpr bool UseRedux = (GroupSize >= 16);
#else
  constexpr bool UseRedux = false;
#endif
  float out = value;
  if constexpr (UseRedux) {
    // CREDUX-based reduction.
    constexpr int GroupExtent = GroupSize * GroupStride;
#pragma unroll
    for (int offset = 0; offset < 32; offset += GroupExtent) {
#pragma unroll
      for (int groupOffset = 0; groupOffset < GroupStride; groupOffset++) {
        bool const isParticipating = (laneIdx >= offset && laneIdx < offset + GroupExtent &&
                                      laneIdx % GroupStride == groupOffset);
        float const sharedValue = isParticipating ? out : 0.0f;
        float reducedValue;
        asm("redux.sync.max.abs.f32 %0, %1, 0xffffffff;\n" : "=f"(reducedValue) : "f"(sharedValue));
        if (isParticipating) {
          out = reducedValue;
        }
      }
    }
    // Remove NaNs if all threads participate in redux instruction.
    if constexpr (GroupStride == 1) {
      out = fmaxf(out, 0.f);
    }
  } else {
    // Ensure absolute value has been taken and filter out NaNs.
    out = fmaxf(fabsf(out), 0.f);
    // SHFL-based reduction.
#pragma unroll
    for (int size = GroupSize; size > 1; size /= 2) {
      int const xorStride = (size / 2) * GroupStride;
      out = fmaxf(out, __shfl_xor_sync(0xffffffffu, out, xorStride));
    }
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float reinterpret_uint32_to_float(uint32_t val) {
  return reinterpret_cast<float&>(val);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns the reciprocal of a non-zero fp32 value where only exponent bits are set.
__forceinline__ __device__ float scale_rcp_exp_only(float val) {
  uint32_t bits = 0x7f000000u - reinterpret_cast<uint32_t&>(val);
  return reinterpret_cast<float&>(bits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ cutlass::Array<float, 2> sigmoid2_base2(cutlass::Array<float, 2> x) {
  cutlass::Array<float, 2> result;
  // Vector of ones.
  cutlass::Array<float, 2> one = {1.0f, 1.0f};
  // Force using FMUL2 or compiler falls back to 2x `FADD Rd, Rz, -Rb`.
  cutlass::Array<float, 2> negX = fmul2(-one, x);
  // 2x MUFU.EX2.
  cutlass::Array<float, 2> exp2NegX = {exp2f(negX[0]), exp2f(negX[1])};
  // Use FADD2.
  cutlass::Array<float, 2> denom = fadd2(one, exp2NegX);
  result[0] = 1.0f / denom[0];
  result[1] = 1.0f / denom[1];
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Returns the largest power-of-2 less than or equal to the absolute value of the input.
__forceinline__ __device__ float trunc_abs_float_to_pow2(float val) {
  // Mask out sign and mantissa bits, only retain exponent.
  constexpr uint32_t mask = 0x7f800000u;
  uint32_t bits = reinterpret_cast<uint32_t&>(val) & mask;
  return reinterpret_cast<float&>(bits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
