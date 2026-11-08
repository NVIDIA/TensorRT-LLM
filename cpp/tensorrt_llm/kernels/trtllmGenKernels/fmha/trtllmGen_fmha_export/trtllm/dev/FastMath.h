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

// Fast Modulo/Division using Precomputation (int32_t only)
//
// Portable standalone implementation based on NVIDIA CCCL library
// Original:
// https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__cmath/fast_modulo_division.h
//
// Implements fast division and modulo operations by precomputing multipliers
// that replace slow division with fast multiplication + bit shifts.
//
// References:
// - Hacker's Delight, Second Edition, Chapter 10
// - Labor of Division (Episode III):
// https://ridiculousfish.com/blog/posts/labor-of-division-episode-iii.html
// - Classic Round-Up Variant: https://arxiv.org/pdf/2412.03680

// Note that this portable header is a WAR for CTK < 13.1. CCCL fast_mod_div is supported since
// CTK 13.1. CCCL fast_mod_div should be adopted in the future.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

#define FMD_HOST_DEVICE __host__ __device__
#define FMD_DEVICE __device__
#define FMD_FORCEINLINE inline

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////////////////////////

FMD_HOST_DEVICE FMD_FORCEINLINE constexpr bool is_power_of_two(int32_t x) noexcept {
  return x > 0 && (x & (x - 1)) == 0;
}

FMD_HOST_DEVICE FMD_FORCEINLINE constexpr int ilog2(int32_t x) noexcept {
  int log = 0;
  uint32_t ux = static_cast<uint32_t>(x);
  while (ux >>= 1) {
    ++log;
  }
  return log;
}

FMD_HOST_DEVICE FMD_FORCEINLINE constexpr int ceil_ilog2(int32_t x) noexcept {
  return is_power_of_two(x) ? ilog2(x) : ilog2(x) + 1;
}

FMD_HOST_DEVICE FMD_FORCEINLINE constexpr uint64_t ceil_div(uint64_t a, uint64_t b) noexcept {
  return (a + b - 1) / b;
}

// Get high 32 bits of 32x32->64 bit multiplication
#if defined(__CUDA_ARCH__)
FMD_DEVICE FMD_FORCEINLINE uint32_t mul_hi(uint32_t a, uint32_t b) noexcept {
  return __umulhi(a, b);
}
#else
FMD_HOST_DEVICE FMD_FORCEINLINE uint32_t mul_hi(uint32_t a, uint32_t b) noexcept {
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// fast_mod_div - Fast division and modulo for int32_t
////////////////////////////////////////////////////////////////////////////////////////////////////

// Fast division and modulo for int32_t using precomputed multipliers
//
// Note: Dividend must be non-negative (>= 0)
//
// Usage:
//   fast_mod_div divisor(7);
//   int32_t result = divisor.div(100);  // Fast division: 14
//   int32_t remainder = divisor.mod(100);  // Fast modulo: 2
//
// Or using operator syntax:
//   int32_t result = 100 / divisor;     // 14
//   int32_t remainder = 100 % divisor;  // 2

class fast_mod_div {
public:
  fast_mod_div() = delete;

  // Constructor - precomputes division constants
  FMD_HOST_DEVICE explicit fast_mod_div(int32_t divisor_arg) noexcept
    : divisor_(divisor_arg) {
    // Algorithm for signed integers (Hacker's Delight approach)
    shift_ = ceil_ilog2(divisor_) - 1;
    // K in range [32, 62]
    int k = 32 + shift_;
    uint64_t multiplier_large = ceil_div(uint64_t{1} << k, static_cast<uint64_t>(divisor_));
    multiplier_ = static_cast<uint32_t>(multiplier_large);
    (void)add_;
  }

  // Fast division
  FMD_HOST_DEVICE FMD_FORCEINLINE int32_t div(int32_t dividend) const noexcept {
    if (divisor_ == 1) {
      return dividend;
    }

    uint32_t udividend = static_cast<uint32_t>(dividend);
    uint32_t higher_bits = mul_hi(udividend, multiplier_);
    return static_cast<int32_t>(higher_bits >> shift_);
  }

  // Fast modulo
  FMD_HOST_DEVICE FMD_FORCEINLINE int32_t mod(int32_t dividend) const noexcept {
    return dividend - div(dividend) * divisor_;
  }

  // Division operator overload
  FMD_HOST_DEVICE FMD_FORCEINLINE friend int32_t operator/(int32_t dividend,
                                                           const fast_mod_div& d) noexcept {
    return d.div(dividend);
  }

  // Modulo operator overload
  FMD_HOST_DEVICE FMD_FORCEINLINE friend int32_t operator%(int32_t dividend,
                                                           const fast_mod_div& d) noexcept {
    return d.mod(dividend);
  }

private:
  // Original divisor
  int32_t divisor_ = 1;
  // Magic multiplier
  uint32_t multiplier_ = 0;
  // Add value (compatible with cccl fast_mod_div)
  uint32_t add_ = 0;
  // Shift amount
  int shift_ = 0;
};

} // namespace dev
} // namespace trtllm

// Clean up macros if desired
#undef FMD_HOST_DEVICE
#undef FMD_DEVICE
#undef FMD_FORCEINLINE
