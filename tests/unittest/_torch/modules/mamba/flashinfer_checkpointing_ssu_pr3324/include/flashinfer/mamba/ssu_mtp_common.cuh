/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Shared definitions for the vertical and horizontal MTP kernels.

#pragma once

#include <cuda/barrier>

#include "conversion.cuh"

namespace flashinfer::mamba::mtp {

// Round up to next power of 2 (compile-time).
constexpr int nextPow2(int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

enum class WarpRole { kCompute, kTMALoad, kEpilogue };

__device__ __forceinline__ WarpRole get_warp_role(int warp) {
  if (warp < 12) return WarpRole::kCompute;
  if (warp < 15) return WarpRole::kTMALoad;
  return WarpRole::kEpilogue;
}

// XOR-based bank-conflict-free swizzle for horizontal state traversal.
// Operates on flat byte addresses: XORs the bank index with the row (cycle) index.
// cycle_length = row stride in bytes, bank_size = sizeof(uint32_t).
template <int cycle_length, int bank_size>
__device__ __forceinline__ int xor_swizzle(int address) {
  int const cycle = address / cycle_length;
  int const delta = address % cycle_length;
  int const bank_idx = delta / bank_size;
  int const intra_bank = delta % bank_size;
  int const new_bank_idx = bank_idx ^ cycle;
  return cycle * cycle_length + new_bank_idx * bank_size + intra_bank;
}

// ── Parity-based barrier helpers (tight spin, no NANOSLEEP) ─────────────────
// More efficient than cuda::barrier::wait() for latency-sensitive pipelines.
// The standard cuda::barrier::wait() adds a NANOSLEEP backoff loop between
// try_wait attempts, which can overshoot and waste cycles. The raw
// mbarrier.try_wait.parity instruction does a tight spin instead.
// See CUDA Programming Guide §4.9.3 "Explicit Phase Tracking".

__device__ __forceinline__ void arrive_and_wait_parity(barrier_t& bar, uint32_t& parity) {
  uint32_t const smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(cuda::device::barrier_native_handle(bar)));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr) : "memory");
  uint32_t ready = 0;
  while (!ready) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(ready)
        : "r"(smem_addr), "r"(parity));
  }
  parity ^= 1;
}

// ── SM100 f32x2 packed SIMD helpers ──────────────────────────────────────────
// On Blackwell (SM100+), {mul,fma}.f32x2 pack two fp32 operations into one
// instruction and issue on the dedicated FMUL2 pipeline, which runs in parallel
// with the regular FMA pipe.  This halves instruction count for element-wise
// fp32 math on independent pairs (e.g. adjacent state-vector components).
// On older architectures the fallback is two scalar ops — zero overhead.
// See: https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/simd_sm100.hpp

__device__ __forceinline__ void mul_f32x2(float2& c, float2 const& a, float2 const& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm("mul.f32x2 %0, %1, %2;\n"
      : "=l"(reinterpret_cast<uint64_t&>(c))
      : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)));
#else
  c.x = a.x * b.x;
  c.y = a.y * b.y;
#endif
}

__device__ __forceinline__ void fma_f32x2(float2& d, float2 const& a, float2 const& b,
                                          float2 const& c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm("fma.rn.f32x2 %0, %1, %2, %3;\n"
      : "=l"(reinterpret_cast<uint64_t&>(d))
      : "l"(reinterpret_cast<uint64_t const&>(a)), "l"(reinterpret_cast<uint64_t const&>(b)),
        "l"(reinterpret_cast<uint64_t const&>(c)));
#else
  d.x = a.x * b.x + c.x;
  d.y = a.y * b.y + c.y;
#endif
}

// =============================================================================
// convertAndStoreSRHorizontal — convert a pair of f32 state values to half.
// When PHILOX_ROUNDS > 0: stochastic rounding via f16x2.
// When PHILOX_ROUNDS == 0: plain nearest-even conversion.
// e is the pair-aligned index within the tile (must be even).
// =============================================================================

template <typename state_t, int DSTATE, int PHILOX_ROUNDS>
__device__ __forceinline__ void convertAndStoreSRHorizontal(state_t& out0, state_t& out1, float s0,
                                                            float s1, int64_t rand_seed,
                                                            int state_ptr_offset, int dd, int col0,
                                                            int e, uint32_t (&rand_ints)[4]) {
  using namespace conversion;
  if constexpr (PHILOX_ROUNDS > 0) {
    if (e % 4 == 0)
      philox_randint4x<PHILOX_ROUNDS>(rand_seed, state_ptr_offset + dd * DSTATE + col0 + e,
                                      rand_ints[0], rand_ints[1], rand_ints[2], rand_ints[3]);
    uint32_t packed = cvt_rs_f16x2_f32(s0, s1, rand_ints[e / 2 % 2]);
    out0 = __ushort_as_half(static_cast<uint16_t>(packed & 0xFFFFu));
    out1 = __ushort_as_half(static_cast<uint16_t>(packed >> 16));
  } else {
    convertAndStore(&out0, s0);
    convertAndStore(&out1, s1);
  }
}

}  // namespace flashinfer::mamba::mtp
