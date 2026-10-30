/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Device helpers for NVFP4 on the SM100 tensor cores (tcgen05.mma kind::mxf4nvf4, block scale 16):
// operand / scale-factor shared-memory descriptors, the scale-factor copy into tensor memory, the
// MMA and the accumulator reads.
//
// Operands are K-major with the 128-byte swizzle: rows of 128 bytes (256 e2m1 values), 8-row
// groups 1024 bytes apart, 16-byte chunk c of row r at r * 128 + ((c ^ (r & 7)) << 4). One MMA
// takes K = 64 values (32 bytes of every row): the descriptor start advances by 32 bytes per step.
//
// Scale factors: for every 64 values of K, the 128 rows of a tile take one 512-byte atom laid out
// as the CUTLASS SM100 scale layout (k::nvfp4_sf_off): row m's 4 scales at (m % 32) * 16 +
// (m / 32) * 4. tcgen05.cp 32x128b.warpx4 copies an atom into 4 tensor-memory columns of all four
// lane quadrants; the MMA of that K step reads its scales from those columns (A: rows of the
// 128-row tile, B: the N <= 32 token rows at lanes 0..N-1, column + 0).
#pragma once

#include <cstdint>

#include "cute/arch/copy_sm100.hpp"
#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/arch/mma_sm100_umma.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cutlass/arch/barrier.h"

namespace glm {
namespace k {
namespace fp4mma {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// byte offset of 16-byte chunk c (0..7) of row r in a 128-byte-swizzled K-major tile
__device__ __forceinline__ int swz128(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

// K-major operand with the 128-byte swizzle (rows 128 B apart, 8-row groups 1024 B apart)
__device__ __forceinline__ uint64_t desc_k_sw128(uint32_t smem_addr) {
  cute::UMMA::SmemDescriptor d;
  d.start_address_ = static_cast<uint16_t>((smem_addr >> 4) & 0x3FFF);
  d.leading_byte_offset_ = 1;
  d.stride_byte_offset_ = 1024 >> 4;
  d.version_ = 1;
  d.base_offset_ = 0;
  d.lbo_mode_ = 0;
  d.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  return d.desc_;
}

// source of a 32x128b scale-factor copy: 32 rows of 16 bytes, contiguous (no swizzle)
__device__ __forceinline__ uint64_t desc_sf_atom(uint32_t smem_addr) {
  cute::UMMA::SmemDescriptor d;
  d.start_address_ = static_cast<uint16_t>((smem_addr >> 4) & 0x3FFF);
  d.leading_byte_offset_ = 1;          // (one 16-byte chunk along K)
  d.stride_byte_offset_ = 128 >> 4;   // 8-row core matrices 128 B apart
  d.version_ = 1;
  d.base_offset_ = 0;
  d.lbo_mode_ = 0;
  d.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_NONE);
  return d.desc_;
}

// one 512-byte scale atom (shared memory, 16-byte aligned) into tensor-memory columns
// [tmem_col, tmem_col + 4) of every lane quadrant
__device__ __forceinline__ void sf_copy(uint32_t tmem_col, uint32_t smem_addr) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n" ::"r"(tmem_col),
               "l"(desc_sf_atom(smem_addr)));
}

// instruction descriptor of an M = 128, N token MMA (e2m1 x e2m1, ue4m3 scales, f32 accumulate)
template <int N>
__device__ __forceinline__ uint64_t idesc_fp4(uint32_t tsfa, uint32_t tsfb) {
  return cute::UMMA::make_runtime_instr_desc_block_scaled<
      cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t, 128, N,
      cute::UMMA::Major::K, cute::UMMA::Major::K>(tsfa, tsfb);
}

// D[128 x N] (+)= A[128 x 64] . B[N x 64]^T with block scales at tensor-memory columns tsfa / tsfb
template <int N>
__device__ __forceinline__ void mma(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_d,
                                    bool accumulate, uint32_t tsfa, uint32_t tsfb) {
  cute::SM100_MMA_MXF4_SS<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
                          cutlass::float_ue4m3_t, 128, N, 16, cute::UMMA::Major::K,
                          cute::UMMA::Major::K>::fma(desc_a, desc_b, tmem_d, accumulate ? 1u : 0u,
                                                     idesc_fp4<N>(tsfa, tsfb), tsfa, tsfb);
}

__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

// this warp's 32 lanes (quadrant = warp % 4) x 8 consecutive fp32 columns
__device__ __forceinline__ void tmem_ld8(uint32_t taddr, float* v) {
  uint32_t r[8];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7])
      : "r"(taddr));
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(r[i]);
}

__device__ __forceinline__ void tmem_ld16(uint32_t taddr, float* v) {
  uint32_t r[16];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,"
      "%14,%15}, [%16];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15])
      : "r"(taddr));
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = __uint_as_float(r[i]);
}

}  // namespace fp4mma
}  // namespace k
}  // namespace glm
