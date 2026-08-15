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

#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_ptx/cuda_ptx.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 1: Types and Descriptors
//
// Pure data types with no dependencies on other code in this file.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct HybridSliceKRegs {
  uint32_t warpIdx;
  uint32_t laneIdx;

  __device__ HybridSliceKRegs()
    : warpIdx(threadIdx.x / 32)
    , laneIdx(threadIdx.x % 32) {}

  __device__ HybridSliceKRegs(uint32_t warp, uint32_t lane)
    : warpIdx(warp)
    , laneIdx(lane) {}
};

union HybridInstrDescriptor {
  uint32_t desc_;

  struct {
    uint16_t sparse_id2_ : 2, // bit [ 0, 2)
      sparse_flag_ : 1,       // bit [ 2, 3)
      saturate_ : 1,          // bit [ 3, 4)
      c_format_ : 2,          // bit [ 4, 6) : 0 = F16, 1 = F32, 2 = S32
      : 1,                    // bit [ 6, 7)
      a_format_ : 3,          // bit [ 7,10)
      b_format_ : 3,          // bit [10,13)
      a_negate_ : 1,          // bit [13,14)
      b_negate_ : 1,          // bit [14,15)
      a_major_ : 1;           // bit [15,16)
    uint16_t b_major_ : 1,    // bit [16,17)
      n_dim_ : 6,             // bit [17,23)
      : 1,                    // bit [23,24)
      m_dim_ : 5,             // bit [24,29)
      : 1,                    // bit [29,30)
      max_shift_ : 2;         // bit [30,32)
  };

  __host__ __device__ constexpr explicit operator uint32_t() const noexcept { return desc_; }
};

union HybridSmemDescriptor {
  uint64_t desc_;
  uint32_t reg32_[2];

  struct {
    uint16_t start_address_ : 14, : 2;
    uint16_t leading_byte_offset_ : 14, : 2;
    uint16_t stride_byte_offset_ : 14, version_ : 2;
    uint8_t : 1, base_offset_ : 3, : 4;
    uint8_t : 5, layout_type_ : 3;
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 2: PTX Primitives
//
// TMEM store/load and fence delegate to cuda_ptx:: library wrappers.
// MMA (f8f6f4 + tmem_a + WDM lane mask) and elect have no cuda_ptx equivalent
// and remain as raw asm volatile.
// No dependencies on anything else in this file.
////////////////////////////////////////////////////////////////////////////////////////////////////

// --- TMEM Store ---
// Thin wrappers around cuda_ptx::tcgen05_st_32x32b / tcgen05_st_16x128b.
// We keep the pointer-based signatures so call sites stay unchanged.

inline __device__ void tcgen05_st_32x32b_x1(uint32_t tmemAddr, uint32_t const* rf) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_st_32x32b(tmemAddr, reinterpret_cast<uint32_t const(&)[1]>(*rf));
#else
  (void)tmemAddr;
  (void)rf;
#endif
}

inline __device__ void tcgen05_st_32x32b_x2(uint32_t tmemAddr, uint32_t const* rf) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_st_32x32b(tmemAddr, reinterpret_cast<uint32_t const(&)[2]>(*rf));
#else
  (void)tmemAddr;
  (void)rf;
#endif
}

inline __device__ void tcgen05_st_32x32b_x4(uint32_t tmemAddr, uint32_t const* rf) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_st_32x32b(tmemAddr, reinterpret_cast<uint32_t const(&)[4]>(*rf));
#else
  (void)tmemAddr;
  (void)rf;
#endif
}

inline __device__ void tcgen05_st_32x32b_x8(uint32_t tmemAddr, uint32_t const* rf) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_st_32x32b(tmemAddr, reinterpret_cast<uint32_t const(&)[8]>(*rf));
#else
  (void)tmemAddr;
  (void)rf;
#endif
}

inline __device__ void tcgen05_st_32x32b_x16(uint32_t tmemAddr, uint32_t const* rf) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_st_32x32b(tmemAddr, reinterpret_cast<uint32_t const(&)[16]>(*rf));
#else
  (void)tmemAddr;
  (void)rf;
#endif
}

inline __device__ void tcgen05_st_32x32b_x32(uint32_t tmemAddr, uint32_t const* rf) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_st_32x32b(tmemAddr, reinterpret_cast<uint32_t const(&)[32]>(*rf));
#else
  (void)tmemAddr;
  (void)rf;
#endif
}

inline __device__ void tcgen05_st_16x128b_x1(uint32_t tmemAddr, void* rf) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_st_16x128b(tmemAddr,
                               reinterpret_cast<uint32_t const(&)[2]>(*static_cast<uint32_t*>(rf)));
#else
  (void)tmemAddr;
  (void)rf;
#endif
}

// Store zeros to 8 TMEM columns (used by accumulator init)
inline __device__ void tcgen05_st_32x32b_x8_zero(uint32_t tmemAddr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  const uint32_t zeros[8] = {};
  cuda_ptx::tcgen05_st_32x32b(tmemAddr, zeros);
#else
  (void)tmemAddr;
#endif
}

// --- TMEM Load ---
// Thin wrappers around cuda_ptx::tcgen05_ld_32x32b.

inline __device__ void tcgen05_ld_32x32b_x1(uint32_t* rf, uint32_t tmemAddr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_ld_32x32b(reinterpret_cast<uint32_t(&)[1]>(*rf), tmemAddr);
#else
  (void)rf;
  (void)tmemAddr;
#endif
}

inline __device__ void tcgen05_ld_32x32b_x8(uint32_t* rf, uint32_t tmemAddr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_ld_32x32b(reinterpret_cast<uint32_t(&)[8]>(*rf), tmemAddr);
#else
  (void)rf;
  (void)tmemAddr;
#endif
}

// --- MMA ---

// FP8 MMA without lane mask.
inline __device__ void tcgen05_mma_1sm_fp8_TS(uint32_t dtmem,
                                              uint32_t atmem,
                                              uint64_t smemDescB,
                                              uint32_t mmaDesc,
                                              int32_t scaleD) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;"
               "}\n" ::"r"(dtmem),
               "r"(atmem),
               "l"(smemDescB),
               "r"(mmaDesc),
               "r"(scaleD));
#else
  (void)dtmem;
  (void)atmem;
  (void)smemDescB;
  (void)mmaDesc;
  (void)scaleD;
#endif
}

// FP8 MMA with ashift variant.
inline __device__ void tcgen05_mma_1sm_fp8_TS_shift(uint32_t dtmem,
                                                    uint32_t atmem,
                                                    uint64_t smemDescB,
                                                    uint32_t mmaDesc,
                                                    int32_t scaleD) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f8f6f4.ashift [%0], [%1], %2, %3, p;"
               "}\n" ::"r"(dtmem),
               "r"(atmem),
               "l"(smemDescB),
               "r"(mmaDesc),
               "r"(scaleD));
#else
  (void)dtmem;
  (void)atmem;
  (void)smemDescB;
  (void)mmaDesc;
  (void)scaleD;
#endif
}

// FP8 MMA with lane mask.
// NOTE: FP8 does NOT use .block_scale or .collector modifiers (those are for FP4).
inline __device__ void tcgen05_mma_1sm_fp8_TS_warp_lane_mask(uint32_t dtmem,
                                                             uint32_t atmem,
                                                             uint64_t smemDescB,
                                                             uint32_t mmaDesc,
                                                             int32_t scaleD,
                                                             uint32_t mask0,
                                                             uint32_t mask1,
                                                             uint32_t mask2,
                                                             uint32_t mask3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p;\n"
               "}\n"
               :
               : "r"(dtmem),
                 "r"(atmem),
                 "l"(smemDescB),
                 "r"(mmaDesc),
                 "r"(scaleD),
                 "r"(mask0),
                 "r"(mask1),
                 "r"(mask2),
                 "r"(mask3));
#else
  (void)dtmem;
  (void)atmem;
  (void)smemDescB;
  (void)mmaDesc;
  (void)scaleD;
  (void)mask0;
  (void)mask1;
  (void)mask2;
  (void)mask3;
#endif
}

// Overload with single mask (replicated to all 4)
inline __device__ void tcgen05_mma_1sm_fp8_TS_warp_lane_mask(uint32_t dtmem,
                                                             uint32_t atmem,
                                                             uint64_t smemDescB,
                                                             uint32_t mmaDesc,
                                                             int32_t scaleD,
                                                             uint32_t mask) {
  tcgen05_mma_1sm_fp8_TS_warp_lane_mask(dtmem,
                                        atmem,
                                        smemDescB,
                                        mmaDesc,
                                        scaleD,
                                        mask,
                                        mask,
                                        mask,
                                        mask);
}

// --- Fence and Elect ---

// Fence: fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster
// Emits a scoreboard req on all preceding LDS instructions.
inline __device__ void hybridSliceKFenceProxyAsync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::fence_proxy_async_generic_sync_restrict(cuda_ptx::sem_acquire,
                                                    cuda_ptx::space_cluster,
                                                    cuda_ptx::scope_cluster);
#endif
}

// Elect sync helper
inline __device__ bool electSync(uint32_t mask = 0xFFFFFFFF) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t is_elected;
  asm volatile("{\n"
               ".reg .pred p;\n"
               "elect.sync _|p, %1;\n"
               "selp.b32 %0, 1, 0, p;\n"
               "}\n"
               : "=r"(is_elected)
               : "r"(mask));
  return is_elected != 0;
#else
  (void)mask;
  return (threadIdx.x % 32) == 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 3: Descriptor Creation
//
// Functions that populate Section 1 descriptor types.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Create MMA instruction descriptor for FP8 UTCQMMA
template <int INST_MMA_M, int INST_MMA_N, bool is_A_E5M2 = false, bool is_B_E5M2 = false>
inline __device__ void makeHybridUtcqmmaDesc(HybridInstrDescriptor& desc) {
  desc.desc_ = 0;
  desc.sparse_flag_ = 0;              // Dense
  desc.a_format_ = is_A_E5M2 ? 1 : 0; // 0 = E4M3, 1 = E5M2
  desc.b_format_ = is_B_E5M2 ? 1 : 0;
  desc.c_format_ = 1; // F32
  desc.a_negate_ = 0;
  desc.b_negate_ = 0;
  desc.a_major_ = 0; // K-major
  desc.b_major_ = 0; // K-major
  // tcgen05.mma descriptor encoding (3 LSBs and 4 LSBs not included):
  // n_dim_ = N / 8, valid range 1 (N=8) to 32 (N=256)
  // m_dim_ = M / 16, valid values: 4 (M=64), 8 (M=128), 16 (M=256)
  desc.n_dim_ = INST_MMA_N / 8;
  desc.m_dim_ = INST_MMA_M / 16;
}

// Create SMEM descriptor with SWIZZLE_32B
inline __device__ void makeHybridSmemDesc32B(HybridSmemDescriptor& desc, void const* smemPtr) {
  desc.desc_ = 0;
  desc.leading_byte_offset_ = 16 >> 4;
  desc.stride_byte_offset_ = (2 * 128) >> 4;
  desc.base_offset_ = 0;
  desc.layout_type_ = 6; // SWIZZLE_32B
  desc.start_address_ = __cvta_generic_to_shared(smemPtr) >> 4;
}

// Create SMEM descriptor with SWIZZLE_64B
inline __device__ void makeHybridSmemDesc64B(HybridSmemDescriptor& desc, void const* smemPtr) {
  desc.desc_ = 0;
  desc.leading_byte_offset_ = 16 >> 4;
  desc.stride_byte_offset_ = (4 * 128) >> 4;
  desc.base_offset_ = 0;
  desc.layout_type_ = 4; // SWIZZLE_64B
  desc.start_address_ = __cvta_generic_to_shared(smemPtr) >> 4;
}

// Create SMEM descriptor with SWIZZLE_128B
inline __device__ void makeHybridSmemDesc128B(HybridSmemDescriptor& desc, void const* smemPtr) {
  desc.desc_ = 0;
  desc.leading_byte_offset_ = 16 >> 4;
  desc.stride_byte_offset_ = (8 * 128) >> 4;
  desc.base_offset_ = 0;
  desc.layout_type_ = 2; // SWIZZLE_128B
  desc.start_address_ = __cvta_generic_to_shared(smemPtr) >> 4;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 4: Configuration
//
// Compile-time constants and the HybridSliceKConfig struct.
// No runtime dependencies.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Compute FundamentalKtileUnit based on Mtile
template <int Mtile> constexpr int computeFundamentalKtileUnit() {
  if constexpr (Mtile < 32) {
    return 32 / Mtile;
  } else {
    return 1;
  }
}

// Compute warp mask based on Mtile (Ntile in LowLatencyKernels terminology)
template <int Mtile> constexpr uint32_t computeHybridWarpMask() {
  if constexpr (Mtile == 128) {
    return 0xFFFFFFFF; // All lanes active
  } else if constexpr (Mtile == 64) {
    return 0x55555555; // Every 2nd lane
  } else if constexpr (Mtile == 32) {
    return 0x11111111; // Every 4th lane
  } else if constexpr (Mtile == 16) {
    return 0x11111111; // Hybrid mode kicks in
  } else if constexpr (Mtile == 8) {
    return 0x11111111;
  } else { // Mtile == 4
    return 0x11111111;
  }
}

// Compute ReRunTimes based on Mtile and HybridSliceK mode
template <int Mtile, bool HybridSliceK = true> constexpr int computeReRunTimes() {
  if constexpr (HybridSliceK) {
    if constexpr (Mtile > 32) {
      return 128 / Mtile;
    } else {
      return 4;
    }
  } else {
    return 128 / Mtile;
  }
}

template <int Mtile_, int Ktile128BNum_, int MaxBS_, int WarpNum_ = 4, bool HybridSliceK_ = true>
struct HybridSliceKConfig {
  static constexpr int Mtile = Mtile_;
  static constexpr int Ktile128BNum = Ktile128BNum_;
  static constexpr int MaxBS = MaxBS_;
  static constexpr int WarpNum = WarpNum_;
  static constexpr bool HybridSliceK = HybridSliceK_;

  // Derived parameters (all based on Mtile, not MaxBS).
  static constexpr int FundamentalKtileUnit = computeFundamentalKtileUnit<Mtile>();
  static constexpr int ReRunTimes = computeReRunTimes<Mtile, HybridSliceK>();
  static constexpr uint32_t WarpMask = computeHybridWarpMask<Mtile>();

  // TMEM column calculations (based on Mtile).
  static constexpr int TmemACols_1inst = 8; // Fixed per single tcgen05_st_32x32b_x8 call
  static constexpr int TmemACols_1set = (Mtile >= 32)
                                          ? (Mtile / 32 * Ktile128BNum * TmemACols_1inst)
                                          : (Ktile128BNum / FundamentalKtileUnit * TmemACols_1inst);

  // Accumulator columns (extended in Hybrid mode for small Mtile)
  static constexpr int AccumColsExtensionFactor = (HybridSliceK && Mtile < 32) ? (32 / Mtile) : 1;

  // Total K iterations per stage
  static constexpr int TotalIterations = Ktile128BNum * 4; // 4 = 128B/32B
  static constexpr int InstNum = TotalIterations / (WarpNum * FundamentalKtileUnit);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 5: SMEM to TMEM Copy (A Operand)
//
// Loads Weight W from SMEM to registers, then stores to TMEM.
// Depends on: Section 2 (PTX primitives).
////////////////////////////////////////////////////////////////////////////////////////////////////

// Swizzle function for 128B swizzle mode.
inline __device__ int swizzle128BFunc(int rowId, int col16BId) {
  int xorFactor = rowId % 8;
  int swizzledCol16BId = col16BId ^ xorFactor;
  return rowId * 128 + swizzledCol16BId * 16;
}

// Load 256 bits from SMEM for 32x128B tile with 4 warps in 4x1 arrangement
// Load pattern for 32x128B tile layout.
template <int Mtile>
inline __device__ void hybridSliceKLoad256bits_4x1(int warpIdx,
                                                   int laneIdx,
                                                   void const* smemPtr,
                                                   uint32_t* regOut,
                                                   int colOffset) {
  // Thread mapping: 4 warps cover 32 rows (8 rows each)
  // Each thread loads 2 x uint4 = 32 bytes
  int rowId = laneIdx / 4 + (warpIdx % 4) * 8;
  int col16BId = (laneIdx % 4) * 2;

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int smemOffset = swizzle128BFunc(rowId, col16BId + colOffset + i);
    uint4* regPtr = reinterpret_cast<uint4*>(&regOut[i * 4]);
    uint4 const* smemSrc =
      reinterpret_cast<uint4 const*>(static_cast<char const*>(smemPtr) + smemOffset);
    *regPtr = *smemSrc;
  }
}

// Load 256 bits from SMEM for 64x64B tile with 4 warps in 4x1 arrangement
// Load pattern for 64x64B tile layout.
inline __device__ void hybridSliceKLoad256bits_64x64B_4x1(int warpIdx,
                                                          int laneIdx,
                                                          void const* smemPtr,
                                                          uint32_t* regOut,
                                                          int colOffset) {
  int rowId = laneIdx / 2 + (warpIdx % 4) * 16;
  int col16BId = (laneIdx % 2) * 4;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int smemOffset = swizzle128BFunc(rowId, col16BId + colOffset + i);
    uint4* regPtr = reinterpret_cast<uint4*>(&regOut[i * 4]);
    uint4 const* smemSrc =
      reinterpret_cast<uint4 const*>(static_cast<char const*>(smemPtr) + smemOffset);
    *regPtr = *smemSrc;
  }
}

// Load 256 bits from SMEM for 128x32B tile with 4 warps in 4x1 arrangement
// Load pattern for 128x32B tile layout.
inline __device__ void hybridSliceKLoad256bits_128x32B_based(int warpIdx,
                                                             int laneIdx,
                                                             void const* smemPtr,
                                                             uint32_t* regOut,
                                                             int colOffset) {
  int rowId = laneIdx + (warpIdx % 4) * 32;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int smemOffset = swizzle128BFunc(rowId, colOffset + i);
    uint4* regPtr = reinterpret_cast<uint4*>(&regOut[i * 4]);
    uint4 const* smemSrc =
      reinterpret_cast<uint4 const*>(static_cast<char const*>(smemPtr) + smemOffset);
    *regPtr = *smemSrc;
  }
}

// Load SmemRows rows from SMEM to registers using 256-bit loads.
//
// SMEM layout: [Ktile_128Bnum][SmemRows][128B]
// Register layout: [Ktile_128Bnum * SmemRows / 32][8] (uint32_t)
//
// Template parameters:
//   SmemRows: Number of rows in SMEM (MaxBS for Activation X, M_tma for Weight W)
//   Ktile128BNum: Number of 128B K-tiles per stage
template <int SmemRows, int Ktile128BNum>
inline __device__ void hybridSliceKLoadA_perfpatch(int warpIdx,
                                                   int laneIdx,
                                                   void const* smemPtr,
                                                   uint32_t (*regOut)[8]) {
  static_assert(SmemRows % 4 == 0, "SmemRows must be a multiple of 4");

  constexpr int SmemRowsDiv32 = SmemRows / 32;
  constexpr int SmemRowsDiv4 = SmemRows / 4;

  // K-slice stride in SMEM: SmemRows * 128B per K-slice
  constexpr int SmemKSliceStride = SmemRows * 128;

  // Select load function based on SmemRows
  if constexpr (SmemRowsDiv32 >= 4) {
    // SmemRows >= 128: use 128x32B load pattern, 4 iterations per K-slice
#pragma unroll
    for (int i = 0; i < Ktile128BNum; ++i) {
#pragma unroll
      for (int kk = 0; kk < SmemRowsDiv32; ++kk) {
        hybridSliceKLoad256bits_128x32B_based(warpIdx,
                                              laneIdx,
                                              static_cast<char const*>(smemPtr) +
                                                i * SmemKSliceStride,
                                              &regOut[i * SmemRowsDiv32 + kk][0],
                                              kk * 2);
      }
    }
  } else if constexpr (SmemRowsDiv32 == 2) {
    // SmemRows == 64: use 64x64B load pattern, 2 iterations per K-slice
#pragma unroll
    for (int i = 0; i < Ktile128BNum; ++i) {
#pragma unroll
      for (int kk = 0; kk < SmemRowsDiv32; ++kk) {
        hybridSliceKLoad256bits_64x64B_4x1(warpIdx,
                                           laneIdx,
                                           static_cast<char const*>(smemPtr) + i * SmemKSliceStride,
                                           &regOut[i * SmemRowsDiv32 + kk][0],
                                           kk * 2);
      }
    }
  } else if constexpr (SmemRowsDiv32 == 1) {
    // SmemRows == 32: use 32x128B load pattern, 1 iteration per K-slice
#pragma unroll
    for (int i = 0; i < Ktile128BNum; ++i) {
      hybridSliceKLoad256bits_4x1<SmemRows>(warpIdx,
                                            laneIdx,
                                            static_cast<char const*>(smemPtr) +
                                              i * SmemKSliceStride,
                                            &regOut[i][0],
                                            0);
    }
  } else {
    // SmemRows < 32: use small tile load pattern with fundamental K-tile grouping
    constexpr int FundamentalKDiv128B = 2 * (4 / SmemRowsDiv4);
    static_assert(Ktile128BNum % FundamentalKDiv128B == 0,
                  "Ktile128BNum must be divisible by FundamentalKDiv128B");
    constexpr int SmallTileKSliceStride = SmemRowsDiv4 * 4 * 128;
#pragma unroll
    for (int i = 0; i < Ktile128BNum; i += FundamentalKDiv128B) {
      hybridSliceKLoad256bits_4x1<SmemRows>(warpIdx,
                                            laneIdx,
                                            static_cast<char const*>(smemPtr) +
                                              i * SmallTileKSliceStride,
                                            &regOut[i / FundamentalKDiv128B][0],
                                            0);
    }
  }
}

// Store registers to TMEM using tcgen05_st_32x32b_x8.
// TmemRows: number of rows to store.
template <int TmemRows, int Ktile128BNum>
inline __device__ void hybridSliceKStoreA_perfpatch(uint32_t (*regIn)[8], uint32_t tmemAddr) {
  static_assert(TmemRows % 4 == 0, "TmemRows must be a multiple of 4");
  if constexpr (TmemRows >= 32) {
    constexpr int TmemRowsDiv32 = TmemRows / 32;
#pragma unroll
    for (int i = 0; i < Ktile128BNum; ++i) {
#pragma unroll
      for (int kk = 0; kk < TmemRowsDiv32; ++kk) {
        tcgen05_st_32x32b_x8(tmemAddr + i * 8 * TmemRowsDiv32 + kk * 8,
                             regIn[i * TmemRowsDiv32 + kk]);
      }
    }
  } else {
    constexpr int TmemRowsDiv4 = TmemRows / 4;
    constexpr int FundamentalKDiv128B = 2 * (4 / TmemRowsDiv4);
    static_assert(Ktile128BNum % FundamentalKDiv128B == 0,
                  "Ktile128BNum must be divisible by FundamentalKDiv128B");
#pragma unroll
    for (int i = 0; i < Ktile128BNum; i += FundamentalKDiv128B) {
      tcgen05_st_32x32b_x8(tmemAddr + (i / FundamentalKDiv128B) * 8,
                           regIn[i / FundamentalKDiv128B]);
    }
  }
}

// Combined LDS+STTM for Hybrid SliceK A (callable from generated code)
//
// Template parameters (in order):
//   MaxBS: token tile (token dimension tile size)
//   Ktile128BNum: tileK / 128 (number of 128B K-tiles per stage)
//   FundamentalKtile128BNum: 32/Mtile for Mtile < 32, else 1
//   Mtile: hidden tile (Weight W rows in SMEM, used for Load/Store)
//
// Semantic alignment: MaxBS = token tile, Mtile = hidden tile (M_tma in LLK).
// The function copies Weight W from SMEM to TMEM, where Weight has Mtile rows.
template <int MaxBS, int Ktile128BNum, int FundamentalKtile128BNum, int Mtile = 32>
inline __device__ void hybridSliceKCopyA(int warpIdx,
                                         int laneIdx,
                                         void const* smemPtr,
                                         uint32_t tmemBaseAddr,
                                         int tmemColOffset) {
  // Register buffer layout based on hidden tile (Mtile = Weight W rows in SMEM).
  static_assert(Mtile % 4 == 0, "Mtile must be a multiple of 4");
  static_assert(MaxBS % 4 == 0, "MaxBS must be a multiple of 4");
  constexpr int RegCount =
    (Mtile >= 32) ? (Ktile128BNum * Mtile / 32) : (Ktile128BNum / (2 * (4 / (Mtile / 4))));
  uint32_t regBuffer[RegCount][8];

  // Load from SMEM to registers.
  // SmemRows = Mtile (Weight W has Mtile rows in SMEM)
  hybridSliceKLoadA_perfpatch<Mtile, Ktile128BNum>(warpIdx, laneIdx, smemPtr, regBuffer);

  // Emit fence.proxy.async to ensure all preceding LDS instructions complete.
  // fence.proxy.async::generic.acquire emits scoreboard dependency on LDS,
  // allowing safe SMEM release to TMA without costly fence.proxy.async.shared::cta.
  hybridSliceKFenceProxyAsync();

  // Compute sub-partition offset for TMEM address.
  // Each warp writes to a different TMEM row range: warp N writes to rows N*32..(N+1)*32-1
  // The offset is: (sp_id * 32) << 16, where sp_id = warpIdx % 4
  int spId = warpIdx % 4;
  uint32_t spOffset = static_cast<uint32_t>(spId * 32) << 16;

  // Store from registers to TMEM using tcgen05_st_32x32b_x8.
  // Use Mtile for TMEM store to match actual data size loaded from SMEM.
  hybridSliceKStoreA_perfpatch<Mtile, Ktile128BNum>(regBuffer,
                                                    tmemBaseAddr + tmemColOffset + spOffset);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 6: Accumulator Initialization
//
// Clears TMEM accumulator region to zeros before the K-loop starts.
// Depends on: Section 2 (PTX primitives).
////////////////////////////////////////////////////////////////////////////////////////////////////

// Initialize TMEM accumulator region to zeros.
// Template parameters:
//   M_tma: hidden dimension tile - determines AccumCols expansion
//   MaxBS: token dimension tile
//   AccumCols: Number of accumulator columns = MaxBS * (M_tma > 32 ? 1 : 32/M_tma)
//
// In Hybrid SliceK, each warp uses a different sub-partition to access the accumulator.
// Each warp uses sp_id = warp_idx % 4 for isolated TMEM row ranges.
template <int M_tma, int MaxBS>
inline __device__ void hybridSliceKInitAccum(int warpIdx, uint32_t tmemAddrAccum) {
  // Compute sub-partition offset for this warp.
  int spId = warpIdx % 4;
  uint32_t spOffset = static_cast<uint32_t>(spId * 32) << 16;
  uint32_t tmemAddrWithSp = tmemAddrAccum + spOffset;

  // For Hybrid SliceK, accumulators are expanded when M_tma < 32.
  //   AccumCols_1set = ((MaxProblemN * (Mtile < 32 ? 32/Mtile : 1)) + 7) / 8 * 8
  // The (+7)/8*8 alignment is required: the MMA instruction's N dimension is
  // EffectiveInstN = ((MaxBS+7)/8*8) * extension, which can exceed MaxBS * extension
  // when MaxBS is not a multiple of 8.  Without alignment we'd leave TMEM columns
  // uncleared, causing the MMA to accumulate into garbage.
  constexpr int AccumColsFactor = (M_tma < 32) ? (32 / M_tma) : 1;
  constexpr int AccumCols = ((MaxBS * AccumColsFactor) + 7) / 8 * 8;

  // Each STTM 32x32b.x8 clears 8 columns (each column is 32 elements of 32-bit = 128 bytes)
  // Total columns to clear: AccumCols
  // We use x8 instruction which clears 8 columns per call
  constexpr int NumClearInsts = (AccumCols + 7) / 8;

  // NOTE: Do NOT use elect_one_sync() here!
  // The tcgen05.st.sync.aligned instruction has .sync.aligned modifier which means
  // it's a warp-level operation that requires ALL threads to participate.
  // The hardware internally elects one thread to perform the actual store.
  // Using elect_one_sync() would cause warp divergence and deadlock!
#pragma unroll
  for (int i = 0; i < NumClearInsts; ++i) {
    tcgen05_st_32x32b_x8_zero(tmemAddrWithSp + i * 8);
  }

  // Wait for TMEM stores to complete
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_wait_st();
#endif
}

// Version with explicit AccumCols parameter for advanced use cases
template <int AccumCols>
inline __device__ void hybridSliceKInitAccumExplicit(uint32_t tmemAddrAccum) {
  constexpr int NumClearInsts = (AccumCols + 7) / 8;

  // NOTE: Do NOT use elect_one_sync() here!
  // The tcgen05.st.sync.aligned instruction requires ALL threads to participate.
  // See hybridSliceKInitAccum for detailed explanation.
#pragma unroll
  for (int i = 0; i < NumClearInsts; ++i) {
    tcgen05_st_32x32b_x8_zero(tmemAddrAccum + i * 8);
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  cuda_ptx::tcgen05_wait_st();
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 7: MMA Execution
//
// The core compute stage: pre-computes WDM masks and operand addresses,
// then issues tcgen05.mma instructions.
// Depends on: Section 2 (PTX), Section 3 (Descriptor Creation), Section 4 (Config).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Config> struct HybridSliceKMmaExecutor {
  static constexpr int WarpNum = Config::WarpNum;
  static constexpr int Mtile = Config::Mtile;
  static constexpr int FundamentalKtileUnit = Config::FundamentalKtileUnit;
  static constexpr int TotalIterations = Config::TotalIterations;
  static constexpr int InstNum = Config::InstNum;
  static constexpr uint32_t WarpMask = Config::WarpMask;
  static constexpr int TmemACols_1inst = Config::TmemACols_1inst;
  static constexpr int ReRunTimes = Config::ReRunTimes;

  // Pre-computed masks for each instruction
  uint32_t mReRunMask[InstNum][4];
  // Pre-computed operand addresses
  uint32_t mOperandATmem[InstNum];
  uint64_t mMatBDesc[InstNum];

  int mWarpIdx;
  uint32_t mMmaDesc; // MMA instruction descriptor (32-bit, not 64-bit)
  int32_t mScaleD;   // scaleD uses int32_t to match PTX predicate comparison

  __device__ HybridSliceKMmaExecutor(int warpIdx,
                                     uint32_t mmaDesc,
                                     uint32_t tmemAddrA,
                                     uint64_t smemDescB,
                                     int32_t scaleD)
    : mWarpIdx(warpIdx)
    , mMmaDesc(mmaDesc)
    , mScaleD(scaleD) {
    // Pre-compute WDM masks: ~(WarpMask << run)
    // WDM mask bit "1" = lane disabled/masked out
#pragma unroll
    for (int iter = 0; iter < TotalIterations; iter += WarpNum * FundamentalKtileUnit) {
      int warpIter = iter + warpIdx;
      int run = warpIter % ReRunTimes;
      uint32_t mask = ~(WarpMask << run);
      int instIdx = iter / (WarpNum * FundamentalKtileUnit);
      mReRunMask[instIdx][0] = mask;
      mReRunMask[instIdx][1] = mask;
      mReRunMask[instIdx][2] = mask;
      mReRunMask[instIdx][3] = mask;
    }

    // Pre-compute operand addresses
    prepareArgs(tmemAddrA, smemDescB);
  }

  __device__ void prepareArgs(uint32_t tmemAddrA, uint64_t smemDescB) {
    union SmemDescOpt {
      uint64_t desc_ = 0;
      struct {
        uint32_t lo;
        uint32_t hi;
      };
    };

    SmemDescOpt smemDesc{smemDescB};

#pragma unroll
    for (int iter = 0; iter < TotalIterations; iter += WarpNum * FundamentalKtileUnit) {
      int warpIter = iter + mWarpIdx;
      // aTmemColumnIter = warp_iter / (128 / Mtile)
      int aTmemColumnIter = warpIter / (128 / Config::Mtile);
      int instIdx = iter / (WarpNum * FundamentalKtileUnit);

      mOperandATmem[instIdx] = tmemAddrA + aTmemColumnIter * TmemACols_1inst;
      mMatBDesc[instIdx] = smemDesc.desc_;

      // Advance SMEM descriptor: smem_desc.lo += (MaxBS * 128 * FundKtile) >> 4
      if constexpr (WarpNum == 4) {
        smemDesc.lo += ((Config::MaxBS * 128 * FundamentalKtileUnit) >> 4);
      } else {
        smemDesc.lo += (32 >> 4);
        if (iter % 4 == 3) {
          smemDesc.lo -= ((32 * 4) >> 4);
          smemDesc.lo += ((Config::MaxBS * 128 * FundamentalKtileUnit) >> 4);
        }
      }
    }
  }

  __device__ void execute(uint32_t tmemAddrAccum) {
#pragma unroll
    for (int iter = 0; iter < TotalIterations; iter += WarpNum * FundamentalKtileUnit) {
      if (electSync(~0u)) {
        int instIdx = iter / (WarpNum * FundamentalKtileUnit);
        tcgen05_mma_1sm_fp8_TS_warp_lane_mask(tmemAddrAccum,
                                              mOperandATmem[instIdx],
                                              mMatBDesc[instIdx],
                                              mMmaDesc,
                                              mScaleD,
                                              mReRunMask[instIdx][0],
                                              mReRunMask[instIdx][1],
                                              mReRunMask[instIdx][2],
                                              mReRunMask[instIdx][3]);
      }
    }
  }
};

// Original version with pre-computed descriptors (for advanced use cases)
// Mtile must be M_tma (weight N-direction tile), MaxBS must be TileM.
template <int Mtile, int Ktile128BNum, int MaxBS, int WarpNum = 4>
inline __device__ void hybridSliceKMmaWithDesc(int warpIdx,
                                               uint32_t mmaDesc,
                                               uint32_t tmemAddrA,
                                               uint64_t smemDescB,
                                               uint32_t tmemAddrAccum,
                                               int32_t scaleD) {
  using Config = HybridSliceKConfig<Mtile, Ktile128BNum, MaxBS, WarpNum>;
  HybridSliceKMmaExecutor<Config> executor(warpIdx, mmaDesc, tmemAddrA, smemDescB, scaleD);
  executor.execute(tmemAddrAccum);
}

// Simplified version that creates descriptors internally (matches LowLatency pattern)
// Template parameters:
//   Mtile: Tile size in M dimension (maps to N in LowLatency notation)
//   Ktile128BNum: Number of 128B K-tiles
//   MaxBS: Maximum batch size (TileM)
//   WarpNum: Number of warps for MMA (default 4)
//   InstM: MMA instruction M size (default 128)
//   InstN: MMA instruction N size (computed from Mtile and FundamentalKtileUnit)
template <int Mtile, int Ktile128BNum, int MaxBS, int WarpNum = 4, int InstM = 128>
inline __device__ void hybridSliceKMma(int warpIdx,
                                       uint32_t tmemAddrA,
                                       void const* smemPtrB,
                                       uint32_t tmemAddrAccum,
                                       uint32_t scaleD) {
  using Config = HybridSliceKConfig<Mtile, Ktile128BNum, MaxBS, WarpNum>;

  // Compute effective InstN: N_TC = ceil8(MaxBS), extended for small Mtile.
  constexpr int N_TC = ((MaxBS + 7) / 8) * 8;
  constexpr int EffectiveInstN = N_TC * ((Mtile > 32) ? 1 : (32 / Mtile));

  // Create MMA descriptor
  HybridInstrDescriptor mmaDescUnion;
  makeHybridUtcqmmaDesc<InstM, EffectiveInstN>(mmaDescUnion);

  // Create SMEM descriptor for B matrix (Activation X)
  // FP8+WDM mode: SW128B with 32B warp offset.
  // When Mtile == 64, apply warp remap: swap bit0 and bit1 of warp index.
  HybridSmemDescriptor smemDescUnion;
  void const* smemPtrBAdj = smemPtrB;
  if constexpr (Mtile == 64) {
    // Apply warp remap when Mtile (M_tma_pad) == 64: swap bit0 and bit1
    int smemWarpIdx = ((warpIdx & 1) << 1) | ((warpIdx & 2) >> 1);
    smemPtrBAdj =
      reinterpret_cast<void const*>(reinterpret_cast<uint8_t const*>(smemPtrB) + smemWarpIdx * 32);
  } else {
    // For Mtile != 64: simple warp * 32 offset
    smemPtrBAdj =
      reinterpret_cast<void const*>(reinterpret_cast<uint8_t const*>(smemPtrB) + warpIdx * 32);
  }
  makeHybridSmemDesc128B(smemDescUnion, smemPtrBAdj);

  // Compute sub-partition offset for both A operand and Accumulator.
  // Each warp uses sp_id = warp_idx % 4 for isolated TMEM row ranges.
  int spId = warpIdx % 4;
  uint32_t spOffset = static_cast<uint32_t>(spId * 32) << 16;
  uint32_t tmemAddrAWithSp = tmemAddrA + spOffset;
  uint32_t tmemAddrAccumWithSp = tmemAddrAccum + spOffset; // Accum also uses sp offset!

  // Execute MMA: Both A and Accum use sub-partition offset
  HybridSliceKMmaExecutor<Config> executor(warpIdx,
                                           mmaDescUnion.desc_, // uint32_t, not uint64_t
                                           tmemAddrAWithSp,    // A: with sub-partition offset
                                           smemDescUnion.desc_,
                                           static_cast<int32_t>(scaleD));
  executor.execute(tmemAddrAccumWithSp); // Accum: with sub-partition offset
}

// Version that takes resIdx to determine which K-slices this warp processes.
// In the 4-independent-MMA-tasks scheme (PLAN_ALT), each Task has its own resIdx (0,1,2,3).
// This replaces warpIdx-based slicing: instead of all 4 warps cooperating on one task,
// each resIdx processes every 4th K-slice: resIdx=0 handles K[0,4,8,...], resIdx=1 handles
// K[1,5,9,...], etc.
template <int Mtile, int Ktile128BNum, int MaxBS, int WarpNum = 4, int InstM = 128>
inline __device__ void hybridSliceKMmaWithResIdx(int resIdx,
                                                 uint32_t tmemAddrA,
                                                 void const* smemPtrB,
                                                 uint32_t tmemAddrAccum,
                                                 uint32_t scaleD) {
  using Config = HybridSliceKConfig<Mtile, Ktile128BNum, MaxBS, WarpNum>;

  // In the 4-independent-tasks scheme, each Task runs on exactly 1 warp.
  // So we use resIdx directly as the "warp index" within the Hybrid SliceK pattern.
  // This means resIdx=0 warp processes K-iterations [0, 4, 8, ...],
  // resIdx=1 warp processes [1, 5, 9, ...], etc.
  int warpIdx = resIdx;

  // Compute effective InstN for Hybrid mode.
  constexpr int N_TC = ((MaxBS + 7) / 8) * 8;
  constexpr int EffectiveInstN = N_TC * ((Mtile > 32) ? 1 : (32 / Mtile));

  // Create MMA descriptor
  HybridInstrDescriptor mmaDescUnion;
  makeHybridUtcqmmaDesc<InstM, EffectiveInstN>(mmaDescUnion);

  // Create SMEM descriptor for B matrix (same logic as hybridSliceKMma)
  // FP8+WDM mode: SW128B with 32B warp offset.
  HybridSmemDescriptor smemDescUnion;
  void const* smemPtrBAdj = smemPtrB;
  if constexpr (Mtile == 64) {
    // Apply warp remap when Mtile (M_tma_pad) == 64: swap bit0 and bit1
    int smemWarpIdx = ((warpIdx & 1) << 1) | ((warpIdx & 2) >> 1);
    smemPtrBAdj =
      reinterpret_cast<void const*>(reinterpret_cast<uint8_t const*>(smemPtrB) + smemWarpIdx * 32);
  } else {
    smemPtrBAdj =
      reinterpret_cast<void const*>(reinterpret_cast<uint8_t const*>(smemPtrB) + warpIdx * 32);
  }
  makeHybridSmemDesc128B(smemDescUnion, smemPtrBAdj);

  // Compute sub-partition offset for BOTH A operand AND Accumulator.
  // Must match hybridSliceKMma — each warp reads/writes its own TMEM rows.
  int spId = warpIdx % 4;
  uint32_t spOffset = static_cast<uint32_t>(spId * 32) << 16;
  uint32_t tmemAddrAWithSp = tmemAddrA + spOffset;
  uint32_t tmemAddrAccumWithSp = tmemAddrAccum + spOffset;

  // Execute MMA with computed descriptors
  HybridSliceKMmaExecutor<Config> executor(warpIdx,
                                           mmaDescUnion.desc_, // uint32_t, not uint64_t
                                           tmemAddrAWithSp,
                                           smemDescUnion.desc_,
                                           static_cast<int32_t>(scaleD));
  executor.execute(tmemAddrAccumWithSp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 8: Epilogue (TMEM to Global)
//
// Reads accumulators from TMEM, applies reduction and scaling,
// then stores results to global memory.
// Depends on: Section 2 (PTX primitives).
////////////////////////////////////////////////////////////////////////////////////////////////////

// Handle duplicated TMEM cols for Mtile < 32.
template <int Mtile, int MaxBS, bool IsHybridSliceK>
inline __device__ void hybridSliceKLoadAccum32x32bX8(uint32_t* rf, uint32_t tmemAddr) {
  if constexpr (IsHybridSliceK && (Mtile < 32)) {
    constexpr int duplicateNum = 32 / Mtile;
    int laneIdx = threadIdx.x % 32;
    int quadIdx = laneIdx / 4;

    uint32_t rfTemp[8];
    tcgen05_ld_32x32b_x8(rf, tmemAddr);

#pragma unroll
    for (int i = 1; i < duplicateNum; ++i) {
      tcgen05_ld_32x32b_x8(rfTemp, tmemAddr + MaxBS * i);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        rf[s] = (quadIdx % duplicateNum == i) ? rfTemp[s] : rf[s];
      }
    }
  } else {
    tcgen05_ld_32x32b_x8(rf, tmemAddr);
  }
}

// Epilogue: TMEM load + reduce + store for Hybrid SliceK (FP8).
// TODO: Implement DSMEM epilogue path for CGA split-K.
// Each warp uses sp_id = warp_idx % 4 for isolated accumulator access.
// Lane-level reduction handles row-folding for Mtile < 128.
template <typename OutType, int Mtile, int MaxBS, int CgaSplitK, int TransposeOut>
inline __device__ void hybridSliceKEpilogueStore(int warpIdx,
                                                 int tokenNums,
                                                 int N,
                                                 uint32_t tmemAddr,
                                                 float scale,
                                                 OutType* Z,
                                                 int mCtaOffset,
                                                 int nCtaOffset) {
  // Compute sub-partition offset for this warp (matches MMA and InitAccum)
  int spId = warpIdx % 4;
  uint32_t spOffset = static_cast<uint32_t>(spId * 32) << 16;
  uint32_t tmemAddrWithSp = tmemAddr + spOffset;

  float accumRegs[MaxBS];

#pragma unroll
  for (int i = 0; i < MaxBS; ++i) {
    accumRegs[i] = 0.0f;
  }

  // Read from warp's sub-partition of accumulator (matches MMA write)
  // TMEM column addressing: addr + m
  // Loop uses m = 0,8,16,24 (column offset in units of 32b elements).
#pragma unroll
  for (int m = 0; m < MaxBS; m += 8) {
    uint32_t rf[8];
    // Address = base with SP offset + m (column offset)
    hybridSliceKLoadAccum32x32bX8<Mtile, MaxBS, true>(rf, tmemAddrWithSp + m);
    float* rfFloat = reinterpret_cast<float*>(rf);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      accumRegs[m + i] = rfFloat[i];
    }

    // Lane-level reduction for Mtile < 128 (reduce_iteration = 64 / Mtile)
    if constexpr (Mtile < 128) {
      constexpr int reduceIteration = 64 / Mtile;
#pragma unroll
      for (int mi = 0; mi < 8; ++mi) {
#pragma unroll
        for (int r = reduceIteration; r > 0; r >>= 1) {
          accumRegs[m + mi] += __shfl_xor_sync(0xFFFFFFFF, accumRegs[m + mi], r);
        }
      }
    }
  }

  // Use mCtaOffset passed from caller (trtllm-gen tile scheduler provides correct tile index)
  // instead of hardcoded blockIdx.x which may not match actual tile assignment
  int mOffset = mCtaOffset;
  int localWarpIdx = warpIdx; // use warp-group local index (handles non-contiguous warps)
  int laneIdx = threadIdx.x % 32;
  int laneThreadOffset = laneIdx / (128 / Mtile);

  int nThreadOffset = localWarpIdx * Mtile / 4 + laneThreadOffset;

  cutlass::NumericConverter<OutType, float> convert;

  int nCtaOffsetLocal = nCtaOffset;

#pragma unroll
  for (int m = 0; m < MaxBS; ++m) {
    int midx = m + mOffset;
    if (midx < tokenNums && nThreadOffset < Mtile && (nCtaOffsetLocal + nThreadOffset) < N) {
      float scaledVal = accumRegs[m] * scale;
      OutType outVal = convert(scaledVal);
      if constexpr (TransposeOut) {
        int outIdx = (nCtaOffsetLocal + nThreadOffset) * tokenNums + midx;
        Z[outIdx] = outVal;
      } else {
        int outIdx = midx * N + nCtaOffsetLocal + nThreadOffset;
        Z[outIdx] = outVal;
      }
    }
  }
}

// Backward-compatible wrappers for E4m3 and E5m2
template <int Mtile, int MaxBS, int CgaSplitK, int TransposeOut>
inline __device__ void hybridSliceKEpilogueStoreE4m3(int warpIdx,
                                                     int tokenNums,
                                                     int N,
                                                     uint32_t tmemAddr,
                                                     float scale,
                                                     cutlass::float_e4m3_t* Z,
                                                     int mCtaOffset,
                                                     int nCtaOffset) {
  hybridSliceKEpilogueStore<cutlass::float_e4m3_t, Mtile, MaxBS, CgaSplitK, TransposeOut>(
    warpIdx,
    tokenNums,
    N,
    tmemAddr,
    scale,
    Z,
    mCtaOffset,
    nCtaOffset);
}

template <int Mtile, int MaxBS, int CgaSplitK, int TransposeOut>
inline __device__ void hybridSliceKEpilogueStoreE5m2(int warpIdx,
                                                     int tokenNums,
                                                     int N,
                                                     uint32_t tmemAddr,
                                                     float scale,
                                                     cutlass::float_e5m2_t* Z,
                                                     int mCtaOffset,
                                                     int nCtaOffset) {
  hybridSliceKEpilogueStore<cutlass::float_e5m2_t, Mtile, MaxBS, CgaSplitK, TransposeOut>(
    warpIdx,
    tokenNums,
    N,
    tmemAddr,
    scale,
    Z,
    mCtaOffset,
    nCtaOffset);
}

// TMEM Reduction structure for different Ntile values
// Uses recursive reduction: 128 -> 64 -> 32 -> 16 -> 8 -> 4
template <int Ntile, int MaxBS> struct HybridSliceKTmemReduce {
  // Number of elements per thread in N dimension
  static constexpr int ElementsInNPerThread = (Ntile >= 64) ? 2 : 1;
  static constexpr int BSPerThread = MaxBS / 8;

  float mAccum[BSPerThread][ElementsInNPerThread];

  __device__ void loadFromTmem(uint32_t tmemAddr) {
    uint32_t rf[8];
    hybridSliceKLoadAccum32x32bX8<Ntile, MaxBS, true>(rf, tmemAddr);

    // Reinterpret as floats
    float* rfFloat = reinterpret_cast<float*>(rf);

#pragma unroll
    for (int b = 0; b < BSPerThread; ++b) {
#pragma unroll
      for (int n = 0; n < ElementsInNPerThread; ++n) {
        mAccum[b][n] = rfFloat[b * ElementsInNPerThread + n];
      }
    }
  }

  // Reduce within Hybrid SliceK (sum adjacent elements)
  __device__ void reduce() {
    // For Ntile < 128, we need to sum adjacent elements
    // The number of elements to sum depends on 128/Ntile
    constexpr int ReductionFactor = 128 / Ntile;
    if constexpr (ReductionFactor > 1) {
      int laneIdx = threadIdx.x % 32;
#pragma unroll
      for (int b = 0; b < BSPerThread; ++b) {
#pragma unroll
        for (int n = 0; n < ElementsInNPerThread; ++n) {
          // Sum across ReductionFactor lanes
          float sum = mAccum[b][n];
#pragma unroll
          for (int r = 1; r < ReductionFactor; ++r) {
            int srcLane = (laneIdx + r) % 32;
            sum += __shfl_sync(0xFFFFFFFF, mAccum[b][n], srcLane);
          }
          // Only the first lane in each group keeps the result
          if ((laneIdx % ReductionFactor) == 0) {
            mAccum[b][n] = sum;
          }
        }
      }
    }
  }

  __device__ void applyScale(float scale) {
#pragma unroll
    for (int b = 0; b < BSPerThread; ++b) {
#pragma unroll
      for (int n = 0; n < ElementsInNPerThread; ++n) {
        mAccum[b][n] *= scale;
      }
    }
  }

  __device__ void storeToRF(float* outRF) {
#pragma unroll
    for (int b = 0; b < BSPerThread; ++b) {
#pragma unroll
      for (int n = 0; n < ElementsInNPerThread; ++n) {
        outRF[b * ElementsInNPerThread + n] = mAccum[b][n];
      }
    }
  }
};

// Main TMEM reduction function (callable from generated code)
template <int Ntile, int MaxBS>
inline __device__ void hybridSliceKTmemReduce(int laneIdx,
                                              uint32_t tmemAddrAccum,
                                              float* accumRF,
                                              float decodeScale = 1.0f) {
  HybridSliceKTmemReduce<Ntile, MaxBS> reducer;
  reducer.loadFromTmem(tmemAddrAccum);
  reducer.reduce();
  reducer.applyScale(decodeScale);
  reducer.storeToRF(accumRF);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 9: TMA Loading (Weight W)
//
// TMA-based loading for Weight W with Hybrid SliceK layout.
// For M_tma < 32: multiple TMA loads to achieve FundamentalKtile-adjacent SMEM layout.
// For M_tma >= 32: single TMA load.
// Depends on: cuda_ptx (external).
////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper: compute FundamentalKtile128BNum based on M_tma
template <int M_tma> struct HybridFundamentalKtile {
  static constexpr int value = (M_tma >= 32) ? 1 : (M_tma == 16 ? 2 : (M_tma == 8 ? 4 : 8));
};

// W TMA load for Hybrid SliceK (FP8 version)
// M_tma: M tile size (maps to mTileM in trtllm-gen)
// K_tile_div_128B: number of 128B K-slices per tile
template <int M_tma, int K_tile_div_128B>
__device__ __forceinline__ void hybridSliceKLoadW(void* smemPtr,
                                                  CUtensorMap const* tmaDesc,
                                                  uint64_t* barrier,
                                                  int32_t kOffset,
                                                  int32_t mOffset) {
  constexpr int FundKtile = HybridFundamentalKtile<M_tma>::value;
  constexpr int SmemBytesPerKSlice = M_tma * 128;
  if (cute::elect_one_sync()) {
    // TMA expects K coord in 128B units for W (Hybrid SliceK).
    int32_t kOffset128 = kOffset / 128;
    if constexpr (M_tma >= 32) {
      // Single TMA load for M_tma >= 32
      // Coords: {0, m, k} where kOffset is in 128B units
      int32_t coord[3] = {0, mOffset, kOffset128};
      cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                     cuda_ptx::space_global_t{},
                                     smemPtr,
                                     tmaDesc,
                                     coord,
                                     barrier);
    } else {
      // Multiple TMA loads for M_tma < 32
      uint8_t* smemBytePtr = reinterpret_cast<uint8_t*>(smemPtr);

#pragma unroll
      for (int ki = 0; ki < K_tile_div_128B; ki += FundKtile) {
        int32_t coord[3] = {0, kOffset128 + ki, mOffset};
        void* smemDst = smemBytePtr + ki * SmemBytesPerKSlice;
        cuda_ptx::cp_async_bulk_tensor(cuda_ptx::space_cluster_t{},
                                       cuda_ptx::space_global_t{},
                                       smemDst,
                                       tmaDesc,
                                       coord,
                                       barrier);
      }
    }
  }
}

} // namespace dev
} // namespace trtllm
