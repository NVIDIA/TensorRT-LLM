// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once

#include "cute/tensor.hpp"

#include "cute/arch/mma_sm100_umma.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"

#if !defined(FMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT)
#error "The BF16X prefill JIT must define FMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT"
#endif
#if !defined(FMHA_SM100_PREFILL_BF16X_SM107)
#error "The BF16X prefill JIT must define FMHA_SM100_PREFILL_BF16X_SM107"
#endif
// Data-path selector: 0 = E4M3 Q / E4M3 dequantized K,V / E4M3 P (kind::f8f6f4, the upstream
// Q8KV4 path), 1 = BF16 Q / BF16 dequantized K,V / BF16 P (kind::f16). The JIT passes it only for
// the BF16 extension, so an E4M3 build sees exactly the upstream configuration.
#if !defined(FMHA_SM100_PREFILL_BF16X_DTYPE)
#define FMHA_SM100_PREFILL_BF16X_DTYPE 0
#endif
#if FMHA_SM100_PREFILL_BF16X_DTYPE != 0 && FMHA_SM100_PREFILL_BF16X_DTYPE != 1
#error "FMHA_SM100_PREFILL_BF16X_DTYPE must be 0 (E4M3) or 1 (BF16)"
#endif

namespace fmha_sm100::prefill_bf16x::detail {

inline constexpr bool kBf16DataPath = FMHA_SM100_PREFILL_BF16X_DTYPE == 1;
#if !defined(FMHA_SM100_PREFILL_BF16X_E4M3X)
#define FMHA_SM100_PREFILL_BF16X_E4M3X 0
#endif
// E4M3X: the E4M3 data path transcribed op for op to the CuTe-DSL fp8 forward (the in-model VR path), the bitwise
// target: K / V dequant with the fp32 global scale before the E4M3 conversion, no P range scaling, the DSL softmax /
// epilogue / LSE roundings (the BF16X M3 set). Off (0) = the upstream Q8KV4 E4M3 kernel, unchanged.
inline constexpr bool kE4M3X = !kBf16DataPath && FMHA_SM100_PREFILL_BF16X_E4M3X == 1;
inline constexpr bool kExactPath = kBf16DataPath || kE4M3X;

#if FMHA_SM100_PREFILL_BF16X_DTYPE == 1
using Element = cutlass::bfloat16_t;
#else
using Element = cutlass::float_e4m3_t;
#endif
using Accumulator = float;

inline constexpr int kQHeadsPerKv = 16;
inline constexpr int kHeadDim = 128;
inline constexpr int kPageSize = 128;
inline constexpr int kQueriesPerGroup = 128 / kQHeadsPerKv;
// SMEM budget (227 KB = 232448 B of dynamic SMEM per CTA). One Q stage is a group of
// kQueriesPerGroup = 8 tokens x 16 heads x 128 dims:
//   E4M3: 8 x 16 x 128 x 1 B = 16 KB per stage, 8 stages = 128 KB; K + V tiles 2 x 16 KB.
//   BF16: 8 x 16 x 128 x 2 B = 32 KB per stage, 4 stages = 128 KB; K + V tiles 2 x 32 KB.
// Common: packed K/V 2 x 8 KB, K/V block scales 2 x 1 KB, qsplit ring 1 KB, row_sum/row_max
// 2 KB, barriers + tmem_base + work tile ~0.4 KB; sizeof(SharedStorage) rounds up to 1 KB.
//   E4M3: 131072 + 32768 + 16384 + 2048 + ~3.4 K = 185748 B -> sizeof 186368 B (182 KB).
//   BF16: 131072 + 65536 + 16384 + 2048 + ~3.4 K = 218452 B -> sizeof 219136 B (214 KB).
// A fifth BF16 stage would need 251904 B > 232448 B, so BF16 runs 4 stages (13 KB headroom).
// The launch static_asserts sizeof(SharedStorage) <= 232448.
inline constexpr int kQStages = kBf16DataPath ? 4 : 8;
inline constexpr int kScoreStages = 2;
inline constexpr int kQMetadataStages = 32;
// Block-scale staging: the dequant divides each E4M3 block scale by 2^kBlockScaleShift so that
// code x scale fits E4M3 for full-range (TransformerEngine-style) scales; the host folds 2^shift
// back into the softmax and output scales. Each shift is its own extension.
inline constexpr int kBlockScaleShift = FMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT;
static_assert(kBlockScaleShift >= 0 && kBlockScaleShift <= 7, "block-scale shift out of range");
// The BF16 dequant is exact (no requantization), so it has no block-scale staging.
static_assert(!kBf16DataPath || kBlockScaleShift == 0, "the BF16 data path requires shift 0");
static_assert(!kE4M3X || kBlockScaleShift == 0, "the E4M3X data path requires shift 0");

inline constexpr int kTmemScoreOffset = 0;
inline constexpr int kTmemScoreStride = 128;
inline constexpr int kTmemOutputOffset = 256;
inline constexpr int kTmemOutputStride = 128;
// P aliases the upper columns of its S stage: E4M3 P (4 per 32-bit column) takes 32 columns at
// +96, BF16 P (2 per column) takes 64 columns at +64. The softmax reads all 128 S columns into
// registers before it writes P.
inline constexpr int kTmemProbabilityOffset = kBf16DataPath ? 64 : 96;
inline constexpr int kTmemProbabilityColumns = kPageSize * cute::sizeof_bits_v<Element> / 32;
static_assert(kTmemProbabilityOffset + kTmemProbabilityColumns == kTmemScoreStride);
inline constexpr int kTmemColumns = 512;

inline constexpr int kPackedKvBytes = kPageSize * kHeadDim / 2;
inline constexpr int kScaleBytes = kPageSize * kHeadDim / 16;

// Warp roles follow the FA4 convention: define ownership once and dispatch
// each role from a small top-level kernel body.
struct WarpSpecializationSm103 {
  static constexpr int kSoftmax0Warp = 0;
  static constexpr int kSoftmax1Warp = 4;
  static constexpr int kEpilogueWarp = 8;
  static constexpr int kMmaWarp = 12;
  static constexpr int kLoadWarp = 13;

  static constexpr int kSoftmaxWarps = 4;
  static constexpr int kEpilogueWarps = 4;
  static constexpr int kLoadWarps = 3;
  static constexpr int kWarps = 16;
  static constexpr int kThreads = kWarps * 32;
  static constexpr int kTmemParticipantWarps = kMmaWarp + 1;

  static constexpr int kLoadKWarp = kLoadWarp;
  static constexpr int kLoadVWarp = kLoadWarp + 1;
  static constexpr int kQLoadWarps = kLoadWarps;
  static constexpr int kQLoadThreads = kQLoadWarps * 32;

  CUTLASS_HOST_DEVICE static constexpr bool is_softmax0_warp(int warp_idx) {
    return warp_idx >= kSoftmax0Warp && warp_idx < kSoftmax0Warp + kSoftmaxWarps;
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_softmax1_warp(int warp_idx) {
    return warp_idx >= kSoftmax1Warp && warp_idx < kSoftmax1Warp + kSoftmaxWarps;
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_softmax_warp(int warp_idx) {
    return is_softmax0_warp(warp_idx) || is_softmax1_warp(warp_idx);
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_epilogue_warp(int warp_idx) {
    return warp_idx >= kEpilogueWarp && warp_idx < kMmaWarp;
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_mma_warp(int warp_idx) {
    return warp_idx == kMmaWarp;
  }

  CUTLASS_HOST_DEVICE static constexpr bool is_load_warp(int warp_idx) {
    return warp_idx >= kLoadWarp && warp_idx < kLoadWarp + kLoadWarps;
  }

  CUTLASS_HOST_DEVICE static constexpr int softmax_stage(int warp_idx) {
    return is_softmax0_warp(warp_idx) ? 0 : 1;
  }

  CUTLASS_HOST_DEVICE static constexpr int softmax_warp_base(int stage) {
    return stage == 0 ? kSoftmax0Warp : kSoftmax1Warp;
  }
};

using WarpSpecialization = WarpSpecializationSm103;

static_assert(WarpSpecialization::kSoftmax0Warp == 0);
static_assert(WarpSpecialization::kSoftmax1Warp ==
              WarpSpecialization::kSoftmax0Warp + WarpSpecialization::kSoftmaxWarps);
static_assert(WarpSpecialization::kEpilogueWarp ==
              WarpSpecialization::kSoftmax1Warp + WarpSpecialization::kSoftmaxWarps);
static_assert(WarpSpecialization::kMmaWarp ==
              WarpSpecialization::kEpilogueWarp + WarpSpecialization::kEpilogueWarps);
static_assert(WarpSpecialization::kLoadWarp == WarpSpecialization::kMmaWarp + 1);
static_assert(WarpSpecialization::kLoadWarp + WarpSpecialization::kLoadWarps ==
              WarpSpecialization::kWarps);

inline constexpr int kSoftmax0Registers = 176;
inline constexpr int kSoftmax1Registers = 176;
inline constexpr int kEpilogueRegisters = 96;
inline constexpr int kOtherRegisters = 64;
static_assert(kSoftmax0Registers + kSoftmax1Registers + kEpilogueRegisters + kOtherRegisters <=
              512);

inline constexpr int kKvDequantKBarrierId = 6;
inline constexpr int kKvDequantVBarrierId = 7;
inline constexpr int kTmemBarrierId = 0;
inline constexpr int kQLoadBarrierId = 1;

// One Q TMA box: a token's 16 heads x kQTmaCols dims. An SW128 K-major atom row is 128 bytes,
// i.e. all 128 E4M3 dims but only 64 BF16 dims; the MMA's stage layout puts the two 64-dim BF16
// k-atoms of all 128 rows 16 KB apart, so a BF16 token is loaded as two boxes (one per k-atom).
inline constexpr int kQTmaCols = kBf16DataPath ? kHeadDim / 2 : kHeadDim;
inline constexpr int kQTmaKBlocks = kHeadDim / kQTmaCols;

using QTokenSmemLayout = decltype(cute::coalesce(
    cute::tile_to_shape(cute::UMMA::Layout_K_SW128_Atom<Element>{},
                        cute::Shape<cute::Int<kQHeadsPerKv>, cute::Int<kQTmaCols>>{},
                        cute::Step<cute::_1, cute::_2>{}),
    cute::Shape<cute::_1, cute::_1>{}));

} // namespace fmha_sm100::prefill_bf16x::detail
