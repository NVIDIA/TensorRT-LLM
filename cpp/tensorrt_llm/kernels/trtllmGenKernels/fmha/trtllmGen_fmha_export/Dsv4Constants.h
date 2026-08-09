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

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

// DSv4 fused inverse-RoPE + FP8 quant ABI constants.
//
// Logical FMHA tensors:
//   Q:         [sumOfSeqLensQ,  numHeadsQ,  headDimQk]
//   K:         [sumOfSeqLensKv, numHeadsKv, headDimQk]
//   V:         [sumOfSeqLensKv, numHeadsKv, headDimV]
//   logical O: [sumOfSeqLensQ,  numHeadsQ,  headDimV]
//
// Fused epilogue outputs:
//   FP8 O:     [numHeadsQ / headsPerGroup, sumOfSeqLensQ, headsPerGroup, headDimV]
//   FP32 scale: [numHeadsQ / headsPerGroup,
//               headsPerGroup * headDimV / quantGroupSize,
//               scaleBufM]
//
// Inverse-RoPE inputs:
//   positionIds:   [sumOfSeqLensQ]
//   cosSin cache:  [maxPosition, cosSinStride], with each row laid out as
//                  [cos(ropeHalf), sin(ropeHalf)].
//
// Dimension mapping:
//   headDimQk is the last dimension of Q/K.
//   headDimV is the last dimension of V, logical O, and FP8 O.
//   headDimPerCtaV is the V/O head-dimension slice per CTA; clusterDimX=2 covers headDimV.
//   headsPerGroup is FP8 O dim2 and the head-in-group factor in FP32 scale dim1.
//   quantGroupSize is the 128-column headDimV block; FP32 scale has one value per block.
//   scaleBufM is FP32 scale dim2, the physical token stride for the scale tensor.
//   ropeStart/ropeHalf cover headDimV[448:512); ropeOffsetInBlock is the offset inside that block.

inline constexpr int32_t kDsv4HeadDimQk = 512;
inline constexpr int32_t kDsv4HeadDimV = 512;
inline constexpr int32_t kDsv4HeadDimPerCtaV = 256;
inline constexpr int32_t kDsv4HeadsPerGroup = 8;

inline constexpr int32_t kDsv4QuantGroupSize = 128;
inline constexpr int32_t kDsv4Log2QuantGroupSize = 7; // log2(128)
inline constexpr int32_t kDsv4RopeStart = 448;
inline constexpr int32_t kDsv4RopeHalf = 32;
inline constexpr int32_t kDsv4RopeOffsetInBlock = kDsv4RopeStart % kDsv4QuantGroupSize;
inline constexpr int32_t kDsv4CosSinStride = kDsv4RopeHalf * 2;

static_assert(kDsv4HeadDimV % kDsv4QuantGroupSize == 0);
static_assert((kDsv4QuantGroupSize & (kDsv4QuantGroupSize - 1)) == 0);
static_assert(kDsv4RopeOffsetInBlock + kDsv4RopeHalf * 2 == kDsv4QuantGroupSize);

} // namespace fmha
