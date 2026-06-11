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

inline constexpr int32_t kDsv4HeadDimQk = 512;
inline constexpr int32_t kDsv4HeadDimV = 512;
inline constexpr int32_t kDsv4HeadDimPerCtaV = 256;
inline constexpr int32_t kDsv4HeadsPerGroup = 8;

inline constexpr int32_t kDsv4QuantGroupSize = 128;
inline constexpr int32_t kDsv4RopeStart = 448;
inline constexpr int32_t kDsv4RopeHalf = 32;
inline constexpr int32_t kDsv4RopeOffsetInChunk = kDsv4RopeStart % kDsv4QuantGroupSize;
inline constexpr int32_t kDsv4CosSinStride = kDsv4RopeHalf * 2;

static_assert(kDsv4HeadDimV % kDsv4QuantGroupSize == 0);
static_assert((kDsv4QuantGroupSize & (kDsv4QuantGroupSize - 1)) == 0);
static_assert(kDsv4RopeOffsetInChunk + kDsv4RopeHalf * 2 == kDsv4QuantGroupSize);

} // namespace fmha
