/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
 *
 * This file contains constants that decoderXQA*.{h,cpp} need.
 */
#pragma once
#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cstdlib>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
inline constexpr int kMinHistoryTokensPerBlock = 128;

inline constexpr float kEnableMinBlockFactor = 4.0;
inline constexpr int kTargetWaveFactor = 8;

inline constexpr int kXqaMlaCgaSize = 4;
inline constexpr int kXqaMlaTokensPerTile = 64;
inline constexpr int kXqaMlaMultiBlockMinTilesPerCta = 2;
inline constexpr int kXqaMlaKernelMinHeadGrpSize = 32;
inline constexpr int kXqaMlaKernelHeadGrpSize = 128;
inline constexpr int kXqaMlaQElemBytes = 1;
inline constexpr int kXqaMlaOutputHeadSize = 512;
inline constexpr int kXqaMlaOutputElemBytes = 2;

inline constexpr int getXqaMlaKernelHeadGrpSize(int runtimeHeadGrpSize)
{
    return runtimeHeadGrpSize <= kXqaMlaKernelMinHeadGrpSize ? kXqaMlaKernelMinHeadGrpSize
        : (runtimeHeadGrpSize <= 64 ? 64 : kXqaMlaKernelHeadGrpSize);
}

inline int getXqaMlaRuntimeKernelHeadGrpSize(int runtimeHeadGrpSize)
{
    int const kernelHeadGrpSize = getXqaMlaKernelHeadGrpSize(runtimeHeadGrpSize);
    if (kernelHeadGrpSize != kXqaMlaKernelMinHeadGrpSize)
    {
        return kernelHeadGrpSize;
    }

    // The 32-head specialization is correct in standalone tests, but current full-model warmup is pathologically slow.
    // Keep production selection on the known-good 64-head specialization unless the experimental path is requested.
    char const* enable32 = std::getenv("XQA_MLA_ENABLE_HEAD_GRP_32");
    char const* disable32 = std::getenv("XQA_MLA_DISABLE_HEAD_GRP_32");
    bool const use32 = enable32 != nullptr && enable32[0] == '1' && enable32[1] == '\0';
    bool const disable32Requested = disable32 != nullptr && disable32[0] == '1' && disable32[1] == '\0';
    return use32 && !disable32Requested ? kXqaMlaKernelMinHeadGrpSize : 64;
}

inline constexpr uint32_t getXqaMlaCgaXBufSize(int kernelHeadGrpSize)
{
    return static_cast<uint32_t>(kernelHeadGrpSize) * 144U;
}

inline constexpr uint32_t getXqaMlaPartialResultSize(int kernelHeadGrpSize)
{
    return static_cast<uint32_t>(kernelHeadGrpSize) * 1032U;
}

inline constexpr uint32_t getXqaMlaPartialResultChunks(int kernelHeadGrpSize)
{
    return static_cast<uint32_t>(kernelHeadGrpSize / 32);
}

// For multi-block mode. We reserve workspace for this amount of sub-sequences.
// This should be enough. Huge batch size may result in larger value, but for large batch size,
// multi-block mode is not useful. For llama v2 70b, 6000 results in ~12MB multi-block
// workspace, and is enough for > 10 waves.
inline constexpr int getXqaMaxNumSubSeq(bool isMLA)
{
    constexpr int kXQA_MAX_NUM_SUB_SEQ = 6000;
    constexpr int kXQA_MLA_MAX_NUM_SUB_SEQ = 500;
    return isMLA ? kXQA_MLA_MAX_NUM_SUB_SEQ : kXQA_MAX_NUM_SUB_SEQ;
}

} // namespace kernels

TRTLLM_NAMESPACE_END
