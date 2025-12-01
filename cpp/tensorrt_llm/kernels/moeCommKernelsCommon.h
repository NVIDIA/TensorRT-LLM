/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <stdint.h>

namespace tensorrt_llm
{
namespace kernels
{

// ============================================================================
// Alignment Macro
// ============================================================================

#ifdef __CUDACC__
#define ALIGN_256 __align__(256)
#else
#define ALIGN_256 alignas(256)
#endif

// ============================================================================
// Warp Constants
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr uint32_t WARP_MASK = 0xffffffff;

// ============================================================================
// Memory Block Constants
// ============================================================================

// Size of a 128-byte aligned block (used for bulk async copies)
constexpr int BYTES_PER_128B_BLOCK = 128;

// Size of a 16-byte aligned block (used for field alignment)
constexpr int BYTES_PER_16B_BLOCK = 16;

// Number of int elements per 128-byte block
constexpr int INTS_PER_128B_BLOCK = BYTES_PER_128B_BLOCK / sizeof(int);

// Number of uint64_t elements per 128-byte block
constexpr int UINT64_PER_128B_BLOCK = BYTES_PER_128B_BLOCK / sizeof(uint64_t);

// ============================================================================
// Block Organization Constants
// ============================================================================

// Maximum number of groups (warps) per CTA for MoE communication kernels
constexpr int MAX_GROUP_COUNT_PER_BLOCK = 8;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Ceiling division: compute ceil(a / b) for integers
 */
template <typename T>
inline constexpr T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

/**
 * Align value up to nearest multiple of alignment
 */
template <typename T>
inline constexpr T align_up(T value, T alignment)
{
    return ceil_div(value, alignment) * alignment;
}

// ============================================================================
// MoE Parallel Info Structures
// ============================================================================

struct MoeEpWorldInfo
{
    int epSize;
    int epRank;
};

struct MoeExpertParallelInfo
{
    int expertCount = -1;
    int topK = 1;
};

} // namespace kernels
} // namespace tensorrt_llm
