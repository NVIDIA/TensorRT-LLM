/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

namespace ada_blockwise_gemm
{

// Shared tile and scale configuration
constexpr int kBlockM = 128;
constexpr int kBlockK = 64;
constexpr int kSmemPad = 8;
constexpr int kPaddedK = kBlockK + kSmemPad;
constexpr int kFragM = 16;
constexpr int kFragN = 16;
constexpr int kFragK = 16;
constexpr int kVectorWidth = 16;
constexpr int kScaleGranularityM = 1;
constexpr int kScaleGranularityN = 128;
constexpr int kScaleGranularityK = 128;
constexpr int kMaxBlockN = 128;

template <typename T>
__host__ __device__ constexpr T ceil_div(T value, T divisor)
{
    return (value + divisor - 1) / divisor;
}

template <typename T>
__host__ __device__ constexpr T align_to(T value, T alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

struct DeepGemmLaunchConfig
{
    int M = 0;
    int N = 0;
    int K = 0;
    int scale_m_stride = 0;
    int scale_n_stride = 0;
    int scale_k_tiles = 0;
    int use_wide = 0;
};

__host__ __device__ inline bool prefer_wide_tile(int m, int n)
{
    if (m <= 0 || n <= 0)
    {
        return false;
    }
    const int bigger = (m > n) ? m : n;
    const int smaller = (m > n) ? n : m;
    if (bigger < 4096)
    {
        return false;
    }
    return static_cast<long long>(bigger) * 2 >= static_cast<long long>(smaller) * 3;
}

template <int BlockN, int WarpN>
struct TileTraits
{
    static constexpr int kBlockN = BlockN;
    static constexpr int kWarpN = WarpN;
    static constexpr int kWarpM = 64;
    static constexpr int kWarpTilesM = kBlockM / kWarpM;
    static constexpr int kWarpTilesN = kBlockN / kWarpN;
    static constexpr int kWarpsPerBlock = kWarpTilesM * kWarpTilesN;
    static constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
    static_assert(kBlockN % kWarpN == 0, "Warp tiles must divide block N");
};

using NarrowTile = TileTraits<64, 16>;
using WideTile = TileTraits<128, 32>;
constexpr int kThreadsPerBlock = NarrowTile::kThreadsPerBlock;
constexpr int kMaxWarps = WideTile::kWarpsPerBlock;
static_assert(kThreadsPerBlock == WideTile::kThreadsPerBlock, "Threadblock shape mismatch");

__host__ __device__ inline int block_n_for_config(bool use_wide)
{
    return use_wide ? WideTile::kBlockN : NarrowTile::kBlockN;
}

constexpr size_t shared_storage_bytes()
{
    return static_cast<size_t>(kBlockM) * kPaddedK * sizeof(short) + static_cast<size_t>(kMaxBlockN) * kPaddedK * sizeof(short)
        + static_cast<size_t>(kMaxWarps) * kFragM * kFragN * sizeof(float);
}

} // namespace ada_blockwise_gemm
