/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#pragma once

#include <cstdint>

namespace gemm
{

namespace gemm
{

enum class AllReduceAlgo : uint32_t
{
    // Does not apply all-reduce.
    None = 0,
    // Reduction occurs at L2 cache; pulls N-1 partial outputs from peer devices. Result is
    // non-deterministic. Potentially lower latency at cost of higher memory traffic.
    OneShot,
    // Reduction occurs at switch; pulls 1/Nth of the output from switch (reduce-scatter phase) and
    // store to multicast mem (all-gather phase). Result is deterministic. Lower memory traffic at
    // cost of potentially higher latency.
    TwoShot,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class MatrixLayout
{
    // K-major layout (default). [Mn, K]
    MajorK = 0,
    // M-major for A and N-major for B. [K, Mn]
    MajorMn,
    // Layout is blocked along the K dimension as seen in the diagram below. [K / blockK, Mn, blockK]
    // where blockK is fixed at 128B
    //
    //         ├──────────────  K  ──────────────┤
    //  ┬  ┬   ├──── K block ───┤
    //  │  │   │ 0   1   2   3  ║ 32  33  34  35 │
    //  │ CTA0 │ 4   5   6   7  ║ 36  37  38  39 │
    //  │  │   │ 8   9   10  11 ║ 40  41  42  43 │
    //  │  ┴   │ 12  13  14  15 ║ 44  45  46  47 │
    //  M  ┬   ├────────────────║────────────────┤
    //  │  │   │ 16  17  18  19 ║ 48  49  50  51 │
    //  │ CTA1 │ 20  21  22  23 ║ 52  53  54  55 │
    //  │  │   │ 24  25  26  27 ║ 56  57  58  59 │
    //  ┴  ┴   │ 28  29  30  31 ║ 60  61  62  63 │
    BlockMajorK
};

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class SplitK : uint32_t
{
    // No split-k is needed. I.e. mNumSlicesForSplitK == 1.
    None = 0,
    // CTAs computing one MN tile save partial results to global memory.
    // Then wait on the barrier and the last CTA in the group loads partial results from gmem,
    // sums them up and writes back to gmem.
    Gmem,
    // All CTAs in one CGA calculate partial sums. Then send the results to the smem of
    // the last CTA in the CGA, which sums them up and writes to gmem.
    Dsmem,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class BiasType : uint32_t
{
    // No bias.
    None = 0,
    // One bias value per N of the output tensor.
    M = 1,
    // One bias value per row M of the output tensor.
    N = 2,
    // One bias value for each element of the output tensor.
    Mn = 3,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class TileScheduler
{
    // Static scheduler (Non-persistent).
    Static = 0,
    // Dynamic persistent scheduler. This is either based on an atomically incremented global work id
    // prior to SM100 archs, or the HW supported work id scheduler based on UGETNEXTWORKID for SM100+.
    Persistent,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the SplitK type.

#define SPLIT_K_FUNCTION(Mode)                                                                                         \
    inline bool doesSplitKUse##Mode(SplitK mode)                                                                       \
    {                                                                                                                  \
        return (mode == SplitK::Mode);                                                                                 \
    }

SPLIT_K_FUNCTION(Gmem)
SPLIT_K_FUNCTION(Dsmem)

#undef SPLIT_K_FUNCTION

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the Bias type.

#define BIAS_TYPE_FUNCTION(Mode)                                                                                       \
    inline bool isBiasType##Mode(BiasType type)                                                                        \
    {                                                                                                                  \
        return (type == BiasType::Mode);                                                                               \
    }

BIAS_TYPE_FUNCTION(None)
BIAS_TYPE_FUNCTION(N)
BIAS_TYPE_FUNCTION(M)
BIAS_TYPE_FUNCTION(Mn)

#undef BIAS_TYPE_FUNCTION

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm

} // namespace gemm
