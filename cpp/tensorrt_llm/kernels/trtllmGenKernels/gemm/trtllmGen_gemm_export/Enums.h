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

} // namespace gemm

} // namespace gemm
