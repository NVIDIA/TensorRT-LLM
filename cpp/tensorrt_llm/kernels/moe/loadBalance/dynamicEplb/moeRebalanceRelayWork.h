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

// Work assignment of the HALO-R relay copy.
//
// A broadcast b sends one expert bundle (every weight plane of one expert) from its owner to the destination set D_b,
// with m = |D_b| and the owner not in D_b. The bundle is cut into slices: each plane, in plane order, into
// ceil(bytes / kRelaySliceBytes) slices, the last one short. The destination with index i = |{d in D_b : d < rank}|
// pulls every slice j with j mod m == i from the owner and stores it into the helper slot of every member of D_b, its
// own included. Every slice is therefore read once from the owner and written once into every destination.
//
// The work items of one rank are (b, k) for every broadcast it receives, in plan order, and k in [0, count(N, m, i)),
// slice j = i + k * m. Items are flattened broadcast-major with prefix sums and dealt round-robin to the grid's issuer
// warps.

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/moe/loadBalance/dynamicEplb/moeRebalanceHaloRPlanner.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace halo_r
{

//! One relay slice: one bulk copy into one shared-memory slot, the slice size of the TMA copy.
constexpr unsigned long long kRelaySliceBytes = 8192ULL;
//! Weight planes per expert bundle the relay accepts.
constexpr int kRelayMaxPlanes = 8;

//! Index of `rank` in a destination mask, or -1 when the rank is not a destination; *count receives |mask|.
TLLM_HALO_R_HOST_DEVICE inline int relayDestinationIndex(unsigned destinationMask, int rank, int* count)
{
    unsigned const below = rank >= 32 ? destinationMask : (destinationMask & ((1U << rank) - 1U));
    int total = 0;
    int before = 0;
    for (int d = 0; d < 32; ++d)
    {
        total += static_cast<int>((destinationMask >> d) & 1U);
        before += static_cast<int>((below >> d) & 1U);
    }
    *count = total;
    return ((destinationMask >> rank) & 1U) ? before : -1;
}

//! Number of slices j in [0, slices) with j mod destinations == index.
TLLM_HALO_R_HOST_DEVICE inline long long relaySliceCount(long long slices, int destinations, int index)
{
    return (index < 0 || destinations <= 0 || index >= slices) ? 0 : (slices - 1 - index) / destinations + 1;
}

//! The k-th slice owned by destination `index`.
TLLM_HALO_R_HOST_DEVICE inline long long relaySliceOf(long long k, int destinations, int index)
{
    return static_cast<long long>(index) + k * static_cast<long long>(destinations);
}

TLLM_HALO_R_HOST_DEVICE inline long long relayPlaneSlices(unsigned long long bytes)
{
    return static_cast<long long>((bytes + kRelaySliceBytes - 1ULL) / kRelaySliceBytes);
}

TLLM_HALO_R_HOST_DEVICE inline long long relayBundleSlices(unsigned long long const* planeBytes, int planes)
{
    long long slices = 0;
    for (int p = 0; p < planes; ++p)
    {
        slices += relayPlaneSlices(planeBytes[p]);
    }
    return slices;
}

//! Slice j of a bundle as (plane, byte offset, bytes). Returns false when j is outside the bundle.
TLLM_HALO_R_HOST_DEVICE inline bool relayLocateSlice(unsigned long long const* planeBytes, int planes, long long slice,
    int* plane, unsigned long long* offset, unsigned long long* bytes)
{
    for (int p = 0; p < planes; ++p)
    {
        long long const slices = relayPlaneSlices(planeBytes[p]);
        if (slice < slices)
        {
            *plane = p;
            *offset = static_cast<unsigned long long>(slice) * kRelaySliceBytes;
            unsigned long long const left = planeBytes[p] - *offset;
            *bytes = left < kRelaySliceBytes ? left : kRelaySliceBytes;
            return true;
        }
        slice -= slices;
    }
    return false;
}

//! Work item t of this rank as (position q in the rank's broadcast list, k). prefix[q] counts the items before list
//! entry q and prefix[entries] is the total.
TLLM_HALO_R_HOST_DEVICE inline int relayLocateItem(int const* prefix, int entries, long long item, long long* k)
{
    int lo = 0;
    int hi = entries;
    while (hi - lo > 1)
    {
        int const mid = (lo + hi) / 2;
        if (static_cast<long long>(prefix[mid]) <= item)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    *k = item - prefix[lo];
    return lo;
}

} // namespace halo_r
} // namespace kernels

TRTLLM_NAMESPACE_END
