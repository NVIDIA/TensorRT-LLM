/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

// Payload packing is unchanged. Full-generation tags occupy separate zero-initialized MNNVL words.
// Two-parity payload reuse still obeys the sampler's existing all-CTA/peer-step completion protocol.
TRTLLM_MV2_NAMESPACE_BEGIN

namespace prob1Exchange
{
using Word = unsigned long long;
constexpr int kSlots = 64;
constexpr int kTagOffset = 128;
constexpr int kAllocationWords = 256;
#if defined(__CUDACC__)
#define PROB1_EX_HD __host__ __device__
#else
#define PROB1_EX_HD
#endif
PROB1_EX_HD constexpr bool ready(Word observed, Word requested)
{
    return requested != 0 && observed == requested;
}

PROB1_EX_HD constexpr int tagIndex(int payloadIndex)
{
    return kTagOffset + payloadIndex;
}

#undef PROB1_EX_HD
#if defined(__CUDACC__)
__device__ __forceinline__ void publish(Word* base, Word* payload, Word value, Word generation)
{
    Word* const tag = base + kTagOffset + (payload - base);
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(payload), "l"(value) : "memory");
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(tag), "l"(generation) : "memory");
}

__device__ __forceinline__ bool readyLoad(Word const* base, Word const* payload, Word generation, Word& value)
{
    Word const* const tag = base + kTagOffset + (payload - base);
    Word observed;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(observed) : "l"(tag) : "memory");
    if (!ready(observed, generation))
        return false;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(payload) : "memory");
    return true;
}
#endif
static_assert(kTagOffset >= kSlots + 11 && kTagOffset + kSlots <= kAllocationWords, "nonoverlapping controls/tags");
} // namespace prob1Exchange

TRTLLM_MV2_NAMESPACE_END
