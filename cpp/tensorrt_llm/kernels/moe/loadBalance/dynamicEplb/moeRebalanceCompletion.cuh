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

// Completion of a copy kernel of per-layer dynamic EPLB, shared by the HALO-Q in-switch copy (moeRebalanceTma.cu) and
// the HALO-R relay (moeRebalanceHaloR.cu): the chain that elects the grid's last CTA and the READY publication through
// the multicast view of the READY rows.

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! One thread per CTA joins a release/acquire RMW chain over the grid's CTAs; returns true in the last CTA to arrive,
//! which then observes every CTA's writes ordered before its arrival. No CTA waits for another, so an oversubscribed
//! grid cannot deadlock here.
__device__ __forceinline__ bool moeRebalanceArriveLastCta(unsigned* completedCtas)
{
    unsigned previous;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(previous) : "l"(completedCtas) : "memory");
    return previous == gridDim.x - 1;
}

//! The last CTA rearms the counter: every CTA has arrived, and the next launch is stream-ordered after this one, so no
//! host memset is needed between launches.
__device__ __forceinline__ void moeRebalanceRearmCtaCounter(unsigned* completedCtas)
{
    asm volatile("st.relaxed.gpu.global.u32 [%0], 0;" ::"l"(completedCtas) : "memory");
}

//! Publishes `value` into a READY cell through its multicast address `flagMc`, after every write the calling thread
//! has ordered before it. The consumer acquires the cell through its unicast alias.
__device__ __forceinline__ void moeRebalancePublishReady(std::uint64_t flagMc, std::uint64_t value)
{
    asm volatile("fence.proxy.alias;" ::: "memory");
    asm volatile("fence.release.sys;" ::: "memory");
    asm volatile("multimem.st.release.sys.global.u64 [%0], %1;" ::"l"(flagMc), "l"(value) : "memory");
    asm volatile("fence.proxy.alias;" ::: "memory");
}

} // namespace kernels

TRTLLM_NAMESPACE_END
