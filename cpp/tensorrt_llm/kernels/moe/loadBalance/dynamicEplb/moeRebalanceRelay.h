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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! HALO-R relay copy for one MoE layer on one EP rank. Each rank pulls its share of every expert bundle it receives
//! from the owner and stores it into the helper slot of every receiver of that bundle, then the last CTA publishes
//! the plan epoch into this rank's READY cell on every rank. A publication carries the READY error bit when the
//! copy could not run or the scheduler reported a failure. Runs on the stream of the HALO-R scheduler, after it.
struct MoeRebalanceRelayParams
{
    int const* plan;                       //!< The scheduler's plan for this layer and epoch.
    std::uint64_t const* sourceTable;      //!< [experts * planes] address of each expert's plane in its owner's home
                                           //!< rows.
    std::uint64_t const* destinationTable; //!< [ep * planes] address A such that A + row * planeBytes[p] is row `row`
                                           //!< of rank d's plane p.
    std::uint64_t const* planeBytes;       //!< [planes] bytes of one expert in each plane.
    int const* status;                     //!< The scheduler's status words.
    unsigned* completedCtas;               //!< [1] zero before the first launch; every launch leaves it zero.
    unsigned* launchCount;                 //!< [1] relay launches of this layer; zero before the first launch.
    std::uint64_t* readyFlagMc;            //!< This rank's READY cell through the multicast view of the READY rows.
    int ep;
    int experts;
    int localRank;
    int planes;
    int ctas;
};

//! Dynamic shared memory of one relay CTA on the current device.
std::size_t getMoeRebalanceRelaySharedMemoryBytes();

//! Threads of one relay CTA.
int getMoeRebalanceRelayThreads();

void invokeMoeRebalanceRelay(MoeRebalanceRelayParams const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
