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

//! Status words of the HALO-R scheduler. Each word holds the epoch of its most recent failure, or 0.
enum MoeRebalanceHaloRStatus : int
{
    kHaloRStatusExchangeTimeout = 0,
    kHaloRStatusGridTimeout = 1,
    kHaloRStatusPlanWaitTimeout = 2,
    kHaloRStatusInvalidRoute = 3,
    kHaloRStatusSlotOutsideMask = 5,
    kHaloRStatusPlanner = 6,
    //! Sticky READY error: (kind << 28) | (epoch & 0x0fffffff); kind 1 = a READY cell carried the error bit,
    //! 2 = a READY cell did not arrive in time.
    kHaloRStatusReady = 7,
    kHaloRStatusWords = 8,
};

//! Optional timing probe of one layer, for tests and benchmarks: one record of kHaloRProbeWords %globaltimer
//! nanosecond stamps per launch, indexed by the relay launch count before the launch. The caller presets the min
//! words to ~0 and the others to 0; launches past the last record are not stamped. A null probe costs one branch.
enum MoeRebalanceHaloRProbeWord : int
{
    kHaloRProbeSchedulerStart = 0, //!< Earliest scheduler CTA start (min).
    kHaloRProbePlanPublished = 1,  //!< CTA 0 released the plan epoch.
    kHaloRProbeSchedulerEnd = 2,   //!< Latest scheduler CTA end (max).
    kHaloRProbeRelayStart = 3,     //!< Earliest relay CTA start (min).
    kHaloRProbeRelayEnd = 4,       //!< The last relay CTA published READY (max).
    kHaloRProbeRelayFallbacks = 5, //!< Relay CTAs that waited for the scheduler grid to complete to read the plan.
    kHaloRProbeWords = 8,
};

//! HALO-R scheduler for one MoE layer on one EP rank: routed-token count exchange, greedy replica plan and physical
//! slot of every route. Collective over the EP group; every rank launches it once per layer forward.
//!
//! The relay copy that follows it on the same stream may be a programmatic dependent launch. Worker CTAs trigger it at
//! entry and CTA 0 right after it releases the plan epoch, so the relay starts while the workers still write routes.
//! The trigger carries no memory ordering: the relay acquires the plan through the plan epoch and waits for the
//! scheduler grid to complete before it reads the status words (moeRebalanceRelay.cu).
struct MoeRebalanceHaloRParams
{
    int const* routes;              //!< [validRouteCount] global logical expert ids, -1 for padding.
    int* outSlots;                  //!< [routeCapacity] physical slot ids; entries past the valid routes become -1.
    int* outIds;                    //!< [helpers] expert held by each local helper slot, or -1.
    int* outOwners;                 //!< [helpers] owner rank of that expert, or -1.
    std::uint64_t const* peerBases; //!< [ep] symmetric count-exchange buffer of every rank, mapped locally.
    int* status;                    //!< [kHaloRStatusWords]
    int* partial;                   //!< [ctas * experts] per-CTA histograms.
    int* routeAux;                  //!< Worker rendezvous, per-worker totals and selected-route indices.
    int* gridSync;                  //!< [4] grid rendezvous words and the route-prefix publication epoch.
    int* plan;                      //!< [halo_r::kPlanWords] plan published to the relay copy.
    int* routePrefix;               //!< [384 * (ep + 1)] per-broadcast destination prefix of this rank's routes.
    std::uint64_t* readyFlags;      //!< [ep] this layer's READY row on this rank; drained before each exchange.
    unsigned* relayResidentCtas;    //!< [1] the relay copy's residency counter, zeroed at entry, before the CTA's
                                    //!< programmatic launch trigger; may be null.
    int ep;
    int experts;
    int helpers;
    int sendCount;
    int sendWeight;
    int localRank;
    int routeCapacity;
    int validRouteCount;
    int ctas;
    std::uint64_t spinNs;             //!< Wall-clock bound of every device wait, in nanoseconds.
    std::uint64_t* probe;             //!< [probeRecords * kHaloRProbeWords] timing records, or null.
    unsigned const* probeLaunchCount; //!< The relay launch count that indexes the probe records.
    int probeRecords;
};

//! Dynamic shared memory of the scheduler kernel for a given geometry.
std::size_t getMoeRebalanceHaloRSharedMemoryBytes(int ep, int experts);

//! Minimum routeAux entries for a given geometry.
std::int64_t getMoeRebalanceHaloRRouteAuxEntries(int experts, int ctas, int routeCapacity);

void invokeMoeRebalanceHaloR(MoeRebalanceHaloRParams const& params, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
