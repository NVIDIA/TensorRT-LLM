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

//! Supported geometry of the HALO-R kernel: EP sizes from kHaloRMinEp (the planner covers up to halo_r::kMaxRanks),
//! and from kHaloRMinCtas CTAs (CTA 0 plans while the others map routes) to kHaloRMaxCtas.
constexpr int kHaloRMinEp = 2;
constexpr int kHaloRMinCtas = 2;
constexpr int kHaloRMaxCtas = 128;

//! Status words of the HALO-R kernel. Each word holds the epoch of its most recent failure, or 0. A launch traps
//! before it releases any route once any word is nonzero, so a nonzero word never outlives the launch that set it.
enum MoeRebalanceHaloRStatus : int
{
    kHaloRStatusExchangeTimeout = 0,
    kHaloRStatusGridTimeout = 1,
    kHaloRStatusPlanWaitTimeout = 2,
    kHaloRStatusInvalidRoute = 3,
    kHaloRStatusSlotOutsideMask = 5,
    //! Words of the status buffer, the CUDA scheduler's: the words above and three unused ones.
    kHaloRStatusWords = 8,
};

//! Diagnostic counters of one layer, int64 each; read only by tests.
enum MoeRebalanceHaloRCounter : int
{
    kHaloRCounterLaunches = 0,       //!< Launches of the layer's kernel, eager and replayed.
    kHaloRCounterReceiverCopies = 1, //!< Sum over the plans' broadcasts of the receivers that were copied to.
    kHaloRCounterReceiverSkips = 2,  //!< Sum over the plans' broadcasts of the receivers that were skipped.
    kHaloRCounters = 3,
};

//! One launch of the fused HALO-R kernel for one MoE layer on one EP rank. The kernel runs on `ctas` CTAs, one per
//! SM, on the copy stream; every rank launches it once per layer forward, ranks without tokens included.
//!
//!   1. Scheduler role. CTA 0 drains this layer's READY row, exchanges the routed-token counts with every EP rank,
//!      plans the layer's replicas with one warp and builds the route prefix of every selected expert; the other CTAs
//!      compute stable per-expert ordinals and write the physical slot of every route.
//!   2. Route release. Each CTA checks every status word and then adds 1 to `routeCounter` with a system-scope
//!      release. A route consumer on another stream waits for the launch's completion event and then until the
//!      counter has grown by `ctas`.
//!   3. Relay. Each CTA copies its share of the expert bundles this rank receives: the receivers of a bundle that do
//!      not hold it yet (all of them without a residency table) split it into slices, and each pulls its slices from
//!      the owner and stores them into the helper slot of every such receiver.
//!   4. Completion. The last CTA to finish its copies records the plan's helper experts in the residency table and
//!      publishes `readyValue` into this rank's READY cell on every rank.
//!
//! A failure before the route release traps before the failing CTA releases its routes, so a consumer never reads
//! routes of a failed launch. A failure after it, in the relay or at completion, traps before READY, so no READY is
//! published for the launch.
//!
//! READY and the route counter take constant values, so that a captured launch replays without host bookkeeping:
//! every launch publishes READY 1 and first waits for every rank's previous READY in this rank's row and zeroes it.
//! The caller must therefore fill the row with 1 before the first launch, as a previous launch would leave it. The
//! route consumer waits for the counter to reach `ctas` and then resets it to 0.
struct MoeRebalanceHaloRParams
{
    int const* routes;                //!< [validRouteCount] global logical expert ids, -1 for padding.
    int* outSlots;                    //!< [routeCapacity] physical slot ids; entries past the valid routes become -1.
    int* outIds;                      //!< [helpers] expert held by each local helper slot, or -1.
    int* outOwners;                   //!< [helpers] owner rank of that expert, or -1.
    std::uint64_t const* peerBases;   //!< [ep] symmetric count-exchange buffer of every rank, mapped locally.
    int* status;                      //!< [kHaloRStatusWords]
    int* partial;                     //!< [ctas * experts] per-CTA histograms.
    int* routeAux;                    //!< Worker rendezvous, per-worker totals and selected-route indices.
    int* gridSync;                    //!< [4] grid rendezvous words and the route-prefix publication epoch.
    int* plan;                        //!< [halo_r::kPlanWords] the layer's plan; word 0 is its publication epoch.
    int* routePrefix;                 //!< [384 * (ep + 1)] per-broadcast destination prefix of this rank's routes.
    unsigned long long* routeCounter; //!< Each CTA adds 1; the consumer waits for `ctas`, then resets it.
    std::uint64_t const* sourceTable; //!< [experts * planes] address of each expert's plane in its owner's home
                                      //!< rows.
    std::uint64_t const* destinationTable; //!< [ep * planes] address A such that A + row * planeBytes[p] is row `row`
                                           //!< of rank d's plane p.
    std::uint64_t const* planeBytes;       //!< [planes] bytes of one expert in each plane.
    int* residency;                        //!< [ep * helpers] expert each rank's helper slot holds, -1 unknown;
                                           //!< null when the helper slots are shared by several layers.
    unsigned* completedCtas;               //!< [1] zero before the first launch; every launch leaves it zero.
    long long* counters;                   //!< [kHaloRCounters] diagnostic counters.
    std::uint64_t* readyFlags;             //!< [ep] this layer's READY row on this rank, drained before each exchange.
    std::uint64_t* readyFlagMc;            //!< This rank's READY cell through the multicast view of the READY rows.
    std::uint64_t readyValue;              //!< READY generation this launch publishes: always 1.
    unsigned long long spinCycles;         //!< Bound of every cross-rank and cross-CTA wait, in SM clock cycles.
    int ep;
    int experts;
    int helpers;
    int sendCount;
    int localRank;
    int routeCapacity;
    int validRouteCount;
    int planes;
};

//! Launch configuration of the fused kernel, prepared once per layer context.
struct MoeRebalanceHaloRLaunch
{
    int ctas = 0;
    int dynamicSharedBytes = 0;
    //! The remote memory-sync domain and its map, when the device has more than one domain.
    cudaLaunchAttribute attributes[2]{};
    unsigned attributeCount = 0;
};

//! Minimum routeAux entries for a given geometry.
std::int64_t getMoeRebalanceHaloRRouteAuxEntries(int experts, int ctas, int routeCapacity);

//! Throws unless the bound arguments of `params` describe a geometry the kernel supports. Routes, the valid route
//! count and the spin bound are per launch and not checked.
void checkMoeRebalanceHaloRParams(MoeRebalanceHaloRParams const& params, int ctas);

//! Loads the fused kernel on the current device, reserves its local memory and sizes its shared memory, so that the
//! first launch does not wait for the device; throws unless exactly one CTA fits each SM. Called once per context, at
//! bind time: a launch-time module load or stack resize would wait for kernels that spin on work queued after it.
MoeRebalanceHaloRLaunch prepareMoeRebalanceHaloRLaunch(int ep, int experts, int ctas);

//! Enqueues one launch on `stream`; returns the launch error, if any. `launched` becomes the launch's completion event
//! (cudaLaunchAttributeLaunchCompletionEvent), which nominally triggers once every CTA has begun (CUDA may delay it to
//! the kernel's end): a route consumer on another stream waits for it before its route wait, so CUDA sees that the
//! wait follows the kernel.
cudaError_t launchMoeRebalanceHaloR(MoeRebalanceHaloRLaunch const& launch, MoeRebalanceHaloRParams const& params,
    cudaStream_t stream, cudaEvent_t launched);

} // namespace kernels

TRTLLM_NAMESPACE_END
