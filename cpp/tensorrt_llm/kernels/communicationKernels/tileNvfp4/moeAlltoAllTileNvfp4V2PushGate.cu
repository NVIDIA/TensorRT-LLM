/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

/*
 * Push-combine control plane: K2 producer-global publish + K3 consumer gate.
 *
 * These two steps live on the NEW raw-u64 epoch flag ABI ("raw_u64_epoch_v1",
 * layout "compact_row_v2") defined in moeAlltoAllTileNvfp4V2PushFlags.h. They
 * are deliberately independent of the PULL-combine uint32 flags
 * (completionFlags / flagVal / dispatch flags), which stay untouched: this file
 * only ever touches the compact_row_v2 control plane via the pushXxxPtr
 * accessors.
 *
 * Memory ordering mirrors the family idiom publishAndAcquireReadiness
 * (moeAlltoAllTileNvfp4V2TmaCommon.cuh): fence.release + st.relaxed for the
 * publish side, spin ld.relaxed + fence.acquire for the observe side. The only
 * change is a scope adaptation to u64 words (and a gpu-scope release for the
 * consumer-local gate, which never leaves the device).
 */

#include "moeAlltoAllTileNvfp4V2PushFlags.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

namespace
{

// POD kernel args (family pointer-table idiom, mirrors K1PushDeviceArgs). Carries
// the already-hopped per-consumer control-row bases so K2/K3 replicate the SAME
// symmetric MNNVL hop the payload plane uses: control_peer_base[C] is the base of
// consumer C's compact_row_v2 row (workspaceBase + C*stride(0) + control offset).
// A producer publishing into consumer C addresses pushReadyPtr(base[C], rankId);
// the consumer gates on its own row base[rankId]. Built on the host from
// PushCombineControlParams::control_peer_base.
struct PushGateDeviceArgs
{
    std::uint8_t* control_peer_base[kMaxRanksTile];
    int ep_size;
    int rank_id;
    // SM-cycle budget of the consumer-gate spin (pushK23ControlKernel<true, ...>);
    // unused by the instantiations without a deadline.
    long long deadline_cycles;
    // NVLinkOneSided A2A round flag of this rank; advanced by one together with the
    // generation open (kOpenGeneration), in place of the D0 kernel.
    std::uint32_t* a2a_flag_val;
};

// D0's rebuild-required sentinel (moeAlltoAllTileNvfp4V2PushD0.cu), for the
// generation open K23 performs when the dispatch skipped D0 (pushK23ControlKernel<*, true>).
constexpr std::uint64_t kPushGenerationRebuildRequiredK23 = ~static_cast<std::uint64_t>(0);
static_assert(kPushGenerationRebuildRequiredK23 != kPushInvalidGeneration,
    "the rebuild sentinel must be distinct from INVALID_GENERATION");

// TRTLLM_PUSH_K23_DEADLINE=1: the family's in-kernel A2A wait budget,
// check_timeout in moeAlltoAllKernels.cu (60 * 2000 * 1000 * 1000 SM cycles).
constexpr long long kPushK23DefaultDeadlineCycles = 60LL * 2000LL * 1000LL * 1000LL;

// Deadline expiry: name every producer slot of this consumer's row that still differs
// from the expected generation, then trap (as the dispatch / combine waits do), so
// a lost or late readiness publish fails loudly instead of spinning forever.
__device__ __noinline__ void pushK23GateTimeout(
    std::uint8_t* rowBase, int consumerC, int epSize, std::uint64_t expected, long long waitedCycles)
{
#if defined(__CUDA_ARCH__)
    for (int producerP = 0; producerP < epSize; ++producerP)
    {
        std::uint64_t observed;
        asm volatile("ld.relaxed.sys.u64 %0, [%1];"
                     : "=l"(observed)
                     : "l"(pushReadyPtr(rowBase, producerP))
                     : "memory");
        if (observed != expected)
        {
            printf(
                "push K23 gate: ---Rank %d timed out after %lld cycles waiting for ready[%d] from producer rank %d "
                "(observed generation %llu, expected %llu)\n",
                consumerC, waitedCycles, producerP, producerP, static_cast<unsigned long long>(observed),
                static_cast<unsigned long long>(expected));
        }
    }
    asm volatile("trap;");
#endif
}

// -----------------------------------------------------------------------------
// K23: fused producer-publish + consumer-gate (the shipped control kernel).
//
// lane0/block0 runs, in order: (1) K2's producer publish -- read this rank's OWN
// device generation g (D0-open RMW), fence.release.sys, then relaxed sys stores
// of g into EVERY consumer row's slot rankId (control_peer_base[C], including
// zero-count consumers); the invalid-generation sentinel guards ONLY the publish
// loop (no early return, so the gate half always runs); then (2) K3's consumer
// gate -- reload this consumer's own generation as expected, spin O(E) on all E
// producer slots of its OWN row with relaxed sys loads until all == expected,
// fence.acquire.sys, then gpu-scope release-store localGate[C] = g for K4.
// Launched <<<1,32,0,stream>>>; no __syncwarp / no shared mem (one lane). The
// self-slot ready store -> own poll read is ordered by program order; the load-
// bearing release/acquire edge is genuinely inter-device (this rank's own row is
// filled by remote ranks' K23 publish halves over MNNVL), so fusing local
// publish+gate cannot self-satisfy the wait -- lane0 still spins at sys scope on
// all E slots.
// -----------------------------------------------------------------------------
//
// kOpenGeneration (TRTLLM_PUSH_D0_FOLD=1): first advance this rank's OWN generation word
// g-1 -> g, exactly as D0's open kernel does at the dispatch tail (same relaxed-sys
// load, same near-overflow sentinel + trap, never 0), so the dispatch can skip D0.
// The word has no other reader: only this rank's K23 loads it (the two loads below),
// and remote ranks never read it (they write only ready[] slots of this row). The
// store is relaxed-sys; the publish half's fence.release.sys below orders it ahead
// of every ready[] store, so the D0 store's release is kept. Both loads below then
// return g by same-thread program order: the same publish and gate values as with D0.
// -----------------------------------------------------------------------------
//
// kPdl (TRTLLM_PUSH_K23_PDL=1): K23 is launched with programmatic stream
// serialization after K1, so its launch overlaps K1's tail. Every thread first waits
// on the grid dependency (K1 complete, its pushes and zero stores performed and
// visible: the same guarantee the plain launch's kernel boundary gave), and only then
// allows its dependent (K4) to launch, so K4's pre-wait prologue (route loads) also
// runs strictly after K1. The body that follows is unchanged.
// -----------------------------------------------------------------------------
template <bool kDeadline, bool kOpenGeneration, bool kPdl = false>
__global__ void pushK23ControlKernel(PushGateDeviceArgs args)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr (kPdl)
    {
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    // Full warp (32) launched but only lane0/block0 works; guard drops lanes
    // 1..31 and any extra CTA so a wider launch cannot double-publish/double-gate.
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }

#if defined(__CUDA_ARCH__)
    int const rankId = args.rank_id; // producer index P == consumer index C here

    if constexpr (kOpenGeneration)
    {
        std::uint64_t* const generationWord = pushGenerationPtr(args.control_peer_base[rankId]);
        std::uint64_t const previous = pushLoadRelaxedSysU64(generationWord);
        if (previous >= kPushGenerationRebuildRequiredK23 - 1)
        {
            pushStoreReleaseSysU64(generationWord, kPushGenerationRebuildRequiredK23);
            asm volatile("trap;");
        }
        std::uint64_t const opened = previous + 1;
        asm volatile("st.relaxed.sys.u64 [%0], %1;" ::"l"(generationWord), "l"(opened) : "memory");
        // Combine half of the NVLinkOneSided A2A round (flag 2N+1 -> 2N+2), as D0 does.
        // K1 has completed (kernel boundary, or the grid-dependency wait under kPdl),
        // and only this rank's own stream reads the word.
        if (args.a2a_flag_val != nullptr)
        {
            *args.a2a_flag_val = *args.a2a_flag_val + 1U;
        }
    }

    // ---- K2 publish half (producer readiness publish) --------
    // This producer's OWN device epoch word (advanced by D0 open RMW; never host).
    std::uint64_t const generation = pushLoadRelaxedSysU64(pushGenerationPtr(args.control_peer_base[rankId]));
    // CRITICAL structural change vs the standalone K2 (which returns here): the
    // invalid-generation sentinel guards ONLY the publish loop. Do NOT return --
    // that would also skip the gate + localGate and change hang semantics. Falling
    // through reproduces today's decomposition where K3 launches/spins regardless
    // of whether K2 published. Sentinel 0 is never published.
    if (generation != kPushInvalidGeneration)
    {
        // ONE hoisted system-scope release orders all prior K1 payload writes
        // ahead of every ready[] store. Single + hoisted, exactly as K2.
        asm volatile("fence.release.sys;" ::: "memory");
#pragma unroll 1
        for (int consumerC = 0; consumerC < args.ep_size; ++consumerC)
        {
            std::uint64_t* address = pushReadyPtr(args.control_peer_base[consumerC], rankId);
            asm volatile("st.relaxed.sys.u64 [%0], %1;" ::"l"(address), "l"(generation) : "memory");
        }
    }

    // ---- K3 gate half (consumer single-gate spin), same lane ----
    int const consumerC = rankId;
    std::uint8_t* const rowBase = args.control_peer_base[consumerC];

    // SECOND relaxed-sys load of the SAME device word into `expected` (kept as a
    // separate load, NOT deduped; no invalid-check on expected -- unchanged vs K3).
    // Slot rankId of this row is satisfied by the local publish above via same-lane
    // program order; slots P != rankId require REMOTE ranks' K23 publish halves.
    std::uint64_t const expected = pushLoadRelaxedSysU64(pushGenerationPtr(rowBase));

    long long const gateStart = kDeadline ? clock64() : 0;
    bool allRanksReady = false;
    while (!allRanksReady)
    {
        allRanksReady = true;
#pragma unroll 1
        for (int producerP = 0; producerP < args.ep_size; ++producerP)
        {
            std::uint64_t* address = pushReadyPtr(rowBase, producerP);
            std::uint64_t observed;
            asm volatile("ld.relaxed.sys.u64 %0, [%1];" : "=l"(observed) : "l"(address) : "memory");
            if (observed != expected)
            {
                allRanksReady = false;
                if constexpr (kDeadline)
                {
                    long long const waited = clock64() - gateStart;
                    if (__builtin_expect(waited > args.deadline_cycles, 0))
                    {
                        pushK23GateTimeout(rowBase, consumerC, args.ep_size, expected, waited);
                    }
                }
                break;
            }
        }
    }

    // Acquire the payloads that the producers released before publishing ready.
    asm volatile("fence.acquire.sys;" ::: "memory");

    // Terminal local handoff to K4: gpu-scope release+store (localGate never leaves
    // the device). Scope stays SPLIT from the sys-scope publish/gate above.
    std::uint64_t* gate = pushLocalGatePtr(rowBase);
    asm volatile("fence.release.gpu;" ::: "memory");
    asm volatile("st.relaxed.gpu.u64 [%0], %1;" ::"l"(gate), "l"(expected) : "memory");
#else
    // Host-side compile placeholder: the device asm above is the real body.
    (void) args;
#endif
}

// Copy the already-hopped per-consumer control-row bases (indices [0, ep_size))
// plus the epoch axes into the by-value device-args struct. Mirrors K1's
// makeDeviceArgs: the pointer table is snapshotted on the host so the tiny K2/K3
// launches carry no separate device buffer.
PushGateDeviceArgs makeGateArgs(PushCombineControlParams const& params)
{
    PushGateDeviceArgs args{};
    for (int consumerC = 0; consumerC < params.ep_size; ++consumerC)
    {
        args.control_peer_base[consumerC] = params.control_peer_base[consumerC];
    }
    args.ep_size = params.ep_size;
    args.rank_id = params.rank_id;
    args.deadline_cycles = 0;
    args.a2a_flag_val = params.a2a_flag_val;
    return args;
}

// The K23 consumer-gate deadline, read once per process. Unset (or 0):
// no deadline, the stock kernel. TRTLLM_PUSH_K23_DEADLINE=1: the family budget
// (kPushK23DefaultDeadlineCycles); TRTLLM_PUSH_K23_DEADLINE_CYCLES=N (> 0): N SM cycles.
long long resolveK23DeadlineCycles()
{
    static long long const cycles = []
    {
        long long value = 0;
        char const* custom = std::getenv("TRTLLM_PUSH_K23_DEADLINE_CYCLES");
        if (custom != nullptr && custom[0] != '\0')
        {
            value = std::strtoll(custom, nullptr, 10);
        }
        if (value <= 0)
        {
            char const* enable = std::getenv("TRTLLM_PUSH_K23_DEADLINE");
            value = (enable != nullptr && enable[0] == '1') ? kPushK23DefaultDeadlineCycles : 0;
        }
        if (value > 0)
        {
            TLLM_LOG_INFO("[TRTLLM_PUSH_K23_DEADLINE] push-combine K23 gate deadline active: %lld SM cycles", value);
        }
        return value;
    }();
    return cycles;
}

bool k23PdlRequested()
{
    char const* env = std::getenv("TRTLLM_PUSH_K23_PDL");
    return env != nullptr && env[0] == '1';
}

using K23Kernel = void (*)(PushGateDeviceArgs);

K23Kernel selectK23Kernel(bool deadline, bool openGeneration, bool pdl)
{
    if (pdl)
    {
        return deadline
            ? (openGeneration ? pushK23ControlKernel<true, true, true> : pushK23ControlKernel<true, false, true>)
            : (openGeneration ? pushK23ControlKernel<false, true, true> : pushK23ControlKernel<false, false, true>);
    }
    return deadline ? (openGeneration ? pushK23ControlKernel<true, true> : pushK23ControlKernel<true, false>)
                    : (openGeneration ? pushK23ControlKernel<false, true> : pushK23ControlKernel<false, false>);
}

} // namespace

// -----------------------------------------------------------------------------
// Host launch. Plain stream launch (no PDL): K23 runs after K1 has fully drained
// on the owner stream (the kernel boundary is the drain proof), publishes this
// rank's readiness, then spins the consumer gate; K4 follows on the same stream.
// -----------------------------------------------------------------------------
// Fused producer-publish + consumer-gate on one launch. Validation is the union
// of the publish and gate host checks: the full [0, ep_size) non-null sweep over
// control_peer_base subsumes the own-row check, and is required because the
// publish half dereferences every consumer row. Geometry is one warp; lane0 works.
void launchK23Control(PushCombineControlParams const& params, cudaStream_t stream, bool openGeneration)
{
    TLLM_CHECK_WITH_INFO(params.ep_size >= 1 && params.ep_size <= kPushReadyCapacity, "K23 ep_size %d out of [1, %d]",
        params.ep_size, kPushReadyCapacity);
    TLLM_CHECK_WITH_INFO(params.rank_id >= 0 && params.rank_id < params.ep_size,
        "K23 rank_id %d out of [0, ep_size=%d)", params.rank_id, params.ep_size);
    // The publish half hops into EVERY consumer row C in [0, ep_size), so every
    // entry of the already-hopped control_peer_base table it touches (which
    // includes this rank's own gate row) must be populated.
    for (int consumerC = 0; consumerC < params.ep_size; ++consumerC)
    {
        TLLM_CHECK_WITH_INFO(
            params.control_peer_base[consumerC] != nullptr, "K23 requires a non-null control_peer_base[%d]", consumerC);
    }

    // ONE CTA of ONE warp; only lane0 publishes then gates. The generation is the
    // DEVICE epoch word (read on-device from this rank's own push_generation slot
    // in both phases); it is deliberately NOT forwarded from the host.
    PushGateDeviceArgs args = makeGateArgs(params);
    long long const deadlineCycles = resolveK23DeadlineCycles();
    if (deadlineCycles > 0)
    {
        args.deadline_cycles = deadlineCycles;
    }
    TLLM_CHECK_WITH_INFO(!openGeneration || params.a2a_flag_val != nullptr,
        "K23 generation open (D0 fold) requires the NVLinkOneSided A2A round flag");
    // TRTLLM_PUSH_K23_PDL=1 launches the <*, *, true> instance with PDL after K1 (read at
    // each launch, so a captured graph bakes the choice; default off).
    bool const usePdl = k23PdlRequested();
    K23Kernel const kernel = selectK23Kernel(deadlineCycles > 0, openGeneration, usePdl);
    if (usePdl)
    {
        static bool const announced = [&]
        {
            TLLM_LOG_INFO(
                "[TRTLLM_PUSH_K23_PDL] push-combine K23 launched with PDL after K1: grid-dependency wait on K1, "
                "then K4 may launch");
            return true;
        }();
        (void) announced;
        tensorrt_llm::common::launchWithPdlWhenEnabled(
            "pushK23ControlKernel<pdl>", kernel, dim3(1), dim3(32), 0, stream, args);
    }
    else
    {
        kernel<<<1, 32, 0, stream>>>(args);
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
