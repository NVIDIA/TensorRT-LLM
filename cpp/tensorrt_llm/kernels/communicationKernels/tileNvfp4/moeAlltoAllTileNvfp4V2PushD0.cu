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
 * D0 control-plane steps for the push-combine plane of the packed-NVFP4
 * tileNvfp4 V2 family. Two control-only device ops live here:
 *
 *   - launchPushControlBootstrap : one-time clear of THIS rank's OWN control
 *     segment (its single compact_row_v2 row at control_peer_base[rankId]:
 *     ready[P] for P in [0, epSize), push_generation, localGate = 0), mirroring
 *     the stock workspace[epRank].zero_() + collective barrier. INVALID_-
 *     GENERATION (0) IS the cleared state, so the zeroed segment is a well-formed
 *     "nothing opened yet" state. Run once before first graph capture / dispatch
 *     collective-init.
 *
 *   - launchD0GenerationOpen : a graph-captured device op that monotonically
 *     advances THIS rank's combine push_generation g-1 -> g (raw u64). It is a
 *     device-side read-modify-write so it stays replay-safe under CUDA graphs
 *     (it does NOT bake in a host scalar); the first open yields g=1; it never
 *     writes 0; it touches ONLY the generation word (never the remote ready
 *     flag plane); and on near-overflow it FAILS CLOSED rather than wrapping.
 *
 * This file is a separate plane. It never touches the uint32
 * completionFlags / flagVal / dispatch flags (those belong to the PULL combine
 * and are addressed only through the untouched MoeA2ACombineParamsV2 path). The
 * u64 memory-ordering here mirrors the release/acquire idiom in
 * moeAlltoAllTileNvfp4V2TmaCommon.cuh::publishAndAcquireReadiness, scope-adapted
 * from u32 to u64 (fence.release.sys + st.relaxed.sys stores).
 */

#include "moeAlltoAllTileNvfp4V2PushFlags.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{
namespace
{

// One control-plane row (compact_row_v2) is exactly this many raw u64 words:
// ready[kPushReadyCapacity] + push_generation + localGate, with no padding.
constexpr int kPushWordsPerRow = kPushControlSlabBytes / static_cast<int>(sizeof(std::uint64_t));
static_assert(kPushControlSlabBytes % static_cast<int>(sizeof(std::uint64_t)) == 0,
    "compact_row_v2 row stride must be a whole number of u64 words");
static_assert(
    kPushWordsPerRow == kPushReadyCapacity + 2, "each row is ready[kPushReadyCapacity] + push_generation + localGate");

// Bootstrap CTA width. The whole plane is tiny (ep_size <= kPushReadyCapacity
// rows, kPushWordsPerRow words each), so a single CTA with a grid-stride loop
// clears it comfortably.
constexpr int kBootstrapThreads = 128;

// Rebuild-required sentinel for the monotonic u64 generation counter (it makes
// the counter fail closed). It is deliberately the maximum u64 so that:
//   - it can never be produced by a legal open (legal opens stay strictly
//     below it, see the near-overflow guard in the open kernel), and
//   - it is trivially distinguishable from kPushInvalidGeneration (0) and from
//     any live generation. A host that observes it must tear down and rebuild
//     the control plane; the plane must NOT be online-reset or wrapped.
constexpr std::uint64_t kPushGenerationRebuildRequired = ~static_cast<std::uint64_t>(0);
static_assert(kPushGenerationRebuildRequired != kPushInvalidGeneration,
    "the rebuild sentinel must be distinct from INVALID_GENERATION");

// System-scope u64 store used to clear a control word. Mirrors the st.relaxed.sys
// stores in publishAndAcquireReadiness, widened to u64. Falls back to a plain
// store on the host-compiler pass (no device asm available there).
__device__ __forceinline__ void pushStoreRelaxedSysU64(std::uint64_t* address, std::uint64_t value)
{
#if defined(__CUDA_ARCH__)
    asm volatile("st.relaxed.sys.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
#else
    *address = value;
#endif
}

// pushLoadRelaxedSysU64 and pushStoreReleaseSysU64 now live in
// moeAlltoAllTileNvfp4V2PushFlags.h (promoted so K2/K3 share the byte-identical
// system-scope idiom); they are used here through that header.

// System-scope release fence, standalone. Publishes the cleared bootstrap state.
__device__ __forceinline__ void pushFenceReleaseSys()
{
#if defined(__CUDA_ARCH__)
    asm volatile("fence.release.sys;" ::: "memory");
#endif
}

// ---- Bootstrap: clear THIS rank's OWN control segment to INVALID_GENERATION ---
//
// Mirrors the stock bootstrap (moeA2AInitializeOp: workspace[epRank].zero_()
// + collective barrier): each rank clears only its OWN segment -- the single
// compact_row_v2 row at ownBase == control_peer_base[rankId]. Zeroes ready[P] for
// every producer slot P in [0, epSize), plus this row's push_generation and
// localGate. All-zero == kPushInvalidGeneration == the "nothing opened yet" state.
// A COLLECTIVE barrier (owned by the op layer) fences every rank's
// clear before any peer flag read. Addresses are formed through the frozen
// compact_row_v2 accessors so this step is independent of inter-row padding.
__global__ __launch_bounds__(kBootstrapThreads) void moeA2APushControlBootstrapKernel(std::uint8_t* ownBase, int epSize)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
    // Clear the active ready slots [0, epSize), then the fixed-offset generation
    // and localGate words. Unused ready slots [epSize, kPushReadyCapacity) remain
    // untouched and are never polled for this EP size.
    long long const totalWords = static_cast<long long>(epSize) + 2;
    long long const stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < totalWords; idx += stride)
    {
        int const wordInRow = static_cast<int>(idx);
        std::uint64_t* address;
        if (wordInRow < epSize)
        {
            address = pushReadyPtr(ownBase, wordInRow);
        }
        else if (wordInRow == epSize)
        {
            address = pushGenerationPtr(ownBase);
        }
        else
        {
            address = pushLocalGatePtr(ownBase);
        }
        pushStoreRelaxedSysU64(address, kPushInvalidGeneration); // == 0
    }

    // Publish the cleared segment before this rank (or a peer) opens the first
    // generation. Bootstrap is a single-CTA launch, so a system release fence on
    // the completing threads is sufficient to order the zeroing globally; kernel
    // completion then orders it against the subsequent dispatch/init launch, and
    // the op-layer collective barrier orders it across ranks.
    pushFenceReleaseSys();
}

// ---- D0 generation-open: monotonic g-1 -> g for THIS rank's combine epoch -----
//
// Requirements, all satisfied device-side so the op is safe to
// capture into a CUDA graph and replay:
//   * REPLAY-SAFE: the new generation is derived from the CURRENT device word
//     via a read-modify-write, NOT from a host scalar baked in at capture. Each
//     graph replay therefore advances the counter by exactly one.
//   * FIRST OPEN == 1: a cleared word holds kPushInvalidGeneration (0), so the
//     first open computes 0 + 1 == kPushFirstGeneration (1).
//   * NEVER 0: the stored value is always >= kPushFirstGeneration.
//   * REMOTE FLAG PLANE UNTOUCHED: only push_generation[C] for C == rankId is
//     written; ready[C][P] is never cleared or advanced here.
//   * FAIL CLOSED ON NEAR-OVERFLOW: rather than wrapping past the u64 range (or
//     online-resetting), the op stamps the rebuild-required sentinel and traps,
//     forcing a host-driven rebuild.
//
// ownBase is this rank's own already-hopped control row (control_peer_base[rankId])
// carrying its combine generation word.
__global__ __launch_bounds__(1) void moeA2APushD0GenerationOpenKernel(std::uint8_t* ownBase, std::uint32_t* a2aFlagVal)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
    // Single thread suffices: this is a scalar advance of one u64 word.
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    std::uint64_t* const generationWord = pushGenerationPtr(ownBase);
    std::uint64_t const previous = pushLoadRelaxedSysU64(generationWord);

    // Near-overflow guard (fail closed). Legal generations occupy
    // [kPushFirstGeneration, kPushGenerationRebuildRequired - 1]. If advancing
    // would reach the reserved sentinel (or we already sit on/above it), do NOT
    // wrap or reset: stamp the sentinel and trap so the failure is loud and the
    // host must rebuild the plane.
    if (previous >= kPushGenerationRebuildRequired - 1)
    {
        pushStoreReleaseSysU64(generationWord, kPushGenerationRebuildRequired);
#if defined(__CUDA_ARCH__)
        asm volatile("trap;");
#endif
        return;
    }

    // Monotonic advance g-1 -> g. previous == 0 (cleared) yields exactly
    // kPushFirstGeneration; the result is always >= kPushFirstGeneration and is
    // published with release ordering so a consumer that later acquires it also
    // observes everything ordered before it on this rank.
    std::uint64_t const opened = previous + 1;
    pushStoreReleaseSysU64(generationWord, opened);

    // Combine half of the NVLinkOneSided A2A round (flag 2N+1 -> 2N+2), exactly
    // what the stock prepare-combine kernel does. Only this rank's own stream
    // reads the word, so a plain store ordered by kernel completion suffices.
    if (a2aFlagVal != nullptr)
    {
        *a2aFlagVal = *a2aFlagVal + 1U;
    }
}

// Shared parameter validation for the two control-only D0 launches. Neither
// step inspects payload counts, so recv_counters is allowed to be null here.
void validatePushControlParams(PushCombineControlParams const& params)
{
    TLLM_CHECK_WITH_INFO(params.ep_size > 0 && params.ep_size <= kPushReadyCapacity,
        "ep_size must be in (0, kPushReadyCapacity]; ready[C][P] is indexed by producer P < ep_size");
    TLLM_CHECK_WITH_INFO(params.rank_id >= 0 && params.rank_id < params.ep_size,
        "rank_id must be a valid consumer/producer index in [0, ep_size)");
    // Both control-only steps touch ONLY this rank's own already-hopped segment
    // control_peer_base[rank_id] (the bootstrap clears its own segment; D0 advances its own
    // generation word), never control_base.
    std::uint8_t* const ownBase = params.control_peer_base[params.rank_id];
    TLLM_CHECK_WITH_INFO(
        ownBase != nullptr, "control_peer_base[rank_id=%d] (this rank's own segment) is required", params.rank_id);
    TLLM_CHECK_WITH_INFO(reinterpret_cast<std::uintptr_t>(ownBase) % alignof(std::uint64_t) == 0,
        "own control segment base must be u64-aligned for the raw-u64 flag ABI");
}

} // namespace

void launchPushControlBootstrap(PushCombineControlParams const& params, cudaStream_t stream)
{
    validatePushControlParams(params);
    // Single CTA clears THIS rank's own (tiny) segment control_peer_base[rank_id]
    // via a grid-stride loop.
    tensorrt_llm::common::launchWithPdlWhenEnabled("moeA2APushControlBootstrapKernel", moeA2APushControlBootstrapKernel,
        dim3(1), dim3(kBootstrapThreads), 0, stream, params.control_peer_base[params.rank_id], params.ep_size);
}

void launchD0GenerationOpen(PushCombineControlParams const& params, cudaStream_t stream)
{
    validatePushControlParams(params);
    // Spec D0: the opened generation is computed on-device from the current word
    // (replay-safe RMW), so no host generation scalar is forwarded to the kernel
    // -- forwarding one would bake a host scalar into the captured graph and break
    // monotonicity across replays. The kernel advances THIS rank's own generation
    // word at control_peer_base[rank_id]. Single-thread launch.
    tensorrt_llm::common::launchWithPdlWhenEnabled("moeA2APushD0GenerationOpenKernel", moeA2APushD0GenerationOpenKernel,
        dim3(1), dim3(1), 0, stream, params.control_peer_base[params.rank_id], params.a2a_flag_val);
}

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
