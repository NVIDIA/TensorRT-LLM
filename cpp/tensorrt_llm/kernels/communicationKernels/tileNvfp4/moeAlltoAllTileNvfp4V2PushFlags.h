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
 * Push-combine control plane for the packed-NVFP4 tileNvfp4 V2 family.
 *
 * This header defines a NEW raw-u64 flag ABI ("raw_u64_epoch_v1", layout
 * "compact_row_v2"). It is deliberately independent of the PULL
 * combine flags (the uint32 completionFlags / flagVal / dispatch flags in
 * MoeA2ACombineParamsV2 and TileNvfp4CombinePointersV2), which stay untouched.
 * The push plane must never alias or reuse those uint32 words.
 */
#pragma once

#include "moeAlltoAllTileNvfp4V2.h"

#include <cuda_runtime_api.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

// -----------------------------------------------------------------------------
// raw-u64 flag ABI (frozen).
//
// flag_abi_version = "raw_u64_epoch_v1"
// flag_layout_id   = "compact_row_v2"
//
// Every control word on this plane is a u64. Generations are monotonic u64:
// INVALID_GENERATION (0) is reserved and never opened; the first opened
// generation is 1. Producers publish readiness into a per-consumer row; a
// consumer gates on the readiness words in its own row before consuming.
// -----------------------------------------------------------------------------

// Reserved sentinel. A ready[]/push_generation word holding this value has not
// yet been opened for any epoch and must not be treated as ready.
constexpr std::uint64_t kPushInvalidGeneration = 0;

// The first (lowest) generation a bootstrap/open step may publish. Generations
// increase monotonically from here as u64.
constexpr std::uint64_t kPushFirstGeneration = 1;

// Number of producer-indexed ready slots per consumer row. Sized to cover the
// target EP range (E in {2,4,8,16,32}); indexed by producer rank P in [0, ep_size).
constexpr int kPushReadyCapacity = 32;

// Total bytes of one per-consumer control row (compact_row_v2):
//   ready[kPushReadyCapacity] u64 (256B) + push_generation u64 (8B)
//   + localGate u64 (8B) = 272B. The live MNNVL tensor uses its fabric
// stride(0) for peer hops; this constant is the size of one rank's segment.
constexpr int kPushControlSlabBytes = 272;

// Frozen string ids for the flag ABI. Downstream code and any serialized
// descriptor must compare against these verbatim.
inline constexpr char const kPushFlagAbiVersion[] = "raw_u64_epoch_v1";
inline constexpr char const kPushFlagLayoutId[] = "compact_row_v2";

// -----------------------------------------------------------------------------
// compact_row_v2 byte layout, per consumer row (stride kPushControlSlabBytes):
//
//   offset   0 : ready[kPushReadyCapacity] u64  (256 bytes, indexed by producer P)
//   offset 256 : push_generation u64            (8 bytes)
//   offset 264 : localGate u64                  (8 bytes)
//   total      : 272 bytes
// -----------------------------------------------------------------------------
constexpr int kPushReadyOffsetBytes = 0;
constexpr int kPushGenerationOffsetBytes
    = kPushReadyOffsetBytes + kPushReadyCapacity * static_cast<int>(sizeof(std::uint64_t));
constexpr int kPushLocalGateOffsetBytes = kPushGenerationOffsetBytes + static_cast<int>(sizeof(std::uint64_t));

static_assert(kPushReadyOffsetBytes == 0, "compact_row_v2 ready[] must start at offset 0");
static_assert(kPushGenerationOffsetBytes == 256, "compact_row_v2 push_generation must be at offset 256");
static_assert(kPushLocalGateOffsetBytes == 264, "compact_row_v2 localGate must be at offset 264");
static_assert(kPushLocalGateOffsetBytes + static_cast<int>(sizeof(std::uint64_t)) == kPushControlSlabBytes,
    "compact_row_v2 row math must total kPushControlSlabBytes (272)");
static_assert(kPushControlSlabBytes == 272, "compact_row_v2 control_slab_bytes is frozen at 272");

// -----------------------------------------------------------------------------
// Row accessors (already-hopped per-consumer base + intra-row offset).
//
// These accessors take `baseC`, the base of consumer C's row that has ALREADY
// had the symmetric MNNVL remote hop applied by the caller (baseC ==
// control_peer_base[C], which the live tensor derives as control_base + C *
// controlPlane.stride(0)). This mirrors the payload plane,
// where K1PushParams::packed_peer_base[C] is the already-hopped landing base
// (PushOp.cpp fillPackedPeerBases: workspaceBase + C*stride(0) + sfOffset). The
// control plane replicates the SAME hop through control_peer_base[C], so these
// accessors only add the intra-row byte offset and never re-apply C*stride.
// -----------------------------------------------------------------------------

// Pointer to producer P's ready slot within consumer C's already-hopped row
// (ready[P] == baseC + kPushReadyOffsetBytes + P*8).
__host__ __device__ inline std::uint64_t* pushReadyPtr(std::uint8_t* baseC, int producerP)
{
    return reinterpret_cast<std::uint64_t*>(
        baseC + kPushReadyOffsetBytes + static_cast<std::size_t>(producerP) * sizeof(std::uint64_t));
}

// Pointer to the push_generation word within an already-hopped consumer row.
__host__ __device__ inline std::uint64_t* pushGenerationPtr(std::uint8_t* baseC)
{
    return reinterpret_cast<std::uint64_t*>(baseC + kPushGenerationOffsetBytes);
}

// Pointer to the localGate word within an already-hopped consumer row.
__host__ __device__ inline std::uint64_t* pushLocalGatePtr(std::uint8_t* baseC)
{
    return reinterpret_cast<std::uint64_t*>(baseC + kPushLocalGateOffsetBytes);
}

// -----------------------------------------------------------------------------
// Device generation load/store helpers (promoted from the D0 open kernel so the
// K2 publish and K3 gate steps can read the DEVICE generation word rather than a
// host scalar). Copied verbatim from moeAlltoAllTileNvfp4V2PushD0.cu so the
// system-scope ordering idiom stays byte-identical across every push step.
// -----------------------------------------------------------------------------

// System-scope u64 load of a control word. Mirrors the ld.relaxed.sys spin loads
// in publishAndAcquireReadiness, widened to u64.
__device__ __forceinline__ std::uint64_t pushLoadRelaxedSysU64(std::uint64_t const* address)
{
#if defined(__CUDA_ARCH__)
    std::uint64_t observed;
    asm volatile("ld.relaxed.sys.u64 %0, [%1];" : "=l"(observed) : "l"(address) : "memory");
    return observed;
#else
    return *address;
#endif
}

// Release fence (system scope), then a relaxed u64 store: exactly the publish
// half of publishAndAcquireReadiness's release/acquire idiom, widened to u64.
// Used to make a freshly opened generation visible to remote consumers with the
// correct ordering against any prior writes on this rank.
__device__ __forceinline__ void pushStoreReleaseSysU64(std::uint64_t* address, std::uint64_t value)
{
#if defined(__CUDA_ARCH__)
    asm volatile("fence.release.sys;" ::: "memory");
    asm volatile("st.relaxed.sys.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
#else
    *address = value;
#endif
}

// -----------------------------------------------------------------------------
// Control-plane launch parameters (POD; pointers + scalars only).
//
// This carries ONLY the push-combine control plane. It deliberately holds no
// completionFlags / flagVal / bulkPeerMask -- those belong to the
// PULL combine and are passed through the untouched MoeA2ACombineParamsV2 path.
// -----------------------------------------------------------------------------
struct PushCombineControlParams
{
    // Base of this rank's raw-u64 control segment (compact_row_v2). Peer segments
    // are reached through control_peer_base[], whose addresses include the MNNVL
    // fabric stride. Use the pushXxx accessors only for offsets within one segment;
    // never index this plane as uint32.
    std::uint8_t* control_base;

    // Per-consumer already-hopped control-row bases, indexed by consumer rank C
    // in [0, ep_size). Mirrors K1PushParams::packed_peer_base: each entry is the
    // base of consumer C's compact_row_v2 row reached through the SAME symmetric
    // MNNVL hop the payload plane uses (workspaceBase + C*stride(0) + control
    // offset). A producer publishing readiness into consumer C's remote row uses
    // pushReadyPtr(control_peer_base[C], producerP); entries beyond ep_size are
    // unused. The control plane must replicate the payload hop exactly.
    std::uint8_t* control_peer_base[kMaxRanksTile];

    // Expansion parallel size E (number of ranks / rows / producer slots used).
    int ep_size;

    // This rank's global id. Serves as the producer index P for the K2 publish
    // step and as the consumer index C for the K3 gate and local steps.
    int rank_id;

    // Per-(producer P, consumer C) receive counts, or null on control-only
    // launches (bootstrap / generation-open) that do not inspect payload counts.
    int const* recv_counters;

    // Runtime fan-out / max-tokens-per-rank knob (r); mirrors the family's
    // runtime sizing scalar used by the payload path.
    int r_runtime;

    // This rank's NVLinkOneSided A2A round flag (moeAlltoAllKernels.cu flag_val),
    // or null. Each A2A round consumes two flag values (2N+1 for dispatch, 2N+2
    // for combine) and the dispatch double-buffers recv_counters by round
    // parity. Push-combine replaces the stock combine, so the D0 combine-open
    // advances this word by one in place of the stock prepare-combine kernel;
    // that keeps the dispatch-side round/parity protocol intact.
    std::uint32_t* a2a_flag_val;

    // NOTE: the generation (epoch) is a DEVICE word, not a host scalar. It lives
    // in the compact_row_v2 push_generation slot and is advanced device-side by
    // the D0 open kernel's captured read-modify-write, then read on-device by K2/
    // K3 via pushGenerationPtr + pushLoadRelaxedSysU64. It is deliberately NOT a
    // field here so a captured graph replay re-advances it by exactly one.
};

// -----------------------------------------------------------------------------
// Host launch declarations. Definitions live in the push-combine kernel .cu
// files. Bootstrap and D0-open are control-only; K23 publishes this rank's
// producer readiness then gates the consumer on its row (fused K2 publish + K3
// gate in one launch) before K4 consumes.
// -----------------------------------------------------------------------------
void launchPushControlBootstrap(PushCombineControlParams const& params, cudaStream_t stream);
void launchD0GenerationOpen(PushCombineControlParams const& params, cudaStream_t stream);
// Fused producer-publish + consumer-gate control kernel (single <<<1,32>>> launch).
// openGeneration=true also advances this rank's generation word (and the A2A round
// flag) first, as the D0 open does, for a dispatch that skipped launchD0GenerationOpen.
void launchK23Control(PushCombineControlParams const& params, cudaStream_t stream, bool openGeneration = false);

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
