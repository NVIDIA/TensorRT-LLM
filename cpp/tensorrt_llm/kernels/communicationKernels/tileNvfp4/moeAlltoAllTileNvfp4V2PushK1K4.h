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
 * Push-combine payload kernels K1 (producer push) and K4 (consumer top-k
 * reduce) for the packed-NVFP4 tileNvfp4 V2 family.
 *
 * These two kernels are the PAYLOAD plane of the push-combine path; the control
 * plane (raw-u64 readiness/generation/localGate flags, "raw_u64_epoch_v1" /
 * "compact_row_v2") lives in moeAlltoAllTileNvfp4V2PushFlags.h and is carried by
 * PushCombineControlParams. The split of responsibilities is strict:
 *
 *   K1 (launchK1Push): a PRODUCER rank P quantizes its BF16 activations with the
 *       family NVFP4 quantizer (quantizeAndStoreTile / quantizeSixteen /
 *       fp32VecToE2m1 / packedAbsMax) into the
 *       packed [Q|SF] 4480-B record layout (kPackedRecordBytes, TmaFormats.cuh)
 *       and PUSHES each record into the destination consumer's packed landing
 *       plane. K1 touches ONLY payload bytes; it never writes the flag plane
 *       (readiness is published by the separate K2 producer-publish step).
 *
 *   K4 (launchK4TopkReduce): a CONSUMER rank C, after gating on its own control
 *       row (localGate[C], acquired from control_base via pushLocalGatePtr), reads
 *       the per-route packed records that landed in its LOCAL packed plane and
 *       top-k reduces them into a BF16 output token, reusing the family decode +
 *       reduce (decodeBlock16 / decodeHalfBlock16 in TmaFormats.cuh,
 *       reduceRoute8Frozen / loadScalePackOncePerRoutePacked / packBf16Pair in
 *       TmaCommon.cuh). K4 holds no remote or completion-flag pointers.
 *
 * The PULL combine (MoeA2ACombineParamsV2, TileNvfp4CombinePointersV2,
 * uint32 completionFlags) stays the bitwise/bytewise reference and is untouched:
 * these structs reuse its route/slot inputs (topk_target_ranks /
 * topk_send_indices / recv_counters) and its packed landing plane (the
 * sf_recv_buffers view), but add no new numeric/quant behavior.
 */
#pragma once

#include "moeAlltoAllTileNvfp4V2.h"
#include "moeAlltoAllTileNvfp4V2PushFlags.h"

#include <cuda_runtime_api.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

// -----------------------------------------------------------------------------
// Packed record geometry (must match TmaFormats.cuh Nvfp4Format).
//
// One pushed/consumed record is the family's packed-interleaved [Q|SF] 4480-B
// block: Q 3584 B (kQAlignedBytes) + SF 896 B. K1 lands, and K4 reads, one such
// record per (producer P, producer token t) slot. Mirrors, and must stay equal
// to, Nvfp4Format::kPackedRecordBytes / kPackedRecordStrideBytes (4480). It is
// redeclared here (not included) so this control-plane-adjacent header keeps no
// device .cuh dependency; the family .cu that defines the kernels includes
// TmaFormats.cuh and static_asserts the two against each other.
// -----------------------------------------------------------------------------
constexpr int kK1PackedRecordBytes = 4480;

// K4 selected-shared-stride option. The default (4480) reads records at the
// natural packed stride; the 4496 variant pads each staged record to a 16-byte
// multiple boundary (shared-memory bank/occupancy tuning). Both decode the SAME 4480
// payload bytes through the frozen decoder; only the staging stride differs.
constexpr int kK1SelectedSharedStrideDefaultBytes = 4480;
constexpr int kK1SelectedSharedStrideAbBytes = 4496;

// -----------------------------------------------------------------------------
// Launch geometry for K1/K4.
//
// threads == 224 is the MANDATORY baseline row (identity string
// "deterministic_224_baseline_v1"): the deterministic, byte-exact-against-the-
// PULL-reference configuration. Any launch with blockDim.x != 224 must be rejected
// on the host (HOST_REJECT) rather than silently run a different shape.
//
// cta_cap and target_sm_count are resolved at launch time (0 here means
// "resolve from the device / occupancy at launch"): cta_cap is the per-wave CTA
// ceiling and target_sm_count the SM count of the resolved device (Rubin
// sm_107a target). s_alloc / s_active are the allocated vs. actively-used
// pipeline stage counts of the two-phase overlap skeleton (see the
// FullTokenWindow/HalfRecordWindow geometry in the TMA template); the baseline
// allocates 2 and uses 2.
// -----------------------------------------------------------------------------
struct K1Geometry
{
    int threads;         // block width; must be 224 or HOST_REJECT
    int cta_cap;         // per-wave CTA ceiling (0 => resolve at launch)
    int target_sm_count; // resolved device SM count (0 => resolve at launch)
    int s_alloc;         // allocated pipeline stages
    int s_active;        // actively-used pipeline stages
};

// The frozen baseline geometry row. cta_cap / target_sm_count are filled in by
// the launcher from the live device; threads/s_alloc/s_active are frozen.
constexpr K1Geometry kK1BaselineGeometry{224, /*cta_cap*/ 0, /*target_sm*/ 0, /*s_alloc*/ 2, /*s_active*/ 2};

// Frozen identity string for the mandatory baseline launch row. Host launch
// code and any serialized descriptor must compare against this verbatim.
inline constexpr char const kK1BaselineIdentity[] = "deterministic_224_baseline_v1";

// The one legal block width. blockDim.x != kK1BaselineThreads => HOST_REJECT.
constexpr int kK1BaselineThreads = 224;
static_assert(kK1BaselineGeometry.threads == kK1BaselineThreads, "the baseline geometry row must pin threads == 224");

// -----------------------------------------------------------------------------
// K1 producer-push launch parameters (POD; pointers + scalars only).
//
// Owned by a single PRODUCER rank P (== rank_id). K1 quantizes this rank's BF16
// activations into packed 4480-B records and pushes each into the destination
// consumer's packed landing plane. It carries NO completion / readiness flags:
// the push plane is payload-only here, and readiness is published separately by
// the fused K23 control kernel (launchK23Control, moeAlltoAllTileNvfp4V2PushFlags.h).
// -----------------------------------------------------------------------------
struct K1PushParams
{
    // Per-consumer destination packed landing bases, indexed by consumer rank C
    // in [0, ep_size). Each entry is the consumer's TIGHT packed [Q|SF] plane --
    // the SAME view the PULL path exposes as MoeA2ACombineParamsV2::
    // sf_recv_buffers[C] (peerBase + sfOffset). K1 writes the pushed record for
    // (this producer P, producer token t) at:
    //     dst = packed_peer_base[C] + (P * r_runtime + t) * kK1PackedRecordBytes
    // i.e. the landing slot is keyed by producer rank and producer token index,
    // not by dstIdx. Entries beyond ep_size are unused.
    std::uint8_t* packed_peer_base[kMaxRanksTile];

    // Per-(producer P, consumer C) receive counts. Reused verbatim from the
    // NVLinkOneSided dispatch recv_counters plane; here it bounds how many of
    // this producer's tokens land in each consumer's plane. May be null only on
    // a degenerate zero-token launch.
    //
    // The NVLinkOneSided dispatch double-buffers this plane by A2A round:
    // recv_counters[parity][source_rank], with parity = ((flag_val - 1) >> 1) & 1
    // (each round consumes two flag values, 2N+1 for dispatch and 2N+2 for
    // combine; see round_parity() in moeAlltoAllKernels.cu). When a2a_flag_val is
    // non-null, K1 selects the bank of the current round on device so captured
    // CUDA graphs stay replay-safe; when it is null the plane is single-banked.
    int const* recv_counters;
    std::uint32_t const* a2a_flag_val;

    // TOP_K route metadata, reused verbatim from MoeA2ACombineParamsV2
    // (topk_target_ranks / topk_send_indices). topk_target_ranks[t * top_k + s]
    // resolves the consumer rank C that this producer's token t routes to on
    // slot s (dstIdx < 0 marks an unused slot). Needed to pick packed_peer_base[C].
    int const* topk_target_ranks;
    int const* topk_send_indices;

    // BF16 source activation plane: the [ep_size*R, 7168] BAND plane (NOT a
    // [local_num_tokens, 7168] compact plane). A record from rank C at send-index
    // t lives at slot C*R + t; band C is valid over slots [C*R, C*R +
    // recv_counters[C]). One token every source_token_stride_bytes. Pointer is
    // void const* to mirror MoeA2ACombineParamsV2::prepare_payload; the kernel
    // reinterprets it as __nv_bfloat16 and runs the family quantizer.
    void const* source_activation;
    std::uint64_t source_token_stride_bytes; // 14336 (kBf16TokenBytes) for H7168 BF16

    int local_num_tokens;                    // number of this producer's local tokens (t in [0, local_num_tokens))
    int top_k;                               // 8 for DeepSeek-R1 (route slots per token)

    int ep_size;                             // expansion parallel size E in {2, 4, 8, 16, 32}
    int rank_id;                             // this rank's global id; serves as the PRODUCER index P
    int r_runtime;                           // runtime max-tokens-per-rank (R); the P*R+t slot stride factor

    K1Geometry geometry;                     // launch shape; threads must be 224 or HOST_REJECT

    // Packed layout identity. records_per_token == min(top_k, ep_size) restricted
    // to {2, 4, 8} (the family dispatch axis). packed_layout_version is the SOLE
    // identity axis pinning the packed-record decode contract: it must carry
    // kTileNvfp4Compact (== 2), i.e. it equals
    // static_cast<int>(MoeA2ACombinePayloadLayout::kTileNvfp4Compact), so producer
    // and consumer agree on the 4480-B [Q|SF] interpretation.
    int records_per_token;
    int packed_layout_version;
};

// -----------------------------------------------------------------------------
// K4 consumer top-k-reduce launch parameters (POD; pointers + scalars only).
//
// Owned by a single CONSUMER rank C (== rank_id). K4 gates on its own control
// row, then reads the packed records that landed LOCALLY and top-k reduces them
// into BF16 output tokens. It holds NO remote peer pointers and NO completion
// flags: it reads only its own landing plane and its own control row.
// -----------------------------------------------------------------------------
struct K4Params
{
    // This consumer's LOCAL packed [Q|SF] landing plane (== this rank's own
    // sf_recv_buffers[rank_id] view). For a route from producer P at producer
    // token t, K4 reads the landed record at:
    //     src = packed_local_base + (P * r_runtime + t) * kK1PackedRecordBytes
    // (the P*R+t slot mirrors K1's landing address).
    std::uint8_t* packed_local_base;

    // This consumer's OWN control row on the raw-u64 control plane (compact_row_v2),
    // ALREADY hopped by the host through the SAME symmetric MNNVL hop K3 uses:
    // control_base == controlBase + rank_id * controlPlane.stride(0) (i.e. it equals
    // control_peer_base[rank_id], NOT row 0 offset by the 272-B slab). K4 acquires
    // this consumer's localGate word via the single-argument, already-hopped
    // pushLocalGatePtr(control_base) (moeAlltoAllTileNvfp4V2PushFlags.h) before
    // reading any landed record; it never re-applies a rank_id row hop.
    std::uint8_t* control_base;

    // TOP_K route / slot metadata, reused verbatim from MoeA2ACombineParamsV2
    // (topk_target_ranks / topk_send_indices) plus the per-(P,C) recv_counters
    // plane. These identify, for each of this consumer's output tokens, which
    // producer routes contribute and where their records landed.
    int const* topk_target_ranks;
    int const* topk_send_indices;
    int const* recv_counters;

    // BF16 output plane for this consumer: contiguous [local_num_tokens, 7168],
    // one token every output_token_stride_bytes. void* mirrors
    // MoeA2ACombineParamsV2::output_data; the kernel writes __nv_bfloat16.
    void* output_data;
    std::uint64_t output_token_stride_bytes; // 14336 (kBf16TokenBytes) for H7168 BF16

    int local_num_tokens;                    // number of this consumer's output tokens
    int top_k;                               // 8 (route slots per token)

    int ep_size;                             // expansion parallel size E in {2, 4, 8, 16, 32}
    int rank_id;                             // this rank's global id; serves as the CONSUMER index C
    int r_runtime;                           // runtime max-tokens-per-rank (R); the P*R+t slot stride factor

    K1Geometry geometry;                     // launch shape; threads must be 224 or HOST_REJECT

    // Packed layout identity (must agree with the producing K1). As in K1,
    // packed_layout_version is the SOLE identity axis and must carry
    // kTileNvfp4Compact (== 2), i.e. it equals
    // static_cast<int>(MoeA2ACombinePayloadLayout::kTileNvfp4Compact).
    int records_per_token;
    int packed_layout_version;

    // Per-record staging stride option. kK1SelectedSharedStrideDefaultBytes
    // (4480) is the natural packed stride; kK1SelectedSharedStrideAbBytes (4496)
    // pads each staged record (bank/occupancy tuning). Both decode
    // the same 4480 payload bytes through the frozen decoder.
    int selected_shared_stride_bytes;
};

// -----------------------------------------------------------------------------
// Host launch declarations. Definitions live in the push-combine kernel .cu.
// Both reject any launch whose block width is not kK1BaselineThreads (224).
// -----------------------------------------------------------------------------
void launchK1Push(K1PushParams const& params, cudaStream_t stream);
void launchK4TopkReduce(K4Params const& params, cudaStream_t stream);

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
