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
 * PUSH-combine Torch/workspace op surface for the packed-NVFP4 tileNvfp4 V2
 * family. This file is ADDITIVE: it registers a NEW set of trtllm ops that drive
 * the push-combine kernels (D0/K1/K2/K3/K4) and never touches the PULL
 * combine surface, which stays the byte/bitwise reference.
 *
 * Ops registered under the `trtllm` library (each independently launchable and
 * microbenchmarkable):
 *
 *   moe_a2a_push_control_bootstrap   -- control-only: clear THIS rank's control
 *                                       segment on-device (paired with a collective
 *                                       barrier by the caller). Wraps
 *                                       launchPushControlBootstrap.
 *   moe_a2a_push_d0_generation_open  -- control-only: monotonically advance THIS
 *                                       rank's push_generation g-1 -> g (device
 *                                       RMW, replay-safe). Wraps launchD0GenerationOpen.
 *   moe_a2a_push_k1                  -- PRODUCER: quantize this rank's BF16 tokens
 *                                       into packed [Q|SF] 4480-B records and push
 *                                       each into the consumer's landing plane.
 *                                       Wraps launchK1Push.
 *   moe_a2a_push_k23                 -- control-only: FUSED producer-publish + consumer
 *                                       gate in one launch -- publish ready[C][P]=g for
 *                                       every consumer, then spin ready[C][*]==g and
 *                                       release localGate[C]. Wraps launchK23Control.
 *   moe_a2a_push_k4_topk_reduce      -- CONSUMER: gate on localGate[C], read the
 *                                       LOCAL landed packed records, top-k reduce to
 *                                       BF16. Wraps launchK4TopkReduce. Returns output.
 *   moe_a2a_push_combine             -- sequential K1 -> K23 -> K4 on the current
 *                                       stream under the common TensorRT-LLM PDL policy.
 *                                       Returns output.
 *
 * Validation split: this file owns the TORCH_CHECK host
 * surface (schema / tensor / layout / range / alignment checks and the
 * region-C alias proof); the launch-time kernel-parameter validators (the
 * blockDim!=224 HOST_REJECT, the {2,4,8} wire-axis checks, geometry pins) live
 * with the launchers in the push-combine .cu files and are NOT duplicated here.
 *
 * Region-C alias check (cft_region_c_alias_v1, host, fail-closed): the push packed
 * landing region C is STORAGE-ALIASED onto the SAME bytes the PULL combine sizes as its
 * per-rank SF plane (sf_recv_buffers[C] == peerBase + sfOffset). This file proves
 * the alias arithmetic before any base-allocation lookup and HOST_REJECTs every
 * unsafe alias; it never changes the PULL op surface nor its execution/sync.
 *
 * register_fake / meta notes: none of these ops reads tensor DATA on the host, and
 * every output shape is a pure function of the runtime scalar `local_num_tokens`
 * and the frozen hidden width (7168). A Python register_fake can therefore mint
 *   - () for control_bootstrap/d0/k1/k2/k3, and
 *   - torch.empty([local_num_tokens, 7168], dtype=bfloat16, device=...) for
 *     k4 / push_combine
 * with no device work and no data dependence (local_num_tokens may be a SymInt).
 */
#include "moeAlltoAllTileNvfp4V2.h"
#include "moeAlltoAllTileNvfp4V2PushFlags.h"
#include "moeAlltoAllTileNvfp4V2PushK1K4.h"

#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>

// Runtime identity for the push-combine op DSO, pinned independently from the
// DSO SHA. It is a DISTINCT symbol so a
// push-combine DSO can never be mistaken for -- or alias into -- the
// unified PULL DSO.
#ifndef TILE_PUSH_COMBINE_BUILD_ID
#define TILE_PUSH_COMBINE_BUILD_ID "tile_push_combine:v2"
#endif

extern "C" __attribute__((visibility("default"))) char const* trtllm_tile_push_combine_build_id()
{
    return TILE_PUSH_COMBINE_BUILD_ID;
}

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext::moe_comm
{
namespace
{

using kernels::moe_comm::K1Geometry;
using kernels::moe_comm::K1PushParams;
using kernels::moe_comm::K4Params;
using kernels::moe_comm::MoeA2ACombinePayloadLayout;
using kernels::moe_comm::MoeA2ACombineQuantFormat;
using kernels::moe_comm::MoeA2ACombineScaleLayout;
using kernels::moe_comm::PushCombineControlParams;
using kernels::moe_comm::kK1BaselineGeometry;
using kernels::moe_comm::kK1BaselineThreads;
using kernels::moe_comm::kK1PackedRecordBytes;
using kernels::moe_comm::kK1SelectedSharedStrideAbBytes;
using kernels::moe_comm::kK1SelectedSharedStrideDefaultBytes;
using kernels::moe_comm::kMaxRanksTile;
using kernels::moe_comm::kPushControlSlabBytes;
using kernels::moe_comm::kPushFirstGeneration;
using kernels::moe_comm::kPushReadyCapacity;

// -----------------------------------------------------------------------------
// Push-op metainfo layout (10 fields). The Python selector projects the
// NVLinkOneSided metainfo tensor (thop/moeAlltoAllMeta.h MoeA2ADataOffsets) BY
// NAME onto this fixed order (NVLinkOneSidedTileNvfp4V2._make_rc11_side_dso_metainfo),
// so this separately built DSO never depends on the drifting field count or
// order of the main library's metainfo enum. Kept file-local (inside the
// anonymous namespace) so this TU never ODR-collides with that enum.
// -----------------------------------------------------------------------------
enum Rc11MoeA2AMetaInfoIndex : int64_t
{
    RC11_FLAG_VAL_OFFSET_INDEX = 0,
    RC11_LOCAL_TOKEN_COUNTER_OFFSET_INDEX = 1,
    RC11_SEND_COUNTERS_OFFSET_INDEX = 2,
    RC11_RECV_COUNTERS_OFFSET_INDEX = 3,
    RC11_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = 4,
    RC11_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = 5,
    RC11_TOPK_TARGET_RANKS_OFFSET_INDEX = 6,
    RC11_TOPK_SEND_INDICES_OFFSET_INDEX = 7,
    RC11_EPLB_GATHERED_STATS_OFFSET_INDEX = 8,
    RC11_PAYLOAD_DATA_OFFSET_INDEX = 9,
    RC11_NUM_METAINFO_FIELDS = 10
};

// -----------------------------------------------------------------------------
// Frozen geometry constants (mirrored from the PULL op / the push headers).
// -----------------------------------------------------------------------------
constexpr std::uint64_t kAlignment = 128;
constexpr std::uint64_t kHiddenElements = 7168;
constexpr std::uint64_t kBf16TokenBytes = 14336;
constexpr std::uint64_t kNvfp4QPayloadBytesPerToken = 3584;
constexpr std::uint64_t kNvfp4SfPayloadBytesPerToken = 896;
constexpr std::uint64_t kQAlignedBytes = 3584;
constexpr std::uint64_t kPackedRecordStrideBytes = kNvfp4QPayloadBytesPerToken + kNvfp4SfPayloadBytesPerToken; // 4480
static_assert(kPackedRecordStrideBytes == static_cast<std::uint64_t>(kK1PackedRecordBytes),
    "op-side packed record stride must mirror the K1/K4 kernel record bytes (4480)");
constexpr int kTopK = 8;
constexpr int64_t kMinEpSize = 2;

// The frozen recipe enum id of the packed-interleaved [Q|SF] recipe (recipe 3).
// The push path is defined ONLY for this recipe; every other recipe is a
// split-plane / cross-recipe alias (HOST_REJECT).
constexpr int64_t kPackedRecipe = 3;

// -----------------------------------------------------------------------------
// RecipeDesc / kRecipeTable identical to the PULL combine's recipe table so
// the push op resolves recipe-IDENTITY against the SAME fixed 4-value enum
// (needed to fail-closed on cross-recipe aliasing). Only the packed recipe [3] is
// a legal push target; the other three rows exist purely so recipe resolution and
// the layout arithmetic stay byte-identical to the PULL op.
// -----------------------------------------------------------------------------
struct RecipeDesc
{
    std::uint32_t quant_format;
    std::uint32_t payload_layout;
    std::uint32_t scale_layout;
    std::uint32_t reserved0;
    std::uint64_t q_payload_bytes_per_token;
    std::uint64_t sf_payload_bytes_per_token;
    std::uint64_t sf_token_stride_bytes;
    std::uint64_t sf_plane_bytes_per_slot;
    bool dynamic_smem;
    char const* name;
};

constexpr RecipeDesc kRecipeTable[] = {
    // [0] nvfp4: tile-wise NVFP4 split-plane.
    {static_cast<std::uint32_t>(MoeA2ACombineQuantFormat::kTileNvfp4),
        static_cast<std::uint32_t>(MoeA2ACombinePayloadLayout::kTileNvfp4SplitPlane),
        static_cast<std::uint32_t>(MoeA2ACombineScaleLayout::kTile64Block16E4M3PlusF32LE), 0,
        kNvfp4QPayloadBytesPerToken, kNvfp4SfPayloadBytesPerToken, kNvfp4SfPayloadBytesPerToken,
        kNvfp4SfPayloadBytesPerToken, false, "nvfp4"},
    // [1] fp8_e4m3: scale-free saturating E4M3 split-plane (no SF plane).
    {static_cast<std::uint32_t>(MoeA2ACombineQuantFormat::kFp8E4M3),
        static_cast<std::uint32_t>(MoeA2ACombinePayloadLayout::kTileNvfp4SplitPlane),
        static_cast<std::uint32_t>(MoeA2ACombineScaleLayout::kNone), 0, 7168, 0, 0, 0, true, "fp8_e4m3"},
    // [2] bf16: passthrough split-plane (no quantization, no SF plane).
    {static_cast<std::uint32_t>(MoeA2ACombineQuantFormat::kNone),
        static_cast<std::uint32_t>(MoeA2ACombinePayloadLayout::kTileNvfp4SplitPlane),
        static_cast<std::uint32_t>(MoeA2ACombineScaleLayout::kNone), 0, kBf16TokenBytes, 0, 0, 0, true, "bf16"},
    // [3] nvfp4_packed: compact [Q|SF] plane; reserved0 carries q_aligned.
    {static_cast<std::uint32_t>(MoeA2ACombineQuantFormat::kTileNvfp4),
        static_cast<std::uint32_t>(MoeA2ACombinePayloadLayout::kTileNvfp4Compact),
        static_cast<std::uint32_t>(MoeA2ACombineScaleLayout::kTile64Block16E4M3PlusF32LE),
        static_cast<std::uint32_t>(kQAlignedBytes), kNvfp4QPayloadBytesPerToken, kPackedRecordStrideBytes,
        kPackedRecordStrideBytes, kPackedRecordStrideBytes, false, "nvfp4_packed"},
};

static_assert(
    sizeof(kRecipeTable) / sizeof(kRecipeTable[0]) == 4, "the user-frozen recipe enum has exactly four values");

RecipeDesc const& validateRecipe(int64_t recipe)
{
    TORCH_CHECK(recipe >= 0 && recipe <= 3,
        "recipe must be 0 (nvfp4) | 1 (fp8_e4m3) | 2 (bf16) | 3 (nvfp4_packed); got ", recipe);
    return kRecipeTable[recipe];
}

// Device dispatch axis: records_per_token = min(top_k, ep_size) restricted to
// {2,4,8} (as in the PULL combine). K1 and K4 both accept this wire axis;
// K4 maps width 2 to Route2 and widths 4/8 to Route8.
int recordsPerTokenFor(int64_t epSize)
{
    return epSize == 2 ? 2 : epSize == 4 ? 4 : 8;
}

// D0/bootstrap/K1/K4 use launchWithPdlWhenEnabled and inherit the common TensorRT-LLM
// getEnvEnablePDL() policy, including its default when TRTLLM_ENABLE_PDL is
// unset. K23 remains the same plain control launch in either mode.

// -----------------------------------------------------------------------------
// Checked-arithmetic helpers (overflow behaviour matches the PULL op's host
// surface).
// -----------------------------------------------------------------------------
std::uint64_t checkedAdd(std::uint64_t left, std::uint64_t right, char const* what)
{
    TORCH_CHECK(left <= std::numeric_limits<std::uint64_t>::max() - right, what, " overflows uint64");
    return left + right;
}

std::uint64_t checkedMul(std::uint64_t left, std::uint64_t right, char const* what)
{
    TORCH_CHECK(left == 0 || right <= std::numeric_limits<std::uint64_t>::max() / left, what, " overflows uint64");
    return left * right;
}

std::uint64_t align128(std::uint64_t value)
{
    return checkedAdd(value, kAlignment - 1, "128-byte alignment") & ~(kAlignment - 1);
}

struct CheckedLayout
{
    std::uint64_t staticSlots;
    std::uint64_t qEnd;
    std::uint64_t sfOffset;
    std::uint64_t sfEnd;
};

// Layout arithmetic (as the PULL op's validateAndCalculateLayout). This
// is PURE arithmetic on the runtime scalars: it performs NO base-allocation
// pointer lookup, so it can (and must) run before any workspace.data_ptr() read
// in the recipe-first ordering.
CheckedLayout validateAndCalculateLayout(RecipeDesc const& desc, int64_t epSize, int64_t runtimeMaxTokensPerRank,
    int64_t graphStaticMaxTokensPerRank, int64_t qPlaneOffset, int64_t workspaceRowBytes)
{
    TORCH_CHECK(epSize >= kMinEpSize && epSize <= kMaxRanksTile,
        "ep_size must be in [2, kMaxRanksTile=", static_cast<int>(kMaxRanksTile), "]; got ", epSize);
    TORCH_CHECK(runtimeMaxTokensPerRank > 0, "runtime_max_tokens_per_rank must be positive");
    TORCH_CHECK(graphStaticMaxTokensPerRank >= runtimeMaxTokensPerRank,
        "graph_static_max_tokens_per_rank must be >= runtime_max_tokens_per_rank");
    TORCH_CHECK(runtimeMaxTokensPerRank <= INT_MAX && graphStaticMaxTokensPerRank <= INT_MAX,
        "R and M must fit the V2 int/uint32 ABI");
    TORCH_CHECK(qPlaneOffset >= 0 && qPlaneOffset % static_cast<int64_t>(kAlignment) == 0,
        "q_plane_offset must be nonnegative and 128-byte aligned");
    TORCH_CHECK(workspaceRowBytes >= 0, "workspace row size must be nonnegative");

    auto const staticSlots = checkedMul(
        static_cast<std::uint64_t>(epSize), static_cast<std::uint64_t>(graphStaticMaxTokensPerRank), "E*M");
    auto const runtimeSlots
        = checkedMul(static_cast<std::uint64_t>(epSize), static_cast<std::uint64_t>(runtimeMaxTokensPerRank), "E*R");
    TORCH_CHECK(runtimeSlots <= static_cast<std::uint64_t>(INT_MAX), "E*R must fit the int-indexed runtime slot bound");
    auto const qBytes = checkedMul(staticSlots, kBf16TokenBytes, "source reservation");
    auto const qEnd = checkedAdd(static_cast<std::uint64_t>(qPlaneOffset), qBytes, "source end");
    auto const sfOffset = align128(qEnd);
    auto const sfBytes = checkedMul(staticSlots, desc.sf_plane_bytes_per_slot, "second-plane reservation");
    auto const sfEnd = checkedAdd(sfOffset, sfBytes, "second-plane end");
    TORCH_CHECK(sfEnd <= static_cast<std::uint64_t>(workspaceRowBytes), "tile push ", desc.name,
        " workspace is too small: plane_end=", sfEnd, ", workspace row bytes=", workspaceRowBytes);
    return {staticSlots, qEnd, sfOffset, sfEnd};
}

void validateWorkspace(torch::Tensor const& workspace, int64_t epRank, int64_t epSize)
{
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be [ep_size, size_per_rank_bytes]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal ep_size");
    TORCH_CHECK(workspace.stride(1) == 1, "workspace rows must be byte-contiguous");
    TORCH_CHECK(workspace.stride(0) >= workspace.size(1), "workspace rank rows must not overlap");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "ep_rank must be in [0, ep_size)");
}

int64_t const* validateMetainfo(torch::Tensor const& metainfo)
{
    CHECK_CPU(metainfo);
    CHECK_TYPE(metainfo, torch::kInt64);
    CHECK_CONTIGUOUS(metainfo);
    TORCH_CHECK(metainfo.dim() == 1 && metainfo.size(0) == static_cast<int64_t>(RC11_NUM_METAINFO_FIELDS),
        "metainfo must be a CPU int64 tensor with RC11_NUM_METAINFO_FIELDS entries");
    return metainfo.data_ptr<int64_t>();
}

void validateRange(int64_t offset, std::uint64_t bytes, int64_t rowBytes, char const* name)
{
    TORCH_CHECK(offset >= 0, name, " offset must be nonnegative");
    auto const end = checkedAdd(static_cast<std::uint64_t>(offset), bytes, name);
    TORCH_CHECK(end <= static_cast<std::uint64_t>(rowBytes), name, " exceeds the workspace row");
}

void validateInt32Range(int64_t offset, std::uint64_t bytes, int64_t rowBytes, char const* name)
{
    validateRange(offset, bytes, rowBytes, name);
    TORCH_CHECK(offset % static_cast<int64_t>(alignof(std::uint32_t)) == 0, name, " must be 4-byte aligned");
}

// Validate the raw-u64 control plane (compact_row_v2). Like the payload workspace,
// this is the SYMMETRIC MNNVL plane: ONE 2-D CUDA uint8 tensor [ep_size, row_bytes]
// where consumer C's control row is reached by the SAME remote hop the payload plane
// uses (base + C*stride(0)). Each row is byte-contiguous and at least
// kPushControlSlabBytes (272B) wide. This is a separate plane and never aliases the
// PULL uint32 flag words. Returns THIS rank's base for the peer table.
std::uint8_t* validateControlPlane(torch::Tensor const& controlPlane, int64_t epSize)
{
    CHECK_TH_CUDA(controlPlane);
    CHECK_TYPE(controlPlane, torch::kUInt8);
    TORCH_CHECK(controlPlane.dim() == 2, "push control plane must be a 2-D [ep_size, control_row_bytes] uint8 tensor");
    TORCH_CHECK(epSize >= 1 && epSize <= kPushReadyCapacity,
        "push control plane ep_size must be in [1, kPushReadyCapacity=", kPushReadyCapacity, "]; got ", epSize);
    TORCH_CHECK(controlPlane.size(0) == epSize, "push control plane first dimension must equal ep_size; got ",
        controlPlane.size(0));
    TORCH_CHECK(controlPlane.stride(1) == 1, "push control plane rows must be byte-contiguous");
    TORCH_CHECK(controlPlane.stride(0) >= static_cast<int64_t>(kPushControlSlabBytes),
        "push control plane row stride must be >= kPushControlSlabBytes (", kPushControlSlabBytes, "); got ",
        controlPlane.stride(0));
    auto* base = controlPlane.data_ptr<std::uint8_t>();
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(base) % alignof(std::uint64_t) == 0,
        "push control plane base must be u64-aligned for the raw-u64 flag ABI");
    return base;
}

// Fill the control-plane peer table: control_peer_base[C] is consumer C's already-
// hopped control-row base (controlBase + C*stride(0)), reached through the SAME
// symmetric MNNVL hop the payload plane uses in fillPackedPeerBases. A producer
// publishing readiness into consumer C's remote row uses pushReadyPtr(
// control_peer_base[C], producerP); entries beyond ep_size stay unused.
void fillControlPeerBases(
    PushCombineControlParams& params, torch::Tensor const& controlPlane, std::uint8_t* controlBase, int epSize)
{
    for (int peer = 0; peer < epSize; ++peer)
    {
        params.control_peer_base[peer] = controlBase + peer * controlPlane.stride(0);
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(params.control_peer_base[peer]) % alignof(std::uint64_t) == 0,
            "every peer control-row base must be u64-aligned for the raw-u64 flag ABI");
    }
}

// -----------------------------------------------------------------------------
// cft_region_c_alias_v1 (host, fail-closed).
//
// Proves that the push packed landing region C is a pure STORAGE ALIAS onto the
// SAME bytes the PULL combine sizes as its per-rank SF plane, and
// HOST_REJECTs every unsafe alias. Storage alias only: it changes neither the
// PULL op surface nor its execution/sync.
//
// Layout (per rank row):
//   c_begin       = sf_offset = align128(q_end)                 (== PULL SF start)
//   packed_live   = [c_begin, c_begin + E*R_runtime*4480)       (records K1 may write)
//   packed_end    = c_begin + E*M_layout*4480                   (== PULL SF end)
//   c_end         = workspace_row_bytes                         (region-C storage end)
// with c_begin <= packed_live_end <= packed_end <= c_end and R_runtime <= M_layout.
//
// POISON contract (asserted structurally, honoured by the kernels): the inactive
// packed capacity [packed_live_end, packed_end) and the unused C tail
// [packed_end, c_end) are NEVER written by K1 (which only lands slots P*R+t with
// t < local_num_tokens <= R, P < E) and NEVER read as live by K4, so both stay
// poison exactly as the PULL combine leaves them.
//
// recipe-identity is resolved (payload_layout == compact) BEFORE any caller does a
// base-allocation lookup; a non-packed recipe is a split-plane / cross-recipe alias
// and is rejected here.
struct RegionCAlias
{
    std::uint64_t cBegin;        // sf_offset == align128(q_end)
    std::uint64_t packedLiveEnd; // sf_offset + E*R_runtime*4480
    std::uint64_t packedEnd;     // sf_offset + E*M_layout*4480 (== PULL SF end)
    std::uint64_t cEnd;          // workspace row bytes
};

RegionCAlias validateRegionCAliasV1(RecipeDesc const& desc, int64_t epSize, int64_t runtimeMaxTokensPerRank,
    int64_t graphStaticMaxTokensPerRank, int recordsPerToken, int blockDimX, bool cftIncumbentLive,
    int64_t workspaceRowBytes, CheckedLayout const& checked)
{
    // (1) recipe-IDENTITY, resolved before base allocation lookup. Only the packed
    // compact recipe may alias region C; the split-plane recipes
    // and any other recipe are cross-recipe aliases -> HOST_REJECT.
    TORCH_CHECK(desc.payload_layout == static_cast<std::uint32_t>(MoeA2ACombinePayloadLayout::kTileNvfp4Compact),
        "cft_region_c_alias_v1: push-combine requires the packed compact recipe "
        "(recipe ",
        kPackedRecipe, ", kTileNvfp4Compact); recipe '", desc.name,
        "' is a legacy-CFT / cross-recipe alias (HOST_REJECT)");
    TORCH_CHECK(desc.sf_plane_bytes_per_slot == kPackedRecordStrideBytes,
        "cft_region_c_alias_v1: packed slot stride must be kK1PackedRecordBytes (4480); got ",
        desc.sf_plane_bytes_per_slot);

    // (2) blockDim != 224 is a HOST_REJECT (the one legal push launch width).
    TORCH_CHECK(blockDimX == kK1BaselineThreads, "cft_region_c_alias_v1: push launch blockDim.x must be ",
        kK1BaselineThreads, " (deterministic_224_baseline_v1); got ", blockDimX, " (HOST_REJECT)");

    // (3) unsupported tuple: the push consumer path (K4) is defined for EP in
    // {2,4,8,16,32} with records_per_token == min(top_k, ep_size) in {2,4,8}. Anything else is
    // an unsupported (recipe, ep_size, records_per_token) tuple -> HOST_REJECT.
    TORCH_CHECK(epSize == 2 || epSize == 4 || epSize == 8 || epSize == 16 || epSize == 32,
        "cft_region_c_alias_v1: push-combine supports EP in {2,4,8,16,32}; got ep_size=", epSize,
        " (unsupported tuple, HOST_REJECT)");
    TORCH_CHECK(recordsPerToken == 2 || recordsPerToken == 4 || recordsPerToken == 8,
        "cft_region_c_alias_v1: records_per_token must be 2, 4, or 8; got ", recordsPerToken,
        " (unsupported tuple, HOST_REJECT)");
    TORCH_CHECK(recordsPerToken == recordsPerTokenFor(epSize), "cft_region_c_alias_v1: records_per_token ",
        recordsPerToken, " != min(top_k, ep_size) for ep_size=", epSize, " (unsupported tuple, HOST_REJECT)");

    // (4) second-live-owner: region C is a single-live-owner alias. If the
    // CFT PULL combine is concurrently a live owner of these bytes, the push alias
    // must NOT run (it would race it) -> HOST_REJECT.
    TORCH_CHECK(!cftIncumbentLive,
        "cft_region_c_alias_v1: region C already has a live incumbent-CFT owner; the push "
        "alias may not claim it as a second live owner (HOST_REJECT)");

    // (5) R_runtime <= M_layout.
    TORCH_CHECK(runtimeMaxTokensPerRank <= graphStaticMaxTokensPerRank, "cft_region_c_alias_v1: R_runtime (",
        runtimeMaxTokensPerRank, ") must be <= M_layout (", graphStaticMaxTokensPerRank, ")");

    // (6) c_begin == sf_offset == align128(q_end).
    auto const cBegin = checked.sfOffset;
    TORCH_CHECK(cBegin == align128(checked.qEnd), "cft_region_c_alias_v1: region-C begin must equal align128(q_end)");

    // (7) c_begin <= packed_live_end <= packed_end <= c_end, with packed_end pinned
    // to the PULL SF-plane end (proof the alias is the SAME storage, not a
    // parallel region).
    auto const liveSlots = checkedMul(
        static_cast<std::uint64_t>(epSize), static_cast<std::uint64_t>(runtimeMaxTokensPerRank), "E*R live slots");
    auto const layoutSlots = checkedMul(static_cast<std::uint64_t>(epSize),
        static_cast<std::uint64_t>(graphStaticMaxTokensPerRank), "E*M layout slots");
    auto const packedLiveEnd
        = checkedAdd(cBegin, checkedMul(liveSlots, kPackedRecordStrideBytes, "live packed bytes"), "packed live end");
    auto const packedEnd
        = checkedAdd(cBegin, checkedMul(layoutSlots, kPackedRecordStrideBytes, "layout packed bytes"), "packed end");
    auto const cEnd = static_cast<std::uint64_t>(workspaceRowBytes);

    TORCH_CHECK(cBegin <= packedLiveEnd, "cft_region_c_alias_v1: c_begin must be <= packed_live_end");
    TORCH_CHECK(packedLiveEnd <= packedEnd, "cft_region_c_alias_v1: packed_live_end (", packedLiveEnd,
        ") must be <= packed_end (", packedEnd, ")");
    TORCH_CHECK(packedEnd <= cEnd, "cft_region_c_alias_v1: packed_end (", packedEnd,
        ") must be <= region-C end / workspace "
        "row (",
        cEnd, ")");
    TORCH_CHECK(packedEnd == checked.sfEnd,
        "cft_region_c_alias_v1: packed_end must equal the incumbent SF-plane end (storage alias "
        "identity); packed_end=",
        packedEnd, " sf_end=", checked.sfEnd);

    return {cBegin, packedLiveEnd, packedEnd, cEnd};
}

// -----------------------------------------------------------------------------
// Shared per-rank base-pointer resolution. Runs AFTER validateRegionCAliasV1 so
// recipe-identity is proven before any base-allocation lookup.
// -----------------------------------------------------------------------------
struct RankBases
{
    std::uint8_t* workspaceBase;
    std::uint8_t* rankBase;        // this rank's row
    std::uint8_t* localQBase;      // rankBase + q_plane_offset (BF16 source plane)
    std::uint8_t* localPackedBase; // rankBase + sf_offset (this rank's landing plane)
    int64_t const* offsets;        // side-DSO metainfo offsets, based at rankBase
};

RankBases resolveRankBases(torch::Tensor const& workspace, int64_t const* offsets, int64_t epRank, int64_t qPlaneOffset,
    CheckedLayout const& checked, int64_t rowBytes)
{
    RankBases bases{};
    bases.workspaceBase = workspace.data_ptr<std::uint8_t>();
    bases.rankBase = bases.workspaceBase + epRank * workspace.stride(0);
    bases.localQBase = bases.rankBase + qPlaneOffset;
    bases.localPackedBase = bases.rankBase + checked.sfOffset;
    bases.offsets = offsets;

    // Auxiliary-metadata range checks (as the PULL host surface).
    validateRange(offsets[RC11_PAYLOAD_DATA_OFFSET_INDEX], 0, rowBytes, "payload_data");
    TORCH_CHECK(offsets[RC11_PAYLOAD_DATA_OFFSET_INDEX] % static_cast<int64_t>(kAlignment) == 0,
        "payload_data must be 128-byte aligned");
    TORCH_CHECK(
        qPlaneOffset >= offsets[RC11_PAYLOAD_DATA_OFFSET_INDEX], "q_plane_offset must not overlap auxiliary metadata");
    // recv_counters[2][ep_size]: the NVLinkOneSided dispatch double-buffers the
    // per-source counts by A2A round parity (see K1PushParams::a2a_flag_val).
    validateInt32Range(offsets[RC11_RECV_COUNTERS_OFFSET_INDEX],
        checkedMul(static_cast<std::uint64_t>(2 * workspace.size(0)), sizeof(int), "recv counters"), rowBytes,
        "recv counters");
    validateInt32Range(offsets[RC11_FLAG_VAL_OFFSET_INDEX], sizeof(std::uint32_t), rowBytes, "flag_val");

    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(bases.localQBase) % kAlignment == 0,
        "workspace Q source base must be 128-byte aligned");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(bases.localPackedBase) % kAlignment == 0,
        "region-C packed landing base must be 128-byte aligned");
    return bases;
}

// Validate + resolve the per-token TOP_K route arrays for `localNumTokens` tokens.
void validateRouteArrays(int64_t const* offsets, int64_t localNumTokens, int64_t rowBytes)
{
    auto const routeBytes = checkedMul(
        checkedMul(static_cast<std::uint64_t>(localNumTokens), static_cast<std::uint64_t>(kTopK), "route entries"),
        sizeof(int), "route bytes");
    validateInt32Range(offsets[RC11_TOPK_TARGET_RANKS_OFFSET_INDEX], routeBytes, rowBytes, "topk_target_ranks");
    validateInt32Range(offsets[RC11_TOPK_SEND_INDICES_OFFSET_INDEX], routeBytes, rowBytes, "topk_send_indices");
}

// Fill the K1 destination landing bases: packed_peer_base[C] is the consumer C's
// TIGHT packed [Q|SF] plane view (peerBase + sf_offset) -- byte-identical to the
// MoeA2ACombineParamsV2::sf_recv_buffers[C]. K1 lands (P*R+t) records there.
void fillPackedPeerBases(K1PushParams& params, torch::Tensor const& workspace, std::uint8_t* workspaceBase,
    CheckedLayout const& checked, int epSize)
{
    for (int peer = 0; peer < epSize; ++peer)
    {
        auto* peerBase = workspaceBase + peer * workspace.stride(0);
        params.packed_peer_base[peer] = peerBase + checked.sfOffset;
        TORCH_CHECK(reinterpret_cast<std::uintptr_t>(params.packed_peer_base[peer]) % kAlignment == 0,
            "every peer region-C packed base must be 128-byte aligned");
    }
}

// Resolve the K1 BF16 source-activation pointer. The source is the [ep_size*R, 7168]
// BAND plane (a record from rank C at send-index t lives at slot C*R+t; band C is
// valid over [C*R, C*R+recv_counters[C])), NOT a compact [local_num_tokens, 7168]
// plane. payload_in_workspace: read the workspace Q plane (rankBase + q_plane_offset)
// which is the [E, R, 7168] band plane; external: a contiguous BF16 [>=E*R, 7168] plane.
void const* resolveK1Source(torch::Tensor const& payload, bool payloadInWorkspace, std::uint8_t* localQBase,
    int64_t epSize, int64_t runtimeMaxTokensPerRank, torch::Tensor const& workspace)
{
    CHECK_TH_CUDA(payload);
    CHECK_CONTIGUOUS(payload);
    CHECK_TYPE(payload, torch::kBFloat16);
    TORCH_CHECK(payload.dim() == 2 && payload.size(1) == static_cast<int64_t>(kHiddenElements),
        "K1 payload must be a contiguous BF16 [ep_size*R, 7168] band plane");
    auto const bandRows = checkedMul(static_cast<std::uint64_t>(epSize),
        static_cast<std::uint64_t>(runtimeMaxTokensPerRank), "K1 band plane rows (E*R)");
    TORCH_CHECK(static_cast<std::uint64_t>(payload.size(0)) >= bandRows,
        "K1 payload band plane must hold >= ep_size*R (", bandRows, ") rows; got ", payload.size(0));
    TORCH_CHECK(payload.device() == workspace.device(), "K1 payload and workspace must be on the same CUDA device");
    if (payloadInWorkspace)
    {
        TORCH_CHECK(payload.data_ptr() == localQBase,
            "payload_in_workspace=true requires payload.data_ptr == workspace local Q base");
        return localQBase;
    }
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(payload.data_ptr()) % 32 == 0,
        "K1 external BF16 source must be 32-byte aligned");
    return payload.data_ptr();
}

// -----------------------------------------------------------------------------
// K1 producer-push param builder.
// -----------------------------------------------------------------------------
K1PushParams buildK1Params(torch::Tensor const& payload, int64_t localNumTokens, torch::Tensor const& workspace,
    RankBases const& bases, CheckedLayout const& checked, int64_t runtimeMaxTokensPerRank, int64_t epRank,
    int64_t epSize, bool payloadInWorkspace, int recordsPerToken)
{
    K1PushParams params{};
    fillPackedPeerBases(params, workspace, bases.workspaceBase, checked, static_cast<int>(epSize));
    params.recv_counters
        = reinterpret_cast<int const*>(bases.rankBase + bases.offsets[RC11_RECV_COUNTERS_OFFSET_INDEX]);
    params.a2a_flag_val
        = reinterpret_cast<std::uint32_t const*>(bases.rankBase + bases.offsets[RC11_FLAG_VAL_OFFSET_INDEX]);
    params.topk_target_ranks
        = reinterpret_cast<int const*>(bases.rankBase + bases.offsets[RC11_TOPK_TARGET_RANKS_OFFSET_INDEX]);
    params.topk_send_indices
        = reinterpret_cast<int const*>(bases.rankBase + bases.offsets[RC11_TOPK_SEND_INDICES_OFFSET_INDEX]);
    params.source_activation
        = resolveK1Source(payload, payloadInWorkspace, bases.localQBase, epSize, runtimeMaxTokensPerRank, workspace);
    params.source_token_stride_bytes = kBf16TokenBytes;
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.top_k = kTopK;
    params.ep_size = static_cast<int>(epSize);
    params.rank_id = static_cast<int>(epRank);
    params.r_runtime = static_cast<int>(runtimeMaxTokensPerRank);
    params.geometry = kK1BaselineGeometry;
    TORCH_CHECK(recordsPerToken == recordsPerTokenFor(epSize), "K1 records_per_token (", recordsPerToken,
        ") must equal min(top_k, ep_size) for ep_size=", epSize);
    params.records_per_token = recordsPerToken;
    // K1 pins the packed identity by the compact layout enum (kTileNvfp4Compact == 2),
    // matching buildK4Params so producer and consumer agree on the 4480-B [Q|SF]
    // record interpretation.
    params.packed_layout_version = static_cast<int>(MoeA2ACombinePayloadLayout::kTileNvfp4Compact);
    return params;
}

// -----------------------------------------------------------------------------
// K4 consumer top-k-reduce param builder.
// -----------------------------------------------------------------------------
K4Params buildK4Params(std::uint8_t* controlBase, torch::Tensor const& controlPlane, torch::Tensor const& output,
    int64_t localNumTokens, RankBases const& bases, int64_t runtimeMaxTokensPerRank, int64_t epRank, int64_t epSize,
    int recordsPerToken, int64_t selectedSharedStrideBytes)
{
    TORCH_CHECK(selectedSharedStrideBytes == kK1SelectedSharedStrideDefaultBytes
            || selectedSharedStrideBytes == kK1SelectedSharedStrideAbBytes,
        "K4 selected_shared_stride_bytes must be ", kK1SelectedSharedStrideDefaultBytes, " (default) or ",
        kK1SelectedSharedStrideAbBytes, " (A/B); got ", selectedSharedStrideBytes);

    K4Params params{};
    params.packed_local_base = bases.localPackedBase;
    // K4 gates on its OWN control row via a single-argument (already-hopped)
    // pushLocalGatePtr, so hand it the own-row base reached through the SAME
    // symmetric MNNVL hop K3 uses for control_peer_base[rank_id] (base +
    // rank_id*stride(0)). Passing row 0 would desync the K3->K4 gate for every
    // rank_id > 0 because stride(0) is the fabric rank stride, not the 272-B segment.
    params.control_base = controlBase + epRank * controlPlane.stride(0);
    params.topk_target_ranks
        = reinterpret_cast<int const*>(bases.rankBase + bases.offsets[RC11_TOPK_TARGET_RANKS_OFFSET_INDEX]);
    params.topk_send_indices
        = reinterpret_cast<int const*>(bases.rankBase + bases.offsets[RC11_TOPK_SEND_INDICES_OFFSET_INDEX]);
    params.recv_counters
        = reinterpret_cast<int const*>(bases.rankBase + bases.offsets[RC11_RECV_COUNTERS_OFFSET_INDEX]);
    params.output_data = output.data_ptr();
    params.output_token_stride_bytes = kBf16TokenBytes;
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.top_k = kTopK;
    params.ep_size = static_cast<int>(epSize);
    params.rank_id = static_cast<int>(epRank);
    params.r_runtime = static_cast<int>(runtimeMaxTokensPerRank);
    params.geometry = kK1BaselineGeometry;
    TORCH_CHECK(recordsPerToken == recordsPerTokenFor(epSize), "K4 records_per_token (", recordsPerToken,
        ") must equal min(top_k, ep_size) for ep_size=", epSize);
    params.records_per_token = recordsPerToken;
    // K4 keys the packed identity by the compact layout enum (its validator pins
    // packed_layout_version == kTileNvfp4Compact), matching buildK1Params.
    params.packed_layout_version = static_cast<int>(MoeA2ACombinePayloadLayout::kTileNvfp4Compact);
    params.selected_shared_stride_bytes = static_cast<int>(selectedSharedStrideBytes);
    return params;
}

// Build the control-plane params. The generation (epoch) is NOT a field: it is a
// DEVICE word in the compact_row_v2 push_generation slot, advanced device-side by
// the D0 open kernel's captured RMW and read on-device by K2/K3. The peer table is
// filled from the symmetric control plane (control_peer_base[C] = base + C*stride(0)).
PushCombineControlParams buildControlParams(std::uint8_t* controlBase, torch::Tensor const& controlPlane,
    int64_t epRank, int64_t epSize, std::uint32_t* a2aFlagVal = nullptr)
{
    PushCombineControlParams params{};
    params.control_base = controlBase;
    fillControlPeerBases(params, controlPlane, controlBase, static_cast<int>(epSize));
    params.ep_size = static_cast<int>(epSize);
    params.rank_id = static_cast<int>(epRank);
    params.recv_counters = nullptr; // control-only steps never inspect payload counts
    params.r_runtime = 0;
    params.a2a_flag_val = a2aFlagVal;
    return params;
}

} // namespace

// =============================================================================
// Op: D0 generation-open (control-only). Device RMW advances THIS rank's combine
// push_generation (the compact_row_v2 device word); there is no host generation
// scalar to bake in. It also advances this rank's NVLinkOneSided A2A round flag
// (flag_val, 2N+1 -> 2N+2) in place of the stock prepare-combine kernel, so the
// dispatch round protocol (including its recv_counters parity banks) stays
// intact when Push-Combine replaces the stock combine. register_fake: nothing.
// =============================================================================
void moeA2APushD0GenerationOpenOp(torch::Tensor const& controlPlane, torch::Tensor const& workspace,
    torch::Tensor const& metainfo, int64_t epRank, int64_t epSize)
{
    auto* controlBase = validateControlPlane(controlPlane, epSize);
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "ep_rank must be in [0, ep_size)");
    validateWorkspace(workspace, epRank, epSize);
    auto const* offsets = validateMetainfo(metainfo);
    validateInt32Range(offsets[RC11_FLAG_VAL_OFFSET_INDEX], sizeof(std::uint32_t), workspace.size(1), "flag_val");
    auto* a2aFlagVal = reinterpret_cast<std::uint32_t*>(
        workspace.data_ptr<std::uint8_t>() + epRank * workspace.stride(0) + offsets[RC11_FLAG_VAL_OFFSET_INDEX]);
    auto const params = buildControlParams(controlBase, controlPlane, epRank, epSize, a2aFlagVal);
    kernels::moe_comm::launchD0GenerationOpen(params, at::cuda::getCurrentCUDAStream());
    auto const result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "push D0 generation-open launch failed: ", cudaGetErrorString(result));
}

// =============================================================================
// Op: control-plane bootstrap (control-only). Clears THIS rank's control segment
// on-device (the caller pairs this with a collective barrier before any peer flag
// read). register_fake: returns nothing.
// =============================================================================
void moeA2APushControlBootstrapOp(torch::Tensor const& controlPlane, int64_t epRank, int64_t epSize)
{
    auto* controlBase = validateControlPlane(controlPlane, epSize);
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "ep_rank must be in [0, ep_size)");
    auto const params = buildControlParams(controlBase, controlPlane, epRank, epSize);
    kernels::moe_comm::launchPushControlBootstrap(params, at::cuda::getCurrentCUDAStream());
    auto const result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "push control bootstrap launch failed: ", cudaGetErrorString(result));
}

// =============================================================================
// Op: K23 fused producer-publish + consumer-gate (control-only). Replaces the
// former separate moe_a2a_push_k2_publish + moe_a2a_push_k3_gate ops: one launch
// runs K2's publish body then K3's gate body on lane0 of a single warp
// (moeAlltoAllTileNvfp4V2PushGate.cu, pushK23ControlKernel), removing one launch
// and the K2->K3 kernel boundary from the control plane. register_fake: nothing.
// =============================================================================
void moeA2APushK23ControlOp(torch::Tensor const& controlPlane, int64_t epRank, int64_t epSize, bool openGeneration,
    torch::optional<torch::Tensor> const& workspace, torch::optional<torch::Tensor> const& metainfo)
{
    auto* controlBase = validateControlPlane(controlPlane, epSize);
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "ep_rank must be in [0, ep_size)");
    // open_generation=True folds the D0 open into K23 (TRTLLM_PUSH_D0_FOLD): K23 then
    // also advances the NVLinkOneSided A2A round flag, which lives in the workspace.
    std::uint32_t* a2aFlagVal = nullptr;
    if (openGeneration)
    {
        TORCH_CHECK(workspace.has_value() && metainfo.has_value(),
            "moe_a2a_push_k23(open_generation=True) requires workspace and metainfo");
        validateWorkspace(*workspace, epRank, epSize);
        auto const* offsets = validateMetainfo(*metainfo);
        validateInt32Range(offsets[RC11_FLAG_VAL_OFFSET_INDEX], sizeof(std::uint32_t), workspace->size(1), "flag_val");
        a2aFlagVal = reinterpret_cast<std::uint32_t*>(
            workspace->data_ptr<std::uint8_t>() + epRank * workspace->stride(0) + offsets[RC11_FLAG_VAL_OFFSET_INDEX]);
    }
    auto const params = buildControlParams(controlBase, controlPlane, epRank, epSize, a2aFlagVal);
    kernels::moe_comm::launchK23Control(params, at::cuda::getCurrentCUDAStream(), openGeneration);
    auto const result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "push K23 control launch failed: ", cudaGetErrorString(result));
}

// =============================================================================
// Op: K1 producer push. Quantizes this rank's BF16 tokens into packed 4480-B
// records and pushes each into the consumer's region-C landing plane.
// register_fake: returns nothing (writes only into peer workspace region C).
// =============================================================================
void moeA2APushK1Op(torch::Tensor const& payload, int64_t localNumTokens, torch::Tensor const& workspace,
    torch::Tensor const& metainfo, int64_t runtimeMaxTokensPerRank, int64_t graphStaticMaxTokensPerRank, int64_t epRank,
    int64_t epSize, int64_t qPlaneOffset, bool payloadInWorkspace, int64_t recipe, bool cftIncumbentLive)
{
    // recipe-identity resolved BEFORE base allocation lookup (recipe-first ordering).
    auto const& desc = validateRecipe(recipe);
    auto const recordsPerToken = recordsPerTokenFor(epSize);
    TORCH_CHECK(localNumTokens >= 0 && localNumTokens <= runtimeMaxTokensPerRank && localNumTokens <= INT_MAX,
        "local_num_tokens must be in [0, runtime_max_tokens_per_rank] and fit int");

    validateWorkspace(workspace, epRank, epSize);
    auto const* offsets = validateMetainfo(metainfo);
    auto const rowBytes = workspace.size(1);
    auto const checked = validateAndCalculateLayout(
        desc, epSize, runtimeMaxTokensPerRank, graphStaticMaxTokensPerRank, qPlaneOffset, rowBytes);
    // Storage-alias proof for the full EP2/4/8/16/32 Push surface.
    validateRegionCAliasV1(desc, epSize, runtimeMaxTokensPerRank, graphStaticMaxTokensPerRank, recordsPerToken,
        kK1BaselineThreads, cftIncumbentLive, rowBytes, checked);

    auto const bases = resolveRankBases(workspace, offsets, epRank, qPlaneOffset, checked, rowBytes);
    validateRouteArrays(offsets, localNumTokens, rowBytes);
    auto const params = buildK1Params(payload, localNumTokens, workspace, bases, checked, runtimeMaxTokensPerRank,
        epRank, epSize, payloadInWorkspace, recordsPerToken);
    kernels::moe_comm::launchK1Push(params, at::cuda::getCurrentCUDAStream());
    auto const result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "push K1 launch failed: ", cudaGetErrorString(result));
}

// =============================================================================
// Op: K4 consumer top-k reduce. Gates on localGate[C], reads the LOCAL landed
// records, top-k reduces to BF16. Returns the BF16 [local_num_tokens, 7168] output.
// register_fake: torch.empty([local_num_tokens, 7168], bfloat16).
// =============================================================================
torch::Tensor moeA2APushK4TopkReduceOp(int64_t localNumTokens, torch::Tensor const& workspace,
    torch::Tensor const& controlPlane, torch::Tensor const& metainfo, int64_t runtimeMaxTokensPerRank,
    int64_t graphStaticMaxTokensPerRank, int64_t epRank, int64_t epSize, int64_t qPlaneOffset, int64_t recipe,
    int64_t selectedSharedStrideBytes, bool cftIncumbentLive)
{
    auto const& desc = validateRecipe(recipe);
    auto const recordsPerToken = recordsPerTokenFor(epSize);
    TORCH_CHECK(localNumTokens >= 0 && localNumTokens <= runtimeMaxTokensPerRank && localNumTokens <= INT_MAX,
        "local_num_tokens must be in [0, runtime_max_tokens_per_rank] and fit int");

    validateWorkspace(workspace, epRank, epSize);
    auto* controlBase = validateControlPlane(controlPlane, epSize);
    auto const* offsets = validateMetainfo(metainfo);
    auto const rowBytes = workspace.size(1);
    auto const checked = validateAndCalculateLayout(
        desc, epSize, runtimeMaxTokensPerRank, graphStaticMaxTokensPerRank, qPlaneOffset, rowBytes);
    validateRegionCAliasV1(desc, epSize, runtimeMaxTokensPerRank, graphStaticMaxTokensPerRank, recordsPerToken,
        kK1BaselineThreads, cftIncumbentLive, rowBytes, checked);

    auto const bases = resolveRankBases(workspace, offsets, epRank, qPlaneOffset, checked, rowBytes);
    validateRouteArrays(offsets, localNumTokens, rowBytes);

    auto output = torch::empty(
        {localNumTokens, static_cast<int64_t>(kHiddenElements)}, workspace.options().dtype(torch::kBFloat16));
    auto const params = buildK4Params(controlBase, controlPlane, output, localNumTokens, bases, runtimeMaxTokensPerRank,
        epRank, epSize, recordsPerToken, selectedSharedStrideBytes);
    kernels::moe_comm::launchK4TopkReduce(params, at::cuda::getCurrentCUDAStream());
    auto const result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "push K4 launch failed: ", cudaGetErrorString(result));
    return output;
}

// =============================================================================
// Op: sequential push combine. Launches K1 -> K23 -> K4 on the CURRENT stream
// under the common TensorRT-LLM PDL policy. Returns the BF16
// [local_num_tokens, 7168] output.
// register_fake: torch.empty([local_num_tokens, 7168], bfloat16).
// =============================================================================
torch::Tensor moeA2APushCombineOp(torch::Tensor const& payload, int64_t localNumTokens, torch::Tensor const& workspace,
    torch::Tensor const& controlPlane, torch::Tensor const& metainfo, int64_t runtimeMaxTokensPerRank,
    int64_t graphStaticMaxTokensPerRank, int64_t epRank, int64_t epSize, int64_t qPlaneOffset, bool payloadInWorkspace,
    int64_t recipe, int64_t selectedSharedStrideBytes, bool cftIncumbentLive, bool openGeneration)
{
    // recipe-identity resolved BEFORE base allocation lookup (recipe-first ordering).
    auto const& desc = validateRecipe(recipe);
    auto const recordsPerToken = recordsPerTokenFor(epSize);
    TORCH_CHECK(localNumTokens >= 0 && localNumTokens <= runtimeMaxTokensPerRank && localNumTokens <= INT_MAX,
        "local_num_tokens must be in [0, runtime_max_tokens_per_rank] and fit int");

    validateWorkspace(workspace, epRank, epSize);
    auto* controlBase = validateControlPlane(controlPlane, epSize);
    auto const* offsets = validateMetainfo(metainfo);
    auto const rowBytes = workspace.size(1);
    auto const checked = validateAndCalculateLayout(
        desc, epSize, runtimeMaxTokensPerRank, graphStaticMaxTokensPerRank, qPlaneOffset, rowBytes);
    // Full push path uses K4 -> EP must be in the published EP2/4/8/16/32 surface.
    validateRegionCAliasV1(desc, epSize, runtimeMaxTokensPerRank, graphStaticMaxTokensPerRank, recordsPerToken,
        kK1BaselineThreads, cftIncumbentLive, rowBytes, checked);

    auto const bases = resolveRankBases(workspace, offsets, epRank, qPlaneOffset, checked, rowBytes);
    validateRouteArrays(offsets, localNumTokens, rowBytes);

    auto output = torch::empty(
        {localNumTokens, static_cast<int64_t>(kHiddenElements)}, workspace.options().dtype(torch::kBFloat16));

    auto const k1 = buildK1Params(payload, localNumTokens, workspace, bases, checked, runtimeMaxTokensPerRank, epRank,
        epSize, payloadInWorkspace, recordsPerToken);
    auto const k4 = buildK4Params(controlBase, controlPlane, output, localNumTokens, bases, runtimeMaxTokensPerRank,
        epRank, epSize, recordsPerToken, selectedSharedStrideBytes);
    // The generation is a device word (D0 open advances it, K2/K3 read it on-device);
    // there is no host `generation` scalar so a captured graph re-reads the device
    // epoch on every replay.
    // open_generation=True (TRTLLM_PUSH_D0_FOLD): the dispatch skipped D0, so K23
    // opens the push generation and advances the A2A round flag instead.
    auto const control = buildControlParams(controlBase, controlPlane, epRank, epSize,
        openGeneration ? reinterpret_cast<std::uint32_t*>(bases.rankBase + offsets[RC11_FLAG_VAL_OFFSET_INDEX])
                       : nullptr);

    // Strict sequential K1 -> K23 -> K4 on the current stream. Ordering is the
    // stream itself: the fused K23 publishes only after K1 has fully drained
    // (kernel boundary), gates on the published readiness in the same lane, and K4
    // acquires localGate[C] after K23. K2/K3 remain separately launchable ops for
    // isolated profiling / fault injection; only the fused full-path op uses K23.
    auto const stream = at::cuda::getCurrentCUDAStream();
    kernels::moe_comm::launchK1Push(k1, stream);
    kernels::moe_comm::launchK23Control(control, stream, openGeneration);
    kernels::moe_comm::launchK4TopkReduce(k4, stream);
    auto const result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "push combine (K1->K23->K4) launch failed: ", cudaGetErrorString(result));
    return output;
}

} // namespace torch_ext::moe_comm

TRTLLM_NAMESPACE_END

// =============================================================================
// Torch registration. TORCH_LIBRARY_FRAGMENT lets this op set coexist with
// other fragments of the `trtllm` library without touching them.
// =============================================================================
TORCH_LIBRARY_FRAGMENT(trtllm, module)
{
    module.def("moe_a2a_push_control_bootstrap(Tensor(a!) control_plane, int ep_rank, int ep_size) -> ()");
    module.def(
        "moe_a2a_push_d0_generation_open(Tensor(a!) control_plane, Tensor(b!) workspace, Tensor metainfo, "
        "int ep_rank, int ep_size) -> ()");
    // K2/K3 read the generation (epoch) as a DEVICE word (compact_row_v2
    // push_generation slot); there is no host generation arg.
    // open_generation=True folds the D0 open (and the A2A round-flag advance) into K23.
    module.def(
        "moe_a2a_push_k23(Tensor(a!) control_plane, int ep_rank, int ep_size, bool open_generation=False, "
        "Tensor(b!)? workspace=None, Tensor? metainfo=None) -> ()");
    // payload and workspace share alias set `a`: when payload_in_workspace=true the
    // BF16 source IS the workspace Q plane (mirrors the PULL combine schema).
    module.def(
        "moe_a2a_push_k1(Tensor(a) payload, int local_num_tokens, Tensor(a!) workspace, "
        "Tensor metainfo, int runtime_max_tokens_per_rank, int graph_static_max_tokens_per_rank, "
        "int ep_rank, int ep_size, int q_plane_offset, bool payload_in_workspace, int recipe, "
        "bool cft_incumbent_live=False) -> ()");
    module.def(
        "moe_a2a_push_k4_topk_reduce(int local_num_tokens, Tensor(a!) workspace, "
        "Tensor(b!) control_plane, Tensor metainfo, int runtime_max_tokens_per_rank, "
        "int graph_static_max_tokens_per_rank, int ep_rank, int ep_size, int q_plane_offset, "
        "int recipe, int selected_shared_stride_bytes, bool cft_incumbent_live=False) -> Tensor");
    // payload/workspace share alias set `a` (payload_in_workspace); control_plane is
    // the distinct mutable raw-u64 flag plane (`b`).
    module.def(
        "moe_a2a_push_combine(Tensor(a) payload, int local_num_tokens, Tensor(a!) workspace, "
        "Tensor(b!) control_plane, Tensor metainfo, int runtime_max_tokens_per_rank, "
        "int graph_static_max_tokens_per_rank, int ep_rank, int ep_size, int q_plane_offset, "
        "bool payload_in_workspace, int recipe, int selected_shared_stride_bytes, "
        "bool cft_incumbent_live=False, bool open_generation=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, module)
{
    module.impl("moe_a2a_push_control_bootstrap", &tensorrt_llm::torch_ext::moe_comm::moeA2APushControlBootstrapOp);
    module.impl("moe_a2a_push_d0_generation_open", &tensorrt_llm::torch_ext::moe_comm::moeA2APushD0GenerationOpenOp);
    module.impl("moe_a2a_push_k23", &tensorrt_llm::torch_ext::moe_comm::moeA2APushK23ControlOp);
    module.impl("moe_a2a_push_k1", &tensorrt_llm::torch_ext::moe_comm::moeA2APushK1Op);
    module.impl("moe_a2a_push_k4_topk_reduce", &tensorrt_llm::torch_ext::moe_comm::moeA2APushK4TopkReduceOp);
    module.impl("moe_a2a_push_combine", &tensorrt_llm::torch_ext::moe_comm::moeA2APushCombineOp);
}
