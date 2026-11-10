/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllSentinel.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"
#include "tensorrt_llm/thop/moe/communication/moeAlltoAllMeta.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <atomic>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <torch/extension.h>
#include <torch/types.h>
#include <unordered_map>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace moe_comm
{

// Process-wide completion-flag budget copied into subsequent dispatch/combine launch arguments,
// so a captured CUDA graph keeps the budget that was current at capture.
//
// No collective separates a rank's first-touch JIT/autotune work from its dispatch launch, so
// this device-side budget is in effect a deadline on the slowest peer's host-side progress.
// Python raises it during warmup (tensorrt_llm/_torch/moe/fused_moe/moe_comm_timeout_guard.py).
static std::atomic<int64_t> gTimeoutCycles{tensorrt_llm::kernels::moe_comm::kDefaultTimeoutCycles};

// Set the budget of subsequent launches in seconds; 0 restores the default.
void moeA2ASetTimeoutOp(int64_t timeoutSec)
{
    constexpr int64_t kMaxTimeoutSec = 24 * 60 * 60;
    TORCH_CHECK(timeoutSec >= 0 && timeoutSec <= kMaxTimeoutSec, "MoE all-to-all timeout must be in 0..",
        kMaxTimeoutSec, " seconds (0 restores the default), got ", timeoutSec);
    int64_t const timeoutCycles = timeoutSec == 0 ? tensorrt_llm::kernels::moe_comm::kDefaultTimeoutCycles
                                                  : timeoutSec * tensorrt_llm::kernels::moe_comm::kAssumedClockHz;
    gTimeoutCycles.store(timeoutCycles, std::memory_order_relaxed);
}

// Return the budget of subsequent launches in seconds.
int64_t moeA2AGetTimeoutOp()
{
    return gTimeoutCycles.load(std::memory_order_relaxed) / tensorrt_llm::kernels::moe_comm::kAssumedClockHz;
}

static constexpr size_t CACHELINE_ALIGNMENT = 128;

// ---------------------------------------------------------------------------
// Communication shims.
//
// The MoE all-to-all workspace setup used tensorrt_llm::mpi::MpiComm directly.
// mpiUtils.h throws "MPI is disabled, DON'T USE MPI" whenever TLLM_DISABLE_MPI=1,
// which the Ray executor sets for every worker (executor/ray_executor.py), so the
// NVLinkOneSided strategy always failed there and silently degraded to
// NVLinkTwoSided. Route through the Torch process group in that case; the same
// pattern is already used by cacheTransceiver / allreduceOp via pg_utils.
// ---------------------------------------------------------------------------

inline void moeA2ABarrier()
{
    if (useMPI())
    {
        tensorrt_llm::mpi::MpiComm::session().barrier();
        return;
    }
    auto const pg = tensorrt_llm::pg_utils::get_world_pg();
    TORCH_CHECK(pg,
        "MoE all-to-all needs either MPI or an initialised Torch process group; neither is available. "
        "Initialise the Torch process group before building the MoE workspace, or unset TLLM_DISABLE_MPI.");
    PGCHECK_THROW(pg->barrier());
}

// TODO: Is Alignment necessary?
// Helper function to align offset to specified byte boundary
inline size_t alignOffset(size_t offset, size_t alignment)
{
    return (offset + alignment - 1) & ~(alignment - 1);
}

MoeA2AWorkspaceLayout const& readWorkspaceLayout(torch::Tensor const& metainfo)
{
    CHECK_CPU(metainfo);
    CHECK_TYPE(metainfo, torch::kInt64);
    CHECK_CONTIGUOUS(metainfo);
    TORCH_CHECK(metainfo.dim() == 1 && metainfo.numel() == NUM_METAINFO_FIELDS,
        "metainfo must contain the complete MoE A2A workspace layout");
    return *reinterpret_cast<MoeA2AWorkspaceLayout const*>(metainfo.data_ptr<int64_t>());
}

inline bool hasActiveRankMask(torch::optional<torch::Tensor> const& maskTensor)
{
    return maskTensor.has_value() && maskTensor.value().defined();
}

// Resolve a provided rank-mask tensor into a fixed-width uint64 array. On failure
// (wrong dtype / device / shape), throw at the Python op boundary rather than launch.
inline void resolveActiveRankMask(torch::optional<torch::Tensor> const& maskTensor, int64_t epRank,
    uint64_t (&out)[tensorrt_llm::kernels::moe_comm::kRankMaskWords])
{
    using tensorrt_llm::kernels::moe_comm::kRankMaskWords;
    using tensorrt_llm::kernels::moe_comm::kMaxRanks;
    TORCH_CHECK(
        epRank >= 0 && epRank < kMaxRanks, "epRank must be in the range [0, ", kMaxRanks, ") for active_rank_mask");
    TORCH_CHECK(hasActiveRankMask(maskTensor), "active_rank_mask must be defined");
    torch::Tensor const& t = maskTensor.value();
    TORCH_CHECK(t.is_cpu(), "active_rank_mask must be a CPU tensor");
    TORCH_CHECK(t.scalar_type() == torch::kUInt64, "active_rank_mask must have dtype uint64");
    TORCH_CHECK(t.dim() == 1, "active_rank_mask must be a 1D tensor");
    TORCH_CHECK(t.numel() == kRankMaskWords, "active_rank_mask must have exactly ", kRankMaskWords, " uint64 elements");
    TORCH_CHECK(t.is_contiguous(), "active_rank_mask must be contiguous");
    auto const* src = static_cast<uint64_t const*>(t.const_data_ptr());
    for (int w = 0; w < kRankMaskWords; ++w)
    {
        out[w] = src[w];
    }
    // Local rank's bit must be set; otherwise the kernel would be running on a "dead" rank.
    TORCH_CHECK((out[epRank >> 6] >> (epRank & 63)) & 1ULL, "active_rank_mask must mark the local ep_rank (", epRank,
        ") as active");
}

// All offsets and capacities are per rank. Payload capacities are padded so
// the following phase's control buffers remain aligned.
MoeA2AWorkspaceLayout calculateWorkspaceLayout(int64_t epSize, int64_t maxNumTokens, int64_t topK,
    int64_t dispatchBytes, int64_t combineInputBytes, int64_t eplbStatsNumExperts)
{
    using tensorrt_llm::kernels::moe_comm::kMaxRanks;
    using tensorrt_llm::kernels::moe_comm::kMaxTopK;
    TORCH_CHECK(epSize > 0 && epSize <= kMaxRanks, "Invalid EP size: ", epSize);
    TORCH_CHECK(maxNumTokens > 0 && maxNumTokens <= std::numeric_limits<int32_t>::max(),
        "Invalid allocation-time token capacity: ", maxNumTokens);
    TORCH_CHECK(topK > 0 && topK <= kMaxTopK, "Invalid top_k: ", topK);
    TORCH_CHECK(eplbStatsNumExperts >= 0 && eplbStatsNumExperts <= std::numeric_limits<int32_t>::max(),
        "Invalid EPLB expert capacity: ", eplbStatsNumExperts);

    auto alignedBytes = [](int64_t bytes)
    {
        TORCH_CHECK(bytes >= 0 && bytes <= std::numeric_limits<int64_t>::max() - kWorkspaceAlignment,
            "Invalid workspace payload capacity: ", bytes);
        return static_cast<int64_t>(alignOffset(bytes, kWorkspaceAlignment));
    };
    dispatchBytes = alignedBytes(dispatchBytes);
    combineInputBytes = alignedBytes(combineInputBytes);

    MoeA2AWorkspaceLayout layout{};
    int64_t offset = 0;
    auto reserve = [&](MoeA2AMetaInfoIndex index, int64_t bytes, int64_t alignment = sizeof(int32_t))
    {
        TORCH_CHECK(
            offset <= std::numeric_limits<int64_t>::max() - alignment, "Workspace layout alignment overflows int64");
        offset = static_cast<int64_t>(alignOffset(offset, alignment));
        TORCH_CHECK(bytes >= 0 && bytes <= std::numeric_limits<int64_t>::max() - offset,
            "Workspace layout size overflows int64");
        layout[index] = offset;
        offset += bytes;
    };
    int64_t const rankCountsBytes = epSize * sizeof(int32_t);

    reserve(FLAG_VAL_OFFSET_INDEX, sizeof(uint32_t));

    // Dispatch writes these counters and routes; combine reuses the completed routes.
    reserve(LOCAL_TOKEN_COUNTER_OFFSET_INDEX, sizeof(int32_t));
    reserve(SEND_COUNTERS_OFFSET_INDEX, rankCountsBytes);
    reserve(RECV_COUNTERS_OFFSET_INDEX, 2 * rankCountsBytes);
    reserve(DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX, rankCountsBytes, CACHELINE_ALIGNMENT);
    reserve(TOPK_TARGET_RANKS_OFFSET_INDEX, maxNumTokens * topK * sizeof(int32_t), CACHELINE_ALIGNMENT);
    reserve(TOPK_TARGET_INDICES_OFFSET_INDEX, maxNumTokens * topK * sizeof(int32_t), CACHELINE_ALIGNMENT);
    reserve(EPLB_GATHERED_STATS_OFFSET_INDEX, epSize * eplbStatsNumExperts * sizeof(int32_t), CACHELINE_ALIGNMENT);
    reserve(DISPATCH_PAYLOAD_OFFSET_INDEX, dispatchBytes, kWorkspaceAlignment);
    layout[DISPATCH_PAYLOAD_SIZE_INDEX] = dispatchBytes;

    // Fence combine pulls from each peer's combine input.
    reserve(COMBINE_COMPLETION_FLAGS_OFFSET_INDEX, rankCountsBytes, CACHELINE_ALIGNMENT);
    reserve(COMBINE_INPUT_OFFSET_INDEX, combineInputBytes, kWorkspaceAlignment);
    layout[COMBINE_INPUT_SIZE_INDEX] = combineInputBytes;

    layout[MAX_NUM_TOKENS_INDEX] = maxNumTokens;
    layout[TOP_K_INDEX] = topK;
    layout[EP_SIZE_INDEX] = epSize;
    layout[EPLB_STATS_NUM_EXPERTS_INDEX] = eplbStatsNumExperts;
    layout[WORKSPACE_SIZE_INDEX] = offset;
    return layout;
}

torch::Tensor moeA2AGetWorkspaceLayoutOp(int64_t epSize, int64_t maxNumTokens, int64_t topK, int64_t dispatchBytes,
    int64_t combineInputBytes, torch::optional<int64_t> eplbStatsNumExperts)
{
    auto const layout = calculateWorkspaceLayout(
        epSize, maxNumTokens, topK, dispatchBytes, combineInputBytes, eplbStatsNumExperts.value_or(0));
    auto metainfo
        = torch::empty({NUM_METAINFO_FIELDS}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    std::copy(layout.begin(), layout.end(), metainfo.data_ptr<int64_t>());
    return metainfo;
}

// ============================================================================
// Sentinel transport: workspace regions
// ============================================================================

namespace sentinel = tensorrt_llm::kernels::moe_comm::sentinel;
// The sentinel kernels take every top_k the one-sided ops do, so the op's top_k check covers them.
static_assert(
    sentinel::kMaxTopK == tensorrt_llm::kernels::moe_comm::kMaxTopK, "sentinel and fence top_k bounds differ");

// Bytes of one row of a sentinel payload format: the payload (landing and receive rows), its
// scales as received, and its scales on the wire (MXFP8 scale bytes travel as 16-bit lanes).
struct SentinelRowBytes
{
    int64_t payload;
    int64_t scale;
    int64_t wireScale;
};

SentinelRowBytes sentinelRowBytes(int64_t payloadFormat, int64_t hiddenSize)
{
    // Row bytes are kernel ints: hidden sizes beyond this bound are rejected, not wrapped.
    TORCH_CHECK(hiddenSize > 0 && hiddenSize <= (1 << 24), "Invalid sentinel hidden size ", hiddenSize);
    switch (payloadFormat)
    {
    case sentinel::kBf16:
        TORCH_CHECK(hiddenSize % 8 == 0, "sentinel BF16 rows need hidden_size % 8 == 0, got ", hiddenSize);
        return {2 * hiddenSize, 0, 0};
    case sentinel::kMxfp8:
        TORCH_CHECK(hiddenSize % 256 == 0, "sentinel MXFP8 rows need hidden_size % 256 == 0, got ", hiddenSize);
        return {hiddenSize, hiddenSize / 32, hiddenSize / 16};
    case sentinel::kFp8Block128:
        TORCH_CHECK(hiddenSize % 512 == 0, "sentinel FP8 block-128 rows need hidden_size % 512 == 0, got ", hiddenSize);
        return {hiddenSize, hiddenSize / 32, hiddenSize / 32};
    case sentinel::kNvfp4:
        TORCH_CHECK(hiddenSize % 256 == 0, "sentinel NVFP4 rows need hidden_size % 256 == 0, got ", hiddenSize);
        return {hiddenSize / 2, hiddenSize / 16, hiddenSize / 16};
    default: TORCH_CHECK(false, "Unknown sentinel payload format ", payloadFormat);
    }
    return {};
}

// Dispatch blocks per token slot: one per kNumThreads 16-byte vectors of the wire row.
int64_t sentinelDispatchBlocksPerToken(SentinelRowBytes const& row)
{
    int64_t const vectors = (row.payload + row.wireScale) / 16;
    return (vectors + sentinel::kNumThreads - 1) / sentinel::kNumThreads;
}

// Round state at SENTINEL_STATE_OFFSET_INDEX: int32 words (RoundStateWord) on their own cache line,
// then the uint64 receive-source mask of every token slot.
constexpr int64_t kSentinelSourceMaskOffset = CACHELINE_ALIGNMENT;

// Sentinel workspace bytes of one (source rank, token) row in each region: the landing rows (one set
// of payload, scale and weight rows, two sets of expert-id rows), the combine landing row and the
// receive rows (payload, scales, expert ids and weights as received). The region sizes and the
// default token limit are both built from it.
struct SentinelBytesPerRow
{
    int64_t payload, scale, weights, idsSet; // landing
    int64_t combine;
    int64_t recv;

    int64_t total() const
    {
        return payload + scale + weights + 2 * idsSet + combine + recv;
    }
};

SentinelBytesPerRow sentinelBytesPerRow(int64_t topK, SentinelRowBytes const& row, int64_t hiddenSize)
{
    int64_t const topkRowBytes = 16 * sentinel::topk_row_vectors(topK);
    SentinelBytesPerRow bytes{};
    bytes.payload = row.payload;
    bytes.scale = row.wireScale;
    bytes.weights = topkRowBytes;
    bytes.idsSet = topkRowBytes;
    bytes.combine = 2 * hiddenSize; // bf16
    bytes.recv = row.payload + row.scale + 2 * topK * static_cast<int64_t>(sizeof(int32_t));
    return bytes;
}

// Bytes of the sentinel regions of one rank for maxNumTokens tokens per source rank: the round
// state, then the per-row bytes times the rows. Each region and each landing row group starts on a
// kWorkspaceAlignment boundary; the receive rows each start on their own cache line.
struct SentinelRegionBytes
{
    int64_t state;
    int64_t payload, scale, weights, idsSet; // landing row groups
    int64_t landing;                         // all of them
    int64_t combine;
    int64_t recv;
};

SentinelRegionBytes sentinelRegionBytes(
    int64_t epSize, int64_t maxNumTokens, int64_t topK, SentinelRowBytes const& row, int64_t hiddenSize)
{
    auto aligned = [](int64_t bytes) { return static_cast<int64_t>(alignOffset(bytes, kWorkspaceAlignment)); };
    int64_t const rows = epSize * maxNumTokens; // (source rank, token)
    auto const perRow = sentinelBytesPerRow(topK, row, hiddenSize);
    SentinelRegionBytes bytes{};
    bytes.state = kSentinelSourceMaskOffset + maxNumTokens * static_cast<int64_t>(sizeof(uint64_t));
    bytes.payload = aligned(rows * perRow.payload);
    bytes.scale = aligned(rows * perRow.scale);
    bytes.weights = aligned(rows * perRow.weights);
    bytes.idsSet = aligned(rows * perRow.idsSet);
    bytes.landing = bytes.payload + bytes.scale + bytes.weights + 2 * bytes.idsSet;
    bytes.combine = aligned(rows * perRow.combine);
    bytes.recv = rows * perRow.recv
        + tensorrt_llm::kernels::moe_comm::kMaxPayloads * static_cast<int64_t>(CACHELINE_ALIGNMENT);
    return bytes;
}

// Append the sentinel regions for steps of at most maxNumTokens tokens per rank to `layout`, after
// every other region (whose offsets do not change).
void appendSentinelRegions(
    MoeA2AWorkspaceLayout& layout, int64_t maxNumTokens, int64_t payloadFormat, int64_t hiddenSize)
{
    int64_t const epSize = layout[EP_SIZE_INDEX];
    int64_t const topK = layout[TOP_K_INDEX];
    TORCH_CHECK(layout[SENTINEL_MAX_NUM_TOKENS_INDEX] == 0, "The workspace layout already has sentinel regions");
    TORCH_CHECK(
        epSize <= sentinel::kMaxRanks, "The sentinel transport supports at most ", sentinel::kMaxRanks, " EP ranks");
    TORCH_CHECK(
        maxNumTokens > 0 && maxNumTokens <= sentinel::kMaxTokensPerRank && maxNumTokens <= layout[MAX_NUM_TOKENS_INDEX],
        "Sentinel token capacity must be in [1, min(", sentinel::kMaxTokensPerRank, ", max_num_tokens)], got ",
        maxNumTokens);
    auto const bytes
        = sentinelRegionBytes(epSize, maxNumTokens, topK, sentinelRowBytes(payloadFormat, hiddenSize), hiddenSize);

    int64_t offset = layout[WORKSPACE_SIZE_INDEX];
    auto reserve = [&](MoeA2AMetaInfoIndex index, int64_t size)
    {
        TORCH_CHECK(offset >= 0 && offset <= std::numeric_limits<int64_t>::max() - kWorkspaceAlignment,
            "Workspace layout alignment overflows int64");
        offset = static_cast<int64_t>(alignOffset(offset, kWorkspaceAlignment));
        TORCH_CHECK(
            size >= 0 && size <= std::numeric_limits<int64_t>::max() - offset, "Workspace layout size overflows int64");
        layout[index] = offset;
        offset += size;
    };
    reserve(SENTINEL_STATE_OFFSET_INDEX, bytes.state);
    reserve(SENTINEL_LANDING_OFFSET_INDEX, bytes.landing);
    reserve(SENTINEL_COMBINE_LANDING_OFFSET_INDEX, bytes.combine);
    reserve(SENTINEL_RECV_OFFSET_INDEX, bytes.recv);
    layout[SENTINEL_RECV_SIZE_INDEX] = bytes.recv;

    layout[SENTINEL_MAX_NUM_TOKENS_INDEX] = maxNumTokens;
    layout[SENTINEL_PAYLOAD_FORMAT_INDEX] = payloadFormat;
    layout[SENTINEL_HIDDEN_SIZE_INDEX] = hiddenSize;
    layout[WORKSPACE_SIZE_INDEX] = offset;
}

// Landing regions of a layout with sentinel regions, as the kernels address them.
sentinel::LandingLayout sentinelLandingLayout(MoeA2AWorkspaceLayout const& layout)
{
    int64_t const hiddenSize = layout[SENTINEL_HIDDEN_SIZE_INDEX];
    auto const bytes = sentinelRegionBytes(layout[EP_SIZE_INDEX], layout[SENTINEL_MAX_NUM_TOKENS_INDEX],
        layout[TOP_K_INDEX], sentinelRowBytes(layout[SENTINEL_PAYLOAD_FORMAT_INDEX], hiddenSize), hiddenSize);
    sentinel::LandingLayout landing{};
    landing.payload_offset = layout[SENTINEL_LANDING_OFFSET_INDEX];
    landing.scale_offset = landing.payload_offset + bytes.payload;
    landing.weights_offset = landing.scale_offset + bytes.scale;
    landing.ids_offset = landing.weights_offset + bytes.weights;
    landing.ids_set_bytes = bytes.idsSet;
    landing.combine_offset = layout[SENTINEL_COMBINE_LANDING_OFFSET_INDEX];
    return landing;
}

// Arm this rank's landing rows: expert-id rows with kIdsSentinel bytes, the others with
// kDataSentinel bytes. The round state stays zero.
void armSentinelRegions(uint8_t* rankWorkspace, MoeA2AWorkspaceLayout const& layout, cudaStream_t stream)
{
    auto const landing = sentinelLandingLayout(layout);
    // cudaMemsetAsync fills bytes, so each sentinel word must be one byte repeated.
    constexpr auto repeatedByte = [](uint32_t word) { return (word & 0xFFu) * 0x01010101u == word; };
    static_assert(repeatedByte(sentinel::kDataSentinel) && repeatedByte(sentinel::kIdsSentinel),
        "sentinel words must be a repeated byte");
    constexpr int kDataByte = static_cast<int>(sentinel::kDataSentinel & 0xFFu);
    constexpr int kIdsByte = static_cast<int>(sentinel::kIdsSentinel & 0xFFu);
    auto fillRange = [&](int64_t begin, int value, int64_t end)
    {
        TORCH_CHECK(cudaMemsetAsync(rankWorkspace + begin, value, end - begin, stream) == cudaSuccess,
            "cudaMemsetAsync failed");
    };
    fillRange(landing.payload_offset, kDataByte, landing.ids_offset); // payload, scale and weight rows
    fillRange(landing.ids_offset, kIdsByte, landing.ids_offset + 2 * landing.ids_set_bytes);
    fillRange(landing.combine_offset, kDataByte, layout[SENTINEL_RECV_OFFSET_INDEX]);
}

// Workspace slices with a sentinel dispatch not combined yet, with the dispatch's step geometry. A
// sentinel combine reads routing state only its dispatch writes and re-arms the rows that dispatch
// used, so it must run on the same geometry, and a sentinel round must end before the next dispatch.
// The ops run on the host in stream order, also while a CUDA graph is captured.
struct SentinelRound
{
    int64_t runtimeMaxTokensPerRank;
    int64_t localNumTokens;
    bool enableRankMask;

    bool operator==(SentinelRound const& other) const
    {
        return runtimeMaxTokensPerRank == other.runtimeMaxTokensPerRank && localNumTokens == other.localNumTokens
            && enableRankMask == other.enableRankMask;
    }
};

std::mutex gOpenSentinelRoundsMutex;
std::unordered_map<uint8_t const*, SentinelRound> gOpenSentinelRounds;

void openSentinelRound(uint8_t const* rankWorkspace, SentinelRound const& round)
{
    std::lock_guard<std::mutex> lock(gOpenSentinelRoundsMutex);
    gOpenSentinelRounds[rankWorkspace] = round;
}

void closeSentinelRound(uint8_t const* rankWorkspace)
{
    std::lock_guard<std::mutex> lock(gOpenSentinelRoundsMutex);
    gOpenSentinelRounds.erase(rankWorkspace);
}

std::optional<SentinelRound> openSentinelRoundOf(uint8_t const* rankWorkspace)
{
    std::lock_guard<std::mutex> lock(gOpenSentinelRoundsMutex);
    auto const it = gOpenSentinelRounds.find(rankWorkspace);
    return it == gOpenSentinelRounds.end() ? std::nullopt : std::optional<SentinelRound>(it->second);
}

// Append the sentinel regions to metainfo, a layout from moe_a2a_get_workspace_layout: landing rows
// for steps of at most sentinelMaxTokens tokens per rank in payloadFormat (sentinel::PayloadFormat) at
// hiddenSize.
void moeA2ASentinelExtendWorkspaceLayoutOp(
    torch::Tensor& metainfo, int64_t sentinelMaxTokens, int64_t payloadFormat, int64_t hiddenSize)
{
    auto layout = readWorkspaceLayout(metainfo);
    auto const base = calculateWorkspaceLayout(layout[EP_SIZE_INDEX], layout[MAX_NUM_TOKENS_INDEX], layout[TOP_K_INDEX],
        layout[DISPATCH_PAYLOAD_SIZE_INDEX], layout[COMBINE_INPUT_SIZE_INDEX], layout[EPLB_STATS_NUM_EXPERTS_INDEX]);
    TORCH_CHECK(layout == base, "metainfo must be a layout from moe_a2a_get_workspace_layout");
    appendSentinelRegions(layout, sentinelMaxTokens, payloadFormat, hiddenSize);
    std::copy(layout.begin(), layout.end(), metainfo.data_ptr<int64_t>());
}

// Row geometry of a sentinel payload format at hiddenSize, for the per-step choice and the default
// token limit: (hidden-state row bytes, scale row bytes, dispatch blocks per token slot, workspace
// bytes per (source rank, token) row). Raises for an unsupported hidden size.
std::vector<int64_t> moeA2ASentinelRowGeometryOp(int64_t payloadFormat, int64_t hiddenSize, int64_t topK)
{
    auto const row = sentinelRowBytes(payloadFormat, hiddenSize);
    return {row.payload, row.scale, sentinelDispatchBlocksPerToken(row),
        sentinelBytesPerRow(topK, row, hiddenSize).total()};
}

// Whether the loaded code carries the sentinel kernels for the current device.
bool moeA2ASentinelAvailableOp()
{
    return sentinel::moe_a2a_sentinel_kernels_available();
}

// Initialize control state after allocation; layout construction itself is CPU-only.
void moeA2AInitializeOp(torch::Tensor const& workspace, torch::Tensor const& metainfo, int64_t epRank, int64_t epSize)
{
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) == epSize && workspace.stride(1) == 1,
        "workspace must have shape [ep_size, bytes_per_rank] with contiguous rank slices");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "Invalid EP rank");
    auto const& layout = readWorkspaceLayout(metainfo);
    TORCH_CHECK(layout[EP_SIZE_INDEX] == epSize, "Workspace layout EP size mismatch");
    auto expected = calculateWorkspaceLayout(epSize, layout[MAX_NUM_TOKENS_INDEX], layout[TOP_K_INDEX],
        layout[DISPATCH_PAYLOAD_SIZE_INDEX], layout[COMBINE_INPUT_SIZE_INDEX], layout[EPLB_STATS_NUM_EXPERTS_INDEX]);
    bool const hasSentinel = layout[SENTINEL_MAX_NUM_TOKENS_INDEX] > 0;
    if (hasSentinel)
    {
        appendSentinelRegions(expected, layout[SENTINEL_MAX_NUM_TOKENS_INDEX], layout[SENTINEL_PAYLOAD_FORMAT_INDEX],
            layout[SENTINEL_HIDDEN_SIZE_INDEX]);
    }
    TORCH_CHECK(layout == expected, "Workspace layout metadata is inconsistent");
    TORCH_CHECK(layout[DISPATCH_PAYLOAD_SIZE_INDEX] > 0 && layout[COMBINE_INPUT_SIZE_INDEX] > 0,
        "Dispatch and combine input payload capacities must be positive");
    TORCH_CHECK(workspace.size(1) >= layout[WORKSPACE_SIZE_INDEX], "Workspace needs ", layout[WORKSPACE_SIZE_INDEX],
        " bytes per rank, got ", workspace.size(1));

    workspace[epRank].narrow(0, 0, layout[WORKSPACE_SIZE_INDEX]).zero_();
    uint8_t* rankWorkspace = workspace.data_ptr<uint8_t>() + epRank * workspace.stride(0);
    cudaMemsetAsync(rankWorkspace + layout[RECV_COUNTERS_OFFSET_INDEX], 0xFF,
        2 * static_cast<size_t>(epSize) * sizeof(int32_t), at::cuda::getCurrentCUDAStream());
    if (hasSentinel)
    {
        armSentinelRegions(rankWorkspace, layout, at::cuda::getCurrentCUDAStream());
    }
    closeSentinelRound(rankWorkspace);
    // Synchronize among ranks. Under a non-MPI orchestrator (Ray) MpiComm throws
    // "MPI is disabled, DON'T USE MPI" from mpiUtils.h, which made the whole
    // NVLinkOneSided strategy unusable there; fall back to the Torch process
    // group the Ray workers already initialise, as pg_utils does elsewhere.
    cudaDeviceSynchronize();
    moeA2ABarrier();
}

// ============================================================================
// Sentinel transport: dispatch and combine
// ============================================================================

// Fills every rank's workspace slice and the round state of a sentinel step's kernels; returns
// this rank's slice.
template <typename Params>
uint8_t* setSentinelWorkspace(
    Params& params, torch::Tensor const& workspace, MoeA2AWorkspaceLayout const& layout, int64_t epRank, int64_t epSize)
{
    TORCH_CHECK(layout[SENTINEL_MAX_NUM_TOKENS_INDEX] > 0, "Workspace was allocated without sentinel regions");
    uint8_t* workspacePtr = workspace.data_ptr<uint8_t>();
    uint8_t* rankWorkspace = workspacePtr + epRank * workspace.stride(0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(rankWorkspace) % CACHELINE_ALIGNMENT == 0, "rankWorkSpacePtr must be ",
        CACHELINE_ALIGNMENT, "-byte aligned");
    for (int rank = 0; rank < epSize; ++rank)
    {
        params.rank_workspaces[rank] = workspacePtr + rank * workspace.stride(0);
    }
    params.landing = sentinelLandingLayout(layout);
    int64_t const state = layout[SENTINEL_STATE_OFFSET_INDEX];
    params.round_state = reinterpret_cast<int32_t*>(rankWorkspace + state);
    params.recv_source_mask = reinterpret_cast<uint64_t*>(rankWorkspace + state + kSentinelSourceMaskOffset);
    // The fence path's routing table: a sentinel round fills and reads it like a fence round.
    params.topk_target_ranks = reinterpret_cast<int32_t*>(rankWorkspace + layout[TOPK_TARGET_RANKS_OFFSET_INDEX]);
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    return rankWorkspace;
}

// A sentinel dispatch. inputPayloads = hidden states, [scales,] expert ids, [weights] (the expert
// ids at expertIdPayloadIndex); recvOffsets = their receive buffers in the sentinel receive region.
void sentinelDispatch(torch::Tensor const& tokenSelectedExperts, std::vector<torch::Tensor> const& inputPayloads,
    int64_t expertIdPayloadIndex, torch::Tensor const& workspace, MoeA2AWorkspaceLayout const& layout,
    std::vector<size_t> const& recvOffsets, int64_t runtimeMaxTokensPerRank, int64_t epRank, int64_t epSize,
    int64_t topK, int64_t numExperts, int64_t invalidExpertId)
{
    sentinel::DispatchParams params{};
    uint8_t* rankWorkspace = setSentinelWorkspace(params, workspace, layout, epRank, epSize);
    TORCH_CHECK(runtimeMaxTokensPerRank <= layout[SENTINEL_MAX_NUM_TOKENS_INDEX], "A sentinel step takes at most ",
        layout[SENTINEL_MAX_NUM_TOKENS_INDEX], " tokens per rank, got ", runtimeMaxTokensPerRank);
    int64_t const payloadFormat = layout[SENTINEL_PAYLOAD_FORMAT_INDEX];
    auto const row = sentinelRowBytes(payloadFormat, layout[SENTINEL_HIDDEN_SIZE_INDEX]);
    int64_t const numPayloads = static_cast<int64_t>(inputPayloads.size());
    bool const hasScale = payloadFormat != sentinel::kBf16;
    TORCH_CHECK(expertIdPayloadIndex == (hasScale ? 2 : 1) && numPayloads >= expertIdPayloadIndex + 1
            && numPayloads <= expertIdPayloadIndex + 2,
        "A sentinel dispatch carries hidden states, ", (hasScale ? "their scales, " : ""),
        "expert ids and optional weights, in this order");
    auto rowBytes = [](torch::Tensor const& t) { return t.size(1) * t.element_size(); };
    TORCH_CHECK(rowBytes(inputPayloads[0]) == row.payload,
        "Hidden states do not match the workspace's sentinel payload format");
    if (hasScale)
    {
        TORCH_CHECK(rowBytes(inputPayloads[1]) == row.scale,
            "Hidden-state scales do not match the workspace's sentinel payload format");
    }
    torch::Tensor const& expertIds = inputPayloads[expertIdPayloadIndex];
    TORCH_CHECK(expertIds.data_ptr() == tokenSelectedExperts.data_ptr(),
        "The expert-id payload must be token_selected_experts");
    bool const hasWeights = numPayloads == expertIdPayloadIndex + 2;
    bool weightsBf16 = false;
    if (hasWeights)
    {
        torch::Tensor const& weights = inputPayloads[expertIdPayloadIndex + 1];
        weightsBf16 = weights.scalar_type() == at::kBFloat16;
        TORCH_CHECK((weights.element_size() == 4 || weightsBf16) && weights.size(1) == topK,
            "Sentinel routing weights must be 4-byte or bf16 [local_num_tokens, top_k]");
    }
    TORCH_CHECK(numExperts % epSize == 0, "The sentinel transport needs num_experts divisible by ep_size");
    // Receivers tell a row without a token by its invalid id, so it must name no expert.
    TORCH_CHECK(invalidExpertId < 0 || invalidExpertId >= numExperts,
        "A sentinel dispatch needs invalid_expert_id outside [0, num_experts), got ", invalidExpertId);
    TORCH_CHECK(static_cast<uint32_t>(invalidExpertId) != sentinel::kIdsSentinel,
        "invalid_expert_id must differ from the sentinel expert-id sentinel");

    params.hidden_states = static_cast<uint8_t const*>(inputPayloads[0].data_ptr());
    params.hidden_states_scale = hasScale ? static_cast<uint8_t const*>(inputPayloads[1].data_ptr()) : nullptr;
    params.token_selected_experts = tokenSelectedExperts.data_ptr<int32_t>();
    params.topk_weights
        = hasWeights ? static_cast<uint32_t const*>(inputPayloads[expertIdPayloadIndex + 1].data_ptr()) : nullptr;
    params.recv_hidden_states = rankWorkspace + recvOffsets[0];
    params.recv_hidden_states_scale = hasScale ? rankWorkspace + recvOffsets[1] : nullptr;
    params.recv_topk_ids = reinterpret_cast<int32_t*>(rankWorkspace + recvOffsets[expertIdPayloadIndex]);
    params.recv_topk_weights
        = hasWeights ? reinterpret_cast<uint32_t*>(rankWorkspace + recvOffsets[expertIdPayloadIndex + 1]) : nullptr;
    params.num_experts = static_cast<int>(numExperts);
    params.experts_per_rank = static_cast<int>(numExperts / epSize);
    params.top_k = static_cast<int>(topK);
    params.local_num_tokens = static_cast<int>(tokenSelectedExperts.size(0));
    params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
    params.has_topk_weights = hasWeights;
    params.weights_bf16 = weightsBf16;
    params.mx_scales = payloadFormat == sentinel::kMxfp8;
    params.payload_row_bytes = static_cast<int>(row.payload);
    params.scale_row_bytes = static_cast<int>(row.scale);
    params.wire_scale_row_bytes = static_cast<int>(row.wireScale);
    params.payload_vectors = static_cast<int>(row.payload / 16);
    params.wire_scale_vectors = static_cast<int>(row.wireScale / 16);
    params.blocks_per_token = static_cast<int>(sentinelDispatchBlocksPerToken(row));
    params.invalid_expert_id = static_cast<int32_t>(invalidExpertId);
    params.active_rank_mask = ~uint64_t{0}; // every rank: a sentinel step takes no rank mask
    params.timeout_cycles = gTimeoutCycles.load(std::memory_order_relaxed);

    sentinel::PayloadKind const payloadKind = payloadFormat == sentinel::kBf16 ? sentinel::kKindBf16
        : payloadFormat == sentinel::kNvfp4                                    ? sentinel::kKindNvfp4
                                                                               : sentinel::kKindFp8;
    sentinel::moe_a2a_sentinel_dispatch_launch(payloadKind, params, at::cuda::getCurrentCUDAStream());
}

// A sentinel combine of payload [ep_size, runtimeMaxTokensPerRank, hidden_size] bf16 into output.
void sentinelCombine(torch::Tensor const& payload, int64_t localNumTokens, torch::Tensor const& workspace,
    MoeA2AWorkspaceLayout const& layout, int64_t runtimeMaxTokensPerRank, int64_t epRank, int64_t epSize, int64_t topK,
    torch::Tensor& output)
{
    sentinel::CombineParams params{};
    setSentinelWorkspace(params, workspace, layout, epRank, epSize);
    int64_t const hiddenSize = layout[SENTINEL_HIDDEN_SIZE_INDEX];
    TORCH_CHECK(runtimeMaxTokensPerRank <= layout[SENTINEL_MAX_NUM_TOKENS_INDEX], "A sentinel step takes at most ",
        layout[SENTINEL_MAX_NUM_TOKENS_INDEX], " tokens per rank");
    TORCH_CHECK(payload.scalar_type() == at::kBFloat16, "A sentinel combine takes bf16 expert outputs");
    TORCH_CHECK(payload.size(2) == hiddenSize, "A sentinel combine takes rows of the workspace's hidden size");

    params.expert_output = static_cast<uint8_t const*>(payload.data_ptr());
    params.output = static_cast<uint8_t*>(output.data_ptr());
    params.top_k = static_cast<int>(topK);
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
    params.hidden_size = static_cast<int>(hiddenSize);
    params.row_vectors = static_cast<int>(hiddenSize / 8);
    params.blocks_per_token = (params.row_vectors + sentinel::kNumThreads - 1) / sentinel::kNumThreads;
    params.active_rank_mask = ~uint64_t{0}; // every rank: a sentinel step takes no rank mask
    params.timeout_cycles = gTimeoutCycles.load(std::memory_order_relaxed);

    sentinel::moe_a2a_sentinel_combine_launch(params, at::cuda::getCurrentCUDAStream());
}

// MoE All-to-All Dispatch Operation
// This operation dispatches tokens and their associated payloads to different expert ranks.
//
// Inputs:
//   - tokenSelectedExperts: [local_num_tokens, top_k] tensor of expert indices
//   - inputPayloads: List of tensors with shape [local_num_tokens, ...] containing data to dispatch
//   - workspace: [ep_size, size_per_rank] unified virtual memory workspace where size_per_rank is large enough to store
//   all the auxiliary data and recv payloads.
//   - metainfo: [NUM_METAINFO_FIELDS] tensor containing offsets for auxiliary data
//   - runtimeMaxTokensPerRank: Maximum of the number of tokens of each DP rank's local batch. This is a dynamic value
//   during runtime.
//   - maxNumTokens: Maximum number of tokens that could be supported. This is a static value that is setup during
//   initialization.
//   - epRank: Current expert parallel rank
//   - epSize: Total expert parallel size
//   - topK: Number of experts selected per token
//   - numExperts: Total number of routing slots (tokenSelectedExperts values are in [0, numExperts))
//   - eplbStatsNumExperts: Number of experts used for EPLB stats (may be <= numExperts)
//   - eplbLocalStats: [eplbStatsNumExperts] tensor containing local statistics for EPLB.
//   - useSentinel: run the sentinel transport; the payloads must be the hidden states, their scales (quantized
//   formats), the expert ids and optionally the weights, in the workspace's sentinel payload format, and
//   expertIdPayloadIndex / invalidTokenExpertId must be given. Received rows are then (source rank, token), not
//   compacted, and rows of tokens not routed here carry invalidTokenExpertId.
//
// Return values:
//   - recvTensors: Vector of receive buffers (one tensor per payload), each [ep_size, runtimeMaxTokensPerRank,
//   elements_per_token]
//   - combinePayloadOffset: Offset into workspace for the combine payload region, to be used by the combine operation
//   - eplbGatheredStats: (Optional) [ep_size, eplbStatsNumExperts] tensor containing gathered statistics for EPLB, or
//   an empty tensor if eplbLocalStats is None.
//
// Note: token_selected_experts is used for routing but is NOT automatically included as a payload.
//       If you want to dispatch token_selected_experts, include it explicitly in inputPayloads.
std::tuple<std::vector<torch::Tensor>, int64_t, torch::Tensor> moeA2ADispatchOp(
    torch::Tensor const& tokenSelectedExperts, std::vector<torch::Tensor> const& inputPayloads,
    torch::Tensor const& workspace, torch::Tensor const& metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank,
    int64_t epSize, int64_t topK, int64_t numExperts, torch::optional<torch::Tensor> eplbLocalStats,
    torch::optional<int64_t> expertIdPayloadIndex, torch::optional<int64_t> invalidTokenExpertId, bool enableRankMask,
    torch::optional<torch::Tensor> activeRankMask, bool useSentinel)
{
    using tensorrt_llm::kernels::moe_comm::PayloadDescriptor;
    using tensorrt_llm::kernels::moe_comm::MoeA2ADispatchParams;
    using tensorrt_llm::kernels::moe_comm::moe_a2a_dispatch_launch;
    using tensorrt_llm::kernels::moe_comm::kMaxTopK;
    using tensorrt_llm::kernels::moe_comm::kMaxPayloads;
    using tensorrt_llm::kernels::moe_comm::kMaxRanks;

    // Validate inputs
    CHECK_INPUT(tokenSelectedExperts, torch::kInt32);
    TORCH_CHECK(tokenSelectedExperts.dim() == 2, "tokenSelectedExperts must be a 2D tensor");
    TORCH_CHECK(tokenSelectedExperts.size(1) == topK, "tokenSelectedExperts must have topK columns");

    auto const& offsets = readWorkspaceLayout(metainfo);
    TORCH_CHECK(
        !useSentinel || offsets[SENTINEL_MAX_NUM_TOKENS_INDEX] > 0, "Workspace was allocated without sentinel regions");

    int64_t localNumTokens = tokenSelectedExperts.size(0);
    TORCH_CHECK(runtimeMaxTokensPerRank > 0, "runtimeMaxTokensPerRank must be positive");
    TORCH_CHECK(runtimeMaxTokensPerRank <= offsets[MAX_NUM_TOKENS_INDEX],
        "runtimeMaxTokensPerRank exceeds the allocation-time token capacity");
    TORCH_CHECK(epSize > 0 && epSize <= kMaxRanks, "epSize must be in the range (0, ", kMaxRanks, "]");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");
    TORCH_CHECK(topK > 0 && topK <= kMaxTopK, "topK must be in the range (0, kMaxTopK]");
    TORCH_CHECK(!inputPayloads.empty(), "inputPayloads must not be empty");
    TORCH_CHECK(inputPayloads.size() <= kMaxPayloads, "Too many input payloads");
    TORCH_CHECK(numExperts >= epSize, "numExperts must be greater than or equal to epSize");
    // numExperts does not need to be divisible by epSize: the kernel performs
    // ceil/floor contiguous partitioning so ranks [0, numExperts % epSize)
    // own (numExperts / epSize + 1) experts and the rest own (numExperts / epSize).

    bool const sanitizeExpertIds = expertIdPayloadIndex.has_value() || invalidTokenExpertId.has_value();
    TORCH_CHECK(expertIdPayloadIndex.has_value() == invalidTokenExpertId.has_value(),
        "expert_id_payload_index and invalid_token_expert_id must be provided together");
    bool enableEplb = eplbLocalStats.has_value();
    int64_t eplbStatsNumExperts = 0;
    if (enableEplb)
    {
        TORCH_CHECK(eplbLocalStats.has_value(), "enable_eplb requires eplb_local_stats");
        torch::Tensor const& eplbLocalStatsTensor = eplbLocalStats.value();
        eplbStatsNumExperts = eplbLocalStatsTensor.size(0);
        TORCH_CHECK(eplbStatsNumExperts > 0, "eplb_local_stats must not be empty");
        TORCH_CHECK(eplbStatsNumExperts <= numExperts, "eplb_local_stats size must be <= numExperts (slots)");
        CHECK_INPUT(eplbLocalStatsTensor, torch::kInt32);
        TORCH_CHECK(eplbLocalStatsTensor.is_contiguous(), "eplb_local_stats must be contiguous");
        TORCH_CHECK(eplbLocalStatsTensor.dim() == 1, "eplb_local_stats must be a 1D tensor");
    }

    // All input payloads must have the same first dimension (localNumTokens)
    for (auto const& payload : inputPayloads)
    {
        TORCH_CHECK(payload.dim() >= 1, "All payloads must have at least 1 dimension");
        TORCH_CHECK(payload.size(0) == localNumTokens,
            "All payloads must have the same first dimension as tokenSelectedExperts");
        TORCH_CHECK(payload.is_contiguous(), "All payloads must be contiguous");
    }

    // Record the cacheline aligned start offset for each payload's recv buffer.
    // 1. We assume the base workspace ptr of each rank is aligned (checked in this OP)
    // 2. offsets[DISPATCH_PAYLOAD_OFFSET_INDEX] is aligned (fixed by the workspace layout)
    // 3. We align the currentOffset during update.
    // In this way, it is guaranteed that the recv buffer is (over-)aligned, sufficient for 128bit vectorized ld/st.

    std::vector<int> payloadElementSizes;
    std::vector<int> payloadElementsPerToken;
    std::vector<size_t> payloadRecvBufferOffsets;

    // Start offset for the first payload. A sentinel step receives into its own region.
    int64_t const recvRegionOffset = offsets[useSentinel ? SENTINEL_RECV_OFFSET_INDEX : DISPATCH_PAYLOAD_OFFSET_INDEX];
    int64_t const recvRegionSize = offsets[useSentinel ? SENTINEL_RECV_SIZE_INDEX : DISPATCH_PAYLOAD_SIZE_INDEX];
    size_t currentOffset = static_cast<size_t>(recvRegionOffset);
    for (auto const& payload : inputPayloads)
    {
        CHECK_CONTIGUOUS(payload);
        CHECK_TH_CUDA(payload);
        TORCH_CHECK(payload.dim() == 2, "payload must be a 2D tensor");
        TORCH_CHECK(
            payload.size(0) == localNumTokens, "payload must have the same first dimension as tokenSelectedExperts");
        // Unlike recv buffer for payloads, payload itself is not allocated by us and we cannot control its alignment.
        // We only make sure the payload start offset is 16-byte aligned, while the actual vectorized ld/st width is
        // dynamically determined based on bytes per token of this payload.
        TORCH_CHECK(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16 == 0, "payload must be 16-byte aligned");

        int elementsPerToken = static_cast<int>(payload.size(1));
        int elementSize = static_cast<int>(payload.dtype().itemsize());
        // Each payload buffer stores data from ALL ranks
        int64_t bytesPerPayload = epSize * runtimeMaxTokensPerRank * elementsPerToken * elementSize;

        payloadElementSizes.push_back(elementSize);
        payloadElementsPerToken.push_back(elementsPerToken);

        payloadRecvBufferOffsets.push_back(currentOffset);

        // Update offset and align to cacheline boundary for the next payload recv buffer.
        currentOffset += bytesPerPayload;
        currentOffset = alignOffset(currentOffset, CACHELINE_ALIGNMENT);
    }

    int expertIdPayloadIdx = -1;
    int32_t invalidExpertId = -1;
    if (sanitizeExpertIds)
    {
        expertIdPayloadIdx = static_cast<int>(*expertIdPayloadIndex);
        TORCH_CHECK(expertIdPayloadIdx >= 0 && expertIdPayloadIdx < static_cast<int>(inputPayloads.size()),
            "expert_id_payload_index out of range");
        auto const& expertIdPayload = inputPayloads[expertIdPayloadIdx];
        CHECK_TYPE(expertIdPayload, torch::kInt32);
        TORCH_CHECK(expertIdPayload.size(1) == topK, "expert-id payload must have topK columns");
        invalidExpertId = static_cast<int32_t>(*invalidTokenExpertId);
    }

    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    // Don't check contiguous - MnnvlMemory creates strided tensors for multi-GPU
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");

    TORCH_CHECK(epSize == offsets[EP_SIZE_INDEX] && topK == offsets[TOP_K_INDEX],
        "Dispatch EP size/top_k differs from its workspace layout");
    TORCH_CHECK(localNumTokens <= runtimeMaxTokensPerRank, "Local token count exceeds the runtime capacity");
    TORCH_CHECK(eplbStatsNumExperts <= offsets[EPLB_STATS_NUM_EXPERTS_INDEX],
        "EPLB statistics exceed their workspace capacity");
    TORCH_CHECK(workspace.size(1) >= offsets[WORKSPACE_SIZE_INDEX], "Workspace is smaller than its layout");
    int64_t const combinePayloadOffset = offsets[COMBINE_INPUT_OFFSET_INDEX];
    TORCH_CHECK(currentOffset <= static_cast<size_t>(recvRegionOffset + recvRegionSize),
        "Dispatch payload exceeds its workspace capacity: need ", currentOffset - recvRegionOffset, " bytes, capacity ",
        recvRegionSize);

    // Get base workspace pointer
    uint8_t* workspacePtr = workspace.data_ptr<uint8_t>();
    uint8_t* rankWorkSpacePtr = workspacePtr + epRank * workspace.stride(0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(rankWorkSpacePtr) % CACHELINE_ALIGNMENT == 0,
        "rankWorkSpacePtr must be %d-byte aligned", CACHELINE_ALIGNMENT);
    TORCH_CHECK(!openSentinelRoundOf(rankWorkSpacePtr).has_value(),
        "A sentinel dispatch must be followed by its sentinel combine before the next dispatch");

    // Create tensor views for the current rank's receive buffers only
    auto makeRecvTensors = [&]()
    {
        std::vector<torch::Tensor> recvTensors;
        for (size_t payload_idx = 0; payload_idx < inputPayloads.size(); payload_idx++)
        {
            void* recvDataPtr = rankWorkSpacePtr + payloadRecvBufferOffsets[payload_idx];
            recvTensors.push_back(
                torch::from_blob(recvDataPtr, {epSize, runtimeMaxTokensPerRank, payloadElementsPerToken[payload_idx]},
                    inputPayloads[payload_idx].options()));
        }
        return recvTensors;
    };

    if (useSentinel)
    {
        TORCH_CHECK(!enableEplb, "A sentinel dispatch does not gather EPLB statistics");
        // The sentinel kernels never drain or re-arm a masked peer's rows; fault tolerance uses fence.
        TORCH_CHECK(
            !enableRankMask && !hasActiveRankMask(activeRankMask), "A sentinel dispatch does not take a rank mask");
        TORCH_CHECK(sanitizeExpertIds,
            "A sentinel dispatch needs expert_id_payload_index and invalid_token_expert_id: it fills the expert ids of "
            "rows without a token");
        sentinelDispatch(tokenSelectedExperts, inputPayloads, expertIdPayloadIdx, workspace, offsets,
            payloadRecvBufferOffsets, runtimeMaxTokensPerRank, epRank, epSize, topK, numExperts, invalidExpertId);
        cudaError_t result = cudaGetLastError();
        TORCH_CHECK(
            result == cudaSuccess, "moe_a2a_dispatch sentinel kernel launch failed: ", cudaGetErrorString(result));
        openSentinelRound(rankWorkSpacePtr, SentinelRound{runtimeMaxTokensPerRank, localNumTokens, enableRankMask});
        return std::make_tuple(
            makeRecvTensors(), combinePayloadOffset, torch::empty({0}, workspace.options().dtype(torch::kInt32)));
    }

    // Setup payload descriptors for source data
    int num_payloads = static_cast<int>(inputPayloads.size());
    std::vector<PayloadDescriptor> payloadDescriptors(num_payloads);
    for (int i = 0; i < num_payloads; i++)
    {
        payloadDescriptors[i].src_data = inputPayloads[i].data_ptr();
        payloadDescriptors[i].element_size = payloadElementSizes[i];
        payloadDescriptors[i].elements_per_token = payloadElementsPerToken[i];
    }

    // Setup dispatch parameters
    MoeA2ADispatchParams params{};
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.num_experts = static_cast<int>(numExperts);
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
    params.top_k = static_cast<int>(topK);
    params.enable_eplb = enableEplb;
    params.eplb_stats_num_experts = static_cast<int>(eplbStatsNumExperts);

    params.token_selected_experts = tokenSelectedExperts.data_ptr<int32_t>();

    params.num_payloads = num_payloads;
    std::copy(payloadDescriptors.begin(), payloadDescriptors.end(), &params.payloads[0]);

    params.flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);
    params.local_token_counter = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[LOCAL_TOKEN_COUNTER_OFFSET_INDEX]);
    params.send_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[SEND_COUNTERS_OFFSET_INDEX]);
    params.topk_target_ranks = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_RANKS_OFFSET_INDEX]);
    params.topk_target_indices = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_INDICES_OFFSET_INDEX]);

    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        uint8_t* targetWorkSpacePtr = workspacePtr + (target_rank * workspace.stride(0));

        params.recv_counters[target_rank]
            = reinterpret_cast<int*>(targetWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);
        params.completion_flags[target_rank]
            = reinterpret_cast<uint32_t*>(targetWorkSpacePtr + offsets[DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX]);
        if (enableEplb)
        {
            params.eplb_gathered_stats[target_rank]
                = reinterpret_cast<int*>(targetWorkSpacePtr + offsets[EPLB_GATHERED_STATS_OFFSET_INDEX]);
        }
        else
        {
            params.eplb_gathered_stats[target_rank] = nullptr;
        }

        for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
        {
            // Store pointer for current payload using pre-calculated aligned offset
            params.recv_buffers[target_rank][payload_idx] = targetWorkSpacePtr + payloadRecvBufferOffsets[payload_idx];
        }
    }

    if (enableEplb)
    {
        params.eplb_local_stats = eplbLocalStats.value().data_ptr<int32_t>();
    }
    else
    {
        params.eplb_local_stats = nullptr;
    }

    // Resolve the optional active-rank mask. Default (no mask) = all bits set, which
    // exactly reproduces the pre-fault-tolerance kernel behavior.
    params.enable_rank_mask = enableRankMask;
    if (params.enable_rank_mask)
    {
        resolveActiveRankMask(activeRankMask, epRank, params.active_rank_mask);
    }
    else
    {
        TORCH_CHECK(!hasActiveRankMask(activeRankMask), "active_rank_mask requires enable_rank_mask=True");
    }

    params.stream = at::cuda::getCurrentCUDAStream();
    params.timeout_cycles = gTimeoutCycles.load(std::memory_order_relaxed);

    // Prepare for dispatch (zero counters/indices and increment flag_val)
    moe_a2a_prepare_dispatch_launch(params);

    // Launch the dispatch kernel
    moe_a2a_dispatch_launch(params);

    cudaError_t result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "moe_a2a_dispatch kernel launch failed: ", cudaGetErrorString(result));

    std::vector<torch::Tensor> recvTensors = makeRecvTensors();

    torch::Tensor eplbGatheredStats;
    if (enableEplb)
    {
        int* gatheredStatsPtr = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[EPLB_GATHERED_STATS_OFFSET_INDEX]);
        auto statsOptions = workspace.options().dtype(torch::kInt32);
        eplbGatheredStats = torch::from_blob(
            gatheredStatsPtr, {static_cast<int64_t>(epSize), static_cast<int64_t>(eplbStatsNumExperts)}, statsOptions);
    }
    else
    {
        eplbGatheredStats = torch::empty({0}, workspace.options().dtype(torch::kInt32));
    }

    return std::make_tuple(std::move(recvTensors), combinePayloadOffset, std::move(eplbGatheredStats));
}

// MoE All-to-All Combine Operation
// Combine the per-rank expert outputs into the originating tokens' buffers on the local rank.
//
// The payload may be external or a view of the combine input region. Recognize the input
// region by its address; payloadInWorkspace=true additionally requires that zero-copy path.
// Other sources, including dispatch payload views, are staged when needed.
// Fence combine reads from 'combinePayloadOffset'. A sentinel combine (useSentinel, after a sentinel
// dispatch) reads the payload where it is and pushes it.
torch::Tensor moeA2ACombineOp(torch::Tensor const& payload, int64_t localNumTokens, torch::Tensor const& workspace,
    torch::Tensor const& metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank, int64_t epSize, int64_t topK,
    int64_t combinePayloadOffset, bool payloadInWorkspace, bool useLowPrecision = false, bool enableRankMask = false,
    torch::optional<torch::Tensor> activeRankMask = torch::nullopt, bool useSentinel = false)
{
    using tensorrt_llm::kernels::moe_comm::MoeA2ACombineParams;
    using tensorrt_llm::kernels::moe_comm::moe_a2a_combine_launch;
    using tensorrt_llm::kernels::moe_comm::kMaxTopK;
    using tensorrt_llm::kernels::moe_comm::kMaxRanks;

    // Validate inputs
    CHECK_TH_CUDA(payload);
    CHECK_CONTIGUOUS(payload);
    TORCH_CHECK(payload.dim() == 3, "payload must be a 3D tensor [ep_size, max_tokens_per_rank, elements_per_token]");
    TORCH_CHECK(payload.size(0) == epSize, "payload first dimension must equal epSize");
    TORCH_CHECK(runtimeMaxTokensPerRank > 0, "runtimeMaxTokensPerRank must be positive");
    TORCH_CHECK(
        payload.size(1) == runtimeMaxTokensPerRank, "payload second dimension must equal runtimeMaxTokensPerRank");
    // We only make sure the payload start offset is 16-byte aligned, while the actual vectorized ld/st width is
    // dynamically determined based on bytes per token of this payload.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16 == 0, "payload must be 16-byte aligned");
    int64_t elementsPerToken = payload.size(2);
    TORCH_CHECK(elementsPerToken > 0, "elementsPerToken must be positive");
    TORCH_CHECK(epSize > 0 && epSize <= kMaxRanks, "epSize must be in the range (0, ", kMaxRanks, "]");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");
    TORCH_CHECK(topK > 0 && topK <= kMaxTopK, "topK must be in the range (0, kMaxTopK]");

    // Map torch dtype to tensorrt_llm::DataType
    tensorrt_llm::DataType nvDtype = tensorrt_llm::DataType::kFLOAT;
    auto scalarType = payload.scalar_type();
    if (scalarType == at::kHalf)
    {
        nvDtype = tensorrt_llm::DataType::kHALF;
    }
    else if (scalarType == at::kBFloat16)
    {
        nvDtype = tensorrt_llm::DataType::kBF16;
    }
    else if (scalarType == at::kFloat)
    {
        nvDtype = tensorrt_llm::DataType::kFLOAT;
    }
    else
    {
        TORCH_CHECK(false, "Unsupported data type for payload");
    }
    // use_low_precision is passed through to the kernel via params.use_low_precision; dtype is not mutated.

    auto const& offsets = readWorkspaceLayout(metainfo);

    // Validate workspace and set synchronization pointers
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) == epSize, "workspace must be [ep_size, size_per_rank]");
    uint8_t* workspacePtr = workspace.data_ptr<uint8_t>();
    int64_t sizePerRank = workspace.size(1);
    uint8_t* rankWorkSpacePtr = workspacePtr + epRank * workspace.stride(0);
    TORCH_CHECK(epSize == offsets[EP_SIZE_INDEX] && topK == offsets[TOP_K_INDEX],
        "Combine EP size/top_k differs from its workspace layout");
    TORCH_CHECK(sizePerRank >= offsets[WORKSPACE_SIZE_INDEX], "Workspace is smaller than its layout");
    int64_t const regionSize = offsets[COMBINE_INPUT_SIZE_INDEX];
    TORCH_CHECK(combinePayloadOffset == offsets[COMBINE_INPUT_OFFSET_INDEX],
        "combinePayloadOffset must address the fixed combine source region");
    TORCH_CHECK(runtimeMaxTokensPerRank <= offsets[MAX_NUM_TOKENS_INDEX],
        "runtimeMaxTokensPerRank exceeds the allocation-time token capacity");
    uint8_t* combinePayloadPtr = rankWorkSpacePtr + combinePayloadOffset;
    // If the caller claims the payload is in the workspace, ensure it really is: a mismatch would
    // otherwise silently fall back to staging and lose the zero-copy path the caller asked for.
    bool const inputIsWorkspace = payload.data_ptr() == combinePayloadPtr;
    TORCH_CHECK(!payloadInWorkspace || inputIsWorkspace,
        "payload_in_workspace is true but payload does not address the combine input region");
    int64_t payloadSize = payload.numel() * payload.element_size();
    TORCH_CHECK(payloadSize <= regionSize, "Combine payload exceeds its fixed workspace region: need ", payloadSize,
        " bytes, capacity ", regionSize);
    auto const openRound = openSentinelRoundOf(rankWorkSpacePtr);
    TORCH_CHECK(useSentinel == openRound.has_value(),
        useSentinel ? "A sentinel combine needs a sentinel dispatch before it"
                    : "A sentinel dispatch must be followed by a sentinel combine");
    SentinelRound const thisRound{runtimeMaxTokensPerRank, localNumTokens, enableRankMask};
    TORCH_CHECK(!useSentinel || *openRound == thisRound,
        "A sentinel combine must use its dispatch's runtime_max_tokens_per_rank, local token count and rank-mask "
        "mode");

    // Create output tensor (local on current rank), no need for initialization
    // Typically, newly allocated GPU torch tensors are at least 16-byte aligned.
    // Output dtype always matches the payload dtype: low-precision accumulates FP8 back to payload dtype.
    auto output_options = payload.options();
    torch::Tensor output = torch::empty({localNumTokens, elementsPerToken}, output_options);

    if (useSentinel)
    {
        TORCH_CHECK(!useLowPrecision, "A sentinel combine does not use the FP8 wire format");
        TORCH_CHECK(
            !enableRankMask && !hasActiveRankMask(activeRankMask), "A sentinel combine does not take a rank mask");
        sentinelCombine(
            payload, localNumTokens, workspace, offsets, runtimeMaxTokensPerRank, epRank, epSize, topK, output);
        cudaError_t result = cudaGetLastError();
        TORCH_CHECK(
            result == cudaSuccess, "moe_a2a_combine sentinel kernel launch failed: ", cudaGetErrorString(result));
        closeSentinelRound(rankWorkSpacePtr);
        return output;
    }

    // Setup combine parameters
    MoeA2ACombineParams params{};
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
    params.top_k = static_cast<int>(topK);
    params.source_payload = payload.data_ptr();
    params.output_data = output.data_ptr();
    params.elements_per_token = static_cast<int>(elementsPerToken);
    params.dtype = nvDtype;
    params.use_low_precision = useLowPrecision;
    params.source_stride_per_token = static_cast<int>(elementsPerToken * payload.element_size());
    // FP8 staged out of an external payload is packed at one byte per element; otherwise the combine
    // input keeps the payload's stride.
    params.workspace_stride_per_token
        = useLowPrecision && !inputIsWorkspace ? static_cast<int>(elementsPerToken) : params.source_stride_per_token;

    params.flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);
    params.topk_target_ranks = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_RANKS_OFFSET_INDEX]);
    params.topk_target_indices = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_INDICES_OFFSET_INDEX]);
    params.recv_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);

    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        uint8_t* target_workspace_ptr = workspacePtr + target_rank * workspace.stride(0);
        params.completion_flags[target_rank]
            = reinterpret_cast<uint32_t*>(target_workspace_ptr + offsets[COMBINE_COMPLETION_FLAGS_OFFSET_INDEX]);
        params.combine_input_buffers[target_rank] = target_workspace_ptr + combinePayloadOffset;
    }

    // Resolve the optional active-rank mask. Default (no mask) = all bits set.
    params.enable_rank_mask = enableRankMask;
    if (params.enable_rank_mask)
    {
        resolveActiveRankMask(activeRankMask, epRank, params.active_rank_mask);
    }
    else
    {
        TORCH_CHECK(!hasActiveRankMask(activeRankMask), "active_rank_mask requires enable_rank_mask=True");
    }

    // Resolve the complete payload plan once. Prepare always launches at least one block to
    // advance flag_val, but prepare_num_tokens=0 performs no payload work.
    if (params.use_low_precision)
    {
        params.prepare_num_tokens = params.ep_size * params.max_tokens_per_rank;
    }
    else
    {
        params.prepare_num_tokens = inputIsWorkspace ? 0 : params.ep_size * params.max_tokens_per_rank;
    }

    params.stream = at::cuda::getCurrentCUDAStream();
    params.timeout_cycles = gTimeoutCycles.load(std::memory_order_relaxed);

    moe_a2a_prepare_combine_launch(params);

    // Launch the combine kernel.
    moe_a2a_combine_launch(params);
    cudaError_t result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "moe_a2a_combine kernel launch failed: ", cudaGetErrorString(result));

    return output;
}

// Op: moe_a2a_sanitize_expert_ids
void moeA2ASanitizeExpertIdsOp(torch::Tensor& expert_ids, torch::Tensor& workspace, torch::Tensor const& metainfo,
    int64_t epRank, int64_t invalid_expert_id)
{
    CHECK_INPUT(expert_ids, torch::kInt32);
    TORCH_CHECK(expert_ids.dim() == 3, "expert_ids must be [ep_size, runtime_max_tokens_per_rank, top_k]");

    int ep_size = static_cast<int>(expert_ids.size(0));
    int runtime_max_tokens_per_rank = static_cast<int>(expert_ids.size(1));
    int top_k = static_cast<int>(expert_ids.size(2));

    auto const& offsets = readWorkspaceLayout(metainfo);

    uint8_t* rankWorkSpacePtr = workspace.data_ptr<uint8_t>() + epRank * workspace.stride(0);
    int* recv_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);
    uint32_t* flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);

    tensorrt_llm::kernels::moe_comm::moe_a2a_sanitize_expert_ids_launch(expert_ids.data_ptr<int32_t>(), recv_counters,
        flag_val, static_cast<int32_t>(invalid_expert_id), ep_size, runtime_max_tokens_per_rank, top_k,
        at::cuda::getCurrentCUDAStream());
}

// Return a workspace-backed tensor for combine payload region using from_blob
torch::Tensor moeA2AGetCombinePayloadTensorOp(torch::Tensor const& workspace, torch::Tensor const& metainfo,
    int64_t epRank, int64_t epSize, int64_t runtimeMaxTokensPerRank, c10::ScalarType outDtype, int64_t hiddenSize)
{
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be [ep_size, size_per_rank_bytes]");
    TORCH_CHECK(epRank >= 0 && epRank < workspace.size(0), "epRank out of range");
    TORCH_CHECK(epSize == workspace.size(0), "epSize mismatch with workspace");
    TORCH_CHECK(runtimeMaxTokensPerRank > 0, "runtimeMaxTokensPerRank must be positive");
    TORCH_CHECK(hiddenSize > 0, "hidden must be positive");

    int64_t sizePerRank = workspace.size(1); // bytes
    int64_t elementSize = static_cast<int64_t>(c10::elementSize(outDtype));
    int64_t bytesNeeded = epSize * runtimeMaxTokensPerRank * hiddenSize * elementSize;
    auto const& layout = readWorkspaceLayout(metainfo);
    TORCH_CHECK(epSize == layout[EP_SIZE_INDEX], "Combine view EP size differs from its workspace layout");
    TORCH_CHECK(runtimeMaxTokensPerRank <= layout[MAX_NUM_TOKENS_INDEX],
        "Combine view exceeds the allocation-time token capacity");
    TORCH_CHECK(sizePerRank >= layout[WORKSPACE_SIZE_INDEX], "Workspace is smaller than its layout");
    int64_t const combinePayloadOffset = layout[COMBINE_INPUT_OFFSET_INDEX];
    TORCH_CHECK(bytesNeeded <= layout[COMBINE_INPUT_SIZE_INDEX], "Combine view exceeds its input region: need ",
        bytesNeeded, " bytes, capacity ", layout[COMBINE_INPUT_SIZE_INDEX]);

    uint8_t* base = workspace.data_ptr<uint8_t>();
    uint8_t* rankBase = base + epRank * workspace.stride(0);
    uint8_t* dataPtr = rankBase + combinePayloadOffset;

    auto options = workspace.options().dtype(outDtype);
    torch::Tensor t = torch::from_blob(dataPtr, {epSize * runtimeMaxTokensPerRank, hiddenSize}, options);
    return t;
}

} // namespace moe_comm

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// PyTorch bindings
TORCH_LIBRARY_FRAGMENT(trtllm, module)
{
    // Note that we returns recv_tensors as a list of views into workspace, we need to upcast its alias
    // group to wildcard (a!->*). See
    // https://github.com/pytorch/pytorch/blob/b1eb6dede556136f9fdcee28415b0358d58ad877/aten/src/ATen/native/README.md#annotations
    module.def(
        "moe_a2a_dispatch(Tensor token_selected_experts, Tensor[] input_payloads, "
        "Tensor(a!->*) workspace, Tensor metainfo, int runtime_max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k, int num_experts, "
        "Tensor? eplb_local_stats=None, "
        "int? expert_id_payload_index=None, "
        "int? invalid_token_expert_id=None, "
        "bool enable_rank_mask=False, "
        "Tensor? active_rank_mask=None, "
        "bool use_sentinel=False) -> (Tensor(a!)[], int, Tensor(a!))");
    module.def(
        "moe_a2a_combine(Tensor(a) payload, int local_num_tokens,"
        "Tensor(a!) workspace, Tensor metainfo, int runtime_max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k, int combine_payload_offset, "
        "bool payload_in_workspace, "
        "bool use_low_precision=False, "
        "bool enable_rank_mask=False, "
        "Tensor? active_rank_mask=None, "
        "bool use_sentinel=False) -> Tensor");
    module.def("moe_a2a_initialize(Tensor(a!) workspace, Tensor metainfo, int ep_rank, int ep_size) -> ()");
    module.def(
        "moe_a2a_sanitize_expert_ids(Tensor(a!) expert_ids, Tensor(a!) workspace, Tensor metainfo, int ep_rank, int "
        "invalid_expert_id) -> ()");
    module.def(
        "moe_a2a_get_combine_payload_tensor(Tensor(a) workspace, Tensor metainfo, int ep_rank, int ep_size, "
        "int runtime_max_tokens_per_rank, ScalarType out_dtype, int hidden_size) -> Tensor(a)");
    module.def("moe_a2a_set_timeout(int timeout_sec) -> ()", &tensorrt_llm::torch_ext::moe_comm::moeA2ASetTimeoutOp);
    module.def("moe_a2a_get_timeout() -> int", &tensorrt_llm::torch_ext::moe_comm::moeA2AGetTimeoutOp);
    module.def(
        "moe_a2a_get_workspace_layout(int ep_size, int max_num_tokens_per_rank, int top_k, "
        "int dispatch_payload_bytes, int combine_input_bytes, int? eplb_stats_num_experts=None) -> Tensor",
        &tensorrt_llm::torch_ext::moe_comm::moeA2AGetWorkspaceLayoutOp);
    module.def(
        "moe_a2a_sentinel_extend_workspace_layout(Tensor(a!) metainfo, int sentinel_max_tokens, int payload_format, "
        "int hidden_size) -> ()",
        &tensorrt_llm::torch_ext::moe_comm::moeA2ASentinelExtendWorkspaceLayoutOp);
    module.def("moe_a2a_sentinel_row_geometry(int payload_format, int hidden_size, int top_k) -> int[]",
        &tensorrt_llm::torch_ext::moe_comm::moeA2ASentinelRowGeometryOp);
    module.def("moe_a2a_sentinel_available() -> bool", &tensorrt_llm::torch_ext::moe_comm::moeA2ASentinelAvailableOp);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, module)
{
    module.impl("moe_a2a_dispatch", &tensorrt_llm::torch_ext::moe_comm::moeA2ADispatchOp);
    module.impl("moe_a2a_combine", &tensorrt_llm::torch_ext::moe_comm::moeA2ACombineOp);
    module.impl("moe_a2a_initialize", &tensorrt_llm::torch_ext::moe_comm::moeA2AInitializeOp);
    module.impl("moe_a2a_sanitize_expert_ids", &tensorrt_llm::torch_ext::moe_comm::moeA2ASanitizeExpertIdsOp);
    module.impl(
        "moe_a2a_get_combine_payload_tensor", &tensorrt_llm::torch_ext::moe_comm::moeA2AGetCombinePayloadTensorOp);
}
