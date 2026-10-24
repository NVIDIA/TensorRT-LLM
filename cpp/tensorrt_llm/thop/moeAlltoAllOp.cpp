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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllCftManager.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernelsLegacy.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/moeAlltoAllMeta.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <torch/extension.h>
#include <torch/types.h>
#include <unordered_map>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace moe_comm
{

static constexpr size_t CACHELINE_ALIGNMENT = 128;

// TODO: Is Alignment necessary?
// Helper function to align offset to specified byte boundary
inline size_t alignOffset(size_t offset, size_t alignment)
{
    return (offset + alignment - 1) & ~(alignment - 1);
}

inline bool hasActiveRankMask(torch::optional<torch::Tensor> const& maskTensor)
{
    return maskTensor.has_value() && maskTensor.value().defined();
}

// TLLM_MOE_A2A_FUSED_ROUND=1 launches the dispatch and combine kernels of moeAlltoAllKernels.cu:
//   - the fence dispatch prepares its own round and fills the received padding rows' expert ids, so no prepare kernel
//     runs before it and no sanitize kernel after it (expert-id buffers up to kFusedSanitizeMaxBytes);
//   - it loads its payloads before routing (TLLM_MOE_A2A_DISPATCH_PRELOAD, on by default);
//   - the fence combine stages each token's routing in shared memory while it waits for its peers;
//   - a dispatch after an aborted round (a dispatch whose combine was skipped) still takes an odd flag.
// Unset or 0 launches the previous kernels (moeAlltoAllKernelsLegacy.cu) exactly as before: a prepare
// kernel before every dispatch and, after a fence dispatch, the sanitize kernel (which the previous version launched
// from Python). Both give the same results. Read once per process, logged, and any value other than 0 or 1 is rejected;
// all ranks of a group must use the same value.
inline bool useFusedRoundKernels()
{
    static bool const enabled = []
    {
        char const* env = std::getenv("TLLM_MOE_A2A_FUSED_ROUND");
        std::string const value = env == nullptr ? "" : env;
        TLLM_CHECK_WITH_INFO(value.empty() || value == "0" || value == "1",
            "TLLM_MOE_A2A_FUSED_ROUND must be 0 or 1, got '%s'", value.c_str());
        bool const fused = value == "1";
        TLLM_LOG_INFO("MoE A2A: %s kernels (TLLM_MOE_A2A_FUSED_ROUND=%s)", fused ? "fused-round" : "legacy",
            value.empty() ? "<unset>" : value.c_str());
        return fused;
    }();
    return enabled;
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

// Calculate auxiliary data offsets
MoeA2ADataOffsets calculateOffsets(int epSize, int maxNumTokens, int eplbStatsNumExperts)
{
    // TODO: Use lambdas to encapsulate offset and alignment for each entry, which is less error prone and easier to
    // read.
    constexpr size_t SIZEOF_INT32 = 4;

    MoeA2ADataOffsets offsets;
    size_t offset = 0;

    // flag_val
    offsets[FLAG_VAL_OFFSET_INDEX] = offset;
    offset += SIZEOF_INT32;

    // local_token_counter
    offsets[LOCAL_TOKEN_COUNTER_OFFSET_INDEX] = offset;
    offset += SIZEOF_INT32;

    // send_counters
    offsets[SEND_COUNTERS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // recv_counters[parity][source_rank] stores the token count received from source_rank.
    // The two parity banks alternate between A2A rounds.
    offsets[RECV_COUNTERS_OFFSET_INDEX] = offset;
    offset += 2 * epSize * SIZEOF_INT32;

    // dispatch completion flags
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // combine completion flags
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[COMBINE_COMPLETION_FLAGS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // topk_target_ranks: [maxNumTokens, kMaxTopK]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[TOPK_TARGET_RANKS_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(maxNumTokens) * static_cast<size_t>(tensorrt_llm::kernels::moe_comm::kMaxTopK)
        * SIZEOF_INT32;

    // topk_send_indices: [maxNumTokens, kMaxTopK]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[TOPK_SEND_INDICES_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(maxNumTokens) * static_cast<size_t>(tensorrt_llm::kernels::moe_comm::kMaxTopK)
        * SIZEOF_INT32;

    // eplb gathered stats: [epSize, eplbStatsNumExperts]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[EPLB_GATHERED_STATS_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(epSize) * static_cast<size_t>(eplbStatsNumExperts) * SIZEOF_INT32;

    // Counted write counters: each 8B counter must be kCftCounterStride-aligned to avoid
    // L2 XBAR port camping when concurrent counters update (see NCCL CFT perf study).
    using tensorrt_llm::kernels::moe_comm::kCftCounterStride;

    // dispatch counted write counters: [ep_size] uint64_t, kCftCounterStride stride
    offset = alignOffset(offset, kCftCounterStride);
    offsets[DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX] = offset;
    offset += epSize * kCftCounterStride;

    // combine counted write counters (CFT combine path): per receive-slot uint64
    // counters, [ep_size * maxNumTokens], kCftCounterStride stride to avoid L2 XBAR camping.
    offset = alignOffset(offset, kCftCounterStride);
    offsets[COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(epSize) * static_cast<size_t>(maxNumTokens) * kCftCounterStride;

    // dispatch counter baseline: [ep_size] uint64
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[DISPATCH_COUNTER_BASELINE_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(epSize) * sizeof(uint64_t);

    // combine counter baseline: [ep_size * maxNumTokens] uint64
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[COMBINE_COUNTER_BASELINE_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(epSize) * static_cast<size_t>(maxNumTokens) * sizeof(uint64_t);

    // payload data
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[PAYLOAD_DATA_OFFSET_INDEX] = offset;

    // Stable combine slot stride (a count, not a byte offset).
    offsets[MAX_NUM_TOKENS_INDEX] = maxNumTokens;

    return offsets;
}

// Initialize auxiliary data in workspace
// This function sets up the initial values for flag_val and completion_flags
//
// Inputs:
//   - workspace: [ep_size, size_per_rank] unified virtual memory workspace
//   - epRank: Current expert parallel rank
//   - epSize: Total expert parallel size
//   - maxNumTokens: Maximum number of tokens supported
//   - eplbStatsNumExperts: (Optional) Number of experts used for EPLB stats
//
// Returns:
//   - metainfo: Tensor containing offsets for auxiliary data
torch::Tensor moeA2AInitializeOp(torch::Tensor const& workspace, int64_t epRank, int64_t epSize, int64_t maxNumTokens,
    torch::optional<int64_t> eplbStatsNumExperts)
{
    using tensorrt_llm::kernels::moe_comm::kMaxRanks;

    // Validate inputs
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");
    TORCH_CHECK(epSize > 0 && epSize <= kMaxRanks, "epSize must be in the range (0, ", kMaxRanks, "]");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

    int64_t eplbStatsNumExpertsValue = eplbStatsNumExperts.value_or(0);
    TORCH_CHECK(eplbStatsNumExpertsValue >= 0, "eplbStatsNumExperts must be positive if not None.");

    // Calculate auxiliary data offsets
    MoeA2ADataOffsets offsets = calculateOffsets(epSize, maxNumTokens, static_cast<int>(eplbStatsNumExpertsValue));

    // Initialize workspace to zero, then mark both recv-counter parities empty.
    workspace[epRank].zero_();
    uint8_t* rankWorkSpacePtr = workspace.data_ptr<uint8_t>() + epRank * workspace.stride(0);
    cudaMemsetAsync(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX], 0xFF,
        2 * static_cast<size_t>(epSize) * sizeof(int32_t), at::cuda::getCurrentCUDAStream());

    // Return metainfo as a tensor containing offsets
    torch::Tensor metainfo = torch::empty(
        {static_cast<int64_t>(NUM_METAINFO_FIELDS)}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    for (int i = 0; i < static_cast<int>(NUM_METAINFO_FIELDS); i++)
    {
        metainfo[i] = static_cast<int64_t>(offsets[i]);
    }

    // Synchronize among ranks
    cudaDeviceSynchronize();
    tensorrt_llm::mpi::MpiComm::session().barrier();

    return metainfo;
}

// ============================================================================
// CFT Handle-Based Counted Writes Initialization
// ============================================================================

using CftLeManager = tensorrt_llm::kernels::moe_comm::CftLeManager;

// Each workspace has a distinct MNNVL allocation handle and needs its own LE.
// An entry owns an endpoint bound to the workspace allocation, so it must be
// erased before that allocation can be freed and its VA reused. The Python
// NVLinkOneSided teardown calls moe_a2a_cft_release while the workspace is
// still alive. MoeAlltoAll retains its shared workspaces for the process
// lifetime, so those entries remain valid until static destruction.
static std::unordered_map<CUdeviceptr, std::unique_ptr<CftLeManager>> g_cft_managers;

static CftLeManager* getCftManager(CUdeviceptr workspaceRankPtr)
{
    auto const manager = g_cft_managers.find(workspaceRankPtr);
    return manager == g_cft_managers.end() ? nullptr : manager->second.get();
}

// Initialize CFT Logical Endpoints for handle-based counted writes.
//
// This op:
//   1. Creates a CftLeManager that loads LE driver APIs
//   2. Allocates fabric memory for the local rank's LE (recv buffer + counters)
//   3. Exchanges LE handles with all ranks via MPI allgather
//   4. Returns a tensor containing peer LE IDs, counter base, and backing pointer
//
// The LE memory layout:
//   [0, payloadSize)                          — payload data (same layout as workspace recv buffers)
//   [counterBase, counterBase + ep_size * 8)  — per-source-rank 8B counted write counters
//
// Args:
//   workspace: the MNNVL workspace tensor (used for dimensions only)
//   metainfo: the metainfo tensor from moe_a2a_initialize
//   inputPayloads: representative payload tensors (for size calculation)
//   runtimeMaxTokensPerRank: max tokens per rank
//   epRank: current EP rank
//   epSize: total EP size
//
// Returns:
//   cft_info: 1D int64 tensor [ep_size + 3] containing:
//     [0..ep_size-1] : peer LE IDs (uint32_t stored as int64)
//     [ep_size]      : counter base offset within LE
//     [ep_size+1]    : local LE backing pointer (CUdeviceptr stored as int64)
//     [ep_size+2]    : total LE alloc size
// Initialize CFT Logical Endpoints by binding the LE to the MNNVL workspace.
// The workspace memory IS the LE backing store — fabric.try_put.counted writes land
// directly in workspace recv_buffers, eliminating the duplicate allocation and the
// need to know payload layout at init time.
//
// Args:
//   workspaceMemHandle: CUmemGenericAllocationHandle (as int64) from cuMemCreate
//   workspaceRankPtr:   VA pointer to this rank's workspace region (as int64)
//   workspaceSizePerRank: size of the workspace per rank in bytes
//   epRank, epSize: EP topology
void moeA2ACftInitializeOp(torch::Tensor const& workspace, int64_t workspaceMemHandle, int64_t workspaceSizePerRank,
    int64_t epRank, int64_t epSize)
{
    CHECK_TH_CUDA(workspace);
    CUdeviceptr workspaceRankPtr
        = reinterpret_cast<CUdeviceptr>(workspace.data_ptr<uint8_t>() + epRank * workspace.stride(0));
    if (auto const* manager = getCftManager(workspaceRankPtr); manager && manager->isInitialized())
        return;

    auto manager = std::make_unique<CftLeManager>();

    TORCH_CHECK(manager->loadApis(),
        "CftLeManager: Failed to load LE driver APIs. "
        "Ensure CUDA driver 615.00+ is installed.");

    int localDevIdx = -1;
    cudaGetDevice(&localDevIdx);
    TORCH_CHECK(manager->createEndpointExternal(localDevIdx,
                    static_cast<CUmemGenericAllocationHandle>(workspaceMemHandle), workspaceRankPtr,
                    static_cast<size_t>(workspaceSizePerRank), static_cast<int>(epRank), static_cast<int>(epSize)),
        "CftLeManager: Failed to create LE endpoint bound to workspace on device ", localDevIdx);

    auto allgatherFn = [](void const* sendBuf, void* recvBuf, size_t bytesPerRank) {
        tensorrt_llm::mpi::MpiComm::world().allgather(
            sendBuf, recvBuf, bytesPerRank, tensorrt_llm::mpi::MpiType::kBYTE);
    };

    TORCH_CHECK(manager->exchangeEndpoints(allgatherFn), "CftLeManager: Failed to exchange LE endpoints");

    cudaError_t initErr = cudaDeviceSynchronize();
    if (initErr != cudaSuccess)
    {
        fprintf(stderr, "CftLeManager[rank%d]: cudaDeviceSynchronize after init FAILED: %s\n", (int) epRank,
            cudaGetErrorString(initErr));
    }
    tensorrt_llm::mpi::MpiComm::world().barrier();
    g_cft_managers.insert_or_assign(workspaceRankPtr, std::move(manager));
}

// Release the CFT manager before the workspace allocation is freed. Erasing
// the pointer-keyed entry destroys the endpoint and prevents a recycled VA
// from resolving to a manager bound to an old allocation.
void moeA2ACftReleaseOp(torch::Tensor const& workspace, int64_t epRank)
{
    CHECK_TH_CUDA(workspace);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(epRank >= 0 && epRank < workspace.size(0), "epRank must be in the range [0, epSize)");

    CUdeviceptr workspaceRankPtr
        = reinterpret_cast<CUdeviceptr>(workspace.data_ptr<uint8_t>() + epRank * workspace.stride(0));
    g_cft_managers.erase(workspaceRankPtr);
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
//   - quantizeGlobalScale: (Optional) quantize on send. inputPayloads[0] is then the BF16 hidden state
//   [local_num_tokens, hidden] (hidden a multiple of 256, 32-byte aligned) and the dispatch kernel sends its NVFP4
//   quantization, the bytes of fp4_quantize(hidden, quantizeGlobalScale, 16, false, false): the E2M1 codes [hidden / 2]
//   as the first received payload and the linear E4M3 scales [hidden / 16] as the second (both uint8); the other
//   payloads follow, so recvTensors has one more tensor than inputPayloads. quantizeGlobalScale is the one-float FP32
//   device tensor of that call. It needs the fused-round kernels (TLLM_MOE_A2A_FUSED_ROUND=1), the fence dispatch (no
//   counted writes), topK 16 and SM10x, and the payload rows [codes | scales | the other payloads] of a token must fit
//   the kernel's staging in shared memory (at most 4 16-byte vectors per thread of TLLM_MOE_A2A_DISPATCH_BLOCK_SIZE,
//   at least 64 threads, 48 KB in all); expertIdPayloadIndex still indexes inputPayloads. nvlink_one_sided.py
//   (quantize_on_send_enabled) mirrors these launch checks.
//   - expertIdPayloadIndex, invalidTokenExpertId: (Optional, given together) index of the int32 [local_num_tokens,
//   topK] expert-id payload. The received expert ids of every padding row (rows at or beyond the source rank's token
//   count) are set to invalidTokenExpertId, as moe_a2a_sanitize_expert_ids does. With TLLM_MOE_A2A_FUSED_ROUND=1 the
//   dispatch kernel fills them when the expert-id receive buffer has at most kFusedSanitizeMaxBytes, otherwise the
//   sanitize kernel after it. With the previous kernels (default) the counted-write dispatch kernel fills them and
//   a fence dispatch is followed by the sanitize kernel.
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
    bool useCftCountedWrites, torch::optional<int64_t> expertIdPayloadIndex,
    torch::optional<int64_t> invalidTokenExpertId, bool enableRankMask, torch::optional<torch::Tensor> activeRankMask,
    torch::optional<torch::Tensor> quantizeGlobalScale)
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

    CHECK_CPU(metainfo);
    CHECK_TYPE(metainfo, torch::kInt64);
    TORCH_CHECK(metainfo.dim() == 1, "metainfo must be a 1D tensor");
    TORCH_CHECK(metainfo.size(0) == static_cast<int64_t>(NUM_METAINFO_FIELDS),
        "metainfo must have NUM_METAINFO_FIELDS elements");
    MoeA2ADataOffsets const& offsets = *reinterpret_cast<MoeA2ADataOffsets const*>(metainfo.data_ptr<int64_t>());

    int64_t localNumTokens = tokenSelectedExperts.size(0);
    TORCH_CHECK(runtimeMaxTokensPerRank > 0, "runtimeMaxTokensPerRank must be positive");
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
    }

    // What is sent: the input payloads, except that quantize on send replaces the BF16 hidden state (payload 0) by its
    // NVFP4 codes and inserts their scales as payload 1.
    struct SentPayload
    {
        void const* src;
        int elementSize;
        int elementsPerToken;
        torch::TensorOptions options;
    };

    bool const quantizeOnSend = quantizeGlobalScale.has_value();
    int quantizeHidden = 0;
    std::vector<SentPayload> sentPayloads;
    for (size_t i = 0; i < inputPayloads.size(); i++)
    {
        auto const& payload = inputPayloads[i];
        if (quantizeOnSend && i == 0)
        {
            TORCH_CHECK(payload.scalar_type() == torch::kBFloat16, "quantize on send needs a BF16 hidden state");
            quantizeHidden = static_cast<int>(payload.size(1));
            TORCH_CHECK(quantizeHidden > 0 && quantizeHidden % 256 == 0,
                "quantize on send needs a hidden size that is a multiple of 256, got ", quantizeHidden);
            TORCH_CHECK(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 32 == 0,
                "quantize on send needs a 32-byte aligned hidden state");
            auto const wireOptions = payload.options().dtype(torch::kUInt8);
            sentPayloads.push_back({payload.data_ptr(), 1, quantizeHidden / 2, wireOptions});
            sentPayloads.push_back({nullptr, 1, quantizeHidden / 16, wireOptions});
        }
        else
        {
            sentPayloads.push_back({payload.data_ptr(), static_cast<int>(payload.dtype().itemsize()),
                static_cast<int>(payload.size(1)), payload.options()});
        }
    }
    TORCH_CHECK(sentPayloads.size() <= kMaxPayloads, "Too many payloads sent");
    if (quantizeOnSend)
    {
        torch::Tensor const& scale = quantizeGlobalScale.value();
        TORCH_CHECK(!useCftCountedWrites, "quantize on send needs the fence dispatch (no CFT counted writes)");
        TORCH_CHECK(useFusedRoundKernels(), "quantize on send needs TLLM_MOE_A2A_FUSED_ROUND=1");
        TORCH_CHECK(topK == 16, "quantize on send is implemented for topK 16, got ", topK);
        TORCH_CHECK(inputPayloads.size() >= 2, "quantize on send needs the hidden state and the expert-id payload");
        CHECK_INPUT(scale, torch::kFloat32);
        TORCH_CHECK(scale.numel() >= 1, "quantizeGlobalScale must hold an FP32 value");
    }

    // Record the cacheline aligned start offset for each payload's recv buffer.
    // 1. We assume the base workspace ptr of each rank is aligned (checked in this OP)
    // 2. offsets[PAYLOAD_DATA_OFFSET_INDEX] is aligned (ensured in calculateOffsets)
    // 3. We align the currentOffset during update.
    // In this way, it is guaranteed that the recv buffer is (over-)aligned, sufficient for 128bit vectorized ld/st.

    std::vector<int> payloadElementSizes;
    std::vector<int> payloadElementsPerToken;
    std::vector<size_t> payloadRecvBufferOffsets;

    // Start offset for the first payload
    size_t currentOffset = static_cast<size_t>(offsets[PAYLOAD_DATA_OFFSET_INDEX]);
    for (auto const& sent : sentPayloads)
    {
        // Each payload buffer stores data from ALL ranks
        int64_t bytesPerPayload = epSize * runtimeMaxTokensPerRank * sent.elementsPerToken * sent.elementSize;

        payloadElementSizes.push_back(sent.elementSize);
        payloadElementsPerToken.push_back(sent.elementsPerToken);

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
        if (quantizeOnSend)
        {
            // The scales are inserted after the codes: the expert ids of the sent payloads are one further on.
            TORCH_CHECK(expertIdPayloadIdx >= 1, "the expert-id payload cannot be the hidden state");
            expertIdPayloadIdx += 1;
        }
    }

    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    // Don't check contiguous - MnnvlMemory creates strided tensors for multi-GPU
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");

    // Validate workspace size - must include space for auxiliary data + payloads
    int64_t sizePerRank = workspace.size(1);
    int64_t requiredSize = static_cast<int64_t>(currentOffset);
    TORCH_CHECK(sizePerRank >= requiredSize,
        "Workspace size per rank insufficient for dispatch. "
        "Need at least ",
        requiredSize, " bytes (", offsets[PAYLOAD_DATA_OFFSET_INDEX], " for auxiliary data + payloads), but got ",
        sizePerRank);

    // Get base workspace pointer
    uint8_t* workspacePtr = workspace.data_ptr<uint8_t>();
    uint8_t* rankWorkSpacePtr = workspacePtr + epRank * workspace.stride(0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(rankWorkSpacePtr) % CACHELINE_ALIGNMENT == 0,
        "rankWorkSpacePtr must be %d-byte aligned", CACHELINE_ALIGNMENT);

    // Setup payload descriptors for source data
    int num_payloads = static_cast<int>(sentPayloads.size());
    std::vector<PayloadDescriptor> payloadDescriptors(num_payloads);
    for (int i = 0; i < num_payloads; i++)
    {
        payloadDescriptors[i].src_data = sentPayloads[i].src;
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
    params.topk_send_indices = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_SEND_INDICES_OFFSET_INDEX]);

    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        uint8_t* targetWorkSpacePtr = workspacePtr + (target_rank * workspace.stride(0));

        params.recv_counters[target_rank]
            = reinterpret_cast<int*>(targetWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);
        params.completion_flags[target_rank]
            = reinterpret_cast<uint32_t*>(targetWorkSpacePtr + offsets[DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX]);
        params.le_dispatch_counters[target_rank]
            = reinterpret_cast<uint64_t*>(targetWorkSpacePtr + offsets[DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX]);
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

    // CFT requires all payloads to be 16B-aligned (fabric.try_put.counted operates on 16B chunks).
    if (useCftCountedWrites)
    {
        for (int i = 0; i < num_payloads; i++)
        {
            int bytesPerToken = payloadElementSizes[i] * payloadElementsPerToken[i];
            TORCH_CHECK(bytesPerToken % 16 == 0, "CFT dispatch payload ", i, " has ", bytesPerToken,
                " bytes per token; CFT counted writes require 16-byte alignment");
        }
    }
    params.use_cft_counted_writes = useCftCountedWrites;
    bool const fusedRound = useFusedRoundKernels();
    // Fused round: the dispatch kernel fills the padding rows' expert ids itself when the expert-id receive buffer is
    // small (decode). A larger one is filled grid-wide by the sanitize kernel after the dispatch (see below).
    // Legacy: only the counted-write dispatch fills them; a fence dispatch is followed by the sanitize kernel.
    bool const fuseSanitize = sanitizeExpertIds
        && (fusedRound ? epSize * runtimeMaxTokensPerRank * topK * static_cast<int64_t>(sizeof(int32_t))
                    <= tensorrt_llm::kernels::moe_comm::kFusedSanitizeMaxBytes
                       : useCftCountedWrites);
    params.sanitize_expert_ids = fuseSanitize;
    params.expert_id_payload_index = expertIdPayloadIdx;
    params.invalid_expert_id = invalidExpertId;
    if (quantizeOnSend)
    {
        params.quantize_global_scale = quantizeGlobalScale.value().data_ptr<float>();
        params.quantize_hidden = quantizeHidden;
    }

    // CFT handle-based counted writes
    if (useCftCountedWrites)
    {
        auto const* cftManager = getCftManager(reinterpret_cast<CUdeviceptr>(rankWorkSpacePtr));
        TORCH_CHECK(cftManager && cftManager->isInitialized(),
            "CFT counted writes requested but moe_a2a_cft_initialize has not been called");

        // Fill peer LE IDs
        auto const* leIds = cftManager->getAllLeIds();
        for (int i = 0; i < static_cast<int>(epSize); i++)
        {
            params.cft_peer_le_ids[i] = leIds[i];
        }

        // LE payload offsets = workspace payload offsets (LE IS the workspace).
        // No separate LE layout — fabric.try_put.counted writes directly into workspace recv_buffers.
        for (int i = 0; i < num_payloads; i++)
        {
            params.cft_le_payload_offsets[i] = payloadRecvBufferOffsets[i];
        }
        params.cft_le_counter_base = offsets[DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX];

        // recv_buffers and le_dispatch_counters already point to the workspace (set above).
        // No override needed — workspace IS the LE backing store.

        params.cft_dispatch_counter_baseline
            = reinterpret_cast<uint64_t*>(rankWorkSpacePtr + offsets[DISPATCH_COUNTER_BASELINE_OFFSET_INDEX]);
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

    if (fusedRound)
    {
        // The counted-write dispatch needs the prepare kernel (zero counters, increment flag_val, mark the next
        // recv_counters bank). The fence dispatch kernel prepares its own round.
        if (useCftCountedWrites)
        {
            moe_a2a_prepare_dispatch_launch(params);
        }

        // Launch the dispatch kernel
        moe_a2a_dispatch_launch(params);
    }
    else
    {
        // Prepare for dispatch (zero counters/indices and increment flag_val), then dispatch.
        tensorrt_llm::kernels::moe_comm::legacy::moe_a2a_prepare_dispatch_launch(params);
        tensorrt_llm::kernels::moe_comm::legacy::moe_a2a_dispatch_launch(params);
    }

    if (sanitizeExpertIds && !fuseSanitize)
    {
        auto* const recvExpertIds
            = reinterpret_cast<int32_t*>(rankWorkSpacePtr + payloadRecvBufferOffsets[expertIdPayloadIdx]);
        auto const* const recvCounters
            = reinterpret_cast<int32_t const*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);
        if (fusedRound)
        {
            tensorrt_llm::kernels::moe_comm::moe_a2a_sanitize_expert_ids_launch(recvExpertIds, recvCounters,
                params.flag_val, invalidExpertId, params.ep_size, params.max_tokens_per_rank, params.top_k,
                params.stream);
        }
        else
        {
            tensorrt_llm::kernels::moe_comm::legacy::moe_a2a_sanitize_expert_ids_launch(recvExpertIds, recvCounters,
                params.flag_val, invalidExpertId, params.ep_size, params.max_tokens_per_rank, params.top_k,
                params.stream);
        }
    }

    cudaError_t result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "moe_a2a_dispatch kernel launch failed: ", cudaGetErrorString(result));

    // Create tensor views for the current rank's receive buffers only
    std::vector<torch::Tensor> recvTensors;
    for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
    {
        void* recvDataPtr;
        if (useCftCountedWrites)
        {
            // LE IS workspace — recv data is at the same workspace offset regardless of CFT.
            recvDataPtr = rankWorkSpacePtr + payloadRecvBufferOffsets[payload_idx];
        }
        else
        {
            recvDataPtr = rankWorkSpacePtr + payloadRecvBufferOffsets[payload_idx];
        }
        auto recvTensor = torch::from_blob(recvDataPtr,
            {epSize, runtimeMaxTokensPerRank, payloadElementsPerToken[payload_idx]}, sentPayloads[payload_idx].options);
        recvTensors.push_back(recvTensor);
    }

    // Place the combine payload region after the dispatch payload region sized for the STATIC maxNumTokens, not for
    // runtimeMaxTokensPerRank. The combine kernel has no end-of-combine barrier and dispatch has no start barrier, so
    // a fast peer may already run the next dispatch while this rank still reads its combine region. If that dispatch
    // has a larger runtimeMaxTokensPerRank, a runtime-sized dispatch region would extend into this rank's combine
    // region and overwrite it mid-read. The workspace already reserves both regions for maxNumTokens
    // (calculate_required_workspace_size).
    // (A runtimeMaxTokensPerRank above maxNumTokens keeps the old, runtime-sized placement.)
    int64_t const staticMaxTokens = std::max<int64_t>(offsets[MAX_NUM_TOKENS_INDEX], runtimeMaxTokensPerRank);
    size_t staticDispatchEnd = static_cast<size_t>(offsets[PAYLOAD_DATA_OFFSET_INDEX]);
    for (size_t i = 0; i < payloadElementSizes.size(); i++)
    {
        staticDispatchEnd += static_cast<size_t>(epSize) * static_cast<size_t>(staticMaxTokens)
            * static_cast<size_t>(payloadElementsPerToken[i]) * static_cast<size_t>(payloadElementSizes[i]);
        staticDispatchEnd = alignOffset(staticDispatchEnd, CACHELINE_ALIGNMENT);
    }
    int64_t combinePayloadOffset = static_cast<int64_t>(alignOffset(staticDispatchEnd, CACHELINE_ALIGNMENT));
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
// The payload may be external or a view of the normal combine workspace region. The op derives
// ownership from its address and resolves all prepare, push, and reduction strides before launch.
// Fence combine reads from 'combinePayloadOffset'. CFT combine stages the local slice and receives
// peer slices in a dedicated counted-write region before reduction.
torch::Tensor moeA2ACombineOp(torch::Tensor const& payload, int64_t localNumTokens, torch::Tensor const& workspace,
    torch::Tensor const& metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank, int64_t epSize, int64_t topK,
    int64_t combinePayloadOffset, bool useLowPrecision = false, bool useCftCountedWrites = false,
    bool enableRankMask = false, torch::optional<torch::Tensor> activeRankMask = torch::nullopt)
{
    using tensorrt_llm::kernels::moe_comm::MoeA2ACombineParams;
    using tensorrt_llm::kernels::moe_comm::moe_a2a_combine_launch;
    using tensorrt_llm::kernels::moe_comm::moe_a2a_cft_combine_push_launch;
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

    CHECK_CPU(metainfo);
    CHECK_TYPE(metainfo, torch::kInt64);
    TORCH_CHECK(metainfo.dim() == 1, "metainfo must be a 1D tensor");
    TORCH_CHECK(metainfo.size(0) == static_cast<int64_t>(NUM_METAINFO_FIELDS),
        "metainfo must have NUM_METAINFO_FIELDS elements");
    MoeA2ADataOffsets const& offsets = *reinterpret_cast<MoeA2ADataOffsets const*>(metainfo.data_ptr<int64_t>());

    // Validate workspace and set synchronization pointers
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) == epSize, "workspace must be [ep_size, size_per_rank]");
    uint8_t* workspacePtr = workspace.data_ptr<uint8_t>();
    int64_t sizePerRank = workspace.size(1);
    uint8_t* rankWorkSpacePtr = workspacePtr + epRank * workspace.stride(0);
    TORCH_CHECK(combinePayloadOffset >= 0, "combinePayloadOffset must be non-negative");
    uint8_t* combinePayloadPtr = rankWorkSpacePtr + combinePayloadOffset;
    bool const payloadInWorkspace = payload.data_ptr() == combinePayloadPtr;

    int64_t payloadSize = payload.numel() * payload.element_size();
    TORCH_CHECK(combinePayloadOffset + payloadSize <= sizePerRank,
        "Workspace size per rank insufficient for combine. "
        "Need at least ",
        combinePayloadOffset + payloadSize, " bytes (", combinePayloadOffset, " for offset + ", payloadSize,
        " for payload), but got ", sizePerRank);

    // Create output tensor (local on current rank), no need for initialization
    // Typically, newly allocated GPU torch tensors are at least 16-byte aligned.
    // Output dtype always matches the payload dtype: low-precision accumulates FP8 back to payload dtype.
    auto output_options = payload.options();
    torch::Tensor output = torch::empty({localNumTokens, elementsPerToken}, output_options);

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
    params.wire_bytes_per_token
        = static_cast<int>(elementsPerToken) * (useLowPrecision ? 1 : static_cast<int>(payload.element_size()));
    params.workspace_stride_per_token
        = useLowPrecision && !payloadInWorkspace ? params.wire_bytes_per_token : params.source_stride_per_token;

    params.flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);
    params.topk_target_ranks = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_RANKS_OFFSET_INDEX]);
    params.topk_send_indices = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_SEND_INDICES_OFFSET_INDEX]);
    params.recv_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);

    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        uint8_t* target_workspace_ptr = workspacePtr + target_rank * workspace.stride(0);
        params.completion_flags[target_rank]
            = reinterpret_cast<uint32_t*>(target_workspace_ptr + offsets[COMBINE_COMPLETION_FLAGS_OFFSET_INDEX]);
        params.recv_buffers[target_rank] = target_workspace_ptr + combinePayloadOffset;
    }

    // CFT requires the payload to be 16B-aligned (fabric.try_put.counted operates on 16B chunks).
    if (useCftCountedWrites)
    {
        TORCH_CHECK(params.wire_bytes_per_token % 16 == 0, "CFT combine payload has ", params.wire_bytes_per_token,
            " bytes per token; CFT counted writes require 16-byte alignment");
    }

    // ---- CFT combine wiring (counted writes). Sets up dedicated receive region C,
    // per-slot combine counters, and single-buffer baselines. Fence combine ignores these. ----
    params.use_cft_for_combine = useCftCountedWrites;
    auto const* cftManager = getCftManager(reinterpret_cast<CUdeviceptr>(rankWorkSpacePtr));
    if (useCftCountedWrites && cftManager && cftManager->isInitialized())
    {
        auto const* leIds = cftManager->getAllLeIds();
        for (int i = 0; i < static_cast<int>(epSize); i++)
            params.cft_peer_le_ids[i] = leIds[i];

        // Dedicated combine receive region: prepare writes the local slice and fabric pushes write peer slices.
        int64_t combineRecvRegionOffset = alignOffset(combinePayloadOffset + payloadSize, CACHELINE_ALIGNMENT);
        TORCH_CHECK(combineRecvRegionOffset + payloadSize <= sizePerRank,
            "CFT combine: workspace too small for combine receive region C: need ",
            combineRecvRegionOffset + payloadSize, " bytes, got ", sizePerRank);
        params.cft_le_combine_payload_base = static_cast<uint64_t>(combineRecvRegionOffset);
        params.cft_le_combine_counter_base = offsets[COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX];
        params.cft_le_combine_counters
            = reinterpret_cast<uint64_t*>(rankWorkSpacePtr + offsets[COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX]);
        params.cft_le_combine_recv = reinterpret_cast<void*>(rankWorkSpacePtr + combineRecvRegionOffset);

        {
            int const staticMaxTokens = static_cast<int>(offsets[MAX_NUM_TOKENS_INDEX]);
            params.combine_counter_ep_stride = staticMaxTokens;
            params.cft_combine_counter_baseline
                = reinterpret_cast<uint64_t*>(rankWorkSpacePtr + offsets[COMBINE_COUNTER_BASELINE_OFFSET_INDEX]);
        }
    }
    else
    {
        params.use_cft_for_combine = false;
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
    params.prepare_first_token = 0;
    if (params.use_low_precision)
    {
        params.prepare_num_tokens = params.ep_size * params.max_tokens_per_rank;
    }
    else if (params.use_cft_for_combine)
    {
        params.prepare_first_token = params.ep_rank * params.max_tokens_per_rank;
        params.prepare_num_tokens = params.max_tokens_per_rank;
    }
    else
    {
        params.prepare_num_tokens = payloadInWorkspace ? 0 : params.ep_size * params.max_tokens_per_rank;
    }

    params.cft_push_payload = params.use_low_precision ? combinePayloadPtr : params.source_payload;
    params.cft_push_stride_per_token
        = params.use_low_precision ? params.workspace_stride_per_token : params.source_stride_per_token;
    params.reduce_stride_per_token
        = params.use_cft_for_combine ? params.wire_bytes_per_token : params.workspace_stride_per_token;

    params.stream = at::cuda::getCurrentCUDAStream();

    if (useFusedRoundKernels())
    {
        moe_a2a_prepare_combine_launch(params);

        // CFT combine push: processing rank pushes results back to originating rank's LE.
        if (params.use_cft_for_combine)
        {
            moe_a2a_cft_combine_push_launch(params);
        }

        // Launch the combine kernel.
        moe_a2a_combine_launch(params);
    }
    else
    {
        namespace legacy = tensorrt_llm::kernels::moe_comm::legacy;
        legacy::moe_a2a_prepare_combine_launch(params);
        if (params.use_cft_for_combine)
        {
            legacy::moe_a2a_cft_combine_push_launch(params);
        }
        legacy::moe_a2a_combine_launch(params);
    }
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

    CHECK_CPU(metainfo);
    CHECK_TYPE(metainfo, torch::kInt64);
    TORCH_CHECK(metainfo.dim() == 1, "metainfo must be a 1D tensor");
    TORCH_CHECK(metainfo.size(0) == static_cast<int64_t>(NUM_METAINFO_FIELDS),
        "metainfo must have NUM_METAINFO_FIELDS elements");
    MoeA2ADataOffsets const& offsets = *reinterpret_cast<MoeA2ADataOffsets const*>(metainfo.data_ptr<int64_t>());

    uint8_t* rankWorkSpacePtr = workspace.data_ptr<uint8_t>() + epRank * workspace.stride(0);
    int* recv_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);
    uint32_t* flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);

    auto const launch = useFusedRoundKernels()
        ? tensorrt_llm::kernels::moe_comm::moe_a2a_sanitize_expert_ids_launch
        : tensorrt_llm::kernels::moe_comm::legacy::moe_a2a_sanitize_expert_ids_launch;
    launch(expert_ids.data_ptr<int32_t>(), recv_counters, flag_val, static_cast<int32_t>(invalid_expert_id), ep_size,
        runtime_max_tokens_per_rank, top_k, at::cuda::getCurrentCUDAStream());
}

// Return a workspace-backed tensor for combine payload region using from_blob
torch::Tensor moeA2AGetCombinePayloadTensorOp(torch::Tensor const& workspace, int64_t epRank, int64_t epSize,
    int64_t runtimeMaxTokensPerRank, int64_t combinePayloadOffset, c10::ScalarType outDtype, int64_t hiddenSize)
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
    TORCH_CHECK(combinePayloadOffset >= 0, "combine_payload_offset must be non-negative");
    TORCH_CHECK(combinePayloadOffset + bytesNeeded <= sizePerRank,
        "workspace does not have enough space for combine payload tensor. combine payload offset=",
        combinePayloadOffset, ", payload size needed=", bytesNeeded, ", workspace size per rank=", sizePerRank);

    uint8_t* base = workspace.data_ptr<uint8_t>();
    uint8_t* rankBase = base + epRank * workspace.stride(0);
    uint8_t* dataPtr = rankBase + combinePayloadOffset;

    auto options = workspace.options().dtype(outDtype);
    torch::Tensor t = torch::from_blob(dataPtr, {epSize * runtimeMaxTokensPerRank, hiddenSize}, options);
    return t;
}

// Return the size of auxiliary data in workspace
int64_t moeA2AGetAuxDataSizeOp(int64_t epSize, int64_t maxNumTokens, torch::optional<int64_t> eplbStatsNumExperts)
{
    int64_t eplbStatsNumExpertsValue = eplbStatsNumExperts.value_or(0);
    TORCH_CHECK(eplbStatsNumExpertsValue >= 0, "eplbStatsNumExperts must be positive if not None.");
    MoeA2ADataOffsets offsets = calculateOffsets(
        static_cast<int>(epSize), static_cast<int>(maxNumTokens), static_cast<int>(eplbStatsNumExpertsValue));
    return static_cast<int64_t>(offsets[PAYLOAD_DATA_OFFSET_INDEX]);
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
        "bool use_cft_counted_writes=False, "
        "int? expert_id_payload_index=None, "
        "int? invalid_token_expert_id=None, "
        "bool enable_rank_mask=False, "
        "Tensor? active_rank_mask=None, "
        "Tensor? quantize_global_scale=None) -> (Tensor(a!)[], int, Tensor(a!))");
    module.def(
        "moe_a2a_combine(Tensor(a) payload, int local_num_tokens,"
        "Tensor(a!) workspace, Tensor metainfo, int runtime_max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k, int combine_payload_offset, "
        "bool use_low_precision=False, "
        "bool use_cft_counted_writes=False, "
        "bool enable_rank_mask=False, "
        "Tensor? active_rank_mask=None) -> Tensor");
    module.def(
        "moe_a2a_cft_initialize(Tensor workspace, int workspace_mem_handle, "
        "int workspace_size_per_rank, int ep_rank, int ep_size) -> ()");
    module.def("moe_a2a_cft_release(Tensor workspace, int ep_rank) -> ()");
    module.def(
        "moe_a2a_initialize(Tensor(a!) workspace, int ep_rank, int ep_size, int max_num_tokens_per_rank, "
        "int? eplb_stats_num_experts=None) -> Tensor");
    module.def(
        "moe_a2a_sanitize_expert_ids(Tensor(a!) expert_ids, Tensor(a!) workspace, Tensor metainfo, int ep_rank, int "
        "invalid_expert_id) -> ()");
    module.def(
        "moe_a2a_get_combine_payload_tensor(Tensor(a) workspace, int ep_rank, int ep_size, int "
        "runtime_max_tokens_per_rank, "
        "int combine_payload_offset, ScalarType out_dtype, int hidden_size) -> Tensor(a)");
    module.def("moe_a2a_get_aux_data_size(int ep_size, int max_num_tokens, int? eplb_stats_num_experts=None) -> int",
        &tensorrt_llm::torch_ext::moe_comm::moeA2AGetAuxDataSizeOp);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, module)
{
    module.impl("moe_a2a_dispatch", &tensorrt_llm::torch_ext::moe_comm::moeA2ADispatchOp);
    module.impl("moe_a2a_combine", &tensorrt_llm::torch_ext::moe_comm::moeA2ACombineOp);
    module.impl("moe_a2a_initialize", &tensorrt_llm::torch_ext::moe_comm::moeA2AInitializeOp);
    module.impl("moe_a2a_sanitize_expert_ids", &tensorrt_llm::torch_ext::moe_comm::moeA2ASanitizeExpertIdsOp);
    module.impl(
        "moe_a2a_get_combine_payload_tensor", &tensorrt_llm::torch_ext::moe_comm::moeA2AGetCombinePayloadTensorOp);
    module.impl("moe_a2a_cft_initialize", &tensorrt_llm::torch_ext::moe_comm::moeA2ACftInitializeOp);
    module.impl("moe_a2a_cft_release", &tensorrt_llm::torch_ext::moe_comm::moeA2ACftReleaseOp);
}
